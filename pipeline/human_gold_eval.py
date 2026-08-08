"""Human gold pass — scores a real hand-annotated set against the LLM labels.

The 300-comment "gold" set was
double-coded by two LLMs (claude-sonnet-5, gemini-3.5-flash), so its
kappa=0.80 / span-F1=57% measure *model-model* agreement. Two models can agree
and both be wrong; that error is invisible to any LLM-vs-LLM statistic. A human
pass over a stratified 100-comment subset makes it visible.

The headline output is the three-way decomposition, computed per decision
(neutral/emotive call, emotion of a matched pair, cause_category of a matched
pair):

  models agree + human agrees      -> consensus VALIDATED
  models agree + human disagrees   -> SHARED MODEL BIAS   (invisible to IAA)
  models disagree                  -> TASK AMBIGUITY      (what IAA already saw)

The shared-bias rate is the number the paper needs: it upper-bounds how much of
the reported agreement is real signal rather than correlated error. Also reports
plain human-vs-model agreement (same metrics as pipeline/gold_report.py, so the
numbers sit next to the existing ones) and validates the sarcasm flag, which
kappa=0.42 between models leaves completely unresolved.

    # score whatever the annotator has finished so far
    python -m pipeline.human_gold_eval

    # self-test with no human labels: replay annotator "A" (= the seeded
    # sonnet rows) out of the 300-comment store. Should show ~perfect agreement
    # with sonnet and near-zero shared bias -- it IS sonnet.
    python -m pipeline.human_gold_eval --self-test

    python -m pipeline.human_gold_eval --export   # dump human labels to JSONL
"""
from __future__ import annotations

import argparse
import json
import os
import sqlite3
from collections import Counter, defaultdict

import config

from .gold_report import cohens_kappa, compare_annotators, load_gold_sample
from .report import match_pairs

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GOLD_DIR = os.path.join(config.LABELS_DIR, "gold")
HUMAN_DB = os.path.join(GOLD_DIR, "human_gold_labels.sqlite")
HUMAN_SAMPLE = os.path.join(config.SAMPLES_DIR, "human_gold_100_seed7.jsonl")
HUMAN_EXPORT = os.path.join(GOLD_DIR, "human_gold_100.jsonl")
REPORT_PATH = os.path.join(GOLD_DIR, "human_gold_report.md")

# The two models whose labels were filed as the "A"/"B" gold
# submissions -- the pair whose agreement the paper currently reports.
MODELS = {
    "sonnet": os.path.join(GOLD_DIR, "anthropic_claude-sonnet-5.labels.jsonl"),
    "gemini": os.path.join(GOLD_DIR, "google_gemini-3.5-flash.labels.jsonl"),
}
LLM_GOLD = os.path.join(GOLD_DIR, "adjudicated.jsonl")  # the current "gold standard"


# ── Loading ──────────────────────────────────────────────────────────────────

def load_human(db_path: str, annotator: str | None = None) -> tuple[str, dict[str, dict]]:
    """(annotator, {comment_id: {"neutral", "pairs"}}) from a gold-tool sqlite."""
    if not os.path.exists(db_path):
        raise SystemExit(f"no human labels yet at {db_path} — run ./run_human_gold.sh first")
    con = sqlite3.connect(db_path)
    if annotator is None:
        names = [r[0] for r in con.execute(
            "SELECT annotator, COUNT(*) n FROM labels GROUP BY annotator ORDER BY n DESC")]
        if not names:
            raise SystemExit(f"{db_path} has no labels yet")
        annotator = names[0]
    out = {}
    for cid, neutral, pairs in con.execute(
            "SELECT comment_id, neutral, pairs FROM labels WHERE annotator = ?", (annotator,)):
        out[cid] = {"neutral": bool(neutral), "pairs": json.loads(pairs)}
    con.close()
    return annotator, out


def load_labels_jsonl(path: str) -> dict[str, dict]:
    out = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            pairs = r.get("pairs") or []
            # adjudicated.jsonl carries an explicit `neutral`; model label files
            # imply it with an empty pair list.
            out[r["comment_id"]] = {"neutral": bool(r.get("neutral", not pairs)),
                                    "pairs": pairs}
    return out


def restrict(labels: dict[str, dict], ids) -> dict[str, dict]:
    return {c: labels[c] for c in ids if c in labels}


# ── Three-way decomposition ──────────────────────────────────────────────────
#
# A "decision" is one comparable categorical call. We only score decisions where
# all three annotators actually have an opinion: the neutral/emotive call always
# qualifies; emotion/cause only where a pair matches across all three by span
# IoU >= 0.5, since a field comparison on unaligned spans is meaningless.

def decisions(human: dict, a: dict, b: dict, thr: float = 0.5) -> list[dict]:
    """One row per comparable decision, with the human's call and both models'."""
    rows = []
    for cid in sorted(set(human) & set(a) & set(b)):
        h, x, y = human[cid], a[cid], b[cid]
        rows.append({"comment_id": cid, "field": "neutral",
                     "human": "neutral" if h["neutral"] else "emotive",
                     "a": "neutral" if x["neutral"] else "emotive",
                     "b": "neutral" if y["neutral"] else "emotive"})
        # Align a<->b first, then require the human to match the same a-pair, so
        # all three rows describe one and the same span.
        ab = match_pairs(x["pairs"], y["pairs"], thr=thr)
        for pa, pb in ab:
            ha = match_pairs(h["pairs"], [pa], thr=thr)
            if not ha:
                continue
            ph = ha[0][0]
            for field in ("emotion", "cause_category"):
                rows.append({"comment_id": cid, "field": field,
                             "human": ph.get(field), "a": pa.get(field), "b": pb.get(field)})
    return rows


def decompose(rows: list[dict]) -> dict:
    """models-agree/human-agrees vs models-agree/human-disagrees vs models-disagree."""
    out = {}
    by_field = defaultdict(list)
    for r in rows:
        by_field[r["field"]].append(r)
        by_field["ALL"].append(r)
    for field, rs in by_field.items():
        agree = [r for r in rs if r["a"] == r["b"]]
        validated = [r for r in agree if r["human"] == r["a"]]
        shared_bias = [r for r in agree if r["human"] != r["a"]]
        ambiguous = [r for r in rs if r["a"] != r["b"]]
        n = len(rs)
        out[field] = {
            "n": n,
            "n_models_agree": len(agree),
            "validated": len(validated),
            "shared_bias": len(shared_bias),
            "ambiguous": len(ambiguous),
            # of the decisions the models agreed on, how often they were both wrong
            "shared_bias_rate": (len(shared_bias) / len(agree)) if agree else None,
            "validated_rate": (len(validated) / len(agree)) if agree else None,
            # on the ambiguous ones, does the human ever back either model?
            "human_backs_a": sum(1 for r in ambiguous if r["human"] == r["a"]),
            "human_backs_b": sum(1 for r in ambiguous if r["human"] == r["b"]),
            "human_backs_neither": sum(1 for r in ambiguous
                                       if r["human"] != r["a"] and r["human"] != r["b"]),
            "bias_pairs": Counter(f"{r['a']} (models) vs {r['human']} (human)"
                                  for r in shared_bias).most_common(10),
        }
    return out


# ── Sarcasm validation ───────────────────────────────────────────────────────

def sarcasm_check(human: dict, models: dict[str, dict], thr: float = 0.5) -> dict:
    """Human-vs-model sarcasm agreement on span-matched pairs.

    kappa=0.42 between two models says the flag is unstable, not whether it's
    *right*. Against a human we get the thing the paper actually needs: is the
    model's sarcasm flag detecting sarcasm, or firing on something else?
    """
    out = {}
    for name, m in models.items():
        h_flags, m_flags = [], []
        for cid in sorted(set(human) & set(m)):
            for ph, pm in match_pairs(human[cid]["pairs"], m[cid]["pairs"], thr=thr):
                h_flags.append(bool(ph.get("sarcasm")))
                m_flags.append(bool(pm.get("sarcasm")))
        tp = sum(1 for h, x in zip(h_flags, m_flags) if h and x)
        fp = sum(1 for h, x in zip(h_flags, m_flags) if not h and x)
        fn = sum(1 for h, x in zip(h_flags, m_flags) if h and not x)
        prec = tp / (tp + fp) if (tp + fp) else None
        rec = tp / (tp + fn) if (tp + fn) else None
        out[name] = {
            "n_matched_pairs": len(h_flags),
            "human_rate": (sum(h_flags) / len(h_flags)) if h_flags else None,
            "model_rate": (sum(m_flags) / len(m_flags)) if m_flags else None,
            "kappa": cohens_kappa(h_flags, m_flags) if h_flags else None,
            "precision": prec, "recall": rec,
            "f1": (2 * prec * rec / (prec + rec)) if prec and rec else None,
        }
    return out


# ── Report ───────────────────────────────────────────────────────────────────

def _f(x, pct=False):
    if x is None:
        return "—"
    return f"{100 * x:.1f}%" if pct else f"{x:.3f}"


def build_report(annotator: str, human: dict, models: dict[str, dict],
                 llm_gold: dict, sample: dict) -> str:
    ids = sorted(human)
    assisted_pairs = sum(1 for c in ids for p in human[c]["pairs"] if p.get("_accepted_from"))
    mode = ("curated from pooled model candidates"
            if assisted_pairs else "blind to model labels")
    L: list[str] = []
    L.append("# Human gold pass — 100-comment hand-annotated check\n")
    L.append(f"- annotator: **{annotator}** (single coder, {mode})")
    L.append(f"- comments labeled: **{len(ids)}**"
             + (f" of {len(sample)}" if sample else ""))
    if sample:
        strata = Counter(sample[c]["stratum"] for c in ids if c in sample)
        L.append(f"- strata covered: {', '.join(f'{k} {v}' for k, v in sorted(strata.items()))}")
    L.append(f"- human neutral rate: {_f(sum(1 for c in ids if human[c]['neutral']) / len(ids), True)}"
             if ids else "")
    L.append("")

    # 1. plain agreement, same metrics as gold_report.py's IAA table
    L.append("## 1. Human vs each labeler\n")
    L.append("Same metrics as the LLM-vs-LLM IAA table, so they're directly comparable.\n")
    L.append("| pair | n | κ neutral | κ emotion | κ cause | span F1 strict | "
             "span F1 relaxed | matched pairs |")
    L.append("|---|---|---|---|---|---|---|---|")

    def row(label, r):
        return (f"| {label} | {r['n']} | {_f(r['kappa_neutral'])} | "
                f"{_f(r['kappa_emotion'])} | {_f(r['kappa_cause'])} | "
                f"{_f(r['span_f1'], True)} | {_f(r['span_f1_relaxed'], True)} | "
                f"{r['n_matched_pairs']} |")

    comparisons = {**models, "llm_gold (adjudicated)": llm_gold}
    for name, other in comparisons.items():
        o = restrict(other, ids)
        if not o:
            continue
        L.append(row(f"human vs {name}", compare_annotators(human, o)))
    if len(models) == 2:
        a, b = list(models)
        L.append(row(f"_{a} vs {b}_ (model–model, for reference)",
                     compare_annotators(restrict(models[a], ids), restrict(models[b], ids))))
    L.append("")
    L.append("> Strict span F1 = emotion_span token IoU ≥ 0.5; relaxed = containment ≥ 0.8, "
             "which counts a span nested inside a longer one as a match. The gap between "
             "them is span-boundary style rather than disagreement about content. Both use "
             "edge-punctuation-stripped tokens, so a trailing `.`/`?` no longer costs "
             "overlap.\n")
    L.append("> If human-vs-model κ lands well below the model-vs-model κ, the reported "
             "agreement is partly models sharing a prior, not the task being easy.\n")

    # 2. shared bias vs ambiguity
    L.append("## 2. Shared model bias vs task ambiguity\n")
    n_pairs = sum(len(human[c]["pairs"]) for c in ids)
    n_assisted = sum(1 for c in ids for p in human[c]["pairs"] if p.get("_accepted_from"))
    assisted_share = (n_assisted / n_pairs) if n_pairs else 0.0
    if n_assisted:
        L.append(f"> **Curated pass:** the annotator worked from a pooled, deduplicated "
                 f"candidate list drawn from every available label source, keeping the "
                 f"correct pairs, discarding the rest, and editing fields where the spans "
                 f"were right but the labels were not "
                 f"({n_assisted}/{n_pairs} kept pairs, {_f(assisted_share, True)}, "
                 f"originate in a candidate). Because the candidates come from the models, "
                 f"the rate below is a **floor** on shared model error — it counts only the "
                 f"decisions where the human rejected a reading both models agreed on. A "
                 f"blind pass (`run_human_gold.sh` without `--assist`) is what turns that "
                 f"floor into an estimate.\n")
    if len(models) == 2:
        a, b = list(models)
        rows = decisions(human, restrict(models[a], ids), restrict(models[b], ids))
        dec = decompose(rows)
        L.append(f"Decision-level, over {dec['ALL']['n']} comparable decisions "
                 f"({a} vs {b} vs human).\n")
        L.append("| field | decisions | models agree | ↳ human agrees (validated) | "
                 "↳ human disagrees (**shared bias**) | models disagree (ambiguity) |")
        L.append("|---|---|---|---|---|---|")
        for field in ("ALL", "neutral", "emotion", "cause_category"):
            d = dec.get(field)
            if not d:
                continue
            label = "**all**" if field == "ALL" else field
            L.append(f"| {label} | {d['n']} | {d['n_models_agree']} | "
                     f"{d['validated']} ({_f(d['validated_rate'], True)}) | "
                     f"{d['shared_bias']} ({_f(d['shared_bias_rate'], True)}) | "
                     f"{d['ambiguous']} |")
        L.append("")
        allf = dec["ALL"]
        L.append(f"**Shared-bias rate: {_f(allf['shared_bias_rate'], True)}** — of the "
                 f"{allf['n_models_agree']} decisions where both models agreed (and where "
                 "LLM-only IAA therefore sees no problem), this is the share the human "
                 "scored differently. That fraction of the reported agreement is correlated "
                 "model error, not task signal.\n")
        amb = allf["ambiguous"]
        if amb:
            L.append(f"On the {amb} decisions where the models disagreed (genuine task "
                     f"ambiguity), the human sided with {a} {allf['human_backs_a']}×, with "
                     f"{b} {allf['human_backs_b']}×, and with neither "
                     f"{allf['human_backs_neither']}×.\n")
        for field in ("neutral", "emotion", "cause_category"):
            bp = dec.get(field, {}).get("bias_pairs")
            if bp:
                L.append(f"Most common shared-bias confusions in `{field}`:\n")
                for k, v in bp:
                    L.append(f"- {k} — {v}×")
                L.append("")

    # 3. sarcasm
    L.append("## 3. Sarcasm flag validation\n")
    sc = sarcasm_check(human, models)
    L.append("| model | matched pairs | human sarcasm rate | model rate | κ | precision | recall | F1 |")
    L.append("|---|---|---|---|---|---|---|---|")
    for name, s in sc.items():
        L.append(f"| {name} | {s['n_matched_pairs']} | {_f(s['human_rate'], True)} | "
                 f"{_f(s['model_rate'], True)} | {_f(s['kappa'])} | {_f(s['precision'], True)} | "
                 f"{_f(s['recall'], True)} | {_f(s['f1'], True)} |")
    L.append("")
    L.append("> Precision here is what decides the sarcasm question: if the flag is mostly "
             "false positives, the market study should not be re-signing valence on it. "
             "Pair this with `python -m pipeline.market_sarcasm`, which shows how much the "
             "euphoria/VIX results actually move when sarcastic pairs are dropped or flipped.\n")
    return "\n".join(L)


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", default=HUMAN_DB, help="gold-tool sqlite with the human labels")
    ap.add_argument("--annotator", default=None,
                    help="which annotator column to score (default: the one with most labels)")
    ap.add_argument("--sample", default=HUMAN_SAMPLE)
    ap.add_argument("--out", default=REPORT_PATH)
    ap.add_argument("--export", action="store_true",
                    help=f"also write the human labels as JSONL to {HUMAN_EXPORT}")
    ap.add_argument("--self-test", action="store_true",
                    help="score the seeded sonnet rows out of the 300-comment store instead "
                         "of real human labels — verifies the harness with no annotation done")
    args = ap.parse_args()

    db, annotator = args.db, args.annotator
    if args.self_test:
        db = os.path.join(REPO_ROOT, "gold-tool", "server", "data", "gold_labels.sqlite")
        annotator = annotator or "A"
        print(f"[human-gold] SELF-TEST: replaying '{annotator}' from {os.path.basename(db)} "
              "— expect ~perfect agreement with sonnet and ~0 shared bias\n")

    annotator, human = load_human(db, annotator)
    if not human:
        raise SystemExit(f"annotator '{annotator}' has no labels in {db}")

    models = {name: load_labels_jsonl(p) for name, p in MODELS.items() if os.path.exists(p)}
    llm_gold = load_labels_jsonl(LLM_GOLD) if os.path.exists(LLM_GOLD) else {}
    sample = load_gold_sample(args.sample) if os.path.exists(args.sample) else {}

    # Score only what's been annotated so far, so this is useful mid-pass.
    if sample:
        extra = set(human) - set(sample)
        if extra:
            print(f"[human-gold] note: {len(extra)} labeled comment(s) are outside "
                  f"{os.path.basename(args.sample)} — scoring the intersection only")
            human = {c: v for c, v in human.items() if c in sample}
    models = {n: restrict(m, human) for n, m in models.items()}
    llm_gold = restrict(llm_gold, human)

    print(f"[human-gold] annotator={annotator}  labeled={len(human)}"
          + (f"/{len(sample)}" if sample else "")
          + f"  models={list(models)}")
    missing = [n for n, m in models.items() if len(m) < len(human)]
    if missing:
        print(f"[human-gold] WARNING: {missing} don't cover every labeled comment")

    report = build_report(annotator, human, models, llm_gold, sample)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"[human-gold] wrote {args.out}\n")
    print(report)

    if args.export:
        os.makedirs(os.path.dirname(HUMAN_EXPORT), exist_ok=True)
        with open(HUMAN_EXPORT, "w", encoding="utf-8") as f:
            for cid in sorted(human):
                f.write(json.dumps({"comment_id": cid, "annotator": annotator,
                                    **human[cid]}, ensure_ascii=False) + "\n")
        print(f"[human-gold] exported {len(human)} human labels to {HUMAN_EXPORT}")


if __name__ == "__main__":
    main()
