"""Human gold-set IAA + silver-vs-gold validation (spec section "Human Gold Set").

Reads the double-coded gold labels straight from gold-tool's sqlite (A +
B, 300/300 each) and:

  * reports inter-annotator agreement (Cohen's kappa on neutral-vs-emotive, on
    emotion and cause_category of matched pairs, plus span-level F1 at
    IoU >= 0.5) -- these are computable now, straight off the raw double
    coding, no adjudication needed.
  * writes a disagreement worklist (`adjudication_worklist.md`) for the
    comments where the two annotators don't fully agree, for the
    discussion-based adjudication pass the paper's methodology calls for.
  * once that pass is done and the worklist's FINAL: tags are filled in,
    `--apply-worklist` merges it with the auto-agreed comments into the
    single adjudicated gold set (`adjudicated.jsonl`).
  * `--score-silver` then scores the silver consensus
    (data/labels/silver/consensus.jsonl) against the adjudicated gold set:
    typed-pair F1 (span IoU >= 0.5 + emotion match) and +cause_category.

    python -m pipeline.gold_report                    # IAA report + worklist
    python -m pipeline.gold_report --apply-worklist data/labels/gold/adjudication_worklist.md
    python -m pipeline.gold_report --score-silver
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sqlite3
from collections import Counter

import config

from .report import match_pairs, span_iou  # noqa: F401  (span_iou re-exported for callers)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GOLD_DB = os.path.join(REPO_ROOT, "gold-tool", "server", "data", "gold_labels.sqlite")
GOLD_SAMPLE = os.path.join(config.SAMPLES_DIR, "gold_300_seed7.jsonl")
GOLD_DIR = os.path.join(config.LABELS_DIR, "gold")
SILVER_CONSENSUS = os.path.join(config.LABELS_DIR, "silver", "consensus.jsonl")
ANNOTATORS = ("A", "B")

# The two "annotator" slots in gold_labels.sqlite hold the output of two frontier
# LLMs, whose labels.jsonl files were promoted into those rows. The DB keys are
# kept (renaming them would invalidate the adjudication trail), but every report
# names the actual source — this set is model-vs-model agreement, not human IAA,
# and reading it as human IAA is exactly the error the human benchmark exists to
# correct (see pipeline/human_gold_eval.py).
ANNOTATOR_PROVENANCE = {"A": "claude-sonnet-5", "B": "gemini-3.5-flash"}


def _provenance(name: str) -> str:
    src = ANNOTATOR_PROVENANCE.get(name)
    return f"{name} ({src})" if src else name


# ── Loading ──────────────────────────────────────────────────────────────────

def load_double_coded(db_path: str = GOLD_DB) -> dict[str, dict[str, dict]]:
    """annotator -> comment_id -> {"neutral": bool, "pairs": [...]}"""
    con = sqlite3.connect(db_path)
    out: dict[str, dict[str, dict]] = {a: {} for a in ANNOTATORS}
    for comment_id, annotator, neutral, pairs in con.execute(
            "SELECT comment_id, annotator, neutral, pairs FROM labels"):
        out.setdefault(annotator, {})[comment_id] = {
            "neutral": bool(neutral), "pairs": json.loads(pairs)}
    con.close()
    return out


def load_gold_sample(path: str = GOLD_SAMPLE) -> dict[str, dict]:
    with open(path, encoding="utf-8") as f:
        return {r["comment_id"]: r for r in map(json.loads, f)}


def _load_labels_file(path: str) -> dict[str, dict]:
    """A per-model <slug>.labels.jsonl (from pipeline.annotate) as an annotator
    record map: {comment_id: {"neutral", "pairs"}}. Empty/None pairs = neutral."""
    out: dict[str, dict] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            pairs = r.get("pairs") or []
            out[r["comment_id"]] = {"neutral": not pairs, "pairs": pairs}
    return out


def load_coded_from_files(labels_a: str, labels_b: str,
                          names: tuple[str, str] = ANNOTATORS
                          ) -> dict[str, dict[str, dict]]:
    """Two annotator slots loaded straight from two model .labels.jsonl files
    instead of the economics gold-tool sqlite — used for per-subreddit gold
    transfer checks where the two 'annotators' are two frontier models."""
    return {names[0]: _load_labels_file(labels_a),
            names[1]: _load_labels_file(labels_b)}


# ── Agreement ────────────────────────────────────────────────────────────────

def cohens_kappa(a: list, b: list) -> float | None:
    """Unweighted Cohen's kappa over paired categorical labels."""
    n = len(a)
    if n == 0:
        return None
    po = sum(1 for x, y in zip(a, b) if x == y) / n
    cats = set(a) | set(b)
    ca, cb = Counter(a), Counter(b)
    pe = sum((ca[c] / n) * (cb[c] / n) for c in cats)
    if pe >= 1.0:
        return 1.0
    return (po - pe) / (1 - pe)


def compare_annotators(a_labels: dict[str, dict], b_labels: dict[str, dict],
                        thr: float = 0.5, relaxed_thr: float = 0.8) -> dict:
    """IAA metrics + per-comment disagreement flags between two annotators'
    {comment_id: {"neutral", "pairs"}} maps.

    Span agreement is reported twice: `span_f1` under the strict token-IoU
    criterion (unchanged, primary) and `span_f1_relaxed` under containment,
    which counts nested spans as matching. Everything else — kappas,
    disagreement flags — is computed on the strict matching only.
    """
    ids = sorted(set(a_labels) & set(b_labels))

    neutral_a = ["neutral" if a_labels[c]["neutral"] else "emotive" for c in ids]
    neutral_b = ["neutral" if b_labels[c]["neutral"] else "emotive" for c in ids]
    k_neutral = cohens_kappa(neutral_a, neutral_b)

    emo_a, emo_b, cause_a, cause_b = [], [], [], []
    total_a_pairs = total_b_pairs = total_matches = 0
    total_matches_relaxed = 0
    disagreements = []
    for cid in ids:
        pa, pb = a_labels[cid]["pairs"], b_labels[cid]["pairs"]
        total_a_pairs += len(pa)
        total_b_pairs += len(pb)
        ms = match_pairs(pa, pb, thr=thr)
        total_matches += len(ms)
        # Same matching under the relaxed span criterion; the gap between the
        # two span F1s is how much "disagreement" is only span boundaries.
        total_matches_relaxed += len(match_pairs(pa, pb, thr=relaxed_thr, relaxed=True))
        field_mismatch = False
        for x, y in ms:
            emo_a.append(x["emotion"]); emo_b.append(y["emotion"])
            cause_a.append(x["cause_category"]); cause_b.append(y["cause_category"])
            if x["emotion"] != y["emotion"] or x["cause_category"] != y["cause_category"]:
                field_mismatch = True
        neutral_mismatch = a_labels[cid]["neutral"] != b_labels[cid]["neutral"]
        count_mismatch = len(ms) != len(pa) or len(ms) != len(pb)
        if neutral_mismatch or count_mismatch or field_mismatch:
            disagreements.append(cid)

    denom = total_a_pairs + total_b_pairs
    span_f1 = (2 * total_matches / denom) if denom else None
    span_f1_relaxed = (2 * total_matches_relaxed / denom) if denom else None

    return {
        "n": len(ids), "ids": ids,
        "kappa_neutral": k_neutral,
        "kappa_emotion": cohens_kappa(emo_a, emo_b),
        "kappa_cause": cohens_kappa(cause_a, cause_b),
        "span_f1": span_f1, "n_matched_pairs": len(emo_a),
        "span_f1_relaxed": span_f1_relaxed, "n_matched_relaxed": total_matches_relaxed,
        "total_a_pairs": total_a_pairs, "total_b_pairs": total_b_pairs,
        "disagreements": disagreements,
    }


# ── IAA report ───────────────────────────────────────────────────────────────

def _fmt(x: float | None, pct: bool = False) -> str:
    if x is None:
        return "—"
    return f"{100*x:.1f}%" if pct else f"{x:.3f}"


def build_iaa_report(out_dir: str = GOLD_DIR, coded: dict | None = None,
                     names: tuple[str, str] = ANNOTATORS) -> dict:
    if coded is None:
        coded = load_double_coded()
    a, b = names
    stats = compare_annotators(coded[a], coded[b])

    os.makedirs(out_dir, exist_ok=True)
    lines = [
        f"# Gold set: cross-annotator agreement ({_provenance(a)} vs {_provenance(b)})", "",
        "> Both slots are frontier-LLM output, so these are **model-vs-model** numbers.",
        "> For agreement against a hand-annotated set see `human_gold_report.md`.",
        "",
        f"- comments double-coded: {stats['n']}",
        f"- matched pairs (IoU >= 0.5): {stats['n_matched_pairs']} "
        f"({a} {stats['total_a_pairs']} pairs, {b} {stats['total_b_pairs']} pairs)",
        f"- neutral/emotive Cohen's kappa: {_fmt(stats['kappa_neutral'])}",
        f"- emotion kappa on matched pairs: {_fmt(stats['kappa_emotion'])}",
        f"- cause_category kappa on matched pairs: {_fmt(stats['kappa_cause'])}",
        f"- span F1, strict (IoU >= 0.5): {_fmt(stats['span_f1'], pct=True)}",
        f"- span F1, relaxed (containment >= 0.8): "
        f"{_fmt(stats['span_f1_relaxed'], pct=True)} "
        f"({stats['n_matched_relaxed']} matched) — the gap over strict is span-boundary "
        f"style (same span marked long vs short), not disagreement about content",
        f"- comments with any disagreement: {len(stats['disagreements'])} "
        f"({_fmt(len(stats['disagreements'])/stats['n'], pct=True)})",
        "",
        "Disagreements need discussion-based adjudication before the gold set "
        "is final (see `adjudication_worklist.md`); these IAA numbers are "
        "computed straight off the raw double coding and don't depend on it.",
        "",
    ]
    out = os.path.join(out_dir, "iaa_report.md")
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"[gold_report] wrote {out}", flush=True)
    return stats


# ── Disagreement worklist ───────────────────────────────────────────────────

def _fmt_pairs(label: str, rec: dict) -> list[str]:
    if rec["neutral"] or not rec["pairs"]:
        return [f"**{label}**: (neutral — no pairs)"]
    out = []
    for p in rec["pairs"]:
        out.append(
            f"**{label}**: **{p['emotion']}** “{p['emotion_span']}” ← "
            f"[{p['cause_category']}/{p.get('cause_source') or '∅'}] "
            f"“{p.get('cause_span') or '∅'}” · {p['target_asset']} "
            f"· i={p['intensity']}" + (" ⛑sarcasm" if p.get("sarcasm") else ""))
    return out


def build_worklist(out_dir: str = GOLD_DIR) -> str:
    coded = load_double_coded()
    stats = compare_annotators(coded["A"], coded["B"])
    sample = load_gold_sample()

    lines = [
        "# Gold set adjudication worklist", "",
        f"{len(stats['disagreements'])}/{stats['n']} comments have some "
        "disagreement between A and B. The other "
        f"{stats['n'] - len(stats['disagreements'])} already agree and need no action.",
        "",
        "**Instructions**: discuss each comment below and write the agreed",
        "call on the `FINAL:` line, using exactly one of:",
        "", "```",
        "FINAL: A      <- A's record is correct as-is",
        "FINAL: B    <- B's record is correct as-is",
        "FINAL: NEUTRAL      <- no pairs (overrides both)",
        "FINAL: CUSTOM",
        "- emotion: fear_anxiety",
        "  emotion_span: verbatim substring of the comment",
        "  cause_span: verbatim substring of comment/parent/post (or null)",
        "  cause_source: comment|parent|post|null",
        "  cause_category: one of the frozen 14 (+other/unclear)",
        "  target_asset: equities|bonds|gold|crypto|USD|housing|none",
        "  intensity: 0..1",
        "  sarcasm: true|false",
        "  (repeat the block above for each pair; omit entirely for a neutral call)",
        "```", "",
        "Leave `FINAL: TBD` (already filled in below) until you've discussed it.",
        "",
    ]

    for cid in stats["disagreements"]:
        item = sample[cid]
        lines += [f"---", f"### `{cid}` · {item['stratum']}/{item.get('topic_bucket', '')}",
                  f"**POST**: {item.get('post_title')}"]
        if item.get("post_selftext"):
            lines.append(f"**POST BODY**: {item['post_selftext']}")
        if item.get("parent_body"):
            lines.append(f"**PARENT**: {item['parent_body']}")
        lines.append(f"**COMMENT**: {item['body']}")
        lines.append("")
        lines += _fmt_pairs("A", coded["A"][cid])
        lines += _fmt_pairs("B", coded["B"][cid])
        lines.append("")
        lines.append("FINAL: TBD")
        lines.append("")

    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, "adjudication_worklist.md")
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"[gold_report] wrote {out} ({len(stats['disagreements'])} items)", flush=True)
    return out


# ── Apply a completed worklist ──────────────────────────────────────────────

_PAIR_FIELD_RE = re.compile(r"^\s*(\w+):\s*(.*)$")
_PAIR_FIELDS = {"emotion", "emotion_span", "cause_span", "cause_source",
                "cause_category", "target_asset", "intensity", "sarcasm"}
_NULLABLE_FIELDS = {"cause_span", "cause_source"}  # "none" is a real enum
                                                   # value for target_asset


def _parse_custom_pairs(block_lines: list[str]) -> list[dict]:
    pairs, cur = [], None
    for line in block_lines:
        if line.strip().startswith("- emotion:"):
            if cur:
                pairs.append(cur)
            cur = {}
            line = line.strip()[2:]  # drop "- "
        m = _PAIR_FIELD_RE.match(line)
        if not m or cur is None:
            continue
        key, val = m.group(1), m.group(2).strip()
        if key not in _PAIR_FIELDS:  # stray prose line, not a field
            continue
        if key == "intensity":
            cur[key] = float(val)
        elif key == "sarcasm":
            cur[key] = val.lower() == "true"
        elif key in _NULLABLE_FIELDS and val.lower() in ("null", "none", "~", ""):
            cur[key] = None
        else:
            cur[key] = val
    if cur:
        pairs.append(cur)
    return pairs


def parse_worklist(path: str) -> dict[str, dict]:
    """Returns comment_id -> {"neutral": bool, "pairs": [...]} for resolved
    items only (raises if any FINAL is still TBD)."""
    coded = load_double_coded()
    text = open(path, encoding="utf-8").read()
    blocks = re.split(r"^---\s*$", text, flags=re.MULTILINE)[1:]

    resolved: dict[str, dict] = {}
    unresolved = []
    for block in blocks:
        m = re.search(r"^### `([^`]+)`", block, flags=re.MULTILINE)
        if not m:
            continue
        cid = m.group(1)
        fm = re.search(r"^FINAL:\s*(\S+)", block, flags=re.MULTILINE)
        final = fm.group(1).upper() if fm else "TBD"
        if final == "TBD":
            unresolved.append(cid)
        elif final == "A":
            resolved[cid] = coded["A"][cid]
        elif final == "B":
            resolved[cid] = coded["B"][cid]
        elif final == "NEUTRAL":
            resolved[cid] = {"neutral": True, "pairs": []}
        elif final == "CUSTOM":
            lines = block.splitlines()
            start = next(i for i, l in enumerate(lines) if l.strip() == "FINAL: CUSTOM") + 1
            pairs = _parse_custom_pairs(lines[start:])
            resolved[cid] = {"neutral": not pairs, "pairs": pairs}
        else:
            unresolved.append(cid)

    if unresolved:
        raise SystemExit(
            f"{len(unresolved)} comments still have FINAL: TBD — finish the "
            f"discussion pass first: {unresolved[:10]}{'...' if len(unresolved) > 10 else ''}")
    return resolved


def build_adjudicated_gold(worklist_path: str, out_dir: str = GOLD_DIR) -> str:
    coded = load_double_coded()
    stats = compare_annotators(coded["A"], coded["B"])
    resolved = parse_worklist(worklist_path)

    adjudicated = {}
    for cid in stats["ids"]:
        if cid in stats["disagreements"]:
            adjudicated[cid] = resolved[cid]
        else:
            adjudicated[cid] = coded["A"][cid]  # the two agree; either works

    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, "adjudicated.jsonl")
    with open(out, "w", encoding="utf-8") as f:
        for cid, rec in adjudicated.items():
            f.write(json.dumps({"comment_id": cid, **rec}) + "\n")
    print(f"[gold_report] wrote {out} ({len(adjudicated)} comments)", flush=True)
    return out


# ── Silver vs gold ───────────────────────────────────────────────────────────

def score_silver_vs_gold(adjudicated_path: str = os.path.join(GOLD_DIR, "adjudicated.jsonl"),
                          consensus_path: str = SILVER_CONSENSUS,
                          thr: float = 0.5) -> dict:
    if not os.path.exists(adjudicated_path):
        raise SystemExit(
            f"{adjudicated_path} doesn't exist yet — finish the discussion pass "
            "and run --apply-worklist first")

    gold = {}
    with open(adjudicated_path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            gold[r["comment_id"]] = r

    silver = {}
    with open(consensus_path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r["comment_id"] in gold:
                silver[r["comment_id"]] = [p for p in r["pairs"] if p.get("included")]

    n_gold_pairs = n_silver_pairs = 0
    n_span_match = n_typed_match = n_full_match = 0
    for cid, g in gold.items():
        gp, sp = g["pairs"], silver.get(cid, [])
        n_gold_pairs += len(gp)
        n_silver_pairs += len(sp)
        ms = match_pairs(gp, sp, thr=thr)
        n_span_match += len(ms)
        for x, y in ms:
            if x["emotion"] == y["emotion"]:
                n_typed_match += 1
                if x["cause_category"] == y["cause_category"]:
                    n_full_match += 1

    def f1(matches, n_gold, n_pred):
        p = matches / n_pred if n_pred else 0.0
        r = matches / n_gold if n_gold else 0.0
        return 2 * p * r / (p + r) if (p + r) else 0.0

    return {
        "n_comments": len(gold), "n_gold_pairs": n_gold_pairs, "n_silver_pairs": n_silver_pairs,
        "span_f1": f1(n_span_match, n_gold_pairs, n_silver_pairs),
        "typed_pair_f1": f1(n_typed_match, n_gold_pairs, n_silver_pairs),
        "full_triplet_f1": f1(n_full_match, n_gold_pairs, n_silver_pairs),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--apply-worklist", metavar="PATH",
                     help="parse a completed adjudication_worklist.md into adjudicated.jsonl")
    ap.add_argument("--score-silver", action="store_true",
                     help="score silver consensus against the adjudicated gold set")
    # Per-subreddit gold transfer check: the two 'annotators' are two frontier
    # models' .labels.jsonl files rather than the economics gold-tool sqlite.
    ap.add_argument("--labels-a", help="annotator-A .labels.jsonl (→ A slot)")
    ap.add_argument("--labels-b", help="annotator-B .labels.jsonl (→ B slot)")
    ap.add_argument("--names", default=",".join(ANNOTATORS),
                    help="two display names for the report (default A,B)")
    ap.add_argument("--out-dir", help=f"IAA report dir (default {GOLD_DIR})")
    args = ap.parse_args()

    if args.apply_worklist:
        build_adjudicated_gold(args.apply_worklist)
        return
    if args.score_silver:
        result = score_silver_vs_gold()
        print(json.dumps(result, indent=2))
        return

    if args.labels_a or args.labels_b:
        if not (args.labels_a and args.labels_b):
            raise SystemExit("--labels-a and --labels-b must be given together")
        names = tuple(s.strip() for s in args.names.split(","))
        if len(names) != 2:
            raise SystemExit("--names needs exactly two comma-separated names")
        coded = load_coded_from_files(args.labels_a, args.labels_b, names)
        build_iaa_report(args.out_dir or GOLD_DIR, coded, names)
        return

    build_iaa_report()
    build_worklist()


if __name__ == "__main__":
    main()
