"""Pilot flag report (spec §10) + joint-review sheet.

Reads every `<model>.labels.jsonl` in a run directory (one per model, from
pipeline.annotate) plus the sample file, and produces:

  * `report.md`  — per-model metrics vs the PRE-COMMITTED gate criteria of
    spec §10 (validity/repair, cause_source distribution, sarcasm rate + Fleiss
    kappa across models, `unclear` usage, pairs-per-comment, interpersonal share,
    label distributions), with PASS/CHECK markers.
  * `review.md`  — (--sheet) human review sheet: each comment with its context
    and every model's pairs side by side, flag-relevant cases marked
    ([sarcasm] [unclear] [interpersonal] [repaired]) for the joint pilot review.

    python -m pipeline.report --run pilot --sample data/samples/pilot_250_seed7.jsonl
    python -m pipeline.report --run pilot --sample ... --sheet
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from collections import Counter

import config

from .textnorm import normalize


def _load_labels(run_dir: str) -> dict[str, dict[str, dict]]:
    """model -> comment_id -> label entry."""
    out: dict[str, dict[str, dict]] = {}
    for path in sorted(glob.glob(os.path.join(run_dir, "*.labels.jsonl"))):
        with open(path, encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                out.setdefault(rec["model"], {})[rec["comment_id"]] = rec
    return out


def _load_sample(path: str) -> dict[str, dict]:
    with open(path, encoding="utf-8") as f:
        return {r["comment_id"]: r for r in map(json.loads, f)}


# ── Agreement helpers ────────────────────────────────────────────────────────

def fleiss_kappa(rows: list[list[int]]) -> float | None:
    """Fleiss' kappa. rows[i] = per-category rating counts for item i (same total n)."""
    rows = [r for r in rows if sum(r) > 1]
    if not rows:
        return None
    n = sum(rows[0])
    N, k = len(rows), len(rows[0])
    p_cat = [sum(r[j] for r in rows) / (N * n) for j in range(k)]
    P_bar = sum((sum(c * c for c in r) - n) / (n * (n - 1)) for r in rows) / N
    Pe = sum(p * p for p in p_cat)
    if Pe >= 1.0:
        return 1.0
    return (P_bar - Pe) / (1 - Pe)


# Edge punctuation is stripped per token before comparison: annotators and
# models routinely select the same phrase with or without its trailing '.'/'?',
# which made "fed." and "fed" distinct tokens and cost IoU for no semantic
# reason (2026-08-06 audit: 23 span pairs on the gold set differed by nothing
# else, 9 comparisons crossed the 0.5 match gate). Applied only here, in the
# scoring path — textnorm.normalize() is left alone so the *validator's*
# verbatim guarantee stays exactly as strict as spec §4 requires.
_EDGE_PUNCT = ".,;:!?'\"()[]{}-…"


def _tokens(span: str | None) -> set[str]:
    if not span:
        return set()
    return {t for t in (w.strip(_EDGE_PUNCT) for w in normalize(span).split()) if t}


def span_iou(a: str | None, b: str | None) -> float:
    """Strict overlap: token Jaccard. Punishes length differences — a short span
    nested in a long one scores |short|/|long|."""
    ta, tb = _tokens(a), _tokens(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def span_containment(a: str | None, b: str | None) -> float:
    """Relaxed overlap: |A∩B| / min(|A|,|B|). Reaches 1.0 when one span's tokens
    are a subset of the other's however different the lengths — i.e. the same
    span marked at different boundaries, which is a style difference rather
    than a disagreement about what the span is."""
    ta, tb = _tokens(a), _tokens(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / min(len(ta), len(tb))


def match_pairs(pa: list[dict], pb: list[dict], thr: float = 0.5,
                relaxed: bool = False) -> list[tuple[dict, dict]]:
    """Greedy 1:1 matching of two pair lists by emotion_span overlap.

    relaxed=False (default, unchanged): token IoU >= thr — the strict criterion
    every previously reported number uses.
    relaxed=True: containment >= thr, which accepts nested spans. The gap
    between the two measures how much apparent disagreement is span-boundary
    style rather than substance.
    """
    score = span_containment if relaxed else span_iou
    cands = sorted(((score(x.get("emotion_span"), y.get("emotion_span")), i, j)
                    for i, x in enumerate(pa) for j, y in enumerate(pb)),
                   reverse=True)
    used_a: set[int] = set()
    used_b: set[int] = set()
    matches = []
    for iou, i, j in cands:
        if iou < thr:
            break
        if i in used_a or j in used_b:
            continue
        used_a.add(i)
        used_b.add(j)
        matches.append((pa[i], pb[j]))
    return matches


# ── Report ───────────────────────────────────────────────────────────────────

def _pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def _dist_line(counter: Counter, total: int, top: int = 20) -> str:
    if not total:
        return "—"
    return ", ".join(f"{k} {_pct(v / total)}" for k, v in counter.most_common(top))


def build_report(run: str, sample_path: str) -> str:
    run_dir = os.path.join(config.LABELS_DIR, run)
    labels = _load_labels(run_dir)
    sample = _load_sample(sample_path)
    if not labels:
        raise SystemExit(f"no *.labels.jsonl found in {run_dir}")

    lines = [f"# Pilot flag report — run `{run}`", "",
             f"Sample: `{sample_path}` ({len(sample)} items) — models: "
             + ", ".join(f"`{m}`" for m in labels), ""]

    per_model_sarcasm: dict[str, dict[str, bool]] = {}
    per_model_neutral: dict[str, dict[str, bool]] = {}

    for model, recs in labels.items():
        n = len(recs)
        valid = [r for r in recs.values() if r["valid"]]
        repaired = sum(1 for r in recs.values() if r.get("repaired"))
        fixes = sum(r.get("source_fixes", 0) for r in recs.values())
        pairs = [(r, p) for r in valid for p in (r.get("pairs") or [])]
        np_ = len(pairs)
        emotive = [r for r in valid if r.get("pairs")]

        emo = Counter(p["emotion"] for _, p in pairs)
        cause = Counter(p["cause_category"] for _, p in pairs)
        src = Counter(p.get("cause_source") for _, p in pairs if p.get("cause_source"))
        asset = Counter(p["target_asset"] for _, p in pairs)
        ppc = Counter(len(r.get("pairs") or []) for r in valid)
        sarcasm_n = sum(1 for _, p in pairs if p.get("sarcasm"))
        unclear_n = cause.get("unclear", 0)
        inter_n = cause.get("interpersonal", 0)
        intens = [p["intensity"] for _, p in pairs]

        per_model_sarcasm[model] = {r["comment_id"]: any(p.get("sarcasm")
                                    for p in (r.get("pairs") or [])) for r in valid}
        per_model_neutral[model] = {r["comment_id"]: not r.get("pairs") for r in valid}

        # Gate checks (spec §10) that are computable per single model.
        first_pass = (len(valid) - repaired) / n if n else 0
        ctx_share = (src.get("post", 0) + src.get("parent", 0)) / np_ if np_ else 0
        sarc_rate = sarcasm_n / np_ if np_ else 0
        uncl_rate = unclear_n / np_ if np_ else 0
        at_cap = sum(v for k, v in ppc.items() if k >= 8)

        def mark(ok: bool) -> str:
            return "PASS" if ok else "**CHECK**"

        lines += [
            f"## `{model}`", "",
            f"- items: {n} | valid: {len(valid)} ({_pct(len(valid)/n)}) | "
            f"via repair: {repaired} | cause_source fixes: {fixes}",
            f"- pairs: {np_} total | emotive comments: {len(emotive)}/{len(valid)} "
            f"({_pct(len(emotive)/len(valid)) if valid else '—'}) | "
            f"mean intensity: {sum(intens)/np_:.2f}" if np_ else
            f"- pairs: 0",
            f"- pairs/comment: " + ", ".join(f"{k}×{v}" for k, v in sorted(ppc.items())),
            f"- emotions: {_dist_line(emo, np_)}",
            f"- causes: {_dist_line(cause, np_)}",
            f"- cause_source: {_dist_line(src, sum(src.values()))}",
            f"- target_asset: {_dist_line(asset, np_)}", "",
            "| §10 gate | value | expectation | status |",
            "|---|---|---|---|",
            f"| first-pass validity (verbatim proxy) | {_pct(first_pass)} | ≥ 95% "
            f"after repair → final {_pct(len(valid)/n)} | {mark(len(valid)/n >= 0.95)} |",
            f"| cause_source ∈ {{post,parent}} | {_pct(ctx_share)} | ≥ 5% | "
            f"{mark(ctx_share >= 0.05)} |",
            f"| sarcasm rate | {_pct(sarc_rate)} | ~10–25%, >40% = over-trigger | "
            f"{mark(sarc_rate <= 0.40)} |",
            f"| unclear pairs | {_pct(uncl_rate)} | < 10% | {mark(uncl_rate < 0.10)} |",
            f"| comments at 8-pair cap | {at_cap} | 0 | {mark(at_cap == 0)} |",
            f"| interpersonal share | {_pct(inter_n/np_) if np_ else '—'} | report | — |",
            "",
        ]

    # Cross-model agreement (needs ≥ 2 models).
    models = list(labels)
    if len(models) >= 2:
        common = set.intersection(*(set(per_model_neutral[m]) for m in models))
        sarc_rows = [[sum(1 for m in models if per_model_sarcasm[m].get(cid)),
                      sum(1 for m in models if not per_model_sarcasm[m].get(cid))]
                     for cid in common]
        neut_rows = [[sum(1 for m in models if per_model_neutral[m].get(cid)),
                      sum(1 for m in models if not per_model_neutral[m].get(cid))]
                     for cid in common]
        k_sarc = fleiss_kappa(sarc_rows)
        k_neut = fleiss_kappa(neut_rows)

        emo_agree, cause_agree, match_rate, mn = [], [], [], 0
        for a in range(len(models)):
            for b in range(a + 1, len(models)):
                ra, rb = labels[models[a]], labels[models[b]]
                for cid in common:
                    pa = ra[cid].get("pairs") or []
                    pb = rb[cid].get("pairs") or []
                    if not pa and not pb:
                        continue
                    ms = match_pairs(pa, pb)
                    mn += 1
                    match_rate.append(2 * len(ms) / (len(pa) + len(pb)))
                    for x, y in ms:
                        emo_agree.append(x["emotion"] == y["emotion"])
                        cause_agree.append(x["cause_category"] == y["cause_category"])

        def avg(xs):
            return sum(xs) / len(xs) if xs else None

        lines += ["## Cross-model agreement", "",
                  f"- common items: {len(common)}",
                  f"- sarcasm Fleiss κ (comment level): "
                  f"{k_sarc:.3f}" if k_sarc is not None else "- sarcasm κ: —",
                  f"- neutral-vs-emotive Fleiss κ: "
                  f"{k_neut:.3f}" if k_neut is not None else "- neutral κ: —",
                  f"- pair match rate (span IoU ≥ 0.5): {_pct(avg(match_rate) or 0)}",
                  f"- emotion agreement on matched pairs: {_pct(avg(emo_agree) or 0)}",
                  f"- cause_category agreement on matched pairs: "
                  f"{_pct(avg(cause_agree) or 0)}",
                  "",
                  "§10 sarcasm rule: κ ≥ 0.5 evaluate in paper · 0.2–0.5 metadata-only "
                  "· < 0.2 drop flag + audit.", ""]

    out = os.path.join(run_dir, "report.md")
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"[report] wrote {out}", flush=True)
    return out


# ── Review sheet ─────────────────────────────────────────────────────────────

def build_sheet(run: str, sample_path: str) -> str:
    run_dir = os.path.join(config.LABELS_DIR, run)
    labels = _load_labels(run_dir)
    sample = _load_sample(sample_path)

    lines = [f"# Pilot review sheet — run `{run}`",
             "", "Markers: ⚑sarcasm ⚑unclear ⚑interpersonal ⚑repaired", ""]
    for cid, item in sample.items():
        entries = {m: recs.get(cid) for m, recs in labels.items() if recs.get(cid)}
        if not entries:
            continue
        lines += [f"---", f"### `{cid}` · {item['stratum']}",
                  f"**POST**: {item.get('post_title')}"]
        if item.get("parent_body"):
            lines.append(f"**PARENT**: {item['parent_body']}")
        lines.append(f"**COMMENT**: {item['body']}")
        for model, rec in entries.items():
            marks = " ⚑repaired" if rec.get("repaired") else ""
            if not rec["valid"]:
                lines.append(f"- `{model}`: INVALID — {rec['errors']}")
                continue
            pairs = rec.get("pairs") or []
            if not pairs:
                lines.append(f"- `{model}`: (neutral — no pairs){marks}")
            for p in pairs:
                fl = "".join(" ⚑" + k for k, v in
                             (("sarcasm", p.get("sarcasm")),
                              ("unclear", p.get("cause_category") == "unclear"),
                              ("interpersonal", p.get("cause_category") == "interpersonal"))
                             if v)
                lines.append(
                    f"- `{model}`: **{p['emotion']}** “{p['emotion_span']}” ← "
                    f"[{p['cause_category']}/{p.get('cause_source') or '∅'}] "
                    f"“{p.get('cause_span') or '∅'}” · {p['target_asset']} · "
                    f"i={p['intensity']}{fl}{marks}")
        lines.append("")

    out = os.path.join(run_dir, "review.md")
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"[report] wrote {out}", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser(description="Pilot flag report + review sheet")
    ap.add_argument("--run", default="pilot")
    ap.add_argument("--sample", required=True)
    ap.add_argument("--sheet", action="store_true", help="also write review.md")
    args = ap.parse_args()
    build_report(args.run, args.sample)
    if args.sheet:
        build_sheet(args.run, args.sample)


if __name__ == "__main__":
    main()
