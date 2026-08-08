"""Recompute every gold-set extraction number in the paper from saved predictions.

Any change to span matching (`pipeline/report.py`) moves every span-F1 in
Table `tab:extraction` at once, and those numbers were produced by separate
runs on separate machines over several weeks. This rescoring pass recomputes
them all from the stored prediction files, so the table stays internally
consistent instead of mixing matcher versions.

`--old` restores the pre-2026-08-06 tokenizer (no edge-punctuation stripping).
Run it first: if a row reproduces its published number under `--old`, the same
row's `--new` number is trustworthy. Rows that don't reproduce are flagged
rather than silently republished.

    python -m pipeline.rescore_gold --old     # verify against the paper
    python -m pipeline.rescore_gold           # the corrected numbers
"""
from __future__ import annotations

import argparse
import json
import os

import config

from . import report
from .report import match_pairs
from .textnorm import normalize

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GOLD = os.path.join(config.LABELS_DIR, "gold", "adjudicated.jsonl")
INV_GOLD = os.path.join(config.LABELS_DIR, "investing_gold", "adjudicated.jsonl")
SILVER = os.path.join(config.LABELS_DIR, "silver")
OUT = os.path.join(REPO_ROOT, "outputs")

# (paper row label, predictions path, gold path, published Pair/+E/+C/Neut.)
# Published values are the paper's Table `tab:extraction` as of 2026-08-06 and
# exist only so `--old` can prove the harness reproduces them.
SYSTEMS = [
    ("DeepSeek-V4-Flash", f"{SILVER}/deepseek_deepseek-v4-flash.labels.jsonl", GOLD,
     (51.8, 46.1, 36.3, 88.7)),
    ("MiMo-V2.5", f"{SILVER}/xiaomi_mimo-v2.5.labels.jsonl", GOLD,
     (53.3, 48.7, 34.5, 87.0)),
    ("MiniMax-M3 (subset)", f"{SILVER}/minimax_minimax-m3.labels.jsonl", GOLD,
     (52.3, 43.4, 30.5, 86.3)),
    ("Consensus (silver)", f"{SILVER}/consensus.jsonl", GOLD,
     (61.4, 56.5, 42.9, 86.7)),
    ("Student zero-shot", f"{OUT}/base_qwen3.5-9b_gold_preds.jsonl", GOLD,
     (45.2, 36.9, 23.0, 83.7)),
    ("Student LoRA support>=2",
     f"{OUT}/qwen3.5-9b-silver_clean-lr1e-4/final-merged-vlm/gold_preds.jsonl", GOLD,
     (53.0, 47.4, 33.8, 86.3)),
    ("Student LoRA support>=1",
     f"{OUT}/qwen3.5-9b-silver_recall-lr1e-4/ckpt1300-merged-vlm/gold_preds.jsonl", GOLD,
     (59.6, 52.4, 37.4, 89.0)),
    ("DeBERTaV3 tagger", f"{OUT}/deberta-v3-tagger/epoch2.gold_preds.jsonl", GOLD,
     (47.7, 43.3, 27.6, 80.7)),
    ("r/investing: Qwen3.5-9B LoRA",
     os.path.join(config.LABELS_DIR, "investing_gold", "student_recall_preds.jsonl"), INV_GOLD,
     (61.3, 53.8, 36.0, 85.5)),
    ("r/investing: DeBERTaV3",
     f"{OUT}/deberta-v3-tagger/epoch2.investing_gold_preds.jsonl", INV_GOLD,
     (37.1, 32.6, 11.1, 72.0)),
]


def load(path: str) -> dict[str, list[dict]]:
    """{comment_id: pairs}. Consensus records carry rejected candidates too, so
    keep only the ones the vote actually included."""
    out: dict[str, list[dict]] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            pairs = r.get("pairs") or []
            if any("included" in p for p in pairs):
                pairs = [p for p in pairs if p.get("included")]
            out[r["comment_id"]] = pairs
    return out


def score(gold: dict, pred: dict, thr: float = 0.5, relaxed: bool = False) -> dict:
    """Pair / typed-pair / full-triplet F1 against the full gold set.

    Iterates every gold comment with `pred.get(cid, [])`, mirroring
    gold_report.score_silver_vs_gold: a system that didn't answer a comment is
    penalised for the gold pairs it missed rather than having them dropped from
    the denominator. This is what MiniMax's row reflects — it votes on only
    183/300, so the other 117 count against it.

    Neutral accuracy is deliberately not recomputed here: it compares pair
    *counts*, never spans, so no span-matching change can move it and the
    published values stand.
    """
    n_gold = n_pred = n_span = n_typed = n_full = 0
    for cid in sorted(gold):
        gp, pp = gold[cid], pred.get(cid, [])
        n_gold += len(gp)
        n_pred += len(pp)
        for x, y in match_pairs(gp, pp, thr=thr, relaxed=relaxed):
            n_span += 1
            if x["emotion"] == y["emotion"]:
                n_typed += 1
                if x["cause_category"] == y["cause_category"]:
                    n_full += 1
    f1 = lambda m: (200.0 * m / (n_gold + n_pred)) if (n_gold + n_pred) else 0.0
    return {"n": len(gold), "pair": f1(n_span), "typed": f1(n_typed), "full": f1(n_full)}


def counts(gold: dict, pred: dict, ids, thr: float = 0.5):
    """Per-comment (gold, pred, span, typed, full) tallies — the pieces the
    bootstrap resamples. Kept separate from score() so a resample is a cheap
    sum over cached per-comment counts rather than a rematch."""
    out = {}
    for cid in ids:
        gp, pp = gold.get(cid, []), pred.get(cid, [])
        ns = nt = nf = 0
        for x, y in match_pairs(gp, pp, thr=thr):
            ns += 1
            if x["emotion"] == y["emotion"]:
                nt += 1
                if x["cause_category"] == y["cause_category"]:
                    nf += 1
        out[cid] = (len(gp), len(pp), ns, nt, nf)
    return out


def bootstrap(gold: dict, pred_a: dict, pred_b: dict, b: int = 2000, seed: int = 7):
    """Paired bootstrap over comments of the span/typed/full F1 difference A-B.

    Resamples the 300 gold comments with replacement (the unit of independence),
    recomputing both systems' F1 on each resample so the pairing is preserved.
    Returns per-metric (delta, lo, hi, two-sided p).
    """
    import random
    ids = sorted(gold)
    ca, cb = counts(gold, pred_a, ids), counts(gold, pred_b, ids)

    def f1s(c, sample):
        ng = np_ = ns = nt = nf = 0
        for cid in sample:
            g, p, s, t, fl = c[cid]
            ng += g; np_ += p; ns += s; nt += t; nf += fl
        d = ng + np_
        return (200.0 * ns / d, 200.0 * nt / d, 200.0 * nf / d) if d else (0.0, 0.0, 0.0)

    obs = tuple(x - y for x, y in zip(f1s(ca, ids), f1s(cb, ids)))
    rng = random.Random(seed)
    draws = [[], [], []]
    for _ in range(b):
        sample = [ids[rng.randrange(len(ids))] for _ in ids]
        fa, fb = f1s(ca, sample), f1s(cb, sample)
        for k in range(3):
            draws[k].append(fa[k] - fb[k])
    out = []
    for k in range(3):
        d = sorted(draws[k])
        lo, hi = d[int(0.025 * b)], d[int(0.975 * b) - 1]
        p = 2 * min(sum(1 for v in d if v <= 0) / b, sum(1 for v in d if v >= 0) / b)
        out.append((obs[k], lo, hi, max(p, 1.0 / b)))
    return out


def use_old_tokenizer():
    """Pre-2026-08-06 behaviour: whitespace split with punctuation attached."""
    report._tokens = lambda s: set(normalize(s).split()) if s else set()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--old", action="store_true",
                    help="score with the old tokenizer to verify published numbers")
    ap.add_argument("--relaxed-thr", type=float, default=0.8)
    ap.add_argument("--bootstrap", action="store_true",
                    help="also recompute the paired bootstrap CIs the paper cites")
    args = ap.parse_args()

    if args.old:
        use_old_tokenizer()
        print("[rescore] OLD tokenizer — these should reproduce the paper\n")
    else:
        print("[rescore] NEW tokenizer (edge punctuation stripped)\n")

    golds = {p: load(p) for p in {s[2] for s in SYSTEMS} if os.path.exists(p)}
    print(f"| {'System':30s} | Pair | +E   | +C   | "
          + ("Δ vs paper |" if args.old else "Pair(relaxed) |"))
    print("|" + "-" * 72 + "|")
    worst = 0.0
    for label, path, gold_path, pub in SYSTEMS:
        if not os.path.exists(path) or gold_path not in golds:
            print(f"| {label:30s} |  MISSING — published value stands  |")
            continue
        s = score(golds[gold_path], load(path))
        if args.old:
            d = max(abs(s["pair"] - pub[0]), abs(s["typed"] - pub[1]), abs(s["full"] - pub[2]))
            worst = max(worst, d)
            flag = "ok" if d < 0.6 else f"** {d:.1f} **"
            print(f"| {label:30s} | {s['pair']:4.1f} | {s['typed']:4.1f} | {s['full']:4.1f} "
                  f"| {flag:>10s} |")
        else:
            r = score(golds[gold_path], load(path), thr=args.relaxed_thr, relaxed=True)
            print(f"| {label:30s} | {s['pair']:4.1f} | {s['typed']:4.1f} | {s['full']:4.1f} "
                  f"| {r['pair']:13.1f} |")
    if args.old:
        print(f"\n[rescore] worst deviation on Pair/+E/+C: {worst:.2f} points "
              + ("(reproduces — the new numbers are trustworthy)" if worst < 0.6
                 else "(INVESTIGATE before republishing)"))
    print("\nNeutral accuracy compares pair counts, never spans — no span-matching change "
          "can move it, so it is not recomputed and the published values stand.")

    if args.bootstrap:
        gold = golds[GOLD]
        byname = {s[0]: load(s[1]) for s in SYSTEMS if os.path.exists(s[1])}
        # Same three comparisons the paper makes: consensus vs the best single
        # teacher by span F1 (MiMo-V2.5), the two student thresholds against each
        # other, and the recall student against the consensus it distils from.
        pairs = [("Consensus vs MiMo-V2.5", "Consensus (silver)", "MiMo-V2.5"),
                 ("Recall student vs support>=2 student",
                  "Student LoRA support>=1", "Student LoRA support>=2"),
                 ("Consensus vs recall student",
                  "Consensus (silver)", "Student LoRA support>=1")]
        print("\n[rescore] paired bootstrap, 2,000 resamples over the 300 gold comments\n")
        for label, a, bname in pairs:
            if a not in byname or bname not in byname:
                continue
            res = bootstrap(gold, byname[a], byname[bname])
            parts = []
            for metric, (d, lo, hi, p) in zip(("span", "typed", "full"), res):
                parts.append(f"{metric} Δ={d:+.1f} [{lo:+.1f},{hi:+.1f}] p={p:.3f}")
            print(f"  {label}\n    " + "\n    ".join(parts))


if __name__ == "__main__":
    main()
