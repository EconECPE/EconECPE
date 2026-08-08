"""Evaluate the schema fields NOT in Table IV — target_asset and intensity —
on the gold set. Both are otherwise unevaluated, and intensity feeds the market
features, so neither should go unscored.

For matched pairs (emotion-span IoU >= 0.5) it reports, per comparison:
  * target_asset: agreement accuracy + Cohen's kappa
  * intensity:    Pearson r + mean absolute error (MAE), scale 0..1

across three comparisons:
  (1) IAA        — the two gold annotators (Claude Sonnet 5 vs Gemini 3.5 Flash)
  (2) silver     — silver consensus vs adjudicated gold
  (3) student    — deployed recall student vs adjudicated gold (its intensity
                   is what feeds the weekly market panel)

Read-only. Run: python -m pipeline.gold_fields_eval
"""
from __future__ import annotations

import json
import os

from scipy import stats

import config
from .report import match_pairs
from .gold_report import cohens_kappa, _load_labels_file

GOLD_DIR = os.path.join(config.LABELS_DIR, "gold")
ADJ = os.path.join(GOLD_DIR, "adjudicated.jsonl")
ANNO_A = os.path.join(GOLD_DIR, "anthropic_claude-sonnet-5.labels.jsonl")
ANNO_B = os.path.join(GOLD_DIR, "google_gemini-3.5-flash.labels.jsonl")
SILVER = os.path.join(config.LABELS_DIR, "silver", "consensus.jsonl")
STUDENT = os.path.join("outputs", "qwen3.5-9b-silver_recall-lr1e-4",
                       "ckpt1300-merged-vlm", "gold_preds.jsonl")


def load_adj(path):
    return {json.loads(l)["comment_id"]: (json.loads(l).get("pairs") or [])
            for l in open(path, encoding="utf-8")}


def load_included(path):
    """consensus-shaped file: keep only pairs flagged included."""
    out = {}
    for l in open(path, encoding="utf-8"):
        r = json.loads(l)
        out[r["comment_id"]] = [p for p in (r.get("pairs") or []) if p.get("included")]
    return out


def field_agreement(src_a, src_b, thr=0.5):
    """src_*: {cid: [pairs]}. Match pairs on shared cids (emotion-span IoU),
    then compare target_asset and intensity on the matched pairs."""
    ids = set(src_a) & set(src_b)
    asset_a, asset_b, int_a, int_b = [], [], [], []
    for cid in ids:
        for x, y in match_pairs(src_a[cid], src_b[cid], thr=thr):
            if x.get("target_asset") is not None and y.get("target_asset") is not None:
                asset_a.append(x["target_asset"]); asset_b.append(y["target_asset"])
            if x.get("intensity") is not None and y.get("intensity") is not None:
                int_a.append(float(x["intensity"])); int_b.append(float(y["intensity"]))
    n = len(asset_a)
    acc = sum(1 for u, v in zip(asset_a, asset_b) if u == v) / n if n else None
    kap = cohens_kappa(asset_a, asset_b)
    if len(int_a) >= 2:
        r, p = stats.pearsonr(int_a, int_b)
        mae = sum(abs(u - v) for u, v in zip(int_a, int_b)) / len(int_a)
        exact = sum(1 for u, v in zip(int_a, int_b) if abs(u - v) < 1e-9) / len(int_a)
    else:
        r = p = mae = exact = None
    return dict(n_matched=n, asset_acc=acc, asset_kappa=kap,
                intensity_r=r, intensity_p=p, intensity_mae=mae,
                intensity_exact=exact, n_int=len(int_a))


def main():
    adj = load_adj(ADJ)
    a = {c: v["pairs"] for c, v in _load_labels_file(ANNO_A).items()}
    b = {c: v["pairs"] for c, v in _load_labels_file(ANNO_B).items()}
    silver = {c: v for c, v in load_included(SILVER).items() if c in adj}
    student = load_included(STUDENT)

    comps = [
        ("IAA (Sonnet vs Gemini)", a, b),
        ("silver vs gold", silver, adj),
        ("recall student vs gold", student, adj),
    ]
    print(f"{'comparison':26s} {'n_pair':>6s} | {'asset_acc':>9s} {'asset_k':>8s} | "
          f"{'int_r':>6s} {'int_MAE':>7s} {'int_exact':>9s}")
    for name, sa, sb in comps:
        r = field_agreement(sa, sb)
        print(f"{name:26s} {r['n_matched']:6d} | "
              f"{(r['asset_acc'] or 0)*100:8.1f}% {r['asset_kappa'] or 0:8.3f} | "
              f"{r['intensity_r'] or 0:6.3f} {r['intensity_mae'] or 0:7.3f} "
              f"{(r['intensity_exact'] or 0)*100:8.1f}%")


if __name__ == "__main__":
    main()
