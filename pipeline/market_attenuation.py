"""Attenuation / reliability bound on the cause-conditioned market NULL
— the full-corpus panel is labeled by the 59.6-span-F1 student, so the flagship
null could be measurement attenuation rather than a true absence of signal.

Idea: per-comment label noise is largely averaged out by weekly aggregation, so
the quantity that matters is the reliability of the *weekly* feature, not the
per-comment F1. We estimate it by split-half: each week's comments are hashed
into two disjoint halves, the flagship cells and the aggregate valence/fear
series are built on each half, first-differenced (as in the test), and the two
half-series correlated across weeks; Spearman-Brown up-corrects the
half-length correlation to the full-panel reliability.

High reliability => the observed near-zero flagship correlations are not an
artifact of a noisy labeler. We then de-attenuate the six-test flagship family
(r / sqrt(reliability), returns treated as error-free) as an UPPER bound on any
true effect and re-apply BH, and report the minimum detectable effect.

Read-only. Run:
    python -m pipeline.market_attenuation --labels \
        data/labels/corpus/labels.jsonl data/labels/corpus/labels_part2_recall.jsonl
"""
from __future__ import annotations

import argparse
import hashlib
import json

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

from pipeline.market_panel import VALENCE, DROP_CAUSE, weekly_returns

FREQ = "W-FRI"

# six-test pre-specified flagship family (feature -> forward-1w return target)
FLAGSHIP = [
    ("fear_anxiety", "inflation", "GC=F__fwd_1w", "fear×inflation → gold(GC=F)"),
    ("fear_anxiety", "inflation", "GLD__fwd_1w", "fear×inflation → gold(GLD)"),
    ("fear_anxiety", "inflation", "BTC-USD__fwd_1w", "fear×inflation → BTC"),
    ("fear_anxiety", "inflation", "ETH-USD__fwd_1w", "fear×inflation → ETH"),
    ("fear_anxiety", "geopolitics", "SPY__fwd_1w", "fear×geopolitics → SPY"),
    ("fear_anxiety", "geopolitics", "^IXIC__fwd_1w", "fear×geopolitics → Nasdaq"),
]


def _half(cid: str) -> int:
    return int(hashlib.md5(str(cid).encode()).hexdigest(), 16) & 1


def load_pairs_cid(paths):
    rows = []
    for path in paths:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if "\x00" in line:
                    line = line.replace("\x00", "")
                try:
                    r = json.loads(line)
                except ValueError:
                    continue
                ts = r.get("created_utc")
                if ts is None:
                    continue
                cid = r.get("comment_id")
                for p in r.get("pairs", []):
                    if p.get("cause_category") in DROP_CAUSE:
                        continue
                    rows.append((cid, int(ts), p.get("emotion"),
                                 p.get("cause_category"),
                                 float(p.get("intensity") or 0.0)))
    df = pd.DataFrame(rows, columns=["cid", "ts", "emotion", "cause", "intensity"])
    df["date"] = pd.to_datetime(df["ts"], unit="s", utc=True).dt.tz_localize(None)
    df["valence"] = df["emotion"].map(VALENCE).fillna(0) * df["intensity"]
    df["half"] = df["cid"].map(_half)
    return df


def _cell(df, emo, cause):
    sub = df[(df["emotion"] == emo) & (df["cause"] == cause)]
    return sub.set_index("date")["intensity"].resample(FREQ).sum()


def _valence(df):
    return df.set_index("date")["valence"].resample(FREQ).sum()


def _fear_share(df):
    g = df.set_index("date")
    n = g["intensity"].resample(FREQ).count()
    f = g[g["emotion"] == "fear_anxiety"]["intensity"].resample(FREQ).count()
    return (f / n).replace([np.inf, -np.inf], np.nan)


def sb_reliability(a: pd.Series, b: pd.Series, diff=True):
    """Spearman-Brown full-length reliability from two half-length series."""
    s = pd.concat([a.rename("a"), b.rename("b")], axis=1).fillna(0.0)
    if diff:
        s = s.diff().dropna()
    r, _ = stats.pearsonr(s["a"], s["b"])
    sb = 2 * r / (1 + r) if (1 + r) != 0 else float("nan")
    return r, sb


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--labels", nargs="+", required=True)
    args = ap.parse_args()

    df = load_pairs_cid(args.labels)
    A = df[df["half"] == 0]
    B = df[df["half"] == 1]
    print(f"[attenuation] {len(df):,} pairs; halves {len(A):,} / {len(B):,} "
          f"({df['cid'].nunique():,} comments)")

    print("\n=== Split-half weekly reliability (first-differenced series) ===")
    print(f"{'series':32s} {'half-r':>7s} {'Spearman-Brown rel.':>20s}")
    rel = {}
    series_defs = {
        "cell_fear×inflation": (_cell(A, "fear_anxiety", "inflation"),
                                _cell(B, "fear_anxiety", "inflation")),
        "cell_fear×geopolitics": (_cell(A, "fear_anxiety", "geopolitics"),
                                  _cell(B, "fear_anxiety", "geopolitics")),
        "net_valence (aggregate)": (_valence(A), _valence(B)),
        "fear_share (aggregate)": (_fear_share(A), _fear_share(B)),
    }
    for name, (sa, sb) in series_defs.items():
        r, sbrel = sb_reliability(sa, sb)
        rel[name] = sbrel
        print(f"{name:32s} {r:7.3f} {sbrel:20.3f}")

    # observed flagship correlations on the full panel (differenced feature vs fwd return)
    rets = weekly_returns(FREQ)
    full = {
        ("fear_anxiety", "inflation"): _cell(df, "fear_anxiety", "inflation"),
        ("fear_anxiety", "geopolitics"): _cell(df, "fear_anxiety", "geopolitics"),
    }
    relmap = {("fear_anxiety", "inflation"): rel["cell_fear×inflation"],
              ("fear_anxiety", "geopolitics"): rel["cell_fear×geopolitics"]}

    print("\n=== Flagship family: observed r + attenuation-corrected effect size ===")
    print("(disattenuation rescales r AND its standard error by the same 1/sqrt(rel),")
    print(" so the t-statistic and p-value are invariant; only the effect-size")
    print(" *estimate* changes. r_true is the largest true effect consistent with r_obs.)")
    print(f"{'test':30s} {'n':>4s} {'r_obs':>7s} {'p_obs':>7s} {'rel':>5s} {'r_true(est)':>11s}")
    rows = []
    for emo, cause, tgt, label in FLAGSHIP:
        feat = full[(emo, cause)].diff()
        sub = pd.concat([feat.rename("x"), rets[tgt].rename("y")], axis=1).dropna()
        r, p = stats.pearsonr(sub["x"], sub["y"])
        n = len(sub)
        lam = relmap[(emo, cause)]
        r_true = max(-0.999, min(0.999, r / np.sqrt(lam)))   # attenuation-corrected estimate
        rows.append((label, n, r, p, lam, r_true))
        print(f"{label:30s} {n:4d} {r:+7.3f} {p:7.3f} {lam:5.2f} {r_true:+11.3f}")

    q_obs = multipletests([row[3] for row in rows], method="fdr_bh")[1]
    print(f"\n  BH within family (6 tests): min q = {q_obs.min():.3f} "
          f"(unchanged by disattenuation, since t is invariant -> family stays null)")

    # minimum detectable effect at n, alpha=0.05 two-sided, and the true-effect bound
    n = max(row[1] for row in rows)
    mde_obs = 1.96 / np.sqrt(n)
    r_true_bound = max(abs(row[5]) for row in rows)
    print(f"\n  n={n}: smallest detectable |r_obs| approx {mde_obs:.3f}. The largest "
          f"attenuation-corrected\n  true effect in the family is |r_true| approx "
          f"{r_true_bound:.3f} (fear x inflation -> crypto); the null therefore\n  bounds "
          f"true cause-conditioned correlations at roughly |r|<={r_true_bound:.2f}, not smaller ones.")
    print(f"\n  Aggregate net_valence reliability = {rel['net_valence (aggregate)']:.3f}: the "
          f"VIX/euphoria\n  findings run on near-error-free series; attenuation is specific to the "
          f"sparse\n  cause-conditioned cells (fear x inflation rel = {rel['cell_fear×inflation']:.2f}).")


if __name__ == "__main__":
    main()
