"""Construct validity by macro regime, delivering on the intro's regime
motivation. Rather than per-regime return PREDICTION (which is
underpowered and null), we test whether the extracted affect series tracks the
market's own fear gauge (VIX) within each of the macro regimes named in Section I.
A coupling that holds regime-by-regime is a positive, not another null.

For each contiguous regime window we report, on the economics recall panel:
  Delta net-valence  vs same-week Delta VIX   (composite construct validity)
  Delta fear share   vs same-week Delta VIX

Read-only. Run: python -m pipeline.market_regime
"""
from __future__ import annotations
import os
import pandas as pd
from scipy import stats

import config
from pipeline.market_signif import vix_weekly

FREQ = "W-FRI"
PANEL = os.path.join(config.DATA_DIR, "market", "panel_weekly.recall.csv")

# contiguous macro regimes aligned with Section I
REGIMES = [
    ("Trade war 2018-19",            "2018-01-01", "2019-12-31"),
    ("COVID + stimulus 2020-21H1",   "2020-01-01", "2021-06-30"),
    ("Inflation/hiking/Ukraine 21H2-22", "2021-07-01", "2022-12-31"),
    ("Banking stress + disinflation 2023", "2023-01-01", "2023-12-31"),
    ("Disinflation + easing 2024-26", "2024-01-01", "2026-12-31"),
]


def pear(x, y):
    d = pd.concat([x, y], axis=1).dropna()
    if len(d) < 12 or d.iloc[:, 0].std() == 0 or d.iloc[:, 1].std() == 0:
        return float("nan"), float("nan"), len(d)
    r, p = stats.pearsonr(d.iloc[:, 0], d.iloc[:, 1])
    return r, p, len(d)


def main():
    panel = pd.read_csv(PANEL, index_col=0, parse_dates=True)
    dvix = vix_weekly().diff().rename("dvix")
    dnv = panel["net_valence"].diff().rename("dnv")
    dfear = panel["share_fear_anxiety"].diff().rename("dfear")

    print(f"Construct validity by regime (economics recall panel), Δ vs same-week ΔVIX\n")
    print(f"  full sample: Δnet_val {tuple(round(v,3) for v in pear(dnv,dvix)[:2])}, "
          f"Δfear_share {tuple(round(v,3) for v in pear(dfear,dvix)[:2])}\n")
    print(f"  {'regime':38s} {'n':>4s} {'Δnetval~ΔVIX':>16s} {'Δfear~ΔVIX':>16s}")
    for name, a, b in REGIMES:
        sl = slice(a, b)
        r1, p1, n = pear(dnv[sl], dvix[sl])
        r2, p2, _ = pear(dfear[sl], dvix[sl])
        s1 = "*" if p1 < 0.05 else " "
        s2 = "*" if p2 < 0.05 else " "
        print(f"  {name:38s} {n:>4d}  r={r1:+.3f} p={p1:.3f}{s1}  r={r2:+.3f} p={p2:.3f}{s2}")
    print("\n  (* p<0.05; net-valence should couple NEGATIVELY, fear share POSITIVELY, in every regime)")

    # ---- pooled WITHIN-regime coupling (regime-demeaned) ----
    # removes cross-regime differences: tests whether the VIX coupling is a genuine
    # within-regime phenomenon rather than an artifact of differences between regimes.
    def regime_label(ts):
        for name, a, b in REGIMES:
            if pd.Timestamp(a) <= ts <= pd.Timestamp(b):
                return name
        return None
    df = pd.concat([dnv, dfear, dvix], axis=1).dropna()
    df["reg"] = [regime_label(t) for t in df.index]
    df = df.dropna(subset=["reg"])
    def within(col):
        x = df[col] - df.groupby("reg")[col].transform("mean")
        v = df["dvix"] - df.groupby("reg")["dvix"].transform("mean")
        return pear(x, v)
    r1, p1, n = within("dnv")
    r2, p2, _ = within("dfear")
    print(f"\n  POOLED within-regime (regime-demeaned, n={n}):")
    print(f"    Δnet_valence ~ ΔVIX : r={r1:+.3f}  p={p1:.2e}")
    print(f"    Δfear_share  ~ ΔVIX : r={r2:+.3f}  p={p2:.2e}")


if __name__ == "__main__":
    main()
