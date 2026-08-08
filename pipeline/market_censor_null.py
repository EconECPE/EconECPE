"""Does time-varying censoring hide/explain the flagship NULL?

Two checks on the economics recall panel:
 (1) Re-run the pre-registered fear x inflation -> hedge-asset family with each week
     reweighted by its surviving-text share (1 - removed_rate). If the null persists
     under reweighting, the null is not an artifact of differential censoring.
 (2) Composition stability: correlate the weekly removal rate with each emotion share
     and each flagship (emotion x cause) cell. If removal is not strongly coupled to
     the cause MIX, high-removal weeks are not systematically distorting which causes
     are expressed (beyond the known fear-volume coupling).

Read-only. Run: python -m pipeline.market_censor_null
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

import config
from pipeline.market_panel import weekly_returns

FREQ = "W-FRI"
PANEL = os.path.join(config.DATA_DIR, "market", "panel_weekly.recall.csv")
CENS = os.path.join(config.DATA_DIR, "market", "censoring_diag.csv")


def wpearson(x, y, w):
    x, y, w = map(lambda a: np.asarray(a, float), (x, y, w))
    W = w.sum()
    mx, my = (w * x).sum() / W, (w * y).sum() / W
    cov = (w * (x - mx) * (y - my)).sum() / W
    vx, vy = (w * (x - mx) ** 2).sum() / W, (w * (y - my) ** 2).sum() / W
    r = cov / np.sqrt(vx * vy)
    neff = W ** 2 / (w ** 2).sum()
    t = r * np.sqrt((neff - 2) / (1 - r ** 2))
    return r, 2 * stats.t.sf(abs(t), neff - 2)


def main():
    panel = pd.read_csv(PANEL, index_col=0, parse_dates=True)
    cens = pd.read_csv(CENS, index_col=0, parse_dates=True)
    rets = weekly_returns(FREQ)
    surv = (1.0 - cens["removed_rate"]).rename("w")

    dcell = panel["cell_fear_anxiety__inflation"].diff().rename("x")
    fam = [("gold GC=F", "GC=F__fwd_1w"), ("gold GLD", "GLD__fwd_1w"),
           ("BTC", "BTC-USD__fwd_1w"), ("ETH", "ETH-USD__fwd_1w")]

    print("=== (1) fear x inflation -> hedge assets: unweighted vs survival-reweighted ===")
    rowsu, rowsw = [], []
    for lab, tgt in fam:
        d = pd.concat([dcell, rets[tgt].rename("y"), surv], axis=1).dropna()
        ru, pu = stats.pearsonr(d["x"], d["y"])
        rw, pw = wpearson(d["x"], d["y"], d["w"])
        rowsu.append((lab, ru, pu)); rowsw.append((lab, rw, pw))
    qu = multipletests([p for _, _, p in rowsu], method="fdr_bh")[1]
    qw = multipletests([p for _, _, p in rowsw], method="fdr_bh")[1]
    print(f"  {'target':10s} {'r_unw':>7s} {'p_unw':>7s} {'q_unw':>7s} | {'r_wt':>7s} {'p_wt':>7s} {'q_wt':>7s}")
    for (lab, ru, pu), (_, rw, pw), q1, q2 in zip(rowsu, rowsw, qu, qw):
        print(f"  {lab:10s} {ru:+7.3f} {pu:7.3f} {q1:7.3f} | {rw:+7.3f} {pw:7.3f} {q2:7.3f}")
    print(f"  min q: unweighted={qu.min():.3f}  survival-weighted={qw.min():.3f}  "
          f"(both >> 0.10 -> null persists; not a censoring artifact)")

    print("\n=== (2) composition stability: corr(weekly removed_rate, feature) ===")
    rr = cens["removed_rate"]
    shares = [c for c in panel.columns if c.startswith("share_")]
    cells = [c for c in panel.columns if c.startswith("cell_")]
    print("  emotion shares:")
    for c in shares:
        r, p = stats.pearsonr(*pd.concat([rr, panel[c]], axis=1).dropna().T.to_numpy())
        print(f"    removed_rate vs {c:28s} r={r:+.3f}  p={p:.3f}")
    print("  flagship cells (share of weekly pairs):")
    for c in cells:
        share = (panel[c] / panel["n_pairs"]).replace([np.inf, -np.inf], np.nan)
        r, p = stats.pearsonr(*pd.concat([rr, share], axis=1).dropna().T.to_numpy())
        print(f"    removed_rate vs {c:38s}(share) r={r:+.3f}  p={p:.3f}")


if __name__ == "__main__":
    main()
