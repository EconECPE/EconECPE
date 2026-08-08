"""Robustness of the cause-free euphoria-reversal OOS signal to time-varying
censoring (Section 8). The weekly [removed]/[deleted] rate co-moves with market
fear, differentially thinning the emotion panel in high-fear weeks. We re-test
the out-of-sample euphoria->equity reversal (a) reweighting each week by its
surviving-text share (1 - removed_rate), so heavily censored weeks contribute
less, and (b) dropping calendar-year 2022 (the inflation-shock peak of both fear
and moderation), and (c) both together. Direction is fixed in-sample (<=2023);
the OOS (>2023) p-value is one-sided in that direction, matching market_signif.

Read-only. Run: python -m pipeline.market_censor_robust
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd
from scipy import stats

import config
from pipeline.market_panel import weekly_returns
from pipeline.market_signif import equity_fwd

FREQ = "W-FRI"
SPLIT = "2023-01-01"
PANEL = os.path.join(config.DATA_DIR, "market", "panel_weekly.recall.csv")   # economics recall
CENS = os.path.join(config.DATA_DIR, "market", "censoring_diag.csv")


def wpearson(x, y, w):
    """Weighted Pearson r + two-sided p via Kish effective-n t-approx."""
    x, y, w = map(lambda a: np.asarray(a, float), (x, y, w))
    W = w.sum()
    mx, my = (w * x).sum() / W, (w * y).sum() / W
    cov = (w * (x - mx) * (y - my)).sum() / W
    vx, vy = (w * (x - mx) ** 2).sum() / W, (w * (y - my) ** 2).sum() / W
    r = cov / np.sqrt(vx * vy)
    neff = W ** 2 / (w ** 2).sum()
    t = r * np.sqrt((neff - 2) / (1 - r ** 2))
    return r, 2 * stats.t.sf(abs(t), neff - 2), neff


def oos(d, weighted):
    IS, OOS = d[d.index <= SPLIT], d[d.index > SPLIT]
    if weighted:
        ri = wpearson(IS.x, IS.y, IS.w)[0]
        ro, po2, no = wpearson(OOS.x, OOS.y, OOS.w)
    else:
        ri = stats.pearsonr(IS.x, IS.y)[0]
        ro, po2 = stats.pearsonr(OOS.x, OOS.y)
        no = len(OOS)
    same = np.sign(ro) == np.sign(ri)
    po1 = po2 / 2 if same else 1 - po2 / 2         # one-sided in the IS direction
    return ri, ro, po1, no, (same and po1 < 0.05)


def main():
    panel = pd.read_csv(PANEL, index_col=0, parse_dates=True)
    cens = pd.read_csv(CENS, index_col=0, parse_dates=True)
    rets = weekly_returns(FREQ)

    x = panel["emo_excitement_euphoria"].diff().rename("x")   # Δ euphoria volume
    y = equity_fwd(rets, "1w").rename("y")                     # fwd 1w mean equity
    w = (1.0 - cens["removed_rate"]).rename("w")               # surviving-text share
    d = pd.concat([x, y, w], axis=1).dropna()
    d["year"] = d.index.year

    print(f"\n[censor-robust] euphoria->equity 1w reversal, {len(d)} weeks "
          f"({d.index.min().date()}..{d.index.max().date()}), split {SPLIT}")
    print(f"  weekly removed_rate: mean {cens['removed_rate'].mean():.3f}, "
          f"2022 mean {cens.loc[cens.index.year==2022,'removed_rate'].mean():.3f}\n")

    variants = [
        ("baseline (reproduce paper)", d, False),
        ("reweighted by surviving share", d, True),
        ("drop 2022", d[d.year != 2022], False),
        ("reweighted + drop 2022", d[d.year != 2022], True),
    ]
    print(f"  {'variant':32s} {'IS r':>7s} {'OOS r':>7s} {'1-sided p':>10s} {'n_oos':>7s}  holds?")
    for name, dd, wt in variants:
        ri, ro, p, no, held = oos(dd, wt)
        print(f"  {name:32s} {ri:+7.3f} {ro:+7.3f} {p:>10.3f} {no:>7.0f}  "
              f"{'HELD' if held else 'not sig'}")
    print()


if __name__ == "__main__":
    main()
