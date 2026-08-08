"""Selection-aware inference for the euphoria-reversal OOS claim.

The concern: euphoria->equity was picked as the "strongest 1-week signal" from the
FULL-sample screen (Table VIII, 444 wks incl. the OOS years), then "confirmed" OOS.
That is circular if selection peeked at the OOS period. Here we (1) redo the
selection using ONLY in-sample 2018-2022, confirming euphoria is still the top
emotion feature for 1w equity; (2) report the clean OOS test on 2023-2026; and
(3) run a White-style reality check: a stationary block bootstrap of the max |r|
over the whole emotion family on the in-sample window, giving a selection-adjusted
p-value for the in-sample euphoria correlation.

Read-only. Run: python -m pipeline.market_selection
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
PANEL = os.path.join(config.DATA_DIR, "market", "panel_weekly.recall.csv")  # economics recall
BLOCK = 8


def main():
    panel = pd.read_csv(PANEL, index_col=0, parse_dates=True)
    rets = weekly_returns(FREQ)
    y = equity_fwd(rets, "1w").rename("y")

    # candidate emotion family: per-emotion volume + share + composite valence
    feats = [c for c in panel.columns if c.startswith("emo_") or c.startswith("share_")]
    feats += ["net_valence"]
    dpanel = panel[feats].diff()

    d = pd.concat([dpanel, y], axis=1).dropna()
    IS, OOS = d[d.index <= SPLIT], d[d.index > SPLIT]

    # ---- (1) in-sample-only selection ----
    rows = []
    for f in feats:
        r, p = stats.pearsonr(IS[f], IS["y"])
        rows.append((f, r, p))
    rank = sorted(rows, key=lambda t: t[1])          # most negative (reversal) first
    print(f"[selection] in-sample 2018..{SPLIT[:4]}  ({len(IS)} wks), {len(feats)} emotion features")
    print("  most-negative (reversal) IS correlations with 1w fwd equity:")
    for f, r, p in rank[:5]:
        print(f"    {f:28s} r={r:+.3f}  p={p:.3f}")
    top = rank[0]
    print(f"  -> top IS reversal feature: {top[0]} (r={top[1]:+.3f})")

    euph = "emo_excitement_euphoria"
    r_is = dict((f, r) for f, r, _ in rows)[euph]
    r_is_share = dict((f, r) for f, r, _ in rows).get("share_excitement_euphoria")
    print(f"  euphoria rank by most-negative r: "
          f"{[f for f,_,_ in rank].index(euph)+1} of {len(feats)} (volume); "
          f"share rank: {[f for f,_,_ in rank].index('share_excitement_euphoria')+1}")

    # ---- (2) clean OOS test (direction fixed in-sample) ----
    ro, po2 = stats.pearsonr(OOS[euph], OOS["y"])
    po1 = po2 / 2 if np.sign(ro) == np.sign(r_is) else 1 - po2 / 2
    print(f"\n[OOS] euphoria->equity 1w: IS r={r_is:+.3f} | OOS r={ro:+.3f} "
          f"one-sided p={po1:.3f} (n_oos={len(OOS)})")

    # ---- (3) White-style reality check over the family, in-sample ----
    rng = np.random.default_rng(7)
    X = IS[feats].to_numpy()
    yv = IS["y"].to_numpy()
    n = len(yv)
    Xz = (X - X.mean(0)) / X.std(0)
    obs_max = np.max(np.abs([np.corrcoef(Xz[:, k], yv)[0, 1] for k in range(len(feats))]))
    euph_absr = abs(r_is)
    nb = int(np.ceil(n / BLOCK))
    B = 5000
    ge_max = 0
    ge_euph = 0
    for _ in range(B):
        starts = rng.integers(0, n, size=nb)
        idx = np.concatenate([(np.arange(s, s + BLOCK) % n) for s in starts])[:n]
        yb = yv[idx]                                   # block-resampled returns -> null
        rs = np.array([np.corrcoef(Xz[:, k], yb)[0, 1] for k in range(len(feats))])
        m = np.max(np.abs(rs))
        ge_max += (m >= euph_absr)
        ge_euph += (abs(rs[feats.index(euph)]) >= euph_absr)
    p_rc = (ge_max + 1) / (B + 1)          # family-wise selection-adjusted
    p_raw = (ge_euph + 1) / (B + 1)        # per-feature block-bootstrap
    print(f"\n[reality check] in-sample, block bootstrap (block={BLOCK}, B={B})")
    print(f"  euphoria |r_IS|={euph_absr:.3f}; observed max|r| over family={obs_max:.3f}")
    print(f"  selection-adjusted (max-stat over {len(feats)} features) p={p_rc:.3f}")
    print(f"  per-feature block-bootstrap p={p_raw:.3f}")


if __name__ == "__main__":
    main()
