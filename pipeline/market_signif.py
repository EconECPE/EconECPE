"""Defensible significance / robustness analyses for the market study, instead of
fishing the 621-cell grid. All on the full economics recall corpus, first-differenced.

  1. VIX construct validity   — contemporaneous emotion vs the market's own fear gauge
  2. Composite a-priori factor — net_valence -> next-week equity (single low-mult. test)
  3. Pre-registered flagship  — fear x inflation -> gold/crypto ; fear x geopolitics -> equities
  4. Block-bootstrap          — autocorrelation-robust p for the top 1-week signals
  5. Sample-split OOS         — form on 2018-2022, confirm on 2023-2026 (one-sided, dir. fixed IS)

Read-only; prints a report.
  python -m pipeline.market_signif                                   # economics (default)
  python -m pipeline.market_signif --labels data/labels/corpus/investing_labels.jsonl --tag investing
  python -m pipeline.market_signif --labels <econ p1> <econ p2> <investing> --tag combined
"""
from __future__ import annotations
import argparse
import os
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

import config
from pipeline.market_panel import load_pairs, build_panel, weekly_returns

FREQ = "W-FRI"
LABELS = [os.path.join(config.LABELS_DIR, "corpus", "labels.jsonl"),
          os.path.join(config.LABELS_DIR, "corpus", "labels_part2_recall.jsonl")]
SPLIT = "2023-01-01"          # in-sample <= ; out-of-sample >
EQUITY = ["SPY", "QQQ", "^GSPC", "^IXIC"]


def equity_fwd(rets, h="1w"):
    cols = [f"{t}__fwd_{h}" for t in EQUITY if f"{t}__fwd_{h}" in rets.columns]
    return rets[cols].mean(axis=1).rename(f"equity_fwd_{h}")


def vix_weekly():
    px = pd.read_parquet(os.path.join(config.DATA_DIR, "market", "prices.parquet"))
    v = px["^VIX_close"].resample(FREQ).last()
    return v.rename("vix")


def block_boot(x, y, B=4000, block=8, seed=0):
    """Paired stationary block bootstrap of Pearson r -> autocorr-robust 95% CI + two-sided p."""
    rng = np.random.default_rng(seed)
    d = pd.concat([x, y], axis=1).dropna()
    X, Y = d.iloc[:, 0].to_numpy(), d.iloc[:, 1].to_numpy()
    n = len(X)
    obs = np.corrcoef(X, Y)[0, 1]
    nb = int(np.ceil(n / block))
    rs = np.empty(B)
    for b in range(B):
        starts = rng.integers(0, n, size=nb)
        idx = np.concatenate([(np.arange(s, s + block) % n) for s in starts])[:n]
        rs[b] = np.corrcoef(X[idx], Y[idx])[0, 1]
    lo, hi = np.percentile(rs, [2.5, 97.5])
    p = 2 * min((rs <= 0).mean(), (rs >= 0).mean())
    return obs, lo, hi, max(p, 1.0 / B), n


def pear(x, y):
    d = pd.concat([x, y], axis=1).dropna()
    if len(d) < 30 or d.iloc[:, 0].std() == 0 or d.iloc[:, 1].std() == 0:
        return np.nan, np.nan, len(d)
    r, p = stats.pearsonr(d.iloc[:, 0], d.iloc[:, 1])
    return r, p, len(d)


def walk_forward(X, y, h, min_train=104, alpha=10.0):
    """Expanding-window one-step-ahead Ridge, no look-ahead: to predict week i the
    model trains only on rows whose h-week target has already realized (j+h<=i).
    Returns the out-of-sample information coefficient IC = corr(pred, realized)."""
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    d = pd.concat([X, y.rename("y")], axis=1).dropna()
    Xv, yv = d.drop(columns="y").to_numpy(), d["y"].to_numpy()
    n = len(yv)
    preds, acts = [], []
    for i in range(min_train, n - h):
        tr = slice(0, i - h + 1)
        if i - h + 1 < 30:
            continue
        sc = StandardScaler().fit(Xv[tr])
        m = Ridge(alpha=alpha).fit(sc.transform(Xv[tr]), yv[tr])
        preds.append(float(m.predict(sc.transform(Xv[i:i + 1]))[0]))
        acts.append(yv[i])
    if len(preds) < 30:
        return np.nan, len(preds)
    ic = np.corrcoef(preds, acts)[0, 1]
    return ic, len(preds)


def run_walk_forward(dpanel, rets, prices_eq_ret):
    print("\n=== Walk-forward prediction (expanding Ridge, OOS information coefficient IC) ===")
    shares = [c for c in dpanel.columns if c.startswith("share_")]
    cells = [c for c in dpanel.columns if c.startswith("cell_")]
    ar = pd.concat({f"ar{k}": prices_eq_ret.shift(k) for k in (1, 2, 3, 4)}, axis=1)
    groups = {
        "Returns-only AR": ar,
        "+ scalar sentiment": pd.concat([ar, dpanel[["net_valence"]]], axis=1),
        "+ emotion (no causes)": pd.concat([ar, dpanel[["net_valence"] + shares]], axis=1),
        "+ emotion x cause (ours)": pd.concat([ar, dpanel[["net_valence"] + shares + cells]], axis=1),
    }
    hs = {"1w": 1, "1m": 4, "3m": 13}
    print(f"  {'feature set':26s}" + "".join(f"{h:>8s}" for h in hs))
    out = {}
    for name, X in groups.items():
        row = {}
        for h, hh in hs.items():
            ic, npred = walk_forward(X, equity_fwd(rets, h), hh)
            row[h] = ic
        out[name] = row
        print(f"  {name:26s}" + "".join(f"{row[h]:+8.3f}" for h in hs))
    return out


def main():
    ap = argparse.ArgumentParser(description="Defensible market significance / robustness analyses")
    ap.add_argument("--labels", nargs="+", default=LABELS,
                    help="one or more corpus label JSONLs to pool "
                         "(default: economics recall part1 + part2)")
    ap.add_argument("--split", default=SPLIT,
                    help="sample-split date: in-sample <= split < out-of-sample")
    ap.add_argument("--tag", default="economics", help="label shown in the report header")
    args = ap.parse_args()
    split = args.split

    df = load_pairs(args.labels)
    panel = build_panel(df, FREQ)
    rets = weekly_returns(FREQ)

    # extra cell not in the default FLAGSHIP: fear x geopolitics
    g = df.set_index("date")
    fg = g[(g["emotion"] == "fear_anxiety") & (g["cause"] == "geopolitics")]["intensity"].resample(FREQ).sum()
    panel["cell_fear_anxiety__geopolitics"] = fg.reindex(panel.index).fillna(0)

    vix = vix_weekly()
    dpanel = panel.diff()
    dvix = vix.diff().rename("dvix")
    equity1w = equity_fwd(rets, "1w")

    print(f"\n[signif:{args.tag}] panel {panel.index.min().date()}..{panel.index.max().date()}  "
          f"{panel.shape[0]} weeks  ({len(df):,} pairs)\n")

    # ---- 1. VIX construct validity (contemporaneous, differenced) ----
    print("=== 1. VIX construct validity (Δemotion vs same-week ΔVIX) ===")
    for feat in ["share_fear_anxiety", "share_anger_disgust", "share_optimism_confidence",
                 "net_valence"]:
        if feat in dpanel:
            r, p, n = pear(dpanel[feat], dvix)
            print(f"  {feat:28s} vs ΔVIX : r={r:+.3f}  p={p:.2e}  n={n}")

    # ---- 2. Composite a-priori factor -> next-week equity (single test/horizon) ----
    print("\n=== 2. Composite net_valence -> forward equity (reversal ⇒ negative) ===")
    for h in ("1w", "1m", "3m"):
        r, p, n = pear(dpanel["net_valence"].diff() if False else dpanel["net_valence"],
                       equity_fwd(rets, h))
        print(f"  Δnet_valence -> equity_{h:2s} : r={r:+.3f}  p={p:.3f}  n={n}")

    # ---- 3. Pre-registered flagship family (BH within this small family only) ----
    print("\n=== 3. Pre-registered flagship (small family, BH within family) ===")
    fam = []
    def add(name, feat, tgt):
        if feat in dpanel and tgt in rets:
            r, p, n = pear(dpanel[feat], rets[tgt])
            fam.append((name, r, p, n))
    add("fear×inflation → gold(GC=F) 1w", "cell_fear_anxiety__inflation", "GC=F__fwd_1w")
    add("fear×inflation → gold(GLD) 1w",  "cell_fear_anxiety__inflation", "GLD__fwd_1w")
    add("fear×inflation → BTC 1w",        "cell_fear_anxiety__inflation", "BTC-USD__fwd_1w")
    add("fear×inflation → ETH 1w",        "cell_fear_anxiety__inflation", "ETH-USD__fwd_1w")
    add("fear×geopolitics → SPY 1w",      "cell_fear_anxiety__geopolitics", "SPY__fwd_1w")
    add("fear×geopolitics → Nasdaq 1w",   "cell_fear_anxiety__geopolitics", "^IXIC__fwd_1w")
    ps = [f[2] for f in fam]
    bh = multipletests(ps, method="fdr_bh")[1] if ps else []
    for (name, r, p, n), q in zip(fam, bh):
        flag = "  <-- q<.10" if q < 0.10 else ""
        print(f"  {name:34s} r={r:+.3f}  p={p:.3f}  q={q:.3f}  n={n}{flag}")

    # ---- 4. Block-bootstrap on the top 1-week signals ----
    print("\n=== 4. Stationary block-bootstrap (autocorr-robust) on top 1w signals ===")
    for feat, lab in [("emo_excitement_euphoria", "euphoria→equity 1w"),
                      ("share_fear_anxiety", "fear share→equity 1w")]:
        obs, lo, hi, p, n = block_boot(dpanel[feat], equity1w)
        naive_r, naive_p, _ = pear(dpanel[feat], equity1w)
        print(f"  {lab:22s} r={obs:+.3f}  95%CI[{lo:+.3f},{hi:+.3f}]  boot_p={p:.3f}  "
              f"(naive_p={naive_p:.3f})  n={n}")

    # ---- 5. Sample-split out-of-sample (direction fixed in-sample -> one-sided OOS) ----
    print("\n=== 5. Sample-split OOS (form 2018-22, test 2023-26; one-sided, dir. fixed IS) ===")
    for feat, lab in [("emo_excitement_euphoria", "euphoria→equity 1w"),
                      ("share_fear_anxiety", "fear share→equity 1w")]:
        j = pd.concat([dpanel[feat].rename("x"), equity1w.rename("y")], axis=1).dropna()
        IS = j.index <= split
        OOS = j.index > split
        ri, pi, ni = pear(j["x"][IS], j["y"][IS])
        ro, po2, no = pear(j["x"][OOS], j["y"][OOS])
        # one-sided OOS p in the in-sample direction
        po1 = po2 / 2 if np.sign(ro) == np.sign(ri) else 1 - po2 / 2
        held = "HELD" if (np.sign(ro) == np.sign(ri) and po1 < 0.05) else "not sig"
        print(f"  {lab:22s} IS r={ri:+.3f}(p={pi:.3f},n={ni}) | "
              f"OOS r={ro:+.3f} one-sided p={po1:.3f} n={no}  [{held}]")

    # ---- Walk-forward (stage ii) ----
    px = pd.read_parquet(os.path.join(config.DATA_DIR, "market", "prices.parquet"))
    eq_close = px[[f"{t}_close" for t in EQUITY if f"{t}_close" in px.columns]].resample(FREQ).last()
    eq_ret = eq_close.pct_change().mean(axis=1)
    run_walk_forward(dpanel, rets, eq_ret)
    print()


if __name__ == "__main__":
    main()
