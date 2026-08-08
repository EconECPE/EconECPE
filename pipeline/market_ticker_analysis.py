"""Stage 2: cross-sectional test on the (ticker x week) emotion panel.

fwd_return[i, t->t+h] ~ emotion[i, t] with ticker AND week fixed effects.
The WEEK fixed effect absorbs the market-wide move, so the emotion coefficient
is the purely IDIOSYNCRATIC (cross-sectional) relation the aggregate index-level
design cannot identify. SEs clustered by week (contemporaneous cross-ticker
correlation). Compare ticker-FE-only (contains the market factor) vs two-way FE.

Run: python -m pipeline.market_ticker_analysis
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import config

ETF = {"SPY", "QQQ", "VTI", "VOO", "DIA", "IWM", "ARKK"}
PANEL = os.path.join(config.DATA_DIR, "market", "ticker_panel.csv")
MIN_PAIRS = 3
HORIZONS = {"0w-contemp": 0, "1w-fwd": 1, "1m-fwd": 4}   # 0 = same-week return


def weekly_fwd_returns(tickers, h_weeks):
    import yfinance as yf
    px = yf.download(sorted(tickers), start="2017-12-01", end="2026-07-31",
                     interval="1d", auto_adjust=True, progress=False)["Close"]
    wk = px.resample("W-FRI").last()
    out = {}
    for h in set(h_weeks):
        fwd = wk.pct_change() if h == 0 else wk.shift(-h) / wk - 1.0
        f = fwd.stack().rename("fwd").reset_index()
        f.columns = ["fri", "ticker", "fwd"]
        f["wk"] = f["fri"].dt.to_period("W-FRI")
        out[h] = f[["wk", "ticker", "fwd"]]
    return out


def main():
    d = pd.read_csv(PANEL, parse_dates=["week"])
    d = d[~d.ticker.isin(ETF) & (d.n_pairs >= MIN_PAIRS)].copy()
    d["wk"] = d["week"].dt.to_period("W-FRI")
    d["net_val_mean"] = d["net_val"] / d["n_pairs"]
    d["share_fear"] = d["n_fear"] / d["n_pairs"]
    tickers = sorted(d.ticker.unique())
    print(f"[xsec] {len(d)} ticker-weeks, {len(tickers)} tickers, {d.wk.nunique()} weeks "
          f"(n_pairs>={MIN_PAIRS})")

    rets = weekly_fwd_returns(tickers, HORIZONS.values())

    for hname, h in HORIZONS.items():
        m = d.merge(rets[h], on=["wk", "ticker"], how="inner").dropna(subset=["fwd"])
        # winsorize returns lightly to tame penny-stock outliers
        lo, hi = m["fwd"].quantile([0.01, 0.99])
        m["fwd"] = m["fwd"].clip(lo, hi)
        m["week_s"] = m["wk"].astype(str)
        print(f"\n===== horizon {hname} : {len(m)} obs, {m.ticker.nunique()} tickers, "
              f"{m.week_s.nunique()} weeks =====")
        for feat in ["net_val_mean", "share_fear"]:
            for label, formula in [
                ("ticker-FE only (has market factor)", f"fwd ~ {feat} + C(ticker)"),
                ("two-way FE (idiosyncratic)",          f"fwd ~ {feat} + C(ticker) + C(week_s)"),
            ]:
                res = smf.ols(formula, data=m).fit(
                    cov_type="cluster", cov_kwds={"groups": m["week_s"]})
                b, se, p = res.params[feat], res.bse[feat], res.pvalues[feat]
                print(f"  {feat:13s} | {label:36s} beta={b:+.4f}  se={se:.4f}  "
                      f"t={b/se:+.2f}  p={p:.3f}  n={int(res.nobs)}")


if __name__ == "__main__":
    main()
