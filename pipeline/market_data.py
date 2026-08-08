"""Phase 4 — market data puller. Fetches the ticker set (2018->now) via
yfinance and writes tidy daily price/return tables for the market study.

    ^GSPC ^IXIC SPY QQQ GC=F GLD BTC-USD ETH-USD ^VIX

Output: data/market/prices.parquet  (index=date, columns = <ticker>_close, <ticker>_ret1d)
and data/market/prices.csv for eyeballing. Forward returns are computed later in
market_panel.py against the aggregated sentiment panel (keeps horizons in one place).
"""
from __future__ import annotations

import argparse
import os

import config

TICKERS = ["^GSPC", "^IXIC", "SPY", "QQQ", "GC=F", "GLD", "BTC-USD", "ETH-USD", "^VIX"]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--start", default="2018-01-01")
    ap.add_argument("--end", default=None, help="default: today")
    ap.add_argument("--out-dir", default=os.path.join(config.DATA_DIR, "market"))
    args = ap.parse_args()

    import yfinance as yf
    import pandas as pd

    os.makedirs(args.out_dir, exist_ok=True)
    raw = yf.download(TICKERS, start=args.start, end=args.end, progress=False, auto_adjust=True)
    close = raw["Close"].copy()
    close.columns = [f"{c}_close" for c in close.columns]
    ret = raw["Close"].pct_change()
    ret.columns = [f"{c}_ret1d" for c in ret.columns]
    df = close.join(ret)
    df.index.name = "date"

    pq = os.path.join(args.out_dir, "prices.parquet")
    df.to_parquet(pq)
    df.to_csv(os.path.join(args.out_dir, "prices.csv"))
    n_by = {c.replace("_close", ""): int(close[c].notna().sum()) for c in close.columns}
    print(f"[market] {df.shape[0]} trading days {df.index.min().date()}..{df.index.max().date()}")
    print(f"[market] non-null closes per ticker: {n_by}")
    print(f"[market] -> {pq}")


if __name__ == "__main__":
    main()
