"""Download the daily prices the market study uses. They are not redistributed.

Yahoo Finance's terms do not allow redistributing its data, so this release
ships the weekly emotion panels but not the prices they are tested against.
This script fetches the same nine series over the same window the paper used
and writes the two files the market scripts read:

    data/market/prices.parquet    read by pipeline/market_panel.py, market_signif.py, ...
    data/market/prices.csv        the same table, for inspection

Usage (from the repository root):

    pip install yfinance pandas pyarrow
    python tools/fetch_prices.py

It then checks *coverage* (trading days per series) against the paper's
snapshot. Values are not compared, because Yahoo revises its history: adjusted
closes are rescaled for dividends, and gold futures (GC=F) are corrected after
the fact. In a test download on 2026-09-23, daily returns matched the paper's
snapshot at r >= 0.99999 for eight of the nine series and at r = 0.995 for
GC=F, so results on gold futures may move slightly; the others reproduce.

Both output files are git-ignored, so a local download cannot be committed back
into the release by accident.
"""
from __future__ import annotations

import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

# The paper's window: 2018-01-01 through 2026-07-11 inclusive (yfinance's `end`
# is exclusive), and the number of non-empty daily closes per series in the
# snapshot the published results were computed on.
START, END = "2018-01-01", "2026-07-12"
EXPECTED_ROWS = 3114
EXPECTED_CLOSES = {
    "BTC-USD": 3114, "ETH-USD": 3114, "GC=F": 2142, "GLD": 2141, "QQQ": 2141,
    "SPY": 2141, "^GSPC": 2141, "^IXIC": 2141, "^VIX": 2142,
}


def coverage(pd, out_dir):
    df = pd.read_parquet(os.path.join(out_dir, "prices.parquet"))
    return df, {c[: -len("_close")]: int(df[c].notna().sum())
                for c in df.columns if c.endswith("_close")}


def main() -> int:
    try:
        import pandas as pd
        import yfinance  # noqa: F401  (imported by pipeline.market_data)
    except ImportError:
        print("needs: pip install yfinance pandas pyarrow", file=sys.stderr)
        return 2

    from pipeline import market_data

    out_dir = os.path.join(ROOT, "data", "market")
    # Yahoo's batch endpoint intermittently returns one ticker empty. The same
    # code path as the paper (pipeline.market_data) is simply re-run until every
    # series has data, so the table is built exactly as it was for the paper.
    for attempt in range(1, 6):
        sys.argv = ["market_data", "--start", START, "--end", END, "--out-dir", out_dir]
        market_data.main()
        df, got = coverage(pd, out_dir)
        empty = [t for t in EXPECTED_CLOSES if got.get(t, 0) == 0]
        if not empty:
            break
        print(f"[fetch] attempt {attempt}: no data for {', '.join(empty)}; retrying in "
              f"{10 * attempt}s")
        time.sleep(10 * attempt)
    else:
        print(f"\nYahoo returned no data for {', '.join(empty)} after 5 attempts; "
              f"try again later.", file=sys.stderr)
        return 1

    # A day or two of difference is Yahoo revising its history; more than that
    # (or a missing series) means the download is not the paper's data.
    diffs = {t: got.get(t, 0) - n for t, n in EXPECTED_CLOSES.items() if got.get(t, 0) != n}
    if len(df) != EXPECTED_ROWS:
        diffs["(trading days)"] = len(df) - EXPECTED_ROWS
    if not diffs:
        print(f"\ncoverage matches the paper's snapshot ({EXPECTED_ROWS} days, 9 series)")
        return 0
    print("\ncoverage differs from the paper's snapshot:")
    for t, d in diffs.items():
        print(f"  {t}: {d:+d} days")
    if max(abs(d) for d in diffs.values()) > 5:
        print("That is more than Yahoo's usual revisions; results may not reproduce.")
        return 1
    print("Within Yahoo's usual history revisions (see the note at the top of this file).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
