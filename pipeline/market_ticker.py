"""Ticker-level cross-section: does emotion directed at a specific company predict
THAT company's forward return, after removing the market-wide move (week fixed
effect)? This is the cross-sectional identification the aggregate index-level
design lacks. r/investing only (cashtags/company names are essentially absent in
r/economics).

Stage 1 (this file, `--extract`): stream investing labels JOIN comments, attribute
each comment's emotion pairs to the single ticker it mentions (cashtag or company
name), and build a weekly (ticker x week) emotion panel ->
data/market/ticker_panel.csv.

Env: ECONECPE_TICKER_LIMIT caps comments scanned (validation).
"""
from __future__ import annotations
import json, os, re, sys
from collections import defaultdict
import config

VALENCE = {"optimism_confidence": 1, "excitement_euphoria": 1, "fear_anxiety": -1,
           "pessimism_despair": -1, "anger_disgust": -1,
           "confusion_uncertainty": 0, "surprise": 0}

# ticker -> company-name aliases (word-boundary, case-insensitive). Cashtag ($TICK)
# is always matched. Single-letter / verb-like ambiguous names are omitted; a few
# common-but-noisy names (apple, google, meta) are kept and flagged in robustness.
TICKERS = {
    "AAPL": ["apple"], "MSFT": ["microsoft"], "GOOGL": ["alphabet", "google"],
    "AMZN": ["amazon"], "NVDA": ["nvidia"], "META": ["facebook", "meta"],
    "TSLA": ["tesla"], "NFLX": ["netflix"], "AMD": ["amd"], "INTC": ["intel"],
    "CRM": ["salesforce"], "ORCL": ["oracle"], "ADBE": ["adobe"], "CSCO": ["cisco"],
    "QCOM": ["qualcomm"], "AVGO": ["broadcom"], "IBM": ["ibm"], "PYPL": ["paypal"],
    "SHOP": ["shopify"], "UBER": ["uber"], "ABNB": ["airbnb"], "COIN": ["coinbase"],
    "HOOD": ["robinhood"], "PLTR": ["palantir"], "SNOW": ["snowflake"],
    "ROKU": ["roku"], "DIS": ["disney"], "BA": ["boeing"], "NIO": ["nio"],
    "RIVN": ["rivian"], "LCID": ["lucid motors"], "GME": ["gamestop"], "AMC": ["amc"],
    "BABA": ["alibaba"], "JPM": ["jpmorgan", "jp morgan"], "GS": ["goldman sachs"],
    "KO": ["coca-cola", "coca cola"], "PEP": ["pepsi"], "WMT": ["walmart"],
    "COST": ["costco"], "MCD": ["mcdonald"], "SBUX": ["starbucks"], "NKE": ["nike"],
    "XOM": ["exxon"], "CVX": ["chevron"], "PFE": ["pfizer"], "MRNA": ["moderna"],
    "F": ["ford motor"], "GM": ["general motors"], "BAC": ["bank of america"],
    "SPY": ["s&p 500", "s&p500", "sp500", "the s&p"], "QQQ": ["nasdaq 100"],
    "VTI": ["total stock market"], "ARKK": ["ark innovation", "arkk"],
}
NAME2TICK = {a.lower(): t for t, al in TICKERS.items() for a in al}
NAME_RE = re.compile("|".join(rf"\b{re.escape(a)}\b" for a in NAME2TICK), re.I)
CASH_RE = re.compile(r"\$([A-Z]{1,5})\b")
UNIVERSE = set(TICKERS)


def tickers_in(body: str):
    hits = set()
    for m in NAME_RE.finditer(body):
        hits.add(NAME2TICK[m.group(0).lower().strip()])
    for c in CASH_RE.findall(body):
        if c in UNIVERSE:
            hits.add(c)
    return hits


def main():
    import pandas as pd
    lab_path = os.path.join(config.LABELS_DIR, "corpus", "investing_labels.jsonl")
    com_path = os.path.join(config.DATA_DIR, "investing_comments.jsonl")
    limit = int(os.environ.get("ECONECPE_TICKER_LIMIT", "0")) or None

    # pass 1: per-comment emotion aggregate {cid: (ts, net_val, n_fear, n_pairs)}
    lab = {}
    for line in open(lab_path):
        r = json.loads(line)
        pairs = r.get("pairs") or []
        if not pairs:
            continue
        nv = sum(VALENCE.get(p.get("emotion"), 0) * float(p.get("intensity") or 0) for p in pairs)
        nf = sum(1 for p in pairs if p.get("emotion") == "fear_anxiety")
        lab[r["comment_id"]] = (int(r["created_utc"]), nv, nf, len(pairs))
    print(f"[ticker] loaded {len(lab):,} emotive investing comments", file=sys.stderr)

    # pass 2: stream comments, attribute single-ticker comments
    cell = defaultdict(lambda: [0.0, 0, 0, 0])   # (week,tick) -> [nv, nfear, npairs, ncomments]
    from collections import Counter
    tick_ct = Counter()
    n = matched = 0
    for i, line in enumerate(open(com_path)):
        if limit and i >= limit:
            break
        try:
            r = json.loads(line)
        except ValueError:
            continue
        cid = r.get("id") or r.get("comment_id")
        rec = lab.get(cid)
        if rec is None:
            continue
        n += 1
        body = r.get("body") or ""
        ticks = tickers_in(body)
        if len(ticks) != 1:
            continue
        t = next(iter(ticks))
        tick_ct[t] += 1
        matched += 1
        ts, nv, nf, npr = rec
        wk = pd.Timestamp(ts, unit="s").to_period("W-FRI").start_time.normalize()
        c = cell[(wk, t)]
        c[0] += nv; c[1] += nf; c[2] += npr; c[3] += 1

    print(f"[ticker] emotive comments seen {n:,}; single-ticker attributed {matched:,} "
          f"({100*matched/max(n,1):.1f}%)", file=sys.stderr)
    print(f"[ticker] top tickers: {tick_ct.most_common(20)}", file=sys.stderr)
    rows = [{"week": wk, "ticker": t, "net_val": v[0], "n_fear": v[1],
             "n_pairs": v[2], "n_comments": v[3]} for (wk, t), v in cell.items()]
    df = pd.DataFrame(rows).sort_values(["ticker", "week"])
    out = os.path.join(config.DATA_DIR, "market", "ticker_panel.csv")
    df.to_csv(out, index=False)
    print(f"[ticker] wrote {out}: {len(df)} ticker-weeks, {df['ticker'].nunique()} tickers", file=sys.stderr)


if __name__ == "__main__":
    main()
