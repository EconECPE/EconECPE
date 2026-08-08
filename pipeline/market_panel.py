"""Phase 4 — aggregate labels into a weekly sentiment panel, align with forward
returns, and test whether cause-conditioned emotion leads market moves.

Pipeline:
  labels JSONL (comment_id, created_utc, neutral, pairs[])   [pipeline.label_corpus]
    -> explode pairs, drop `interpersonal` (filtered from market aggregation)
    -> weekly features: overall net-valence/volume/intensity, per-emotion volume,
       per-cause volume, and the flagship (emotion x cause) cells (e.g. fear x inflation)
    -> join weekly forward returns (1w/1m/3m) for each ticker  [data/market/prices.parquet]
    -> lead-lag Pearson + Granger causality, Benjamini-Hochberg across all tests
Outputs: data/market/panel_weekly.csv, data/market/analysis.csv, and a printed report.

Headline hypothesis: *cause-conditioned* emotion (e.g. fear x inflation)
beats scalar sentiment at predicting forward returns.

    python -m pipeline.market_panel --labels data/labels/corpus/labels_s20.jsonl
"""
from __future__ import annotations

import argparse
import json
import os

import config

VALENCE = {  # net-valence sign per emotion
    "optimism_confidence": 1, "excitement_euphoria": 1,
    "fear_anxiety": -1, "pessimism_despair": -1, "anger_disgust": -1,
    "confusion_uncertainty": 0, "surprise": 0,
}
DROP_CAUSE = {"interpersonal"}          # filtered from market aggregation
# flagship cause-conditioned cells to test explicitly
FLAGSHIP = [("fear_anxiety", "inflation"), ("fear_anxiety", "monetary_policy"),
            ("optimism_confidence", "growth_gdp"), ("fear_anxiety", "markets_themselves"),
            ("anger_disgust", "fiscal_taxes"), ("optimism_confidence", "corporate_earnings")]
HORIZONS = {"1w": 1, "1m": 4, "3m": 13}  # in weeks


def load_pairs(paths):
    """Load one or more corpus label files into a flat pair-level frame.

    Accepts a single path or a list — multiple recall shards (e.g. part1 +
    part2) are concatenated in memory, never on disk, so the canonical
    append-only JSONLs are left untouched. Tolerant of a partial trailing line
    (part2 may be actively appended by the running labeler) and of stray NUL
    bytes seen in the raw corpus, so a live/dirty file can't abort the build.
    """
    import pandas as pd
    if isinstance(paths, str):
        paths = [paths]
    rows, n_read, n_bad = [], 0, 0
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
                    n_bad += 1           # partial trailing line / corrupt record
                    continue
                n_read += 1
                ts = r.get("created_utc")
                if ts is None:
                    continue
                for p in r.get("pairs", []):
                    if p.get("cause_category") in DROP_CAUSE:
                        continue
                    rows.append((int(ts), p.get("emotion"), p.get("cause_category"),
                                 float(p.get("intensity") or 0.0), bool(p.get("sarcasm"))))
        print(f"[panel] loaded {os.path.basename(path)}")
    if n_bad:
        print(f"[panel] skipped {n_bad} unparseable line(s) (live/partial tail is expected)")
    # `sarcasm` is carried through unused by the main panel (valence takes the
    # emotion at face value); pipeline/market_sarcasm.py is what acts on it.
    df = pd.DataFrame(rows, columns=["ts", "emotion", "cause", "intensity", "sarcasm"])
    df["date"] = pd.to_datetime(df["ts"], unit="s", utc=True).dt.tz_localize(None)
    df["valence"] = df["emotion"].map(VALENCE).fillna(0) * df["intensity"]
    return df


def build_panel(df, freq="W-FRI"):
    import pandas as pd
    g = df.set_index("date")
    wk = lambda s: s.resample(freq)
    panel = pd.DataFrame(index=wk(g["intensity"]).mean().index)
    panel["n_pairs"] = wk(g["intensity"]).count()
    panel["net_valence"] = wk(g["valence"]).sum()          # scalar sentiment baseline
    panel["net_valence_mean"] = wk(g["valence"]).mean()
    panel["mean_intensity"] = wk(g["intensity"]).mean()
    # per-emotion volume AND activity-normalized share (share is stationary-ish;
    # raw counts track total posting activity, which trends over the years and
    # produces spurious Granger hits — see pilot analysis).
    for emo in VALENCE:
        cnt = g[g["emotion"] == emo]["intensity"].resample(freq).count()
        panel[f"emo_{emo}"] = cnt
        panel[f"share_{emo}"] = cnt / panel["n_pairs"]
    # flagship cause-conditioned cells (volume + signed intensity)
    for emo, cause in FLAGSHIP:
        sub = g[(g["emotion"] == emo) & (g["cause"] == cause)]
        panel[f"cell_{emo}__{cause}"] = sub["intensity"].resample(freq).sum()
    return panel.fillna(0)


def weekly_returns(freq="W-FRI"):
    import pandas as pd
    px = pd.read_parquet(os.path.join(config.DATA_DIR, "market", "prices.parquet"))
    close = px[[c for c in px.columns if c.endswith("_close")]].copy()
    close.columns = [c.replace("_close", "") for c in close.columns]
    wk = close.resample(freq).last()
    out = {}
    for tk in wk.columns:
        s = wk[tk]
        for name, h in HORIZONS.items():
            out[f"{tk}__fwd_{name}"] = s.shift(-h) / s - 1.0
    return pd.DataFrame(out, index=wk.index)


def analyze(panel, rets, maxlag=4, min_weeks=60, diff=False):
    """diff=True first-differences each feature (Δ) to enforce stationarity before
    testing — kills shared-trend / activity-growth confounds behind spurious Granger."""
    import pandas as pd, numpy as np
    from scipy import stats
    from statsmodels.tsa.stattools import grangercausalitytests
    from statsmodels.stats.multitest import multipletests

    feats = [c for c in panel.columns if c not in ("n_pairs",)]
    panel = panel.diff().dropna() if diff else panel
    joined = panel.join(rets, how="inner")
    results = []
    for f in feats:
        x = joined[f]
        if x.std() == 0:
            continue
        for r in rets.columns:
            sub = joined[[f, r]].dropna()
            if len(sub) < min_weeks or sub[f].std() == 0 or sub[r].std() == 0:
                continue
            pr, pp = stats.pearsonr(sub[f], sub[r])
            gp = np.nan
            try:  # Granger: does feature help predict the return? (min p over lags)
                gt = grangercausalitytests(sub[[r, f]], maxlag=maxlag, verbose=False)
                gp = min(gt[l][0]["ssr_ftest"][1] for l in gt)
            except Exception:
                pass
            results.append({"feature": f, "target": r, "n": len(sub),
                            "pearson_r": pr, "pearson_p": pp, "granger_p_min": gp})
    res = pd.DataFrame(results)
    if not len(res):
        return res
    res["pearson_p_bh"] = multipletests(res["pearson_p"], method="fdr_bh")[1]
    gmask = res["granger_p_min"].notna()
    res.loc[gmask, "granger_p_bh"] = multipletests(res.loc[gmask, "granger_p_min"], method="fdr_bh")[1]
    return res.sort_values("pearson_p")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--labels", nargs="+",
                    default=[os.path.join(config.LABELS_DIR, "corpus", "labels_s20.jsonl")],
                    help="one or more corpus label JSONLs; multiple are pooled (e.g. recall "
                         "part1 + part2)")
    ap.add_argument("--freq", default="W-FRI")
    ap.add_argument("--maxlag", type=int, default=4)
    ap.add_argument("--tag", default="",
                    help="suffix for output files, e.g. --tag recall -> panel_weekly.recall.csv "
                         "(keeps the previous draft's csvs intact for comparison)")
    ap.add_argument("--diff", action="store_true",
                    help="first-difference features (Δ) to enforce stationarity — the honest "
                         "test; removes shared-trend/activity confounds")
    args = ap.parse_args()

    import pandas as pd
    suffix = f".{args.tag}" if args.tag else ""
    df = load_pairs(args.labels)
    print(f"[panel] {len(df):,} pairs, {df['date'].min().date()}..{df['date'].max().date()}")
    panel = build_panel(df, args.freq)
    rets = weekly_returns(args.freq)
    out_dir = os.path.join(config.DATA_DIR, "market")
    panel.to_csv(os.path.join(out_dir, f"panel_weekly{suffix}.csv"))
    print(f"[panel] {panel.shape[0]} weeks x {panel.shape[1]} features "
          f"(weeks with >=10 pairs: {(panel['n_pairs']>=10).sum()})")

    res = analyze(panel, rets, args.maxlag, diff=args.diff)
    res.to_csv(os.path.join(out_dir, f"analysis{suffix}.csv"), index=False)
    if not len(res):
        print("[panel] no testable feature/target pairs yet (need more labeled weeks)")
        return
    sig = res[(res.get("pearson_p_bh", 1) < 0.10)]
    print(f"\n[panel] {len(res)} tests | {len(sig)} significant after BH (q<0.10)")
    cols = ["feature", "target", "n", "pearson_r", "pearson_p", "pearson_p_bh", "granger_p_min"]
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print("\nTop 12 by raw p:\n", res[cols].head(12).to_string(index=False))
        if len(sig):
            print("\nBH-significant (q<0.10):\n", sig[cols].to_string(index=False))


if __name__ == "__main__":
    main()
