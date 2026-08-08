"""Stress-stratified pilot sampler (spec §10) + hygiene filters.

Pilot strata (defaults, over the currently indexed comments):
  link_top  40% — top-level comments on link posts   → stresses post-title causes
  reply_arg 40% — replies addressing another user    → stresses parent causes +
                  interpersonal affect ("you/your" proxy)
  random    20% — pure random emotive-unfiltered     → honest base rates

Hygiene (all strata): body 60–1500 chars, not [deleted]/[removed], author not
AutoModerator/*bot. Reproducible via --seed (SQL-side deterministic ordering).

Output JSONL: one item per comment with pre-assembled context, ready for
pipeline.annotate:
    {comment_id, stratum, body, author, created_utc, score, link_id, parent_id,
     post_title, post_selftext, post_is_self, parent_body}

NOTE: until the Arctic Shift comments ingest completes, the index only covers
2018→(ingest frontier); the pilot doesn't need full coverage, the 20–50k
annotation sample does — re-run sampling after ingest completion.

    python -m pipeline.sample --pilot --n 250 --seed 7
"""
from __future__ import annotations

import argparse
import json
import os

import config

from .context import ContextAssembler, connect

HYGIENE = """
    LENGTH(c.body) BETWEEN 60 AND 1500
    AND c.body NOT IN ('[removed]', '[deleted]')
    AND c.author NOT IN ('AutoModerator', '[deleted]')
    AND LOWER(c.author) NOT LIKE '%bot'
    AND c.body NOT LIKE 'Rule %'
    AND c.body NOT LIKE '%[contact the mods](%'
"""

# Deterministic pseudo-random order: hash the id with the seed. abs() to avoid
# negative/positive interleave differences across SQLite versions.
_ORDER = "ORDER BY ABS((c.rowid * 2654435761 + :seed) % 2147483647)"

_STRATA_SQL = {
    # Top-level comment on a link post: parent is the post, post is not a self post.
    "link_top": f"""
        SELECT c.* FROM comments c
        JOIN posts p ON p.id = SUBSTR(c.link_id, 4)
        WHERE c.parent_id LIKE 't3\\_%' ESCAPE '\\'
          AND p.is_self = 0 AND {HYGIENE} {_ORDER} LIMIT :n""",
    # Reply that addresses another user (interpersonal/argument proxy) and whose
    # parent body is present in the index (else the parent-cause stress is moot).
    "reply_arg": f"""
        SELECT c.* FROM comments c
        JOIN comments pc ON pc.id = SUBSTR(c.parent_id, 4)
        WHERE c.parent_id LIKE 't1\\_%' ESCAPE '\\'
          AND (' ' || LOWER(c.body) || ' ') LIKE '% you %'
          AND {HYGIENE} {_ORDER} LIMIT :n""",
    "random": f"SELECT c.* FROM comments c WHERE {HYGIENE} {_ORDER} LIMIT :n",
}


def sample_pilot(n: int = 250, seed: int = 7, out_path: str | None = None) -> str:
    os.makedirs(config.SAMPLES_DIR, exist_ok=True)
    out_path = out_path or os.path.join(config.SAMPLES_DIR, f"pilot_{n}_seed{seed}.jsonl")

    conn = connect()
    cur = conn.cursor()
    cur.row_factory = lambda c, row: {d[0]: row[i] for i, d in enumerate(c.description)}
    asm = ContextAssembler(conn)  # plain tuple rows; dict rows stay cursor-local

    quotas = {"link_top": round(n * 0.4), "reply_arg": round(n * 0.4)}
    quotas["random"] = n - sum(quotas.values())

    seen: set[str] = set()
    items = []
    for stratum, quota in quotas.items():
        got = 0
        # Over-fetch to survive cross-stratum dedup.
        rows = cur.execute(_STRATA_SQL[stratum], {"n": quota * 2, "seed": seed}).fetchall()
        for c in rows:
            if got >= quota or c["id"] in seen:
                continue
            seen.add(c["id"])
            items.append({
                "comment_id": c["id"], "stratum": stratum, "body": c["body"],
                "author": c["author"], "created_utc": c["created_utc"],
                "score": c["score"], "link_id": c["link_id"],
                "parent_id": c["parent_id"], **asm.assemble(c),
            })
            got += 1
        if got < quota:
            print(f"[sample] WARNING: stratum {stratum} short: {got}/{quota}", flush=True)

    with open(out_path, "w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    counts = {s: sum(1 for i in items if i["stratum"] == s) for s in quotas}
    print(f"[sample] wrote {len(items)} items to {out_path}  {counts}", flush=True)
    conn.close()
    return out_path


# ── Full annotation sample (Phase 2 silver run) ──────────────────────────────
#
# Design (spec §10 rationale):
#   15%  random   — pure random hygiene-passing slice → honest base rates
#   42.5% topic   — keyword-proxy buckets for the economic cause taxonomy
#                   (can't stratify by cause_category before labels exist)
#   42.5% emotive — lexicon/punctuation-scored, boosts pair density
# Per-year quotas proportional to hygiene-passing volume (floor applied) so all
# market regimes (pre-COVID, COVID, 2022 inflation, 2024-26) are covered.
# One deterministic scan in hash order fills all quotas: random slice first
# (unconditional → unbiased), then topic, then emotive.

TOPIC_KEYWORDS = {  # first-match order; high-precision, lowercase substrings
    "trade_tariffs": ["tariff", "trade war", "trade deal", "supply chain"],
    "monetary_policy": ["the fed", "federal reserve", "interest rate", "rate hike",
                        "rate cut", "powell", "quantitative easing", "central bank"],
    "inflation": ["inflation", " cpi", "cost of living", "purchasing power"],
    "employment": ["unemployment", "layoff", "jobs report", "labor market", "wages",
                   "minimum wage"],
    "growth_gdp": ["gdp", "recession", "economic growth", "productivity"],
    "fiscal_taxes": ["deficit", "national debt", "government spending", "stimulus",
                     "tax cut", "taxes", "budget bill"],
    "energy": ["oil price", "gas price", "opec", "energy price", "crude oil"],
    "housing": ["housing", "mortgage", "home price", "real estate", "landlord",
                "rent is", "rent has"],
    "markets_themselves": ["stock market", "bubble", "market crash", "s&p", "nasdaq",
                           "valuation", "bull market", "bear market"],
    "crypto": ["bitcoin", "crypto", "ethereum", "btc "],
    "corporate_earnings": ["earnings", "buyback", "profit margin", "quarterly report"],
    "inequality_distribution": ["inequality", "wealth gap", "billionaire", "top 1%",
                                "wealth tax", "rich get richer"],
    "personal_finance": ["my savings", "my paycheck", "paycheck to paycheck",
                         "student loan", "credit card debt", "emergency fund",
                         "my 401k", "my retirement", "my salary", "my finances",
                         "personal debt", "my credit score", "can't afford"],
    "geopolitics": ["sanction", "ukraine", "geopolit", "election", " war "],
}

EMOTIVE_WORDS = [
    "terrified", "scared", "afraid", "worried", "worry", "anxious", "panic", "fear",
    "angry", "furious", "outrage", "disgust", "insane", "ridiculous", "absurd",
    "criminal", "scam", "hate", "awful", "terrible", "horrible", "devastat",
    "depress", "hopeless", "doomed", "screwed", "cooked", "brutal", "excited",
    "thrilled", "amazing", "incredible", "euphori", "to the moon", "all in", "yolo",
    "unbelievable", "shocking", "stunned", "wtf", "wow", "damn", "holy", "crazy",
]
_EMOJI = ("🚀", "📉", "📈", "😭", "🙄", "💀", "🤡", "😂", "🤦")


def emotive_score(body: str) -> int:
    low = body.lower()
    score = sum(1 for w in EMOTIVE_WORDS if w in low)
    score += min(2, body.count("!"))
    if any(e in body for e in _EMOJI):
        score += 1
    if sum(1 for t in body.split() if len(t) > 3 and t.isupper()) >= 1:
        score += 1
    return score


def _hygiene_py(body: str, author: str) -> bool:
    return (60 <= len(body) <= 1500
            and body not in ("[removed]", "[deleted]")
            and author not in ("AutoModerator", "[deleted]")
            and not author.lower().endswith("bot")
            and not body.startswith("Rule ")
            and "[contact the mods](" not in body)


def sample_full(n: int = 20_000, seed: int = 7, out_path: str | None = None) -> str:
    import datetime as dt

    os.makedirs(config.SAMPLES_DIR, exist_ok=True)
    out_path = out_path or os.path.join(config.SAMPLES_DIR, f"silver_{n}_seed{seed}.jsonl")
    conn = connect()

    # Per-year hygiene-passing volume → proportional quotas with a floor.
    print("[sample] counting per-year volume...", flush=True)
    vol: dict[int, int] = {}
    for ts, cnt in conn.execute(
            f"SELECT (created_utc/31556952)+1970, COUNT(*) FROM comments c "
            f"WHERE {HYGIENE} GROUP BY 1"):
        vol[int(ts)] = cnt
    total_vol = sum(vol.values())

    def year_quotas(slice_n: int, floor: int) -> dict[int, int]:
        q = {y: max(floor, round(slice_n * v / total_vol)) for y, v in vol.items()}
        scale = slice_n / sum(q.values())
        return {y: max(floor, round(c * scale)) for y, c in q.items()}

    n_rand = round(n * 0.15)
    n_topic = round(n * 0.425)
    n_emo = n - n_rand - n_topic
    rand_q = year_quotas(n_rand, 200)
    emo_q = year_quotas(n_emo, 500)
    topic_q = {t: n_topic // len(TOPIC_KEYWORDS) for t in TOPIC_KEYWORDS}
    # per-(topic, year) cap keeps buckets from clumping into one era
    topic_year_cap = {(t, y): max(60, int(2 * q * vol[y] / total_vol))
                      for t, q in topic_q.items() for y in vol}

    picked: dict[str, dict] = {}
    counts: dict = {"random": {y: 0 for y in vol}, "emotive": {y: 0 for y in vol},
                    "topic": {t: 0 for t in TOPIC_KEYWORDS}}
    ty_counts = {k: 0 for k in topic_year_cap}
    need_rand = n_rand
    need_topic = n_topic
    need_emo = n_emo

    print(f"[sample] scanning in hash order (quotas: random {n_rand}, "
          f"topic {n_topic}, emotive {n_emo})...", flush=True)
    cur = conn.execute(
        "SELECT id, created_utc, link_id, parent_id, author, score, body FROM comments "
        "ORDER BY ABS((rowid * 2654435761 + ?) % 2147483647)", (seed,))
    scanned = 0
    while need_rand or need_topic or need_emo:
        rows = cur.fetchmany(50_000)
        if not rows:
            break
        for cid, ts, link_id, parent_id, author, score, body in rows:
            scanned += 1
            if cid in picked or not _hygiene_py(body or "", author or ""):
                continue
            y = dt.datetime.fromtimestamp(ts, dt.timezone.utc).year
            rec = {"comment_id": cid, "body": body, "author": author,
                   "created_utc": ts, "score": score, "link_id": link_id,
                   "parent_id": parent_id}
            if need_rand and counts["random"].get(y, 0) < rand_q.get(y, 0):
                rec["stratum"] = "random"
                picked[cid] = rec
                counts["random"][y] += 1
                need_rand -= 1
                continue
            if need_topic:
                low = body.lower()
                for t, kws in TOPIC_KEYWORDS.items():
                    if (counts["topic"][t] < topic_q[t]
                            and ty_counts[(t, y)] < topic_year_cap[(t, y)]
                            and any(k in low for k in kws)):
                        rec["stratum"] = "topic"
                        rec["topic_bucket"] = t
                        picked[cid] = rec
                        counts["topic"][t] += 1
                        ty_counts[(t, y)] += 1
                        need_topic -= 1
                        break
                if cid in picked:
                    continue
            if need_emo and counts["emotive"].get(y, 0) < emo_q.get(y, 0) \
                    and emotive_score(body) >= 2:
                rec["stratum"] = "emotive"
                picked[cid] = rec
                counts["emotive"][y] += 1
                need_emo -= 1
    print(f"[sample] scanned {scanned:,} rows; picked {len(picked):,} "
          f"(short: rand {need_rand}, topic {need_topic}, emo {need_emo})", flush=True)

    print("[sample] assembling context...", flush=True)
    asm = ContextAssembler(conn)
    with open(out_path, "w", encoding="utf-8") as f:
        for rec in picked.values():
            f.write(json.dumps({**rec, **asm.assemble(rec)}, ensure_ascii=False) + "\n")
    print(f"[sample] wrote {len(picked):,} items to {out_path}", flush=True)
    print(f"[sample] topic fill: { {t: c for t, c in counts['topic'].items()} }", flush=True)
    conn.close()
    return out_path


# ── Gold subsample (Phase 2 human double-coding) ─────────────────────────────
#
# Draws from an already-materialized full sample (no DB access needed): mirrors
# its random/topic/emotive proportions (15/42.5/42.5), and spreads the topic
# slice evenly across whichever cause-taxonomy keyword buckets are present so
# every category gets a gold-label floor for per-category kappa.

def sample_gold(n: int = 300, seed: int = 7, src_path: str | None = None,
                out_path: str | None = None) -> str:
    import random as _random

    os.makedirs(config.SAMPLES_DIR, exist_ok=True)
    src_path = src_path or os.path.join(config.SAMPLES_DIR, f"silver_20000_seed{seed}.jsonl")
    out_path = out_path or os.path.join(config.SAMPLES_DIR, f"gold_{n}_seed{seed}.jsonl")

    by_stratum: dict[str, list[dict]] = {"random": [], "topic": [], "emotive": []}
    by_bucket: dict[str, list[dict]] = {}
    with open(src_path, encoding="utf-8") as f:
        for line in f:
            it = json.loads(line)
            by_stratum.setdefault(it["stratum"], []).append(it)
            if it["stratum"] == "topic":
                by_bucket.setdefault(it["topic_bucket"], []).append(it)

    rng = _random.Random(seed)
    n_rand = round(n * 0.15)
    n_topic = round(n * 0.425)
    n_emo = n - n_rand - n_topic

    picked: list[dict] = list(rng.sample(by_stratum["random"],
                                         min(n_rand, len(by_stratum["random"]))))

    buckets = sorted(by_bucket)
    base, extra = divmod(n_topic, len(buckets))
    for i, b in enumerate(buckets):
        quota = base + (1 if i < extra else 0)
        picked += rng.sample(by_bucket[b], min(quota, len(by_bucket[b])))

    picked += rng.sample(by_stratum["emotive"], min(n_emo, len(by_stratum["emotive"])))
    rng.shuffle(picked)  # de-block so the labeling UI doesn't see runs of one stratum

    with open(out_path, "w", encoding="utf-8") as f:
        for it in picked:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    print(f"[sample] wrote {len(picked)} gold items to {out_path}  "
          f"target(random={n_rand}, topic={n_topic}, emotive={n_emo})", flush=True)
    return out_path


def main():
    ap = argparse.ArgumentParser(description="Pilot + full + gold annotation samplers")
    ap.add_argument("--pilot", action="store_true", help="draw the pilot sample")
    ap.add_argument("--full", action="store_true", help="draw the full silver sample")
    ap.add_argument("--gold", action="store_true", help="draw the gold double-coding subsample")
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default=None)
    ap.add_argument("--src", default=None,
                    help="(--gold) full sample JSONL to subsample from "
                         "(default data/samples/silver_20000_seed<seed>.jsonl)")
    args = ap.parse_args()
    if args.pilot:
        sample_pilot(args.n or 250, args.seed, args.out)
    elif args.full:
        sample_full(args.n or 20_000, args.seed, args.out)
    elif args.gold:
        sample_gold(args.n or 300, args.seed, args.src, args.out)
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
