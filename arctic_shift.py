#!/usr/bin/env python3
"""Ingest the FULL history of r/economics (submissions + comments) from the
Arctic Shift archive — the community successor to Pushshift, covering every
public subreddit from Dec 2005 to the current month.

Design goals:
  * Complete   — walks the whole archive ascending by created_utc.
  * Resumable  — checkpoints after every page; kill it and re-run to continue.
  * Robust     — retries with backoff on 429/5xx/network errors.
  * Streaming  — appends JSONL (one record per line), so multi-GB output and
                 interruptions are both fine.

Pagination: the API caps `limit` at 100 and treats `after` as an EXCLUSIVE
lower bound on created_utc. We query `after = last_max - 1` (which returns
created_utc >= last_max) and dedupe the boundary second by id, so we never skip
items that share the last page's final timestamp.

Usage:
    python arctic_shift.py                 # both streams, full history
    python arctic_shift.py --only posts    # submissions only
    python arctic_shift.py --only comments # comments only
    python arctic_shift.py --status        # print progress and exit
"""
import argparse
import json
import os
import sys
import time

import requests

import config

STREAMS = {
    "posts": ("posts/search", config.SUBMISSIONS_FILE),
    "comments": ("comments/search", config.COMMENTS_FILE),
}


def _checkpoint_path(stream):
    # Per-stream AND per-subreddit files so posts + comments (and different
    # subreddits) can run as independent processes without clobbering each
    # other's progress. Namespacing by subreddit prevents a new sub's ingest
    # from overwriting another sub's cursor (which would append duplicates on a
    # future re-run). r/economics keeps its original (pre-namespacing) filenames
    # for backward compatibility, so its already-completed checkpoints stay valid
    # without any file renaming.
    if config.SUBREDDIT == "economics":
        return os.path.join(config.DATA_DIR, f"arctic_checkpoint_{stream}.json")
    return os.path.join(config.DATA_DIR,
                        f"arctic_checkpoint_{config.SUBREDDIT}_{stream}.json")


def _load_checkpoint(stream):
    path = _checkpoint_path(stream)
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return {}


def _save_checkpoint(stream, state):
    path = _checkpoint_path(stream)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(state, f)
    os.replace(tmp, path)  # atomic


def _request(endpoint, params):
    """GET with retry/backoff. Returns the `data` list, or None on fatal error."""
    url = f"{config.ARCTIC_BASE}/{endpoint}"
    headers = {"User-Agent": config.ARCTIC_USER_AGENT}
    for attempt in range(config.ARCTIC_MAX_RETRIES):
        try:
            r = requests.get(url, params=params, headers=headers, timeout=60)
        except requests.RequestException as e:
            wait = min(60, 2 ** attempt * 2)
            print(f"    net error: {e} -> retry in {wait}s", flush=True)
            time.sleep(wait)
            continue
        if r.status_code == 200:
            try:
                return r.json().get("data", [])
            except ValueError:
                print(f"    bad JSON (len={len(r.text)}) -> retry", flush=True)
                time.sleep(3)
                continue
        # Transient: 408 (req timeout), 422 ("slow down a bit"), 429 (rate limit),
        # and ALL 5xx (incl. Cloudflare 520-527/530).
        if r.status_code in (408, 422, 429) or r.status_code >= 500:
            wait = min(120, 2 ** attempt * 3)
            print(f"    HTTP {r.status_code} -> backing off {wait}s", flush=True)
            time.sleep(wait)
            continue
        # Other 4xx (e.g. 400 bad params) -> genuinely not retryable
        print(f"    HTTP {r.status_code}: {r.text[:200]}", flush=True)
        return None
    return None


def ingest(stream):
    endpoint, outfile = STREAMS[stream]
    state = _load_checkpoint(stream)
    if state.get("complete"):
        print(f"[{stream}] already complete ({state.get('total', 0)} items). "
              f"Delete {_checkpoint_path(stream)} to re-pull.", flush=True)
        return

    # Start at the later of the checkpoint and the configured window start, so a
    # fresh run begins at START_TS (2018) and a resume continues where it left off.
    after = max(int(state.get("after", 0)), config.START_TS)  # exclusive lower bound
    boundary_ids = set(state.get("boundary_ids", []))          # ids already saved at ts == after
    total = int(state.get("total", 0))

    print(f"[{stream}] resuming: created_utc>{after} "
          f"({time.strftime('%Y-%m-%d', time.gmtime(after))}), "
          f"{total} saved so far", flush=True)

    out = open(outfile, "a", encoding="utf-8")
    fails = 0
    try:
        while True:
            params = {"subreddit": config.SUBREDDIT,
                      "limit": config.ARCTIC_PAGE_LIMIT,   # "auto" -> up to 1000/page
                      "sort": "asc",
                      "after": after - 1}  # returns created_utc >= after; boundary deduped by id

            data = _request(endpoint, params)
            if data is None:
                # Retries for this batch are exhausted. Rather than kill a
                # multi-hour run on a transient outage, sleep and resume from
                # the checkpoint. Give up only after many consecutive failures.
                fails += 1
                if fails > config.ARCTIC_MAX_FATAL:
                    print(f"[{stream}] {fails} consecutive failed batches — "
                          f"giving up (safe to re-run to resume).", flush=True)
                    return
                wait = min(600, 30 * fails)
                print(f"[{stream}] batch failed ({fails}/{config.ARCTIC_MAX_FATAL}); "
                      f"sleeping {wait}s then resuming.", flush=True)
                time.sleep(wait)
                continue
            fails = 0

            new_items = 0
            for item in data:
                ts = item.get("created_utc")
                iid = item.get("id")
                if ts is None or iid is None:
                    continue
                if ts < after:
                    continue                       # older than window (safety)
                if ts == after and iid in boundary_ids:
                    continue                       # already saved this exact item
                out.write(json.dumps(item, ensure_ascii=False))
                out.write("\n")
                new_items += 1
            out.flush()
            total += new_items

            # No new items beyond what's already saved => end of the window/archive.
            # (For this subreddit a single second can't fill a whole 1000-item page,
            # so re-querying the boundary second and finding only dupes is a reliable
            # end signal. The boundary re-query above is what guarantees we never drop
            # items that were truncated at a page boundary.)
            if new_items == 0:
                state.update(after=after, boundary_ids=list(boundary_ids),
                             total=total, complete=True)
                _save_checkpoint(stream, state)
                print(f"[{stream}] DONE — reached end. Total: {total}", flush=True)
                return

            # Advance cursor + rebuild the boundary-id set for the new max second.
            batch_max = max(it["created_utc"] for it in data if it.get("created_utc") is not None)
            max_ids = {it["id"] for it in data if it.get("created_utc") == batch_max}
            boundary_ids = (boundary_ids | max_ids) if batch_max == after else max_ids
            after = batch_max

            state.update(after=after, boundary_ids=list(boundary_ids), total=total)
            _save_checkpoint(stream, state)

            print(f"[{stream}] +{new_items:4d}  total={total:<9d} "
                  f"@ {time.strftime('%Y-%m-%d', time.gmtime(after))}", flush=True)
            time.sleep(config.ARCTIC_SLEEP)
    finally:
        out.close()


def print_status():
    print(f"Subreddit: r/{config.SUBREDDIT}")
    for stream in STREAMS:
        s = _load_checkpoint(stream)
        upto = s.get("after", 0)
        when = time.strftime("%Y-%m-%d", time.gmtime(upto)) if upto else "—"
        flag = "COMPLETE" if s.get("complete") else "in progress"
        print(f"  {stream:9s}: {s.get('total', 0):>9,} saved | up to {when} | {flag}")


def main():
    ap = argparse.ArgumentParser(description="Arctic Shift full-history ingester")
    ap.add_argument("--only", choices=list(STREAMS), help="ingest just one stream")
    ap.add_argument("--status", action="store_true", help="print progress and exit")
    args = ap.parse_args()

    if args.status:
        print_status()
        return

    streams = [args.only] if args.only else list(STREAMS)
    for stream in streams:
        ingest(stream)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted — progress is checkpointed; re-run to resume.", file=sys.stderr)
