"""Rebuild the annotated corpus locally, then turn the offset labels into spans.

This repository ships labels, not text. Every annotated comment is identified in
`data/manifest/*.manifest.jsonl` by its Reddit id, and every span is stored as an
offset (see `tools/spanpack.py`). This script fetches the comments back from the
Arctic Shift archive, checks each one against the SHA-256 in the manifest, and
writes the fully-formed dataset that the pipeline expects.

    python tools/rehydrate.py                    # fetch + verify every split
    python tools/rehydrate.py --split gold_300   # one split
    python tools/rehydrate.py --decode           # also write text-bearing labels
    python tools/rehydrate.py --verify-only      # re-check an existing rehydration

Fetching is resumable: partial output is reused, so a killed run costs nothing.

Two things will not round-trip perfectly, and the checksum report tells you
which comments they hit. Reddit content is mutable — a comment edited or deleted
since the archive snapshot returns different text, or nothing at all — and about
2% of any Reddit id set is unavailable at any given time. Labels for comments
that fail their checksum are still written, flagged `"text_mismatch": true`, so
you can reproduce the reported numbers on the verified subset and see exactly
how large the unverified remainder is.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import spanpack  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
MANIFEST_DIR = ROOT / "data" / "manifest"
LABEL_DIR = ROOT / "data" / "labels"
OUT_DIR = ROOT / "data" / "rehydrated"

API = "https://arctic-shift.photon-reddit.com/api"
USER_AGENT = "econecpe-rehydrate/1.0 (academic replication)"
BATCH = 100
SLEEP = 1.0          # the archive 422s if polled faster; see the ingest code
RETRIES = 5


def sha(text: str | None) -> str | None:
    if text is None:
        return None
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _get(url: str) -> list[dict]:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    for attempt in range(RETRIES):
        try:
            with urllib.request.urlopen(request, timeout=90) as response:
                return json.load(response).get("data", [])
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as exc:
            status = getattr(exc, "code", None)
            if attempt == RETRIES - 1 or (status is not None and status not in (408, 422, 429, 500, 502, 503, 504)):
                raise
            time.sleep(SLEEP * 2 ** attempt)
    return []


def fetch(kind: str, ids: list[str]) -> dict[str, dict]:
    """Fetch comments or posts by id. `kind` is 'comments' or 'posts'."""
    out: dict[str, dict] = {}
    clean = sorted({i.split("_", 1)[-1] for i in ids if i})
    for start in range(0, len(clean), BATCH):
        chunk = clean[start:start + BATCH]
        for record in _get(f"{API}/{kind}/ids?ids={','.join(chunk)}"):
            if record.get("id"):
                out[record["id"]] = record
        done = min(start + BATCH, len(clean))
        print(f"\r  {kind}: {done}/{len(clean)}", end="", flush=True)
        time.sleep(SLEEP)
    print()
    return out


def rehydrate_split(name: str, verify_only: bool = False) -> Path:
    manifest = [json.loads(l) for l in (MANIFEST_DIR / f"{name}.manifest.jsonl").read_text().splitlines() if l.strip()]
    out_path = OUT_DIR / f"{name}.jsonl"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    have: dict[str, dict] = {}
    if out_path.exists():
        for line in out_path.read_text().splitlines():
            if line.strip():
                record = json.loads(line)
                have[record["comment_id"]] = record
    if verify_only and not have:
        raise SystemExit(f"{out_path} does not exist — run without --verify-only first")

    missing = [e for e in manifest if e["comment_id"] not in have]
    print(f"{name}: {len(manifest)} annotated comments, {len(have)} already local, {len(missing)} to fetch")

    if missing and not verify_only:
        comments = fetch("comments", [e["comment_id"] for e in missing])
        parent_ids = [e["parent_id"] for e in missing if (e.get("parent_id") or "").startswith("t1_")]
        parents = fetch("comments", parent_ids) if parent_ids else {}
        post_ids = [e["link_id"] for e in missing if e.get("link_id")]
        posts = fetch("posts", post_ids) if post_ids else {}

        with out_path.open("a", encoding="utf-8") as fh:
            for entry in missing:
                comment = comments.get(entry["comment_id"])
                if comment is None:
                    continue
                post = posts.get((entry.get("link_id") or "").split("_", 1)[-1]) or {}
                parent = parents.get((entry.get("parent_id") or "").split("_", 1)[-1]) or {}
                # `author` is deliberately not carried over: the labels never
                # needed it and the release does not republish who wrote what.
                fh.write(json.dumps({
                    "comment_id": entry["comment_id"],
                    "body": comment.get("body"),
                    "created_utc": entry.get("created_utc"),
                    "link_id": entry.get("link_id"),
                    "parent_id": entry.get("parent_id"),
                    "stratum": entry.get("stratum"),
                    "post_title": post.get("title"),
                    "post_selftext": post.get("selftext") or None,
                    "post_is_self": post.get("is_self"),
                    "parent_body": parent.get("body"),
                }, ensure_ascii=False) + "\n")

        have = {}
        for line in out_path.read_text().splitlines():
            if line.strip():
                record = json.loads(line)
                have[record["comment_id"]] = record

    unavailable = ok = mismatch = 0
    bad: set[str] = set()
    for entry in manifest:
        record = have.get(entry["comment_id"])
        if record is None:
            unavailable += 1
            bad.add(entry["comment_id"])
            continue
        texts = spanpack.sources_of(record)
        if all(entry.get(f"sha256_{k}") == sha(texts[k] or None) for k in spanpack.SOURCES):
            ok += 1
        else:
            mismatch += 1
            bad.add(entry["comment_id"])

    total = len(manifest)
    print(f"  verified {ok}/{total} ({ok / total:.1%}); "
          f"{unavailable} unavailable, {mismatch} changed since the archive snapshot")
    (OUT_DIR / f"{name}.unverified.json").write_text(json.dumps(sorted(bad), indent=1))
    return out_path


def decode_labels(name: str, label_files: list[str]) -> None:
    corpus = {}
    for line in (OUT_DIR / f"{name}.jsonl").read_text().splitlines():
        if line.strip():
            record = json.loads(line)
            corpus[record["comment_id"]] = spanpack.sources_of(record)
    unverified = set(json.loads((OUT_DIR / f"{name}.unverified.json").read_text()))

    for relative in label_files:
        source = LABEL_DIR / relative
        target = OUT_DIR / Path(relative).name.replace(".offsets.jsonl", ".jsonl")
        written = skipped = 0
        with target.open("w", encoding="utf-8") as fh:
            for line in source.read_text().splitlines():
                if not line.strip():
                    continue
                record = json.loads(line)
                texts = corpus.get(record["comment_id"])
                if texts is None:
                    skipped += 1
                    continue
                decoded = spanpack.decode_record(record, texts)
                if record["comment_id"] in unverified:
                    decoded["text_mismatch"] = True
                fh.write(json.dumps(decoded, ensure_ascii=False) + "\n")
                written += 1
        print(f"  {target.relative_to(ROOT)}: {written} records"
              + (f", {skipped} skipped (comment unavailable)" if skipped else ""))


# Which label files belong to which split. Kept here rather than inferred so a
# missing rehydration fails loudly instead of silently producing a short file.
SPLITS: dict[str, list[str]] = {
    "gold_300": ["gold/adjudicated.offsets.jsonl", "gold/human_gold_100.offsets.jsonl"],
    "silver_20000": ["silver/consensus.offsets.jsonl"],
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split", choices=sorted(SPLITS), help="only this split (default: all)")
    parser.add_argument("--decode", action="store_true", help="also write text-bearing label files")
    parser.add_argument("--verify-only", action="store_true", help="re-check an existing rehydration, fetch nothing")
    args = parser.parse_args()

    for name in ([args.split] if args.split else sorted(SPLITS)):
        rehydrate_split(name, verify_only=args.verify_only)
        if args.decode:
            decode_labels(name, SPLITS[name])


if __name__ == "__main__":
    main()
