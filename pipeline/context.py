"""SQLite index over the scraped JSONL + context assembly (spec §1).

The raw archive is append-only JSONL (747MB+ of comments); annotation needs fast
random access: comment -> its post's title/selftext, reply -> its parent's body.
This module maintains `data/index.sqlite` incrementally: each `--build` run reads
only the bytes appended since the last run (byte offsets kept in the `meta`
table), so it can be re-run cheaply while the Arctic Shift ingest is still
appending — same kill-and-rerun property as the scrapers.

    python -m pipeline.context --build     # (re)index newly appended records
    python -m pipeline.context --status    # row counts + indexed offsets

Assembly (`ContextAssembler`): the model input context is the post title
(+ selftext, truncated) and, for replies, the immediate parent comment body
(truncated) — per the locked context design.
"""
from __future__ import annotations

import argparse
import json
import os
import sqlite3
import time

import config

_SCHEMA = """
CREATE TABLE IF NOT EXISTS posts (
    id          TEXT PRIMARY KEY,
    created_utc INTEGER,
    title       TEXT,
    selftext    TEXT,
    is_self     INTEGER,
    num_comments INTEGER,
    score       INTEGER
);
CREATE TABLE IF NOT EXISTS comments (
    id          TEXT PRIMARY KEY,
    created_utc INTEGER,
    link_id     TEXT,
    parent_id   TEXT,
    author      TEXT,
    score       INTEGER,
    body        TEXT
);
CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT
);
"""

_BATCH = 20_000


def connect(db_path: str | None = None) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path or config.INDEX_DB)
    conn.executescript(_SCHEMA)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    return conn


def _get_offset(conn: sqlite3.Connection, key: str) -> int:
    row = conn.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
    return int(row[0]) if row else 0


def _set_offset(conn: sqlite3.Connection, key: str, offset: int) -> None:
    conn.execute("INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)", (key, str(offset)))


def _ingest_stream(conn, jsonl_path, offset_key, table, row_of) -> int:
    """Append records added to `jsonl_path` since the recorded byte offset."""
    if not os.path.exists(jsonl_path):
        return 0
    start = _get_offset(conn, offset_key)
    added = 0
    cols = {"posts": "(?,?,?,?,?,?,?)", "comments": "(?,?,?,?,?,?,?)"}[table]
    sql = f"INSERT OR REPLACE INTO {table} VALUES {cols}"
    batch = []
    with open(jsonl_path, encoding="utf-8") as f:
        f.seek(start)
        while True:
            line = f.readline()
            if not line:
                break
            if not line.endswith("\n"):
                break  # partial trailing line mid-append; picked up next run
            start = f.tell()
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            row = row_of(rec)
            if row is None:
                continue
            batch.append(row)
            if len(batch) >= _BATCH:
                conn.executemany(sql, batch)
                _set_offset(conn, offset_key, start)
                conn.commit()
                added += len(batch)
                batch = []
    if batch:
        conn.executemany(sql, batch)
        added += len(batch)
    _set_offset(conn, offset_key, start)
    conn.commit()
    return added


def _post_row(rec: dict):
    pid = rec.get("id")
    if not pid:
        return None
    return (pid, rec.get("created_utc"), rec.get("title") or "",
            rec.get("selftext") or "", 1 if rec.get("is_self") else 0,
            rec.get("num_comments"), rec.get("score"))


def _comment_row(rec: dict):
    cid = rec.get("id")
    if not cid:
        return None
    return (cid, rec.get("created_utc"), rec.get("link_id") or "",
            rec.get("parent_id") or "", rec.get("author") or "",
            rec.get("score"), rec.get("body") or "")


def build(db_path: str | None = None) -> None:
    conn = connect(db_path)
    t0 = time.time()
    np = _ingest_stream(conn, config.SUBMISSIONS_FILE, "posts_offset", "posts", _post_row)
    nc = _ingest_stream(conn, config.COMMENTS_FILE, "comments_offset", "comments", _comment_row)
    print(f"[index] +{np} posts, +{nc} comments in {time.time()-t0:.1f}s", flush=True)
    conn.close()


def status(db_path: str | None = None) -> None:
    conn = connect(db_path)
    for table in ("posts", "comments"):
        n = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        print(f"  {table:9s}: {n:>9,} rows")
    conn.close()


def strip_kind(thing_id: str | None) -> str:
    """'t3_abc' -> 'abc' (Reddit fullname -> bare id)."""
    if not thing_id:
        return ""
    return thing_id.split("_", 1)[1] if "_" in thing_id else thing_id


class ContextAssembler:
    """Look up the context texts for a comment row (dict with link_id/parent_id)."""

    def __init__(self, conn: sqlite3.Connection | None = None):
        self.conn = conn or connect()

    def post(self, link_id: str) -> dict | None:
        row = self.conn.execute(
            "SELECT title, selftext, is_self FROM posts WHERE id=?",
            (strip_kind(link_id),)).fetchone()
        if row is None:
            return None
        return {"title": row[0], "selftext": row[1], "is_self": bool(row[2])}

    def parent_body(self, parent_id: str) -> str | None:
        if not parent_id.startswith("t1_"):
            return None  # top-level comment: parent is the post itself
        row = self.conn.execute(
            "SELECT body FROM comments WHERE id=?", (strip_kind(parent_id),)).fetchone()
        return row[0] if row else None

    def assemble(self, comment: dict) -> dict:
        """Context fields for one comment record (JSONL dict or equivalent)."""
        post = self.post(comment.get("link_id") or "") or {}
        selftext = (post.get("selftext") or "")[: config.CTX_SELFTEXT_MAX] or None
        parent = self.parent_body(comment.get("parent_id") or "")
        return {
            "post_title": post.get("title"),
            "post_selftext": selftext,
            "post_is_self": post.get("is_self"),
            "parent_body": parent[: config.CTX_PARENT_MAX] if parent else None,
        }


def main():
    ap = argparse.ArgumentParser(description="Build/inspect the post+comment lookup index")
    ap.add_argument("--build", action="store_true", help="index newly appended records")
    ap.add_argument("--status", action="store_true", help="print row counts")
    args = ap.parse_args()
    if args.build:
        build()
    if args.status or not args.build:
        status()


if __name__ == "__main__":
    main()
