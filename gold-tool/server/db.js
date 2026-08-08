import Database from "better-sqlite3";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
// GOLD_DB lets the human-gold pass (100 comments, single annotator) keep its
// own store, so it can never collide with — or be confused for — the 300-comment
// LLM double-coding already sitting in gold_labels.sqlite. See README "Human gold".
const DB_PATH = process.env.GOLD_DB
  ? path.resolve(process.env.GOLD_DB)
  : path.join(__dirname, "data", "gold_labels.sqlite");
fs.mkdirSync(path.dirname(DB_PATH), { recursive: true });
const db = new Database(DB_PATH);

db.pragma("journal_mode = WAL");
db.exec(`
  CREATE TABLE IF NOT EXISTS labels (
    comment_id TEXT NOT NULL,
    annotator  TEXT NOT NULL,
    neutral    INTEGER NOT NULL DEFAULT 0,
    pairs      TEXT NOT NULL DEFAULT '[]',
    updated_at TEXT NOT NULL,
    PRIMARY KEY (comment_id, annotator)
  );

  CREATE TABLE IF NOT EXISTS adjudications (
    comment_id  TEXT PRIMARY KEY,
    final       TEXT NOT NULL,      -- A | B | NEUTRAL | CUSTOM
    neutral     INTEGER NOT NULL DEFAULT 0,
    pairs       TEXT NOT NULL DEFAULT '[]',
    resolved_by TEXT NOT NULL,
    updated_at  TEXT NOT NULL
  );

  CREATE TABLE IF NOT EXISTS discussion (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    comment_id TEXT NOT NULL,
    author     TEXT NOT NULL,
    body       TEXT NOT NULL,
    created_at TEXT NOT NULL
  );
  CREATE INDEX IF NOT EXISTS idx_discussion_comment ON discussion (comment_id, id);
`);

export default db;
