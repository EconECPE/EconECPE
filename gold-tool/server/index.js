import express from "express";
import cors from "cors";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

import db from "./db.js";
import { loadGoldItems, getGoldItem, getModelPredictions, getConsensusPrediction,
         getProposals } from "./data.js";
import { validatePairs } from "./validate.js";
import { hasDisagreement } from "./agreement.js";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const REPO_ROOT = path.resolve(__dirname, "..", "..");
const ADJUDICATED_PATH = path.join(REPO_ROOT, "data", "labels", "gold", "adjudicated.jsonl");
const SCHEMA = JSON.parse(
  fs.readFileSync(path.join(__dirname, "..", "shared", "schema.json"), "utf-8"));

// ── Human-gold mode ─────────────────────────────────────────────────────────
// The 300-comment pass in gold_labels.sqlite was double-coded by two LLMs
// (claude-sonnet-5, gemini-3.5-flash) whose rows were filed under the human
// annotator names. This mode runs a genuine single-annotator human pass over a
// stratified 100-comment subset, and it must be BLIND: if the annotator can see
// what the models said, human-vs-model agreement stops measuring anything.
//
//   GOLD_ANNOTATORS=A-human   replaces the annotator roster (also gates the
//                                  adjudication tab, which is a 2-coder concept)
//   GOLD_BLIND=1                   strips model + consensus predictions from the API
const BLIND = process.env.GOLD_BLIND === "1";
if (process.env.GOLD_ANNOTATORS) {
  SCHEMA.annotators = process.env.GOLD_ANNOTATORS.split(",")
    .map((s) => s.trim()).filter(Boolean);
}
// The client hides the adjudication UI and the "model said" panels on these.
SCHEMA.blind = BLIND;
SCHEMA.adjudication = SCHEMA.annotators.length > 1;

const NO_PREDICTIONS = { models: [], consensus: null, proposals: null };
function predictionsFor(commentId) {
  if (BLIND) return NO_PREDICTIONS;
  return {
    models: getModelPredictions(commentId),
    consensus: getConsensusPrediction(commentId),
    proposals: getProposals(commentId),
  };
}

const app = express();
app.use(cors());
app.use(express.json());

const getLabelStmt = db.prepare(
  "SELECT * FROM labels WHERE comment_id = ? AND annotator = ?");
const upsertLabelStmt = db.prepare(`
  INSERT INTO labels (comment_id, annotator, neutral, pairs, updated_at)
  VALUES (@comment_id, @annotator, @neutral, @pairs, @updated_at)
  ON CONFLICT(comment_id, annotator) DO UPDATE SET
    neutral = excluded.neutral, pairs = excluded.pairs, updated_at = excluded.updated_at
`);
const progressStmt = db.prepare(
  "SELECT annotator, COUNT(*) AS n FROM labels GROUP BY annotator");

const getAdjudicationStmt = db.prepare(
  "SELECT * FROM adjudications WHERE comment_id = ?");
const upsertAdjudicationStmt = db.prepare(`
  INSERT INTO adjudications (comment_id, final, neutral, pairs, resolved_by, updated_at)
  VALUES (@comment_id, @final, @neutral, @pairs, @resolved_by, @updated_at)
  ON CONFLICT(comment_id) DO UPDATE SET
    final = excluded.final, neutral = excluded.neutral, pairs = excluded.pairs,
    resolved_by = excluded.resolved_by, updated_at = excluded.updated_at
`);
const listMessagesStmt = db.prepare(
  "SELECT author, body, created_at FROM discussion WHERE comment_id = ? ORDER BY id ASC");
const insertMessageStmt = db.prepare(
  "INSERT INTO discussion (comment_id, author, body, created_at) VALUES (?, ?, ?, ?)");

function requireAnnotator(req, res) {
  const { annotator } = req.query.annotator ? req.query : req.body;
  if (!SCHEMA.annotators.includes(annotator)) {
    res.status(400).json({ error: `annotator must be one of ${SCHEMA.annotators.join(", ")}` });
    return null;
  }
  return annotator;
}

function myLabel(commentId, annotator) {
  const row = getLabelStmt.get(commentId, annotator);
  if (!row) return null;
  return { neutral: !!row.neutral, pairs: JSON.parse(row.pairs), updatedAt: row.updated_at };
}

function adjudicationRow(commentId) {
  const row = getAdjudicationStmt.get(commentId);
  if (!row) return null;
  return {
    final: row.final, neutral: !!row.neutral, pairs: JSON.parse(row.pairs),
    resolvedBy: row.resolved_by, updatedAt: row.updated_at,
  };
}

// Comment ids where A and B's independent labels disagree —
// mirrors pipeline/gold_report.py's compare_annotators disagreement flag.
function disagreementIds() {
  return loadGoldItems()
    .map((it) => it.comment_id)
    .filter((cid) => {
      const s = myLabel(cid, "A");
      const f = myLabel(cid, "B");
      return s && f && hasDisagreement(s, f);
    });
}

app.get("/api/schema", (_req, res) => res.json(SCHEMA));

app.get("/api/comments", (req, res) => {
  const annotator = requireAnnotator(req, res);
  if (!annotator) return;
  const items = loadGoldItems().map((it) => {
    const label = myLabel(it.comment_id, annotator);
    return {
      comment_id: it.comment_id,
      stratum: it.stratum,
      topic_bucket: it.topic_bucket || null,
      done: label !== null,
    };
  });
  res.json(items);
});

app.get("/api/comments/:id", (req, res) => {
  const annotator = requireAnnotator(req, res);
  if (!annotator) return;
  const item = getGoldItem(req.params.id);
  if (!item) return res.status(404).json({ error: "unknown comment_id" });
  res.json({
    comment_id: item.comment_id,
    stratum: item.stratum,
    topic_bucket: item.topic_bucket || null,
    post_title: item.post_title,
    post_selftext: item.post_selftext,
    post_is_self: item.post_is_self,
    parent_body: item.parent_body,
    body: item.body,
    author: item.author,
    created_utc: item.created_utc,
    score: item.score,
    ...predictionsFor(item.comment_id),
    myLabel: myLabel(item.comment_id, annotator),
  });
});

app.put("/api/labels/:id", (req, res) => {
  const annotator = requireAnnotator(req, res);
  if (!annotator) return;
  const item = getGoldItem(req.params.id);
  if (!item) return res.status(404).json({ error: "unknown comment_id" });

  const neutral = !!req.body.neutral;
  const rawPairs = neutral ? [] : (req.body.pairs || []);
  const { pairs, errors } = validatePairs(rawPairs, item);
  if (!neutral && pairs.length === 0 && errors.length === 0) {
    errors.push("provide at least one pair, or mark the comment neutral");
  }
  if (errors.length) return res.status(422).json({ errors });

  upsertLabelStmt.run({
    comment_id: item.comment_id,
    annotator,
    neutral: neutral ? 1 : 0,
    pairs: JSON.stringify(pairs),
    updated_at: new Date().toISOString(),
  });
  res.json({ ok: true, neutral, pairs });
});

app.get("/api/progress", (_req, res) => {
  const total = loadGoldItems().length;
  const counts = Object.fromEntries(SCHEMA.annotators.map((a) => [a, 0]));
  for (const row of progressStmt.all()) counts[row.annotator] = row.n;
  res.json({ total, counts });
});

// ── Adjudication: A/B discuss the 173 disagreements and settle
// on the official gold call (spec: "adjudicated by discussion"). ──────────
//
// Meaningless with a single annotator, and its export would overwrite the
// 300-comment adjudicated.jsonl — so it's off in human-gold mode.
app.use("/api/adjudication", (_req, res, next) => {
  if (!SCHEMA.adjudication) {
    return res.status(404).json({ error: "adjudication is disabled in single-annotator mode" });
  }
  next();
});

app.get("/api/adjudication", (_req, res) => {
  const ids = disagreementIds();
  const byId = new Map(loadGoldItems().map((it) => [it.comment_id, it]));
  const items = ids.map((cid) => {
    const item = byId.get(cid);
    const adj = adjudicationRow(cid);
    return {
      comment_id: cid,
      stratum: item.stratum,
      topic_bucket: item.topic_bucket || null,
      resolved: adj !== null,
      final: adj?.final ?? null,
    };
  });
  res.json(items);
});

app.get("/api/adjudication/progress", (_req, res) => {
  const ids = disagreementIds();
  const resolved = ids.filter((cid) => adjudicationRow(cid) !== null).length;
  res.json({ total: ids.length, resolved });
});

app.get("/api/adjudication/:id", (req, res) => {
  const item = getGoldItem(req.params.id);
  if (!item) return res.status(404).json({ error: "unknown comment_id" });
  if (!disagreementIds().includes(item.comment_id)) {
    return res.status(404).json({ error: "this comment has no disagreement to adjudicate" });
  }
  res.json({
    comment_id: item.comment_id,
    stratum: item.stratum,
    topic_bucket: item.topic_bucket || null,
    post_title: item.post_title,
    post_selftext: item.post_selftext,
    post_is_self: item.post_is_self,
    parent_body: item.parent_body,
    body: item.body,
    author: item.author,
    created_utc: item.created_utc,
    score: item.score,
    a: myLabel(item.comment_id, "A"),
    b: myLabel(item.comment_id, "B"),
    consensus: getConsensusPrediction(item.comment_id),
    adjudication: adjudicationRow(item.comment_id),
    messages: listMessagesStmt.all(item.comment_id),
  });
});

app.get("/api/adjudication/:id/messages", (req, res) => {
  res.json(listMessagesStmt.all(req.params.id));
});

app.post("/api/adjudication/:id/messages", (req, res) => {
  const author = requireAnnotator(req, res);
  if (!author) return;
  const body = (req.body.body || "").trim();
  if (!body) return res.status(400).json({ error: "message body is required" });
  const createdAt = new Date().toISOString();
  insertMessageStmt.run(req.params.id, author, body, createdAt);
  res.json(listMessagesStmt.all(req.params.id));
});

app.put("/api/adjudication/:id", (req, res) => {
  const resolvedBy = requireAnnotator(req, res);
  if (!resolvedBy) return;
  const item = getGoldItem(req.params.id);
  if (!item) return res.status(404).json({ error: "unknown comment_id" });

  const final = String(req.body.final || "").toUpperCase();
  let neutral, pairs;
  if (final === "A" || final === "B") {
    const label = myLabel(item.comment_id, final === "A" ? "A" : "B");
    if (!label) return res.status(422).json({ errors: [`${final} has no label for this comment`] });
    ({ neutral, pairs } = label);
  } else if (final === "NEUTRAL") {
    neutral = true;
    pairs = [];
  } else if (final === "CUSTOM") {
    const rawPairs = req.body.pairs || [];
    const validated = validatePairs(rawPairs, item);
    if (validated.errors.length) return res.status(422).json({ errors: validated.errors });
    pairs = validated.pairs;
    neutral = pairs.length === 0;
  } else {
    return res.status(400).json({ error: "final must be one of A, B, NEUTRAL, CUSTOM" });
  }

  upsertAdjudicationStmt.run({
    comment_id: item.comment_id,
    final,
    neutral: neutral ? 1 : 0,
    pairs: JSON.stringify(pairs),
    resolved_by: resolvedBy,
    updated_at: new Date().toISOString(),
  });
  res.json(adjudicationRow(item.comment_id));
});

// Merges the 127 auto-agreed comments with the resolved disagreements into
// the official gold set — same output path/shape as
// `python -m pipeline.gold_report --apply-worklist`, so downstream tooling
// (pipeline/gold_report.py --score-silver) keeps working unchanged.
app.post("/api/adjudication/export", (_req, res) => {
  const disagreements = new Set(disagreementIds());
  const unresolved = [...disagreements].filter((cid) => adjudicationRow(cid) === null);
  if (unresolved.length) {
    return res.status(409).json({
      error: `${unresolved.length} comment(s) still need a final call before the gold set can be exported`,
      unresolved,
    });
  }

  const ids = loadGoldItems().map((it) => it.comment_id).sort();
  const lines = ids.map((cid) => {
    const rec = disagreements.has(cid) ? adjudicationRow(cid) : myLabel(cid, "A");
    return JSON.stringify({ comment_id: cid, neutral: rec.neutral, pairs: rec.pairs });
  });
  fs.mkdirSync(path.dirname(ADJUDICATED_PATH), { recursive: true });
  fs.writeFileSync(ADJUDICATED_PATH, lines.join("\n") + "\n", "utf-8");
  res.json({ ok: true, path: path.relative(REPO_ROOT, ADJUDICATED_PATH), n: ids.length });
});

const PORT = process.env.PORT || 4000;
app.listen(PORT, () => console.log(`[gold-tool] server listening on :${PORT}`));
