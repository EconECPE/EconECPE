import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const REPO_ROOT = path.resolve(__dirname, "..", "..");
const SILVER_DIR = path.join(REPO_ROOT, "data", "labels", "silver");

// GOLD_SAMPLE swaps the item set the tool serves — used by the human-gold pass
// to restrict labeling to data/samples/human_gold_100_seed7.jsonl (a stratified
// 100-comment subset of the 300).
const GOLD_PATH = process.env.GOLD_SAMPLE
  ? path.resolve(process.env.GOLD_SAMPLE)
  : path.join(REPO_ROOT, "data", "samples", "gold_300_seed7.jsonl");

// One entry per model we want the "what the model said" panel to show, in
// display order. A source is either slug+dir (the annotate.py convention:
// <dir>/<slug>.labels.jsonl plus an incremental <slug>.raw.jsonl) or an
// explicit `labels` path for outputs that don't follow it. Add one here and it
// shows up automatically — no other code changes needed.
//
// Every source below covers all 100 comments of the human-gold sample except
// minimax-m3 (57/100), which shows "pending" on the rest.
const GOLD_DIR = path.join(REPO_ROOT, "data", "labels", "gold");
const OUTPUTS_DIR = path.join(REPO_ROOT, "outputs");

const MODELS = [
  // The two frontier families whose labels became the 300-comment gold set.
  { model: "claude-sonnet-5 (gold A)", slug: "anthropic_claude-sonnet-5", dir: GOLD_DIR },
  { model: "gemini-3.5-flash (gold B)", slug: "google_gemini-3.5-flash", dir: GOLD_DIR },
  // Their debate-adjudicated merge — the current LLM "gold standard".
  { model: "LLM-adjudicated gold", labels: path.join(GOLD_DIR, "adjudicated.jsonl") },
  // The three silver consensus-pipeline models (run over the full 20k).
  { model: "deepseek/deepseek-v4-flash", slug: "deepseek_deepseek-v4-flash", dir: SILVER_DIR },
  { model: "xiaomi/mimo-v2.5", slug: "xiaomi_mimo-v2.5", dir: SILVER_DIR },
  { model: "minimax/minimax-m3", slug: "minimax_minimax-m3", dir: SILVER_DIR },
  // The distilled students actually deployed on the corpus.
  { model: "Qwen3.5-9B student (fine-tuned)",
    labels: path.join(OUTPUTS_DIR, "qwen3.5-9b-silver_clean-lr1e-4",
                      "final-merged-vlm", "gold_preds.new.jsonl") },
  { model: "DeBERTaV3 tagger", labels: path.join(OUTPUTS_DIR, "deberta-v3-tagger",
                                                 "epoch2.gold_preds.jsonl") },
];

function readJsonl(p) {
  if (!fs.existsSync(p)) return [];
  return fs.readFileSync(p, "utf-8")
    .split("\n")
    .filter(Boolean)
    .map((line) => {
      try { return JSON.parse(line); } catch { return null; }
    })
    .filter(Boolean);
}

function statMtime(p) {
  try { return fs.statSync(p).mtimeMs; } catch { return 0; }
}

let goldItems = null;
export function loadGoldItems() {
  if (!goldItems) {
    goldItems = readJsonl(GOLD_PATH);
    if (!goldItems.length) {
      throw new Error(`no gold items found at ${GOLD_PATH} — run ` +
        "'python -m pipeline.sample --gold' first");
    }
  }
  return goldItems;
}

export function getGoldItem(commentId) {
  return loadGoldItems().find((it) => it.comment_id === commentId) || null;
}

// Predictions cache, one slot per model: re-parses only when the underlying
// JSONL file's mtime changes, since annotation runs keep appending in the
// background.
const caches = new Map(MODELS.map((m) => [m.model, {
  // Sources given an explicit `labels` path have no incremental raw file;
  // rawPath points at a nonexistent file, which readJsonl treats as empty.
  rawPath: m.slug ? path.join(m.dir, `${m.slug}.raw.jsonl`) : "",
  labelsPath: m.labels ?? path.join(m.dir, `${m.slug}.labels.jsonl`),
  rawMtime: 0, raw: new Map(), labelsMtime: 0, labels: new Map(),
}]));

function refresh(cache) {
  const rawMtime = statMtime(cache.rawPath);
  if (rawMtime !== cache.rawMtime) {
    const map = new Map();
    for (const rec of readJsonl(cache.rawPath)) {
      if (rec?.id) map.set(String(rec.id), rec); // later lines win (resume reruns)
    }
    cache.raw = map;
    cache.rawMtime = rawMtime;
  }
  const labelsMtime = statMtime(cache.labelsPath);
  if (labelsMtime !== cache.labelsMtime) {
    const map = new Map();
    for (const rec of readJsonl(cache.labelsPath)) {
      if (rec?.comment_id) map.set(String(rec.comment_id), rec);
    }
    cache.labels = map;
    cache.labelsMtime = labelsMtime;
  }
}

// Mirrors the precedence pipeline/annotate.py uses when merging: the
// validated `.labels.jsonl` entry wins once it exists; until the full run
// finishes, fall back to the incrementally-written `.raw.jsonl`.
function predictionFor(modelName, cache, commentId) {
  refresh(cache);
  const validated = cache.labels.get(commentId);
  if (validated) {
    // Exports that aren't annotate.py output (adjudicated gold, student preds)
    // carry no `valid` field — they're already-validated results, so absence
    // means valid, not invalid.
    const ok = validated.valid ?? true;
    return {
      model: modelName,
      status: ok ? (validated.repaired ? "validated_repaired" : "validated") : "invalid",
      pairs: validated.pairs || [],
      errors: validated.errors || [],
    };
  }
  const raw = cache.raw.get(commentId);
  if (raw) {
    if (raw.error) {
      return { model: modelName, status: "error", pairs: [], errors: [raw.error] };
    }
    return {
      model: modelName,
      status: "raw_unvalidated",
      pairs: raw.parsed?.pairs || [],
      errors: [],
    };
  }
  return { model: modelName, status: "pending", pairs: [], errors: [] };
}

// The judge (deepseek-v4-pro) doesn't produce its own full annotation — it
// only rules on individual disputed fields within pipeline/consensus.py's
// voted pair clusters. So there's no "judge.raw.jsonl" comparable to the
// other three models. What's actually useful to show is the FINAL voted
// result from consensus.jsonl, which is where the judge's verdicts land
// (merged into whichever pairs had a field with agreement: "judge").
const CONSENSUS_PATH = path.join(SILVER_DIR, "consensus.jsonl");
const consensusCache = { mtime: 0, byId: new Map() };

function refreshConsensus() {
  const mtime = statMtime(CONSENSUS_PATH);
  if (mtime !== consensusCache.mtime) {
    const map = new Map();
    for (const rec of readJsonl(CONSENSUS_PATH)) {
      if (rec?.comment_id) map.set(rec.comment_id, rec);
    }
    consensusCache.byId = map;
    consensusCache.mtime = mtime;
  }
}

const CONSENSUS_LABEL = "Silver consensus (judge: deepseek-v4-pro)";

export function getConsensusPrediction(commentId) {
  refreshConsensus();
  const rec = consensusCache.byId.get(commentId);
  if (!rec) {
    return { model: CONSENSUS_LABEL, status: "pending", pairs: [], errors: [] };
  }
  return {
    model: CONSENSUS_LABEL,
    status: "validated",
    pairs: (rec.pairs || []).filter((p) => p.included),
    errors: [],
  };
}

// The three independent models' predictions for one comment, in MODELS order.
// The consensus/judge result is fetched separately via getConsensusPrediction
// — it's not "another model", it's the synthesized answer, so the UI shows it
// in its own prominent spot rather than as a 4th tile among raw model reads.
export function getModelPredictions(commentId) {
  return MODELS.map(({ model }) => predictionFor(model, caches.get(model), commentId));
}

// ── Deduplicated proposals ──────────────────────────────────────────────────
//
// Nine panels mostly repeat each other, and reading them all to find the one
// pair worth accepting is the slow part. So collapse every proposed pair across
// all sources (including the consensus) into distinct proposals, keyed on the
// fields that define a pair, and report who backs each. One click on the
// best-supported proposal replaces reading nine panels.
//
// Spans are keyed on normalized text so trivial whitespace/case differences
// don't split an otherwise identical proposal into two rows.
const norm = (s) => (s || "").toLowerCase().replace(/\s+/g, " ").trim();

export function getProposals(commentId) {
  const sources = [...getModelPredictions(commentId), getConsensusPrediction(commentId)];
  const byKey = new Map();
  let nSaidNeutral = 0;
  let nAnswered = 0;

  for (const src of sources) {
    if (src.status === "pending" || src.status === "error") continue;
    nAnswered += 1;
    if (!src.pairs.length) nSaidNeutral += 1;
    for (const p of src.pairs) {
      const key = [norm(p.emotion_span), norm(p.cause_span), p.emotion,
                   p.cause_category, p.target_asset].join(" ");
      if (!byKey.has(key)) {
        byKey.set(key, { pair: { ...p }, backers: [], intensities: [], sarcasm: 0 });
      }
      const slot = byKey.get(key);
      slot.backers.push(src.model);
      slot.intensities.push(typeof p.intensity === "number" ? p.intensity : 0.5);
      if (p.sarcasm) slot.sarcasm += 1;
    }
  }

  const proposals = [...byKey.values()].map(({ pair, backers, intensities, sarcasm }) => ({
    ...pair,
    // Median intensity across backers rather than whichever source happened to
    // be first, and sarcasm by majority of the sources proposing this pair.
    intensity: intensities.slice().sort((a, b) => a - b)[Math.floor(intensities.length / 2)],
    sarcasm: sarcasm * 2 > backers.length,
    backers,
    support: backers.length,
  })).sort((a, b) => b.support - a.support);

  return { proposals, nAnswered, nSaidNeutral, nSources: sources.length };
}
