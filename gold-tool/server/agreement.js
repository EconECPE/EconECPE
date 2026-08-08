// Port of pipeline/report.py's span_iou/match_pairs + pipeline/gold_report.py's
// disagreement check, so the server can tell which of the 300 double-coded
// comments need adjudication without shelling out to Python. Must stay in
// lockstep with those (spec-adjacent scoring logic).
import { normalize } from "./textnorm.js";

// Mirrors pipeline/report.py's _EDGE_PUNCT stripping: a trailing '.'/'?' on the
// last word is a selection-boundary difference, not a content difference, and
// must not cost token overlap.
const EDGE_PUNCT = /^[.,;:!?'"()[\]{}\-…]+|[.,;:!?'"()[\]{}\-…]+$/g;

function tokens(span) {
  if (!span) return new Set();
  return new Set(
    normalize(span).split(" ").map((w) => w.replace(EDGE_PUNCT, "")).filter(Boolean));
}

export function spanIoU(a, b) {
  const ta = tokens(a);
  const tb = tokens(b);
  if (ta.size === 0 || tb.size === 0) return 0;
  let inter = 0;
  for (const t of ta) if (tb.has(t)) inter++;
  const union = new Set([...ta, ...tb]).size;
  return inter / union;
}

// Greedy 1:1 matching of two pair lists by emotion_span token IoU.
export function matchPairs(pa, pb, thr = 0.5) {
  const cands = [];
  pa.forEach((x, i) => {
    pb.forEach((y, j) => {
      cands.push([spanIoU(x.emotion_span, y.emotion_span), i, j]);
    });
  });
  cands.sort((a, b) => b[0] - a[0]);
  const usedA = new Set();
  const usedB = new Set();
  const matches = [];
  for (const [iou, i, j] of cands) {
    if (iou < thr) break;
    if (usedA.has(i) || usedB.has(j)) continue;
    usedA.add(i);
    usedB.add(j);
    matches.push([pa[i], pb[j]]);
  }
  return matches;
}

// Same criterion as pipeline/gold_report.py's compare_annotators: neutral
// flag mismatch, pair-count mismatch (some pair went unmatched), or a field
// mismatch (emotion/cause_category) on a matched pair.
export function hasDisagreement(a, b, thr = 0.5) {
  if (a.neutral !== b.neutral) return true;
  const pa = a.pairs || [];
  const pb = b.pairs || [];
  const ms = matchPairs(pa, pb, thr);
  if (ms.length !== pa.length || ms.length !== pb.length) return true;
  return ms.some(([x, y]) => x.emotion !== y.emotion || x.cause_category !== y.cause_category);
}
