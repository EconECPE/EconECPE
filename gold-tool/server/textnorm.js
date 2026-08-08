// Port of pipeline/textnorm.py — must stay in lockstep with it (spec §4).
// Normalizes markdown-escape/quote/whitespace/case noise, then checks span
// containment. Source precedence: comment > parent > post (closest wins).

const QUOTE_MAP = {
  "‘": "'", "’": "'", "‚": "'", "‛": "'",
  "“": '"', "”": '"', "„": '"',
  "–": "-", "—": "-", "−": "-",
  " ": " ",
  "…": "...",
};
const QUOTE_RE = new RegExp(`[${Object.keys(QUOTE_MAP).join("")}]`, "g");

const HTML_ENTITIES = { amp: "&", lt: "<", gt: ">", quot: '"', "#39": "'", apos: "'", nbsp: " " };

function unescapeHtml(text) {
  return text.replace(/&(amp|lt|gt|quot|#39|apos|nbsp);/g, (_, e) => HTML_ENTITIES[e]);
}

export function normalize(text) {
  if (!text) return "";
  let t = unescapeHtml(text);
  t = t.replace(/\\/g, "");
  t = t.replace(QUOTE_RE, (ch) => QUOTE_MAP[ch]);
  t = t.replace(/\s+/g, " ").trim();
  return t.toLowerCase();
}

export function spanIn(span, text) {
  if (!span || !text) return false;
  return normalize(text).includes(normalize(span));
}

const SOURCE_PRECEDENCE = ["comment", "parent", "post"];

export function findSpanSource(span, sources) {
  for (const name of SOURCE_PRECEDENCE) {
    if (spanIn(span, sources[name])) return name;
  }
  return null;
}
