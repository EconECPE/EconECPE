// Mirrors pipeline/validate.py's span rules for human-submitted pairs: spans
// must be verbatim, cause_source is derived (never trusted from the client).
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { spanIn, findSpanSource } from "./textnorm.js";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const SCHEMA = JSON.parse(
  fs.readFileSync(path.join(__dirname, "..", "shared", "schema.json"), "utf-8"));

function sourcesOf(item) {
  const post = [item.post_title, item.post_selftext].filter(Boolean).join(" \n ");
  return { comment: item.body, parent: item.parent_body, post: post || null };
}

// Returns { pairs, errors }. `pairs` has cause_source silently corrected to the
// derived value, same as the Python validator does for the model.
export function validatePairs(pairs, item) {
  const errors = [];
  const sources = sourcesOf(item);
  if (!Array.isArray(pairs)) return { pairs: [], errors: ["pairs must be an array"] };

  const out = pairs.map((pair, i) => {
    const p = { ...pair };
    if (!SCHEMA.emotions.includes(p.emotion)) {
      errors.push(`pairs[${i}].emotion: invalid value ${JSON.stringify(p.emotion)}`);
    }
    if (typeof p.emotion_span !== "string" || !p.emotion_span) {
      errors.push(`pairs[${i}].emotion_span is required`);
    } else if (!spanIn(p.emotion_span, sources.comment)) {
      errors.push(`pairs[${i}].emotion_span is not a verbatim substring of the comment`);
    }
    if (p.cause_span !== null && typeof p.cause_span !== "string") {
      errors.push(`pairs[${i}].cause_span must be a string or null`);
    } else if (p.cause_span === null) {
      p.cause_source = null;
      if (p.cause_category !== "unclear") {
        errors.push(`pairs[${i}].cause_category must be "unclear" when cause_span is null`);
      }
    } else {
      const derived = findSpanSource(p.cause_span, sources);
      if (derived === null) {
        errors.push(`pairs[${i}].cause_span is not a verbatim substring of the comment, ` +
          "parent comment, or post");
      } else {
        p.cause_source = derived; // validator-enforced, spec §4
      }
      if (p.cause_category === "unclear") {
        errors.push(`pairs[${i}].cause_category can't be "unclear" when cause_span is set`);
      }
    }
    if (!SCHEMA.causeCategories.includes(p.cause_category)) {
      errors.push(`pairs[${i}].cause_category: invalid value ${JSON.stringify(p.cause_category)}`);
    }
    if (!SCHEMA.targetAssets.includes(p.target_asset)) {
      errors.push(`pairs[${i}].target_asset: invalid value ${JSON.stringify(p.target_asset)}`);
    }
    if (typeof p.intensity !== "number" || p.intensity < 0 || p.intensity > 1) {
      errors.push(`pairs[${i}].intensity must be a number in [0, 1]`);
    }
    if (typeof p.sarcasm !== "boolean") {
      errors.push(`pairs[${i}].sarcasm must be a boolean`);
    }
    return p;
  });

  return { pairs: out, errors };
}
