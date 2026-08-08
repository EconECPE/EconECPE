"""Validate model annotations against spec/schema.json + the span rules (spec §4).

Two layers:
  1. JSON Schema (structure, enums, null-cause consistency) — authoritative file
     is spec/schema.json.
  2. Span rules — emotion_span must occur verbatim (normalized) in the comment;
     cause_span must occur in comment/parent/post. cause_source is DERIVED from
     the texts and silently corrected when the model mislabels it (recorded in
     `source_fixes`, it is not an error).

`validate_annotation` returns (annotation, errors, source_fixes): errors non-empty
→ the item goes to the repair loop with those messages.
"""
from __future__ import annotations

import json
from functools import lru_cache

import config

from .textnorm import find_span_source, span_in


@lru_cache(maxsize=1)
def _validator():
    from jsonschema import Draft202012Validator
    with open(config.SPEC_SCHEMA, encoding="utf-8") as f:
        schema = json.load(f)
    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(schema)


def _sources_of(item: dict) -> dict:
    post_text = " \n ".join(x for x in (item.get("post_title"), item.get("post_selftext")) if x)
    return {
        "comment": item.get("body") or item.get("comment"),
        "parent": item.get("parent_body"),
        "post": post_text or None,
    }


def validate_annotation(ann: object, item: dict) -> tuple[dict | None, list[str], int]:
    """Check one parsed model output for one item.

    Returns (annotation, errors, source_fixes). `annotation` is the (possibly
    source-corrected) dict, or None if it isn't even a dict. Non-empty `errors`
    means the annotation needs the repair loop.
    """
    if not isinstance(ann, dict):
        return None, [f"output is {type(ann).__name__}, expected a JSON object"], 0

    errors = [f"schema: {e.message}" for e in _validator().iter_errors(ann)]
    fixes = 0
    sources = _sources_of(item)

    for i, pair in enumerate(ann.get("pairs") or []):
        if not isinstance(pair, dict):
            continue
        espan = pair.get("emotion_span")
        if isinstance(espan, str) and not span_in(espan, sources["comment"]):
            errors.append(
                f"pairs[{i}].emotion_span is not a verbatim substring of the comment: "
                f"{espan!r}")
        cspan = pair.get("cause_span")
        if isinstance(cspan, str):
            derived = find_span_source(cspan, sources)
            if derived is None:
                errors.append(
                    f"pairs[{i}].cause_span is not a verbatim substring of the comment, "
                    f"parent comment, or post: {cspan!r}")
            elif pair.get("cause_source") != derived:
                pair["cause_source"] = derived  # spec §4: validator-enforced
                fixes += 1

    return ann, errors, fixes
