"""Span normalization and source derivation (spec §4).

Spans must be verbatim, but "verbatim" has to survive the noise Reddit and LLMs
introduce: markdown escapes (\\_ \\* \\[), smart quotes, HTML entities, casing and
whitespace differences. `normalize()` folds all of that away; containment is then
checked on normalized text.

`find_span_source()` derives where a span came from, with the spec's precedence
comment > parent > post — the closest text to the author wins.
"""
from __future__ import annotations

import html
import re

_WS_RE = re.compile(r"\s+")

# Folds curly quotes AND straight " onto straight ' — models reliably copy a
# "double-quoted" source phrase back as 'single-quoted' (2026-07-04 audit),
# a punctuation-only verbatim mismatch, not a content change.
_QUOTE_MAP = str.maketrans({
    "‘": "'", "’": "'", "‚": "'", "‛": "'",
    "“": "'", "”": "'", "„": "'",
    '"': "'",
    "–": "-", "—": "-", "−": "-",
    " ": " ",
    "…": "...",
})


def normalize(text: str | None) -> str:
    """Fold escaping/quote/whitespace/case noise; keep word content intact."""
    if not text:
        return ""
    text = html.unescape(text)
    # Backslashes only ever appear as markdown escapes (possibly stacked, e.g. the
    # shrug emoticon ¯\\\_(ツ)\_/¯); they are never semantic for span matching.
    text = text.replace("\\", "")
    text = text.translate(_QUOTE_MAP)
    text = _WS_RE.sub(" ", text).strip()
    return text.casefold()


def span_in(span: str | None, text: str | None) -> bool:
    """True if `span` occurs verbatim (normalized) inside `text`."""
    if not span or not text:
        return False
    return normalize(span) in normalize(text)


# Spec §4 precedence: the closest text to the author wins.
SOURCE_PRECEDENCE = ("comment", "parent", "post")


def find_span_source(span: str | None, sources: dict[str, str | None]) -> str | None:
    """Derive which source text a span was copied from, or None if none match.

    `sources` maps source name -> text, e.g. {"comment": body, "parent": parent_body,
    "post": title + selftext}. Missing/None texts are skipped.
    """
    for name in SOURCE_PRECEDENCE:
        if span_in(span, sources.get(name)):
            return name
    return None
