"""Encode annotation spans as offsets, so the release carries no Reddit text.

The labels of this dataset *are* spans: `emotion_span` and `cause_span` are
verbatim quotations of a comment, its parent, or the linked post. Shipping them
as strings would republish Reddit content — measured over the 300-comment gold
set, the spans alone reproduce 35% of an average comment and 100% of twenty of
them. So the release stores each span as a `(source, start, end, occurrence)`
offset instead, and `tools/rehydrate.py` turns it back into text against a
locally fetched copy of the corpus.

Offsets index into `pipeline.textnorm.normalize(source_text)`, not the raw text,
because that is the space in which spans were validated: normalization folds
markdown escapes, HTML entities, smart quotes, whitespace and case, and roughly
10% of the spans match only after it. Measured on gold + human gold: every span
is recoverable this way, 99.6% of them at a unique offset.

Decoding still returns the *raw* substring, not the normalized one. `index_map`
tracks, for every character of the normalized text, which raw character it came
from, so the original casing and punctuation survive the round trip.

`encode_record` is build-time (it needs the corpus); `decode_record` ships and
runs against whatever the user rehydrated.
"""
from __future__ import annotations

import html
import re

_WS_RE = re.compile(r"\s+")

# Mirrors pipeline.textnorm._QUOTE_MAP. Kept as an explicit dict rather than a
# translation table so the index-tracking walk can see the replacement lengths.
_QUOTE_MAP = {
    "‘": "'", "’": "'", "‚": "'", "‛": "'",
    "“": "'", "”": "'", "„": "'",
    '"': "'",
    "–": "-", "—": "-", "−": "-",
    " ": " ",
    "…": "...",
}

# CPython's html.unescape charref pattern, so step 1 below decodes exactly the
# entities the stdlib does. Any divergence is caught by the assertion in
# normalize_indexed rather than silently shifting every later offset.
_CHARREF_RE = re.compile(
    r"&(#[0-9]+;?|#[xX][0-9a-fA-F]+;?|[^\t\n\f <&#;]{1,32};?)"
)

SOURCES = ("comment", "parent", "post")


class SpanEncodeError(RuntimeError):
    """A span could not be located in any of its candidate source texts."""


def normalize_indexed(text: str) -> tuple[str, list[int]]:
    """Normalize `text` and return it with a per-character map back into `text`.

    The returned string is byte-for-byte what `pipeline.textnorm.normalize`
    produces — asserted by the caller — and `index_map[i]` is the index in
    `text` of the character that produced `normalized[i]`.
    """
    chars: list[str] = []
    src: list[int] = []

    # 1 — HTML entities. A real entity collapses to one character, mapped to the
    #     '&' that started it. The pattern also fires on non-entities — "R&D"
    #     matches the no-semicolon branch and swallows the rest of the word —
    #     where html.unescape hands the text straight back; those must stay
    #     mapped one-to-one or every later offset in the comment shifts.
    pos = 0
    for match in _CHARREF_RE.finditer(text):
        for i in range(pos, match.start()):
            chars.append(text[i])
            src.append(i)
        decoded = html.unescape(match.group(0))
        if decoded == match.group(0):
            for offset, ch in enumerate(decoded):
                chars.append(ch)
                src.append(match.start() + offset)
        else:
            for ch in decoded:
                chars.append(ch)
                src.append(match.start())
        pos = match.end()
    for i in range(pos, len(text)):
        chars.append(text[i])
        src.append(i)

    # 2 — markdown escapes: backslashes are dropped outright.
    # 3 — quote/dash/ellipsis folding, which is 1:1 except '…' -> '...'.
    out_chars: list[str] = []
    out_src: list[int] = []
    for ch, i in zip(chars, src):
        if ch == "\\":
            continue
        replacement = _QUOTE_MAP.get(ch, ch)
        for rch in replacement:
            out_chars.append(rch)
            out_src.append(i)

    # 4 — whitespace runs collapse to a single space, mapped to the run's start.
    ws_chars: list[str] = []
    ws_src: list[int] = []
    run = False
    for ch, i in zip(out_chars, out_src):
        if ch.isspace() or _WS_RE.fullmatch(ch):
            if not run:
                ws_chars.append(" ")
                ws_src.append(i)
                run = True
        else:
            ws_chars.append(ch)
            ws_src.append(i)
            run = False

    # 5 — strip.
    start, end = 0, len(ws_chars)
    while start < end and ws_chars[start] == " ":
        start += 1
    while end > start and ws_chars[end - 1] == " ":
        end -= 1
    ws_chars, ws_src = ws_chars[start:end], ws_src[start:end]

    # 6 — casefold. Almost always 1:1, but 'ß' -> 'ss' and the ligatures are not,
    #     so expand per character and keep the map aligned.
    fin_chars: list[str] = []
    fin_src: list[int] = []
    for ch, i in zip(ws_chars, ws_src):
        for fch in ch.casefold():
            fin_chars.append(fch)
            fin_src.append(i)

    return "".join(fin_chars), fin_src


def sources_of(record: dict) -> dict[str, str]:
    """The three candidate source texts for a rehydrated corpus record."""
    post = ((record.get("post_title") or "") + " " + (record.get("post_selftext") or "")).strip()
    return {
        "comment": record.get("body") or "",
        "parent": record.get("parent_body") or "",
        "post": post,
    }


def encode_span(span: str, texts: dict[str, str]) -> dict:
    """Locate `span` in the candidate texts and return its offset form.

    Precedence is the spec's: comment > parent > post, the closest text to the
    author wins. `occurrence` disambiguates the 0.4% of spans that appear more
    than once in their source.
    """
    from pipeline.textnorm import normalize  # build-time only

    needle = normalize(span)
    if not needle:
        raise SpanEncodeError("empty span")
    for name in SOURCES:
        raw = texts.get(name) or ""
        if not raw:
            continue
        norm, _ = normalize_indexed(raw)
        assert norm == normalize(raw), f"index-tracked normalize diverged on {name!r}"
        if needle not in norm:
            continue
        return {
            "src": name,
            "start": norm.index(needle),
            "end": norm.index(needle) + len(needle),
            "occurrence": norm.count(needle) - 1 if norm.count(needle) > 1 else 0,
        }
    raise SpanEncodeError(f"span not found in any source: {span[:80]!r}")


def decode_span(offset: dict, texts: dict[str, str]) -> str:
    """Turn an offset back into the raw substring of its source text."""
    raw = texts.get(offset["src"]) or ""
    norm, index_map = normalize_indexed(raw)
    start, end = offset["start"], offset["end"]
    if end > len(norm):
        raise SpanEncodeError(
            f"offset {start}:{end} runs past the {len(norm)}-char {offset['src']} text; "
            "the rehydrated copy differs from the one the labels were built on"
        )
    lo = index_map[start]
    hi = index_map[end - 1] + 1
    return raw[lo:hi]


def _walk(record: dict):
    for pair in record.get("pairs") or []:
        for key in ("emotion_span", "cause_span"):
            if pair.get(key) is not None:
                yield pair, key


def encode_record(record: dict, texts: dict[str, str]) -> dict:
    out = dict(record)
    out["pairs"] = [dict(p) for p in record.get("pairs") or []]
    for pair, key in _walk(out):
        pair[key] = encode_span(pair[key], texts)
    return out


def decode_record(record: dict, texts: dict[str, str]) -> dict:
    out = dict(record)
    out["pairs"] = [dict(p) for p in record.get("pairs") or []]
    for pair, key in _walk(out):
        pair[key] = decode_span(pair[key], texts)
    return out
