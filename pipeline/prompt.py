"""Build the annotation chat messages: system + few-shot + formatted item.

The system prompt (prompts/system.md) is the distilled spec; the few-shot
examples (prompts/fewshot.json) are the spec's worked examples E1–E5, sent as
alternating user/assistant turns so every model sees the exact output contract.

An "item" is one sampled comment dict with its pre-assembled context fields:
    {comment_id, body, post_title, post_selftext, parent_body, ...}
"""
from __future__ import annotations

import json
import os
from functools import lru_cache

import config

SYSTEM_PATH = os.path.join(config.PROMPTS_DIR, "system.md")
FEWSHOT_PATH = os.path.join(config.PROMPTS_DIR, "fewshot.json")


@lru_cache(maxsize=1)
def system_prompt() -> str:
    with open(SYSTEM_PATH, encoding="utf-8") as f:
        return f.read()


@lru_cache(maxsize=1)
def fewshot() -> tuple:
    with open(FEWSHOT_PATH, encoding="utf-8") as f:
        return tuple(json.load(f))


def format_input(item: dict) -> str:
    """Render one comment + context as the user message. Field order and labels
    are part of the prompt contract — few-shot inputs use the same rendering."""
    parts = [f"POST TITLE: {item.get('post_title') or '(unavailable)'}"]
    if item.get("post_selftext"):
        parts.append(f"POST BODY: {item['post_selftext']}")
    if item.get("parent_body"):
        parts.append(f"PARENT COMMENT: {item['parent_body']}")
    parts.append(f"COMMENT (annotate this): {item.get('body') or item.get('comment')}")
    return "\n\n".join(parts)


def build_messages(item: dict) -> list[dict]:
    msgs = [{"role": "system", "content": system_prompt()}]
    for ex in fewshot():
        msgs.append({"role": "user", "content": format_input(ex["input"])})
        msgs.append({"role": "assistant",
                     "content": json.dumps(ex["output"], ensure_ascii=False)})
    msgs.append({"role": "user", "content": format_input(item)})
    return msgs


def build_repair_messages(item: dict, raw_output: str, errors: list[str]) -> list[dict]:
    """One-shot repair turn: original conversation + the failed output + errors."""
    msgs = build_messages(item)
    msgs.append({"role": "assistant", "content": raw_output})
    msgs.append({"role": "user", "content": (
        "Your annotation failed validation:\n- "
        + "\n- ".join(errors)
        + "\n\nRe-emit the corrected JSON object only. Remember: spans must be "
          "copied VERBATIM from the comment or its context; follow the field "
          "rules exactly."
    )})
    return msgs
