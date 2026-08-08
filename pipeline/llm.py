"""Minimal LLM client: concurrent, resumable labelling over any OpenAI-compatible endpoint.

This replaces the private wrapper the study originally ran on. It is deliberately
small — the interesting parts of this repository are the annotation spec, the
consensus protocol and the debate adjudication, not the transport — but it is
complete enough to re-run every model-calling step here without a vendor SDK.

No provider is assumed. Point it at anything that speaks the OpenAI chat
completions API:

    export LLM_BASE_URL=https://your-endpoint/v1
    export LLM_API_KEY=...

Models are named with whatever slug your endpoint expects; the slugs recorded in
the released labels (`deepseek/deepseek-v4-flash`, `xiaomi/mimo-v2.5`,
`minimax/minimax-m3`, `deepseek/deepseek-v4-pro`, `anthropic/claude-sonnet-5`,
`google/gemini-3.5-flash`) are the ones used for the study and follow the
provider/model convention of an aggregator.

Three entry points:

    make_client()                    an OpenAI-compatible client from the env
    call_chat(client, msgs, model)   one call -> (content, usage, finish_reason)
    label(items, build_messages...)  concurrent + resumable map over items

`label()` is the workhorse. It appends one JSON line per item to `out_path`:

    {"id", "parsed", "raw_output", "error", "cost_usd", "ts"}

and on a rerun it skips ids already present, so a killed run resumes for free
and the file is the unit of progress. Costs are recorded only if the endpoint
reports them (some aggregators return `usage.cost`); otherwise the field is null
and nothing downstream depends on it.
"""
from __future__ import annotations

import json
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone

# Unset -> whatever the installed `openai` client defaults to. No provider is
# named or preferred anywhere in this module.
DEFAULT_BASE_URL = os.environ.get("LLM_BASE_URL") or None
DEFAULT_TIMEOUT = float(os.environ.get("LLM_TIMEOUT", "180"))
MAX_RETRIES = int(os.environ.get("LLM_MAX_RETRIES", "5"))


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


# ── Client ───────────────────────────────────────────────────────────────────

def make_client(base_url: str | None = None, api_key: str | None = None):
    """An OpenAI-compatible client. Requires the `openai` package and a key."""
    try:
        from openai import OpenAI
    except ImportError as e:  # pragma: no cover - install-time guidance
        raise SystemExit(
            "the `openai` package is required for live calls: pip install openai\n"
            "(every offline path in this repo runs under --mock and needs no key)"
        ) from e
    key = api_key or os.environ.get("LLM_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not key:
        raise SystemExit("set LLM_API_KEY (or use --mock, which needs no key)")
    url = base_url or DEFAULT_BASE_URL
    kwargs = {"api_key": key, "timeout": DEFAULT_TIMEOUT}
    if url:
        kwargs["base_url"] = url
    return OpenAI(**kwargs)


# ── JSON extraction ──────────────────────────────────────────────────────────

_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)


def extract_json(text: str | None):
    """First JSON object in a model reply, or None.

    Models wrap JSON in prose or fences even when told not to, and reasoning
    models prepend commentary. Try the whole string, then any fenced block, then
    the first brace-balanced span (string-aware, so a `}` inside a quoted value
    doesn't end the scan early).
    """
    if not text:
        return None
    for candidate in (text, *(m.group(1) for m in _FENCE_RE.finditer(text))):
        try:
            return json.loads(candidate.strip())
        except (json.JSONDecodeError, AttributeError):
            pass
    start = text.find("{")
    while start != -1:
        depth, in_str, esc = 0, False, False
        for i in range(start, len(text)):
            c = text[i]
            if in_str:
                if esc:
                    esc = False
                elif c == "\\":
                    esc = True
                elif c == '"':
                    in_str = False
            elif c == '"':
                in_str = True
            elif c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start:i + 1])
                    except json.JSONDecodeError:
                        break
        start = text.find("{", start + 1)
    return None


# ── Single call ──────────────────────────────────────────────────────────────

def call_chat(client, messages, model, *, max_tokens: int | None = None,
              temperature: float = 0.0, timeout: float = DEFAULT_TIMEOUT,
              extra_body: dict | None = None):
    """One chat completion -> (content, usage, finish_reason).

    `usage` is always a dict with prompt_tokens / completion_tokens / cost;
    values are None when the endpoint does not report them. Transient failures
    are retried with exponential backoff; the last exception propagates.
    """
    kwargs: dict = {"model": model, "messages": messages, "temperature": temperature}
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens
    if extra_body:
        kwargs["extra_body"] = extra_body
    if timeout is not None:
        kwargs["timeout"] = timeout

    last = None
    for attempt in range(MAX_RETRIES):
        try:
            resp = client.chat.completions.create(**kwargs)
            break
        except TypeError:
            # A stub client (the --mock paths) may not accept timeout/extra_body.
            kwargs.pop("timeout", None)
            kwargs.pop("extra_body", None)
            resp = client.chat.completions.create(**kwargs)
            break
        except Exception as e:  # noqa: BLE001 — retry anything transport-shaped
            last = e
            if attempt == MAX_RETRIES - 1:
                raise
            time.sleep(min(2 ** attempt, 30))
    else:  # pragma: no cover - loop always breaks or raises
        raise last  # type: ignore[misc]

    choice = resp.choices[0]
    content = getattr(choice.message, "content", None)
    finish = getattr(choice, "finish_reason", None)
    raw = getattr(resp, "usage", None)
    usage = {
        "prompt_tokens": getattr(raw, "prompt_tokens", None) if raw else None,
        "completion_tokens": getattr(raw, "completion_tokens", None) if raw else None,
        # Aggregators may attach a per-call price; plain OpenAI-compatible
        # servers do not. Nothing downstream requires it.
        "cost": getattr(raw, "cost", None) if raw else None,
    }
    return content, usage, finish


# ── Concurrent, resumable map ────────────────────────────────────────────────

def _done_ids(out_path: str) -> set[str]:
    done: set[str] = set()
    if not os.path.exists(out_path):
        return done
    with open(out_path, encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("id") is not None and not rec.get("error"):
                done.add(str(rec["id"]))
    return done


def label(items, build_messages, *, id_of, out_path: str, model: str,
          client=None, max_workers: int = 8, temperature: float = 0.0,
          parse=extract_json, max_tokens: int | None = None,
          extra_body: dict | None = None) -> list[dict]:
    """Call `model` once per item, appending results to `out_path`.

    `build_messages(item)` returns the chat messages; `id_of(item)` its id.
    Items whose id already has a successful record in `out_path` are skipped, so
    reruns resume. Returns the records produced by *this* call only — callers
    that need the full set (after a resume) re-read `out_path`.
    """
    if client is None:
        client = make_client()
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)

    done = _done_ids(out_path)
    todo = [it for it in items if str(id_of(it)) not in done]
    if done:
        print(f"[llm] {model}: {len(done)} already done, {len(todo)} to call",
              flush=True)
    if not todo:
        return []

    write_lock = threading.Lock()
    out_file = open(out_path, "a", encoding="utf-8")
    results: list[dict] = []

    def one(item) -> dict:
        cid = str(id_of(item))
        try:
            content, usage, finish = call_chat(
                client, build_messages(item), model, max_tokens=max_tokens,
                temperature=temperature, extra_body=extra_body)
            rec = {"id": cid, "parsed": parse(content), "raw_output": content,
                   "error": None if content else f"empty output (finish={finish})",
                   "cost_usd": usage.get("cost"), "ts": _now()}
        except Exception as e:  # noqa: BLE001 — one bad item must not kill the run
            rec = {"id": cid, "parsed": None, "raw_output": None,
                   "error": f"{type(e).__name__}: {e}", "cost_usd": None,
                   "ts": _now()}
        with write_lock:
            out_file.write(json.dumps(rec, ensure_ascii=False) + "\n")
            out_file.flush()
        return rec

    try:
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = [pool.submit(one, it) for it in todo]
            for i, fut in enumerate(as_completed(futures), 1):
                rec = fut.result()
                results.append(rec)
                if i % 50 == 0 or i == len(todo):
                    n_err = sum(1 for r in results if r.get("error"))
                    print(f"[llm] {model}: {i}/{len(todo)} ({n_err} errors)",
                          flush=True)
    finally:
        out_file.close()
    return results
