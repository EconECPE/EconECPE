"""Annotate a sample file with one model: call → validate → one repair pass → merge.

Plumbing (concurrency, retries, resume) is `pipeline.llm.label()`; this module
adds the EconECPE domain layer: prompt building, schema+span validation, the
single repair turn (spec §4), and the merged per-model output file.

Outputs under data/labels/<run>/:
    <model-slug>.raw.jsonl      phase-A raw calls   (resumable)
    <model-slug>.repair.jsonl   phase-B repair calls (resumable)
    <model-slug>.labels.jsonl   merged, validated — one line per item:
        {comment_id, stratum, model, pairs, valid, repaired, source_fixes,
         errors, cost_usd}

    python -m pipeline.annotate --sample data/samples/pilot_250_seed7.jsonl \\
        --run pilot --model deepseek/deepseek-v4-flash [--limit 20] [--mock]

--mock runs the whole flow against a canned fake client (no key, no cost): it
answers {"pairs": []} but every 4th item gets a pair whose spans are cut from
the real comment/title, exercising validation and cause_source derivation.
"""
from __future__ import annotations

import argparse
import json
import os

import config

from .llm import extract_json, label
from .prompt import build_messages, build_repair_messages
from .validate import validate_annotation


def _slug(model: str) -> str:
    return model.replace("/", "_").replace(":", "_")


def load_sample(path: str, limit: int | None = None) -> list[dict]:
    items = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if limit is not None and len(items) >= limit:
                break
            items.append(json.loads(line))
    return items


# ── Mock client (key-free smoke tests) ───────────────────────────────────────

class _FakeCompletions:
    def __init__(self):
        self.calls = 0

    def create(self, *, messages, **kw):
        self.calls += 1
        from types import SimpleNamespace as NS
        if "Your annotation failed validation" in messages[-1]["content"]:
            return NS(choices=[NS(message=NS(content='{"pairs": []}'),
                                  finish_reason="stop")], usage=None)  # repair succeeds
        if self.calls % 10 == 0:  # paraphrased span → must trigger the repair loop
            bad = ('{"pairs": [{"emotion": "surprise", "emotion_span": "not in the text", '
                   '"cause_span": null, "cause_source": null, "cause_category": "unclear", '
                   '"target_asset": "none", "intensity": 0.5, "sarcasm": false}]}')
            return NS(choices=[NS(message=NS(content=bad), finish_reason="stop")],
                      usage=None)
        comment = messages[-1]["content"].rsplit("COMMENT (annotate this): ", 1)[-1]
        title_line = messages[-1]["content"].split("\n", 1)[0]
        title = title_line.replace("POST TITLE: ", "")
        ann: dict = {"pairs": []}
        if self.calls % 4 == 0:
            pair = {
                "emotion": "pessimism_despair",
                "emotion_span": " ".join(comment.split()[:6]),
                "cause_span": None, "cause_source": None,
                "cause_category": "unclear", "target_asset": "none",
                "intensity": 0.5, "sarcasm": False,
            }
            if title and title != "(unavailable)":
                pair.update(cause_span=" ".join(title.split()[:5]),
                            cause_source="comment",  # wrong on purpose → validator fixes
                            cause_category="other")
            ann["pairs"] = [pair]

        from types import SimpleNamespace as NS  # minimal OpenAI-shaped response
        return NS(choices=[NS(message=NS(content=json.dumps(ann)),
                              finish_reason="stop")],
                  usage=None)


class FakeClient:
    """OpenAI-shaped stub: deterministic, free, offline."""
    def __init__(self):
        class _Chat:
            completions = _FakeCompletions()
        self.chat = _Chat()


# ── Orchestration ────────────────────────────────────────────────────────────

def annotate(sample_path: str, model: str, run: str = "pilot",
             limit: int | None = None, mock: bool = False,
             max_workers: int | None = None, reasoning: str | None = None,
             max_tokens: int | None = None) -> str:
    items = load_sample(sample_path, limit)
    by_id = {it["comment_id"]: it for it in items}
    run_dir = os.path.join(config.LABELS_DIR, run)
    os.makedirs(run_dir, exist_ok=True)
    slug = _slug(model)
    raw_path = os.path.join(run_dir, f"{slug}.raw.jsonl")
    repair_path = os.path.join(run_dir, f"{slug}.repair.jsonl")
    out_path = os.path.join(run_dir, f"{slug}.labels.jsonl")

    client = FakeClient() if mock else None
    common = dict(model=model, client=client,
                  max_workers=max_workers or config.ANNOTATE_MAX_WORKERS,
                  temperature=0.0, parse=extract_json,
                  # Reasoning models think inside the completion budget, so a
                  # small one starves them on hard comments (observed:
                  # mimo-v2.5 finish_reason=length → empty output; even 12_000
                  # still starves ~6% of mimo-medium calls, 2026-07-04).
                  max_tokens=max_tokens or 12_000)
    if reasoning:
        # Reasoning-effort passthrough. Endpoints that don't understand the
        # field ignore it; some (e.g. deepseek-v4-flash) want "high".
        common["extra_body"] = {"reasoning": {"effort": reasoning}}

    # Phase A — first attempt for every item.
    results = label(items, build_messages, id_of=lambda it: it["comment_id"],
                    out_path=raw_path, **common)
    # Resumed runs skip already-done ids; reload the full raw file for merging.
    raw_by_id = _load_records(raw_path)

    # Validate; collect repair candidates.
    merged: dict[str, dict] = {}
    to_repair: list[dict] = []
    for cid, rec in raw_by_id.items():
        item = by_id.get(cid)
        if item is None:
            continue
        entry = _entry(item, model)
        if rec.get("error"):
            entry.update(valid=False, errors=[f"call: {rec['error']}"])
        else:
            ann, errors, fixes = validate_annotation(rec.get("parsed"), item)
            entry.update(pairs=(ann or {}).get("pairs"), errors=errors,
                         source_fixes=fixes, valid=not errors,
                         cost_usd=rec.get("cost_usd"))
            if errors:
                to_repair.append({**item, "_raw": rec.get("raw_output") or "",
                                  "_errors": errors})
        merged[cid] = entry

    # Phase B — one repair turn for invalid outputs (spec §4).
    if to_repair:
        print(f"[annotate] repairing {len(to_repair)} invalid outputs", flush=True)
        label(to_repair,
              lambda it: build_repair_messages(it, it["_raw"], it["_errors"]),
              id_of=lambda it: it["comment_id"], out_path=repair_path, **common)
        for cid, rec in _load_records(repair_path).items():
            item, entry = by_id.get(cid), merged.get(cid)
            if item is None or entry is None or rec.get("error"):
                continue
            ann, errors, fixes = validate_annotation(rec.get("parsed"), item)
            if not errors:  # repaired output wins only if now valid
                entry.update(pairs=(ann or {}).get("pairs"), errors=[],
                             source_fixes=entry.get("source_fixes", 0) + fixes,
                             valid=True, repaired=True,
                             cost_usd=(entry.get("cost_usd") or 0)
                                      + (rec.get("cost_usd") or 0))

    with open(out_path, "w", encoding="utf-8") as f:
        for cid in by_id:  # sample order
            if cid in merged:
                f.write(json.dumps(merged[cid], ensure_ascii=False) + "\n")

    n = len(merged)
    n_valid = sum(1 for e in merged.values() if e["valid"])
    n_rep = sum(1 for e in merged.values() if e.get("repaired"))
    n_fix = sum(e.get("source_fixes", 0) for e in merged.values())
    print(f"[annotate] {model}: {n_valid}/{n} valid ({n_rep} via repair), "
          f"{n_fix} cause_source fixes → {out_path}", flush=True)
    return out_path


def _entry(item: dict, model: str) -> dict:
    return {"comment_id": item["comment_id"], "stratum": item.get("stratum"),
            "model": model, "pairs": None, "valid": False, "repaired": False,
            "source_fixes": 0, "errors": [], "cost_usd": None}


def _load_records(path: str) -> dict[str, dict]:
    """id -> last successful record (later lines win: reruns supersede)."""
    recs: dict[str, dict] = {}
    if not os.path.exists(path):
        return recs
    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("id"):
                recs[str(rec["id"])] = rec
    return recs


def main():
    ap = argparse.ArgumentParser(description="Annotate a sample with one model")
    ap.add_argument("--sample", required=True, help="sample JSONL from pipeline.sample")
    ap.add_argument("--model", default=config.ANNOTATE_MODEL)
    ap.add_argument("--run", default="pilot", help="run name → data/labels/<run>/")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--mock", action="store_true", help="fake client: free, offline")
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--reasoning", default=None,
                    choices=["minimal", "low", "medium", "high", "max"],
                    help="reasoning effort passthrough (deepseek-v4-flash: use high)")
    ap.add_argument("--max-tokens", type=int, default=None,
                    help="completion token budget (default 12000; bump if a reasoning "
                         "model exhausts it thinking and returns empty output)")
    args = ap.parse_args()
    annotate(args.sample, args.model, args.run, args.limit, args.mock, args.workers,
             args.reasoning, args.max_tokens)


if __name__ == "__main__":
    main()
