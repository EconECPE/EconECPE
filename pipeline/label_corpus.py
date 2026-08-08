"""Phase 3B — scale: label the full r/economics comment corpus with the distilled
student (local vLLM), producing the labels-of-record panel for the market study.

Streams data/economics_comments.jsonl (~5.16M rows), assembles each comment's
context from the SQLite index (pipeline.context.ContextAssembler), runs the merged
fine-tuned model with the SAME contract as training (system + user, no few-shot,
enable_thinking=False), validates each output (derives cause_source / drops
non-verbatim spans), and appends one record per comment to a JSONL:

    {"comment_id","created_utc","link_id","neutral","pairs":[...]}

Follows the arctic_shift.py durability pattern: append-only JSONL output + a tiny
JSON checkpoint (byte offset into the comments file) written after every batch —
safe to kill and re-run; it resumes from the offset. Deleted/empty comments are
skipped (recorded in the checkpoint's `skipped` count, not written).

    # benchmark throughput on the first 2000 comments
    venv-vllm/bin/python -m pipeline.label_corpus --limit 2000

    # full run (background; resumable)
    venv-vllm/bin/python -m pipeline.label_corpus
"""
from __future__ import annotations

import argparse
import json
import os
import time

import config
from .prompt import format_input, system_prompt
from .validate import validate_annotation
from .infer_gold import parse_pairs
from .context import ContextAssembler, connect

DEAD = {"[deleted]", "[removed]", "", None}
OUT_DEFAULT = os.path.join(config.LABELS_DIR, "corpus", "labels.jsonl")

# Per-field char caps so no prompt exceeds max_model_len. System prompt is ~1351
# tokens; with max_model_len 8192 - 768 gen there's ~6000 tokens (~22k chars) for
# the user message. Training truncated at 4096 tokens anyway, so capping here just
# bounds pathological long posts/comments (rare). Sum of caps ~19.3k chars (~5.1k tok).
CAP_TITLE, CAP_SELFTEXT, CAP_PARENT, CAP_BODY = 300, 4000, 3000, 12000


def _cap(item: dict) -> dict:
    if item.get("post_title"): item["post_title"] = item["post_title"][:CAP_TITLE]
    if item.get("post_selftext"): item["post_selftext"] = item["post_selftext"][:CAP_SELFTEXT]
    if item.get("parent_body"): item["parent_body"] = item["parent_body"][:CAP_PARENT]
    if item.get("body"): item["body"] = item["body"][:CAP_BODY]
    return item


def _atomic_write_json(path: str, obj: dict) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f)
    os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    # DEPLOYED student is the RECALL variant (support>=1); it is the labeler of
    # record for the whole corpus. The default was previously the clean variant,
    # which silently mislabeled part of the corpus when a resume omitted --model.
    ap.add_argument("--model", default=os.path.join(
        config.BASE_DIR, "outputs", "qwen3.5-9b-silver_recall-lr1e-4", "ckpt1300-merged-vlm"))
    ap.add_argument("--comments", default=config.COMMENTS_FILE)
    ap.add_argument("--out", default=OUT_DEFAULT)
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--batch-size", type=int, default=4096)
    ap.add_argument("--limit", type=int, default=None, help="stop after N comments processed (benchmark)")
    ap.add_argument("--stride", type=int, default=1,
                    help="systematic sample: label every Nth comment line for uniform temporal "
                         "coverage (stride 20 over 5.16M ~= 250k, spread across 2018->2026)")
    ap.add_argument("--max-model-len", type=int, default=8192)
    ap.add_argument("--max-new-tokens", type=int, default=768)
    ap.add_argument("--gpu-mem-util", type=float, default=0.90)
    args = ap.parse_args()

    ckpt_path = args.checkpoint or (args.out + ".checkpoint.json")
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    start_offset, n_done, n_skipped, line_idx = 0, 0, 0, 0
    if os.path.exists(ckpt_path) and args.limit is None:
        c = json.load(open(ckpt_path))
        start_offset, n_done, n_skipped = c["offset"], c["n_done"], c.get("skipped", 0)
        line_idx = c.get("line_idx", 0)
        print(f"[label] resuming from offset {start_offset} ({n_done} done, {n_skipped} skipped)")

    from vllm import LLM, SamplingParams
    llm = LLM(model=args.model, max_model_len=args.max_model_len,
              gpu_memory_utilization=args.gpu_mem_util, trust_remote_code=True,
              enable_prefix_caching=True)  # shared system prompt amortized across the corpus
    sampling = SamplingParams(temperature=0.0, max_tokens=args.max_new_tokens)
    sys_prompt = system_prompt()
    ctx = ContextAssembler(connect())
    tok = llm.get_tokenizer()
    # Token budget for the input side; char caps don't bound token count, and vLLM
    # hard-errors on an over-length prompt (that crashed the earlier run at ~91%).
    prompt_budget = args.max_model_len - args.max_new_tokens

    out_mode = "a" if start_offset else "w"
    fout = open(args.out, out_mode, encoding="utf-8")
    fin = open(args.comments, "rb")
    fin.seek(start_offset)

    batch_recs, batch_convs = [], []
    processed_this_run = 0
    t0 = time.time()

    def flush(batch_recs, batch_convs, offset_after):
        nonlocal n_done, n_skipped
        if not batch_recs:
            return
        # Drop prompts over the context budget (vLLM raises instead of truncating,
        # which would kill the whole run); count them as skipped.
        kept_recs, kept_convs = [], []
        for rec, conv in zip(batch_recs, batch_convs):
            try:
                ntok = len(tok.apply_chat_template(conv, add_generation_prompt=True,
                                                   tokenize=True, enable_thinking=False))
            except Exception:
                # tokenization failed -> assume over-length and SKIP. Do NOT fall
                # through with ntok=0 (that kept the prompt and let an 8193-tok input
                # reach vLLM, raising VLLMValidationError and killing the engine at ~91%).
                ntok = prompt_budget + 1
            if ntok > prompt_budget:
                n_skipped += 1
                print(f"[label] skipping over-length prompt ({ntok} tok > {prompt_budget}) "
                      f"comment {rec['id']}", flush=True)
                continue
            kept_recs.append(rec)
            kept_convs.append(conv)
        if not kept_convs:
            fout.flush()
            _atomic_write_json(ckpt_path, {"offset": offset_after, "n_done": n_done,
                                           "skipped": n_skipped, "line_idx": line_idx})
            return
        def _chat_safe(convs, recs):
            # vLLM raises VLLMValidationError and kills the engine if ANY prompt in the
            # batch exceeds max_model_len. The pre-filter above counts tokens with
            # tok.apply_chat_template, which can UNDERCOUNT vLLM's own renderer, so an
            # over-length prompt occasionally slips through and crashed the run at ~91%.
            # Bisect on failure to isolate and skip the offending prompt(s) — vLLM's own
            # tokenizer is the source of truth for what's too long; never crash the run.
            nonlocal n_skipped
            if not convs:
                return [], []
            try:
                return recs, llm.chat(convs, sampling, add_generation_prompt=True,
                                      chat_template_kwargs={"enable_thinking": False})
            except Exception as e:
                if len(convs) == 1:
                    n_skipped += 1
                    print(f"[label] skipping vLLM-rejected prompt comment {recs[0]['id']} "
                          f"({type(e).__name__})", flush=True)
                    return [], []
                mid = len(convs) // 2
                r1, o1 = _chat_safe(convs[:mid], recs[:mid])
                r2, o2 = _chat_safe(convs[mid:], recs[mid:])
                return r1 + r2, o1 + o2

        kept_recs, outs = _chat_safe(kept_convs, kept_recs)
        for rec, o in zip(kept_recs, outs):
            pairs, _ok = parse_pairs(o.outputs[0].text)
            ann, _e, _f = validate_annotation({"pairs": pairs}, rec["_item"])
            pairs = (ann or {}).get("pairs", [])
            fout.write(json.dumps({
                "comment_id": rec["id"], "created_utc": rec.get("created_utc"),
                "link_id": rec.get("link_id"), "neutral": len(pairs) == 0, "pairs": pairs,
            }, ensure_ascii=False) + "\n")
            n_done += 1
        fout.flush()
        _atomic_write_json(ckpt_path, {"offset": offset_after, "n_done": n_done,
                                       "skipped": n_skipped, "line_idx": line_idx})

    while True:
        if args.limit is not None and processed_this_run >= args.limit:
            break
        line = fin.readline()
        if not line:
            break
        offset_after = fin.tell()
        this_idx = line_idx
        line_idx += 1
        if args.stride > 1 and this_idx % args.stride != 0:
            continue  # systematic sample: skip non-selected lines (not counted as skipped)
        try:
            rec = json.loads(line)
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            # crash-during-write can leave zero-filled holes in the corpus (a NUL run
            # fuses with the next record into one line that json can't decode). Skip
            # it rather than dying; log the offset so corruption stays visible.
            print(f"[label] skipping unparseable line ~offset {offset_after} "
                  f"({type(e).__name__}) — likely a crash-corrupted record", flush=True)
            n_skipped += 1
            continue
        body = rec.get("body")
        if body in DEAD or not (rec.get("id")):
            n_skipped += 1
            continue
        c = ctx.assemble(rec)
        item = _cap({"comment_id": rec["id"], "body": body, "created_utc": rec.get("created_utc"),
                     "link_id": rec.get("link_id"), **c})
        rec["_item"] = item
        batch_recs.append(rec)
        batch_convs.append([{"role": "system", "content": sys_prompt},
                            {"role": "user", "content": format_input(item)}])
        processed_this_run += 1
        if len(batch_recs) >= args.batch_size:
            flush(batch_recs, batch_convs, offset_after)
            rate = n_done / max(1e-9, time.time() - t0) if not start_offset else \
                processed_this_run / max(1e-9, time.time() - t0)
            print(f"[label] {n_done:,} labeled | {n_skipped:,} skipped | "
                  f"{processed_this_run/max(1e-9,time.time()-t0):.1f} comments/s", flush=True)
            batch_recs, batch_convs = [], []

    flush(batch_recs, batch_convs, fin.tell())
    fout.close(); fin.close()
    dt = time.time() - t0
    print(f"[label] DONE run: processed {processed_this_run} this run, {n_done:,} total labeled, "
          f"{n_skipped:,} skipped, {dt:.1f}s ({processed_this_run/max(1e-9,dt):.1f} comments/s)")


if __name__ == "__main__":
    main()
