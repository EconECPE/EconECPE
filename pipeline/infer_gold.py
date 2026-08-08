"""Local-GPU (vLLM) inference of the fine-tuned Qwen3.5-9B LoRA on the 300 gold labels.

Serves the base model (from the shared HF cache) with the LoRA adapter attached
at runtime (vLLM native LoRA — no merge step) and runs the *fine-tuned* prompt
contract: system + user only, NO few-shot, exactly as pipeline.finetune_dataset
built the training targets. Predictions are then scored against the adjudicated
gold with the existing span/typed/full-triplet F1 machinery
(pipeline.gold_report.score_silver_vs_gold).

Predictions are written consensus-shaped ({comment_id, neutral, pairs:[{...,
"included": true}]}) so they plug straight into the scorer.

Run with the vLLM venv's python:

    # smoke test on the mid-run checkpoint already on disk (5 items)
    venv-vllm/bin/python -m pipeline.infer_gold \\
        --adapter outputs/qwen3.5-9b-silver_clean-lr1e-4/checkpoint-1600 --limit 5

    # full 300-item gold eval on the final adapter
    venv-vllm/bin/python -m pipeline.infer_gold \\
        --adapter outputs/qwen3.5-9b-silver_clean-lr1e-4/final
"""
from __future__ import annotations

import argparse
import json
import os
import re

import config
from .prompt import format_input, system_prompt
from .validate import validate_annotation
from .gold_report import score_silver_vs_gold

GOLD_SAMPLE = os.path.join(config.DATA_DIR, "samples", "gold_300_seed7.jsonl")
GOLD_ADJUDICATED = os.path.join(config.LABELS_DIR, "gold", "adjudicated.jsonl")

_JSON_OBJ = re.compile(r"\{.*\}", re.DOTALL)


def load_items(path: str, limit: int | None) -> list[dict]:
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if limit is not None and len(out) >= limit:
                break
            out.append(json.loads(line))
    return out


def read_base_model(adapter_dir: str) -> str:
    with open(os.path.join(adapter_dir, "adapter_config.json"), encoding="utf-8") as f:
        return json.load(f)["base_model_name_or_path"]


def parse_pairs(text: str) -> tuple[list[dict], bool]:
    """Extract {"pairs": [...]} from the model completion. Returns (pairs, ok)."""
    m = _JSON_OBJ.search(text)
    if not m:
        return [], False
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError:
        return [], False
    pairs = obj.get("pairs") if isinstance(obj, dict) else None
    return (pairs, True) if isinstance(pairs, list) else ([], False)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--adapter", required=True,
                    help="model directory: a LoRA adapter dir, or a merged full model with --merged")
    ap.add_argument("--no-lora", action="store_true",
                    help="serve the base model only (diagnostic: isolates the LoRA's effect)")
    ap.add_argument("--merged", action="store_true",
                    help="--adapter is a full merged model — serve it directly, no runtime LoRA "
                         "(vLLM can't parse Unsloth's regex target_modules, so merged is the "
                         "reliable path; see pipeline.merge_adapter)")
    ap.add_argument("--base", default=None, help="override base model (default: read from adapter_config)")
    ap.add_argument("--sample", default=GOLD_SAMPLE)
    ap.add_argument("--gold", default=GOLD_ADJUDICATED)
    ap.add_argument("--out", default=None, help="predictions JSONL (default: <adapter>/gold_preds.jsonl)")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--max-model-len", type=int, default=4096)
    ap.add_argument("--max-new-tokens", type=int, default=1024)
    ap.add_argument("--max-lora-rank", type=int, default=16)
    ap.add_argument("--gpu-mem-util", type=float, default=0.90)
    args = ap.parse_args()

    from vllm import LLM, SamplingParams

    out_path = args.out or os.path.join(args.adapter, "gold_preds.jsonl")
    items = load_items(args.sample, args.limit)

    chat_kwargs = {}
    if args.no_lora:
        base = args.base or read_base_model(args.adapter)
        print(f"[infer_gold] {len(items)} gold items | BASE ONLY (no LoRA) {base}")
        llm = LLM(model=base, max_model_len=args.max_model_len,
                  gpu_memory_utilization=args.gpu_mem_util, trust_remote_code=True)
    elif args.merged:
        print(f"[infer_gold] {len(items)} gold items | merged model {args.adapter}")
        llm = LLM(
            model=args.adapter,
            max_model_len=args.max_model_len,
            gpu_memory_utilization=args.gpu_mem_util,
            trust_remote_code=True,
        )
    else:
        from vllm.lora.request import LoRARequest
        base = args.base or read_base_model(args.adapter)
        print(f"[infer_gold] {len(items)} gold items | base {base} | adapter {args.adapter}")
        llm = LLM(
            model=base,
            enable_lora=True,
            max_lora_rank=args.max_lora_rank,
            max_model_len=args.max_model_len,
            gpu_memory_utilization=args.gpu_mem_util,
            trust_remote_code=True,
        )
        chat_kwargs["lora_request"] = LoRARequest("gold_adapter", 1, args.adapter)

    sampling = SamplingParams(temperature=0.0, max_tokens=args.max_new_tokens)
    sys_prompt = system_prompt()

    conversations = [
        [{"role": "system", "content": sys_prompt},
         {"role": "user", "content": format_input(item)}]
        for item in items
    ]
    # Qwen3.5 is a reasoning model: add_generation_prompt alone force-opens a
    # <think> block and the model reverts to base reasoning ("Thinking Process:").
    # Training rendered an EMPTY think block then JSON, so suppress thinking at
    # inference (enable_thinking=False) to reproduce the exact training prefix.
    outputs = llm.chat(conversations, sampling, add_generation_prompt=True,
                       chat_template_kwargs={"enable_thinking": False}, **chat_kwargs)

    raw_path = out_path.replace(".jsonl", ".raw.jsonl")
    n_parse_fail = 0
    with open(out_path, "w", encoding="utf-8") as fout, open(raw_path, "w", encoding="utf-8") as fraw:
        for item, out in zip(items, outputs):
            completion = out.outputs[0].text
            fraw.write(json.dumps({"comment_id": item["comment_id"], "raw": completion},
                                  ensure_ascii=False) + "\n")
            pairs, ok = parse_pairs(completion)
            if not ok:
                n_parse_fail += 1
            # derive cause_source / flag span issues (pairs kept regardless — a bad
            # span simply won't match gold under IoU>=0.5)
            ann, _errs, _fixes = validate_annotation({"pairs": pairs}, item)
            pairs = (ann or {}).get("pairs", [])
            for p in pairs:
                p["included"] = True
            rec = {"comment_id": item["comment_id"], "neutral": len(pairs) == 0, "pairs": pairs}
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"[infer_gold] predictions -> {out_path} | parse failures: {n_parse_fail}/{len(items)}")

    # Score against gold when the full set was run (subsets would skew F1 denominators).
    if args.limit is None:
        scores = score_silver_vs_gold(adjudicated_path=args.gold, consensus_path=out_path)
        gold = {json.loads(l)["comment_id"]: json.loads(l) for l in open(args.gold)}
        preds = {json.loads(l)["comment_id"]: json.loads(l) for l in open(out_path)}
        common = [c for c in preds if c in gold]
        neutral_acc = sum(gold[c]["neutral"] == preds[c]["neutral"] for c in common) / len(common)
        scores["neutral_accuracy"] = round(neutral_acc, 4)
        scores["parse_failures"] = n_parse_fail
        print(json.dumps(scores, indent=2))
        with open(os.path.join(args.adapter, "gold_scores.json"), "w") as f:
            json.dump(scores, f, indent=2)
    else:
        print("[infer_gold] --limit set: skipping F1 scoring (subset would skew denominators)")


if __name__ == "__main__":
    main()
