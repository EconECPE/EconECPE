"""Merge a LoRA adapter into its base model and save fp16 weights for vLLM.

vLLM's runtime-LoRA path can't parse Unsloth's regex-string `target_modules`, so
it silently serves the un-adapted base. Merging sidesteps that entirely: the
adapter is folded into the base weights and vLLM serves a plain model.

Uses the training venv (Unsloth/peft, torch 2.10). Writes ~18GB (9B fp16).

    venv/bin/python -m pipeline.merge_adapter \\
        --adapter outputs/qwen3.5-9b-silver_clean-lr1e-4/checkpoint-1600 \\
        --out     outputs/qwen3.5-9b-silver_clean-lr1e-4/checkpoint-1600-merged
"""
from __future__ import annotations

import unsloth  # noqa: F401  (must precede transformers/peft)
from unsloth import FastLanguageModel

import argparse


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-seq-length", type=int, default=4096)
    args = ap.parse_args()

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.adapter,
        max_seq_length=args.max_seq_length,
        load_in_4bit=False,
        load_in_16bit=True,
        full_finetuning=False,
        text_only=True,
    )
    print(f"[merge] loaded {args.adapter}; merging to 16bit -> {args.out}")
    model.save_pretrained_merged(args.out, tokenizer, save_method="merged_16bit")
    print(f"[merge] done -> {args.out}")


if __name__ == "__main__":
    main()
