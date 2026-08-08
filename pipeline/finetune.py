"""Phase 3 distillation: LoRA SFT of a small model on the silver consensus labels.

Base model: Qwen3.5 (dense 4B/9B).
Framework: Unsloth + TRL SFTTrainer. Per Unsloth's own Qwen3.5 guidance, this
uses **bf16 LoRA, not 4-bit QLoRA** — Unsloth explicitly flags Qwen3.5 (dense or
MoE) as having "higher than normal quantization differences" under 4-bit, so
QLoRA is left available via --load-in-4bit but NOT the default. Approx VRAM
(Unsloth's published numbers): 4B bf16 LoRA ~10GB, 9B bf16 LoRA ~22GB — both
fit the RTX 5090's 32GB with headroom once it isn't shared with another job.

Input: data/finetune/<run>/{train,dev}.jsonl from pipeline.finetune_dataset,
each line {"messages": [system, user, assistant]}. The assistant turn is the
schema-shaped `{"pairs": [...]}` JSON distilled from the 3-model consensus.

Loss is masked to the assistant turn only (Unsloth's train_on_responses_only)
so the model isn't spending capacity learning to reproduce the (long, static)
system prompt — only the input->JSON mapping.

Every logged metric (train loss per --logging-steps, eval loss per
--eval-strategy, final summary + peak VRAM) is appended to
<output_dir>/metrics.jsonl as it happens — durable across an unattended
screen/nohup run, independent of scrollback and of checkpoint cadence.

Crash recovery: a checkpoint (adapter + optimizer/scheduler/RNG state) is
saved every --save-steps (default 100) under <output_dir>/checkpoint-N, last 3
kept — except the best-eval_loss checkpoint, which load_best_model_at_end
pins so it survives rotation even if it's not among the last 3 written. If
the process dies (OOM, box reboot, etc.), rerun the exact same command with
--resume added — it picks up the latest checkpoint in output_dir and
continues from that step (not from scratch), and metrics.jsonl is appended to
rather than overwritten.

Overfitting guard: with a --dev set, training evals every --eval-steps
(default 100, which --save-steps must be a multiple of) and
--early-stopping-patience (default 10 evals) stops the run once eval_loss
quits improving; either way,
the final saved adapter is the best-by-eval_loss checkpoint, not necessarily
the last one trained. (The first real 9B/3-epoch run predates this — its
eval_loss rose 21% in the final epoch and the better epoch-2 checkpoint had
already rotated off disk by the time anyone looked.)

    # environment/mechanics check — safe on a nearly-full GPU, no checkpoint saved
    python -m pipeline.finetune --smoke-test --model unsloth/Qwen3-0.6B-unsloth-bnb-4bit

    # real run (needs the GPU free)
    python -m pipeline.finetune --model Qwen/Qwen3.5-9B --run silver \\
        --epochs 3 --batch-size 2 --grad-accum 8

    # after a crash — same command plus --resume
    python -m pipeline.finetune --model Qwen/Qwen3.5-9B --run silver \\
        --epochs 3 --batch-size 2 --grad-accum 8 --resume
"""
from __future__ import annotations

import argparse
import json
import os

# Unsloth must be imported before transformers/peft/trl to apply its patches.
from unsloth import FastLanguageModel, is_bfloat16_supported
from unsloth.chat_templates import train_on_responses_only
from transformers import EarlyStoppingCallback, TrainerCallback

import config

INSTRUCTION_MARK = "<|im_start|>user\n"
RESPONSE_MARK = "<|im_start|>assistant\n"


class JSONLMetricsLogger(TrainerCallback):
    """Appends every logged metric dict (train-step loss + per-epoch eval) to a
    JSONL file. For an unattended overnight run in screen, this is the durable
    record — independent of scrollback and of checkpoint cadence (save_steps)."""

    def __init__(self, path: str, resume: bool = False):
        self.path = path
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if not (resume and os.path.exists(path)):
            open(path, "w", encoding="utf-8").close()  # fresh file, unless resuming

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs:
            return
        record = {"step": state.global_step, "epoch": round(state.epoch or 0, 4), **logs}
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_messages_jsonl(path: str, limit: int | None = None) -> list[list[dict]]:
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if limit is not None and len(out) >= limit:
                break
            out.append(json.loads(line)["messages"])
    return out


def to_text_dataset(tokenizer, messages_list: list[list[dict]]):
    from datasets import Dataset

    texts = [tokenizer.apply_chat_template(m, tokenize=False, add_generation_prompt=False)
             for m in messages_list]
    return Dataset.from_dict({"text": texts})


def _assert_chatml(tokenizer) -> None:
    """Fail loudly (not silently mis-mask the loss) if the chat template isn't
    ChatML-shaped the way INSTRUCTION_MARK/RESPONSE_MARK assume."""
    probe = tokenizer.apply_chat_template(
        [{"role": "system", "content": "s"}, {"role": "user", "content": "u"},
         {"role": "assistant", "content": "a"}],
        tokenize=False, add_generation_prompt=False)
    if INSTRUCTION_MARK not in probe or RESPONSE_MARK not in probe:
        raise SystemExit(
            "chat template doesn't contain the expected ChatML markers "
            f"({INSTRUCTION_MARK!r}, {RESPONSE_MARK!r}) — rendered:\n{probe}\n"
            "Update INSTRUCTION_MARK/RESPONSE_MARK in pipeline/finetune.py to match.")


def build(args) -> None:
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model,
        max_seq_length=args.max_seq_length,
        load_in_4bit=args.load_in_4bit,
        load_in_16bit=not args.load_in_4bit,
        full_finetuning=False,
        text_only=True,  # Qwen3.5 checkpoints are VLM-tagged; we never pass images
    )
    _assert_chatml(tokenizer)

    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_r,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=args.seed,
        max_seq_length=args.max_seq_length,
    )

    train_msgs = load_messages_jsonl(args.train, args.limit)
    train_ds = to_text_dataset(tokenizer, train_msgs)
    eval_ds = None
    if args.dev and not args.smoke_test:
        eval_msgs = load_messages_jsonl(args.dev, args.eval_limit)
        eval_ds = to_text_dataset(tokenizer, eval_msgs)

    if args.smoke_test:
        print(f"[finetune] smoke test: {len(train_ds)} examples loaded, "
              f"rendering example 0 (truncated to 800 chars):\n"
              f"{train_ds[0]['text'][:800]}\n...")

    from trl import SFTConfig, SFTTrainer

    # load_best_model_at_end needs a checkpoint to exist at every eval point, so
    # save/eval strategies must line up: "epoch"+"epoch", or "steps" with
    # save_steps a multiple of eval_steps (transformers' own _validate_args
    # enforces this direction — checkpoints may only happen at a subset of
    # eval-aligned steps, not the reverse). Without this, the run that
    # overfits in its last epoch (as the first real 9B run did) silently
    # ships the overfit final weights instead of the best-by-dev-loss
    # checkpoint.
    eval_strategy = args.eval_strategy if eval_ds is not None else "no"
    if eval_strategy == "steps" and args.save_steps % args.eval_steps != 0:
        raise SystemExit(
            f"--save-steps ({args.save_steps}) must be a multiple of --eval-steps "
            f"({args.eval_steps}) so a checkpoint exists at every eval point.")
    if args.smoke_test:
        save_strategy = "no"
    elif eval_strategy == "epoch":
        save_strategy = "epoch"
    else:
        save_strategy = "steps"
    load_best = eval_ds is not None and not args.smoke_test

    sft_args = SFTConfig(
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        per_device_eval_batch_size=max(1, args.batch_size),
        warmup_steps=args.warmup_steps,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps if args.max_steps else -1,
        learning_rate=args.lr,
        logging_steps=args.logging_steps,
        optim="adamw_8bit",
        weight_decay=args.weight_decay,
        lr_scheduler_type="cosine",
        seed=args.seed,
        output_dir=args.output_dir,
        report_to="none",
        dataset_num_proc=1,
        max_seq_length=args.max_seq_length,
        packing=False,  # multi-pair JSON targets need exact per-example boundaries
        eval_strategy=eval_strategy,
        eval_steps=(args.eval_steps if eval_strategy == "steps" else None),
        save_strategy=save_strategy,
        save_steps=args.save_steps,
        save_total_limit=3,
        load_best_model_at_end=load_best,
        metric_for_best_model="eval_loss" if load_best else None,
        greater_is_better=False if load_best else None,
        bf16=is_bfloat16_supported(),
        fp16=not is_bfloat16_supported(),
    )

    resume_ckpt = None
    if args.resume:
        from transformers.trainer_utils import get_last_checkpoint
        resume_ckpt = get_last_checkpoint(args.output_dir)
        if resume_ckpt is None:
            print(f"[finetune] --resume passed but no checkpoint found in "
                  f"{args.output_dir} — starting fresh")
        else:
            print(f"[finetune] resuming from {resume_ckpt}")

    metrics_path = os.path.join(args.output_dir, "metrics.jsonl")
    callbacks = [JSONLMetricsLogger(metrics_path, resume=resume_ckpt is not None)]
    if load_best and args.early_stopping_patience:
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience))

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        args=sft_args,
        callbacks=callbacks,
    )
    trainer = train_on_responses_only(
        trainer, instruction_part=INSTRUCTION_MARK, response_part=RESPONSE_MARK,
    )

    stats = trainer.train(resume_from_checkpoint=resume_ckpt)
    print(f"[finetune] done: {stats.metrics}")
    import torch
    peak_vram_gb = torch.cuda.max_memory_allocated() / 1e9
    print(f"[finetune] peak VRAM: {peak_vram_gb:.2f} GB")
    with open(metrics_path, "a", encoding="utf-8") as f:
        f.write(json.dumps({"final": True, "peak_vram_gb": round(peak_vram_gb, 2),
                            **stats.metrics}, ensure_ascii=False) + "\n")
    print(f"[finetune] metrics -> {metrics_path}")

    if not args.smoke_test:
        model.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)
        print(f"[finetune] LoRA adapter saved -> {args.output_dir}")


def main():
    ap = argparse.ArgumentParser(description="LoRA SFT distillation on silver consensus labels")
    ap.add_argument("--model", default="Qwen/Qwen3.5-9B")
    ap.add_argument("--run", default="silver", help="data/finetune/<run>/{train,dev}.jsonl")
    ap.add_argument("--train", default=None, help="override train.jsonl path")
    ap.add_argument("--dev", default=None, help="override dev.jsonl path")
    ap.add_argument("--output-dir", default=None)
    ap.add_argument("--max-seq-length", type=int, default=4096)
    ap.add_argument("--load-in-4bit", action="store_true",
                    help="QLoRA instead of bf16 LoRA — Unsloth advises against this "
                         "for Qwen3.5 (higher quantization error); off by default")
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=16)
    ap.add_argument("--lora-dropout", type=float, default=0.0)
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--warmup-steps", type=int, default=10)
    ap.add_argument("--logging-steps", type=int, default=10)
    ap.add_argument("--eval-strategy", choices=["epoch", "steps"], default="steps",
                    help="'steps' (default) evals every --eval-steps, which --save-steps "
                         "must be a multiple of so load_best_model_at_end can restore the "
                         "best checkpoint even if it wasn't the last one written; 'epoch' "
                         "evals less often (cheaper) but disables checkpoint-cadence crash "
                         "recovery finer than one epoch, since save_strategy must then "
                         "match eval_strategy")
    ap.add_argument("--eval-steps", type=int, default=100,
                    help="--save-steps must be a multiple of this when --eval-strategy steps")
    ap.add_argument("--early-stopping-patience", type=int, default=10,
                    help="stop if eval_loss doesn't improve for this many consecutive "
                         "evals (counted in evals, not epochs — with the default "
                         "eval-steps=100 that's roughly one epoch on the ~18.5k silver "
                         "set); 0 disables early stopping (load_best_model_at_end still "
                         "runs, so the best checkpoint is still restored at the end)")
    ap.add_argument("--save-steps", type=int, default=100,
                    help="checkpoint cadence — a LoRA adapter checkpoint is cheap to write, "
                         "so this defaults tighter than usual to bound crash-recovery loss "
                         "on an unattended run (~13min of progress at ~7.7s/step); must be "
                         "a multiple of --eval-steps when --eval-strategy steps")
    ap.add_argument("--resume", action="store_true",
                    help="resume from the latest checkpoint-N in --output-dir instead of "
                         "starting fresh (use after a crash, with the same command line)")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--limit", type=int, default=None, help="cap #train examples")
    ap.add_argument("--eval-limit", type=int, default=None, help="cap #dev examples")
    ap.add_argument("--max-steps", type=int, default=None, help="override epoch-based length")
    ap.add_argument("--smoke-test", action="store_true",
                    help="tiny run (5 steps, batch 1, 16 examples, no save/eval) "
                         "to validate the environment + data format")
    args = ap.parse_args()

    if args.smoke_test:
        args.limit = args.limit or 16
        args.batch_size = 1
        args.grad_accum = 1
        args.max_steps = args.max_steps or 5
        args.epochs = 1.0

    args.train = args.train or os.path.join(config.DATA_DIR, "finetune", args.run, "train.jsonl")
    args.dev = args.dev or os.path.join(config.DATA_DIR, "finetune", args.run, "dev.jsonl")
    args.output_dir = args.output_dir or os.path.join(
        config.BASE_DIR, "outputs", args.model.replace("/", "_"))

    build(args)


if __name__ == "__main__":
    main()
