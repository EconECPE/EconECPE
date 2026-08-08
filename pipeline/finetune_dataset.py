"""Build the SFT dataset for Phase 3 distillation from the silver consensus labels.

Joins data/labels/<run>/consensus.jsonl (labels, keyed by comment_id — see
pipeline.consensus) with the sample JSONL (context fields) on comment_id, keeps
only *included* pairs (support >= min_support, decided upstream by consensus),
and renders each comment as one SFT example:

    {"messages": [
        {"role": "system", "content": <prompts/system.md, verbatim>},
        {"role": "user", "content": <format_input(item), same renderer the
                                      annotator LLMs saw>},
        {"role": "assistant", "content": <json.dumps({"pairs": [...]})>},
    ]}

Deliberately NO few-shot turns: the point of fine-tuning is to bake the
input->output mapping into weights instead of re-teaching it in-context every
call. Dropping the ~500 tokens of worked examples shortens every training
step and — more importantly — every call of the eventual 5M-comment inference
pass (Phase 3 contribution B). The system prompt itself is kept verbatim
(same contract the consensus teachers saw, and a shared prefix that inference
engines with prefix caching, e.g. vLLM, amortize to near-zero across the
5M-comment pass).

Comments with zero included pairs (neutral, or every candidate pair was an
excluded singleton) train on {"pairs": []} — that's the correct target, not a
gap: ~42% of the corpus really is neutral/low-confidence.

Gold labels aren't ready yet, so the held-out dev split (stratified by
`stratum`, proportional, seeded) is the only signal for monitoring training —
it's silver-vs-silver, not a substitute for the eventual gold eval.

    python -m pipeline.finetune_dataset --run silver \\
        --sample data/samples/silver_20000_seed7.jsonl [--dev-size 1500] [--limit 50]
"""
from __future__ import annotations

import argparse
import json
import os
import random
from collections import Counter, defaultdict

import config

from .prompt import format_input, system_prompt
from .report import _load_sample

PAIR_FIELDS = (
    "emotion", "emotion_span", "cause_span", "cause_source",
    "cause_category", "target_asset", "intensity", "sarcasm",
)


def _clean_pair(pair: dict) -> dict:
    """Strip consensus voting metadata (support/models/agreement/included),
    keep exactly the schema fields in schema order."""
    return {f: pair[f] for f in PAIR_FIELDS}


def build_examples(consensus_path: str, sample_path: str,
                   limit: int | None = None,
                   exclude_ids: set[str] | None = None,
                   min_support: int | None = None) -> list[dict]:
    sample = _load_sample(sample_path)
    sys_prompt = system_prompt()
    examples = []
    exclude_ids = exclude_ids or set()

    def _keep(p: dict) -> bool:
        # default: consensus-decided `included` (support>=2). --min-support overrides
        # it to a support threshold — min_support=1 keeps single-model pairs too,
        # raising the target pairs/comment (a recall lever; see gold-eval analysis).
        if min_support is None:
            return bool(p.get("included"))
        return (p.get("support") or 0) >= min_support

    with open(consensus_path, encoding="utf-8") as f:
        for line in f:
            if limit is not None and len(examples) >= limit:
                break
            rec = json.loads(line)
            if rec["comment_id"] in exclude_ids:
                continue
            item = sample.get(rec["comment_id"])
            if item is None:
                continue
            pairs = [_clean_pair(p) for p in rec["pairs"] if _keep(p)]
            target = json.dumps({"pairs": pairs}, ensure_ascii=False)
            examples.append({
                "comment_id": rec["comment_id"],
                "stratum": rec.get("stratum") or item.get("stratum"),
                "n_pairs": len(pairs),
                "messages": [
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": format_input(item)},
                    {"role": "assistant", "content": target},
                ],
            })
    return examples


def stratified_split(examples: list[dict], dev_size: int, seed: int) -> tuple[list[dict], list[dict]]:
    """Proportional-by-stratum split, seeded and deterministic."""
    by_stratum: dict[str, list[dict]] = defaultdict(list)
    for ex in examples:
        by_stratum[ex["stratum"]].append(ex)

    rng = random.Random(seed)
    dev_ids: set[str] = set()
    n_total = len(examples)
    for stratum, items in by_stratum.items():
        items = items[:]
        rng.shuffle(items)
        quota = round(dev_size * len(items) / n_total)
        dev_ids.update(ex["comment_id"] for ex in items[:quota])

    train = [ex for ex in examples if ex["comment_id"] not in dev_ids]
    dev = [ex for ex in examples if ex["comment_id"] in dev_ids]
    return train, dev


def _write_jsonl(path: str, examples: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps({"messages": ex["messages"]}, ensure_ascii=False) + "\n")


def build(run: str, sample_path: str, dev_size: int = 1500, seed: int = 7,
         limit: int | None = None, out_dir: str | None = None,
         exclude_ids_path: str | None = None,
         min_support: int | None = None) -> tuple[str, str]:
    consensus_path = os.path.join(config.LABELS_DIR, run, "consensus.jsonl")
    if not os.path.exists(consensus_path):
        raise SystemExit(f"no consensus file at {consensus_path} — run pipeline.consensus first")

    exclude_ids = set()
    if exclude_ids_path:
        with open(exclude_ids_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                exclude_ids.add(rec.get("comment_id") or rec.get("id"))
        print(f"[finetune_dataset] excluding {len(exclude_ids)} comment_ids from {exclude_ids_path}")

    examples = build_examples(consensus_path, sample_path, limit, exclude_ids, min_support)
    if not examples:
        raise SystemExit("no examples built — check --sample matches the consensus run")
    if exclude_ids:
        kept_ids = {ex["comment_id"] for ex in examples}
        leaked = exclude_ids & kept_ids
        if leaked:
            raise SystemExit(f"BUG: {len(leaked)} excluded ids still present in examples: {sorted(leaked)[:5]}")

    train, dev = stratified_split(examples, dev_size, seed)

    out_dir = out_dir or os.path.join(config.DATA_DIR, "finetune", run)
    os.makedirs(out_dir, exist_ok=True)
    train_path = os.path.join(out_dir, "train.jsonl")
    dev_path = os.path.join(out_dir, "dev.jsonl")
    _write_jsonl(train_path, train)
    _write_jsonl(dev_path, dev)

    manifest_path = os.path.join(out_dir, "split_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump({
            "seed": seed, "dev_size_requested": dev_size,
            "train": {ex["comment_id"]: ex["stratum"] for ex in train},
            "dev": {ex["comment_id"]: ex["stratum"] for ex in dev},
        }, f)

    def _stats(name: str, split: list[dict]) -> None:
        n = len(split)
        empty = sum(1 for ex in split if ex["n_pairs"] == 0)
        avg_pairs = sum(ex["n_pairs"] for ex in split) / n if n else 0
        by_stratum = Counter(ex["stratum"] for ex in split)
        print(f"[finetune_dataset] {name}: {n} examples | empty-pairs "
              f"{empty} ({100*empty/n:.1f}%) | avg pairs/comment {avg_pairs:.2f} | "
              f"strata {dict(by_stratum)}")

    _stats("train", train)
    _stats("dev", dev)
    print(f"[finetune_dataset] wrote {train_path}, {dev_path}, {manifest_path}")
    return train_path, dev_path


def main():
    ap = argparse.ArgumentParser(description="Build the SFT dataset from silver consensus labels")
    ap.add_argument("--run", default="silver", help="consensus run name (data/labels/<run>/consensus.jsonl)")
    ap.add_argument("--sample", required=True, help="sample JSONL matching the consensus run")
    ap.add_argument("--dev-size", type=int, default=1500)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--limit", type=int, default=None, help="cap #comments processed (smoke test)")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--exclude-ids", default=None,
                     help="JSONL with comment_id/id per line; excluded from both train and dev "
                          "(use to keep a gold eval set as a clean, never-trained-on holdout)")
    ap.add_argument("--min-support", type=int, default=None,
                     help="keep consensus pairs with support>=N instead of the default "
                          "`included` filter (support>=2). --min-support 1 keeps single-model "
                          "pairs too → more pairs/comment → higher-recall student.")
    args = ap.parse_args()
    build(args.run, args.sample, args.dev_size, args.seed, args.limit, args.out_dir,
          args.exclude_ids, args.min_support)


if __name__ == "__main__":
    main()
