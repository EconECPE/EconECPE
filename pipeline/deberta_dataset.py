"""Build the token-classification dataset for the DeBERTaV3 encoder tagger
(the efficiency-variant student, Section 5 of the paper).

Converts silver consensus labels into per-comment span-tagging examples: the same
rendered input as the generative student (system context is dropped — the encoder
sees only POST/PARENT/COMMENT text), with each silver pair's emotion span and cause
span located as CHARACTER offsets into that rendered text, tagged with the emotion
label and cause category respectively, plus the pair's intensity. Alignment is by
verbatim substring match (spans are validator-guaranteed verbatim); a span that
doesn't fall inside the rendered+truncated text is dropped and counted.

Output JSONL, one comment per line:
    {"comment_id", "text",
     "emo_spans":   [[start,end,emotion,intensity], ...],
     "cause_spans": [[start,end,cause_category], ...]}

The train script (pipeline.deberta_train) tokenizes `text` with an offset mapping
and turns these char spans into token-level BIO tags at load time, so the tokenizer
choice isn't baked in here.

    python -m pipeline.deberta_dataset --run silver \\
        --sample data/samples/silver_20000_seed7.jsonl --min-support 1 \\
        --exclude-ids data/samples/gold_300_seed7.jsonl \\
        --out-dir data/finetune/deberta_recall
"""
from __future__ import annotations

import argparse
import json
import os
from collections import Counter

import config
from .prompt import format_input
from .report import _load_sample
from .finetune_dataset import stratified_split

PAIR_MIN_SUPPORT_DEFAULT = 1  # match the deployed recall generative student


def _locate(text: str, span: str) -> tuple[int, int] | None:
    if not span:
        return None
    i = text.find(span)
    return (i, i + len(span)) if i >= 0 else None


def build_examples(consensus_path, sample_path, min_support, exclude_ids, limit=None):
    sample = _load_sample(sample_path)
    exclude_ids = exclude_ids or set()
    examples, stats = [], Counter()
    with open(consensus_path, encoding="utf-8") as f:
        for line in f:
            if limit and len(examples) >= limit:
                break
            rec = json.loads(line)
            cid = rec["comment_id"]
            if cid in exclude_ids:
                continue
            item = sample.get(cid)
            if item is None:
                continue
            text = format_input(item)
            emo_spans, cause_spans = [], []
            for p in rec["pairs"]:
                if (p.get("support") or 0) < min_support:
                    continue
                stats["pairs"] += 1
                es = _locate(text, p.get("emotion_span"))
                if es:
                    emo_spans.append([es[0], es[1], p["emotion"], float(p.get("intensity") or 0.0)])
                    stats["emo_found"] += 1
                else:
                    stats["emo_missing"] += 1
                cs = _locate(text, p.get("cause_span"))
                if cs:
                    cause_spans.append([cs[0], cs[1], p["cause_category"]])
                    stats["cause_found"] += 1
                else:
                    stats["cause_missing"] += 1
            examples.append({
                "comment_id": cid, "stratum": rec.get("stratum") or item.get("stratum"),
                "n_pairs": len([p for p in rec["pairs"] if (p.get("support") or 0) >= min_support]),
                "text": text, "emo_spans": emo_spans, "cause_spans": cause_spans,
            })
    return examples, stats


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", default="silver")
    ap.add_argument("--sample", required=True)
    ap.add_argument("--min-support", type=int, default=PAIR_MIN_SUPPORT_DEFAULT)
    ap.add_argument("--exclude-ids", default=None)
    ap.add_argument("--dev-size", type=int, default=1500)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    consensus_path = os.path.join(config.LABELS_DIR, args.run, "consensus.jsonl")
    exclude_ids = set()
    if args.exclude_ids:
        for line in open(args.exclude_ids, encoding="utf-8"):
            line = line.strip()
            if line:
                r = json.loads(line)
                exclude_ids.add(r.get("comment_id") or r.get("id"))

    examples, stats = build_examples(consensus_path, args.sample, args.min_support,
                                     exclude_ids, args.limit)
    train, dev = stratified_split(examples, args.dev_size, args.seed)
    out_dir = args.out_dir or os.path.join(config.DATA_DIR, "finetune", f"deberta_{args.run}")
    os.makedirs(out_dir, exist_ok=True)

    def _write(name, split):
        path = os.path.join(out_dir, name)
        with open(path, "w", encoding="utf-8") as f:
            for ex in split:
                f.write(json.dumps({k: ex[k] for k in ("comment_id", "text", "emo_spans", "cause_spans")},
                                   ensure_ascii=False) + "\n")
        return path

    tp, dp = _write("train.jsonl", train), _write("dev.jsonl", dev)
    emo_cov = stats["emo_found"] / max(1, stats["emo_found"] + stats["emo_missing"])
    cau_cov = stats["cause_found"] / max(1, stats["cause_found"] + stats["cause_missing"])
    print(f"[deberta_dataset] train {len(train)} / dev {len(dev)} | pairs {stats['pairs']} | "
          f"emotion-span alignment {emo_cov:.1%} | cause-span alignment {cau_cov:.1%}")
    print(f"[deberta_dataset] wrote {tp}, {dp}")


if __name__ == "__main__":
    main()
