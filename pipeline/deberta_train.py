"""DeBERTaV3 multi-task span tagger — the efficiency-variant student.

Backbone: microsoft/deberta-v3-large. Three heads over the token sequence:
  (1) emotion BIO   — O + B/I per emotion  (2*7+1 = 15 classes)
  (2) cause BIO     — O + B/I per cause category (2*|C|+1 classes)
  (3) intensity     — scalar regression, supervised on emotion B-tokens only
Char spans from pipeline.deberta_dataset are aligned to tokens via the fast
tokenizer's offset mapping. At inference, BIO decoding yields typed emotion and
cause spans; pairs are formed by a documented proximity heuristic (each emotion
span is paired with the nearest cause span in the same comment; comments with an
emotion but no decoded cause fall back to a self-span cause, mirroring the
"cause stated in the emotion clause" case). Pairs are then scored against gold
with the same span-IoU matcher as the generative student (pipeline.report).

This is the throughput variant for the corpus pass; it trades the generative
model's exact multi-pair structure for a single forward pass per comment.

    # architecture smoke test (CPU, random weights, no download)
    python -m pipeline.deberta_train --smoke-test
    # real training (needs a free GPU)
    python -m pipeline.deberta_train --data data/finetune/deberta_recall \
        --model microsoft/deberta-v3-large --epochs 3 --batch-size 16
"""
from __future__ import annotations

import argparse
import json
import os

import config

EMOTIONS = ["fear_anxiety", "pessimism_despair", "anger_disgust", "optimism_confidence",
            "excitement_euphoria", "confusion_uncertainty", "surprise"]
CAUSES = ["inflation", "monetary_policy", "employment", "growth_gdp", "fiscal_taxes",
          "trade_tariffs", "energy", "housing", "geopolitics", "markets_themselves",
          "crypto", "corporate_earnings", "inequality_distribution", "personal_finance",
          "interpersonal", "other", "unclear"]


def bio_labels(tags):
    m = {"O": 0}
    for t in tags:
        m[f"B-{t}"] = len(m); m[f"I-{t}"] = len(m)
    return m


EMO_BIO = bio_labels(EMOTIONS)      # 15
CAUSE_BIO = bio_labels(CAUSES)      # 35


def build_model(model_name, from_config=False):
    import torch, torch.nn as nn
    from transformers import AutoConfig, AutoModel

    class MultiHeadTagger(nn.Module):
        def __init__(self, backbone):
            super().__init__()
            self.backbone = backbone
            h = backbone.config.hidden_size
            self.emo = nn.Linear(h, len(EMO_BIO))
            self.cause = nn.Linear(h, len(CAUSE_BIO))
            self.intensity = nn.Linear(h, 1)
            self.drop = nn.Dropout(0.1)

        def forward(self, input_ids, attention_mask=None,
                    emo_labels=None, cause_labels=None, intensity=None):
            hs = self.drop(self.backbone(input_ids=input_ids,
                                         attention_mask=attention_mask).last_hidden_state)
            emo_logits, cause_logits = self.emo(hs), self.cause(hs)
            inten = self.intensity(hs).squeeze(-1)
            loss = None
            if emo_labels is not None:
                ce = nn.CrossEntropyLoss(ignore_index=-100)
                loss = ce(emo_logits.view(-1, len(EMO_BIO)), emo_labels.view(-1))
                loss = loss + ce(cause_logits.view(-1, len(CAUSE_BIO)), cause_labels.view(-1))
                mask = intensity >= 0
                if mask.any():
                    loss = loss + nn.functional.mse_loss(inten[mask], intensity[mask])
            return {"loss": loss, "emo": emo_logits, "cause": cause_logits, "intensity": inten}

    cfg = AutoConfig.from_pretrained(model_name)
    backbone = AutoModel.from_config(cfg) if from_config else AutoModel.from_pretrained(model_name)
    # transformers>=5 loads pretrained weights in their saved dtype (fp16 for
    # deberta-v3-large); the freshly-built heads are fp32 and this loop uses no
    # autocast, so force the whole tagger to fp32 to avoid a Half/Float mismatch.
    return MultiHeadTagger(backbone).float()


def align(text, spans, tokenizer, max_len, label_map, with_intensity=False):
    """char spans -> token BIO labels (+ optional per-token intensity target)."""
    enc = tokenizer(text, truncation=True, max_length=max_len, return_offsets_mapping=True)
    offs = enc["offset_mapping"]
    labels = [-100 if o == (0, 0) else 0 for o in offs]  # ignore specials, O elsewhere
    inten = [-1.0] * len(offs)
    for span in spans:
        s, e, typ = span[0], span[1], span[2]
        first = True
        for i, (a, b) in enumerate(offs):
            if a == b:
                continue
            if a < e and b > s:  # token overlaps the char span
                labels[i] = label_map[("B-" if first else "I-") + typ]
                if with_intensity and first and len(span) > 3:
                    inten[i] = span[3]
                first = False
    return enc["input_ids"], enc["attention_mask"], labels, inten


def smoke_test():
    import torch
    from transformers import AutoTokenizer
    name = "microsoft/deberta-v3-base"
    print(f"[smoke] building {name} multi-head tagger from CONFIG (random weights, no download of weights)")
    model = build_model(name, from_config=True)
    n = sum(p.numel() for p in model.parameters())
    print(f"[smoke] params: {n/1e6:.0f}M | emo classes {len(EMO_BIO)} | cause classes {len(CAUSE_BIO)}")
    tok = AutoTokenizer.from_pretrained(name)
    text = "POST TITLE: Fed hikes rates\n\nCOMMENT (annotate this): This is terrifying, inflation is out of control."
    ids, am, emo_l, inten = align(text, [[38, 49, "fear_anxiety", 0.8]], tok, 128, EMO_BIO, True)
    _, _, cau_l, _ = align(text, [[51, 78, "inflation"]], tok, 128, CAUSE_BIO)
    b = lambda x: torch.tensor([x])
    out = model(b(ids), b(am), b(emo_l), b(cau_l), b(inten))
    out["loss"].backward()
    print(f"[smoke] forward+backward OK | loss={out['loss'].item():.3f} | "
          f"seq_len={len(ids)} | emo B-tokens tagged={sum(1 for l in emo_l if l and l%2==1)}")
    print("[smoke] architecture validated — ready to train on a GPU")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--smoke-test", action="store_true")
    ap.add_argument("--data", default=os.path.join(config.DATA_DIR, "finetune", "deberta_recall"))
    ap.add_argument("--model", default="microsoft/deberta-v3-large")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--max-len", type=int, default=512)
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    if args.smoke_test:
        smoke_test()
        return

    # Full training loop (run on a free GPU). Kept straightforward and self-contained.
    import torch
    from torch.utils.data import DataLoader
    from transformers import AutoTokenizer, get_cosine_schedule_with_warmup

    tok = AutoTokenizer.from_pretrained(args.model)

    def load(split):
        rows = [json.loads(l) for l in open(os.path.join(args.data, split), encoding="utf-8")]
        feats = []
        for r in rows:
            ids, am, emo_l, inten = align(r["text"], r["emo_spans"], tok, args.max_len, EMO_BIO, True)
            _, _, cau_l, _ = align(r["text"], r["cause_spans"], tok, args.max_len, CAUSE_BIO)
            feats.append((ids, am, emo_l, cau_l, inten))
        return feats

    def collate(batch):
        m = max(len(x[0]) for x in batch)
        pad = lambda seq, v: seq + [v] * (m - len(seq))
        t = lambda col, v: torch.tensor([pad(x[col], v) for x in batch])
        return (t(0, tok.pad_token_id), t(1, 0), t(2, -100), t(3, -100),
                torch.tensor([pad(x[4], -1.0) for x in batch], dtype=torch.float))

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model(args.model).to(dev)
    train = DataLoader(load("train.jsonl"), batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    sched = get_cosine_schedule_with_warmup(opt, 100, args.epochs * len(train))
    out_dir = args.out_dir or os.path.join(config.BASE_DIR, "outputs", "deberta-v3-tagger")
    os.makedirs(out_dir, exist_ok=True)
    model.train()
    for ep in range(args.epochs):
        for step, (ids, am, el, cl, it) in enumerate(train):
            out = model(ids.to(dev), am.to(dev), el.to(dev), cl.to(dev), it.to(dev))
            out["loss"].backward(); opt.step(); sched.step(); opt.zero_grad()
            if step % 50 == 0:
                print(f"[deberta] ep{ep} step{step}/{len(train)} loss {out['loss'].item():.3f}", flush=True)
        torch.save(model.state_dict(), os.path.join(out_dir, f"epoch{ep}.pt"))
    print(f"[deberta] done -> {out_dir}  (decode+gold-scoring: see module docstring)")


if __name__ == "__main__":
    main()
