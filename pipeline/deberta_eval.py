"""Decode + gold-scoring for the DeBERTaV3 span tagger (companion to
pipeline.deberta_train). Loads a trained checkpoint, decodes typed emotion and
cause spans from the three heads, forms pairs by the proximity heuristic
described in deberta_train, writes predictions in the consensus JSONL schema
(with `included: true` so they plug into the shared scorer), and scores them
against the adjudicated gold set with the SAME span-IoU matcher used for the
generative student (pipeline.gold_report.score_silver_vs_gold).

Decoding:
  * per token, argmax the emotion head and the cause head; group maximal runs of
    tokens sharing a non-O base category into typed char spans via the fast
    tokenizer's offset mapping (lenient BIO: B/I collapsed, robust to missing B-).
  * a decoded span is kept only if it spans >= MIN_TOKENS tokens and its mean
    per-token max-softmax confidence >= a per-head threshold. This suppresses the
    single-token fragments a token tagger scatters; the thresholds are tuned on
    the held-out silver DEV split (`--tune-dev`), never on gold.
  * pairing: each emotion span takes the nearest decoded cause span in the same
    comment (min char-gap); a comment with an emotion but no decoded cause falls
    back to a self-span cause (cause_span = the emotion span; cause_category =
    argmax non-O of the cause head over the emotion tokens).
  * neutral := no emotion span decoded.

    # tune decode thresholds on the silver dev split (no gold involved)
    python -m pipeline.deberta_eval --checkpoint outputs/deberta-v3-tagger/epoch2.pt \\
        --tune-dev data/finetune/deberta_recall/dev.jsonl
    # score on gold with chosen thresholds
    python -m pipeline.deberta_eval --checkpoint outputs/deberta-v3-tagger/epoch2.pt \\
        --emo-min-prob 0.9 --cause-min-prob 0.6
"""
from __future__ import annotations

import argparse
import json
import os

import config
from .prompt import format_input
from .report import match_pairs
from .gold_report import score_silver_vs_gold
from .deberta_train import build_model, EMO_BIO, CAUSE_BIO

MIN_TOKENS = 2  # a valid emotion/cause clause is not a lone token


def _id2cat(bio: dict[str, int]) -> dict[int, str | None]:
    out = {}
    for name, idx in bio.items():
        out[idx] = None if name == "O" else name.split("-", 1)[1]
    return out


EMO_ID2CAT = _id2cat(EMO_BIO)
CAUSE_ID2CAT = _id2cat(CAUSE_BIO)


def _decode_spans(pred_ids, probs, offsets, id2cat, text, min_tokens, min_prob):
    """Maximal same-category runs -> (text, start, end, cat), filtered by length + confidence."""
    runs = []
    cur_cat = cur_start = cur_end = None
    cur_probs = []
    for idx, p, (a, b) in zip(pred_ids, probs, offsets):
        cat = None if a == b else id2cat.get(int(idx))
        if cat != cur_cat:
            if cur_cat is not None:
                runs.append((cur_start, cur_end, cur_cat, cur_probs))
            cur_cat, cur_start, cur_probs = (cat, a, []) if cat is not None else (None, None, [])
        if cat is not None:
            cur_end = b
            cur_probs.append(p)
    if cur_cat is not None:
        runs.append((cur_start, cur_end, cur_cat, cur_probs))
    out = []
    for s, e, c, ps in runs:
        span = text[s:e].strip()
        if not span or len(ps) < min_tokens:
            continue
        if (sum(ps) / len(ps)) < min_prob:
            continue
        out.append((span, s, e, c))
    return out


def _gap(s1, e1, s2, e2):
    if e1 <= s2:
        return s2 - e1
    if e2 <= s1:
        return s1 - e2
    return 0


def decode_comment(text, emo_pred, emo_prob, cause_pred, cause_prob, cause_logits,
                   offsets, min_tokens, emo_min_prob, cause_min_prob):
    import torch
    emo_spans = _decode_spans(emo_pred, emo_prob, offsets, EMO_ID2CAT, text, min_tokens, emo_min_prob)
    cause_spans = _decode_spans(cause_pred, cause_prob, offsets, CAUSE_ID2CAT, text, min_tokens, cause_min_prob)
    if not emo_spans:
        return {"neutral": True, "pairs": []}
    pairs = []
    for etext, es, ee, emo in emo_spans:
        if cause_spans:
            ctext, cs, ce, ccat = min(cause_spans, key=lambda c: _gap(es, ee, c[1], c[2]))
            cause_span, cause_cat = ctext, ccat
        else:
            tok_idx = [i for i, (a, b) in enumerate(offsets) if a < ee and b > es and a != b]
            if tok_idx:
                sub = cause_logits[tok_idx][:, 1:].sum(0)
                cause_cat = CAUSE_ID2CAT[int(torch.argmax(sub).item()) + 1]
            else:
                cause_cat = CAUSE_ID2CAT[1]
            cause_span = etext
        pairs.append({"emotion": emo, "emotion_span": etext,
                      "cause_category": cause_cat, "cause_span": cause_span,
                      "intensity": None, "target_asset": None, "included": True})
    return {"neutral": False, "pairs": pairs}


def forward_all(checkpoint, sample_items, model_name, max_len):
    """Run the model once over items; cache decode inputs so threshold sweeps are free."""
    import torch
    from transformers import AutoTokenizer

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(model_name)
    model = build_model(model_name).to(dev)
    if str(checkpoint).endswith(".safetensors"):  # the Hugging Face release format
        from safetensors.torch import load_file
        model.load_state_dict(load_file(checkpoint, device=str(dev)))
    else:
        model.load_state_dict(torch.load(checkpoint, map_location=dev))
    model.eval()

    cached = []
    for item in sample_items:
        text = item["text"] if "text" in item else format_input(item)
        enc = tok(text, truncation=True, max_length=max_len, return_offsets_mapping=True)
        offsets = enc["offset_mapping"]
        ids = torch.tensor([enc["input_ids"]]).to(dev)
        am = torch.tensor([enc["attention_mask"]]).to(dev)
        with torch.no_grad():
            out = model(ids, am)
        emo_sm = torch.softmax(out["emo"][0], -1)
        cause_sm = torch.softmax(out["cause"][0], -1)
        cached.append({
            "comment_id": item["comment_id"], "text": text, "offsets": offsets,
            "emo_pred": emo_sm.argmax(-1).tolist(), "emo_prob": emo_sm.max(-1).values.tolist(),
            "cause_pred": cause_sm.argmax(-1).tolist(), "cause_prob": cause_sm.max(-1).values.tolist(),
            "cause_logits": out["cause"][0].cpu(),
        })
    return cached


def decode_cached(cached, min_tokens, emo_min_prob, cause_min_prob):
    preds = {}
    for c in cached:
        dec = decode_comment(c["text"], c["emo_pred"], c["emo_prob"], c["cause_pred"],
                             c["cause_prob"], c["cause_logits"], c["offsets"],
                             min_tokens, emo_min_prob, cause_min_prob)
        preds[c["comment_id"]] = dec
    return preds


def _ref_pairs_from_dev(rec):
    """Ground-truth pair list from a dev record's silver spans (same proximity pairing)."""
    emo = [(s[0], s[1], s[2]) for s in rec["emo_spans"]]
    cau = [(s[0], s[1], s[2]) for s in rec["cause_spans"]]
    text = rec["text"]
    pairs = []
    for es, ee, e in emo:
        if cau:
            cs, ce, cc = min(cau, key=lambda c: _gap(es, ee, c[0], c[1]))
            csp, ccat = text[cs:ce].strip(), cc
        else:
            csp, ccat = text[es:ee].strip(), (cau[0][2] if cau else "other")
        pairs.append({"emotion": e, "emotion_span": text[es:ee].strip(),
                      "cause_category": ccat, "cause_span": csp})
    return pairs


def _score(pred_pairs_by_cid, gold_pairs_by_cid, thr=0.5):
    ng = ns = span = typed = full = 0
    for cid, gp in gold_pairs_by_cid.items():
        sp = pred_pairs_by_cid.get(cid, [])
        ng += len(gp); ns += len(sp)
        for x, y in match_pairs(gp, sp, thr=thr):
            span += 1
            if x["emotion"] == y["emotion"]:
                typed += 1
                if x["cause_category"] == y["cause_category"]:
                    full += 1
    f1 = lambda m: (2 * (m / ns) * (m / ng) / ((m / ns) + (m / ng))) if (ns and ng and m) else 0.0
    return {"n_gold_pairs": ng, "n_pred_pairs": ns, "span_f1": round(f1(span), 4),
            "typed_pair_f1": round(f1(typed), 4), "full_triplet_f1": round(f1(full), 4)}


def tune_dev(cached, dev_items):
    gold_pairs = {r["comment_id"]: _ref_pairs_from_dev(r) for r in dev_items}
    print(f"[tune-dev] {len(dev_items)} dev comments; sweeping emo_min_prob (cause fixed 0.5, min_tokens {MIN_TOKENS})")
    best = None
    for emo_thr in [0.0, 0.5, 0.7, 0.8, 0.9, 0.95]:
        preds = decode_cached(cached, MIN_TOKENS, emo_thr, 0.5)
        pred_pairs = {cid: d["pairs"] for cid, d in preds.items()}
        s = _score(pred_pairs, gold_pairs)
        print(f"  emo_min_prob={emo_thr:<4} -> typed_f1 {s['typed_pair_f1']:.4f} "
              f"span_f1 {s['span_f1']:.4f} | pred_pairs {s['n_pred_pairs']} (gold {s['n_gold_pairs']})")
        if best is None or s["typed_pair_f1"] > best[1]:
            best = (emo_thr, s["typed_pair_f1"])
    print(f"[tune-dev] best emo_min_prob = {best[0]} (dev typed_f1 {best[1]:.4f})")
    return best[0]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--sample", default=os.path.join(config.SAMPLES_DIR, "gold_300_seed7.jsonl"))
    ap.add_argument("--gold", default=os.path.join(config.LABELS_DIR, "gold", "adjudicated.jsonl"))
    ap.add_argument("--model", default="microsoft/deberta-v3-large")
    ap.add_argument("--max-len", type=int, default=512)
    ap.add_argument("--tune-dev", default=None, help="dev.jsonl; sweep thresholds on silver dev and exit")
    ap.add_argument("--emo-min-prob", type=float, default=0.0)
    ap.add_argument("--cause-min-prob", type=float, default=0.5)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if args.tune_dev:
        dev_items = [json.loads(l) for l in open(args.tune_dev, encoding="utf-8") if l.strip()]
        cached = forward_all(args.checkpoint, dev_items, args.model, args.max_len)
        tune_dev(cached, dev_items)
        return

    items = [json.loads(l) for l in open(args.sample, encoding="utf-8") if l.strip()]
    cached = forward_all(args.checkpoint, items, args.model, args.max_len)
    preds = decode_cached(cached, MIN_TOKENS, args.emo_min_prob, args.cause_min_prob)
    out_path = args.out or (os.path.splitext(args.checkpoint)[0] + ".gold_preds.jsonl")
    with open(out_path, "w", encoding="utf-8") as f:
        for cid, dec in preds.items():
            f.write(json.dumps({"comment_id": cid, **dec}, ensure_ascii=False) + "\n")

    scores = score_silver_vs_gold(adjudicated_path=args.gold, consensus_path=out_path)
    gold = {json.loads(l)["comment_id"]: json.loads(l) for l in open(args.gold, encoding="utf-8")}
    common = [c for c in gold if c in preds]
    scores["neutral_accuracy"] = round(
        sum(bool(gold[c].get("neutral")) == bool(preds[c].get("neutral")) for c in common) / max(1, len(common)), 4)
    scores["emo_min_prob"] = args.emo_min_prob
    scores["cause_min_prob"] = args.cause_min_prob
    scores["checkpoint"] = os.path.basename(args.checkpoint)
    print(json.dumps(scores, indent=2))
    side = os.path.splitext(args.checkpoint)[0] + ".gold_scores.json"
    with open(side, "w") as f:
        json.dump(scores, f, indent=2)
    print(f"[deberta_eval] preds -> {out_path} | scores -> {side}", flush=True)


if __name__ == "__main__":
    main()
