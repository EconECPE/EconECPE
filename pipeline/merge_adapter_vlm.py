"""Merge a LoRA adapter into the Qwen3.5-9B **VLM** base by shard surgery.

Why not peft/Unsloth merge: the adapter was trained on the text causal-LM view
(keys `base_model.model.model.layers.N.<proj>`), but vLLM 0.24 only runs Qwen3.5
as `Qwen3_5ForConditionalGeneration`, whose language decoder lives at
`model.language_model.layers.N.<proj>`. Unsloth's merge saves arch
`Qwen3_5TextModel` (vLLM rejects it) and vLLM's runtime LoRA can't match the
key prefixes. So we add the deltas straight into the base safetensors shards at
the remapped keys, copy the base config/tokenizer verbatim (keeps the servable
VLM arch + vision_config), and let vLLM serve a plain merged model — no LoRA.

delta = (lora_alpha / r) * (B @ A), added to each target weight (spec: peft LoRA).

    venv/bin/python -m pipeline.merge_adapter_vlm \\
        --adapter outputs/qwen3.5-9b-silver_clean-lr1e-4/final \\
        --out     outputs/qwen3.5-9b-silver_clean-lr1e-4/final-merged-vlm
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import shutil

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file


def base_snapshot(base_id: str) -> str:
    cache = os.environ.get("HF_HUB_CACHE", "/huggingface_shared/hub")
    d = os.path.join(cache, "models--" + base_id.replace("/", "--"), "snapshots")
    snaps = sorted(glob.glob(os.path.join(d, "*")))
    if not snaps:
        raise SystemExit(f"base snapshot not found under {d}")
    return snaps[-1]


def load_deltas(adapter_dir: str) -> dict[str, torch.Tensor]:
    cfg = json.load(open(os.path.join(adapter_dir, "adapter_config.json")))
    scaling = cfg["lora_alpha"] / cfg["r"]
    sd = load_file(os.path.join(adapter_dir, "adapter_model.safetensors"))
    # group A/B by module core: base_model.model.<core>.lora_{A,B}.weight
    cores: dict[str, dict[str, torch.Tensor]] = {}
    for k, v in sd.items():
        assert k.startswith("base_model.model."), k
        core = k[len("base_model.model."):]
        core, ab = core.rsplit(".lora_", 1)   # ab in {"A.weight","B.weight"}
        cores.setdefault(core, {})[ab[0]] = v.float()
    deltas = {}
    for core, ab in cores.items():
        A, B = ab["A"], ab["B"]
        delta = scaling * (B @ A)             # [out,r]@[r,in] -> [out,in]
        # core: "model.layers.N.<sub>.<proj>"  ->  VLM: "model.language_model.layers..."
        assert core.startswith("model.layers."), core
        vlm_key = core.replace("model.layers.", "model.language_model.layers.", 1) + ".weight"
        deltas[vlm_key] = delta
    print(f"[merge-vlm] {len(deltas)} target weights, scaling={scaling}")
    return deltas


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--base", default=None)
    args = ap.parse_args()

    base_id = args.base or json.load(open(os.path.join(args.adapter, "adapter_config.json")))["base_model_name_or_path"]
    snap = base_snapshot(base_id)
    print(f"[merge-vlm] base snapshot: {snap}")
    deltas = load_deltas(args.adapter)

    os.makedirs(args.out, exist_ok=True)
    # copy all non-weight files verbatim (config keeps the servable VLM arch)
    for f in os.listdir(snap):
        if f.endswith(".safetensors"):
            continue
        src = os.path.join(snap, f)
        if os.path.isfile(src) or os.path.islink(src):
            shutil.copy(src, os.path.join(args.out, f))

    remaining = set(deltas)
    shards = sorted(glob.glob(os.path.join(snap, "*.safetensors")))
    for shard in shards:
        tensors = {}
        applied = 0
        with safe_open(shard, framework="pt") as f:
            meta = f.metadata()
            for k in f.keys():
                t = f.get_tensor(k)
                if k in deltas:
                    t = (t.float() + deltas[k]).to(t.dtype)
                    remaining.discard(k)
                    applied += 1
                tensors[k] = t
        out_shard = os.path.join(args.out, os.path.basename(shard))
        save_file(tensors, out_shard, metadata=meta or {"format": "pt"})
        print(f"[merge-vlm] {os.path.basename(shard)}: applied {applied} deltas")

    if remaining:
        raise SystemExit(f"[merge-vlm] ERROR: {len(remaining)} deltas never matched a base key, "
                         f"e.g. {sorted(remaining)[:3]}")
    print(f"[merge-vlm] done — all deltas applied -> {args.out}")


if __name__ == "__main__":
    main()
