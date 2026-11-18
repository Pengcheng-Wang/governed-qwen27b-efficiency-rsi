"""
quantize_v2.py — Training-only calibrated GPTQ W4A16 export for Qwen/Qwen3.8-27B
with bitwise native BF16 MTP preservation AND repaired compressed-tensors `ignore`
list so stock vLLM 0.29 loads 1-GPU INT4 W4A16 + Native BF16 MTP3 without patching vLLM!
"""
from __future__ import annotations

import argparse
import json
import shutil
import struct
import time
from pathlib import Path

import torch
from datasets import Dataset
from llmcompressor import oneshot
from llmcompressor.modifiers.gptq import GPTQModifier
from safetensors import safe_open
from safetensors.torch import save_file
from transformers import AutoModelForImageTextToText, AutoTokenizer

from common import file_hash, object_hash, read_jsonl
from stage import write_json

ROOT = Path(__file__).resolve().parents[1]

# Exact vLLM Qwen3_5MTP module names & regexes dropped by llmcompressor because HF
# Qwen3_5ForConditionalGeneration ignores `mtp.*` weights during oneshot calibration:
VLLM_MTP_IGNORE_ENTRIES = [
    "re:^mtp\\..*",
    "re:.*mtp.*",
    "mtp.fc",
    "mtp.layers.0.self_attn.qkv_proj",
    "mtp.layers.0.self_attn.q_proj",
    "mtp.layers.0.self_attn.k_proj",
    "mtp.layers.0.self_attn.v_proj",
    "mtp.layers.0.self_attn.o_proj",
    "mtp.layers.0.mlp.gate_up_proj",
    "mtp.layers.0.mlp.gate_proj",
    "mtp.layers.0.mlp.up_proj",
    "mtp.layers.0.mlp.down_proj",
    "mtp.layers.0.mlp.experts",
    "mtp.layers.0.mlp.shared_expert.gate_up_proj",
    "mtp.layers.0.mlp.shared_expert.down_proj",
    "mtp.layers.0.mlp.shared_expert_gate",
    "mtp.layers.0.linear_attn.in_proj_qkv",
    "mtp.layers.0.linear_attn.in_proj_qkvz",
    "mtp.layers.0.linear_attn.in_proj_ba",
    "mtp.layers.0.linear_attn.in_proj_z",
    "mtp.layers.0.linear_attn.in_proj_a",
    "mtp.layers.0.linear_attn.in_proj_b",
    "mtp.layers.0.linear_attn.out_proj",
    "model.mtp.fc",
]


def build_weight_index(directory: Path) -> dict:
    mapping, total = {}, 0
    files = sorted(directory.glob("*.safetensors"))
    for path in files:
        with path.open("rb") as stream:
            header_len = struct.unpack("<Q", stream.read(8))[0]
            header = json.loads(stream.read(header_len))
        for name, tensor in header.items():
            if name == "__metadata__":
                continue
            start, end = tensor["data_offsets"]
            mapping[name] = path.name
            total += end - start
    return {"metadata": {"total_size": total}, "weight_map": mapping}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--base", default="models/dense27_base")
    p.add_argument("--output", default="models/gptq_w4a16_mtp_ready_v2")
    p.add_argument("--calibration-data", default="data/splits/sft_systems1024_v2.jsonl")
    p.add_argument("--samples", type=int, default=128)
    p.add_argument("--seed", type=int, default=20261117)
    args = p.parse_args()

    base = ROOT / args.base
    output = ROOT / args.output
    if output.exists():
        raise RuntimeError(f"Output directory already exists: {output}")

    rows = read_jsonl(ROOT / args.calibration_data)[: args.samples]
    if len(rows) != args.samples:
        raise RuntimeError("Insufficient calibration rows")
    dataset = Dataset.from_list([
        {"input_ids": r["input_ids"], "attention_mask": [1] * len(r["input_ids"])}
        for r in rows
    ])
    torch.manual_seed(args.seed)
    torch.set_num_threads(8)
    started = time.monotonic()

    model = AutoModelForImageTextToText.from_pretrained(
        base,
        dtype=torch.bfloat16,
        device_map={"": 0},
        local_files_only=True,
        attn_implementation="flash_attention_2",
    )
    model.config.use_cache = False
    model.config.text_config.use_cache = False
    layer_type = type(model.model.language_model.layers[0]).__name__
    ignore_rules = [
        "re:.*lm_head",
        "re:.*visual.*",
        "re:.*mtp.*",
        "re:.*linear_attn.in_proj_a",
        "re:.*linear_attn.in_proj_b",
    ]
    recipe = GPTQModifier(targets="Linear", scheme="W4A16", ignore=ignore_rules)
    tokenizer = AutoTokenizer.from_pretrained(base, local_files_only=True)
    oneshot(
        model=model,
        dataset=dataset,
        recipe=recipe,
        max_seq_length=4096,
        num_calibration_samples=args.samples,
        tokenizer=tokenizer,
        precision="bfloat16",
        tie_word_embeddings=model.config.tie_word_embeddings,
        pipeline="sequential",
        sequential_targets=[layer_type],
        pad_to_max_length=False,
        shuffle_calibration_samples=False,
        log_dir=str(ROOT / "logs" / "gptq_w4a16_v2"),
    )
    model.config.use_cache = True
    model.config.text_config.use_cache = True
    model.save_pretrained(output, save_compressed=True, max_shard_size="4GB")
    tokenizer.save_pretrained(output)
    for fname in ("generation_config.json", "chat_template.jinja"):
        if (base / fname).is_file():
            shutil.copy2(base / fname, output / fname)

    # Bitwise preserve all 15 native BF16 MTP tensors
    source_index = json.loads((base / "model.safetensors.index.json").read_text(encoding="utf-8"))
    native_mtp = {}
    for tname, sfile in source_index["weight_map"].items():
        if tname.startswith("mtp."):
            with safe_open(base / sfile, framework="pt", device="cpu") as reader:
                native_mtp[tname] = reader.get_tensor(tname).contiguous()
    if len(native_mtp) != 15:
        raise RuntimeError(f"Expected 15 native MTP tensors, got {len(native_mtp)}")
    mtp_path = output / "native-mtp.safetensors"
    save_file(native_mtp, mtp_path, metadata={"format": "pt"})
    with safe_open(mtp_path, framework="pt", device="cpu") as reader:
        if not all(torch.equal(v, reader.get_tensor(k)) for k, v in native_mtp.items()):
            raise RuntimeError("Native MTP tensors modified during save")

    # Rebuild weight index including native-mtp.safetensors
    index = build_weight_index(output)
    write_json(output / "model.safetensors.index.json", index)

    # CRITICAL FIX FOR v1 BLOCKER: Restore explicit MTP ignore entries into config.json["quantization_config"]["ignore"]
    cfg_path = output / "config.json"
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    qcfg = cfg["quantization_config"]
    existing_ignore = list(qcfg.get("ignore", []))
    for entry in VLLM_MTP_IGNORE_ENTRIES:
        if entry not in existing_ignore:
            existing_ignore.append(entry)
    qcfg["ignore"] = existing_ignore
    write_json(cfg_path, cfg)

    elapsed = time.monotonic() - started
    weight_files = [
        {"file": p.name, "bytes": p.stat().st_size, "sha256": file_hash(p)}
        for p in sorted(output.glob("*.safetensors"))
    ]
    manifest = {x["file"]: x["sha256"] for x in weight_files}
    for fname in ("config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "model.safetensors.index.json"):
        if (output / fname).is_file():
            manifest[fname] = file_hash(output / fname)
    write_json(
        output / "complete.json",
        {
            "kind": "quantized_model_mtp_ignore_repaired",
            "resume_capable": False,
            "files": manifest,
            "base_revision": json.loads((ROOT / "configs/model.json").read_text(encoding="utf-8"))["revision"],
        },
    )
    report = {
        "status": "QUANTIZED_MTP_IGNORE_REPAIRED",
        "calibration_seconds": elapsed,
        "calibration_rows": len(rows),
        "calibration_tokens": sum(len(r["input_ids"]) for r in rows),
        "calibration_ids_hash": object_hash([r["id"] for r in rows]),
        "native_mtp_tensors": len(native_mtp),
        "native_mtp_bitwise_preserved": True,
        "mtp_ignore_entries_injected": VLLM_MTP_IGNORE_ENTRIES,
        "weight_file_bytes": sum(x["bytes"] for x in weight_files),
    }
    write_json(ROOT / "results" / "quantization_v2.json", report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
