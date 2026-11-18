"""
train_systems_v2.py — Exact-exposure BF16 LoRA SFT & ReST^EM / MoE trace trainer for AILab_Qwen27B_Governed_v2.

Supports:
  - Dense `Qwen/Qwen3.8-27B` (1-GPU or 8-GPU DDP, Liger fused CE, FP32 LoRA, disable_input_dtype_casting,
    optional parameter-aware activation offloading for 8K-16K ReST^EM verified traces).
  - MoE `Qwen/Qwen3.5-35B-A3B` (1-GPU or DDP, attention+GDN 190-module FP32 LoRA, frozen experts/router/MTP,
    phase-boundary empty_cache(), isolated fresh-process post-training NLL qualification).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import subprocess
import sys
import time
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.distributed as dist
from peft import LoraConfig, PeftModel, get_peft_model, get_peft_model_state_dict
from peft.helpers import disable_input_dtype_casting
from transformers import AutoModelForImageTextToText, AutoTokenizer
from trl.models import get_act_offloading_ctx_manager

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

from common import (  # noqa: E402
    assign_rank,
    file_hash,
    global_update_groups,
    loss_tokens,
    object_hash,
    read_jsonl,
    scaled_loss,
    validate_token_row,
)
from flops import estimated_mfu, training_matmul_flops  # noqa: E402
from stage import write_json  # noqa: E402
from telemetry import BoardTelemetry  # noqa: E402


def collate(rows: list[dict], pad_token_id: int, device: torch.device | str) -> dict[str, torch.Tensor]:
    width = max(len(r["input_ids"]) for r in rows)
    input_ids = torch.full((len(rows), width), pad_token_id, dtype=torch.long, device=device)
    attention_mask = torch.zeros((len(rows), width), dtype=torch.long, device=device)
    labels = torch.full((len(rows), width), -100, dtype=torch.long, device=device)
    for idx, row in enumerate(rows):
        n = len(row["input_ids"])
        input_ids[idx, :n] = torch.tensor(row["input_ids"], dtype=torch.long, device=device)
        attention_mask[idx, :n] = 1
        labels[idx, :n] = torch.tensor(row["labels"], dtype=torch.long, device=device)
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


@torch.no_grad()
def evaluate_nll(model, rows: list[dict], pad_token_id: int, device: torch.device | str) -> float:
    was_training = model.training
    model.eval()
    total_nll = 0.0
    total_toks = 0
    for row in rows:
        toks = loss_tokens(row)
        batch = collate([row], pad_token_id, device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(**batch, skip_logits=True)
            total_nll += float(out.loss.item()) * toks
            total_toks += toks
    if was_training:
        model.train()
    return total_nll / max(1, total_toks)


def run_isolated_nll_only(args: argparse.Namespace) -> None:
    spec = json.loads((ROOT / args.model_config).read_text(encoding="utf-8"))
    is_moe = "Moe" in spec["architecture"]
    if is_moe:
        from liger_kernel.transformers.monkey_patch import apply_liger_kernel_to_qwen3_5_moe
        apply_liger_kernel_to_qwen3_5_moe()
    else:
        from liger_kernel.transformers.monkey_patch import apply_liger_kernel_to_qwen3_5
        apply_liger_kernel_to_qwen3_5()

    tokenizer = AutoTokenizer.from_pretrained(ROOT / spec["model_directory"], local_files_only=True)
    if args.qualify_merged_dir is not None:
        from safetensors import safe_open
        merged_dir = Path(args.qualify_merged_dir).resolve(strict=True)
        src_dir = (ROOT / spec["model_directory"]).resolve(strict=True)
        exp_idx = json.loads((merged_dir / "model.safetensors.index.json").read_text(encoding="utf-8"))
        src_idx = json.loads((src_dir / "model.safetensors.index.json").read_text(encoding="utf-8"))

        # 1. Exact uint8-byte + SHA256 verification of ALL native mtp.* tensors (count, key set, shapes, dtypes, raw bytes)
        expected_mtp = int(spec.get("native_mtp_tensor_count", 15))
        src_mtp_map = {k: v for k, v in src_idx["weight_map"].items() if k.startswith("mtp.")}
        exp_mtp_map = {k: v for k, v in exp_idx["weight_map"].items() if k.startswith("mtp.")}
        if len(src_mtp_map) != expected_mtp or set(src_mtp_map.keys()) != set(exp_mtp_map.keys()):
            raise RuntimeError(
                f"Native MTP key set/count mismatch: expected {expected_mtp} (src={len(src_mtp_map)}), got {len(exp_mtp_map)}"
            )
        for tname, src_shard in sorted(src_mtp_map.items()):
            exp_shard = exp_mtp_map[tname]
            with safe_open(src_dir / src_shard, framework="pt", device="cpu") as sr, safe_open(
                merged_dir / exp_shard, framework="pt", device="cpu"
            ) as er:
                t_src = sr.get_tensor(tname).contiguous()
                t_exp = er.get_tensor(tname).contiguous()
                if (
                    t_src.shape != t_exp.shape
                    or t_src.dtype != t_exp.dtype
                    or not torch.equal(t_src.view(torch.uint8), t_exp.view(torch.uint8))
                ):
                    raise RuntimeError(f"Native MTP tensor exact uint8-byte mismatch on {tname}")

        # 2. Exhaustive independent parameter-level <= 1 BF16 ULP merge oracle across ALL adapted Linear weights
        param_max_bf16_ulp = 0
        checked_adapted_tensors = 0
        if args.qualify_adapter_dir is None:
            raise RuntimeError("--qualify-merged-dir requires --qualify-adapter-dir for exhaustive <=1 BF16 ULP parameter verification")
        adapter_dir = Path(args.qualify_adapter_dir).resolve(strict=True)
        adapter_cfg = json.loads((adapter_dir / "adapter_config.json").read_text(encoding="utf-8"))
        lora_r_val = float(adapter_cfg.get("r", 0))
        if not math.isfinite(lora_r_val) or lora_r_val <= 0:
            raise RuntimeError(f"Invalid adapter rank r={lora_r_val}")
        scaling = float(adapter_cfg["lora_alpha"]) / lora_r_val
        if not math.isfinite(scaling):
            raise RuntimeError(f"Non-finite LoRA scaling: {scaling}")
        adapter_sd: dict[str, torch.Tensor] = {}
        with safe_open(adapter_dir / "adapter_model.safetensors", framework="pt", device="cpu") as ar:
            for ak in ar.keys():
                adapter_sd[ak] = ar.get_tensor(ak)
        lora_a_keys = sorted(k for k in adapter_sd if k.endswith(".lora_A.weight"))
        lora_b_keys = sorted(k for k in adapter_sd if k.endswith(".lora_B.weight"))
        if not lora_a_keys or len(lora_a_keys) != len(lora_b_keys):
            raise RuntimeError(
                f"Malformed or empty adapter weights: lora_A={len(lora_a_keys)}, lora_B={len(lora_b_keys)}"
            )
        for ak in lora_a_keys:
            bk = ak[: -len(".lora_A.weight")] + ".lora_B.weight"
            if bk not in adapter_sd:
                raise RuntimeError(f"Missing paired lora_B key for {ak}")
            mod_path = ak.removeprefix("base_model.model.")[: -len(".lora_A.weight")]
            wk = mod_path + ".weight"
            if wk not in src_idx["weight_map"] or wk not in exp_idx["weight_map"]:
                raise RuntimeError(f"Adapted weight {wk} missing from source or exported index")
            with safe_open(src_dir / src_idx["weight_map"][wk], framework="pt", device="cpu") as sr, safe_open(
                merged_dir / exp_idx["weight_map"][wk], framework="pt", device="cpu"
            ) as er:
                w_base_raw = sr.get_tensor(wk)
                w_exp = er.get_tensor(wk)
            if w_exp.dtype != torch.bfloat16 or w_exp.shape != w_base_raw.shape:
                raise RuntimeError(f"Exported weight {wk} shape/dtype mismatch: {w_exp.shape}/{w_exp.dtype} vs {w_base_raw.shape}")
            la = adapter_sd[ak].to(torch.float32)
            lb = adapter_sd[bk].to(torch.float32)
            w_base = w_base_raw.to(torch.float32)
            if lb.ndim != 2 or la.ndim != 2 or lb.shape[1] != la.shape[0] or (lb.shape[0], la.shape[1]) != tuple(w_base.shape):
                raise RuntimeError(f"LoRA factor shape mismatch on {wk}: B={lb.shape}, A={la.shape}, W={w_base.shape}")
            if not (torch.isfinite(w_base).all() and torch.isfinite(la).all() and torch.isfinite(lb).all() and torch.isfinite(w_exp).all()):
                raise RuntimeError(f"Non-finite parameter tensor encountered on {wk}")
            w_ref_bf16 = (w_base + scaling * (lb @ la)).to(torch.bfloat16)
            if not torch.isfinite(w_ref_bf16).all():
                raise RuntimeError(f"Non-finite BF16 merge reference on {wk}")
            u_ref = w_ref_bf16.view(torch.int16).to(torch.int32)
            u_exp = w_exp.view(torch.int16).to(torch.int32)
            ord_ref = torch.where(u_ref < 0, -32768 - u_ref, u_ref)
            ord_exp = torch.where(u_exp < 0, -32768 - u_exp, u_exp)
            max_ulp = int((ord_ref - ord_exp).abs().max().item())
            if max_ulp > 1:
                raise RuntimeError(f"Parameter merge ULP oracle failed on {wk}: max_bf16_ulp={max_ulp} > 1")
            param_max_bf16_ulp = max(param_max_bf16_ulp, max_ulp)
            checked_adapted_tensors += 1

        # 3. Separate functional held-out NLL measurement in fresh isolated process
        model = AutoModelForImageTextToText.from_pretrained(
            merged_dir,
            dtype=torch.bfloat16,
            local_files_only=True,
            attn_implementation="flash_attention_2",
            device_map={"": 0},
        )
        heldout = read_jsonl(ROOT / args.heldout)
        nll = evaluate_nll(model, heldout, tokenizer.pad_token_id, "cuda:0")
        out_path = Path(args.qualify_output)
        res_payload = {
            "status": "ISOLATED_MERGED_PARAMETER_AND_FUNCTIONAL_QUALIFIED",
            "merged_after_nll": nll,
            "native_mtp_tensors_bitwise_verified": len(src_mtp_map),
            "checked_adapted_linear_tensors": checked_adapted_tensors,
            "param_merge_max_bf16_ulp": param_max_bf16_ulp if checked_adapted_tensors > 0 else None,
            "merged_dir": str(merged_dir),
        }
        write_json(out_path, res_payload)
        print(json.dumps(res_payload))
        return

    base_model = AutoModelForImageTextToText.from_pretrained(
        ROOT / spec["model_directory"],
        dtype=torch.bfloat16,
        local_files_only=True,
        attn_implementation="flash_attention_2",
        device_map={"": 0},
    )
    model = PeftModel.from_pretrained(base_model, Path(args.qualify_checkpoint), is_trainable=False)
    heldout = read_jsonl(ROOT / args.heldout)
    nll = evaluate_nll(model, heldout, tokenizer.pad_token_id, "cuda:0")
    out_path = Path(args.qualify_output)
    write_json(out_path, {"status": "ISOLATED_NLL_OK", "after_nll": nll, "checkpoint": str(args.qualify_checkpoint)})
    print(json.dumps({"after_nll": nll}))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--run-name", required=True)
    p.add_argument("--model-config", default="configs/model.json")
    p.add_argument("--dataset", default="data/splits/sft_systems1024_v2.jsonl")
    p.add_argument("--heldout", default="data/splits/sft_heldout64_v2.jsonl")
    p.add_argument("--global-batch", type=int, default=64)
    p.add_argument("--micro-batch", type=int, default=4)
    p.add_argument("--length-bucketing", action="store_true")
    p.add_argument("--activation-offloading", action="store_true")
    p.add_argument("--max-length", type=int, default=4096)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--rank", type=int, default=16)
    p.add_argument("--alpha", type=int, default=32)
    p.add_argument("--seed", type=int, default=20261117)
    p.add_argument("--save-checkpoint", action="store_true")
    p.add_argument("--merge-fp32-export", type=Path, default=None)
    p.add_argument("--qualify-checkpoint", type=Path, default=None)
    p.add_argument("--qualify-merged-dir", type=Path, default=None)
    p.add_argument("--qualify-adapter-dir", type=Path, default=None)
    p.add_argument("--qualify-output", type=Path, default=None)
    args = p.parse_args()

    if args.qualify_checkpoint is not None or args.qualify_merged_dir is not None:
        run_isolated_nll_only(args)
        return

    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    if world > 1:
        dist.init_process_group("nccl", device_id=device)

    spec = json.loads((ROOT / args.model_config).read_text(encoding="utf-8"))
    is_moe = "Moe" in spec["architecture"]
    if is_moe:
        from liger_kernel.transformers.monkey_patch import apply_liger_kernel_to_qwen3_5_moe
        apply_liger_kernel_to_qwen3_5_moe()
    else:
        from liger_kernel.transformers.monkey_patch import apply_liger_kernel_to_qwen3_5
        apply_liger_kernel_to_qwen3_5()

    rows = read_jsonl(ROOT / args.dataset)
    heldout = read_jsonl(ROOT / args.heldout)
    for r in rows + heldout:
        validate_token_row(r, args.max_length, spec["eos_token_ids"])
    groups = global_update_groups(rows, args.global_batch)

    out_dir = ROOT / "results" / "training" / args.run_name
    ckpt_dir = ROOT / "checkpoints" / args.run_name
    if rank == 0:
        out_dir.mkdir(parents=True, exist_ok=False)

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = False

    proc_start = time.monotonic()
    tokenizer = AutoTokenizer.from_pretrained(ROOT / spec["model_directory"], local_files_only=True)
    model = AutoModelForImageTextToText.from_pretrained(
        ROOT / spec["model_directory"],
        dtype=torch.bfloat16,
        local_files_only=True,
        attn_implementation="flash_attention_2",
        device_map={"": local_rank},
    )
    if model.config.architectures != [spec["architecture"]]:
        raise RuntimeError("Native model architecture wrapper changed")

    linear_names = {
        mname + ".weight": mod.weight.numel()
        for mname, mod in model.named_modules()
        if isinstance(mod, torch.nn.Linear) and (mname.startswith("model.language_model.") or mname == "lm_head")
    }
    if is_moe:
        packed = {
            k: v.numel()
            for k, v in model.named_parameters()
            if k.startswith("model.language_model.layers.") and ".mlp.experts." in k and v.ndim == 3
        }
        router_params = sum(
            v.numel()
            for k, v in model.named_parameters()
            if k.startswith("model.language_model.layers.") and ".mlp." in k and v.ndim == 2 and k not in linear_names
        )
        base_active_params = sum(linear_names.values()) + router_params + sum(packed.values()) * 8 // 256
        targets = sorted(
            k.removesuffix(".weight")
            for k in linear_names
            if k.startswith("model.language_model.layers.") and (".self_attn." in k or ".linear_attn." in k)
        )
        lora_r, lora_alpha = spec["adapter"]["rank"], spec["adapter"]["alpha"]
        attn_layers, q_heads, h_dim = 10, 16, 256
    else:
        base_active_params = sum(linear_names.values())
        targets = sorted(
            k.removesuffix(".weight")
            for k in linear_names
            if k.startswith("model.language_model.layers.")
        )
        lora_r, lora_alpha = args.rank, args.alpha
        attn_layers, q_heads, h_dim = 16, 24, 256

    model = get_peft_model(
        model,
        LoraConfig(r=lora_r, lora_alpha=lora_alpha, lora_dropout=0.0, bias="none", task_type="CAUSAL_LM", target_modules=targets),
    )
    trainable = [p_t for p_t in model.parameters() if p_t.requires_grad]
    trainable_count = sum(p_t.numel() for p_t in trainable)
    if any(p_t.dtype != torch.float32 for p_t in trainable):
        raise RuntimeError("FP32 LoRA parameter invariant violated")

    model.enable_input_require_grads()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.config.use_cache = False
    model.config.text_config.use_cache = False
    model.train()

    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.0)
    before_nll = evaluate_nll(model, heldout, tokenizer.pad_token_id, device) if rank == 0 else 0.0
    model.train()
    torch.cuda.empty_cache()
    if world > 1:
        dist.barrier()
        ddp_model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local_rank],
            broadcast_buffers=False,
            find_unused_parameters=False,
        )
    else:
        ddp_model = model

    vis_env = os.environ.get("CUDA_VISIBLE_DEVICES")
    if vis_env:
        phys_gpus = [int(x) for x in vis_env.split(",")]
    else:
        phys_gpus = list(range(world))
    telemetry = BoardTelemetry(phys_gpus) if rank == 0 else None
    if telemetry:
        telemetry.start()

    torch.cuda.reset_peak_memory_stats(device)
    step_records = []
    loop_start = time.monotonic()
    for step, group in enumerate(groups, 1):
        t0 = time.monotonic()
        global_toks = sum(loss_tokens(r) for r in group)
        rank_rows = assign_rank(group, rank, world, args.length_bucketing)
        chunks = [rank_rows[s_idx : s_idx + args.micro_batch] for s_idx in range(0, len(rank_rows), args.micro_batch)]
        optimizer.zero_grad(set_to_none=True)
        local_loss_sum = 0.0
        local_padded = 0
        for c_idx, chunk in enumerate(chunks):
            ltoks = sum(loss_tokens(r) for r in chunk)
            local_padded += len(chunk) * max(len(r["input_ids"]) for r in chunk)
            is_last_chunk = c_idx == len(chunks) - 1
            sync_ctx = ddp_model.no_sync() if (world > 1 and not is_last_chunk) else nullcontext()
            cast_ctx = disable_input_dtype_casting(model) if args.activation_offloading else nullcontext()
            offload_ctx = get_act_offloading_ctx_manager(model) if args.activation_offloading else nullcontext()
            with sync_ctx, cast_ctx, offload_ctx, torch.autocast("cuda", dtype=torch.bfloat16):
                loss_val = ddp_model(**collate(chunk, tokenizer.pad_token_id, device)).loss
                scaled_loss(loss_val, ltoks, global_toks, world).backward()
            local_loss_sum += float(loss_val.detach().float().item()) * ltoks
        grad_norm = float(torch.nn.utils.clip_grad_norm_(trainable, 1.0, error_if_nonfinite=True).item())
        optimizer.step()
        torch.cuda.synchronize(device)

        loss_tensor = torch.tensor([local_loss_sum, float(local_padded)], dtype=torch.float64, device=device)
        if world > 1:
            dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
        if rank == 0:
            rec = {
                "step": step,
                "seconds": time.monotonic() - t0,
                "loss": float(loss_tensor[0].item()) / global_toks,
                "grad_norm": grad_norm,
                "global_group_hash": object_hash([r["id"] for r in group]),
                "input_tokens": sum(len(r["input_ids"]) for r in group),
                "supervised_tokens": global_toks,
                "padded_tokens": int(loss_tensor[1].item()),
            }
            step_records.append(rec)
            print(json.dumps(rec), flush=True)

    if world > 1:
        dist.barrier()
    torch.cuda.synchronize(device)
    # Stop pure training-loop timer and board-power telemetry BEFORE any checkpoint/export I/O!
    loop_seconds = time.monotonic() - loop_start
    energy = telemetry.stop() if telemetry else {}

    ckpt_t0 = time.monotonic()
    if rank == 0 and (args.save_checkpoint or is_moe or args.merge_fp32_export):
        ckpt_step_dir = ckpt_dir / f"checkpoint-{len(groups)}"
        ckpt_step_dir.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "adapter": {k: v.detach().cpu() for k, v in get_peft_model_state_dict(model).items()},
                "optimizer": optimizer.state_dict(),
                "rng_cpu": torch.get_rng_state(),
                "rng_cuda": torch.cuda.get_rng_state(device),
                "rng_python": random.getstate(),
                "step": len(groups),
            },
            ckpt_step_dir / "state.pt",
        )
        model.save_pretrained(ckpt_step_dir)
        files_map = {f.name: file_hash(f) for f in ckpt_step_dir.iterdir() if f.is_file()}
        write_json(
            ckpt_step_dir / "complete.json",
            {"step": len(groups), "world_size": world, "files": files_map, "resume_capable": True},
        )
    checkpoint_save_seconds = (time.monotonic() - ckpt_t0) if rank == 0 else 0.0

    peak_alloc = torch.tensor([torch.cuda.max_memory_allocated(device) / (1024**3)], dtype=torch.float64, device=device)
    if world > 1:
        dist.all_reduce(peak_alloc, op=dist.ReduceOp.MAX)
    max_peak_gib = float(peak_alloc[0].item())
    total_gib = torch.cuda.get_device_properties(device).total_memory / (1024**3)
    headroom_gib = total_gib - max_peak_gib

    if world > 1:
        dist.barrier()
        dist.destroy_process_group()

    if rank != 0:
        return

    # Fresh-process NLL for MoE (avoids learner CUDA context spill); direct or fresh for Dense
    ckpt_step_dir = ckpt_dir / f"checkpoint-{len(groups)}"
    if is_moe:
        del ddp_model, model, optimizer
        torch.cuda.empty_cache()
        qual_out = out_dir / "isolated_nll.json"
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(phys_gpus[0])
        subprocess.run(
            [
                sys.executable, str( Path(__file__).resolve() ),
                "--run-name", args.run_name,
                "--model-config", args.model_config,
                "--heldout", args.heldout,
                "--qualify-checkpoint", str(ckpt_step_dir),
                "--qualify-output", str(qual_out),
            ],
            env=env, check=True, timeout=600,
        )
        after_nll = json.loads(qual_out.read_text(encoding="utf-8"))["after_nll"]
    else:
        torch.cuda.empty_cache()
        after_nll = evaluate_nll(model, heldout, tokenizer.pad_token_id, device)

    # Optional FP32 merge + native MTP preservation export for vLLM serving, qualified out-of-process!
    merge_qualification = None
    if args.merge_fp32_export is not None:
        if not is_moe:
            del ddp_model, model, optimizer
        torch.cuda.empty_cache()
        export_dir = ROOT / args.merge_fp32_export
        export_dir.mkdir(parents=True, exist_ok=False)
        base_cpu = AutoModelForImageTextToText.from_pretrained(
            ROOT / spec["model_directory"], dtype=torch.float32, local_files_only=True, device_map="cpu"
        )
        peft_cpu = PeftModel.from_pretrained(base_cpu, ckpt_step_dir, is_trainable=False)
        merged_cpu = peft_cpu.merge_and_unload()
        merged_bf16 = merged_cpu.to(torch.bfloat16)
        merged_bf16.save_pretrained(export_dir, max_shard_size="4GB")
        del base_cpu, peft_cpu, merged_cpu, merged_bf16
        tokenizer.save_pretrained(export_dir)
        for fname in ("generation_config.json", "chat_template.jinja"):
            if (ROOT / spec["model_directory"] / fname).is_file():
                import shutil
                shutil.copy2(ROOT / spec["model_directory"] / fname, export_dir / fname)
        # Copy native MTP tensors bitwise into export_dir and update weight index!
        from safetensors import safe_open
        from safetensors.torch import save_file
        from quantize_v2 import build_weight_index
        src_idx = json.loads((ROOT / spec["model_directory"] / "model.safetensors.index.json").read_text(encoding="utf-8"))
        mtp_dict = {}
        for tname, sfile in src_idx["weight_map"].items():
            if tname.startswith("mtp."):
                with safe_open(ROOT / spec["model_directory"] / sfile, framework="pt", device="cpu") as rdr:
                    mtp_dict[tname] = rdr.get_tensor(tname).contiguous()
        save_file(mtp_dict, export_dir / "native-mtp.safetensors", metadata={"format": "pt"})
        write_json(export_dir / "model.safetensors.index.json", build_weight_index(export_dir))
        exp_files = {p_f.name: file_hash(p_f) for p_f in export_dir.iterdir() if p_f.is_file()}
        write_json(export_dir / "complete.json", {"kind": "fp32_merged_model_with_native_mtp", "resume_capable": False, "files": exp_files})

        # Out-of-process functional NLL parity qualification on Search-only heldout!
        m_qual_path = out_dir / "merge_qualification.json"
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(phys_gpus[0])
        subprocess.run(
            [
                sys.executable, "-I", "-B", str(Path(__file__).resolve()),
                "--run-name", args.run_name,
                "--model-config", args.model_config,
                "--heldout", args.heldout,
                "--qualify-merged-dir", str(export_dir),
                "--qualify-adapter-dir", str(ckpt_step_dir),
                "--qualify-output", str(m_qual_path),
            ],
            env=env, check=True, timeout=900,
        )
        mq_raw = json.loads(m_qual_path.read_text(encoding="utf-8"))
        merged_nll = float(mq_raw["merged_after_nll"])
        nll_abs_diff = abs(merged_nll - after_nll)
        nll_rel_diff = nll_abs_diff / max(1e-9, abs(after_nll))
        if nll_rel_diff > 0.01:
            raise RuntimeError(
                f"Merged BF16 export failed functional NLL parity qualification: unmerged={after_nll:.6f}, merged={merged_nll:.6f}"
            )
        merge_qualification = {
            "status": "OUT_OF_PROCESS_PARAMETER_ULP_AND_FUNCTIONAL_NLL_QUALIFIED",
            "param_merge_max_bf16_ulp": mq_raw["param_merge_max_bf16_ulp"],
            "checked_adapted_linear_tensors": int(mq_raw["checked_adapted_linear_tensors"]),
            "native_mtp_tensors_bitwise_verified": int(mq_raw["native_mtp_tensors_bitwise_verified"]),
            "unmerged_peft_after_nll": after_nll,
            "merged_bf16_after_nll": merged_nll,
            "functional_nll_abs_diff": nll_abs_diff,
            "functional_nll_rel_diff": nll_rel_diff,
        }
        write_json(m_qual_path, merge_qualification)

    flops = training_matmul_flops(
        [len(r["input_ids"]) for r in rows],
        base_active_params,
        trainable_count,
        full_attention_layers=attn_layers,
        query_heads=q_heads,
        head_dim=h_dim,
    )
    in_toks = sum(len(r["input_ids"]) for r in rows)
    sup_toks = sum(loss_tokens(r) for r in rows)
    pad_toks = sum(s["padded_tokens"] for s in step_records)
    summary = {
        "status": "COMPLETE",
        "run_name": args.run_name,
        "architecture": spec["architecture"],
        "world_size": world,
        "micro_batch": args.micro_batch,
        "length_bucketing": args.length_bucketing,
        "activation_offloading": args.activation_offloading,
        "presentations": len(rows),
        "updates": len(groups),
        "input_tokens": in_toks,
        "supervised_tokens": sup_toks,
        "padded_tokens": pad_toks,
        "padding_over_input_fraction": (pad_toks - in_toks) / max(1, in_toks),
        "loop_seconds": loop_seconds,
        "checkpoint_save_seconds": checkpoint_save_seconds,
        "process_seconds": time.monotonic() - proc_start,
        "assigned_loop_gpu_hours": loop_seconds * world / 3600.0,
        "supervised_tokens_per_second": sup_toks / loop_seconds,
        "input_tokens_per_second": in_toks / loop_seconds,
        "before_nll": before_nll,
        "after_nll": after_nll,
        "peak_allocated_gib": max_peak_gib,
        "allocated_headroom_gib": headroom_gib,
        "board_joules": energy.get("board_joules"),
        "mfu": estimated_mfu(flops, loop_seconds, world),
        "merge_qualification": merge_qualification,
        "steps": step_records,
    }
    write_json(out_dir / "board_energy.json", energy)
    write_json(out_dir / "summary.json", summary)
    print(json.dumps({k: v for k, v in summary.items() if k != "steps"}, indent=2), flush=True)


if __name__ == "__main__":
    main()
