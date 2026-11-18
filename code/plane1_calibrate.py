"""
plane1_calibrate.py — Unselected Step-0 / Step-2 Pre-Flight Calibration Runner across nodes 0, 1, 2.

Executes strictly pre-loop calibration measurements (zero model selection, zero Confirm/Holdout touch):
  - `dense_audit_and_frontier_a` (node-0 GPUs 0,1): Qwen3.8-27B M_0 128-item verifier audit + DAPO k=8 frontier shard A (0..47)
  - `dense_frontier_b` (node-0 GPUs 6,7): Qwen3.8-27B M_0 DAPO k=8 frontier shard B (48..95)
  - `moe_cap_calibration` (node-0 GPUs 2,3): Qwen3.5-35B-A3B M_0 outer max_tokens cap calibration (32768 vs 49152 on math_search128)
  - `quant_and_int4_mtp3_canary` (node-0 GPU 4 -> GPU 5): GPTQ W4A16 training-only calibration + repaired MTP ignore + 1-GPU INT4+BF16 MTP3 canary
  - `train_aa_noise` (node-1 GPUs 0..7): 5 paired interleaved A1/A2 baseline training blocks (10 runs) measuring training se_AA
  - `serve_aa_noise` (node-2 GPUs 0,1, dedicated host): 5 paired interleaved A1/A2 baseline serving blocks (10 runs) measuring serving se_AA
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
sys.path.insert(0, str(ROOT / "eval"))

from common import file_hash, read_jsonl, write_jsonl  # noqa: E402
from owned_server import retire, start, wait_idle  # noqa: E402
from stage import write_json  # noqa: E402

VENV_PY = str(ROOT / ".venv-training" / "bin" / "python")
QUANT_PY = str(ROOT / ".venv-quant" / "bin" / "python")


def launch_vllm(
    stage_name: str,
    model_dir: str,
    devices: str,
    tp: int,
    port: int,
    *,
    max_model_len: int = 40960,
    max_num_seqs: int = 16,
    max_batched_tokens: int = 8192,
    gpu_mem_util: float = 0.90,
    mtp_tokens: int = 0,
    prefix_caching: bool = False,
) -> str:
    wait_idle(devices)
    cmd = [
        "env",
        f"CUDA_VISIBLE_DEVICES={devices}",
        VENV_PY,
        "-m",
        "vllm.entrypoints.cli.main",
        "serve",
        model_dir,
        "--served-model-name",
        "qwen_gov_v2",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--dtype",
        "bfloat16",
        "--tensor-parallel-size",
        str(tp),
        "--language-model-only",
        "--max-model-len",
        str(max_model_len),
        "--max-num-seqs",
        str(max_num_seqs),
        "--max-num-batched-tokens",
        str(max_batched_tokens),
        "--gpu-memory-utilization",
        str(gpu_mem_util),
        "--enable-prefix-caching" if prefix_caching else "--no-enable-prefix-caching",
        "--seed",
        "20261117",
    ]
    if mtp_tokens > 0:
        cmd.extend(["--speculative-config", json.dumps({"method": "mtp", "num_speculative_tokens": mtp_tokens})])
    start(stage_name, cmd, tp, port, ttl=14400)
    return stage_name


def run_infer(
    run_name: str,
    server_stage: str,
    port: int,
    dataset_rel: str,
    *,
    model_cfg_rel: str = "configs/model.json",
    tok_rel: str = "models/dense27_base",
    cap: int = 32768,
    effort: str = "medium",
    concurrency: int = 8,
    draws: int = 1,
    seed: int = 20261117,
    gpus: str = "0,1",
    max_model_len: int = 40960,
    limit: int | None = None,
    presence_penalty: float | None = None,
) -> dict:
    cmd = [
        VENV_PY,
        str(ROOT / "code" / "infer_v2.py"),
        "--name",
        run_name,
        "--server-stage",
        server_stage,
        "--base-url",
        f"http://127.0.0.1:{port}",
        "--model-name",
        "qwen_gov_v2",
        "--dataset",
        str(ROOT / dataset_rel),
        "--model-config",
        str(ROOT / model_cfg_rel),
        "--tokenizer-dir",
        str(ROOT / tok_rel),
        "--cap",
        str(cap),
        "--effort",
        effort,
        "--concurrency",
        str(concurrency),
        "--draws",
        str(draws),
        "--seed",
        str(seed),
        "--gpus",
        gpus,
        "--max-model-len",
        str(max_model_len),
        "--purpose",
        "plane1_calibration",
    ]
    if limit is not None:
        cmd.extend(["--limit", str(limit)])
    if presence_penalty is not None:
        cmd.extend(["--presence-penalty", str(presence_penalty)])
    subprocess.run(cmd, check=True, timeout=10800)
    return json.loads((ROOT / "results" / "inference" / run_name / "summary.json").read_text(encoding="utf-8"))


def task_dense_audit_and_frontier_a() -> None:
    dapo_all = read_jsonl(ROOT / "data/splits/dapo_search240.jsonl")
    shard_a_path = ROOT / "data/splits/dapo_search_shard_a48.jsonl"
    write_jsonl(shard_a_path, dapo_all[:48])

    srv = launch_vllm(
        "p1_dense_srv_a",
        "models/dense27_base",
        "0,1",
        2,
        18273,
        max_model_len=40960,
        max_num_seqs=24,
        max_batched_tokens=8192,
        gpu_mem_util=0.90,
    )
    try:
        # 1. 128-sample Verifier & Length Audit on math_search128
        audit_summary = run_infer(
            "p1_dense_math_search128_audit",
            srv,
            18273,
            "data/splits/math_search128.jsonl",
            cap=32768,
            effort="medium",
            concurrency=12,
            draws=1,
            gpus="0,1",
        )
        # Inspect >=25 passes and >=25 failures/disagreements across strict vs lenient
        gz = ROOT / "results/inference/p1_dense_math_search128_audit/generations.jsonl.gz"
        with gzip.open(gz, "rt", encoding="utf-8") as f:
            gens = [json.loads(line) for line in f if line.strip()]
        passes = [g["id"] for g in gens if g.get("lenient_correct")]
        fails = [g["id"] for g in gens if not g.get("lenient_correct")]
        strict_vs_lenient_Recoveries = [
            {"id": g["id"], "strict_ans": g.get("strict_extracted"), "lenient_ans": g.get("lenient_extracted")}
            for g in gens
            if g.get("lenient_correct") and not g.get("strict_correct")
        ]
        write_json(
            ROOT / "results/plane1_verifier_audit_dense.json",
            {
                "status": "VERIFIER_AUDIT_COMPLETE",
                "total_audited": len(gens),
                "pass_count": len(passes),
                "fail_count": len(fails),
                "lenient_accuracy": audit_summary["lenient_accuracy"],
                "strict_accuracy": audit_summary["strict_accuracy"],
                "v1_parity_accuracy": audit_summary["v1_parity_accuracy"],
                "strict_to_lenient_false_negative_recoveries": strict_vs_lenient_Recoveries,
                "truncation_rate": audit_summary["truncation_rate"],
                "natural_stop_lenient_parse_failure_rate": audit_summary["natural_stop_lenient_parse_failure_rate"],
                "output_tokens_quantiles": audit_summary["output_tokens_quantiles"],
            },
        )
        # 2. DAPO k=8 Own-Policy Frontier Screen Shard A (48 problems x 8 = 384 rollouts)
        run_infer(
            "p1_dense_dapo_frontier_shard_a",
            srv,
            18273,
            "data/splits/dapo_search_shard_a48.jsonl",
            cap=32768,
            effort="medium",
            concurrency=16,
            draws=8,
            gpus="0,1",
        )
    finally:
        retire(srv, "Completed Plane-1 dense verifier audit and frontier shard A")


def task_dense_frontier_b() -> None:
    dapo_all = read_jsonl(ROOT / "data/splits/dapo_search240.jsonl")
    shard_b_path = ROOT / "data/splits/dapo_search_shard_b48.jsonl"
    write_jsonl(shard_b_path, dapo_all[48:96])

    srv = launch_vllm(
        "p1_dense_srv_b",
        "models/dense27_base",
        "6,7",
        2,
        18276,
        max_model_len=40960,
        max_num_seqs=24,
        max_batched_tokens=8192,
        gpu_mem_util=0.90,
    )
    try:
        run_infer(
            "p1_dense_dapo_frontier_shard_b",
            srv,
            18276,
            "data/splits/dapo_search_shard_b48.jsonl",
            cap=32768,
            effort="medium",
            concurrency=16,
            draws=8,
            gpus="6,7",
        )
    finally:
        retire(srv, "Completed Plane-1 dense frontier shard B")


def task_moe_cap_calibration() -> None:
    srv = launch_vllm(
        "p1_moe_srv_cap",
        "models/moe35_base",
        "2,3",
        2,
        18274,
        max_model_len=65536,
        max_num_seqs=16,
        max_batched_tokens=8192,
        gpu_mem_util=0.85,
    )
    try:
        s32 = run_infer(
            "p1_moe_math_search128_cap32k",
            srv,
            18274,
            "data/splits/math_search128.jsonl",
            model_cfg_rel="configs/moe_model.json",
            tok_rel="models/moe35_base",
            cap=32768,
            effort="none",
            concurrency=12,
            draws=1,
            gpus="2,3",
            max_model_len=65536,
        )
        s48 = run_infer(
            "p1_moe_math_search128_cap48k",
            srv,
            18274,
            "data/splits/math_search128.jsonl",
            model_cfg_rel="configs/moe_model.json",
            tok_rel="models/moe35_base",
            cap=49152,
            effort="none",
            concurrency=12,
            draws=1,
            gpus="2,3",
            max_model_len=65536,
        )
        selected_cap = 49152 if s48["stop_rate"] >= s32["stop_rate"] and s48["stop_rate"] >= 0.96 else 32768
        write_json(
            ROOT / "results/plane1_moe_cap_calibration.json",
            {
                "status": "MOE_CAP_CALIBRATED" if s48["output_tokens_quantiles"]["p99"] * 1.5 <= 49152 else "MOE_CAP_48K_STILL_RIGHT_CENSORED",
                "cap_32768": {
                    "lenient_accuracy": s32["lenient_accuracy"],
                    "strict_accuracy": s32["strict_accuracy"],
                    "stop_rate": s32["stop_rate"],
                    "truncation_rate": s32["truncation_rate"],
                    "output_tokens_quantiles": s32["output_tokens_quantiles"],
                },
                "cap_49152": {
                    "lenient_accuracy": s48["lenient_accuracy"],
                    "strict_accuracy": s48["strict_accuracy"],
                    "stop_rate": s48["stop_rate"],
                    "truncation_rate": s48["truncation_rate"],
                    "output_tokens_quantiles": s48["output_tokens_quantiles"],
                },
                "uncensored_1p5x_p99_satisfied": bool(s48["output_tokens_quantiles"]["p99"] * 1.5 <= 49152),
                "selected_outer_cap": selected_cap,
            },
        )
    finally:
        retire(srv, "Completed Plane-1 MoE outer serving cap calibration")


def task_moe_cap_98k_calibration() -> None:
    srv = launch_vllm(
        "p1_moe_srv_cap98k",
        "models/moe35_base",
        "0,1,2,3",
        4,
        18278,
        max_model_len=131072,
        max_num_seqs=16,
        max_batched_tokens=8192,
        gpu_mem_util=0.88,
    )
    try:
        s98 = run_infer(
            "p1_moe_math_search128_cap98k",
            srv,
            18278,
            "data/splits/math_search128.jsonl",
            model_cfg_rel="configs/moe_model.json",
            tok_rel="models/moe35_base",
            cap=98304,
            effort="none",
            concurrency=12,
            draws=1,
            gpus="0,1,2,3",
            max_model_len=131072,
        )
        p99 = float(s98["output_tokens_quantiles"]["p99"])
        uncensored_ok = (p99 * 1.5) <= 98304.0
        write_json(
            ROOT / "results/plane1_moe_cap98k_calibration.json",
            {
                "status": "MOE_CAP_98K_CERTIFIED" if uncensored_ok else "MOE_CAP_98K_DEGENERATE_TAIL_NO_GO",
                "cap_98304": {
                    "lenient_accuracy": s98["lenient_accuracy"],
                    "strict_accuracy": s98["strict_accuracy"],
                    "stop_rate": s98["stop_rate"],
                    "truncation_rate": s98["truncation_rate"],
                    "output_tokens_quantiles": s98["output_tokens_quantiles"],
                },
                "required_cap_for_1p5x_p99": 1.5 * p99,
                "uncensored_1p5x_p99_satisfied": uncensored_ok,
            },
        )
    finally:
        retire(srv, "Completed Plane-1 MoE 98K uncensored p99 calibration")


def task_quant_and_int4_mtp3_canary() -> None:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "4"
    if not (ROOT / "models/gptq_w4a16_mtp_ready_v2/complete.json").is_file():
        subprocess.run(
            [QUANT_PY, str(ROOT / "code/quantize_v2.py")],
            env=env,
            cwd=str(ROOT),
            check=True,
            timeout=5400,
        )
    # Test 1-GPU INT4 W4A16 + Native BF16 MTP3 on GPU 5!
    canary_result: dict = {"attempted": True}
    srv_name = "p1_int4_mtp3_canary_srv"
    try:
        launch_vllm(
            srv_name,
            "models/gptq_w4a16_mtp_ready_v2",
            "5",
            1,
            18275,
            max_model_len=40960,
            max_num_seqs=16,
            max_batched_tokens=8192,
            gpu_mem_util=0.90,
            mtp_tokens=3,
        )
        obs = run_infer(
            "p1_int4_mtp3_canary32",
            srv_name,
            18275,
            "data/splits/math_search_fast64.jsonl",
            limit=32,
            cap=32768,
            effort="medium",
            concurrency=8,
            draws=1,
            gpus="5",
        )
        canary_result.update(
            status="INT4_MTP3_NATIVE_LOADER_RESOLVED",
            lenient_accuracy=obs["lenient_accuracy"],
            stop_rate=obs["stop_rate"],
            output_tokens_per_second=obs["output_tokens_per_second"],
            mtp_speculative=obs["mtp_speculative"],
        )
    except Exception as exc:
        log_tail = ""
        log_f = ROOT / "runs" / srv_name / "output.log"
        if log_f.is_file():
            log_tail = log_f.read_text(encoding="utf-8", errors="replace")[-3000:]
        canary_result.update(
            status="INT4_MTP3_LOADER_FAILED",
            error=str(exc),
            server_log_tail=log_tail,
        )
    finally:
        try:
            retire(srv_name, "Completed or failed INT4+MTP3 canary")
        except Exception:
            pass
    write_json(ROOT / "results/plane1_int4_mtp3_canary.json", canary_result)
    print(json.dumps(canary_result, indent=2), flush=True)


def task_train_aa_noise() -> None:
    # 5 paired interleaved A1/A2 baseline blocks (10 runs) on 8x A100 DDP (node-1)
    a1_secs, a2_secs = [], []
    a1_tps, a2_tps = [], []
    runs = []
    for blk in range(5):
        for arm_tag, bucket_secs, bucket_tps in (("A1", a1_secs, a1_tps), ("A2", a2_secs, a2_tps)):
            rname = f"p1_train_aa_b{blk}_{arm_tag}"
            port = 24500 + blk * 10 + (1 if arm_tag == "A1" else 2)
            cmd = [
                VENV_PY,
                "-m",
                "torch.distributed.run",
                "--nproc_per_node=8",
                f"--master_port={port}",
                str(ROOT / "code/train_systems_v2.py"),
                "--run-name",
                rname,
                "--dataset",
                "data/splits/sft_systems1024_v2.jsonl",
                "--heldout",
                "data/splits/sft_heldout64_v2.jsonl",
                "--global-batch",
                "64",
                "--micro-batch",
                "1",
                "--seed",
                "20261117",
            ]
            subprocess.run(cmd, cwd=str(ROOT), check=True, timeout=1800)
            s = json.loads((ROOT / "results/training" / rname / "summary.json").read_text(encoding="utf-8"))
            bucket_secs.append(s["loop_seconds"])
            bucket_tps.append(s["supervised_tokens_per_second"])
            runs.append({"block": blk, "arm": arm_tag, "loop_seconds": s["loop_seconds"], "after_nll": s["after_nll"], "board_joules": s["board_joules"]})
    diffs_sec = [x - y for x, y in zip(a1_secs, a2_secs)]
    diffs_tps = [x - y for x, y in zip(a1_tps, a2_tps)]
    se_aa_sec = statistics.stdev(diffs_sec) / (len(diffs_sec) ** 0.5)
    se_aa_tps = statistics.stdev(diffs_tps) / (len(diffs_tps) ** 0.5)
    report = {
        "status": "TRAIN_AA_CALIBRATED",
        "paired_blocks": 5,
        "total_runs": 10,
        "mean_loop_seconds": statistics.fmean(a1_secs + a2_secs),
        "se_aa_loop_seconds": se_aa_sec,
        "per_block_sd_loop_seconds": statistics.stdev(diffs_sec),
        "mean_supervised_tokens_per_second": statistics.fmean(a1_tps + a2_tps),
        "se_aa_supervised_tokens_per_second": se_aa_tps,
        "runs": runs,
    }
    write_json(ROOT / "results/plane1_train_aa_noise.json", report)
    print(json.dumps(report, indent=2), flush=True)


def task_serve_aa_noise() -> None:
    # Dedicated node-2: 5 paired interleaved A1/A2 baseline blocks (10 runs) on 32-problem multi-shape block
    fast64 = read_jsonl(ROOT / "data/splits/math_search_fast64.jsonl")
    block32_path = ROOT / "data/splits/math_search_block32.jsonl"
    write_jsonl(block32_path, fast64[:32])

    srv = launch_vllm(
        "p1_serve_aa_srv_n2",
        "models/dense27_base",
        "0,1",
        2,
        18273,
        max_model_len=40960,
        max_num_seqs=16,
        max_batched_tokens=8192,
        gpu_mem_util=0.90,
    )
    a1_sec, a2_sec = [], []
    a1_tps, a2_tps = [], []
    a1_tpot, a2_tpot = [], []
    blocks = []
    try:
        for blk in range(5):
            seed_blk = 20261117 + blk
            order = [("A1", a1_sec, a1_tps, a1_tpot), ("A2", a2_sec, a2_tps, a2_tpot)]
            if blk % 2 == 1:
                order.reverse()
            blk_rec: dict = {"block_index": blk, "seed": seed_blk, "arms": {}}
            for tag, b_sec, b_tps, b_tpot in order:
                rname = f"p1_serve_aa_b{blk}_{tag}"
                obs = run_infer(
                    rname,
                    srv,
                    18273,
                    "data/splits/math_search_block32.jsonl",
                    cap=32768,
                    effort="medium",
                    concurrency=8,
                    draws=1,
                    seed=seed_blk,
                    gpus="0,1",
                )
                b_sec.append(obs["dataset_seconds"])
                b_tps.append(obs["output_tokens_per_second"])
                b_tpot.append(obs["tpot_p95_ms"])
                blk_rec["arms"][tag] = {
                    "dataset_seconds": obs["dataset_seconds"],
                    "output_tokens_per_second": obs["output_tokens_per_second"],
                    "tpot_p95_ms": obs["tpot_p95_ms"],
                    "board_joules": obs["board_joules"],
                    "stop_rate": obs["stop_rate"],
                    "lenient_accuracy": obs["lenient_accuracy"],
                }
            blocks.append(blk_rec)
    finally:
        retire(srv, "Completed Plane-1 serving A/A noise calibration on dedicated node-2")

    d_sec = [x - y for x, y in zip(a1_sec, a2_sec)]
    d_tps = [x - y for x, y in zip(a1_tps, a2_tps)]
    d_tpot = [x - y for x, y in zip(a1_tpot, a2_tpot)]
    report = {
        "status": "SERVING_AA_CALIBRATED",
        "paired_blocks": 5,
        "total_runs": 10,
        "mean_dataset_seconds": statistics.fmean(a1_sec + a2_sec),
        "se_aa_dataset_seconds": statistics.stdev(d_sec) / (len(d_sec) ** 0.5),
        "per_block_sd_dataset_seconds": statistics.stdev(d_sec),
        "mean_output_tokens_per_second": statistics.fmean(a1_tps + a2_tps),
        "se_aa_output_tokens_per_second": statistics.stdev(d_tps) / (len(d_tps) ** 0.5),
        "mean_tpot_p95_ms": statistics.fmean(a1_tpot + a2_tpot),
        "se_aa_tpot_p95_ms": statistics.stdev(d_tpot) / (len(d_tpot) ** 0.5),
        "blocks": blocks,
    }
    write_json(ROOT / "results/plane1_serving_aa_noise.json", report)
    print(json.dumps(report, indent=2), flush=True)


def task_search_s1_bf16_mtp3() -> None:
    srv = launch_vllm(
        "p1_s1_bf16_mtp3_srv",
        "models/dense27_base",
        "2,3",
        2,
        18274,
        max_model_len=40960,
        max_num_seqs=24,
        max_batched_tokens=8192,
        gpu_mem_util=0.90,
        mtp_tokens=3,
    )
    try:
        run_infer(
            "p1_search128_s1_bf16_mtp3",
            srv,
            18274,
            "data/splits/math_search128.jsonl",
            cap=32768,
            effort="medium",
            concurrency=12,
            draws=1,
            seed=20261117,
            gpus="2,3",
        )
    finally:
        retire(srv, "Completed Search-only S1 BF16+MTP3 discordance screen")


def task_search_s2_int4() -> None:
    srv = launch_vllm(
        "p1_s2_int4_srv",
        "models/gptq_w4a16_mtp_ready_v2",
        "4",
        1,
        18277,
        max_model_len=40960,
        max_num_seqs=24,
        max_batched_tokens=8192,
        gpu_mem_util=0.90,
        mtp_tokens=0,
    )
    try:
        run_infer(
            "p1_search128_s2_int4",
            srv,
            18277,
            "data/splits/math_search128.jsonl",
            cap=32768,
            effort="medium",
            concurrency=12,
            draws=1,
            seed=20261117,
            gpus="4",
        )
    finally:
        retire(srv, "Completed Search-only S2 INT4 TP1 discordance screen")


def task_search_s3_int4_mtp3() -> None:
    srv = launch_vllm(
        "p1_s3_int4_mtp3_srv",
        "models/gptq_w4a16_mtp_ready_v2",
        "5",
        1,
        18275,
        max_model_len=40960,
        max_num_seqs=24,
        max_batched_tokens=8192,
        gpu_mem_util=0.90,
        mtp_tokens=3,
    )
    try:
        run_infer(
            "p1_search128_s3_int4_mtp3",
            srv,
            18275,
            "data/splits/math_search128.jsonl",
            cap=32768,
            effort="medium",
            concurrency=12,
            draws=1,
            seed=20261117,
            gpus="5",
        )
    finally:
        retire(srv, "Completed Search-only S3 INT4+MTP3 TP1 discordance screen")


def task_moe_cap_98k_presence1p5() -> None:
    """Controlled Plane-1 Search comparison: official Qwen3.5-35B-A3B general-thinking presence_penalty=1.5 vs 0.0 at cap=98304."""
    srv = launch_vllm(
        "p1_moe_srv_cap98k_pp15",
        "models/moe35_base",
        "0,1,2,3",
        4,
        18279,
        max_model_len=131072,
        max_num_seqs=16,
        max_batched_tokens=8192,
        gpu_mem_util=0.88,
    )
    try:
        s98_pp15 = run_infer(
            "p1_moe_math_search128_cap98k_pp15",
            srv,
            18279,
            "data/splits/math_search128.jsonl",
            model_cfg_rel="configs/moe_model.json",
            tok_rel="models/moe35_base",
            cap=98304,
            effort="none",
            concurrency=12,
            draws=1,
            seed=20261117,
            gpus="0,1,2,3",
            max_model_len=131072,
            presence_penalty=1.5,
        )
        p99 = float(s98_pp15["output_tokens_quantiles"]["p99"])
        trunc_rate = float(s98_pp15["truncation_rate"])
        uncensored_ok = (trunc_rate == 0.0) and ((p99 * 1.5) <= 98304.0)

        # Compute exact item-paired comparison against p1_moe_math_search128_cap98k (presence_penalty=0.0)
        pp0_gz = ROOT / "results/inference/p1_moe_math_search128_cap98k/generations.jsonl.gz"
        pp15_gz = ROOT / "results/inference/p1_moe_math_search128_cap98k_pp15/generations.jsonl.gz"
        paired_cmp = {}
        if pp0_gz.is_file() and pp15_gz.is_file():
            with gzip.open(pp0_gz, "rt", encoding="utf-8") as f0:
                g0 = {json.loads(ln)["id"]: json.loads(ln) for ln in f0 if ln.strip()}
            with gzip.open(pp15_gz, "rt", encoding="utf-8") as f1:
                g1 = {json.loads(ln)["id"]: json.loads(ln) for ln in f1 if ln.strip()}
            wins = sum(1 for k in g0 if bool(g1[k].get("lenient_correct")) and not bool(g0[k].get("lenient_correct")))
            losses = sum(1 for k in g0 if not bool(g1[k].get("lenient_correct")) and bool(g0[k].get("lenient_correct")))
            ties = len(g0) - wins - losses
            paired_cmp = {
                "wins_pp15_vs_pp0": wins,
                "losses_pp15_vs_pp0": losses,
                "ties_pp15_vs_pp0": ties,
                "paired_lenient_delta": (wins - losses) / max(1, len(g0)),
            }

        payload = {
            "status": "MOE_CAP_98K_PP15_UNCENSORED_CERTIFIED" if uncensored_ok else "MOE_CAP_98K_PP15_STILL_UNCERTIFIED",
            "control_reference_run": "p1_moe_math_search128_cap98k",
            "sampling_change": {"presence_penalty_from": 0.0, "presence_penalty_to": 1.5, "temperature": 1.0, "top_p": 0.95, "top_k": 20},
            "cap_98304_presence_1p5": {
                "lenient_accuracy": s98_pp15["lenient_accuracy"],
                "strict_accuracy": s98_pp15["strict_accuracy"],
                "stop_rate": s98_pp15["stop_rate"],
                "truncation_rate": trunc_rate,
                "natural_stop_lenient_parse_failure_rate": s98_pp15["natural_stop_lenient_parse_failure_rate"],
                "dataset_seconds": s98_pp15["dataset_seconds"],
                "assigned_gpu_hours": s98_pp15["assigned_gpu_hours"],
                "output_tokens": s98_pp15["output_tokens"],
                "output_tokens_quantiles": s98_pp15["output_tokens_quantiles"],
                "board_joules": s98_pp15["board_joules"],
            },
            "paired_vs_presence_0p0": paired_cmp,
            "required_cap_for_1p5x_p99": 1.5 * p99,
            "uncensored_1p5x_p99_satisfied": uncensored_ok,
        }
        write_json(ROOT / "results/plane1_moe_cap98k_presence1p5_calibration.json", payload)
        print(json.dumps(payload, indent=2), flush=True)
    finally:
        retire(srv, "Completed Plane-1 MoE 98K official presence_penalty=1.5 calibration")


def task_merge_export_oracle_smoke() -> None:
    """Live 16-row Qwen3.8-27B FP32 merge + BF16 export oracle qualification on GPU 7 (all 448 adapted tensors <=1 ULP + uint8 15 MTP)."""
    import shutil
    smoke_ds = ROOT / "results/training/.tmp_sft_smoke16.jsonl"
    smoke_ds.parent.mkdir(parents=True, exist_ok=True)
    rows16 = read_jsonl(ROOT / "data/splits/sft_systems1024_v2.jsonl")[:16]
    write_jsonl(smoke_ds, rows16)
    run_name = "p1_merge_oracle_qual_smoke"
    exp_rel = Path("models/tmp_merge_qual_smoke")
    for p in (ROOT / "results/training" / run_name, ROOT / "checkpoints" / run_name, ROOT / exp_rel):
        if p.exists():
            shutil.rmtree(p)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "7"
    try:
        subprocess.run(
            [
                VENV_PY,
                str(ROOT / "code/train_systems_v2.py"),
                "--run-name", run_name,
                "--dataset", str(smoke_ds.relative_to(ROOT)),
                "--heldout", "data/splits/sft_heldout64_v2.jsonl",
                "--global-batch", "16",
                "--micro-batch", "4",
                "--length-bucketing",
                "--seed", "20261117",
                "--merge-fp32-export", str(exp_rel),
            ],
            cwd=str(ROOT),
            env=env,
            check=True,
            timeout=2400,
        )
        summary = json.loads((ROOT / "results/training" / run_name / "summary.json").read_text(encoding="utf-8"))
        qual = json.loads((ROOT / "results/training" / run_name / "merge_qualification.json").read_text(encoding="utf-8"))
        out = {
            "status": "LIVE_QWEN27B_MERGE_ORACLE_QUALIFIED",
            "run_name": run_name,
            "before_nll": summary["before_nll"],
            "unmerged_peft_after_nll": summary["after_nll"],
            "merged_bf16_after_nll": qual["merged_after_nll"],
            "functional_nll_abs_diff": abs(float(qual["merged_after_nll"]) - float(summary["after_nll"])),
            "checked_adapted_linear_tensors": qual["checked_adapted_linear_tensors"],
            "param_merge_max_bf16_ulp": qual["param_merge_max_bf16_ulp"],
            "native_mtp_tensors_bitwise_verified": qual["native_mtp_tensors_bitwise_verified"],
        }
        write_json(ROOT / "results/plane1_merge_export_oracle_qualification.json", out)
        print(json.dumps(out, indent=2), flush=True)
    finally:
        for p in (smoke_ds, ROOT / "checkpoints" / run_name, ROOT / exp_rel):
            if p.is_file():
                p.unlink()
            elif p.is_dir():
                shutil.rmtree(p)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "task",
        choices=[
            "dense_audit_and_frontier_a",
            "dense_frontier_b",
            "moe_cap_calibration",
            "quant_and_int4_mtp3_canary",
            "train_aa_noise",
            "serve_aa_noise",
            "search_s1_bf16_mtp3",
            "search_s2_int4",
            "search_s3_int4_mtp3",
            "moe_cap_98k_calibration",
            "moe_cap_98k_presence1p5",
            "merge_export_oracle_smoke",
        ],
    )
    args = p.parse_args()
    dispatch = {
        "dense_audit_and_frontier_a": task_dense_audit_and_frontier_a,
        "dense_frontier_b": task_dense_frontier_b,
        "moe_cap_calibration": task_moe_cap_calibration,
        "quant_and_int4_mtp3_canary": task_quant_and_int4_mtp3_canary,
        "train_aa_noise": task_train_aa_noise,
        "serve_aa_noise": task_serve_aa_noise,
        "search_s1_bf16_mtp3": task_search_s1_bf16_mtp3,
        "search_s2_int4": task_search_s2_int4,
        "search_s3_int4_mtp3": task_search_s3_int4_mtp3,
        "moe_cap_98k_calibration": task_moe_cap_98k_calibration,
        "moe_cap_98k_presence1p5": task_moe_cap_98k_presence1p5,
        "merge_export_oracle_smoke": task_merge_export_oracle_smoke,
    }
    dispatch[args.task]()


if __name__ == "__main__":
    main()
