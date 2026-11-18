#!/usr/bin/env python3
"""
isolated_scorer.py — Out-of-process hermetic evaluator for AILab_Qwen27B_Governed_v2.

Always invoke with `python -I -B eval/isolated_scorer.py` so candidate working directories,
user site-packages, and environment hooks cannot shadow scoring logic.

Supports two evaluation modes:
  1. `capability`: Grades raw generation JSONL/JSONL.GZ against a frozen split (or externally stored
     Protected Holdout labels), verifying 100% item-ID coverage, zero protocol/verifier errors,
     decoupled strict vs lenient accuracy, separated truncation_rate vs natural_stop_parse_failure_rate,
     and problem-level cluster IDs (`cluster_id = problem_id`, NEVER rollout draw ID).
  2. `systems_abab`: Aggregates interleaved ABAB benchmark block summaries (`>=10-20` paired blocks),
     verifying every block passed completion/quality/NLL admission, computing per-block latency/throughput/
     energy/goodput metrics with `cluster_id = block_id` (NEVER request ID).
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any

EVAL_DIR = Path(__file__).resolve().parent
ROOT = EVAL_DIR.parent
sys.path.insert(0, str(ROOT / "code"))
sys.path.insert(0, str(EVAL_DIR))

from common import read_jsonl  # noqa: E402
from verifiers_v2 import score_response  # noqa: E402


def _read_generations(path: Path) -> list[dict[str, Any]]:
    if path.name.endswith(".gz"):
        with gzip.open(path, "rt", encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]
    return read_jsonl(path)


def _write_strict_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def score_capability(args: argparse.Namespace) -> int:
    prompts = read_jsonl(Path(args.split_prompts).resolve(strict=True))
    labels_map: dict[str, str] = {}
    task_source_map: dict[str, str] = {}
    mcq_options_map: dict[str, int] = {}

    pending_holdout_commit: tuple[Path, dict[str, Any]] | None = None
    if args.external_labels:
        ext_path = Path(args.external_labels).resolve(strict=True)
        # B8: Pre-check single-touch budget BEFORE scoring, but commit touch ONLY AFTER full input validation!
        if args.holdout_registry:
            reg_path = Path(args.holdout_registry).resolve()
            reg = json.loads(reg_path.read_text(encoding="utf-8")) if reg_path.is_file() else {"touches": {}}
            split_key = ext_path.name
            touched_candidates = reg["touches"].setdefault(split_key, [])
            if args.candidate_id in touched_candidates:
                raise RuntimeError(f"SINGLE_TOUCH_VIOLATION: {args.candidate_id} already evaluated on holdout {split_key}")
            if len(touched_candidates) >= 2:
                raise RuntimeError(f"SINGLE_TOUCH_BUDGET_EXCEEDED: holdout {split_key} allows only [B_0, final_A*], got {touched_candidates}")
            touched_candidates.append(args.candidate_id)
            pending_holdout_commit = (reg_path, reg)
        ext = json.loads(ext_path.read_text(encoding="utf-8"))
        labels_map = {str(k): str(v) for k, v in ext["labels"].items()}
    for row in prompts:
        pid = str(row["id"])
        task_source_map[pid] = str(row.get("source", "math"))
        if "options" in row and isinstance(row["options"], list):
            mcq_options_map[pid] = len(row["options"])
        if not args.external_labels:
            if "gold_answer" not in row or row["gold_answer"] is None:
                raise ValueError(f"Missing gold_answer for {pid}; pass --external-labels for holdout splits")
            labels_map[pid] = str(row["gold_answer"])

    if set(labels_map.keys()) != {str(r["id"]) for r in prompts}:
        raise RuntimeError("Prompt set and gold label set mismatch")

    gens = _read_generations(Path(args.generations).resolve(strict=True))
    expected_draws = int(args.draws)
    expected_pairs: dict[str, list[dict[str, Any]]] = {str(r["id"]): [] for r in prompts}

    infra_errors = 0
    for g in gens:
        pid = str(g["id"])
        if pid not in expected_pairs:
            raise RuntimeError(f"Unexpected generation problem id: {pid}")
        if g.get("error") is not None or g.get("finish_reason") not in ("stop", "length"):
            infra_errors += 1
        expected_pairs[pid].append(g)

    for pid, glist in expected_pairs.items():
        if len(glist) != expected_draws:
            raise RuntimeError(f"Anti-survivorship check failed: {pid} has {len(glist)} draws != {expected_draws}")

    items_out: list[dict[str, Any]] = []
    strict_scores: list[float] = []
    v1_scores: list[float] = []
    lenient_scores: list[float] = []
    stop_flags: list[float] = []
    trunc_flags: list[float] = []
    nat_strict_pf_flags: list[float] = []
    nat_lenient_pf_flags: list[float] = []
    verifier_errors = 0
    output_tokens_list: list[float] = []
    problem_any_solved_lenient = 0
    problem_any_solved_strict = 0

    for row in sorted(prompts, key=lambda x: str(x["id"])):
        pid = str(row["id"])
        gold = labels_map[pid]
        src = task_source_map[pid]
        is_int = src in {"dapo", "aime24", "aime25"}
        is_mcq = pid in mcq_options_map
        glist = sorted(expected_pairs[pid], key=lambda x: int(x["seed"]))
        seen_seeds = [int(g["seed"]) for g in glist]
        if len(set(seen_seeds)) != expected_draws:
            raise RuntimeError(f"Duplicate or missing explicit draw seeds for {pid}: {seen_seeds}")
        if args.base_seed is not None:
            from common import object_hash
            req_seeds = sorted(
                int(object_hash({"seed": int(args.base_seed), "id": pid, "draw": d})[:8], 16)
                for d in range(expected_draws)
            )
            if seen_seeds != req_seeds:
                raise RuntimeError(f"Seed pairing mismatch for {pid}: expected {req_seeds}, got {seen_seeds}")

        prob_lenient_hits = 0
        prob_strict_hits = 0
        for d_idx, g in enumerate(glist):
            text = str(g.get("text", ""))
            fin = str(g.get("finish_reason", "error"))
            toks = float(g.get("output_tokens") or 0)
            output_tokens_list.append(toks)
            if is_mcq:
                from mcq_contract import mcq_answer
                ans = mcq_answer(text, thinking=args.thinking, option_count=mcq_options_map[pid]) if fin == "stop" else None
                ok = bool(fin == "stop" and g.get("error") is None and ans == gold)
                s_res = {
                    "truncated": fin == "length",
                    "stopped": fin == "stop" and g.get("error") is None,
                    "strict_correct": ok,
                    "v1_correct": ok,
                    "lenient_correct": ok,
                    "natural_stop_strict_parse_failure": bool(fin == "stop" and ans is None),
                    "natural_stop_lenient_parse_failure": bool(fin == "stop" and ans is None),
                    "verifier_error": None,
                }
            else:
                s_res = score_response(
                    text,
                    gold,
                    finish_reason=fin if g.get("error") is None else "error",
                    thinking=args.thinking,
                    integer_task=is_int,
                    problem=row["problem"],
                )
            if s_res["verifier_error"] is not None:
                verifier_errors += 1
            val = 1.0 if s_res["lenient_correct"] else 0.0
            s_val = 1.0 if s_res["strict_correct"] else 0.0
            v1_val = 1.0 if s_res["v1_correct"] else 0.0
            prob_lenient_hits += int(s_res["lenient_correct"])
            prob_strict_hits += int(s_res["strict_correct"])

            item_id = pid if expected_draws == 1 else f"{pid}#d{d_idx}"
            # CRITICAL: cluster_id is ALWAYS the unique problem ID `pid`, NEVER `item_id`!
            items_out.append({
                "id": item_id,
                "cluster_id": pid,
                "value": val if args.primary_flavor == "lenient" else s_val,
            })
            lenient_scores.append(val)
            strict_scores.append(s_val)
            v1_scores.append(v1_val)
            stop_flags.append(1.0 if s_res["stopped"] else 0.0)
            trunc_flags.append(1.0 if s_res["truncated"] else 0.0)
            nat_strict_pf_flags.append(1.0 if s_res["natural_stop_strict_parse_failure"] else 0.0)
            nat_lenient_pf_flags.append(1.0 if s_res["natural_stop_lenient_parse_failure"] else 0.0)

        if prob_lenient_hits > 0:
            problem_any_solved_lenient += 1
        if prob_strict_hits > 0:
            problem_any_solved_strict += 1

    n_total = len(lenient_scores)
    n_probs = len(prompts)
    metrics = {
        "lenient_accuracy": statistics.fmean(lenient_scores),
        "strict_accuracy": statistics.fmean(strict_scores),
        "v1_parity_accuracy": statistics.fmean(v1_scores),
        "pass_at_k_lenient": problem_any_solved_lenient / n_probs,
        "pass_at_k_strict": problem_any_solved_strict / n_probs,
        "stop_rate": statistics.fmean(stop_flags),
        "truncation_rate": statistics.fmean(trunc_flags),
        "natural_stop_strict_parse_failure_rate": statistics.fmean(nat_strict_pf_flags),
        "natural_stop_lenient_parse_failure_rate": statistics.fmean(nat_lenient_pf_flags),
        "combined_unusable_rate": statistics.fmean(trunc_flags) + statistics.fmean(nat_lenient_pf_flags),
        "infrastructure_error_count": float(infra_errors),
        "verifier_error_count": float(verifier_errors),
        "mean_output_tokens": statistics.fmean(output_tokens_list),
        "median_output_tokens": statistics.median(output_tokens_list),
        "unique_problems": float(n_probs),
        "total_generations": float(n_total),
    }
    reserved_keys = set(metrics.keys())
    if args.extra_metrics_json:
        extra = json.loads(Path(args.extra_metrics_json).read_text(encoding="utf-8"))
        for k, v in extra.items():
            if k in reserved_keys:
                raise ValueError(f"B6_RESERVED_METRIC_OVERRIDE_FORBIDDEN:{k}")
            if isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(float(v)):
                metrics[k] = float(v)

    payload = {
        "candidate_id": args.candidate_id,
        "protocol_id": args.protocol_id,
        "env_fingerprint": args.env_fingerprint,
        "scorer_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "generations_sha256": hashlib.sha256(Path(args.generations).read_bytes()).hexdigest(),
        "metrics": metrics,
        "items": items_out,
    }
    if pending_holdout_commit is not None:
        _write_strict_json(pending_holdout_commit[0], pending_holdout_commit[1])
    _write_strict_json(Path(args.output), payload)
    print(json.dumps({"status": "SCORED", "output": args.output, "metrics": metrics}, indent=2))
    return 0 if (infra_errors == 0 and verifier_errors == 0) else 2


def _require_strict_finite_float(val: Any, label: str) -> float:
    if isinstance(val, bool) or not isinstance(val, (int, float)) or not math.isfinite(float(val)):
        raise ValueError(f"ABAB_NONFINITE_OR_INVALID_METRIC:{label}={val!r}")
    return float(val)


def score_systems_abab(args: argparse.Namespace) -> int:
    def _reject_nonfinite(tok: str) -> Any:
        raise ValueError(f"Non-finite JSON constant {tok!r} forbidden in {args.blocks_json}")

    blocks_data = json.loads(
        Path(args.blocks_json).resolve(strict=True).read_text(encoding="utf-8"),
        parse_constant=_reject_nonfinite,
    )
    arm = args.arm
    blocks = blocks_data["blocks"]
    if not isinstance(blocks, list) or len(blocks) < 2:
        raise RuntimeError("Systems ABAB scoring requires >= 2 paired blocks")

    n_blocks = len(blocks)
    items_out: list[dict[str, Any]] = []
    block_vals: list[float] = []
    seen_bids: set[str] = set()
    infra_errs = 0

    optional_series: dict[str, list[float]] = {
        "stop_rate": [],
        "lenient_accuracy": [],
        "after_nll": [],
        "peak_allocated_gib": [],
        "dataset_seconds": [],
        "board_joules": [],
        "tpot_p95_ms": [],
        "ttft_p95_ms": [],
    }
    gpu_hours_list: list[float] = []

    metric_key = args.block_metric
    for blk in sorted(blocks, key=lambda x: int(x["block_index"])):
        b_idx = blk.get("block_index")
        if isinstance(b_idx, bool) or not isinstance(b_idx, int):
            raise ValueError(f"ABAB_INVALID_BLOCK_INDEX:{b_idx!r}")
        bid = f"abab_block_{b_idx:02d}"
        if bid in seen_bids:
            raise ValueError(f"ABAB_DUPLICATE_BLOCK_ID:{bid}")
        seen_bids.add(bid)

        arms_map = blk.get("arms")
        if not isinstance(arms_map, dict) or arm not in arms_map or not isinstance(arms_map[arm], dict):
            raise ValueError(f"ABAB_MISSING_ARM_OBSERVATION:{bid}:{arm}")
        obs = arms_map[arm]
        if metric_key not in obs:
            raise ValueError(f"ABAB_MISSING_PRIMARY_BLOCK_METRIC:{bid}:{arm}:{metric_key}")
        val = _require_strict_finite_float(obs[metric_key], f"{bid}.{arm}.{metric_key}")
        items_out.append({"id": bid, "cluster_id": bid, "value": val})
        block_vals.append(val)

        if "infrastructure_errors" not in obs:
            raise ValueError(f"ABAB_MISSING_EXPLICIT_INFRASTRUCTURE_ERRORS:{bid}.{arm}")
        raw_ie = obs["infrastructure_errors"]
        if isinstance(raw_ie, bool) or not isinstance(raw_ie, int) or raw_ie < 0:
            raise ValueError(f"ABAB_INVALID_INFRASTRUCTURE_ERRORS:{bid}.{arm}={raw_ie!r}")
        infra_errs += raw_ie

        for opt_k, bucket in optional_series.items():
            if opt_k in obs:
                bucket.append(_require_strict_finite_float(obs[opt_k], f"{bid}.{arm}.{opt_k}"))

        has_gh = "assigned_gpu_hours" in obs
        has_lgh = "assigned_loop_gpu_hours" in obs
        if has_gh or has_lgh:
            gh_key = "assigned_gpu_hours" if has_gh else "assigned_loop_gpu_hours"
            gpu_hours_list.append(_require_strict_finite_float(obs[gh_key], f"{bid}.{arm}.{gh_key}"))

    for opt_k, bucket in optional_series.items():
        if 0 < len(bucket) < n_blocks:
            raise ValueError(f"ABAB_PARTIAL_BLOCK_METRIC_COVERAGE:{opt_k}:present={len(bucket)}/{n_blocks}")
    if 0 < len(gpu_hours_list) < n_blocks:
        raise ValueError(f"ABAB_PARTIAL_BLOCK_METRIC_COVERAGE:gpu_hours:present={len(gpu_hours_list)}/{n_blocks}")

    stop_rates = optional_series["stop_rate"]
    accuracies = optional_series["lenient_accuracy"]
    after_nlls = optional_series["after_nll"]
    peak_gibs = optional_series["peak_allocated_gib"]
    dataset_secs = optional_series["dataset_seconds"]
    energies = optional_series["board_joules"]
    tpot_p95_list = optional_series["tpot_p95_ms"]
    ttft_p95_list = optional_series["ttft_p95_ms"]

    metrics: dict[str, float] = {
        metric_key: statistics.fmean(block_vals) if args.statistic == "mean" else statistics.median(block_vals),
        "mean_" + metric_key: statistics.fmean(block_vals),
        "median_" + metric_key: statistics.median(block_vals),
        "infrastructure_error_count": float(infra_errs),
        "paired_blocks": float(len(blocks)),
    }
    if stop_rates:
        metrics["stop_rate"] = min(stop_rates)
        metrics["mean_stop_rate"] = statistics.fmean(stop_rates)
    if accuracies:
        metrics["lenient_accuracy"] = statistics.fmean(accuracies)
    if after_nlls:
        metrics["after_nll"] = max(after_nlls)
        metrics["mean_after_nll"] = statistics.fmean(after_nlls)
    if peak_gibs:
        metrics["peak_allocated_gib"] = max(peak_gibs)
    if dataset_secs:
        metrics["dataset_seconds"] = statistics.fmean(dataset_secs)
    if gpu_hours_list:
        metrics["mean_assigned_gpu_hours"] = statistics.fmean(gpu_hours_list)
        metrics["assigned_gpu_seconds"] = statistics.fmean(gpu_hours_list) * 3600.0
    if energies:
        metrics["mean_board_joules"] = statistics.fmean(energies)
    if tpot_p95_list:
        metrics["median_tpot_p95_ms"] = statistics.median(tpot_p95_list)
    if ttft_p95_list:
        metrics["median_ttft_p95_ms"] = statistics.median(ttft_p95_list)

    payload = {
        "candidate_id": args.candidate_id,
        "protocol_id": args.protocol_id,
        "env_fingerprint": blocks_data["env_fingerprint"],
        "scorer_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "metrics": metrics,
        "items": items_out,
    }
    _write_strict_json(Path(args.output), payload)
    print(json.dumps({"status": "SCORED_ABAB", "arm": arm, "metrics": metrics}, indent=2))
    return 0 if infra_errs == 0 else 2


def main() -> int:
    p = argparse.ArgumentParser(description="Out-of-process hermetic scorer")
    sub = p.add_subparsers(dest="mode", required=True)

    p_cap = sub.add_parser("capability")
    p_cap.add_argument("--split-prompts", required=True)
    p_cap.add_argument("--external-labels", default=None)
    p_cap.add_argument("--generations", required=True)
    p_cap.add_argument("--draws", type=int, default=1)
    p_cap.add_argument("--thinking", action="store_true", default=True)
    p_cap.add_argument("--no-thinking", action="store_false", dest="thinking")
    p_cap.add_argument("--primary-flavor", choices=["lenient", "strict"], default="lenient")
    p_cap.add_argument("--candidate-id", required=True)
    p_cap.add_argument("--protocol-id", required=True)
    p_cap.add_argument("--env-fingerprint", required=True)
    p_cap.add_argument("--extra-metrics-json", default=None)
    p_cap.add_argument("--base-seed", type=int, default=None)
    p_cap.add_argument("--holdout-registry", default=None)
    p_cap.add_argument("--output", required=True)
    p_cap.set_defaults(func=score_capability)

    p_sys = sub.add_parser("systems_abab")
    p_sys.add_argument("--blocks-json", required=True)
    p_sys.add_argument("--arm", required=True)
    p_sys.add_argument("--block-metric", default="dataset_seconds")
    p_sys.add_argument("--statistic", choices=["mean", "median"], default="mean")
    p_sys.add_argument("--candidate-id", required=True)
    p_sys.add_argument("--protocol-id", required=True)
    p_sys.add_argument("--output", required=True)
    p_sys.set_defaults(func=score_systems_abab)

    args = p.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
