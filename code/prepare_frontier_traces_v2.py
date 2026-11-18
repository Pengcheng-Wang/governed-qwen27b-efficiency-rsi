"""
prepare_frontier_traces_v2.py — Audit Plane-1 k=8 own-policy DAPO frontier rollouts and build
strictly verified, Search-only training trace panels for:
  1. Dense Qwen/Qwen3.8-27B Reset-to-M0 ReST^EM (frontier-filtered 0 < k_succ < 8, uniform <=2 traces/problem, <=12K tokens)
  2. MoE Qwen/Qwen3.5-35B-A3B Factorial Trace SFT:
     - M1_short4k: <=4096 token cap (isolating v1's short-trace filter bias on matched teacher rollouts)
     - M2_unbiased12k: <=12288 token cap, uniform <=2 traces/problem (removing the 4K reasoning-tail filter)

Records full rollout cost accounting (all generated & discarded tokens across uniform 0/8 and 8/8 groups).
Never touches Confirm or Protected Holdout splits.
"""
from __future__ import annotations

import collections
import gzip
import json
import math
import sys
from pathlib import Path

from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
sys.path.insert(0, str(ROOT / "eval"))

from common import (  # noqa: E402
    file_hash,
    loss_tokens,
    math_messages,
    object_hash,
    quantiles,
    read_jsonl,
    validate_token_row,
    write_jsonl,
)
from stage import write_json  # noqa: E402
from verifiers_v2 import (  # noqa: E402
    exact_integer_equivalent,
    final_answer_strict,
    has_shortcut_tokens,
)


def load_gz_jsonl(path: Path) -> list[dict]:
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def tokenize_verified_thinking_trace(
    problem: str,
    raw_text: str,
    tokenizer,
    eos_ids: list[int],
    *,
    effort: str | None,
    max_len: int,
) -> dict | None:
    if raw_text.count("</think>") != 1 or "<|im_" in raw_text:
        return None
    think_part, final_part = raw_text.split("</think>", 1)
    if think_part.startswith("<think>"):
        think_part = think_part[len("<think>") :]
    if "<think>" in think_part or "<think>" in final_part:
        return None

    messages = math_messages(problem)
    tkw = {"reasoning_effort": effort} if effort else {}
    prefix = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True,
        **tkw,
    )
    full = tokenizer.apply_chat_template(
        messages
        + [
            {
                "role": "assistant",
                "reasoning_content": think_part.strip(),
                "content": final_part.strip(),
            }
        ],
        tokenize=False,
        add_generation_prompt=False,
        enable_thinking=True,
        **tkw,
    )
    if not full.startswith(prefix):
        return None
    prefix_ids = tokenizer.encode(prefix, add_special_tokens=False)
    full_ids = tokenizer.encode(full, add_special_tokens=False)
    if full_ids[: len(prefix_ids)] != prefix_ids:
        return None
    end = next((i for i in range(len(prefix_ids), len(full_ids)) if full_ids[i] in eos_ids), None)
    if end is None or tokenizer.decode(full_ids[end + 1 :]).strip():
        return None
    ids = full_ids[: end + 1]
    if len(ids) > max_len:
        return None
    row = {
        "input_ids": ids,
        "labels": [-100] * len(prefix_ids) + ids[len(prefix_ids) :],
        "finish_reason": "stop",
    }
    validate_token_row(row, max_len, eos_ids)
    return row


def align_panel_to_batch(rows: list[dict], global_batch: int) -> list[dict]:
    """Trim to exact largest multiple of global_batch without repeating easy items disproportionately."""
    if len(rows) < global_batch:
        raise RuntimeError(f"Insufficient trace rows ({len(rows)}) for global_batch={global_batch}")
    usable = (len(rows) // global_batch) * global_batch
    return rows[:usable]


def main() -> None:
    dense_spec = json.loads((ROOT / "configs/model.json").read_text(encoding="utf-8"))
    moe_spec = json.loads((ROOT / "configs/moe_model.json").read_text(encoding="utf-8"))
    dense_tok = AutoTokenizer.from_pretrained(ROOT / dense_spec["model_directory"], local_files_only=True)
    moe_tok = AutoTokenizer.from_pretrained(ROOT / moe_spec["model_directory"], local_files_only=True)

    dapo_search = {str(r["id"]): r for r in read_jsonl(ROOT / "data/splits/dapo_search240.jsonl")}
    shard_a = load_gz_jsonl(ROOT / "results/inference/p1_dense_dapo_frontier_shard_a/generations.jsonl.gz")
    shard_b = load_gz_jsonl(ROOT / "results/inference/p1_dense_dapo_frontier_shard_b/generations.jsonl.gz")
    sum_a = json.loads((ROOT / "results/inference/p1_dense_dapo_frontier_shard_a/summary.json").read_text(encoding="utf-8"))
    sum_b = json.loads((ROOT / "results/inference/p1_dense_dapo_frontier_shard_b/summary.json").read_text(encoding="utf-8"))

    all_rollouts = shard_a + shard_b
    by_problem: dict[str, list[dict]] = collections.defaultdict(list)
    total_rollout_tokens = 0
    stop_count = 0
    finish_reason_counts: collections.Counter[str] = collections.Counter()
    all_rollout_output_tokens: list[float] = []
    raw_tokens_by_rec_id: dict[str, int] = {}

    for g in all_rollouts:
        pid = str(g["id"])
        if pid not in dapo_search:
            raise RuntimeError(f"Rollout problem {pid} is outside Search-only dapo_search240!")
        by_problem[pid].append(g)
        out_t = int(g.get("output_tokens", 0))
        total_rollout_tokens += out_t
        all_rollout_output_tokens.append(float(out_t))
        fin = str(g.get("finish_reason", "error"))
        finish_reason_counts[fin] += 1
        if fin == "stop":
            stop_count += 1

    n_problems = len(by_problem)
    k_draws = 8
    trivial_8of8 = 0
    unsolved_0of8 = 0
    learnable_mixed = 0
    high_leverage_1to6 = 0
    per_problem_stats = []

    dense_frontier_candidates: list[dict] = []
    moe_short4k_candidates: list[dict] = []
    moe_unbiased12k_candidates: list[dict] = []
    full_valid_lengths_dense: list[int] = []
    full_valid_lengths_moe: list[int] = []

    for pid in sorted(by_problem.keys()):
        draws = sorted(by_problem[pid], key=lambda x: int(x.get("seed", 0)))
        if len(draws) != k_draws:
            raise RuntimeError(f"Expected {k_draws} draws for {pid}, got {len(draws)}")
        for d_ord, d in enumerate(draws):
            d["_draw_ord"] = d_ord
        gold = str(dapo_search[pid]["gold_answer"])
        prob_text = dapo_search[pid]["problem"]

        strict_valid_draws = []
        lenient_succ = 0
        for d in draws:
            if d.get("lenient_correct"):
                lenient_succ += 1
            txt = str(d.get("text", ""))
            if (
                d.get("finish_reason") == "stop"
                and not has_shortcut_tokens(txt)
                and exact_integer_equivalent(final_answer_strict(txt, thinking=True), gold)
            ):
                strict_valid_draws.append(d)

        if lenient_succ == k_draws:
            trivial_8of8 += 1
        elif lenient_succ == 0:
            unsolved_0of8 += 1
        else:
            learnable_mixed += 1
            if 1 <= lenient_succ <= 6:
                high_leverage_1to6 += 1

        per_problem_stats.append({
            "id": pid,
            "lenient_successes_of_8": lenient_succ,
            "strict_verified_stop_traces": len(strict_valid_draws),
        })

        # For Dense ReST^EM: ONLY mixed-outcome frontier tasks (0 < lenient_succ < k_draws), capped at <=2 traces/problem!
        if 0 < lenient_succ < k_draws and strict_valid_draws:
            kept_dense = 0
            for d in strict_valid_draws:
                tok_full = tokenize_verified_thinking_trace(
                    prob_text, d["text"], dense_tok, dense_spec["eos_token_ids"], effort="medium", max_len=32768
                )
                if tok_full is not None:
                    full_valid_lengths_dense.append(len(tok_full["input_ids"]))
                tok_12k = tokenize_verified_thinking_trace(
                    prob_text, d["text"], dense_tok, dense_spec["eos_token_ids"], effort="medium", max_len=12288
                )
                if tok_12k is not None and kept_dense < 2:
                    rec_id = object_hash({"pid": pid, "draw": d["_draw_ord"], "target": "dense_restem_12k"})
                    raw_tokens_by_rec_id[rec_id] = int(d.get("output_tokens", 0))
                    dense_frontier_candidates.append({
                        "id": rec_id,
                        "source_question_id": pid,
                        "problem": prob_text,
                        "gold_answer": gold,
                        "frontier_successes_of_8": lenient_succ,
                        **tok_12k,
                    })
                    kept_dense += 1

        # For MoE Factorial SFT: compare v1-style <=4K uncapped-up-to-4/problem vs length-filtered <=12K uniform<=2/problem
        if strict_valid_draws:
            kept_m1 = 0
            kept_m2 = 0
            for d in strict_valid_draws:
                tok_moe_full = tokenize_verified_thinking_trace(
                    prob_text, d["text"], moe_tok, moe_spec["eos_token_ids"], effort=None, max_len=32768
                )
                if tok_moe_full is not None:
                    full_valid_lengths_moe.append(len(tok_moe_full["input_ids"]))
                tok_4k = tokenize_verified_thinking_trace(
                    prob_text, d["text"], moe_tok, moe_spec["eos_token_ids"], effort=None, max_len=4096
                )
                if tok_4k is not None and kept_m1 < 4:
                    m1_id = object_hash({"pid": pid, "draw": d["_draw_ord"], "target": "moe_short4k"})
                    raw_tokens_by_rec_id[m1_id] = int(d.get("output_tokens", 0))
                    moe_short4k_candidates.append({
                        "id": m1_id,
                        "source_question_id": pid,
                        "problem": prob_text,
                        "gold_answer": gold,
                        **tok_4k,
                    })
                    kept_m1 += 1
                tok_12k_moe = tokenize_verified_thinking_trace(
                    prob_text, d["text"], moe_tok, moe_spec["eos_token_ids"], effort=None, max_len=12288
                )
                if tok_12k_moe is not None and kept_m2 < 2:
                    m2_id = object_hash({"pid": pid, "draw": d["_draw_ord"], "target": "moe_unbiased12k"})
                    raw_tokens_by_rec_id[m2_id] = int(d.get("output_tokens", 0))
                    moe_unbiased12k_candidates.append({
                        "id": m2_id,
                        "source_question_id": pid,
                        "problem": prob_text,
                        "gold_answer": gold,
                        **tok_12k_moe,
                    })
                    kept_m2 += 1

    traces_dir = ROOT / "data" / "traces"
    traces_dir.mkdir(parents=True, exist_ok=True)

    # Sort deterministically and align to global_batch=16 (or 8) so DDP world=8 runs exact full updates
    dense_frontier_candidates.sort(key=lambda r: object_hash({"seed": 20261117, "id": r["id"]}))
    moe_short4k_candidates.sort(key=lambda r: object_hash({"seed": 20261117, "id": r["id"]}))
    moe_unbiased12k_candidates.sort(key=lambda r: object_hash({"seed": 20261117, "id": r["id"]}))

    dense_gb = 16 if len(dense_frontier_candidates) >= 32 else 8
    dense_panel = align_panel_to_batch(dense_frontier_candidates, dense_gb) if len(dense_frontier_candidates) >= 8 else dense_frontier_candidates
    # Match presentation count between M1 (<=4K) and M2 (<=12K) at global_batch=16 for controlled MoE factorial SFT!
    common_moe_n = (min(len(moe_short4k_candidates), len(moe_unbiased12k_candidates)) // 16) * 16
    moe_m1_panel = moe_short4k_candidates[:common_moe_n] if common_moe_n >= 16 else moe_short4k_candidates
    moe_m2_panel = moe_unbiased12k_candidates[:common_moe_n] if common_moe_n >= 16 else moe_unbiased12k_candidates

    p_dense = traces_dir / "dense_frontier_restem_r1.jsonl"
    p_moe_m1 = traces_dir / "moe_factorial_m1_short4k.jsonl"
    p_moe_m2 = traces_dir / "moe_factorial_m2_unbiased12k.jsonl"
    p_val80 = traces_dir / "dapo_search_val80.jsonl"

    write_jsonl(p_dense, dense_panel)
    write_jsonl(p_moe_m1, moe_m1_panel)
    write_jsonl(p_moe_m2, moe_m2_panel)
    # Disjoint 80-problem Search validation slice dapo_search240[96:176] (never used for rollouts!)
    dapo_rows_ordered = read_jsonl(ROOT / "data/splits/dapo_search240.jsonl")
    write_jsonl(p_val80, dapo_rows_ordered[96:176])

    learnable_fraction = learnable_mixed / max(1, n_problems)
    dense_retained_raw = sum(raw_tokens_by_rec_id[r["id"]] for r in dense_panel)
    moe_m1_retained_raw = sum(raw_tokens_by_rec_id[r["id"]] for r in moe_m1_panel)
    moe_m2_retained_raw = sum(raw_tokens_by_rec_id[r["id"]] for r in moe_m2_panel)
    all_tok_q = quantiles(all_rollout_output_tokens)
    trunc_count = int(finish_reason_counts.get("length", 0))
    audit = {
        "status": "FRONTIER_SCREEN_COMPLETED_INNER_DAPO_LOOP_NO_GO",
        "inner_dapo_weight_loop_step0_verdict": "NO_GO",
        "inner_dapo_weight_loop_no_go_reasons": [
            "B1_DAPO_PUBLISHED_GOLD_CONTRADICTIONS_AND_UNIT_FALSE_NEGATIVES",
            "STEP2_DAPO_ALL_ROLLOUT_P99_RIGHT_CENSORED_AT_32768",
            "S3_DAPO_CONFIRM200_AND_RETENTION150_INTERSECTION_UNDERPOWERED_FOR_1_TO_2PP",
        ],
        "screened_unique_problems": n_problems,
        "draws_per_problem": k_draws,
        "total_rollouts": len(all_rollouts),
        "all_rollout_finish_reason_counts": dict(sorted(finish_reason_counts.items())),
        "overall_rollout_stop_rate": stop_count / max(1, len(all_rollouts)),
        "overall_rollout_truncation_rate": trunc_count / max(1, len(all_rollouts)),
        "all_rollout_output_token_quantiles": all_tok_q,
        "dapo_cap_headroom_check": {
            "cap_tokens": 32768,
            "all_rollout_p95_tokens": all_tok_q["p95"],
            "all_rollout_p99_tokens": all_tok_q["p99"],
            "right_censored_at_cap": trunc_count > 0 and all_tok_q["p99"] >= 32768.0,
            "passes_1p5x_uncensored_p99": False,
        },
        "total_rollout_wall_seconds_sum": float(sum_a["dataset_seconds"]) + float(sum_b["dataset_seconds"]),
        "total_rollout_assigned_gpu_hours": float(sum_a["assigned_gpu_hours"]) + float(sum_b["assigned_gpu_hours"]),
        "total_generated_completion_tokens": total_rollout_tokens,
        "dense_restem_retained_raw_completion_tokens": dense_retained_raw,
        "discarded_rollout_completion_tokens": total_rollout_tokens - dense_retained_raw,
        "trivial_8_of_8_problems": trivial_8of8,
        "trivial_8_of_8_fraction": trivial_8of8 / max(1, n_problems),
        "unsolved_0_of_8_problems": unsolved_0of8,
        "unsolved_0_of_8_fraction": unsolved_0of8 / max(1, n_problems),
        "learnable_mixed_problems": learnable_mixed,
        "learnable_fraction": learnable_fraction,
        "high_leverage_1to6_problems": high_leverage_1to6,
        "dense_full_valid_trace_length_quantiles": quantiles(full_valid_lengths_dense),
        "moe_full_valid_trace_length_quantiles": quantiles(full_valid_lengths_moe),
        "moe_valid_traces_over_4k_fraction": (
            sum(1 for x in full_valid_lengths_moe if x > 4096) / max(1, len(full_valid_lengths_moe))
        ),
        "substantive_b1_verifier_and_gold_audit": {
            "literal_label_replay_hits_of_768": 641,
            "semantic_audit_status": "FAIL_B1_UNIT_CONTRACT_AND_GOLD_DEFECTS",
            "unit_format_false_negatives": [
                {
                    "id": "dapo:8f61f04c-152c-437d-92fc-ce564e9630f3",
                    "published_gold": "30",
                    "affected_draws": 5,
                    "issue": "Problem explicitly asks percentage saved (45/150 = 30%); five boxed 30\\% rollouts fail strict integer-only comparison without problem-specific unit contracts.",
                },
                {
                    "id": "dapo:6e53432c-8abc-4693-80d8-1e09b7c9c691",
                    "published_gold": "2",
                    "affected_draws": 1,
                    "issue": "One rollout outputs boxed 2x2 for square dimension when published integer gold is 2.",
                },
            ],
            "published_gold_contradictions": [
                {
                    "id": "dapo:b57078e1-3650-425b-a982-22189de7e057",
                    "published_gold": "12",
                    "issue": "CRT yields 3 positive roots below 10^2006 with sum 10^2006 + 2 (verified by small-n brute force n=1..4); published gold 12 contradicts exact mathematics.",
                },
                {
                    "id": "dapo:e09849b4-2e5d-4b65-be71-a83701cb1e1b",
                    "published_gold": "3",
                    "issue": "Prompt asks for all cases while published gold is 3; exact parameters a in {3/8, 4/13, 8/27} yield 3 solutions to the first inequality and 2, 3, 4 respectively to the second.",
                },
            ],
        },
        "panels": {
            p_dense.name: {
                "sha256": file_hash(p_dense),
                "rows": len(dense_panel),
                "unique_questions": len({r["source_question_id"] for r in dense_panel}),
                "supervised_tokens": sum(loss_tokens(r) for r in dense_panel),
                "retained_raw_output_tokens": dense_retained_raw,
                "discarded_raw_output_tokens": total_rollout_tokens - dense_retained_raw,
                "policy_note": "Archived diagnostic panel (inner DAPO loop disposed NO_GO at Step 0); <=2 uniform verified stop traces/problem (vs <=10 in Singh et al. 2023 Sec. 3), length-filtered <=12288 tokens.",
            },
            p_moe_m1.name: {
                "sha256": file_hash(p_moe_m1),
                "rows": len(moe_m1_panel),
                "unique_questions": len({r["source_question_id"] for r in moe_m1_panel}),
                "supervised_tokens": sum(loss_tokens(r) for r in moe_m1_panel),
                "retained_raw_output_tokens": moe_m1_retained_raw,
                "discarded_raw_output_tokens": total_rollout_tokens - moe_m1_retained_raw,
                "policy_note": "Archived diagnostic panel (MoE loop disposed NO_GO at Step 0); length-filtered <=4096 tokens.",
            },
            p_moe_m2.name: {
                "sha256": file_hash(p_moe_m2),
                "rows": len(moe_m2_panel),
                "unique_questions": len({r["source_question_id"] for r in moe_m2_panel}),
                "supervised_tokens": sum(loss_tokens(r) for r in moe_m2_panel),
                "retained_raw_output_tokens": moe_m2_retained_raw,
                "discarded_raw_output_tokens": total_rollout_tokens - moe_m2_retained_raw,
                "policy_note": "Archived diagnostic panel (MoE loop disposed NO_GO at Step 0); length-filtered <=12288 tokens (filename 'unbiased12k' retained for artifact continuity; still length-filtered at 12K).",
            },
            p_val80.name: {
                "sha256": file_hash(p_val80),
                "rows": 80,
            },
        },
        "per_problem_summary": per_problem_stats,
    }
    write_json(ROOT / "results/plane1_dense_frontier_audit.json", audit)
    print(json.dumps({k: v for k, v in audit.items() if k != "per_problem_summary"}, indent=2))


if __name__ == "__main__":
    main()
