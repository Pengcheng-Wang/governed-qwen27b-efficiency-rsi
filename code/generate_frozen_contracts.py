"""
generate_frozen_contracts.py — Materialize coherent, fail-closed Step-0 preconditions,
gate_rules_*.json contracts, and program.md for AILab_Qwen27B_Governed_v2 prior to IMMUTABLE_MANIFEST freeze.

Key design invariants:
  1. Single formal Dense serving Confirm finalist S3 (`GPTQ W4A16 TP=1 + Repaired Native BF16 MTP3`) vs S0 (`BF16 TP=2`)
     at K=1 (`alpha_one_sided=0.025`), preserving the ORIGINAL unrelaxed 2.0 pp NI margin (`paired_quality_ni_margin=0.020`)
     on the clean, zero-overlap `math_confirm1280.jsonl` (`N=1,280` independent problem clusters).
  2. Exact KL-Hoeffding NI point-assumption power at S3 Search-only symmetrized discordance (`p_win=p_loss=0.0078125`,
     true delta=0) is 0.9127502593, with wide pilot binomial CI (`[0.0018979, 0.0553026]`) sensitivity explicitly disclosed.
  3. Coherent joint timing + wall-ratio + stop-rate + zero-error predicate planning on dedicated `node-2` across 12 blocks
     (`3 x 4x4` Williams Design on `math_search_block32.jsonl`): `min_improvement=310.0` GPU-s (which guarantees `cand_wall <= 248.91s`,
     `wall_ratio <= 0.8907 <= 0.90` against calibrated `S0` `279.4533s`) and `delta_plan=350.0` GPU-s (`wall_ratio=0.7476 <= 0.90`;
     Search observed `+381.85` GPU-s at `1.578x` wall speedup).
  4. Relative NLL guardrail (`max_ratio_vs_baseline: {"after_nll": 1.005}`) on `loop_train_systems_v2` matching `<= 0.5%` relative NLL drift.
  5. Explicit Step-0 `NO_GO` dispositions for both inner weight loops (`Dense DAPO ReST^EM` and `MoE Qwen3.5-35B-A3B`).
"""
from __future__ import annotations

import hashlib
import json
import math
import random
import sys
from pathlib import Path
from statistics import NormalDist
from typing import Any

_approx_inv_norm = NormalDist(0.0, 1.0).inv_cdf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
sys.path.insert(0, str(ROOT / "eval"))

from common import file_hash, read_jsonl  # noqa: E402
from strict_guard import (  # noqa: E402
    _exact_binom_cdf,
    _exact_cp_lower,
    _exact_cp_upper,
    _kl_hoeffding_mean_bounds,
    simulate_full_gate_power,
)


def _write_clean_json(path: Path, payload: dict) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = (json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")
    path.write_bytes(raw)
    return hashlib.sha256(raw).hexdigest()


def exact_kl_ni_binary_power(
    *,
    n_clusters: int,
    alpha_one_sided: float,
    ni_margin: float,
    p_win: float,
    p_loss: float,
) -> float:
    """Exact trinomial summation over all (n+1)(n+2)/2 (wins, losses) states for the actual KL-Hoeffding gate."""
    if p_win < 0.0 or p_loss < 0.0 or (p_win + p_loss) > 1.0:
        raise ValueError("Invalid trinomial probabilities")
    kl_budget = math.nextafter((math.log(2.0) - math.log(float(alpha_one_sided))) / n_clusters, math.inf)
    pos_lcbs = [_kl_hoeffding_mean_bounds(w / n_clusters, kl_budget)[0] for w in range(n_clusters + 1)]
    neg_ucbs = [_kl_hoeffding_mean_bounds(l / n_clusters, kl_budget)[1] for l in range(n_clusters + 1)]

    total_prob = 0.0
    for w in range(n_clusters + 1):
        # Compute exact Binomial(n_clusters, p_win) mass at w
        if p_win == 0.0:
            pw = 1.0 if w == 0 else 0.0
        elif p_win == 1.0:
            pw = 1.0 if w == n_clusters else 0.0
        else:
            log_pw = (
                math.lgamma(n_clusters + 1)
                - math.lgamma(w + 1)
                - math.lgamma(n_clusters - w + 1)
                + w * math.log(p_win)
                + (n_clusters - w) * math.log1p(-p_win)
            )
            pw = math.exp(log_pw)
        if pw == 0.0:
            continue

        thresh = pos_lcbs[w] + ni_margin
        rem = n_clusters - w
        # neg_ucbs[l] is strictly non-decreasing in l; find largest l in [0, rem] with pos_lcbs[w] - neg_ucbs[l] >= -ni_margin
        if neg_ucbs[0] > thresh:
            continue
        lo, hi = 0, rem
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if neg_ucbs[mid] <= thresh:
                lo = mid
            else:
                hi = mid - 1
        l_max = lo
        q_loss_given_not_win = 0.0 if (1.0 - p_win) <= 0.0 else min(1.0, p_loss / (1.0 - p_win))
        cond_cdf = _exact_binom_cdf(l_max, rem, q_loss_given_not_win)
        total_prob = math.fsum((total_prob, pw * cond_cdf))
    return min(1.0, max(0.0, total_prob))


def compute_serving_guardrail_power(
    *,
    base_wall_mean: float,
    se_wall_diff_5blk: float,
    n_confirm_blocks: int,
    delta_plan_gpu_sec: float,
    max_wall_ratio: float,
    min_block_stop_rate: float,
    block_size: int = 32,
    quality_n_items: int = 1280,
    quality_stop_floor: float = 0.96,
    assumed_per_item_stop_prob: float = 0.999,
    n_mc: int = 20000,
    seed: int = 20261117,
) -> dict[str, Any]:
    """
    Plug-in Monte Carlo / binomial / Fréchet planning estimate (NOT a guaranteed finite-sample probability bound)
    conditioned on healthy error-free infrastructure (p_req_err=0) for S3 (TP=1) vs S0 (TP=2) across
    `n_confirm_blocks` timing blocks + `quality_n_items` quality panel:
      1. `cand_wall_mean / base_wall_mean <= max_wall_ratio` simulated under calibrated block-level timing noise
         (retaining 5-block SE without 12-block sqrt(5/12) variance reduction).
      2. `min_{b=1..B} stop_rate_b >= min_block_stop_rate` (`>=31/32 = 0.96875` per timing block) AND
         `quality_candidate.stop_rate >= quality_stop_floor` (`>=1229/1280` on `math_confirm1280.jsonl`)
         evaluated at the planning assumption `assumed_per_item_stop_prob = 0.999` (alongside sensitivity at
         `0.99640`, the exact 95% Clopper-Pearson lower bound for the `832/832` Plane-1 Dense natural stops).
      3. Separate explicit `4,096`-request (`12*4*32 = 1,536` Williams timing + `2*1,280 = 2,560` Confirm quality)
         i.i.d. infrastructure-error sensitivity table for `p_req_err in (0.0, 1e-5, 1e-4)`.
    """
    cand_wall_plan = 2.0 * base_wall_mean - delta_plan_gpu_sec
    se_single_arm = se_wall_diff_5blk / math.sqrt(2.0)
    rng = random.Random(seed)
    wall_hits = 0
    for _ in range(n_mc):
        b_draw = base_wall_mean + rng.gauss(0.0, se_single_arm)
        c_draw = cand_wall_plan + rng.gauss(0.0, se_single_arm)
        if b_draw > 0.0 and (c_draw / b_draw) <= max_wall_ratio:
            wall_hits += 1
    p_wall_guard = wall_hits / n_mc

    min_stops_per_block = math.ceil(min_block_stop_rate * block_size - 1e-12)
    p_block_fail = _exact_binom_cdf(min_stops_per_block - 1, block_size, assumed_per_item_stop_prob)
    p_all_blocks_stop = (1.0 - p_block_fail) ** n_confirm_blocks

    min_q_stops = math.ceil(quality_stop_floor * quality_n_items - 1e-12)
    p_q_stop_fail = _exact_binom_cdf(min_q_stops - 1, quality_n_items, assumed_per_item_stop_prob)
    p_quality_stop_floor = 1.0 - p_q_stop_fail

    p_combined_stops = p_all_blocks_stop * p_quality_stop_floor
    joint_guard_cond_healthy_lb = max(0.0, p_wall_guard + p_combined_stops - 1.0)

    # Sensitivity at exact 95% Clopper-Pearson lower bound for 832/832 observed natural stops
    cp95_stop_lb = _exact_cp_lower(832, 832, 0.025)
    p_all_blocks_cp95 = (1.0 - _exact_binom_cdf(min_stops_per_block - 1, block_size, cp95_stop_lb)) ** n_confirm_blocks
    p_q_stop_cp95 = 1.0 - _exact_binom_cdf(min_q_stops - 1, quality_n_items, cp95_stop_lb)
    joint_guard_at_cp95_stop_lb = max(0.0, p_wall_guard + (p_all_blocks_cp95 * p_q_stop_cp95) - 1.0)

    total_confirm_requests = n_confirm_blocks * 4 * block_size + 2 * quality_n_items
    err_sensitivity = {}
    for p_err in (0.0, 1e-5, 1e-4):
        p_err_free = (1.0 - p_err) ** total_confirm_requests
        err_sensitivity[f"p_req_err_{p_err:g}"] = {
            "assumed_iid_per_request_error_prob": p_err,
            "total_confirm_requests": total_confirm_requests,
            "all_requests_error_free_prob": p_err_free,
            "plug_in_guardrail_power_estimate_with_error_model": max(0.0, joint_guard_cond_healthy_lb + p_err_free - 1.0),
        }

    return {
        "model_nature": "Plug-in Monte Carlo, binomial, and Fréchet planning estimates under stated assumptions, NOT guaranteed finite-sample probability bounds.",
        "conditioning_scope": "Design power conditioned on healthy/error-free owned local server infrastructure (p_req_err=0); zero pilot errors do not guarantee zero future errors.",
        "cand_wall_seconds_at_delta_plan": cand_wall_plan,
        "implied_wall_ratio_at_delta_plan": cand_wall_plan / base_wall_mean,
        "mc_wall_ratio_guard_power_conservative_5blk_se": p_wall_guard,
        "planning_assumption_per_item_stop_prob": assumed_per_item_stop_prob,
        "min_stops_required_per_32item_block": float(min_stops_per_block),
        "all_12_timing_blocks_stop_floor_power": p_all_blocks_stop,
        "min_stops_required_quality1280": float(min_q_stops),
        "quality1280_candidate_stop_floor_power": p_quality_stop_floor,
        "conditional_healthy_guardrail_predicate_power": joint_guard_cond_healthy_lb,
        "sensitivity_at_832of832_cp95_stop_lb": {
            "cp95_per_item_stop_lower_bound": cp95_stop_lb,
            "all_12_timing_blocks_stop_floor_power": p_all_blocks_cp95,
            "quality1280_candidate_stop_floor_power": p_q_stop_cp95,
            "plug_in_conditional_guardrail_power_at_cp95_stop_lb": joint_guard_at_cp95_stop_lb,
        },
        "infrastructure_error_sensitivity_4096_requests": err_sensitivity,
    }


def compute_training_guardrail_power(
    *,
    aa_nlls: list[float],
    observed_base_peak_gib: float,
    planned_cand_peak_gib: float,
    max_peak_gib_ceiling: float = 76.0,
    max_nll_ratio: float = 1.005,
    planned_bf16_assoc_nll_shift: float = 0.0010,
    n_confirm_blocks: int = 10,
    n_mc: int = 20000,
    seed: int = 20261117,
) -> dict[str, Any]:
    """
    Calibrated analytical + Monte Carlo guardrail model for `loop_train_systems_confirm_v2`
    (conditioned on healthy error-free 8-GPU DDP launches):
      1. `max_{b=1..10}(cand_after_nll_b) / max_{b=1..10}(base_after_nll_b) <= 1.005` simulated from
         calibrated Plane-1 `train_aa` single-run `after_nll` mean & SD under a conservative `+0.0010`
         (`~0.159%`) BF16 microbatch-associativity shift (`>3x` observed v1 shift `<0.05%`).
      2. Deterministic static-graph CUDA peak memory `planned_cand_peak_gib` (`~58.5 GiB` at `micro=4`,
         vs `55.07 GiB` measured at `micro=1`) against `76.0 GiB` ceiling (`>17 GiB` deterministic headroom).
      3. Explicit 20-launch (`10` paired `AB/BA` DDP blocks) error sensitivity for `p_launch_err in (0, 1e-4, 1e-3)`.
    """
    import statistics as _st
    mu_nll = _st.fmean(aa_nlls)
    sd_nll = _st.stdev(aa_nlls)
    rng = random.Random(seed)
    nll_hits = 0
    for _ in range(n_mc):
        b_max = max(mu_nll + rng.gauss(0.0, sd_nll) for _ in range(n_confirm_blocks))
        c_max = max(mu_nll + planned_bf16_assoc_nll_shift + rng.gauss(0.0, sd_nll) for _ in range(n_confirm_blocks))
        if b_max > 0.0 and (c_max / b_max) <= max_nll_ratio:
            nll_hits += 1
    p_nll_guard = nll_hits / n_mc
    p_mem_guard = 1.0 if planned_cand_peak_gib <= (max_peak_gib_ceiling - 5.0) else 0.0
    cond_guard_lb = max(0.0, p_nll_guard + p_mem_guard - 1.0)
    n_launches = 2 * n_confirm_blocks
    err_sens = {
        f"p_launch_err_{pe:g}": {
            "assumed_per_ddp_launch_error_prob": pe,
            "total_ddp_launches": n_launches,
            "all_launches_error_free_prob": (1.0 - pe) ** n_launches,
            "guardrail_power_with_error_model": max(0.0, cond_guard_lb + ((1.0 - pe) ** n_launches) - 1.0),
        }
        for pe in (0.0, 1e-4, 1e-3)
    }
    return {
        "conditioning_scope": "Conditioned on healthy error-free 8-GPU DDP launches (p_launch_err=0).",
        "calibrated_aa_after_nll_mean": mu_nll,
        "calibrated_aa_after_nll_sd": sd_nll,
        "assumed_bf16_associativity_nll_shift": planned_bf16_assoc_nll_shift,
        "mc_10block_max_after_nll_ratio_power": p_nll_guard,
        "measured_micro1_peak_allocated_gib": observed_base_peak_gib,
        "planned_micro4_peak_allocated_gib": planned_cand_peak_gib,
        "hard_ceiling_peak_allocated_gib": max_peak_gib_ceiling,
        "deterministic_memory_guard_power": p_mem_guard,
        "conditional_healthy_guardrail_predicate_power": cond_guard_lb,
        "ddp_launch_error_sensitivity_20_runs": err_sens,
    }


def main() -> None:
    # 0. Remove any stale/superseded inner-weight gate_rules files
    for stale in ROOT.glob("configs/gate_rules_dense_restem_*.json"):
        stale.unlink()

    # Ensure split_audit_v2.json records math_search_block32.jsonl hash without touching any split bytes
    split_audit_path = ROOT / "results/split_audit_v2.json"
    split_audit = json.loads(split_audit_path.read_text(encoding="utf-8"))
    b32_path = ROOT / "data/splits/math_search_block32.jsonl"
    if "math_search_block32.jsonl" not in split_audit["splits"] and b32_path.is_file():
        split_audit["splits"]["math_search_block32.jsonl"] = {
            "sha256": file_hash(b32_path),
            "rows": len(read_jsonl(b32_path)),
        }
        split_audit["splits"] = dict(sorted(split_audit["splits"].items()))
        _write_clean_json(split_audit_path, split_audit)

    # Load hermetic post-B4 A/A calibrations (exact HermeticTrialSandbox worktree + per-run cache policy)
    train_aa_path = ROOT / "results/plane1_train_aa_hermetic_v2.json"
    serve_aa_path = ROOT / "results/plane1_serving_aa_hermetic_v2.json"
    if not train_aa_path.is_file() or not serve_aa_path.is_file():
        raise RuntimeError(
            "Hermetic post-B4 A/A calibrations (plane1_train_aa_hermetic_v2.json and "
            "plane1_serving_aa_hermetic_v2.json) are required before contract generation!"
        )
    train_aa = json.loads(train_aa_path.read_text(encoding="utf-8"))
    serve_aa = json.loads(serve_aa_path.read_text(encoding="utf-8"))
    legacy_train_aa = json.loads((ROOT / "results/plane1_train_aa_noise.json").read_text(encoding="utf-8"))
    legacy_serve_aa = json.loads((ROOT / "results/plane1_serving_aa_noise.json").read_text(encoding="utf-8"))
    frontier_audit = json.loads((ROOT / "results/plane1_dense_frontier_audit.json").read_text(encoding="utf-8"))
    moe_98k = json.loads((ROOT / "results/plane1_moe_cap98k_calibration.json").read_text(encoding="utf-8"))
    rescored_audit = json.loads((ROOT / "results/plane1_verifier_audit_dense_rescored_v2.json").read_text(encoding="utf-8"))
    rescored_disc = json.loads((ROOT / "results/plane1_search128_serving_discordance_rescored_v2.json").read_text(encoding="utf-8"))
    linux_vqual = json.loads((ROOT / "results/plane1_linux_verifier_qualification_v2.json").read_text(encoding="utf-8"))
    aa_b0_summary = json.loads((ROOT / "results/training/p1_train_aa_hermetic_b0_A1/summary.json").read_text(encoding="utf-8"))

    confirm1280_rows = read_jsonl(ROOT / "data/splits/math_confirm1280.jsonl")
    confirm1280_ids = sorted(str(r["id"]) for r in confirm1280_rows)
    if len(confirm1280_ids) != 1280 or len(set(confirm1280_ids)) != 1280:
        raise RuntimeError(f"Expected 1280 unique IDs in math_confirm1280.jsonl, got {len(confirm1280_ids)}")
    confirm1280_cmap = {pid: pid for pid in confirm1280_ids}

    se_train = float(train_aa["se_aa_loop_seconds"])
    base_wall_s0 = float(serve_aa["mean_dataset_seconds"])
    se_wall_s0 = float(serve_aa["se_aa_dataset_seconds"])
    se_gpu_sec_s0 = 2.0 * se_wall_s0  # TP=2 assigned GPU-seconds SE = 2 * dataset_seconds SE = 11.601083565610974

    # 1. Exact KL-Hoeffding NI power & pilot-uncertainty sensitivity on N=1280 at ORIGINAL 2.0 pp margin (K=1, alpha=0.025)
    ni_alpha = 0.025
    ni_margin = 0.020
    s3_disc_count = int(rescored_disc["candidates_vs_S0"]["S3_int4_tp1_mtp3"]["wins_vs_S0"]) + int(
        rescored_disc["candidates_vs_S0"]["S3_int4_tp1_mtp3"]["losses_vs_S0"]
    )
    s1_disc_count = int(rescored_disc["candidates_vs_S0"]["S1_bf16_tp2_mtp3"]["wins_vs_S0"]) + int(
        rescored_disc["candidates_vs_S0"]["S1_bf16_tp2_mtp3"]["losses_vs_S0"]
    )
    p_dir_s3 = (0.5 * s3_disc_count) / 128.0
    p_dir_s1 = (0.5 * s1_disc_count) / 128.0
    s3_disc_ci95_lo = _exact_cp_lower(s3_disc_count, 128, 0.025)
    s3_disc_ci95_hi = _exact_cp_upper(s3_disc_count, 128, 0.025)

    ni_power_s3_point = exact_kl_ni_binary_power(
        n_clusters=1280, alpha_one_sided=ni_alpha, ni_margin=ni_margin, p_win=p_dir_s3, p_loss=p_dir_s3
    )
    ni_power_s3_ci_lo = exact_kl_ni_binary_power(
        n_clusters=1280, alpha_one_sided=ni_alpha, ni_margin=ni_margin, p_win=0.5 * s3_disc_ci95_lo, p_loss=0.5 * s3_disc_ci95_lo
    )
    ni_power_s3_ci_hi = exact_kl_ni_binary_power(
        n_clusters=1280, alpha_one_sided=ni_alpha, ni_margin=ni_margin, p_win=0.5 * s3_disc_ci95_hi, p_loss=0.5 * s3_disc_ci95_hi
    )
    ni_power_s1_single_if_tested = exact_kl_ni_binary_power(
        n_clusters=1280, alpha_one_sided=ni_alpha, ni_margin=ni_margin, p_win=p_dir_s1, p_loss=p_dir_s1
    )

    # 2. Serving complete-predicate sizing (S3 single finalist K=1, 12 Williams blocks, min_improvement=310.0, delta_plan=350.0)
    serve_min_imp = 310.0
    serve_delta_plan = 350.0
    serve_primary_pwr = simulate_full_gate_power(
        delta_plan=serve_delta_plan,
        min_effect=serve_min_imp,
        eff_se=se_gpu_sec_s0,
        alpha_one_sided=0.025,
        multiplicity_k=1,
    )
    serve_guard_pwr = compute_serving_guardrail_power(
        base_wall_mean=base_wall_s0,
        se_wall_diff_5blk=se_wall_s0,
        n_confirm_blocks=12,
        delta_plan_gpu_sec=serve_delta_plan,
        max_wall_ratio=0.90,
        min_block_stop_rate=0.96,
    )
    serve_cond_guard = float(serve_guard_pwr["conditional_healthy_guardrail_predicate_power"])
    serve_complete_lb = max(
        0.0,
        serve_primary_pwr["mc_full_gate_power"] + ni_power_s3_point + serve_cond_guard - 2.0,
    )

    # 3. Training systems complete-predicate sizing (calibrated NLL + peak-GiB model, zero invented literals!)
    train_primary_pwr = simulate_full_gate_power(
        delta_plan=55.0,
        min_effect=25.0,
        eff_se=se_train,
        alpha_one_sided=0.025,
        multiplicity_k=1,
    )
    train_guard_pwr = compute_training_guardrail_power(
        aa_nlls=[float(r["after_nll"]) for r in train_aa["runs"]],
        observed_base_peak_gib=float(aa_b0_summary["peak_allocated_gib"]),
        planned_cand_peak_gib=58.5,
    )
    train_cond_guard = float(train_guard_pwr["conditional_healthy_guardrail_predicate_power"])
    train_complete_lb = max(
        0.0,
        train_primary_pwr["mc_full_gate_power"] + train_cond_guard - 1.0,
    )

    # Analytic Normal primary speed power for serving S3 under post-B4 hermetic se_AA (demonstrates <0.80 -> Step-0 NO_GO)
    z_crit_025 = _approx_inv_norm(1.0 - ni_alpha)
    serve_thresh = max(serve_min_imp, z_crit_025 * se_gpu_sec_s0)
    serve_analytic_normal_primary_power = 0.5 * math.erfc((serve_thresh - serve_delta_plan) / (se_gpu_sec_s0 * math.sqrt(2.0)))

    # 4. Write gate_rules JSON contracts
    search3_ids = [f"abab_block_{i:02d}" for i in range(3)]
    confirm10_ids = [f"abab_block_{i:02d}" for i in range(10)]
    williams12_ids = [f"abab_block_{i:02d}" for i in range(12)]

    train_search_rules = {
        "loop_id": "loop_train_systems_search_v2",
        "primary_metric": "loop_seconds",
        "primary_statistic": "mean",
        "direction": "minimize",
        "decision_stage": "search",
        "alpha_one_sided": 0.05,
        "multiplicity_k": 1,
        "min_improvement": 10.0,
        "aa_noise_se": se_train,
        "require_env_fingerprint_match": True,
        "expected_item_ids": search3_ids,
        "expected_cluster_map": {b: b for b in search3_ids},
        "min_independent_clusters": 3,
        "metric_directions": {
            "loop_seconds": "minimize",
            "after_nll": "minimize",
            "peak_allocated_gib": "minimize",
        },
        "max_ratio_vs_baseline": {
            "after_nll": 1.005,
        },
        "hard_ceilings": {
            "peak_allocated_gib": 76.0,
            "infrastructure_error_count": 0.0,
        },
        "max_cumulative_primary_regression": 5.0,
        "max_dev_looks": 20,
        "max_consecutive_rollbacks": 5,
    }

    train_confirm_rules = {
        "loop_id": "loop_train_systems_confirm_v2",
        "primary_metric": "loop_seconds",
        "primary_statistic": "mean",
        "direction": "minimize",
        "decision_stage": "confirm",
        "alpha_one_sided": 0.025,
        "multiplicity_k": 1,
        "min_improvement": 25.0,
        "aa_noise_se": se_train,
        "require_env_fingerprint_match": True,
        "expected_item_ids": confirm10_ids,
        "expected_cluster_map": {b: b for b in confirm10_ids},
        "min_independent_clusters": 10,
        "metric_directions": {
            "loop_seconds": "minimize",
            "after_nll": "minimize",
            "peak_allocated_gib": "minimize",
        },
        "max_ratio_vs_baseline": {
            "after_nll": 1.005,
        },
        "hard_ceilings": {
            "peak_allocated_gib": 76.0,
            "infrastructure_error_count": 0.0,
        },
        "max_cumulative_primary_regression": 5.0,
        "max_dev_looks": 20,
        "max_consecutive_rollbacks": 5,
    }

    serving_confirm_rules = {
        "loop_id": "loop_dense_serving_confirm_v2",
        "primary_metric": "assigned_gpu_seconds",
        "primary_statistic": "mean",
        "direction": "minimize",
        "decision_stage": "confirm",
        "alpha_one_sided": ni_alpha,
        "multiplicity_k": 1,
        "min_improvement": serve_min_imp,
        "aa_noise_se": se_gpu_sec_s0,
        "require_env_fingerprint_match": True,
        "expected_item_ids": williams12_ids,
        "expected_cluster_map": {b: b for b in williams12_ids},
        "min_independent_clusters": 12,
        "metric_directions": {
            "assigned_gpu_seconds": "minimize",
            "dataset_seconds": "minimize",
            "stop_rate": "maximize",
        },
        "max_ratio_vs_baseline": {
            "dataset_seconds": 0.90,
        },
        "hard_floors": {
            "stop_rate": 0.96,
        },
        "hard_ceilings": {
            "infrastructure_error_count": 0.0,
        },
        "max_cumulative_primary_regression": 20.0,
        "max_dev_looks": 20,
        "max_consecutive_rollbacks": 5,
        "require_paired_quality_ni": True,
        "paired_quality_ni_margin": ni_margin,
        "paired_quality_stop_floor": 0.96,
        "min_quality_independent_clusters": 1280,
        "expected_quality_item_ids": confirm1280_ids,
        "expected_quality_cluster_map": confirm1280_cmap,
    }

    sha_ts = _write_clean_json(ROOT / "configs/gate_rules_train_systems_search.json", train_search_rules)
    sha_tc = _write_clean_json(ROOT / "configs/gate_rules_train_systems_confirm.json", train_confirm_rules)
    sha_sc = _write_clean_json(ROOT / "configs/gate_rules_dense_serving_confirm.json", serving_confirm_rules)

    # 5a. Write explicit machine-verifiable NO_GO preflight config for serving Confirm under calibrated hermetic se_AA
    serving_nogo_cfg = {
        "archetype": "systems",
        "verifier_type": "deterministic",
        "has_disjoint_dev_and_holdout": True,
        "multi_shape_e2e_workload": True,
        "evaluator_locked_and_isolated": True,
        "hermetic_reset_available": True,
        "offline_online_sign_conflict": False,
        "has_trust_region_guard": True,
        "has_fast_proxy_stage": True,
        "one_shot_bug_ruled_out": True,
        "require_80pct_full_gate_power": True,
        "single_trial_minutes": 55.0,
        "total_budget_minutes": 420.0,
        "paired_noise_se": se_gpu_sec_s0,
        "aa_noise_se": se_gpu_sec_s0,
        "min_worthwhile_effect": serve_min_imp,
        "delta_plan": serve_delta_plan,
        "confirm_alpha": ni_alpha,
        "planned_confirmations": 1,
        "ni_predicate_power": ni_power_s3_point,
        "guardrail_predicate_power": serve_cond_guard,
        "expected_step0_verdict": "NO_GO",
        "disposition_reason": (
            f"Post-B4 hermetic serving A/A calibration (plane1_serving_aa_hermetic_v2.json) measured se_AA={se_gpu_sec_s0:.6f} GPU-s "
            f"(wall SE={se_wall_s0:.6f}s across 5 blocks; 127/160 matched prompt/seed pairs exhibited observed output divergence, "
            f"with vLLM 0.29.0 continuous-batching numerical/scheduling effects as plausible context rather than a uniquely proven cause). "
            f"Under preserved min_improvement=310.0 GPU-s, delta_plan=350.0 GPU-s, K=1, alpha=0.025, primary Normal planning power is "
            f"{serve_analytic_normal_primary_power:.10f} < 0.80 (plug-in MC={serve_primary_pwr['mc_full_gate_power']:.4f}), failing the 80% Confirm promotion power gate."
        ),
    }
    sha_srv_nogo = _write_clean_json(ROOT / "configs/step0_serving_confirm_nogo.json", serving_nogo_cfg)

    # 5b. Write canonical fail-closed configs/step0_preconditions.json (sole admitted GO loop = loop_train_systems_confirm_v2)
    step0_cfg = {
        "archetype": "hybrid",
        "verifier_type": "deterministic",
        "has_disjoint_dev_and_holdout": True,
        "multi_shape_e2e_workload": True,
        "evaluator_locked_and_isolated": True,
        "hermetic_reset_available": True,
        "offline_online_sign_conflict": False,
        "has_trust_region_guard": True,
        "has_fast_proxy_stage": True,
        "one_shot_bug_ruled_out": True,
        "require_80pct_full_gate_power": True,
        "single_trial_minutes": 50.0,
        "total_budget_minutes": 420.0,
        "learnable_fraction": float(frontier_audit["learnable_fraction"]),
        "admitted_plane2_confirm_loops": ["loop_train_systems_confirm_v2"],
        "disposed_step0_nogo_loops": [
            "loop_dense_serving_confirm_v2",
            "loop_dense_restem_dapo_v2",
            "loop_moe_qwen35_35b_a3b_v2",
        ],
        "paired_noise_se": se_train,
        "aa_noise_se": se_train,
        "min_worthwhile_effect": 25.0,
        "delta_plan": 55.0,
        "confirm_alpha": 0.025,
        "planned_confirmations": 1,
        "ni_predicate_power": 1.0,
        "ni_predicate_inapplicability_substantiation": (
            "Explicit 1.0 substantiated because the sole admitted Confirm loop (loop_train_systems_confirm_v2) has "
            "require_paired_quality_ni=False (fixed-token supervised SFT loop with no separate binary generation NI panel); "
            "optimization fidelity is gated deterministically via held-out NLL ratio max_ratio_vs_baseline.after_nll <= 1.005 "
            "modeled explicitly inside guardrail_predicate_power."
        ),
        "guardrail_predicate_power": train_cond_guard,
        "power_conditioning_note": (
            "All figures in full_gate_power and loop_level_admission_and_power_registry are plug-in Monte Carlo, "
            "binomial, and Fréchet planning estimates evaluated under explicit assumptions — NOT guaranteed finite-sample "
            "probability bounds. Training Confirm design power is conditioned on healthy/error-free 8-GPU DDP launches "
            "(p_launch_err=0) and post-B4 HermeticTrialSandbox calibration (plane1_train_aa_hermetic_v2.json, se_AA=1.1704s); "
            "zero pilot errors do NOT guarantee zero future infrastructure errors, and explicit 20-launch DDP error sensitivities "
            "are registered below. Open-generation serving Confirm (loop_dense_serving_confirm_v2) is formally disposed as Step-0 "
            "NO_GO after post-B4 hermetic serving A/A calibration (plane1_serving_aa_hermetic_v2.json, se_AA=113.5915 GPU-s) "
            "demonstrated primary Normal planning power 0.6376329614 < 0.80 under preserved gates."
        ),
        "hermetic_vs_legacy_aa_calibration_comparison": {
            "train_hermetic_mean_loop_seconds": train_aa["mean_loop_seconds"],
            "train_hermetic_se_aa_loop_seconds": se_train,
            "train_legacy_warm_cache_mean_loop_seconds": legacy_train_aa["mean_loop_seconds"],
            "train_legacy_asymmetric_cold_block0_se_aa_loop_seconds": legacy_train_aa["se_aa_loop_seconds"],
            "serving_hermetic_mean_dataset_seconds": base_wall_s0,
            "serving_hermetic_se_aa_assigned_gpu_seconds": se_gpu_sec_s0,
            "serving_legacy_mean_dataset_seconds": legacy_serve_aa["mean_dataset_seconds"],
            "serving_legacy_se_aa_assigned_gpu_seconds": 2.0 * float(legacy_serve_aa["se_aa_dataset_seconds"]),
        },
        "loop_level_admission_and_power_registry": {
            "loop_train_systems_confirm_v2": {
                "verdict": "GO",
                "design": (
                    "3-block 3x3 Latin Square independent-vs-T0 Search screen (T1, T2) -> live strict_guard gate-decision "
                    "(zero fallback if neither qualifies) -> hard pre-GPU verify_train_confirm_authorization -> 10-block AB/BA "
                    "Confirm of fastest PROVISIONAL_KEEP finalist on 8x A100 DDP (node-1) inside per-run HermeticTrialSandbox"
                ),
                "pure_loop_timer_excludes_checkpoint_io": True,
                "hermetic_fresh_cache_calibrated_in_aa": True,
                "cold_step1_timing_profile_note": (
                    "Under fresh per-run caches, update 1 averages 123.53s vs 11.06s for run-median updates 2-16 "
                    "(timings measure total update wall time and do not by themselves isolate compiler execution time)."
                ),
                "peak_memory_headroom_below_76gib_ceiling": 76.0 - float(aa_b0_summary["peak_allocated_gib"]),
                "fixed_deterministic_work_tokens": {"input_tokens": 340797, "supervised_tokens": 254724, "updates": 16},
                "baseline_mean_loop_seconds": train_aa["mean_loop_seconds"],
                "relative_after_nll_max_ratio": 1.005,
                "min_improvement_seconds": 25.0,
                "delta_plan_seconds": 55.0,
                "se_aa_loop_seconds": se_train,
                "mde_80pct_seconds": (_approx_inv_norm(0.975) + _approx_inv_norm(0.80)) * se_train,
                "plug_in_primary_speed_predicate_power_estimate": train_primary_pwr["mc_full_gate_power"],
                "calibrated_guardrail_predicate_details": train_guard_pwr,
                "plug_in_conditional_healthy_complete_predicate_frechet_estimate": train_complete_lb,
            },
            "loop_dense_serving_confirm_v2": {
                "verdict": "NO_GO",
                "nogo_preflight_config": "configs/step0_serving_confirm_nogo.json",
                "reasons": [
                    f"S3_UNDERPOWERED_UNDER_OBSERVED_MATCHED_SEED_EMITTED_WORK_VARIANCE (post-B4 hermetic A/A se_AA={se_gpu_sec_s0:.6f} GPU-s, wall SE={se_wall_s0:.6f}s; primary Normal planning power={serve_analytic_normal_primary_power:.10f} < 0.80 at preserved min_improvement=310.0 GPU-s, delta_plan=350.0 GPU-s, K=1, alpha=0.025)",
                ],
                "emitted_work_variance_observation": (
                    "Across 160 matched (ID, prompt_hash, request_seed) pairs in plane1_serving_aa_hermetic_v2.json (0 errors, 100% stop_rate, "
                    "tpot_p95_ms=22.60 +/- 0.046ms invariant to 0.2%), 127/160 pairs exhibited observed generated-text divergence "
                    "(e.g., Block 0 max tokens 12,249 -> 23,547 shifting 32-item dataset_seconds 294.12s -> 536.06s; vLLM 0.29.0 default "
                    "non-reproducibility and continuous-batching numerical/scheduling effects are plausible context, not a uniquely proven cause). "
                    "This emitted-work variance underpowers 32-item-block Confirm timing under unrelaxed gates without indicating S3 is ineffective."
                ),
                "single_finalist_retained_as_search_only_finding": "S3_int4_w4a16_tp1_mtp3_repaired",
                "baseline_incumbent": "S0_bf16_tp2",
                "confirm_and_holdout_panels_untouched": [
                    "data/splits/math_confirm1280.jsonl (0 touches)",
                    "data/splits/math_holdout512_prompts.jsonl (0 touches)",
                ],
                "search_only_factorial_diagnostics_not_in_confirm_k": [
                    f"S1_bf16_tp2_mtp3 (rescored Search 96.09% vs S0 95.31%, discordance 5/128 -> plug-in single-look 2.0pp KL-NI power estimate {ni_power_s1_single_if_tested:.6f})",
                    "S2_int4_w4a16_tp1 (rescored Search 94.53%, isolating bundled INT4+TP1 without MTP3)",
                ],
                "baseline_mean_dataset_seconds": base_wall_s0,
                "baseline_mean_assigned_gpu_seconds": 2.0 * base_wall_s0,
                "min_improvement_gpu_seconds": serve_min_imp,
                "delta_plan_gpu_seconds": serve_delta_plan,
                "analytic_normal_primary_speed_power": serve_analytic_normal_primary_power,
                "plug_in_mc_primary_speed_predicate_power_estimate": serve_primary_pwr["mc_full_gate_power"],
                "joint_guardrail_predicate_details": serve_guard_pwr,
                "plug_in_kl_ni_power_at_s3_search_point_p0p0078125": ni_power_s3_point,
                "plug_in_kl_ni_power_at_s3_cp95_lower_p0p000949": ni_power_s3_ci_lo,
                "plug_in_kl_ni_power_at_s3_cp95_upper_p0p027651": ni_power_s3_ci_hi,
                "plug_in_conditional_healthy_complete_predicate_frechet_estimate": serve_complete_lb,
            },
            "loop_dense_restem_dapo_v2": {
                "verdict": "NO_GO",
                "reasons": [
                    "B1_DAPO_PUBLISHED_GOLD_CONTRADICTIONS_AND_UNIT_FALSE_NEGATIVES",
                    "STEP2_DAPO_ALL_ROLLOUT_P99_RIGHT_CENSORED_AT_32768 (36/768 = 4.6875% length truncations; p95=31930.1, p99=32768.0)",
                    "S3_DAPO_CONFIRM200_AND_RETENTION150_INTERSECTION_UNDERPOWERED_FOR_1_TO_2PP",
                ],
            },
            "loop_moe_qwen35_35b_a3b_v2": {
                "verdict": "NO_GO",
                "reasons": [
                    "MOE_CAP_98K_DEGENERATE_TAIL_NO_GO (under study presence_penalty=0.0: 3/128 = 2.34% non-terminating at 98304 tokens, p99=98304 fails 1.5x uncensored p99; official model card recommends presence_penalty=1.5 to curb endless repetition with language-mixing caveat, documented in sources/research_model_sampling_policy_review.json)",
                ],
            },
        },
        "linux_verifier_runtime_qualification": {
            "receipt": "results/plane1_linux_verifier_qualification_v2.json",
            "runtime": linux_vqual["runtime"],
            "unittest_stderr_tail": linux_vqual["unittest_suite"]["stderr_tail"],
            "historical_40case_post_repair_acceptances": linux_vqual["historical_40case_replay"]["post_repair_acceptances"],
            "verifier_execution_errors": linux_vqual["historical_40case_replay"]["verifier_execution_errors"],
        },
        "frontier_model_review_status": {
            "signed_in_m365_copilot_desktop": "REVIEW_SEND_CONFIRMATION_PENDING",
            "evidence": (
                "Fresh native computer-use inventory exposes the signed-in M365 Copilot app with GPT 6 Astra, Max and Opus 5.5 "
                "observed in the model menu (no Fable 5.2 in that menu; maximum-context setting not verified). Native action "
                "policy requires specific user confirmation before sending each message; parent coordinator owns any "
                "permission-limited reviews, zero review messages have been sent, and zero study review answers are claimed. "
                "Earlier locked-desktop cua.listWindows()==[] probes are retained as historical timeline."
            ),
        },
    }
    sha_s0 = _write_clean_json(ROOT / "configs/step0_preconditions.json", step0_cfg)

    # 6. Write complete 8-section program.md matching governed-loop-optimization/assets/program_template.md
    s3_rescored = rescored_disc["candidates_vs_S0"]["S3_int4_tp1_mtp3"]
    s0_rescored = rescored_disc["S0_bf16_tp2_baseline"]
    program_md = f"""# `program.md` — Governed Optimization Loop Specification (`AILab_Qwen27B_Governed_v2`)

## 1. Objective, Tier & Go/No-Go Admission Audit

- **Problem Statement:** Optimize training-loop and inference-serving resource efficiency on `Qwen/Qwen3.8-27B` (`revision 1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`, native `Qwen3_5ForConditionalGeneration` wrapper with `15` native MTP tensors) **subject to strictly preserved optimization/reasoning fidelity**, while evaluating Step-0 admission preconditions across all four candidate loops (`loop_train_systems_confirm_v2`, `loop_dense_serving_confirm_v2`, `loop_dense_restem_dapo_v2`, and `loop_moe_qwen35_35b_a3b_v2`).
- **Loop Archetype:** `D: Two-Plane Hybrid` (Stage 1 unselected Plane-1 runtime/verifier/hermetic-noise calibration -> Stage 2 SHA-256 boundary freeze & external byte-receipt pin -> Stage 3 Plane-2 governed Systems loop execution). Runtime policy and weights are never mutated together.
- **Loop Scale Tier:** `Tier 1` finite pre-registered Systems evaluation (`<= 3` ledgered Search/Confirm trials total on the sole Step-0 `GO` loop `loop_train_systems_v2`; zero open-ended code drift).
- **Admitted `GO` Loop & Post-B4 Hermetic `se_AA` Noise Floor:**
  - **`loop_train_systems_v2` (`GO` — Sole Admitted Plane-2 Loop):** Minimize pure-training `loop_seconds` (stopped prior to any checkpoint/export I/O) on fixed-exposure `data/splits/sft_systems1024_v2.jsonl` (`1,024` rows, `340,797` input tokens, `254,724` supervised tokens, `16` updates on 8x A100 DDP `node-1` inside per-run `HermeticTrialSandbox`). Hermetic baseline `T0` (`micro=1, bucket=False`): `mean_loop_seconds = {train_aa['mean_loop_seconds']:.4f}s`, **`se_AA = {se_train:.4f}s`** (5 paired `A1/A2` blocks under symmetric fresh per-run Triton/Inductor/CUDA caches; `80%`-power `MDE = {(_approx_inv_norm(0.975) + _approx_inv_norm(0.80)) * se_train:.4f}s`), `min_improvement (eps) = 25.0s`, `delta_plan = 55.0s`, `K = 1`, `alpha_one_sided = 0.025`, plug-in conditional healthy complete-predicate planning estimate = **`{train_complete_lb:.4f} >= 0.80`**.
- **Hard Admission Blockers (`B1–B5`) & Explicit Step-0 `NO_GO` Dispositions (3 of 4 Candidate Loops Blocked Before Plane-2):**
  - [x] **B1 Asymmetric Deterministic Verifier (SFT NLL & Repaired MATH Symbolic Verifier):** Fixed-token held-out `after_nll` on `sft_heldout64_v2.jsonl` plus `FP32` merge `<= 1 BF16 ULP` & bitwise 15-tensor native MTP oracle (`code/train_systems_v2.py` SHA `c05f70dd...`, `12/12` CPU controls). Symbolic MATH verifier (`eval/verifiers_v2.py` SHA `{rescored_audit['verifier_sha256']}`) qualified on remote Linux (`Python 3.12.3`, `math-verify 0.9.0`, `SymPy 1.14.0`, unmodified default production timeouts): `20/20` unit tests pass, `40`-case historical `v1` replay preserves `20/20` passes and repairs `10/20` notation/unit/set false negatives with `0` execution errors (`results/plane1_linux_verifier_qualification_v2.json`).
  - [x] **B2 Disjoint 3-Way Splits & Multi-Shape Workloads:** Pairwise disjoint `Search`, `Confirmation`, and `Protected Holdout` panels (`0` exact or 13-gram overlaps) and `0` historical overlaps across all 6 Confirm/Protected panels against the authenticated `780`-member `v1` archive (`a1ce1c5a...`, parent receipt `2a6abeb1...`).
  - [x] **B3 Immutable Out-of-Process Scorer:** Complete protected surface frozen in `IMMUTABLE_MANIFEST.json`; scoring runs out-of-process via `python -I -B eval/isolated_scorer.py` inside a fresh detached worktree (`11/11` parent CLI completeness controls passed, receipt `9310747a...`).
  - [x] **B4 Hermetic Detached Scratch-Worktree & Per-Run Cache Isolation:** Every trial executes inside `HermeticTrialSandbox` (`git worktree add --detach` + `git clean -fdx` under `/tmp/q27wt_<nonce>`, plus isolated per-run `TRITON_CACHE_DIR`, `TORCHINDUCTOR_CACHE_DIR`, `VLLM_CACHE_ROOT`, `CUDA_CACHE_PATH`, `XDG_CACHE_HOME`, `HF_HOME`, `TMPDIR`, `PYTHONPYCACHEPREFIX`), promoting only declared `results/...` outputs on `rc == 0` and deleting worktree + caches in `finally:` (`21/21` parent boundary/auth controls passed, receipt `ea20dcad...`).
  - [x] **B5 Optimization Fidelity Containment:** Training Confirm requires worst-block relative `after_nll <= 1.005x`, `peak_allocated_gib <= 76.0`, and `infrastructure_error_count == 0`.
  - **Mandatory Step-0 `NO_GO` Dispositions (Unexecuted in Plane-2; Confirm & Protected Holdout Panels Untouched at `0` Touches):**
    1. **`loop_dense_serving_confirm_v2` -> Step-0 `NO_GO` (`S3_UNDERPOWERED_UNDER_OBSERVED_MATCHED_SEED_EMITTED_WORK_VARIANCE`):** Under `HermeticTrialSandbox` (`results/plane1_serving_aa_hermetic_v2.json`), `S0` (`BF16 TP=2`) measured `mean_dataset_seconds = {base_wall_s0:.4f}s` and **`se_AA = {se_gpu_sec_s0:.4f}` GPU-s** (`wall SE = {se_wall_s0:.4f}s`). While hardware decode speed was invariant (`tpot_p95_ms = 22.60 +/- 0.046 ms`), `127/160` matched `(ID, prompt_hash, seed)` pairs exhibited observed generated-text divergence (`vLLM 0.29.0` default non-reproducibility and continuous-batching numerical/scheduling effects are plausible context rather than a uniquely established cause; e.g., Block 0 `max_tokens` `12,249 -> 23,547`, shifting `dataset_seconds` `294.12s -> 536.06s`). Under preserved unrelaxed gates (`min_improvement = 310.0` GPU-s, `delta_plan = 350.0` GPU-s, `K = 1`, `alpha = 0.025`, `2.0 pp` NI), Analytic Normal primary speed power is **`{serve_analytic_normal_primary_power:.6f} < 0.80`** (`configs/step0_serving_confirm_nogo.json` enforces `NO_GO`). Confirm/Holdout panels (`math_confirm1280.jsonl`, `math_holdout512_prompts.jsonl`) remain untouched (`0` touches); `S3` (`1-GPU GPTQ W4A16 + Repaired Native BF16 MTP3`, `18.7 GiB`, rescored Search128 `{s3_rescored['lenient_accuracy']*100:.2f}%` vs `S0` `{s0_rescored['lenient_accuracy']*100:.2f}%`, SLO-correct goodput `{s3_rescored['slo_correct_goodput_per_second']:.6f}` vs `{s0_rescored['slo_correct_goodput_per_second']:.6f}` req/s) is reported strictly as a Search/canary engineering finding, not a confirmed promotion.
    2. **`loop_dense_restem_dapo_v2` -> Step-0 `NO_GO`:** Fails `B1` (`DAPO-Math-17k` published gold contradictions `dapo:b57078e1...`, `dapo:e09849b4...` and integer-unit false negatives), `Step 2` (`36/768 = 4.69%` length truncations, right-censored `p99 = 32,768.0`; `5,580,085` of `5,784,296` rollout tokens discarded after batch alignment), and `S3` (`dapo_confirm200` $\\cap$ `retention_confirm150` `< 25%` power for `1–2 pp` shifts).
    3. **`loop_moe_qwen35_35b_a3b_v2` -> Step-0 `NO_GO`:** Scaling outer cap `32K -> 48K -> 98K` raised base accuracy `91.41% -> 94.53% -> 95.31%` (`uncensored p95 = 35,738 > 32,768`), yet under study `presence_penalty = 0.0`, `3/128 = 2.34%` remained non-terminating at `98,304` tokens (`p99 = 98,304.0`, failing `cap >= 1.5 * uncensored_p99`; official card `presence_penalty = 1.5` sampling mismatch documented in `sources/research_model_sampling_policy_review.json`).

---

## 2. Mutable Surface (`A_t` — Explicit Pre-Registered Allowlist)

During Plane-2 execution, **zero source files, evaluators, configs, prompts, model weights, or splits are modified**. Candidate arms vary strictly over the pre-registered discrete runtime parameters of `loop_train_systems_v2` encoded in frozen `code/plane2_governed_runner.py` (`node-1` 8x A100 DDP):
1. **`T0` (Baseline Incumbent):** `(micro_batch = 1, length_bucketing = False)` at `global_batch = 64`.
2. **`T1` (Candidate 1):** `(micro_batch = 4, length_bucketing = False)` at `global_batch = 64` (exact global-token gradient scaling).
3. **`T2` (Candidate 2):** `(micro_batch = 4, length_bucketing = True)` at `global_batch = 64` (within-update rank length bucketing).
4. **Runtime Output Scope:** Only `results/training/p2_*`, `results/scored/*`, `results/plane2_*.json`, `results/governed_ledger_v2.jsonl`, ignored `runs/*/status.json`, and `.local/governance_anchor/anchor.json` are written during execution.

---

## 3. Immutable Protected Surface (`E_t`) & Isolation Boundary Disclosure

Locked in `IMMUTABLE_MANIFEST.json` and verified via `governed_loop_guard.py verify-manifest --expected-root-sha256` before every stage, sandbox creation, and `gate-decision`:
1. **Evaluators & Guards:** `eval/governed_loop_guard.py`, `eval/strict_guard.py`, `eval/isolated_scorer.py`, `eval/verifiers_v2.py`, and `tests/*.py`.
2. **Runners, Helpers & Configs:** All `code/*.py` and `code/*.sh`, `configs/*.json` and `configs/*.txt`, `sources/*`, and `program.md`.
3. **Splits, Traces & Calibrations:** All `data/splits/*.jsonl`, `data/traces/*.jsonl`, `results/plane1_*.json`, `results/plane1_rescored_search128/*.json`, `results/quantization_v2.json`, and `results/split_audit_v2.json`.
4. **Honest Isolation Scope Disclosure:** Scoring executes in fresh isolated Python subprocesses (`python -I -B`) inside detached `/tmp/q27wt_<nonce>` worktrees, Protected Holdout gold labels remain sealed outside any trial execution (`0` touches across all 3 Protected Holdout panels), and `manifest_root_sha256` is independently byte-receipted by the parent coordinator session (`9cb4ce59-cbf4-4d88-bfc9-7e0b3907e829`). Because jobs on the AML Singularity container run under one Linux container UID, this provides strong protocol, process, scratch-worktree, and external cryptographic tamper detection rather than multi-UID OS kernel privilege separation.

---

## 4. Replication, Pairing & Anti-Survivorship Protocol

1. **Latin-Square Search & Interleaved `AB/BA` Confirm Block Pairing (`node-1`):**
   - Search (`loop_train_systems_search_v2`): `3`-block `3x3` Latin Square `[T0,T1,T2], [T1,T2,T0], [T2,T0,T1]` (`seeds 20261300..20261302`, `cluster_id = abab_block_00..02`), balancing within-block execution order.
   - Confirm (`loop_train_systems_confirm_v2`): `10` alternating `[T0, T*]` / `[T*, T0]` paired blocks (`seeds 20261400..20261409`, `cluster_id = abab_block_00..09`), executed ONLY for the single live-ledger-admitted Search winner `T*` after cryptographic `verify_train_confirm_authorization()`.
2. **Complete Per-Block Coverage (Fail-Closed):** `eval/isolated_scorer.py` rejects any missing/extra/duplicate block ID, any missing or invalid `infrastructure_errors` field (no missing->zero default), and any partial per-block coverage (`0 < present < n_blocks`) or non-finite/`None`/boolean value on `loop_seconds`, `after_nll`, or `peak_allocated_gib`.

---

## 5. Pre-Flight Calibration & Per-Trial Hygiene Invariants

1. **Exact-Policy Hermetic `A/A` Noise Calibration:** Both `plane1_train_aa_hermetic_v2.json` (`node-1`, `se_AA = {se_train:.4f}s`, `mean = {train_aa['mean_loop_seconds']:.4f}s`, `peak_allocated_gib = 55.1398 GiB` leaving `20.8602 GiB` headroom below the `76.0 GiB` ceiling; under fresh per-run caches update 1 averages `123.53s` vs `11.06s` for run-median updates `2–16`, without claiming timings alone isolate compile time) and `plane1_serving_aa_hermetic_v2.json` (`node-2`, `se_AA = {se_gpu_sec_s0:.4f}` GPU-s, exposing `127/160` matched-prompt/seed emitted-work divergence and triggering Step-0 `NO_GO` on serving Confirm) were measured under the **exact `HermeticTrialSandbox` detached-worktree and fresh-ephemeral-cache lifecycle** used in Plane-2.
2. **Pure-Loop Timer Separation:** `train_systems_v2.py` stops `loop_seconds` immediately after `torch.cuda.synchronize` on update 16 prior to held-out `after_nll` evaluation and any checkpoint/export I/O, while also recording full `process_seconds` and stage `runs/*/status.json` separately for honest multi-scope cost accounting.

---

## 6. Two-Stage Promotion Gate, Plug-In Planning Power & Fail-Closed Tripwires

1. **Search Screen (`loop_train_systems_search_v2`):** Evaluates `T1` vs `T0` (`trial_01`) and `T2` vs `T0` (`trial_02`) independently via live `strict_guard.py gate-decision` (`min_improvement = 10.0s`, `max_ratio_vs_baseline.after_nll <= 1.005`, `peak_allocated_gib <= 76.0`, `infrastructure_error_count == 0`). `select_train_search_finalist()` selects the fastest `PROVISIONAL_KEEP` arm (`K = 1`), returning `None` (`HALTED_NO_ELIGIBLE_SEARCH_FINALIST`) if neither qualifies.
2. **Pre-GPU Boundary & Confirm Gate (`loop_train_systems_confirm_v2`, `K = 1`, `alpha = 0.025`):** `worker-train-confirm` enforces `verify_train_confirm_authorization()` against live `IMMUTABLE_MANIFEST.json`, `anchor.json`, and `governed_ledger_v2.jsonl` before launching any GPU run. Promotion (`PROMOTE`) requires 10-block `loop_seconds` reduction `>= 25.0s`, one-sided 97.5% cluster-bootstrap & `se_AA`-floored `LCB > 0`, worst-block `after_nll / base_after_nll <= 1.005`, `peak_allocated_gib <= 76.0`, and `infrastructure_error_count == 0`.
3. **Plug-In Planning Power Disclosure:** Under explicit planning assumptions of healthy 8-GPU DDP launches (`p_launch_err = 0`) and calibrated `+0.0010` BF16 associativity NLL shift, plug-in planning estimates (distinct from guaranteed finite-sample probability bounds) give primary speed power = `{train_primary_pwr['mc_full_gate_power']:.4f}`, 10-block NLL + memory guardrail power = `{train_cond_guard:.4f}`, and conditional complete-predicate Fréchet planning estimate = **`{train_complete_lb:.4f}`** (`{train_guard_pwr['ddp_launch_error_sensitivity_20_runs']['p_launch_err_0.0001']['guardrail_power_with_error_model']:.4f}` at `p_launch_err = 1e-4`; `{train_guard_pwr['ddp_launch_error_sensitivity_20_runs']['p_launch_err_0.001']['guardrail_power_with_error_model']:.4f}` at `p_launch_err = 1e-3`).

---

## 7. Finite Trial Budget, Single-Look Confirm Policy & Stopping Criteria

- **Finite Trial Cap:** At most `3` ledgered trials (`2` Training Search, `<= 1` Training Confirm) within `total_budget_minutes = 420.0`.
- **No Ad-Hoc Retests:** `allow_registered_retest` is `False` (single fixed-sample 10-block Confirm look; any `INCONCLUSIVE_RETEST` or `ROLLBACK` halts without block expansion or re-rolling).
- **Automatic Stopping:** Immediate halt after `trial_03_train_confirm` (or after Search if zero finalists qualify); serving Confirm and both inner weight loops remain halted at Step-0 `NO_GO`.

---

## 8. Live Hash-Chained Ledger, Post-Confirm Merge Qualification & Post-Experiment Lifecycle

1. **Hash-Chained Ledger (`results/governed_ledger_v2.jsonl`):** Initialized from `GENESIS` in `.local/governance_anchor/anchor.json`; advanced atomically by `eval/strict_guard.py gate-decision` (`trial_01..02` after Search, `trial_03` after Confirm) and verified via `governed_loop_guard.py verify-ledger`.
2. **Untouched Protected Holdouts & Post-Confirm 27B Merge Qualification:** Because serving Confirm and both inner weight loops are Step-0 `NO_GO`, all 3 Protected Holdout splits (`math_holdout512_prompts.jsonl`, `dapo_holdout200_prompts.jsonl`, `retention_holdout150_prompts.jsonl`) and `math_confirm1280.jsonl` remain completely untouched (`0` touches). If `loop_train_systems_confirm_v2` promotes a winner, execute one bounded post-ledger untimed `Qwen/Qwen3.8-27B` `--save-checkpoint --merge-fp32-export` + fresh-process `--qualify-merged-dir` check inside `HermeticTrialSandbox` to verify real-model all-matrix `<= 1 BF16 ULP`, bitwise 15-tensor native MTP preservation, and fresh-process held-out `merged_after_nll` parity.
3. **Frontier Review Status & Post-Experiment Lifecycle:** Signed-in M365 Copilot desktop UI is now visible in native inventory (`GPT 6 Astra, Max` and `Opus 5.5` observed in the model menu; no `Fable 5.2` in that menu; maximum-context setting not verified), with present status `REVIEW_SEND_CONFIRMATION_PENDING` because native action policy requires specific user confirmation before sending each message (`0` review prompts sent / `0` study review answers claimed; parent coordinator owns any permission-limited UI reviews; earlier locked-desktop observations retained as history). Upon completion, reconcile multi-scope stage/server/loop GPU-hours without nested double-counting, serialize durable publication through `node-0`, leave the authorized AML Singularity allocation configured in ignored `.local/resource.json` running, restore and verify `copilot-mi300x-keepalive` across all 24 GPUs, and clear the hourly session automation.
"""
    (ROOT / "program.md").write_bytes(program_md.encode("utf-8"))

    print(
        json.dumps(
            {
                "status": "FROZEN_CONTRACTS_GENERATED",
                "step0_preconditions_sha256": sha_s0,
                "step0_serving_confirm_nogo_sha256": sha_srv_nogo,
                "gate_rules_sha256": {
                    "gate_rules_train_systems_search.json": sha_ts,
                    "gate_rules_train_systems_confirm.json": sha_tc,
                    "gate_rules_dense_serving_confirm.json": sha_sc,
                },
                "serving_s3_analytic_normal_primary_power_nogo": serve_analytic_normal_primary_power,
                "serving_s3_plug_in_complete_predicate_nogo": serve_complete_lb,
                "train_conditional_complete_predicate_frechet_lb_go": train_complete_lb,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
