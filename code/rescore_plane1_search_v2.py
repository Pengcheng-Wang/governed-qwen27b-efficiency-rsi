"""
rescore_plane1_search_v2.py — Linux default-timeout verifier qualification + deterministic out-of-process
rescoring of preserved Plane-1 Search arms under the parent-repaired semantic verifier (`verifiers_v2.py` 1961b86e).

Preserves all original raw `generations.jsonl.gz`, `summary.json`, `plane1_verifier_audit_dense.json`,
and `plane1_search128_serving_discordance.json` untouched, writing separate provenance-rich v2 rescore receipts.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
import tarfile
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
sys.path.insert(0, str(ROOT / "eval"))

from common import file_hash, read_jsonl  # noqa: E402
from stage import write_json  # noqa: E402
from verifiers_v2 import score_response  # noqa: E402

EXPECTED_V1_ARCHIVE_SHA256 = "a1ce1c5a3ea7019c72b1596b62b921d0b43b9907cbaceff2f1e5b8efd3525bfa"
EXPECTED_VERIFIER_SHA256 = "a3aa8c7127f830c0d57257075c29422f942e3453e0608ea72225f64a89880ce0"
EXPECTED_SCORER_SHA256 = "f2048ed0a3541f9e9fbbef36cd408629450ce30ae8eeab3abec2a79e847f868c"


def extract_historical40_fixture() -> None:
    audit = json.loads((ROOT / "sources" / "research_baseline_verifier_semantic_audit.json").read_text(encoding="utf-8"))
    pass_ids = list(audit["historical_passes_preserved"])
    fail_disps = {x["id"]: x["disposition"] for x in audit["historical_failure_dispositions"]}
    fail_ids = list(fail_disps.keys())

    tf_path = ROOT / ".local" / "historical_v1" / "evidence_20261101T060439Z_f538d3.tar.gz"
    actual_sha = file_hash(tf_path)
    if actual_sha != EXPECTED_V1_ARCHIVE_SHA256:
        raise RuntimeError(f"Historical v1 archive SHA mismatch: {actual_sha}")

    dataset_Specs = [
        ("math500", "data/sealed_math500.jsonl", "results/inference/math500_base_medium32_v1/generations.jsonl.gz"),
        ("dev200", "data/development_math200.jsonl", "results/inference/reasoning_base_dev200_v1/generations.jsonl.gz"),
        ("aime24", "data/sealed_aime24.jsonl", "results/inference/aime24_base_medium64_v1/generations.jsonl.gz"),
        ("aime25", "data/sealed_aime25.jsonl", "results/inference/aime25_base_medium64_v1/generations.jsonl.gz"),
    ]

    problems_by_id: dict[str, dict[str, str]] = {}
    selected_records: dict[str, dict[str, Any]] = {}
    total_authenticated_records = 0

    with tarfile.open(tf_path, "r:gz") as tf:
        for tag, ds_member, gen_member in dataset_Specs:
            ds_lines = tf.extractfile(ds_member).read().decode("utf-8").splitlines()
            for ln in ds_lines:
                if ln.strip():
                    r = json.loads(ln)
                    problems_by_id[str(r["id"])] = {
                        "problem": str(r["problem"]),
                        "gold_answer": str(r["gold_answer"]),
                        "source": str(r.get("source", tag)),
                    }
            gen_raw = gzip.decompress(tf.extractfile(gen_member).read()).decode("utf-8").splitlines()
            gen_rows = [json.loads(ln) for ln in gen_raw if ln.strip()]
            total_authenticated_records += len(gen_rows)
            for row in gen_rows:
                rid = str(row["id"])
                if row.get("finish_reason") != "stop" or row.get("error") is not None:
                    continue
                if rid in pass_ids and bool(row.get("correct")) is True and rid not in selected_records:
                    selected_records[rid] = {"cohort": "historical_pass", "run_member": gen_member, **row}
                elif rid in fail_disps and bool(row.get("correct")) is False and rid not in selected_records:
                    selected_records[rid] = {
                        "cohort": "historical_natural_stop_failure",
                        "expected_disposition": fail_disps[rid],
                        "run_member": gen_member,
                        **row,
                    }

    if total_authenticated_records != 940:
        raise RuntimeError(f"Expected 940 archived baseline records, got {total_authenticated_records}")
    if len(selected_records) != 40:
        missing = (set(pass_ids) | set(fail_ids)) - set(selected_records.keys())
        raise RuntimeError(f"Selected {len(selected_records)} != 40 historical records; missing={sorted(missing)}")

    cases_out = []
    for rid in pass_ids + fail_ids:
        rec = selected_records[rid]
        pmeta = problems_by_id[rid]
        cases_out.append({
            "id": rid,
            "cohort": rec["cohort"],
            "expected_disposition": rec.get("expected_disposition", "HISTORICAL_PASS_PRESERVED"),
            "run_member": rec["run_member"],
            "seed": int(rec["seed"]),
            "source": pmeta["source"],
            "problem": pmeta["problem"],
            "gold_answer": pmeta["gold_answer"],
            "finish_reason": str(rec["finish_reason"]),
            "archived_correct": bool(rec["correct"]),
            "archived_extracted_answer": rec.get("extracted_answer"),
            "text": str(rec["text"]),
        })

    out_path = ROOT / "sources" / "historical_v1_verifier_replay40_fixture.json"
    write_json(
        out_path,
        {
            "schema_version": 1,
            "source_archive_sha256": actual_sha,
            "archive_records_structurally_authenticated": total_authenticated_records,
            "case_count": len(cases_out),
            "cases": cases_out,
        },
    )
    print(json.dumps({"status": "FIXTURE_EXTRACTED", "path": str(out_path), "bytes": out_path.stat().st_size, "sha256": file_hash(out_path)}, indent=2))


def run_linux_qualification_and_rescore() -> None:
    verifier_sha = file_hash(ROOT / "eval" / "verifiers_v2.py")
    scorer_sha = file_hash(ROOT / "eval" / "isolated_scorer.py")
    if verifier_sha != EXPECTED_VERIFIER_SHA256:
        raise RuntimeError(f"verifiers_v2.py SHA mismatch: {verifier_sha}")
    if scorer_sha != EXPECTED_SCORER_SHA256:
        raise RuntimeError(f"isolated_scorer.py SHA mismatch: {scorer_sha}")

    import sympy  # noqa: F401
    runtime_versions = {
        "platform": platform.platform(),
        "system": platform.system(),
        "python": sys.version.split()[0],
        "math_verify": importlib.metadata.version("math-verify"),
        "sympy": importlib.metadata.version("sympy"),
        "antlr4_python3_runtime": importlib.metadata.version("antlr4-python3-runtime"),
        "production_default_timeouts_unmodified": True,
    }

    # 1. Run full 11-test repository unittest suite on Linux with DEFAULT production timeouts
    t0 = time.monotonic()
    ut_proc = subprocess.run(
        [sys.executable, "-B", "-m", "unittest", "discover", "-s", "tests", "-v"],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    ut_elapsed = time.monotonic() - t0
    if ut_proc.returncode != 0:
        raise RuntimeError(f"Linux default-timeout unittest suite failed (rc={ut_proc.returncode}):\n{ut_proc.stdout}\n{ut_proc.stderr}")

    # 2. Replay exact 40-case historical panel with DEFAULT production timeouts
    fixture_path = ROOT / "sources" / "historical_v1_verifier_replay40_fixture.json"
    fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
    replay_t0 = time.monotonic()
    case_results = []
    exec_errors = 0
    hist_passes_kept = 0
    post_repair_accepts = 0
    legacy_v1_accepts = 0
    newly_repaired_ids = []
    already_supported_ids = []

    for c in fixture["cases"]:
        sr = score_response(
            c["text"],
            c["gold_answer"],
            finish_reason=c["finish_reason"],
            thinking=True,
            integer_task=False,
            problem=c["problem"],
        )
        is_aime = c["source"] in {"aime24", "aime25"}
        sr_int = (
            score_response(
                c["text"],
                c["gold_answer"],
                finish_reason=c["finish_reason"],
                thinking=True,
                integer_task=True,
                problem=c["problem"],
            )
            if is_aime
            else None
        )
        if sr["verifier_error"] is not None or (sr_int is not None and sr_int["verifier_error"] is not None):
            exec_errors += 1
        if sr["v1_correct"]:
            legacy_v1_accepts += 1
        if sr["lenient_correct"]:
            post_repair_accepts += 1
            if c["cohort"] == "historical_pass":
                hist_passes_kept += 1
            elif sr["v1_correct"]:
                already_supported_ids.append(c["id"])
            else:
                newly_repaired_ids.append(c["id"])
        case_results.append({
            "id": c["id"],
            "cohort": c["cohort"],
            "expected_disposition": c["expected_disposition"],
            "archived_correct": c["archived_correct"],
            "v1_legacy_correct": sr["v1_correct"],
            "repaired_strict_correct": sr["strict_correct"],
            "repaired_lenient_correct": sr["lenient_correct"],
            "exact_integer_route_correct": sr_int["lenient_correct"] if sr_int is not None else None,
            "verifier_error": sr["verifier_error"],
        })
    replay_elapsed = time.monotonic() - replay_t0

    if (
        exec_errors != 0
        or hist_passes_kept != 20
        or post_repair_accepts != 30
        or legacy_v1_accepts not in (20, 21)
        or (legacy_v1_accepts + len(newly_repaired_ids)) != post_repair_accepts
    ):
        raise RuntimeError(
            f"Linux 40-case replay mismatch: exec_errors={exec_errors}, hist_passes_kept={hist_passes_kept}, "
            f"post_repair={post_repair_accepts}, legacy_v1={legacy_v1_accepts}, newly_repaired={newly_repaired_ids}"
        )

    linux_qual = {
        "status": "LINUX_DEFAULT_TIMEOUT_VERIFIER_QUALIFIED",
        "runtime": runtime_versions,
        "math_verify_0p9_percent_parity_finding": (
            "Remote cluster math-verify 0.9.0 parses r'\\boxed{10\\%}' only as string fallback \"'10\\\\%'\" (unlike 0.8.0 which "
            "converted X\\% to X/100); this both explains why historical v1 (also on 0.9.0) rejected math_train:1374 ('80\\%' vs "
            "'80 \\%') so legacy_v1_mode_acceptances=20 on Linux 0.9.0, and required normalizing non-percent-prompt numeric X\\% "
            "to (X)/100 in _semantic_math_pair so all 11/11 tests and 30/40 historical acceptances match identically across 0.8.0 and 0.9.0."
        ),
        "verifier_sha256": verifier_sha,
        "isolated_scorer_sha256": scorer_sha,
        "fixture_sha256": file_hash(fixture_path),
        "unittest_suite": {
            "returncode": ut_proc.returncode,
            "elapsed_seconds": ut_elapsed,
            "test_strict_guard_sha256": file_hash(ROOT / "tests" / "test_strict_guard.py"),
            "stderr_tail": ut_proc.stderr.strip().splitlines()[-5:],
        },
        "historical_40case_replay": {
            "elapsed_seconds": replay_elapsed,
            "cases_regraded": len(case_results),
            "historical_passes_preserved": hist_passes_kept,
            "previously_accepted_cases_lost": 20 - hist_passes_kept,
            "legacy_v1_mode_acceptances": legacy_v1_accepts,
            "post_repair_acceptances": post_repair_accepts,
            "newly_repaired_semantic_false_negatives": len(newly_repaired_ids),
            "newly_repaired_ids": newly_repaired_ids,
            "already_supported_ids": already_supported_ids,
            "verifier_execution_errors": exec_errors,
            "cases": case_results,
        },
    }
    write_json(ROOT / "results" / "plane1_linux_verifier_qualification_v2.json", linux_qual)

    # 3. Deterministic out-of-process rescore of all 4 preserved Dense Search arms (zero mutation of raw runs or v1 summaries)
    rescored_dir = ROOT / "results" / "plane1_rescored_search128"
    rescored_dir.mkdir(parents=True, exist_ok=True)
    split_path = ROOT / "data" / "splits" / "math_search128.jsonl"
    env_fp = "plane1_search128_deterministic_rescore_linux_mv090_sympy114"

    arms_map = [
        ("S0_bf16_tp2", "p1_dense_math_search128_audit"),
        ("S1_bf16_tp2_mtp3", "p1_search128_s1_bf16_mtp3"),
        ("S2_int4_tp1", "p1_search128_s2_int4"),
        ("S3_int4_tp1_mtp3", "p1_search128_s3_int4_mtp3"),
    ]
    prompt_by_id = {str(r["id"]): r for r in read_jsonl(split_path)}
    scored_arms: dict[str, dict[str, Any]] = {}
    orig_summaries: dict[str, dict[str, Any]] = {}
    raw_rows_by_arm: dict[str, dict[str, dict[str, Any]]] = {}
    recomputed_slo_by_arm: dict[str, dict[str, Any]] = {}

    for cand_id, run_dir_name in arms_map:
        rdir = ROOT / "results" / "inference" / run_dir_name
        gz_path = rdir / "generations.jsonl.gz"
        orig_summaries[cand_id] = json.loads((rdir / "summary.json").read_text(encoding="utf-8"))
        out_json = rescored_dir / f"{cand_id}_scored_v2.json"
        subprocess.run(
            [
                sys.executable,
                "-I",
                "-B",
                str(ROOT / "eval" / "isolated_scorer.py"),
                "capability",
                "--split-prompts",
                str(split_path),
                "--generations",
                str(gz_path),
                "--draws",
                "1",
                "--base-seed",
                "20261117",
                "--candidate-id",
                cand_id,
                "--protocol-id",
                "plane1_search128_rescored_v2",
                "--env-fingerprint",
                env_fp,
                "--output",
                str(out_json),
            ],
            cwd=str(ROOT),
            check=True,
            timeout=180,
        )
        scored_arms[cand_id] = json.loads(out_json.read_text(encoding="utf-8"))
        with gzip.open(gz_path, "rt", encoding="utf-8") as gf:
            arm_raw = {str(r["id"]): r for r in (json.loads(ln) for ln in gf if ln.strip())}
        raw_rows_by_arm[cand_id] = arm_raw

        # Recompute SLO-qualified correct goodput (infer_v2.py lines 319-334) from unchanged raw timings + repaired grades
        sc_items_map = {str(it["id"]): float(it["value"]) for it in scored_arms[cand_id]["items"]}
        duration_sec = float(orig_summaries[cand_id]["dataset_seconds"])
        slo_ok_count = 0
        slo_lenient_correct_count = 0
        slo_strict_correct_count = 0
        for pid, r_rec in arm_raw.items():
            p_row = prompt_by_id[pid]
            g_new = score_response(
                str(r_rec.get("text", "")),
                str(p_row["gold_answer"]),
                finish_reason=str(r_rec.get("finish_reason", "")),
                thinking=True,
                integer_task=False,
                problem=str(p_row.get("problem") or p_row.get("prompt") or ""),
            )
            if bool(g_new["lenient_correct"]) != (sc_items_map[pid] > 0.5):
                raise RuntimeError(f"Score mismatch vs isolated_scorer on {cand_id}:{pid}")
            is_stopped = (r_rec.get("finish_reason") == "stop" and r_rec.get("error") is None)
            if (
                is_stopped
                and not bool(g_new["natural_stop_lenient_parse_failure"])
                and r_rec.get("ttft_seconds") is not None
                and float(r_rec["ttft_seconds"]) <= 2.0
                and (r_rec.get("tpot_seconds") is None or float(r_rec["tpot_seconds"]) <= 0.1)
                and float(r_rec["including_client_queue_seconds"]) <= 600.0
            ):
                slo_ok_count += 1
                if bool(g_new["lenient_correct"]):
                    slo_lenient_correct_count += 1
                if bool(g_new["strict_correct"]):
                    slo_strict_correct_count += 1

        recomputed_slo_by_arm[cand_id] = {
            "pre_repair_slo_correct_goodput_per_second": float(orig_summaries[cand_id]["slo_correct_goodput_per_second"]),
            "slo_qualified_request_count": slo_ok_count,
            "slo_correct_request_count": slo_lenient_correct_count,
            "slo_correct_goodput_per_second": slo_lenient_correct_count / duration_sec,
            "strict_slo_correct_request_count": slo_strict_correct_count,
            "strict_slo_correct_goodput_per_second": slo_strict_correct_count / duration_sec,
        }

    # Inspect per-item transitions on S0 relative to pre-repair v1_correct
    s0_scored = scored_arms["S0_bf16_tp2"]
    s0_items = {r["id"]: float(r["value"]) for r in s0_scored["items"]}
    s0_raw_rows = raw_rows_by_arm["S0_bf16_tp2"]
    repaired_s0_ids = sorted(
        pid for pid, new_val in s0_items.items()
        if (new_val > 0.5) != bool(s0_raw_rows[pid].get("lenient_correct"))
    )

    orig_audit = json.loads((ROOT / "results" / "plane1_verifier_audit_dense.json").read_text(encoding="utf-8"))
    s0_slo = recomputed_slo_by_arm["S0_bf16_tp2"]
    rescored_audit = {
        "status": "VERIFIER_AUDIT_DENSE_RESCORED_V2",
        "preserves_original_artifact": "results/plane1_verifier_audit_dense.json",
        "original_artifact_sha256": file_hash(ROOT / "results" / "plane1_verifier_audit_dense.json"),
        "raw_generations_sha256": s0_scored["generations_sha256"],
        "verifier_sha256": verifier_sha,
        "isolated_scorer_sha256": scorer_sha,
        "runtime": runtime_versions,
        "total_audited": int(s0_scored["metrics"]["total_generations"]),
        "pre_repair_lenient_accuracy": orig_audit["lenient_accuracy"],
        "rescored_lenient_accuracy": s0_scored["metrics"]["lenient_accuracy"],
        "rescored_strict_accuracy": s0_scored["metrics"]["strict_accuracy"],
        "rescored_v1_parity_accuracy": s0_scored["metrics"]["v1_parity_accuracy"],
        "rescored_pass_count": sum(1 for v in s0_items.values() if v > 0.5),
        "rescored_fail_count": sum(1 for v in s0_items.values() if v < 0.5),
        "changed_item_ids_vs_pre_repair": repaired_s0_ids,
        "truncation_rate": s0_scored["metrics"]["truncation_rate"],
        "natural_stop_lenient_parse_failure_rate": s0_scored["metrics"]["natural_stop_lenient_parse_failure_rate"],
        "natural_stop_strict_parse_failure_rate": s0_scored["metrics"]["natural_stop_strict_parse_failure_rate"],
        "combined_unusable_rate": s0_scored["metrics"]["combined_unusable_rate"],
        "verifier_error_count": s0_scored["metrics"]["verifier_error_count"],
        **s0_slo,
        "output_tokens_quantiles": orig_audit["output_tokens_quantiles"],
    }
    write_json(ROOT / "results" / "plane1_verifier_audit_dense_rescored_v2.json", rescored_audit)

    # Recompute paired wins/losses/ties, discordance rates, and repaired SLO goodput across S1, S2, S3 vs S0
    s0_sum = orig_summaries["S0_bf16_tp2"]
    cands_vs_s0: dict[str, Any] = {}
    for cand_id in ("S1_bf16_tp2_mtp3", "S2_int4_tp1", "S3_int4_tp1_mtp3"):
        sc = scored_arms[cand_id]
        os_sum = orig_summaries[cand_id]
        c_slo = recomputed_slo_by_arm[cand_id]
        c_items = {r["id"]: float(r["value"]) for r in sc["items"]}
        wins = sum(1 for k in s0_items if c_items[k] > 0.5 and s0_items[k] < 0.5)
        losses = sum(1 for k in s0_items if c_items[k] < 0.5 and s0_items[k] > 0.5)
        ties = len(s0_items) - wins - losses
        n = len(s0_items)
        raw_c = raw_rows_by_arm[cand_id]
        changed_vs_pre = sorted(pid for pid, nv in c_items.items() if (nv > 0.5) != bool(raw_c[pid].get("lenient_correct")))
        cands_vs_s0[cand_id] = {
            "raw_generations_sha256": sc["generations_sha256"],
            "scored_artifact": f"results/plane1_rescored_search128/{cand_id}_scored_v2.json",
            "scored_artifact_sha256": file_hash(rescored_dir / f"{cand_id}_scored_v2.json"),
            "pre_repair_lenient_accuracy": os_sum["lenient_accuracy"],
            "lenient_accuracy": sc["metrics"]["lenient_accuracy"],
            "strict_accuracy": sc["metrics"]["strict_accuracy"],
            "v1_parity_accuracy": sc["metrics"]["v1_parity_accuracy"],
            "natural_stop_lenient_parse_failure_rate": sc["metrics"]["natural_stop_lenient_parse_failure_rate"],
            "natural_stop_strict_parse_failure_rate": sc["metrics"]["natural_stop_strict_parse_failure_rate"],
            "combined_unusable_rate": sc["metrics"]["combined_unusable_rate"],
            "changed_item_ids_vs_pre_repair": changed_vs_pre,
            "stop_rate": sc["metrics"]["stop_rate"],
            "verifier_error_count": sc["metrics"]["verifier_error_count"],
            "wins_vs_S0": wins,
            "losses_vs_S0": losses,
            "ties_vs_S0": ties,
            "p_win_hat": wins / n,
            "p_loss_hat": losses / n,
            "discordance_rate_hat": (wins + losses) / n,
            "paired_delta_hat": (wins - losses) / n,
            "dataset_seconds": os_sum["dataset_seconds"],
            "speedup_wall_vs_S0": float(s0_sum["dataset_seconds"]) / float(os_sum["dataset_seconds"]),
            "assigned_gpu_hours": os_sum["assigned_gpu_hours"],
            "gpu_hour_reduction_vs_S0": float(s0_sum["assigned_gpu_hours"]) / float(os_sum["assigned_gpu_hours"]),
            "output_tokens_per_second": os_sum["output_tokens_per_second"],
            "tpot_p95_ms": os_sum["tpot_p95_ms"],
            "board_joules": os_sum["board_joules"],
            "mtp_speculative": os_sum["mtp_speculative"],
            **c_slo,
            "slo_correct_goodput_ratio_vs_S0": c_slo["slo_correct_goodput_per_second"] / s0_slo["slo_correct_goodput_per_second"],
        }

    rescored_disc = {
        "status": "SEARCH128_SERVING_DISCORDANCE_RESCORED_V2",
        "preserves_original_artifact": "results/plane1_search128_serving_discordance.json",
        "original_artifact_sha256": file_hash(ROOT / "results" / "plane1_search128_serving_discordance.json"),
        "panel": "math_search128.jsonl",
        "panel_sha256": file_hash(split_path),
        "n_problems": len(s0_items),
        "verifier_sha256": verifier_sha,
        "isolated_scorer_sha256": scorer_sha,
        "runtime": runtime_versions,
        "S0_bf16_tp2_baseline": {
            "raw_generations_sha256": s0_scored["generations_sha256"],
            "scored_artifact": "results/plane1_rescored_search128/S0_bf16_tp2_scored_v2.json",
            "scored_artifact_sha256": file_hash(rescored_dir / "S0_bf16_tp2_scored_v2.json"),
            "pre_repair_lenient_accuracy": s0_sum["lenient_accuracy"],
            "lenient_accuracy": s0_scored["metrics"]["lenient_accuracy"],
            "strict_accuracy": s0_scored["metrics"]["strict_accuracy"],
            "v1_parity_accuracy": s0_scored["metrics"]["v1_parity_accuracy"],
            "natural_stop_lenient_parse_failure_rate": s0_scored["metrics"]["natural_stop_lenient_parse_failure_rate"],
            "natural_stop_strict_parse_failure_rate": s0_scored["metrics"]["natural_stop_strict_parse_failure_rate"],
            "combined_unusable_rate": s0_scored["metrics"]["combined_unusable_rate"],
            "changed_item_ids_vs_pre_repair": repaired_s0_ids,
            "stop_rate": s0_scored["metrics"]["stop_rate"],
            "verifier_error_count": s0_scored["metrics"]["verifier_error_count"],
            "dataset_seconds": s0_sum["dataset_seconds"],
            "assigned_gpu_hours": s0_sum["assigned_gpu_hours"],
            "output_tokens_per_second": s0_sum["output_tokens_per_second"],
            "tpot_p95_ms": s0_sum["tpot_p95_ms"],
            "board_joules": s0_sum["board_joules"],
            **s0_slo,
        },
        "candidates_vs_S0": cands_vs_s0,
    }
    write_json(ROOT / "results" / "plane1_search128_serving_discordance_rescored_v2.json", rescored_disc)
    print(json.dumps({"linux_qualification": linux_qual["historical_40case_replay"], "rescored_discordance": cands_vs_s0}, indent=2))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--extract-historical40-fixture", action="store_true")
    p.add_argument("--run-linux-qualification-and-rescore", action="store_true")
    args = p.parse_args()
    if args.extract_historical40_fixture:
        extract_historical40_fixture()
    if args.run_linux_qualification_and_rescore:
        run_linux_qualification_and_rescore()


if __name__ == "__main__":
    main()
