#!/usr/bin/env python3
"""
strict_guard.py — Experiment-owned hardened enforcement wrapper around bundled governed_loop_guard.py.

Closes all 8 parent-qualified admission probes + minimize-ratio direction trap + problem-cluster rule
WITHOUT modifying the global skill script:
  G1. Non-empty structured env_fingerprint required on candidate, incumbent, and baseline.
  G2. Mandatory keyed items across ALL THREE profiles (candidate, incumbent, baseline) matching expected_item_ids.
  G3. Authoritative frozen expected_cluster_map (item_id -> problem_id or paired_block_id) verified across all 3 profiles.
  G4. Hard minimum independent clusters (min_independent_clusters >= 2 always; >=10 mean/median, >=20 tail quantiles by default).
  G5. frontier_explore requires explicit frontier_coverage_metric and positive min_frontier_gain > 0.
  G6. Strict JSON bool typing on all check-preconditions flags + programmatic filesystem/git/manifest substantiation.
  G7. Mandatory external anchor file pinning manifest_root_sha256 and advancing ledger_head_sha256 outside candidate worktrees.
  G8. At most ONE registered block-expansion retest per candidate, permitted ONLY immediately after INCONCLUSIVE_RETEST with expanded clusters.
  G9. Rejects direction-blind min_ratio_vs_baseline on minimize metrics; requires explicit metric_directions for relative guardrails.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any

EXPECTED_BUNDLED_GUARD_SHA256 = "eb8eaf8b102a29cdd8e85323eb94fdd493873404abce43ed506f1fe252791c0e"

PRECONDITION_BOOL_KEYS = (
    "has_disjoint_dev_and_holdout",
    "multi_shape_e2e_workload",
    "evaluator_locked_and_isolated",
    "evaluator_locked",
    "hermetic_reset_available",
    "clean_reset_available",
    "offline_online_sign_conflict",
    "has_trust_region_guard",
    "has_fast_proxy_stage",
    "one_shot_bug_ruled_out",
    "require_80pct_full_gate_power",
)


def _approx_inv_norm(p: float) -> float:
    p = min(max(p, 1e-12), 1.0 - 1e-12)
    a = (-39.69683028665376, 220.9460984245205, -275.9285104469687, 138.3577518672690, -30.66479806614716, 2.506628277459239)
    b = (-54.47609879822406, 161.5858368580409, -155.6989798598866, 66.80131188771972, -13.28068155288572)
    c = (-0.007784894002430293, -0.3223964580411365, -2.400758277161838, -2.549732539343734, 4.374664141464968, 2.938163982698783)
    d = (0.007784695709041462, 0.3224671290700398, 2.445134137142996, 3.754408661907416)
    plow = 0.02425
    phigh = 1.0 - plow
    if p < plow:
        q = math.sqrt(-2.0 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
            (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0
        )
    if p <= phigh:
        q = p - 0.5
        r = q * q
        return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / (
            ((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0
        )
    q = math.sqrt(-2.0 * math.log(1.0 - p))
    return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
        (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0
    )


def _exact_binom_cdf(k: int, n: int, p: float) -> float:
    if k < 0:
        return 0.0
    if k >= n:
        return 1.0
    if p <= 0.0:
        return 1.0
    if p >= 1.0:
        return 0.0
    terms = [
        math.lgamma(n + 1) - math.lgamma(j + 1) - math.lgamma(n - j + 1)
        + j * math.log(p) + (n - j) * math.log1p(-p)
        for j in range(k + 1)
    ]
    peak = max(terms)
    return min(1.0, math.exp(peak) * sum(math.exp(x - peak) for x in terms))


def _exact_binom_sf(k: int, n: int, p: float) -> float:
    if k <= 0:
        return 1.0
    if k > n:
        return 0.0
    if p <= 0.0:
        return 0.0
    if p >= 1.0:
        return 1.0
    terms = [
        math.lgamma(n + 1) - math.lgamma(j + 1) - math.lgamma(n - j + 1)
        + j * math.log(p) + (n - j) * math.log1p(-p)
        for j in range(k, n + 1)
    ]
    peak = max(terms)
    return min(1.0, math.exp(peak) * sum(math.exp(x - peak) for x in terms))


def _exact_cp_lower(k: int, n: int, alpha: float) -> float:
    if k <= 0:
        return 0.0
    if k > n:
        return 1.0
    lo, hi = 0.0, k / n
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if _exact_binom_sf(k, n, mid) >= alpha:
            hi = mid
        else:
            lo = mid
    return lo


def _exact_cp_upper(k: int, n: int, alpha: float) -> float:
    if k >= n:
        return 1.0
    if k < 0:
        return 0.0
    lo, hi = k / n, 1.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if _exact_binom_cdf(k, n, mid) >= alpha:
            lo = mid
        else:
            hi = mid
    return hi


def _bernoulli_kl(p: float, q: float) -> float:
    if p == q:
        return 0.0
    if q <= 0.0 or q >= 1.0:
        return math.inf
    if p <= 0.0:
        return -math.log1p(-q)
    if p >= 1.0:
        return -math.log(q)
    return math.fsum((
        p * (math.log(p) - math.log(q)),
        (1.0 - p) * (math.log1p(-p) - math.log1p(-q)),
    ))


def _kl_hoeffding_mean_bounds(emp_mean: float, kl_budget: float) -> tuple[float, float]:
    if emp_mean <= 0.0:
        return 0.0, min(1.0, math.nextafter(-math.expm1(-kl_budget), 1.0))
    if emp_mean >= 1.0:
        return max(0.0, math.nextafter(math.exp(-kl_budget), 0.0)), 1.0
    lo, hi = 0.0, emp_mean
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if _bernoulli_kl(emp_mean, mid) > kl_budget:
            lo = mid
        else:
            hi = mid
    lower = max(0.0, math.nextafter(lo, 0.0))

    lo, hi = emp_mean, 1.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if _bernoulli_kl(emp_mean, mid) > kl_budget:
            hi = mid
        else:
            lo = mid
    upper = min(1.0, math.nextafter(hi, 1.0))
    return lower, upper


def conservative_paired_cluster_ni_lcb(
    cand_vals: dict[str, float],
    ref_vals: dict[str, float],
    clusters: dict[str, str],
    *,
    alpha_one_sided: float,
    seed: int = 20261117,
    n_boot: int = 4000,
) -> dict[str, Any]:
    """
    Model-free finite-sample one-sided (1 - alpha) lower confidence bound on the equally weighted
    mean of independent paired problem-cluster deltas delta_c = cand_c - ref_c in [-1, 1]
    (valid for heterogeneous non-IID problem difficulties AND fractional multi-draw cluster means):
      Set X_c = max(delta_c, 0) in [0, 1] and Y_c = max(-delta_c, 0) in [0, 1].
      By convexity of u -> exp(t*u) on [0, 1] and AM-GM across independent clusters,
        E[exp(t * sum_c X_c)] <= prod_c (1 - mu_{X,c} + mu_{X,c} e^t) <= (1 - bar_mu_X + bar_mu_X e^t)^N.
      Chernoff KL inversion bounds bar_mu_X >= L_X at failure <= alpha/2 and bar_mu_Y <= U_Y at
      failure <= alpha/2; Boole's union bound guarantees P(E[delta_bar] < L_X - U_Y) <= alpha.
    Also records exact Clopper-Pearson union bound when all cluster outcomes are binary {0, 1}.
    """
    if isinstance(alpha_one_sided, bool) or not isinstance(alpha_one_sided, (int, float)) or not (0.0 < float(alpha_one_sided) < 0.5):
        raise ValueError(f"alpha_one_sided must lie in (0, 0.5), got {alpha_one_sided!r}")

    by_cluster_c: dict[str, list[float]] = {}
    by_cluster_r: dict[str, list[float]] = {}
    for iid in sorted(cand_vals.keys()):
        cid = clusters[iid]
        by_cluster_c.setdefault(cid, []).append(cand_vals[iid])
        by_cluster_r.setdefault(cid, []).append(ref_vals[iid])

    ckeys = sorted(by_cluster_c.keys())
    n_c = len(ckeys)
    if n_c < 2:
        raise ValueError("Paired NI bound requires >= 2 independent clusters")

    c_means = [statistics.fmean(by_cluster_c[k]) for k in ckeys]
    r_means = [statistics.fmean(by_cluster_r[k]) for k in ckeys]
    for cid, cm, rm in zip(ckeys, c_means, r_means):
        if not (0.0 - 1e-12 <= cm <= 1.0 + 1e-12) or not (0.0 - 1e-12 <= rm <= 1.0 + 1e-12):
            raise ValueError(
                f"NI_OUT_OF_BOUNDS_CLUSTER_VALUE:cluster={cid} has (cand={cm}, ref={rm}) outside [0, 1]"
            )
    deltas = [max(-1.0, min(1.0, cm - rm)) for cm, rm in zip(c_means, r_means)]
    obs_delta = math.fsum(deltas) / n_c
    discordant = sum(1 for d in deltas if abs(d) > 1e-12)

    kl_budget = math.nextafter((math.log(2.0) - math.log(float(alpha_one_sided))) / n_c, math.inf)
    pos_mean = math.fsum(max(d, 0.0) for d in deltas) / n_c
    neg_mean = math.fsum(max(-d, 0.0) for d in deltas) / n_c
    pos_lcb, _ = _kl_hoeffding_mean_bounds(pos_mean, kl_budget)
    _, neg_ucb = _kl_hoeffding_mean_bounds(neg_mean, kl_budget)
    kl_union_lcb = pos_lcb - neg_ucb

    is_binary = all(
        (abs(cm) <= 1e-12 or abs(cm - 1.0) <= 1e-12) and (abs(rm) <= 1e-12 or abs(rm - 1.0) <= 1e-12)
        for cm, rm in zip(c_means, r_means)
    )
    cp_union_lcb: float | None = None
    if is_binary:
        wins = sum(1 for c, r in zip(c_means, r_means) if c > 0.5 and r < 0.5)
        losses = sum(1 for c, r in zip(c_means, r_means) if c < 0.5 and r > 0.5)
        half_alpha = 0.5 * float(alpha_one_sided)
        cp_union_lcb = _exact_cp_lower(wins, n_c, half_alpha) - _exact_cp_upper(losses, n_c, half_alpha)

    return {
        "method": "paired_positive_negative_kl_hoeffding_union",
        "n_clusters": n_c,
        "discordant_clusters": discordant,
        "is_binary_clusters": is_binary,
        "obs_mean_delta": obs_delta,
        "alpha_one_sided": float(alpha_one_sided),
        "positive_lcb": pos_lcb,
        "negative_ucb": neg_ucb,
        "cp_union_lcb_if_iid_binary": cp_union_lcb,
        "conservative_lcb": kl_union_lcb,
    }


def simulate_full_gate_power(
    *,
    delta_plan: float,
    min_effect: float,
    eff_se: float,
    alpha_one_sided: float,
    multiplicity_k: int,
    n_mc: int = 10000,
    seed: int = 20261117,
) -> dict[str, float]:
    eff_alpha = alpha_one_sided / max(1, multiplicity_k)
    z_crit = _approx_inv_norm(1.0 - eff_alpha)
    z_80 = _approx_inv_norm(0.80)
    se_max_analytic = (
        min(delta_plan / (z_crit + z_80), (delta_plan - min_effect) / z_80)
        if delta_plan > min_effect
        else 0.0
    )
    rng = random.Random(seed)
    promotions = 0
    null_fp = 0
    for _ in range(n_mc):
        noise = rng.gauss(0.0, eff_se)
        obs = delta_plan + noise
        if obs >= min_effect and obs > 0.0 and (obs - z_crit * eff_se) > 0.0:
            promotions += 1
        obs_null = noise
        if obs_null >= min_effect and obs_null > 0.0 and (obs_null - z_crit * eff_se) > 0.0:
            null_fp += 1
    return {
        "mc_full_gate_power": promotions / n_mc,
        "mc_null_false_positive_rate": null_fp / n_mc,
        "analytic_max_se_for_80pct": se_max_analytic,
        "z_crit": z_crit,
    }


def _sha256_bytes(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_strict_json(path: Path) -> Any:
    def _reject_nonfinite(tok: str) -> Any:
        raise ValueError(f"Non-finite JSON constant {tok!r} forbidden in {path}")

    return json.loads(path.read_text(encoding="utf-8"), parse_constant=_reject_nonfinite)


def _write_atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True, allow_nan=False)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(path)


def _verify_guard_binary(guard_path: Path) -> None:
    actual = _sha256_bytes(guard_path)
    if actual != EXPECTED_BUNDLED_GUARD_SHA256:
        raise RuntimeError(
            f"Bundled guard SHA256 mismatch: expected {EXPECTED_BUNDLED_GUARD_SHA256}, got {actual}"
        )


def _run_guard(guard_path: Path, args: list[str], cwd: Path | None = None) -> tuple[int, dict[str, Any]]:
    _verify_guard_binary(guard_path)
    env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONDONTWRITEBYTECODE="1")
    proc = subprocess.run(
        [sys.executable, "-I", str(guard_path), *args],
        cwd=str(cwd) if cwd else None,
        env=env,
        capture_output=True,
         encoding="utf-8",
        timeout=120,
        check=False,
    )
    raw = proc.stdout.strip()
    try:
        parsed = json.loads(raw) if raw else {"stdout": proc.stdout, "stderr": proc.stderr}
    except json.JSONDecodeError:
        parsed = {"stdout": proc.stdout, "stderr": proc.stderr}
    return proc.returncode, parsed


def _extract_strict_items(payload: dict[str, Any], label: str) -> tuple[dict[str, float], dict[str, str]]:
    if "items" not in payload or not isinstance(payload["items"], list) or not payload["items"]:
        raise ValueError(f"G2_MISSING_KEYED_ITEMS:{label}.items must be a non-empty list of {{id, cluster_id, value}}")
    vals: dict[str, float] = {}
    clusters: dict[str, str] = {}
    for idx, row in enumerate(payload["items"]):
        if not isinstance(row, dict) or not {"id", "cluster_id", "value"} <= set(row.keys()):
            raise ValueError(f"G2_INVALID_ITEM_SCHEMA:{label}.items[{idx}] requires explicit id, cluster_id, value")
        iid = str(row["id"]).strip()
        cid = str(row["cluster_id"]).strip()
        if not iid or not cid:
            raise ValueError(f"G3_EMPTY_ID_OR_CLUSTER:{label}.items[{idx}]")
        if iid in vals:
            raise ValueError(f"G2_DUPLICATE_ITEM_ID:{label}.items[{iid}]")
        val = row["value"]
        if isinstance(val, bool) or not isinstance(val, (int, float)) or not math.isfinite(float(val)):
            raise ValueError(f"G2_NONFINITE_ITEM_VALUE:{label}.items[{iid}]={val!r}")
        vals[iid] = float(val)
        clusters[iid] = cid
    return vals, clusters


def validate_rules_schema(rules: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    pm = str(rules.get("primary_metric", "")).strip()
    if not pm:
        errors.append("G9_MISSING_PRIMARY_METRIC")
    direction = str(rules.get("direction", "")).lower()
    if direction not in ("maximize", "minimize"):
        errors.append(f"G9_INVALID_DIRECTION:{direction!r}")
    stat_name = str(rules.get("primary_statistic", "mean")).lower()
    stage = str(rules.get("decision_stage", "")).lower()
    if stage not in ("search", "confirm", "frontier_explore"):
        errors.append(f"G9_INVALID_DECISION_STAGE:{stage!r}")

    if not isinstance(rules.get("require_env_fingerprint_match"), bool) or not rules["require_env_fingerprint_match"]:
        errors.append("G1_ENV_FINGERPRINT_MATCH_MUST_BE_TRUE_BOOL")

    exp_ids = rules.get("expected_item_ids")
    if not isinstance(exp_ids, list) or not exp_ids or len(set(map(str, exp_ids))) != len(exp_ids):
        errors.append("G2_EXPECTED_ITEM_IDS_REQUIRED_UNIQUE_NONEMPTY_LIST")
        exp_id_set: set[str] = set()
    else:
        exp_id_set = {str(x) for x in exp_ids}

    cmap = rules.get("expected_cluster_map")
    if not isinstance(cmap, dict) or set(cmap.keys()) != exp_id_set or any(not str(v).strip() for v in cmap.values()):
        errors.append("G3_AUTHORITATIVE_EXPECTED_CLUSTER_MAP_REQUIRED_MATCHING_EXPECTED_ITEM_IDS")
        n_clusters = 0
    else:
        n_clusters = len({str(v) for v in cmap.values()})

    tail_stat = stat_name in ("p90", "p95", "p99")
    default_min_clusters = 20 if (tail_stat and stage == "confirm") else (10 if stage == "confirm" else 2)
    min_clusters = rules.get("min_independent_clusters", default_min_clusters)
    if isinstance(min_clusters, bool) or not isinstance(min_clusters, int) or min_clusters < 2:
        errors.append(f"G4_INVALID_MIN_INDEPENDENT_CLUSTERS:{min_clusters!r}<2")
    elif n_clusters > 0 and n_clusters < min_clusters:
        errors.append(f"G4_CLUSTER_MAP_BELOW_MINIMUM:n_clusters={n_clusters}<min_independent_clusters={min_clusters}")

    if stage == "frontier_explore":
        fc = str(rules.get("frontier_coverage_metric", "")).strip()
        min_fg = rules.get("min_frontier_gain")
        if not fc:
            errors.append("G5_FRONTIER_EXPLORE_REQUIRES_FRONTIER_COVERAGE_METRIC")
        if isinstance(min_fg, bool) or not isinstance(min_fg, (int, float)) or float(min_fg) <= 0:
            errors.append(f"G5_FRONTIER_EXPLORE_REQUIRES_POSITIVE_MIN_FRONTIER_GAIN:{min_fg!r}")

    metric_dirs = {str(k): str(v).lower() for k, v in rules.get("metric_directions", {}).items()}
    if pm:
        metric_dirs.setdefault(pm, direction)
    for m_name in rules.get("min_ratio_vs_baseline", {}):
        if m_name not in metric_dirs:
            errors.append(f"G9_MISSING_METRIC_DIRECTION:{m_name}")
        elif metric_dirs[m_name] == "minimize":
            errors.append(
                f"G9_MIN_RATIO_ON_MINIMIZE_METRIC_FORBIDDEN:{m_name} "
                "(bundled guard checks raw cv/bv < min_ratio; use max_ratio_vs_baseline or hard_ceilings)"
            )
    for section in ("max_ratio_vs_baseline", "max_degradation_vs_baseline"):
        for m_name in rules.get(section, {}):
            if m_name not in metric_dirs:
                errors.append(f"G9_MISSING_METRIC_DIRECTION:{section}.{m_name}")

    if rules.get("require_paired_quality_ni") is not None and not isinstance(rules.get("require_paired_quality_ni"), bool):
        errors.append("B2_REQUIRE_PAIRED_QUALITY_NI_MUST_BE_BOOL")
    if rules.get("require_paired_quality_ni") is True:
        q_ids = rules.get("expected_quality_item_ids")
        if not isinstance(q_ids, list) or not q_ids or len(set(map(str, q_ids))) != len(q_ids):
            errors.append("B2_EXPECTED_QUALITY_ITEM_IDS_REQUIRED_UNIQUE_NONEMPTY_LIST")
            q_id_set: set[str] = set()
        else:
            q_id_set = {str(x) for x in q_ids}
        q_cmap = rules.get("expected_quality_cluster_map")
        if not isinstance(q_cmap, dict) or set(q_cmap.keys()) != q_id_set or any(not str(v).strip() for v in q_cmap.values()):
            errors.append("B2_AUTHORITATIVE_EXPECTED_QUALITY_CLUSTER_MAP_REQUIRED_MATCHING_QUALITY_IDS")
        else:
            min_q_clus = int(rules.get("min_quality_independent_clusters", 2))
            n_q_clus = len({str(v) for v in q_cmap.values()})
            if n_q_clus < max(2, min_q_clus):
                errors.append(f"B2_QUALITY_CLUSTER_MAP_BELOW_MINIMUM:{n_q_clus}<{max(2, min_q_clus)}")

    return errors


def _validate_finite_metrics(payload: dict[str, Any], label: str) -> list[str]:
    metrics = payload.get("metrics")
    if not isinstance(metrics, dict) or not metrics:
        return [f"B2_MISSING_OR_EMPTY_METRICS_DICT:{label}"]
    errs: list[str] = []
    for mk, mv in metrics.items():
        if isinstance(mv, bool) or not isinstance(mv, (int, float)) or not math.isfinite(float(mv)):
            errs.append(f"B2_NONFINITE_OR_INVALID_METRIC:{label}.metrics.{mk}={mv!r}")
    return errs


def cmd_init_anchor(args: argparse.Namespace) -> int:
    guard_path = Path(args.guard).resolve(strict=True)
    manifest_path = Path(args.manifest).resolve()
    root = Path(args.root).resolve(strict=True)
    anchor_path = Path(args.anchor_file).resolve()
    if anchor_path.exists() and not args.new_epoch:
        print(json.dumps({"status": "ERROR", "error": "Anchor file already exists; pass --new-epoch only before trial ledger execution"}), indent=2)
        return 2
    try:
        root_rel = anchor_path.relative_to(root)
        if not str(root_rel).startswith(".local"):
            print(json.dumps({"status": "ERROR", "error": "Anchor file must reside outside candidate-writable tracked tree (.local or external)"}), indent=2)
            return 2
    except ValueError:
        pass

    rc, res = _run_guard(
        guard_path,
        ["freeze-manifest", "--manifest", str(manifest_path), "--root", str(root), "--paths", *args.paths],
        cwd=root,
    )
    if rc != 0:
        print(json.dumps(res, indent=2))
        return rc

    registered_rules_hashes = [
        _sha256_bytes((Path(rp) if Path(rp).is_absolute() else (root / rp)).resolve(strict=True))
        for rp in (args.rules_files or [])
    ]
    anchor_payload = {
        "schema_version": "1.1",
        "protocol_epoch": args.protocol_epoch,
        "guard_sha256": _sha256_bytes(guard_path),
        "manifest_path": manifest_path.as_posix(),
        "manifest_root_sha256": res["manifest_root_sha256"],
        "registered_rules_sha256": registered_rules_hashes,
        "ledger_head_sha256": "GENESIS",
        "ledger_entry_count": 0,
        "retest_cluster_history": {},
    }
    _write_atomic_json(anchor_path, anchor_payload)
    print(json.dumps({"status": "ANCHOR_PINNED", "anchor_file": anchor_path.as_posix(), **anchor_payload}, indent=2))
    return 0


def _verify_splits_on_disk(splits_dir: Path, holdout_labels_dir: Path) -> list[str]:
    errs: list[str] = []
    if not splits_dir.is_dir():
        return [f"G6_SPLITS_DIR_MISSING:{splits_dir}"]
    subset_parents = {
        "math_search_fast64.jsonl": "math_search128.jsonl",
        "math_search_block32.jsonl": "math_search128.jsonl",
        "dapo_search_shard_a48.jsonl": "dapo_search240.jsonl",
        "dapo_search_shard_b48.jsonl": "dapo_search240.jsonl",
    }
    seen_hashes: dict[str, str] = {}
    hashes_by_file: dict[str, set[str]] = {}
    holdout_prompt_files = list(splits_dir.glob("*holdout*prompts.jsonl"))
    if not holdout_prompt_files:
        errs.append("G6_NO_HOLDOUT_PROMPT_SPLITS_FOUND")
    for jf in sorted(splits_dir.glob("*.jsonl")):
        is_subset_alias = jf.name in subset_parents
        f_hashes: set[str] = set()
        for idx, line in enumerate(jf.read_text(encoding="utf-8").splitlines()):
            if not line.strip():
                continue
            row = json.loads(line)
            if "holdout" in jf.name and ("gold_answer" in row or "solution" in row):
                errs.append(f"G6_HOLDOUT_PROMPT_LEAKS_GOLD:{jf.name}:line={idx}")
            ph = str(row.get("prompt_hash", ""))
            if ph:
                f_hashes.add(ph)
                if not is_subset_alias:
                    if ph in seen_hashes and seen_hashes[ph] != jf.name:
                        errs.append(f"G6_CROSS_SPLIT_PROMPT_HASH_OVERLAP:{jf.name}<->{seen_hashes[ph]}")
                        break
                    seen_hashes[ph] = jf.name
        hashes_by_file[jf.name] = f_hashes
    for sub_name, par_name in subset_parents.items():
        if sub_name in hashes_by_file:
            if par_name not in hashes_by_file or not (hashes_by_file[sub_name] <= hashes_by_file[par_name]):
                errs.append(f"G6_SUBSET_ALIAS_NOT_CONTAINED_IN_PARENT:{sub_name}!={par_name}")
    for hp in holdout_prompt_files:
        expected_lbl = hp.name.replace("_prompts.jsonl", "_labels.json")
        if not (holdout_labels_dir / expected_lbl).is_file():
            errs.append(f"G6_MISSING_EXTERNAL_HOLDOUT_LABELS:{expected_lbl}")
    return errs


def cmd_check_preconditions(args: argparse.Namespace) -> int:
    guard_path = Path(args.guard).resolve(strict=True)
    cfg_path = Path(args.config).resolve(strict=True)
    cfg = _load_strict_json(cfg_path)
    strict_blockers: list[str] = []
    power_report: dict[str, Any] | None = None

    # G6a: Strict JSON boolean type check (rejects "false", 0, 1, None)
    for k in PRECONDITION_BOOL_KEYS:
        if k in cfg and not isinstance(cfg[k], bool):
            strict_blockers.append(
                f"G6_STRICT_BOOL_TYPE_VIOLATION:{k}={cfg[k]!r} (type={type(cfg[k]).__name__}; must be JSON boolean)"
            )

    # Reject mismatched/silent power key aliases so required power checks never silently skip!
    for bad_alias, canonical in (
        ("planned_true_effect", "delta_plan"),
        ("alpha_one_sided", "confirm_alpha"),
        ("confirm_multiplicity_k", "planned_confirmations"),
    ):
        if bad_alias in cfg and canonical not in cfg:
            strict_blockers.append(f"B3_UNRECOGNIZED_POWER_KEY_ALIAS:{bad_alias}->use_{canonical}")

    req_80 = cfg.get("require_80pct_full_gate_power") is True
    if req_80 and "delta_plan" not in cfg:
        strict_blockers.append("B3_MISSING_REQUIRED_DELTA_PLAN_FOR_POWER_GATE")
    if req_80:
        for component in ("ni_predicate_power", "guardrail_predicate_power"):
            if component not in cfg:
                strict_blockers.append(f"B3_MISSING_COMPONENT_PREDICATE_POWER:{component}")

    # Full-gate promotion power validation (including joint NI & guardrail predicate power when supplied)
    if "delta_plan" in cfg and not strict_blockers:
        delta_plan = float(cfg["delta_plan"])
        min_eff = float(cfg.get("min_worthwhile_effect", 0.0))
        eff_se = max(float(cfg.get("paired_noise_se", 0.0)), float(cfg.get("aa_noise_se", 0.0)))
        alpha = float(cfg.get("confirm_alpha", 0.025))
        k_conf = max(1, int(cfg.get("planned_confirmations", 2)))
        if delta_plan <= min_eff:
            strict_blockers.append(
                f"B3_INVALID_DELTA_PLAN:delta_plan={delta_plan}<=min_worthwhile_effect={min_eff} "
                "(full-gate promotion power cannot exceed 50% at delta_plan<=eps)"
            )
        elif eff_se <= 0:
            strict_blockers.append("B3_NONPOSITIVE_EFFECTIVE_SE_FOR_POWER_GATE")
        else:
            power_report = simulate_full_gate_power(
                delta_plan=delta_plan,
                min_effect=min_eff,
                eff_se=eff_se,
                alpha_one_sided=alpha,
                multiplicity_k=k_conf,
            )
            ni_pwr = float(cfg.get("ni_predicate_power", 1.0))
            guard_pwr = float(cfg.get("guardrail_predicate_power", 1.0))
            if not (0.0 <= ni_pwr <= 1.0) or not (0.0 <= guard_pwr <= 1.0):
                strict_blockers.append("B3_INVALID_COMPONENT_PREDICATE_POWER")
            joint_frechet_lb = max(0.0, power_report["mc_full_gate_power"] + ni_pwr + guard_pwr - 2.0)
            power_report["ni_predicate_power"] = ni_pwr
            power_report["guardrail_predicate_power"] = guard_pwr
            power_report["complete_predicate_frechet_lower_bound"] = joint_frechet_lb
            if req_80 and joint_frechet_lb < 0.80:
                strict_blockers.append(
                    f"B3_UNDERPOWERED_COMPLETE_PROMOTION_GATE:complete_power_lb={joint_frechet_lb:.4f}<0.80 "
                    f"(primary_mc={power_report['mc_full_gate_power']:.4f}, ni={ni_pwr:.4f}, guard={guard_pwr:.4f})"
                )

    # G6b: Programmatic filesystem / manifest / git / split substantiation
    if args.substantiate_root:
        root = Path(args.substantiate_root).resolve(strict=True)
        git_proc = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=str(root), capture_output=True, text=True, check=False
        )
        if git_proc.returncode != 0 or Path(git_proc.stdout.strip()).resolve() != root:
            strict_blockers.append("G6_UNSUBSTANTIATED_GIT_ISOLATION:Experiment root lacks isolated Git repository")
        else:
            status_proc = subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=str(root), capture_output=True, text=True, check=False
            )
            if status_proc.returncode != 0 or status_proc.stdout.strip():
                strict_blockers.append("G6_DIRTY_GIT_TREE:Hermetic baseline requires clean git status before trials")

        if not args.anchor_file:
            strict_blockers.append("G7_MISSING_ANCHOR_FILE:Substantiated preflight requires --anchor-file")
        else:
            anchor = _load_strict_json(Path(args.anchor_file).resolve(strict=True))
            v_rc, v_res = _run_guard(
                guard_path,
                [
                    "verify-manifest",
                    "--manifest", anchor["manifest_path"],
                    "--root", str(root),
                    "--expected-root-sha256", anchor["manifest_root_sha256"],
                ],
                cwd=root,
            )
            if v_rc != 0:
                strict_blockers.append(f"G7_MANIFEST_VERIFICATION_FAILED:{v_res}")

        if args.split_audit:
            audit = _load_strict_json(Path(args.split_audit).resolve(strict=True))
            if audit.get("status") != "DISJOINT_THREE_WAY_VERIFIED":
                strict_blockers.append(f"G6_SPLIT_AUDIT_NOT_VERIFIED:{audit.get('status')!r}")
            if audit.get("cross_split_exact_or_13gram_overlaps", 1) != 0:
                strict_blockers.append("G6_CROSS_SPLIT_OVERLAP_NONZERO")
            if audit.get("confirm_and_holdout_vs_all_v1_eval_and_train_overlaps", 1) != 0:
                strict_blockers.append("G6_CONFIRM_OR_HOLDOUT_HISTORICAL_V1_OVERLAP_NONZERO")
            if not audit.get("holdout_labels_outside_candidate_scope", False):
                strict_blockers.append("G6_HOLDOUT_LABELS_EXPOSED_IN_CANDIDATE_SCOPE")
            if (root / "data" / "splits").is_dir():
                strict_blockers.extend(
                    _verify_splits_on_disk(
                        root / "data" / "splits",
                        root / ".local" / "governance_anchor" / "holdout_labels",
                    )
                )

    if strict_blockers:
        out = {
            "verdict": "NO_GO",
            "enforced_by": "strict_guard.py",
            "blockers": strict_blockers,
            "full_gate_power": power_report,
            "warnings": [],
        }
        print(json.dumps(out, indent=2))
        return 1

    rc, res = _run_guard(guard_path, ["check-preconditions", "--config", str(cfg_path)])
    if power_report is not None:
        res["full_gate_power"] = power_report
    print(json.dumps(res, indent=2))
    return rc


def cmd_gate_decision(args: argparse.Namespace) -> int:
    guard_path = Path(args.guard).resolve(strict=True)
    cand_path = Path(args.candidate).resolve(strict=True)
    inc_path = Path(args.incumbent).resolve(strict=True)
    base_path = Path(args.baseline).resolve(strict=True)
    rules_path = Path(args.rules).resolve(strict=True)
    ledger_path = Path(args.ledger).resolve()
    anchor_path = Path(args.anchor_file).resolve(strict=True)

    anchor = _load_strict_json(anchor_path)
    cand = _load_strict_json(cand_path)
    inc = _load_strict_json(inc_path)
    base = _load_strict_json(base_path)
    rules = _load_strict_json(rules_path)

    pre_tripwires: list[str] = []

    # G7a: Always verify immutable manifest against externally pinned root SHA-256 (zero bypass flags!)
    m_rc, m_res = _run_guard(
        guard_path,
        [
            "verify-manifest",
            "--manifest", anchor["manifest_path"],
            "--expected-root-sha256", anchor["manifest_root_sha256"],
        ],
    )
    if m_rc != 0:
        pre_tripwires.append(f"G7_MANIFEST_TAMPERED:{m_res.get('error', 'modified_or_added_files')}")

    # Verify rules SHA256 if registered_rules_sha256 is populated in anchor
    reg_rules = anchor.get("registered_rules_sha256") or []
    if reg_rules:
        actual_rules_sha = _sha256_bytes(rules_path)
        if actual_rules_sha not in reg_rules:
            pre_tripwires.append(f"G7_UNREGISTERED_RULES_SHA256:{actual_rules_sha}")

    # G7b: Verify ledger head against externally pinned head SHA-256 (blocks GENESIS truncation!)
    expected_head = str(anchor["ledger_head_sha256"])
    expected_count = int(anchor["ledger_entry_count"])
    if expected_count > 0 and not ledger_path.is_file():
        pre_tripwires.append("G7_LEDGER_DELETED_OR_TRUNCATED_TO_MISSING")
    elif ledger_path.is_file() or expected_head != "GENESIS":
        l_rc, l_res = _run_guard(
            guard_path,
            ["verify-ledger", "--ledger", str(ledger_path), "--expected-head-sha256", expected_head],
        )
        if l_rc != 0 or int(l_res.get("entries", -1)) != expected_count:
            pre_tripwires.append(
                f"G7_LEDGER_ANCHOR_MISMATCH:expected_head={expected_head},expected_entries={expected_count},got={l_res}"
            )

    # G9 + G1..G5: Validate rules schema
    pre_tripwires.extend(validate_rules_schema(rules))

    # G1: Non-empty structured env_fingerprint + strict finite metrics on all 3 primary profiles
    for label, obj in (("candidate", cand), ("incumbent", inc), ("baseline", base)):
        fp = obj.get("env_fingerprint")
        if not isinstance(fp, str) or len(fp.strip()) < 8:
            pre_tripwires.append(f"G1_MISSING_OR_WEAK_ENV_FINGERPRINT:{label}={fp!r}")
        pre_tripwires.extend(_validate_finite_metrics(obj, label))
    if isinstance(cand.get("env_fingerprint"), str) and isinstance(inc.get("env_fingerprint"), str):
        if cand["env_fingerprint"] != inc["env_fingerprint"]:
            pre_tripwires.append(
                f"G1_ENV_FINGERPRINT_MISMATCH:cand={cand['env_fingerprint']!r}!=inc={inc['env_fingerprint']!r}"
            )

    # G2 + G3 + G4: Full keyed items on ALL THREE profiles + authoritative cluster map + min clusters
    n_observed_clusters = 0
    observed_cluster_set: set[str] = set()
    c_vals: dict[str, float] = {}
    i_vals: dict[str, float] = {}
    c_clusters: dict[str, str] = {}
    try:
        c_vals, c_clusters = _extract_strict_items(cand, "candidate")
        i_vals, i_clusters = _extract_strict_items(inc, "incumbent")
        b_vals, b_clusters = _extract_strict_items(base, "baseline")
        exp_ids = {str(x) for x in rules.get("expected_item_ids", [])}
        cmap = {str(k): str(v) for k, v in rules.get("expected_cluster_map", {}).items()}
        for label, vals, clus in (
            ("candidate", c_vals, c_clusters),
            ("incumbent", i_vals, i_clusters),
            ("baseline", b_vals, b_clusters),
        ):
            if set(vals.keys()) != exp_ids:
                pre_tripwires.append(f"G2_ITEM_COVERAGE_MISMATCH:{label}")
            if clus != cmap:
                pre_tripwires.append(f"G3_CLUSTER_MAP_TAMPERED_OR_MISMATCHED:{label}")
        observed_cluster_set = set(c_clusters.values())
        n_observed_clusters = len(observed_cluster_set)
        min_clusters = int(rules.get("min_independent_clusters", 2))
        if n_observed_clusters < max(2, min_clusters):
            pre_tripwires.append(
                f"G4_INSUFFICIENT_INDEPENDENT_CLUSTERS:{n_observed_clusters}<{max(2, min_clusters)}"
            )
    except ValueError as exc:
        pre_tripwires.append(str(exc))

    # G8 + Sequential Alpha-Spending & Strict Cluster-Superset Check on Retest
    prior_rows: list[dict[str, Any]] = []
    if ledger_path.is_file() and not any(t.startswith("G7_LEDGER") for t in pre_tripwires):
        for line in ledger_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                prior_rows.append(json.loads(line))
    cand_id = str(cand.get("candidate_id", args.trial_id))
    protocol_id = str(cand.get("protocol_id", rules.get("decision_stage", "confirm")))
    retest_key = f"{cand_id}::{protocol_id}"
    retest_hist = anchor.get("retest_cluster_history", {})
    matching_prior = [
        r for r in prior_rows
        if r.get("candidate_id") == cand_id and r.get("protocol_id") == protocol_id
    ]
    is_second_look_retest = False
    effective_rules_path = rules_path
    tmp_retest_rules: Path | None = None
    if matching_prior:
        if not rules.get("allow_registered_retest", False):
            pre_tripwires.append(f"G8_DUPLICATE_EVAL_FORBIDDEN:{cand_id}:{protocol_id}")
        elif len(matching_prior) >= 2:
            pre_tripwires.append(
                f"G8_RETEST_CAP_EXCEEDED:{cand_id}:{protocol_id}:prior_evals={len(matching_prior)}>=2"
            )
        elif not prior_rows or prior_rows[-1].get("candidate_id") != cand_id or prior_rows[-1].get("decision") != "INCONCLUSIVE_RETEST":
            pre_tripwires.append(
                "G8_RETEST_REQUIRES_IMMEDIATE_PRECEDING_INCONCLUSIVE"
            )
        else:
            prev_clusters = set(retest_hist.get(retest_key, []))
            req_retest_clusters = int(rules.get("registered_retest_clusters", 0))
            if req_retest_clusters <= 0 or n_observed_clusters < req_retest_clusters:
                pre_tripwires.append(
                    f"G8_RETEST_REQUIRES_BLOCK_EXPANSION:observed={n_observed_clusters}<required={req_retest_clusters}"
                )
            elif not prev_clusters or not (prev_clusters < observed_cluster_set):
                pre_tripwires.append("G8_RETEST_REQUIRES_STRICT_CLUSTER_SUPERSET")
            else:
                is_second_look_retest = True

    # Optional Joint Paired Quality Non-Inferiority Gate (required for Systems confirm promotions!)
    paired_ni_report: dict[str, Any] | None = None
    if rules.get("require_paired_quality_ni") is True or args.quality_candidate or args.quality_incumbent:
        if rules.get("require_paired_quality_ni") is not True:
            pre_tripwires.append("B2_UNREGISTERED_QUALITY_PAYLOAD_REQUIRES_RULE_CONTRACT")
        elif not (args.quality_candidate and args.quality_incumbent):
            pre_tripwires.append("B2_MISSING_PAIRED_QUALITY_PAYLOADS_FOR_JOINT_GATE")
        else:
            qc = _load_strict_json(Path(args.quality_candidate).resolve(strict=True))
            qi = _load_strict_json(Path(args.quality_incumbent).resolve(strict=True))
            for qlabel, qobj in (("quality_candidate", qc), ("quality_incumbent", qi)):
                qfp = qobj.get("env_fingerprint")
                if not isinstance(qfp, str) or len(qfp.strip()) < 8:
                    pre_tripwires.append(f"B2_MISSING_OR_WEAK_QUALITY_ENV_FINGERPRINT:{qlabel}={qfp!r}")
                pre_tripwires.extend(_validate_finite_metrics(qobj, qlabel))
            if isinstance(qc.get("env_fingerprint"), str) and isinstance(qi.get("env_fingerprint"), str):
                if qc["env_fingerprint"] != qi["env_fingerprint"]:
                    pre_tripwires.append("B2_QUALITY_ENV_FINGERPRINT_MISMATCH")
            try:
                qc_vals, qc_clus = _extract_strict_items(qc, "quality_candidate")
                qi_vals, qi_clus = _extract_strict_items(qi, "quality_incumbent")
                exp_q_ids = {str(x) for x in rules.get("expected_quality_item_ids", [])}
                exp_q_cmap = {str(k): str(v) for k, v in rules.get("expected_quality_cluster_map", {}).items()}
                if set(qc_vals.keys()) != exp_q_ids or set(qi_vals.keys()) != exp_q_ids:
                    pre_tripwires.append("B2_QUALITY_ITEM_COVERAGE_MISMATCH")
                elif qc_clus != exp_q_cmap or qi_clus != exp_q_cmap:
                    pre_tripwires.append("B2_QUALITY_CLUSTER_MAP_TAMPERED_OR_MISMATCHED")
                else:
                    q_spend_mult = (
                        (0.6 if is_second_look_retest else 0.4)
                        if (rules.get("allow_registered_retest", False) and str(rules.get("decision_stage", "")).lower() == "confirm")
                        else 1.0
                    )
                    q_alpha = (float(rules.get("alpha_one_sided", 0.025)) * q_spend_mult) / max(
                        1, int(rules.get("multiplicity_k", 1))
                    )
                    ni_margin = float(rules.get("paired_quality_ni_margin", 0.02))
                    q_stop_floor = float(rules.get("paired_quality_stop_floor", 0.96))
                    paired_ni_report = conservative_paired_cluster_ni_lcb(
                        qc_vals, qi_vals, qc_clus, alpha_one_sided=q_alpha
                    )
                    raw_q_stop = qc.get("metrics", {}).get("stop_rate")
                    if (
                        isinstance(raw_q_stop, bool)
                        or not isinstance(raw_q_stop, (int, float))
                        or not math.isfinite(float(raw_q_stop))
                        or not (0.0 <= float(raw_q_stop) <= 1.0)
                    ):
                        pre_tripwires.append(f"B2_INVALID_QUALITY_STOP_RATE:{raw_q_stop!r}")
                    elif not (float(raw_q_stop) >= q_stop_floor):
                        pre_tripwires.append(f"B2_QUALITY_STOP_FLOOR_FAILED:{float(raw_q_stop):.4f}<{q_stop_floor}")
                    if paired_ni_report["conservative_lcb"] < -ni_margin:
                        pre_tripwires.append(
                            f"B2_PAIRED_QUALITY_NI_FAILED:conservative_lcb={paired_ni_report['conservative_lcb']:.5f}<-{ni_margin}"
                        )
            except ValueError as exc:
                pre_tripwires.append(f"B2_QUALITY_EXTRACT_FAILED:{exc}")

    if pre_tripwires:
        print(
            json.dumps(
                {
                    "decision": "ROLLBACK",
                    "enforced_by": "strict_guard.py",
                    "reason": "strict_contract_violation",
                    "tripwires": pre_tripwires,
                    "paired_quality_ni": paired_ni_report,
                },
                indent=2,
            )
        )
        return 1

    # Apply Bonferroni/Pocock two-look alpha spending when allow_registered_retest is active on confirm
    if rules.get("allow_registered_retest", False) and str(rules.get("decision_stage", "")).lower() == "confirm":
        spend_mult = 0.6 if is_second_look_retest else 0.4
        spent_rules = dict(rules, alpha_one_sided=float(rules.get("alpha_one_sided", 0.025)) * spend_mult)
        tmp_retest_rules = anchor_path.parent / f".tmp_spent_rules_{os.getpid()}.json"
        _write_atomic_json(tmp_retest_rules, spent_rules)
        effective_rules_path = tmp_retest_rules

    try:
        rc, res = _run_guard(
            guard_path,
            [
                "gate-decision",
                "--candidate", str(cand_path),
                "--incumbent", str(inc_path),
                "--baseline", str(base_path),
                "--rules", str(effective_rules_path),
                "--ledger", str(ledger_path),
                "--trial-id", args.trial_id,
                "--hypothesis", args.hypothesis or "",
            ],
        )
    finally:
        if tmp_retest_rules and tmp_retest_rules.exists():
            tmp_retest_rules.unlink()

    # B7: Mark frontier_explore PROMOTE as requiring subsequent confirm + retention before final claim (keep ledger decision exact!)
    if str(rules.get("decision_stage", "")).lower() == "frontier_explore" and res.get("decision") == "PROMOTE":
        res["requires_confirm_and_retention_before_final_claim"] = True

    if paired_ni_report is not None:
        res["paired_quality_ni"] = paired_ni_report

    if "entry_sha256" in res:
        anchor["ledger_head_sha256"] = res["entry_sha256"]
        anchor["ledger_entry_count"] = int(anchor["ledger_entry_count"]) + 1
        retest_hist[retest_key] = sorted(observed_cluster_set)
        anchor["retest_cluster_history"] = retest_hist
        _write_atomic_json(anchor_path, anchor)
    print(json.dumps(res, indent=2))
    return rc


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Strict enforcement wrapper around governed_loop_guard.py")
    p.add_argument(
        "--guard",
        default=str(Path(__file__).resolve().parent / "governed_loop_guard.py"),
        help="Path to unmodified bundled governed_loop_guard.py",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    p_anc = sub.add_parser("init-anchor", help="Freeze manifest and write external anchor state")
    p_anc.add_argument("--manifest", required=True)
    p_anc.add_argument("--root", default=".")
    p_anc.add_argument("--paths", nargs="+", required=True)
    p_anc.add_argument("--anchor-file", required=True)
    p_anc.add_argument("--protocol-epoch", default="plane1_calibrated_v1")
    p_anc.add_argument("--rules-files", nargs="*", default=[])
    p_anc.add_argument("--new-epoch", action="store_true")
    p_anc.set_defaults(func=cmd_init_anchor)

    p_pre = sub.add_parser("check-preconditions", help="Strict type-checked and substantiated precondition gate")
    p_pre.add_argument("--config", required=True)
    p_pre.add_argument("--substantiate-root", default=None)
    p_pre.add_argument("--anchor-file", default=None)
    p_pre.add_argument("--split-audit", default=None)
    p_pre.set_defaults(func=cmd_check_preconditions)

    p_gate = sub.add_parser("gate-decision", help="Strict paired, cluster-locked, externally anchored gate decision")
    p_gate.add_argument("--candidate", required=True)
    p_gate.add_argument("--incumbent", required=True)
    p_gate.add_argument("--baseline", required=True)
    p_gate.add_argument("--rules", required=True)
    p_gate.add_argument("--ledger", required=True)
    p_gate.add_argument("--anchor-file", required=True)
    p_gate.add_argument("--trial-id", required=True)
    p_gate.add_argument("--hypothesis", default="")
    p_gate.add_argument("--quality-candidate", default=None)
    p_gate.add_argument("--quality-incumbent", default=None)
    p_gate.set_defaults(func=cmd_gate_decision)

    return p


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
