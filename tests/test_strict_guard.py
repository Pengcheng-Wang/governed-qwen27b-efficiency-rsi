"""Regression suite verifying strict_guard.py passes all 18 valid controls and blocks all 10 adversarial bypasses."""
from __future__ import annotations

import argparse
import copy
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STRICT_GUARD = ROOT / "eval" / "strict_guard.py"
BUNDLED_GUARD = ROOT / "eval" / "governed_loop_guard.py"
sys.path.insert(0, str(ROOT / "code"))
sys.path.insert(0, str(ROOT / "eval"))

from common import final_answer_lenient, final_answer_strict  # noqa: E402
from strict_guard import conservative_paired_cluster_ni_lcb, simulate_full_gate_power  # noqa: E402
from verifiers_v2 import score_response  # noqa: E402


class StatisticalBoundsTests(unittest.TestCase):
    def test_zero_and_sparse_discordance_ni_bounds(self):
        # n=30 zero-discordance at alpha=0.05 -> -(1 - (0.025)^(1/30)) = -11.55 pp (must NOT pass -2.0 pp NI!)
        ids30 = [f"p{i}" for i in range(30)]
        vals30 = {k: 1.0 for k in ids30}
        cmap30 = {k: k for k in ids30}
        r30 = conservative_paired_cluster_ni_lcb(vals30, vals30, cmap30, alpha_one_sided=0.05)
        self.assertAlmostEqual(r30["conservative_lcb"], -(1.0 - 0.025 ** (1.0 / 30)), places=6)
        self.assertLess(r30["conservative_lcb"], -0.11)

        # n=200 zero-discordance at alpha=0.0125 -> -(1 - (0.00625)^(1/200)) = -2.51 pp (< -0.02, fails 2pp NI)
        ids200 = [f"p{i}" for i in range(200)]
        vals200 = {k: 1.0 for k in ids200}
        cmap200 = {k: k for k in ids200}
        r200 = conservative_paired_cluster_ni_lcb(vals200, vals200, cmap200, alpha_one_sided=0.0125)
        self.assertLess(r200["conservative_lcb"], -0.02)

        # n=256 zero-discordance at alpha=0.0125 -> -(1 - (0.00625)^(1/256)) = -1.963 pp (>= -0.02, passes 2pp NI)
        ids256 = [f"p{i}" for i in range(256)]
        vals256 = {k: 1.0 for k in ids256}
        cmap256 = {k: k for k in ids256}
        r256 = conservative_paired_cluster_ni_lcb(vals256, vals256, cmap256, alpha_one_sided=0.0125)
        self.assertGreaterEqual(r256["conservative_lcb"], -0.02)

        # Sparse nonzero loss: 0 wins, 1 loss on n=256 -> must fail -2.0 pp NI
        cand_loss1 = dict(vals256)
        cand_loss1["p0"] = 0.0
        r_loss1 = conservative_paired_cluster_ni_lcb(cand_loss1, vals256, cmap256, alpha_one_sided=0.0125)
        self.assertLess(r_loss1["conservative_lcb"], -0.02)

        # Non-binary fractional cluster scores in [0, 1] are supported by KL-Hoeffding;
        # out-of-[0, 1] cluster scores MUST fail closed!
        frac_cand = dict(vals30)
        frac_cand["p0"] = 0.5
        r_frac = conservative_paired_cluster_ni_lcb(frac_cand, vals30, cmap30, alpha_one_sided=0.05)
        self.assertFalse(r_frac["is_binary_clusters"])
        self.assertLess(r_frac["conservative_lcb"], -0.11)

        oob_cand = dict(vals30)
        oob_cand["p0"] = 1.25
        with self.assertRaisesRegex(ValueError, "NI_OUT_OF_BOUNDS_CLUSTER_VALUE"):
            conservative_paired_cluster_ni_lcb(oob_cand, vals30, cmap30, alpha_one_sided=0.05)

        # Exact 33,153-state trinomial enumeration on parent's adversarial null:
        # N=256, alpha=0.0125, margin=0.02, p_win=0.004, p_loss=0.0241 (true delta = -0.0201 < -0.02)
        import math as _m
        from strict_guard import _kl_hoeffding_mean_bounds
        N = 256
        budget = _m.nextafter((_m.log(2.0) - _m.log(0.0125)) / N, _m.inf)
        pos_lcb_by_w = {w: _kl_hoeffding_mean_bounds(w / N, budget)[0] for w in range(N + 1)}
        neg_ucb_by_l = {l: _kl_hoeffding_mean_bounds(l / N, budget)[1] for l in range(N + 1)}
        pw, pl, pt = 0.004, 0.0241, 1.0 - 0.004 - 0.0241
        exact_false_ni_prob = 0.0
        for w in range(N + 1):
            for l in range(N - w + 1):
                if (pos_lcb_by_w[w] - neg_ucb_by_l[l]) >= -0.02:
                    t = N - w - l
                    lp = (
                        _m.lgamma(N + 1) - _m.lgamma(w + 1) - _m.lgamma(l + 1) - _m.lgamma(t + 1)
                        + (w * _m.log(pw) if w else 0.0)
                        + (l * _m.log(pl) if l else 0.0)
                        + (t * _m.log(pt) if t else 0.0)
                    )
                    exact_false_ni_prob += _m.exp(lp)
        self.assertLessEqual(exact_false_ni_prob, 0.0125)
        self.assertAlmostEqual(exact_false_ni_prob, 0.00193982, places=6)

    def test_full_gate_promotion_power(self):
        # At delta_plan == min_effect, power cannot exceed ~50%
        p_eq = simulate_full_gate_power(
            delta_plan=0.05, min_effect=0.05, eff_se=0.005, alpha_one_sided=0.025, multiplicity_k=2
        )
        self.assertLessEqual(p_eq["mc_full_gate_power"], 0.53)
        # Well-powered planning margin delta_plan=0.08 > min_effect=0.05 with eff_se=0.01 -> >80%
        p_hi = simulate_full_gate_power(
            delta_plan=0.08, min_effect=0.05, eff_se=0.01, alpha_one_sided=0.025, multiplicity_k=2
        )
        self.assertGreaterEqual(p_hi["mc_full_gate_power"], 0.80)


class VerifierDecouplingTests(unittest.TestCase):
    def _semantic_score(self, answer, gold, *, problem=None):
        kwargs = {"finish_reason": "stop", "thinking": True, "integer_task": False}
        if problem is not None:
            kwargs["problem"] = problem
        try:
            result = score_response("reasoning</think>\n" + rf"\boxed{{{answer}}}", gold, **kwargs)
        except TypeError as error:
            self.fail(f"Semantic scorer must accept public problem context: {error}")
        self.assertIsNone(result["verifier_error"])
        return result

    def test_semantic_notation_equivalence(self):
        cases = [
            (r"3 < \lambda \le 4", "(3,4]", True),
            (r"3 \le \lambda \le 4", "(3,4]", False),
            ("55\u00b0", r"55^\circ", True),
            ("56\u00b0", r"55^\circ", False),
            ("80\u00b0", "80", True),
            (
                r"\begin{pmatrix}\frac15\\[1mm]-\frac{18}5\end{pmatrix}",
                r"\begin{pmatrix}1/5\\-18/5\end{pmatrix}", True,
            ),
            (
                r"\begin{pmatrix}\frac15\\[2pt]\frac{18}5\end{pmatrix}",
                r"\begin{pmatrix}1/5\\-18/5\end{pmatrix}", False,
            ),
        ]
        for answer, gold, expected in cases:
            with self.subTest(answer=answer, gold=gold):
                result = self._semantic_score(answer, gold)
                self.assertEqual(result["lenient_correct"], expected)
                if expected:
                    self.assertFalse(result["v1_correct"], "Keep the historical grading mode separate")

    def test_semantic_numeric_text_preserves_list_structure(self):
        for answer, gold, expected in [
            ("4", r"\mbox{four}", True),
            ("5", r"\mbox{four}", False),
            ("81,256", "81, 256", True),
            ("81256", "81, 256", False),
            ("81,257", "81, 256", False),
            ("(4,3)", "(3,4)", False),
        ]:
            with self.subTest(answer=answer, gold=gold):
                result = self._semantic_score(answer, gold)
                self.assertEqual(result["lenient_correct"], expected)
                if expected:
                    self.assertFalse(result["v1_correct"], "Keep the historical grading mode separate")

    def test_units_follow_public_question_context(self):
        for answer, gold, problem, expected in [
            (r"10\%", "10", "By what percent does the area decrease?", True),
            (r"11\%", "10", "By what percent does the area decrease?", False),
            ("0.1", "10", "By what percent does the area decrease?", False),
            (r"10\%", "10", "Compute 5+5.", False),
            (r"10\%", "0.1", "Express the probability as a decimal.", True),
            (r"12^{\text{th}} \text{ grade}", "12", "For which grade is the ratio closest?", True),
            (r"11^{\text{th}} \text{ grade}", "12", "For which grade is the ratio closest?", False),
            (r"12^{\text{th}} \text{ grade}", "12", "Compute 6+6.", False),
        ]:
            with self.subTest(answer=answer, problem=problem):
                result = self._semantic_score(answer, gold, problem=problem)
                self.assertEqual(result["lenient_correct"], expected)

    def test_strict_vs_lenient_extraction(self):
        # Clean single </think> + braced \boxed{} passes both
        clean = "<think>\n step 1 \n</think>\nTherefore \\boxed{42}."
        self.assertEqual(final_answer_strict(clean, thinking=True), "42")
        self.assertEqual(final_answer_lenient(clean, thinking=True), "42")

        # Duplicate </think> rejected by strict SFT admission, recovered by lenient eval
        dup_think = "<think>draft </think> wait </think>\nFinal: \\boxed{108}"
        self.assertIsNone(final_answer_strict(dup_think, thinking=True))
        self.assertEqual(final_answer_lenient(dup_think, thinking=True), "108")

        # Variable prefix inside \boxed{x = 17} stripped cleanly by lenient
        var_box = "<think>solve</think>\n\\boxed{x = 17}"
        self.assertEqual(final_answer_lenient(var_box, thinking=True), "17")

        # Answer: line fallback after </think> when \boxed{} omitted on natural stop
        ans_line = "<think>done</think>\nFinal Answer: -2015028."
        self.assertIsNone(final_answer_strict(ans_line, thinking=True))
        self.assertEqual(final_answer_lenient(ans_line, thinking=True), "-2015028")

        # Unclosed <think> (truncated) returns None in both
        trunc = "<think>still thinking \\boxed{42}"
        self.assertIsNone(final_answer_strict(trunc, thinking=True))
        self.assertIsNone(final_answer_lenient(trunc, thinking=True))


class StrictGuardEnforcementTests(unittest.TestCase):
    def _invoke(self, cwd: Path, *args: str) -> tuple[int, dict]:
        env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONDONTWRITEBYTECODE="1")
        proc = subprocess.run(
            [sys.executable, "-I", str(STRICT_GUARD), "--guard", str(BUNDLED_GUARD), *args],
            cwd=str(cwd),
            env=env,
            capture_output=True,
            encoding="utf-8",
            timeout=60,
            check=False,
        )
        try:
            body = json.loads(proc.stdout)
        except json.JSONDecodeError:
            body = {"stdout": proc.stdout, "stderr": proc.stderr}
        return proc.returncode, body

    def test_controls_and_all_adversarial_probes_blocked(self):
        with tempfile.TemporaryDirectory(prefix="strict_guard_test_") as tmp:
            root = Path(tmp) / "workspace"
            ext_dir = Path(tmp) / "external_anchor"
            root.mkdir()
            ext_dir.mkdir()
            (root / "protected").mkdir()
            scorer = root / "protected" / "scorer.py"
            scorer.write_text("EXPECTED = 20\n", encoding="utf-8")

            # Initialize isolated git repo for substantiation tests
            subprocess.run(["git", "init", "-b", "main"], cwd=str(root), check=True, capture_output=True)
            subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=str(root), check=True)
            subprocess.run(["git", "config", "user.name", "Test"], cwd=str(root), check=True)

            manifest_path = root / "IMMUTABLE_MANIFEST.json"
            anchor_path = ext_dir / "anchor.json"
            rc, init_res = self._invoke(
                root,
                "init-anchor",
                "--manifest", str(manifest_path),
                "--root", str(root),
                "--paths", "protected",
                "--anchor-file", str(anchor_path),
            )
            self.assertEqual(rc, 0, init_res)

            subprocess.run(["git", "add", "."], cwd=str(root), check=True)
            subprocess.run(["git", "commit", "-m", "init"], cwd=str(root), check=True, capture_output=True)

            ids = [f"item-{i:02d}" for i in range(20)]
            cmap = {iid: f"block-{i:02d}" for i, iid in enumerate(ids)}

            def make_payload(name: str, val: float) -> dict:
                return {
                    "candidate_id": name,
                    "protocol_id": "qualification",
                    "env_fingerprint": "a100-sxm4-80gb-cu129-node0",
                    "metrics": {"score": val, "stop_rate": 1.0, "peak_gib": 60.0},
                    "items": [{"id": iid, "cluster_id": cmap[iid], "value": val} for iid in ids],
                }

            cand = make_payload("cand-a", 0.65)
            inc = make_payload("incumbent", 0.50)
            base = make_payload("baseline", 0.50)
            rules = {
                "primary_metric": "score",
                "primary_statistic": "mean",
                "direction": "maximize",
                "decision_stage": "confirm",
                "alpha_one_sided": 0.025,
                "multiplicity_k": 2,
                "min_improvement": 0.05,
                "aa_noise_se": 0.01,
                "require_env_fingerprint_match": True,
                "expected_item_ids": ids,
                "expected_cluster_map": cmap,
                "min_independent_clusters": 10,
                "hard_floors": {"stop_rate": 0.95},
                "hard_ceilings": {"peak_gib": 75.0},
                "max_cumulative_primary_regression": 0.05,
                "max_dev_looks": 20,
                "max_consecutive_rollbacks": 4,
            }

            def write_j(fname: str, obj: dict) -> Path:
                p = ext_dir / fname
                p.write_text(json.dumps(obj, indent=2) + "\n", encoding="utf-8")
                return p

            def fresh_anchor(fname: str) -> Path:
                ap = ext_dir / fname
                ap.write_text(anchor_path.read_text(encoding="utf-8"), encoding="utf-8")
                return ap

            def run_gate(tag: str, c=None, i=None, b=None, r=None, ap=None, ledger_name=None):
                anc = ap if ap is not None else fresh_anchor(f"{tag}-anchor.json")
                led = ext_dir / (ledger_name or f"{tag}-ledger.jsonl")
                return self._invoke(
                    root,
                    "gate-decision",
                    "--candidate", str(write_j(f"{tag}-c.json", cand if c is None else c)),
                    "--incumbent", str(write_j(f"{tag}-i.json", inc if i is None else i)),
                    "--baseline", str(write_j(f"{tag}-b.json", base if b is None else b)),
                    "--rules", str(write_j(f"{tag}-r.json", rules if r is None else r)),
                    "--ledger", str(led),
                    "--anchor-file", str(anc),
                    "--trial-id", tag,
                )

            # Valid positive controls
            rc, res = run_gate("ctrl-promote")
            self.assertEqual(rc, 0, res)
            self.assertEqual(res["decision"], "PROMOTE")

            rc, res = run_gate("ctrl-search", r=dict(rules, decision_stage="search"))
            self.assertEqual(rc, 0, res)
            self.assertEqual(res["decision"], "PROVISIONAL_KEEP")

            # Probe 1: Both missing env_fingerprint -> MUST FAIL CLOSED
            c_no_env = copy.deepcopy(cand)
            i_no_env = copy.deepcopy(inc)
            del c_no_env["env_fingerprint"]
            del i_no_env["env_fingerprint"]
            rc, res = run_gate("probe1-missing-env", c=c_no_env, i=i_no_env)
            self.assertNotEqual(rc, 0)
            self.assertEqual(res["decision"], "ROLLBACK")

            # Probe 2: Missing items on incumbent/baseline -> MUST FAIL CLOSED
            i_no_items = copy.deepcopy(inc)
            b_no_items = copy.deepcopy(base)
            del i_no_items["items"]
            del b_no_items["items"]
            rc, res = run_gate("probe2-missing-ref-items", i=i_no_items, b=b_no_items)
            self.assertNotEqual(rc, 0)
            self.assertEqual(res["decision"], "ROLLBACK")

            # Probe 3: Cluster label disagreement or candidate relabeling -> MUST FAIL CLOSED
            i_bad_clus = copy.deepcopy(inc)
            for row in i_bad_clus["items"]:
                row["cluster_id"] = "fake-cluster"
            rc, res = run_gate("probe3-cluster-mismatch", i=i_bad_clus)
            self.assertNotEqual(rc, 0)
            self.assertEqual(res["decision"], "ROLLBACK")

            # Probe 4: Single independent cluster -> MUST FAIL CLOSED
            one_cmap = {iid: "single-run" for iid in ids}
            one_c, one_i, one_b = [
                dict(
                    copy.deepcopy(x),
                    items=[{"id": iid, "cluster_id": "single-run", "value": x["metrics"]["score"]} for iid in ids],
                )
                for x in (cand, inc, base)
            ]
            one_r = dict(rules, expected_cluster_map=one_cmap, min_independent_clusters=10)
            rc, res = run_gate("probe4-single-cluster", c=one_c, i=one_i, b=one_b, r=one_r)
            self.assertNotEqual(rc, 0)
            self.assertEqual(res["decision"], "ROLLBACK")

            # Probe 5: frontier_explore without frontier_coverage_metric -> MUST FAIL CLOSED
            fr_bad = dict(rules, decision_stage="frontier_explore", max_cumulative_primary_regression=0.1)
            rc, res = run_gate("probe5-missing-frontier-metric", c=inc, r=fr_bad)
            self.assertNotEqual(rc, 0)
            self.assertEqual(res["decision"], "ROLLBACK")

            # Probe 6: String "false" in check-preconditions -> MUST FAIL CLOSED
            preflight = {
                "archetype": "hybrid",
                "verifier_type": "deterministic",
                "has_disjoint_dev_and_holdout": True,
                "evaluator_locked_and_isolated": "false",
                "hermetic_reset_available": True,
                "has_trust_region_guard": True,
                "single_trial_minutes": 10,
                "total_budget_minutes": 200,
                "learnable_fraction": 0.25,
                "paired_noise_se": 0.01,
                "aa_noise_se": 0.01,
                "min_worthwhile_effect": 0.05,
            }
            rc, res = self._invoke(
                root,
                "check-preconditions",
                "--config", str(write_j("probe6-preflight.json", preflight)),
            )
            self.assertNotEqual(rc, 0)
            self.assertEqual(res["verdict"], "NO_GO")

            # Substantiated valid preflight -> MUST PASS with explicit complete_predicate_frechet_lower_bound >= 0.80
            valid_pre = dict(
                preflight,
                evaluator_locked_and_isolated=True,
                delta_plan=0.10,
                confirm_alpha=0.025,
                planned_confirmations=1,
                ni_predicate_power=0.95,
                guardrail_predicate_power=0.99,
                require_80pct_full_gate_power=True,
            )
            split_audit = {
                "status": "DISJOINT_THREE_WAY_VERIFIED",
                "cross_split_exact_or_13gram_overlaps": 0,
                "confirm_and_holdout_vs_all_v1_eval_and_train_overlaps": 0,
                "holdout_labels_outside_candidate_scope": True,
            }
            rc, res = self._invoke(
                root,
                "check-preconditions",
                "--config", str(write_j("valid-preflight.json", valid_pre)),
                "--substantiate-root", str(root),
                "--anchor-file", str(anchor_path),
                "--split-audit", str(write_j("split-audit.json", split_audit)),
            )
            self.assertEqual(rc, 0, res)
            self.assertEqual(res["verdict"], "GO")
            self.assertGreaterEqual(res["full_gate_power"]["complete_predicate_frechet_lower_bound"], 0.80)

            for missing_keys in (
                ("ni_predicate_power",),
                ("guardrail_predicate_power",),
                ("ni_predicate_power", "guardrail_predicate_power"),
            ):
                with self.subTest(missing_component_powers=missing_keys):
                    incomplete_power = {k: v for k, v in valid_pre.items() if k not in missing_keys}
                    rc_missing, res_missing = self._invoke(
                        root,
                        "check-preconditions",
                        "--config", str(write_j("missing-component-power.json", incomplete_power)),
                    )
                    self.assertNotEqual(rc_missing, 0, res_missing)
                    self.assertEqual(res_missing["verdict"], "NO_GO")
                    for missing_key in missing_keys:
                        self.assertIn(
                            f"B3_MISSING_COMPONENT_PREDICATE_POWER:{missing_key}",
                            res_missing["blockers"],
                        )

            # Probe 6b: require_80pct_full_gate_power=True WITHOUT delta_plan -> MUST FAIL CLOSED!
            missing_dp = dict(preflight, evaluator_locked_and_isolated=True, require_80pct_full_gate_power=True)
            rc_mdp, res_mdp = self._invoke(
                root,
                "check-preconditions",
                "--config", str(write_j("missing-dp.json", missing_dp)),
            )
            self.assertNotEqual(rc_mdp, 0)
            self.assertEqual(res_mdp["verdict"], "NO_GO")

            for bad_required_power in ("true", "false", 1, 0, None):
                with self.subTest(required_power=bad_required_power):
                    bad_power_flag = dict(
                        valid_pre,
                        require_80pct_full_gate_power=bad_required_power,
                        ni_predicate_power=0.1,
                        guardrail_predicate_power=0.1,
                    )
                    rc_bad, res_bad = self._invoke(
                        root,
                        "check-preconditions",
                        "--config", str(write_j("bad-power-flag.json", bad_power_flag)),
                    )
                    self.assertNotEqual(rc_bad, 0, res_bad)
                    self.assertTrue(any(
                        item.startswith("G6_STRICT_BOOL_TYPE_VIOLATION:require_80pct_full_gate_power=")
                        for item in res_bad["blockers"]
                    ), res_bad)

            # Probe 7: Ledger truncation to empty after 1 entry -> MUST FAIL CLOSED via anchor head
            shared_anc = fresh_anchor("probe7-anchor.json")
            rc1, _ = run_gate("probe7-step1", ap=shared_anc, ledger_name="probe7.jsonl")
            self.assertEqual(rc1, 0)
            (ext_dir / "probe7.jsonl").write_text("", encoding="utf-8")
            cand2 = dict(copy.deepcopy(cand), candidate_id="cand-b")
            rc2, res2 = run_gate("probe7-step2", c=cand2, ap=shared_anc, ledger_name="probe7.jsonl")
            self.assertNotEqual(rc2, 0)
            self.assertEqual(res2["decision"], "ROLLBACK")

            # Probe 8: Unlimited retests with allow_registered_retest=True -> MUST BLOCK >1 or after PROMOTE
            retest_anc = fresh_anchor("probe8-anchor.json")
            retest_rules = dict(rules, allow_registered_retest=True, registered_retest_clusters=20)
            rc_r0, _ = run_gate("probe8-r0", r=retest_rules, ap=retest_anc, ledger_name="probe8.jsonl")
            self.assertEqual(rc_r0, 0)
            # Second eval right after PROMOTE (not INCONCLUSIVE_RETEST) -> MUST FAIL CLOSED
            rc_r1, res_r1 = run_gate("probe8-r1", r=retest_rules, ap=retest_anc, ledger_name="probe8.jsonl")
            self.assertNotEqual(rc_r1, 0)
            self.assertEqual(res_r1["decision"], "ROLLBACK")

            # Probe 9: Minimize metric with min_ratio_vs_baseline -> MUST FAIL CLOSED
            min_bad_rules = dict(
                rules,
                direction="minimize",
                metric_directions={"score": "minimize"},
                min_ratio_vs_baseline={"score": 0.9},
            )
            rc9, res9 = run_gate("probe9-min-ratio-trap", r=min_bad_rules)
            self.assertNotEqual(rc9, 0)
            self.assertEqual(res9["decision"], "ROLLBACK")

            # Probe 10: Tampering protected scorer.py -> MUST FAIL CLOSED before gate-decision
            scorer.write_text("EXPECTED = 0\n", encoding="utf-8")
            rc10, res10 = run_gate("probe10-tampered-scorer")
            self.assertNotEqual(rc10, 0)
            self.assertEqual(res10["decision"], "ROLLBACK")


class IsolatedScorerSystemsAbabTests(unittest.TestCase):
    SCORER = ROOT / "eval" / "isolated_scorer.py"

    def _run_abab_cli(self, blocks_payload: dict, out_path: Path, block_metric: str = "assigned_gpu_seconds") -> subprocess.CompletedProcess:
        in_path = out_path.with_suffix(".in.json")
        in_path.write_text(json.dumps(blocks_payload), encoding="utf-8")
        return subprocess.run(
            [
                sys.executable,
                "-I",
                "-B",
                str(self.SCORER),
                "systems_abab",
                "--blocks-json",
                str(in_path),
                "--arm",
                "S3",
                "--block-metric",
                block_metric,
                "--statistic",
                "mean",
                "--candidate-id",
                "S3_cand",
                "--protocol-id",
                "confirm_test",
                "--output",
                str(out_path),
            ],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            check=False,
        )

    def _valid_12block_payload(self) -> dict:
        return {
            "env_fingerprint": "node2_dedicated_williams12_test_fp",
            "blocks": [
                {
                    "block_index": idx,
                    "arms": {
                        "S3": {
                            "assigned_gpu_seconds": 175.0 + idx,
                            "assigned_gpu_hours": (175.0 + idx) / 3600.0,
                            "dataset_seconds": 175.0 + idx,
                            "stop_rate": 1.0,
                            "after_nll": 0.629 + 0.0001 * idx,
                            "peak_allocated_gib": 54.5,
                            "board_joules": 160000.0,
                            "tpot_p95_ms": 16.7,
                            "ttft_p95_ms": 120.0,
                            "infrastructure_errors": 0,
                        }
                    },
                }
                for idx in range(12)
            ],
        }

    def test_systems_abab_enforces_complete_per_block_guardrails_and_explicit_infra_errors(self):
        with tempfile.TemporaryDirectory(prefix="abab_scorer_cli_") as tmp:
            tdir = Path(tmp)
            valid = self._valid_12block_payload()
            out_ok = tdir / "valid_out.json"
            proc_ok = self._run_abab_cli(valid, out_ok)
            self.assertEqual(proc_ok.returncode, 0, proc_ok.stderr)
            self.assertTrue(out_ok.is_file())

            for omitted_key in ("stop_rate", "after_nll", "infrastructure_errors", "dataset_seconds", "peak_allocated_gib"):
                with self.subTest(partial_omit_block5=omitted_key):
                    bad = copy.deepcopy(valid)
                    del bad["blocks"][5]["arms"]["S3"][omitted_key]
                    out_bad = tdir / f"omit_b5_{omitted_key}.json"
                    proc_bad = self._run_abab_cli(bad, out_bad)
                    self.assertNotEqual(proc_bad.returncode, 0, f"Partial omission of {omitted_key} in block 5 must fail closed!")
                    self.assertFalse(out_bad.is_file())

            # Omitting infrastructure_errors across ALL blocks must ALSO fail closed (no missing->zero default!)
            no_ie = copy.deepcopy(valid)
            for b in no_ie["blocks"]:
                del b["arms"]["S3"]["infrastructure_errors"]
            out_no_ie = tdir / "no_infra_all.json"
            proc_no_ie = self._run_abab_cli(no_ie, out_no_ie)
            self.assertNotEqual(proc_no_ie.returncode, 0, "Missing infrastructure_errors across all blocks must fail closed!")
            self.assertFalse(out_no_ie.is_file())

            # Explicit None, bool, or string values in tracked metrics must fail closed
            for bad_field, bad_val in (("stop_rate", None), ("after_nll", "0.629"), ("infrastructure_errors", True), ("infrastructure_errors", -1)):
                with self.subTest(malformed_field=bad_field, val=bad_val):
                    mal = copy.deepcopy(valid)
                    mal["blocks"][5]["arms"]["S3"][bad_field] = bad_val
                    out_mal = tdir / f"mal_{bad_field}.json"
                    proc_mal = self._run_abab_cli(mal, out_mal)
                    self.assertNotEqual(proc_mal.returncode, 0)
                    self.assertFalse(out_mal.is_file())


class TrainSearchStateMachineTests(unittest.TestCase):
    def test_search_finalist_selector_never_falls_back_or_bypasses_gate(self):
        # Import only pure selector logic without requiring remote vllm/httpx runtime
        import ast
        src = (ROOT / "code" / "plane2_governed_runner.py").read_text(encoding="utf-8")
        mod = ast.parse(src)
        fn_node = next(n for n in mod.body if isinstance(n, ast.FunctionDef) and n.name == "select_train_search_finalist")
        ns: dict = {"Any": object}
        exec(compile(ast.Module(body=[fn_node], type_ignores=[]), "plane2_governed_runner.py", "exec"), ns)
        select_fn = ns["select_train_search_finalist"]

        scored = {
            "T1": {"metrics": {"loop_seconds": 118.0}},
            "T2": {"metrics": {"loop_seconds": 105.0}},
        }
        # 1. Empty eligible set -> MUST return None (never arbitrary T2 fallback!)
        self.assertIsNone(select_fn({"T1": {"decision": "ROLLBACK"}, "T2": {"decision": "ROLLBACK"}}, scored))
        # 2. Faster T2 failed strict gate while slower T1 passed -> MUST select T1, never T2!
        self.assertEqual(select_fn({"T1": {"decision": "PROVISIONAL_KEEP"}, "T2": {"decision": "ROLLBACK"}}, scored), "T1")
        # 3. Both passed strict Search gate -> selects fastest admitted finalist T2
        self.assertEqual(select_fn({"T1": {"decision": "PROVISIONAL_KEEP"}, "T2": {"decision": "PROVISIONAL_KEEP"}}, scored), "T2")


class HermeticWorktreeAndConfirmBoundaryTests(unittest.TestCase):
    def _init_mini_governed_repo(self, base: Path) -> tuple[Path, Path]:
        repo = base / "study"
        (repo / "code").mkdir(parents=True)
        (repo / "eval").mkdir(parents=True)
        (repo / "configs").mkdir(parents=True)
        (repo / "data" / "splits").mkdir(parents=True)
        (repo / "models" / "dummy_model").mkdir(parents=True)
        (repo / "results" / "scored").mkdir(parents=True)
        (repo / ".local" / "governance_anchor").mkdir(parents=True)

        (repo / ".gitignore").write_text(
            ".local/\n.venv-*/\nmodels/\ndata/splits/\nresults/\nruns/\nlogs/\nworktrees/\n__pycache__/\n",
            encoding="utf-8",
        )
        shutil.copy2(ROOT / "eval" / "governed_loop_guard.py", repo / "eval" / "governed_loop_guard.py")
        shutil.copy2(ROOT / "eval" / "strict_guard.py", repo / "eval" / "strict_guard.py")
        (repo / "code" / "helper_mod.py").write_text("VALUE = 'CLEAN_HEAD'\n", encoding="utf-8")
        (repo / "data" / "splits" / "sample.jsonl").write_text('{"id": "q1"}\n', encoding="utf-8")
        (repo / "models" / "dummy_model" / "weights.bin").write_bytes(b"IMMUTABLE_MODEL_BYTES")

        subprocess.run(["git", "init"], cwd=str(repo), check=True, capture_output=True)
        subprocess.run(["git", "config", "user.email", "test@local"], cwd=str(repo), check=True, capture_output=True)
        subprocess.run(["git", "config", "user.name", "Test"], cwd=str(repo), check=True, capture_output=True)
        subprocess.run(["git", "add", ".gitignore", "code/helper_mod.py", "eval/governed_loop_guard.py", "eval/strict_guard.py"], cwd=str(repo), check=True, capture_output=True)
        subprocess.run(["git", "commit", "-m", "initial"], cwd=str(repo), check=True, capture_output=True)

        anchor = repo / ".local" / "governance_anchor" / "anchor.json"
        subprocess.run(
            [
                sys.executable,
                "-I",
                "-B",
                str(repo / "eval" / "strict_guard.py"),
                "init-anchor",
                "--manifest",
                str(repo / "IMMUTABLE_MANIFEST.json"),
                "--root",
                str(repo),
                "--paths",
                "code/helper_mod.py",
                "eval/governed_loop_guard.py",
                "eval/strict_guard.py",
                "--anchor-file",
                str(anchor),
                "--protocol-epoch",
                "unit_test_epoch_v1",
            ],
            cwd=str(repo),
            check=True,
            capture_output=True,
        )
        subprocess.run(["git", "add", "IMMUTABLE_MANIFEST.json"], cwd=str(repo), check=True, capture_output=True)
        subprocess.run(["git", "commit", "-m", "lock manifest"], cwd=str(repo), check=True, capture_output=True)
        return repo, anchor

    def test_hermetic_worktree_sandbox_discards_failed_trial_shadow_and_cache_residue(self):
        sys.path.insert(0, str(ROOT / "code"))
        import plane2_governed_runner as p2r

        with tempfile.TemporaryDirectory() as tmp:
            repo, anchor = self._init_mini_governed_repo(Path(tmp))

            # Run 1: Failing trial mutates tracked code, drops untracked shadow conftest.py,
            # pollutes Triton/Inductor/vLLM/TMP caches, writes partial output, and fails!
            with self.assertRaises( subprocess.CalledProcessError ):
                with p2r.HermeticTrialSandbox(repo, "bad_trial_01", anchor) as sbx:
                    bad_script = (
                        "import os, pathlib\n"
                        "pathlib.Path('code/helper_mod.py').write_text('VALUE = \"POISONED\"\\n', encoding='utf-8')\n"
                        "pathlib.Path('conftest.py').write_text('raise SystemExit(99)\\n', encoding='utf-8')\n"
                        "for k in ('TRITON_CACHE_DIR', 'TORCHINDUCTOR_CACHE_DIR', 'VLLM_CACHE_ROOT', 'TMPDIR'):\n"
                        "    p = pathlib.Path(os.environ[k]) / 'poison.cache'\n"
                        "    p.parent.mkdir(parents=True, exist_ok=True)\n"
                        "    p.write_text('STALE_CACHE', encoding='utf-8')\n"
                        "out = pathlib.Path('results/training/bad_run')\n"
                        "out.mkdir(parents=True, exist_ok=True)\n"
                        "(out / 'partial.json').write_text('{\"corrupt\": true}', encoding='utf-8')\n"
                        "raise SystemExit(1)\n"
                    )
                    subprocess.run([sys.executable, "-c", bad_script], cwd=str(sbx.repo), env=sbx.env, check=True)
                    sbx.promote_declared_output("results/training/bad_run")

            # Live study root must remain 100% untouched, with external assets intact and zero dangling worktrees
            self.assertEqual((repo / "code" / "helper_mod.py").read_text(encoding="utf-8").strip(), "VALUE = 'CLEAN_HEAD'")
            self.assertFalse((repo / "conftest.py").exists())
            self.assertFalse((repo / "results" / "training" / "bad_run").exists())
            self.assertEqual((repo / "models" / "dummy_model" / "weights.bin").read_bytes(), b"IMMUTABLE_MODEL_BYTES")
            wt_list = subprocess.run(["git", "worktree", "list"], cwd=str(repo), check=True, capture_output=True, text=True).stdout
            self.assertEqual(len([ln for ln in wt_list.splitlines() if ln.strip()]), 1)

            # Run 2: Immediately subsequent healthy trial sees pristine HEAD code, read-only external data/models,
            # completely empty per-run caches, and promotes ONLY its declared output!
            with p2r.HermeticTrialSandbox(repo, "good_trial_02", anchor) as sbx2:
                good_script = (
                    "import json, os, pathlib, sys\n"
                    "sys.path.insert(0, 'code')\n"
                    "import helper_mod\n"
                    "assert helper_mod.VALUE == 'CLEAN_HEAD', helper_mod.VALUE\n"
                    "assert not pathlib.Path('conftest.py').exists()\n"
                    "assert pathlib.Path('models/dummy_model/weights.bin').read_bytes() == b'IMMUTABLE_MODEL_BYTES'\n"
                    "assert pathlib.Path('data/splits/sample.jsonl').is_file()\n"
                    "for k in ('TRITON_CACHE_DIR', 'TORCHINDUCTOR_CACHE_DIR', 'VLLM_CACHE_ROOT', 'TMPDIR'):\n"
                    "    assert not (pathlib.Path(os.environ[k]) / 'poison.cache').exists()\n"
                    "out = pathlib.Path('results/training/good_run')\n"
                    "out.mkdir(parents=True, exist_ok=True)\n"
                    "(out / 'summary.json').write_text(json.dumps({'status': 'COMPLETE', 'val': helper_mod.VALUE}), encoding='utf-8')\n"
                )
                subprocess.run([sys.executable, "-c", good_script], cwd=str(sbx2.repo), env=sbx2.env, check=True)
                sbx2.promote_declared_output("results/training/good_run")

            promoted = json.loads((repo / "results" / "training" / "good_run" / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(promoted, {"status": "COMPLETE", "val": "CLEAN_HEAD"})

    def test_worker_train_confirm_blocks_unauthorized_stale_rollback_or_mismatched_finalist_before_gpu(self):
        sys.path.insert(0, str(ROOT / "code"))
        import plane2_governed_runner as p2r

        with tempfile.TemporaryDirectory() as tmp:
            repo, anchor = self._init_mini_governed_repo(Path(tmp))
            gpu_launches = 0

            def _forbid_gpu(*_a, **_kw):
                nonlocal gpu_launches
                gpu_launches += 1
                raise AssertionError("GPU launch should NEVER be reached on unauthorized Confirm!")

            # 1. Missing Search adjudication & ledger -> blocked before any GPU launch
            ns_t2 = argparse.Namespace(finalist="T2", anchor_file=str(anchor))
            orig_root, orig_ddp = p2r.ROOT, p2r._run_one_ddp_train
            try:
                p2r.ROOT = repo
                p2r._run_one_ddp_train = _forbid_gpu
                with self.assertRaisesRegex(RuntimeError, "TRAIN_CONFIRM_UNAUTHORIZED"):
                    p2r.cmd_worker_train_confirm(ns_t2)
                self.assertEqual(gpu_launches, 0)

                # Create real Search scored files & real hash-chained Search ledger via strict_guard.py
                rules_search = {
                    "loop_id": "loop_train_systems_search_v2",
                    "primary_metric": "loop_seconds",
                    "primary_statistic": "mean",
                    "direction": "minimize",
                    "decision_stage": "search",
                    "alpha_one_sided": 0.05,
                    "multiplicity_k": 1,
                    "min_improvement": 10.0,
                    "aa_noise_se": 1.0,
                    "require_env_fingerprint_match": True,
                    "expected_item_ids": ["b0", "b1", "b2"],
                    "expected_cluster_map": {"b0": "b0", "b1": "b1", "b2": "b2"},
                    "min_independent_clusters": 3,
                    "metric_directions": {"loop_seconds": "minimize"},
                    "hard_ceilings": {"infrastructure_error_count": 0.0},
                }
                rules_path = repo / "configs" / "gate_rules_train_systems_search.json"
                rules_path.write_text(json.dumps(rules_search), encoding="utf-8")
                subprocess.run(["git", "add", "configs/gate_rules_train_systems_search.json"], cwd=str(repo), check=True, capture_output=True)
                subprocess.run(["git", "commit", "-m", "add search rules"], cwd=str(repo), check=True, capture_output=True)
                anchor.unlink()
                subprocess.run(
                    [
                        sys.executable, "-I", "-B", str(repo / "eval" / "strict_guard.py"), "init-anchor",
                        "--manifest", str(repo / "IMMUTABLE_MANIFEST.json"), "--root", str(repo),
                        "--paths", "code/helper_mod.py", "eval/governed_loop_guard.py", "eval/strict_guard.py", "configs/gate_rules_train_systems_search.json",
                        "--anchor-file", str(anchor), "--protocol-epoch", "unit_test_epoch_search",
                    ],
                    cwd=str(repo), check=True, capture_output=True,
                )
                subprocess.run(["git", "add", "IMMUTABLE_MANIFEST.json"], cwd=str(repo), check=True, capture_output=True)
                subprocess.run(["git", "commit", "-m", "relock search manifest"], cwd=str(repo), check=True, capture_output=True)

                def _mk_scored(cid: str, vals: list[float]) -> Path:
                    p = repo / "results" / "scored" / f"p2_train_search_{cid[:2]}.json"
                    p.write_text(
                        json.dumps(
                            {
                                "candidate_id": cid,
                                "protocol_id": "loop_train_systems_search_v2",
                                "env_fingerprint": "env_unit_test_train_v2",
                                "metrics": {"loop_seconds": sum(vals) / len(vals), "infrastructure_error_count": 0.0},
                                "items": [{"id": f"b{i}", "cluster_id": f"b{i}", "value": v} for i, v in enumerate(vals)],
                            }
                        ),
                        encoding="utf-8",
                    )
                    return p

                p_t0 = _mk_scored("T0_ddp8_micro1_nobucket", [170.0, 171.0, 169.0])
                # Case A: T1 passes (120s), T2 FAILS (168s -> improvement < 10s -> ROLLBACK!)
                p_t1 = _mk_scored("T1_ddp8_micro4_nobucket", [120.0, 121.0, 119.0])
                p_t2 = _mk_scored("T2_ddp8_micro4_length_bucket", [168.0, 167.0, 169.0])
                ledger = repo / "results" / "governed_ledger_v2.jsonl"

                d1_out = subprocess.run(
                    [
                        sys.executable, "-I", "-B", str(repo / "eval" / "strict_guard.py"), "gate-decision",
                        "--candidate", str(p_t1), "--incumbent", str(p_t0), "--baseline", str(p_t0),
                        "--rules", str(rules_path), "--ledger", str(ledger), "--anchor-file", str(anchor),
                        "--trial-id", "t1", "--hypothesis", "h1",
                    ],
                    cwd=str(repo), capture_output=True, text=True, check=False,
                )
                d2_out = subprocess.run(
                    [
                        sys.executable, "-I", "-B", str(repo / "eval" / "strict_guard.py"), "gate-decision",
                        "--candidate", str(p_t2), "--incumbent", str(p_t0), "--baseline", str(p_t0),
                        "--rules", str(rules_path), "--ledger", str(ledger), "--anchor-file", str(anchor),
                        "--trial-id", "t2", "--hypothesis", "h2",
                    ],
                    cwd=str(repo), capture_output=True, text=True, check=False,
                )
                d1 = json.loads(d1_out.stdout)
                d2 = json.loads(d2_out.stdout)
                self.assertEqual(d1["decision"], "PROVISIONAL_KEEP")
                self.assertEqual(d2["decision"], "ROLLBACK")

                anch_data = json.loads(anchor.read_text(encoding="utf-8"))
                adj_path = repo / "results" / "plane2_train_search_adjudication.json"
                adj_payload = {
                    "status": "TRAIN_SEARCH_ADJUDICATED",
                    "protocol_epoch": anch_data["protocol_epoch"],
                    "manifest_root_sha256": anch_data["manifest_root_sha256"],
                    "ledger_head_sha256_after_search": anch_data["ledger_head_sha256"],
                    "ledger_entry_count_after_search": anch_data["ledger_entry_count"],
                    "selected_finalist_arm": "T1",
                    "selected_finalist_candidate_id": "T1_ddp8_micro4_nobucket",
                    "decisions": {"T1": d1, "T2": d2},
                }
                adj_path.write_text(json.dumps(adj_payload), encoding="utf-8")

                # 2. Requesting --finalist T2 (which got ROLLBACK while T1 was selected) -> MUST fail closed with 0 GPU launches!
                with self.assertRaisesRegex(RuntimeError, "TRAIN_CONFIRM_UNAUTHORIZED"):
                    p2r.cmd_worker_train_confirm(argparse.Namespace(finalist="T2", anchor_file=str(anchor)))
                self.assertEqual(gpu_launches, 0)

                # 3. Stale manifest_root_sha256 in adjudication -> MUST fail closed with 0 GPU launches!
                stale_adj = dict(adj_payload, manifest_root_sha256="0" * 64)
                adj_path.write_text(json.dumps(stale_adj), encoding="utf-8")
                with self.assertRaisesRegex(RuntimeError, "TRAIN_CONFIRM_UNAUTHORIZED"):
                    p2r.verify_train_confirm_authorization("T1", root=repo, anchor_path=anchor)
                self.assertEqual(gpu_launches, 0)

                # 4. Restoring exact valid adjudication + live verified ledger for T1 -> AUTHORIZED!
                adj_path.write_text(json.dumps(adj_payload), encoding="utf-8")
                auth_ok = p2r.verify_train_confirm_authorization("T1", root=repo, anchor_path=anchor)
                self.assertEqual(auth_ok["authorized_finalist_arm"], "T1")
                self.assertEqual(gpu_launches, 0)
            finally:
                p2r.ROOT = orig_root
                p2r._run_one_ddp_train = orig_ddp

    def test_single_touch_holdout_requires_verified_live_s3_promote_ledger_and_rejects_string_false_or_stale(self):
        sys.path.insert(0, str(ROOT / "code"))
        import plane2_governed_runner as p2r

        with tempfile.TemporaryDirectory() as tmp:
            tbase = Path(tmp)
            repo, anchor = self._init_mini_governed_repo(tbase)
            vault = tbase / "external_vault"
            vault.mkdir(parents=True)
            (vault / "math_holdout512_labels.json").write_text('{"q1": "42"}\n', encoding="utf-8")

            holdout_gpu_calls = 0

            def _forbid_holdout_gpu(*_a, **_kw):
                nonlocal holdout_gpu_calls
                holdout_gpu_calls += 1
                raise AssertionError("Holdout GPU generation MUST NOT execute without verified live S3 PROMOTE ledger!")

            orig_root, orig_panel = p2r.ROOT, p2r.run_sharded_panel_inference
            try:
                p2r.ROOT = repo
                p2r.run_sharded_panel_inference = _forbid_holdout_gpu
                adj_sum_path = repo / "results" / "plane2_adjudication_summary.json"
                ns = argparse.Namespace(anchor_file=str(anchor), vault_dir=str(vault))

                # 1. Mutable summary claims True with NO ledger -> MUST fail closed with 0 GPU calls & 0 registry touches!
                adj_sum_path.write_text(json.dumps({"serving_s3_promoted": True}), encoding="utf-8")
                with self.assertRaisesRegex(RuntimeError, "HOLDOUT_UNAUTHORIZED"):
                    p2r.cmd_single_touch_holdout(ns)
                self.assertEqual(holdout_gpu_calls, 0)
                self.assertFalse((vault / "holdout_touch_registry.json").exists())

                # 2. Malformed string 'false' (truthy in Python bool('false')!) -> MUST fail closed with 0 GPU calls!
                adj_sum_path.write_text(json.dumps({"serving_s3_promoted": "false"}), encoding="utf-8")
                with self.assertRaisesRegex(RuntimeError, "HOLDOUT_UNAUTHORIZED"):
                    p2r.cmd_single_touch_holdout(ns)
                self.assertEqual(holdout_gpu_calls, 0)

                # Build a real Confirm ledger via strict_guard.py gate-decision (first ROLLBACK, then PROMOTE in 2nd repo)
                s_rules = {
                    "loop_id": "loop_dense_serving_confirm_v2",
                    "primary_metric": "assigned_gpu_seconds",
                    "primary_statistic": "mean",
                    "direction": "minimize",
                    "decision_stage": "confirm",
                    "alpha_one_sided": 0.025,
                    "multiplicity_k": 1,
                    "min_improvement": 310.0,
                    "aa_noise_se": 5.0,
                    "require_env_fingerprint_match": True,
                    "expected_item_ids": [f"b{i}" for i in range(5)],
                    "expected_cluster_map": {f"b{i}": f"b{i}" for i in range(5)},
                    "min_independent_clusters": 5,
                    "metric_directions": {"assigned_gpu_seconds": "minimize"},
                    "hard_ceilings": {"infrastructure_error_count": 0.0},
                }
                s_rules_path = repo / "configs" / "gate_rules_dense_serving_confirm.json"
                s_rules_path.write_text(json.dumps(s_rules), encoding="utf-8")
                # Re-pin manifest so configs/gate_rules_dense_serving_confirm.json is locked in IMMUTABLE_MANIFEST.json
                subprocess.run(["git", "add", "configs/gate_rules_dense_serving_confirm.json"], cwd=str(repo), check=True, capture_output=True)
                subprocess.run(["git", "commit", "-m", "add serving confirm rules"], cwd=str(repo), check=True, capture_output=True)
                anchor.unlink()
                subprocess.run(
                    [
                        sys.executable, "-I", "-B", str(repo / "eval" / "strict_guard.py"), "init-anchor",
                        "--manifest", str(repo / "IMMUTABLE_MANIFEST.json"), "--root", str(repo),
                        "--paths", "code/helper_mod.py", "eval/governed_loop_guard.py", "eval/strict_guard.py", "configs/gate_rules_dense_serving_confirm.json",
                        "--anchor-file", str(anchor), "--protocol-epoch", "unit_test_epoch_v2",
                    ],
                    cwd=str(repo), check=True, capture_output=True,
                )
                subprocess.run(["git", "add", "IMMUTABLE_MANIFEST.json"], cwd=str(repo), check=True, capture_output=True)
                subprocess.run(["git", "commit", "-m", "relock manifest"], cwd=str(repo), check=True, capture_output=True)

                def _mk_srv(cid: str, vals: list[float], fname: str) -> Path:
                    p = repo / "results" / "scored" / fname
                    p.write_text(
                        json.dumps(
                            {
                                "candidate_id": cid,
                                "protocol_id": "loop_dense_serving_confirm_v2",
                                "env_fingerprint": "env_unit_test_serving_v2",
                                "metrics": {"assigned_gpu_seconds": sum(vals) / len(vals), "infrastructure_error_count": 0.0},
                                "items": [{"id": f"b{i}", "cluster_id": f"b{i}", "value": v} for i, v in enumerate(vals)],
                            }
                        ),
                        encoding="utf-8",
                    )
                    return p

                p_s0 = _mk_srv("S0_bf16_tp2", [560.0] * 5, "s0.json")
                p_s3_bad = _mk_srv("S3_int4_w4a16_tp1_mtp3_repaired", [400.0] * 5, "s3_bad.json")  # gain=160 < 310 -> ROLLBACK!
                ledger = repo / "results" / "governed_ledger_v2.jsonl"
                d4_rb_proc = subprocess.run(
                    [
                        sys.executable, "-I", "-B", str(repo / "eval" / "strict_guard.py"), "gate-decision",
                        "--candidate", str(p_s3_bad), "--incumbent", str(p_s0), "--baseline", str(p_s0),
                        "--rules", str(s_rules_path), "--ledger", str(ledger), "--anchor-file", str(anchor),
                        "--trial-id", "trial_04_serving_confirm_s3_int4_tp1_mtp3", "--hypothesis", "s3",
                    ],
                    cwd=str(repo), capture_output=True, text=True, check=False,
                )
                d4_rb = json.loads(d4_rb_proc.stdout)
                self.assertEqual(d4_rb["decision"], "ROLLBACK")
                anch_rb = json.loads(anchor.read_text(encoding="utf-8"))

                # 3. Summary falsely claims serving_s3_promoted=True when verified ledger has ROLLBACK -> MUST fail closed!
                adj_sum_path.write_text(
                    json.dumps(
                        {
                            "status": "LEDGER_ADJUDICATED_AND_VERIFIED",
                            "protocol_epoch": anch_rb["protocol_epoch"],
                            "manifest_root_sha256": anch_rb["manifest_root_sha256"],
                            "ledger_head_sha256_after_confirm": anch_rb["ledger_head_sha256"],
                            "serving_s3_promoted": True,
                            "serving_confirm_entry_sha256": d4_rb["entry_sha256"],
                        }
                    ),
                    encoding="utf-8",
                )
                with self.assertRaisesRegex(RuntimeError, "HOLDOUT_UNAUTHORIZED"):
                    p2r.cmd_single_touch_holdout(ns)
                self.assertEqual(holdout_gpu_calls, 0)

                # 4. Honest False matching verified ROLLBACK ledger -> skips cleanly with 0 GPU calls and writes untouched receipt!
                adj_sum_path.write_text(
                    json.dumps(
                        {
                            "status": "LEDGER_ADJUDICATED_AND_VERIFIED",
                            "protocol_epoch": anch_rb["protocol_epoch"],
                            "manifest_root_sha256": anch_rb["manifest_root_sha256"],
                            "ledger_head_sha256_after_confirm": anch_rb["ledger_head_sha256"],
                            "serving_s3_promoted": False,
                            "serving_confirm_entry_sha256": d4_rb["entry_sha256"],
                        }
                    ),
                    encoding="utf-8",
                )
                rc_skip = p2r.cmd_single_touch_holdout(ns)
                self.assertEqual(rc_skip, 0)
                self.assertEqual(holdout_gpu_calls, 0)
                skip_rec = json.loads((repo / "results" / "single_touch_holdout_v2.json").read_text(encoding="utf-8"))
                self.assertEqual(skip_rec["status"], "SKIPPED_NO_SERVING_PROMOTION")
            finally:
                p2r.ROOT = orig_root
                p2r.run_sharded_panel_inference = orig_panel

    def test_williams_2x2_factorial_synergy_is_true_did_not_average_main_effect(self):
        sys.path.insert(0, str(ROOT / "code"))
        import plane2_governed_runner as p2r

        def _mk_blocks(s0: float, s1: float, s2: float, s3: float) -> list[dict]:
            return [
                {
                    "block_index": i,
                    "arms": {
                        "S0": {"assigned_gpu_seconds": s0},
                        "S1": {"assigned_gpu_seconds": s1},
                        "S2": {"assigned_gpu_seconds": s2},
                        "S3": {"assigned_gpu_seconds": s3},
                    },
                }
                for i in range(4)
            ]

        # Additive costs S0=100, S1=80, S2=70, S3=50 -> exact DiD synergy = (70-50) - (100-80) = 0.0!
        add_eff = p2r.compute_williams_2x2_factorial_effects(_mk_blocks(100.0, 80.0, 70.0, 50.0))
        self.assertAlmostEqual(add_eff["beneficial_synergy_did_S1_plus_S2_minus_S0_minus_S3"]["mean"], 0.0, places=12)

        # Super-additive synergy S3=45 -> +5.0 extra GPU-second reduction
        syn_eff = p2r.compute_williams_2x2_factorial_effects(_mk_blocks(100.0, 80.0, 70.0, 45.0))
        self.assertAlmostEqual(syn_eff["beneficial_synergy_did_S1_plus_S2_minus_S0_minus_S3"]["mean"], 5.0, places=12)

        # Sub-additive antagonism S3=60 -> -10.0 GPU-seconds
        ant_eff = p2r.compute_williams_2x2_factorial_effects(_mk_blocks(100.0, 80.0, 70.0, 60.0))
        self.assertAlmostEqual(ant_eff["beneficial_synergy_did_S1_plus_S2_minus_S0_minus_S3"]["mean"], -10.0, places=12)

    def test_retire_owned_vllm_failure_is_never_silently_swallowed(self):
        sys.path.insert(0, str(ROOT / "code"))
        import plane2_governed_runner as p2r

        with tempfile.TemporaryDirectory() as tmp:
            wt = Path(tmp)
            (wt / "code").mkdir(parents=True)
            # Fake owned_server.py whose retire() raises RuntimeError (unresolved server!)
            (wt / "code" / "owned_server.py").write_text(
                "def retire(name, reason):\n    raise RuntimeError(f'UNRESOLVED_SERVER:{name}')\n"
                "def wait_idle(devices, timeout=120):\n    pass\n",
                encoding="utf-8",
            )
            env = p2r.build_isolated_trial_env(wt / "caches")
            with self.assertRaises(RuntimeError):
                p2r._retire_owned_vllm_in_sandbox(wt, env, "stuck_srv", "cleanup", devices="0", vpy=sys.executable)


if __name__ == "__main__":
    unittest.main()
