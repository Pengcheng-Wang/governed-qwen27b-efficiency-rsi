# `program.md` — Governed Optimization Loop Specification (`AILab_Qwen27B_Governed_v2`)

## 1. Objective, Tier & Go/No-Go Admission Audit

- **Problem Statement:** Optimize training-loop and inference-serving resource efficiency on `Qwen/Qwen3.8-27B` (`revision 1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`, native `Qwen3_5ForConditionalGeneration` wrapper with `15` native MTP tensors) **subject to strictly preserved optimization/reasoning fidelity**, while evaluating Step-0 admission preconditions across all four candidate loops (`loop_train_systems_confirm_v2`, `loop_dense_serving_confirm_v2`, `loop_dense_restem_dapo_v2`, and `loop_moe_qwen35_35b_a3b_v2`).
- **Loop Archetype:** `D: Two-Plane Hybrid` (Stage 1 unselected Plane-1 runtime/verifier/hermetic-noise calibration -> Stage 2 SHA-256 boundary freeze & external byte-receipt pin -> Stage 3 Plane-2 governed Systems loop execution). Runtime policy and weights are never mutated together.
- **Loop Scale Tier:** `Tier 1` finite pre-registered Systems evaluation (`<= 3` ledgered Search/Confirm trials total on the sole Step-0 `GO` loop `loop_train_systems_v2`; zero open-ended code drift).
- **Admitted `GO` Loop & Post-B4 Hermetic `se_AA` Noise Floor:**
  - **`loop_train_systems_v2` (`GO` — Sole Admitted Plane-2 Loop):** Minimize pure-training `loop_seconds` (stopped prior to any checkpoint/export I/O) on fixed-exposure `data/splits/sft_systems1024_v2.jsonl` (`1,024` rows, `340,797` input tokens, `254,724` supervised tokens, `16` updates on 8x A100 DDP `node-1` inside per-run `HermeticTrialSandbox`). Hermetic baseline `T0` (`micro=1, bucket=False`): `mean_loop_seconds = 308.1175s`, **`se_AA = 1.1704s`** (5 paired `A1/A2` blocks under symmetric fresh per-run Triton/Inductor/CUDA caches; `80%`-power `MDE = 3.2789s`), `min_improvement (eps) = 25.0s`, `delta_plan = 55.0s`, `K = 1`, `alpha_one_sided = 0.025`, plug-in conditional healthy complete-predicate planning estimate = **`0.9996 >= 0.80`**.
- **Hard Admission Blockers (`B1–B5`) & Explicit Step-0 `NO_GO` Dispositions (3 of 4 Candidate Loops Blocked Before Plane-2):**
  - [x] **B1 Asymmetric Deterministic Verifier (SFT NLL & Repaired MATH Symbolic Verifier):** Fixed-token held-out `after_nll` on `sft_heldout64_v2.jsonl` plus `FP32` merge `<= 1 BF16 ULP` & bitwise 15-tensor native MTP oracle (`code/train_systems_v2.py` SHA `c05f70dd...`, `12/12` CPU controls). Symbolic MATH verifier (`eval/verifiers_v2.py` SHA `a3aa8c7127f830c0d57257075c29422f942e3453e0608ea72225f64a89880ce0`) qualified on remote Linux (`Python 3.12.3`, `math-verify 0.9.0`, `SymPy 1.14.0`, unmodified default production timeouts): `20/20` unit tests pass, `40`-case historical `v1` replay preserves `20/20` passes and repairs `10/20` notation/unit/set false negatives with `0` execution errors (`results/plane1_linux_verifier_qualification_v2.json`).
  - [x] **B2 Disjoint 3-Way Splits & Multi-Shape Workloads:** Pairwise disjoint `Search`, `Confirmation`, and `Protected Holdout` panels (`0` exact or 13-gram overlaps) and `0` historical overlaps across all 6 Confirm/Protected panels against the authenticated `780`-member `v1` archive (`a1ce1c5a...`, parent receipt `2a6abeb1...`).
  - [x] **B3 Immutable Out-of-Process Scorer:** Complete protected surface frozen in `IMMUTABLE_MANIFEST.json`; scoring runs out-of-process via `python -I -B eval/isolated_scorer.py` inside a fresh detached worktree (`11/11` parent CLI completeness controls passed, receipt `9310747a...`).
  - [x] **B4 Hermetic Detached Scratch-Worktree & Per-Run Cache Isolation:** Every trial executes inside `HermeticTrialSandbox` (`git worktree add --detach` + `git clean -fdx` under `/tmp/q27wt_<nonce>`, plus isolated per-run `TRITON_CACHE_DIR`, `TORCHINDUCTOR_CACHE_DIR`, `VLLM_CACHE_ROOT`, `CUDA_CACHE_PATH`, `XDG_CACHE_HOME`, `HF_HOME`, `TMPDIR`, `PYTHONPYCACHEPREFIX`), promoting only declared `results/...` outputs on `rc == 0` and deleting worktree + caches in `finally:` (`21/21` parent boundary/auth controls passed, receipt `ea20dcad...`).
  - [x] **B5 Optimization Fidelity Containment:** Training Confirm requires worst-block relative `after_nll <= 1.005x`, `peak_allocated_gib <= 76.0`, and `infrastructure_error_count == 0`.
  - **Mandatory Step-0 `NO_GO` Dispositions (Unexecuted in Plane-2; Confirm & Protected Holdout Panels Untouched at `0` Touches):**
    1. **`loop_dense_serving_confirm_v2` -> Step-0 `NO_GO` (`S3_UNDERPOWERED_UNDER_OBSERVED_MATCHED_SEED_EMITTED_WORK_VARIANCE`):** Under `HermeticTrialSandbox` (`results/plane1_serving_aa_hermetic_v2.json`), `S0` (`BF16 TP=2`) measured `mean_dataset_seconds = 250.5675s` and **`se_AA = 113.5915` GPU-s** (`wall SE = 56.7958s`). While hardware decode speed was invariant (`tpot_p95_ms = 22.60 +/- 0.046 ms`), `127/160` matched `(ID, prompt_hash, seed)` pairs exhibited observed generated-text divergence (`vLLM 0.29.0` default non-reproducibility and continuous-batching numerical/scheduling effects are plausible context rather than a uniquely established cause; e.g., Block 0 `max_tokens` `12,249 -> 23,547`, shifting `dataset_seconds` `294.12s -> 536.06s`). Under preserved unrelaxed gates (`min_improvement = 310.0` GPU-s, `delta_plan = 350.0` GPU-s, `K = 1`, `alpha = 0.025`, `2.0 pp` NI), Analytic Normal primary speed power is **`0.637633 < 0.80`** (`configs/step0_serving_confirm_nogo.json` enforces `NO_GO`). Confirm/Holdout panels (`math_confirm1280.jsonl`, `math_holdout512_prompts.jsonl`) remain untouched (`0` touches); `S3` (`1-GPU GPTQ W4A16 + Repaired Native BF16 MTP3`, `18.7 GiB`, rescored Search128 `96.88%` vs `S0` `95.31%`, SLO-correct goodput `0.272057` vs `0.152911` req/s) is reported strictly as a Search/canary engineering finding, not a confirmed promotion.
    2. **`loop_dense_restem_dapo_v2` -> Step-0 `NO_GO`:** Fails `B1` (`DAPO-Math-17k` published gold contradictions `dapo:b57078e1...`, `dapo:e09849b4...` and integer-unit false negatives), `Step 2` (`36/768 = 4.69%` length truncations, right-censored `p99 = 32,768.0`; `5,580,085` of `5,784,296` rollout tokens discarded after batch alignment), and `S3` (`dapo_confirm200` $\cap$ `retention_confirm150` `< 25%` power for `1–2 pp` shifts).
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

1. **Exact-Policy Hermetic `A/A` Noise Calibration:** Both `plane1_train_aa_hermetic_v2.json` (`node-1`, `se_AA = 1.1704s`, `mean = 308.1175s`, `peak_allocated_gib = 55.1398 GiB` leaving `20.8602 GiB` headroom below the `76.0 GiB` ceiling; under fresh per-run caches update 1 averages `123.53s` vs `11.06s` for run-median updates `2–16`, without claiming timings alone isolate compile time) and `plane1_serving_aa_hermetic_v2.json` (`node-2`, `se_AA = 113.5915` GPU-s, exposing `127/160` matched-prompt/seed emitted-work divergence and triggering Step-0 `NO_GO` on serving Confirm) were measured under the **exact `HermeticTrialSandbox` detached-worktree and fresh-ephemeral-cache lifecycle** used in Plane-2.
2. **Pure-Loop Timer Separation:** `train_systems_v2.py` stops `loop_seconds` immediately after `torch.cuda.synchronize` on update 16 prior to held-out `after_nll` evaluation and any checkpoint/export I/O, while also recording full `process_seconds` and stage `runs/*/status.json` separately for honest multi-scope cost accounting.

---

## 6. Two-Stage Promotion Gate, Plug-In Planning Power & Fail-Closed Tripwires

1. **Search Screen (`loop_train_systems_search_v2`):** Evaluates `T1` vs `T0` (`trial_01`) and `T2` vs `T0` (`trial_02`) independently via live `strict_guard.py gate-decision` (`min_improvement = 10.0s`, `max_ratio_vs_baseline.after_nll <= 1.005`, `peak_allocated_gib <= 76.0`, `infrastructure_error_count == 0`). `select_train_search_finalist()` selects the fastest `PROVISIONAL_KEEP` arm (`K = 1`), returning `None` (`HALTED_NO_ELIGIBLE_SEARCH_FINALIST`) if neither qualifies.
2. **Pre-GPU Boundary & Confirm Gate (`loop_train_systems_confirm_v2`, `K = 1`, `alpha = 0.025`):** `worker-train-confirm` enforces `verify_train_confirm_authorization()` against live `IMMUTABLE_MANIFEST.json`, `anchor.json`, and `governed_ledger_v2.jsonl` before launching any GPU run. Promotion (`PROMOTE`) requires 10-block `loop_seconds` reduction `>= 25.0s`, one-sided 97.5% cluster-bootstrap & `se_AA`-floored `LCB > 0`, worst-block `after_nll / base_after_nll <= 1.005`, `peak_allocated_gib <= 76.0`, and `infrastructure_error_count == 0`.
3. **Plug-In Planning Power Disclosure:** Under explicit planning assumptions of healthy 8-GPU DDP launches (`p_launch_err = 0`) and calibrated `+0.0010` BF16 associativity NLL shift, plug-in planning estimates (distinct from guaranteed finite-sample probability bounds) give primary speed power = `1.0000`, 10-block NLL + memory guardrail power = `0.9996`, and conditional complete-predicate Fréchet planning estimate = **`0.9996`** (`0.9977` at `p_launch_err = 1e-4`; `0.9798` at `p_launch_err = 1e-3`).

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
