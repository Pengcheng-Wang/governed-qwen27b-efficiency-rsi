# Experiment Learnings, Forensic Synthesis & Practitioner Playbook (`v1` -> `v2`)

This document provides a self-contained technical synthesis of our two-generation study on **`Qwen/Qwen3.8-27B`** (`revision 1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`, `Qwen3_5ForConditionalGeneration`, `18` shards, `15` native MTP tensors) and **`Qwen/Qwen3.5-35B-A3B`** (`revision 59d61f3ce65a6d9863b86d2e96597125219dc754`, `Qwen3_5MoeForConditionalGeneration`, `14` shards, `785` MTP tensors): how **`/governed-loop-optimization`** exposed and fixed subtle failure modes from `v1`, what worked, what halted honestly, and reusable engineering recipes directly backed by the code in this repository.

---

## 1. Baseline Forensics: What `v1` Found vs. What Was Missing

In `AILab_Qwen27B_Efficiency_v1`, five empirical threads were explored without formal pre-registration, `A/A` noise floors, or hermetic worktree/cache resets:

1. **Dense SFT Loop Speedup (`1.576x` claimed):** `micro=4 + length balancing` beat `micro=1` across 3 fixed-exposure epochs (`7.8% -> 12.1%` partial matmul MFU), **but** bundled two factors (`micro_batch: 1 -> 4` AND `length_bucketing: False -> True`) in warm-cache runs without isolating padding waste vs kernel-launch amortization.
2. **Dense Serving Speedups (`1.642x` MTP3; `1.498x` latency / `2.996x` GPU-h for GPTQ W4A16 TP=1 vs BF16 TP=2):** Useful practical wins, **but**:
   - Combining **1-GPU `GPTQ W4A16` + Native `MTP3` failed** in `vLLM`'s stock loader and was abandoned.
   - Timing blocks ran sequentially (`BBBAAA`, `n=3`) with no `se_AA` measurement.
3. **Dense 50-Update Fresh-Base GRPO/TIS (`0` net `MATH-500` gain at `97.6%`):** `77.5%` (`155/200`) prompt groups produced uniform rewards (`0` gradient signal on `23/50` updates while Adam momentum continued drifting weights), burning `5.50M` completion tokens on an already ceiling-saturated benchmark.
4. **MoE (`Qwen3.5-35B-A3B`) Attention/GDN-Only LoRA (`8.13M` FP32 params):** Achieved `2.002x` loop speedup and `0.145%` held-out NLL agreement, **yet regressed `MATH-500` (`94.0% -> 93.2%`) and failed completion gates (`5.2% -> 6.2%` length/parse failures)**. Forensics revealed two root causes:
   - *Violation of Outer-Harness-Before-Inner-Weights:* Untuned base `Qwen3.5-35B-A3B` **already** violated the `5.0%` truncation gate at `32,768` tokens (`5.2%`) prior to training.
   - *Short-Trace Selection Bias & NLL Proxy Conflict:* `v1` filtered out all teacher traces `>4,096` tokens (`260/693 = 37.5%` dropped) while keeping up to `4` repeats on easy short problems, proving held-out NLL agreement (`0.145%`) is **not** a sufficient proxy for long-horizon reasoning completion.
   - Cross-study synthesis with read-only MI300X `AILab_MultiLingal_v1` (`Qwen3.8-27B` LoRA-SFT) showed the same hazard on general capabilities: unanchored SFT left thinking math flat (`93.6 -> 93.7` MGSM) while inducing collateral regressions on non-thinking benchmarks (`-4.8` GMMLU, `-5.2` IFEval, `-10.9` mArena winrate).

---

## 2. How `/governed-loop-optimization` Reshaped the Experiment (`Archetype D`)

Applying `/governed-loop-optimization` enforced five structural upgrades before a single Plane-2 trial was allowed to run:

### 2.1 Complete Historical Quarantine (`B2` — `code/prepare_splits_v2.py`)
- Reconstructed the exact prompt-hash union of all `780` archived `v1` evaluation and training files (`MATH-500`, `AIME`, `dev200`, `sft_systems1024`, `dapo256`, `mmlu_pro`).
- Built pairwise disjoint `Search`, `Confirmation` (`math_confirm1280.jsonl`, `N=1,280`), and sealed prompt-only `Protected Holdout` (`math_holdout512_prompts.jsonl`, `dapo_holdout200_prompts.jsonl`, `retention_holdout150_prompts.jsonl`) panels with **`0` cross-split exact/13-gram overlaps and `0` historical `v1` overlaps** (`results/split_audit_v2.json`).

### 2.2 Asymmetric, Set-Safe & Percent-Safe Symbolic Verifier (`B1` — `eval/verifiers_v2.py`)
- Audited `20` `v1` baseline passes + `20` `v1` baseline failures (`sources/research_baseline_verifier_semantic_audit.json`).
- Discovered two subtle `math-verify 0.9.0` traps and fixed both cleanly:
  1. **Tuple/Interval vs. Finite-Set Trap:** Bare `"81, 256"` parses in `math-verify` as the finite set `{81, 256}`, whereas rewriting to parenthesized `"(81, 256)"` parses as the continuous **open interval** $(81, 256)$ — breaking reversed solution lists (`"256, 81"`) and risking interval false positives! `eval/verifiers_v2.py` preserves explicit finite-set semantics (`tests/test_verifier_semantic_sets.py`).
  2. **Percent String-Fallback Trap:** Single `number\%` answers triggered `math-verify 0.9.0` string fallback against decimal gold targets (`0.25` vs `25\%`); explicit numeric-percent canonicalization resolved it without loosening multi-part expressions.
- Outcome on Linux (`Python 3.12.3`, default production timeouts): **`20/20` unit tests pass, `20/20` historical passes preserved, `10/20` historical false negatives recovered, `0` false positives on genuinely wrong completions**.

### 2.3 Out-of-Process Per-Block Fail-Closed Scoring & KL-Hoeffding Paired NI (`B3` — `eval/isolated_scorer.py`, `eval/strict_guard.py`)
- `isolated_scorer.py systems_abab` rejects any missing/extra/duplicate block ID, any missing or defaulted `infrastructure_errors` field (forbidding silent `missing -> 0` defaults), and any partial per-block guardrail coverage (`11/11` adversarial CLI controls verified).
- Paired binary quality non-inferiority (`paired_quality_ni_margin = 0.020` at `N = 1,280`, one-sided `alpha = 0.025`) uses conservative problem-clustered **KL-Hoeffding union bounds** (`pos_lcb - neg_ucb >= -0.020`), valid even at zero or sparse discordance where naive i.i.d. bootstrap collapses to `[0, 0]`.
- Pre-GPU (`verify_train_confirm_authorization`) and pre-holdout (`verify_single_touch_holdout_authorization`) gates cryptographically verify `IMMUTABLE_MANIFEST.json`, `anchor.json`, and hash-chained `governed_ledger_v2.jsonl` *before* any GPU process launches or any sealed label is read (`21/21` boundary controls pass in `tests/test_strict_guard.py`).

### 2.4 True Hermetic Worktree & Compiler-Cache Isolation (`B4` — `HermeticTrialSandbox`)
- Rather than trusting `git diff` in a shared directory with warm Triton/Inductor/vLLM caches, every single DDP run and vLLM server/client stage executes inside `HermeticTrialSandbox` (`code/plane2_governed_runner.py`):
  - Spawns `git worktree add --detach` + `git clean -fdx` under `/tmp/q27wt_<nonce>` (explicitly rooted at `/tmp` on POSIX so `vLLM V1` ZeroMQ `ipc://` `AF_UNIX` socket paths stay well under Linux's `107`-byte `sockaddr_un.sun_path` limit).
  - Isolates `TRITON_CACHE_DIR`, `TORCHINDUCTOR_CACHE_DIR`, `VLLM_CACHE_ROOT`, `CUDA_CACHE_PATH`, `XDG_CACHE_HOME`, `HF_HOME`, `TMPDIR`, and `PYTHONPYCACHEPREFIX` per run.
  - Promotes only declared `results/...` outputs on clean `rc == 0`, archives server retirement receipts (`runs/<srv>/{status.json,retired.json}`) fail-closed (`OWNED_SERVER_RETIRE_FAILED` on non-zero drain), and destroys worktrees + caches in `finally:`.

---

## 3. Deep-Dive Into the Four Experimental Tracks

### Track 1: 1-GPU `GPTQ W4A16 + Native BF16 MTP3` (`S3`) & Continuous-Batching Variance Discovery
- **Engineering breakthrough (`code/quantize_v2.py`):** Stock `vLLM 0.29` now serves `Qwen/Qwen3.8-27B` on a **single 80GB A100** (`18.7 GiB` weights) with `3`-token native BF16 Multi-Token Prediction (`62.7%` draft token acceptance). On `math_search128`, `S3` delivered **`3.16x` lower participant GPU-hours** than 2-GPU `BF16 TP=2` (`0.1266` vs `0.3997` GPU-h), **`1.58x` faster dataset wall time** (`455.79s` vs `719.37s`), **`1.78x` higher SLO-correct goodput** (`0.2721` vs `0.1529 req/s`), and **`96.88%` vs `95.31%` lenient accuracy** (`2` wins, `0` losses, `126` ties).
- **Methodological discovery (`results/plane1_serving_aa_hermetic_v2.json`):** Why did `S0` vs `S0` hermetic `A/A` have `se_AA = 113.59` GPU-s (`wall SE = 56.80s`) when hardware `tpot_p95_ms` was `22.60 +/- 0.046 ms` (`0.2%` jitter)?
  - Inspecting all `320` generations across `160` matched `(ID, prompt_hash, seed)` pairs revealed **`127/160` pairs diverged in generated text**. Once tiny floating-point batch-reduction differences alter a single sampled token at `temperature = 1.0`, reasoning chains branch onto shorter or longer derivations (e.g., Block-0 max output tokens `12,249` vs `23,547`, shifting 32-item block wall time `294.12s -> 536.06s`).
  - Under unrelaxed Confirm gates (`min_improvement = 310` GPU-s, `delta_plan = 350` GPU-s, `K = 1`, `alpha = 0.025`), primary Normal power dropped to **`63.76% < 80%`** (`complete_power_lb = 46.11%`). Honoring Step-0 `NO_GO` preserved statistical integrity without falsely claiming `S3` failed.

### Track 2: Decomposing Microbatching (`T1`) vs. Length Bucketing (`T2`) on 8-GPU DDP (`loop_train_systems_v2`)
- **Trainer precision & memory hygiene (`code/train_systems_v2.py`):**
  - HuggingFace `from_pretrained()` loads models in `.eval()` mode by default; calling `model.train()` explicitly *before* wrapping in `DDP` plus Liger fused cross-entropy (`skip_logits=True`, avoiding `7.6 GiB` full-vocabulary logits materialization) and `ddp_model.no_sync()` across microsteps kept 8-GPU DDP `Qwen3.8-27B` at **`55.14 GiB`** (`20.86 GiB` below the `76.0 GiB` ceiling).
  - Gradient normalization divides strictly by global supervised token count across all 8 ranks so `T0`, `T1`, and `T2` execute the **exact same mathematical optimization objective** (`144/144` optimizer-group hashes reconciled across all 9 Search runs; held-out `after_nll` `0.63065` vs `0.62919` vs `0.62931`, well inside `<= 1.005x`).
- **Why `v2`'s 3-arm Latin-Square Search improved on `v1`'s bundled result:**
  - **`T1` (`micro=4, length_bucketing=False`) regressed (`316.58s` vs `309.33s`, `+7.25s`, `ROLLBACK`):** Grouping 4 random-length reasoning traces padded total input tokens from `340,797` to **`585,964` (`+71.94%`)**, burning `+10.28%` more board energy (`460.59 kJ` vs `417.66 kJ`) and `+5.28 GiB` memory.
  - **`T2` (`micro=4, length_bucketing=True`) improved point speed (`299.91s` vs `309.33s`, `-9.43s`) but missed the `10.0s` Search floor (`ROLLBACK`):** Sorting sequences by length within each 64-example global batch cut padded tokens to `535,180` (`+57.04%`) and reduced descriptive updates `2–16` run-median step duration from **`11.24s` (`T0`) to `6.00s` (`T2`)**, while incurring longer cold Step-1/shape initialization (`146.57s` vs `123.36s`), `+3.31%` higher total board energy (`431.47 kJ`), and `60.90 GiB` peak memory.

### Tracks 3 & 4: Pre-Flight Firewalls on Weight-Space RSI (`DAPO-Math-17k` & `Qwen3.5-35B-A3B`)
- **Dense `DAPO-Math-17k` (`results/plane1_dense_frontier_audit.json`):** Auditing `768` frontier rollouts (`16` draws x `48` prompts on `node-0`) revealed `36/768 = 4.69%` right-censored `32,768`-token truncations (`p99 = 32,768.0`), two contradictory published gold answers, and `5,580,085` discarded tokens out of `5,784,296` generated (`96.5%` waste), blocking an underpowered RL loop before burning 24-GPU days.
- **MoE `Qwen3.5-35B-A3B` (`results/plane1_moe_cap_calibration.json`, `results/plane1_moe_cap98k_calibration.json`, `sources/research_model_sampling_policy_review.json`):** Expanding `max_tokens` `32,768 -> 49,152 -> 98,304` improved untuned accuracy **`91.41% -> 94.53% -> 95.31%`**, proving `uncensored p95 = 35,738` exceeds `32K`. Remaining `3/128` (`2.34%`) `98K` non-terminations traced directly to `presence_penalty = 0.0` vs the pinned model card's `presence_penalty = 1.5` general-thinking recommendation.

---

## 4. Actionable Recommendations for Future Governed Loops

1. **Powered Confirmation for Open-Generation Serving:**
   - Enable `vLLM` batch-invariant kernels (where supported for `CC >= 8.0` and target quantization/MTP ops) **or** split serving confirmation into (a) **fixed-trajectory prefill/decode replay** for low-noise systems latency/GPU-hour gating (`se_AA` drops by `>10x`) plus (b) **paired `N >= 1,280` open-generation accuracy/stop-rate KL-Hoeffding NI**.
2. **Sequence Packing + Warm-Cache Multi-Epoch Horizon for Hybrid GDN Training:**
   - Replace right-padded microbatches (`+57%` to `+72%` token waste) with **ragged/packed varlen attention & Gated-DeltaNet state-reset kernels** (`0%` padding), and separate **explicit pre-timer compiler warmup** (or persistent validated compiler caches across `>= 100` updates) from steady-state step timing.
3. **True `2x2` Factorial Attribution (`compute_williams_2x2_factorial_effects`):**
   - Always compute interaction synergy as the unscaled difference-in-differences `(S2 - S3) - (S0 - S1) = S1 + S2 - S0 - S3` (`0` under purely additive savings), never `.5 * ((S0 - S3) - (S1 - S2))` (which algebraically collapses to the average main effect of Factor B!).
