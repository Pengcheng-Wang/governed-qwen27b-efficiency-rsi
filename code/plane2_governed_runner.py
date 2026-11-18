"""
plane2_governed_runner.py — Frozen, manifest-verified Plane-2 execution, hermetic scratch-worktree
sandbox, and single-touch holdout runner for AILab_Qwen27B_Governed_v2 across nodes 0, 1, and 2.

Hermetic B4 & Live-Ledger Invariants:
  - Pure Python stdlib at module import time: zero repo helper modules execute before `IMMUTABLE_MANIFEST.json`
    is cryptographically verified against external `.local/governance_anchor/anchor.json`.
  - Every trial and server stage executes inside `HermeticTrialSandbox`: a fresh detached Git scratch worktree
    (`git worktree add --detach` + `git clean -fdx`) paired with fresh empty per-run cache directories
    (`TRITON_CACHE_DIR`, `TORCHINDUCTOR_CACHE_DIR`, `VLLM_CACHE_ROOT`, `CUDA_CACHE_PATH`, `XDG_CACHE_HOME`,
    `HF_HOME`, `TMPDIR`, `PYTHONPYCACHEPREFIX`, `PYTHONDONTWRITEBYTECODE=1`, `PYTHONNOUSERSITE=1`).
  - Only explicitly declared validated `results/...` outputs are promoted upon `rc == 0`; failed runs unwind
    in `finally:`, removing both the detached worktree and per-run caches while leaving live `.venv-*`,
    `models/`, `.local/`, and external vault directories untouched.
  - `worker-train-confirm` calls `verify_train_confirm_authorization(...)` BEFORE any GPU launch, verifying
    live `IMMUTABLE_MANIFEST.json`, hash-chained `results/governed_ledger_v2.jsonl`, `anchor.json`, and
    `results/plane2_train_search_adjudication.json` so missing/stale/`ROLLBACK`/mismatched requests launch
    zero training processes.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import os
import random
import re
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any

os.environ["PYTHONDONTWRITEBYTECODE"] = "1"

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_VAULT = Path("/scratch/qwen27b_gov_v2_sealed_holdout_vault_c4e9b1")

WILLIAMS_4X4: list[list[str]] = [
    ["S0", "S1", "S3", "S2"],
    ["S1", "S2", "S0", "S3"],
    ["S2", "S3", "S1", "S0"],
    ["S3", "S0", "S2", "S1"],
]

TRAIN_ARMS: dict[str, dict[str, Any]] = {
    "T0": {"candidate_id": "T0_ddp8_micro1_nobucket", "micro_batch": 1, "length_bucketing": False},
    "T1": {"candidate_id": "T1_ddp8_micro4_nobucket", "micro_batch": 4, "length_bucketing": False},
    "T2": {"candidate_id": "T2_ddp8_micro4_length_bucket", "micro_batch": 4, "length_bucketing": True},
}

SERVING_ARMS: dict[str, dict[str, Any]] = {
    "S0": {
        "candidate_id": "S0_bf16_tp2",
        "model_dir": "models/dense27_base",
        "tp": 2,
        "mtp_tokens": 0,
        "williams_gpus": "0,1",
        "williams_port": 18280,
    },
    "S1": {
        "candidate_id": "S1_bf16_tp2_mtp3_search_diag",
        "model_dir": "models/dense27_base",
        "tp": 2,
        "mtp_tokens": 3,
        "williams_gpus": "2,3",
        "williams_port": 18281,
    },
    "S2": {
        "candidate_id": "S2_int4_w4a16_tp1_search_diag",
        "model_dir": "models/gptq_w4a16_mtp_ready_v2",
        "tp": 1,
        "mtp_tokens": 0,
        "williams_gpus": "4",
        "williams_port": 18282,
    },
    "S3": {
        "candidate_id": "S3_int4_w4a16_tp1_mtp3_repaired",
        "model_dir": "models/gptq_w4a16_mtp_ready_v2",
        "tp": 1,
        "mtp_tokens": 3,
        "williams_gpus": "5",
        "williams_port": 18283,
    },
}


def _default_anchor(root: Path | None = None) -> Path:
    base = root if root is not None else ROOT
    return base / ".local" / "governance_anchor" / "anchor.json"


def _venv_py(root: Path | None = None) -> str:
    base = root if root is not None else ROOT
    cand = base / ".venv-training" / "bin" / "python"
    return str(cand) if cand.exists() else sys.executable


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}.{uuid.uuid4().hex[:6]}")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    tmp.replace(path)


def build_isolated_trial_env(cache_root: Path, extra_env: dict[str, str] | None = None) -> dict[str, str]:
    """Construct a hermetic per-trial environment with isolated JIT/compiler/runtime cache roots."""
    sub_dirs = {
        "pycache": cache_root / "pycache",
        "triton": cache_root / "triton",
        "inductor": cache_root / "inductor",
        "compile_debug": cache_root / "compile_debug",
        "nv_compute": cache_root / "nv_compute",
        "vllm": cache_root / "vllm",
        "xdg": cache_root / "xdg",
        "hf": cache_root / "hf",
        "tmp": cache_root / "tmp",
    }
    for d in sub_dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    env = dict(os.environ)
    env.update(
        {
            "PYTHONNOUSERSITE": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONPYCACHEPREFIX": str(sub_dirs["pycache"]),
            "TRITON_CACHE_DIR": str(sub_dirs["triton"]),
            "TORCHINDUCTOR_CACHE_DIR": str(sub_dirs["inductor"]),
            "TORCH_COMPILE_DEBUG_DIR": str(sub_dirs["compile_debug"]),
            "CUDA_CACHE_PATH": str(sub_dirs["nv_compute"]),
            "VLLM_CACHE_ROOT": str(sub_dirs["vllm"]),
            "XDG_CACHE_HOME": str(sub_dirs["xdg"]),
            "HF_HOME": str(sub_dirs["hf"]),
            "TMPDIR": str(sub_dirs["tmp"]),
            "TEMP": str(sub_dirs["tmp"]),
            "TMP": str(sub_dirs["tmp"]),
        }
    )
    if extra_env:
        env.update(extra_env)
    return env


def enforce_hermetic_pre_trial_hygiene(
    anchor_path: Path | None = None,
    *,
    root: Path | None = None,
    verify_manifest: bool = True,
) -> dict[str, Any]:
    """
    Verifies clean live study repository state and immutable manifest root BEFORE spawning a detached worktree:
      1. Purges any `__pycache__` directories under `code/`, `eval/`, and `tests/`.
      2. Verifies zero tracked Git file diffs (`git diff --exit-code` and `--cached`) and forbids any
         untracked shadow files outside `results/`.
      3. When `verify_manifest=True`, verifies `IMMUTABLE_MANIFEST.json` out-of-process (`python -I -B`)
         against the externally pinned `manifest_root_sha256` in `anchor_path`.
    """
    repo_root = (root if root is not None else ROOT).resolve(strict=True)
    anch = (anchor_path if anchor_path is not None else _default_anchor(repo_root)).resolve()

    for sub_name in ("code", "eval", "tests"):
        sub_dir = repo_root / sub_name
        if sub_dir.is_dir():
            for pyc_dir in list(sub_dir.rglob("__pycache__")):
                if pyc_dir.is_dir():
                    shutil.rmtree(pyc_dir)

    for diff_cmd in (["git", "diff", "--exit-code"], ["git", "diff", "--cached", "--exit-code"]):
        gp = subprocess.run(diff_cmd, cwd=str(repo_root), capture_output=True, text=True, check=False)
        if gp.returncode != 0:
            raise RuntimeError(f"Dirty tracked Git tree forbidden before Plane-2 step: {gp.stdout[:1000]}")

    untracked_proc = subprocess.run(
        ["git", "ls-files", "--others", "--exclude-standard"],
        cwd=str(repo_root),
        capture_output=True,
        text=True,
        check=True,
    )
    bad_untracked = [
        ln.strip()
        for ln in untracked_proc.stdout.splitlines()
        if ln.strip() and not ln.strip().replace("\\", "/").startswith("results/")
    ]
    if bad_untracked:
        raise RuntimeError(f"Untracked shadow files outside results/ forbidden: {bad_untracked}")

    if not verify_manifest:
        return {"status": "CLEAN_GIT_PRE_FREEZE_CALIBRATION"}

    if not anch.is_file():
        raise RuntimeError(f"Governance anchor file missing: {anch}")
    anchor = json.loads(anch.read_text(encoding="utf-8"))
    proc = subprocess.run(
        [
            sys.executable,
            "-I",
            "-B",
            str(repo_root / "eval" / "governed_loop_guard.py"),
            "verify-manifest",
            "--manifest",
            str(repo_root / "IMMUTABLE_MANIFEST.json"),
            "--root",
            str(repo_root),
            "--expected-root-sha256",
            str(anchor["manifest_root_sha256"]),
        ],
        cwd=str(repo_root),
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"IMMUTABLE_MANIFEST verification failed before Plane-2 step:\n{proc.stdout}\n{proc.stderr}")
    return json.loads(proc.stdout)


def _link_external_readonly_dir(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.symlink(str(src), str(dst), target_is_directory=True)
    except OSError:
        if os.name == "nt":
            subprocess.run(["cmd", "/c", "mklink", "/J", str(dst), str(src)], check=True, capture_output=True, text=True)
        else:
            raise


def _unlink_external_dir(dst: Path) -> None:
    if not dst.exists() and not dst.is_symlink():
        return
    if dst.is_symlink():
        dst.unlink()
    elif os.name == "nt":
        os.rmdir(dst)
    else:
        dst.unlink()


class HermeticTrialSandbox:
    """
    B4 Enforced Ephemeral Detached Scratch-Worktree & Isolated Cache Sandbox:
      - Materializes a brand-new detached `git worktree` checked out cleanly from `HEAD` (`git clean -fdx`)
        under `<root>/worktrees/wt_<tag>_<pid>_<nonce>/repo`.
      - Links ONLY immutable external read-only asset directories (`models`, gitignored `data/splits`, or
        active server `runs` status descriptors when explicitly requested), never `.local/`, `.venv-*`,
        `results/`, or shared caches.
      - Allocates fresh empty per-trial compiler/JIT/runtime caches (`TRITON_CACHE_DIR`,
        `TORCHINDUCTOR_CACHE_DIR`, `VLLM_CACHE_ROOT`, `CUDA_CACHE_PATH`, `XDG_CACHE_HOME`, `HF_HOME`,
        `TMPDIR`, `PYTHONPYCACHEPREFIX`).
      - Promotes ONLY explicitly declared `results/...` subdirectories upon caller validation, and
        unconditionally destroys both the detached Git worktree and per-trial caches in `__exit__`.
    """

    def __init__(
        self,
        root: Path,
        trial_tag: str,
        anchor_path: Path | None = None,
        *,
        verify_manifest: bool = True,
        link_runs_from: Path | None = None,
    ) -> None:
        self.root = root.resolve(strict=True)
        safe_tag = re.sub(r"[^A-Za-z0-9_.-]+", "_", trial_tag)[:48]
        self.anchor_path = (anchor_path if anchor_path is not None else _default_anchor(self.root)).resolve()
        self.verify_manifest = verify_manifest
        self.link_runs_from = link_runs_from.resolve(strict=True) if link_runs_from is not None else None
        nonce = uuid.uuid4().hex[:8]
        # Use literal /tmp on POSIX (ignoring stage.py's longer TMPDIR override) so vLLM ZMQ ipc:// paths stay << 107 bytes
        scratch_parent = Path("/tmp") if os.name != "nt" else Path(tempfile.gettempdir())
        self.base_dir = scratch_parent / f"q27wt_{nonce}"
        self.repo = self.base_dir / "r"
        self.caches = self.base_dir / "c"
        self.trial_tag = safe_tag
        self.env: dict[str, str] = {}
        self._linked_dirs: list[Path] = []

    def __enter__(self) -> HermeticTrialSandbox:
        enforce_hermetic_pre_trial_hygiene(
            self.anchor_path,
            root=self.root,
            verify_manifest=self.verify_manifest,
        )
        self.base_dir.mkdir(parents=True, exist_ok=False)
        subprocess.run(
            ["git", "worktree", "add", "--detach", str(self.repo), "HEAD"],
            cwd=str(self.root),
            check=True,
            capture_output=True,
            text=True,
        )
        subprocess.run(
            ["git", "clean", "-fdx"],
            cwd=str(self.repo),
            check=True,
            capture_output=True,
            text=True,
        )
        for rel_ext in ("models", "data/splits", "data/traces"):
            src = self.root / rel_ext
            dst = self.repo / rel_ext
            if src.is_dir() and not dst.exists():
                _link_external_readonly_dir(src, dst)
                self._linked_dirs.append(dst)
        if self.link_runs_from is not None:
            dst_runs = self.repo / "runs"
            if not dst_runs.exists():
                _link_external_readonly_dir(self.link_runs_from, dst_runs)
                self._linked_dirs.append(dst_runs)
        self.env = build_isolated_trial_env(self.caches)
        return self

    def promote_declared_output(self, rel_subpath: str) -> Path:
        norm = rel_subpath.replace("\\", "/").strip("/")
        if not norm.startswith("results/"):
            raise RuntimeError(f"HermeticTrialSandbox may only promote declared results/... paths, got {rel_subpath!r}")
        src = (self.repo / norm).resolve(strict=True)
        wt_results = (self.repo / "results").resolve()
        if wt_results not in src.parents:
            raise RuntimeError(f"Declared output escapes sandbox results tree: {src}")
        dst = self.root / norm
        if dst.exists():
            raise RuntimeError(f"Refusing to overwrite existing promoted output in live study root: {dst}")
        dst.parent.mkdir(parents=True, exist_ok=True)
        if src.is_dir():
            shutil.copytree(src, dst)
        else:
            shutil.copy2(src, dst)
        return dst

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> bool:
        for linked in reversed(self._linked_dirs):
            try:
                _unlink_external_dir(linked)
            except Exception:
                pass
        subprocess.run(
            ["git", "worktree", "remove", "--force", str(self.repo)],
            cwd=str(self.root),
            check=False,
            capture_output=True,
            text=True,
        )
        if self.base_dir.exists():
            shutil.rmtree(self.base_dir, ignore_errors=True)
        if self.caches.exists():
            shutil.rmtree(self.caches, ignore_errors=True)
        subprocess.run(
            ["git", "worktree", "prune"],
            cwd=str(self.root),
            check=False,
            capture_output=True,
            text=True,
        )
        return False


def select_train_search_finalist(
    gate_decisions: dict[str, dict[str, Any]],
    scored_payloads: dict[str, dict[str, Any]],
) -> str | None:
    """
    Registered finite independent-vs-baseline Search finalist selector:
      - Admits ONLY candidates whose actual `strict_guard.py gate-decision` verdict vs baseline `T0`
        is `PROVISIONAL_KEEP` (never `ROLLBACK` or errored runs).
      - Selects the single fastest admitted finalist (`K=1` for Confirm) by lowest `loop_seconds`
        (tie-breaking deterministically by arm name).
      - Returns `None` if zero Search candidates qualify (no arbitrary `T2` fallback!).
    """
    admitted: list[tuple[float, str]] = []
    for arm_k in ("T1", "T2"):
        verdict = gate_decisions.get(arm_k, {}).get("decision")
        if verdict == "PROVISIONAL_KEEP":
            loop_sec = float(scored_payloads[arm_k]["metrics"]["loop_seconds"])
            admitted.append((loop_sec, arm_k))
    if not admitted:
        return None
    admitted.sort()
    return admitted[0][1]


def verify_train_confirm_authorization(
    finalist_arm: str,
    *,
    root: Path | None = None,
    anchor_path: Path | None = None,
) -> dict[str, Any]:
    """
    Fail-closed pre-GPU authorization check for `worker-train-confirm`:
      1. Verifies live `IMMUTABLE_MANIFEST.json` against `anchor_path`.
      2. Verifies hash-chained `results/governed_ledger_v2.jsonl` via `governed_loop_guard.py verify-ledger`
         against `anchor["ledger_head_sha256"]`.
      3. Verifies `results/plane2_train_search_adjudication.json` binds to the exact current
         `manifest_root_sha256`, `protocol_epoch`, and `ledger_head_sha256`.
      4. Reads the verified Search ledger entries directly, ensures zero prior Confirm entries exist,
         recomputes `select_train_search_finalist(...)` from the verified ledger verdicts and scored Search
         payloads, and requires `finalist_arm == expected_arm` with `decision == "PROVISIONAL_KEEP"`.
      Any violation raises `RuntimeError("TRAIN_CONFIRM_UNAUTHORIZED: ...")` BEFORE launching any GPU run.
    """
    repo_root = (root if root is not None else ROOT).resolve(strict=True)
    anch_file = (anchor_path if anchor_path is not None else _default_anchor(repo_root)).resolve()

    if finalist_arm not in ("T1", "T2"):
        raise RuntimeError(f"TRAIN_CONFIRM_UNAUTHORIZED: invalid finalist arm {finalist_arm!r}")

    if not anch_file.is_file():
        raise RuntimeError(f"TRAIN_CONFIRM_UNAUTHORIZED: missing governance anchor {anch_file}")

    try:
        enforce_hermetic_pre_trial_hygiene(anch_file, root=repo_root, verify_manifest=True)
    except Exception as exc:
        raise RuntimeError(f"TRAIN_CONFIRM_UNAUTHORIZED: pre-trial manifest/tree check failed: {exc}") from exc

    anchor = json.loads(anch_file.read_text(encoding="utf-8"))
    ledger_path = repo_root / "results" / "governed_ledger_v2.jsonl"
    adj_path = repo_root / "results" / "plane2_train_search_adjudication.json"
    scored_dir = repo_root / "results" / "scored"
    req_files = [
        ledger_path,
        adj_path,
        scored_dir / "p2_train_search_T0.json",
        scored_dir / "p2_train_search_T1.json",
        scored_dir / "p2_train_search_T2.json",
    ]
    missing = [str(p) for p in req_files if not p.is_file()]
    if missing:
        raise RuntimeError(f"TRAIN_CONFIRM_UNAUTHORIZED: missing required Search governance artifacts: {missing}")

    v_proc = subprocess.run(
        [
            sys.executable,
            "-I",
            "-B",
            str(repo_root / "eval" / "governed_loop_guard.py"),
            "verify-ledger",
            "--ledger",
            str(ledger_path),
            "--expected-head-sha256",
            str(anchor.get("ledger_head_sha256", "")),
        ],
        cwd=str(repo_root),
        capture_output=True,
        text=True,
        check=False,
    )
    if v_proc.returncode != 0:
        raise RuntimeError(f"TRAIN_CONFIRM_UNAUTHORIZED: ledger verification failed: {v_proc.stdout} {v_proc.stderr}")
    v_led = json.loads(v_proc.stdout)
    if int(v_led.get("entries", 0)) != int(anchor.get("ledger_entry_count", -1)) or int(v_led.get("entries", 0)) < 2:
        raise RuntimeError(f"TRAIN_CONFIRM_UNAUTHORIZED: invalid ledger entry count {v_led}")

    adj = json.loads(adj_path.read_text(encoding="utf-8"))
    if adj.get("status") != "TRAIN_SEARCH_ADJUDICATED":
        raise RuntimeError(f"TRAIN_CONFIRM_UNAUTHORIZED: Search status is {adj.get('status')!r}, not TRAIN_SEARCH_ADJUDICATED")
    if adj.get("manifest_root_sha256") != anchor.get("manifest_root_sha256"):
        raise RuntimeError("TRAIN_CONFIRM_UNAUTHORIZED: adjudication manifest_root_sha256 does not match active anchor")
    if adj.get("protocol_epoch") != anchor.get("protocol_epoch"):
        raise RuntimeError("TRAIN_CONFIRM_UNAUTHORIZED: adjudication protocol_epoch does not match active anchor")
    if adj.get("ledger_head_sha256_after_search") != anchor.get("ledger_head_sha256"):
        raise RuntimeError("TRAIN_CONFIRM_UNAUTHORIZED: adjudication ledger head does not match active anchor")

    ledger_rows = _read_jsonl(ledger_path)
    if any(r.get("protocol_id") == "loop_train_systems_confirm_v2" for r in ledger_rows):
        raise RuntimeError("TRAIN_CONFIRM_UNAUTHORIZED: single-look Training Confirm entry already exists in ledger")

    cid_to_arm = {TRAIN_ARMS[k]["candidate_id"]: k for k in ("T1", "T2")}
    search_by_arm: dict[str, dict[str, Any]] = {}
    for row in ledger_rows:
        if row.get("protocol_id") == "loop_train_systems_search_v2":
            cid = str(row.get("candidate_id", ""))
            if cid not in cid_to_arm:
                raise RuntimeError(f"TRAIN_CONFIRM_UNAUTHORIZED: unexpected Search candidate_id {cid!r}")
            arm_k = cid_to_arm[cid]
            if arm_k in search_by_arm:
                raise RuntimeError(f"TRAIN_CONFIRM_UNAUTHORIZED: duplicate Search ledger entry for {arm_k}")
            search_by_arm[arm_k] = row

    if set(search_by_arm.keys()) != {"T1", "T2"}:
        raise RuntimeError(f"TRAIN_CONFIRM_UNAUTHORIZED: incomplete Search ledger arms {sorted(search_by_arm.keys())}")

    for arm_k in ("T1", "T2"):
        adj_dec = adj.get("decisions", {}).get(arm_k, {})
        if adj_dec.get("entry_sha256") != search_by_arm[arm_k].get("entry_sha256"):
            raise RuntimeError(f"TRAIN_CONFIRM_UNAUTHORIZED: entry_sha256 mismatch between ledger and adjudication on {arm_k}")

    scored_payloads = {
        k: json.loads((scored_dir / f"p2_train_search_{k}.json").read_text(encoding="utf-8"))
        for k in ("T1", "T2")
    }
    for arm_k in ("T1", "T2"):
        led_mean = float(search_by_arm[arm_k]["candidate_primary"])
        sc_mean = float(scored_payloads[arm_k]["metrics"]["loop_seconds"])
        if not math.isclose(led_mean, sc_mean, rel_tol=0.0, abs_tol=1e-9):
            raise RuntimeError(f"TRAIN_CONFIRM_UNAUTHORIZED: scored Search metric mismatch vs ledger on {arm_k}")

    expected_arm = select_train_search_finalist(search_by_arm, scored_payloads)
    if expected_arm is None:
        raise RuntimeError("TRAIN_CONFIRM_UNAUTHORIZED: zero Search arms achieved PROVISIONAL_KEEP")
    if adj.get("selected_finalist_arm") != expected_arm:
        raise RuntimeError(
            f"TRAIN_CONFIRM_UNAUTHORIZED: adjudication selected {adj.get('selected_finalist_arm')!r} != ledger expected {expected_arm!r}"
        )
    if finalist_arm != expected_arm or search_by_arm[finalist_arm].get("decision") != "PROVISIONAL_KEEP":
        raise RuntimeError(
            f"TRAIN_CONFIRM_UNAUTHORIZED: requested finalist {finalist_arm!r} != ledger-admitted finalist {expected_arm!r}"
        )

    return {
        "authorized_finalist_arm": expected_arm,
        "authorized_candidate_id": TRAIN_ARMS[expected_arm]["candidate_id"],
        "manifest_root_sha256": anchor["manifest_root_sha256"],
        "ledger_head_sha256": anchor["ledger_head_sha256"],
        "search_entry_sha256": search_by_arm[expected_arm]["entry_sha256"],
    }


def _launch_owned_vllm_in_sandbox(
    wt_repo: Path,
    env: dict[str, str],
    stage_name: str,
    model_dir: str,
    devices: str,
    tp: int,
    port: int,
    *,
    mtp_tokens: int = 0,
    max_model_len: int = 40960,
    max_num_seqs: int = 16,
    max_batched_tokens: int = 8192,
    gpu_mem_util: float = 0.90,
    ttl: int = 21600,
    vpy: str | None = None,
) -> str:
    py_bin = vpy or _venv_py(ROOT)
    iso_keys = (
        "PYTHONNOUSERSITE",
        "PYTHONDONTWRITEBYTECODE",
        "PYTHONPYCACHEPREFIX",
        "TRITON_CACHE_DIR",
        "TORCHINDUCTOR_CACHE_DIR",
        "TORCH_COMPILE_DEBUG_DIR",
        "CUDA_CACHE_PATH",
        "VLLM_CACHE_ROOT",
        "XDG_CACHE_HOME",
        "HF_HOME",
        "TMPDIR",
    )
    env_assigns = [f"CUDA_VISIBLE_DEVICES={devices}"] + [f"{k}={env[k]}" for k in iso_keys if k in env]
    cmd = [
        "env",
        *env_assigns,
        py_bin,
        "-B",
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
        "--no-enable-prefix-caching",
        "--seed",
        "20261117",
    ]
    if mtp_tokens > 0:
        cmd.extend(["--speculative-config", json.dumps({"method": "mtp", "num_speculative_tokens": mtp_tokens})])
    spec = json.dumps({"devices": devices, "name": stage_name, "cmd": cmd, "tp": tp, "port": port, "ttl": ttl})
    helper = (
        "import json, sys\n"
        "sys.path.insert(0, 'code')\n"
        "import owned_server\n"
        "d = json.loads(sys.argv[1])\n"
        "owned_server.wait_idle(d['devices'])\n"
        "owned_server.start(d['name'], d['cmd'], int(d['tp']), int(d['port']), ttl=int(d['ttl']))\n"
    )
    proc = subprocess.run(
        [py_bin, "-B", "-c", helper, spec],
        cwd=str(wt_repo),
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=1020,
    )
    if proc.returncode != 0:
        sbx_stage = wt_repo / "runs" / stage_name
        out_tail = ""
        if sbx_stage.is_dir():
            dst_stage = ROOT / "runs" / stage_name
            dst_stage.mkdir(parents=True, exist_ok=True)
            for fname in ("status.json", "retired.json", "output.log"):
                sf = sbx_stage / fname
                if sf.is_file():
                    shutil.copy2(sf, dst_stage / fname)
            if (sbx_stage / "output.log").is_file():
                out_tail = (sbx_stage / "output.log").read_text(encoding="utf-8", errors="replace")[-3000:]
        raise RuntimeError(
            f"OWNED_SERVER_LAUNCH_FAILED ({stage_name}, devices={devices}): rc={proc.returncode}\n"
            f"stdout={proc.stdout[-1000:]}\nstderr={proc.stderr[-1000:]}\nserver_output_tail={out_tail}"
        )
    return stage_name


def _retire_owned_vllm_in_sandbox(
    wt_repo: Path,
    env: dict[str, str],
    stage_name: str,
    reason: str,
    *,
    devices: str | None = None,
    vpy: str | None = None,
) -> None:
    """
    Fail-closed server retirement inside a sandbox worktree:
      - Invokes `owned_server.retire(stage_name, reason)` and, when `devices` is provided,
        `owned_server.wait_idle(devices, timeout=120)`.
      - NEVER silently swallows retirement or device-drain failures (`check=False` -> explicit
        `RuntimeError("OWNED_SERVER_RETIRE_FAILED: ...")` on non-zero exit).
    """
    py_bin = vpy or _venv_py(ROOT)
    spec = json.dumps({"name": stage_name, "reason": reason, "devices": devices})
    helper = (
        "import json, sys\n"
        "sys.path.insert(0, 'code')\n"
        "import owned_server\n"
        "d = json.loads(sys.argv[1])\n"
        "owned_server.retire(d['name'], d['reason'])\n"
        "if d.get('devices'):\n"
        "    owned_server.wait_idle(str(d['devices']), timeout=120)\n"
    )
    proc = subprocess.run(
        [py_bin, "-B", "-c", helper, spec],
        cwd=str(wt_repo),
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=240,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"OWNED_SERVER_RETIRE_FAILED ({stage_name}, devices={devices}): rc={proc.returncode}\n"
            f"stdout={proc.stdout[-1000:]}\nstderr={proc.stderr[-1000:]}"
        )


def _retire_all_launched_servers_fail_closed(
    wt_repo: Path,
    servers: list[tuple[str, dict[str, str], str]],
    reason: str,
    *,
    live_root: Path | None = None,
    vpy: str | None = None,
    active_exc: BaseException | None = None,
) -> None:
    """
    Attempt retirement + GPU idle drain on every launched server, archive `runs/<srv_name>/{status.json,retired.json}`
    into ignored `<live_root>/runs/<srv_name>` for complete stage accounting before scratch-worktree deletion,
    and raise loudly if any server remains unresolved.
    """
    archive_base = ((live_root if live_root is not None else ROOT) / "runs").resolve()
    errors: list[str] = []
    for srv_name, srv_env, devs in reversed(servers):
        try:
            _retire_owned_vllm_in_sandbox(wt_repo, srv_env, srv_name, reason, devices=devs, vpy=vpy)
        except Exception as exc:
            errors.append(f"{srv_name}[gpus={devs}]: {exc}")
        finally:
            sbx_stage = wt_repo / "runs" / srv_name
            if sbx_stage.is_dir():
                dst_stage = archive_base / srv_name
                dst_stage.mkdir(parents=True, exist_ok=True)
                for fname in ("status.json", "retired.json"):
                    sf = sbx_stage / fname
                    if sf.is_file():
                        shutil.copy2(sf, dst_stage / fname)
    if errors:
        msg = "OWNED_SERVER_RETIRE_FAILED: " + " | ".join(errors)
        if active_exc is not None:
            if hasattr(active_exc, "add_note"):
                active_exc.add_note(msg)
            return
        raise RuntimeError(msg)


def run_isolated_systems_scorer(
    *,
    blocks_json: Path,
    arm: str,
    block_metric: str,
    candidate_id: str,
    protocol_id: str,
    output_path: Path,
    root: Path | None = None,
    anchor_path: Path | None = None,
    verify_manifest: bool = True,
) -> dict[str, Any]:
    repo_root = (root if root is not None else ROOT).resolve(strict=True)
    with HermeticTrialSandbox(
        repo_root,
        f"score_{protocol_id}_{arm}",
        anchor_path=anchor_path,
        verify_manifest=verify_manifest,
    ) as sbx:
        subprocess.run(
            [
                sys.executable,
                "-I",
                "-B",
                str(sbx.repo / "eval" / "isolated_scorer.py"),
                "systems_abab",
                "--blocks-json",
                str(blocks_json),
                "--arm",
                arm,
                "--block-metric",
                block_metric,
                "--statistic",
                "mean",
                "--candidate-id",
                candidate_id,
                "--protocol-id",
                protocol_id,
                "--output",
                str(output_path),
            ],
            cwd=str(sbx.repo),
            env=sbx.env,
            check=True,
            timeout=180,
        )
    return json.loads(output_path.read_text(encoding="utf-8"))


def run_sharded_panel_inference(
    *,
    run_prefix: str,
    dataset_path: Path,
    arm_key: str,
    purpose: str,
    env_fingerprint: str,
    scored_output: Path,
    anchor_path: Path | None = None,
    task: str = "math",
    seed: int = 20261117,
    concurrency_per_replica: int = 8,
    external_labels: Path | None = None,
    holdout_registry: Path | None = None,
) -> dict[str, Any]:
    """Run deterministic data-parallel replicas inside a fresh detached Git worktree & isolated per-replica caches."""
    arm = SERVING_ARMS[arm_key]
    tp = int(arm["tp"])
    replica_devices = ["0,1", "2,3", "4,5", "6,7"] if tp == 2 else [str(i) for i in range(8)]
    n_rep = len(replica_devices)

    merged_live_dir = ROOT / "results" / "inference" / run_prefix
    if merged_live_dir.exists():
        raise RuntimeError(f"Inference output directory already exists: {merged_live_dir}")

    all_rows = _read_jsonl(dataset_path)
    py_bin = _venv_py(ROOT)

    with HermeticTrialSandbox(ROOT, f"panel_{run_prefix}", anchor_path=anchor_path, verify_manifest=True) as sbx:
        tmp_shard_dir = sbx.caches / f"shards_{run_prefix}"
        tmp_shard_dir.mkdir(parents=True, exist_ok=True)
        launched_srvs: list[tuple[str, dict[str, str], str]] = []
        procs: list[tuple[int, subprocess.Popen[bytes], Path, Any]] = []

        t0 = time.monotonic()
        body_exc: BaseException | None = None
        try:
            for r_idx, devs in enumerate(replica_devices):
                shard_rows = [row for idx, row in enumerate(all_rows) if (idx % n_rep) == r_idx]
                if not shard_rows:
                    continue
                shard_ds = tmp_shard_dir / f"shard_{r_idx}.jsonl"
                _write_jsonl(shard_ds, shard_rows)
                srv_name = f"{run_prefix}_srv_r{r_idx}"
                port = 18310 + r_idx
                srv_env = build_isolated_trial_env(sbx.caches / f"srv_r{r_idx}")
                _launch_owned_vllm_in_sandbox(
                    sbx.repo,
                    srv_env,
                    srv_name,
                    arm["model_dir"],
                    devs,
                    tp,
                    port,
                    mtp_tokens=int(arm["mtp_tokens"]),
                    max_model_len=40960,
                    max_num_seqs=16,
                    max_batched_tokens=8192,
                    gpu_mem_util=0.90,
                    vpy=py_bin,
                )
                launched_srvs.append((srv_name, srv_env, devs))

            for r_idx, devs in enumerate(replica_devices):
                shard_ds = tmp_shard_dir / f"shard_{r_idx}.jsonl"
                if not shard_ds.is_file():
                    continue
                srv_name = f"{run_prefix}_srv_r{r_idx}"
                port = 18310 + r_idx
                shard_run_name = f"{run_prefix}_shard_r{r_idx}"
                log_path = tmp_shard_dir / f"infer_r{r_idx}.log"
                cli_env = build_isolated_trial_env(sbx.caches / f"cli_r{r_idx}")
                cmd = [
                    py_bin,
                    "-B",
                    str(sbx.repo / "code" / "infer_v2.py"),
                    "--name",
                    shard_run_name,
                    "--server-stage",
                    srv_name,
                    "--base-url",
                    f"http://127.0.0.1:{port}",
                    "--model-name",
                    "qwen_gov_v2",
                    "--dataset",
                    str(shard_ds),
                    "--model-config",
                    str(sbx.repo / "configs" / "model.json"),
                    "--tokenizer-dir",
                    str(sbx.repo / "models" / "dense27_base"),
                    "--task",
                    task,
                    "--cap",
                    "32768",
                    "--effort",
                    "medium",
                    "--concurrency",
                    str(concurrency_per_replica),
                    "--draws",
                    "1",
                    "--seed",
                    str(seed),
                    "--gpus",
                    devs,
                    "--max-model-len",
                    "40960",
                    "--purpose",
                    purpose,
                    "--env-fingerprint",
                    env_fingerprint,
                ]
                log_fh = log_path.open("wb")
                proc = subprocess.Popen(cmd, cwd=str(sbx.repo), env=cli_env, stdout=log_fh, stderr=subprocess.STDOUT)
                procs.append((r_idx, proc, log_path, log_fh))

            failed = []
            for r_idx, proc, log_path, log_fh in procs:
                rc = proc.wait(timeout=14400)
                log_fh.close()
                if rc != 0:
                    tail = log_path.read_text(encoding="utf-8", errors="replace")[-2000:]
                    failed.append(f"replica {r_idx} rc={rc}:\n{tail}")
            if failed:
                raise RuntimeError("Data-parallel shard inference failed:\n" + "\n".join(failed))
        except BaseException as exc:
            body_exc = exc
            raise
        finally:
            for _r_idx, _p, _lp, lfh in procs:
                try:
                    lfh.close()
                except Exception:
                    pass
            _retire_all_launched_servers_fail_closed(
                sbx.repo,
                launched_srvs,
                f"Completed sharded inference {run_prefix}",
                vpy=py_bin,
                active_exc=body_exc,
            )

        wall_elapsed = time.monotonic() - t0
        merged_sbx_dir = sbx.repo / "results" / "inference" / run_prefix
        merged_sbx_dir.mkdir(parents=True, exist_ok=False)
        all_recs: list[dict[str, Any]] = []
        shard_summaries: list[dict[str, Any]] = []
        total_joules = 0.0
        total_gpu_hours = 0.0
        total_draft = 0.0
        total_accepted = 0.0
        for r_idx in range(n_rep):
            sdir = sbx.repo / "results" / "inference" / f"{run_prefix}_shard_r{r_idx}"
            if not sdir.is_dir():
                continue
            ssum = json.loads((sdir / "summary.json").read_text(encoding="utf-8"))
            shard_summaries.append(ssum)
            total_joules += float(ssum["board_joules"])
            total_gpu_hours += float(ssum["assigned_gpu_hours"])
            total_draft += float(ssum["mtp_speculative"]["draft_tokens"] or 0.0)
            total_accepted += float(ssum["mtp_speculative"]["accepted_tokens"] or 0.0)
            with gzip.open(sdir / "generations.jsonl.gz", "rt", encoding="utf-8") as gf:
                for ln in gf:
                    if ln.strip():
                        all_recs.append(json.loads(ln))
            sbx.promote_declared_output(f"results/inference/{run_prefix}_shard_r{r_idx}")

        if len(all_recs) != len(all_rows):
            raise RuntimeError(f"Merged record count mismatch: got {len(all_recs)}, expected {len(all_rows)}")

        merged_gz = merged_sbx_dir / "generations.jsonl.gz"
        with gzip.open(merged_gz, "wt", encoding="utf-8") as out_gz:
            for rec in sorted(all_recs, key=lambda x: (str(x["id"]), int(x["seed"]))):
                out_gz.write(json.dumps(rec, ensure_ascii=False) + "\n")

        score_env = build_isolated_trial_env(sbx.caches / "scorer")
        scored_output.parent.mkdir(parents=True, exist_ok=True)
        score_cmd = [
            py_bin,
            "-I",
            "-B",
            str(sbx.repo / "eval" / "isolated_scorer.py"),
            "capability",
            "--split-prompts",
            str(sbx.repo / dataset_path.relative_to(ROOT)),
            "--generations",
            str(merged_gz),
            "--draws",
            "1",
            "--base-seed",
            str(seed),
            "--candidate-id",
            arm["candidate_id"],
            "--protocol-id",
            purpose,
            "--env-fingerprint",
            env_fingerprint,
            "--output",
            str(scored_output),
        ]
        if external_labels is not None:
            score_cmd.extend(["--external-labels", str(external_labels)])
        if holdout_registry is not None:
            score_cmd.extend(["--holdout-registry", str(holdout_registry)])
        subprocess.run(score_cmd, cwd=str(sbx.repo), env=score_env, check=True, timeout=300)

        scored_payload = json.loads(scored_output.read_text(encoding="utf-8"))
        merged_summary = {
            "run": run_prefix,
            "candidate_id": arm["candidate_id"],
            "dataset": dataset_path.name,
            "dataset_sha256": _file_sha256(dataset_path),
            "replicas": len(shard_summaries),
            "tp_per_replica": tp,
            "mtp_tokens": int(arm["mtp_tokens"]),
            "parallel_wall_seconds": wall_elapsed,
            "max_replica_dataset_seconds": max(float(s["dataset_seconds"]) for s in shard_summaries),
            "sum_assigned_gpu_hours": total_gpu_hours,
            "sum_board_joules": total_joules,
            "mtp_speculative": {
                "draft_tokens": total_draft,
                "accepted_tokens": total_accepted,
                "acceptance_rate": (total_accepted / total_draft) if total_draft > 0 else None,
            },
            "raw_generations_sha256": _file_sha256(merged_gz),
            "scored_metrics": scored_payload["metrics"],
        }
        _write_json(merged_sbx_dir / "summary.json", merged_summary)
        sbx.promote_declared_output(f"results/inference/{run_prefix}")
        return merged_summary


def cmd_stage_holdout_vault(args: argparse.Namespace) -> int:
    src = ROOT / ".local" / "governance_anchor" / "holdout_labels"
    vault = Path(args.vault_dir).resolve()
    if ROOT in vault.parents or vault == ROOT:
        raise RuntimeError("External holdout vault must reside OUTSIDE workspace ROOT")
    vault.mkdir(parents=True, exist_ok=True)
    os.chmod(vault, 0o700)
    copied = []
    if src.is_dir():
        for jf in sorted(src.glob("*_labels.json")):
            dst = vault / jf.name
            shutil.copy2(jf, dst)
            os.chmod(dst, 0o600)
            copied.append({"file": jf.name, "sha256": _file_sha256(dst)})
        if args.remove_workspace_copy:
            shutil.rmtree(src)
    elif not list(vault.glob("*_labels.json")):
        raise RuntimeError(f"Neither {src} nor {vault} contains holdout label files")
    print(json.dumps({"status": "VAULT_STAGED", "vault_dir": str(vault), "files": copied}, indent=2))
    return 0


def _run_one_ddp_train(
    run_name: str,
    arm_key: str,
    seed: int,
    port: int,
    *,
    anchor_path: Path | None = None,
    verify_manifest: bool = True,
) -> dict[str, Any]:
    """Execute one 8-GPU DDP training run inside a fresh detached Git worktree & empty per-run JIT/compiler cache."""
    arm = TRAIN_ARMS[arm_key]
    py_bin = _venv_py(ROOT)
    with HermeticTrialSandbox(
        ROOT,
        f"train_{run_name}",
        anchor_path=anchor_path,
        verify_manifest=verify_manifest,
    ) as sbx:
        cmd = [
            py_bin,
            "-B",
            "-m",
            "torch.distributed.run",
            "--nproc_per_node=8",
            f"--master_port={port}",
            str(sbx.repo / "code" / "train_systems_v2.py"),
            "--run-name",
            run_name,
            "--dataset",
            "data/splits/sft_systems1024_v2.jsonl",
            "--heldout",
            "data/splits/sft_heldout64_v2.jsonl",
            "--global-batch",
            "64",
            "--micro-batch",
            str(arm["micro_batch"]),
            "--seed",
            str(seed),
        ]
        if arm["length_bucketing"]:
            cmd.append("--length-bucketing")
        subprocess.run(cmd, cwd=str(sbx.repo), env=sbx.env, check=True, timeout=1800)
        s_path = sbx.repo / "results" / "training" / run_name / "summary.json"
        s = json.loads(s_path.read_text(encoding="utf-8"))
        if s.get("status") != "COMPLETE":
            raise RuntimeError(f"DDP run {run_name} did not complete cleanly: {s.get('status')!r}")
        sbx.promote_declared_output(f"results/training/{run_name}")
        return {
            "loop_seconds": float(s["loop_seconds"]),
            "checkpoint_save_seconds": float(s.get("checkpoint_save_seconds", 0.0)),
            "process_seconds": float(s["process_seconds"]),
            "before_nll": float(s["before_nll"]),
            "after_nll": float(s["after_nll"]),
            "peak_allocated_gib": float(s["peak_allocated_gib"]),
            "board_joules": float(s["board_joules"]),
            "assigned_loop_gpu_hours": float(s["assigned_loop_gpu_hours"]),
            "supervised_tokens_per_second": float(s["supervised_tokens_per_second"]),
            "padding_over_input_fraction": float(s["padding_over_input_fraction"]),
            "partial_matmul_mfu_lower": float(s["mfu"]["lower"]),
            "partial_matmul_mfu_upper": float(s["mfu"]["upper"]),
            "infrastructure_errors": 0,
        }


def cmd_calibrate_train_aa_hermetic(args: argparse.Namespace) -> int:
    """Plane-1 preselection 5-block AB/BA baseline A/A noise calibration on node-1 under exact HermeticTrialSandbox policy."""
    enforce_hermetic_pre_trial_hygiene(verify_manifest=False)
    blocks = []
    flat_runs = []
    diffs_sec = []
    diffs_tps = []
    for b_idx in range(5):
        seed_b = 20261150 + b_idx
        order = ["A1", "A2"] if (b_idx % 2 == 0) else ["A2", "A1"]
        arms_rec: dict[str, Any] = {}
        for pos, label in enumerate(order):
            rname = f"p1_train_aa_hermetic_b{b_idx}_{label}"
            port = 24900 + b_idx * 10 + pos
            rec = _run_one_ddp_train(rname, "T0", seed_b, port, verify_manifest=False)
            arms_rec[label] = rec
            flat_runs.append(
                {
                    "block": b_idx,
                    "arm": label,
                    "run_name": rname,
                    "loop_seconds": rec["loop_seconds"],
                    "after_nll": rec["after_nll"],
                    "peak_allocated_gib": rec["peak_allocated_gib"],
                    "board_joules": rec["board_joules"],
                    "supervised_tokens_per_second": rec["supervised_tokens_per_second"],
                }
            )
        d_sec = arms_rec["A1"]["loop_seconds"] - arms_rec["A2"]["loop_seconds"]
        d_tps = arms_rec["A1"]["supervised_tokens_per_second"] - arms_rec["A2"]["supervised_tokens_per_second"]
        diffs_sec.append(d_sec)
        diffs_tps.append(d_tps)
        blocks.append({"block_index": b_idx, "seed": seed_b, "order": order, "arms": arms_rec, "diff_A1_minus_A2_seconds": d_sec})

    sd_diff = statistics.stdev(diffs_sec)
    se_aa = sd_diff / math.sqrt(len(diffs_sec))
    se_tps = statistics.stdev(diffs_tps) / math.sqrt(len(diffs_tps))
    out = {
        "status": "TRAIN_AA_HERMETIC_WORKTREE_CACHE_CALIBRATED",
        "preserves_prior_calibration": "results/plane1_train_aa_noise.json",
        "cache_and_worktree_policy": "Exact HermeticTrialSandbox per-run detached git worktree + fresh empty TRITON_CACHE_DIR/TORCHINDUCTOR_CACHE_DIR/CUDA_CACHE_PATH/TMPDIR/PYTHONPYCACHEPREFIX",
        "paired_blocks": len(blocks),
        "total_runs": len(flat_runs),
        "mean_loop_seconds": statistics.fmean(r["loop_seconds"] for r in flat_runs),
        "se_aa_loop_seconds": se_aa,
        "per_block_sd_loop_seconds": sd_diff,
        "mean_supervised_tokens_per_second": statistics.fmean(r["supervised_tokens_per_second"] for r in flat_runs),
        "se_aa_supervised_tokens_per_second": se_tps,
        "runs": flat_runs,
        "blocks": blocks,
    }
    _write_json(ROOT / "results" / "plane1_train_aa_hermetic_v2.json", out)
    print(json.dumps({k: v for k, v in out.items() if k != "blocks"}, indent=2), flush=True)
    return 0


def cmd_calibrate_serving_aa_hermetic(args: argparse.Namespace) -> int:
    """Plane-1 preselection 5-block AB/BA baseline S0 A/A noise calibration on dedicated node-2 under exact HermeticTrialSandbox policy."""
    enforce_hermetic_pre_trial_hygiene(verify_manifest=False)
    py_bin = _venv_py(ROOT)
    env_fp = "node2_dedicated_8xa100_sxm4_80gb_vllm029_hermetic_aa5_block32_c8"
    aa_arms = {
        "A1": {"gpus": "0,1", "port": 18270},
        "A2": {"gpus": "2,3", "port": 18271},
    }
    blocks = []
    with HermeticTrialSandbox(ROOT, "p1_serve_aa_hermetic_suite", verify_manifest=False) as w_sbx:
        launched: dict[str, tuple[str, dict[str, str], str]] = {}
        body_exc: BaseException | None = None
        try:
            for label, cfg in aa_arms.items():
                sname = f"p1_serve_aa_hermetic_srv_{label}"
                srv_env = build_isolated_trial_env(w_sbx.caches / f"srv_{label}")
                _launch_owned_vllm_in_sandbox(
                    w_sbx.repo,
                    srv_env,
                    sname,
                    "models/dense27_base",
                    cfg["gpus"],
                    2,
                    int(cfg["port"]),
                    mtp_tokens=0,
                    max_model_len=40960,
                    max_num_seqs=16,
                    max_batched_tokens=8192,
                    gpu_mem_util=0.90,
                    vpy=py_bin,
                )
                launched[label] = (sname, srv_env, str(cfg["gpus"]))

            for b_idx in range(5):
                seed_b = 20261117 + b_idx
                order = ["A1", "A2"] if (b_idx % 2 == 0) else ["A2", "A1"]
                arms_rec: dict[str, Any] = {}
                for label in order:
                    cfg = aa_arms[label]
                    rname = f"p1_serve_aa_hermetic_b{b_idx}_{label}"
                    with HermeticTrialSandbox(
                        ROOT,
                        f"aa_cli_{rname}",
                        verify_manifest=False,
                        link_runs_from=w_sbx.repo / "runs",
                    ) as c_sbx:
                        cmd = [
                            py_bin,
                            "-B",
                            str(c_sbx.repo / "code" / "infer_v2.py"),
                            "--name",
                            rname,
                            "--server-stage",
                            launched[label][0],
                            "--base-url",
                            f"http://127.0.0.1:{cfg['port']}",
                            "--model-name",
                            "qwen_gov_v2",
                            "--dataset",
                            str(c_sbx.repo / "data" / "splits" / "math_search_block32.jsonl"),
                            "--model-config",
                            str(c_sbx.repo / "configs" / "model.json"),
                            "--tokenizer-dir",
                            str(c_sbx.repo / "models" / "dense27_base"),
                            "--cap",
                            "32768",
                            "--effort",
                            "medium",
                            "--concurrency",
                            "8",
                            "--draws",
                            "1",
                            "--seed",
                            str(seed_b),
                            "--gpus",
                            cfg["gpus"],
                            "--max-model-len",
                            "40960",
                            "--purpose",
                            "plane1_serving_aa_hermetic_v2",
                            "--env-fingerprint",
                            env_fp,
                        ]
                        subprocess.run(cmd, cwd=str(c_sbx.repo), env=c_sbx.env, check=True, timeout=7200)
                        obs = json.loads((c_sbx.repo / "results" / "inference" / rname / "summary.json").read_text(encoding="utf-8"))
                        c_sbx.promote_declared_output(f"results/inference/{rname}")
                        arms_rec[label] = {
                            "dataset_seconds": float(obs["dataset_seconds"]),
                            "assigned_gpu_seconds": float(obs["assigned_gpu_hours"]) * 3600.0,
                            "output_tokens_per_second": float(obs["output_tokens_per_second"]),
                            "tpot_p95_ms": float(obs["tpot_p95_ms"]),
                            "board_joules": float(obs["board_joules"]),
                            "stop_rate": float(obs["stop_rate"]),
                            "lenient_accuracy": float(obs["lenient_accuracy"]),
                            "infrastructure_errors": int(obs["infrastructure_errors"]),
                        }
                    time.sleep(2.0)
                blocks.append({"block_index": b_idx, "seed": seed_b, "order": order, "arms": arms_rec})
        except BaseException as exc:
            body_exc = exc
            raise
        finally:
            _retire_all_launched_servers_fail_closed(
                w_sbx.repo,
                list(launched.values()),
                "Completed Plane-1 hermetic serving A/A calibration",
                vpy=py_bin,
                active_exc=body_exc,
            )

    diffs_wall = [b["arms"]["A1"]["dataset_seconds"] - b["arms"]["A2"]["dataset_seconds"] for b in blocks]
    diffs_tps = [b["arms"]["A1"]["output_tokens_per_second"] - b["arms"]["A2"]["output_tokens_per_second"] for b in blocks]
    diffs_tpot = [b["arms"]["A1"]["tpot_p95_ms"] - b["arms"]["A2"]["tpot_p95_ms"] for b in blocks]
    all_arms = [b["arms"][lb] for b in blocks for lb in ("A1", "A2")]
    sd_wall = statistics.stdev(diffs_wall)
    se_wall = sd_wall / math.sqrt(len(diffs_wall))
    out = {
        "status": "SERVING_AA_HERMETIC_WORKTREE_CACHE_CALIBRATED",
        "preserves_prior_calibration": "results/plane1_serving_aa_noise.json",
        "cache_and_worktree_policy": "Exact HermeticTrialSandbox suite detached git worktree + per-server isolated VLLM_CACHE_ROOT/TRITON_CACHE_DIR/TORCHINDUCTOR_CACHE_DIR + per-block detached client worktree/caches",
        "paired_blocks": len(blocks),
        "total_runs": len(all_arms),
        "mean_dataset_seconds": statistics.fmean(r["dataset_seconds"] for r in all_arms),
        "se_aa_dataset_seconds": se_wall,
        "per_block_sd_dataset_seconds": sd_wall,
        "mean_assigned_gpu_seconds": 2.0 * statistics.fmean(r["dataset_seconds"] for r in all_arms),
        "se_aa_assigned_gpu_seconds": 2.0 * se_wall,
        "mean_output_tokens_per_second": statistics.fmean(r["output_tokens_per_second"] for r in all_arms),
        "se_aa_output_tokens_per_second": statistics.stdev(diffs_tps) / math.sqrt(len(diffs_tps)),
        "mean_tpot_p95_ms": statistics.fmean(r["tpot_p95_ms"] for r in all_arms),
        "se_aa_tpot_p95_ms": statistics.stdev(diffs_tpot) / math.sqrt(len(diffs_tpot)),
        "blocks": blocks,
    }
    _write_json(ROOT / "results" / "plane1_serving_aa_hermetic_v2.json", out)
    print(json.dumps({k: v for k, v in out.items() if k != "blocks"}, indent=2), flush=True)
    return 0


def cmd_worker_train_search(args: argparse.Namespace) -> int:
    """Stage 1 on node-1: 3-block 3x3 Latin Square Search over [T0, T1, T2] inside fresh per-run HermeticTrialSandboxes."""
    anchor_path = Path(args.anchor_file).resolve(strict=True)
    enforce_hermetic_pre_trial_hygiene(anchor_path, root=ROOT, verify_manifest=True)
    scored_dir = ROOT / "results" / "scored"
    scored_dir.mkdir(parents=True, exist_ok=True)

    latin_3x3 = [
        ["T0", "T1", "T2"],
        ["T1", "T2", "T0"],
        ["T2", "T0", "T1"],
    ]
    search_env = "node1_8xa100_sxm4_80gb_ddp8_torch213_cu129_sft1024_search3_hermetic"
    search_blocks = []
    for b_idx, order in enumerate(latin_3x3):
        seed_b = 20261300 + b_idx
        arms_rec: dict[str, Any] = {}
        for pos, arm_k in enumerate(order):
            rname = f"p2_train_search_b{b_idx:02d}_{arm_k}"
            port = 25100 + b_idx * 10 + pos
            arms_rec[arm_k] = _run_one_ddp_train(rname, arm_k, seed_b, port, anchor_path=anchor_path, verify_manifest=True)
        search_blocks.append({"block_index": b_idx, "seed": seed_b, "order": order, "arms": arms_rec})

    search_blocks_path = scored_dir / "p2_train_search3_blocks.json"
    _write_json(search_blocks_path, {"env_fingerprint": search_env, "blocks": search_blocks})

    scored_search = {}
    for arm_k in ("T0", "T1", "T2"):
        scored_search[arm_k] = run_isolated_systems_scorer(
            blocks_json=search_blocks_path,
            arm=arm_k,
            block_metric="loop_seconds",
            candidate_id=TRAIN_ARMS[arm_k]["candidate_id"],
            protocol_id="loop_train_systems_search_v2",
            output_path=scored_dir / f"p2_train_search_{arm_k}.json",
            root=ROOT,
            anchor_path=anchor_path,
        )

    summary = {
        "status": "WORKER_TRAIN_SEARCH_COMPLETE",
        "search_mean_loop_seconds": {k: scored_search[k]["metrics"]["loop_seconds"] for k in ("T0", "T1", "T2")},
        "search_blocks_file": str(search_blocks_path.relative_to(ROOT)),
    }
    _write_json(ROOT / "results" / "plane2_worker_train_search.json", summary)
    print(json.dumps(summary, indent=2), flush=True)
    return 0


def cmd_worker_train_confirm(args: argparse.Namespace) -> int:
    """Stage 2 on node-1: 10-block interleaved AB/BA Confirm ONLY after cryptographic Search ledger verification."""
    anch_arg = getattr(args, "anchor_file", None)
    anchor_path = Path(anch_arg).resolve() if anch_arg else _default_anchor(ROOT)
    winner_arm = str(args.finalist)

    # Hard pre-GPU gate: verifies live manifest, hash-chained ledger, anchor, and Search adjudication BEFORE any GPU run!
    auth = verify_train_confirm_authorization(winner_arm, root=ROOT, anchor_path=anchor_path)

    scored_dir = ROOT / "results" / "scored"
    scored_dir.mkdir(parents=True, exist_ok=True)
    confirm_env = "node1_8xa100_sxm4_80gb_ddp8_torch213_cu129_sft1024_confirm10_hermetic"
    confirm_blocks = []
    for b_idx in range(10):
        seed_b = 20261400 + b_idx
        order = ["T0", winner_arm] if (b_idx % 2 == 0) else [winner_arm, "T0"]
        arms_rec = {}
        for pos, arm_k in enumerate(order):
            rname = f"p2_train_confirm_b{b_idx:02d}_{arm_k}"
            port = 25300 + b_idx * 10 + pos
            arms_rec[arm_k] = _run_one_ddp_train(rname, arm_k, seed_b, port, anchor_path=anchor_path, verify_manifest=True)
        confirm_blocks.append({"block_index": b_idx, "seed": seed_b, "order": order, "arms": arms_rec})

    confirm_blocks_path = scored_dir / "p2_train_confirm10_blocks.json"
    _write_json(
        confirm_blocks_path,
        {"env_fingerprint": confirm_env, "winner_arm": winner_arm, "authorization": auth, "blocks": confirm_blocks},
    )

    for arm_k in ("T0", winner_arm):
        run_isolated_systems_scorer(
            blocks_json=confirm_blocks_path,
            arm=arm_k,
            block_metric="loop_seconds",
            candidate_id=TRAIN_ARMS[arm_k]["candidate_id"],
            protocol_id="loop_train_systems_confirm_v2",
            output_path=scored_dir / f"p2_train_confirm_{arm_k}.json",
            root=ROOT,
            anchor_path=anchor_path,
        )

    summary = {
        "status": "WORKER_TRAIN_CONFIRM_COMPLETE",
        "confirmed_finalist_arm": winner_arm,
        "confirmed_finalist_candidate_id": TRAIN_ARMS[winner_arm]["candidate_id"],
        "authorization": auth,
        "confirm_blocks_file": str(confirm_blocks_path.relative_to(ROOT)),
    }
    _write_json(ROOT / "results" / "plane2_worker_train_confirm.json", summary)
    print(json.dumps(summary, indent=2), flush=True)
    return 0


def compute_williams_2x2_factorial_effects(blocks: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Compute paired 2x2 factorial GPU-second cost reductions across Williams Latin Square blocks:
      - Factor A (bundled precision + TP layout): BF16 TP=2 -> GPTQ W4A16 TP=1 (S0 -> S2, S1 -> S3).
        Precision and TP count are intentionally bundled and never claimed separately.
      - Factor B (speculative decoding): No-MTP -> Repaired Native BF16 MTP3 (S0 -> S1, S2 -> S3).
      - True unscaled beneficial 2x2 difference-in-differences synergy:
            (S2 - S3) - (S0 - S1) = S1 + S2 - S0 - S3
        which equals 0 under purely additive GPU-second savings, > 0 when MTP3 saves strictly more
        GPU-seconds on INT4+TP1 than on BF16+TP2 (super-additive synergy), and < 0 under sub-additivity.
    """
    q_no_mtp = [float(b["arms"]["S0"]["assigned_gpu_seconds"]) - float(b["arms"]["S2"]["assigned_gpu_seconds"]) for b in blocks]
    q_with_mtp = [float(b["arms"]["S1"]["assigned_gpu_seconds"]) - float(b["arms"]["S3"]["assigned_gpu_seconds"]) for b in blocks]
    m_bf16 = [float(b["arms"]["S0"]["assigned_gpu_seconds"]) - float(b["arms"]["S1"]["assigned_gpu_seconds"]) for b in blocks]
    m_int4 = [float(b["arms"]["S2"]["assigned_gpu_seconds"]) - float(b["arms"]["S3"]["assigned_gpu_seconds"]) for b in blocks]
    synergy_did = [
        (float(b["arms"]["S2"]["assigned_gpu_seconds"]) - float(b["arms"]["S3"]["assigned_gpu_seconds"]))
        - (float(b["arms"]["S0"]["assigned_gpu_seconds"]) - float(b["arms"]["S1"]["assigned_gpu_seconds"]))
        for b in blocks
    ]

    def _mean_se(xs: list[float]) -> dict[str, float]:
        return {"mean": statistics.fmean(xs), "se": (statistics.stdev(xs) / math.sqrt(len(xs))) if len(xs) > 1 else 0.0}

    return {
        "formula_note": (
            "Quantization (GPTQ W4A16) and tensor-parallel count (TP=2 -> TP=1) are bundled as Factor A. "
            "Beneficial GPU-second synergy is the exact unscaled difference-in-differences "
            "(S2 - S3) - (S0 - S1) = S1 + S2 - S0 - S3 (0 under additive costs)."
        ),
        "bundled_quant_plus_tp1_without_mtp_S0_minus_S2": _mean_se(q_no_mtp),
        "bundled_quant_plus_tp1_with_mtp3_S1_minus_S3": _mean_se(q_with_mtp),
        "mtp3_effect_on_bf16_tp2_S0_minus_S1": _mean_se(m_bf16),
        "mtp3_effect_on_bundled_int4_tp1_S2_minus_S3": _mean_se(m_int4),
        "beneficial_synergy_did_S1_plus_S2_minus_S0_minus_S3": _mean_se(synergy_did),
    }


def cmd_worker_serving_williams12(args: argparse.Namespace) -> int:
    anchor_path = Path(args.anchor_file).resolve(strict=True)
    enforce_hermetic_pre_trial_hygiene(anchor_path, root=ROOT, verify_manifest=True)
    scored_dir = ROOT / "results" / "scored"
    scored_dir.mkdir(parents=True, exist_ok=True)
    env_fp = "node2_dedicated_8xa100_sxm4_80gb_vllm029_williams12_block32_c8_hermetic"
    py_bin = _venv_py(ROOT)

    blocks = []
    with HermeticTrialSandbox(ROOT, "p2_williams12_suite", anchor_path=anchor_path, verify_manifest=True) as w_sbx:
        launched_srvs: dict[str, tuple[str, dict[str, str], str]] = {}
        body_exc: BaseException | None = None
        try:
            for arm_k in ("S0", "S1", "S2", "S3"):
                cfg = SERVING_ARMS[arm_k]
                sname = f"p2_williams12_srv_{arm_k}"
                srv_env = build_isolated_trial_env(w_sbx.caches / f"srv_{arm_k}")
                _launch_owned_vllm_in_sandbox(
                    w_sbx.repo,
                    srv_env,
                    sname,
                    cfg["model_dir"],
                    cfg["williams_gpus"],
                    int(cfg["tp"]),
                    int(cfg["williams_port"]),
                    mtp_tokens=int(cfg["mtp_tokens"]),
                    max_model_len=40960,
                    max_num_seqs=16,
                    max_batched_tokens=8192,
                    gpu_mem_util=0.90,
                    vpy=py_bin,
                )
                launched_srvs[arm_k] = (sname, srv_env, str(cfg["williams_gpus"]))

            for b_idx in range(12):
                seed_b = 20261500 + b_idx
                order = WILLIAMS_4X4[b_idx % 4]
                arms_rec: dict[str, Any] = {}
                for arm_k in order:
                    cfg = SERVING_ARMS[arm_k]
                    rname = f"p2_serve_w12_b{b_idx:02d}_{arm_k}"
                    with HermeticTrialSandbox(
                        ROOT,
                        f"cli_{rname}",
                        anchor_path=anchor_path,
                        verify_manifest=True,
                        link_runs_from=w_sbx.repo / "runs",
                    ) as c_sbx:
                        cmd = [
                            py_bin,
                            "-B",
                            str(c_sbx.repo / "code" / "infer_v2.py"),
                            "--name",
                            rname,
                            "--server-stage",
                            launched_srvs[arm_k][0],
                            "--base-url",
                            f"http://127.0.0.1:{cfg['williams_port']}",
                            "--model-name",
                            "qwen_gov_v2",
                            "--dataset",
                            str(c_sbx.repo / "data" / "splits" / "math_search_block32.jsonl"),
                            "--model-config",
                            str(c_sbx.repo / "configs" / "model.json"),
                            "--tokenizer-dir",
                            str(c_sbx.repo / "models" / "dense27_base"),
                            "--cap",
                            "32768",
                            "--effort",
                            "medium",
                            "--concurrency",
                            "8",
                            "--draws",
                            "1",
                            "--seed",
                            str(seed_b),
                            "--gpus",
                            cfg["williams_gpus"],
                            "--max-model-len",
                            "40960",
                            "--purpose",
                            "loop_dense_serving_confirm_v2",
                            "--env-fingerprint",
                            env_fp,
                        ]
                        subprocess.run(cmd, cwd=str(c_sbx.repo), env=c_sbx.env, check=True, timeout=7200)
                        obs = json.loads((c_sbx.repo / "results" / "inference" / rname / "summary.json").read_text(encoding="utf-8"))
                        c_sbx.promote_declared_output(f"results/inference/{rname}")
                        arms_rec[arm_k] = {
                            "assigned_gpu_seconds": float(obs["assigned_gpu_hours"]) * 3600.0,
                            "assigned_gpu_hours": float(obs["assigned_gpu_hours"]),
                            "dataset_seconds": float(obs["dataset_seconds"]),
                            "output_tokens_per_second": float(obs["output_tokens_per_second"]),
                            "tpot_p95_ms": float(obs["tpot_p95_ms"]),
                            "ttft_p95_ms": float(obs["ttft_p95_ms"]),
                            "board_joules": float(obs["board_joules"]),
                            "stop_rate": float(obs["stop_rate"]),
                            "lenient_accuracy": float(obs["lenient_accuracy"]),
                            "infrastructure_errors": int(obs["infrastructure_errors"]),
                            "mtp_acceptance_rate": obs["mtp_speculative"]["acceptance_rate"],
                        }
                    time.sleep(2.0)
                blocks.append({"block_index": b_idx, "seed": seed_b, "order": order, "arms": arms_rec})
        except BaseException as exc:
            body_exc = exc
            raise
        finally:
            _retire_all_launched_servers_fail_closed(
                w_sbx.repo,
                list(launched_srvs.values()),
                "Completed 12-block Williams serving confirmation on node-2",
                vpy=py_bin,
                active_exc=body_exc,
            )

    blocks_path = scored_dir / "p2_serving_williams12_blocks.json"
    _write_json(blocks_path, {"env_fingerprint": env_fp, "blocks": blocks})

    scored = {}
    for arm_k in ("S0", "S1", "S2", "S3"):
        scored[arm_k] = run_isolated_systems_scorer(
            blocks_json=blocks_path,
            arm=arm_k,
            block_metric="assigned_gpu_seconds",
            candidate_id=SERVING_ARMS[arm_k]["candidate_id"],
            protocol_id="loop_dense_serving_confirm_v2",
            output_path=scored_dir / f"p2_serving_williams12_{arm_k}.json",
            root=ROOT,
            anchor_path=anchor_path,
        )

    factorial = {
        "status": "WORKER_SERVING_WILLIAMS12_COMPLETE",
        "n_blocks": len(blocks),
        "arm_means": {
            k: {
                "assigned_gpu_seconds": scored[k]["metrics"]["assigned_gpu_seconds"],
                "dataset_seconds": scored[k]["metrics"]["dataset_seconds"],
                "mean_board_joules": scored[k]["metrics"]["mean_board_joules"],
                "median_tpot_p95_ms": scored[k]["metrics"]["median_tpot_p95_ms"],
                "min_block_stop_rate": scored[k]["metrics"]["stop_rate"],
                "mean_lenient_accuracy_block32": scored[k]["metrics"]["lenient_accuracy"],
            }
            for k in ("S0", "S1", "S2", "S3")
        },
        "paired_2x2_factorial_gpu_second_reductions": compute_williams_2x2_factorial_effects(blocks),
    }
    _write_json(ROOT / "results" / "plane2_worker_serving_williams12.json", factorial)
    print(json.dumps(factorial, indent=2), flush=True)
    return 0


def cmd_worker_serving_quality1280(args: argparse.Namespace) -> int:
    anchor_path = Path(args.anchor_file).resolve(strict=True)
    enforce_hermetic_pre_trial_hygiene(anchor_path, root=ROOT, verify_manifest=True)
    scored_dir = ROOT / "results" / "scored"
    scored_dir.mkdir(parents=True, exist_ok=True)
    q_env = "node0_8xa100_sxm4_80gb_vllm029_qwen38_27b_confirm1280_cap32768_medium_s20261117_hermetic"
    ds_path = ROOT / "data" / "splits" / "math_confirm1280.jsonl"

    s0_sum = run_sharded_panel_inference(
        run_prefix="p2_quality_confirm1280_S0",
        dataset_path=ds_path,
        arm_key="S0",
        purpose="loop_dense_serving_confirm_v2_quality",
        env_fingerprint=q_env,
        scored_output=scored_dir / "p2_quality_confirm1280_S0.json",
        anchor_path=anchor_path,
    )
    s3_sum = run_sharded_panel_inference(
        run_prefix="p2_quality_confirm1280_S3",
        dataset_path=ds_path,
        arm_key="S3",
        purpose="loop_dense_serving_confirm_v2_quality",
        env_fingerprint=q_env,
        scored_output=scored_dir / "p2_quality_confirm1280_S3.json",
        anchor_path=anchor_path,
    )
    out = {"status": "WORKER_SERVING_QUALITY1280_COMPLETE", "S0": s0_sum, "S3": s3_sum}
    _write_json(ROOT / "results" / "plane2_worker_serving_quality1280.json", out)
    print(json.dumps(out, indent=2), flush=True)
    return 0


def _run_gate_decision(
    *,
    trial_id: str,
    hypothesis: str,
    cand_json: Path,
    inc_json: Path,
    base_json: Path,
    rules_json: Path,
    ledger_path: Path,
    anchor_path: Path,
    quality_cand: Path | None = None,
    quality_inc: Path | None = None,
) -> dict[str, Any]:
    with HermeticTrialSandbox(ROOT, f"gate_{trial_id}", anchor_path=anchor_path, verify_manifest=True) as sbx:
        cmd = [
            sys.executable,
            "-I",
            "-B",
            str(sbx.repo / "eval" / "strict_guard.py"),
            "gate-decision",
            "--candidate",
            str(cand_json),
            "--incumbent",
            str(inc_json),
            "--baseline",
            str(base_json),
            "--rules",
            str(sbx.repo / rules_json.relative_to(ROOT)),
            "--ledger",
            str(ledger_path),
            "--anchor-file",
            str(anchor_path),
            "--trial-id",
            trial_id,
            "--hypothesis",
            hypothesis,
        ]
        if quality_cand is not None and quality_inc is not None:
            cmd.extend(["--quality-candidate", str(quality_cand), "--quality-incumbent", str(quality_inc)])
        proc = subprocess.run(cmd, cwd=str(sbx.repo), env=sbx.env, capture_output=True, text=True, check=False)
    if not proc.stdout.strip():
        raise RuntimeError(f"Empty gate-decision output for {trial_id}: stderr={proc.stderr}")
    res = json.loads(proc.stdout)
    res["cli_returncode"] = proc.returncode
    return res


def cmd_adjudicate_train_search(args: argparse.Namespace) -> int:
    """Live pre-Confirm ledger adjudication of the registered 2-candidate independent-vs-baseline Search screen."""
    anchor_path = Path(args.anchor_file).resolve(strict=True)
    enforce_hermetic_pre_trial_hygiene(anchor_path, root=ROOT, verify_manifest=True)
    scored_dir = ROOT / "results" / "scored"
    ledger_path = ROOT / "results" / "governed_ledger_v2.jsonl"

    d1 = _run_gate_decision(
        trial_id="trial_01_train_search_t1_vs_t0",
        hypothesis="Registered independent-vs-baseline Search screen: per-rank micro-batch=4 under exact global-token scaling cuts DDP loop_seconds by >=10s vs T0 without >0.5% relative NLL drift.",
        cand_json=scored_dir / "p2_train_search_T1.json",
        inc_json=scored_dir / "p2_train_search_T0.json",
        base_json=scored_dir / "p2_train_search_T0.json",
        rules_json=ROOT / "configs" / "gate_rules_train_systems_search.json",
        ledger_path=ledger_path,
        anchor_path=anchor_path,
    )
    d2 = _run_gate_decision(
        trial_id="trial_02_train_search_t2_vs_t0",
        hypothesis="Registered independent-vs-baseline Search screen: micro-batch=4 plus within-update length bucketing cuts DDP loop_seconds by >=10s vs T0 without >0.5% relative NLL drift.",
        cand_json=scored_dir / "p2_train_search_T2.json",
        inc_json=scored_dir / "p2_train_search_T0.json",
        base_json=scored_dir / "p2_train_search_T0.json",
        rules_json=ROOT / "configs" / "gate_rules_train_systems_search.json",
        ledger_path=ledger_path,
        anchor_path=anchor_path,
    )

    post_anchor = json.loads(anchor_path.read_text(encoding="utf-8"))
    scored_payloads = {
        k: json.loads((scored_dir / f"p2_train_search_{k}.json").read_text(encoding="utf-8"))
        for k in ("T1", "T2")
    }
    selected_finalist = select_train_search_finalist({"T1": d1, "T2": d2}, scored_payloads)
    out = {
        "status": "TRAIN_SEARCH_ADJUDICATED" if selected_finalist else "HALTED_NO_ELIGIBLE_SEARCH_FINALIST",
        "protocol_epoch": post_anchor["protocol_epoch"],
        "manifest_root_sha256": post_anchor["manifest_root_sha256"],
        "ledger_head_sha256_after_search": post_anchor["ledger_head_sha256"],
        "ledger_entry_count_after_search": post_anchor["ledger_entry_count"],
        "selected_finalist_arm": selected_finalist,
        "selected_finalist_candidate_id": TRAIN_ARMS[selected_finalist]["candidate_id"] if selected_finalist else None,
        "decisions": {"T1": d1, "T2": d2},
    }
    _write_json(ROOT / "results" / "plane2_train_search_adjudication.json", out)
    print(json.dumps(out, indent=2), flush=True)
    return 0


def cmd_adjudicate_confirm(args: argparse.Namespace) -> int:
    """Live Confirm ledger adjudication after contemporaneous Confirm measurements complete."""
    anchor_path = Path(args.anchor_file).resolve(strict=True)
    enforce_hermetic_pre_trial_hygiene(anchor_path, root=ROOT, verify_manifest=True)
    scored_dir = ROOT / "results" / "scored"
    ledger_path = ROOT / "results" / "governed_ledger_v2.jsonl"

    search_adj = json.loads((ROOT / "results" / "plane2_train_search_adjudication.json").read_text(encoding="utf-8"))
    winner_arm = search_adj.get("selected_finalist_arm")
    decisions: list[dict[str, Any]] = [search_adj["decisions"]["T1"], search_adj["decisions"]["T2"]]

    d3 = None
    if winner_arm is not None:
        d3 = _run_gate_decision(
            trial_id=f"trial_03_train_confirm_{winner_arm.lower()}",
            hypothesis=f"Contemporaneous 10-block AB/BA confirmation of ledger-admitted Search finalist {winner_arm} vs baseline T0 on 8x A100 DDP.",
            cand_json=scored_dir / f"p2_train_confirm_{winner_arm}.json",
            inc_json=scored_dir / "p2_train_confirm_T0.json",
            base_json=scored_dir / "p2_train_confirm_T0.json",
            rules_json=ROOT / "configs" / "gate_rules_train_systems_confirm.json",
            ledger_path=ledger_path,
            anchor_path=anchor_path,
        )
        decisions.append(d3)

    d4 = _run_gate_decision(
        trial_id="trial_04_serving_confirm_s3_int4_tp1_mtp3",
        hypothesis="1-GPU GPTQ W4A16 + Repaired Native BF16 MTP3 (S3) cuts assigned GPU-seconds by >=310 GPU-s and wall time to <=0.90x vs 2-GPU BF16 TP=2 (S0) while preserving accuracy within strict 2.0 pp KL-Hoeffding NI on math_confirm1280.",
        cand_json=scored_dir / "p2_serving_williams12_S3.json",
        inc_json=scored_dir / "p2_serving_williams12_S0.json",
        base_json=scored_dir / "p2_serving_williams12_S0.json",
        rules_json=ROOT / "configs" / "gate_rules_dense_serving_confirm.json",
        ledger_path=ledger_path,
        anchor_path=anchor_path,
        quality_cand=scored_dir / "p2_quality_confirm1280_S3.json",
        quality_inc=scored_dir / "p2_quality_confirm1280_S0.json",
    )
    decisions.append(d4)

    anchor = json.loads(anchor_path.read_text(encoding="utf-8"))
    v_proc = subprocess.run(
        [
            sys.executable,
            "-I",
            "-B",
            str(ROOT / "eval" / "governed_loop_guard.py"),
            "verify-ledger",
            "--ledger",
            str(ledger_path),
            "--expected-head-sha256",
            str(anchor["ledger_head_sha256"]),
        ],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        check=True,
    )
    train_promoted = bool(winner_arm is not None and d3 is not None and d3.get("decision") == "PROMOTE")
    serving_promoted = bool(d4.get("decision") == "PROMOTE")
    out = {
        "status": "LEDGER_ADJUDICATED_AND_VERIFIED",
        "protocol_epoch": anchor["protocol_epoch"],
        "manifest_root_sha256": anchor["manifest_root_sha256"],
        "ledger_head_sha256_after_confirm": anchor["ledger_head_sha256"],
        "serving_confirm_entry_sha256": d4.get("entry_sha256"),
        "serving_confirm_rules_sha256": _file_sha256(ROOT / "configs" / "gate_rules_dense_serving_confirm.json"),
        "ledger_verification": json.loads(v_proc.stdout),
        "train_search_selected_finalist": winner_arm,
        "promoted_train_winner": TRAIN_ARMS[winner_arm]["candidate_id"] if train_promoted else "T0_ddp8_micro1_nobucket",
        "train_winner_promoted": train_promoted,
        "promoted_serving_winner": SERVING_ARMS["S3"]["candidate_id"] if serving_promoted else "S0_bf16_tp2",
        "serving_s3_promoted": serving_promoted,
        "decisions": decisions,
    }
    _write_json(ROOT / "results" / "plane2_adjudication_summary.json", out)
    print(json.dumps(out, indent=2), flush=True)
    return 0


def verify_single_touch_holdout_authorization(
    *,
    root: Path | None = None,
    anchor_path: Path | None = None,
    vault_dir: Path | None = None,
) -> dict[str, Any]:
    """
    Fail-closed pre-GPU / pre-label authorization check for `cmd_single_touch_holdout`:
      1. Requires external `vault_dir` outside `repo_root` with untouched holdout registry.
      2. Verifies live `IMMUTABLE_MANIFEST.json` against `anchor_path`.
      3. Verifies hash-chained `results/governed_ledger_v2.jsonl` via `governed_loop_guard.py verify-ledger`
         against `anchor["ledger_head_sha256"]`.
      4. Requires strict boolean `adj["serving_s3_promoted"]` (`type(val) is bool` — string `'false'` is rejected!)
         plus exact matching `protocol_epoch`, `manifest_root_sha256`, `ledger_head_sha256_after_confirm`, and
         `serving_confirm_entry_sha256`.
      5. Reads the verified ledger directly to inspect the exact single `loop_dense_serving_confirm_v2` entry
         (`trial_id == "trial_04_serving_confirm_s3_int4_tp1_mtp3"`,
         `candidate_id == "S3_int4_w4a16_tp1_mtp3_repaired"`):
           - If verified ledger decision is `PROMOTE` and `serving_s3_promoted is True` -> `EXECUTE_SINGLE_TOUCH`.
           - If verified ledger decision is NOT `PROMOTE` and `serving_s3_promoted is False` -> `SKIP_NOT_PROMOTED`.
           - Any missing, stale, malformed, tampered, or contradictory state raises
             `RuntimeError("HOLDOUT_UNAUTHORIZED: ...")` BEFORE launching any request or touching vault labels!
    """
    repo_root = (root if root is not None else ROOT).resolve(strict=True)
    anch_file = (anchor_path if anchor_path is not None else _default_anchor(repo_root)).resolve()
    vault = (vault_dir if vault_dir is not None else DEFAULT_VAULT).resolve()

    if repo_root in vault.parents or vault == repo_root:
        raise RuntimeError("HOLDOUT_UNAUTHORIZED: holdout vault must reside strictly outside workspace ROOT")
    if not (vault / "math_holdout512_labels.json").is_file():
        raise RuntimeError(f"HOLDOUT_UNAUTHORIZED: missing external holdout labels in {vault}")
    if (vault / "holdout_touch_registry.json").exists():
        raise RuntimeError("HOLDOUT_UNAUTHORIZED: holdout_touch_registry.json already exists (single-touch cap)")
    if (repo_root / "results" / "single_touch_holdout_v2.json").exists():
        raise RuntimeError("HOLDOUT_UNAUTHORIZED: single_touch_holdout_v2.json already exists")

    if not anch_file.is_file():
        raise RuntimeError(f"HOLDOUT_UNAUTHORIZED: missing governance anchor {anch_file}")

    try:
        enforce_hermetic_pre_trial_hygiene(anch_file, root=repo_root, verify_manifest=True)
    except Exception as exc:
        raise RuntimeError(f"HOLDOUT_UNAUTHORIZED: pre-holdout manifest/tree check failed: {exc}") from exc

    anchor = json.loads(anch_file.read_text(encoding="utf-8"))
    ledger_path = repo_root / "results" / "governed_ledger_v2.jsonl"
    adj_path = repo_root / "results" / "plane2_adjudication_summary.json"
    rules_path = repo_root / "configs" / "gate_rules_dense_serving_confirm.json"
    missing = [str(p) for p in (ledger_path, adj_path, rules_path) if not p.is_file()]
    if missing:
        raise RuntimeError(f"HOLDOUT_UNAUTHORIZED: missing Confirm governance artifacts: {missing}")

    adj = json.loads(adj_path.read_text(encoding="utf-8"))
    promoted_flag = adj.get("serving_s3_promoted")
    if type(promoted_flag) is not bool:
        raise RuntimeError(f"HOLDOUT_UNAUTHORIZED: serving_s3_promoted must be strict bool, got {promoted_flag!r}")
    if adj.get("status") != "LEDGER_ADJUDICATED_AND_VERIFIED":
        raise RuntimeError(f"HOLDOUT_UNAUTHORIZED: invalid adjudication status {adj.get('status')!r}")
    if adj.get("manifest_root_sha256") != anchor.get("manifest_root_sha256"):
        raise RuntimeError("HOLDOUT_UNAUTHORIZED: adjudication manifest_root_sha256 does not match active anchor")
    if adj.get("protocol_epoch") != anchor.get("protocol_epoch"):
        raise RuntimeError("HOLDOUT_UNAUTHORIZED: adjudication protocol_epoch does not match active anchor")
    if adj.get("ledger_head_sha256_after_confirm") != anchor.get("ledger_head_sha256"):
        raise RuntimeError("HOLDOUT_UNAUTHORIZED: adjudication ledger head does not match active anchor")

    expected_head = str(anchor.get("ledger_head_sha256", ""))
    expected_count = int(anchor.get("ledger_entry_count", -1))
    if expected_count < 0:
        raise RuntimeError(f"HOLDOUT_UNAUTHORIZED: invalid anchor ledger_entry_count={expected_count}")
    if expected_count > 0 or ledger_path.is_file() or expected_head != "GENESIS":
        if not ledger_path.is_file():
            raise RuntimeError("HOLDOUT_UNAUTHORIZED: missing ledger file for non-GENESIS anchor")
        v_proc = subprocess.run(
            [
                sys.executable,
                "-I",
                "-B",
                str(repo_root / "eval" / "governed_loop_guard.py"),
                "verify-ledger",
                "--ledger",
                str(ledger_path),
                "--expected-head-sha256",
                expected_head,
            ],
            cwd=str(repo_root),
            capture_output=True,
            text=True,
            check=False,
        )
        if v_proc.returncode != 0:
            raise RuntimeError(f"HOLDOUT_UNAUTHORIZED: ledger verification failed: {v_proc.stdout} {v_proc.stderr}")
        v_led = json.loads(v_proc.stdout)
        if int(v_led.get("entries", -1)) != expected_count:
            raise RuntimeError(f"HOLDOUT_UNAUTHORIZED: ledger entry count mismatch {v_led}")
        ledger_rows = _read_jsonl(ledger_path)
    else:
        ledger_rows = []

    rules_sha = _file_sha256(rules_path)
    if "serving_confirm_rules_sha256" in adj and adj["serving_confirm_rules_sha256"] != rules_sha:
        raise RuntimeError("HOLDOUT_UNAUTHORIZED: serving_confirm_rules_sha256 mismatch vs locked rules file")

    srv_rows = [r for r in ledger_rows if r.get("protocol_id") == "loop_dense_serving_confirm_v2"]
    if len(srv_rows) > 1:
        raise RuntimeError(f"HOLDOUT_UNAUTHORIZED: duplicate loop_dense_serving_confirm_v2 rows ({len(srv_rows)})")

    if len(srv_rows) == 0:
        # Allowed ONLY when strict_guard pre-tripwire rejected S3 before ledger write AND promoted_flag is False
        if promoted_flag is True or adj.get("serving_confirm_entry_sha256") is not None:
            raise RuntimeError("HOLDOUT_UNAUTHORIZED: missing loop_dense_serving_confirm_v2 ledger row for claimed promotion")
        return {
            "action": "SKIP_NOT_PROMOTED",
            "ledger_decision": "PRE_TRIPWIRE_ROLLBACK",
            "serving_confirm_entry_sha256": None,
            "manifest_root_sha256": anchor["manifest_root_sha256"],
            "ledger_head_sha256": expected_head,
        }

    s_row = srv_rows[0]
    if s_row.get("trial_id") != "trial_04_serving_confirm_s3_int4_tp1_mtp3":
        raise RuntimeError(f"HOLDOUT_UNAUTHORIZED: unexpected serving Confirm trial_id {s_row.get('trial_id')!r}")
    if s_row.get("candidate_id") != SERVING_ARMS["S3"]["candidate_id"]:
        raise RuntimeError(f"HOLDOUT_UNAUTHORIZED: unexpected serving Confirm candidate_id {s_row.get('candidate_id')!r}")
    if s_row.get("decision_stage") != "confirm" or s_row.get("primary_metric") != "assigned_gpu_seconds":
        raise RuntimeError("HOLDOUT_UNAUTHORIZED: ledger row stage/metric mismatch for serving Confirm")
    if adj.get("serving_confirm_entry_sha256") != s_row.get("entry_sha256"):
        raise RuntimeError("HOLDOUT_UNAUTHORIZED: serving_confirm_entry_sha256 mismatch vs verified ledger")

    ledger_promoted = s_row.get("decision") == "PROMOTE"
    if promoted_flag != ledger_promoted:
        raise RuntimeError(
            f"HOLDOUT_UNAUTHORIZED: summary serving_s3_promoted={promoted_flag!r} contradicts verified ledger decision={s_row.get('decision')!r}"
        )

    return {
        "action": "EXECUTE_SINGLE_TOUCH" if ledger_promoted else "SKIP_NOT_PROMOTED",
        "ledger_decision": s_row.get("decision"),
        "serving_confirm_entry_sha256": s_row["entry_sha256"],
        "manifest_root_sha256": anchor["manifest_root_sha256"],
        "ledger_head_sha256": anchor["ledger_head_sha256"],
    }


def _paired_bootstrap_ci95(c_vals: dict[str, float], r_vals: dict[str, float], n_boot: int = 10000, seed: int = 20261117) -> dict[str, float]:
    keys = sorted(c_vals.keys())
    deltas = [c_vals[k] - r_vals[k] for k in keys]
    n = len(deltas)
    rng = random.Random(seed)
    means = sorted(math.fsum(deltas[rng.randrange(n)] for _ in range(n)) / n for _ in range(n_boot))
    return {
        "obs_paired_delta": math.fsum(deltas) / n,
        "ci95_low": means[int(0.025 * n_boot)],
        "ci95_high": means[int(0.975 * n_boot)],
    }


def _compute_kl_ni_via_strict_guard(
    sbx: HermeticTrialSandbox,
    v_cand: dict[str, float],
    v_inc: dict[str, float],
    clus: dict[str, str],
    alpha_one_sided: float = 0.025,
) -> dict[str, Any]:
    in_path = sbx.caches / "ni_input.json"
    in_path.write_text(
        json.dumps({"cand": v_cand, "inc": v_inc, "clus": clus, "alpha": alpha_one_sided}),
        encoding="utf-8",
    )
    helper = (
        "import json, pathlib, sys\n"
        "sys.path.insert(0, 'eval')\n"
        "from strict_guard import conservative_paired_cluster_ni_lcb\n"
        "d = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding='utf-8'))\n"
        "print(json.dumps(conservative_paired_cluster_ni_lcb(d['cand'], d['inc'], d['clus'], alpha_one_sided=float(d['alpha']))))\n"
    )
    proc = subprocess.run(
        [sys.executable, "-I", "-B", "-c", helper, str(in_path)],
        cwd=str(sbx.repo),
        env=sbx.env,
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(proc.stdout)


def cmd_single_touch_holdout(args: argparse.Namespace) -> int:
    anchor_path = Path(args.anchor_file).resolve()
    vault = Path(args.vault_dir).resolve()
    auth = verify_single_touch_holdout_authorization(root=ROOT, anchor_path=anchor_path, vault_dir=vault)

    if auth["action"] == "SKIP_NOT_PROMOTED":
        untouched = {
            "status": "SKIPPED_NO_SERVING_PROMOTION",
            "authorization": auth,
            "note": "Verified ledger did not PROMOTE S3 at Confirm; Protected Holdout remains completely untouched (0 touches).",
        }
        _write_json(ROOT / "results" / "single_touch_holdout_v2.json", untouched)
        print(json.dumps(untouched, indent=2), flush=True)
        return 0

    scored_dir = ROOT / "results" / "scored"
    ds_holdout = ROOT / "data" / "splits" / "math_holdout512_prompts.jsonl"
    lbl_path = vault / "math_holdout512_labels.json"
    reg_path = vault / "holdout_touch_registry.json"
    h_env = "node0_8xa100_sxm4_80gb_vllm029_qwen38_27b_holdout512_cap32768_medium_s20261117_hermetic"

    s0_out = scored_dir / "holdout512_S0.json"
    s3_out = scored_dir / "holdout512_S3.json"
    s0_sum = run_sharded_panel_inference(
        run_prefix="holdout512_S0",
        dataset_path=ds_holdout,
        arm_key="S0",
        purpose="single_touch_protected_holdout_v2",
        env_fingerprint=h_env,
        scored_output=s0_out,
        anchor_path=anchor_path,
        external_labels=lbl_path,
        holdout_registry=reg_path,
    )
    s3_sum = run_sharded_panel_inference(
        run_prefix="holdout512_S3",
        dataset_path=ds_holdout,
        arm_key="S3",
        purpose="single_touch_protected_holdout_v2",
        env_fingerprint=h_env,
        scored_output=s3_out,
        anchor_path=anchor_path,
        external_labels=lbl_path,
        holdout_registry=reg_path,
    )

    p0 = json.loads(s0_out.read_text(encoding="utf-8"))
    p3 = json.loads(s3_out.read_text(encoding="utf-8"))
    v0 = {r["id"]: float(r["value"]) for r in p0["items"]}
    v3 = {r["id"]: float(r["value"]) for r in p3["items"]}
    clus = {r["id"]: str(r["cluster_id"]) for r in p3["items"]}
    with HermeticTrialSandbox(ROOT, "holdout_ni_stats", anchor_path=anchor_path, verify_manifest=True) as sbx:
        kl_ni = _compute_kl_ni_via_strict_guard(sbx, v3, v0, clus, alpha_one_sided=0.025)
    boot_ci = _paired_bootstrap_ci95(v3, v0)
    wins = sum(1 for k in v0 if v3[k] > 0.5 and v0[k] < 0.5)
    losses = sum(1 for k in v0 if v3[k] < 0.5 and v0[k] > 0.5)
    ties = len(v0) - wins - losses
    reg_state = json.loads(reg_path.read_text(encoding="utf-8"))

    report = {
        "status": "SINGLE_TOUCH_PROTECTED_HOLDOUT_COMPLETE",
        "panel": "data/splits/math_holdout512_prompts.jsonl",
        "panel_sha256": _file_sha256(ds_holdout),
        "n_problems": len(v0),
        "holdout_touch_registry": reg_state,
        "untouched_holdout_panels": [
            "data/splits/dapo_holdout200_prompts.jsonl (0 touches; inner DAPO loop Step-0 NO_GO)",
            "data/splits/retention_holdout150_prompts.jsonl (0 touches; inner DAPO loop Step-0 NO_GO)",
        ],
        "S0_bf16_tp2_baseline": s0_sum,
        "S3_int4_w4a16_tp1_mtp3_promoted": s3_sum,
        "paired_outcomes": {"wins": wins, "losses": losses, "ties": ties},
        "paired_cluster_bootstrap_95ci": boot_ci,
        "kl_hoeffding_one_sided_975_ni": kl_ni,
    }
    _write_json(ROOT / "results" / "single_touch_holdout_v2.json", report)
    print(json.dumps(report, indent=2), flush=True)
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Frozen Plane-2 Governed Loop & Single-Touch Holdout Runner")
    sub = p.add_subparsers(dest="cmd", required=True)

    p_v = sub.add_parser("stage-holdout-vault")
    p_v.add_argument("--vault-dir", default=str(DEFAULT_VAULT))
    p_v.add_argument("--remove-workspace-copy", action="store_true")
    p_v.set_defaults(func=cmd_stage_holdout_vault)

    p_cta = sub.add_parser("calibrate-train-aa-hermetic")
    p_cta.set_defaults(func=cmd_calibrate_train_aa_hermetic)

    p_csa = sub.add_parser("calibrate-serving-aa-hermetic")
    p_csa.set_defaults(func=cmd_calibrate_serving_aa_hermetic)

    p_ts = sub.add_parser("worker-train-search")
    p_ts.add_argument("--anchor-file", default=str(_default_anchor()))
    p_ts.set_defaults(func=cmd_worker_train_search)

    p_ats = sub.add_parser("adjudicate-train-search")
    p_ats.add_argument("--anchor-file", default=str(_default_anchor()))
    p_ats.set_defaults(func=cmd_adjudicate_train_search)

    p_tc = sub.add_parser("worker-train-confirm")
    p_tc.add_argument("--finalist", required=True, choices=["T1", "T2"])
    p_tc.add_argument("--anchor-file", default=str(_default_anchor()))
    p_tc.set_defaults(func=cmd_worker_train_confirm)

    p_w = sub.add_parser("worker-serving-williams12")
    p_w.add_argument("--anchor-file", default=str(_default_anchor()))
    p_w.set_defaults(func=cmd_worker_serving_williams12)

    p_q = sub.add_parser("worker-serving-quality1280")
    p_q.add_argument("--anchor-file", default=str(_default_anchor()))
    p_q.set_defaults(func=cmd_worker_serving_quality1280)

    p_ac = sub.add_parser("adjudicate-confirm")
    p_ac.add_argument("--anchor-file", default=str(_default_anchor()))
    p_ac.set_defaults(func=cmd_adjudicate_confirm)

    p_h = sub.add_parser("single-touch-holdout")
    p_h.add_argument("--anchor-file", default=str(_default_anchor()))
    p_h.add_argument("--vault-dir", default=str(DEFAULT_VAULT))
    p_h.set_defaults(func=cmd_single_touch_holdout)

    return p


def main() -> int:
    args = build_parser().parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
