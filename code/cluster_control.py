"""Identity-verified remote keepalive & stale-process lifecycle management on the authorized node."""
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
KEEPALIVE = Path("/scratch/copilot-mi300x-keepalive")
TMUX_SESSION = "copilot-mi300x-keepalive"


def log_event(event: str, **details) -> dict:
    payload = {"utc": datetime.now(timezone.utc).isoformat(), "node": os.uname().nodename, "event": event, **details}
    out_dir = ROOT / "runs" / "cluster_control"
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "events.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(payload) + "\n")
    print(json.dumps(payload), flush=True)
    return payload


def proc_identity(pid: int) -> dict | None:
    proc = Path("/proc") / str(pid)
    if not (proc / "stat").is_file():
        return None
    try:
        stat_text = (proc / "stat").read_text(encoding="utf-8")
        start_ticks = stat_text.rsplit(") ", 1)[1].split()[19]
        cmdline = (proc / "cmdline").read_bytes().replace(b"\0", b" ").decode("utf-8", "replace").strip()
        cwd = os.readlink(proc / "cwd") if (proc / "cwd").exists() else None
        return {"pid": pid, "start_ticks": start_ticks, "cmdline": cmdline, "cwd": cwd}
    except (FileNotFoundError, ProcessLookupError, PermissionError, IndexError):
        return None


def quiesce_keepalive(stop_verified_stale_pid: int | None = None,
                      expected_stale_cwd_substr: str | None = None,
                      expected_stale_cmd_substr: str | None = None) -> dict:
    actions = []
    sup_pid_file = KEEPALIVE / "supervisor.pid"
    sup_start_file = KEEPALIVE / "supervisor.starttime"
    if sup_pid_file.is_file() and sup_start_file.is_file():
        pid = int(sup_pid_file.read_text().strip())
        expected_ticks = sup_start_file.read_text().strip()
        ident = proc_identity(pid)
        if ident and ident["start_ticks"] == expected_ticks and str(KEEPALIVE / "remote_supervisor.sh") in ident["cmdline"]:
            os.kill(pid, signal.SIGTERM)
            for _ in range(40):
                if proc_identity(pid) is None:
                    break
                time.sleep(0.25)
            actions.append(log_event("STOPPED_VERIFIED_KEEPALIVE", **ident))
        else:
            actions.append(log_event("KEEPALIVE_IDENTITY_SKIP", pid=pid, observed=ident))

    if stop_verified_stale_pid is not None:
        ident = proc_identity(stop_verified_stale_pid)
        if ident is not None:
            if "jupyter" in ident["cmdline"].lower():
                raise RuntimeError("Refusing to touch Singularity jupyter daemon")
            cwd_ok = expected_stale_cwd_substr and ident["cwd"] and expected_stale_cwd_substr in ident["cwd"]
            cmd_ok = expected_stale_cmd_substr and expected_stale_cmd_substr in ident["cmdline"]
            if not (cwd_ok and cmd_ok):
                raise RuntimeError(f"Stale PID identity mismatch: {ident}")
            os.kill(stop_verified_stale_pid, signal.SIGTERM)
            for _ in range(40):
                if proc_identity(stop_verified_stale_pid) is None:
                    break
                time.sleep(0.25)
            actions.append(log_event("STOPPED_VERIFIED_STALE_PROCESS", **ident))

    # Wait up to 20s for all 8 GPUs to report zero compute apps
    for _ in range(20):
        apps = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader"],
            capture_output=True, text=True, check=True, timeout=15,
        ).stdout.strip()
        if not apps:
            break
        time.sleep(1.0)
    apps = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader"],
        capture_output=True, text=True, check=True, timeout=15,
    ).stdout.strip()
    return log_event("NODE_QUIESCED_STATUS", remaining_compute_apps=apps, actions_count=len(actions))


def restore_keepalive(profile: str = "medium") -> dict:
    if not (KEEPALIVE / "remote_supervisor.sh").is_file():
        raise RuntimeError(f"Missing keepalive supervisor at {KEEPALIVE / 'remote_supervisor.sh'}")
    # Ensure no compute apps remain before starting idle keepalive
    apps = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader"],
        capture_output=True, text=True, check=True, timeout=15,
    ).stdout.strip()
    sup_pid_file = KEEPALIVE / "supervisor.pid"
    sup_start_file = KEEPALIVE / "supervisor.starttime"
    already_alive = False
    if sup_pid_file.is_file() and sup_start_file.is_file():
        pid = int(sup_pid_file.read_text().strip())
        ident = proc_identity(pid)
        if ident and ident["start_ticks"] == sup_start_file.read_text().strip():
            already_alive = True
    if not already_alive:
        # Remove stale tmux session if dead, then start fresh
        subprocess.run(["tmux", "kill-session", "-t", TMUX_SESSION], capture_output=True)
        cmd = [
            "tmux", "new-session", "-d", "-s", TMUX_SESSION,
            f"exec env KEEPALIVE_IDLE_PROFILE={profile} KEEPALIVE_INITIAL_IDLE_GRACE_SECONDS=5 "
            f"KEEPALIVE_IDLE_GRACE_SECONDS=5 bash '{KEEPALIVE}/remote_supervisor.sh'"
        ]
        subprocess.run(cmd, check=True, timeout=15)
    # Poll up to 60s for worker_pid and GPU occupancy on all 8 GPUs
    state_text = ""
    gpu_rows = []
    for _ in range(30):
        time.sleep(2.0)
        if (KEEPALIVE / "supervisor.state").is_file():
            state_text = (KEEPALIVE / "supervisor.state").read_text(encoding="utf-8").strip()
        smi = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,memory.used,utilization.gpu", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=True, timeout=15,
        ).stdout.strip().splitlines()
        gpu_rows = []
        for line in smi:
            parts = [p.strip() for p in line.split(",")]
            gpu_rows.append({"gpu": int(parts[0]), "memory_mib": int(parts[1]), "util_pct": int(parts[2])})
        if len(gpu_rows) == 8 and all(g["memory_mib"] >= 1000 for g in gpu_rows):
            break
    all_busy = len(gpu_rows) == 8 and all(g["memory_mib"] >= 1000 for g in gpu_rows)
    if not all_busy:
        raise RuntimeError(f"Keepalive failed to occupy all 8 GPUs: state={state_text!r}, gpus={gpu_rows}, prior_apps={apps!r}")
    return log_event("KEEPALIVE_RESTORED_AND_VERIFIED", supervisor_state=state_text, gpus=gpu_rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    p_q = sub.add_parser("quiesce")
    p_q.add_argument("--stale-pid", type=int)
    p_q.add_argument("--stale-cwd-substr")
    p_q.add_argument("--stale-cmd-substr")
    p_r = sub.add_parser("restore-keepalive")
    p_r.add_argument("--profile", default="medium")
    args = parser.parse_args()
    if args.cmd == "quiesce":
        quiesce_keepalive(args.stale_pid, args.stale_cwd_substr, args.stale_cmd_substr)
    else:
        restore_keepalive(args.profile)


if __name__ == "__main__":
    main()
