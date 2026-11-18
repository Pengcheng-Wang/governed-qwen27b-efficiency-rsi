"""Bounded node-local stage runner with durable status and explicit failure records."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import re
import signal
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--timeout", type=int, required=True)
    parser.add_argument("--gpus", type=int, default=0)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if not re.fullmatch(r"[a-zA-Z0-9_-]+", args.name) or args.timeout <= 0:
        raise ValueError("A unique simple stage name and positive timeout are required")
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        raise ValueError("A command is required")
    root = args.root.resolve()
    directory = root / "runs" / args.name
    directory.mkdir(parents=True, exist_ok=False)
    env = os.environ.copy()
    for key in ("RANK", "WORLD_SIZE", "LOCAL_RANK", "LOCAL_WORLD_SIZE", "NODE_RANK",
                "NODE_COUNT", "MASTER_ADDR", "MASTER_PORT", "GROUP_RANK", "ROLE_RANK"):
        env.pop(key, None)
    for name in ("cache", "tmp", "results"):
        (root / name).mkdir(exist_ok=True)
    env.update({
        "HF_HOME": str(root / "cache" / "huggingface"),
        "UV_CACHE_DIR": str(root / "cache" / "uv"),
        "TRITON_CACHE_DIR": str(root / "cache" / "triton"),
        "TORCHINDUCTOR_CACHE_DIR": str(root / "cache" / "torchinductor"),
        "TORCH_EXTENSIONS_DIR": str(root / "cache" / "torch_extensions"),
        "VLLM_CACHE_ROOT": str(root / "cache" / "vllm"),
        "XDG_CACHE_HOME": str(root / "cache"),
        "CUDA_CACHE_PATH": str(root / "cache" / "cuda"),
        "TMPDIR": str(root / "tmp"),
        "TOKENIZERS_PARALLELISM": "false",
        "PYTHONUNBUFFERED": "1",
    })
    cudnn = Path(importlib.metadata.distribution("nvidia-cudnn-cu12").locate_file("nvidia/cudnn"))
    if not (cudnn / "lib" / "libcudnn.so.9").is_file():
        raise RuntimeError("The vendor cuDNN library is missing")
    env["CUDNN_HOME"] = str(cudnn)
    env["LD_LIBRARY_PATH"] = str(cudnn / "lib") + ":" + env.get("LD_LIBRARY_PATH", "")
    state = {
        "name": args.name,
        "status": "starting",
        "supervisor_pid": os.getpid(),
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "assigned_gpus": args.gpus,
        "command": command,
    }
    start = time.monotonic()
    received_signal = None

    def on_signal(number, _frame):
        nonlocal received_signal
        received_signal = number

    signal.signal(signal.SIGTERM, on_signal)
    signal.signal(signal.SIGINT, on_signal)
    with (directory / "output.log").open("wb") as log:
        child = subprocess.Popen(
            command, cwd=root, env=env, stdin=subprocess.DEVNULL,
            stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
        )
        state.update(status="running", child_pid=child.pid)
        stop_requested_at = None
        while child.poll() is None:
            elapsed = time.monotonic() - start
            if (received_signal is not None or elapsed > args.timeout) and stop_requested_at is None:
                state["stop_reason"] = (
                    f"signal_{received_signal}" if received_signal is not None else "timeout"
                )
                stop_requested_at = time.monotonic()
                os.killpg(child.pid, signal.SIGTERM)
            elif stop_requested_at is not None and time.monotonic() - stop_requested_at > 60:
                if child.poll() is None:
                    os.killpg(child.pid, signal.SIGKILL)
            state.update(
                elapsed_seconds=elapsed,
                heartbeat_utc=datetime.now(timezone.utc).isoformat(),
            )
            write_json(directory / "status.json", state)
            time.sleep(2)
        elapsed = time.monotonic() - start
        state.update(
            status="completed" if child.returncode == 0 and not state.get("stop_reason") else "failed",
            exit_code=child.returncode,
            elapsed_seconds=elapsed,
            assigned_gpu_hours=elapsed * args.gpus / 3600,
            ended_at_utc=datetime.now(timezone.utc).isoformat(),
        )
        write_json(directory / "status.json", state)
    print(json.dumps(state))
    raise SystemExit(0 if state["status"] == "completed" else 1)


if __name__ == "__main__":
    main()
