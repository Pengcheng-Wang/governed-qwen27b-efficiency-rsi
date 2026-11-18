"""Scoped lifecycle helpers for experiment-owned, bounded vLLM servers."""
from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import httpx

from stage import write_json

ROOT = Path(__file__).resolve().parents[1]


class DeviceBusyError(RuntimeError):
    pass


def require_idle(devices: str) -> None:
    result = subprocess.run(
        ["nvidia-smi", f"--id={devices}", "--query-compute-apps=pid", "--format=csv,noheader"],
        capture_output=True,
        text=True,
        timeout=20,
        check=True,
    )
    if result.stdout.strip():
        raise DeviceBusyError(f"Requested devices still have compute processes: {result.stdout.strip()}")


def wait_idle(devices: str, timeout: int = 120) -> None:
    deadline = time.monotonic() + timeout
    while True:
        try:
            require_idle(devices)
            return
        except DeviceBusyError as error:
            if time.monotonic() >= deadline:
                raise TimeoutError(f"Devices did not drain; no unrelated process was signaled: {error}") from error
            time.sleep(1)


def retire(name: str, reason: str) -> None:
    directory = ROOT / "runs" / name
    if not (directory / "status.json").is_file():
        return
    state = json.loads((directory / "status.json").read_text(encoding="utf-8"))
    if "vllm.entrypoints.cli.main" not in state["command"] or "serve" not in state["command"]:
        raise RuntimeError("Refusing to retire a non-vLLM stage")
    if state["status"] not in {"running", "starting"}:
        return
    pid = state["supervisor_pid"]
    proc = Path("/proc") / str(pid) / "cmdline"
    if proc.is_file():
        command = proc.read_bytes().decode("utf-8", "replace").split("\0")
        if str(ROOT / "code/stage.py") not in command or "--name" not in command:
            raise RuntimeError("Server supervisor ownership changed")
        if command[command.index("--name") + 1] != name:
            raise RuntimeError("Server supervisor name changed")
        write_json(directory / "retired.json", {"reason": reason})
        os.kill(pid, signal.SIGTERM)
    for _ in range(120):
        state = json.loads((directory / "status.json").read_text(encoding="utf-8"))
        if state["status"] not in {"running", "starting"}:
            return
        time.sleep(1)
    raise TimeoutError("The owned server did not stop; no unrelated processes were signaled")


def start(name: str, command: list[str], gpu_count: int, port: int, ttl: int = 7200) -> None:
    if (ROOT / "runs" / name).exists():
        raise RuntimeError(f"Stage already exists: {name}; inspect before relaunch")
    invocation = [
        sys.executable,
        str(ROOT / "code/stage.py"),
        "--root",
        str(ROOT),
        "--name",
        name,
        "--timeout",
        str(ttl),
        "--gpus",
        str(gpu_count),
        "--",
        *command,
    ]
    (ROOT / "logs").mkdir(exist_ok=True)
    with (ROOT / "logs" / f"{name}.supervisor.log").open("wb") as log:
        process = subprocess.Popen(
            invocation,
            cwd=ROOT,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    deadline = time.monotonic() + 900
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"Server failed before readiness: {name}")
        try:
            response = httpx.get(f"http://127.0.0.1:{port}/health", timeout=3)
            if response.status_code == 200:
                return
        except httpx.TransportError:
            pass
        time.sleep(5)
    retire(name, "Readiness timed out; retain the failure and release only this owned server")
    raise TimeoutError(f"Server readiness timeout: {name}; inspect its owned stage")
