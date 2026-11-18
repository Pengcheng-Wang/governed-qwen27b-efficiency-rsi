"""Low-frequency board-power and memory sampling; no synthetic load or clock changes."""
from __future__ import annotations

import csv
import io
import subprocess
import threading
import time


def integrate_power(samples: list[dict]) -> dict:
    if len(samples) < 2:
        return {"board_joules": None, "sampled_seconds": 0.0, "samples": len(samples)}
    joules, seconds = 0.0, 0.0
    for previous, current in zip(samples, samples[1:]):
        delta = current["t"] - previous["t"]
        if delta < 0:
            raise ValueError("Power samples are not monotonic")
        seconds += delta
        joules += delta * (previous["total_watts"] + current["total_watts"]) / 2
    return {"board_joules": joules, "sampled_seconds": seconds, "samples": len(samples)}


class BoardTelemetry:
    def __init__(self, physical_gpu_ids: list[int], interval: float = 1.0):
        if not physical_gpu_ids:
            raise ValueError("Telemetry needs the actual participating physical GPU IDs")
        self.ids = physical_gpu_ids
        self.interval = interval
        self.samples: list[dict] = []
        self.errors: list[dict] = []
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._loop, daemon=True)

    def _sample(self) -> None:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--id=" + ",".join(map(str, self.ids)),
                "--query-gpu=index,power.draw,memory.used,utilization.gpu,clocks.sm",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        timestamp = time.monotonic()
        if result.returncode:
            self.errors.append({"t": timestamp, "error": result.stderr})
            return
        try:
            devices = [
                {
                    "gpu": int(row[0]),
                    "watts": float(row[1]),
                    "memory_mib": float(row[2]),
                    "utilization_percent": float(row[3]),
                    "sm_clock_mhz": float(row[4]),
                }
                for row in csv.reader(io.StringIO(result.stdout))
            ]
        except (ValueError, IndexError) as error:
            self.errors.append({"t": timestamp, "error": str(error), "raw": result.stdout})
            return
        if {device["gpu"] for device in devices} != set(self.ids):
            self.errors.append({"t": timestamp, "error": "GPU inventory mismatch"})
            return
        self.samples.append({
            "t": timestamp,
            "total_watts": sum(x["watts"] for x in devices),
            "devices": devices,
        })

    def _loop(self) -> None:
        while not self.stop_event.wait(self.interval):
            try:
                self._sample()
            except subprocess.TimeoutExpired as error:
                self.errors.append({"t": time.monotonic(), "error": str(error)})

    def start(self) -> None:
        self._sample()
        self.thread.start()

    def stop(self) -> dict:
        self.stop_event.set()
        self.thread.join(timeout=15)
        if self.thread.is_alive():
            raise RuntimeError("Power sampler did not stop")
        self._sample()
        return {
            **integrate_power(self.samples),
            "errors": self.errors,
            "scope": "Participating GPU boards only, not whole-machine energy",
            "power_samples": self.samples,
        }
