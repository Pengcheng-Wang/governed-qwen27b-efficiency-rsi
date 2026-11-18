"""Download and LFS-verify official Qwen/Qwen3.8-27B and Qwen/Qwen3.5-35B-A3B checkpoints."""
from __future__ import annotations

import argparse
import collections
import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download
from safetensors import safe_open

from common import file_hash
from stage import write_json

ROOT = Path(__file__).resolve().parents[1]


def verify_and_download(spec_path: Path, manifest_out: Path) -> dict:
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    destination = ROOT / spec["model_directory"]
    info = HfApi().model_info(spec["repo_id"], revision=spec["revision"], files_metadata=True)
    if info.sha != spec["revision"]:
        raise RuntimeError(f"Resolved revision {info.sha} != pinned {spec['revision']}")
    started = time.monotonic()
    snapshot_download(
        spec["repo_id"],
        revision=info.sha,
        local_dir=destination,
        allow_patterns=["*.json", "*.jinja", "*.safetensors", "README.md", "LICENSE", "*.txt"],
        max_workers=6,
    )
    config = json.loads((destination / "config.json").read_text(encoding="utf-8"))
    if config["architectures"] != [spec["architecture"]]:
        raise RuntimeError(f"Architecture mismatch: {config['architectures']} != {[spec['architecture']]}")
    index = json.loads((destination / "model.safetensors.index.json").read_text(encoding="utf-8"))
    if index["metadata"]["total_size"] != spec["expected_checkpoint_bytes"]:
        raise RuntimeError("Checkpoint tensor byte count mismatch")
    mtp_names = sorted(k for k in index["weight_map"] if k.startswith("mtp."))
    if len(mtp_names) != spec["expected_mtp_tensor_count"]:
        raise RuntimeError(f"MTP count mismatch: {len(mtp_names)} != {spec['expected_mtp_tensor_count']}")
    by_shard: dict[str, list[str]] = collections.defaultdict(list)
    for name in mtp_names:
        by_shard[index["weight_map"][name]].append(name)
    files = []
    mtp_tensors = []
    for filename in sorted(set(index["weight_map"].values())):
        path = destination / filename
        digest = file_hash(path)
        sibling = next(s for s in info.siblings if s.rfilename == filename)
        if sibling.lfs is None or sibling.lfs.sha256 != digest:
            raise RuntimeError(f"Official LFS checksum mismatch: {filename}")
        files.append({"file": filename, "bytes": path.stat().st_size, "sha256": digest})
        print(f"VERIFIED {spec['repo_id']} {filename} {digest}", flush=True)
        if filename in by_shard:
            with safe_open(path, framework="pt", device="cpu") as reader:
                for mname in by_shard[filename]:
                    shape = reader.get_slice(mname).get_shape()
                    mtp_tensors.append({"name": mname, "shape": shape, "parameters": math.prod(shape)})
    meta_files = [
        "config.json",
        "generation_config.json",
        "tokenizer_config.json",
        "tokenizer.json",
        "chat_template.jinja",
        "model.safetensors.index.json",
    ]
    report = {
        "status": "VERIFIED",
        "repo_id": info.id,
        "revision": info.sha,
        "architecture": spec["architecture"],
        "download_and_verification_seconds": time.monotonic() - started,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "tensor_bytes": index["metadata"]["total_size"],
        "weight_files": files,
        "native_mtp_tensor_count": len(mtp_tensors),
        "native_mtp_parameters": sum(x["parameters"] for x in mtp_tensors),
        "metadata_sha256": {
            name: file_hash(destination / name) for name in meta_files if (destination / name).is_file()
        },
    }
    write_json(manifest_out, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", choices=["dense", "moe", "both"], default="both")
    args = parser.parse_args()
    if args.target in ("dense", "both"):
        verify_and_download(ROOT / "configs/model.json", ROOT / "results/dense_model_manifest.json")
    if args.target in ("moe", "both"):
        verify_and_download(ROOT / "configs/moe_model.json", ROOT / "results/moe_model_manifest.json")
    print("ALL_REQUESTED_MODELS_VERIFIED", flush=True)


if __name__ == "__main__":
    main()
