"""Single-node publisher for complete, checksummed AILab_Qwen27B_Governed_v2 evidence."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import io
import json
import os
import platform
import shutil
import tarfile
import uuid
from datetime import datetime, timezone
from pathlib import Path

from stage import write_json


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


README = """# Governed Qwen reasoning efficiency rerun archive (AILab_Qwen27B_Governed_v2)

This unique directory belongs to AILab_Qwen27B_Governed_v2, started 2026-11-17.
It preserves v1 untouched and does not overwrite or relabel v1 outcomes.

Node 0 is the sole publisher. Every other node stages outputs on node-local
storage and transfers them to node 0 over the job-internal connection. Never
write to this directory from multiple nodes or publish large bundles concurrently.

Start with LATEST.json. It points to a complete checksummed metadata archive
and verified complete optimizer/RNG checkpoint bundles; older archives remain
immutable. Verify archive SHA256 and all member hashes before restoring.

Private local progress.md, .local/resource.json, and holdout gold keys stored
in .local/governance_anchor/ are excluded from public source bundles unless an
explicit sealed verification receipt is inspected. Preserving artifacts never
authorizes canceling the user-owned AML allocation; keepalive must remain active.
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--initialize", action="store_true")
    parser.add_argument("--checkpoint", type=Path)
    args = parser.parse_args()
    if platform.node() != "node-0":
        raise RuntimeError("Only node 0 may publish to shared durable storage")
    root = args.root.resolve()
    destination = args.destination
    if not destination.is_absolute() or ".." in destination.parts:
        raise ValueError("An explicit absolute durable destination is required")
    with (root / ".publisher.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.initialize and not destination.exists():
            destination.mkdir(parents=True, exist_ok=False)
            with (destination / "README.md").open("x", encoding="utf-8") as stream:
                stream.write(README)
                stream.flush()
                os.fsync(stream.fileno())
        if not (destination / "README.md").is_file():
            raise RuntimeError("Durable destination has not been initialized")
        if (destination / "README.md").read_text(encoding="utf-8") != README:
            raise RuntimeError("Durable destination belongs to a different publisher")
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:6]
        staging = root / "tmp" / f"evidence_{stamp}.tar.gz"
        staging.parent.mkdir(exist_ok=True)
        if args.checkpoint:
            checkpoint = args.checkpoint.resolve()
            if root not in checkpoint.parents or not (checkpoint / "complete.json").is_file():
                raise RuntimeError("Only a complete experiment-owned checkpoint can be banked")
            completion = json.loads((checkpoint / "complete.json").read_text(encoding="utf-8"))
            if "state_sha256" in completion:
                if digest(checkpoint / "state.pt") != completion["state_sha256"]:
                    raise RuntimeError("Checkpoint state does not match its completion checksum")
            elif completion.get("files"):
                for name, expected in completion["files"].items():
                    if Path(name).name != name or digest(checkpoint / name) != expected:
                        raise RuntimeError(f"Artifact file {name} does not match completion manifest")
            else:
                raise RuntimeError("Completion metadata does not identify verifiable artifact files")
            paths = sorted(checkpoint.rglob("*"))
        else:
            paths = []
            for top in ("README.md", "program.md", "gate_rules.json", "IMMUTABLE_MANIFEST.json", "trial_ledger.jsonl"):
                if (root / top).is_file():
                    paths.append(root / top)
            for name in ("code", "configs", "eval", "tests", "sources", "results", "data"):
                if (root / name).exists():
                    paths.extend(sorted((root / name).rglob("*")))
        members = []
        with tarfile.open(staging, "w:gz", compresslevel=1) as archive:
            for path in paths:
                if not path.is_file() or "__pycache__" in path.parts or path.is_symlink():
                    continue
                if path.relative_to(root).parts[:2] == ("data", "raw"):
                    continue
                if not args.checkpoint and (
                    any(part.startswith("checkpoint-") for part in path.parts)
                    or path.suffix in {".pt", ".bin", ".safetensors"}
                ):
                    continue
                if path.name in {"progress.md", "PROGRESS.md", "resource.json"} or ".local" in path.parts:
                    raise RuntimeError("Private local state cannot enter a durable evidence bundle")
                relative = path.relative_to(root).as_posix()
                if not args.checkpoint:
                    data = path.read_bytes()
                    members.append({
                        "file": relative,
                        "bytes": len(data),
                        "sha256": hashlib.sha256(data).hexdigest(),
                    })
                    info = tarfile.TarInfo(relative)
                    info.size, info.mode = len(data), 0o644
                    archive.addfile(info, io.BytesIO(data))
                else:
                    members.append({"file": relative, "bytes": path.stat().st_size, "sha256": digest(path)})
                    archive.add(path, arcname=relative, recursive=False)
        checksum = digest(staging)
        final = destination / staging.name
        partial = destination / (staging.name + ".partial")
        with staging.open("rb") as source, partial.open("xb") as target:
            shutil.copyfileobj(source, target, length=8 * 1024 * 1024)
            target.flush()
            os.fsync(target.fileno())
        if digest(partial) != checksum:
            raise RuntimeError("Durable archive checksum mismatch")
        partial.rename(final)
        manifest = {
            "archive": final.name,
            "sha256": checksum,
            "bytes": final.stat().st_size,
            "members": members,
            "published_at_utc": datetime.now(timezone.utc).isoformat(),
            "kind": "checkpoint" if args.checkpoint else "metadata",
        }
        if args.checkpoint:
            manifest["artifact_type"] = completion.get("kind", "training_checkpoint")
            manifest["resume_capable"] = completion.get("resume_capable", True)
        write_json(destination / f"{staging.stem}.manifest.json", manifest)
        latest_path = destination / "LATEST.json"
        latest = json.loads(latest_path.read_text(encoding="utf-8")) if latest_path.exists() else {}
        if "archive" in latest:
            latest = {"metadata": latest, "checkpoints": {}}
        latest.setdefault("checkpoints", {})
        if args.checkpoint:
            latest["checkpoints"][checkpoint.relative_to(root).as_posix()] = manifest
        else:
            latest["metadata"] = manifest
        latest["published_at_utc"] = manifest["published_at_utc"]
        write_json(latest_path, latest)
        staging.unlink()
        print(json.dumps({"published": final.name, "sha256": checksum, "members": len(members)}))


if __name__ == "__main__":
    main()
