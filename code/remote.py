"""Experiment-local AML transport for AILab_Qwen27B_Governed_v2 with LF framing and verified transfers."""
from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import re
import shlex
import shutil
import subprocess
import tarfile
import uuid
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

if hasattr(sys.stdout if "sys" in globals() else __import__("sys").stdout, "reconfigure"):
    __import__("sys").stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parents[1]


def resource() -> dict:
    config = json.loads((ROOT / ".local" / "resource.json").read_text(encoding="utf-8"))
    if config["may_cancel_allocation"] or config["owns_allocation"]:
        raise ValueError("This experiment cannot own or release the allocation")
    return config


def az_args(operation: str, config: dict) -> list[str]:
    if operation not in {"show", "connect-ssh"}:
        raise ValueError("Only read-status and SSH operations are supported")
    az = shutil.which("az")
    if not az:
        raise RuntimeError("Azure CLI is unavailable; no other allocation is authorized")
    return [
        az, "ml", "job", operation, "--name", config["job_name"],
        "--workspace-name", config["workspace"], "--resource-group",
        config["resource_group"], "--subscription", config["subscription"],
    ]


def parse_output(stdout: bytes, tag: str, cli_code: int) -> str:
    text = stdout.decode("utf-8", "replace").replace("\r", "")
    match = re.search(
        rf"^__Q27V2_BEGIN_{tag}__\n(.*?)^__Q27V2_END_{tag}__:(\d+)\s*$",
        text, re.MULTILINE | re.DOTALL,
    )
    if match is None:
        raise RuntimeError("No complete SSH result frame; inspect the private transport log")
    if cli_code != 0 or int(match[2]) != 0:
        raise RuntimeError(
            f"Remote exit={match[2]}, CLI exit={cli_code}\n{match[1][-6000:]}"
        )
    return match[1]


def execute(script: str, node: int = 0, timeout: int = 300) -> str:
    config = resource()
    if node not in config["allowed_nodes"]:
        raise ValueError("Node is not authorized")
    if not 1 <= timeout <= 3600:
        raise ValueError("SSH commands must be bounded to 1-3600 seconds")
    tag = uuid.uuid4().hex
    script = script.replace("\r\n", "\n").replace("\r", "\n")
    script = f'test "$(hostname)" = node-{node}\n{script}'
    if node != 0:
        temporary = f"{config['remote_root']}/.hop-{tag}.sh"
        temp = shlex.quote(temporary)
        root = shlex.quote(config["remote_root"])
        encoded = base64.encodebytes(script.encode("utf-8")).decode("ascii")
        script = (
            f"mkdir -p {root}\n"
            f"base64 -d > {temp} <<'HOP_{tag}'\n{encoded}HOP_{tag}\n"
            f"ssh -n -o BatchMode=yes -o StrictHostKeyChecking=no -o ConnectTimeout=20 node-{node} "
            f"{shlex.quote('mkdir -p ' + root)}\n"
            f"scp -q -o BatchMode=yes -o StrictHostKeyChecking=no -o ConnectTimeout=20 {temp} "
            f"node-{node}:{temp} </dev/null\n"
            "set +e\n"
            f"ssh -n -o BatchMode=yes -o StrictHostKeyChecking=no -o ConnectTimeout=20 node-{node} "
            f"{shlex.quote(f'timeout --kill-after=15 {timeout} bash -euo pipefail {temp}')}\n"
            "hop_rc=$?\nset -e\n"
            f"ssh -n -o BatchMode=yes -o StrictHostKeyChecking=no -o ConnectTimeout=20 node-{node} "
            f"{shlex.quote('rm -f -- ' + temp)}\n"
            f"rm -f -- {temp}\nexit \"$hop_rc\""
        )
    body = (
        f"echo __Q27V2_BEGIN_{tag}__\n"
        f"timeout --signal=TERM --kill-after=15 {timeout} "
        f"bash -euo pipefail 2>&1 <<'SCRIPT_{tag}'\n"
        "test \"$(hostname)\" = node-0\n"
        f"{script}\nSCRIPT_{tag}\n"
        f"rc=$?\nprintf '\\n__Q27V2_END_{tag}__:%s\\n' \"$rc\"\nexit \"$rc\"\n"
    )
    args = az_args("connect-ssh", config) + [
        "--node-index", "0",
        "--private-key-file-path", config["private_key_path"],
    ]
    directory = ROOT / ".local" / "transport"
    directory.mkdir(parents=True, exist_ok=True)
    log = directory / f"{tag}.log"
    try:
        result = subprocess.run(
            args, input=body.encode("utf-8"), capture_output=True, timeout=timeout + 120
        )
    except subprocess.TimeoutExpired as error:
        log.write_bytes((error.stdout or b"") + b"\nSTDERR\n" + (error.stderr or b""))
        raise RuntimeError(
            f"SSH timed out; inspect owned remote state before retrying. Log: {log.name}"
        ) from error
    log.write_bytes(result.stdout + b"\nSTDERR\n" + result.stderr)
    try:
        return parse_output(result.stdout, tag, result.returncode)
    except RuntimeError as error:
        raise RuntimeError(f"{error}\nPrivate transport log: {log.name}") from error


def allowed_remote(path: str, config: dict) -> str:
    target = PurePosixPath(path)
    root = PurePosixPath(config["remote_root"])
    if not target.is_absolute() or ".." in target.parts:
        raise ValueError("Remote paths must be absolute without parent traversal")
    if target != root and root not in target.parents:
        raise ValueError("Transfers are limited to this experiment's node-local root")
    return shlex.quote(str(target))


def push_tree(node: int) -> str:
    config = resource()
    buffer = io.BytesIO()
    paths = []
    for top_file in (".gitignore", "README.md", "program.md", "gate_rules.json"):
        if (ROOT / top_file).is_file():
            paths.append(ROOT / top_file)
    for name in ("code", "configs", "eval", "tests", "sources"):
        folder = ROOT / name
        if folder.exists():
            paths.extend(sorted(folder.rglob("*")))
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        for path in paths:
            if not path.is_file() or "__pycache__" in path.parts:
                continue
            data = path.read_bytes()
            item = tarfile.TarInfo(path.relative_to(ROOT).as_posix())
            item.size = len(data)
            item.mode = 0o644
            archive.addfile(item, io.BytesIO(data))
    payload = buffer.getvalue()
    if len(payload) >= 5 * 1024 * 1024:
        raise ValueError("Source bundle exceeds the small-transfer limit")
    digest = hashlib.sha256(payload).hexdigest()
    destination = allowed_remote(config["remote_root"], config)
    temporary = f"{config['remote_root']}/.transfer-{uuid.uuid4().hex}.tgz"
    temp = allowed_remote(temporary, config)
    encoded = base64.encodebytes(payload).decode("ascii")
    return execute(
        f"mkdir -p {destination}\n"
        f"base64 -d > {temp} <<'Q27V2_PAYLOAD'\n{encoded}Q27V2_PAYLOAD\n"
        f"printf '%s  %s\\n' {digest} {temp} | sha256sum -c -\n"
        f"tar xzf {temp} -C {destination}\nrm -f -- {temp}\necho SOURCE_SYNC_OK",
        node,
    )


def pull_file(remote: str, destination: Path, node: int = 0) -> None:
    source = allowed_remote(remote, resource())
    destination = destination.resolve()
    if ROOT not in destination.parents:
        raise ValueError("Pull destinations must be inside the new experiment")
    output = execute(
        f"test -f {source}\ntest \"$(stat -c %s {source})\" -lt 20971520\n"
        f"sha256sum {source}\necho Q27V2_FILE_BEGIN\nbase64 -w 76 {source}\n"
        "echo Q27V2_FILE_END", node,
    )
    match = re.search(
        r"^([a-f0-9]{64})  .*\nQ27V2_FILE_BEGIN\n(.*?)^Q27V2_FILE_END",
        output, re.MULTILINE | re.DOTALL,
    )
    if match is None:
        raise RuntimeError("Missing file payload or checksum")
    data = base64.b64decode("".join(match[2].split()), validate=True)
    if hashlib.sha256(data).hexdigest() != match[1]:
        raise RuntimeError("Transfer checksum mismatch")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(data)


def status() -> dict:
    result = subprocess.run(
        az_args("show", resource()) + [
            "--query",
            "{name:name,display_name:display_name,status:status,"
            "instance_count:resources.instance_count,"
            "instance_type:resources.instance_type,environment:environment}",
            "-o", "json", "--only-show-errors",
        ], capture_output=True, timeout=120,
    )
    if result.returncode:
        raise RuntimeError(result.stderr.decode("utf-8", "replace"))
    record = json.loads(result.stdout)
    record["observed_at_utc"] = datetime.now(timezone.utc).isoformat()
    path = ROOT / ".local" / "status.json"
    path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    return record


def launch_stage(name: str, command: list[str], *, node: int = 0,
                 timeout: int = 3600, gpus: int = 0) -> str:
    if not re.fullmatch(r"[a-zA-Z0-9_-]+", name):
        raise ValueError("Use a unique simple stage name")
    root = resource()["remote_root"]
    invocation = shlex.join([
        "python3", f"{root}/code/stage.py", "--root", root, "--name", name,
        "--timeout", str(timeout), "--gpus", str(gpus), "--", *command,
    ])
    return execute(
        f"mkdir -p {shlex.quote(root + '/logs')}\n"
        f"test ! -e {shlex.quote(root + '/runs/' + name)}\n"
        f"nohup {invocation} >{shlex.quote(root + '/logs/' + name + '.supervisor.log')} "
        "2>&1 </dev/null &\n"
        "echo SUPERVISOR_PID=$!\nsleep 3\n"
        f"cat {shlex.quote(root + '/runs/' + name + '/status.json')}\n"
        "python3 -c 'import json,sys; s=json.load(open(sys.argv[1])); "
        "assert s[\"status\"] in (\"running\",\"completed\"), s' "
        f"{shlex.quote(root + '/runs/' + name + '/status.json')}",
        node=node, timeout=45,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("operation", choices=["status", "exec", "sync", "pull"])
    parser.add_argument("arguments", nargs="*")
    parser.add_argument("--node", type=int, default=0)
    parser.add_argument("--file", type=Path)
    parser.add_argument("--save", type=Path)
    parser.add_argument("--timeout", type=int, default=300)
    args = parser.parse_args()
    if args.operation == "status":
        output = json.dumps(status(), indent=2)
    elif args.operation == "sync":
        output = push_tree(args.node)
    elif args.operation == "pull":
        if len(args.arguments) != 2:
            parser.error("pull requires a remote file and a local destination")
        pull_file(args.arguments[0], Path(args.arguments[1]), args.node)
        output = "PULL_OK"
    else:
        script = args.file.read_text(encoding="utf-8") if args.file else " ".join(args.arguments)
        output = execute(script, args.node, args.timeout)
    if args.save:
        destination = args.save.resolve()
        if ROOT not in destination.parents:
            raise ValueError("Saved output must remain inside the experiment")
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(output, encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
