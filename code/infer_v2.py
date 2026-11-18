"""
infer_v2.py — Policy-pinned streaming evaluation for AILab_Qwen27B_Governed_v2
with request-level TTFT/TPOT/final-channel latency, GPU board energy integration,
vLLM speculative-decoding counter deltas, and decoupled strict/lenient grading.
"""
from __future__ import annotations

import argparse
import asyncio
import gzip
import importlib.metadata
import json
import re
import subprocess
import sys
import time
from pathlib import Path

import httpx
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
sys.path.insert(0, str(ROOT / "eval"))

from common import file_hash, math_messages, object_hash, quantiles, read_jsonl  # noqa: E402
from inference_policy import template_options  # noqa: E402
from mcq_contract import mcq_answer, mcq_messages  # noqa: E402
from stage import write_json  # noqa: E402
from telemetry import BoardTelemetry  # noqa: E402
from verifiers_v2 import score_response  # noqa: E402


def parse_prometheus_spec_counters(metrics_text: str) -> dict[str, float]:
    totals = {"draft_tokens": 0.0, "accepted_tokens": 0.0, "num_drafts": 0.0}
    for line in metrics_text.splitlines():
        if line.startswith("#") or not line.strip():
            continue
        if "spec_decode_num_draft_tokens" in line:
            totals["draft_tokens"] += float(line.rsplit(None, 1)[-1])
        elif "spec_decode_num_accepted_tokens" in line and "per_pos" not in line:
            totals["accepted_tokens"] += float(line.rsplit(None, 1)[-1])
        elif "spec_decode_num_drafts" in line:
            totals["num_drafts"] += float(line.rsplit(None, 1)[-1])
    return totals


async def one_request(
    client: httpx.AsyncClient,
    semaphore: asyncio.Semaphore,
    row: dict,
    ids: list[int],
    policy: dict,
    base_url: str,
    seed: int,
    start: float,
    due: float,
    model_name: str,
) -> dict:
    if due > time.monotonic():
        await asyncio.sleep(due - time.monotonic())
    queued = time.monotonic()
    async with semaphore:
        dispatched = time.monotonic()
        record = {
            "id": row["id"],
            "seed": seed,
            "prompt_hash": row["prompt_hash"],
            "prompt_tokens_local": len(ids),
            "client_wait_seconds": dispatched - queued,
            "dispatch_offset_seconds": dispatched - start,
            "text": "",
            "error": None,
            "finish_reason": None,
            "output_tokens": None,
            "prompt_tokens_server": None,
            "ttft_seconds": None,
            "time_to_final_channel_seconds": None,
        }
        chunks, arrival_times = [], []
        text = ""
        payload = {
            "model": model_name,
            "prompt": ids,
            "seed": seed,
            "stream": True,
            "stream_options": {"include_usage": True},
            **{k: v for k, v in policy.items() if k not in {"thinking", "effort"}},
        }
        try:
            async with client.stream("POST", base_url + "/v1/completions", json=payload) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    raw = line[6:]
                    if raw == "[DONE]":
                        break
                    event = json.loads(raw)
                    if event.get("usage") is not None:
                        usage = event["usage"]
                        record["output_tokens"] = usage["completion_tokens"]
                        record["prompt_tokens_server"] = usage["prompt_tokens"]
                    for choice in event.get("choices", []):
                        chunk = choice.get("text", "")
                        if chunk:
                            now = time.monotonic()
                            chunks.append(chunk)
                            arrival_times.append(now)
                            text += chunk
                            if record["ttft_seconds"] is None:
                                record["ttft_seconds"] = now - dispatched
                            final_visible = (
                                ("</think>" in text and bool(text.rsplit("</think>", 1)[1].strip()))
                                if policy["thinking"]
                                else bool(text.strip())
                            )
                            if final_visible and record["time_to_final_channel_seconds"] is None:
                                record["time_to_final_channel_seconds"] = now - dispatched
                        if choice.get("finish_reason") is not None:
                            record["finish_reason"] = choice["finish_reason"]
        except (httpx.HTTPError, json.JSONDecodeError, KeyError) as error:
            record["error"] = f"{type(error).__name__}: {error}"
        ended = time.monotonic()
        record.update(
            text="".join(chunks),
            request_seconds=ended - dispatched,
            including_client_queue_seconds=ended - queued,
            completion_offset_seconds=ended - start,
            stream_chunk_count=len(chunks),
        )
        tokens = record["output_tokens"]
        record["tpot_seconds"] = (
            (ended - dispatched - record["ttft_seconds"]) / (tokens - 1)
            if tokens is not None and tokens > 1 and record["ttft_seconds"] is not None
            else None
        )
        if record["error"] is None and (
            record["finish_reason"] not in {"stop", "length"}
            or tokens is None
            or record["prompt_tokens_server"] != len(ids)
        ):
            record["error"] = "PROTOCOL_ERROR: missing usage/finish or prompt-token mismatch"
        return record


async def run_eval(args, rows: list[dict], tokenizer, policy: dict, directory: Path) -> dict:
    rendered = [
        tokenizer.apply_chat_template(
            mcq_messages(row) if args.task == "mcq" else math_messages(row["problem"]),
            tokenize=False,
            add_generation_prompt=True,
            **template_options(policy),
        )
        for row in rows
    ]
    prompts = [tokenizer.encode(text, add_special_tokens=False) for text in rendered]
    if any(len(ids) + policy["max_tokens"] > args.max_model_len for ids in prompts):
        raise RuntimeError("Prompt plus output cap exceeds configured max_model_len")

    write_json(
        directory / "template_probe.json",
        {
            "first_id": rows[0]["id"],
            "rendered_prompt": rendered[0],
            "prompt_token_len": len(prompts[0]),
            "thinking": policy["thinking"],
            "effort": policy["effort"],
        },
    )

    semaphore = asyncio.Semaphore(args.concurrency)
    telemetry = BoardTelemetry([int(x) for x in args.gpus.split(",")])
    limits = httpx.Limits(max_connections=max(24, args.concurrency + 4))
    async with httpx.AsyncClient(timeout=httpx.Timeout(args.request_timeout), limits=limits) as client:
        health = await client.get(args.base_url + "/health")
        health.raise_for_status()
        before = await client.get(args.base_url + "/metrics")
        before.raise_for_status()
        (directory / "metrics_before.txt").write_text(before.text, encoding="utf-8")
        spec_before = parse_prometheus_spec_counters(before.text)

        telemetry.start()
        started = time.monotonic()
        tasks = []
        req_idx = 0
        for draw in range(args.draws):
            for row, ids in zip(rows, prompts):
                seed = int(object_hash({"seed": args.seed, "id": row["id"], "draw": draw})[:8], 16)
                due = started + req_idx / args.request_rate if args.request_rate else started
                tasks.append(
                    asyncio.create_task(
                        one_request(
                            client, semaphore, row, ids, policy, args.base_url, seed, started, due, args.model_name
                        )
                    )
                )
                req_idx += 1
        outputs = []
        with (directory / "generations.partial.jsonl").open("w", encoding="utf-8") as stream:
            for task in asyncio.as_completed(tasks):
                rec = await task
                outputs.append(rec)
                stream.write(json.dumps(rec, ensure_ascii=False) + "\n")
                stream.flush()
                if len(outputs) % 25 == 0 or len(outputs) == len(tasks):
                    print(
                        json.dumps({
                            "finished": len(outputs),
                            "total": len(tasks),
                            "last_id": rec["id"],
                            "tokens": rec["output_tokens"],
                            "finish": rec["finish_reason"],
                            "err": rec["error"],
                        }),
                        flush=True,
                    )
        ended = time.monotonic()
        energy = telemetry.stop()
        after = await client.get(args.base_url + "/metrics")
        after.raise_for_status()
        (directory / "metrics_after.txt").write_text(after.text, encoding="utf-8")
        spec_after = parse_prometheus_spec_counters(after.text)

    has_inline_gold = all("gold_answer" in r and r["gold_answer"] is not None for r in rows)
    if has_inline_gold:
        answers = {r["id"]: str(r["gold_answer"]) for r in rows}
        problems = {r["id"]: r["problem"] for r in rows}
        int_ids = {r["id"] for r in rows if r.get("source") in {"dapo", "aime24", "aime25"}}
        opt_counts = {r["id"]: len(r["options"]) for r in rows} if args.task == "mcq" else {}
        for rec in outputs:
            fin = rec["finish_reason"] if rec["error"] is None else "error"
            if args.task == "mcq":
                ans = mcq_answer(rec["text"], thinking=policy["thinking"], option_count=opt_counts[rec["id"]]) if fin == "stop" else None
                ok = bool(fin == "stop" and ans == answers[rec["id"]])
                rec.update(
                    lenient_extracted=ans,
                    strict_extracted=ans,
                    lenient_correct=ok,
                    strict_correct=ok,
                    v1_correct=ok,
                    natural_stop_lenient_parse_failure=bool(fin == "stop" and ans is None),
                    verifier_error=None,
                )
            else:
                sr = score_response(
                    rec["text"],
                    answers[rec["id"]],
                    finish_reason=fin,
                    thinking=policy["thinking"],
                    integer_task=(rec["id"] in int_ids),
                    problem=problems[rec["id"]],
                )
                rec.update(sr)

    gz_path = directory / "generations.jsonl.gz"
    with gzip.open(gz_path, "wt", encoding="utf-8") as stream:
        for rec in sorted(outputs, key=lambda x: (x["id"], x["seed"])):
            stream.write(json.dumps(rec, ensure_ascii=False) + "\n")
    (directory / "generations.partial.jsonl").unlink()

    duration = max(r["completion_offset_seconds"] for r in outputs) - min(r["dispatch_offset_seconds"] for r in outputs)
    valid = [r for r in outputs if r["error"] is None]
    stopped = [r for r in valid if r["finish_reason"] == "stop"]
    trunc_rate = sum(r["finish_reason"] == "length" for r in outputs) / len(outputs)
    stop_rate = len(stopped) / len(outputs)
    out_toks = [r["output_tokens"] or 0 for r in outputs]
    total_toks = sum(out_toks)
    draft_delta = max(0.0, spec_after["draft_tokens"] - spec_before["draft_tokens"])
    acc_delta = max(0.0, spec_after["accepted_tokens"] - spec_before["accepted_tokens"])

    tpot_q = quantiles([r["tpot_seconds"] * 1000.0 for r in valid if r["tpot_seconds"] is not None])
    ttft_q = quantiles([r["ttft_seconds"] * 1000.0 for r in valid if r["ttft_seconds"] is not None])
    req_q = quantiles([r["request_seconds"] for r in valid])
    tok_q = quantiles([float(t) for t in out_toks])

    report = {
        "run": args.name,
        "purpose": args.purpose,
        "task": args.task,
        "dataset": args.dataset.name,
        "dataset_sha256": file_hash(args.dataset),
        "server_stage": args.server_stage,
        "served_model": args.model_name,
        "policy": policy,
        "concurrency": args.concurrency,
        "draws": args.draws,
        "seed": args.seed,
        "unique_problems": len(rows),
        "requests": len(outputs),
        "wall_seconds": ended - started,
        "dataset_seconds": duration,
        "output_tokens": total_toks,
        "output_tokens_quantiles": tok_q,
        "output_tokens_per_second": total_toks / duration,
        "requests_per_second": len(outputs) / duration,
        "assigned_gpu_hours": duration * len(args.gpus.split(",")) / 3600.0,
        "infrastructure_errors": len(outputs) - len(valid),
        "stop_rate": stop_rate,
        "truncation_rate": trunc_rate,
        "tpot_ms": tpot_q,
        "ttft_ms": ttft_q,
        "tpot_p95_ms": tpot_q["p95"],
        "ttft_p95_ms": ttft_q["p95"],
        "request_seconds": req_q,
        "board_joules": energy["board_joules"],
        "energy": {k: v for k, v in energy.items() if k != "power_samples"},
        "mtp_speculative": {
            "draft_tokens": draft_delta,
            "accepted_tokens": acc_delta,
            "acceptance_rate": (acc_delta / draft_delta) if draft_delta > 0 else None,
        },
        "raw_generations_sha256": file_hash(gz_path),
    }
    if has_inline_gold:
        lenient_acc = sum(bool(r.get("lenient_correct")) for r in outputs) / len(outputs)
        strict_acc = sum(bool(r.get("strict_correct")) for r in outputs) / len(outputs)
        v1_acc = sum(bool(r.get("v1_correct")) for r in outputs) / len(outputs)
        nat_pf = sum(bool(r.get("natural_stop_lenient_parse_failure")) for r in outputs) / len(outputs)
        verr = sum(r.get("verifier_error") is not None for r in outputs)
        slo_ok = [
            r for r in stopped
            if not r.get("natural_stop_lenient_parse_failure")
            and r["ttft_seconds"] is not None and r["ttft_seconds"] <= 2.0
            and (r["tpot_seconds"] is None or r["tpot_seconds"] <= 0.1)
            and r["including_client_queue_seconds"] <= 600.0
        ]
        report.update(
            lenient_accuracy=lenient_acc,
            strict_accuracy=strict_acc,
            v1_parity_accuracy=v1_acc,
            natural_stop_lenient_parse_failure_rate=nat_pf,
            verifier_errors=verr,
            slo_correct_goodput_per_second=sum(bool(r.get("lenient_correct")) for r in slo_ok) / duration,
        )
    write_json(directory / "telemetry.json", energy)
    write_json(directory / "summary.json", report)

    if args.scored_output and (has_inline_gold or args.external_labels):
        cmd = [
            sys.executable, "-I", "-B", str(ROOT / "eval" / "isolated_scorer.py"),
            "capability",
            "--split-prompts", str(args.dataset),
            "--generations", str(gz_path),
            "--draws", str(args.draws),
            "--base-seed", str(args.seed),
            "--candidate-id", args.candidate_id or args.name,
            "--protocol-id", args.protocol_id or args.purpose,
            "--env-fingerprint", args.env_fingerprint,
            "--output", str(args.scored_output),
        ]
        if not policy["thinking"]:
            cmd.append("--no-thinking")
        if args.external_labels:
            cmd.extend(["--external-labels", str(args.external_labels)])
        if args.holdout_registry:
            cmd.extend(["--holdout-registry", str(args.holdout_registry)])
        subprocess.run(cmd, check=True, timeout=300)

    return report


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--name", required=True)
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--model-config", type=Path, default=ROOT / "configs/model.json")
    p.add_argument("--tokenizer-dir", type=Path, default=ROOT / "models/dense27_base")
    p.add_argument("--task", choices=["math", "mcq"], default="math")
    p.add_argument("--base-url", default="http://127.0.0.1:18273")
    p.add_argument("--server-stage", required=True)
    p.add_argument("--model-name", default="qwen27b")
    p.add_argument("--cap", type=int, default=32768)
    p.add_argument("--effort", choices=["xhigh", "medium", "low", "none"], default="xhigh")
    p.add_argument("--no-thinking", action="store_true")
    p.add_argument("--concurrency", type=int, default=8)
    p.add_argument("--draws", type=int, default=1)
    p.add_argument("--seed", type=int, default=20261117)
    p.add_argument("--limit", type=int)
    p.add_argument("--gpus", default="0,1")
    p.add_argument("--request-rate", type=float)
    p.add_argument("--request-timeout", type=float, default=7200)
    p.add_argument("--max-model-len", type=int, default=40960)
    p.add_argument("--purpose", default="calibration")
    p.add_argument("--temperature", type=float)
    p.add_argument("--top-p", type=float)
    p.add_argument("--top-k", type=int)
    p.add_argument("--presence-penalty", type=float)
    p.add_argument("--scored-output", type=Path)
    p.add_argument("--candidate-id")
    p.add_argument("--protocol-id")
    p.add_argument("--env-fingerprint", default="a100-sxm4-80gb-cu129-vllm029")
    p.add_argument("--external-labels", type=Path)
    p.add_argument("--holdout-registry", type=Path)
    args = p.parse_args()

    rows = read_jsonl(args.dataset)
    if args.limit:
        rows = rows[: args.limit]
    spec = json.loads(args.model_config.read_text(encoding="utf-8"))
    effort = None if args.effort == "none" else args.effort
    if spec.get("reasoning_effort_supported") is False and effort is not None:
        raise ValueError("Model template does not support reasoning_effort; pass --effort none")
    policy = {
        **spec["sampling"],
        "max_tokens": args.cap,
        "stop_token_ids": spec["eos_token_ids"],
        "thinking": not args.no_thinking,
        "effort": effort,
    }
    if args.no_thinking:
        policy.update(temperature=0.7, top_p=0.8, top_k=20, presence_penalty=1.5)
    for k, v in (
        ("temperature", args.temperature),
        ("top_p", args.top_p),
        ("top_k", args.top_k),
        ("presence_penalty", args.presence_penalty),
    ):
        if v is not None:
            policy[k] = v

    directory = ROOT / "results" / "inference" / args.name
    directory.mkdir(parents=True, exist_ok=False)
    server = json.loads((ROOT / "runs" / args.server_stage / "status.json").read_text(encoding="utf-8"))
    if server["status"] != "running":
        raise RuntimeError(f"Owned server {args.server_stage} is not running")
    write_json(
        directory / "request_contract.json",
        {
            "policy": policy,
            "task": args.task,
            "ids": [r["id"] for r in rows],
            "dataset_sha256": file_hash(args.dataset),
            "model_revision": spec["revision"],
            "server_stage": args.server_stage,
            "server_command": server["command"],
            "served_model": args.model_name,
            "vllm_version": importlib.metadata.version("vllm"),
            "env_fingerprint": args.env_fingerprint,
        },
    )
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_dir, local_files_only=True)
    report = asyncio.run(run_eval(args, rows, tokenizer, policy, directory))
    print(json.dumps({k: v for k, v in report.items() if k not in {"energy"}}, indent=2))


if __name__ == "__main__":
    main()
