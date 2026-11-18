from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path

BOX_INSTRUCTION = "Solve the following problem. Put your final answer in \\boxed{}."


def math_messages(problem: str) -> list[dict]:
    return [{"role": "user", "content": f"{problem.strip()}\n\n{BOX_INSTRUCTION}"}]


def object_hash(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


def file_hash(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def normalize_problem(text: str) -> str:
    return re.sub(r"\s+", "", text).casefold()


def text_ngrams(text: str, width: int = 13) -> set[tuple[str, ...]]:
    words = re.findall(r"\w+|[^\w\s]", text.casefold())
    return {tuple(words[i:i + width]) for i in range(len(words) - width + 1)}


def last_boxed(text: str) -> str | None:
    starts = list(re.finditer(r"\\(?:boxed|fbox)\s*\{", text))
    for match in reversed(starts):
        start, depth = match.end(), 1
        for index in range(start, len(text)):
            if text[index] == "{" and (index == 0 or text[index - 1] != "\\"):
                depth += 1
            elif text[index] == "}" and (index == 0 or text[index - 1] != "\\"):
                depth -= 1
                if depth == 0:
                    answer = text[start:index].strip()
                    if answer:
                        return answer
                    break
    # Unbraced single-token LaTeX fallback: e.g. \boxed 42 or \boxed -7
    unbraced = list(re.finditer(r"\\(?:boxed|fbox)\s+([+-]?\d+(?:\.\d+)?)\b", text))
    if unbraced:
        return unbraced[-1].group(1).strip()
    return None


def final_answer_strict(text: str, *, thinking: bool) -> str | None:
    """Strict channel contract required for SFT/ReST^EM training trace admission."""
    if thinking:
        if text.count("</think>") != 1:
            return None
        after = text.split("</think>", 1)[1]
    else:
        after = text
    starts = list(re.finditer(r"\\(?:boxed|fbox)\s*\{", after))
    for match in reversed(starts):
        start, depth = match.end(), 1
        for index in range(start, len(after)):
            if after[index] == "{" and (index == 0 or after[index - 1] != "\\"):
                depth += 1
            elif after[index] == "}" and (index == 0 or after[index - 1] != "\\"):
                depth -= 1
                if depth == 0:
                    ans = after[start:index].strip()
                    if ans:
                        return ans
                    break
    return None


def final_answer_lenient(text: str, *, thinking: bool) -> str | None:
    """Fault-tolerant semantic answer extraction after </think> for evaluation scoring."""
    if thinking:
        if "</think>" not in text:
            return None
        after = text.rsplit("</think>", 1)[1]
    else:
        after = text
    boxed = last_boxed(after)
    if boxed is not None:
        # Strip trivial variable assignment prefixes inside \boxed{x = 8/5}
        cleaned = re.sub(r"^[a-zA-Z]\s*=\s*", "", boxed).strip()
        return cleaned if cleaned else boxed
    # Explicit Answer: <expr> line fallback after thinking close
    ans_lines = list(re.finditer(r"(?:^|\n)\s*(?:Final\s+)?Answer\s*:\s*([^\n]+)", after, re.IGNORECASE))
    if ans_lines:
        cand = ans_lines[-1].group(1).strip().rstrip(".")
        cand = cand.strip("$").strip()
        if cand:
            return cand
    return None


def final_answer(text: str, *, thinking: bool) -> str | None:
    """v1-compatible strict boxed-after-think extractor (preserved for parity audits)."""
    if thinking:
        if "</think>" not in text:
            return None
        text = text.rsplit("</think>", 1)[1]
    starts = list(re.finditer(r"\\(?:boxed|fbox)\s*\{", text))
    for match in reversed(starts):
        start, depth = match.end(), 1
        for index in range(start, len(text)):
            if text[index] == "{" and (index == 0 or text[index - 1] != "\\"):
                depth += 1
            elif text[index] == "}" and (index == 0 or text[index - 1] != "\\"):
                depth -= 1
                if depth == 0:
                    answer = text[start:index].strip()
                    if answer:
                        return answer
                    break
    return None


def quantiles(values: list[float]) -> dict:
    if not values:
        return {"count": 0, "p50": None, "p90": None, "p95": None, "p99": None, "max": None}
    ordered = sorted(values)

    def at(fraction: float) -> float:
        position = fraction * (len(ordered) - 1)
        left, right = math.floor(position), math.ceil(position)
        return ordered[left] + (ordered[right] - ordered[left]) * (position - left)

    return {
        "count": len(values),
        "p50": at(0.5),
        "p90": at(0.9),
        "p95": at(0.95),
        "p99": at(0.99),
        "max": ordered[-1],
    }


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)


def loss_tokens(row: dict) -> int:
    return sum(value != -100 for value in row["labels"][1:])


def validate_token_row(row: dict, cap: int, eos_ids: list[int]) -> None:
    ids, labels = row["input_ids"], row["labels"]
    if not ids or len(ids) != len(labels) or len(ids) > cap:
        raise ValueError("Invalid token lengths or overlength row")
    if ids[-1] not in eos_ids or labels[-1] != ids[-1]:
        raise ValueError("The final native EOS must be supervised")
    if not loss_tokens(row) or labels[0] != -100:
        raise ValueError("Missing response supervision or unmasked prompt")
    first = next(index for index, value in enumerate(labels) if value != -100)
    if labels[first:] != ids[first:]:
        raise ValueError("Response labels must preserve every target token")
    if row["finish_reason"] != "N/A" and row["finish_reason"] != "stop":
        raise ValueError("Generated training targets must have an explicit stop")


def global_update_groups(rows: list[dict], batch_size: int) -> list[list[dict]]:
    if batch_size <= 0 or len(rows) % batch_size:
        raise ValueError("Exact-exposure panels must divide into whole global updates")
    return [rows[start:start + batch_size] for start in range(0, len(rows), batch_size)]


def assign_rank(group: list[dict], rank: int, world: int, bucket: bool) -> list[dict]:
    if len(group) % world or not 0 <= rank < world:
        raise ValueError("Update cannot be evenly assigned without sampler duplication")
    ordered = sorted(group, key=lambda row: len(row["input_ids"])) if bucket else group
    return ordered[rank::world]


def scaled_loss(mean_loss, local_tokens: int, global_tokens: int, world: int):
    if min(local_tokens, global_tokens, world) <= 0:
        raise ValueError("Loss normalization requires positive token counts")
    return mean_loss * (local_tokens * world / global_tokens)
