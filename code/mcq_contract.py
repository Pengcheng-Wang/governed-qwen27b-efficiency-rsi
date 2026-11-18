"""Multiple-choice prompt and answer extractor for non-math retention guardrails."""
from __future__ import annotations

import re


def mcq_messages(row: dict) -> list[dict]:
    options = row["options"]
    if not 2 <= len(options) <= 10 or any(not option.strip() for option in options):
        raise ValueError("Invalid multiple-choice options")
    choices = "\n".join(f"{chr(65 + i)}. {option.strip()}" for i, option in enumerate(options))
    return [{
        "role": "user",
        "content": (
            f"{row['problem'].strip()}\n\n{choices}\n\n"
            "Think through the question, then finish with exactly 'Answer: X', "
            "where X is the correct option letter."
        ),
    }]


def mcq_answer(text: str, *, thinking: bool, option_count: int) -> str | None:
    if thinking:
        if "</think>" not in text:
            return None
        text = text.rsplit("</think>", 1)[1]
    text = text.replace("**", "").replace("`", "")
    answers = [
        value.upper()
        for value in re.findall(r"(?im)^\s*Answer\s*:\s*\(?([A-J])\)?(?:[.\s]|<\||$)", text)
    ]
    if len(set(answers)) != 1:
        return None
    answer = answers[-1]
    return answer if ord(answer) - 65 < option_count else None
