"""Hardened math & integer verifiers with decoupled strict SFT admission vs lenient semantic scoring."""
from __future__ import annotations

import re

from common import final_answer, final_answer_lenient, final_answer_strict

FORBIDDEN_SHORTCUT_PATTERNS = (
    r"\bsorry\b",
    r"\badmit\b",
    r"\bnative_decide\b",
    r"os\._exit",
    r"sys\.exit",
)


def has_shortcut_tokens(text: str) -> bool:
    for pat in FORBIDDEN_SHORTCUT_PATTERNS:
        if re.search(pat, text):
            return True
    return False


def exact_integer_equivalent(answer: str | None, gold: str) -> bool:
    if not re.fullmatch(r"[+-]?\d+", gold.strip()):
        raise ValueError("Integer scorer requires an exact integer gold string")
    expected = int(gold)
    if answer is None:
        return False
    value = answer.strip().strip("$").strip()
    value = re.sub(r"^[a-zA-Z]\s*=\s*", "", value).strip()
    if re.fullmatch(r"[+-]?\d+(?:\.0+)?", value):
        return int(value.split(".", 1)[0]) == expected
    if re.fullmatch(r"[+-]?\d+\.\d+", value):
        return False
    if re.fullmatch(r"[+-]?\d{1,3}(?:,\d{3})+", value):
        return int(value.replace(",", "")) == expected
    from math_verify import LatexExtractionConfig, parse
    from sympy import Expr, Float, Integer, simplify

    parsed = parse(
        rf"\boxed{{{value}}}",
        extraction_config=[LatexExtractionConfig()],
        fallback_mode="no_fallback",
        parsing_timeout=3,
    )
    for expression in parsed:
        if isinstance(expression, Expr) and not expression.has(Float) and expression.is_number:
            if simplify(expression - Integer(expected)) == 0:
                return True
    return False


def _semantic_math_pair(answer: str, gold: str, problem: str) -> tuple[str, str]:
    words = {
        word: str(index) for index, word in enumerate(
            "zero one two three four five six seven eight nine ten eleven twelve "
            "thirteen fourteen fifteen sixteen seventeen eighteen nineteen twenty".split()
        )
    }

    def notation(value: str) -> str:
        value = value.strip().replace("\u00b0", r"^\circ")
        value = re.sub(
            r"\\\\\s*\[[+-]?(?:\d+(?:\.\d*)?|\.\d+)\s*(?:pt|mm|cm|em|ex)\]",
            r"\\\\", value,
        )
        word = re.fullmatch(r"\\(?:mbox|text)\{\s*([A-Za-z]+)\s*\}", value)
        name = word.group(1).casefold() if word else value.casefold()
        return words.get(name, value)

    answer, gold = notation(answer), notation(gold)
    if re.search(r"\b(?:what|which)\s+(?:percentage|percent)\b", problem, re.IGNORECASE):
        answer = re.sub(r"\\?%\s*$", "", answer).strip()
        gold = re.sub(r"\\?%\s*$", "", gold).strip()
    else:
        answer = re.sub(r"^([+-]?(?:\d+(?:\.\d*)?|\.\d+))\s*\\?%$", r"(\1)/100", answer)
        gold = re.sub(r"^([+-]?(?:\d+(?:\.\d*)?|\.\d+))\s*\\?%$", r"(\1)/100", gold)
    if re.search(r"\bwhich\s+grade\b", problem, re.IGNORECASE):
        answer = re.sub(
            r"^(\d+)(?:\^\{\\(?:text|mathrm)\{(?:st|nd|rd|th)\}\}|st|nd|rd|th)"
            r"\s*(?:\\(?:text|mathrm)\{\s*grade\s*\}|grade)$",
            r"\1", answer, flags=re.IGNORECASE,
        )
    # A spaced numeric reference list is not a thousands-grouped scalar.
    if re.fullmatch(r"[+-]?\d+(?:,\s+[+-]?\d+)+", gold):
        if re.fullmatch(r"[+-]?\d+(?:,\s*[+-]?\d+)+", answer):
            answer = ", ".join(part.strip() for part in answer.split(","))
    return answer, gold


def grade_symbolic_math(
    answer: str | None, gold: str, *, problem: str = "", legacy: bool = False,
) -> dict:
    if answer is None:
        return {"correct": False, "parse_failure": True, "verifier_error": None}
    from math_verify import LatexExtractionConfig, parse, verify
    from sympy import FiniteSet, Interval

    try:
        if not legacy:
            answer, gold = _semantic_math_pair(answer, gold, problem)
        expected = parse(rf"\boxed{{{gold}}}", extraction_config=[LatexExtractionConfig()])
        actual = parse(rf"\boxed{{{answer}}}", extraction_config=[LatexExtractionConfig()])
        if not expected:
            raise ValueError("Verifier could not parse published gold answer")
        # Do not use coordinate-pair/interval aliases for finite solution sets.
        discrete_reference = not legacy and isinstance(expected[0], FiniteSet)
        compatible_type = not (
            discrete_reference and actual and isinstance(actual[0], Interval)
        )
        return {
            "correct": bool(actual and compatible_type and verify(
                expected, actual, allow_set_relation_comp=not legacy and not discrete_reference,
            )),
            "parse_failure": not bool(actual),
            "verifier_error": None,
        }
    except (ValueError, TypeError, TimeoutError) as error:
        return {
            "correct": False,
            "parse_failure": True,
            "verifier_error": f"{type(error).__name__}: {error}",
        }


def score_response(
    text: str,
    gold: str,
    *,
    finish_reason: str,
    thinking: bool,
    integer_task: bool,
    problem: str = "",
) -> dict:
    """
    Computes decoupled strict (admission/v1-parity) and lenient (fault-tolerant semantic) grades,
    and cleanly separates truncation (`finish_reason == 'length'`) from natural-stop parse failure.
    """
    truncated = finish_reason == "length"
    stopped = finish_reason == "stop"
    strict_ans = final_answer_strict(text, thinking=thinking) if stopped else None
    v1_ans = final_answer(text, thinking=thinking) if stopped else None
    lenient_ans = final_answer_lenient(text, thinking=thinking) if stopped else None

    if integer_task:
        strict_ok = bool(stopped and strict_ans is not None and exact_integer_equivalent(strict_ans, gold))
        v1_ok = bool(stopped and v1_ans is not None and exact_integer_equivalent(v1_ans, gold))
        lenient_ok = bool(stopped and lenient_ans is not None and exact_integer_equivalent(lenient_ans, gold))
        strict_pf = bool(stopped and strict_ans is None)
        lenient_pf = bool(stopped and lenient_ans is None)
        verr = None
    else:
        g_strict = grade_symbolic_math(strict_ans, gold, problem=problem) if stopped else {"correct": False, "parse_failure": True, "verifier_error": None}
        g_v1 = grade_symbolic_math(v1_ans, gold, legacy=True) if stopped else {"correct": False, "parse_failure": True, "verifier_error": None}
        g_lenient = grade_symbolic_math(lenient_ans, gold, problem=problem) if stopped else {"correct": False, "parse_failure": True, "verifier_error": None}
        strict_ok = bool(stopped and g_strict["correct"])
        v1_ok = bool(stopped and g_v1["correct"])
        lenient_ok = bool(stopped and g_lenient["correct"])
        strict_pf = bool(stopped and g_strict["parse_failure"])
        lenient_pf = bool(stopped and g_lenient["parse_failure"])
        verr = g_lenient["verifier_error"] or g_strict["verifier_error"]

    clean_sft_admissible = bool(
        stopped
        and strict_ok
        and not has_shortcut_tokens(text)
        and ("<|im_" not in text)
        and (not thinking or (text.count("</think>") == 1 and "<think>" not in text))
    )
    return {
        "finish_reason": finish_reason,
        "truncated": truncated,
        "stopped": stopped,
        "strict_extracted": strict_ans,
        "v1_extracted": v1_ans,
        "lenient_extracted": lenient_ans,
        "strict_correct": strict_ok,
        "v1_correct": v1_ok,
        "lenient_correct": lenient_ok,
        "natural_stop_strict_parse_failure": strict_pf,
        "natural_stop_lenient_parse_failure": lenient_pf,
        "combined_unusable_rate_indicator": bool(truncated or lenient_pf or verr is not None),
        "verifier_error": verr,
        "sft_clean_admissible": clean_sft_admissible,
    }
