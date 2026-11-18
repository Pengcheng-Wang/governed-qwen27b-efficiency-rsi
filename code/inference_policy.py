"""Native template options: unsupported effort labels must not masquerade as active controls."""
from __future__ import annotations


def template_options(policy: dict) -> dict:
    result = {"enable_thinking": policy["thinking"]}
    if policy["effort"] is not None:
        result["reasoning_effort"] = policy["effort"]
    return result
