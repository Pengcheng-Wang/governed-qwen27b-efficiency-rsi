"""Transparent partial algorithmic FLOP accounting, preserved from v1 with explicit omissions."""
from __future__ import annotations


def training_matmul_flops(
    lengths: list[int],
    base_linear_parameters: int,
    adapter_parameters: int,
    *,
    full_attention_layers: int = 16,
    query_heads: int = 24,
    head_dim: int = 256,
) -> tuple[float, float]:
    if any(length <= 0 for length in lengths) or min(base_linear_parameters, adapter_parameters) < 0:
        raise ValueError("Invalid FLOP inventory")
    tokens = sum(lengths)
    projections = tokens * (4 * base_linear_parameters + 6 * adapter_parameters)
    attention_lower = sum(length * (length + 1) for length in lengths) * (
        2 * query_heads * head_dim * full_attention_layers * 3
    )
    attention_upper = sum(length * length for length in lengths) * (
        4 * query_heads * head_dim * full_attention_layers * 3
    )
    return projections + attention_lower, projections + attention_upper


def estimated_mfu(flops: tuple[float, float], seconds: float, gpus: int) -> dict:
    if seconds <= 0 or gpus <= 0:
        raise ValueError("MFU needs positive elapsed time and participating GPU count")
    denominator = seconds * gpus * 312e12
    return {
        "lower": flops[0] / denominator,
        "upper": flops[1] / denominator,
        "peak_dense_bf16_tflops_per_a100": 312,
        "label": "Partial algorithmic matmul-only MFU estimate, not complete hybrid-model MFU",
        "coverage": "LM projection, MLP, logits head and full attention; no vision or unused MTP",
        "omitted": "Gated DeltaNet recurrence, convolution, normalization, activation and optimizer arithmetic",
        "excluded_overhead": "Padding and checkpoint recomputation excluded from FLOPs but included in time",
        "interval_kind": "Causal/full-square attention accounting conventions, not a confidence interval",
        "hfu": None,
    }
