#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export UV_CACHE_DIR="$ROOT/cache/uv"
export CAUSAL_CONV1D_FORCE_BUILD=TRUE
export TORCH_CUDA_ARCH_LIST=8.0
export MAX_JOBS=4
if [[ ! -x .venv-training/bin/python ]]; then
  python3 -m venv --system-site-packages .venv-training
fi
.venv-training/bin/python -c 'import torch; assert torch.__version__ == "2.13.0+cu129"'
uv pip install --python .venv-training/bin/python \
  --constraint configs/constraints-image.txt \
  --requirement configs/requirements-training.txt \
  --no-build-isolation --no-deps
# Verify math-verify is importable (install cleanly if absent in base image)
if ! .venv-training/bin/python -c 'import math_verify' 2>/dev/null; then
  uv pip install --python .venv-training/bin/python --constraint configs/constraints-image.txt math-verify
fi
.venv-training/bin/python - <<'PY'
import importlib.metadata as metadata
import json
import torch
import transformers
from packaging.requirements import Requirement
for package in ("fla-core", "flash-linear-attention", "causal-conv1d", "trl"):
    for raw in metadata.requires(package) or []:
        requirement = Requirement(raw)
        if requirement.marker and not requirement.marker.evaluate():
            continue
        actual = metadata.version(requirement.name)
        assert requirement.specifier.contains(actual), (package, raw, actual)
for line in open("configs/constraints-image.txt"):
    if not line.strip():
        continue
    requirement = Requirement(line.strip())
    actual = metadata.version(requirement.name)
    assert requirement.specifier.contains(actual), (line, actual)
from fla.ops.gated_delta_rule import chunk_gated_delta_rule
from causal_conv1d import causal_conv1d_fn
from trl import GRPOConfig, GRPOTrainer
import math_verify
from transformers.models.qwen3_5 import modeling_qwen3_5 as model
assert model.chunk_gated_delta_rule is not None
assert model.causal_conv1d_fn is not None
assert torch.__version__ == "2.13.0+cu129"
assert transformers.__version__ == "5.10.4"
versions = {p: metadata.version(p) for p in [
    "torch", "transformers", "vllm", "trl", "peft", "fla-core",
    "flash-linear-attention", "causal-conv1d", "triton", "math-verify"
]}
from pathlib import Path
Path("results").mkdir(exist_ok=True)
Path("results/training_environment.json").write_text(json.dumps({
    "status": "PASS",
    "versions": versions,
    "base_packages_replaced": False,
    "cuDNN_repair": "CUDNN_HOME and process-local loader path point at the inherited vendor library",
}, indent=2))
print(json.dumps(versions, indent=2))
print("TRAINING_IMPORTS_OK")
PY
