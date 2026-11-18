#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export UV_CACHE_DIR="$ROOT/cache/uv"
if [[ ! -x .venv-quant/bin/python ]]; then
  python3 -m venv --system-site-packages .venv-quant
fi
.venv-quant/bin/python - <<'PY'
import subprocess
import sysconfig
from pathlib import Path
root = Path.cwd()
kernel_site = subprocess.check_output([
    str(root / ".venv-training/bin/python"), "-c",
    "import sysconfig; print(sysconfig.get_path('purelib'))",
], text=True).strip()
Path(sysconfig.get_path("purelib"), "study_training_kernels.pth").write_text(kernel_site + "\n")
PY
uv pip install --python .venv-quant/bin/python --no-deps \
  --requirement configs/requirements-quant.txt
.venv-quant/bin/python - <<'PY'
import importlib.metadata as metadata
import json
from pathlib import Path
from packaging.requirements import Requirement
for package in ("llmcompressor", "compressed-tensors", "auto-round", "accelerate"):
    for raw in metadata.requires(package) or []:
        requirement = Requirement(raw)
        if requirement.marker and not requirement.marker.evaluate():
            continue
        actual = metadata.version(requirement.name)
        assert requirement.specifier.contains(actual), (package, raw, actual)
for package, expected in {
    "torch": "2.13.0+cu129", "transformers": "5.10.4",
    "vllm": "0.29.0+cu129", "triton": "3.7.1", "numpy": "2.2.6"
}.items():
    assert metadata.version(package) == expected
from llmcompressor import oneshot
from llmcompressor.modifiers.gptq import GPTQModifier
from transformers.models.qwen3_5 import modeling_qwen3_5
assert modeling_qwen3_5.chunk_gated_delta_rule is not None
assert modeling_qwen3_5.causal_conv1d_fn is not None
versions = {name: metadata.version(name) for name in (
    "torch", "transformers", "llmcompressor", "compressed-tensors", "auto-round", "accelerate"
)}
Path("results").mkdir(exist_ok=True)
Path("results/quant_environment.json").write_text(json.dumps({
    "status": "PASS",
    "versions": versions,
    "scope": "Quantization-only venv; base serving environment is untouched",
    "kernel_overlay": "Study-local validated training venv; identical vendor ABI",
}, indent=2))
print(json.dumps(versions, indent=2))
print("QUANT_IMPORTS_OK")
PY
