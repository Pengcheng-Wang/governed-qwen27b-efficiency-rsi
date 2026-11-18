"""CPU regression controls for the actual parameter-export qualification prefix."""
from __future__ import annotations

import ast
import json
import math
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
from safetensors.torch import save_file

TRAINER = Path(__file__).resolve().parents[1] / "code" / "train_systems_v2.py"


class MergeExportFiniteTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        module = ast.parse(TRAINER.read_text(encoding="utf-8"), str(TRAINER))
        function = next(
            node for node in module.body
            if isinstance(node, ast.FunctionDef) and node.name == "run_isolated_nll_only"
        )
        branch = next(
            node for node in function.body
            if isinstance(node, ast.If)
            and any(
                isinstance(part, ast.Attribute) and part.attr == "qualify_merged_dir"
                for part in ast.walk(node.test)
            )
        )
        end = next(
            index for index, node in enumerate(branch.body)
            if isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id == "model" for target in node.targets)
        )
        cls.prefix = compile(
            ast.Module(body=branch.body[:end], type_ignores=[]), str(TRAINER), "exec"
        )

    def _qualify(self, *, overflow=False, alpha=1):
        with tempfile.TemporaryDirectory(prefix="merge_export_finite_") as tmp:
            root = Path(tmp)
            source, export, adapter = (root / name for name in ("source", "export", "adapter"))
            for directory in (source, export, adapter):
                directory.mkdir()
            state = {"layer.weight": torch.ones(2, 2, dtype=torch.bfloat16)}
            state.update({
                f"mtp.tensor{index}": torch.tensor([0.0, 1.0], dtype=torch.bfloat16)
                for index in range(15)
            })
            exported = {key: value.clone() for key, value in state.items()}
            if overflow or not math.isfinite(alpha):
                exported["layer.weight"].fill_(torch.finfo(torch.bfloat16).max)
            else:
                exported["layer.weight"].fill_(2.0)
            for directory, tensors in ((source, state), (export, exported)):
                save_file(tensors, directory / "model.safetensors")
                (directory / "model.safetensors.index.json").write_text(json.dumps({
                    "weight_map": {key: "model.safetensors" for key in tensors}
                }), encoding="utf-8")
            value = 1e20 if overflow else 1.0
            save_file({
                "base_model.model.layer.lora_A.weight": torch.full((1, 2), value),
                "base_model.model.layer.lora_B.weight": torch.full((2, 1), value),
            }, adapter / "adapter_model.safetensors")
            (adapter / "adapter_config.json").write_text(
                json.dumps({"r": 1, "lora_alpha": alpha}), encoding="utf-8"
            )
            namespace = {
                "ROOT": root,
                "Path": Path,
                "json": json,
                "math": math,
                "torch": torch,
                "spec": {"model_directory": "source", "native_mtp_tensor_count": 15},
                "args": SimpleNamespace(
                    qualify_merged_dir=str(export), qualify_adapter_dir=str(adapter)
                ),
            }
            exec(self.prefix, namespace)
            return namespace["checked_adapted_tensors"], namespace["param_max_bf16_ulp"]

    def test_exact_finite_merge(self):
        self.assertEqual(self._qualify(), (1, 0))

    def test_finite_factors_overflowing_reference_are_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "Non-finite BF16 merge reference"):
            self._qualify(overflow=True)

    def test_nonfinite_scaling_is_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "Non-finite LoRA scaling"):
            self._qualify(alpha=float("inf"))


if __name__ == "__main__":
    unittest.main()
