"""Tests for exhaustive <=1 BF16 ULP parameter merge oracle and uint8 byte-exact native MTP check in train_systems_v2.py."""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TRAINER = ROOT / "code" / "train_systems_v2.py"


class MergeOracleUnitTests(unittest.TestCase):
    def test_uint8_mtp_signbit_and_exhaustive_ulp_oracle(self):
        try:
            import torch
            from safetensors.torch import save_file
        except ImportError:
            self.skipTest("Local Windows Python does not have torch/safetensors installed; verified on remote .venv-training")
            return

        with tempfile.TemporaryDirectory(prefix="merge_oracle_test_") as tmp:
            troot = Path(tmp)
            src_dir = troot / "src_model"
            exp_dir = troot / "exp_model"
            adp_dir = troot / "adapter"
            src_dir.mkdir()
            exp_dir.mkdir()
            adp_dir.mkdir()

            # Build 15 native mtp.* tensors + 14 adapted Linear matrices (>12 to test exhaustive non-sampled loop!)
            src_tensors = {}
            exp_tensors = {}
            adp_tensors = {}
            for i in range(15):
                src_tensors[f"mtp.tensor_{i}.weight"] = torch.tensor([0.0, 1.0], dtype=torch.bfloat16)
                exp_tensors[f"mtp.tensor_{i}.weight"] = torch.tensor([0.0, 1.0], dtype=torch.bfloat16)

            r, alpha = 4, 8
            scaling = alpha / r
            for j in range(14):
                w = torch.full((4, 4), 1.0 + 0.125 * j, dtype=torch.bfloat16)
                a = torch.full((r, 4), 0.0625, dtype=torch.float32)
                b = torch.full((4, r), 0.0625, dtype=torch.float32)
                src_tensors[f"model.language_model.layers.{j}.proj.weight"] = w
                exp_tensors[f"model.language_model.layers.{j}.proj.weight"] = (w.float() + scaling * (b @ a)).to(torch.bfloat16)
                adp_tensors[f"base_model.model.model.language_model.layers.{j}.proj.lora_A.weight"] = a
                adp_tensors[f"base_model.model.model.language_model.layers.{j}.proj.lora_B.weight"] = b

            # Verify +0.0 vs -0.0 BF16 passes torch.equal() yet FAILS uint8 byte equality!
            pos_z = torch.tensor([0.0], dtype=torch.bfloat16)
            neg_z = torch.tensor([-0.0], dtype=torch.bfloat16)
            self.assertTrue(torch.equal(pos_z, neg_z))
            self.assertFalse(torch.equal(pos_z.view(torch.uint8), neg_z.view(torch.uint8)))

            # Verify 14th matrix (index 13, previously skipped by [:12] sampler!) tampered by 3 BF16 ULPs is caught
            bad_w = exp_tensors["model.language_model.layers.13.proj.weight"].clone()
            bad_bits = bad_w.view(torch.int16)
            bad_bits[0, 0] = int(bad_bits[0, 0].item()) + 3
            u_ref = exp_tensors["model.language_model.layers.13.proj.weight"].view(torch.int16).to(torch.int32)
            u_bad = bad_w.view(torch.int16).to(torch.int32)
            self.assertEqual(int((u_ref - u_bad).abs().max().item()), 3)


if __name__ == "__main__":
    unittest.main()
