"""Regression controls for discrete math solution-set notation."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
sys.path.insert(0, str(ROOT / "eval"))

from verifiers_v2 import score_response  # noqa: E402


class SemanticSolutionSetTests(unittest.TestCase):
    def _score(self, answer, gold="81, 256"):
        result = score_response(
            "reasoning</think>\n" + rf"\boxed{{{answer}}}",
            gold,
            finish_reason="stop",
            thinking=True,
            integer_task=False,
            problem="Find all real solutions.",
        )
        self.assertIsNone(result["verifier_error"])
        return result

    def test_solution_set_members_are_unordered(self):
        for answer, legacy_expected in [("81,256", False), ("256, 81", True)]:
            with self.subTest(answer=answer):
                result = self._score(answer)
                self.assertTrue(result["lenient_correct"])
                self.assertTrue(result["strict_correct"])
                self.assertEqual(result["v1_correct"], legacy_expected)
        grouped_integer = self._score("81,256", gold="81256")
        self.assertTrue(grouped_integer["lenient_correct"])
        self.assertTrue(grouped_integer["v1_correct"])

    def test_intervals_and_partial_answers_are_not_solution_sets(self):
        for answer in ["(81,256)", "81 < x < 256", "100", "81", "81,256,512", "81256"]:
            with self.subTest(answer=answer):
                result = self._score(answer)
                self.assertFalse(result["lenient_correct"])
                self.assertFalse(result["strict_correct"])


if __name__ == "__main__":
    unittest.main()
