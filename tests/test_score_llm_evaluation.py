"""Tests for importing the completed blind human review."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import unittest

from scripts.score_llm_evaluation import _load_human_scores


class ScoreLlmEvaluationTests(unittest.TestCase):
    def test_tool_only_rows_do_not_require_human_scores(self) -> None:
        review = Path("test_score_review.csv")
        key = Path("test_score_key.json")
        self.addCleanup(review.unlink, missing_ok=True)
        self.addCleanup(key.unlink, missing_ok=True)
        key.write_text(json.dumps({"Modelo A": "model-a"}), encoding="utf-8")
        with review.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=(
                    "alias",
                    "case_id",
                    "respuesta",
                    "calidad_1_a_5",
                    "personalidad_1_a_5",
                ),
            )
            writer.writeheader()
            writer.writerow(
                {
                    "alias": "Modelo A",
                    "case_id": "persona_01",
                    "respuesta": "Respuesta visible",
                    "calidad_1_a_5": "4",
                    "personalidad_1_a_5": "5",
                }
            )
            writer.writerow(
                {
                    "alias": "Modelo A",
                    "case_id": "tools_01",
                    "respuesta": "",
                    "calidad_1_a_5": "",
                    "personalidad_1_a_5": "",
                }
            )

        scores = _load_human_scores(review, key)

        self.assertEqual(scores, {"model-a": 4.5})


if __name__ == "__main__":
    unittest.main()
