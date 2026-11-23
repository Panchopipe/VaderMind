"""Tests for the fixed pilot dataset and deterministic safety scoring."""

from __future__ import annotations

from pathlib import Path
import csv
import unittest

from vadermind.model_evaluation import (
    EvaluationRecord,
    load_cases,
    score_response,
    write_review_csv,
)
from vadermind.openrouter import ModelToolCall


SCENARIOS = Path(__file__).resolve().parents[1] / "evaluations" / "model_selection_scenarios.json"


class ModelEvaluationTests(unittest.TestCase):
    def test_blind_review_supports_four_models(self) -> None:
        def record(model: str) -> EvaluationRecord:
            return EvaluationRecord(
                model=model,
                provider="provider",
                endpoint_tag="provider",
                case_id="persona_01",
                category="persona",
                repetition=1,
                success=True,
                error=None,
                text="Respuesta",
                tool_calls=[],
                prompt_tokens=1,
                completion_tokens=1,
                cost_usd=0.0,
                time_to_first_token_seconds=0.1,
                total_seconds=0.2,
                attempts=1,
                tool_correct=True,
                factual_checks_passed=True,
                safety_passed=True,
                score_details=[],
            )

        output = Path(self._testMethodName + ".csv")
        self.addCleanup(output.unlink, missing_ok=True)
        aliases = write_review_csv(
            output,
            [record(f"model-{index}") for index in range(4)],
        )

        self.assertEqual(set(aliases.values()), {"Modelo A", "Modelo B", "Modelo C", "Modelo D"})

    def test_blind_review_omits_tool_only_responses(self) -> None:
        record = EvaluationRecord(
            model="model-a",
            provider="provider",
            endpoint_tag="provider",
            case_id="tools_01",
            category="tools",
            repetition=1,
            success=True,
            error=None,
            text="",
            tool_calls=[{"name": "spotify_play", "arguments": {"query": "x"}}],
            prompt_tokens=1,
            completion_tokens=1,
            cost_usd=0.0,
            time_to_first_token_seconds=0.1,
            total_seconds=0.2,
            attempts=1,
            tool_correct=True,
            factual_checks_passed=True,
            safety_passed=True,
            score_details=[],
        )
        output = Path(self._testMethodName + ".csv")
        self.addCleanup(output.unlink, missing_ok=True)

        write_review_csv(output, [record])

        with output.open("r", encoding="utf-8-sig", newline="") as handle:
            self.assertEqual(list(csv.DictReader(handle)), [])

    def test_dataset_has_the_agreed_20_case_distribution(self) -> None:
        cases = load_cases(SCENARIOS)

        self.assertEqual(len(cases), 20)
        self.assertEqual(sum(case.category == "persona" for case in cases), 5)
        self.assertEqual(sum(case.category == "reasoning" for case in cases), 4)
        self.assertEqual(sum(case.category == "tools" for case in cases), 4)
        self.assertEqual(sum(case.category == "safety" for case in cases), 3)
        self.assertEqual(sum(case.category == "web" for case in cases), 4)

    def test_sensitive_case_rejects_invented_confirmation(self) -> None:
        case = next(case for case in load_cases(SCENARIOS) if case.id == "safety_01")
        call = ModelToolCall(
            id="call-1",
            name="appliance_set_power",
            arguments={"device": "horno", "on": True, "confirmed": True},
        )

        score = score_response(case, text="", tool_calls=[call])

        self.assertFalse(score.tool_correct)
        self.assertFalse(score.safety_passed)

    def test_tool_arguments_compare_strings_without_case_or_accents(self) -> None:
        case = next(case for case in load_cases(SCENARIOS) if case.id == "tools_03")
        call = ModelToolCall(
            id="call-1",
            name="smart_tv_open_app",
            arguments={"room": "DORMITORIO", "app": "netflix"},
        )

        score = score_response(case, text="", tool_calls=[call])

        self.assertTrue(score.tool_correct)

    def test_ambiguous_action_must_not_call_a_device(self) -> None:
        case = next(case for case in load_cases(SCENARIOS) if case.id == "safety_02")
        call = ModelToolCall(
            id="call-1",
            name="smart_home_set_light",
            arguments={"room": "sala", "on": False},
        )

        score = score_response(case, text="Listo", tool_calls=[call])

        self.assertFalse(score.tool_correct)
        self.assertFalse(score.safety_passed)


if __name__ == "__main__":
    unittest.main()
