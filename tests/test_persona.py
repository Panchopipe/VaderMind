"""Tests for the persona prompts that shape Vader's answers."""

from __future__ import annotations

import unittest

from vadermind.persona import build_system_prompt
from vadermind.persona_evaluation import PERSONA_QUESTIONS


class PersonaPromptTests(unittest.TestCase):
    def test_both_modes_set_the_short_answer_rule(self) -> None:
        for mode in ("canon", "chileno"):
            for language, rule in (("es", "una o dos oraciones"), ("en", "one or two sentences")):
                with self.subTest(mode=mode, language=language):
                    prompt = build_system_prompt(language=language, mode=mode)
                    self.assertIn(rule, prompt)
                    # A numeric length reference made the model aim for it.
                    self.assertNotIn("400", prompt)

    def test_both_modes_keep_voice_only_rule(self) -> None:
        self.assertIn("respiración", build_system_prompt(mode="canon"))
        self.assertIn("respiración", build_system_prompt(mode="chileno"))

    def test_only_chileno_allows_swearing(self) -> None:
        self.assertIn("garabatos", build_system_prompt(mode="chileno"))
        self.assertIn("nunca vulgar", build_system_prompt(mode="canon"))

    def test_tone_examples_do_not_reuse_benchmark_questions(self) -> None:
        for mode in ("canon", "chileno"):
            prompt = build_system_prompt(mode=mode).casefold()
            for question in PERSONA_QUESTIONS:
                with self.subTest(mode=mode, question=question):
                    self.assertNotIn(question.casefold(), prompt)

    def test_unknown_mode_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            build_system_prompt(mode="ninos")


if __name__ == "__main__":
    unittest.main()
