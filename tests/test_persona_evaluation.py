"""Tests for the persona benchmark text metrics."""

from __future__ import annotations

import unittest

from vadermind.persona_evaluation import (
    PERSONA_QUESTIONS,
    count_sentences,
    measure_answer,
    summarize,
)


class PersonaEvaluationTests(unittest.TestCase):
    def test_bank_has_fifteen_unique_questions(self) -> None:
        self.assertEqual(len(PERSONA_QUESTIONS), 15)
        self.assertEqual(len(set(PERSONA_QUESTIONS)), 15)

    def test_counts_sentences_by_terminal_punctuation(self) -> None:
        self.assertEqual(count_sentences("Mi estado es irrelevante. Habla."), 2)
        self.assertEqual(count_sentences("¿Me desafías? Imprudente!"), 2)
        self.assertEqual(count_sentences(""), 0)

    def test_detects_canon_habits(self) -> None:
        metrics = measure_answer(
            "La debilidad no se tolera en el Imperio. "
            "Respirando hondo, te pregunto: ¿obedecerás?"
        )

        self.assertTrue(metrics.ends_with_question)
        self.assertEqual(metrics.filler_words, 2)
        self.assertEqual(metrics.physical_descriptions, 1)
        self.assertFalse(metrics.breaks_character)

    def test_flags_character_break_and_markdown(self) -> None:
        metrics = measure_answer("**Nota:** soy un modelo de lenguaje.")

        self.assertTrue(metrics.breaks_character)
        self.assertGreater(metrics.markdown_marks, 0)

    def test_counts_chilean_terms_voseo_and_swears(self) -> None:
        metrics = measure_answer("Ya po, weón, ¿querís que te explique? El país es fome.")

        self.assertEqual(metrics.swears, 1)
        # po, weon, fome and the voseo form "queris"; "pais" is not voseo.
        self.assertEqual(metrics.chilean_terms, 4)

    def test_weon_punctuation_follows_spoken_chilean_rhythm(self) -> None:
        good = measure_answer(
            "Me tiene trabajando por el sueldo mínimo weón. Weón, la vida no es pura tragedia."
        )
        comma_before_end = measure_answer("Me tiene trabajando por el sueldo mínimo, weón.")
        wrapped = measure_answer("Con cumbia, weón, bailo con los stormtroopers.")
        opening_without_comma = measure_answer("Bailo. Weón la vida no es pura tragedia.")

        self.assertEqual((good.weon_uses, good.weon_bad_punctuation), (2, 0))
        self.assertEqual(comma_before_end.weon_bad_punctuation, 1)
        self.assertEqual(wrapped.weon_bad_punctuation, 1)
        self.assertEqual(opening_without_comma.weon_bad_punctuation, 1)

    def test_summary_reports_medians_and_rates(self) -> None:
        summary = summarize([measure_answer("Silencio."), measure_answer("x" * 250 + ".")])

        self.assertEqual(summary["respuestas"], 2)
        self.assertEqual(summary["sobre_200_pct"], 50)
        self.assertEqual(summary["caracteres_max"], 251)


if __name__ == "__main__":
    unittest.main()
