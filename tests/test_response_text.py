"""Tests for assistant response cleanup."""

from __future__ import annotations

import unittest

from vadermind.response_text import clean_assistant_text


class ResponseTextTests(unittest.TestCase):
    """Verify that display and speech text stays voice-friendly."""

    def test_clean_assistant_text_removes_stage_directions(self) -> None:
        """Physical action notes should not appear in spoken output."""

        text = "*respira pesadamente* Estoy presente. *cruza los brazos amenazadoramente*"

        self.assertEqual(clean_assistant_text(text), "Estoy presente.")

    def test_clean_assistant_text_removes_markdown_emphasis(self) -> None:
        """Markdown emphasis should not create awkward TTS pauses."""

        text = "Soy **Lord Vader**, no tu amigo de copete del fin de semana."

        self.assertEqual(
            clean_assistant_text(text),
            "Soy Lord Vader, no tu amigo de copete del fin de semana.",
        )

    def test_weon_follows_spoken_chilean_comma_rule(self) -> None:
        """A closing weón loses its comma; an opening Weón gains one."""

        cases = (
            (
                "Me tiene trabajando por el sueldo mínimo, weón.",
                "Me tiene trabajando por el sueldo mínimo weón.",
            ),
            (
                "Aquí andamos, weón, lidiando con la inflación.",
                "Aquí andamos weón, lidiando con la inflación.",
            ),
            (
                "Bailo con los stormtroopers. Weón si la vida no es pura tragedia.",
                "Bailo con los stormtroopers. Weón, si la vida no es pura tragedia.",
            ),
            (
                "Huevón la vida es corta.",
                "Huevón, la vida es corta.",
            ),
        )
        for text, expected in cases:
            with self.subTest(text=text):
                self.assertEqual(clean_assistant_text(text), expected)

    def test_correct_weon_and_canon_text_stay_unchanged(self) -> None:
        for text in (
            "Weón, la vida no es pura tragedia. El Emperador me explota weón.",
            "Fue mi maestro. Luego, mi enemigo.",
        ):
            with self.subTest(text=text):
                self.assertEqual(clean_assistant_text(text), text)

if __name__ == "__main__":
    unittest.main()
