"""Tests for the local wake word and the fixed voice commands."""

from __future__ import annotations

import json
from pathlib import Path
import queue
import unittest
from unittest import mock

from vadermind.listening import ListeningError
from vadermind.wake_word import (
    DECOYS,
    VoskWakeWord,
    is_confirmation,
    is_shutdown_request,
    is_silence_request,
    normalize,
    persona_mode_request,
    wait_for_wake_word,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
VOSK_MODEL = PROJECT_ROOT / "models" / "vosk" / "vosk-model-small-es-0.42"


class FakeRecognizer:
    """Scripted Vosk: each frame gives (is_final, words, confidence)."""

    def __init__(self, script: list[tuple[bool, str, float]]) -> None:
        self.script = list(script)
        self.current: tuple[bool, str, float] = (False, "", 1.0)
        self.resets = 0

    def AcceptWaveform(self, frame: bytes) -> bool:  # noqa: N802 (Vosk API)
        self.current = self.script.pop(0) if self.script else (False, "", 1.0)
        return self.current[0]

    def Result(self) -> str:  # noqa: N802
        _, text, conf = self.current
        words = [{"word": word, "conf": conf} for word in text.split()]
        return json.dumps({"text": text, "result": words})

    def PartialResult(self) -> str:  # noqa: N802
        return json.dumps({"partial": self.current[1]})

    def Reset(self) -> None:  # noqa: N802
        self.resets += 1


def _frames(count: int) -> "queue.Queue[bytes]":
    source: "queue.Queue[bytes]" = queue.Queue()
    for _ in range(count):
        source.put(b"\x00" * 1024)
    return source


class NormalizeTests(unittest.TestCase):
    def test_accents_case_and_punctuation(self) -> None:
        self.assertEqual(normalize("¡Oye,  VÁDER!"), "oye vader")


class VoskWakeWordTests(unittest.TestCase):
    def test_partial_guess_never_triggers(self) -> None:
        # A lone "oye" used to show up as a partial "oye vader" and wake Vader.
        detector = VoskWakeWord("", recognizer=FakeRecognizer([(False, "oye vader", 1.0)]))
        self.assertFalse(detector.feed(b""))

    def test_confident_final_result_triggers(self) -> None:
        detector = VoskWakeWord("", recognizer=FakeRecognizer([(True, "oye vader", 0.9)]))
        self.assertTrue(detector.feed(b""))

    def test_phrase_inside_longer_result_triggers(self) -> None:
        detector = VoskWakeWord("", recognizer=FakeRecognizer([(True, "ya oye vader", 0.8)]))
        self.assertTrue(detector.feed(b""))

    def test_low_confidence_does_not_trigger(self) -> None:
        detector = VoskWakeWord("", recognizer=FakeRecognizer([(True, "oye vader", 0.46)]))
        self.assertFalse(detector.feed(b""))
        stricter = VoskWakeWord(
            "", recognizer=FakeRecognizer([(True, "oye vader", 0.7)]), min_confidence=0.8
        )
        self.assertFalse(stricter.feed(b""))

    def test_vader_alone_or_split_does_not_trigger(self) -> None:
        detector = VoskWakeWord(
            "",
            recognizer=FakeRecognizer(
                [(True, "vader", 1.0), (True, "oye", 1.0), (True, "vas a ver", 1.0)]
            ),
        )
        self.assertFalse(any(detector.feed(b"") for _ in range(3)))

    def test_veider_variant_wakes(self) -> None:
        detector = VoskWakeWord("", recognizer=FakeRecognizer([(True, "oye bei der", 0.8)]))
        self.assertTrue(detector.feed(b""))
        self.assertEqual(detector.accepted, ["oye vader", "oye bei der"])

    def test_variants_can_be_turned_off(self) -> None:
        detector = VoskWakeWord("", recognizer=FakeRecognizer([(True, "oye bei der", 1.0)]), variants=())
        self.assertFalse(detector.feed(b""))

    def test_custom_phrase(self) -> None:
        detector = VoskWakeWord("", "Darth Vader", recognizer=FakeRecognizer([(True, "darth vader", 1.0)]))
        self.assertTrue(detector.feed(b""))

    def test_decoys_cover_chilean_look_alikes_but_not_the_phrase(self) -> None:
        self.assertNotIn("oye vader", DECOYS)
        for decoy in ("vas a ver", "a ver", "oye", "vader", "va a llover"):
            self.assertIn(decoy, DECOYS)
        self.assertNotIn("oye vale", DECOYS)  # it swallowed a real "Oye, Vader"

    def test_missing_model_is_a_clear_error(self) -> None:
        try:
            import vosk  # noqa: F401
        except ImportError:
            self.skipTest("vosk no instalado")
        with self.assertRaises(ListeningError) as caught:
            VoskWakeWord(PROJECT_ROOT / "models" / "no-existe")
        self.assertIn("VADER_WAKE_MODEL_PATH", str(caught.exception))

    @unittest.skipUnless(VOSK_MODEL.is_dir(), "modelo Vosk no descargado")
    def test_real_model_ignores_silence(self) -> None:
        detector = VoskWakeWord(VOSK_MODEL)
        self.assertFalse(any(detector.feed(b"\x00" * 1024) for _ in range(60)))


class WaitForWakeWordTests(unittest.TestCase):
    def test_returns_true_on_detection_and_resets(self) -> None:
        recognizer = FakeRecognizer([(False, "", 1.0), (False, "", 1.0), (True, "oye vader", 1.0)])
        detector = VoskWakeWord("", recognizer=recognizer)
        self.assertTrue(wait_for_wake_word(detector, frames=lambda: _frames(5)))
        self.assertEqual(recognizer.resets, 2)  # before listening and after hearing it

    def test_stops_when_asked(self) -> None:
        detector = VoskWakeWord("", recognizer=FakeRecognizer([]))
        calls = iter([False, False, True])
        self.assertFalse(
            wait_for_wake_word(detector, frames=lambda: _frames(5), should_stop=lambda: next(calls))
        )

    def test_silent_microphone_raises(self) -> None:
        detector = VoskWakeWord("", recognizer=FakeRecognizer([]))
        empty: "queue.Queue[bytes]" = queue.Queue()
        original = queue.Queue.get

        def fast_get(self, block=True, timeout=None):  # type: ignore[no-untyped-def]
            return original(self, block, 0.01)

        with mock.patch.object(queue.Queue, "get", fast_get):
            with self.assertRaises(ListeningError):
                wait_for_wake_word(detector, frames=lambda: empty)


class LocalCommandTests(unittest.TestCase):
    def test_shutdown_phrases(self) -> None:
        for text in ("Vader, apágate.", "apagate", "Apaga la Raspberry", "Desconéctate, Vader"):
            self.assertTrue(is_shutdown_request(text), text)

    def test_tv_and_music_are_not_shutdown(self) -> None:
        for text in ("Apaga la tele", "apaga el televisor en 10 minutos", "pausa la música"):
            self.assertFalse(is_shutdown_request(text), text)

    def test_silence_commands(self) -> None:
        for text in ("Silencio", "Vader, silencio.", "Oye Vader, cállate", "Basta ya",
                     "eso es todo, Vader", "Olvídalo", "silencio por favor"):
            self.assertTrue(is_silence_request(text), text)

    def test_silence_inside_another_order_is_not_a_command(self) -> None:
        for text in ("Pon la canción Silencio", "baja el volumen que está muy fuerte",
                     "basta de música, pon noticias", "¿qué es el silencio?", ""):
            self.assertFalse(is_silence_request(text), text)

    def test_persona_mode_commands(self) -> None:
        cases = {
            "Modo canon.": "canon",
            "Oye Vader, modo chileno": "chileno",
            "Vader, ponte en modo chileno por favor": "chileno",
            "Cambia a modo canon": "canon",
            "Modo cañón": "canon",  # how the transcription may spell it
            "vuelve al modo chileno": "chileno",
        }
        for text, mode in cases.items():
            self.assertEqual(persona_mode_request(text), mode, text)

    def test_mode_inside_another_question_is_not_a_command(self) -> None:
        for text in ("¿Qué es el modo canon?", "explícame el modo chileno de Vader",
                     "modo avión", "pon la canción Modo Chileno de Los Tres", ""):
            self.assertIsNone(persona_mode_request(text), text)

    def test_confirmation(self) -> None:
        for text in ("Sí.", "si, apágate", "Dale", "Confirmo", "ya"):
            self.assertTrue(is_confirmation(text), text)
        for text in ("No", "mejor no", "", "espera"):
            self.assertFalse(is_confirmation(text), text)


if __name__ == "__main__":
    unittest.main()
