"""Tests for speech-to-text transcription and its benchmark metrics."""

from __future__ import annotations

import base64
import json
import os
from pathlib import Path
import unittest
from unittest import mock

import httpx

from vadermind.stt_evaluation import (
    STT_PHRASES,
    SttRow,
    normalize_transcript,
    summarize_model,
    word_errors,
)
from vadermind.config import load_settings
from vadermind.voice_input import (
    CHILEAN_STT_PROMPT,
    FallbackSpeechToText,
    TranscriptionError,
    VercelSpeechToText,
    build_speech_to_text,
)


def _client(handler) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler))


class VercelSpeechToTextTests(unittest.TestCase):
    def test_sends_base64_wav_with_model_header_and_no_training(self) -> None:
        captured: dict[str, object] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured["model"] = request.headers["ai-model-id"]
            captured["body"] = json.loads(request.content)
            return httpx.Response(
                200,
                json={
                    "text": " Pon el HDMI dos. ",
                    "language": "es",
                    "warnings": [],
                    "providerMetadata": {"gateway": {"cost": "0.0001"}},
                },
            )

        stt = VercelSpeechToText(
            "test-key", model="openai/gpt-4o-mini-transcribe", http_client=_client(handler)
        )
        result = stt.transcribe_detailed(b"RIFFfake")

        body = captured["body"]
        self.assertEqual(captured["model"], "openai/gpt-4o-mini-transcribe")
        self.assertEqual(base64.b64decode(body["audio"]), b"RIFFfake")
        self.assertEqual(body["mediaType"], "audio/wav")
        self.assertTrue(body["providerOptions"]["gateway"]["disallowPromptTraining"])
        self.assertEqual(body["providerOptions"]["openai"]["language"], "es")
        self.assertEqual(result.text, "Pon el HDMI dos.")
        self.assertEqual(result.cost_usd, 0.0001)

    def test_language_hint_only_goes_to_openai_models(self) -> None:
        captured: dict[str, object] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured.update(json.loads(request.content))
            return httpx.Response(200, json={"text": "hola"})

        stt = VercelSpeechToText(
            "test-key", model="spacexai/grok-stt", http_client=_client(handler)
        )
        self.assertEqual(stt.transcribe(b"RIFF"), "hola")
        self.assertNotIn("openai", captured["providerOptions"])

    def test_context_prompt_goes_to_the_model_provider(self) -> None:
        bodies: list[dict[str, object]] = []

        def handler(request: httpx.Request) -> httpx.Response:
            bodies.append(json.loads(request.content))
            return httpx.Response(200, json={"text": "ok"})

        for model in ("openai/gpt-4o-transcribe", "spacexai/grok-stt"):
            VercelSpeechToText(
                "test-key", model=model, prompt="weón, cachai", http_client=_client(handler)
            ).transcribe(b"RIFF")
        self.assertEqual(
            bodies[0]["providerOptions"]["openai"], {"language": "es", "prompt": "weón, cachai"}
        )
        self.assertEqual(bodies[1]["providerOptions"]["spacexai"], {"prompt": "weón, cachai"})

    def test_prompt_setting_default_uses_the_chilean_context(self) -> None:
        previous = dict(os.environ)
        try:
            os.environ["AI_GATEWAY_API_KEY"] = "test-key"
            os.environ["VADER_STT_PROMPT"] = "default"
            settings = load_settings(dotenv_path=Path("missing.env"))
        finally:
            os.environ.clear()
            os.environ.update(previous)
        self.assertEqual(build_speech_to_text(settings).primary.prompt, CHILEAN_STT_PROMPT)

    def test_http_error_becomes_transcription_error(self) -> None:
        stt = VercelSpeechToText(
            "test-key",
            http_client=_client(lambda _request: httpx.Response(403, text="no access")),
        )
        with self.assertRaises(TranscriptionError):
            stt.transcribe(b"RIFF")

    def test_missing_key_is_rejected(self) -> None:
        with self.assertRaises(TranscriptionError):
            VercelSpeechToText("")

    def test_zdr_is_requested_only_when_required(self) -> None:
        bodies: list[dict[str, object]] = []

        def handler(request: httpx.Request) -> httpx.Response:
            bodies.append(json.loads(request.content))
            return httpx.Response(200, json={"text": "ok"})

        for require_zdr in (False, True):
            VercelSpeechToText(
                "test-key", require_zdr=require_zdr, http_client=_client(handler)
            ).transcribe(b"RIFF")

        self.assertNotIn("zeroDataRetention", bodies[0]["providerOptions"]["gateway"])
        self.assertTrue(bodies[1]["providerOptions"]["gateway"]["zeroDataRetention"])


class FallbackSpeechToTextTests(unittest.TestCase):
    def _stt(self, model: str, status: int, text: str, calls: list[str]) -> VercelSpeechToText:
        def handler(request: httpx.Request) -> httpx.Response:
            calls.append(request.headers["ai-model-id"])
            return httpx.Response(status, json={"text": text})

        return VercelSpeechToText("test-key", model=model, http_client=_client(handler))

    def test_backup_is_not_called_when_primary_works(self) -> None:
        calls: list[str] = []
        stt = FallbackSpeechToText(
            self._stt("spacexai/grok-stt", 200, "hola", calls),
            self._stt("openai/gpt-4o-transcribe", 200, "respaldo", calls),
        )
        self.assertEqual(stt.transcribe(b"RIFF"), "hola")
        self.assertEqual(calls, ["spacexai/grok-stt"])
        self.assertIsNone(stt.last_primary_error)

    def test_backup_answers_when_primary_fails(self) -> None:
        calls: list[str] = []
        stt = FallbackSpeechToText(
            self._stt("spacexai/grok-stt", 503, "", calls),
            self._stt("openai/gpt-4o-transcribe", 200, "respaldo", calls),
        )
        result = stt.transcribe_detailed(b"RIFF")
        self.assertEqual(result.text, "respaldo")
        self.assertEqual(result.model, "openai/gpt-4o-transcribe")
        self.assertIn("503", stt.last_primary_error or "")

    def test_without_backup_the_error_propagates(self) -> None:
        stt = FallbackSpeechToText(self._stt("spacexai/grok-stt", 503, "", []), None)
        with self.assertRaises(TranscriptionError):
            stt.transcribe(b"RIFF")

    def test_settings_defaults_use_gpt4o_with_context_and_grok_backup(self) -> None:
        # Chosen on the Pi (16 Oct 2026): gpt-4o + Chilean context 0 % WER at 1 m.
        with mock.patch.dict("os.environ", {"AI_GATEWAY_API_KEY": "test-key"}, clear=True):
            settings = load_settings(Path("__missing__.env"))
        stt = build_speech_to_text(settings)
        self.assertEqual(stt.primary.model, "openai/gpt-4o-transcribe")
        self.assertEqual(stt.primary.prompt, CHILEAN_STT_PROMPT)
        self.assertNotIn("Spotify", CHILEAN_STT_PROMPT)
        self.assertEqual(stt.fallback.model, "spacexai/grok-stt")
        self.assertTrue(stt.primary.require_zdr)  # AI_GATEWAY_REQUIRE_ZDR defaults to true
        # A hung primary must hand over to the backup quickly, not after 30 s.
        self.assertEqual(stt.primary._client.timeout.read, 8.0)
        stt.close()


class SttEvaluationTests(unittest.TestCase):
    def test_numbered_phrases_keep_the_windows_fifteen_first(self) -> None:
        self.assertEqual([p.number for p in STT_PHRASES], list(range(1, 19)))
        self.assertEqual(STT_PHRASES[0].text, "Pon música de Los Prisioneros.")
        self.assertEqual({p.kind for p in STT_PHRASES[15:]}, {"chileno_natural"})

    def test_normalization_ignores_case_accents_punctuation_and_digits(self) -> None:
        self.assertEqual(
            normalize_transcript("Sube el volumen a 60."),
            normalize_transcript("sube el volumen a sesenta"),
        )
        self.assertEqual(
            normalize_transcript("¿Cómo va a estar el clima?"),
            ["como", "va", "a", "estar", "el", "clima"],
        )
        self.assertEqual(normalize_transcript("Apaga en 31"), ["apaga", "en", "treinta", "y", "uno"])

    def test_chilean_spelling_variants_are_not_errors(self) -> None:
        self.assertEqual(
            word_errors("Oye weón, ¿cachai lo que pasó ayer?", "Oye huevón, cachay lo que pasó ayer"),
            (0, 7),
        )
        self.assertEqual(word_errors("Ya po, ponte algo.", "Ya, pues ponte algo."), (0, 4))
        self.assertEqual(word_errors("Ya po.", "Ya poh."), (0, 2))

    def test_wrong_channel_number_counts_as_error(self) -> None:
        self.assertEqual(word_errors("Cambia al canal trece.", "Cambia al canal 3."), (1, 4))

    def test_summary_counts_failures_and_perfect_phrases(self) -> None:
        rows = [
            SttRow("m", 1, "Pon el HDMI dos.", "pon el hdmi 2", 1.0, 0.001),
            SttRow("m", 2, "Cambia al canal trece.", "cambia al canal tres", 2.0, 0.001),
            SttRow("m", 3, "Hola", None, None, None, "HTTP 500"),
        ]
        summary = summarize_model(rows)
        self.assertEqual(summary["fallidas"], 1)
        self.assertEqual(summary["frases_perfectas"], 1)
        self.assertEqual(summary["error_palabras_pct"], 12.5)
        self.assertEqual(summary["latencia_mediana_s"], 1.5)
        self.assertEqual(summary["costo_total_usd"], 0.002)


if __name__ == "__main__":
    unittest.main()
