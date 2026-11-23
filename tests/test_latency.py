"""Tests for the latency work: persistent Piper, warm connections, speculation."""

from __future__ import annotations

from pathlib import Path
import os
import queue
import sys
import tempfile
import threading
import time
import unittest
from unittest import mock

import httpx

from vadermind.config import load_settings
from vadermind.listening import Utterance, VoiceListener, record_utterance
from vadermind.telemetry import TURN_TIMELINE
from vadermind.tools.information import InformationToolsExecutor
from vadermind.vad import FRAME_SAMPLES, EndOfSpeechDetector
from vadermind.voice_input import TranscriptionResult, VercelSpeechToText
from vadermind.voice_output import PiperTextToSpeech, VoiceError, _PiperWorker, split_sentences
from vadermind.warmup import origin

# Stands in for `piper --output_dir DIR`: one WAV and one printed path per line.
FAKE_PIPER = r"""
import sys, pathlib, time
out = pathlib.Path(sys.argv[sys.argv.index("--output_dir") + 1])
for line in sys.stdin:
    if line.strip() == "morir":
        sys.exit(3)
    path = out / f"{time.perf_counter_ns()}.wav"
    path.write_bytes(b"RIFF" + b"\0" * 64)
    print(path, flush=True)
"""


class PiperWorkerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.worker = _PiperWorker(
            [sys.executable, "-c", FAKE_PIPER], Path(self.folder.name), timeout_seconds=10
        )
        self.addCleanup(self.worker.close)

    def test_one_process_serves_every_reply(self) -> None:
        first = self.worker.synthesize("Hola.")
        process = self.worker._process
        second = self.worker.synthesize("Dos lineas\nse vuelven una.")
        self.assertIs(self.worker._process, process)
        self.assertNotEqual(first, second)
        self.assertTrue(first.is_file() and second.is_file())

    def test_a_dead_process_raises_and_the_next_call_restarts_it(self) -> None:
        with self.assertRaises(VoiceError):
            self.worker.synthesize("morir")
        self.assertTrue(self.worker.synthesize("de nuevo").is_file())


class PiperFallbackTests(unittest.TestCase):
    def test_persistent_failure_falls_back_to_one_process_per_reply(self) -> None:
        previous = dict(os.environ)
        try:
            os.environ["PIPER_EXECUTABLE"] = "no-existe-piper"
            settings = load_settings(dotenv_path=Path("missing.env"))
        finally:
            os.environ.clear()
            os.environ.update(previous)
        tts = PiperTextToSpeech(settings)
        tts.warm()  # cannot start: must not raise, only disable persistence
        self.assertTrue(tts._persistent_failed)
        self.assertFalse(tts._use_persistent())


class SentenceSplitTests(unittest.TestCase):
    def test_splits_on_sentence_ends_but_not_inside_times_or_numbers(self) -> None:
        self.assertEqual(
            split_sentences("Acá andamos weón. ¿Qué querís hacer hoy? ¡Vamos!"),
            ["Acá andamos weón.", "¿Qué querís hacer hoy?", "¡Vamos!"],
        )
        self.assertEqual(split_sentences("Son las 10:32. Hace 3.5 grados."), ["Son las 10:32.", "Hace 3.5 grados."])
        self.assertEqual(split_sentences("Hecho."), ["Hecho."])

    def test_long_sentences_split_at_commas_without_tiny_pieces(self) -> None:
        # Real chileno-mode reply that took 2.85 s to synthesize whole on the Pi.
        self.assertEqual(
            split_sentences(
                "Más aburrido que monje tibetano weón, entre reuniones de pauta y "
                "aguantando al Emperador que anda insoportable con los recortes."
            ),
            [
                "Más aburrido que monje tibetano weón,",
                "entre reuniones de pauta y aguantando al Emperador que anda "
                "insoportable con los recortes.",
            ],
        )
        self.assertEqual(
            split_sentences("Weón, no tengo idea de qué me hablai, pero suena a puro webeo."),
            ["Weón, no tengo idea de qué me hablai,", "pero suena a puro webeo."],
        )
        short = "Ya po, dale nomás."  # short sentences keep their commas
        self.assertEqual(split_sentences(short), [short])
        self.assertEqual(
            split_sentences("Hace 3,5 grados en Punta Arenas y 12,4 grados en Santiago hoy."),
            ["Hace 3,5 grados en Punta Arenas y 12,4 grados en Santiago hoy."],
        )


class SentenceStreamingTests(unittest.TestCase):
    """Vader starts talking when the first sentence is ready, not the whole reply."""

    REPLY = "Primera oracion. Segunda oracion. Tercera oracion."

    def setUp(self) -> None:
        previous = dict(os.environ)
        try:
            os.environ["VADER_SAVE_LAST_AUDIO"] = "false"
            self.tts = PiperTextToSpeech(load_settings(dotenv_path=Path("missing.env")))
        finally:
            os.environ.clear()
            os.environ.update(previous)
        self.synthesized: list[str] = []
        self.played: list[str] = []
        self.first_played = threading.Event()
        self.waited_in_vain = False

    def _synthesize(self, sentence: str, target: Path, *, fail_on: str = "") -> Path:
        if sentence == fail_on:
            raise VoiceError("Piper fallo")
        # Later sentences only finish once the first is playing; checked by the test.
        if self.synthesized and not self.first_played.wait(5):
            self.waited_in_vain = True
        self.synthesized.append(sentence)
        target.write_bytes(b"RIFF" + b"\0" * 64)
        self._paths.append(target)
        return target

    def _play(self, path: Path) -> None:
        self.played.append(path.name)
        self.first_played.set()

    def _speak(self, text: str, fail_on: str = "") -> None:
        self._paths: list[Path] = []
        synth = lambda _tts, sentence, target: self._synthesize(sentence, target, fail_on=fail_on)  # noqa: E731
        with mock.patch.object(PiperTextToSpeech, "_synthesize", synth), mock.patch(
            "vadermind.voice_output._play_wav_file", self._play
        ):
            self.tts.speak(text)

    def test_first_sentence_plays_while_the_rest_is_synthesized(self) -> None:
        self._speak(self.REPLY)
        self.assertFalse(self.waited_in_vain, "la primera oracion no sono antes")
        self.assertEqual(self.synthesized, split_sentences(self.REPLY))
        self.assertEqual(len(self.played), 3)
        self.assertFalse(any(path.exists() for path in self._paths))

    def test_a_failure_mid_reply_keeps_what_played_and_leaves_no_files(self) -> None:
        with self.assertRaises(VoiceError):
            self._speak(self.REPLY, fail_on="Segunda oracion.")
        self.assertEqual(len(self.played), 1)
        self.assertFalse(any(path.exists() for path in self._paths))

    def test_without_the_persistent_worker_the_reply_is_one_piece(self) -> None:
        self.tts._persistent_failed = True  # each sentence would reload the model
        self.first_played.set()
        self._speak(self.REPLY)
        self.assertEqual(self.synthesized, [self.REPLY])


class _ScriptedVad:
    """Return a scripted probability per frame."""

    def __init__(self, probabilities: list[float]) -> None:
        self.probabilities = list(probabilities)

    def reset(self) -> None:
        pass

    def speech_probability(self, frame: bytes) -> float:
        return self.probabilities.pop(0)


def _record(probabilities: list[float], pause_ms: int = 320):
    source: "queue.Queue[bytes]" = queue.Queue()
    for _ in probabilities:
        source.put(b"\x01\x00" * FRAME_SAMPLES)
    snapshots: list[int] = []
    starts: list[str] = []
    utterance = record_utterance(
        _ScriptedVad(probabilities),
        EndOfSpeechDetector(silence_ms=800),
        frames=lambda: source,
        on_speech_start=lambda: starts.append("start"),
        on_pause=lambda pcm: snapshots.append(len(pcm)),
        pause_ms=pause_ms,
    )
    return utterance, snapshots, starts


class SpeculativeRecordingTests(unittest.TestCase):
    def test_clean_pause_speculates_once_and_covers_all_speech(self) -> None:
        utterance, snapshots, starts = _record([0.9] * 20 + [0.0] * 30)
        self.assertEqual(utterance.reason, "fin_de_habla")
        self.assertEqual(len(snapshots), 1)
        self.assertTrue(utterance.speculation_complete)
        self.assertEqual(starts, ["start"])

    def test_resumed_speech_discards_the_first_guess_and_speculates_again(self) -> None:
        utterance, snapshots, _ = _record([0.9] * 20 + [0.0] * 12 + [0.9] * 10 + [0.0] * 30)
        self.assertEqual(len(snapshots), 2)
        self.assertLess(snapshots[0], snapshots[1])
        self.assertTrue(utterance.speculation_complete)

    def test_uncertain_frame_after_the_snapshot_voids_it(self) -> None:
        # 0.4 is neither speech nor clear silence: it might be a soft word.
        utterance, snapshots, _ = _record([0.9] * 20 + [0.0] * 12 + [0.4] + [0.0] * 30)
        self.assertEqual(utterance.reason, "fin_de_habla")
        self.assertEqual(len(snapshots), 2)  # it guessed again after the doubt
        self.assertTrue(utterance.speculation_complete)

    def test_no_speculation_when_disabled(self) -> None:
        utterance, snapshots, _ = _record([0.9] * 20 + [0.0] * 30, pause_ms=0)
        self.assertEqual(snapshots, [])
        self.assertFalse(utterance.speculation_complete)


class _CountingTranscriber:
    def __init__(self) -> None:
        self.calls: list[bytes] = []
        self.last_primary_error = None

    def transcribe_detailed(self, audio: bytes) -> TranscriptionResult:
        self.calls.append(audio)
        return TranscriptionResult(f"texto {len(self.calls)}", "spacexai/grok-stt", 0.1)

    def close(self) -> None:
        pass


class _SilentCues:
    def start(self) -> None:
        pass

    def end(self, wait: bool = True) -> None:
        pass


def _listener(complete: bool) -> tuple[VoiceListener, _CountingTranscriber]:
    transcriber = _CountingTranscriber()

    def record(_vad, _detector, *, on_pause=None, **_hooks) -> Utterance:
        on_pause(b"\x01\x00" * FRAME_SAMPLES)
        return Utterance(
            b"RIFF-final",
            "fin_de_habla",
            1.0,
            speech_ended_at=time.perf_counter(),
            speculation_complete=complete,
        )

    listener = VoiceListener(
        _ScriptedVad([]), transcriber, _SilentCues(), speculative_ms=320, record=record
    )
    return listener, transcriber


class SpeculativeListenerTests(unittest.TestCase):
    def test_complete_speculation_is_used_without_a_second_request(self) -> None:
        listener, transcriber = _listener(complete=True)
        turn = listener.listen()
        listener.close()
        self.assertEqual(turn.text, "texto 1")
        self.assertEqual(len(transcriber.calls), 1)
        self.assertEqual(TURN_TIMELINE.notes["especulaciones"], "1")

    def test_stale_speculation_is_ignored_and_the_full_audio_is_transcribed(self) -> None:
        listener, transcriber = _listener(complete=False)
        turn = listener.listen()
        listener.close()
        self.assertEqual(len(transcriber.calls), 2)
        self.assertEqual(transcriber.calls[-1], b"RIFF-final")
        self.assertEqual(turn.text, "texto 2")

    def test_speculation_needs_a_pause_shorter_than_the_end_of_speech_wait(self) -> None:
        listener = VoiceListener(
            _ScriptedVad([]), _CountingTranscriber(), _SilentCues(),
            silence_ms=800, speculative_ms=800,
        )
        self.assertEqual(listener.speculative_ms, 0)
        listener.close()


class WarmConnectionTests(unittest.TestCase):
    def test_stt_warm_up_opens_the_gateway_origin(self) -> None:
        seen: list[tuple[str, str]] = []

        def handler(request: httpx.Request) -> httpx.Response:
            seen.append((request.method, str(request.url)))
            return httpx.Response(404)

        stt = VercelSpeechToText("key", http_client=httpx.Client(transport=httpx.MockTransport(handler)))
        stt.warm()
        self.assertEqual(seen, [("HEAD", "https://ai-gateway.vercel.sh/")])

    def test_warm_up_failures_are_silent(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("sin red")

        stt = VercelSpeechToText("key", http_client=httpx.Client(transport=httpx.MockTransport(handler)))
        stt.warm()  # must not raise

    def test_origin_keeps_only_scheme_and_host(self) -> None:
        self.assertEqual(origin("https://api.open-meteo.com/v1/forecast?x=1"), "https://api.open-meteo.com/")


class _GeoClient:
    def __init__(self) -> None:
        self.urls: list[str] = []

    def get(self, url: str, params: dict[str, object]):
        self.urls.append(url)
        if "geocoding" in url:
            data = {"results": [{"latitude": -33.45, "longitude": -70.67}]}
        else:
            data = {
                "daily": {
                    "weather_code": [1, 61],
                    "temperature_2m_max": [22.0, 19.0],
                    "temperature_2m_min": [9.0, 8.0],
                    "precipitation_probability_max": [10, 70],
                }
            }

        class _Response:
            def raise_for_status(self) -> None:
                pass

            def json(self):
                return data

        return _Response()


class GeocodingCacheTests(unittest.TestCase):
    def test_city_coordinates_are_looked_up_once_and_survive_restarts(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            cache = Path(folder) / "geo.json"
            client = _GeoClient()
            executor = InformationToolsExecutor(client=client, geocoding_cache_path=cache)  # type: ignore[arg-type]
            args = {"location": "Santiago de Chile", "when": "tomorrow"}
            executor.execute("information.weather", args)
            executor.execute("information.weather", {**args, "location": " santiago  de CHILE"})
            self.assertEqual(sum("geocoding" in url for url in client.urls), 1)
            self.assertTrue(cache.is_file())

            restarted = _GeoClient()
            again = InformationToolsExecutor(client=restarted, geocoding_cache_path=cache)  # type: ignore[arg-type]
            result = again.execute("information.weather", args)
            self.assertTrue(result["ok"])
            self.assertFalse(any("geocoding" in url for url in restarted.urls))


if __name__ == "__main__":
    unittest.main()
