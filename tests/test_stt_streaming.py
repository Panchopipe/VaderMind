"""Tests for streaming transcription: the WebSocket client and the listener's use of it."""

from __future__ import annotations

import asyncio
from concurrent.futures import Future
import json
import os
from pathlib import Path
import queue
import threading
import time
import unittest
from types import SimpleNamespace

from aiohttp import WSMsgType, web

from vadermind.config import load_settings
from vadermind.listening import Utterance, VoiceListener, record_utterance
from vadermind.stt_streaming import (
    CHILEAN_KEYTERMS,
    StreamCancelled,
    StreamingSpeechToText,
    build_streaming_speech_to_text,
)
from vadermind.telemetry import TURN_TIMELINE
from vadermind.vad import FRAME_SAMPLES, EndOfSpeechDetector
from vadermind.voice_input import TranscriptionError, TranscriptionResult

KEY = "clave-de-prueba"
FRAME = b"\x01\x00" * FRAME_SAMPLES


class FakeGateway:
    """A local stand-in for the AI Gateway WebSocket, run on its own thread.

    `mode`: "ok" answers `finish` after "audio done"; "error" sends an error
    part right after the start frame; "silent" never answers; "reject"
    refuses the handshake with HTTP 401.
    """

    def __init__(self) -> None:
        self.mode = "ok"
        self.protocols: list[str] = []
        self.start: dict | None = None
        self.audio = bytearray()
        self.audio_done = False
        self.closed = threading.Event()
        self._loop = asyncio.new_event_loop()
        self._ready = threading.Event()
        self._thread = threading.Thread(target=self._serve, daemon=True)
        self._thread.start()
        self._ready.wait(5)

    def _serve(self) -> None:
        asyncio.set_event_loop(self._loop)
        app = web.Application()
        app.router.add_get("/v4/ai/transcription-model", self._handle)
        self._runner = web.AppRunner(app)
        self._loop.run_until_complete(self._runner.setup())
        site = web.TCPSite(self._runner, "127.0.0.1", 0)
        self._loop.run_until_complete(site.start())
        self.port = site._server.sockets[0].getsockname()[1]  # type: ignore[union-attr]
        self._ready.set()
        self._loop.run_forever()

    @property
    def url(self) -> str:
        return f"ws://127.0.0.1:{self.port}/v4/ai/transcription-model"

    async def _handle(self, request: web.Request) -> web.StreamResponse:
        self.protocols = [
            item.strip() for item in request.headers.get("Sec-WebSocket-Protocol", "").split(",")
        ]
        self.model = request.query.get("ai-model-id")
        if self.mode == "reject":
            return web.Response(status=401, text="sin permiso")
        ws = web.WebSocketResponse(protocols=("ai-gateway-transcription.v1",))
        await ws.prepare(request)
        try:
            async for message in ws:
                if message.type == WSMsgType.BINARY:
                    self.audio.extend(message.data)
                    continue
                if message.type != WSMsgType.TEXT:
                    continue
                frame = json.loads(message.data)
                if frame["type"] == "transcription-stream.start":
                    self.start = frame
                    await ws.send_str(json.dumps({"type": "stream-start", "warnings": []}))
                    if self.mode == "error":
                        await ws.send_str(
                            json.dumps({"type": "error", "error": {"name": "X", "message": "fallo upstream"}})
                        )
                        await ws.close(code=1011)
                elif frame["type"] == "transcription-stream.audio-done":
                    self.audio_done = True
                    if self.mode == "ok":
                        await ws.send_str(json.dumps({"type": "transcript-final", "text": "Hola Vader."}))
                        await ws.send_str(
                            json.dumps(
                                {"type": "finish", "text": " Hola Vader. ", "segments": [], "language": "es"}
                            )
                        )
                        await ws.close()
        finally:
            self.closed.set()
        return ws

    def stop(self) -> None:
        asyncio.run_coroutine_threadsafe(self._runner.cleanup(), self._loop).result(5)
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join(5)


class StreamingClientTests(unittest.TestCase):
    def setUp(self) -> None:
        self.gateway = FakeGateway()
        self.stt = StreamingSpeechToText(
            KEY, url=self.gateway.url, connect_timeout=2.0, finish_timeout=0.5
        )

    def tearDown(self) -> None:
        self.stt.close()
        self.gateway.stop()

    def test_audio_streams_and_the_final_text_arrives_after_audio_done(self) -> None:
        stream = self.stt.open()
        for _ in range(3):  # fed before the socket is open: it waits in the queue
            stream.feed(FRAME)
        result = stream.finish().result(timeout=3)
        self.assertEqual(result.text, "Hola Vader.")
        self.assertEqual(result.language, "es")
        self.assertEqual(result.model, "spacexai/grok-stt")
        self.assertEqual(bytes(self.gateway.audio), FRAME * 3)
        self.assertEqual(self.gateway.model, "spacexai/grok-stt")
        self.assertIn("ai-gateway-transcription.v1", self.gateway.protocols)
        self.assertIn(f"ai-gateway-auth.{KEY}", self.gateway.protocols)
        self.assertIsNotNone(stream.connect_seconds)
        self.assertIsNotNone(stream.finish_seconds)

    def test_start_frame_sends_spanish_keyterms_under_xai_and_no_training(self) -> None:
        self.stt.open().finish().result(timeout=3)
        start = self.gateway.start
        self.assertEqual(start["inputAudioFormat"], {"type": "audio/pcm", "rate": 16000})
        options = start["providerOptions"]
        self.assertEqual(options["xai"], {"language": "es", "keyterm": list(CHILEAN_KEYTERMS)})
        self.assertNotIn("spacexai", options)
        self.assertEqual(options["gateway"], {"disallowPromptTraining": True})

    def test_other_models_get_no_grok_options(self) -> None:
        other = StreamingSpeechToText(KEY, model="openai/gpt-realtime-whisper", url=self.gateway.url)
        try:
            self.assertEqual(set(other.start_frame()["providerOptions"]), {"gateway"})
        finally:
            other.close()

    def test_a_server_error_fails_the_stream_before_the_pause(self) -> None:
        self.gateway.mode = "error"
        stream = self.stt.open()
        stream.feed(FRAME)
        deadline = time.monotonic() + 3
        while not stream.failed and time.monotonic() < deadline:
            time.sleep(0.02)
        self.assertTrue(stream.failed)
        with self.assertRaisesRegex(TranscriptionError, "fallo upstream"):
            stream.finish().result(timeout=1)

    def test_a_silent_server_times_out(self) -> None:
        self.gateway.mode = "silent"
        stream = self.stt.open()
        stream.feed(FRAME)
        with self.assertRaisesRegex(TranscriptionError, "no respondio"):
            stream.finish().result(timeout=3)

    def test_a_rejected_key_is_reported_without_the_key(self) -> None:
        self.gateway.mode = "reject"
        with self.assertRaises(TranscriptionError) as caught:
            self.stt.open().finish().result(timeout=3)
        self.assertIn("401", str(caught.exception))
        self.assertNotIn(KEY, str(caught.exception))

    def test_cancel_closes_without_asking_for_text(self) -> None:
        stream = self.stt.open()
        stream.feed(FRAME)
        time.sleep(0.2)
        stream.cancel()
        self.assertTrue(self.gateway.closed.wait(3))
        self.assertFalse(self.gateway.audio_done)
        with self.assertRaises(StreamCancelled):
            stream.result.result(timeout=3)

    def test_an_unreachable_gateway_fails_fast(self) -> None:
        closed = StreamingSpeechToText(KEY, url="ws://127.0.0.1:9/v4/ai/transcription-model")
        try:
            with self.assertRaises(TranscriptionError):
                closed.open().finish().result(timeout=5)
        finally:
            closed.close()


def _settings(**changes: object) -> SimpleNamespace:
    values = dict(
        stt_streaming=True,
        stt_streaming_model="spacexai/grok-stt",
        stt_speculative_ms=192,
        stt_prompt="default",
        ai_gateway_api_key=KEY,
        ai_gateway_require_zdr=False,
        language="es",
    )
    values.update(changes)
    return SimpleNamespace(**values)


class BuildStreamingTests(unittest.TestCase):
    def test_on_by_default_and_off_with_false(self) -> None:
        previous = dict(os.environ)
        try:
            os.environ.pop("VADER_STT_STREAMING", None)
            enabled = load_settings(dotenv_path=Path("missing.env"))
            os.environ["VADER_STT_STREAMING"] = "false"
            disabled = load_settings(dotenv_path=Path("missing.env"))
        finally:
            os.environ.clear()
            os.environ.update(previous)
        self.assertTrue(enabled.stt_streaming)
        self.assertEqual(enabled.stt_streaming_model, "spacexai/grok-stt")
        self.assertFalse(disabled.stt_streaming)
        self.assertIsNone(build_streaming_speech_to_text(disabled))

    def test_needs_the_speculative_pause(self) -> None:
        with self.assertLogs("vadermind.stt_streaming", "WARNING"):
            self.assertIsNone(build_streaming_speech_to_text(_settings(stt_speculative_ms=0)))

    def test_custom_prompt_sends_no_keyterms(self) -> None:
        stt = build_streaming_speech_to_text(_settings(stt_prompt=""))
        try:
            self.assertEqual(stt.keyterms, ())
        finally:
            stt.close()

    def test_missing_key_keeps_rest(self) -> None:
        with self.assertLogs("vadermind.stt_streaming", "WARNING"):
            self.assertIsNone(build_streaming_speech_to_text(_settings(ai_gateway_api_key="")))


class _FakeStream:
    def __init__(self, answer: TranscriptionResult | None, fail: bool = False) -> None:
        self.result: Future = Future()
        self.fed = bytearray()
        self.finished = False
        self.cancelled = False
        self.connect_seconds = 0.4
        self.finish_seconds = 0.3
        self._answer = answer
        if fail:
            self.result.set_exception(TranscriptionError("sin conexion"))

    @property
    def failed(self) -> bool:
        return self.result.done() and self.result.exception() is not None

    def feed(self, pcm: bytes) -> None:
        self.fed.extend(pcm)

    def finish(self) -> Future:
        self.finished = True
        if self._answer is not None and not self.result.done():
            self.result.set_result(self._answer)
        return self.result

    def cancel(self) -> None:
        self.cancelled = True


class _FakeStreamer:
    def __init__(self, stream: _FakeStream) -> None:
        self.stream = stream
        self.model = "spacexai/grok-stt"

    def open(self) -> _FakeStream:
        return self.stream

    def close(self) -> None:
        pass


class _RestTranscriber:
    def __init__(self) -> None:
        self.calls: list[bytes] = []
        self.last_primary_error = None

    def transcribe_detailed(self, audio: bytes) -> TranscriptionResult:
        self.calls.append(audio)
        return TranscriptionResult("texto rest", "openai/gpt-4o-transcribe", 0.9)

    def close(self) -> None:
        pass


class _SilentCues:
    def start(self) -> None:
        pass

    def end(self, wait: bool = True) -> None:
        pass


def _listen(stream: _FakeStream, pauses: int = 1, reason: str = "fin_de_habla"):
    rest = _RestTranscriber()

    def record(_vad, _detector, *, on_pause=None, on_audio=None, **_hooks) -> Utterance:
        captured = b""
        for _ in range(pauses):  # speech, a pause, maybe more speech and another pause
            on_audio(FRAME)
            captured += FRAME
            if reason != "sin_voz":
                on_pause(captured)
        return Utterance(
            b"RIFF-final",
            reason,
            1.0,
            speech_ended_at=time.perf_counter(),
            speculation_complete=reason == "fin_de_habla",
        )

    listener = VoiceListener(
        SimpleNamespace(), rest, _SilentCues(), speculative_ms=192,
        record=record, streamer=_FakeStreamer(stream),
    )
    turn = listener.listen()
    listener.close()
    return turn, rest


STREAMED = TranscriptionResult("Hola Vader.", "spacexai/grok-stt", 0.3, language="es")


class StreamingListenerTests(unittest.TestCase):
    def test_the_stream_answers_at_the_pause_without_rest(self) -> None:
        stream = _FakeStream(STREAMED)
        turn, rest = _listen(stream)
        self.assertEqual(turn.text, "Hola Vader.")
        self.assertEqual(rest.calls, [])
        self.assertEqual(bytes(stream.fed), FRAME)
        self.assertEqual(TURN_TIMELINE.notes["transcriptor"], "streaming")
        self.assertIn("stt_conexion=0.40", TURN_TIMELINE.report())
        self.assertIn("stt_streaming=0.30", TURN_TIMELINE.report())

    def test_a_failed_stream_leaves_the_pause_to_rest(self) -> None:
        with self.assertLogs("vadermind.listening", "WARNING"):
            turn, rest = _listen(_FakeStream(None, fail=True))
        self.assertEqual(turn.text, "texto rest")
        self.assertEqual(len(rest.calls), 1)
        self.assertEqual(TURN_TIMELINE.notes["transcriptor"], "rest")

    def test_a_slow_stream_races_a_rest_request(self) -> None:
        stream = _FakeStream(None)  # never answers
        turn, rest = _listen(stream)
        self.assertEqual(turn.text, "texto rest")
        self.assertEqual(rest.calls, [b"RIFF-final"])

    def test_speech_after_the_pause_goes_to_rest(self) -> None:
        stream = _FakeStream(STREAMED)
        turn, rest = _listen(stream, pauses=2)
        self.assertTrue(stream.finished)
        self.assertEqual(turn.text, "texto rest")  # the stream only heard the first part
        self.assertEqual(len(rest.calls), 1)
        self.assertEqual(TURN_TIMELINE.notes["especulaciones"], "2")

    def test_no_speech_cancels_the_stream(self) -> None:
        stream = _FakeStream(STREAMED)
        _listen(stream, reason="sin_voz")
        self.assertTrue(stream.cancelled)
        self.assertFalse(stream.finished)

    def test_streaming_needs_the_speculative_pause(self) -> None:
        listener = VoiceListener(
            SimpleNamespace(), _RestTranscriber(), _SilentCues(), speculative_ms=0,
            streamer=_FakeStreamer(_FakeStream(STREAMED)),
        )
        self.assertIsNone(listener.streamer)
        listener.close()


class _ScriptedVad:
    def __init__(self, probabilities: list[float]) -> None:
        self.probabilities = list(probabilities)

    def reset(self) -> None:
        pass

    def speech_probability(self, _frame: bytes) -> float:
        return self.probabilities.pop(0)


class RecordingAudioHookTests(unittest.TestCase):
    def test_audio_hook_holds_exactly_what_the_pause_snapshot_sends(self) -> None:
        probabilities = [0.0] * 5 + [0.9] * 20 + [0.0] * 30
        source: "queue.Queue[bytes]" = queue.Queue()
        for index, _ in enumerate(probabilities):
            source.put(index.to_bytes(2, "little") * FRAME_SAMPLES)
        fed: list[bytes] = []
        snapshots: list[bytes] = []
        record_utterance(
            _ScriptedVad(probabilities),
            EndOfSpeechDetector(silence_ms=800),
            frames=lambda: source,
            on_pause=lambda pcm: snapshots.append(b"".join(fed) == pcm and pcm),
            pause_ms=192,
            on_audio=fed.append,
        )
        self.assertEqual(len(snapshots), 1)
        self.assertTrue(snapshots[0], "the streamed audio differs from the snapshot")
        self.assertTrue(snapshots[0].startswith(FRAME_SAMPLES * (0).to_bytes(2, "little")))


if __name__ == "__main__":
    unittest.main()
