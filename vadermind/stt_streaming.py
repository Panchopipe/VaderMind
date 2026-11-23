"""Streaming transcription through Vercel AI Gateway's WebSocket (beta).

The REST transcriber uploads the recording when the user pauses and waits
~0.9 s for the text. Here the audio travels while the user speaks; at the
pause Vader only says "audio done" and the text arrives ~0.3 s later.

Protocol, read from the open-source AI SDK (`@ai-sdk/gateway` 4.0.150 and
`@ai-sdk/provider-utils`); it is a beta and Vercel does not document it:

1. WebSocket to `.../v4/ai/transcription-model?ai-model-id=<model>` with the
   subprotocols `ai-gateway-transcription.v1` and `ai-gateway-auth.<key>`.
2. One JSON text frame `transcription-stream.start`: audio format and options.
3. Raw PCM in binary frames of at most 64 KiB.
4. One JSON text frame `transcription-stream.audio-done`; closing without it
   aborts the stream.
5. The server answers with JSON parts and ends with `finish` (whole text and
   language) or `error`, then closes.

One background thread runs an asyncio loop for all streams. The listener calls
`open()` when it starts listening (the handshake takes 0.4-0.9 s and is hidden
while the user starts talking), `feed()` with each captured frame, and
`finish()` at the pause, which returns a Future like the REST speculation.
"""

from __future__ import annotations

import asyncio
from concurrent.futures import Future
import json
import logging
import threading
import time
from typing import TYPE_CHECKING
from urllib.parse import quote

from vadermind.vad import SAMPLE_RATE
from vadermind.voice_input import TranscriptionError, TranscriptionResult

if TYPE_CHECKING:
    from vadermind.config import Settings

LOGGER = logging.getLogger(__name__)

STREAM_URL = "wss://ai-gateway.vercel.sh/v4/ai/transcription-model"
SUBPROTOCOL = "ai-gateway-transcription.v1"
AUTH_SUBPROTOCOL_PREFIX = "ai-gateway-auth."
MAX_FRAME_BYTES = 65536
DEFAULT_STREAMING_MODEL = "spacexai/grok-stt"

# The words of the Chilean REST prompt, as grok-stt key terms. On the user's
# 126 recordings they took grok from 16.9 % to 5.8 % word error with
# `language: "es"`; generic verb forms ("estai", "tenis") did not help.
CHILEAN_KEYTERMS = (
    "Vader", "Darth Vader", "weón", "cachai", "po", "ya po", "al tiro", "pucha",
    "vai", "hai", "querís", "sabís", "la pega", "fome", "bacán",
)

# grok-stt reads its options under "xai"; under "spacexai" they are ignored
# without any warning.
_OPTION_KEYS = {"spacexai": "xai", "xai": "xai"}

_FINISH = object()  # queue markers: send "audio done" / abort the stream
_CANCEL = object()


class StreamCancelled(TranscriptionError):
    """The stream was aborted on purpose (no pause asked for its text)."""


class TranscriptionStream:
    """One utterance: `feed()` audio while the user speaks, then `finish()`."""

    def __init__(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop
        self._queue: asyncio.Queue[object] = asyncio.Queue()
        self.result: Future[TranscriptionResult] = Future()
        self.finished = False  # finish() was called
        self.connect_seconds: float | None = None  # WebSocket handshake
        self.finish_seconds: float | None = None  # "audio done" -> text

    @property
    def failed(self) -> bool:
        return self.result.done() and self.result.exception() is not None

    def feed(self, pcm: bytes) -> None:
        """Queue 16 kHz mono int16 PCM; safe from any thread, never blocks."""

        if not self.finished and not self.result.done():
            self._put(pcm)

    def finish(self) -> "Future[TranscriptionResult]":
        """Close the audio; the Future gets the text or a TranscriptionError."""

        if not self.finished:
            self.finished = True
            self._put(_FINISH)
        return self.result

    def cancel(self) -> None:
        """Abort without asking for text (no speech, or not needed)."""

        if not self.result.done():
            self._put(_CANCEL)

    def _put(self, item: object) -> None:
        try:
            self._loop.call_soon_threadsafe(self._queue.put_nowait, item)
        except RuntimeError:  # the loop is closed: the stream is over anyway
            pass

    def _settle(self, value: TranscriptionResult | None, error: BaseException | None) -> None:
        if self.result.done():
            return
        if error is not None:
            self.result.set_exception(error)
        else:
            self.result.set_result(value)  # type: ignore[arg-type]


class StreamingSpeechToText:
    """Open streaming transcriptions to one model through Vercel AI Gateway."""

    def __init__(
        self,
        api_key: str,
        *,
        model: str = DEFAULT_STREAMING_MODEL,
        language: str = "es",
        keyterms: tuple[str, ...] = CHILEAN_KEYTERMS,
        require_zdr: bool = False,
        connect_timeout: float = 5.0,
        finish_timeout: float = 3.0,
        url: str = STREAM_URL,
    ) -> None:
        if not api_key:
            raise TranscriptionError("Falta AI_GATEWAY_API_KEY para transcribir.")
        import aiohttp  # installed with aiowebostv; imported here to fail early

        self._aiohttp = aiohttp
        self.api_key = api_key
        self.model = model
        self.language = language
        self.keyterms = tuple(keyterms)
        self.require_zdr = require_zdr
        self.connect_timeout = connect_timeout
        self.finish_timeout = finish_timeout
        self.url = f"{url}?ai-model-id={quote(model, safe='/')}"
        self._session: object | None = None
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(
            target=self._loop.run_forever, name="stt-streaming", daemon=True
        )
        self._thread.start()

    def start_frame(self) -> dict[str, object]:
        gateway: dict[str, object] = {"disallowPromptTraining": True}
        if self.require_zdr:
            gateway["zeroDataRetention"] = True
        options: dict[str, object] = {"gateway": gateway}
        key = _OPTION_KEYS.get(self.model.split("/", 1)[0])
        if key:
            provider: dict[str, object] = {}
            if self.language:
                provider["language"] = self.language
            if self.keyterms:
                provider["keyterm"] = list(self.keyterms)
            options[key] = provider
        return {
            "type": "transcription-stream.start",
            "inputAudioFormat": {"type": "audio/pcm", "rate": SAMPLE_RATE},
            "providerOptions": options,
        }

    def open(self) -> TranscriptionStream:
        """Start connecting now; audio fed before the socket is ready waits in a queue."""

        stream = TranscriptionStream(self._loop)
        asyncio.run_coroutine_threadsafe(self._run(stream), self._loop)
        return stream

    def close(self) -> None:
        async def shutdown() -> None:
            for task in asyncio.all_tasks():
                if task is not asyncio.current_task():
                    task.cancel()  # streams still open when Vader exits
            if self._session is not None:
                await self._session.close()  # type: ignore[attr-defined]

        if self._loop.is_closed():
            return
        try:
            asyncio.run_coroutine_threadsafe(shutdown(), self._loop).result(timeout=2.0)
        except Exception:  # closing must never block Vader from exiting
            pass
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join(timeout=2.0)
        if not self._thread.is_alive():
            self._loop.close()

    async def _run(self, stream: TranscriptionStream) -> None:
        try:
            result = await self._transcribe(stream)
        except TranscriptionError as error:
            stream._settle(None, error)
        except Exception as error:  # any network surprise must reach the turn as an error
            stream._settle(None, TranscriptionError(f"Transcripcion en streaming: {error!r}"))
        else:
            if result is None:
                stream._settle(None, StreamCancelled("Transcripcion en streaming cancelada."))
            else:
                stream._settle(result, None)

    async def _transcribe(self, stream: TranscriptionStream) -> TranscriptionResult | None:
        aiohttp = self._aiohttp
        if self._session is None:
            self._session = aiohttp.ClientSession()
        started = time.perf_counter()
        try:
            ws = await asyncio.wait_for(
                self._session.ws_connect(  # type: ignore[attr-defined]
                    self.url,
                    protocols=(SUBPROTOCOL, f"{AUTH_SUBPROTOCOL_PREFIX}{self.api_key}"),
                    autoclose=True,
                ),
                self.connect_timeout,
            )
        except asyncio.TimeoutError as error:
            raise TranscriptionError(
                f"La transcripcion en streaming no conecto en {self.connect_timeout:g} s."
            ) from error
        except aiohttp.WSServerHandshakeError as error:
            # The message never includes the protocols header, so the key stays out of logs.
            raise TranscriptionError(
                f"La transcripcion en streaming con {self.model} rechazo la conexion "
                f"(HTTP {error.status})."
            ) from error
        except (aiohttp.ClientError, OSError) as error:
            raise TranscriptionError(f"No se pudo abrir la transcripcion en streaming: {error}") from error
        stream.connect_seconds = time.perf_counter() - started

        async with ws:
            await ws.send_str(json.dumps(self.start_frame()))
            receiver = asyncio.ensure_future(self._receive(ws, stream))
            try:
                if not await self._send_audio(ws, stream._queue):
                    if receiver.done():
                        await receiver  # re-raises the server's error
                    return None
                done_at = time.perf_counter()
                try:
                    part = await asyncio.wait_for(receiver, self.finish_timeout)
                except asyncio.TimeoutError as error:
                    raise TranscriptionError(
                        f"La transcripcion en streaming no respondio en {self.finish_timeout:g} s."
                    ) from error
            finally:
                receiver.cancel()  # no-op once it finished
                if receiver.done() and not receiver.cancelled():
                    receiver.exception()  # already reported; keeps asyncio from warning
            stream.finish_seconds = time.perf_counter() - done_at
            metadata = part.get("providerMetadata") or {}
            cost = (metadata.get("gateway") or {}).get("cost") if isinstance(metadata, dict) else None
            language = part.get("language")
            result = TranscriptionResult(
                text=str(part.get("text") or "").strip(),
                model=self.model,
                seconds=stream.finish_seconds,
                cost_usd=float(cost) if cost is not None else None,
                language=str(language) if language else None,
            )
            # Hand the text over now: closing the socket costs another round trip.
            stream._settle(result, None)
        return result

    async def _send_audio(self, ws: object, queue: "asyncio.Queue[object]") -> bool:
        """Send queued audio; True once "audio done" went out, False if aborted."""

        while True:
            item = await queue.get()
            if item is _CANCEL:
                return False
            if item is _FINISH:
                await ws.send_str(json.dumps({"type": "transcription-stream.audio-done"}))  # type: ignore[attr-defined]
                return True
            data = bytes(item)  # type: ignore[arg-type]
            for offset in range(0, len(data), MAX_FRAME_BYTES):
                await ws.send_bytes(data[offset : offset + MAX_FRAME_BYTES])  # type: ignore[attr-defined]

    async def _receive(self, ws: object, stream: TranscriptionStream) -> dict[str, object]:
        """Read server parts until `finish`; an error also stops the sender."""

        aiohttp = self._aiohttp
        try:
            async for message in ws:  # type: ignore[attr-defined]
                if message.type != aiohttp.WSMsgType.TEXT:
                    continue
                try:
                    part = json.loads(message.data)
                except ValueError:
                    continue
                if not isinstance(part, dict):
                    continue
                kind = part.get("type")
                if kind == "finish":
                    return part
                if kind == "error":
                    detail = part.get("error")
                    if isinstance(detail, dict):
                        detail = detail.get("message") or detail
                    raise TranscriptionError(
                        f"La transcripcion en streaming con {self.model} fallo: {detail}"
                    )
                if kind == "stream-start" and part.get("warnings"):
                    LOGGER.warning("Avisos de la transcripcion en streaming: %s", part["warnings"])
            raise TranscriptionError(
                "La transcripcion en streaming se cerro sin texto "
                f"(codigo {getattr(ws, 'close_code', None)})."
            )
        except TranscriptionError as error:
            # Fail the turn's Future now (so the listener stops using this stream)
            # and wake the sender so it stops sending audio.
            stream._settle(None, error)
            stream._queue.put_nowait(_CANCEL)
            raise


def build_streaming_speech_to_text(settings: "Settings") -> StreamingSpeechToText | None:
    """The streaming transcriber when `VADER_STT_STREAMING` is on, else None.

    Any problem building it is logged and Vader keeps the REST transcriber.
    """

    if not settings.stt_streaming:
        return None
    if settings.stt_speculative_ms <= 0:
        LOGGER.warning(
            "VADER_STT_STREAMING necesita VADER_STT_SPECULATIVE_MS mayor que 0; se usa REST."
        )
        return None
    try:
        return StreamingSpeechToText(
            settings.ai_gateway_api_key or "",
            model=settings.stt_streaming_model,
            language=settings.language,
            keyterms=CHILEAN_KEYTERMS if settings.stt_prompt == "default" else (),
            require_zdr=settings.ai_gateway_require_zdr,
        )
    except (TranscriptionError, ImportError) as error:
        LOGGER.warning("Sin transcripcion en streaming (%s); se usa REST.", error)
        return None
