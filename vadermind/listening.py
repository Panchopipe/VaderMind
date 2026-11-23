"""Record one spoken turn from the microphone, framed by optional cue sounds."""

from __future__ import annotations

from collections import deque
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
import io
import logging
import math
from pathlib import Path
import queue
import re
import struct
import tempfile
import threading
import time
from typing import TYPE_CHECKING, Callable
import unicodedata
import wave

from vadermind.telemetry import TURN_TIMELINE, timed_operation
from vadermind.vad import FRAME_MS, FRAME_SAMPLES, SAMPLE_RATE, EndOfSpeechDetector, SileroVad
from vadermind.voice_output import VoiceError, _play_wav_file

if TYPE_CHECKING:
    from vadermind.config import Settings
    from vadermind.stt_streaming import StreamingSpeechToText, TranscriptionStream
    from vadermind.voice_input import FallbackSpeechToText, TranscriptionResult

LOGGER = logging.getLogger(__name__)

PRE_ROLL_MS = 320  # audio kept from just before speech was detected

# Phrases speech models invent from silence or noise; never send them to Vader.
_HALLUCINATIONS = {
    "gracias por ver",
    "gracias por ver el video",
    "subtitulos realizados por la comunidad de amara org",
    "suscribete",
}


class ListeningError(RuntimeError):
    """Raised when the microphone cannot record a turn."""


@dataclass(slots=True, frozen=True)
class Utterance:
    wav: bytes
    reason: str  # "fin_de_habla", "maximo" or "sin_voz"
    seconds: float
    speech_ended_at: float | None = None  # perf_counter when the user went quiet
    closed_at: float | None = None  # perf_counter when recording stopped
    # True when the last speculative snapshot already holds all the speech: only
    # clear silence followed it, so its transcription is the final one.
    speculation_complete: bool = False


def wav_bytes(pcm: bytes, sample_rate: int = SAMPLE_RATE) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(pcm)
    return buffer.getvalue()


def resolve_input_device(name: str) -> tuple[int | None, str]:
    """Find a microphone whose name contains `name`; empty means system default."""

    try:
        import sounddevice as sd
    except ImportError as error:
        raise ListeningError("Falta sounddevice; instala requirements.txt.") from error
    if not name:
        return None, str(sd.query_devices(kind="input")["name"])
    inputs = [
        (index, str(device["name"]))
        for index, device in enumerate(sd.query_devices())
        if device["max_input_channels"] > 0
    ]
    for index, device_name in inputs:
        if name.lower() in device_name.lower():
            return index, device_name
    available = "; ".join(sorted({device_name for _, device_name in inputs}))
    raise ListeningError(
        f"No encontre un microfono que contenga '{name}' (VADER_MIC_DEVICE). "
        f"Disponibles: {available}"
    )


class FrameResampler:
    """Turn microphone blocks at any rate into 16 kHz frames for Silero VAD.

    Many USB microphones only record at 44.1 or 48 kHz. Each block is low-pass
    filtered (so nothing above 8 kHz folds back into the speech band), then
    sampled every `in_rate / 16000` input samples. Filter history and the read
    position carry over between blocks, so the joins are seamless.
    """

    _TAPS = 63

    def __init__(self, in_rate: int) -> None:
        import numpy as np

        self._np = np
        self._step = in_rate / SAMPLE_RATE
        cutoff = 0.45 / self._step  # ~7.2 kHz, in cycles per input sample
        n = np.arange(self._TAPS) - (self._TAPS - 1) / 2
        taps = 2 * cutoff * np.sinc(2 * cutoff * n) * np.hamming(self._TAPS)
        self._taps = taps / taps.sum()
        self._history = np.zeros(self._TAPS - 1)
        self._pending = np.zeros(0)  # filtered samples not yet read
        self._position = 0.0  # next read position inside `_pending`
        self._output = np.zeros(0, dtype=np.int16)

    def process(self, block: bytes) -> list[bytes]:
        """Feed one int16 mono block; return the complete 16 kHz frames so far."""

        np = self._np
        samples = np.frombuffer(block, dtype=np.int16).astype(np.float64)
        padded = np.concatenate([self._history, samples])
        self._history = padded[-(self._TAPS - 1) :]
        buffer = np.concatenate([self._pending, np.convolve(padded, self._taps, mode="valid")])
        # Read positions must leave one sample to the right for interpolation.
        count = max(0, math.ceil((len(buffer) - 1 - self._position) / self._step))
        positions = self._position + self._step * np.arange(count)
        resampled = np.interp(positions, np.arange(len(buffer)), buffer)
        end = self._position + self._step * count
        # `end` may land past this buffer; keep the remainder for the next block.
        dropped = min(int(end), len(buffer))
        self._pending = buffer[dropped:]
        self._position = end - dropped
        self._output = np.concatenate(
            [self._output, np.clip(np.round(resampled), -32768, 32767).astype(np.int16)]
        )
        complete = len(self._output) // FRAME_SAMPLES * FRAME_SAMPLES
        frames = [
            self._output[start : start + FRAME_SAMPLES].tobytes()
            for start in range(0, complete, FRAME_SAMPLES)
        ]
        self._output = self._output[complete:]
        return frames


class HighPassFilter:
    """Remove rumble below ~90 Hz from 16 kHz frames; speech sits above it.

    On the Pi most of the residual noise was below 200 Hz while the voice
    barely uses the band under 90 Hz. Second-order Butterworth biquad (RBJ
    cookbook); its state carries over between frames.
    """

    def __init__(self, cutoff_hz: float = 90.0, sample_rate: int = SAMPLE_RATE) -> None:
        w0 = 2 * math.pi * cutoff_hz / sample_rate
        cos_w0 = math.cos(w0)
        alpha = math.sin(w0) / (2 * (1 / math.sqrt(2)))
        a0 = 1 + alpha
        self._b0 = (1 + cos_w0) / 2 / a0
        self._b1 = -(1 + cos_w0) / a0
        self._b2 = self._b0
        self._a1 = -2 * cos_w0 / a0
        self._a2 = (1 - alpha) / a0
        self._x1 = self._x2 = self._y1 = self._y2 = 0.0

    def process(self, frame: bytes) -> bytes:
        b0, b1, b2, a1, a2 = self._b0, self._b1, self._b2, self._a1, self._a2
        x1, x2, y1, y2 = self._x1, self._x2, self._y1, self._y2
        samples = struct.unpack(f"<{len(frame) // 2}h", frame)
        output = []
        for x0 in samples:
            y0 = b0 * x0 + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2
            x2, x1, y2, y1 = x1, x0, y1, y0
            output.append(max(-32768, min(32767, int(y0))))
        self._x1, self._x2, self._y1, self._y2 = x1, x2, y1, y2
        return struct.pack(f"<{len(output)}h", *output)


def frame_converter(rate: int) -> Callable[[bytes], list[bytes]]:
    """Microphone blocks at `rate` -> high-passed 16 kHz audio, as Vader hears it."""

    resampler = FrameResampler(rate) if rate != SAMPLE_RATE else None
    high_pass = HighPassFilter()

    def convert(block: bytes) -> list[bytes]:
        frames = [block] if resampler is None else resampler.process(block)
        return [high_pass.process(frame) for frame in frames]

    return convert


def capture_rate(sd: object, device: int | None) -> int:
    """16 kHz when the microphone accepts it, then 48 kHz, else its default rate.

    48 kHz is an exact multiple of 16 kHz, and cheap USB microphones often
    record natively at 48 kHz and convert to 44.1 kHz internally.
    """

    for rate in (SAMPLE_RATE, 48_000):
        try:
            sd.check_input_settings(  # type: ignore[attr-defined]
                device=device, samplerate=rate, channels=1, dtype="int16"
            )
            return rate
        except Exception:
            continue
    info = sd.query_devices(device, "input")  # type: ignore[attr-defined]
    return int(info["default_samplerate"])


LAST_UTTERANCE_PATH = Path(tempfile.gettempdir()) / "vadermind_last_utterance.wav"


def log_utterance_audio(wav: bytes) -> None:
    """In debug mode, keep the last utterance and log how loud it was.

    The WAV is exactly what the transcriber receives, so it can be copied and
    heard when transcriptions go wrong. Peak near 100 % means clipping.
    """

    if not LOGGER.isEnabledFor(logging.DEBUG):
        return
    import numpy as np

    with wave.open(io.BytesIO(wav), "rb") as handle:
        samples = np.frombuffer(handle.readframes(handle.getnframes()), dtype=np.int16)
    if samples.size:
        values = samples.astype(np.float64) / 32768.0
        peak = float(np.abs(values).max())
        rms = float(np.sqrt(np.mean(values**2)))
        LOGGER.debug(
            "Audio enviado a transcribir: %.1f s, pico %.0f %%, promedio %.1f dBFS, "
            "saturadas %d muestras.",
            samples.size / SAMPLE_RATE,
            peak * 100,
            20 * math.log10(max(rms, 1e-9)),
            int(np.count_nonzero(np.abs(samples) >= 32700)),
        )
    try:
        LAST_UTTERANCE_PATH.write_bytes(wav)
    except OSError as error:
        LOGGER.debug("No se pudo guardar la ultima frase: %s", error)


def record_utterance(
    vad: SileroVad,
    detector: EndOfSpeechDetector,
    *,
    device: int | None = None,
    on_ready: Callable[[], None] | None = None,
    frames: Callable[[], "queue.Queue[bytes]"] | None = None,
    on_speech_start: Callable[[], None] | None = None,
    on_pause: Callable[[bytes], None] | None = None,
    pause_ms: int = 0,
    on_audio: Callable[[bytes], None] | None = None,
) -> Utterance:
    """Listen until the detector says the user finished; return a WAV in memory.

    The microphone opens first and `on_ready` (the start cue) plays afterwards,
    so listening is live the moment the cue ends. Audio captured while the cue
    plays is discarded, so the cue itself never reaches the transcriber.

    `on_speech_start` fires once, when the user starts talking. `on_pause`
    receives the PCM captured so far each time a pause reaches `pause_ms`, so
    transcription can start before the end of speech is confirmed. `on_audio`
    receives each piece of PCM as it joins the capture (the pre-roll first),
    so a streaming transcriber holds exactly what `on_pause` would send.
    """

    vad.reset()
    pre_roll: deque[bytes] = deque(maxlen=PRE_ROLL_MS // FRAME_MS)
    captured: list[bytes] = []
    started = time.perf_counter()
    speculating = False  # a snapshot was sent and only clear silence followed it

    def consume(source: "queue.Queue[bytes]") -> str:
        nonlocal speculating
        while True:
            try:
                frame = source.get(timeout=2.0)
            except queue.Empty as error:
                raise ListeningError("El microfono dejo de entregar audio.") from error
            probability = vad.speech_probability(frame)
            reason = detector.update(probability)
            if detector.speech_started:
                if not captured:
                    captured.extend(pre_roll)
                    pre_roll.clear()
                    if on_speech_start:
                        on_speech_start()
                    if on_audio and captured:
                        on_audio(b"".join(captured))
                captured.append(frame)
                if on_audio:
                    on_audio(frame)
                if on_pause and pause_ms > 0:
                    if not detector.is_clear_silence(probability):
                        speculating = False  # maybe more speech: the snapshot is stale
                    elif not speculating and detector.silence_run_ms >= pause_ms and not reason:
                        speculating = True
                        on_pause(b"".join(captured))
            else:
                pre_roll.append(frame)
            if reason:
                return reason

    if frames is not None:
        if on_ready:
            on_ready()
        reason = consume(frames())
    else:
        try:
            import sounddevice as sd
        except ImportError as error:
            raise ListeningError("Falta sounddevice; instala requirements.txt.") from error
        source: "queue.Queue[bytes]" = queue.Queue()
        try:
            rate = capture_rate(sd, device)
            if rate != SAMPLE_RATE:
                LOGGER.debug("Microfono a %d Hz; se convierte a %d Hz.", rate, SAMPLE_RATE)
            convert = frame_converter(rate)

            def callback(data: object, *_: object) -> None:
                for frame in convert(bytes(data)):  # type: ignore[arg-type]
                    source.put(frame)

            with sd.RawInputStream(
                samplerate=rate,
                channels=1,
                dtype="int16",
                blocksize=round(FRAME_SAMPLES * rate / SAMPLE_RATE),
                device=device,
                callback=callback,
            ):
                if on_ready:
                    on_ready()
                while not source.empty():
                    source.get_nowait()
                reason = consume(source)
        except sd.PortAudioError as error:
            raise ListeningError(f"No se pudo abrir el microfono: {error}") from error

    finished = time.perf_counter()
    return Utterance(
        wav=wav_bytes(b"".join(captured)),
        reason=reason,
        seconds=finished - started,
        speech_ended_at=(
            finished - detector.silence_ms / 1000 if reason == "fin_de_habla" else finished
        ),
        closed_at=finished,
        speculation_complete=speculating and reason == "fin_de_habla",
    )


# Muffled audio made the models invent other languages (English, Japanese,
# Russian). These words are frequent in English and rare in Spanish speech.
_ENGLISH_WORDS = {
    "the", "what", "who", "you", "your", "are", "is", "can", "i", "to", "of", "and",
    "when", "were", "was", "my", "it", "this", "with", "for", "do", "favorite",
}
_SPANISH_WORDS = {
    "el", "la", "los", "las", "que", "de", "en", "y", "es", "un", "una", "por", "con",
    "me", "te", "se", "lo", "mi", "como", "pon", "ya", "po", "al", "del", "para",
}


def matches_language(text: str, detected: str | None, expected: str = "es") -> bool:
    """False when a transcript is clearly not in the expected language.

    Uses the language the model reports (grok-stt does, gpt-4o does not), any
    letters outside the Latin alphabet, and, for Spanish, a count of English
    function words.
    """

    if detected and not detected.lower().startswith(expected.lower()):
        return False
    if any(char.isalpha() and "LATIN" not in unicodedata.name(char, "") for char in text):
        return False
    if expected.lower().startswith("es"):
        plain = unicodedata.normalize("NFD", text.lower())
        words = re.findall(r"[a-z]+", "".join(c for c in plain if unicodedata.category(c) != "Mn"))
        english = sum(word in _ENGLISH_WORDS for word in words)
        spanish = sum(word in _SPANISH_WORDS for word in words)
        if english >= 2 and english > spanish:
            return False
    return True


def is_usable_transcript(text: str) -> bool:
    """Reject empty text, lone symbols and phrases models invent from silence."""

    plain = unicodedata.normalize("NFD", text.lower())
    plain = "".join(char for char in plain if unicodedata.category(char) != "Mn")
    words = re.findall(r"[a-z0-9]+", plain)
    if not any(len(word) >= 2 for word in words):
        return False
    return " ".join(words) not in _HALLUCINATIONS


@dataclass(slots=True, frozen=True)
class VoiceTurn:
    """What the user said, or a short notice explaining why nothing was sent."""

    text: str | None
    notice: str | None = None
    model: str | None = None
    # Why there is no text: "sin_voz", "sin_transcripcion" or "no_entendido".
    reason: str | None = None


class VoiceListener:
    """One spoken turn: cue, record until silence, cue, transcribe, filter."""

    def __init__(
        self,
        vad: SileroVad,
        transcriber: "FallbackSpeechToText",
        cues: ListeningCues,
        *,
        silence_ms: int = 800,
        max_seconds: float = 15.0,
        device: int | None = None,
        device_name: str = "",
        speculative_ms: int = 0,
        language: str = "es",
        record: Callable[..., Utterance] = record_utterance,
        streamer: "StreamingSpeechToText | None" = None,
    ) -> None:
        self.vad = vad
        self.transcriber = transcriber
        self.cues = cues
        self.silence_ms = silence_ms
        self.max_seconds = max_seconds
        self.device = device
        self.device_name = device_name
        self.language = language
        # Speculating only helps if the pause is shorter than the end-of-speech wait.
        self.speculative_ms = speculative_ms if 0 < speculative_ms < silence_ms else 0
        # Streaming asks for the text at that same pause, so it needs one too.
        self.streamer = streamer if self.speculative_ms else None
        self.on_speech_start: Callable[[], None] | None = None
        self.on_recorded: Callable[[], None] | None = None  # recording closed
        self._record = record
        self._executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="stt")

    def close(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)
        self.transcriber.close()
        if self.streamer is not None:
            self.streamer.close()

    def listen(self, no_speech_seconds: float | None = None) -> VoiceTurn:
        """Record one turn; give up after `no_speech_seconds` without speech."""

        from vadermind.voice_input import TranscriptionError

        TURN_TIMELINE.reset()
        # A streamed turn never calls the REST transcriber: drop an earlier turn's error.
        self.transcriber.last_primary_error = None
        detector = EndOfSpeechDetector(
            silence_ms=self.silence_ms, max_seconds=self.max_seconds
        )
        if no_speech_seconds is not None:
            detector.no_speech_timeout_seconds = no_speech_seconds
        speculations: list[Future] = []
        # The socket opens while the start cue plays and the user starts talking.
        stream = self.streamer.open() if self.streamer is not None else None

        def on_pause(pcm: bytes) -> None:
            # Transcribe what was said so far while the detector confirms the end.
            if stream is not None and not stream.finished and not stream.failed:
                # The stream already holds exactly `pcm`: only ask for its text.
                speculations.append(stream.finish())
                return
            speculations.append(
                self._executor.submit(self.transcriber.transcribe_detailed, wav_bytes(pcm))
            )

        def on_ready() -> None:
            # The microphone is open; listening is live once the start cue ends.
            TURN_TIMELINE.mark("micro_abierto")
            self.cues.start()
            TURN_TIMELINE.mark("escucha_lista")

        def on_speech_start() -> None:
            TURN_TIMELINE.mark("inicio_habla")
            if self.on_speech_start:
                self.on_speech_start()

        try:
            with timed_operation("escucha"):
                utterance = self._record(
                    self.vad,
                    detector,
                    device=self.device,
                    on_ready=on_ready,
                    on_speech_start=on_speech_start,
                    on_pause=on_pause if self.speculative_ms else None,
                    pause_ms=self.speculative_ms,
                    on_audio=stream.feed if stream is not None else None,
                )
        finally:
            # The microphone is already closed: the end cue can overlap transcription.
            self.cues.end(wait=False)
            if stream is not None and not stream.finished:
                stream.cancel()  # no pause asked for its text
            if self.on_recorded:
                self.on_recorded()
        if utterance.reason == "sin_voz":
            return VoiceTurn(
                None, "No escuche nada. Presiona Enter para intentarlo de nuevo.", reason="sin_voz"
            )
        log_utterance_audio(utterance.wav)
        if utterance.speech_ended_at is not None:
            TURN_TIMELINE.mark("fin_habla", utterance.speech_ended_at)
        if utterance.closed_at is not None:
            TURN_TIMELINE.mark("grabacion_cerrada", utterance.closed_at)

        result = None
        if utterance.speculation_complete and speculations:
            try:
                with timed_operation("espera_transcripcion_especulativa"):
                    result = self._settle(speculations[-1], stream, utterance.wav)
            except TranscriptionError as error:
                LOGGER.debug("Speculative transcription failed, retrying: %s", error)
        if stream is not None:
            self._note_stream(stream, result)
        if speculations:
            LOGGER.debug(
                "Transcripciones especulativas: %d lanzadas, %s.",
                len(speculations),
                "se uso la ultima" if result is not None else "ninguna se uso",
            )
        TURN_TIMELINE.note("especulacion", "si" if result is not None else "no")
        # More than one means the user paused and went on: the earlier ones were discarded.
        TURN_TIMELINE.note("especulaciones", len(speculations))
        try:
            if result is None:
                with timed_operation("transcripcion"):
                    result = self.transcriber.transcribe_detailed(utterance.wav)
            TURN_TIMELINE.mark("transcripcion_lista")
        except TranscriptionError as error:
            LOGGER.warning("Transcription failed: %s", error)
            return VoiceTurn(
                None,
                "No pude transcribir tu voz. Intentalo de nuevo o escribe.",
                reason="sin_transcripcion",
            )
        if self.transcriber.last_primary_error:
            LOGGER.warning(
                "Primary STT failed, used %s: %s",
                result.model,
                self.transcriber.last_primary_error,
            )
        if not is_usable_transcript(result.text) or not matches_language(
            result.text, result.language, self.language
        ):
            heard = f" (escuche: \"{result.text}\")" if result.text.strip() else ""
            return VoiceTurn(
                None, f"No entendi lo que dijiste{heard}. Intentalo de nuevo.", reason="no_entendido"
            )
        notice = "Llegaste al maximo de grabacion; te corte." if utterance.reason == "maximo" else None
        return VoiceTurn(result.text, notice, result.model)

    def _settle(
        self, last: Future, stream: "TranscriptionStream | None", wav: bytes
    ) -> "TranscriptionResult":
        """Text of the last speculation; a slow stream races a REST request."""

        if stream is None or last is not stream.result or last.done():
            return last.result()
        # Usually the stream answered while the end of speech was confirmed.
        LOGGER.info("Transcripcion en streaming lenta: se pide tambien por REST.")
        rest = self._executor.submit(self.transcriber.transcribe_detailed, wav)
        return first_success([last, rest])

    @staticmethod
    def _note_stream(stream: "TranscriptionStream", result: "TranscriptionResult | None") -> None:
        from vadermind.stt_streaming import StreamCancelled

        streamed = (
            result is not None
            and stream.result.done()
            and not stream.failed
            and stream.result.result() is result
        )
        TURN_TIMELINE.note("transcriptor", "streaming" if streamed else "rest")
        if stream.connect_seconds is not None:
            TURN_TIMELINE.add_duration("conexion_stt_streaming", stream.connect_seconds)
        if streamed and stream.finish_seconds is not None:
            TURN_TIMELINE.add_duration("respuesta_stt_streaming", stream.finish_seconds)
        if stream.failed and not isinstance(stream.result.exception(), StreamCancelled):
            LOGGER.warning(
                "Transcripcion en streaming fallo; se uso REST: %s", stream.result.exception()
            )


def first_success(futures: list[Future]) -> "TranscriptionResult":
    """The first transcription to arrive (earlier futures win ties); raise if all fail."""

    from vadermind.voice_input import TranscriptionError

    pending = set(futures)
    error: TranscriptionError | None = None
    while pending:
        done, pending = wait(pending, return_when=FIRST_COMPLETED)
        for future in sorted(done, key=futures.index):
            try:
                return future.result()
            except TranscriptionError as failure:
                error = failure
    raise error or TranscriptionError("Ninguna transcripcion respondio.")


def build_voice_listener(settings: "Settings") -> VoiceListener:
    """Create the listener from settings; raises ListeningError if not ready."""

    from vadermind.stt_streaming import build_streaming_speech_to_text
    from vadermind.vad import VadError
    from vadermind.voice_input import TranscriptionError, build_speech_to_text

    device, device_name = resolve_input_device(settings.mic_device)
    try:
        vad = SileroVad(settings.vad_model_path)
        transcriber = build_speech_to_text(settings)
    except (VadError, TranscriptionError) as error:
        raise ListeningError(str(error)) from error
    return VoiceListener(
        vad,
        transcriber,
        ListeningCues(
            settings.listen_cues,
            settings.listen_start_sound,
            settings.listen_end_sound,
            first_sound=settings.listen_first_sound,
        ),
        silence_ms=settings.listen_silence_ms,
        max_seconds=settings.listen_max_seconds,
        device=device,
        device_name=device_name,
        speculative_ms=settings.stt_speculative_ms,
        language=settings.language,
        streamer=build_streaming_speech_to_text(settings),
    )


def _tone(path: Path, frequency: float, milliseconds: int) -> None:
    samples = SAMPLE_RATE * milliseconds // 1000
    fade = SAMPLE_RATE // 200  # 5 ms ramps avoid clicks
    pcm = bytearray()
    for index in range(samples):
        envelope = min(1.0, index / fade, (samples - index) / fade)
        value = 0.35 * envelope * math.sin(2 * math.pi * frequency * index / SAMPLE_RATE)
        pcm += struct.pack("<h", int(value * 32767))
    path.write_bytes(wav_bytes(bytes(pcm)))


_NO_SOUND = {"none", "off", "no", "ninguno"}


class ListeningCues:
    """Play a sound when listening starts and when it ends.

    Each sound is a WAV path, empty for the default beep, or "none" for silence.
    `first_sound`, if set, replaces the start sound only on the session's first
    spoken turn (for example, Vader's full breath when he first listens).
    """

    def __init__(
        self,
        enabled: bool,
        start_sound: str = "",
        end_sound: str = "",
        cache_dir: Path | None = None,
        first_sound: str = "",
    ) -> None:
        self.enabled = enabled
        cache = cache_dir or Path.cwd() / ".runtime_cache"
        self.start_path = self._resolve(start_sound, cache / "cue_listen_start.wav")
        self.end_path = self._resolve(end_sound, cache / "cue_listen_end.wav")
        self.first_path = self._resolve(first_sound, None)
        self._first_pending = self.first_path is not None
        if enabled:
            if not start_sound and self.start_path and not self.start_path.exists():
                cache.mkdir(parents=True, exist_ok=True)
                _tone(self.start_path, 880.0, 120)
            if not end_sound and self.end_path and not self.end_path.exists():
                cache.mkdir(parents=True, exist_ok=True)
                _tone(self.end_path, 587.0, 120)

    @staticmethod
    def _resolve(value: str, default: Path | None) -> Path | None:
        value = value.strip()
        if value.lower() in _NO_SOUND:
            return None
        return Path(value) if value else default

    def start(self) -> None:
        if self._first_pending:
            self._first_pending = False
            self._play(self.first_path)
            return
        self._play(self.start_path)

    def end(self, wait: bool = True) -> None:
        if self.end_path is None:
            return
        if not wait and self.enabled:
            threading.Thread(target=self._play, args=(self.end_path,), daemon=True).start()
            return
        self._play(self.end_path)

    def _play(self, path: Path | None) -> None:
        if not self.enabled or path is None:
            return
        try:
            _play_wav_file(path)
        except (VoiceError, OSError, RuntimeError):
            pass  # a missing cue must never block a turn
