"""Optional text-to-speech output for VaderMind."""

from __future__ import annotations

import atexit
from collections import deque
from dataclasses import dataclass, field
import io
from pathlib import Path
import logging
import queue
import re
import shutil
import subprocess
import tempfile
import threading
import time
from typing import Callable
import uuid
import wave

from vadermind.config import Settings
from vadermind.contracts import TextToSpeech
from vadermind.telemetry import STARTUP, TURN_TIMELINE, timed_operation

LOGGER = logging.getLogger("vadermind.voice")


SpeechObserver = Callable[[Path], None]
_SPEECH_OBSERVERS: list[SpeechObserver] = []


def add_speech_observer(observer: SpeechObserver) -> None:
    """Call `observer(wav_path)` right before each piece of Vader's voice plays."""

    _SPEECH_OBSERVERS.append(observer)


def remove_speech_observer(observer: SpeechObserver) -> None:
    if observer in _SPEECH_OBSERVERS:
        _SPEECH_OBSERVERS.remove(observer)


def _wav_seconds(file_path: Path) -> float | None:
    try:
        with wave.open(str(file_path), "rb") as source:
            return source.getnframes() / source.getframerate()
    except (wave.Error, OSError, ZeroDivisionError):
        return None


def _play_speech(file_path: Path) -> None:
    """Play Vader's voice, recording when the first sound of the turn began."""

    started = time.perf_counter()
    for observer in list(_SPEECH_OBSERVERS):
        try:
            observer(file_path)
        except Exception as error:  # the lights must never silence Vader
            LOGGER.debug("Observador de voz fallo: %s", error)
    TURN_TIMELINE.add_duration("envolvente_luces", time.perf_counter() - started)
    # How long the audio itself lasts: playback minus this is the player's overhead.
    seconds = _wav_seconds(file_path)
    if seconds is not None:
        TURN_TIMELINE.add_duration("audio_vader", seconds)
    TURN_TIMELINE.mark("primer_sonido")
    with timed_operation("reproduccion"):
        _play_wav_file(file_path)


class VoiceError(RuntimeError):
    """Raised when spoken output cannot be generated or played."""


# A sentence ends at . ! ? or … followed by a space; "10:32." or "3.5" do not split.
_SENTENCE_END = re.compile(r"(?<=[.!?…])\s+")
# Clauses: a comma, semicolon or colon followed by a space ("3,5" does not split).
_CLAUSE_END = re.compile(r"(?<=[,;:])\s+")
# The Chilean mode writes long sentences joined by commas; on the Pi a
# 130-character one took 2.8 s before any sound. Split those at commas.
LONG_SENTENCE_CHARS = 60
MIN_PIECE_CHARS = 25  # "Weón," alone would sound choppy: join it to what follows
MIN_TAIL_CHARS = 12  # a last clause like "weón." joins the piece before it


def split_sentences(text: str) -> list[str]:
    """Split a reply into short pieces so Piper can voice the first one right away."""

    pieces: list[str] = []
    for sentence in _SENTENCE_END.split(text.strip()):
        sentence = sentence.strip()
        if not sentence:
            continue
        if len(sentence) <= LONG_SENTENCE_CHARS:
            pieces.append(sentence)
            continue
        current = ""
        for clause in _CLAUSE_END.split(sentence):
            current = f"{current} {clause}".strip()
            if len(current) >= MIN_PIECE_CHARS:
                pieces.append(current)
                current = ""
        if current:  # a short tail joins the previous piece of this sentence
            if pieces and len(current) < MIN_TAIL_CHARS and pieces[-1][-1] in ",;:":
                pieces[-1] = f"{pieces[-1]} {current}"
            else:
                pieces.append(current)
    return pieces or [text.strip()]


def _join_wavs(parts: list[bytes], target: Path) -> None:
    """Write the sentences of one reply as a single WAV (for VADER_SAVE_LAST_AUDIO)."""

    with wave.open(str(target), "wb") as output:
        for index, part in enumerate(parts):
            with wave.open(io.BytesIO(part), "rb") as source:
                if index == 0:
                    output.setparams(source.getparams())
                output.writeframes(source.readframes(source.getnframes()))


class SilentTextToSpeech(TextToSpeech):
    """No-op text-to-speech implementation used when voice mode is disabled."""

    def speak(self, text: str) -> None:
        """Intentionally do nothing."""


class _PiperWorker:
    """One long-lived Piper process, so the voice model loads once per session.

    Launching Piper for every reply reloads the ONNX model: about 0.5 s on the
    Windows notebook, even for a one-word answer, and more on a Raspberry Pi.
    In this mode Piper reads one line of text per utterance from stdin, writes a
    WAV into `output_dir` and prints its path on stdout.
    """

    def __init__(self, command: list[str], output_dir: Path, timeout_seconds: float) -> None:
        self.command = command
        self.output_dir = output_dir
        self.timeout_seconds = timeout_seconds
        self.lock = threading.Lock()
        self._process: subprocess.Popen[str] | None = None
        self._paths: queue.Queue[str | None] = queue.Queue()
        self._stderr: deque[str] = deque(maxlen=20)

    def _ensure_started(self) -> subprocess.Popen[str]:
        if self._process is not None and self._process.poll() is None:
            return self._process
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._paths = queue.Queue()
        process = subprocess.Popen(
            [*self.command, "--output_dir", str(self.output_dir)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            bufsize=1,
        )
        threading.Thread(
            target=self._read_paths, args=(process, self._paths), daemon=True
        ).start()
        threading.Thread(target=self._read_errors, args=(process,), daemon=True).start()
        self._process = process
        return process

    @staticmethod
    def _read_paths(process: subprocess.Popen[str], paths: "queue.Queue[str | None]") -> None:
        assert process.stdout is not None
        for line in process.stdout:
            line = line.strip()
            if line:
                paths.put(line)
        process.stdout.close()
        paths.put(None)  # the process ended

    def _read_errors(self, process: subprocess.Popen[str]) -> None:
        assert process.stderr is not None
        for line in process.stderr:
            self._stderr.append(line.rstrip())
        process.stderr.close()

    def synthesize(self, text: str) -> Path:
        """Return the WAV Piper wrote for `text`; the caller owns and deletes it."""

        line = " ".join(text.split())  # one utterance per line
        with self.lock:
            try:
                process = self._ensure_started()
                assert process.stdin is not None
                process.stdin.write(line + "\n")
                process.stdin.flush()
            except OSError as error:
                self.close()
                raise VoiceError(f"No se pudo iniciar Piper persistente: {error}") from error
            try:
                printed = self._paths.get(timeout=self.timeout_seconds)
            except queue.Empty as error:
                self.close()
                raise VoiceError(
                    f"Piper excedio el limite de {self.timeout_seconds:g} segundos."
                ) from error
            if printed is None:
                detail = " | ".join(self._stderr) or "sin detalle"
                self.close()
                raise VoiceError(f"Piper persistente termino inesperadamente: {detail}")
        path = Path(printed)
        if not path.is_file():
            path = self.output_dir / path.name
        if not path.is_file() or path.stat().st_size <= 44:
            raise VoiceError("Piper genero un archivo de audio vacio o invalido.")
        return path

    def close(self) -> None:
        process, self._process = self._process, None
        if process is None:
            return
        try:
            if process.stdin is not None:
                process.stdin.close()
            process.wait(timeout=2)
        except (OSError, subprocess.TimeoutExpired):
            process.kill()


@dataclass(slots=True)
class PiperTextToSpeech(TextToSpeech):
    """Generate speech locally with Piper and play it on the host device."""

    settings: Settings
    _worker: _PiperWorker | None = field(default=None, init=False, repr=False)
    _persistent_failed: bool = field(default=False, init=False, repr=False)
    _worker_lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)

    def _base_command(self) -> list[str]:
        return [
            _resolve_piper_executable(self.settings.piper_executable),
            "--model",
            self.settings.piper_model_path,
            "--config",
            self.settings.piper_config_path,
            "--length_scale",
            str(self.settings.piper_length_scale),
            "--noise_scale",
            str(self.settings.piper_noise_scale),
            "--noise_w",
            str(self.settings.piper_noise_w),
            "--sentence_silence",
            str(self.settings.piper_sentence_silence),
            "--quiet",
        ]

    def _use_persistent(self) -> bool:
        return self.settings.piper_persistent and not self._persistent_failed

    def _get_worker(self) -> _PiperWorker:
        with self._worker_lock:  # the startup warm-up runs in another thread
            if self._worker is None:
                self._worker = _PiperWorker(
                    self._base_command(),
                    Path.cwd() / ".runtime_cache" / "piper_out",
                    self.settings.piper_timeout_seconds,
                )
                atexit.register(self._worker.close)
            return self._worker

    def warm(self) -> None:
        """Load the voice model and run one throwaway sentence before Vader speaks."""

        if not self._use_persistent():
            return
        try:
            with timed_operation("precalentamiento_piper"):
                self._get_worker().synthesize("Listo.").unlink(missing_ok=True)
            STARTUP.piper_ready()
        except VoiceError as error:
            LOGGER.warning("Piper persistente no arranco; uso un proceso por respuesta: %s", error)
            self._persistent_failed = True

    def close(self) -> None:
        if self._worker is not None:
            self._worker.close()

    def speak(self, text: str) -> None:
        """Synthesize with Piper and play it, one sentence at a time.

        With the persistent worker, the first sentence plays as soon as it is
        ready while the next ones are synthesized in the background: on a
        Raspberry Pi 4 a long reply took up to 4 s before any sound, but its
        first sentence takes about 1 s.
        """

        if not text.strip():
            raise VoiceError("Piper no recibio texto para sintetizar.")

        runtime_cache = Path.cwd() / ".runtime_cache"
        runtime_cache.mkdir(parents=True, exist_ok=True)
        # Without the persistent worker every sentence would reload the model.
        sentences = split_sentences(text) if self._use_persistent() else [text.strip()]
        ready: "queue.Queue[Path | VoiceError | None]" = queue.Queue()
        stop = threading.Event()

        def produce() -> None:
            try:
                for index, sentence in enumerate(sentences):
                    if stop.is_set():
                        return
                    target = runtime_cache / f"piper_{uuid.uuid4().hex}.wav"
                    if index == 0:
                        with timed_operation("sintesis_voz"):
                            ready.put(self._synthesize(sentence, target))
                    else:
                        ready.put(self._synthesize(sentence, target))
            except VoiceError as error:
                ready.put(error)
            finally:
                ready.put(None)

        TURN_TIMELINE.mark("sintesis_inicio")
        TURN_TIMELINE.note("trozos", len(sentences))
        if len(sentences) > 1:
            LOGGER.debug("Voz por oraciones: %d.", len(sentences))
        producer = threading.Thread(target=produce, daemon=True, name="piper-sentences")
        producer.start()
        played: list[bytes] = []
        try:
            pieces = 0
            while True:
                waiting = time.perf_counter()
                item = ready.get()
                if pieces and isinstance(item, Path):
                    # Silence between two pieces while Piper finishes the next one.
                    TURN_TIMELINE.add_duration("espera_trozos", time.perf_counter() - waiting)
                if item is None:
                    break
                if isinstance(item, VoiceError):
                    raise item
                pieces += 1
                try:
                    if self.settings.save_last_audio:
                        played.append(item.read_bytes())
                    _play_speech(item)
                finally:
                    item.unlink(missing_ok=True)
        finally:
            stop.set()
            producer.join()
            while not ready.empty():  # sentences never played after an error
                leftover = ready.get_nowait()
                if isinstance(leftover, Path):
                    leftover.unlink(missing_ok=True)
        if played:
            try:
                _join_wavs(played, Path.cwd() / "last_vader_reply.wav")
            except (wave.Error, OSError) as error:
                LOGGER.debug("No se pudo guardar la ultima respuesta: %s", error)

    def _synthesize(self, text: str, playback_path: Path) -> Path:
        if self._use_persistent():
            try:
                return self._get_worker().synthesize(text)
            except VoiceError as error:
                LOGGER.warning(
                    "Piper persistente fallo; sigo con un proceso por respuesta: %s", error
                )
                self._persistent_failed = True
        return self._synthesize_once(text, playback_path.with_suffix(".partial.wav"))

    def _synthesize_once(self, text: str, partial_path: Path) -> Path:
        """Launch Piper for a single reply (the original mode, kept as fallback)."""

        command = self._base_command()
        command[-1:-1] = ["--output_file", str(partial_path)]
        try:
            result = subprocess.run(
                command,
                input=f"{text.strip()}\n",
                text=True,
                encoding="utf-8",
                capture_output=True,
                check=False,
                timeout=self.settings.piper_timeout_seconds,
            )
        except subprocess.TimeoutExpired as error:
            partial_path.unlink(missing_ok=True)
            raise VoiceError(
                f"Piper excedio el limite de {self.settings.piper_timeout_seconds:g} segundos."
            ) from error
        except OSError as error:
            raise VoiceError(
                "No se pudo ejecutar Piper. Revisa PIPER_EXECUTABLE y sus permisos."
            ) from error
        if result.returncode != 0:
            partial_path.unlink(missing_ok=True)
            detail = (result.stderr or result.stdout).strip()
            raise VoiceError(
                f"Piper termino con codigo {result.returncode}: {detail or 'sin detalle'}"
            )
        if not partial_path.is_file() or partial_path.stat().st_size <= 44:
            partial_path.unlink(missing_ok=True)
            raise VoiceError("Piper genero un archivo de audio vacio o invalido.")
        return partial_path


@dataclass(slots=True)
class OpenAITextToSpeech(TextToSpeech):
    """Generate speech with OpenAI and play it locally on Windows."""

    settings: Settings

    def __post_init__(self) -> None:
        """Create the speech client lazily after dependency checks."""

        try:
            from openai import OpenAI
        except ImportError as error:
            raise VoiceError(
                "Falta la libreria 'openai'. Ejecuta 'py -m pip install -r requirements.txt'."
            ) from error

        self._client = OpenAI(api_key=self.settings.openai_api_key)

    def speak(self, text: str) -> None:
        """Generate a WAV file and play it with the local operating system."""

        temp_path = Path(tempfile.gettempdir()) / "vadermind_reply.wav"
        playback_path = Path(tempfile.gettempdir()) / "vadermind_reply_normalized.wav"
        TURN_TIMELINE.mark("sintesis_inicio")
        try:
            with self._client.audio.speech.with_streaming_response.create(
                model=self.settings.openai_tts_model,
                voice=self.settings.openai_tts_voice,
                input=text,
                response_format="wav",
                instructions=(
                    "Speak in a deep, controlled, solemn tone with mechanical calm. "
                    "Do not imitate any real actor. Do not shout. "
                    "Keep the delivery clear, slow, intimidating, and intelligible."
                ),
            ) as response:
                response.stream_to_file(temp_path)

            if temp_path.stat().st_size == 0:
                raise VoiceError("OpenAI genero un archivo de audio vacio.")

            _normalize_wav_file(temp_path, playback_path)

            if self.settings.save_last_audio:
                shutil.copyfile(playback_path, Path.cwd() / "last_vader_reply.wav")

            _play_speech(playback_path)
        except Exception as error:  # pragma: no cover - depends on local audio and network
            LOGGER.exception("Voice output failed.")
            raise VoiceError(
                "La voz no pudo reproducirse. El texto si fue generado, pero la salida de audio fallo."
            ) from error
        finally:
            temp_path.unlink(missing_ok=True)
            playback_path.unlink(missing_ok=True)


@dataclass(slots=True)
class ElevenLabsTextToSpeech(TextToSpeech):
    """Generate speech with ElevenLabs and play it locally on Windows."""

    settings: Settings

    def __post_init__(self) -> None:
        """Create the HTTP client lazily after dependency checks."""

        try:
            import httpx
        except ImportError as error:
            raise VoiceError(
                "Falta la libreria 'httpx'. Ejecuta 'py -m pip install -r requirements.txt'."
            ) from error

        self._httpx = httpx

    def speak(self, text: str) -> None:
        """Generate audio through ElevenLabs and play it as a WAV file."""

        TURN_TIMELINE.mark("sintesis_inicio")
        if self.settings.tts_streaming:
            self._speak_streaming(text)
            return

        output_format = self.settings.elevenlabs_output_format
        temp_path = Path(tempfile.gettempdir()) / "vadermind_elevenlabs_reply.wav"
        try:
            response = self._httpx.post(
                f"https://api.elevenlabs.io/v1/text-to-speech/{self.settings.elevenlabs_voice_id}",
                params={"output_format": output_format},
                headers={
                    "xi-api-key": self.settings.elevenlabs_api_key,
                    "Content-Type": "application/json",
                },
                json={
                    "text": text,
                    "model_id": self.settings.elevenlabs_model_id,
                    "voice_settings": {
                        "stability": self.settings.elevenlabs_stability,
                        "similarity_boost": self.settings.elevenlabs_similarity_boost,
                        "style": self.settings.elevenlabs_style,
                        "use_speaker_boost": self.settings.elevenlabs_use_speaker_boost,
                    },
                },
                timeout=60,
            )
            try:
                response.raise_for_status()
            except self._httpx.HTTPStatusError as error:
                raise VoiceError(_describe_elevenlabs_http_error(error.response.status_code)) from error

            if not response.content:
                raise VoiceError("ElevenLabs genero un archivo de audio vacio.")

            if output_format.startswith("pcm_"):
                _write_pcm_wav_file(response.content, temp_path, output_format)
            elif output_format.startswith("wav_"):
                temp_path.write_bytes(response.content)
            else:
                raise VoiceError(
                    "ElevenLabs debe usar un formato pcm_* o wav_* para reproducir audio en Windows."
                )

            if self.settings.save_last_audio:
                shutil.copyfile(temp_path, Path.cwd() / "last_vader_reply.wav")

            _play_speech(temp_path)
        except Exception as error:  # pragma: no cover - depends on local audio and network
            LOGGER.exception("ElevenLabs voice output failed.")
            raise VoiceError(
                "La voz de ElevenLabs no pudo reproducirse. El texto si fue generado, pero la salida de audio fallo."
            ) from error
        finally:
            temp_path.unlink(missing_ok=True)

    def _speak_streaming(self, text: str) -> None:
        """Stream ElevenLabs PCM audio directly to the local output device."""

        output_format = self.settings.elevenlabs_output_format
        if not output_format.startswith("pcm_"):
            raise VoiceError("El streaming de ElevenLabs requiere ELEVENLABS_OUTPUT_FORMAT=pcm_24000.")

        try:
            import sounddevice
        except ImportError as error:
            raise VoiceError(
                "Falta la libreria 'sounddevice'. Ejecuta 'py -m pip install -r requirements.txt'."
            ) from error

        sample_rate = int(output_format.split("_", 1)[1])
        audio_chunks: list[bytes] = []
        first_chunk_at: float | None = None
        started_at = time.perf_counter()
        stream_path = Path.cwd() / "last_vader_reply.wav"

        try:
            with sounddevice.RawOutputStream(
                samplerate=sample_rate,
                channels=1,
                dtype="int16",
                blocksize=0,
            ) as audio_stream:
                pending_audio = b""
                with self._httpx.stream(
                    "POST",
                    f"https://api.elevenlabs.io/v1/text-to-speech/{self.settings.elevenlabs_voice_id}/stream",
                    params={"output_format": output_format},
                    headers={
                        "xi-api-key": self.settings.elevenlabs_api_key,
                        "Content-Type": "application/json",
                    },
                    json={
                        "text": text,
                        "model_id": self.settings.elevenlabs_model_id,
                        "voice_settings": {
                            "stability": self.settings.elevenlabs_stability,
                            "similarity_boost": self.settings.elevenlabs_similarity_boost,
                            "style": self.settings.elevenlabs_style,
                            "use_speaker_boost": self.settings.elevenlabs_use_speaker_boost,
                        },
                    },
                    timeout=60,
                ) as response:
                    try:
                        response.raise_for_status()
                    except self._httpx.HTTPStatusError as error:
                        raise VoiceError(
                            _describe_elevenlabs_http_error(error.response.status_code)
                        ) from error

                    for chunk in response.iter_bytes():
                        if not chunk:
                            continue
                        if first_chunk_at is None:
                            first_chunk_at = time.perf_counter()
                            LOGGER.info(
                                "Primer chunk de audio ElevenLabs recibido en %.2f segundos.",
                                first_chunk_at - started_at,
                            )
                        audio_chunks.append(chunk)
                        pending_audio += chunk
                        aligned_byte_count = len(pending_audio) - (len(pending_audio) % 2)
                        if aligned_byte_count:
                            audio_stream.write(pending_audio[:aligned_byte_count])
                            pending_audio = pending_audio[aligned_byte_count:]

                if pending_audio:
                    audio_stream.write(pending_audio + b"\x00")

            if not audio_chunks:
                raise VoiceError("ElevenLabs no envio audio por streaming.")

            if self.settings.save_last_audio:
                _write_pcm_wav_file(b"".join(audio_chunks), stream_path, output_format)
        except Exception as error:  # pragma: no cover - depends on local audio and network
            LOGGER.exception("ElevenLabs streaming voice output failed.")
            raise VoiceError(
                "La voz por streaming de ElevenLabs no pudo reproducirse."
            ) from error


def _resolve_piper_executable(executable: str) -> str:
    """Return an absolute path when PIPER_EXECUTABLE is a relative file path.

    Windows can skip the working directory when launching a relative path written
    with forward slashes, for example when NoDefaultCurrentDirectoryInExePath is
    set by the terminal. Bare names such as ``piper`` keep using PATH lookup.
    """

    path = Path(executable)
    if len(path.parts) > 1 and not path.is_absolute() and path.is_file():
        return str(path.absolute())
    return executable


def _describe_elevenlabs_http_error(status_code: int) -> str:
    """Return a clear Spanish explanation for common ElevenLabs API failures."""

    if status_code == 401:
        return "ElevenLabs rechazo la API key. Revisa ELEVENLABS_API_KEY en .env."
    if status_code == 402:
        return "ElevenLabs requiere credito, plan activo o permisos de pago para generar esta voz."
    if status_code == 403:
        return "ElevenLabs no permite usar esta voz con tu cuenta o plan actual."
    if status_code == 404:
        return "ElevenLabs no encontro la voz configurada. Revisa ELEVENLABS_VOICE_ID."
    return f"ElevenLabs devolvio un error HTTP {status_code}."


def _write_pcm_wav_file(pcm_audio: bytes, target_path: Path, output_format: str) -> None:
    """Wrap ElevenLabs raw PCM output in a standard WAV container."""

    sample_rate = int(output_format.split("_", 1)[1])
    with wave.open(str(target_path), "wb") as target:
        target.setnchannels(1)
        target.setsampwidth(2)
        target.setframerate(sample_rate)
        target.writeframes(pcm_audio)


def _normalize_wav_file(source_path: Path, target_path: Path) -> None:
    """Rewrite streaming WAV files with a standard header for Windows playback."""

    with wave.open(str(source_path), "rb") as source:
        params = source.getparams()
        with wave.open(str(target_path), "wb") as target:
            target.setnchannels(params.nchannels)
            target.setsampwidth(params.sampwidth)
            target.setframerate(params.framerate)

            while True:
                chunk = source.readframes(8192)
                if not chunk:
                    break
                target.writeframes(chunk)


def _play_wav_file(file_path: Path) -> None:
    """Play a WAV file using Windows audio or a common Linux player."""

    try:
        import winsound
    except ImportError:
        for player, arguments in (
            ("aplay", ["-q"]),
            ("paplay", []),
            ("ffplay", ["-nodisp", "-autoexit", "-loglevel", "error"]),
        ):
            executable = shutil.which(player)
            if executable:
                result = subprocess.run(
                    [executable, *arguments, str(file_path)],
                    capture_output=True,
                    check=False,
                )
                if result.returncode == 0:
                    return
                detail = result.stderr.decode(errors="replace").strip()
                raise VoiceError(f"{player} no pudo reproducir el audio: {detail}")

        raise VoiceError(
            "No se encontro un reproductor WAV. En Raspberry instala 'alsa-utils' para usar aplay."
        )

    winsound.PlaySound(str(file_path), winsound.SND_FILENAME)
