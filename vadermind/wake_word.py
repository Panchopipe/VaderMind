"""Local wake word: wait for "Oye Vader" without sending audio anywhere.

Vosk (a small offline Spanish recognizer, ~40 MB) runs with a closed grammar.
Nothing leaves the device until the phrase is heard; only then does Vader
record the turn that goes to the transcription service.

A grammar with only "oye vader" and "[unk]" forces the recognizer to pick the
closest of two options, so "vai a ver" or a lone "oye" woke Vader (42 of 84
look-alike phrases in a test with Piper's Chilean voice). Two fixes, measured
on the same clips (12/12 hits kept, false wakes 42 -> 2 of 84):

- Decoys: similar-sounding phrases ("vas a ver", "oye", "vader", "hoy va a
  llover"...) give the recognizer a better match than the wake phrase.
- Only final results (after a short pause) count, never the early partial
  guess, and both words need `min_confidence`.

The detector is a small interface (`feed` / `reset`), so a trained
openWakeWord model can replace Vosk later without touching the rest.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
import queue
import re
import time
import unicodedata
from typing import TYPE_CHECKING, Callable, Protocol

from vadermind.listening import ListeningError, capture_rate, frame_converter
from vadermind.vad import FRAME_SAMPLES, SAMPLE_RATE

if TYPE_CHECKING:
    from vadermind.config import Settings

LOGGER = logging.getLogger("vadermind.wake")


def normalize(text: str) -> str:
    """Lowercase, no accents, no punctuation, single spaces."""

    text = unicodedata.normalize("NFKD", text.lower())
    text = "".join(char for char in text if not unicodedata.combining(char))
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9ñ ]", " ", text)).strip()


class WakeWordDetector(Protocol):
    def feed(self, frame: bytes) -> bool:
        """Take 16 kHz mono int16 audio; True when the wake phrase was heard."""

    def reset(self) -> None:
        """Forget audio heard so far (call after each detection)."""


# Look-alikes of "oye vader" in Chilean speech. "oye vale" is left out on
# purpose: as a decoy it also swallowed a real "Oye, Vader".
DECOYS = (
    "vas a ver", "va a ver", "a ver", "vamos a ver", "voy a ver", "hay que ver", "ver",
    "verla", "oye", "oye ya", "oye ven", "vale", "verde", "de verdad", "baile", "vader",
    "darth vader", "va", "vas", "ya", "venir", "hoy", "hoy va", "llover", "va a llover",
    "hacer", "que", "la", "tele", "pa", "o sea",
)


# The model's dictionary says "Vader" the Spanish way ("ba-der"). People say
# "Veider": "oye bei der" spells that sound with words the model knows. With
# Piper clips of "Oye Veider" it went from 1/36 to 29/36 detections with the
# same false wakes. Only ONE variant: similar alternatives split the
# recognizer's confidence between them (three together fell to 5/36).
DEFAULT_VARIANTS = ("oye bei der",)


class VoskWakeWord:
    """Spot the wake phrase with Vosk restricted to a closed grammar with decoys."""

    def __init__(
        self,
        model_path: str | Path,
        phrase: str = "oye vader",
        recognizer=None,
        min_confidence: float = 0.5,
        variants: tuple[str, ...] = DEFAULT_VARIANTS,
    ) -> None:
        self.phrase = normalize(phrase)
        # Every accepted way to say the phrase; any of them wakes Vader.
        self.accepted = [self.phrase] + [
            normalize(variant) for variant in variants if normalize(variant) not in ("", self.phrase)
        ]
        self.min_confidence = min_confidence
        if recognizer is None:
            try:
                from vosk import KaldiRecognizer, Model, SetLogLevel
            except ImportError as error:
                raise ListeningError("Falta vosk; instala requirements.txt.") from error
            if not Path(model_path).is_dir():
                raise ListeningError(
                    f"No encontre el modelo Vosk en {model_path} (VADER_WAKE_MODEL_PATH)."
                )
            SetLogLevel(-1)
            phrases = [*self.accepted, *(d for d in DECOYS if d not in self.accepted), "[unk]"]
            recognizer = KaldiRecognizer(
                Model(str(model_path)), SAMPLE_RATE, json.dumps(phrases, ensure_ascii=False)
            )
            recognizer.SetWords(True)  # per-word confidence in final results
        self._recognizer = recognizer
        # Vosk times each word in seconds of audio fed since the recognizer was
        # created (Reset does not restart that count; checked with vosk 0.3.x).
        self._fed_seconds = 0.0
        # Audio heard after the phrase ended before Vosk confirmed it, for timing.
        self.last_lag_seconds: float | None = None

    def _heard(self, payload: str) -> bool:
        """True if the final result holds the phrase with every word confident."""

        try:
            words = json.loads(payload).get("result", [])
        except (ValueError, AttributeError):
            return False
        heard = [normalize(str(word.get("word", ""))) for word in words]
        for phrase in self.accepted:
            target = phrase.split()
            for start in range(len(heard) - len(target) + 1):
                if heard[start : start + len(target)] == target:
                    matched = words[start : start + len(target)]
                    confidence = min(float(word.get("conf", 0.0)) for word in matched)
                    if confidence >= self.min_confidence:
                        try:
                            self.last_lag_seconds = max(
                                0.0, self._fed_seconds - float(matched[-1]["end"])
                            )
                        except (KeyError, TypeError, ValueError):
                            self.last_lag_seconds = None
                        return True
                    LOGGER.debug("'%s' con confianza baja (%.2f); no despierto.", phrase, confidence)
        return False

    def feed(self, frame: bytes) -> bool:
        self._fed_seconds += len(frame) / 2 / SAMPLE_RATE
        # Partial results are early guesses: in the test they woke Vader on a lone "oye".
        if self._recognizer.AcceptWaveform(frame):
            return self._heard(self._recognizer.Result())
        return False

    def reset(self) -> None:
        self._recognizer.Reset()


def wait_for_wake_word(
    detector: WakeWordDetector,
    *,
    device: int | None = None,
    frames: Callable[[], "queue.Queue[bytes]"] | None = None,
    should_stop: Callable[[], bool] = lambda: False,
    timing: dict[str, float] | None = None,
) -> bool:
    """Block until the wake phrase is heard (True) or `should_stop()` (False).

    The microphone is open only while waiting; Vader's own speech never
    reaches the detector because turns are half-duplex.

    `timing`, if given, receives perf_counter marks: `palabra_detectada` (Vosk
    confirmed the phrase) and `fin_palabra` (estimated end of the phrase:
    detection minus the audio Vosk needed after it and any frames still queued).
    """

    detector.reset()
    if timing is not None:
        timing.clear()

    def consume(source: "queue.Queue[bytes]") -> bool:
        while not should_stop():
            try:
                frame = source.get(timeout=2.0)
            except queue.Empty as error:
                raise ListeningError("El microfono dejo de entregar audio.") from error
            if detector.feed(frame):
                if timing is not None:
                    _record_wake_timing(detector, source, timing)
                detector.reset()
                return True
        return False

    if frames is not None:
        return consume(frames())
    try:
        import sounddevice as sd
    except ImportError as error:
        raise ListeningError("Falta sounddevice; instala requirements.txt.") from error
    source: "queue.Queue[bytes]" = queue.Queue()
    try:
        rate = capture_rate(sd, device)
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
            return consume(source)
    except sd.PortAudioError as error:
        raise ListeningError(f"No se pudo abrir el microfono: {error}") from error


def _record_wake_timing(
    detector: WakeWordDetector, source: "queue.Queue[bytes]", timing: dict[str, float]
) -> None:
    """Note when the phrase was recognized and estimate when the user finished it."""

    now = time.perf_counter()
    timing["palabra_detectada"] = now
    lag = getattr(detector, "last_lag_seconds", None)
    if lag is None:
        return
    # Frames still waiting in the queue were captured after the one just read.
    backlog = source.qsize() * FRAME_SAMPLES / SAMPLE_RATE
    timing["fin_palabra"] = now - backlog - lag


def build_wake_word(settings: "Settings") -> VoskWakeWord:
    return VoskWakeWord(
        settings.wake_model_path,
        settings.wake_phrase,
        min_confidence=settings.wake_min_confidence,
        variants=settings.wake_variants,
    )


# --- Local voice commands that never reach the language model ---------------

_SHUTDOWN = re.compile(r"\bapagate\b|\bapaga (la )?raspberry\b|\bdesconectate\b")
_YES = {"si", "confirmo", "dale", "ya", "afirmativo", "claro", "hazlo"}


def is_shutdown_request(text: str) -> bool:
    """"Vader, apágate" asks Vader to power off the Pi ("apaga la tele" does not)."""

    return bool(_SHUTDOWN.search(normalize(text)))


_SILENCE = {"silencio", "callate", "basta", "eso es todo", "olvidalo", "descansa"}
# Words that may surround the command without changing it ("Oye Vader, silencio ya").
_FILLER = {"oye", "vader", "darth", "ya", "por", "favor", "ahora", "mejor"}


def is_silence_request(text: str) -> bool:
    """True when the whole utterance only asks Vader to stop listening.

    "Vader, silencio" -> True; "pon la cancion Silencio" -> False, because the
    command must be the only meaningful content, not part of another order.
    """

    words = [word for word in normalize(text).split() if word not in _FILLER]
    return " ".join(words) in _SILENCE


# "modo canon" / "ponte en modo chileno". The transcription may write "cañón"
# for "canon", so that spelling counts too.
_PERSONA_MODE = re.compile(
    r"(?:(?:cambia|cambiate|pasa|pasate|ponte|vuelve|activa)\s+)?(?:(?:a|al|en)\s+)?"
    r"modo\s+(canon|canonico|cañon|chileno|chilena)"
)
_PERSONA_NAMES = {"canon": "canon", "canonico": "canon", "cañon": "canon",
                  "chileno": "chileno", "chilena": "chileno"}


def persona_mode_request(text: str) -> str | None:
    """'canon' or 'chileno' when the whole utterance only asks to switch mode.

    "Oye Vader, modo chileno" -> "chileno"; "¿qué es el modo canon?" -> None,
    because like "silencio" the order must be the only meaningful content.
    """

    words = [word for word in normalize(text).split() if word not in _FILLER]
    match = _PERSONA_MODE.fullmatch(" ".join(words))
    return _PERSONA_NAMES[match.group(1)] if match else None


def is_confirmation(text: str) -> bool:
    words = normalize(text).split()
    return bool(words) and words[0] in _YES
