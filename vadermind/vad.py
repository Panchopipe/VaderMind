"""Voice activity detection: Silero VAD plus the end-of-speech rule."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

SAMPLE_RATE = 16_000
FRAME_SAMPLES = 512  # Silero VAD v5+ expects 512 samples (32 ms) at 16 kHz.
FRAME_MS = FRAME_SAMPLES * 1000 // SAMPLE_RATE
_CONTEXT_SAMPLES = 64


class VadError(RuntimeError):
    """Raised when the VAD model cannot be loaded or run."""


class SileroVad:
    """Run the Silero ONNX model on 32 ms frames and return speech probability."""

    def __init__(self, model_path: str | Path) -> None:
        path = Path(model_path)
        if not path.is_file():
            raise VadError(
                f"No se encontro el modelo VAD en {path}. Ver docs/local-artifacts.md."
            )
        try:
            import numpy as np
            import onnxruntime as ort
        except ImportError as error:
            raise VadError("Faltan onnxruntime o numpy; instala requirements.txt.") from error

        options = ort.SessionOptions()
        options.inter_op_num_threads = 1
        options.intra_op_num_threads = 1
        self._np = np
        self._session = ort.InferenceSession(
            str(path), sess_options=options, providers=["CPUExecutionProvider"]
        )
        self._sr = np.array(SAMPLE_RATE, dtype=np.int64)
        self.reset()

    def reset(self) -> None:
        """Forget previous audio before a new utterance."""

        self._state = self._np.zeros((2, 1, 128), dtype=self._np.float32)
        self._context = self._np.zeros((1, _CONTEXT_SAMPLES), dtype=self._np.float32)

    def speech_probability(self, frame: bytes) -> float:
        """Return 0..1 for one frame of 512 int16 mono samples."""

        np = self._np
        samples = np.frombuffer(frame, dtype=np.int16).astype(np.float32) / 32768.0
        if samples.size != FRAME_SAMPLES:
            raise VadError(f"El VAD necesita {FRAME_SAMPLES} muestras por frame.")
        window = np.concatenate([self._context, samples[np.newaxis, :]], axis=1)
        output, self._state = self._session.run(
            None, {"input": window, "state": self._state, "sr": self._sr}
        )
        self._context = window[:, -_CONTEXT_SAMPLES:]
        return float(output[0][0])


@dataclass(slots=True)
class EndOfSpeechDetector:
    """Decide, frame by frame, when the user started and finished talking.

    Speech starts after `min_speech_ms` of frames above `threshold`; it ends
    after `silence_ms` below `threshold - 0.15` (hysteresis avoids flicker).
    """

    threshold: float = 0.5
    silence_ms: int = 800
    min_speech_ms: int = 96
    max_seconds: float = 15.0
    no_speech_timeout_seconds: float = 6.0
    speech_started: bool = False
    _elapsed_ms: int = 0
    _voiced_run_ms: int = 0
    _silence_run_ms: int = 0

    def update(self, probability: float) -> str | None:
        """Feed one frame; return None to keep listening or the reason to stop.

        Reasons: "fin_de_habla", "maximo", "sin_voz".
        """

        self._elapsed_ms += FRAME_MS
        if not self.speech_started:
            self._voiced_run_ms = self._voiced_run_ms + FRAME_MS if probability >= self.threshold else 0
            if self._voiced_run_ms >= self.min_speech_ms:
                self.speech_started = True
            elif self._elapsed_ms >= self.no_speech_timeout_seconds * 1000:
                return "sin_voz"
        else:
            if probability < self.threshold - 0.15:
                self._silence_run_ms += FRAME_MS
                if self._silence_run_ms >= self.silence_ms:
                    return "fin_de_habla"
            elif probability >= self.threshold:
                self._silence_run_ms = 0
        if self._elapsed_ms >= self.max_seconds * 1000:
            return "maximo"
        return None

    @property
    def silence_run_ms(self) -> int:
        """How long the current pause has lasted, counting only clear silence."""

        return self._silence_run_ms

    def is_clear_silence(self, probability: float) -> bool:
        """True for frames below the lower threshold: certainly not speech."""

        return probability < self.threshold - 0.15
