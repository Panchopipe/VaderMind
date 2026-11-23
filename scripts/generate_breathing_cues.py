"""Generate original mechanical-respirator breathing cues for --listen.

Nothing is sampled from films: every sound is shaped noise built here, so the
WAVs can be regenerated anywhere (for example on the Raspberry Pi).

Usage from the project root:

    .\\.venv\\Scripts\\python.exe -m scripts.generate_breathing_cues
    .\\.venv\\Scripts\\python.exe -m scripts.generate_breathing_cues --out-dir .runtime_cache

Writes, for each variant, an inhale (start cue), an exhale (end cue) and a
preview with both, into audio_tests/vader_breathing/ by default (outside Git).
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import wave

import numpy as np

SAMPLE_RATE = 22_050
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = PROJECT_ROOT / "audio_tests" / "vader_breathing"


@dataclass(frozen=True)
class Voice:
    """Tone of one respirator: where the air resonates and how much it rattles."""

    name: str
    inhale_formants: tuple[tuple[float, float, float], ...]  # (Hz, width Hz, gain)
    exhale_formants: tuple[tuple[float, float, float], ...]
    tube_ms: float  # length of the mask/tube echo; gives the hollow sound
    tube_feedback: float
    rattle_hz: float  # mechanical flutter of the valve
    rattle_depth: float
    whistle_hz: float  # faint tonal whistle on the inhale (0 disables it)


VOICES = (
    Voice(
        name="clasica",
        inhale_formants=((900, 500, 1.0), (1900, 700, 0.8), (3200, 900, 0.35)),
        exhale_formants=((450, 350, 1.0), (1100, 600, 0.7), (2400, 900, 0.3)),
        tube_ms=3.4,
        tube_feedback=0.55,
        rattle_hz=38.0,
        rattle_depth=0.35,
        whistle_hz=1450.0,
    ),
    Voice(
        name="mecanica",
        inhale_formants=((700, 300, 1.0), (1500, 400, 0.9), (2800, 600, 0.5)),
        exhale_formants=((350, 250, 1.0), (900, 400, 0.8), (2000, 700, 0.4)),
        tube_ms=5.2,
        tube_feedback=0.7,
        rattle_hz=52.0,
        rattle_depth=0.55,
        whistle_hz=0.0,
    ),
)


def _shaped_noise(
    rng: np.random.Generator,
    seconds: float,
    formants: tuple[tuple[float, float, float], ...],
    voice: Voice,
) -> np.ndarray:
    """White noise filtered in the frequency domain by formants and a tube comb."""

    size = int(seconds * SAMPLE_RATE)
    spectrum = np.fft.rfft(rng.standard_normal(size))
    freqs = np.fft.rfftfreq(size, 1 / SAMPLE_RATE)
    shape = np.zeros_like(freqs)
    for center, width, gain in formants:
        shape += gain * np.exp(-0.5 * ((freqs - center) / width) ** 2)
    # Feedback comb: echoes inside a short tube, heard as a hollow, masked tone.
    delay = voice.tube_ms / 1000
    comb = 1 / np.abs(1 - voice.tube_feedback * np.exp(-2j * np.pi * freqs * delay))
    shape *= comb / comb.max()
    shape[freqs < 120] = 0  # no rumble
    return np.fft.irfft(spectrum * shape, n=size)


def _flutter(rng: np.random.Generator, size: int, voice: Voice) -> np.ndarray:
    """Irregular amplitude wobble around rattle_hz, like a valve flap."""

    t = np.arange(size) / SAMPLE_RATE
    drift = np.cumsum(rng.standard_normal(size)) / SAMPLE_RATE * 6
    return 1 - voice.rattle_depth * (0.5 + 0.5 * np.sin(2 * np.pi * voice.rattle_hz * t + drift))


def _envelope(size: int, attack: float, release: float) -> np.ndarray:
    """Smooth swell and fade (fractions of the length)."""

    x = np.linspace(0, 1, size)
    rise = np.clip(x / attack, 0, 1) ** 1.5
    fall = np.clip((1 - x) / release, 0, 1) ** 2
    return np.sin(np.pi / 2 * rise) * fall


def _valve_click(rng: np.random.Generator, seconds: float, pitch: float) -> np.ndarray:
    """Short mechanical 'tick' of the demand valve opening."""

    size = int(seconds * SAMPLE_RATE)
    t = np.arange(size) / SAMPLE_RATE
    ring = np.sin(2 * np.pi * pitch * t) * np.exp(-t * 180)
    thump = np.sin(2 * np.pi * 110 * t) * np.exp(-t * 60) * 0.6
    grit = rng.standard_normal(size) * np.exp(-t * 400) * 0.3
    return (ring + thump + grit) * 0.5


def inhale(voice: Voice, seconds: float, seed: int = 7) -> np.ndarray:
    rng = np.random.default_rng(seed)
    air = _shaped_noise(rng, seconds, voice.inhale_formants, voice)
    size = air.size
    air *= _envelope(size, attack=0.35, release=0.25) * _flutter(rng, size, voice)
    if voice.whistle_hz:
        t = np.arange(size) / SAMPLE_RATE
        glide = voice.whistle_hz * (1 + 0.06 * t / seconds)  # rises as air speeds up
        whistle = np.sin(2 * np.pi * np.cumsum(glide) / SAMPLE_RATE)
        air += 0.08 * np.std(air) * whistle * _envelope(size, 0.5, 0.3) * 3
    click = _valve_click(rng, 0.06, 2600)
    out = air / np.max(np.abs(air))
    out[: click.size] += click
    return out


def exhale(voice: Voice, seconds: float, seed: int = 11) -> np.ndarray:
    rng = np.random.default_rng(seed)
    air = _shaped_noise(rng, seconds, voice.exhale_formants, voice)
    size = air.size
    # Exhale rattles more and fades slowly, like air pushed out through the valve.
    stronger = Voice(**{**voice.__dict__, "rattle_depth": min(0.9, voice.rattle_depth * 1.4)})
    air *= _envelope(size, attack=0.12, release=0.55) * _flutter(rng, size, stronger)
    click = _valve_click(rng, 0.05, 1800)
    out = air / np.max(np.abs(air))
    out[: click.size] += click * 0.8
    return out


def _write(path: Path, audio: np.ndarray, peak: float = 0.5) -> None:
    audio = audio / np.max(np.abs(audio)) * peak
    fade = int(0.005 * SAMPLE_RATE)
    audio[:fade] *= np.linspace(0, 1, fade)
    audio[-fade:] *= np.linspace(1, 0, fade)
    pcm = (audio * 32767).astype("<i2").tobytes()
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(SAMPLE_RATE)
        handle.writeframes(pcm)


def main() -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--inhale-seconds", type=float, default=0.65)
    parser.add_argument("--exhale-seconds", type=float, default=1.0)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    for voice in VOICES:
        start = inhale(voice, args.inhale_seconds)
        end = exhale(voice, args.exhale_seconds)
        _write(args.out_dir / f"{voice.name}_inhala.wav", start)
        _write(args.out_dir / f"{voice.name}_exhala.wav", end)
        gap = np.zeros(int(0.25 * SAMPLE_RATE))
        cycle = np.concatenate([start / np.max(np.abs(start)), gap, end / np.max(np.abs(end)), gap * 3])
        _write(args.out_dir / f"{voice.name}_muestra.wav", np.tile(cycle, 2))
        print(f"{voice.name}: {args.inhale_seconds:.2f} s inhala, {args.exhale_seconds:.2f} s exhala")
    print(f"Archivos en {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
