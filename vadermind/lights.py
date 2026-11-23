"""Status lights for the helmet: Vader's eyes show what he is doing.

Three interchangeable versions share one small interface, so Vader works the
same with or without LEDs:

- `NullLights`: does nothing (the default; laptop or helmet without LEDs).
- `ConsoleLights`: prints each change of state (to try it on Windows).
- `Ws2812Lights`: two WS2812 modules of 7 LEDs (center "pupil" + ring of 6)
  chained on the Raspberry Pi SPI bus (GPIO10, physical pin 19).

States, all in Vader red with a different pattern each:

- reposo: only the pupils, dim (waiting for "Oye Vader")
- escuchando: pupils and ring steady
- pensando: the ring breathes slowly
- hablando: the ring follows the volume of the sentence being played
- error: two quick blinks, then back to reposo
- apagado: everything off (also while Vader loads at start-up)

Speech is an overlay: `speak()` lights the ring with a volume envelope and,
when the envelope ends, the eyes go back to the state they had before.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
import math
import os
from pathlib import Path
import threading
import time
from typing import TYPE_CHECKING, Callable, Sequence
import wave

if TYPE_CHECKING:
    from vadermind.config import Settings

LOGGER = logging.getLogger("vadermind.lights")

REPOSO = "reposo"
ESCUCHANDO = "escuchando"
PENSANDO = "pensando"
HABLANDO = "hablando"
ERROR = "error"
APAGADO = "apagado"
STATES = (REPOSO, ESCUCHANDO, PENSANDO, HABLANDO, ERROR, APAGADO)

RED = (255, 0, 0)
MAX_BRIGHTNESS = 0.25  # the eyes share the Pi's 5 V pin: never more than this
LEDS_PER_EYE = 7
IDLE_PUPIL = 0.35  # reposo: pupil at this fraction of the brightness
LISTEN_RING = 0.55
THINK_PERIOD_S = 1.6
THINK_FLOOR = 0.15
SPEAK_FLOOR = 0.15  # the ring never goes fully dark while Vader speaks
ERROR_BLINK_S = 0.15  # on, off, on, off
ERROR_SECONDS = 4 * ERROR_BLINK_S
ENVELOPE_WINDOW_S = 0.04
FRAME_INTERVAL_S = 1 / 30

Color = tuple[int, int, int]


# --- WS2812 over SPI ---------------------------------------------------------
# At 2.4 MHz each SPI bit lasts ~417 ns; each LED bit is sent as three SPI bits:
# 0 -> 100 (short high pulse), 1 -> 110 (long high pulse). Colors go in
# green-red-blue order and ~333 us of low line latches them.

SPI_SPEED_HZ = 2_400_000
RESET_BYTES = 100


def _encode_byte(value: int) -> bytes:
    bits = 0
    for position in range(7, -1, -1):
        bits = (bits << 3) | (0b110 if (value >> position) & 1 else 0b100)
    return bits.to_bytes(3, "big")


_ENCODED = [_encode_byte(value) for value in range(256)]


def encode(colors: Sequence[Color]) -> bytes:
    """Full SPI frame for a chain of LEDs given as RGB tuples."""

    frame = bytearray(b"\x00")
    for red, green, blue in colors:
        for channel in (green, red, blue):
            frame += _ENCODED[channel]
    frame += bytes(RESET_BYTES)
    return bytes(frame)


def _ioc_write(number: int, size: int) -> int:
    """Linux _IOW('k', number, size) for the spidev ioctls."""

    return (1 << 30) | (size << 16) | (ord("k") << 8) | number


SPI_IOC_WR_MODE = _ioc_write(1, 1)
SPI_IOC_WR_MAX_SPEED_HZ = _ioc_write(4, 4)


class SpiDevice:
    """Write raw bytes to /dev/spidevB.D without the `spidev` package.

    Vader runs inside its virtual environment, where the Debian package
    python3-spidev is not visible. The kernel driver only needs the mode and
    speed set once; after that, a plain write() sends the bytes.
    """

    def __init__(self, path: str = "/dev/spidev0.0", speed_hz: int = SPI_SPEED_HZ) -> None:
        import fcntl
        import struct

        self._fd = os.open(path, os.O_WRONLY)
        try:
            fcntl.ioctl(self._fd, SPI_IOC_WR_MODE, struct.pack("B", 0))
            fcntl.ioctl(self._fd, SPI_IOC_WR_MAX_SPEED_HZ, struct.pack("I", speed_hz))
        except OSError:
            os.close(self._fd)
            raise

    def write(self, data: bytes) -> None:
        os.write(self._fd, data)

    def close(self) -> None:
        os.close(self._fd)


# --- What each state looks like ----------------------------------------------


def scale(color: Color, level: float) -> Color:
    level = min(max(level, 0.0), MAX_BRIGHTNESS)
    return tuple(round(channel * level) for channel in color)  # type: ignore[return-value]


def eye_frame(
    state: str,
    elapsed: float,
    brightness: float,
    *,
    level: float = 0.0,
    eyes: int = 2,
    center_index: int = 0,
    color: Color = RED,
) -> list[Color]:
    """Colors for the whole chain at `elapsed` seconds into `state`.

    `level` (0 to 1) is the speech volume, used only while speaking.
    """

    pupil, ring = 0.0, 0.0
    if state == REPOSO:
        pupil = IDLE_PUPIL
    elif state == ESCUCHANDO:
        pupil, ring = 1.0, LISTEN_RING
    elif state == PENSANDO:
        wave_ = (1 - math.cos(2 * math.pi * elapsed / THINK_PERIOD_S)) / 2
        pupil, ring = 1.0, THINK_FLOOR + (1 - THINK_FLOOR) * wave_
    elif state == HABLANDO:
        pupil, ring = 1.0, SPEAK_FLOOR + (1 - SPEAK_FLOOR) * min(max(level, 0.0), 1.0)
    elif state == ERROR:
        lit = int(elapsed / ERROR_BLINK_S) % 2 == 0 and elapsed < ERROR_SECONDS
        pupil = ring = 1.0 if lit else 0.0
    pupil_color = scale(color, brightness * pupil)
    ring_color = scale(color, brightness * ring)
    eye = [pupil_color if index == center_index else ring_color for index in range(LEDS_PER_EYE)]
    return eye * eyes


def speech_envelope(wav_path: Path, window_s: float = ENVELOPE_WINDOW_S) -> list[float]:
    """Loudness of a 16-bit WAV every `window_s`, from 0 to 1 (1 = its loud parts)."""

    with wave.open(str(wav_path), "rb") as source:
        if source.getsampwidth() != 2:
            return []
        rate, channels = source.getframerate(), source.getnchannels()
        raw = source.readframes(source.getnframes())
    import numpy as np

    samples = np.frombuffer(raw[: len(raw) - len(raw) % 2], dtype="<i2")[::channels]
    step = max(1, int(rate * window_s))
    if not len(samples):
        return []
    padded = np.zeros(-(-len(samples) // step) * step, dtype=np.float64)
    padded[: len(samples)] = samples
    levels = np.sqrt(np.mean(padded.reshape(-1, step) ** 2, axis=1))
    reference = float(np.percentile(levels, 95)) or 1.0
    return [float(value) for value in np.minimum(1.0, np.sqrt(levels / reference))]


# --- The three versions ----------------------------------------------------------


class Lights:
    """Interface: Vader calls these without knowing which version is behind."""

    def set_state(self, state: str) -> None:
        """Change the base state (reposo, escuchando, pensando, error, apagado)."""

    def speak(self, envelope: Sequence[float], window_s: float = ENVELOPE_WINDOW_S) -> None:
        """Follow this volume envelope from now; then return to the base state."""

    def close(self) -> None:
        """Turn everything off and release the hardware."""


class NullLights(Lights):
    """No LEDs: every call does nothing."""


class ConsoleLights(Lights):
    """Print the state changes, to follow the logic on a computer."""

    def __init__(self, write: Callable[[str], None] = print) -> None:
        self._write = write
        self._state = ""

    def set_state(self, state: str) -> None:
        if state != self._state:
            self._state = state
            self._write(f"[luces] {state}")

    def speak(self, envelope: Sequence[float], window_s: float = ENVELOPE_WINDOW_S) -> None:
        self._write(f"[luces] hablando ({len(envelope) * window_s:.1f} s)")

    def close(self) -> None:
        self.set_state(APAGADO)


@dataclass(slots=True)
class _Speech:
    envelope: Sequence[float]
    window_s: float
    started: float


class Ws2812Lights(Lights):
    """Animate the eyes from a background thread at ~30 frames per second."""

    def __init__(
        self,
        device: SpiDevice,
        *,
        brightness: float = 0.10,
        eyes: int = 2,
        center_index: int = 0,
        clock: Callable[[], float] = time.monotonic,
        interval_s: float = FRAME_INTERVAL_S,
    ) -> None:
        self._device = device
        self.brightness = min(max(brightness, 0.0), MAX_BRIGHTNESS)
        self.eyes = eyes
        self.center_index = center_index
        self._clock = clock
        self._interval_s = interval_s
        self._lock = threading.Lock()
        # Dark while Vader loads: the dim pupils (reposo) must mean "ready", so
        # only the session turns them on, once it is waiting for "Oye Vader".
        self._state = APAGADO
        self._since = clock()
        self._speech: _Speech | None = None
        self._last_frame: bytes | None = None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True, name="luces")
        self._thread.start()

    def set_state(self, state: str) -> None:
        if state not in STATES or state == HABLANDO:
            raise ValueError(f"Estado de luces no valido: {state}")
        with self._lock:
            if state != self._state or state == ERROR:
                self._state, self._since = state, self._clock()
            self._speech = None

    def speak(self, envelope: Sequence[float], window_s: float = ENVELOPE_WINDOW_S) -> None:
        with self._lock:
            self._speech = _Speech(list(envelope), window_s, self._clock())

    def frame(self) -> list[Color]:
        """Colors for this instant (also advances error -> reposo and ends speech)."""

        with self._lock:
            now = self._clock()
            if self._state == ERROR and now - self._since >= ERROR_SECONDS:
                self._state, self._since = REPOSO, now
            if self._speech is not None:
                index = int((now - self._speech.started) / self._speech.window_s)
                if index < len(self._speech.envelope):
                    return eye_frame(
                        HABLANDO, 0.0, self.brightness, level=self._speech.envelope[index],
                        eyes=self.eyes, center_index=self.center_index,
                    )
                self._speech = None
            return eye_frame(
                self._state, now - self._since, self.brightness,
                eyes=self.eyes, center_index=self.center_index,
            )

    def _show(self, colors: list[Color]) -> None:
        data = encode(colors)
        if data != self._last_frame:  # a still frame is sent only once
            self._device.write(data)
            self._last_frame = data

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                self._show(self.frame())
            except OSError as error:
                LOGGER.warning("Las luces dejaron de responder: %s", error)
                return
            self._stop.wait(self._interval_s)

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=1.0)
        try:
            self._device.write(encode([(0, 0, 0)] * LEDS_PER_EYE * self.eyes))
        finally:
            self._device.close()


def build_lights(settings: "Settings") -> Lights:
    """Lights chosen by VADER_LIGHTS: none, console or ws2812."""

    kind = settings.lights
    if kind == "console":
        return ConsoleLights()
    if kind == "ws2812":
        try:
            device = SpiDevice(settings.lights_spi_device)
        except (OSError, ImportError) as error:
            LOGGER.warning("No pude abrir %s; sigo sin luces: %s", settings.lights_spi_device, error)
            return NullLights()
        return Ws2812Lights(
            device,
            brightness=settings.lights_brightness,
            eyes=settings.lights_eyes,
            center_index=settings.lights_center_index,
        )
    return NullLights()
