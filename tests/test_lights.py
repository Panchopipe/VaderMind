"""Tests for the helmet status lights (no hardware needed)."""

from __future__ import annotations

from pathlib import Path
import tempfile
import threading
import unittest
import wave

from types import SimpleNamespace
from vadermind.lights import (
    APAGADO,
    ERROR,
    ERROR_SECONDS,
    ESCUCHANDO,
    LEDS_PER_EYE,
    MAX_BRIGHTNESS,
    PENSANDO,
    REPOSO,
    SPI_IOC_WR_MAX_SPEED_HZ,
    SPI_IOC_WR_MODE,
    ConsoleLights,
    NullLights,
    Ws2812Lights,
    build_lights,
    encode,
    eye_frame,
    speech_envelope,
)


class FakeDevice:
    def __init__(self) -> None:
        self.writes: list[bytes] = []
        self.closed = False

    def write(self, data: bytes) -> None:
        self.writes.append(data)

    def close(self) -> None:
        self.closed = True


class FakeClock:
    def __init__(self) -> None:
        self.now = 100.0

    def __call__(self) -> float:
        return self.now


def _lights(clock: FakeClock, device: FakeDevice | None = None) -> Ws2812Lights:
    lights = Ws2812Lights(device or FakeDevice(), brightness=0.1, clock=clock, interval_s=3600)
    return lights


class EncodingTests(unittest.TestCase):
    def test_ioctl_numbers_match_linux_spidev(self) -> None:
        self.assertEqual(SPI_IOC_WR_MODE, 0x40016B01)
        self.assertEqual(SPI_IOC_WR_MAX_SPEED_HZ, 0x40046B04)

    def test_encode_sends_green_red_blue_with_latch(self) -> None:
        frame = encode([(255, 0, 0)])
        self.assertEqual(frame[0], 0)
        self.assertEqual(frame[1:4], bytes.fromhex("924924"))  # green 0
        self.assertEqual(frame[4:7], bytes.fromhex("db6db6"))  # red 255
        self.assertEqual(len(frame), 1 + 9 + 100)


class EyeFrameTests(unittest.TestCase):
    def test_idle_lights_only_the_pupils_dimly(self) -> None:
        frame = eye_frame(REPOSO, 0.0, 0.1)
        self.assertEqual(len(frame), 2 * LEDS_PER_EYE)
        self.assertEqual(frame[0], frame[LEDS_PER_EYE])
        self.assertGreater(frame[0][0], 0)
        self.assertLess(frame[0][0], round(255 * 0.1))
        self.assertTrue(all(color == (0, 0, 0) for i, color in enumerate(frame) if i % 7))

    def test_center_index_moves_the_pupil(self) -> None:
        frame = eye_frame(REPOSO, 0.0, 0.1, center_index=3, eyes=1)
        self.assertEqual([i for i, color in enumerate(frame) if color != (0, 0, 0)], [3])

    def test_all_states_are_red_and_never_exceed_the_limit(self) -> None:
        for state in (REPOSO, ESCUCHANDO, PENSANDO, "hablando", ERROR, APAGADO):
            for elapsed in (0.0, 0.3, 0.8, 1.2):
                for color in eye_frame(state, elapsed, 1.0, level=1.0):
                    self.assertEqual(color[1:], (0, 0))
                    self.assertLessEqual(color[0], round(255 * MAX_BRIGHTNESS))

    def test_thinking_ring_breathes(self) -> None:
        low = eye_frame(PENSANDO, 0.0, 0.1)[1][0]
        high = eye_frame(PENSANDO, 0.8, 0.1)[1][0]
        self.assertLess(low, high)

    def test_speaking_ring_follows_level(self) -> None:
        quiet = eye_frame("hablando", 0.0, 0.2, level=0.0)[1][0]
        loud = eye_frame("hablando", 0.0, 0.2, level=1.0)[1][0]
        self.assertGreater(quiet, 0)
        self.assertGreater(loud, quiet * 3)

    def test_error_blinks_twice_then_goes_dark(self) -> None:
        self.assertGreater(eye_frame(ERROR, 0.05, 0.1)[1][0], 0)
        self.assertEqual(eye_frame(ERROR, 0.2, 0.1)[1][0], 0)
        self.assertGreater(eye_frame(ERROR, 0.35, 0.1)[1][0], 0)
        self.assertEqual(eye_frame(ERROR, ERROR_SECONDS + 0.01, 0.1)[1][0], 0)

    def test_off_is_dark(self) -> None:
        self.assertTrue(all(color == (0, 0, 0) for color in eye_frame(APAGADO, 0.0, 0.1)))


class Ws2812LightsTests(unittest.TestCase):
    def test_eyes_stay_dark_until_vader_is_ready(self) -> None:
        # The dim pupils mean "ready": they used to light up while Vosk loaded
        # and the network was still missing (11.5 s early on the Pi, 20/11).
        clock = FakeClock()
        lights = _lights(clock)
        try:
            self.assertTrue(all(color == (0, 0, 0) for color in lights.frame()))
            lights.set_state(REPOSO)  # what the session does once it is listening
            self.assertEqual(lights.frame(), eye_frame(REPOSO, 0.0, 0.1))
        finally:
            lights.close()

    def test_speech_overlay_returns_to_previous_state(self) -> None:
        clock = FakeClock()
        lights = _lights(clock)
        try:
            lights.set_state(PENSANDO)
            lights.speak([1.0, 0.0], window_s=0.5)
            self.assertEqual(lights.frame(), eye_frame("hablando", 0, 0.1, level=1.0))
            clock.now += 0.6
            self.assertEqual(lights.frame(), eye_frame("hablando", 0, 0.1, level=0.0))
            clock.now += 0.6
            self.assertEqual(lights.frame(), eye_frame(PENSANDO, 1.2, 0.1))
        finally:
            lights.close()

    def test_error_returns_to_idle(self) -> None:
        clock = FakeClock()
        lights = _lights(clock)
        try:
            lights.set_state(ERROR)
            clock.now += ERROR_SECONDS + 0.01
            self.assertEqual(lights.frame(), eye_frame(REPOSO, 0.0, 0.1))
        finally:
            lights.close()

    def test_set_state_cancels_speech(self) -> None:
        clock = FakeClock()
        lights = _lights(clock)
        try:
            lights.speak([1.0] * 100)
            lights.set_state(ESCUCHANDO)
            self.assertEqual(lights.frame(), eye_frame(ESCUCHANDO, 0.0, 0.1))
        finally:
            lights.close()

    def test_rejects_unknown_state(self) -> None:
        lights = _lights(FakeClock())
        try:
            with self.assertRaises(ValueError):
                lights.set_state("bailando")
        finally:
            lights.close()

    def test_brightness_is_capped(self) -> None:
        lights = Ws2812Lights(FakeDevice(), brightness=0.9, interval_s=3600)
        try:
            self.assertEqual(lights.brightness, MAX_BRIGHTNESS)
        finally:
            lights.close()

    def test_thread_writes_frames_and_close_turns_off(self) -> None:
        device = FakeDevice()
        written = threading.Event()
        original = device.write

        def write(data: bytes) -> None:
            original(data)
            written.set()

        device.write = write  # type: ignore[method-assign]
        lights = Ws2812Lights(device, brightness=0.1, interval_s=0.01)
        self.assertTrue(written.wait(1.0))
        lights.close()
        self.assertTrue(device.closed)
        self.assertEqual(device.writes[-1], encode([(0, 0, 0)] * 14))

    def test_still_frames_are_sent_once(self) -> None:
        device = FakeDevice()
        lights = _lights(FakeClock(), device)
        try:
            lights._show(lights.frame())
            lights._show(lights.frame())
            self.assertEqual(len([w for w in device.writes if w == encode(lights.frame())]), 1)
        finally:
            lights.close()


class EnvelopeTests(unittest.TestCase):
    def test_envelope_rises_with_the_voice(self) -> None:
        import math

        rate = 16000
        quiet = [0] * (rate // 5)
        loud = [int(12000 * math.sin(2 * math.pi * 200 * i / rate)) for i in range(rate // 5)]
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "voz.wav"
            with wave.open(str(path), "wb") as target:
                target.setnchannels(1)
                target.setsampwidth(2)
                target.setframerate(rate)
                import struct

                target.writeframes(struct.pack(f"<{len(quiet) + len(loud)}h", *(quiet + loud)))
            envelope = speech_envelope(path, window_s=0.04)
        self.assertEqual(len(envelope), 10)
        self.assertLess(max(envelope[:5]), 0.05)
        self.assertGreater(min(envelope[5:]), 0.9)


def _settings(kind: str, device: str = "/dev/spidev0.0") -> SimpleNamespace:
    return SimpleNamespace(
        lights=kind,
        lights_spi_device=device,
        lights_brightness=0.1,
        lights_eyes=2,
        lights_center_index=0,
    )


class FactoryTests(unittest.TestCase):
    def test_console_and_null(self) -> None:
        self.assertIsInstance(build_lights(_settings("none")), NullLights)
        self.assertIsInstance(build_lights(_settings("console")), ConsoleLights)

    def test_missing_spi_device_falls_back_to_no_lights(self) -> None:
        settings = _settings("ws2812", "/no/existe/spidev9.9")
        with self.assertLogs("vadermind.lights", level="WARNING"):
            self.assertIsInstance(build_lights(settings), NullLights)

    def test_console_prints_changes_once(self) -> None:
        lines: list[str] = []
        lights = ConsoleLights(lines.append)
        lights.set_state(REPOSO)
        lights.set_state(REPOSO)
        lights.set_state(PENSANDO)
        self.assertEqual(lines, ["[luces] reposo", "[luces] pensando"])


if __name__ == "__main__":
    unittest.main()
