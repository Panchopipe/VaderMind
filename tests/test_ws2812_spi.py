"""Tests for the WS2812 SPI encoding used by the helmet light test."""

from __future__ import annotations

import argparse
import unittest

from scripts.ws2812_spi_test import (
    MAX_BRIGHTNESS,
    BREATH_FLOOR,
    BREATH_PERIOD_S,
    RESET_BYTES,
    _encode_byte,
    breath_level,
    encode,
    parse_color,
    scale,
)


class EncodeByteTests(unittest.TestCase):
    def test_zero_is_eight_short_pulses(self) -> None:
        # 100 repeated eight times: 100100100100100100100100
        self.assertEqual(_encode_byte(0x00), bytes([0b10010010, 0b01001001, 0b00100100]))

    def test_full_is_eight_long_pulses(self) -> None:
        # 110 repeated eight times: 110110110110110110110110
        self.assertEqual(_encode_byte(0xFF), bytes([0b11011011, 0b01101101, 0b10110110]))

    def test_most_significant_bit_goes_first(self) -> None:
        self.assertEqual(_encode_byte(0x80)[0] >> 5, 0b110)
        self.assertEqual(_encode_byte(0x80)[2] & 0b111, 0b100)


class FrameTests(unittest.TestCase):
    def test_frame_layout_for_one_led(self) -> None:
        frame = encode([(10, 20, 30)])
        self.assertEqual(len(frame), 1 + 9 + RESET_BYTES)
        self.assertEqual(frame[0], 0)
        self.assertEqual(frame[-RESET_BYTES:], bytes(RESET_BYTES))

    def test_channels_go_green_red_blue(self) -> None:
        frame = encode([(0xFF, 0x00, 0x00)])
        self.assertEqual(frame[1:4], _encode_byte(0x00))  # green
        self.assertEqual(frame[4:7], _encode_byte(0xFF))  # red
        self.assertEqual(frame[7:10], _encode_byte(0x00))  # blue


class BrightnessAndColorTests(unittest.TestCase):
    def test_brightness_is_capped(self) -> None:
        self.assertEqual(scale((255, 255, 255), 1.0), scale((255, 255, 255), MAX_BRIGHTNESS))
        self.assertEqual(scale((255, 0, 0), 0.1), (26, 0, 0))

    def test_parse_color_names_and_triplets(self) -> None:
        self.assertEqual(parse_color("Rojo"), (255, 0, 0))
        self.assertEqual(parse_color("10, 20, 30"), (10, 20, 30))
        with self.assertRaises(argparse.ArgumentTypeError):
            parse_color("300,0,0")
        with self.assertRaises(argparse.ArgumentTypeError):
            parse_color("magenta")


class BreathTests(unittest.TestCase):
    def test_pulse_starts_low_and_peaks_mid_period(self) -> None:
        self.assertAlmostEqual(breath_level(0.0), BREATH_FLOOR)
        self.assertAlmostEqual(breath_level(BREATH_PERIOD_S / 2), 1.0)
        self.assertAlmostEqual(breath_level(BREATH_PERIOD_S), BREATH_FLOOR)

    def test_pulse_never_exceeds_requested_brightness(self) -> None:
        for step in range(100):
            level = breath_level(step * 0.07)
            self.assertGreaterEqual(level, BREATH_FLOOR - 1e-9)
            self.assertLessEqual(level, 1.0 + 1e-9)
        self.assertEqual(scale((255, 0, 0), MAX_BRIGHTNESS * breath_level(1.5)), (64, 0, 0))


if __name__ == "__main__":
    unittest.main()
