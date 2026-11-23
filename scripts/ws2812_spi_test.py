"""Light WS2812 LEDs with a fixed color through the Raspberry Pi SPI bus.

Hardware test for the helmet lights (stage 3). Data goes out on GPIO10
(SPI0 MOSI, physical pin 19) through a 330 ohm resistor. Run it on the Pi:

    python3 scripts/ws2812_spi_test.py --color rojo
    python3 scripts/ws2812_spi_test.py --off
    python3 scripts/ws2812_spi_test.py --leds 14 --animar 60

With --animar the color "breathes" (rises and falls slowly) for that many
seconds and then turns off; Ctrl+C also turns it off. It is meant for
measuring power and microphone noise while the lights keep changing.

How it works: the WS2812 reads short and long pulses, not real SPI. At
2.4 MHz every SPI bit lasts ~417 ns, so each LED bit is sent as three SPI
bits: 0 -> 100 (short high pulse), 1 -> 110 (long high pulse). Colors go
in green-red-blue order. A run of zeros at the end (low line for >280 us)
tells the LEDs to show the new color, which they keep after the script ends.

Needs `dtparam=spi=on` plus `core_freq=500` and `core_freq_min=500` in
/boot/firmware/config.txt, so the SPI clock does not change with load.
"""

from __future__ import annotations

import argparse
import math
import sys
import time

SPI_BUS = 0
SPI_DEVICE = 0
SPI_SPEED_HZ = 2_400_000
RESET_BYTES = 100  # 100 bytes x 8 bits / 2.4 MHz = ~333 us of low line
MAX_BRIGHTNESS = 0.25
DEFAULT_BRIGHTNESS = 0.10
BREATH_PERIOD_S = 3.0
BREATH_FLOOR = 0.10  # lowest point of the pulse, as a fraction of the brightness
FRAME_INTERVAL_S = 0.02  # ~50 frames per second

COLORS = {
    "rojo": (255, 0, 0),
    "verde": (0, 255, 0),
    "azul": (0, 0, 255),
    "blanco": (255, 255, 255),
    "apagado": (0, 0, 0),
}


def _encode_byte(value: int) -> bytes:
    """Turn one color byte into three SPI bytes (three SPI bits per LED bit)."""
    bits = 0
    for position in range(7, -1, -1):
        bits = (bits << 3) | (0b110 if (value >> position) & 1 else 0b100)
    return bits.to_bytes(3, "big")


_ENCODED = [_encode_byte(value) for value in range(256)]


def scale(color: tuple[int, int, int], brightness: float) -> tuple[int, int, int]:
    """Apply the brightness limit to an RGB color."""
    level = min(max(brightness, 0.0), MAX_BRIGHTNESS)
    return tuple(round(channel * level) for channel in color)  # type: ignore[return-value]


def encode(colors: list[tuple[int, int, int]]) -> bytes:
    """Build the full SPI frame for a chain of LEDs given as RGB tuples."""
    frame = bytearray(b"\x00")  # start from a low line
    for red, green, blue in colors:
        for channel in (green, red, blue):
            frame += _ENCODED[channel]
    frame += bytes(RESET_BYTES)
    return bytes(frame)


def breath_level(elapsed: float, period: float = BREATH_PERIOD_S) -> float:
    """Pulse factor between BREATH_FLOOR and 1.0; starts and ends each period low."""
    wave = (1 - math.cos(2 * math.pi * elapsed / period)) / 2
    return BREATH_FLOOR + (1 - BREATH_FLOOR) * wave


def parse_color(text: str) -> tuple[int, int, int]:
    """Accept a color name or 'r,g,b' with values from 0 to 255."""
    name = text.strip().lower()
    if name in COLORS:
        return COLORS[name]
    parts = name.split(",")
    if len(parts) == 3 and all(part.strip().isdigit() for part in parts):
        values = tuple(int(part) for part in parts)
        if all(0 <= value <= 255 for value in values):
            return values  # type: ignore[return-value]
    raise argparse.ArgumentTypeError(
        f"color no valido: {text!r}; usa {', '.join(COLORS)} o r,g,b (0-255)"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--color", type=parse_color, default=COLORS["rojo"])
    parser.add_argument("--brillo", type=float, default=DEFAULT_BRIGHTNESS,
                        help=f"0 a {MAX_BRIGHTNESS} (por defecto {DEFAULT_BRIGHTNESS})")
    parser.add_argument("--leds", type=int, default=1, help="cantidad de LEDs en la cadena")
    parser.add_argument("--off", action="store_true", help="apagar los LEDs")
    parser.add_argument("--animar", type=float, default=0, metavar="SEGUNDOS",
                        help="respiracion lenta durante esos segundos; al final se apagan")
    args = parser.parse_args()

    if args.brillo > MAX_BRIGHTNESS:
        print(f"Brillo limitado a {MAX_BRIGHTNESS} para cuidar la energia de la Pi.")
    color = (0, 0, 0) if args.off else scale(args.color, args.brillo)

    import spidev

    spi = spidev.SpiDev()
    spi.open(SPI_BUS, SPI_DEVICE)
    try:
        spi.max_speed_hz = SPI_SPEED_HZ
        spi.mode = 0
        if args.animar > 0 and not args.off:
            _animate(spi, args.color, args.brillo, args.leds, args.animar)
            return 0
        spi.writebytes2(encode([color] * args.leds))
    finally:
        spi.close()
    print(f"Enviado {color} (rojo, verde, azul) a {args.leds} LED(s).")
    return 0


def _animate(spi, base: tuple[int, int, int], brightness: float, leds: int, seconds: float) -> None:
    """Breathe the color for `seconds`, then always leave the LEDs off."""
    print(f"Animando {leds} LED(s) durante {seconds:g} s (Ctrl+C para detener)...")
    start = time.monotonic()
    try:
        while (elapsed := time.monotonic() - start) < seconds:
            color = scale(base, brightness * breath_level(elapsed))
            spi.writebytes2(encode([color] * leds))
            time.sleep(FRAME_INTERVAL_S)
    except KeyboardInterrupt:
        print("Detenido.")
    finally:
        spi.writebytes2(encode([(0, 0, 0)] * leds))
    print("Animacion terminada; LEDs apagados.")


if __name__ == "__main__":
    sys.exit(main())
