"""Show each state of Vader's eyes for a few seconds, using the real component.

Run it on the Pi, from ~/VaderMind, with the virtual environment (it does not
need the spidev package):

    .venv/bin/python scripts/lights_demo.py
    .venv/bin/python scripts/lights_demo.py --estado pensando --segundos 10
    .venv/bin/python scripts/lights_demo.py --pupila 3

--pupila N lights only LED N of each eye, to find which one is the center
(set it in VADER_LIGHTS_CENTER_INDEX). On a computer use --consola.
The LEDs are always left off at the end.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vadermind.lights import (  # noqa: E402
    ERROR,
    ESCUCHANDO,
    LEDS_PER_EYE,
    PENSANDO,
    RED,
    REPOSO,
    ConsoleLights,
    SpiDevice,
    Ws2812Lights,
    encode,
    scale,
)

DEMO = (REPOSO, ESCUCHANDO, PENSANDO, "hablando", ERROR)


def fake_voice(seconds: float, window_s: float = 0.04) -> list[float]:
    """A syllable-like envelope (~4 per second) to preview the speaking ring."""

    steps = int(seconds / window_s)
    return [abs(math.sin(math.pi * 4 * index * window_s)) ** 0.7 for index in range(steps)]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--estado", choices=DEMO, help="mostrar solo este estado")
    parser.add_argument("--segundos", type=float, default=4.0, help="duracion de cada estado")
    parser.add_argument("--brillo", type=float, default=0.10)
    parser.add_argument("--centro", type=int, default=0, help="indice del LED pupila (0-6)")
    parser.add_argument("--pupila", type=int, metavar="N", help="encender solo el LED N de cada ojo")
    parser.add_argument("--consola", action="store_true", help="sin LEDs: imprime los estados")
    args = parser.parse_args()

    if args.pupila is not None:
        device = SpiDevice()
        try:
            eye = [scale(RED, args.brillo) if i == args.pupila else (0, 0, 0) for i in range(LEDS_PER_EYE)]
            device.write(encode(eye * 2))
            print(f"LED {args.pupila} de cada ojo encendido {args.segundos:g} s.")
            time.sleep(args.segundos)
            device.write(encode([(0, 0, 0)] * LEDS_PER_EYE * 2))
        finally:
            device.close()
        return 0

    lights = (
        ConsoleLights()
        if args.consola
        else Ws2812Lights(SpiDevice(), brightness=args.brillo, center_index=args.centro)
    )
    try:
        for state in (args.estado,) if args.estado else DEMO:
            print(f"Estado: {state}")
            if state == "hablando":
                lights.set_state(PENSANDO)
                lights.speak(fake_voice(args.segundos))
            else:
                lights.set_state(state)
            time.sleep(args.segundos)
    except KeyboardInterrupt:
        print("Detenido.")
    finally:
        lights.close()
    print("Luces apagadas.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
