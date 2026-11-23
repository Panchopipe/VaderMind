"""Measure the microphone's background noise and mains hum during a few seconds of silence.

Run it from VaderMind/ while the room is quiet and nobody speaks:

    .venv/bin/python scripts/measure_mic_noise.py

Compare setups (wall supply vs power bank, speakers unplugged, another USB
port) by the printed numbers. Mains hum shows up at 50 Hz and its odd
harmonics (150, 250 Hz...) in Chile.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from vadermind.config import load_settings  # noqa: E402
from vadermind.listening import capture_rate, resolve_input_device  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seconds", type=float, default=3.0)
    args = parser.parse_args()

    import sounddevice as sd

    settings = load_settings()
    device, name = resolve_input_device(settings.mic_device)
    rate = capture_rate(sd, device)
    print(f"Microfono: {name} a {rate} Hz. Silencio durante {args.seconds:g} s...")
    audio = sd.rec(int(args.seconds * rate), samplerate=rate, channels=1, dtype="int16", device=device)
    sd.wait()

    samples = audio[:, 0].astype(np.float64) / 32768.0
    samples = samples[int(0.2 * rate) :]  # skip the stream start-up
    rms_db = 10 * np.log10(max(float(np.mean(samples**2)), 1e-12))
    power = np.abs(np.fft.rfft(samples * np.hanning(len(samples)))) ** 2
    freqs = np.fft.rfftfreq(len(samples), 1 / rate)
    audible = power[freqs >= 20].sum()
    low_share = 100 * power[(freqs >= 20) & (freqs < 200)].sum() / audible
    hum_share = 100 * sum(
        power[(freqs > h - 3) & (freqs < h + 3)].sum() for h in (50, 150, 250, 350)
    ) / audible

    print(f"Ruido de fondo: {rms_db:.1f} dBFS")
    print(f"Energia bajo 200 Hz: {low_share:.0f} %")
    print(f"Zumbido electrico (50/150/250/350 Hz): {hum_share:.0f} % del ruido")
    if rms_db < -50:
        print("Veredicto: silencioso. El microfono esta limpio.")
    elif hum_share > 30:
        print("Veredicto: zumbido electrico. Revisar fuente, parlantes o puerto USB.")
    else:
        print("Veredicto: ruido de fondo sin zumbido claro (sala, ventilador o microfono).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
