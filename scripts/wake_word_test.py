"""Check the "Oye Vader" detector alone: it prints every time it hears the phrase.

Run it from VaderMind/ (laptop or Pi) and talk normally for a while:

    .venv/bin/python scripts/wake_word_test.py          (Pi)
    .venv\\Scripts\\python.exe scripts\\wake_word_test.py  (Windows)

Say "Oye Vader" several times at different distances (hits), then talk or
play music without saying it (false activations should stay at zero).
Uses VADER_MIC_DEVICE and VADER_WAKE_MODEL_PATH from .env. Ctrl+C ends it.
"""

from __future__ import annotations

from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vadermind.config import load_settings  # noqa: E402
from vadermind.listening import ListeningError, resolve_input_device  # noqa: E402
from vadermind.wake_word import build_wake_word, wait_for_wake_word  # noqa: E402


def main() -> int:
    settings = load_settings()
    try:
        device, name = resolve_input_device(settings.mic_device)
        detector = build_wake_word(settings)
    except ListeningError as error:
        print(f"Error: {error}")
        return 1
    print(f"Microfono: {name}. Di \"{settings.wake_phrase}\" (Ctrl+C para terminar).")
    start = time.monotonic()
    hits = 0
    try:
        while wait_for_wake_word(detector, device=device):
            hits += 1
            print(f"[{time.monotonic() - start:7.1f} s] Escuchado #{hits}")
    except KeyboardInterrupt:
        pass
    except ListeningError as error:
        print(f"Error: {error}")
        return 1
    minutes = (time.monotonic() - start) / 60
    print(f"\n{hits} deteccion(es) en {minutes:.1f} min.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
