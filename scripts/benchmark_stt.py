"""Record the fixed STT phrases and compare transcription models on them.

Usage from the project root:

    .\\.venv\\Scripts\\python.exe -m scripts.benchmark_stt record
    .\\.venv\\Scripts\\python.exe -m scripts.benchmark_stt record --only 6 13
    .\\.venv\\Scripts\\python.exe -m scripts.benchmark_stt run
    .\\.venv\\Scripts\\python.exe -m scripts.benchmark_stt run --local

On the Raspberry Pi, keep each setup in its own folder:

    .venv/bin/python -m scripts.benchmark_stt record --folder pi_1m
    .venv/bin/python -m scripts.benchmark_stt run --folder pi_1m

`record` saves one WAV per phrase in evaluations/stt_recordings/[folder]/
(outside Git), using VADER_MIC_DEVICE and the same 16 kHz conversion and
high-pass filter as Vader. `run` sends the recordings that exist to each model
and writes a report in evaluations/results/stt_<fecha>/. `--local` adds
faster-whisper if installed.
"""

from __future__ import annotations

import argparse
from array import array
from dataclasses import asdict
from datetime import datetime
import json
from pathlib import Path
import sys
import time
import wave

from vadermind.config import ConfigurationError, load_dotenv_file, load_settings
from vadermind.listening import capture_rate, frame_converter, resolve_input_device
from vadermind.stt_evaluation import STT_PHRASES, SttRow, summarize_model
from vadermind.voice_input import CHILEAN_STT_PROMPT, TranscriptionError, VercelSpeechToText

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RECORDINGS_ROOT = PROJECT_ROOT / "evaluations" / "stt_recordings"
RECORDINGS = RECORDINGS_ROOT
RESULTS_ROOT = PROJECT_ROOT / "evaluations" / "results"
SAMPLE_RATE = 16_000
DEFAULT_MODELS = [
    "openai/gpt-4o-mini-transcribe",
    "openai/gpt-4o-transcribe",
    "google/gemini-3.5-transcribe",
    "spacexai/grok-stt",
]
LOCAL_MODEL = "local/faster-whisper-base"

SUMMARY_LABELS = {
    "error_palabras_pct": "Error por palabra (%)",
    "frases_perfectas": "Frases perfectas",
    "fallidas": "Llamadas fallidas",
    "latencia_mediana_s": "Latencia mediana (s)",
    "latencia_max_s": "Latencia maxima (s)",
    "costo_total_usd": "Costo total (US$)",
}


def recording_path(number: int) -> Path:
    return RECORDINGS / f"frase_{number:02d}.wav"


def record(only: list[int] | None) -> None:
    import sounddevice as sd

    settings = load_settings(PROJECT_ROOT / ".env")
    device, name = resolve_input_device(settings.mic_device)
    rate = capture_rate(sd, device)
    print(f"Microfono: {name} a {rate} Hz (se guarda a {SAMPLE_RATE} Hz, como lo oye Vader)")
    print("Habla como hablas siempre, a la distancia que quieras medir.")
    print("Enter para empezar a hablar, Enter otra vez para terminar.\n")
    RECORDINGS.mkdir(parents=True, exist_ok=True)

    for phrase in STT_PHRASES:
        path = recording_path(phrase.number)
        if only is not None and phrase.number not in only:
            continue
        if only is None and path.exists():
            print(f"{phrase.number:2d}. ya grabada, se omite (usa --only para repetirla)")
            continue
        while True:
            input(f'{phrase.number:2d}. Di: "{phrase.text}"  [Enter para grabar]')
            chunks: list[bytes] = []
            convert = frame_converter(rate)
            with sd.RawInputStream(
                samplerate=rate,
                channels=1,
                dtype="int16",
                device=device,
                callback=lambda data, *_: chunks.extend(convert(bytes(data))),
            ):
                input("    Grabando... [Enter para terminar]")
            audio = b"".join(chunks)
            samples = array("h", audio)
            seconds = len(samples) / SAMPLE_RATE
            peak = max((abs(sample) for sample in samples), default=0)
            print(f"    {seconds:.1f} s, nivel maximo {100 * peak / 32767:.0f}%")
            if seconds < 0.5 or peak < 1000:
                print("    Muy corta o sin sonido; repitamosla.")
                continue
            with wave.open(str(path), "wb") as handle:
                handle.setnchannels(1)
                handle.setsampwidth(2)
                handle.setframerate(SAMPLE_RATE)
                handle.writeframes(audio)
            break
    print(f"\nGrabaciones en {RECORDINGS}")


def _local_transcriber():
    from faster_whisper import WhisperModel

    model = WhisperModel("base", device="cpu", compute_type="int8")

    def transcribe(path: Path) -> tuple[str, float]:
        started = time.perf_counter()
        segments, _info = model.transcribe(str(path), language="es", beam_size=5)
        text = " ".join(segment.text.strip() for segment in segments)
        return text, time.perf_counter() - started

    return transcribe


def run(models: list[str], include_local: bool, compare_prompt: bool = False) -> Path:
    phrases = [p for p in STT_PHRASES if recording_path(p.number).exists()]
    if not phrases:
        raise ValueError(f"No hay grabaciones en {RECORDINGS}; usa 'record'.")
    missing = [p.number for p in STT_PHRASES if p not in phrases]
    if missing:
        print(f"Sin grabacion, se omiten las frases {missing}.")
    load_dotenv_file(PROJECT_ROOT / ".env")
    settings = load_settings(PROJECT_ROOT / ".env")
    if not settings.ai_gateway_api_key:
        raise ConfigurationError("Falta AI_GATEWAY_API_KEY.")

    rows: list[SttRow] = []
    # --compare-prompt runs every model twice: plain and with the Chilean context.
    variants = [(model, "") for model in models]
    if compare_prompt:
        variants += [(model, CHILEAN_STT_PROMPT) for model in models]
    models = [model + (" +contexto" if prompt else "") for model, prompt in variants]
    for (model_id, prompt), model in zip(variants, models):
        with VercelSpeechToText(
            settings.ai_gateway_api_key,
            model=model_id,
            prompt=prompt,
            timeout_seconds=settings.ai_gateway_timeout_seconds,
        ) as stt:
            for phrase in phrases:
                try:
                    result = stt.transcribe_detailed(recording_path(phrase.number))
                    row = SttRow(model, phrase.number, phrase.text, result.text,
                                 round(result.seconds, 3), result.cost_usd)
                except (TranscriptionError, OSError) as error:
                    row = SttRow(model, phrase.number, phrase.text, None, None, None,
                                 str(error))
                rows.append(row)
                print(f"[{model}] {phrase.number:2d} {row.hypothesis or 'ERROR: ' + str(row.error)}")

    if include_local:
        try:
            local = _local_transcriber()
        except ImportError:
            print("faster-whisper no esta instalado; se omite el modelo local.")
        else:
            models = [*models, LOCAL_MODEL]
            for phrase in phrases:
                text, seconds = local(recording_path(phrase.number))
                rows.append(SttRow(LOCAL_MODEL, phrase.number, phrase.text, text,
                                   round(seconds, 3), 0.0))
                print(f"[{LOCAL_MODEL}] {phrase.number:2d} {text}")

    summaries = {
        model: summarize_model([row for row in rows if row.model == model])
        for model in models
    }
    output = RESULTS_ROOT / f"stt_{datetime.now():%Y%m%d_%H%M%S}"
    output.mkdir(parents=True)
    with (output / "transcripciones.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(asdict(row), ensure_ascii=False) + "\n")
    (output / "resumen.json").write_text(
        json.dumps(summaries, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    report = _report(rows, summaries, phrases)
    (output / "resumen.md").write_text(report, encoding="utf-8")
    print("\n" + report)
    return output


def _report(rows: list[SttRow], summaries: dict[str, dict[str, float]], phrases: list) -> str:
    names = list(summaries)
    lines = [
        "# Benchmark de transcripcion", "",
        "| Medida | " + " | ".join(names) + " |",
        "| --- |" + " ---: |" * len(names),
    ]
    for key, title in SUMMARY_LABELS.items():
        values = [f"{summaries[name].get(key, '-'):g}"
                  if isinstance(summaries[name].get(key), (int, float)) else "-"
                  for name in names]
        lines.append(f"| {title} | " + " | ".join(values) + " |")
    lines += ["", "## Transcripciones", ""]
    for phrase in phrases:
        lines.append(f"**{phrase.number}. {phrase.text}**")
        for row in rows:
            if row.phrase_number == phrase.number:
                lines.append(f"- {row.model}: {row.hypothesis if row.hypothesis is not None else 'ERROR: ' + str(row.error)}")
        lines.append("")
    return "\n".join(lines)


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    record_parser = commands.add_parser("record", help="Grabar las frases fijas.")
    record_parser.add_argument("--only", nargs="+", type=int, help="Frases a regrabar.")
    run_parser = commands.add_parser("run", help="Comparar modelos de transcripcion.")
    run_parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    run_parser.add_argument("--local", action="store_true", help="Agregar faster-whisper base.")
    run_parser.add_argument(
        "--compare-prompt", action="store_true",
        help="Repetir cada modelo con el contexto chileno (VADER_STT_PROMPT=default).",
    )
    for sub in (record_parser, run_parser):
        sub.add_argument(
            "--folder", default="", help="Subcarpeta de grabaciones, por ejemplo pi_1m."
        )
    args = parser.parse_args()
    global RECORDINGS
    RECORDINGS = RECORDINGS_ROOT / args.folder if args.folder else RECORDINGS_ROOT
    try:
        if args.command == "record":
            record(args.only)
        else:
            output = run(args.models, args.local, args.compare_prompt)
            print(f"\nResultados en {output}")
    except (ConfigurationError, OSError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\nInterrumpido.")
        return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
