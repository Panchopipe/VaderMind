"""Benchmark the production-like Piper and ElevenLabs TTS latency paths."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import platform
import statistics
import subprocess
import time
import wave

import httpx

from vadermind.config import load_settings


DEFAULT_TEXT = (
    "Los sistemas estan operativos, Francisco. La red imperial permanece bajo "
    "vigilancia. Si detecto una anomalia, te informare de inmediato y esperare "
    "nuevas instrucciones."
)


def _summary(values: list[float]) -> dict[str, float]:
    """Return compact latency statistics rounded to milliseconds."""

    return {
        "median_seconds": round(statistics.median(values), 3),
        "mean_seconds": round(statistics.mean(values), 3),
        "min_seconds": round(min(values), 3),
        "max_seconds": round(max(values), 3),
    }


def _piper_once(settings, text: str, output_path: Path) -> dict[str, float | int]:
    """Measure the current Piper path: process start through complete WAV."""

    command = [
        settings.piper_executable,
        "--model",
        settings.piper_model_path,
        "--config",
        settings.piper_config_path,
        "--output_file",
        str(output_path),
        "--length_scale",
        str(settings.piper_length_scale),
        "--noise_scale",
        str(settings.piper_noise_scale),
        "--noise_w",
        str(settings.piper_noise_w),
        "--sentence_silence",
        str(settings.piper_sentence_silence),
        "--quiet",
    ]

    started_at = time.perf_counter()
    result = subprocess.run(
        command,
        input=f"{text}\n",
        text=True,
        encoding="utf-8",
        capture_output=True,
        check=False,
        timeout=settings.piper_timeout_seconds,
    )
    completed_at = time.perf_counter()
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip()
        raise RuntimeError(f"Piper failed with code {result.returncode}: {detail}")

    with wave.open(str(output_path), "rb") as wav_file:
        audio_seconds = wav_file.getnframes() / wav_file.getframerate()

    latency = completed_at - started_at
    return {
        "time_to_audio_seconds": latency,
        "complete_seconds": latency,
        "audio_seconds": audio_seconds,
        "bytes": output_path.stat().st_size,
        "real_time_factor": latency / audio_seconds,
    }


def _elevenlabs_once(settings, text: str) -> dict[str, float | int]:
    """Measure ElevenLabs streaming from request start to first and final PCM byte."""

    started_at = time.perf_counter()
    first_audio_at: float | None = None
    total_bytes = 0
    with httpx.stream(
        "POST",
        f"https://api.elevenlabs.io/v1/text-to-speech/{settings.elevenlabs_voice_id}/stream",
        params={"output_format": settings.elevenlabs_output_format},
        headers={
            "xi-api-key": settings.elevenlabs_api_key,
            "Content-Type": "application/json",
        },
        json={
            "text": text,
            "model_id": settings.elevenlabs_model_id,
            "voice_settings": {
                "stability": settings.elevenlabs_stability,
                "similarity_boost": settings.elevenlabs_similarity_boost,
                "style": settings.elevenlabs_style,
                "use_speaker_boost": settings.elevenlabs_use_speaker_boost,
            },
        },
        timeout=60,
    ) as response:
        response.raise_for_status()
        for chunk in response.iter_bytes():
            if not chunk:
                continue
            if first_audio_at is None:
                first_audio_at = time.perf_counter()
            total_bytes += len(chunk)
    completed_at = time.perf_counter()

    if first_audio_at is None or total_bytes == 0:
        raise RuntimeError("ElevenLabs returned no audio.")
    if not settings.elevenlabs_output_format.startswith("pcm_"):
        raise RuntimeError("This benchmark expects an ElevenLabs pcm_* format.")

    sample_rate = int(settings.elevenlabs_output_format.split("_", 1)[1])
    audio_seconds = total_bytes / (sample_rate * 2)
    return {
        "time_to_audio_seconds": first_audio_at - started_at,
        "complete_seconds": completed_at - started_at,
        "audio_seconds": audio_seconds,
        "bytes": total_bytes,
        "real_time_factor": (completed_at - started_at) / audio_seconds,
    }


def main() -> int:
    """Run an alternating multi-trial benchmark and save non-sensitive JSON."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--text", default=DEFAULT_TEXT)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("audio_tests/tts_latency_comparison/benchmark_2026-09-03.json"),
    )
    args = parser.parse_args()
    if args.iterations < 3:
        parser.error("Use at least 3 iterations.")

    settings = load_settings()
    if not settings.elevenlabs_api_key:
        raise RuntimeError("ELEVENLABS_API_KEY is required for this benchmark.")

    scratch = Path(".runtime_cache/tts_latency_benchmark")
    scratch.mkdir(parents=True, exist_ok=True)

    # Warm only the local OS/model-file cache. Production Piper still launches a
    # fresh process for every measured request.
    warmup_path = scratch / "piper_warmup.wav"
    _piper_once(settings, "Prueba de inicio.", warmup_path)
    warmup_path.unlink(missing_ok=True)

    piper_results: list[dict[str, float | int]] = []
    elevenlabs_results: list[dict[str, float | int]] = []
    for index in range(args.iterations):
        piper_path = scratch / f"piper_{index}.wav"
        if index % 2 == 0:
            piper_results.append(_piper_once(settings, args.text, piper_path))
            elevenlabs_results.append(_elevenlabs_once(settings, args.text))
        else:
            elevenlabs_results.append(_elevenlabs_once(settings, args.text))
            piper_results.append(_piper_once(settings, args.text, piper_path))
        piper_path.unlink(missing_ok=True)
        print(f"Trial {index + 1}/{args.iterations} complete", flush=True)

    report = {
        "benchmark_date": "2026-09-03",
        "scope": "TTS only; Claude generation and audio playback excluded",
        "system": {
            "platform": platform.platform(),
            "processor": platform.processor(),
        },
        "text": args.text,
        "text_characters": len(args.text),
        "iterations": args.iterations,
        "providers": {
            "piper": {
                "mode": "fresh process, complete WAV before playback",
                "model": Path(settings.piper_model_path).name,
                "time_to_audio": _summary(
                    [float(item["time_to_audio_seconds"]) for item in piper_results]
                ),
                "complete": _summary(
                    [float(item["complete_seconds"]) for item in piper_results]
                ),
                "audio_duration": _summary(
                    [float(item["audio_seconds"]) for item in piper_results]
                ),
                "raw_trials": piper_results,
            },
            "elevenlabs": {
                "mode": "streaming PCM, new HTTP connection per request",
                "model": settings.elevenlabs_model_id,
                "output_format": settings.elevenlabs_output_format,
                "time_to_audio": _summary(
                    [float(item["time_to_audio_seconds"]) for item in elevenlabs_results]
                ),
                "complete": _summary(
                    [float(item["complete_seconds"]) for item in elevenlabs_results]
                ),
                "audio_duration": _summary(
                    [float(item["audio_seconds"]) for item in elevenlabs_results]
                ),
                "raw_trials": elevenlabs_results,
            },
        },
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
