#!/usr/bin/env python3
"""Assemble the reviewed Vader WAV batches into a Piper training dataset."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import subprocess
import sys
import wave
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Clip:
    filename: str
    text: str
    batch: str
    source: Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def wav_info(path: Path) -> dict[str, int | float]:
    with wave.open(str(path), "rb") as wav:
        frames = wav.getnframes()
        rate = wav.getframerate()
        return {
            "sample_rate": rate,
            "channels": wav.getnchannels(),
            "sample_width_bits": wav.getsampwidth() * 8,
            "frames": frames,
            "duration_seconds": frames / rate,
        }


def load_batch(batch_dir: Path) -> tuple[list[Clip], list[dict[str, str]]]:
    metadata = batch_dir / "metadata_atempo085.csv"
    audio_dir = batch_dir / "atempo085_24k"
    if not metadata.is_file():
        raise FileNotFoundError(f"No existe {metadata}")
    if not audio_dir.is_dir():
        raise FileNotFoundError(f"No existe {audio_dir}")

    clips: list[Clip] = []
    rejected: list[dict[str, str]] = []
    with metadata.open("r", encoding="utf-8-sig", newline="") as handle:
        for line_number, row in enumerate(csv.reader(handle, delimiter="|"), start=1):
            if len(row) != 2:
                raise ValueError(
                    f"Fila inválida en {metadata}:{line_number}; se esperaban 2 columnas"
                )
            filename, text = (value.strip() for value in row)
            source = audio_dir / filename
            if not source.is_file():
                rejected.append(
                    {
                        "filename": filename,
                        "text": text,
                        "batch": batch_dir.name,
                        "reason": "missing_from_reviewed_atempo085_directory",
                    }
                )
                continue
            clips.append(Clip(filename, text, batch_dir.name, source))
    return clips, rejected


def write_pipe_csv(path: Path, rows: list[list[str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="|", lineterminator="\n")
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-dir", action="append", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--sample-rate", type=int, default=22050)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("FFmpeg no está disponible en PATH")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise RuntimeError(f"La carpeta de salida no está vacía: {args.output_dir}")

    all_clips: list[Clip] = []
    rejected: list[dict[str, str]] = []
    for batch_dir in args.batch_dir:
        clips, batch_rejected = load_batch(batch_dir.resolve())
        all_clips.extend(clips)
        rejected.extend(batch_rejected)

    filenames = [clip.filename for clip in all_clips]
    duplicates = sorted({name for name in filenames if filenames.count(name) > 1})
    if duplicates:
        raise ValueError(f"Nombres duplicados: {', '.join(duplicates)}")
    all_clips.sort(key=lambda clip: clip.filename)

    wav_dir = args.output_dir / "wav"
    wav_dir.mkdir(parents=True, exist_ok=True)
    manifest_rows: list[list[str]] = [[
        "filename",
        "text",
        "source_batch",
        "source_path",
        "duration_seconds",
        "sample_rate",
        "channels",
        "sample_width_bits",
        "source_sha256",
        "dataset_sha256",
    ]]
    metadata_rows: list[list[str]] = []
    total_duration = 0.0

    for index, clip in enumerate(all_clips, start=1):
        destination = wav_dir / clip.filename
        command = [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-nostdin",
            "-i",
            str(clip.source),
            "-ar",
            str(args.sample_rate),
            "-ac",
            "1",
            "-c:a",
            "pcm_s16le",
            "-y",
            str(destination),
        ]
        subprocess.run(command, check=True)
        info = wav_info(destination)
        if (
            info["sample_rate"] != args.sample_rate
            or info["channels"] != 1
            or info["sample_width_bits"] != 16
        ):
            raise RuntimeError(f"Formato inesperado después de convertir {clip.filename}")
        duration = float(info["duration_seconds"])
        total_duration += duration
        metadata_rows.append([clip.filename, clip.text])
        manifest_rows.append([
            clip.filename,
            clip.text,
            clip.batch,
            str(clip.source),
            f"{duration:.6f}",
            str(info["sample_rate"]),
            str(info["channels"]),
            str(info["sample_width_bits"]),
            sha256(clip.source),
            sha256(destination),
        ])
        if index % 25 == 0 or index == len(all_clips):
            print(f"Convertidos {index}/{len(all_clips)}", flush=True)

    write_pipe_csv(args.output_dir / "metadata.csv", metadata_rows)
    write_pipe_csv(args.output_dir / "manifest.csv", manifest_rows)
    write_pipe_csv(
        args.output_dir / "rejected.csv",
        [["filename", "text", "source_batch", "reason"]]
        + [
            [item["filename"], item["text"], item["batch"], item["reason"]]
            for item in rejected
        ],
    )
    summary = {
        "dataset_name": args.output_dir.name,
        "approved_clips": len(all_clips),
        "excluded_clips": len(rejected),
        "total_duration_seconds": round(total_duration, 6),
        "total_duration_minutes": round(total_duration / 60, 3),
        "audio_format": {
            "sample_rate": args.sample_rate,
            "channels": 1,
            "encoding": "PCM signed 16-bit little-endian",
        },
        "metadata_format": "filename.wav|text",
        "source_variant": "atempo=0.85 reviewed WAVs",
        "source_batches": [str(path.resolve()) for path in args.batch_dir],
    }
    (args.output_dir / "dataset_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1)
