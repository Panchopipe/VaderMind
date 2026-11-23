#!/usr/bin/env python3
"""Fail-fast validation for the packaged legacy Piper 1.0 dataset."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import wave
from pathlib import Path


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", required=True, type=Path)
    args = parser.parse_args()
    root = args.dataset_dir.resolve()
    metadata_path = root / "metadata.csv"
    manifest_path = root / "manifest.csv"
    wav_dir = root / "wav"

    errors: list[str] = []
    metadata: dict[str, str] = {}
    with metadata_path.open("r", encoding="utf-8-sig", newline="") as handle:
        for line_number, row in enumerate(csv.reader(handle, delimiter="|"), 1):
            if len(row) != 2:
                errors.append(f"metadata.csv:{line_number} no tiene 2 columnas")
                continue
            clip_id, text = (value.strip() for value in row)
            if clip_id.lower().endswith(".wav"):
                errors.append(f"{clip_id}: Piper 1.0 espera el identificador sin .wav")
            if clip_id in metadata:
                errors.append(f"Identificador duplicado: {clip_id}")
            if not text:
                errors.append(f"Transcripción vacía: {clip_id}")
            metadata[clip_id] = text

    expected_hashes: dict[str, str] = {}
    with manifest_path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="|"):
            expected_hashes[Path(row["filename"]).stem] = row["dataset_sha256"]

    wav_paths = sorted(wav_dir.glob("*.wav"))
    durations: list[float] = []
    for clip_id in sorted(metadata):
        wav_path = wav_dir / f"{clip_id}.wav"
        if not wav_path.is_file():
            errors.append(f"Falta WAV: {wav_path.name}")
            continue
        with wave.open(str(wav_path), "rb") as wav:
            rate = wav.getframerate()
            channels = wav.getnchannels()
            bits = wav.getsampwidth() * 8
            duration = wav.getnframes() / rate
        durations.append(duration)
        if (rate, channels, bits) != (22050, 1, 16):
            errors.append(
                f"Formato inválido {wav_path.name}: {rate} Hz, {channels} canal(es), {bits} bits"
            )
        if not 1.0 <= duration <= 15.0:
            errors.append(f"Duración fuera de rango {wav_path.name}: {duration:.3f}s")
        expected = expected_hashes.get(clip_id)
        if not expected:
            errors.append(f"Falta hash en manifest.csv: {clip_id}")
        elif file_sha256(wav_path) != expected:
            errors.append(f"Hash incorrecto: {wav_path.name}")

    wav_ids = {path.stem for path in wav_paths}
    orphan_ids = sorted(wav_ids - set(metadata))
    if orphan_ids:
        errors.append(f"WAV sin transcripción: {', '.join(orphan_ids)}")

    summary = {
        "metadata_rows": len(metadata),
        "wav_files": len(wav_paths),
        "duration_minutes": round(sum(durations) / 60, 3),
        "min_seconds": round(min(durations), 3) if durations else None,
        "max_seconds": round(max(durations), 3) if durations else None,
        "errors": errors,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
