#!/usr/bin/env python3
"""Build the uploadable RunPod ZIP without duplicating the approved dataset."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import uuid
import zipfile
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", required=True, type=Path)
    parser.add_argument("--training-files", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    dataset = args.dataset_dir.resolve()
    training_files = args.training_files.resolve()
    output = args.output.resolve()
    root_name = output.stem
    if root_name.endswith(".tar"):
        root_name = root_name[:-4]

    temp = output.parent / ".package_build" / uuid.uuid4().hex
    try:
        stage = temp / root_name
        shutil.copytree(training_files, stage)
        staged_dataset = stage / "dataset"
        staged_dataset.mkdir()
        shutil.copytree(dataset / "wav", staged_dataset / "wav")
        for name in ("manifest.csv", "rejected.csv", "dataset_summary.json"):
            shutil.copy2(dataset / name, staged_dataset / name)

        metadata_rows: list[list[str]] = []
        with (dataset / "metadata.csv").open(
            "r", encoding="utf-8-sig", newline=""
        ) as source:
            for row in csv.reader(source, delimiter="|"):
                if len(row) != 2:
                    raise ValueError(f"Fila de metadata inválida: {row}")
                metadata_rows.append([Path(row[0]).stem, row[1]])
        with (staged_dataset / "metadata.csv").open(
            "w", encoding="utf-8", newline=""
        ) as target:
            csv.writer(target, delimiter="|", lineterminator="\n").writerows(
                metadata_rows
            )

        package_info = {
            "package": root_name,
            "clips": len(metadata_rows),
            "dataset_source": dataset.name,
            "metadata_format": "id_without_extension|text",
            "target": "RunPod RTX 3090 / Piper 1.0 fine-tuning",
        }
        (stage / "PACKAGE.json").write_text(
            json.dumps(package_info, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

        output.parent.mkdir(parents=True, exist_ok=True)
        if output.exists():
            output.unlink()
        with zipfile.ZipFile(
            output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6
        ) as archive:
            for path in sorted(stage.rglob("*")):
                if not path.is_file():
                    continue
                relative = path.relative_to(stage.parent).as_posix()
                info = zipfile.ZipInfo.from_file(path, relative)
                if "/scripts/" in f"/{relative}" and path.suffix in {".sh", ".py"}:
                    info.external_attr = (0o100755 << 16)
                with path.open("rb") as handle:
                    archive.writestr(
                        info,
                        handle.read(),
                        compress_type=zipfile.ZIP_DEFLATED,
                        compresslevel=6,
                    )
    finally:
        shutil.rmtree(temp, ignore_errors=True)

    checksum = sha256(output)
    output.with_suffix(output.suffix + ".sha256").write_text(
        f"{checksum}  {output.name}\n", encoding="ascii"
    )
    print(
        json.dumps(
            {
                "output": str(output),
                "size_bytes": output.stat().st_size,
                "sha256": checksum,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
