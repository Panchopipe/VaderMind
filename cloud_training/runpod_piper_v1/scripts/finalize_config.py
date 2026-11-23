#!/usr/bin/env python3
"""Add Chilean language metadata while retaining es-419 phonemization."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


parser = argparse.ArgumentParser()
parser.add_argument("--config", required=True, type=Path)
args = parser.parse_args()

config = json.loads(args.config.read_text(encoding="utf-8"))
config["language"] = {
    "code": "es_CL",
    "family": "es",
    "region": "CL",
    "name_native": "Español",
    "name_english": "Spanish",
    "country_english": "Chile",
}
config.setdefault("espeak", {})["voice"] = "es-419"
args.config.write_text(
    json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
)
