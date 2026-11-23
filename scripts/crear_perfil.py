"""Create a home profile in profiles/<name>/ (outside Git).

Run from the VaderMind folder, on the laptop or on the Pi:

    # Pancho: today's configuration, copied (not moved) from .env and the state files
    python scripts/crear_perfil.py pancho --wifi RedPancho --desde-env

    # Pablo: a new home, chileno personality, Samsung TV noted for later
    python scripts/crear_perfil.py pablo --wifi RedPablo --personalidad chileno \
        --extra SAMSUNG_TV_HOST=192.0.2.23 --extra SAMSUNG_TV_MAC=00:00:5E:00:53:03

Copying keeps the old files in place, so going back is just deleting
profiles/. Values are never printed, only the names of what was copied.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vadermind.config import (  # noqa: E402
    PROFILE_KEYS,
    PROFILE_STATE_FILES,
    ConfigurationError,
    read_env_file,
)
from vadermind.profiles import PROFILE_FILE, PROFILES_DIR, SSIDS_KEY, validate_profile_name  # noqa: E402

# Where each state file lived before profiles, if .env doesn't say otherwise.
LEGACY_STATE_FILES = {
    "SPOTIFY_TOKEN_CACHE_PATH": ".spotify_token.json",
    "SPOTIFY_PLAYLIST_ALIASES_PATH": ".spotify_playlist_aliases.json",
    "LG_TV_CLIENT_KEY_PATH": ".lg_tv_client_key",
    "LG_TV_SCHEDULES_PATH": ".lg_tv_schedules.json",
}
PRIVATE_FILES = {"spotify_token.json", "lg_tv_client_key"}


def build_profile_lines(
    wifi: list[str], personality: str | None, extras: list[str], env: dict[str, str]
) -> list[str]:
    lines = ["# Perfil de hogar de VaderMind (fuera de Git). Ver docs/profiles-handoff.md", ""]
    lines.append(f"{SSIDS_KEY}={','.join(wifi)}")
    values = {key: env[key] for key in PROFILE_KEYS if env.get(key)}
    if personality:
        values["VADER_PERSONA_MODE"] = personality
    for key in PROFILE_KEYS:
        if key in values:
            lines.append(f"{key}={values[key]}")
    if extras:
        lines += ["", "# Datos anotados para integraciones futuras (Vader aun no los usa)"]
        lines += extras
    return lines


def main() -> int:
    parser = argparse.ArgumentParser(description="Crea un perfil de hogar en profiles/.")
    parser.add_argument("nombre", help="nombre del perfil, en minusculas (pancho, pablo)")
    parser.add_argument("--wifi", action="append", required=True, help="nombre exacto de la red (repetible)")
    parser.add_argument("--personalidad", choices=("canon", "chileno"))
    parser.add_argument(
        "--desde-env",
        action="store_true",
        help="copia del .env actual Spotify/TV/personalidad y los archivos de estado",
    )
    parser.add_argument("--extra", action="append", default=[], metavar="CLAVE=VALOR")
    parser.add_argument("--reemplazar", action="store_true", help="sobrescribe un perfil.env existente")
    args = parser.parse_args()

    root = Path.cwd()
    try:
        name = validate_profile_name(args.nombre)
    except ConfigurationError as error:
        print(error)
        return 1
    if not (root / "vadermind").is_dir():
        print("Ejecuta este script desde la carpeta VaderMind.")
        return 1
    for extra in args.extra:
        if "=" not in extra:
            print(f"--extra debe ser CLAVE=VALOR: {extra}")
            return 1

    folder = root / PROFILES_DIR / name
    profile_file = folder / PROFILE_FILE
    if profile_file.exists() and not args.reemplazar:
        print(f"Ya existe {profile_file}. Usa --reemplazar para sobrescribirlo.")
        return 1

    env = read_env_file(root / ".env") if args.desde_env else {}
    folder.mkdir(parents=True, exist_ok=True)
    lines = build_profile_lines(args.wifi, args.personalidad, args.extra, env)
    profile_file.write_text("\n".join(lines) + "\n", encoding="utf-8")
    copied_keys = [key for key in PROFILE_KEYS if env.get(key)]
    print(f"Creado {profile_file}: redes {', '.join(args.wifi)}.")
    if copied_keys:
        print("Copiado del .env: " + ", ".join(copied_keys))

    if args.desde_env:
        for key, target_name in PROFILE_STATE_FILES.items():
            source = root / (env.get(key) or LEGACY_STATE_FILES[key])
            target = folder / target_name
            if not source.is_file():
                print(f"No hay {source.name}: se omite.")
                continue
            if target.exists() and not args.reemplazar:
                print(f"Ya existe {target.name} en el perfil: se deja como esta.")
                continue
            shutil.copy2(source, target)
            if target_name in PRIVATE_FILES and os.name == "posix":
                target.chmod(0o600)
            print(f"Copiado {source.name} -> {PROFILES_DIR}/{name}/{target_name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
