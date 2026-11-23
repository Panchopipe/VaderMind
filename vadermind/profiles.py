"""User profiles: one folder per home (Pancho, Pablo) chosen by the Wi-Fi network.

The Pi travels between homes. Everything that belongs to a home (Spotify
account, TV, timers, personality) lives in profiles/<name>/, and the network
the Pi is connected to says which home it is in:

1. --perfil on the command line, or VADER_PROFILE in .env, forces a profile.
2. Otherwise, the Wi-Fi network (SSID) listed in a profile's VADER_WIFI_SSIDS.
3. On an unknown network (the iPhone hotspot) or none, the last profile used,
   saved in .perfil_activo.
4. If nothing was ever used, "pancho".

Asking the network name is local: it works without internet. Without a
profiles/ folder Vader behaves as before, with everything in .env.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
from typing import Callable

from vadermind.config import ConfigurationError, read_env_file

LOGGER = logging.getLogger(__name__)

PROFILES_DIR = "profiles"
PROFILE_FILE = "perfil.env"
LAST_PROFILE_FILE = ".perfil_activo"
DEFAULT_PROFILE = "pancho"
SSIDS_KEY = "VADER_WIFI_SSIDS"
FORCE_KEY = "VADER_PROFILE"
NAME_PATTERN = re.compile(r"^[a-z0-9_-]+$")


@dataclass(frozen=True, slots=True)
class ProfileChoice:
    """Which profile Vader runs with, and why."""

    name: str | None
    directory: Path | None
    reason: str
    forced: bool = False

    def describe(self) -> str:
        if self.name is None:
            return "sin perfiles (todo en .env)"
        return f"{self.name} ({self.reason})"


def available_profiles(root: Path) -> dict[str, Path]:
    """Profiles under root/profiles: each folder with a perfil.env."""

    base = root / PROFILES_DIR
    if not base.is_dir():
        return {}
    return {
        folder.name: folder
        for folder in sorted(base.iterdir())
        if folder.is_dir() and (folder / PROFILE_FILE).is_file()
    }


def profile_ssids(directory: Path) -> set[str]:
    raw = read_env_file(directory / PROFILE_FILE).get(SSIDS_KEY, "")
    return {ssid.strip() for ssid in raw.split(",") if ssid.strip()}


def profile_for_ssid(ssid: str | None, profiles: dict[str, Path]) -> str | None:
    """The profile whose network list contains ssid (exact match, as SSIDs are)."""

    if not ssid:
        return None
    for name, directory in profiles.items():
        if ssid in profile_ssids(directory):
            return name
    return None


def parse_nmcli_ssid(output: str) -> str | None:
    """SSID of the active network from `nmcli -t -f ACTIVE,SSID device wifi list`."""

    for line in output.splitlines():
        active, _, ssid = line.partition(":")
        if active.strip() == "yes" and ssid:
            # Terse mode escapes ':' and '\' inside values.
            return ssid.replace("\\:", ":").replace("\\\\", "\\").strip() or None
    return None


_NETSH_SSID = re.compile(r"^\s*SSID\s*:\s*(.*?)\s*$", re.MULTILINE)


def parse_netsh_ssid(output: str) -> str | None:
    """SSID from `netsh wlan show interfaces` (Windows); BSSID lines don't match."""

    match = _NETSH_SSID.search(output)
    return (match.group(1) or None) if match else None


def current_ssid(
    run: Callable[..., subprocess.CompletedProcess] = subprocess.run,
    platform: str = sys.platform,
) -> str | None:
    """The Wi-Fi network this machine is connected to, or None."""

    if platform.startswith("linux"):
        command = ["nmcli", "-t", "-f", "ACTIVE,SSID", "device", "wifi", "list", "--rescan", "no"]
        parse = parse_nmcli_ssid
    elif platform.startswith("win"):
        command = ["netsh", "wlan", "show", "interfaces"]
        parse = parse_netsh_ssid
    else:
        return None
    if run is subprocess.run and not shutil.which(command[0]):
        return None
    try:
        result = run(command, capture_output=True, check=False, timeout=5)
    except (OSError, subprocess.SubprocessError) as error:
        LOGGER.warning("No pude preguntar a que red Wi-Fi estoy conectado: %s", error)
        return None
    if result.returncode != 0:
        return None
    output = result.stdout
    if isinstance(output, bytes):
        output = output.decode("utf-8", errors="replace")
    return parse(output)


def last_profile(root: Path) -> str | None:
    try:
        return (root / LAST_PROFILE_FILE).read_text(encoding="utf-8").strip() or None
    except OSError:
        return None


def remember_profile(root: Path, name: str) -> None:
    """Save the profile for the next start on an unknown network."""

    if last_profile(root) == name:
        return
    try:
        (root / LAST_PROFILE_FILE).write_text(name + "\n", encoding="utf-8")
    except OSError as error:
        LOGGER.warning("No pude guardar el perfil activo: %s", error)


def resolve_profile(
    requested: str | None = None,
    *,
    root: Path | None = None,
    ssid_lookup: Callable[[], str | None] = current_ssid,
) -> ProfileChoice:
    """Choose the profile for this start (see the module docstring)."""

    root = root or Path.cwd()
    profiles = available_profiles(root)
    forced = (requested or "").strip().lower()
    forced_by = "linea de comandos"
    if not forced:
        forced = (os.environ.get(FORCE_KEY) or read_env_file(root / ".env").get(FORCE_KEY, ""))
        forced = forced.strip().lower()
        forced_by = FORCE_KEY
    if forced:
        if forced not in profiles:
            known = ", ".join(profiles) or "ninguno"
            raise ConfigurationError(
                f"No existe el perfil '{forced}' en {PROFILES_DIR}/ (hay: {known})."
            )
        return ProfileChoice(forced, profiles[forced], forced_by, forced=True)
    if not profiles:
        return ProfileChoice(None, None, "sin perfiles")

    ssid = ssid_lookup()
    by_network = profile_for_ssid(ssid, profiles)
    if by_network:
        remember_profile(root, by_network)
        return ProfileChoice(by_network, profiles[by_network], f"red {ssid}")
    last = last_profile(root)
    if last in profiles:
        where = f"red {ssid} desconocida" if ssid else "sin red Wi-Fi"
        return ProfileChoice(last, profiles[last], f"ultimo usado, {where}")
    fallback = DEFAULT_PROFILE if DEFAULT_PROFILE in profiles else next(iter(profiles))
    return ProfileChoice(fallback, profiles[fallback], "por defecto")


def profile_after_network(
    choice: ProfileChoice,
    *,
    root: Path | None = None,
    ssid_lookup: Callable[[], str | None] = current_ssid,
) -> str | None:
    """Once the Wi-Fi is up: the profile of this network, if it isn't the one in use.

    At a cold boot Vader loads before the Wi-Fi connects, so the first choice
    may be the last home's profile. A forced profile never changes.
    """

    if choice.forced or choice.name is None:
        return None
    root = root or Path.cwd()
    by_network = profile_for_ssid(ssid_lookup(), available_profiles(root))
    if by_network and by_network != choice.name:
        return by_network
    return None


def validate_profile_name(name: str) -> str:
    name = name.strip().lower()
    if not NAME_PATTERN.fullmatch(name):
        raise ConfigurationError(
            "El nombre del perfil solo puede tener minusculas, numeros, '-' y '_'."
        )
    return name
