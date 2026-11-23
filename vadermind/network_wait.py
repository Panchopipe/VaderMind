"""Wait for the network inside Vader, after loading, instead of before starting.

The systemd service used to start Vader only after network-online.target (Wi-Fi
joined and with an address): on a cold boot that was ~14 s in which nothing
loaded, and then 7.3 s of loading. Now the service starts Vader right away and
Vader asks systemd the same question just before the dim pupils, so the models
load while the Wi-Fi connects and the pupils still mean "ready, with network".
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import sys
import time
from typing import Callable

LOGGER = logging.getLogger(__name__)

TARGET = "network-online.target"
# NetworkManager stops waiting for the Wi-Fi on its own and systemd then reaches
# the target anyway; this limit only covers a target nobody asked systemd for.
DEFAULT_TIMEOUT = 90.0
POLL_SECONDS = 0.1


def _target_active(systemctl: str) -> bool:
    try:
        result = subprocess.run([systemctl, "is-active", "--quiet", TARGET], check=False)
    except OSError as error:  # a broken check must not hold Vader back
        LOGGER.warning("No pude preguntar por la red a systemd: %s", error)
        return True
    return result.returncode == 0


def wait_for_network(
    timeout: float = DEFAULT_TIMEOUT,
    *,
    is_ready: Callable[[], bool] | None = None,
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> bool:
    """Block until systemd reaches network-online.target; True if it did.

    Without systemd (the laptop) there is nothing to wait for: it returns at once.
    """

    if is_ready is None:
        systemctl = shutil.which("systemctl") if sys.platform == "linux" else None
        if systemctl is None:
            return True
        is_ready = lambda: _target_active(systemctl)  # noqa: E731
    deadline = clock() + timeout
    while not is_ready():
        if clock() >= deadline:
            LOGGER.warning("Sigo sin red tras %.0f s: arranco igual.", timeout)
            return False
        sleep(POLL_SECONDS)
    return True
