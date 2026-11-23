"""Keep network connections and local engines warm so a spoken turn never waits on setup.

A new HTTPS connection costs one TLS handshake (about 0.13 s to Vercel and up to
0.8 s to Open-Meteo, measured from Chile). httpx closes idle connections after
5 s by default, while the servers keep them open for at least 50 s, so every
turn used to pay the handshake again.
"""

from __future__ import annotations

import logging
import threading
from typing import Callable
from urllib.parse import urlparse

import httpx

LOGGER = logging.getLogger(__name__)

# Long enough to span a whole spoken turn (talking, transcribing, thinking).
KEEPALIVE_SECONDS = 120.0
LONG_KEEPALIVE = httpx.Limits(
    max_connections=10, max_keepalive_connections=4, keepalive_expiry=KEEPALIVE_SECONDS
)


def origin(url: str) -> str:
    """Return scheme://host of a URL: the cheapest place to open a connection."""

    parsed = urlparse(url)
    return f"{parsed.scheme}://{parsed.netloc}/"


def warm_connection(client: httpx.Client, url: str) -> None:
    """Open (or refresh) a pooled connection to url's host; failures are harmless."""

    try:
        client.head(origin(url), timeout=5.0)
    except httpx.HTTPError as error:
        LOGGER.debug("No pude precalentar %s: %s", origin(url), error)


def in_background(*tasks: Callable[[], None]) -> None:
    """Run each warm-up in its own daemon thread; never block the caller."""

    def guarded(task: Callable[[], None]) -> None:
        try:
            task()
        except Exception as error:  # a warm-up must never break a turn
            LOGGER.debug("Precalentamiento fallido: %s", error)

    for task in tasks:
        threading.Thread(target=guarded, args=(task,), daemon=True).start()
