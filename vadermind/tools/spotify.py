"""Spotify Web API integration for VaderMind."""

from __future__ import annotations

from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlencode, urlparse
import base64
import json
import logging
import re
import secrets
import time
import unicodedata
import webbrowser

from vadermind.config import Settings

LOGGER = logging.getLogger("vadermind.spotify")

# Lookups done while Vader speaks are trusted only briefly, then fetched again.
PREFETCH_TTL_SECONDS = 15.0
# Spoken-ack tools that start by listing devices; pause is silent, so no window.
PREFETCH_DEVICE_TOOLS = {
    "spotify.play",
    "spotify.play_playlist",
    "spotify.play_reference",
    "spotify.resume",
    "spotify.next",
    "spotify.previous",
    "spotify.volume",
    "spotify.shuffle",
}

SPOTIFY_AUTH_URL = "https://accounts.spotify.com/authorize"
SPOTIFY_TOKEN_URL = "https://accounts.spotify.com/api/token"
SPOTIFY_API_URL = "https://api.spotify.com/v1"
SPOTIFY_SCOPES = (
    "user-read-playback-state "
    "user-modify-playback-state "
    "user-read-currently-playing "
    "user-read-private "
    "playlist-read-private "
    "playlist-read-collaborative "
    "playlist-modify-public "
    "playlist-modify-private "
    "user-library-modify"
)


def _httpx():
    """Import httpx only when Spotify network operations actually run."""

    try:
        import httpx
    except ImportError as error:
        raise SpotifyError(
            "Falta la libreria 'httpx'. Ejecuta '.\\.venv\\Scripts\\python.exe -m pip install -r requirements.txt'."
        ) from error
    return httpx


class SpotifyError(RuntimeError):
    """Raised when Spotify cannot complete a requested operation."""


@dataclass(slots=True)
class SpotifyAuthResult:
    """Stores the result of the one-time browser authorization."""

    authorized: bool
    message: str


class SpotifyTokenStore:
    """Small JSON token cache for Spotify OAuth tokens."""

    def __init__(self, cache_path: str) -> None:
        """Build a token store from a path relative to the current project."""

        self.path = Path(cache_path)

    def load(self) -> dict[str, Any] | None:
        """Read cached token data if it exists."""

        if not self.path.exists():
            return None
        try:
            return json.loads(self.path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            raise SpotifyError(
                f"El archivo de token de Spotify esta corrupto: {self.path}"
            ) from error

    def save(self, token_data: dict[str, Any]) -> None:
        """Persist token data locally."""

        self.path.write_text(json.dumps(token_data, indent=2), encoding="utf-8")


class SpotifyOAuth:
    """Handles Spotify's local Authorization Code flow."""

    def __init__(self, settings: Settings) -> None:
        """Store Spotify settings and the token cache."""

        self.settings = settings
        self.token_store = SpotifyTokenStore(settings.spotify_token_cache_path)

    def authorization_url(self, state: str) -> str:
        """Build the Spotify authorization URL the user must approve."""

        params = {
            "client_id": self.settings.spotify_client_id,
            "response_type": "code",
            "redirect_uri": self.settings.spotify_redirect_uri,
            "scope": SPOTIFY_SCOPES,
            "state": state,
            "show_dialog": "false",
        }
        return f"{SPOTIFY_AUTH_URL}?{urlencode(params)}"

    def authorize_with_local_callback(self, timeout_seconds: int = 180) -> SpotifyAuthResult:
        """Open the browser, wait for Spotify's callback, and cache the token."""

        state = secrets.token_urlsafe(16)
        auth_url = self.authorization_url(state)
        callback = self._wait_for_callback(auth_url, state, timeout_seconds)
        if "error" in callback:
            return SpotifyAuthResult(
                authorized=False,
                message=f"Spotify rechazo la autorizacion: {callback['error']}",
            )

        code = callback.get("code")
        if not code:
            return SpotifyAuthResult(
                authorized=False,
                message="Spotify no devolvio un codigo de autorizacion.",
            )

        token_data = self.exchange_code(code)
        self.token_store.save(token_data)
        return SpotifyAuthResult(
            authorized=True,
            message="Spotify quedo autorizado. Vader ya puede controlar playback.",
        )

    def exchange_code(self, code: str) -> dict[str, Any]:
        """Exchange an authorization code for access and refresh tokens."""

        httpx = _httpx()
        response = httpx.post(
            SPOTIFY_TOKEN_URL,
            headers={"Authorization": self._basic_auth_header()},
            data={
                "grant_type": "authorization_code",
                "code": code,
                "redirect_uri": self.settings.spotify_redirect_uri,
            },
            timeout=20,
        )
        return self._token_from_response(response)

    def refresh_access_token(self, refresh_token: str) -> dict[str, Any]:
        """Refresh an expired Spotify access token."""

        previous_token = self.token_store.load() or {}
        httpx = _httpx()
        response = httpx.post(
            SPOTIFY_TOKEN_URL,
            headers={"Authorization": self._basic_auth_header()},
            data={
                "grant_type": "refresh_token",
                "refresh_token": refresh_token,
            },
            timeout=20,
        )
        new_token = self._token_from_response(response)
        if "refresh_token" not in new_token:
            new_token["refresh_token"] = refresh_token
        if "scope" not in new_token and previous_token.get("scope"):
            new_token["scope"] = previous_token["scope"]
        self.token_store.save(new_token)
        return new_token

    def access_token(self) -> str:
        """Return a valid access token, refreshing it if needed."""

        token_data = self.token_store.load()
        if not token_data:
            raise SpotifyError(
                "Spotify no esta autorizado. Ejecuta primero: py main.py --spotify-auth"
            )

        granted_scopes = set(str(token_data.get("scope", "")).split())
        missing_scopes = set(SPOTIFY_SCOPES.split()) - granted_scopes
        if missing_scopes:
            raise SpotifyError(
                "La autorizacion de Spotify es anterior a las nuevas capacidades. "
                "Ejecuta nuevamente: py main.py --spotify-auth"
            )

        expires_at = float(token_data.get("expires_at", 0))
        if expires_at > time.time():
            return str(token_data["access_token"])

        refresh_token = token_data.get("refresh_token")
        if not refresh_token:
            raise SpotifyError(
                "Spotify no tiene refresh token. Ejecuta nuevamente: py main.py --spotify-auth"
            )

        refreshed = self.refresh_access_token(str(refresh_token))
        return str(refreshed["access_token"])

    def _basic_auth_header(self) -> str:
        """Return the Basic auth header Spotify expects for token requests."""

        raw = f"{self.settings.spotify_client_id}:{self.settings.spotify_client_secret}"
        encoded = base64.b64encode(raw.encode("utf-8")).decode("ascii")
        return f"Basic {encoded}"

    def _token_from_response(self, response: httpx.Response) -> dict[str, Any]:
        """Parse and normalize Spotify token responses."""

        try:
            response.raise_for_status()
        except _httpx().HTTPStatusError as error:
            raise SpotifyError(
                f"Spotify no pudo entregar token OAuth: HTTP {response.status_code}."
            ) from error

        token_data = response.json()
        expires_in = int(token_data.get("expires_in", 3600))
        token_data["expires_at"] = time.time() + expires_in - 30
        return token_data

    def _wait_for_callback(
        self, auth_url: str, expected_state: str, timeout_seconds: int
    ) -> dict[str, str]:
        """Run a one-request localhost callback server for Spotify OAuth."""

        parsed_redirect = urlparse(self.settings.spotify_redirect_uri)
        port = parsed_redirect.port or 80
        callback_path = parsed_redirect.path or "/callback"
        result: dict[str, str] = {}

        class CallbackHandler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:  # noqa: N802 - stdlib callback name
                parsed_url = urlparse(self.path)
                if parsed_url.path != callback_path:
                    self.send_response(404)
                    self.end_headers()
                    return

                query = parse_qs(parsed_url.query)
                returned_state = query.get("state", [""])[0]
                if returned_state != expected_state:
                    result["error"] = "state_mismatch"
                else:
                    result["code"] = query.get("code", [""])[0]
                    result["error"] = query.get("error", [""])[0]

                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.end_headers()
                self.wfile.write(
                    b"<html><body><h1>Spotify autorizado.</h1>"
                    b"<p>Ya puedes volver a VaderMind.</p></body></html>"
                )

            def log_message(self, format: str, *args: object) -> None:
                return

        server = HTTPServer(("127.0.0.1", port), CallbackHandler)
        server.timeout = timeout_seconds

        print("Abriendo Spotify para autorizar VaderMind.")
        print(f"Si el navegador no se abre, pega esta URL:\n{auth_url}")
        webbrowser.open(auth_url)
        server.handle_request()
        server.server_close()

        if not result:
            raise SpotifyError("Tiempo agotado esperando autorizacion de Spotify.")
        return {key: value for key, value in result.items() if value}


class SpotifyClient:
    """Small Spotify Web API client for playback-oriented commands."""

    def __init__(self, settings: Settings) -> None:
        """Store settings and OAuth helper."""

        self.settings = settings
        self.oauth = SpotifyOAuth(settings)

    def prefetch(self, tool_name: str, args: dict[str, object]) -> None:
        """Run the read-only lookups of a command while Vader is still talking.

        Only GET requests happen here: the search and the device list. Choosing
        or transferring the device and starting playback still wait for execute.
        Failures are ignored; execute repeats the lookup and reports the error.
        """

        try:
            if tool_name == "spotify.play":
                query = str(args.get("query", "")).strip()
                item_type = str(args.get("item_type", "auto"))
                if query:
                    self._store_prefetch(
                        ("search", query, item_type),
                        self.search_items(query=query, item_type=item_type, limit=5),
                    )
            if tool_name in PREFETCH_DEVICE_TOOLS:
                self._store_prefetch(("devices",), self._fetch_devices())
        except SpotifyError as error:
            LOGGER.debug("Spotify prefetch skipped: %s", error)

    def _store_prefetch(self, key: tuple[str, ...], value: object) -> None:
        if not hasattr(self, "_prefetched"):
            self._prefetched = {}
        self._prefetched[key] = (time.monotonic(), value)

    def _take_prefetch(self, key: tuple[str, ...]) -> Any | None:
        """Use a prefetched result once, and only while it is still fresh."""

        stored = getattr(self, "_prefetched", {}).pop(key, None)
        if stored and time.monotonic() - stored[0] <= PREFETCH_TTL_SECONDS:
            return stored[1]
        return None

    def get_devices(self) -> list[dict[str, Any]]:
        """Return the user's available Spotify Connect devices."""

        prefetched = self._take_prefetch(("devices",))
        if prefetched is not None:
            return list(prefetched)
        return self._fetch_devices()

    def _fetch_devices(self) -> list[dict[str, Any]]:
        data = self._request("GET", "/me/player/devices")
        return list(data.get("devices", []))

    def wait_for_tv_device(
        self,
        *,
        preferred_name: str | None = None,
        previous_device_ids: set[str] | None = None,
        timeout_seconds: float = 20.0,
    ) -> dict[str, Any]:
        """Wait for the Spotify app on a TV to publish a controllable device."""

        previous = previous_device_ids or set()
        preferred = _normalize_match_text(preferred_name or "")
        deadline = time.monotonic() + timeout_seconds
        last_names: list[str] = []
        while time.monotonic() < deadline:
            devices = [
                device
                for device in self.get_devices()
                if device.get("id")
                and not device.get("is_restricted")
            ]
            last_names = [str(device.get("name") or "") for device in devices]
            if preferred:
                match = next(
                    (
                        device
                        for device in devices
                        if _normalize_match_text(str(device.get("name") or ""))
                        == preferred
                    ),
                    None,
                )
                if match:
                    return match
            else:
                tv_candidates = [
                    device
                    for device in devices
                    if self._looks_like_tv_device(device)
                ]
                if len(tv_candidates) == 1:
                    return tv_candidates[0]
                if len(tv_candidates) > 1:
                    names = ", ".join(str(item.get("name")) for item in tv_candidates)
                    raise SpotifyError(
                        "Hay mas de un televisor Spotify disponible: "
                        f"{names}. Configura SPOTIFY_TV_DEVICE_NAME."
                    )
                new_devices = [
                    device
                    for device in devices
                    if str(device.get("id")) not in previous
                ]
                if len(new_devices) == 1:
                    return new_devices[0]
            time.sleep(0.75)

        if preferred:
            raise SpotifyError(
                f"Spotify no publico el dispositivo {preferred_name}. "
                "Comprueba el nombre configurado y que la app este abierta en la TV."
            )
        visible = ", ".join(name for name in last_names if name)
        suffix = f" Dispositivos visibles: {visible}." if visible else ""
        raise SpotifyError(
            "Spotify no publico un dispositivo de television controlable." + suffix
        )

    def play_search_on_device(
        self,
        query: str,
        device: dict[str, Any],
        *,
        item_type: str = "auto",
        shuffle: bool = False,
    ) -> dict[str, Any]:
        """Start a search result on an explicit device and verify playback."""

        candidates = self.search_items(query=query, item_type=item_type, limit=5)
        selected = (
            self._best_auto_candidate(query, candidates)
            if item_type.strip().lower() == "auto"
            else candidates[0]
        )
        return self._play_resolved_on_device(
            query,
            selected,
            device,
            shuffle=shuffle,
            candidates=candidates,
        )

    def play_user_playlist_on_device(
        self,
        query: str,
        device: dict[str, Any],
        *,
        shuffle: bool = False,
    ) -> dict[str, Any]:
        """Resolve a saved playlist and play it on one explicit TV device."""

        selected = self._playlist_summary(self.find_user_playlist(query))
        return self._play_resolved_on_device(
            query,
            selected,
            device,
            shuffle=shuffle,
            candidates=[selected],
        )

    def _play_resolved_on_device(
        self,
        query: str,
        selected: dict[str, Any],
        device: dict[str, Any],
        *,
        shuffle: bool,
        candidates: list[dict[str, Any]],
    ) -> dict[str, Any]:
        playable = (
            self._first_show_episode(selected)
            if selected["type"] == "show"
            else selected
        )
        device_id = str(device.get("id") or "").strip()
        if not device_id:
            raise SpotifyError("Spotify no entrego un identificador para la television.")
        active_device = self._activate_device_if_needed(device)
        if shuffle:
            if playable["type"] not in {"artist", "playlist", "album"}:
                raise SpotifyError(
                    "El modo aleatorio requiere un artista, una playlist o un album."
                )
            self._request(
                "PUT",
                "/me/player/shuffle",
                params={"state": "true", "device_id": device_id},
                expect_json=False,
            )

        playback: dict[str, Any] | None = None
        for attempt in range(2):
            self._play_item(playable, device_id=device_id)
            playback = self._wait_for_playback_on_device(device_id)
            if playback:
                break
            if attempt == 0:
                time.sleep(0.5)
        if not playback:
            raise SpotifyError(
                "Envie la reproduccion a la television, pero Spotify no la confirmo."
            )

        raw_item = playback.get("item") or {}
        confirmed_name = str(raw_item.get("name") or playable.get("name") or "musica")
        artists = ", ".join(
            str(artist.get("name") or "") for artist in raw_item.get("artists", [])
        ).strip(", ")
        device_state = playback.get("device") or active_device
        device_name = str(device_state.get("name") or device.get("name") or "television")
        volume = device_state.get("volume_percent")
        artist_text = f" de {artists}" if artists else ""
        return {
            "ok": True,
            "tool_name": "media.spotify_on_tv",
            "message": (
                f"Spotify confirmo {confirmed_name}{artist_text} en {device_name}."
            ),
            "item": playable,
            "confirmed_item": {
                "name": confirmed_name,
                "artists": artists,
            },
            "device": {
                "id": device_id,
                "name": device_name,
                "type": device_state.get("type", device.get("type", "unknown")),
                "spotify_volume_percent": volume,
            },
            "candidates": candidates,
            "requested_query": query,
        }

    def _wait_for_playback_on_device(
        self, device_id: str, timeout_seconds: float = 6.0
    ) -> dict[str, Any] | None:
        deadline = time.monotonic() + timeout_seconds
        while time.monotonic() < deadline:
            state = self._request(
                "GET",
                "/me/player",
                params={"additional_types": "track,episode"},
                allow_no_content=True,
            )
            active = state.get("device") or {}
            if (
                str(active.get("id") or "") == device_id
                and state.get("is_playing") is True
                and state.get("item")
            ):
                return state
            time.sleep(0.4)
        return None

    @staticmethod
    def _looks_like_tv_device(device: dict[str, Any]) -> bool:
        device_type = _normalize_match_text(str(device.get("type") or ""))
        name = _normalize_match_text(str(device.get("name") or ""))
        return device_type in {"tv", "television"} or any(
            token in name for token in ("webos", "lg tv", "television")
        )

    def get_user_playlists(self, limit: int = 50) -> list[dict[str, Any]]:
        """Return the user's saved, owned, and collaborative playlists."""

        playlists: list[dict[str, Any]] = []
        offset = 0
        page_size = min(max(limit, 1), 50)

        while True:
            data = self._request(
                "GET",
                "/me/playlists",
                params={"limit": page_size, "offset": offset},
            )
            items = [item for item in data.get("items", []) if item]
            playlists.extend(items)

            if len(playlists) >= limit or not data.get("next"):
                return playlists[:limit]
            offset += page_size

    def list_user_playlists(self, limit: int = 10) -> dict[str, Any]:
        """Return a concise summary of the user's playlists."""

        playlists = self.get_user_playlists(limit=limit)
        if not playlists:
            return {
                "ok": True,
                "tool_name": "spotify.list_playlists",
                "message": "No encontre playlists en tu biblioteca de Spotify.",
                "playlists": [],
            }

        summaries = [self._playlist_summary(playlist) for playlist in playlists]
        names = ", ".join(summary["name"] for summary in summaries[:8])
        extra = len(summaries) - 8
        suffix = f" y {extra} mas" if extra > 0 else ""
        return {
            "ok": True,
            "tool_name": "spotify.list_playlists",
            "message": f"Tus playlists visibles son: {names}{suffix}.",
            "playlists": summaries,
        }

    def play_user_playlist(self, query: str, shuffle: bool = False) -> dict[str, Any]:
        """Find one of the user's playlists by name and start playback."""

        playlist = self.find_user_playlist(query)
        device = self._select_device()
        item = self._playlist_summary(playlist)
        if shuffle:
            self._request(
                "PUT",
                "/me/player/shuffle",
                params={"state": "true", "device_id": device["id"]},
                expect_json=False,
            )
        self._play_item(item, device_id=device["id"])

        message = self._play_message(item, device)
        if shuffle:
            message += " Modo aleatorio activado."

        return {
            "ok": True,
            "tool_name": "spotify.play_playlist",
            "message": message,
            "playlist": item,
            "device": {
                "id": device["id"],
                "name": device["name"],
                "type": device.get("type", "unknown"),
            },
        }

    def find_user_playlist(self, query: str) -> dict[str, Any]:
        """Find the best matching user playlist by exact or partial name."""

        normalized_query = _normalize_match_text(query)
        if not normalized_query:
            raise SpotifyError("Necesito el nombre de la playlist que quieres reproducir.")

        # A saved playlist may be beyond the first Spotify API page. Search a
        # generous session-sized window before falling back to an error.
        playlists = self.get_user_playlists(limit=1000)
        exact_matches = [
            playlist
            for playlist in playlists
            if _normalize_match_text(playlist.get("name", "")) == normalized_query
        ]
        if exact_matches:
            return exact_matches[0]

        alias = self._find_playlist_alias(query)
        if alias:
            return alias

        partial_matches = [
            playlist
            for playlist in playlists
            if normalized_query in _normalize_match_text(playlist.get("name", ""))
        ]
        if partial_matches:
            return partial_matches[0]

        visible_names = ", ".join(
            playlist.get("name", "Sin nombre") for playlist in playlists[:5]
        )
        hint = f" Vi estas: {visible_names}." if visible_names else ""
        raise SpotifyError(f"No encontre una playlist tuya llamada {query}.{hint}")

    def _find_playlist_alias(self, query: str) -> dict[str, Any] | None:
        """Resolve a locally remembered playlist that Spotify does not list."""

        alias_path = Path(
            getattr(
                self.settings,
                "spotify_playlist_aliases_path",
                ".spotify_playlist_aliases.json",
            )
        )
        if not alias_path.is_file():
            return None
        try:
            aliases = json.loads(alias_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        if not isinstance(aliases, dict):
            return None
        normalized_query = _normalize_match_text(query)
        for name, uri in aliases.items():
            if _normalize_match_text(str(name)) != normalized_query:
                continue
            uri_text = str(uri)
            if not re.fullmatch(r"spotify:playlist:[A-Za-z0-9]+", uri_text):
                continue
            return {
                "id": uri_text.rsplit(":", 1)[-1],
                "type": "playlist",
                "uri": uri_text,
                "name": str(name),
                "owner": {"display_name": "Spotify"},
                "items": {"total": 0},
            }
        return None

    def search_items(
        self, query: str, item_type: str = "auto", limit: int = 5
    ) -> list[dict[str, Any]]:
        """Search Spotify and return compact, playable results."""

        prefetched = self._take_prefetch(("search", query, item_type))
        if prefetched is not None:
            return list(prefetched)

        normalized_type = item_type.strip().lower()
        supported = {"track", "artist", "playlist", "album", "show", "episode"}
        if normalized_type == "auto":
            # Include artists and albums so a broad command such as "pon Juan
            # Gabriel" can start a continuous context instead of one isolated
            # track. Exact-name ranking happens in play_search().
            search_types = ["track", "artist", "album", "playlist"]
        elif normalized_type in supported:
            search_types = [normalized_type]
        else:
            choices = ", ".join(sorted(supported))
            raise SpotifyError(f"El tipo de busqueda debe ser auto o uno de: {choices}.")

        result_limit = min(max(limit, 1), 10)
        data = self._request(
            "GET",
            "/search",
            params={
                "q": query,
                "type": ",".join(search_types),
                "market": self.settings.spotify_market,
                "limit": result_limit,
            },
        )

        results: list[dict[str, Any]] = []
        plural_keys = {
            "track": "tracks",
            "artist": "artists",
            "playlist": "playlists",
            "album": "albums",
            "show": "shows",
            "episode": "episodes",
        }
        for result_type in search_types:
            raw_items = data.get(plural_keys[result_type], {}).get("items", [])
            results.extend(
                self._item_summary(item, result_type)
                for item in raw_items
                if item and item.get("uri")
            )
        if not results:
            raise SpotifyError(f"No encontre resultados en Spotify para: {query}")
        return results if normalized_type == "auto" else results[:result_limit]

    def search_first(self, query: str, item_type: str = "auto") -> dict[str, Any]:
        """Search Spotify and return the first result."""

        return self.search_items(query=query, item_type=item_type, limit=5)[0]

    def play_search(
        self, query: str, item_type: str = "auto", shuffle: bool = False
    ) -> dict[str, Any]:
        """Search for a track or playlist and start playback."""

        candidates = self.search_items(query=query, item_type=item_type, limit=5)
        selected = (
            self._best_auto_candidate(query, candidates)
            if item_type.strip().lower() == "auto"
            else candidates[0]
        )
        result = self.play_item(selected, shuffle=shuffle)
        result["candidates"] = candidates
        result["requested_query"] = query
        result["explanation"] = (
            f'Busque "{query}" en Spotify y elegi {selected["name"]} '
            f'como {self._type_label(selected["type"])}. La reproduccion se inicio '
            "usando su contexto para que pueda continuar."
        )
        return result

    def _best_auto_candidate(
        self, query: str, candidates: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """Prefer an exact artist/track/context match over mixed search order."""

        normalized_query = _normalize_match_text(query)
        exact = [
            item
            for item in candidates
            if _normalize_match_text(item.get("name", "")) == normalized_query
        ]
        type_priority = {"artist": 0, "track": 1, "album": 2, "playlist": 3}
        if exact:
            return min(exact, key=lambda item: type_priority.get(str(item["type"]), 9))
        return candidates[0]

    def list_search(self, query: str, item_type: str) -> dict[str, Any]:
        """Search without starting playback so a follow-up can select a result."""

        candidates = self.search_items(query=query, item_type=item_type, limit=5)
        names = "; ".join(
            f"{index}. {item['name']} de {item['subtitle']}"
            for index, item in enumerate(candidates, start=1)
        )
        return {
            "ok": True,
            "tool_name": "spotify.search",
            "message": f"Encontre: {names}. Puedes decir ahora pon la segunda.",
            "candidates": candidates,
        }

    def play_item(
        self, item: dict[str, Any], shuffle: bool = False
    ) -> dict[str, Any]:
        """Play a previously resolved search result."""

        device = self._select_device()
        playable_item = item
        if item["type"] == "show":
            playable_item = self._first_show_episode(item)
        if shuffle:
            if playable_item["type"] not in {"artist", "playlist", "album"}:
                raise SpotifyError(
                    "El modo aleatorio requiere un artista, una playlist o un album."
                )
            self._request(
                "PUT",
                "/me/player/shuffle",
                params={"state": "true", "device_id": device["id"]},
                expect_json=False,
            )
        self._play_item(playable_item, device_id=device["id"])

        message = self._play_message(playable_item, device)
        if shuffle:
            message += " Modo aleatorio activado."

        return {
            "ok": True,
            "tool_name": "spotify.play",
            "message": message,
            "item": playable_item,
            "device": {
                "id": device["id"],
                "name": device["name"],
                "type": device.get("type", "unknown"),
            },
        }

    def play_reference(self, reference: str, shuffle: bool = False) -> dict[str, Any]:
        """Play a Spotify URL or URI without requiring catalog visibility."""

        match = re.search(
            r"(?:open\.spotify\.com/|spotify:)(playlist|album|track|episode|show)(?:/|:)([A-Za-z0-9]+)",
            reference,
            re.IGNORECASE,
        )
        if not match:
            raise SpotifyError("El enlace o URI de Spotify no es valido.")
        item_type = match.group(1).lower()
        item_id = match.group(2)
        uri = f"spotify:{item_type}:{item_id}"
        public_url = f"https://open.spotify.com/{item_type}/{item_id}"
        title = self._reference_title(public_url) or f"{item_type} de Spotify"
        item = {"type": item_type, "uri": uri, "name": title, "subtitle": "Spotify"}

        if item_type == "playlist":
            self._remember_playlist_alias(title, uri)

        device = self._select_device()
        if shuffle:
            if item_type not in {"playlist", "album"}:
                raise SpotifyError("El modo aleatorio requiere una playlist o un album.")
            self._request(
                "PUT",
                "/me/player/shuffle",
                params={"state": "true", "device_id": device["id"]},
                expect_json=False,
            )
        playable_item = self._first_show_episode(item) if item_type == "show" else item
        self._play_item(playable_item, device_id=device["id"])
        message = self._play_message(playable_item, device)
        if shuffle:
            message += " Modo aleatorio activado."
        return self._ok("spotify.play_reference", message, item=playable_item)

    def _reference_title(self, public_url: str) -> str | None:
        """Resolve a public Spotify title through its oEmbed metadata."""

        try:
            response = _httpx().get(
                "https://open.spotify.com/oembed",
                params={"url": public_url},
                timeout=20,
            )
            response.raise_for_status()
            title = str(response.json().get("title", "")).strip()
            return title or None
        except Exception:
            return None

    def _remember_playlist_alias(self, name: str, uri: str) -> None:
        """Persist a public playlist name so later natural commands can find it."""

        alias_path = Path(
            getattr(
                self.settings,
                "spotify_playlist_aliases_path",
                ".spotify_playlist_aliases.json",
            )
        )
        aliases: dict[str, str] = {}
        if alias_path.is_file():
            try:
                loaded = json.loads(alias_path.read_text(encoding="utf-8"))
                if isinstance(loaded, dict):
                    aliases = {str(key): str(value) for key, value in loaded.items()}
            except (OSError, json.JSONDecodeError):
                pass
        aliases[name] = uri
        try:
            alias_path.write_text(
                json.dumps(aliases, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
        except OSError as error:
            raise SpotifyError(f"No pude guardar el alias local de la playlist: {error}") from error

    def _play_item(self, item: dict[str, Any], device_id: str) -> None:
        """Send the playback command to Spotify."""

        payload: dict[str, Any]
        if item["type"] == "track" and item.get("context_uri"):
            payload = {
                "context_uri": item["context_uri"],
                "offset": {"uri": item["uri"]},
            }
        elif item["type"] in {"track", "episode"}:
            payload = {"uris": [item["uri"]]}
        else:
            payload = {"context_uri": item["uri"]}

        self._request(
            "PUT",
            "/me/player/play",
            params={"device_id": device_id},
            json=payload,
            expect_json=False,
        )

    def pause(self) -> dict[str, Any]:
        """Pause playback on the selected device."""

        device = self._select_device()
        try:
            self._request(
                "PUT",
                "/me/player/pause",
                params={"device_id": device["id"]},
                expect_json=False,
            )
        except SpotifyError as error:
            # Spotify Connect can apply pause but still answer 403 Restriction
            # violated. Trust the observed player state over the contradictory
            # transport response, but fail normally when it cannot be verified.
            if "restriction" not in str(error).casefold():
                raise
            state = self._request("GET", "/me/player/currently-playing")
            if "is_playing" not in state or state.get("is_playing") is not False:
                raise
        return self._ok("spotify.pause", f"Reproduccion pausada en {device['name']}.")

    def resume(self) -> dict[str, Any]:
        """Resume playback on the selected device."""

        device = self._select_device()
        self._request("PUT", "/me/player/play", params={"device_id": device["id"]}, expect_json=False)
        return self._ok("spotify.resume", f"Reproduccion reanudada en {device['name']}.")

    def skip(self, direction: str) -> dict[str, Any]:
        """Skip to the next or previous item."""

        device = self._select_device()
        endpoint = "/me/player/next" if direction == "next" else "/me/player/previous"
        self._request("POST", endpoint, params={"device_id": device["id"]}, expect_json=False)
        message = "Pase a la siguiente pista." if direction == "next" else "Volvi a la pista anterior."
        return self._ok(f"spotify.{direction}", message)

    def set_volume(self, volume_percent: int | None = None, delta: int | None = None) -> dict[str, Any]:
        """Set an absolute volume or adjust it relative to the current device."""

        device = self._select_device()
        if volume_percent is None:
            current = device.get("volume_percent")
            if current is None:
                raise SpotifyError("El dispositivo no informa su volumen actual.")
            volume_percent = int(current) + int(delta or 0)
        volume_percent = min(max(int(volume_percent), 0), 100)
        if device.get("supports_volume") is False:
            raise SpotifyError(f"El dispositivo {device['name']} no permite cambiar volumen desde Spotify.")
        self._request(
            "PUT",
            "/me/player/volume",
            params={"volume_percent": volume_percent, "device_id": device["id"]},
            expect_json=False,
        )
        return self._ok("spotify.volume", f"Volumen ajustado a {volume_percent} por ciento.")

    def set_shuffle(self, state: bool) -> dict[str, Any]:
        """Enable or disable shuffle."""

        device = self._select_device()
        self._request("PUT", "/me/player/shuffle", params={"state": str(state).lower(), "device_id": device["id"]}, expect_json=False)
        return self._ok("spotify.shuffle", "Modo aleatorio activado." if state else "Modo aleatorio desactivado.")

    def set_repeat(self, state: str) -> dict[str, Any]:
        """Set repeat mode to track, context, or off."""

        if state not in {"track", "context", "off"}:
            raise SpotifyError("La repeticion debe ser cancion, contexto o desactivada.")
        device = self._select_device()
        self._request("PUT", "/me/player/repeat", params={"state": state, "device_id": device["id"]}, expect_json=False)
        messages = {"track": "Repetire esta pista.", "context": "Repetire la playlist o album actual.", "off": "Repeticion desactivada."}
        return self._ok("spotify.repeat", messages[state])

    def currently_playing(self) -> dict[str, Any]:
        """Return the current track or podcast episode."""

        data = self._request(
            "GET", "/me/player/currently-playing", params={"additional_types": "track,episode"}, allow_no_content=True
        )
        raw_item = data.get("item")
        if not raw_item:
            return {"ok": True, "tool_name": "spotify.current", "message": "No hay nada sonando ahora.", "item": None}
        item_type = raw_item.get("type", "track")
        item = self._item_summary(raw_item, item_type)
        state = "sonando" if data.get("is_playing") else "en pausa"
        return {
            "ok": True,
            "tool_name": "spotify.current",
            "message": f"Esta {state}: {item['name']} de {item['subtitle']}.",
            "item": item,
        }

    def add_to_queue(self, query: str, item_type: str = "track") -> dict[str, Any]:
        """Search for a track or episode and add it to the playback queue."""

        item = self.search_first(query, item_type)
        device = self._select_device()
        self._request("POST", "/me/player/queue", params={"uri": item["uri"], "device_id": device["id"]}, expect_json=False)
        return self._ok("spotify.queue", f"Agregue {item['name']} de {item['subtitle']} a la cola.", item=item)

    def save_item(self, item: dict[str, Any]) -> dict[str, Any]:
        """Save an item to the user's Spotify library using the 2026 endpoint."""

        self._request("PUT", "/me/library", params={"uris": item["uri"]}, expect_json=False)
        return self._ok("spotify.save", f"Guarde {item['name']} de {item['subtitle']} en tu biblioteca.", item=item)

    def save_search(self, query: str, item_type: str = "track") -> dict[str, Any]:
        """Search for an item and save the first match."""

        return self.save_item(self.search_first(query, item_type))

    def save_current(self) -> dict[str, Any]:
        """Save the item that is currently playing."""

        current = self.currently_playing()
        item = current.get("item")
        if not isinstance(item, dict):
            raise SpotifyError("No hay una cancion o episodio sonando para guardar.")
        return self.save_item(item)

    def play_similar(self) -> dict[str, Any]:
        """Play another track by the current track's primary artist."""

        data = self._request("GET", "/me/player/currently-playing", params={"additional_types": "track"}, allow_no_content=True)
        raw_item = data.get("item")
        if not raw_item or raw_item.get("type") != "track":
            raise SpotifyError("Necesito una cancion sonando para buscar algo parecido.")
        artists = raw_item.get("artists", [])
        if not artists:
            raise SpotifyError("Spotify no informo el artista de la cancion actual.")
        artist_name = artists[0].get("name", "")
        candidates = self.search_items(f'artist:"{artist_name}"', "track", limit=10)
        different = next((item for item in candidates if item["uri"] != raw_item.get("uri")), None)
        if not different:
            raise SpotifyError(f"No encontre otra cancion de {artist_name}.")
        result = self.play_item(different)
        result["message"] = f"Elegí algo relacionado por artista: {different['name']} de {different['subtitle']}."
        result["candidates"] = candidates
        return result

    def create_playlist(self, name: str, public: bool = False, description: str = "") -> dict[str, Any]:
        """Create an empty playlist for the current user."""

        playlist = self._request("POST", "/me/playlists", json={"name": name, "public": public, "description": description})
        item = self._playlist_summary(playlist)
        visibility = "publica" if public else "privada"
        return self._ok("spotify.create_playlist", f"Cree la playlist {name}, {visibility}.", playlist=item)

    def update_playlist(self, query: str, changes: dict[str, Any]) -> dict[str, Any]:
        """Update the name, description, or visibility of a playlist."""

        playlist = self.find_user_playlist(query)
        payload = {key: value for key, value in changes.items() if key in {"name", "description", "public", "collaborative"}}
        if not payload:
            raise SpotifyError("No recibi ningun cambio para la playlist.")
        self._request("PUT", f"/playlists/{playlist['id']}", json=payload, expect_json=False)
        return self._ok("spotify.update_playlist", f"Actualice la playlist {playlist.get('name', query)}.")

    def add_to_playlist(self, playlist_query: str, item_query: str) -> dict[str, Any]:
        """Append the first matching track to a user playlist."""

        playlist = self.find_user_playlist(playlist_query)
        item = self.search_first(item_query, "track")
        self._request("POST", f"/playlists/{playlist['id']}/items", json={"uris": [item["uri"]]})
        return self._ok("spotify.add_to_playlist", f"Agregue {item['name']} a la playlist {playlist.get('name', playlist_query)}.", item=item)

    def remove_from_playlist(self, playlist_query: str, item_query: str) -> dict[str, Any]:
        """Remove every occurrence of the first matching track from a playlist."""

        playlist = self.find_user_playlist(playlist_query)
        item = self.search_first(item_query, "track")
        self._request("DELETE", f"/playlists/{playlist['id']}/items", json={"items": [{"uri": item["uri"]}]})
        return self._ok("spotify.remove_from_playlist", f"Quite {item['name']} de la playlist {playlist.get('name', playlist_query)}.", item=item)

    def request_delete_playlist(self, query: str) -> dict[str, Any]:
        """Resolve a playlist and request explicit confirmation before removal."""

        playlist = self.find_user_playlist(query)
        item = self._playlist_summary(playlist)
        return {
            "ok": False,
            "confirmation_required": True,
            "tool_name": "spotify.delete_playlist",
            "message": f"Confirma: quieres quitar la playlist {item['name']} de tu biblioteca? Di si, eliminar o cancelar.",
            "pending_action": {"name": "spotify.delete_playlist", "args": {"uri": item["uri"], "name": item["name"], "confirmed": True}},
        }

    def delete_playlist(self, uri: str, name: str) -> dict[str, Any]:
        """Remove a playlist from the user's library after confirmation."""

        self._request("DELETE", "/me/library", params={"uris": uri}, expect_json=False)
        return self._ok("spotify.delete_playlist", f"Quite la playlist {name} de tu biblioteca.")

    def _first_show_episode(self, show: dict[str, Any]) -> dict[str, Any]:
        """Resolve a podcast show to its first available episode."""

        show_id = str(show["uri"]).rsplit(":", 1)[-1]
        data = self._request("GET", f"/shows/{show_id}/episodes", params={"market": self.settings.spotify_market, "limit": 10})
        raw_episode = next((item for item in data.get("items", []) if item and item.get("uri") and item.get("is_playable", True)), None)
        if not raw_episode:
            raise SpotifyError(f"No encontre episodios reproducibles de {show['name']}.")
        episode = self._item_summary(raw_episode, "episode")
        if episode["subtitle"] == "Spotify":
            episode["subtitle"] = show["name"]
        return episode

    def _ok(self, tool_name: str, message: str, **details: Any) -> dict[str, Any]:
        """Build a consistent successful tool response."""

        return {"ok": True, "tool_name": tool_name, "message": message, **details}

    def _select_device(self) -> dict[str, Any]:
        """Choose the preferred Spotify Connect device."""

        devices = self.get_devices()
        if not devices:
            raise SpotifyError(
                "No hay dispositivos Spotify disponibles. Abre Spotify en este computador o en tu telefono."
            )

        preferred_name = (self.settings.spotify_device_name or "").strip().lower()
        if preferred_name:
            for device in devices:
                if device.get("name", "").strip().lower() == preferred_name:
                    return self._activate_device_if_needed(device)

        for device in devices:
            if device.get("is_active"):
                return device

        return self._activate_device_if_needed(devices[0])

    def _activate_device_if_needed(self, device: dict[str, Any]) -> dict[str, Any]:
        """Transfer playback to the selected device when it is inactive."""

        if device.get("is_active"):
            return device

        self._request(
            "PUT",
            "/me/player",
            json={"device_ids": [device["id"]], "play": False},
            expect_json=False,
        )
        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline:
            time.sleep(0.35)
            refreshed = next(
                (
                    candidate
                    for candidate in self.get_devices()
                    if candidate.get("id") == device.get("id")
                ),
                None,
            )
            if refreshed and refreshed.get("is_active"):
                return refreshed
        return device

    def _play_message(self, item: dict[str, Any], device: dict[str, Any]) -> str:
        """Build a short user-facing confirmation."""

        if item["type"] == "track":
            return f"Reproduciendo {item['name']} de {item['subtitle']} en {device['name']}."
        labels = {
            "artist": "musica del artista",
            "playlist": "la playlist",
            "album": "el album",
            "episode": "el episodio",
        }
        label = labels.get(item["type"], "")
        return f"Reproduciendo {label} {item['name']} de {item['subtitle']} en {device['name']}."

    def _item_summary(self, item: dict[str, Any], item_type: str) -> dict[str, Any]:
        """Return a stable representation for any supported Spotify item."""

        if item_type == "playlist":
            return self._playlist_summary(item)
        if item_type in {"track", "album"}:
            subtitle = ", ".join(artist.get("name", "") for artist in item.get("artists", []))
        elif item_type == "artist":
            subtitle = "Spotify"
        elif item_type == "show":
            subtitle = item.get("publisher") or "Spotify"
        elif item_type == "episode":
            show = item.get("show") or {}
            subtitle = show.get("name") or item.get("publisher") or "Spotify"
        else:
            subtitle = "Spotify"
        summary = {
            "type": item_type,
            "uri": item["uri"],
            "name": item.get("name", "Sin nombre"),
            "subtitle": subtitle or "Spotify",
        }
        if item_type == "track":
            album_uri = (item.get("album") or {}).get("uri")
            if album_uri:
                summary["context_uri"] = album_uri
        return summary

    def _type_label(self, item_type: str) -> str:
        """Return a short Spanish label for selection explanations."""

        return {
            "artist": "artista",
            "track": "cancion",
            "album": "album",
            "playlist": "playlist",
            "show": "podcast",
            "episode": "episodio",
        }.get(item_type, "resultado")

    def _playlist_summary(self, playlist: dict[str, Any]) -> dict[str, Any]:
        """Return a stable, compact playlist representation."""

        owner = playlist.get("owner", {}).get("display_name") or "Spotify"
        collection = playlist.get("items") or playlist.get("tracks") or {}
        tracks = collection.get("total", 0)
        return {
            "type": "playlist",
            "uri": playlist["uri"],
            "name": playlist.get("name", "Sin nombre"),
            "subtitle": owner,
            "tracks": tracks,
        }

    def _request(
        self,
        method: str,
        path: str,
        expect_json: bool = True,
        allow_no_content: bool = False,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Execute an authenticated Spotify API request."""

        url = f"{SPOTIFY_API_URL}{path}"
        headers = kwargs.pop("headers", {})
        headers["Authorization"] = f"Bearer {self.oauth.access_token()}"

        httpx = _httpx()
        # One reused client: a new one costs ~165 ms of SSL setup plus a new TLS handshake.
        if getattr(self, "_http", None) is None:
            self._http = httpx.Client()
        response = self._http.request(
            method,
            url,
            headers=headers,
            timeout=20,
            **kwargs,
        )
        if response.status_code == 401:
            token_data = self.oauth.token_store.load()
            refresh_token = token_data.get("refresh_token") if token_data else None
            if refresh_token:
                headers["Authorization"] = (
                    f"Bearer {self.oauth.refresh_access_token(str(refresh_token))['access_token']}"
                )
                response = self._http.request(
                    method,
                    url,
                    headers=headers,
                    timeout=20,
                    **kwargs,
                )

        if response.status_code == 403:
            detail = ""
            try:
                error_data = response.json().get("error", {})
                api_message = str(error_data.get("message", "")).strip()
                api_reason = str(error_data.get("reason", "")).strip()
                if api_message:
                    detail = api_message
                    if api_reason and api_reason != "UNKNOWN":
                        detail += f" ({api_reason})"
            except (ValueError, AttributeError):
                pass
            if "restriction violated" in detail.lower():
                raise SpotifyError(
                    "Spotify autorizo la cuenta, pero el dispositivo activo rechazo este control "
                    f"con una restriccion del reproductor: {detail}."
                )
            raise SpotifyError(
                "Spotify rechazo la operacion. Revisa la cuenta Premium y los permisos OAuth."
                + (f" Detalle: {detail}." if detail else "")
            )
        if response.status_code == 404:
            if not path.startswith("/me/player"):
                raise SpotifyError(
                    f"Spotify no encontro el recurso solicitado en {path}."
                )
            raise SpotifyError(
                "Spotify no encontro un dispositivo activo. Abre Spotify y vuelve a intentarlo."
            )

        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as error:
            raise SpotifyError(
                f"Spotify fallo con HTTP {response.status_code}: {response.text[:200]}"
            ) from error

        if allow_no_content and response.status_code == 204:
            return {}
        if not expect_json or not response.content:
            return {}
        return response.json()


class SpotifyToolsExecutor:
    """Tool executor for Spotify capabilities."""

    supports_model_tools = True
    supports_spotify_tools = True

    def __init__(self, settings: Settings) -> None:
        """Build the Spotify API client."""

        self.client = SpotifyClient(settings)

    def prefetch(self, tool_name: str, args: dict[str, object]) -> None:
        """Warm read-only lookups while the spoken confirmation plays."""

        self.client.prefetch(tool_name, args)

    def execute(self, tool_name: str, args: dict[str, object]) -> dict[str, object]:
        """Execute a named Spotify tool."""
        try:
            return self._execute(tool_name, args)
        except (SpotifyError, ValueError, TypeError) as error:
            return {
                "ok": False,
                "tool_name": tool_name,
                "message": str(error),
            }
        finally:
            # Unused lookups belong to this command only.
            getattr(self.client, "_prefetched", {}).clear()

    def _execute(self, tool_name: str, args: dict[str, object]) -> dict[str, object]:
        """Dispatch a validated action to the Spotify client."""

        if tool_name == "spotify.list_playlists":
            return self.client.list_user_playlists(limit=int(args.get("limit", 10)))
        if tool_name == "spotify.play_playlist":
            return self.client.play_user_playlist(
                query=self._required(args, "query"),
                shuffle=bool(args.get("shuffle", False)),
            )
        if tool_name == "spotify.play":
            return self.client.play_search(
                query=self._required(args, "query"),
                item_type=str(args.get("item_type", "auto")),
                shuffle=bool(args.get("shuffle", False)),
            )
        if tool_name == "spotify.play_reference":
            return self.client.play_reference(
                self._required(args, "reference"),
                shuffle=bool(args.get("shuffle", False)),
            )
        if tool_name == "spotify.search":
            return self.client.list_search(query=self._required(args, "query"), item_type=str(args.get("item_type", "track")))
        if tool_name == "spotify.play_item":
            item = args.get("item")
            if not isinstance(item, dict):
                raise SpotifyError("No tengo un resultado anterior que pueda reproducir.")
            return self.client.play_item(item)
        if tool_name == "spotify.pause":
            return self.client.pause()
        if tool_name == "spotify.resume":
            return self.client.resume()
        if tool_name == "spotify.next":
            return self.client.skip("next")
        if tool_name == "spotify.previous":
            return self.client.skip("previous")
        if tool_name == "spotify.current":
            return self.client.currently_playing()
        if tool_name == "spotify.volume":
            absolute = args.get("volume_percent")
            delta = args.get("delta")
            return self.client.set_volume(
                volume_percent=int(absolute) if absolute is not None else None,
                delta=int(delta) if delta is not None else None,
            )
        if tool_name == "spotify.shuffle":
            return self.client.set_shuffle(bool(args.get("state")))
        if tool_name == "spotify.repeat":
            return self.client.set_repeat(str(args.get("state", "off")))
        if tool_name == "spotify.queue":
            return self.client.add_to_queue(self._required(args, "query"), str(args.get("item_type", "track")))
        if tool_name == "spotify.save":
            return self.client.save_search(self._required(args, "query"), str(args.get("item_type", "track")))
        if tool_name == "spotify.save_current":
            return self.client.save_current()
        if tool_name == "spotify.play_similar":
            return self.client.play_similar()
        if tool_name == "spotify.create_playlist":
            return self.client.create_playlist(
                self._required(args, "name"),
                public=bool(args.get("public", False)),
                description=str(args.get("description", "")),
            )
        if tool_name == "spotify.update_playlist":
            query = self._required(args, "query")
            return self.client.update_playlist(query, dict(args))
        if tool_name == "spotify.add_to_playlist":
            return self.client.add_to_playlist(self._required(args, "playlist_query"), self._required(args, "item_query"))
        if tool_name == "spotify.remove_from_playlist":
            return self.client.remove_from_playlist(self._required(args, "playlist_query"), self._required(args, "item_query"))
        if tool_name == "spotify.delete_playlist":
            if bool(args.get("confirmed")):
                return self.client.delete_playlist(self._required(args, "uri"), self._required(args, "name"))
            return self.client.request_delete_playlist(self._required(args, "query"))
        raise SpotifyError(f"Spotify no conoce la tool: {tool_name}")

    @staticmethod
    def _required(args: dict[str, object], key: str) -> str:
        """Return a required non-empty text argument."""

        value = str(args.get(key, "")).strip()
        if not value:
            raise SpotifyError(f"Falta el dato requerido: {key}.")
        return value


def _normalize_match_text(text: str) -> str:
    """Normalize names for forgiving playlist matching."""

    without_accents = "".join(
        char
        for char in unicodedata.normalize("NFKD", text)
        if not unicodedata.combining(char)
    )
    return " ".join(without_accents.casefold().split())
