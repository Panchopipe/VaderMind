"""Validated orchestration spanning the local LG TV and Spotify clients."""

from __future__ import annotations

from typing import Any

from vadermind.config import Settings
from vadermind.tools.lg_tv import LgTvController, LgTvError
from vadermind.tools.spotify import SpotifyClient, SpotifyError


class MediaToolsExecutor:
    """Coordinate only the explicitly supported Spotify-to-TV workflow."""

    supports_model_tools = True
    supports_media_tools = True

    def __init__(
        self,
        settings: Settings,
        *,
        tv: LgTvController,
        spotify: SpotifyClient,
    ) -> None:
        self.tv = tv
        self.spotify = spotify
        self.preferred_tv_device = settings.spotify_tv_device_name

    def execute(self, tool_name: str, args: dict[str, object]) -> dict[str, object]:
        try:
            if tool_name != "media.spotify_on_tv":
                raise ValueError(f"Media no conoce la tool: {tool_name}")
            query = self._required_text(args, "query")
            item_type = str(args.get("item_type", "auto"))
            if item_type not in {
                "auto",
                "track",
                "artist",
                "playlist",
                "album",
                "show",
                "episode",
            }:
                raise ValueError("El tipo de contenido de Spotify no esta permitido.")
            shuffle = args.get("shuffle", False)
            if not isinstance(shuffle, bool):
                raise ValueError("El estado de reproduccion aleatoria no es valido.")
            library_only = args.get("library_only", False)
            if not isinstance(library_only, bool):
                raise ValueError("El origen de la playlist no es valido.")
            if library_only and item_type not in {"auto", "playlist"}:
                raise ValueError(
                    "library_only solo es valido para playlists personales."
                )
            return self._play_on_tv(
                query,
                item_type=item_type,
                shuffle=shuffle,
                library_only=library_only,
            )
        except (LgTvError, SpotifyError, ValueError, TypeError) as error:
            return {"ok": False, "tool_name": tool_name, "message": str(error)}

    def _play_on_tv(
        self, query: str, *, item_type: str, shuffle: bool, library_only: bool
    ) -> dict[str, Any]:
        before_devices = self.spotify.get_devices()
        previous_ids = {
            str(device.get("id"))
            for device in before_devices
            if device.get("id")
        }

        state = self.tv.status()
        if state.get("reachable") is False:
            if not self.tv.power_on():
                raise LgTvError(
                    "No pude confirmar que la television despertara para abrir Spotify."
                )
        self.tv.launch_app("Spotify")
        device = self.spotify.wait_for_tv_device(
            preferred_name=self.preferred_tv_device,
            previous_device_ids=previous_ids,
        )
        if library_only:
            result = self.spotify.play_user_playlist_on_device(
                query,
                device,
                shuffle=shuffle,
            )
        else:
            result = self.spotify.play_search_on_device(
                query,
                device,
                item_type=item_type,
                shuffle=shuffle,
            )
        try:
            tv_state = self.tv.status()
        except LgTvError:
            return result
        result["tv_state"] = tv_state
        tv_volume = tv_state.get("volume")
        if tv_volume is not None:
            result["message"] = (
                str(result.get("message", "")).rstrip(".")
                + f"; la television informa volumen {tv_volume}."
            )
        return result

    @staticmethod
    def _required_text(args: dict[str, object], key: str) -> str:
        value = args.get(key)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"Falta el dato requerido: {key}.")
        cleaned = " ".join(value.split())
        if len(cleaned) > 200:
            raise ValueError("La busqueda de Spotify es demasiado larga.")
        return cleaned
