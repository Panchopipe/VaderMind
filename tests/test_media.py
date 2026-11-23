"""Tests for the bounded Spotify-to-LG-TV orchestration."""

from __future__ import annotations

from types import SimpleNamespace
import unittest

from vadermind.tools.media import MediaToolsExecutor


class RecordingTv:
    def __init__(self, reachable: bool = True) -> None:
        self.reachable = reachable
        self.calls: list[tuple[str, object]] = []

    def status(self) -> dict[str, object]:
        self.calls.append(("status", None))
        return {"reachable": self.reachable, "volume": 20}

    def power_on(self) -> bool:
        self.calls.append(("power_on", None))
        self.reachable = True
        return True

    def launch_app(self, name: str) -> str:
        self.calls.append(("launch_app", name))
        return name


class RecordingSpotify:
    def __init__(self) -> None:
        self.calls: list[tuple[str, object]] = []

    def get_devices(self) -> list[dict[str, object]]:
        self.calls.append(("get_devices", None))
        return [{"id": "phone", "name": "Telefono"}]

    def wait_for_tv_device(self, **kwargs: object) -> dict[str, object]:
        self.calls.append(("wait_for_tv_device", kwargs))
        return {"id": "tv", "name": "LG webOS TV", "type": "TV"}

    def play_search_on_device(
        self,
        query: str,
        device: dict[str, object],
        **kwargs: object,
    ) -> dict[str, object]:
        self.calls.append(
            ("play_search_on_device", {"query": query, "device": device, **kwargs})
        )
        return {
            "ok": True,
            "tool_name": "media.spotify_on_tv",
            "message": "Spotify confirmo Imperial March en LG webOS TV, con volumen 25.",
        }

    def play_user_playlist_on_device(
        self,
        query: str,
        device: dict[str, object],
        **kwargs: object,
    ) -> dict[str, object]:
        self.calls.append(
            (
                "play_user_playlist_on_device",
                {"query": query, "device": device, **kwargs},
            )
        )
        return {
            "ok": True,
            "tool_name": "media.spotify_on_tv",
            "message": "Spotify confirmo la playlist Timba en LG webOS TV.",
        }


class MediaToolsTests(unittest.TestCase):
    def test_spotify_tv_workflow_wakes_opens_waits_plays_and_confirms(self) -> None:
        tv = RecordingTv(reachable=False)
        spotify = RecordingSpotify()
        executor = MediaToolsExecutor(
            SimpleNamespace(spotify_tv_device_name=None),
            tv=tv,  # type: ignore[arg-type]
            spotify=spotify,  # type: ignore[arg-type]
        )

        result = executor.execute(
            "media.spotify_on_tv",
            {"query": "Imperial March", "item_type": "track", "shuffle": False},
        )

        self.assertTrue(result["ok"])
        self.assertIn("confirmo", str(result["message"]))
        self.assertIn("volumen 20", str(result["message"]))
        self.assertEqual(
            tv.calls,
            [
                ("status", None),
                ("power_on", None),
                ("launch_app", "Spotify"),
                ("status", None),
            ],
        )
        self.assertEqual(spotify.calls[0], ("get_devices", None))
        self.assertEqual(spotify.calls[-1][0], "play_search_on_device")

    def test_media_executor_fails_closed_for_unknown_operations(self) -> None:
        executor = MediaToolsExecutor(
            SimpleNamespace(spotify_tv_device_name=None),
            tv=RecordingTv(),  # type: ignore[arg-type]
            spotify=RecordingSpotify(),  # type: ignore[arg-type]
        )

        result = executor.execute("media.raw", {"uri": "something"})

        self.assertFalse(result["ok"])

    def test_personal_playlist_uses_library_lookup_on_tv(self) -> None:
        spotify = RecordingSpotify()
        executor = MediaToolsExecutor(
            SimpleNamespace(spotify_tv_device_name=None),
            tv=RecordingTv(),  # type: ignore[arg-type]
            spotify=spotify,  # type: ignore[arg-type]
        )

        result = executor.execute(
            "media.spotify_on_tv",
            {
                "query": "Timba",
                "item_type": "playlist",
                "library_only": True,
            },
        )

        self.assertTrue(result["ok"])
        self.assertEqual(spotify.calls[-1][0], "play_user_playlist_on_device")


if __name__ == "__main__":
    unittest.main()
