"""Unit tests for expanded Spotify Web API capabilities."""

from __future__ import annotations

import time
from types import SimpleNamespace
import unittest
from unittest import mock

from vadermind.tools.spotify import SpotifyClient, SpotifyError, SpotifyToolsExecutor


class RecordingSpotifyClient(SpotifyClient):
    """Spotify client test double that records requests without network access."""

    def __init__(self) -> None:
        self.settings = SimpleNamespace(spotify_market="CL", spotify_device_name=None)
        self.requests: list[tuple[str, str, dict[str, object]]] = []
        self.responses: dict[str, dict[str, object]] = {}

    def _select_device(self) -> dict[str, object]:
        return {"id": "device-1", "name": "Living Room", "volume_percent": 40, "supports_volume": True}

    def _request(self, method: str, path: str, expect_json: bool = True, allow_no_content: bool = False, **kwargs):
        self.requests.append((method, path, kwargs))
        return self.responses.get(path, {})


class PrefetchingSpotifyClient(SpotifyClient):
    """Real device selection and search, with a fake transport that records calls."""

    def __init__(self) -> None:
        self.settings = SimpleNamespace(spotify_market="CL", spotify_device_name=None)
        self.requests: list[tuple[str, str]] = []

    def _request(self, method: str, path: str, expect_json: bool = True, allow_no_content: bool = False, **kwargs):
        self.requests.append((method, path))
        if path == "/me/player/devices":
            return {"devices": [{"id": "d1", "name": "PC", "is_active": True}]}
        if path == "/search":
            return {"artists": {"items": [{"uri": "spotify:artist:m", "name": "Metallica", "type": "artist"}]}}
        return {}


class SpotifyPrefetchTests(unittest.TestCase):
    def test_prefetch_only_reads_and_execute_reuses_its_results(self) -> None:
        client = PrefetchingSpotifyClient()
        client.prefetch("spotify.play", {"query": "Metallica"})

        # While Vader speaks: only GET requests, never a playback change.
        self.assertTrue(all(method == "GET" for method, _ in client.requests))
        self.assertEqual(
            [path for _, path in client.requests], ["/search", "/me/player/devices"]
        )

        client.requests.clear()
        client.play_search("Metallica")
        # After speaking: no repeated lookups, straight to playback.
        self.assertEqual(client.requests, [("PUT", "/me/player/play")])

    def test_stale_prefetch_is_ignored(self) -> None:
        client = PrefetchingSpotifyClient()
        client.prefetch("spotify.next", {})
        with mock.patch("vadermind.tools.spotify.time.monotonic", return_value=time.monotonic() + 60):
            client.requests.clear()
            client.skip("next")
        self.assertEqual(
            client.requests, [("GET", "/me/player/devices"), ("POST", "/me/player/next")]
        )

    def test_prefetch_errors_are_swallowed(self) -> None:
        client = PrefetchingSpotifyClient()
        with mock.patch.object(client, "_request", side_effect=SpotifyError("sin red")):
            client.prefetch("spotify.play", {"query": "Metallica"})  # must not raise

    def test_executor_discards_unused_prefetch_after_each_command(self) -> None:
        executor = SpotifyToolsExecutor.__new__(SpotifyToolsExecutor)
        executor.client = PrefetchingSpotifyClient()
        executor.prefetch("spotify.play", {"query": "Metallica"})
        executor.execute("spotify.pause", {})
        self.assertEqual(executor.client._prefetched, {})


class SpotifyCapabilityTests(unittest.TestCase):
    """Verify commands map to the current Spotify endpoints and payloads."""

    def test_transport_controls_use_player_endpoints(self) -> None:
        client = RecordingSpotifyClient()

        client.pause()
        client.resume()
        client.skip("next")
        client.skip("previous")

        self.assertEqual(
            [(method, path) for method, path, _ in client.requests],
            [
                ("PUT", "/me/player/pause"),
                ("PUT", "/me/player/play"),
                ("POST", "/me/player/next"),
                ("POST", "/me/player/previous"),
            ],
        )

    def test_pause_accepts_verified_state_after_contradictory_restriction(self) -> None:
        """A false-negative 403 should not become a spoken error if playback stopped."""

        client = RecordingSpotifyClient()
        with mock.patch.object(
            client,
            "_request",
            side_effect=[
                SpotifyError("Restriction violated"),
                {"is_playing": False, "item": {"name": "Track"}},
            ],
        ):
            result = client.pause()

        self.assertTrue(result["ok"])
        self.assertIn("pausada", result["message"])

    def test_volume_shuffle_and_repeat_send_valid_values(self) -> None:
        client = RecordingSpotifyClient()

        client.set_volume(delta=10)
        client.set_shuffle(True)
        client.set_repeat("track")

        self.assertEqual(client.requests[0][2]["params"]["volume_percent"], 50)
        self.assertEqual(client.requests[1][2]["params"]["state"], "true")
        self.assertEqual(client.requests[2][2]["params"]["state"], "track")

    def test_current_track_can_be_saved_with_generic_library_endpoint(self) -> None:
        client = RecordingSpotifyClient()
        client.responses["/me/player/currently-playing"] = {
            "is_playing": True,
            "item": {
                "type": "track",
                "uri": "spotify:track:1",
                "name": "One",
                "artists": [{"name": "Metallica"}],
            },
        }

        result = client.save_current()

        self.assertTrue(result["ok"])
        self.assertEqual(client.requests[-1][0:2], ("PUT", "/me/library"))
        self.assertEqual(client.requests[-1][2]["params"]["uris"], "spotify:track:1")

    def test_queue_searches_a_track_and_adds_its_uri(self) -> None:
        client = RecordingSpotifyClient()
        client.responses["/search"] = {
            "tracks": {"items": [{"type": "track", "uri": "spotify:track:1", "name": "One", "artists": [{"name": "U2"}]}]}
        }

        result = client.add_to_queue("One")

        self.assertTrue(result["ok"])
        queue_request = client.requests[-1]
        self.assertEqual(queue_request[0:2], ("POST", "/me/player/queue"))
        self.assertEqual(queue_request[2]["params"]["uri"], "spotify:track:1")

    def test_album_search_returns_five_candidates_for_followup(self) -> None:
        client = RecordingSpotifyClient()
        client.responses["/search"] = {
            "albums": {
                "items": [
                    {"type": "album", "uri": f"spotify:album:{index}", "name": f"Album {index}", "artists": [{"name": "Artist"}]}
                    for index in range(1, 6)
                ]
            }
        }

        result = client.list_search("Album", "album")

        self.assertEqual(len(result["candidates"]), 5)
        self.assertIn("2. Album 2", result["message"])

    def test_auto_search_prefers_exact_artist_and_plays_continuous_context(self) -> None:
        """A bare artist name should play the artist context, not one isolated track."""

        client = RecordingSpotifyClient()
        client.responses["/search"] = {
            "tracks": {
                "items": [
                    {
                        "type": "track",
                        "uri": "spotify:track:querida",
                        "name": "Querida",
                        "artists": [{"name": "Juan Gabriel"}],
                        "album": {"uri": "spotify:album:album1"},
                    }
                ]
            },
            "artists": {
                "items": [
                    {
                        "type": "artist",
                        "uri": "spotify:artist:juan",
                        "name": "Juan Gabriel",
                    }
                ]
            },
            "albums": {"items": []},
            "playlists": {"items": []},
        }

        result = client.play_search("Juan Gabriel")

        self.assertEqual(result["item"]["type"], "artist")
        play_request = client.requests[-1]
        self.assertEqual(
            play_request[2]["json"], {"context_uri": "spotify:artist:juan"}
        )

    def test_track_playback_uses_album_context_at_requested_track(self) -> None:
        """A selected song should continue through its album after the requested track."""

        client = RecordingSpotifyClient()
        item = {
            "type": "track",
            "uri": "spotify:track:querida",
            "name": "Querida",
            "subtitle": "Juan Gabriel",
            "context_uri": "spotify:album:album1",
        }

        client.play_item(item)

        self.assertEqual(
            client.requests[-1][2]["json"],
            {
                "context_uri": "spotify:album:album1",
                "offset": {"uri": "spotify:track:querida"},
            },
        )

    def test_playlist_mutations_use_2026_items_and_library_endpoints(self) -> None:
        client = RecordingSpotifyClient()
        playlist = {"id": "p1", "uri": "spotify:playlist:p1", "name": "Prueba", "owner": {"display_name": "Me"}, "items": {"total": 0}}
        client.responses["/me/playlists"] = {"items": [playlist], "next": None}
        client.responses["/search"] = {
            "tracks": {"items": [{"type": "track", "uri": "spotify:track:t1", "name": "One", "artists": [{"name": "U2"}]}]}
        }

        client.add_to_playlist("Prueba", "One")
        client.remove_from_playlist("Prueba", "One")
        client.delete_playlist("spotify:playlist:p1", "Prueba")

        endpoints = [(method, path) for method, path, _ in client.requests]
        self.assertIn(("POST", "/playlists/p1/items"), endpoints)
        self.assertIn(("DELETE", "/playlists/p1/items"), endpoints)
        self.assertIn(("DELETE", "/me/library"), endpoints)

    def test_saved_playlist_can_start_with_shuffle_enabled(self) -> None:
        """Shuffle should be enabled before the requested playlist starts."""

        client = RecordingSpotifyClient()
        client.responses["/me/playlists"] = {
            "items": [
                {
                    "id": "p1",
                    "uri": "spotify:playlist:p1",
                    "name": "New UK Garage",
                    "owner": {"display_name": "Francisco"},
                    "items": {"total": 20},
                }
            ],
            "next": None,
        }

        result = client.play_user_playlist("New UK Garage", shuffle=True)

        commands = [(method, path) for method, path, _ in client.requests]
        shuffle_index = commands.index(("PUT", "/me/player/shuffle"))
        play_index = commands.index(("PUT", "/me/player/play"))
        self.assertLess(shuffle_index, play_index)
        self.assertIn("Modo aleatorio activado", result["message"])

    def test_catalog_playlist_can_start_with_shuffle_enabled(self) -> None:
        """A model-selected catalog playlist should preserve its shuffle argument."""

        client = RecordingSpotifyClient()
        client.responses["/search"] = {
            "playlists": {
                "items": [
                    {
                        "id": "p1",
                        "uri": "spotify:playlist:p1",
                        "name": "Juan Gabriel Exitos",
                        "owner": {"display_name": "Spotify"},
                        "items": {"total": 40},
                    }
                ]
            }
        }

        result = client.play_search(
            "Juan Gabriel", item_type="playlist", shuffle=True
        )

        commands = [(method, path) for method, path, _ in client.requests]
        self.assertLess(
            commands.index(("PUT", "/me/player/shuffle")),
            commands.index(("PUT", "/me/player/play")),
        )
        self.assertIn("Modo aleatorio activado", result["message"])

    def test_playlist_lookup_is_not_limited_to_first_api_page(self) -> None:
        """A saved playlist beyond the first 50 entries should remain discoverable."""

        client = RecordingSpotifyClient()
        expected = {
            "id": "p51",
            "uri": "spotify:playlist:p51",
            "name": "New UK Garage",
        }
        playlists = [
            {"id": f"p{index}", "uri": f"spotify:playlist:p{index}", "name": f"List {index}"}
            for index in range(50)
        ] + [expected]

        with mock.patch.object(
            client, "get_user_playlists", return_value=playlists
        ) as get_playlists:
            result = client.find_user_playlist("New UK Garage")

        get_playlists.assert_called_once_with(limit=1000)
        self.assertEqual(result, expected)

    def test_playlist_lookup_uses_local_alias_when_api_omits_playlist(self) -> None:
        """Public editorial playlists can be resolved from a remembered URI."""

        client = RecordingSpotifyClient()
        client.settings.spotify_playlist_aliases_path = "aliases.json"
        with (
            mock.patch("vadermind.tools.spotify.Path.is_file", return_value=True),
            mock.patch(
                "vadermind.tools.spotify.Path.read_text",
                return_value=(
                    '{"New UK Garage": '
                    '"spotify:playlist:37i9dQZF1DX6MNYaY0PcFh"}'
                ),
            ),
            mock.patch.object(client, "get_user_playlists", return_value=[]),
        ):
            result = client.find_user_playlist("New UK Garage")

        self.assertEqual(result["name"], "New UK Garage")
        self.assertEqual(result["uri"], "spotify:playlist:37i9dQZF1DX6MNYaY0PcFh")

    def test_direct_playlist_link_plays_uri_and_remembers_title(self) -> None:
        """A direct link should work even when GET playlist metadata returns 404."""

        client = RecordingSpotifyClient()
        with mock.patch.object(client, "_reference_title", return_value="New UK Garage"), mock.patch.object(
            client, "_remember_playlist_alias"
        ) as remember:
            result = client.play_reference(
                "https://open.spotify.com/playlist/37i9dQZF1DX6MNYaY0PcFh?si=test"
            )

        remember.assert_called_once_with(
            "New UK Garage", "spotify:playlist:37i9dQZF1DX6MNYaY0PcFh"
        )
        play_request = client.requests[-1]
        self.assertEqual(play_request[0:2], ("PUT", "/me/player/play"))
        self.assertEqual(
            play_request[2]["json"]["context_uri"],
            "spotify:playlist:37i9dQZF1DX6MNYaY0PcFh",
        )
        self.assertTrue(result["ok"])

    def test_wait_for_tv_device_prefers_a_visible_tv(self) -> None:
        client = RecordingSpotifyClient()
        client.responses["/me/player/devices"] = {
            "devices": [
                {"id": "phone", "name": "Telefono", "type": "Smartphone"},
                {"id": "tv", "name": "LG webOS TV", "type": "TV"},
            ]
        }

        device = client.wait_for_tv_device(timeout_seconds=0.1)

        self.assertEqual(device["id"], "tv")

    def test_play_on_explicit_tv_device_requires_playback_confirmation(self) -> None:
        client = RecordingSpotifyClient()
        selected = {
            "type": "track",
            "uri": "spotify:track:1",
            "name": "Imperial March",
            "subtitle": "John Williams",
        }
        playback = {
            "is_playing": True,
            "device": {
                "id": "tv",
                "name": "LG webOS TV",
                "type": "TV",
                "volume_percent": 25,
            },
            "item": {
                "name": "Imperial March",
                "artists": [{"name": "John Williams"}],
            },
        }
        with (
            mock.patch.object(client, "search_items", return_value=[selected]),
            mock.patch.object(client, "_activate_device_if_needed", side_effect=lambda value: value),
            mock.patch.object(client, "_wait_for_playback_on_device", return_value=playback),
        ):
            result = client.play_search_on_device(
                "Imperial March",
                {"id": "tv", "name": "LG webOS TV", "type": "TV"},
                item_type="track",
            )

        self.assertTrue(result["ok"])
        self.assertNotIn("volumen 25", result["message"])
        self.assertEqual(result["device"]["spotify_volume_percent"], 25)
        self.assertIn(("PUT", "/me/player/play"), [(a, b) for a, b, _ in client.requests])


if __name__ == "__main__":
    unittest.main()
