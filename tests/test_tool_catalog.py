"""Tests for strict model-tool mapping into internal Spotify operations."""

from __future__ import annotations

import unittest

from vadermind.tools.catalog import (
    capability_model_tools,
    information_model_tools,
    lg_tv_model_tools,
    resolve_capability_tool_call,
    resolve_information_tool_call,
    resolve_lg_tv_tool_call,
    resolve_model_tool_call,
    spotify_model_tools,
)


class SpotifyToolCatalogTests(unittest.TestCase):
    def test_saved_playlist_maps_to_library_playback(self) -> None:
        call = resolve_model_tool_call(
            "spotify_play",
            {
                "query": "Jazz",
                "item_type": "playlist",
                "library_only": True,
                "shuffle": True,
            },
        )

        self.assertEqual(call.name, "spotify.play_playlist")
        self.assertEqual(call.args, {"query": "Jazz", "shuffle": True})

    def test_catalog_playlist_stays_a_typed_catalog_search(self) -> None:
        call = resolve_model_tool_call(
            "spotify_play",
            {
                "query": "Juan Gabriel",
                "item_type": "playlist",
                "library_only": False,
                "shuffle": False,
            },
        )

        self.assertEqual(call.name, "spotify.play")
        self.assertEqual(call.args["item_type"], "playlist")

    def test_unknown_or_invalid_model_calls_fail_closed(self) -> None:
        with self.assertRaises(ValueError):
            resolve_model_tool_call("spotify_destroy_account", {})
        with self.assertRaises(ValueError):
            resolve_model_tool_call(
                "spotify_control", {"action": "set_volume", "volume_percent": 500}
            )

    def test_catalog_exposes_only_known_function_names(self) -> None:
        names = {
            str(item["function"]["name"])
            for item in spotify_model_tools()
        }

        self.assertIn("spotify_play", names)
        self.assertIn("spotify_playlist_manage", names)
        self.assertNotIn("spotify_destroy_account", names)

    def test_information_catalog_contains_only_read_operations(self) -> None:
        names = {
            str(item["function"]["name"])
            for item in information_model_tools()
        }

        self.assertEqual(
            names,
            {
                "information_time",
                "information_weather",
                "information_sports_query",
                "information_web_query",
            },
        )
        self.assertFalse(any("buy" in name or "login" in name for name in names))

    def test_information_calls_are_strictly_mapped(self) -> None:
        call = resolve_information_tool_call(
            "information_weather", {"location": "Santiago", "when": "tomorrow"}
        )

        self.assertEqual(call.name, "information.weather")
        today = resolve_information_tool_call(
            "information_weather", {"location": "Santiago", "when": "today"}
        )
        self.assertEqual(today.args["when"], "today")
        sports = resolve_information_tool_call(
            "information_sports_query",
            {
                "intent": "event_result",
                "sport": "american_football",
                "event": "Super Bowl",
                "competition": "NFL",
                "date_reference": "most recent",
            },
        )
        self.assertEqual(sports.name, "information.sports_query")
        self.assertEqual(sports.args["event"], "Super Bowl")
        rugby = resolve_information_tool_call(
            "information_sports_query",
            {
                "intent": "latest_result",
                "sport": "rugby",
                "participant": "Chile XV",
                "competition": "Americas Rugby Championship",
                "country": "Chile",
                "date_reference": "ultimo partido",
            },
        )
        self.assertEqual(rugby.args["sport"], "rugby")
        competition = resolve_information_tool_call(
            "information_sports_query",
            {
                "intent": "latest_result",
                "sport": "football",
                "participant": "unspecified",
                "event": "final",
                "competition": "UEFA Champions League",
                "country": "unspecified",
                "date_reference": "2026",
            },
        )
        self.assertNotIn("participant", competition.args)
        self.assertNotIn("country", competition.args)
        competition_only = resolve_information_tool_call(
            "information_sports_query",
            {
                "intent": "event_result",
                "sport": "football",
                "competition": "UEFA Champions League",
                "date_reference": "ultimo campeon",
            },
        )
        self.assertEqual(
            competition_only.args["competition"], "UEFA Champions League"
        )
        with self.assertRaises(ValueError):
            resolve_information_tool_call("information_purchase", {"item": "algo"})

    def test_capability_router_combines_safe_information_and_optional_spotify(self) -> None:
        without_spotify = {
            str(item["function"]["name"])
            for item in capability_model_tools(include_spotify=False)
        }
        with_spotify = {
            str(item["function"]["name"])
            for item in capability_model_tools(include_spotify=True)
        }

        self.assertIn("information_time", without_spotify)
        self.assertIn("information_weather", without_spotify)
        self.assertNotIn("information_deferred", without_spotify)
        self.assertIn("information_sports_query", without_spotify)
        self.assertIn("information_web_query", without_spotify)
        self.assertIn("external_action_blocked", without_spotify)
        self.assertNotIn("spotify_play", without_spotify)
        self.assertIn("spotify_play", with_spotify)
        self.assertFalse(any("purchase_execute" in name for name in with_spotify))

    def test_capability_router_validates_information_and_blocked_actions(self) -> None:
        deferred = resolve_capability_tool_call(
            "information_deferred",
            {
                "query": "calendario de los Juegos Olimpicos de hoy",
                "category": "sports",
            },
        )
        legacy_web = resolve_capability_tool_call(
            "information_web_query",
            {
                "query": "valor vigente del abono de transporte",
                "category": "current_fact",
            },
        )
        blocked = resolve_capability_tool_call(
            "external_action_blocked", {"action": "booking"}
        )

        self.assertEqual(deferred.name, "information.deferred")
        self.assertEqual(deferred.args["category"], "sports")
        self.assertEqual(legacy_web.name, "information.web_query")
        self.assertEqual(legacy_web.args["category"], "current_fact")
        self.assertEqual(blocked.name, "external.blocked")
        with self.assertRaises(ValueError):
            resolve_capability_tool_call(
                "external_action_blocked", {"action": "transfer_money"}
            )

    def test_lg_tv_catalog_exposes_only_bounded_operations(self) -> None:
        names = {str(item["function"]["name"]) for item in lg_tv_model_tools()}

        self.assertEqual(
            names,
            {
                "lg_tv_status",
                "lg_tv_power",
                "lg_tv_volume",
                "lg_tv_mute",
                "lg_tv_launch_app",
                "lg_tv_select_input",
                "lg_tv_playback",
                "lg_tv_navigation",
                "lg_tv_channel",
                "lg_tv_screen",
                "lg_tv_show_message",
                "lg_tv_timer",
            },
        )
        self.assertNotIn("lg_tv_raw_command", names)

    def test_lg_tv_model_calls_are_strictly_mapped(self) -> None:
        status = resolve_lg_tv_tool_call(
            "lg_tv_status", {"detail": "channel"}
        )
        self.assertEqual(status.args, {"detail": "channel"})
        default_status = resolve_lg_tv_tool_call("lg_tv_status", {})
        self.assertEqual(default_status.args, {"detail": "full"})
        self.assertEqual(
            resolve_lg_tv_tool_call("lg_tv_power", {"action": "off"}).name,
            "lg_tv.power_off",
        )
        volume = resolve_lg_tv_tool_call(
            "lg_tv_volume", {"action": "set", "volume": 27}
        )
        self.assertEqual(volume.args, {"volume": 27})
        input_call = resolve_lg_tv_tool_call(
            "lg_tv_select_input", {"input": "HDMI 2"}
        )
        self.assertEqual(input_call.name, "lg_tv.select_input")
        playback = resolve_lg_tv_tool_call(
            "lg_tv_playback", {"action": "pause"}
        )
        navigation = resolve_lg_tv_tool_call(
            "lg_tv_navigation", {"action": "close_app"}
        )
        channel = resolve_lg_tv_tool_call(
            "lg_tv_channel", {"action": "up"}
        )
        selected_channel = resolve_lg_tv_tool_call(
            "lg_tv_channel", {"action": "select", "query": "T13"}
        )
        timer = resolve_lg_tv_tool_call(
            "lg_tv_timer", {"action": "power_off", "minutes": 30}
        )
        screen = resolve_lg_tv_tool_call("lg_tv_screen", {"on": False})
        message = resolve_lg_tv_tool_call(
            "lg_tv_show_message", {"message": "  Hola   Vader  "}
        )
        self.assertEqual(playback.name, "lg_tv.playback")
        self.assertEqual(navigation.name, "lg_tv.close_app")
        self.assertEqual(channel.name, "lg_tv.channel_up")
        self.assertEqual(selected_channel.name, "lg_tv.select_channel")
        self.assertEqual(timer.name, "lg_tv.schedule_power_off")
        self.assertEqual(screen.name, "lg_tv.screen_off")
        self.assertEqual(message.args, {"message": "Hola Vader"})
        with self.assertRaises(ValueError):
            resolve_lg_tv_tool_call(
                "lg_tv_volume", {"action": "set", "volume": 500}
            )
        with self.assertRaises(ValueError):
            resolve_lg_tv_tool_call(
                "lg_tv_select_input", {"input": "AV 1"}
            )
        with self.assertRaises(ValueError):
            resolve_lg_tv_tool_call("lg_tv_raw_command", {"uri": "system/turnOff"})
        with self.assertRaises(ValueError):
            resolve_lg_tv_tool_call(
                "lg_tv_navigation", {"action": "service_menu"}
            )
        with self.assertRaises(ValueError):
            resolve_lg_tv_tool_call(
                "lg_tv_show_message", {"message": "x" * 161}
            )
        with self.assertRaises(ValueError):
            resolve_lg_tv_tool_call(
                "lg_tv_status", {"detail": "everything"}
            )

    def test_capability_router_can_include_lg_tv_without_spotify(self) -> None:
        names = {
            str(item["function"]["name"])
            for item in capability_model_tools(
                include_spotify=False, include_lg_tv=True
            )
        }

        self.assertIn("lg_tv_power", names)
        self.assertIn("lg_tv_launch_app", names)
        self.assertNotIn("spotify_play_on_tv", names)
        self.assertNotIn("spotify_play", names)

    def test_spotify_tv_orchestration_requires_both_integrations(self) -> None:
        names = {
            str(item["function"]["name"])
            for item in capability_model_tools(
                include_spotify=True, include_lg_tv=True
            )
        }

        self.assertIn("spotify_play_on_tv", names)
        call = resolve_capability_tool_call(
            "spotify_play_on_tv",
            {"query": "Imperial March", "item_type": "track"},
        )
        self.assertEqual(call.name, "media.spotify_on_tv")


if __name__ == "__main__":
    unittest.main()
