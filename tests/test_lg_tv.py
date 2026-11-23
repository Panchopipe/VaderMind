"""Tests for the bounded LG webOS TV executor."""

from __future__ import annotations

import asyncio
from pathlib import Path
import time
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from vadermind.tools.lg_tv import LgTvClient, LgTvError, LgTvToolsExecutor
from vadermind.tools.lg_tv_scheduler import LgTvScheduleManager


class RecordingLgTvClient:
    def __init__(self) -> None:
        self.calls: list[tuple[str, object]] = []

    def status(self) -> dict[str, object]:
        self.calls.append(("status", None))
        return {
            "is_on": True,
            "screen_on": True,
            "current_app": "Home",
            "channel_name": "Canal de prueba",
            "channel_number": "7.1",
            "sound_output": "tv_speaker",
            "volume": 29,
            "muted": False,
        }

    def set_volume(self, volume: int) -> int:
        self.calls.append(("set_volume", volume))
        return volume

    def step_volume(self, direction: str) -> int:
        self.calls.append(("step_volume", direction))
        return 30 if direction == "up" else 28

    def set_mute(self, muted: bool) -> bool:
        self.calls.append(("set_mute", muted))
        return muted

    def launch_app(self, app_name: str) -> str:
        self.calls.append(("launch_app", app_name))
        return app_name

    def select_input(self, input_name: str) -> str:
        self.calls.append(("select_input", input_name))
        return input_name

    def playback(self, action: str) -> str:
        self.calls.append(("playback", action))
        return action

    def navigate(self, action: str) -> str:
        self.calls.append(("navigate", action))
        return action

    def step_channel(self, direction: str) -> dict[str, str]:
        self.calls.append(("step_channel", direction))
        return {"id": "channel-8", "number": "8.1", "name": "Canal nuevo"}

    def select_channel(self, query: str) -> dict[str, str]:
        self.calls.append(("select_channel", query))
        return {"id": "channel-13", "number": "13-2", "name": "T13 en vivo"}

    def set_screen(self, on: bool) -> bool:
        self.calls.append(("set_screen", on))
        return on

    def close_current_app(self) -> str:
        self.calls.append(("close_current_app", None))
        return "Home"

    def show_message(self, message: str) -> str:
        self.calls.append(("show_message", message))
        return message

    def power_off(self) -> bool:
        self.calls.append(("power_off", None))
        return True

    def power_on(self) -> bool:
        self.calls.append(("power_on", None))
        return True


class OfflineLgTvClient(RecordingLgTvClient):
    def status(self) -> dict[str, object]:
        self.calls.append(("status", None))
        return {
            "is_on": False,
            "reachable": False,
            "current_app": "",
            "volume": None,
            "muted": None,
        }


class UnconfirmedPowerOffLgTvClient(RecordingLgTvClient):
    def power_off(self) -> bool:
        self.calls.append(("power_off", None))
        return False


class UnconfirmedWakeLgTvClient(RecordingLgTvClient):
    def power_on(self) -> bool:
        self.calls.append(("power_on", None))
        return False


class RecordingScheduler:
    def __init__(self) -> None:
        self.items: list[dict[str, object]] = []

    def schedule(
        self, action: str, delay_minutes: int, *, message: str = ""
    ) -> dict[str, object]:
        item = {
            "id": "timer-1",
            "action": action,
            "execute_at": time.time() + delay_minutes * 60,
            "message": message,
        }
        self.items.append(item)
        return item

    def pending(self, action: str | None = None) -> list[dict[str, object]]:
        return [
            dict(item)
            for item in self.items
            if action is None or item["action"] == action
        ]

    def cancel(self, action: str | None = None) -> int:
        before = len(self.items)
        self.items = [
            item
            for item in self.items
            if action is not None and item["action"] != action
        ]
        return before - len(self.items)


class AsyncRecordingWebOsDevice:
    def __init__(self) -> None:
        self.calls: list[tuple[str, object]] = []
        self.tv_state = SimpleNamespace(
            current_app_id="youtube",
            apps={"youtube": {"title": "YouTube"}},
            inputs={"com.webos.app.hdmi1": {"label": "HDMI 1", "id": "HDMI_1"}},
            channels=[],
        )
        self.current_channel = {
            "channelId": "channel-7",
            "channelNumber": "7.1",
            "channelName": "Canal anterior",
        }
        self.screen_on = True

    async def pause(self) -> None:
        self.calls.append(("pause", None))

    async def play(self) -> None:
        self.calls.append(("play", None))

    async def stop(self) -> None:
        self.calls.append(("stop", None))

    async def rewind(self) -> None:
        self.calls.append(("rewind", None))

    async def fast_forward(self) -> None:
        self.calls.append(("fast_forward", None))

    async def button(self, button: str) -> None:
        self.calls.append(("button", button))

    async def channel_up(self) -> None:
        self.calls.append(("channel_up", None))
        self.current_channel = {
            "channelId": "channel-8",
            "channelNumber": "8.1",
            "channelName": "Canal nuevo",
        }

    async def get_current_channel(self) -> dict[str, str]:
        self.calls.append(("get_current_channel", None))
        return dict(self.current_channel)

    async def set_screen_state(self, on: bool) -> None:
        self.calls.append(("set_screen_state", on))
        self.screen_on = on

    async def get_power_state(self) -> dict[str, str]:
        self.calls.append(("get_power_state", None))
        return {"state": "Active" if self.screen_on else "Screen Off"}

    async def close_app(self, app_id: str) -> None:
        self.calls.append(("close_app", app_id))

    async def send_message(self, message: str) -> None:
        self.calls.append(("send_message", message))

    async def launch_app(self, app_id: str) -> None:
        self.calls.append(("launch_app", app_id))
        self.tv_state.current_app_id = app_id

    async def get_current_app(self) -> str:
        self.calls.append(("get_current_app", None))
        return self.tv_state.current_app_id

    async def set_input(self, input_id: str) -> None:
        self.calls.append(("set_input", input_id))
        self.tv_state.current_app_id = "com.webos.app.hdmi1"

    async def get_input(self) -> str:
        self.calls.append(("get_input", None))
        return self.tv_state.current_app_id

    async def get_channels(self) -> list[dict[str, str]]:
        self.calls.append(("get_channels", None))
        return [
            {
                "channelId": "channel-13",
                "channelNumber": "13-2",
                "channelName": "T13 en vivo",
            }
        ]

    async def set_channel(self, channel_id: str) -> None:
        self.calls.append(("set_channel", channel_id))
        self.current_channel = {
            "channelId": channel_id,
            "channelNumber": "13-2",
            "channelName": "T13 en vivo",
        }


class LgTvToolsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = RecordingLgTvClient()
        self.executor = LgTvToolsExecutor(client=self.client)

    def test_status_is_a_direct_factual_answer(self) -> None:
        result = self.executor.execute("lg_tv.status", {})

        self.assertTrue(result["ok"])
        self.assertTrue(result["direct_answer"])
        self.assertIn("volumen 29", result["message"])
        self.assertIn("canal 7.1 Canal de prueba", result["message"])
        self.assertIn("altavoces del televisor", result["message"])

    def test_status_answers_only_the_requested_detail(self) -> None:
        expected = {
            "power": "La television esta encendida.",
            "app": "La aplicacion activa es Home.",
            "channel": "El canal actual es 7.1 Canal de prueba.",
            "screen": "La pantalla de la television esta encendida.",
            "sound": "La television esta usando los altavoces del televisor.",
            "volume": "El volumen de la television esta en 29.",
            "mute": "La television no esta en silencio.",
        }

        for detail, message in expected.items():
            with self.subTest(detail=detail):
                result = self.executor.execute(
                    "lg_tv.status", {"detail": detail}
                )
                self.assertEqual(result["message"], message)

        channel = self.executor.execute(
            "lg_tv.status", {"detail": "channel"}
        )
        self.assertNotIn("volumen", channel["message"])
        self.assertNotIn("altavoces", channel["message"])

    def test_channel_detail_explains_when_no_channel_is_active(self) -> None:
        self.client.status = lambda: {
            "is_on": True,
            "reachable": True,
            "screen_on": True,
            "current_app": "Inicio",
            "channel_name": "",
            "channel_number": "",
            "sound_output": "tv_speaker",
            "volume": 29,
            "muted": False,
        }

        result = self.executor.execute("lg_tv.status", {"detail": "channel"})

        self.assertEqual(
            result["message"],
            "No estas viendo un canal; la television esta en Inicio.",
        )

    def test_unreachable_status_is_reported_as_probably_off(self) -> None:
        executor = LgTvToolsExecutor(client=OfflineLgTvClient())

        result = executor.execute("lg_tv.status", {})

        self.assertTrue(result["ok"])
        self.assertTrue(result["direct_answer"])
        self.assertIn("probablemente esta apagada", result["message"])

        channel = executor.execute("lg_tv.status", {"detail": "channel"})
        self.assertIn("No pude consultar el canal", channel["message"])

    def test_power_on_requires_network_confirmation(self) -> None:
        executor = LgTvToolsExecutor(client=UnconfirmedWakeLgTvClient())

        result = executor.execute("lg_tv.power_on", {})

        self.assertFalse(result["ok"])
        self.assertTrue(result["direct_answer"])
        self.assertIn("no pude confirmar", result["message"])

    def test_power_off_reports_tv_that_stays_on(self) -> None:
        executor = LgTvToolsExecutor(client=UnconfirmedPowerOffLgTvClient())

        result = executor.execute("lg_tv.power_off", {})

        self.assertFalse(result["ok"])
        self.assertTrue(result["direct_answer"])
        self.assertIn("sigue encendida", result["message"])

    def test_mutations_dispatch_only_known_operations(self) -> None:
        cases = (
            ("lg_tv.set_volume", {"volume": 24}, ("set_volume", 24)),
            ("lg_tv.volume_up", {}, ("step_volume", "up")),
            ("lg_tv.volume_down", {}, ("step_volume", "down")),
            ("lg_tv.set_mute", {"muted": True}, ("set_mute", True)),
            ("lg_tv.launch_app", {"app_name": "YouTube"}, ("launch_app", "YouTube")),
            ("lg_tv.select_input", {"input": "HDMI 1"}, ("select_input", "HDMI 1")),
            ("lg_tv.playback", {"action": "pause"}, ("playback", "pause")),
            ("lg_tv.navigate", {"action": "home"}, ("navigate", "home")),
            ("lg_tv.channel_up", {}, ("step_channel", "up")),
            ("lg_tv.channel_down", {}, ("step_channel", "down")),
            (
                "lg_tv.select_channel",
                {"query": "T13"},
                ("select_channel", "T13"),
            ),
            ("lg_tv.screen_on", {}, ("set_screen", True)),
            ("lg_tv.screen_off", {}, ("set_screen", False)),
            ("lg_tv.close_app", {}, ("close_current_app", None)),
            (
                "lg_tv.show_message",
                {"message": "  Hola   desde Vader  "},
                ("show_message", "Hola desde Vader"),
            ),
            ("lg_tv.power_off", {}, ("power_off", None)),
            ("lg_tv.power_on", {}, ("power_on", None)),
        )
        for name, args, expected_call in cases:
            with self.subTest(name=name):
                result = self.executor.execute(name, args)
                self.assertTrue(result["ok"])
                self.assertEqual(self.client.calls[-1], expected_call)

    def test_unknown_and_invalid_operations_fail_closed(self) -> None:
        unknown = self.executor.execute("lg_tv.raw", {"uri": "system/turnOff"})
        invalid_volume = self.executor.execute("lg_tv.set_volume", {"volume": 101})
        invalid_input = self.executor.execute("lg_tv.select_input", {"input": "AV 1"})
        invalid_navigation = self.executor.execute(
            "lg_tv.navigate", {"action": "service_menu"}
        )
        long_message = self.executor.execute(
            "lg_tv.show_message", {"message": "x" * 161}
        )
        invalid_detail = self.executor.execute(
            "lg_tv.status", {"detail": "everything"}
        )

        self.assertFalse(unknown["ok"])
        self.assertFalse(invalid_volume["ok"])
        self.assertFalse(invalid_input["ok"])
        self.assertFalse(invalid_navigation["ok"])
        self.assertFalse(long_message["ok"])
        self.assertFalse(invalid_detail["ok"])

    def test_app_and_input_resolution_uses_tv_catalogs(self) -> None:
        app = LgTvClient._resolve_app(
            {
                "youtube.leanback.v4": {"title": "YouTube"},
                "netflix": {"title": "Netflix"},
            },
            "youtube",
        )
        input_item = LgTvClient._resolve_input(
            {
                "com.webos.app.hdmi1": {
                    "label": "HDMI 1",
                    "id": "HDMI_1",
                }
            },
            "HDMI 1",
        )

        self.assertEqual(app, ("youtube.leanback.v4", "YouTube"))
        self.assertEqual(input_item, ("HDMI_1", "HDMI 1"))
        home_client = SimpleNamespace(tv_state=SimpleNamespace(apps={}))
        self.assertEqual(
            LgTvClient._app_title(home_client, "com.webos.app.home"), "Inicio"
        )

    def test_power_on_retries_wol_and_waits_for_webos(self) -> None:
        settings = SimpleNamespace(
            lg_tv_host="192.0.2.162",
            lg_tv_mac="00:00:5e:00:53:01",
            lg_tv_broadcast_address="192.0.2.255",
            lg_tv_client_key_path=".test-lg-key",
            lg_tv_timeout_seconds=5.0,
            lg_tv_wake_timeout_seconds=20.0,
        )
        client = LgTvClient(settings)

        with (
            patch.object(
                client,
                "_is_webos_reachable",
                side_effect=[False, False, True],
            ),
            patch.object(client, "_send_magic_packet") as send_packet,
            patch.object(client, "_connect_in_background") as connect_early,
            patch("vadermind.tools.lg_tv.time.sleep"),
        ):
            confirmed = client.power_on()

        self.assertTrue(confirmed)
        send_packet.assert_called_once()
        # The shared connection opens right away, ready for the next command.
        connect_early.assert_called_once()

    def _power_client(self) -> LgTvClient:
        return LgTvClient(
            SimpleNamespace(
                lg_tv_host="192.0.2.167",
                lg_tv_mac="00:00:5e:00:53:01",
                lg_tv_broadcast_address="192.0.2.255",
                lg_tv_client_key_path=".test-lg-key",
                lg_tv_timeout_seconds=5.0,
                lg_tv_wake_timeout_seconds=20.0,
            )
        )

    def test_power_off_confirms_state_and_retries_once(self) -> None:
        client = self._power_client()

        with (
            patch.object(client, "_is_webos_reachable", return_value=True),
            patch.object(client, "_run_connected") as run_connected,
            patch.object(client, "_wait_until_off", side_effect=[False, True]),
        ):
            confirmed = client.power_off()

        self.assertTrue(confirmed)
        self.assertEqual(run_connected.call_count, 2)

    def test_power_off_fails_when_tv_stays_on_after_retry(self) -> None:
        client = self._power_client()

        with (
            patch.object(client, "_is_webos_reachable", return_value=True),
            patch.object(client, "_run_connected") as run_connected,
            patch.object(client, "_wait_until_off", return_value=False),
        ):
            confirmed = client.power_off()

        self.assertFalse(confirmed)
        self.assertEqual(run_connected.call_count, 2)

    def test_power_off_treats_dropped_connection_as_pending_confirmation(self) -> None:
        client = self._power_client()

        with (
            patch.object(client, "_is_webos_reachable", return_value=True),
            patch.object(
                client, "_run_connected", side_effect=LgTvError("interrumpida")
            ) as run_connected,
            patch.object(client, "_wait_until_off", return_value=True),
        ):
            confirmed = client.power_off()

        self.assertTrue(confirmed)
        run_connected.assert_called_once()

    def test_standby_or_unreachable_tv_counts_as_off(self) -> None:
        client = self._power_client()

        with patch.object(client, "_is_webos_reachable", return_value=False):
            self.assertTrue(client._is_confirmed_off())
        with (
            patch.object(client, "_is_webos_reachable", return_value=True),
            patch.object(client, "status", return_value={"is_on": False}),
        ):
            self.assertTrue(client._is_confirmed_off())
        with (
            patch.object(client, "_is_webos_reachable", return_value=True),
            patch.object(client, "status", return_value={"is_on": True}),
        ):
            self.assertFalse(client._is_confirmed_off())

    def test_status_skips_websocket_when_tv_is_unreachable(self) -> None:
        settings = SimpleNamespace(
            lg_tv_host="192.0.2.162",
            lg_tv_mac="00:00:5e:00:53:01",
            lg_tv_broadcast_address="192.0.2.255",
            lg_tv_client_key_path=".test-lg-key",
            lg_tv_timeout_seconds=5.0,
            lg_tv_wake_timeout_seconds=20.0,
        )
        client = LgTvClient(settings)

        with (
            patch.object(client, "_is_webos_reachable", return_value=False),
            patch.object(client, "_run_connected") as run_connected,
        ):
            status = client.status()

        self.assertFalse(status["reachable"])
        run_connected.assert_not_called()

    def test_client_maps_second_layer_to_specific_webos_methods(self) -> None:
        settings = SimpleNamespace(
            lg_tv_host="192.0.2.162",
            lg_tv_mac="00:00:5e:00:53:01",
            lg_tv_broadcast_address="192.0.2.255",
            lg_tv_client_key_path=".test-lg-key",
            lg_tv_timeout_seconds=5.0,
            lg_tv_wake_timeout_seconds=40.0,
        )
        client = LgTvClient(settings)
        device = AsyncRecordingWebOsDevice()

        def run(operation: object) -> object:
            return asyncio.run(operation(device))  # type: ignore[operator]

        with patch.object(client, "_run_connected", side_effect=run):
            client.playback("pause")
            client.navigate("home")
            client.step_channel("up")
            client.set_screen(False)
            client.close_current_app()
            client.show_message("Hola")

        self.assertEqual(
            device.calls,
            [
                ("pause", None),
                ("button", "HOME"),
                ("get_current_channel", None),
                ("channel_up", None),
                ("get_current_channel", None),
                ("set_screen_state", False),
                ("get_power_state", None),
                ("close_app", "youtube"),
                ("send_message", "Hola"),
            ],
        )

    def test_client_confirms_app_input_and_direct_channel(self) -> None:
        settings = SimpleNamespace(
            lg_tv_host="192.0.2.162",
            lg_tv_mac="00:00:5e:00:53:01",
            lg_tv_broadcast_address="192.0.2.255",
            lg_tv_client_key_path=".test-lg-key",
            lg_tv_timeout_seconds=5.0,
            lg_tv_wake_timeout_seconds=40.0,
        )
        client = LgTvClient(settings)
        device = AsyncRecordingWebOsDevice()

        def run(operation: object) -> object:
            return asyncio.run(operation(device))  # type: ignore[operator]

        with patch.object(client, "_run_connected", side_effect=run):
            self.assertEqual(client.launch_app("YouTube"), "YouTube")
            self.assertEqual(client.select_input("HDMI 1"), "HDMI 1")
            selected = client.select_channel("T13")

        self.assertEqual(selected["number"], "13-2")
        self.assertIn(("get_current_app", None), device.calls)
        self.assertIn(("get_input", None), device.calls)
        self.assertIn(("set_channel", "channel-13"), device.calls)

    def test_app_launch_survives_webos_transition_cancellation(self) -> None:
        settings = SimpleNamespace(
            lg_tv_host="192.0.2.162",
            lg_tv_mac="00:00:5e:00:53:01",
            lg_tv_broadcast_address="192.0.2.255",
            lg_tv_client_key_path=".test-lg-key",
            lg_tv_timeout_seconds=5.0,
            lg_tv_wake_timeout_seconds=40.0,
        )
        client = LgTvClient(settings)
        device = AsyncRecordingWebOsDevice()

        async def cancelled_launch(app_id: str) -> None:
            device.calls.append(("launch_app", app_id))
            device.tv_state.current_app_id = app_id
            raise asyncio.CancelledError

        def run(operation: object) -> object:
            return asyncio.run(operation(device))  # type: ignore[operator]

        with (
            patch.object(client, "_run_connected", side_effect=run),
            patch.object(device, "launch_app", side_effect=cancelled_launch),
        ):
            launched = client.launch_app("YouTube")

        self.assertEqual(launched, "YouTube")
        self.assertIn(("get_current_app", None), device.calls)

    def test_channel_resolution_is_exact_and_fails_when_ambiguous(self) -> None:
        channels = [
            {"channelId": "a", "channelNumber": "13-2", "channelName": "T13"},
            {"channelId": "b", "channelNumber": "13-3", "channelName": "T13 HD"},
        ]

        selected = LgTvClient._resolve_channel(channels, "13-2")
        self.assertEqual(selected["channelId"], "a")
        with self.assertRaisesRegex(Exception, "ambiguo"):
            LgTvClient._resolve_channel(channels, "13")

    def test_persistent_tv_timers_execute_overdue_actions_once(self) -> None:
        calls: list[tuple[str, str]] = []
        path = Path(".test-lg-timers.json")
        try:
            manager = LgTvScheduleManager(
                path,
                power_off=lambda: calls.append(("power_off", "")),
                show_message=lambda message: calls.append(("message", message)),
                start_worker=False,
            )
            item = manager.schedule("show_message", 1, message="Hola Vader")
            self.assertTrue(path.is_file())

            restarted = LgTvScheduleManager(
                path,
                power_off=lambda: calls.append(("power_off", "")),
                show_message=lambda message: calls.append(("message", message)),
                start_worker=False,
            )
            executed = restarted.run_due(now=float(item["execute_at"]) + 1)
            self.assertEqual(executed, 1)
            self.assertEqual(calls, [("message", "Hola Vader")])
            self.assertEqual(restarted.run_due(now=time.time() + 600), 0)
        finally:
            path.unlink(missing_ok=True)
            path.with_name(path.name + ".tmp").unlink(missing_ok=True)

    def test_timer_tools_schedule_report_and_cancel(self) -> None:
        scheduler = RecordingScheduler()
        executor = LgTvToolsExecutor(client=self.client, scheduler=scheduler)

        scheduled = executor.execute(
            "lg_tv.schedule_power_off", {"minutes": 30}
        )
        status = executor.execute("lg_tv.timer_status", {})
        cancelled = executor.execute(
            "lg_tv.cancel_timers", {"scope": "power_off"}
        )

        self.assertTrue(scheduled["ok"])
        self.assertIn("apagado en 30 minutos", status["message"])
        self.assertIn("Cancele 1 temporizador", cancelled["message"])


if __name__ == "__main__":
    unittest.main()
