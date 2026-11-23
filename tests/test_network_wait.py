"""Tests for loading Vader while the Wi-Fi connects (`--esperar-red`)."""

from __future__ import annotations

from types import SimpleNamespace
import unittest
from unittest import mock

from vadermind import network_wait
from vadermind.network_wait import wait_for_network


class _Clock:
    """Fake monotonic clock that only moves when the code sleeps."""

    def __init__(self) -> None:
        self.now = 0.0
        self.sleeps = 0

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps += 1
        self.now += seconds


class WaitForNetworkTests(unittest.TestCase):
    def test_returns_at_once_when_the_network_is_already_up(self) -> None:
        clock = _Clock()
        self.assertTrue(wait_for_network(is_ready=lambda: True, clock=clock, sleep=clock.sleep))
        self.assertEqual(clock.sleeps, 0)

    def test_polls_until_systemd_reaches_the_target(self) -> None:
        clock = _Clock()
        answers = iter([False, False, False, True])
        self.assertTrue(wait_for_network(is_ready=lambda: next(answers), clock=clock, sleep=clock.sleep))
        self.assertEqual(clock.sleeps, 3)

    def test_gives_up_after_the_timeout_and_starts_anyway(self) -> None:
        clock = _Clock()
        with self.assertLogs("vadermind.network_wait", level="WARNING") as logs:
            ready = wait_for_network(5.0, is_ready=lambda: False, clock=clock, sleep=clock.sleep)
        self.assertFalse(ready)
        self.assertGreaterEqual(clock.now, 5.0)
        self.assertIn("arranco igual", logs.output[0])

    def test_without_systemd_there_is_nothing_to_wait_for(self) -> None:
        with mock.patch.object(network_wait.shutil, "which", return_value=None):
            self.assertTrue(wait_for_network(timeout=0.0))

    def test_asks_systemd_about_network_online(self) -> None:
        calls: list[list[str]] = []

        def run(args, check):
            calls.append(args)
            return SimpleNamespace(returncode=0 if len(calls) > 1 else 3)  # 3: inactive

        with mock.patch.object(network_wait.subprocess, "run", side_effect=run):
            self.assertFalse(network_wait._target_active("/usr/bin/systemctl"))
            self.assertTrue(network_wait._target_active("/usr/bin/systemctl"))
        self.assertEqual(calls[0], ["/usr/bin/systemctl", "is-active", "--quiet", "network-online.target"])

    def test_a_broken_check_does_not_hold_vader_back(self) -> None:
        with mock.patch.object(network_wait.subprocess, "run", side_effect=OSError("sin systemctl")):
            with self.assertLogs("vadermind.network_wait", level="WARNING"):
                self.assertTrue(network_wait._target_active("/usr/bin/systemctl"))


class _Warm:
    def __init__(self, name: str, events: list[str]) -> None:
        self.name = name
        self.events = events

    def warm(self) -> None:
        self.events.append(self.name)


class HandsFreeStartupTests(unittest.TestCase):
    def _app(self, events: list[str]) -> SimpleNamespace:
        return SimpleNamespace(
            tts=_Warm("piper", events),
            engine=_Warm("modelo", events),
            tool_executor=_Warm("herramientas", events),
        )

    def test_flag_is_parsed(self) -> None:
        from vadermind.orchestrator import build_parser

        args = build_parser().parse_args(["--manos-libres", "--esperar-red"])
        self.assertTrue(args.esperar_red)
        self.assertFalse(build_parser().parse_args(["--manos-libres"]).esperar_red)

    def test_session_warm_up_can_leave_the_network_for_later(self) -> None:
        from vadermind import orchestrator

        events: list[str] = []
        listener = SimpleNamespace(transcriber=_Warm("stt", events), on_speech_start=None)
        run_now = lambda *tasks: [task() for task in tasks]  # noqa: E731
        with mock.patch.object(orchestrator, "in_background", side_effect=run_now):
            orchestrator._warm_up_session(self._app(events), listener, network=False)
            self.assertEqual(events, ["piper"])
            orchestrator._warm_up_session(self._app(events), listener)
        self.assertEqual(events, ["piper", "piper", "modelo", "herramientas", "stt"])

    def _run_hands_free(self, wait_network: bool) -> list[str]:
        from vadermind import orchestrator

        events: list[str] = []
        settings = SimpleNamespace(
            wake_phrase="Oye Vader", persona_mode="chileno", follow_up_seconds=10.0,
            allow_shutdown=False,
        )
        listener = SimpleNamespace(
            transcriber=_Warm("stt", events), device_name="micro", device=None, on_recorded=None
        )

        class Session:
            def __init__(self, *args, **kwargs) -> None:
                pass

            def run(self) -> int:
                events.append("pupilas")
                return 0

        def load_vosk(_settings):
            events.append("vosk")
            return object()

        run_now = lambda *tasks: [task() for task in tasks]  # noqa: E731
        with mock.patch.object(orchestrator, "build_wake_word", side_effect=load_vosk), \
                mock.patch.object(orchestrator, "wait_for_network", side_effect=lambda: events.append("red")), \
                mock.patch.object(orchestrator, "in_background", side_effect=run_now), \
                mock.patch.object(orchestrator, "HandsFreeSession", Session), \
                mock.patch.object(orchestrator.signal, "signal"), \
                mock.patch.object(orchestrator, "_safe_print"):
            code = orchestrator._run_hands_free(
                settings, self._app(events), listener, SimpleNamespace(), wait_network=wait_network
            )
        self.assertEqual(code, 0)
        return events

    def test_waits_for_the_network_after_loading_and_before_the_dim_pupils(self) -> None:
        self.assertEqual(
            self._run_hands_free(wait_network=True),
            ["vosk", "red", "modelo", "herramientas", "stt", "pupilas"],
        )

    def test_without_the_flag_nothing_waits(self) -> None:
        self.assertEqual(self._run_hands_free(wait_network=False), ["vosk", "pupilas"])


if __name__ == "__main__":
    unittest.main()
