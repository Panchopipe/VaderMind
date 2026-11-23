"""Tests for the persistent webOS connection used by LgTvClient."""

from __future__ import annotations

import threading
import unittest

from vadermind.tools.lg_tv import LgTvError, _WebOsSession


class _FakeWebOs:
    def __init__(self, number: int) -> None:
        self.number = number
        self.connected = True
        self.probe_fails = False
        self.disconnects = 0

    def is_connected(self) -> bool:
        return self.connected

    async def get_power_state(self) -> dict[str, str]:
        if self.probe_fails:
            raise ConnectionResetError("socket muerto")
        return {"state": "Active"}

    async def disconnect(self) -> None:
        self.disconnects += 1
        self.connected = False


class WebOsSessionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.opened: list[_FakeWebOs] = []

        async def connect() -> _FakeWebOs:
            client = _FakeWebOs(len(self.opened) + 1)
            self.opened.append(client)
            return client

        self.session = _WebOsSession(connect)
        self.addCleanup(self.session.close)

    def _run(self, operation):
        return self.session.run(operation, timeout=5)

    def test_one_connection_serves_many_commands(self) -> None:
        async def which(client):
            return client.number

        self.assertEqual([self._run(which) for _ in range(3)], [1, 1, 1])
        self.assertEqual(len(self.opened), 1)

    def test_a_dead_connection_is_replaced_and_the_command_runs_once(self) -> None:
        sent: list[int] = []

        async def command(client):
            sent.append(client.number)
            return client.number

        self._run(command)
        self.opened[0].probe_fails = True
        self.assertEqual(self._run(command), 2)
        self.assertEqual(sent, [1, 2])  # one send per call, never repeated
        self.assertEqual(self.opened[0].disconnects, 1)

    def test_a_closed_connection_is_reopened(self) -> None:
        async def which(client):
            return client.number

        self._run(which)
        self.opened[0].connected = False
        with self.assertLogs("vadermind.tools.lg_tv", "INFO") as logs:
            self.assertEqual(self._run(which), 2)
        # the journal shows who paid the ~4 s reconnection (warm-up or command)
        self.assertIn("conexion webOS abierta", logs.output[0])

    def test_transport_error_during_a_command_drops_the_connection(self) -> None:
        async def broken(client):
            raise ConnectionResetError("el TV corto la conexion")

        async def which(client):
            return client.number

        with self.assertRaises(ConnectionResetError):
            self._run(broken)
        self.assertEqual(self._run(which), 2)

    def test_tv_refusal_keeps_the_connection(self) -> None:
        async def refused(client):
            raise LgTvError("No encontre esa aplicacion.")

        async def which(client):
            return client.number

        with self.assertRaises(LgTvError):
            self._run(refused)
        self.assertEqual(self._run(which), 1)

    def test_close_can_run_twice(self) -> None:
        async def which(client):
            return client.number

        self._run(which)
        self.session.close()
        self.session.close()  # explicit close plus the atexit hook
        self.assertEqual(self.opened[0].disconnects, 1)

    def test_calls_from_several_threads_run_one_at_a_time(self) -> None:
        import asyncio

        active = 0
        peak = 0

        async def slow(client):
            nonlocal active, peak
            active += 1
            peak = max(peak, active)
            await asyncio.sleep(0.05)
            active -= 1

        threads = [threading.Thread(target=self._run, args=(slow,)) for _ in range(4)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        self.assertEqual(peak, 1)
        self.assertEqual(len(self.opened), 1)


if __name__ == "__main__":
    unittest.main()
