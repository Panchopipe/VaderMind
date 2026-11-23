"""Local, allowlisted control for LG televisions running webOS."""

from __future__ import annotations

import asyncio
import atexit
from collections.abc import Awaitable, Callable
from contextlib import suppress
import logging
from pathlib import Path
import socket
import stat
import threading
import time
import unicodedata
from typing import Any, Protocol

from vadermind.config import Settings
from vadermind.tools.lg_tv_scheduler import (
    LgTvScheduleError,
    LgTvScheduleManager,
)


LOGGER = logging.getLogger(__name__)


class LgTvError(RuntimeError):
    """Raised when a validated LG TV operation cannot be completed."""


class LgTvController(Protocol):
    """Small synchronous surface used by the tool executor and its tests."""

    def status(self) -> dict[str, object]: ...

    def set_volume(self, volume: int) -> int: ...

    def step_volume(self, direction: str) -> int | None: ...

    def set_mute(self, muted: bool) -> bool: ...

    def launch_app(self, app_name: str) -> str: ...

    def select_input(self, input_name: str) -> str: ...

    def playback(self, action: str) -> str: ...

    def navigate(self, action: str) -> str: ...

    def step_channel(self, direction: str) -> dict[str, str]: ...

    def select_channel(self, query: str) -> dict[str, str]: ...

    def set_screen(self, on: bool) -> bool: ...

    def close_current_app(self) -> str: ...

    def show_message(self, message: str) -> str: ...

    def power_off(self) -> bool: ...

    def power_on(self) -> bool: ...


class LgTvScheduler(Protocol):
    """Small scheduler surface used by the executor and test doubles."""

    def schedule(
        self, action: str, delay_minutes: int, *, message: str = ""
    ) -> dict[str, object]: ...

    def pending(self, action: str | None = None) -> list[dict[str, object]]: ...

    def cancel(self, action: str | None = None) -> int: ...


class _WebOsSession:
    """Keep one webOS connection open on a background event loop.

    Opening a connection costs about 4 s on the living-room TV: aiowebostv
    downloads the whole TV state (apps, channels, programme guide) and sleeps a
    fixed 2 s while Live TV is showing. The command itself takes about 0.1 s.

    Before each operation a read-only request checks the connection; if it
    fails, a new connection replaces it. Only that probe is ever retried, never
    the command, so a dead connection cannot make the TV act twice. Operations
    run one at a time, including those from the timer thread.
    """

    PROBE_TIMEOUT_SECONDS = 2.0

    def __init__(self, connect: Callable[[], Awaitable[Any]]) -> None:
        self._connect = connect
        self._client: Any = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._operation_lock: asyncio.Lock | None = None
        self._start_lock = threading.Lock()

    def run(self, operation: Callable[[Any], Awaitable[Any]], timeout: float) -> Any:
        future = asyncio.run_coroutine_threadsafe(self._run(operation), self._event_loop())
        try:
            return future.result(timeout)
        except TimeoutError:
            future.cancel()
            raise

    def close(self) -> None:
        with self._start_lock:
            loop, self._loop = self._loop, None
        if loop is None or loop.is_closed():
            return
        with suppress(Exception):
            asyncio.run_coroutine_threadsafe(self._drop(), loop).result(4)
        loop.call_soon_threadsafe(loop.stop)
        if self._thread is not None:
            self._thread.join(timeout=4)

    @staticmethod
    def _serve(loop: asyncio.AbstractEventLoop) -> None:
        asyncio.set_event_loop(loop)
        try:
            loop.run_forever()
        finally:
            loop.close()

    def _event_loop(self) -> asyncio.AbstractEventLoop:
        with self._start_lock:
            if self._loop is None:
                loop = asyncio.new_event_loop()
                self._thread = threading.Thread(
                    target=self._serve, args=(loop,), name="webos", daemon=True
                )
                self._thread.start()
                self._loop = loop
            return self._loop

    async def _run(self, operation: Callable[[Any], Awaitable[Any]]) -> Any:
        if self._operation_lock is None:
            self._operation_lock = asyncio.Lock()
        async with self._operation_lock:
            client = await self._live_client()
            try:
                return await operation(client)
            except LgTvError:
                raise  # the TV answered; the connection is fine
            except BaseException:
                await self._drop()  # transport trouble: reconnect next time
                raise

    async def _live_client(self) -> Any:
        client = self._client
        if client is not None and client.is_connected():
            try:
                await asyncio.wait_for(client.get_power_state(), self.PROBE_TIMEOUT_SECONDS)
                return client
            except asyncio.CancelledError:
                task = asyncio.current_task()
                if task is not None and task.cancelling():
                    raise  # our own caller gave up; do not swallow it
            except Exception:
                pass
        await self._drop()
        started = time.perf_counter()
        self._client = await self._connect()
        # Timing only: shows whether a command or the "Oye Vader" warm-up paid it.
        LOGGER.info("TV: conexion webOS abierta en %.2f s.", time.perf_counter() - started)
        return self._client

    async def _drop(self) -> None:
        client, self._client = self._client, None
        if client is not None:
            with suppress(Exception, asyncio.CancelledError):
                await asyncio.wait_for(client.disconnect(), timeout=3)


class LgTvClient:
    """Connect to one configured webOS TV and persist its pairing key locally."""

    def __init__(self, settings: Settings) -> None:
        if not settings.lg_tv_host:
            raise LgTvError("LG_TV_HOST no esta configurado.")
        self.host = settings.lg_tv_host
        self.mac = settings.lg_tv_mac
        self.broadcast_address = settings.lg_tv_broadcast_address
        self.client_key_path = Path(settings.lg_tv_client_key_path)
        self.connect_timeout = settings.lg_tv_timeout_seconds
        self.wake_timeout = settings.lg_tv_wake_timeout_seconds
        self.confirm_timeout = min(max(self.connect_timeout, 4.0), 10.0)
        self._session = _WebOsSession(self._open_client)
        atexit.register(self._session.close)

    def warm(self) -> None:
        """Connect ahead of the first command when the TV is already on."""

        if not self._is_webos_reachable():
            return

        async def noop(client: Any) -> None:
            return None

        with suppress(LgTvError):
            self._run_connected(noop)

    def close(self) -> None:
        self._session.close()

    def status(self) -> dict[str, object]:
        """Read the current TV state without changing it."""

        if not self._is_webos_reachable():
            return {
                "is_on": False,
                "reachable": False,
                "screen_on": False,
                "current_app": "",
                "current_app_id": "",
                "channel_name": "",
                "channel_number": "",
                "sound_output": "",
                "volume": None,
                "muted": None,
            }

        async def operation(client: Any) -> dict[str, object]:
            state = client.tv_state
            current_app_id = state.current_app_id or ""
            current_app = self._app_title(client, current_app_id) or current_app_id
            current_channel = state.current_channel or {}
            if current_app_id == "com.webos.app.livetv" and not current_channel:
                from aiowebostv.exceptions import WebOsTvError

                with suppress(WebOsTvError):
                    current_channel = await client.get_current_channel()
            return {
                "is_on": bool(state.is_on),
                "reachable": True,
                "screen_on": bool(state.is_screen_on),
                "current_app": current_app,
                "current_app_id": current_app_id,
                "channel_name": str(current_channel.get("channelName") or "").strip(),
                "channel_number": str(
                    current_channel.get("channelNumber") or ""
                ).strip(),
                "sound_output": str(state.sound_output or "").strip(),
                "volume": state.volume,
                "muted": state.muted,
            }

        return self._run_connected(operation)

    def set_volume(self, volume: int) -> int:
        """Set and then read back an absolute volume from 0 to 100."""

        async def operation(client: Any) -> int:
            await client.set_volume(volume)
            confirmed = await client.get_volume()
            return volume if confirmed is None else int(confirmed)

        return self._run_connected(operation)

    def step_volume(self, direction: str) -> int | None:
        """Move the TV volume by one native webOS step."""

        async def operation(client: Any) -> int | None:
            if direction == "up":
                await client.volume_up()
            elif direction == "down":
                await client.volume_down()
            else:
                raise LgTvError("La direccion de volumen no es valida.")
            confirmed = await client.get_volume()
            return None if confirmed is None else int(confirmed)

        return self._run_connected(operation)

    def set_mute(self, muted: bool) -> bool:
        """Set mute explicitly and read back the resulting state."""

        async def operation(client: Any) -> bool:
            await client.set_mute(muted)
            confirmed = await client.get_muted()
            return muted if confirmed is None else bool(confirmed)

        return self._run_connected(operation)

    def launch_app(self, app_name: str) -> str:
        """Launch a catalog app and confirm that it became the foreground app."""

        async def send_launch(client: Any) -> tuple[str, str]:
            app_id, title = self._resolve_app(client.tv_state.apps, app_name)
            try:
                await client.launch_app(app_id)
            except asyncio.CancelledError:
                # Some webOS versions tear down the request task while switching
                # applications, even though the launch itself was applied. Treat
                # that as an indeterminate transport result and verify afresh.
                pass
            return app_id, title

        async def confirm_launch(client: Any, app_id: str) -> bool:
            return await self._wait_for_current_app(client, app_id)

        title = app_name
        for attempt in range(2):
            app_id, title = self._run_connected(send_launch)
            try:
                confirmed = self._run_connected(
                    lambda client: confirm_launch(client, app_id)
                )
            except LgTvError:
                confirmed = False
            if confirmed:
                return title
            if attempt == 0:
                time.sleep(0.4)
        raise LgTvError(
            f"Envie la orden para abrir {title}, pero la television no la dejo activa."
        )

    def select_input(self, input_name: str) -> str:
        """Switch to a TV-published HDMI input and confirm the active input."""

        async def operation(client: Any) -> str:
            input_id, label = self._resolve_input(client.tv_state.inputs, input_name)
            for attempt in range(2):
                await client.set_input(input_id)
                if await self._wait_for_input(client, input_id):
                    return label
                if attempt == 0:
                    await asyncio.sleep(0.4)
            raise LgTvError(
                f"Envie la orden para cambiar a {label}, pero no pude confirmar esa entrada."
            )

        return self._run_connected(operation)

    def playback(self, action: str) -> str:
        """Run one allowlisted media command against the foreground TV app."""

        async def operation(client: Any) -> str:
            operations = {
                "play": client.play,
                "pause": client.pause,
                "stop": client.stop,
                "rewind": client.rewind,
                "fast_forward": client.fast_forward,
            }
            selected = operations.get(action)
            if selected is None:
                raise LgTvError("El control de reproduccion no es valido.")
            await selected()
            return action

        return self._run_connected(operation)

    def navigate(self, action: str) -> str:
        """Press one safe remote-control button on the TV."""

        buttons = {
            "home": "HOME",
            "back": "BACK",
            "up": "UP",
            "down": "DOWN",
            "left": "LEFT",
            "right": "RIGHT",
            "ok": "ENTER",
        }
        button = buttons.get(action)
        if button is None:
            raise LgTvError("El control de navegacion no es valido.")

        async def operation(client: Any) -> str:
            await client.button(button)
            return action

        return self._run_connected(operation)

    def step_channel(self, direction: str) -> dict[str, str]:
        """Move one broadcast channel and confirm the resulting channel."""

        async def operation(client: Any) -> dict[str, str]:
            if direction not in {"up", "down"}:
                raise LgTvError("La direccion de canal no es valida.")
            before = await client.get_current_channel()
            before_key = self._channel_key(before)
            for attempt in range(2):
                if direction == "up":
                    await client.channel_up()
                else:
                    await client.channel_down()
                confirmed = await self._wait_for_channel_change(client, before_key)
                if confirmed:
                    return self._channel_summary(confirmed)
                if attempt == 0:
                    await asyncio.sleep(0.4)
            raise LgTvError(
                "Envie la orden de cambio de canal, pero el canal informado no cambio."
            )

        return self._run_connected(operation)

    def select_channel(self, query: str) -> dict[str, str]:
        """Select a TV-published channel by number or name and confirm it."""

        async def operation(client: Any) -> dict[str, str]:
            channels = await client.get_channels() or client.tv_state.channels or []
            channel = self._resolve_channel(channels, query)
            channel_id = str(channel.get("channelId") or "").strip()
            if not channel_id:
                raise LgTvError("El canal elegido no entrego un identificador utilizable.")
            for attempt in range(2):
                await client.set_channel(channel_id)
                confirmed = await self._wait_for_selected_channel(client, channel)
                if confirmed:
                    return self._channel_summary(confirmed)
                if attempt == 0:
                    await asyncio.sleep(0.4)
            label = self._channel_label(channel)
            raise LgTvError(
                f"Envie la orden para cambiar al canal {label}, pero no pude confirmarlo."
            )

        return self._run_connected(operation)

    def set_screen(self, on: bool) -> bool:
        """Turn only the panel on or off and confirm the resulting power state."""

        async def operation(client: Any) -> bool:
            for attempt in range(2):
                await client.set_screen_state(on)
                if await self._wait_for_screen_state(client, on):
                    return on
                if attempt == 0:
                    await asyncio.sleep(0.4)
            state = "encendida" if on else "apagada"
            raise LgTvError(
                f"Envie la orden, pero no pude confirmar la pantalla {state}."
            )

        return self._run_connected(operation)

    def close_current_app(self) -> str:
        """Close the foreground app identified by the TV itself."""

        async def operation(client: Any) -> str:
            app_id = str(client.tv_state.current_app_id or "").strip()
            if not app_id:
                raise LgTvError("El televisor no informo una aplicacion activa.")
            title = self._app_title(client, app_id) or app_id
            await client.close_app(app_id)
            return title

        return self._run_connected(operation)

    def show_message(self, message: str) -> str:
        """Show one bounded plain-text notification on the TV."""

        async def operation(client: Any) -> str:
            await client.send_message(message)
            return message

        return self._run_connected(operation)

    def power_off(self) -> bool:
        """Turn off the TV, confirm it left the active state and retry once."""

        if not self._is_webos_reachable():
            return True

        async def operation(client: Any) -> None:
            await client.power_off()

        for _attempt in range(2):
            try:
                self._run_connected(operation)
            except LgTvError:
                # webOS may drop the socket while shutting down; the state check decides.
                pass
            if self._wait_until_off():
                return True
        return False

    def _wait_until_off(self) -> bool:
        deadline = time.monotonic() + self.confirm_timeout
        while time.monotonic() < deadline:
            if self._is_confirmed_off():
                return True
            time.sleep(0.5)
        return self._is_confirmed_off()

    def _is_confirmed_off(self) -> bool:
        """Treat an unreachable service or a webOS standby state as powered off."""

        if not self._is_webos_reachable():
            return True
        try:
            return not self.status()["is_on"]
        except LgTvError:
            return False

    def power_on(self) -> bool:
        """Wake the TV and confirm that its webOS control service becomes reachable."""

        if not self.mac:
            raise LgTvError("LG_TV_MAC no esta configurado para encender la television.")
        compact_mac = self.mac.replace(":", "").replace("-", "")
        try:
            mac_bytes = bytes.fromhex(compact_mac)
        except ValueError as error:
            raise LgTvError("LG_TV_MAC no tiene un formato valido.") from error
        if len(mac_bytes) != 6:
            raise LgTvError("LG_TV_MAC no tiene un formato valido.")

        if self._is_webos_reachable():
            self._connect_in_background()
            return True

        packet =b"\xff" * 6 + mac_bytes * 16
        deadline = time.monotonic() + self.wake_timeout
        next_send = 0.0
        while time.monotonic() < deadline:
            now = time.monotonic()
            if now >= next_send:
                self._send_magic_packet(packet)
                next_send = now + 2.0
            if self._is_webos_reachable():
                self._connect_in_background()
                return True
            time.sleep(0.5)
        if self._is_webos_reachable():
            self._connect_in_background()
            return True
        return False

    def _connect_in_background(self) -> None:
        """Open the shared connection right after waking, before the next command.

        A command that arrives meanwhile waits for this connection instead of
        opening a second one.
        """

        threading.Thread(target=self.warm, name="webos-warm", daemon=True).start()

    def _send_magic_packet(self, packet: bytes) -> None:
        """Send one WoL burst to both configured and limited broadcasts."""

        destinations = tuple(
            dict.fromkeys((self.broadcast_address, "255.255.255.255"))
        )
        sent = False
        last_error: OSError | None = None
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as wake_socket:
                wake_socket.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
                for destination in destinations:
                    for port in (9, 7):
                        try:
                            wake_socket.sendto(packet, (destination, port))
                            sent = True
                        except OSError as error:
                            last_error = error
        except OSError as error:
            raise LgTvError("No pude enviar la señal Wake-on-LAN.") from error
        if not sent:
            raise LgTvError("No pude enviar la señal Wake-on-LAN.") from last_error

    def _is_webos_reachable(self) -> bool:
        """Return whether either standard local webOS control port accepts TCP."""

        probe_timeout = min(self.connect_timeout, 0.4)
        for port in (3000, 3001):
            try:
                with socket.create_connection(
                    (self.host, port), timeout=probe_timeout
                ):
                    return True
            except OSError:
                continue
        return False

    def _run_connected(
        self, operation: Callable[[Any], Awaitable[Any]]
    ) -> Any:
        """Run one operation on the shared connection, opening it if needed."""

        try:
            return self._session.run(operation, timeout=90)
        except LgTvError:
            raise
        except asyncio.CancelledError as error:
            raise LgTvError(
                "La conexion con webOS se interrumpio durante la transicion; "
                "no pude confirmar la operacion."
            ) from error
        except TimeoutError as error:
            raise LgTvError(
                "La television no respondio a tiempo; comprueba que este encendida y en la misma red."
            ) from error
        except OSError as error:
            raise LgTvError(
                "No pude comunicarme con la television en la red local."
            ) from error
        except Exception as error:
            raise LgTvError(f"La television rechazo la operacion: {error}") from error

    async def _open_client(self) -> Any:
        """Open and register a new webOS connection; close it if anything fails."""

        try:
            from aiowebostv import WebOsClient
        except ImportError as error:
            raise LgTvError(
                "Falta aiowebostv. Instala las dependencias de requirements.txt."
            ) from error

        client = WebOsClient(
            self.host,
            client_key=self._read_client_key(),
            connect_timeout=self.connect_timeout,
        )
        try:
            connected = await asyncio.wait_for(client.connect(), timeout=60)
            if not connected:
                raise LgTvError("La television no acepto la conexion.")
        except BaseException:
            with suppress(Exception, asyncio.CancelledError):
                await asyncio.wait_for(client.disconnect(), timeout=3)
            raise
        if client.client_key:
            self._save_client_key(client.client_key)
        return client

    def _read_client_key(self) -> str | None:
        if not self.client_key_path.is_file():
            return None
        key = self.client_key_path.read_text(encoding="utf-8").strip()
        return key or None

    def _save_client_key(self, client_key: str) -> None:
        existing = self._read_client_key()
        if existing == client_key:
            return
        self.client_key_path.parent.mkdir(parents=True, exist_ok=True)
        self.client_key_path.write_text(client_key, encoding="utf-8")
        with suppress(OSError):
            self.client_key_path.chmod(stat.S_IRUSR | stat.S_IWUSR)

    @staticmethod
    def _app_title(client: Any, app_id: str) -> str:
        app = client.tv_state.apps.get(app_id, {})
        if isinstance(app, dict):
            title = str(app.get("title") or app.get("name") or "").strip()
            if title:
                return title
        return {
            "com.webos.app.home": "Inicio",
            "com.webos.app.livetv": "TV en Vivo",
        }.get(app_id, "")

    async def _wait_for_current_app(self, client: Any, app_id: str) -> bool:
        deadline = time.monotonic() + self.confirm_timeout
        while time.monotonic() < deadline:
            current = str(await client.get_current_app() or "").strip()
            if current == app_id:
                return True
            await asyncio.sleep(0.35)
        return False

    async def _wait_for_input(self, client: Any, input_id: str) -> bool:
        expected = _compact_name(input_id)
        deadline = time.monotonic() + self.confirm_timeout
        while time.monotonic() < deadline:
            current = _compact_name(str(await client.get_input() or ""))
            if current and (expected in current or current in expected):
                return True
            await asyncio.sleep(0.35)
        return False

    async def _wait_for_screen_state(self, client: Any, on: bool) -> bool:
        deadline = time.monotonic() + self.confirm_timeout
        while time.monotonic() < deadline:
            payload = await client.get_power_state()
            state = _normalize_name(str(payload.get("state") or ""))
            screen_on = state not in {
                "",
                "screen off",
                "power off",
                "suspend",
                "active standby",
            }
            if screen_on is on:
                return True
            await asyncio.sleep(0.35)
        return False

    async def _wait_for_channel_change(
        self, client: Any, before_key: tuple[str, str, str]
    ) -> dict[str, Any] | None:
        deadline = time.monotonic() + self.confirm_timeout
        while time.monotonic() < deadline:
            current = await client.get_current_channel()
            if current and self._channel_key(current) != before_key:
                return current
            await asyncio.sleep(0.35)
        return None

    async def _wait_for_selected_channel(
        self, client: Any, selected: dict[str, Any]
    ) -> dict[str, Any] | None:
        expected = self._channel_key(selected)
        deadline = time.monotonic() + self.confirm_timeout
        while time.monotonic() < deadline:
            current = await client.get_current_channel()
            current_key = self._channel_key(current)
            if current and any(
                actual and wanted and actual == wanted
                for actual, wanted in zip(current_key, expected)
            ):
                return current
            await asyncio.sleep(0.35)
        return None

    @staticmethod
    def _resolve_app(
        apps: dict[str, Any], requested_name: str
    ) -> tuple[str, str]:
        aliases = {
            "amazon prime": "prime video",
            "amazon prime video": "prime video",
            "disney plus": "disney+",
        }
        requested = _normalize_name(requested_name)
        requested = aliases.get(requested, requested)
        exact: list[tuple[str, str]] = []
        partial: list[tuple[str, str]] = []
        for app_id, app in apps.items():
            if not isinstance(app, dict):
                continue
            title = str(app.get("title") or app.get("name") or app_id).strip()
            searchable = {_normalize_name(app_id), _normalize_name(title)}
            candidate = (str(app_id), title)
            if requested in searchable:
                exact.append(candidate)
            elif any(requested in value for value in searchable):
                partial.append(candidate)
        matches = exact or partial
        if not matches:
            raise LgTvError(
                f"No encontre una aplicacion instalada llamada {requested_name}."
            )
        unique = list(dict.fromkeys(matches))
        if len(unique) > 1:
            names = ", ".join(title for _, title in unique[:5])
            raise LgTvError(f"La aplicacion es ambigua. Coinciden: {names}.")
        return unique[0]

    @staticmethod
    def _resolve_input(
        inputs: dict[str, Any], requested_name: str
    ) -> tuple[str, str]:
        requested = _normalize_name(requested_name).replace("hdmi_", "hdmi ")
        for app_id, item in inputs.items():
            if not isinstance(item, dict):
                continue
            label = str(item.get("label") or item.get("name") or app_id).strip()
            values = {
                _normalize_name(label).replace("hdmi_", "hdmi "),
                _normalize_name(str(app_id)).replace("hdmi_", "hdmi "),
            }
            if requested not in values and not any(value.endswith(requested) for value in values):
                continue
            input_id = str(item.get("id") or "").strip()
            if not input_id:
                raise LgTvError(f"La entrada {label} no entrego un identificador utilizable.")
            return input_id, label
        raise LgTvError(f"El televisor no publico la entrada {requested_name}.")

    @staticmethod
    def _resolve_channel(
        channels: list[dict[str, Any]], requested: str
    ) -> dict[str, Any]:
        query = _normalize_name(requested)
        if not query:
            raise LgTvError("Necesito el numero o nombre del canal.")
        exact: list[dict[str, Any]] = []
        partial: list[dict[str, Any]] = []
        for channel in channels:
            if not isinstance(channel, dict):
                continue
            number = _normalize_name(str(channel.get("channelNumber") or ""))
            name = _normalize_name(str(channel.get("channelName") or ""))
            if query in {number, name}:
                exact.append(channel)
            elif query and (query in name or query in number):
                partial.append(channel)
        matches = exact or partial
        if not matches:
            raise LgTvError(f"No encontre un canal llamado {requested}.")
        if len(matches) > 1:
            labels = ", ".join(
                LgTvClient._channel_label(channel) for channel in matches[:5]
            )
            raise LgTvError(f"El canal es ambiguo. Coinciden: {labels}.")
        return matches[0]

    @staticmethod
    def _channel_key(channel: dict[str, Any] | None) -> tuple[str, str, str]:
        channel = channel or {}
        return (
            str(channel.get("channelId") or "").strip(),
            _normalize_name(str(channel.get("channelNumber") or "")),
            _normalize_name(str(channel.get("channelName") or "")),
        )

    @staticmethod
    def _channel_summary(channel: dict[str, Any]) -> dict[str, str]:
        return {
            "id": str(channel.get("channelId") or "").strip(),
            "number": str(channel.get("channelNumber") or "").strip(),
            "name": str(channel.get("channelName") or "").strip(),
        }

    @staticmethod
    def _channel_label(channel: dict[str, Any]) -> str:
        return " ".join(
            value
            for value in (
                str(channel.get("channelNumber") or "").strip(),
                str(channel.get("channelName") or "").strip(),
            )
            if value
        ) or "desconocido"


class LgTvToolsExecutor:
    """Dispatch only the LG TV operations exposed by the validated catalog."""

    supports_model_tools = True
    supports_lg_tv_tools = True

    def __init__(
        self,
        settings: Settings | None = None,
        client: LgTvController | None = None,
        scheduler: LgTvScheduler | None = None,
    ) -> None:
        if client is None:
            if settings is None:
                raise ValueError("Se necesita configuracion para construir LG TV.")
            client = LgTvClient(settings)
        self.client = client
        self.scheduler_error = ""
        if scheduler is None and settings is not None:
            try:
                scheduler = LgTvScheduleManager(
                    getattr(settings, "lg_tv_schedules_path", ".lg_tv_schedules.json"),
                    power_off=client.power_off,
                    show_message=client.show_message,
                )
            except LgTvScheduleError as error:
                self.scheduler_error = str(error)
        self.scheduler = scheduler

    def warm(self) -> None:
        warm = getattr(self.client, "warm", None)
        if callable(warm):
            warm()

    def execute(self, tool_name: str, args: dict[str, object]) -> dict[str, object]:
        """Execute a validated LG TV tool and return a spoken-safe result."""

        try:
            return self._execute(tool_name, args)
        except (LgTvError, LgTvScheduleError, ValueError, TypeError) as error:
            return {"ok": False, "tool_name": tool_name, "message": str(error)}

    def _execute(self, tool_name: str, args: dict[str, object]) -> dict[str, object]:
        if tool_name == "lg_tv.status":
            detail = _enum_arg(
                args,
                "detail",
                {"full", "power", "app", "channel", "screen", "sound", "volume", "mute"},
                default="full",
            )
            status = self.client.status()
            if status.get("reachable") is False:
                if detail in {"full", "power"}:
                    message = (
                        "La television no responde en la red local; "
                        "probablemente esta apagada."
                    )
                else:
                    labels = {
                        "app": "la aplicacion activa",
                        "channel": "el canal",
                        "screen": "el estado de la pantalla",
                        "sound": "la salida de sonido",
                        "volume": "el volumen",
                        "mute": "el estado de silencio",
                    }
                    message = (
                        f"No pude consultar {labels[detail]} porque la television "
                        "no responde; probablemente esta apagada."
                    )
                return {
                    "ok": True,
                    "direct_answer": True,
                    "tool_name": tool_name,
                    "message": message,
                    "state": status,
                }
            power = "encendida" if status.get("is_on") else "apagada"
            app = str(status.get("current_app", "")).strip()
            screen_on = status.get("screen_on")
            channel_name = str(status.get("channel_name", "")).strip()
            channel_number = str(status.get("channel_number", "")).strip()
            sound_output = _sound_output_name(
                str(status.get("sound_output", "")).strip()
            )
            volume = status.get("volume")
            muted = status.get("muted")
            if detail != "full":
                messages = {
                    "power": f"La television esta {power}.",
                    "app": (
                        f"La aplicacion activa es {app}."
                        if app
                        else "No pude determinar la aplicacion activa."
                    ),
                    "channel": (
                        "El canal actual es "
                        + " ".join(
                            value for value in (channel_number, channel_name) if value
                        )
                        + "."
                        if channel_name or channel_number
                        else (
                            f"No estas viendo un canal; la television esta en {app}."
                            if app
                            else "La television no informo un canal activo."
                        )
                    ),
                    "screen": (
                        "La pantalla de la television esta encendida."
                        if screen_on is True
                        else "La pantalla de la television esta apagada."
                    ),
                    "sound": (
                        f"La television esta usando {sound_output}."
                        if sound_output
                        else "La television no informo la salida de sonido."
                    ),
                    "volume": (
                        f"El volumen de la television esta en {volume}."
                        if volume is not None
                        else "La television no informo el volumen."
                    ),
                    "mute": (
                        "La television esta en silencio."
                        if muted is True
                        else (
                            "La television no esta en silencio."
                            if muted is False
                            else "La television no informo el estado de silencio."
                        )
                    ),
                }
                return {
                    "ok": True,
                    "direct_answer": True,
                    "tool_name": tool_name,
                    "message": messages[detail],
                    "state": status,
                }
            details = [f"La television esta {power}"]
            if app:
                details.append(f"con {app} activo")
            if channel_name or channel_number:
                channel = " ".join(
                    value for value in (channel_number, channel_name) if value
                )
                details.append(f"en el canal {channel}")
            if screen_on is False:
                details.append("con la pantalla apagada")
            if sound_output:
                details.append(f"usando {sound_output}")
            if volume is not None:
                details.append(f"y volumen {volume}")
            if muted is True:
                details.append("en silencio")
            return {
                "ok": True,
                "direct_answer": True,
                "tool_name": tool_name,
                "message": ", ".join(details) + ".",
                "state": status,
            }
        if tool_name == "lg_tv.set_volume":
            volume = _volume_arg(args)
            confirmed = self.client.set_volume(volume)
            return _success(tool_name, f"Volumen de la television ajustado a {confirmed}.")
        if tool_name == "lg_tv.volume_up":
            confirmed = self.client.step_volume("up")
            suffix = f" Ahora esta en {confirmed}." if confirmed is not None else ""
            return _success(tool_name, f"Subi el volumen de la television.{suffix}")
        if tool_name == "lg_tv.volume_down":
            confirmed = self.client.step_volume("down")
            suffix = f" Ahora esta en {confirmed}." if confirmed is not None else ""
            return _success(tool_name, f"Baje el volumen de la television.{suffix}")
        if tool_name == "lg_tv.set_mute":
            muted = args.get("muted")
            if not isinstance(muted, bool):
                raise LgTvError("El estado de silencio debe ser verdadero o falso.")
            confirmed = self.client.set_mute(muted)
            message = "Television silenciada." if confirmed else "Quite el silencio de la television."
            return _success(tool_name, message)
        if tool_name == "lg_tv.launch_app":
            app_name = _required_name(args, "app_name")
            launched = self.client.launch_app(app_name)
            return _success(tool_name, f"Abri {launched} en la television.")
        if tool_name == "lg_tv.select_input":
            input_name = _required_name(args, "input")
            if input_name not in {"HDMI 1", "HDMI 2", "HDMI 3", "HDMI 4"}:
                raise LgTvError("Solo se permiten las entradas HDMI 1 a HDMI 4.")
            selected = self.client.select_input(input_name)
            return _success(tool_name, f"Cambie la television a {selected}.")
        if tool_name == "lg_tv.playback":
            action = _enum_arg(
                args,
                "action",
                {"play", "pause", "stop", "rewind", "fast_forward"},
            )
            self.client.playback(action)
            messages = {
                "play": "Reanude la reproduccion de la television.",
                "pause": "Pause la reproduccion de la television.",
                "stop": "Detuve la reproduccion de la television.",
                "rewind": "Retrocedi la reproduccion de la television.",
                "fast_forward": "Adelante la reproduccion de la television.",
            }
            return _success(tool_name, messages[action])
        if tool_name == "lg_tv.navigate":
            action = _enum_arg(
                args,
                "action",
                {"home", "back", "up", "down", "left", "right", "ok"},
            )
            self.client.navigate(action)
            messages = {
                "home": "Abri el inicio de la television.",
                "back": "Volvi atras en la television.",
                "up": "Movi la seleccion hacia arriba.",
                "down": "Movi la seleccion hacia abajo.",
                "left": "Movi la seleccion hacia la izquierda.",
                "right": "Movi la seleccion hacia la derecha.",
                "ok": "Confirme la seleccion en la television.",
            }
            return _success(tool_name, messages[action])
        if tool_name in {"lg_tv.channel_up", "lg_tv.channel_down"}:
            direction = "up" if tool_name.endswith("_up") else "down"
            channel = self.client.step_channel(direction)
            message = "Subi" if direction == "up" else "Baje"
            return _success(
                tool_name,
                f"{message} un canal; la television confirmo {_channel_result_label(channel)}.",
            )
        if tool_name == "lg_tv.select_channel":
            query = _required_name(args, "query")
            channel = self.client.select_channel(query)
            return _success(
                tool_name,
                f"Cambie al canal {_channel_result_label(channel)} y la television lo confirmo.",
            )
        if tool_name in {"lg_tv.screen_on", "lg_tv.screen_off"}:
            on = tool_name.endswith("_on")
            self.client.set_screen(on)
            message = "Encendi" if on else "Apague"
            return _success(
                tool_name,
                f"{message} la pantalla sin apagar la television.",
            )
        if tool_name == "lg_tv.close_app":
            app = self.client.close_current_app()
            return _success(tool_name, f"Cerre {app} en la television.")
        if tool_name == "lg_tv.show_message":
            message = _message_arg(args)
            self.client.show_message(message)
            return _success(tool_name, "Mostre el mensaje en la television.")
        if tool_name in {
            "lg_tv.schedule_power_off",
            "lg_tv.schedule_message",
            "lg_tv.cancel_timers",
            "lg_tv.timer_status",
        }:
            return self._execute_timer(tool_name, args)
        if tool_name == "lg_tv.power_off":
            if self.client.power_off():
                return _success(tool_name, "Apague la television.")
            return {
                "ok": False,
                "direct_answer": True,
                "tool_name": tool_name,
                "message": (
                    "Envie la orden de apagado dos veces, pero la television "
                    "sigue encendida."
                ),
            }
        if tool_name == "lg_tv.power_on":
            confirmed = self.client.power_on()
            if confirmed:
                return {
                    "ok": True,
                    "direct_answer": True,
                    "tool_name": tool_name,
                    "message": "La television respondio en la red local; esta encendida.",
                }
            return {
                "ok": False,
                "direct_answer": True,
                "tool_name": tool_name,
                "message": (
                    "Envie varias señales de encendido, pero no pude confirmar "
                    "que la television despertara."
                ),
            }
        raise LgTvError(f"LG TV no conoce la tool: {tool_name}")

    def _execute_timer(
        self, tool_name: str, args: dict[str, object]
    ) -> dict[str, object]:
        if self.scheduler is None:
            raise LgTvError(
                self.scheduler_error
                or "Los temporizadores de television no estan configurados."
            )
        if tool_name == "lg_tv.schedule_power_off":
            minutes = _minutes_arg(args)
            item = self.scheduler.schedule("power_off", minutes)
            result = _success(
                tool_name,
                f"Programe el apagado de la television para dentro de {minutes} minutos.",
            )
            result["timer"] = item
            return result
        if tool_name == "lg_tv.schedule_message":
            minutes = _minutes_arg(args)
            message = _message_arg(args)
            item = self.scheduler.schedule(
                "show_message", minutes, message=message
            )
            result = _success(
                tool_name,
                f"Programe el mensaje para dentro de {minutes} minutos.",
            )
            result["timer"] = item
            return result
        if tool_name == "lg_tv.cancel_timers":
            scope = _enum_arg(
                args,
                "scope",
                {"all", "power_off", "messages"},
                default="all",
            )
            action = {
                "all": None,
                "power_off": "power_off",
                "messages": "show_message",
            }[scope]
            removed = self.scheduler.cancel(action)
            if not removed:
                return {
                    "ok": True,
                    "direct_answer": True,
                    "tool_name": tool_name,
                    "message": "No habia temporizadores de television que cancelar.",
                }
            noun = "temporizador" if removed == 1 else "temporizadores"
            return _success(tool_name, f"Cancele {removed} {noun} de television.")

        pending = self.scheduler.pending()
        if not pending:
            return {
                "ok": True,
                "direct_answer": True,
                "tool_name": tool_name,
                "message": "No hay temporizadores pendientes para la television.",
                "timers": [],
            }
        now = time.time()
        descriptions = []
        for item in pending:
            remaining = max(0, round((float(item["execute_at"]) - now) / 60))
            if item.get("action") == "power_off":
                descriptions.append(f"apagado en {remaining} minutos")
            else:
                descriptions.append(f"mensaje en {remaining} minutos")
        return {
            "ok": True,
            "direct_answer": True,
            "tool_name": tool_name,
            "message": "Temporizadores pendientes: " + "; ".join(descriptions) + ".",
            "timers": pending,
        }


def _success(tool_name: str, message: str) -> dict[str, object]:
    return {"ok": True, "tool_name": tool_name, "message": message}


def _volume_arg(args: dict[str, object]) -> int:
    value = args.get("volume")
    if not isinstance(value, int) or isinstance(value, bool) or not 0 <= value <= 100:
        raise LgTvError("El volumen debe ser un entero entre 0 y 100.")
    return value


def _minutes_arg(args: dict[str, object]) -> int:
    value = args.get("minutes")
    if (
        not isinstance(value, int)
        or isinstance(value, bool)
        or not 1 <= value <= 1440
    ):
        raise LgTvError("El temporizador debe estar entre 1 minuto y 24 horas.")
    return value


def _required_name(args: dict[str, object], key: str) -> str:
    value = args.get(key)
    if not isinstance(value, str) or not value.strip():
        raise LgTvError(f"Falta el dato requerido: {key}.")
    cleaned = value.strip()
    if len(cleaned) > 80:
        raise LgTvError(f"El dato {key} es demasiado largo.")
    return cleaned


def _enum_arg(
    args: dict[str, object],
    key: str,
    allowed: set[str],
    default: str | None = None,
) -> str:
    value = args.get(key, default)
    if not isinstance(value, str) or value not in allowed:
        raise LgTvError(f"El valor de {key} no esta permitido.")
    return value


def _message_arg(args: dict[str, object]) -> str:
    value = args.get("message")
    if not isinstance(value, str):
        raise LgTvError("Falta el mensaje para la television.")
    cleaned = " ".join(value.split())
    if not cleaned:
        raise LgTvError("El mensaje para la television no puede estar vacio.")
    if len(cleaned) > 160:
        raise LgTvError("El mensaje para la television no puede superar 160 caracteres.")
    return cleaned


def _sound_output_name(value: str) -> str:
    return {
        "tv_speaker": "los altavoces del televisor",
        "external_arc": "la salida HDMI ARC o eARC",
        "external_optical": "la salida optica",
        "headphone": "los audifonos",
        "bt_soundbar": "la barra de sonido Bluetooth",
    }.get(value, value.replace("_", " "))


def _normalize_name(value: str) -> str:
    decomposed = unicodedata.normalize("NFKD", value.casefold())
    without_accents = "".join(
        char for char in decomposed if not unicodedata.combining(char)
    )
    return " ".join(without_accents.replace("-", " ").split())


def _compact_name(value: str) -> str:
    return "".join(char for char in _normalize_name(value) if char.isalnum())


def _channel_result_label(channel: dict[str, str]) -> str:
    return " ".join(
        value
        for value in (channel.get("number", ""), channel.get("name", ""))
        if value
    ) or "el nuevo canal"
