"""Persistent, local timers for a bounded set of LG TV actions."""

from __future__ import annotations

from collections.abc import Callable
import json
import logging
from pathlib import Path
import threading
import time
from uuid import uuid4


LOGGER = logging.getLogger("vadermind.lg_tv_scheduler")


class LgTvScheduleError(RuntimeError):
    """Raised when a local TV schedule cannot be stored safely."""


class LgTvScheduleManager:
    """Persist TV timers and execute overdue items in a daemon worker."""

    ALLOWED_ACTIONS = {"power_off", "show_message"}

    def __init__(
        self,
        path: str | Path,
        *,
        power_off: Callable[[], object],
        show_message: Callable[[str], object],
        start_worker: bool = True,
    ) -> None:
        self.path = Path(path)
        self._power_off = power_off
        self._show_message = show_message
        self._condition = threading.Condition()
        self._items = self._load()
        if start_worker:
            threading.Thread(
                target=self._worker,
                name="vadermind-lg-tv-timers",
                daemon=True,
            ).start()

    def schedule(
        self,
        action: str,
        delay_minutes: int,
        *,
        message: str = "",
    ) -> dict[str, object]:
        """Create one future action after validating its bounded payload."""

        if action not in self.ALLOWED_ACTIONS:
            raise LgTvScheduleError("La accion programada no esta permitida.")
        if (
            not isinstance(delay_minutes, int)
            or isinstance(delay_minutes, bool)
            or not 1 <= delay_minutes <= 1440
        ):
            raise LgTvScheduleError(
                "El temporizador debe estar entre 1 minuto y 24 horas."
            )
        cleaned_message = " ".join(message.split())
        if action == "show_message" and not cleaned_message:
            raise LgTvScheduleError("Falta el mensaje que debo mostrar mas tarde.")
        if len(cleaned_message) > 160:
            raise LgTvScheduleError(
                "El mensaje para la television no puede superar 160 caracteres."
            )

        item: dict[str, object] = {
            "id": uuid4().hex,
            "action": action,
            "execute_at": time.time() + delay_minutes * 60,
            "message": cleaned_message,
        }
        with self._condition:
            if action == "power_off":
                self._items = [
                    existing
                    for existing in self._items
                    if existing.get("action") != "power_off"
                ]
            self._items.append(item)
            self._items.sort(key=lambda value: float(value["execute_at"]))
            self._save_locked()
            self._condition.notify_all()
        return dict(item)

    def pending(self, action: str | None = None) -> list[dict[str, object]]:
        """Return a snapshot of pending timers in execution order."""

        if action is not None and action not in self.ALLOWED_ACTIONS:
            raise LgTvScheduleError("La accion programada no esta permitida.")
        with self._condition:
            return [
                dict(item)
                for item in self._items
                if action is None or item.get("action") == action
            ]

    def cancel(self, action: str | None = None) -> int:
        """Cancel all timers, or only those for one allowlisted action."""

        if action is not None and action not in self.ALLOWED_ACTIONS:
            raise LgTvScheduleError("La accion programada no esta permitida.")
        with self._condition:
            before = len(self._items)
            if action is None:
                self._items.clear()
            else:
                self._items = [
                    item for item in self._items if item.get("action") != action
                ]
            removed = before - len(self._items)
            if removed:
                self._save_locked()
                self._condition.notify_all()
            return removed

    def run_due(self, now: float | None = None) -> int:
        """Execute all currently overdue items; useful at startup and in tests."""

        completed = 0
        current_time = time.time() if now is None else now
        while True:
            with self._condition:
                due_index = next(
                    (
                        index
                        for index, item in enumerate(self._items)
                        if float(item["execute_at"]) <= current_time
                    ),
                    None,
                )
                if due_index is None:
                    return completed
                item = self._items.pop(due_index)
                # Persist before execution so a crash cannot repeat a power action.
                self._save_locked()
            try:
                self._execute(item)
            except Exception:
                LOGGER.exception(
                    "No se pudo ejecutar temporizador LG TV id=%s action=%s",
                    item.get("id"),
                    item.get("action"),
                )
            completed += 1

    def _worker(self) -> None:
        while True:
            self.run_due()
            with self._condition:
                if not self._items:
                    self._condition.wait(timeout=60.0)
                    continue
                delay = max(0.0, float(self._items[0]["execute_at"]) - time.time())
                self._condition.wait(timeout=min(delay, 60.0))

    def _execute(self, item: dict[str, object]) -> None:
        action = item.get("action")
        if action == "power_off":
            self._power_off()
            return
        if action == "show_message":
            self._show_message(str(item.get("message", "")))
            return
        raise LgTvScheduleError("El archivo contiene una accion no permitida.")

    def _load(self) -> list[dict[str, object]]:
        if not self.path.is_file():
            return []
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise LgTvScheduleError(
                f"No pude leer los temporizadores locales: {error}"
            ) from error
        raw_items = payload.get("items", []) if isinstance(payload, dict) else []
        items: list[dict[str, object]] = []
        for raw in raw_items:
            if not isinstance(raw, dict):
                continue
            action = raw.get("action")
            execute_at = raw.get("execute_at")
            message = str(raw.get("message", ""))
            if action not in self.ALLOWED_ACTIONS:
                continue
            if not isinstance(execute_at, (int, float)) or isinstance(execute_at, bool):
                continue
            if len(message) > 160:
                continue
            items.append(
                {
                    "id": str(raw.get("id") or uuid4().hex),
                    "action": action,
                    "execute_at": float(execute_at),
                    "message": message,
                }
            )
        return sorted(items, key=lambda value: float(value["execute_at"]))

    def _save_locked(self) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            temporary = self.path.with_name(self.path.name + ".tmp")
            temporary.write_text(
                json.dumps(
                    {"version": 1, "items": self._items},
                    indent=2,
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )
            temporary.replace(self.path)
        except OSError as error:
            raise LgTvScheduleError(
                f"No pude guardar los temporizadores locales: {error}"
            ) from error
