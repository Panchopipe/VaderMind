"""Base tool abstractions for future integrations."""

from __future__ import annotations

from vadermind.contracts import ToolExecutor


class NoToolsExecutor(ToolExecutor):
    """Default tool executor for early phases where tools are disabled."""

    def execute(self, tool_name: str, args: dict[str, object]) -> dict[str, object]:
        """Return a clear disabled-tool result instead of silently failing."""

        return {
            "ok": False,
            "tool_name": tool_name,
            "message": "Las tools externas todavía no están activadas en esta versión.",
            "args": args,
        }


class RoutedToolsExecutor(ToolExecutor):
    """Dispatch provider-neutral tool names to their local executor by prefix."""

    def __init__(self, executors: dict[str, ToolExecutor]) -> None:
        self.executors = dict(executors)
        self.supports_spotify_tools = "spotify." in self.executors
        self.supports_lg_tv_tools = "lg_tv." in self.executors
        self.supports_media_tools = "media." in self.executors
        self.supports_model_tools = bool(self.executors)

    def execute(self, tool_name: str, args: dict[str, object]) -> dict[str, object]:
        """Execute only a tool whose namespace was registered explicitly."""

        for prefix, executor in self.executors.items():
            if tool_name.startswith(prefix):
                return executor.execute(tool_name, args)
        return {
            "ok": False,
            "tool_name": tool_name,
            "message": "La herramienta solicitada no esta habilitada en esta sesion.",
        }

    def warm(self) -> None:
        """Let each executor open its connections before the first command."""

        for executor in self.executors.values():
            warm = getattr(executor, "warm", None)
            if callable(warm):
                warm()

    def prefetch(self, tool_name: str, args: dict[str, object]) -> None:
        """Forward read-only warm-up to the owning executor, if it offers one."""

        for prefix, executor in self.executors.items():
            if tool_name.startswith(prefix):
                prefetch = getattr(executor, "prefetch", None)
                if prefetch is not None:
                    prefetch(tool_name, args)
                return
