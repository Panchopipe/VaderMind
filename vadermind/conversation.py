"""Conversation memory helpers."""

from __future__ import annotations

from vadermind.contracts import SessionState


MAX_HISTORY_MESSAGES = 8


def append_user_message(session_state: SessionState, text: str) -> None:
    """Append a user message to the current session."""

    session_state.messages.append({"role": "user", "content": text})


def append_assistant_message(session_state: SessionState, text: str) -> None:
    """Append an assistant message and increase the completed turn count."""

    session_state.messages.append({"role": "assistant", "content": text})
    session_state.total_turns += 1


def recent_history(session_state: SessionState, limit: int = MAX_HISTORY_MESSAGES) -> list[dict[str, str]]:
    """Return the most recent messages to keep prompts small and predictable."""

    if limit <= 0:
        return []
    return session_state.messages[-limit:]
