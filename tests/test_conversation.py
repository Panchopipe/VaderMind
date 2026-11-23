"""Tests for session memory helpers."""

from __future__ import annotations

import unittest

from vadermind.conversation import (
    append_assistant_message,
    append_user_message,
    recent_history,
)
from vadermind.contracts import SessionState


class ConversationTests(unittest.TestCase):
    """Verify that memory behaves predictably."""

    def test_recent_history_keeps_latest_messages(self) -> None:
        """Only the most recent messages should be returned."""

        session_state = SessionState()
        for index in range(6):
            append_user_message(session_state, f"user-{index}")
            append_assistant_message(session_state, f"assistant-{index}")

        history = recent_history(session_state, limit=4)

        self.assertEqual(len(history), 4)
        self.assertEqual(history[0]["content"], "user-4")
        self.assertEqual(history[-1]["content"], "assistant-5")

    def test_total_turns_grows_after_assistant_reply(self) -> None:
        """Completed turns should count assistant answers."""

        session_state = SessionState()
        append_user_message(session_state, "Hola")
        append_assistant_message(session_state, "Saludos")

        self.assertEqual(session_state.total_turns, 1)


if __name__ == "__main__":
    unittest.main()
