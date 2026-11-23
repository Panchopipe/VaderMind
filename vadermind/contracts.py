"""Stable public contracts for the core application."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol


@dataclass(slots=True)
class ModelTurn:
    """One model turn containing either conversational text or one tool request."""

    text: str = ""
    tool_name: str | None = None
    tool_args: dict[str, object] = field(default_factory=dict)


@dataclass(slots=True)
class SessionState:
    """Stores the in-memory conversation state for the current session."""

    language: str = "es"
    messages: list[dict[str, str]] = field(default_factory=list)
    total_turns: int = 0
    tool_context: dict[str, object] = field(default_factory=dict)


class ChatEngine(Protocol):
    """Generates assistant messages from user input and session state."""

    def respond(self, user_text: str, session_state: SessionState) -> str:
        """Return the assistant answer for the current turn."""


class TextToSpeech(Protocol):
    """Converts text into audible output."""

    def speak(self, text: str) -> None:
        """Play the spoken version of the provided text."""


class SpeechToText(Protocol):
    """Transforms audio input into text."""

    def transcribe(self, audio_input: object) -> str:
        """Return the transcription for the provided audio input."""


class ToolExecutor(Protocol):
    """Runs external tools behind a stable interface."""

    def execute(self, tool_name: str, args: dict[str, object]) -> dict[str, object]:
        """Execute a named tool and return structured output."""
