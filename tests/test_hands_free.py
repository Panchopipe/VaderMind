"""Tests for the hands-free loop: wake word, follow-up window, errors, power off."""

from __future__ import annotations

import unittest

from vadermind.brain import BrainError
from vadermind.hands_free import (
    MIC_FAILED,
    MODE_SWITCHED,
    NOT_UNDERSTOOD,
    SHUTDOWN_CANCELLED,
    SHUTDOWN_FAREWELL,
    SHUTDOWN_QUESTION,
    HandsFreeSession,
)
from vadermind.lights import APAGADO, ERROR, ESCUCHANDO, PENSANDO, REPOSO, Lights
from vadermind.listening import ListeningError, VoiceTurn
from vadermind.voice_output import VoiceError


class RecordingLights(Lights):
    def __init__(self) -> None:
        self.states: list[str] = []

    def set_state(self, state: str) -> None:
        self.states.append(state)


class FakeTts:
    def __init__(self, fail: bool = False) -> None:
        self.spoken: list[str] = []
        self.fail = fail

    def speak(self, text: str) -> None:
        if self.fail:
            raise VoiceError("sin parlantes")
        self.spoken.append(text)


class FakeApp:
    def __init__(self, answers: dict[str, object] | None = None, tts: FakeTts | None = None) -> None:
        self.answers = answers or {}
        self.tts = tts or FakeTts()
        self.spoken_answer: str | None = None
        self.skip_post_turn_speech = False
        self.turns: list[str] = []

    def process_turn(self, user_text: str) -> str:
        self.turns.append(user_text)
        answer = self.answers.get(user_text, f"Respuesta a {user_text}")
        if isinstance(answer, Exception):
            raise answer
        return str(answer)


class FakeListener:
    def __init__(self, turns: list[object]) -> None:
        self.turns = list(turns)
        self.timeouts: list[float | None] = []

    def listen(self, no_speech_seconds: float | None = None) -> VoiceTurn:
        self.timeouts.append(no_speech_seconds)
        item = self.turns.pop(0)
        if isinstance(item, Exception):
            raise item
        return item  # type: ignore[return-value]


SILENCE = VoiceTurn(None, "No escuche nada.", reason="sin_voz")


def _session(app, listener, wakes=1, **options):
    lights = RecordingLights()
    remaining = [True] * wakes + [False]
    session = HandsFreeSession(
        app,
        listener,
        lights,
        lambda: remaining.pop(0),
        write=lambda _text: None,
        sleep=lambda _seconds: None,
        **options,
    )
    return session, lights


class HandsFreeTests(unittest.TestCase):
    def test_wake_warm_up_runs_once_per_wake_word_not_on_follow_ups(self) -> None:
        app = FakeApp()
        listener = FakeListener(
            [VoiceTurn("prende la tele"), VoiceTurn("sube el volumen"), SILENCE, VoiceTurn("hola"), SILENCE]
        )
        wakes: list[int] = []
        session, _lights = _session(
            app, listener, wakes=2, follow_up_seconds=10, on_wake=lambda: wakes.append(len(app.turns))
        )
        self.assertEqual(session.run(), 0)
        # before the first turn and before "hola"; never inside the follow-up window
        self.assertEqual(wakes, [0, 2])

    def test_conversation_with_follow_up_then_back_to_idle(self) -> None:
        app = FakeApp()
        listener = FakeListener([VoiceTurn("que hora es"), VoiceTurn("y en Madrid"), SILENCE])
        session, lights = _session(app, listener, follow_up_seconds=10)
        self.assertEqual(session.run(), 0)
        self.assertEqual(app.turns, ["que hora es", "y en Madrid"])
        self.assertEqual(app.tts.spoken, ["Respuesta a que hora es", "Respuesta a y en Madrid"])
        # first turn after the wake word, then two follow-up windows
        self.assertEqual(listener.timeouts, [None, 10, 10])
        self.assertEqual(
            lights.states,
            [REPOSO, ESCUCHANDO, PENSANDO, ESCUCHANDO, PENSANDO, ESCUCHANDO, REPOSO],
        )

    def test_spoken_answer_and_skip_are_respected(self) -> None:
        app = FakeApp()
        app.spoken_answer = "Texto para la voz"
        session, _ = _session(app, FakeListener([VoiceTurn("hola"), SILENCE]))
        session.run()
        self.assertEqual(app.tts.spoken, ["Texto para la voz"])

        quiet = FakeApp()
        quiet.skip_post_turn_speech = True
        session, _ = _session(quiet, FakeListener([VoiceTurn("pausa"), SILENCE]))
        session.run()
        self.assertEqual(quiet.tts.spoken, [])

    def test_not_understood_blinks_speaks_and_lets_user_repeat(self) -> None:
        app = FakeApp()
        listener = FakeListener(
            [VoiceTurn(None, "No entendi", reason="no_entendido"), VoiceTurn("hola"), SILENCE]
        )
        session, lights = _session(app, listener)
        session.run()
        self.assertEqual(app.tts.spoken[0], NOT_UNDERSTOOD)
        self.assertIn(ERROR, lights.states)
        self.assertEqual(app.turns, ["hola"])  # repeated without the wake word

    def test_model_error_is_spoken_and_returns_to_waiting(self) -> None:
        app = FakeApp({"hola": BrainError("Sin conexion con el modelo.")})
        session, lights = _session(app, FakeListener([VoiceTurn("hola")]))
        session.run()
        self.assertEqual(app.tts.spoken, ["Sin conexion con el modelo."])
        self.assertIn(ERROR, lights.states)

    def test_microphone_failure_is_spoken(self) -> None:
        app = FakeApp()
        session, lights = _session(app, FakeListener([ListeningError("se desconecto")]))
        self.assertEqual(session.run(), 0)
        self.assertEqual(app.tts.spoken, [MIC_FAILED])
        self.assertIn(ERROR, lights.states)

    def test_playback_failure_shows_error(self) -> None:
        app = FakeApp(tts=FakeTts(fail=True))
        session, lights = _session(app, FakeListener([VoiceTurn("hola")]))
        session.run()
        self.assertEqual(lights.states[-2:], [ERROR, REPOSO])


class SilenceTests(unittest.TestCase):
    def test_silence_in_follow_up_goes_back_to_idle_without_answer(self) -> None:
        app = FakeApp()
        listener = FakeListener([VoiceTurn("que hora es"), VoiceTurn("Vader, silencio.")])
        session, lights = _session(app, listener)
        session.run()
        self.assertEqual(app.turns, ["que hora es"])
        self.assertEqual(app.tts.spoken, ["Respuesta a que hora es"])
        self.assertEqual(lights.states[-1], REPOSO)
        self.assertEqual(len(listener.turns), 0)

    def test_silence_right_after_wake_word(self) -> None:
        app = FakeApp()
        session, _ = _session(app, FakeListener([VoiceTurn("Silencio")]))
        session.run()
        self.assertEqual(app.turns, [])
        self.assertEqual(app.tts.spoken, [])

    def test_zero_seconds_disables_follow_up(self) -> None:
        app = FakeApp()
        listener = FakeListener([VoiceTurn("hola")])
        session, _ = _session(app, listener, follow_up_seconds=0)
        session.run()
        self.assertEqual(listener.timeouts, [None])  # no second listen without the wake word


class PersonaModeTests(unittest.TestCase):
    def test_voice_switches_mode_and_keeps_listening(self) -> None:
        app = FakeApp()
        modes: list[str] = []
        listener = FakeListener([VoiceTurn("Modo canon"), VoiceTurn("hola"), SILENCE])
        session, _ = _session(app, listener, set_mode=modes.append)
        session.run()
        self.assertEqual(modes, ["canon"])
        self.assertEqual(app.tts.spoken, [MODE_SWITCHED["canon"], "Respuesta a hola"])
        self.assertEqual(app.turns, ["hola"])  # the command never reached the model
        self.assertEqual(listener.timeouts, [None, 10.0, 10.0])  # follow-up stays open

    def test_without_switcher_the_phrase_goes_to_the_model(self) -> None:
        app = FakeApp()
        session, _ = _session(app, FakeListener([VoiceTurn("modo chileno"), SILENCE]))
        session.run()
        self.assertEqual(app.turns, ["modo chileno"])


class ShutdownTests(unittest.TestCase):
    def test_confirmed_shutdown_powers_off(self) -> None:
        app = FakeApp()
        calls: list[bool] = []
        listener = FakeListener([VoiceTurn("Vader, apágate"), VoiceTurn("Sí")])
        session, lights = _session(
            app, listener, wakes=5, allow_shutdown=True, power_off=lambda: calls.append(True) or True
        )
        self.assertEqual(session.run(), 0)
        self.assertEqual(calls, [True])
        self.assertEqual(app.tts.spoken, [SHUTDOWN_QUESTION, SHUTDOWN_FAREWELL])
        self.assertEqual(lights.states[-1], APAGADO)
        self.assertEqual(app.turns, [])  # never reached the model

    def test_anything_but_yes_cancels(self) -> None:
        app = FakeApp()
        calls: list[bool] = []
        listener = FakeListener([VoiceTurn("apágate"), VoiceTurn("no, espera")])
        session, _ = _session(app, listener, allow_shutdown=True, power_off=lambda: calls.append(True) or True)
        session.run()
        self.assertEqual(calls, [])
        self.assertEqual(app.tts.spoken, [SHUTDOWN_QUESTION, SHUTDOWN_CANCELLED])

    def test_silence_after_question_cancels(self) -> None:
        app = FakeApp()
        listener = FakeListener([VoiceTurn("apágate"), SILENCE])
        session, _ = _session(app, listener, allow_shutdown=True, power_off=lambda: True)
        session.run()
        self.assertEqual(app.tts.spoken, [SHUTDOWN_QUESTION, SHUTDOWN_CANCELLED])

    def test_disabled_shutdown_goes_to_the_model(self) -> None:
        app = FakeApp()
        session, _ = _session(app, FakeListener([VoiceTurn("apágate"), SILENCE]))
        session.run()
        self.assertEqual(app.turns, ["apágate"])


if __name__ == "__main__":
    unittest.main()
