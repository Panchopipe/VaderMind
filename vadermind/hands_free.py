"""Hands-free Vader: wait for "Oye Vader", talk, and keep listening a little.

The loop, with what the eyes show at each step:

1. reposo: dim pupils while the local detector waits for "Oye Vader".
2. escuchando: start cue, record until the user goes quiet.
3. pensando: transcription and the model work on the answer.
4. hablando: the ring follows Vader's voice (driven by the speech observer).
5. Follow-up window: Vader listens again for `follow_up_seconds` without the
   wake word; if nobody speaks, back to 1. `follow_up_seconds=0` disables it.
   "Vader, silencio" (or "Oye Vader... silencio") goes back to 1 at once, so
   people can talk next to Vader without him taking it as an order.
   "Vader, modo canon" / "modo chileno" switches the personality until the
   next restart; the default stays in VADER_PERSONA_MODE.

Failures are never silent: the eyes blink (error) and Vader says what
failed, because nobody is reading a console. "Vader, apágate" is a fixed
local command, confirmed by voice, that never reaches the language model.
"""

from __future__ import annotations

import logging
import time
from typing import Callable, Protocol

from vadermind.brain import BrainError
from vadermind.lights import (
    APAGADO,
    ERROR,
    ERROR_SECONDS,
    ESCUCHANDO,
    PENSANDO,
    REPOSO,
    Lights,
)
from vadermind.listening import ListeningError, VoiceTurn
from vadermind.telemetry import TURN_TIMELINE, cpu_temperature
from vadermind.voice_output import VoiceError
from vadermind.wake_word import (
    is_confirmation,
    is_shutdown_request,
    is_silence_request,
    persona_mode_request,
)

LOGGER = logging.getLogger("vadermind.hands_free")

NOT_UNDERSTOOD = "No te entendi. Repitelo."
NOT_TRANSCRIBED = "No pude escucharte bien. Repitelo."
MIC_FAILED = "Mi microfono no responde."
SHUTDOWN_QUESTION = "¿Ordenas que me apague? Responde si o no."
SHUTDOWN_CANCELLED = "Como desees. Sigo aqui."
SHUTDOWN_FAREWELL = "Me retiro."
SHUTDOWN_FAILED = "No pude apagarme."
# Said in the new mode, so the answer itself shows the change worked.
MODE_SWITCHED = {"canon": "Como ordenes. Modo canon.", "chileno": "Ya po. Modo chileno."}


class _App(Protocol):
    tts: object
    spoken_answer: str | None
    skip_post_turn_speech: bool

    def process_turn(self, user_text: str) -> str: ...


class _Listener(Protocol):
    def listen(self, no_speech_seconds: float | None = None) -> VoiceTurn: ...


class HandsFreeSession:
    def __init__(
        self,
        app: _App,
        listener: _Listener,
        lights: Lights,
        wait_for_wake: Callable[[], bool],
        *,
        follow_up_seconds: float = 10.0,
        allow_shutdown: bool = False,
        power_off: Callable[[], bool] = lambda: False,
        set_mode: Callable[[str], None] | None = None,
        write: Callable[[str], None] = print,
        sleep: Callable[[float], None] = time.sleep,
        wake_marks: Callable[[], dict[str, float]] = dict,
        on_idle: Callable[[], None] | None = None,
        on_wake: Callable[[], None] | None = None,
    ) -> None:
        self.app = app
        self.listener = listener
        self.lights = lights
        self.wait_for_wake = wait_for_wake
        self.follow_up_seconds = follow_up_seconds
        self.allow_shutdown = allow_shutdown
        self.power_off = power_off
        self.set_mode = set_mode
        self.write = write
        self.sleep = sleep
        self.powered_off = False
        # Timing only (see `_log_timings`): it never changes what Vader does.
        self.wake_marks = wake_marks
        self.on_idle = on_idle
        # Read-only warm-ups that must not delay listening (e.g. the TV
        # connection): it runs once per "Oye Vader", not on follow-up turns.
        self.on_wake = on_wake
        self._carry: dict[str, float] = {}  # marks from before the turn's recording
        self._turns = 0
        self._kind: str | None = None

    def run(self) -> int:
        """Loop until the wake wait gives up or Vader powers off."""

        follow_up = False
        while not self.powered_off:
            if not follow_up:
                self.lights.set_state(REPOSO)
                self.write("(esperando \"Oye Vader\"...)")
                if self.on_idle:
                    self.on_idle()
                if not self.wait_for_wake():
                    return 0
                if self.on_wake:
                    self.on_wake()
                self.write("(Oye Vader)")
                self._carry = self.wake_marks()
            follow_up = self.turn(follow_up) and self.follow_up_seconds > 0
        return 0

    def turn(self, follow_up: bool = False) -> bool:
        """One spoken exchange; True keeps the follow-up window open."""

        self._kind = None
        keep_listening = self._turn(follow_up)
        ended = time.perf_counter()
        TURN_TIMELINE.mark("fin_turno", ended)
        self._log_timings()
        # A follow-up turn measures how long Vader takes to listen again.
        self._carry = {"turno_anterior_fin": ended}
        return keep_listening

    def _turn(self, follow_up: bool) -> bool:
        heard = self._listen(self.follow_up_seconds if follow_up else None)
        if heard is None:
            return False
        if heard.text is None:
            if heard.reason == "sin_voz":
                return False  # nobody spoke: quietly back to waiting
            self._kind = heard.reason or "no_entendido"
            self.write(f"VaderMind > {heard.notice}")
            self._fail(NOT_TRANSCRIBED if heard.reason == "sin_transcripcion" else NOT_UNDERSTOOD)
            return True  # let the user repeat without the wake word
        self.write(f"Tu (voz) > {heard.text}")
        if is_silence_request(heard.text):
            self._kind = "local_silencio"
            self.write("(silencio: vuelvo al reposo)")
            return False  # no answer: the end cue and the dim pupils confirm it
        mode = persona_mode_request(heard.text) if self.set_mode else None
        if mode:
            self._kind = "local_modo"
            return self._switch_mode(mode)
        if self.allow_shutdown and is_shutdown_request(heard.text):
            self._kind = "local_apagado"
            return self._shutdown()
        return self._respond(heard.text)

    def _log_timings(self) -> None:
        """One `Tiempos [...]` line per turn the user spoke in (journalctl keeps it)."""

        if "fin_habla" not in TURN_TIMELINE.marks:
            return  # nobody spoke: nothing to measure
        self._turns += 1
        TURN_TIMELINE.note("turno", self._turns)
        TURN_TIMELINE.note("via", "seguimiento" if "turno_anterior_fin" in TURN_TIMELINE.marks else "palabra")
        temperature = cpu_temperature()
        if temperature:
            TURN_TIMELINE.note("temp", temperature)
        LOGGER.info(TURN_TIMELINE.report(self._kind))

    def _switch_mode(self, mode: str) -> bool:
        """Change the personality until the next restart (the default lives in .env)."""

        self.set_mode(mode)  # type: ignore[misc]
        self.write(f"(modo {mode})")
        self._say(MODE_SWITCHED[mode])
        return True  # keep listening: the next question already gets the new mode

    def _listen(self, no_speech_seconds: float | None) -> VoiceTurn | None:
        self.lights.set_state(ESCUCHANDO)
        try:
            heard = self.listener.listen(no_speech_seconds=no_speech_seconds)
        except ListeningError as error:
            LOGGER.warning("Escucha fallida: %s", error)
            self.write(f"VaderMind > {error}")
            self._fail(MIC_FAILED)
            return None
        # listen() starts a fresh timeline: add what happened before it began.
        for name, at in self._carry.items():
            TURN_TIMELINE.mark(name, at)
        self._carry = {}
        return heard

    def _respond(self, text: str) -> bool:
        self.lights.set_state(PENSANDO)
        try:
            answer = self.app.process_turn(text)
        except ValueError as error:
            self.write(f"VaderMind > {error}")
            self.lights.set_state(ERROR)
            return False
        except BrainError as error:
            self._kind = "error_modelo"
            self.write(f"Vader > {error}")
            self._fail(str(error))
            return False
        if answer:
            self.write(f"Vader > {answer}")
        if answer and not self.app.skip_post_turn_speech:
            if not self._say(self.app.spoken_answer or answer):
                return False
        return True

    def _shutdown(self) -> bool:
        self._say(SHUTDOWN_QUESTION)
        reply = self._listen(None)
        if reply is None:
            return False
        if reply.text is None or not is_confirmation(reply.text):
            if reply.text:
                self.write(f"Tu (voz) > {reply.text}")
            self._say(SHUTDOWN_CANCELLED)
            return False
        self.write(f"Tu (voz) > {reply.text}")
        self._say(SHUTDOWN_FAREWELL)
        self.lights.set_state(APAGADO)
        if self.power_off():
            self.powered_off = True
            return False
        self._fail(SHUTDOWN_FAILED)
        return False

    def _say(self, text: str) -> bool:
        try:
            self.app.tts.speak(text)  # type: ignore[attr-defined]
            return True
        except VoiceError as error:
            LOGGER.warning("Audio playback failed: %s", error)
            self.write(f"VaderMind > Aviso: {error}")
            self.lights.set_state(ERROR)
            return False

    def _fail(self, spoken: str) -> None:
        """Blink the error, then say it (speaking right away would hide the blink)."""

        self.lights.set_state(ERROR)
        self.sleep(ERROR_SECONDS)
        self._say(spoken)
