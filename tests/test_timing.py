"""Tests for the speed-stage timing: turn lines, start-up line and the report script."""

from __future__ import annotations

import json
import os
from pathlib import Path
import queue
import tempfile
import threading
import unittest
from unittest import mock

from scripts.latency_report import parse, report
from vadermind.config import load_settings
from vadermind.hands_free import HandsFreeSession
from vadermind.lights import Lights
from vadermind.listening import Utterance, VoiceListener, VoiceTurn
from vadermind.telemetry import TURN_TIMELINE, StartupTimeline, TurnTimeline, timed_operation
from vadermind.voice_output import PiperTextToSpeech
from vadermind.wake_word import VoskWakeWord, wait_for_wake_word


class TurnTimelineTests(unittest.TestCase):
    def _timeline(self) -> TurnTimeline:
        timeline = TurnTimeline()
        for name, at in {
            "fin_palabra": 10.0,
            "palabra_detectada": 10.7,
            "micro_abierto": 11.0,
            "escucha_lista": 11.0,
            "inicio_habla": 11.5,
            "fin_habla": 13.0,
            "grabacion_cerrada": 13.8,
            "transcripcion_lista": 14.6,
            "sintesis_inicio": 15.5,
            "primer_sonido": 15.9,
            "fin_turno": 19.0,
        }.items():
            timeline.mark(name, at)
        return timeline

    def test_report_lists_every_span_that_happened(self) -> None:
        line = self._timeline().report()
        self.assertTrue(line.startswith("Tiempos [charla] "))
        for expected in (
            "despertar=1.00", "vosk=0.70", "abrir_micro=0.30", "sonido_inicio=0.00",
            "espera_usuario=0.50", "calla_suena=2.90", "silencio=0.80", "stt=0.80",
            "pensar=0.90", "voz=0.40", "completa=6.00", "hablar=3.10",
        ):
            self.assertIn(expected, line)
        self.assertNotIn("reabrir", line)  # woken by the phrase, not a follow-up

    def test_tool_steps_and_notes(self) -> None:
        timeline = self._timeline()
        timeline.add_duration("tool_spotify.play", 0.4)
        timeline.add_duration("ttft", 0.6)
        timeline.add_duration("reproduccion", 1.0)
        timeline.add_duration("reproduccion", 1.5)
        timeline.add_duration("audio_vader", 2.2)
        timeline.note("nivel", "fast")
        line = timeline.report()
        self.assertTrue(line.startswith("Tiempos [spotify.play] "))
        self.assertIn("herramienta=0.40", line)
        self.assertIn("ttft=0.60", line)
        self.assertIn("aplay_extra=0.30", line)
        self.assertTrue(line.endswith("| nivel=fast"))
        self.assertTrue(timeline.report("local_modo").startswith("Tiempos [local_modo] "))

    def test_timed_operations_feed_the_turn(self) -> None:
        TURN_TIMELINE.reset()
        with timed_operation("tool_lg_tv.status"):
            pass
        self.assertEqual(TURN_TIMELINE.tool_name(), "lg_tv.status")
        TURN_TIMELINE.reset()
        self.assertEqual(TURN_TIMELINE.durations, {})


class StartupTimelineTests(unittest.TestCase):
    def test_steps_add_up_from_process_start_to_dim_pupils(self) -> None:
        startup = StartupTimeline()
        startup.marks = {"proceso": 100.0, "main": 101.0, "imports": 103.0, "config": 103.1,
                         "motor_voz": 103.5, "escucha": 104.5, "luces": 104.6, "vosk": 106.6,
                         "reposo": 106.7, "piper_listo": 107.2}
        startup._boot_offset = -80.0  # perf_counter 100 = 20 s after the kernel started
        line = startup.report()
        for expected in (
            "kernel_a_reposo=26.70", "kernel_a_proceso=20.00", "proceso_a_reposo=6.70",
            "python=1.00", "imports=2.00", "escucha=1.00", "vosk=2.00", "piper_tras_reposo=0.50",
        ):
            self.assertIn(expected, line)

    def test_network_wait_shows_between_vosk_and_dim_pupils(self) -> None:
        startup = StartupTimeline()
        startup.marks = {"proceso": 100.0, "main": 100.1, "vosk": 107.4, "red": 112.9,
                         "reposo": 113.0}
        startup._boot_offset = None
        line = startup.report()
        for expected in ("proceso_a_reposo=13.00", "vosk=7.30", "red=5.50", "reposo=0.10"):
            self.assertIn(expected, line)

    def test_ready_line_is_logged_once(self) -> None:
        startup = StartupTimeline()
        with self.assertLogs("vadermind.telemetry", level="INFO") as logs:
            startup.log_ready()
            startup.log_ready()
            startup.piper_ready()  # Piper finished after the pupils were dim
        self.assertEqual(len(logs.output), 2)
        self.assertIn("Arranque ", logs.output[0])
        self.assertIn("piper_tras_reposo=", logs.output[1])


class _TimedRecognizer:
    """Vosk-like: a final "oye vader" ending at `end` seconds of fed audio."""

    def __init__(self, final_at_frame: int, end: float) -> None:
        self.frames = 0
        self.final_at_frame = final_at_frame
        self.end = end

    def AcceptWaveform(self, frame: bytes) -> bool:  # noqa: N802 (Vosk API)
        self.frames += 1
        return self.frames == self.final_at_frame

    def Result(self) -> str:  # noqa: N802
        words = [{"word": "oye", "conf": 1.0, "end": self.end - 0.3},
                 {"word": "vader", "conf": 1.0, "end": self.end}]
        return json.dumps({"result": words})

    def Reset(self) -> None:  # noqa: N802
        pass


class WakeTimingTests(unittest.TestCase):
    def test_lag_is_the_audio_heard_after_the_phrase(self) -> None:
        # 25 frames of 32 ms = 0.80 s fed; the phrase ended at 0.30 s.
        detector = VoskWakeWord("", recognizer=_TimedRecognizer(25, end=0.30))
        source: "queue.Queue[bytes]" = queue.Queue()
        for _ in range(30):
            source.put(b"\x00" * 1024)
        timing: dict[str, float] = {}
        self.assertTrue(wait_for_wake_word(detector, frames=lambda: source, timing=timing))
        self.assertAlmostEqual(detector.last_lag_seconds, 0.50, places=3)
        # 5 frames still queued (0.16 s) were captured after the detection point.
        self.assertAlmostEqual(
            timing["palabra_detectada"] - timing["fin_palabra"], 0.66, places=3
        )

    def test_without_word_times_only_the_detection_is_marked(self) -> None:
        class NoTimes(_TimedRecognizer):
            def Result(self) -> str:  # noqa: N802
                return json.dumps({"result": [{"word": "oye", "conf": 1.0},
                                              {"word": "vader", "conf": 1.0}]})

        detector = VoskWakeWord("", recognizer=NoTimes(1, end=0.0))
        source: "queue.Queue[bytes]" = queue.Queue()
        source.put(b"\x00" * 1024)
        timing: dict[str, float] = {}
        self.assertTrue(wait_for_wake_word(detector, frames=lambda: source, timing=timing))
        self.assertEqual(set(timing), {"palabra_detectada"})


class _Cues:
    def start(self) -> None:
        pass

    def end(self, wait: bool = True) -> None:
        pass


class _Transcriber:
    last_primary_error = None

    def transcribe_detailed(self, _wav: bytes):
        return mock.Mock(text="que hora es", language="es", model="fake")

    def close(self) -> None:
        pass


class ListenerMarksTests(unittest.TestCase):
    def test_listen_marks_microphone_cue_and_speech_start(self) -> None:
        started: list[bool] = []

        def record(_vad, _detector, *, on_ready, on_speech_start, **_hooks) -> Utterance:
            on_ready()
            on_speech_start()
            return Utterance(b"RIFF", "fin_de_habla", 1.0, speech_ended_at=1.0, closed_at=1.8)

        listener = VoiceListener(mock.Mock(), _Transcriber(), _Cues(), record=record)
        listener.on_speech_start = lambda: started.append(True)
        turn = listener.listen()
        listener.close()
        self.assertEqual(turn.text, "que hora es")
        self.assertEqual(started, [True])  # the original callback still runs
        for name in ("micro_abierto", "escucha_lista", "inicio_habla", "fin_habla",
                     "grabacion_cerrada", "transcripcion_lista"):
            self.assertIn(name, TURN_TIMELINE.marks)
        self.assertEqual(TURN_TIMELINE.notes["especulacion"], "no")


class _TimedListener:
    """Fake listener that, like the real one, starts a fresh timeline."""

    def __init__(self, turns: list[VoiceTurn]) -> None:
        self.turns = list(turns)

    def listen(self, no_speech_seconds: float | None = None) -> VoiceTurn:
        TURN_TIMELINE.reset()
        turn = self.turns.pop(0)
        TURN_TIMELINE.mark("micro_abierto")
        TURN_TIMELINE.mark("escucha_lista")
        if turn.text is not None:
            TURN_TIMELINE.mark("fin_habla")
        return turn


class _App:
    def __init__(self) -> None:
        self.tts = mock.Mock()
        self.spoken_answer = None
        self.skip_post_turn_speech = False

    def process_turn(self, text: str) -> str:
        with timed_operation("tool_information.time"):
            pass
        return "Son las diez."


class HandsFreeTimingTests(unittest.TestCase):
    def test_one_line_per_spoken_turn_with_its_kind(self) -> None:
        listener = _TimedListener([
            VoiceTurn("que hora es"), VoiceTurn("modo canon"), VoiceTurn("silencio"),
        ])
        wakes = [True, False]
        session = HandsFreeSession(
            _App(), listener, Lights(), lambda: wakes.pop(0),
            set_mode=lambda _mode: None, write=lambda _text: None,
            wake_marks=lambda: {"fin_palabra": 0.0, "palabra_detectada": 0.5},
        )
        with self.assertLogs("vadermind.hands_free", level="INFO") as logs:
            session.run()
        lines = [entry.split(":", 2)[2] for entry in logs.output]
        self.assertEqual(len(lines), 3)
        self.assertTrue(lines[0].startswith("Tiempos [information.time] "))
        self.assertIn("despertar=", lines[0])
        self.assertIn("via=palabra", lines[0])
        self.assertTrue(lines[1].startswith("Tiempos [local_modo] "))
        self.assertIn("reabrir=", lines[1])  # follow-up: no wake word
        self.assertIn("via=seguimiento", lines[1])
        self.assertTrue(lines[2].startswith("Tiempos [local_silencio] "))
        self.assertIn("turno=3", lines[2])

    def test_nobody_speaking_logs_nothing(self) -> None:
        listener = _TimedListener([VoiceTurn(None, "nada", reason="sin_voz")])
        wakes = [True, False]
        session = HandsFreeSession(
            _App(), listener, Lights(), lambda: wakes.pop(0), write=lambda _text: None
        )
        with mock.patch("vadermind.hands_free.LOGGER") as logger:
            session.run()
        logger.info.assert_not_called()


class PiperPieceTimingTests(unittest.TestCase):
    def test_pieces_and_wait_between_them_are_recorded(self) -> None:
        previous = dict(os.environ)
        try:
            os.environ["VADER_SAVE_LAST_AUDIO"] = "false"
            tts = PiperTextToSpeech(load_settings(dotenv_path=Path("missing.env")))
        finally:
            os.environ.clear()
            os.environ.update(previous)
        played = threading.Event()

        def synthesize(_tts, sentence: str, target: Path) -> Path:
            if sentence.startswith("Segunda"):
                played.wait(5)  # the second piece is late: Vader waits in silence
            target.write_bytes(b"RIFF" + b"\0" * 64)
            return target

        TURN_TIMELINE.reset()
        with tempfile.TemporaryDirectory() as folder, mock.patch(
            "pathlib.Path.cwd", return_value=Path(folder)
        ), mock.patch.object(PiperTextToSpeech, "_synthesize", synthesize), mock.patch(
            "vadermind.voice_output._play_wav_file", lambda _path: played.set()
        ):
            tts.speak("Primera oracion. Segunda oracion.")
        self.assertEqual(TURN_TIMELINE.notes["trozos"], "2")
        self.assertIn("espera_trozos", TURN_TIMELINE.durations)
        self.assertIn("envolvente_luces", TURN_TIMELINE.durations)


SAMPLE_LOG = """\
Arranque kernel_a_reposo=31.20 kernel_a_proceso=22.00 proceso_a_reposo=9.20 python=0.90 imports=3.10 | gobernador=ondemand temp=45.0
Arranque piper_tras_reposo=1.40
Tu (voz) > Hola
Tiempos [charla] despertar=3.40 calla_suena=3.10 completa=5.00 | turno=1 via=palabra
Tu (voz) > ¿Qué hora es?
Tiempos [information.time] despertar=1.10 vosk=0.70 calla_suena=2.50 silencio=0.80 stt=0.90 pensar=0.60 voz=0.20 completa=4.00 | turno=2 via=palabra
Tu (voz) > Pon Metallica
Tiempos [spotify.play] reabrir=0.40 calla_suena=2.00 silencio=0.80 stt=0.70 pensar=0.30 voz=0.20 herramienta=0.90 completa=4.40 | turno=3 via=seguimiento
"""


class LatencyReportTests(unittest.TestCase):
    def test_parse_turns_startups_and_texts(self) -> None:
        turns, startups = parse(SAMPLE_LOG.splitlines())
        self.assertEqual([turn.kind for turn in turns], ["charla", "information.time", "spotify.play"])
        self.assertEqual(turns[1].text, "¿Qué hora es?")
        self.assertEqual(turns[2].values["herramienta"], 0.9)
        self.assertTrue(turns[0].first)
        self.assertEqual(len(startups), 1)
        self.assertEqual(startups[0]["piper_tras_reposo"], 1.4)

    def test_report_leaves_the_first_turn_out_of_the_medians(self) -> None:
        text = report(*parse(SAMPLE_LOG.splitlines()))
        self.assertIn("== hora (1 turnos: information.time) ==", text)
        self.assertIn("== spotify (1 turnos: spotify.play) ==", text)
        self.assertNotIn("== charla", text)  # its only turn was the first one
        self.assertIn("primer turno, fuera de las medianas", text)
        self.assertIn("kernel_a_reposo", text)
        ranking = text.split("== Tramos ordenados")[1]
        self.assertLess(ranking.index("stt"), ranking.index("pensar"))  # 0.80 before 0.45

    def test_streaming_turns_are_counted(self) -> None:
        log = SAMPLE_LOG.replace("turno=2 via=palabra", "turno=2 via=palabra transcriptor=streaming")
        log = log.replace("herramienta=0.90", "herramienta=0.90 stt_conexion=0.55 stt_streaming=0.31")
        log = log.replace("turno=3 via=seguimiento", "turno=3 via=seguimiento transcriptor=rest")
        text = report(*parse(log.splitlines()))
        self.assertIn("== Transcriptor (turnos con voz): rest 1, streaming 1 ==", text)
        self.assertIn("streaming: abrir la conexion", text)


if __name__ == "__main__":
    unittest.main()
