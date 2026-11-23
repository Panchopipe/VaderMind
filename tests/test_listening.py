"""Tests for end-of-speech detection, microphone turns and transcript filtering."""

from __future__ import annotations

import io
from pathlib import Path
import queue
import tempfile
import time
import unittest
from unittest import mock
import wave

from vadermind.listening import (
    FrameResampler,
    HighPassFilter,
    ListeningCues,
    ListeningError,
    Utterance,
    VoiceListener,
    capture_rate,
    is_usable_transcript,
    log_utterance_audio,
    matches_language,
    record_utterance,
    resolve_input_device,
    wav_bytes,
)
from vadermind.telemetry import TURN_TIMELINE, TurnTimeline
from vadermind.vad import FRAME_MS, FRAME_SAMPLES, EndOfSpeechDetector, SileroVad
from vadermind.voice_input import TranscriptionError, TranscriptionResult

PROJECT_ROOT = Path(__file__).resolve().parents[1]
VAD_MODEL = PROJECT_ROOT / "models" / "vad" / "silero_vad.onnx"


def _run(detector: EndOfSpeechDetector, probabilities: list[float]) -> tuple[str | None, int]:
    for index, probability in enumerate(probabilities, start=1):
        reason = detector.update(probability)
        if reason:
            return reason, index
    return None, len(probabilities)


def _frames(ms: int) -> int:
    return -(-ms // FRAME_MS)


class EndOfSpeechDetectorTests(unittest.TestCase):
    def test_silence_only_times_out_as_no_speech(self) -> None:
        reason, frames = _run(EndOfSpeechDetector(no_speech_timeout_seconds=1.0), [0.0] * 100)
        self.assertEqual(reason, "sin_voz")
        self.assertEqual(frames, _frames(1000))

    def test_speech_then_silence_ends_after_configured_pause(self) -> None:
        detector = EndOfSpeechDetector(silence_ms=800)
        reason, frames = _run(detector, [0.9] * 30 + [0.05] * 100)
        self.assertEqual(reason, "fin_de_habla")
        self.assertEqual(frames, 30 + _frames(800))

    def test_short_pause_mid_sentence_does_not_cut(self) -> None:
        detector = EndOfSpeechDetector(silence_ms=800)
        pause = [0.05] * _frames(500)
        reason, _ = _run(detector, [0.9] * 20 + pause + [0.9] * 20 + [0.05] * 5)
        self.assertIsNone(reason)

    def test_single_noisy_frame_does_not_start_speech(self) -> None:
        detector = EndOfSpeechDetector()
        _run(detector, [0.0, 0.9, 0.0, 0.0])
        self.assertFalse(detector.speech_started)

    def test_uncertain_frames_neither_end_nor_reset_the_pause(self) -> None:
        detector = EndOfSpeechDetector(silence_ms=800)
        # 0.4 sits between the two thresholds: it keeps the pause counter as is.
        reason, _ = _run(detector, [0.9] * 10 + [0.05] * 20 + [0.4] * 60)
        self.assertIsNone(reason)

    def test_endless_speech_stops_at_maximum(self) -> None:
        reason, frames = _run(EndOfSpeechDetector(max_seconds=2.0), [0.9] * 200)
        self.assertEqual(reason, "maximo")
        self.assertEqual(frames, _frames(2000))


class _FakeVad:
    """Treat frames full of non-zero samples as speech."""

    def reset(self) -> None:
        self.resets = getattr(self, "resets", 0) + 1

    def speech_probability(self, frame: bytes) -> float:
        return 0.9 if any(frame) else 0.0


def _queue(pattern: list[bool]) -> "queue.Queue[bytes]":
    source: "queue.Queue[bytes]" = queue.Queue()
    for voiced in pattern:
        source.put((b"\x10\x00" if voiced else b"\x00\x00") * FRAME_SAMPLES)
    return source


class RecordUtteranceTests(unittest.TestCase):
    def test_start_cue_runs_before_frames_are_consumed(self) -> None:
        order: list[str] = []
        source = _queue([True] * 5 + [False] * 30)

        def frames() -> "queue.Queue[bytes]":
            order.append("frames")
            return source

        utterance = record_utterance(
            _FakeVad(),
            EndOfSpeechDetector(silence_ms=320),
            on_ready=lambda: order.append("cue"),
            frames=frames,
        )
        self.assertEqual(order, ["cue", "frames"])
        # The user went quiet one pause (320 ms) before recording stopped.
        self.assertIsNotNone(utterance.speech_ended_at)

    def test_wav_keeps_pre_roll_and_speech(self) -> None:
        pattern = [False] * 20 + [True] * 10 + [False] * 40
        utterance = record_utterance(
            _FakeVad(),
            EndOfSpeechDetector(silence_ms=320),
            frames=lambda: _queue(pattern),
        )
        self.assertEqual(utterance.reason, "fin_de_habla")
        with wave.open(io.BytesIO(utterance.wav)) as handle:
            self.assertEqual(handle.getframerate(), 16_000)
            frames = handle.getnframes() // FRAME_SAMPLES
        # Speech is confirmed on its 3rd frame: 10 pre-roll frames (8 silence +
        # 2 speech onset) + 8 more speech + 10 trailing silence frames.
        self.assertEqual(frames, 28)


class LanguageFilterTests(unittest.TestCase):
    def test_spanish_and_chilean_pass(self) -> None:
        for text in (
            "Oye weón, cachai lo que pasó ayer?",
            "Pon música de Los Prisioneros.",
            "Reproduce mi playlist de trote en el televisor.",
            "Pon el HDMI dos.",
            "Lord Vader, ¿qué opinas de la Alianza Rebelde?",
        ):
            with self.subTest(text=text):
                self.assertTrue(matches_language(text, "es-mx"))
                self.assertTrue(matches_language(text, None))

    def test_reported_language_other_scripts_and_english_fail(self) -> None:
        self.assertFalse(matches_language("What language, Filipino, can I use?", "en"))
        self.assertFalse(matches_language("モンスターハンターライズ。", None))
        self.assertFalse(matches_language("When you were young, what was your favorite player?", None))
        self.assertTrue(matches_language("Play the music", "en", expected="en"))


class TranscriptFilterTests(unittest.TestCase):
    def test_real_commands_pass(self) -> None:
        self.assertTrue(is_usable_transcript("Pon el HDMI dos."))
        self.assertTrue(is_usable_transcript("Gracias, Vader."))

    def test_empty_symbols_and_known_hallucinations_are_rejected(self) -> None:
        for text in ("", "  ", "...", "¿?", "a", "¡Gracias por ver el video!"):
            self.assertFalse(is_usable_transcript(text), text)


class _FakeTranscriber:
    def __init__(self, text: str = "", error: bool = False) -> None:
        self.text = text
        self.error = error
        self.last_primary_error = None
        self.calls = 0

    def transcribe_detailed(self, audio: bytes) -> TranscriptionResult:
        self.calls += 1
        if self.error:
            raise TranscriptionError("HTTP 503")
        return TranscriptionResult(self.text, "spacexai/grok-stt", 0.5)

    def close(self) -> None:
        pass


class _Cues:
    def __init__(self) -> None:
        self.events: list[str] = []

    def start(self) -> None:
        self.events.append("start")

    def end(self, wait: bool = True) -> None:
        self.events.append("end" if wait else "end_async")


def _listener(reason: str, transcriber: _FakeTranscriber, cues: _Cues) -> VoiceListener:
    def record(_vad, _detector, *, device=None, on_ready=None, **_hooks) -> Utterance:
        if on_ready:
            on_ready()
        cues.events.append("recorded")
        return Utterance(b"RIFF", reason, 1.0, speech_ended_at=time.perf_counter())

    return VoiceListener(_FakeVad(), transcriber, cues, record=record)


class VoiceListenerTests(unittest.TestCase):
    def test_successful_turn_plays_both_cues_and_returns_text(self) -> None:
        cues = _Cues()
        turn = _listener("fin_de_habla", _FakeTranscriber("Pasa a la siguiente canción."), cues).listen()
        self.assertEqual(turn.text, "Pasa a la siguiente canción.")
        self.assertIsNone(turn.notice)
        # Start cue plays once the mic is open; the end cue does not block transcription.
        self.assertEqual(cues.events, ["start", "recorded", "end_async"])

    def test_turn_marks_speech_end_and_transcription_on_the_timeline(self) -> None:
        _listener("fin_de_habla", _FakeTranscriber("Hola"), _Cues()).listen()
        self.assertIn("fin_habla", TURN_TIMELINE.marks)
        self.assertIn("transcripcion_lista", TURN_TIMELINE.marks)

    def test_no_speech_skips_transcription(self) -> None:
        transcriber = _FakeTranscriber("hola")
        turn = _listener("sin_voz", transcriber, _Cues()).listen()
        self.assertIsNone(turn.text)
        self.assertEqual(transcriber.calls, 0)

    def test_transcription_failure_sends_nothing(self) -> None:
        turn = _listener("fin_de_habla", _FakeTranscriber(error=True), _Cues()).listen()
        self.assertIsNone(turn.text)
        self.assertIn("transcribir", turn.notice or "")

    def test_hallucinated_text_is_not_sent_to_vader_but_is_shown(self) -> None:
        turn = _listener("fin_de_habla", _FakeTranscriber("Gracias por ver."), _Cues()).listen()
        self.assertIsNone(turn.text)
        self.assertIn('"Gracias por ver."', turn.notice or "")

    def test_other_languages_are_not_sent_to_vader(self) -> None:
        # Real outputs with the microphone inside the helmet.
        for text in ("今日はいい天気よ。", "Я был в подвале.", "Who are your favorite comedians?"):
            with self.subTest(text=text):
                turn = _listener("fin_de_habla", _FakeTranscriber(text), _Cues()).listen()
                self.assertIsNone(turn.text)

    def test_maximum_length_still_sends_text_with_notice(self) -> None:
        turn = _listener("maximo", _FakeTranscriber("Apaga la tele"), _Cues()).listen()
        self.assertEqual(turn.text, "Apaga la tele")
        self.assertIn("maximo", turn.notice or "")


class ListeningCuesTests(unittest.TestCase):
    def test_default_beeps_are_generated_and_disabled_cues_stay_silent(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            cues = ListeningCues(True, cache_dir=Path(folder))
            self.assertTrue(cues.start_path.is_file())
            self.assertTrue(cues.end_path.is_file())
            ListeningCues(False, cache_dir=Path(folder) / "off").start()
            self.assertFalse((Path(folder) / "off").exists())

    def test_full_breath_only_on_first_turn_then_silence(self) -> None:
        """User's setup: breath when Vader first listens, no sound afterwards."""

        played: list[str] = []
        with tempfile.TemporaryDirectory() as folder, mock.patch(
            "vadermind.listening._play_wav_file", side_effect=lambda p: played.append(Path(p).name)
        ):
            cues = ListeningCues(
                True, "none", "none", cache_dir=Path(folder), first_sound="respiracion.wav"
            )
            for _ in range(3):
                cues.start()
                cues.end(wait=False)
            self.assertFalse(Path(folder).exists() and any(Path(folder).iterdir()))
        self.assertEqual(played, ["respiracion.wav"])

    def test_first_sound_then_regular_start_sound(self) -> None:
        played: list[str] = []
        with tempfile.TemporaryDirectory() as folder, mock.patch(
            "vadermind.listening._play_wav_file", side_effect=lambda p: played.append(Path(p).name)
        ):
            cues = ListeningCues(True, "bip.wav", "none", cache_dir=Path(folder), first_sound="resp.wav")
            cues.start()
            cues.start()
        self.assertEqual(played, ["resp.wav", "bip.wav"])

    def test_custom_sounds_replace_the_beeps(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            cues = ListeningCues(True, "respiracion.wav", "", cache_dir=Path(folder))
            self.assertEqual(cues.start_path, Path("respiracion.wav"))


class TurnTimelineTests(unittest.TestCase):
    def test_summary_splits_latency_from_speech_end_to_first_sound(self) -> None:
        timeline = TurnTimeline()
        for name, at in (
            ("fin_habla", 10.0),
            ("grabacion_cerrada", 10.8),
            ("transcripcion_lista", 11.1),
            ("sintesis_inicio", 12.3),
            ("primer_sonido", 13.5),
            ("primer_sonido", 20.0),  # later sounds do not move the first one
        ):
            timeline.mark(name, at)
        self.assertEqual(
            timeline.summary(),
            "fin_habla->primer_sonido 3.50 s "
            "(silencio 0.80 · stt 0.30 · modelo 1.20 · voz 1.20)",
        )

    def test_silent_success_and_text_turns(self) -> None:
        timeline = TurnTimeline()
        self.assertIsNone(timeline.summary())
        timeline.mark("fin_habla", 1.0)
        timeline.mark("grabacion_cerrada", 1.8)
        timeline.mark("transcripcion_lista", 3.0)
        self.assertEqual(
            timeline.summary(), "fin_habla->sin voz de Vader (silencio 0.80 · stt 1.20)"
        )


class ResolveInputDeviceTests(unittest.TestCase):
    def _devices(self, kind=None):
        devices = [
            {"name": "Speakers (Realtek)", "max_input_channels": 0},
            {"name": "Microphone Array (Intel Smart Sound)", "max_input_channels": 2},
            {"name": "Auriculares (AirPods Hands-Free)", "max_input_channels": 1},
        ]
        return devices[1] if kind == "input" else devices

    def test_finds_microphone_by_partial_name_and_default_when_empty(self) -> None:
        with mock.patch("sounddevice.query_devices", side_effect=self._devices):
            self.assertEqual(resolve_input_device("microphone array")[0], 1)
            self.assertEqual(
                resolve_input_device(""), (None, "Microphone Array (Intel Smart Sound)")
            )

    def test_unknown_name_lists_available_inputs_only(self) -> None:
        with mock.patch("sounddevice.query_devices", side_effect=self._devices):
            with self.assertRaises(ListeningError) as caught:
                resolve_input_device("USB PnP")
        self.assertIn("AirPods", str(caught.exception))
        self.assertNotIn("Speakers", str(caught.exception))


def _tone(rate: int, seconds: float, frequency: float, amplitude: float = 8000.0):
    import numpy as np

    t = np.arange(int(rate * seconds)) / rate
    return (amplitude * np.sin(2 * np.pi * frequency * t)).astype(np.int16)


def _resample(rate: int, signal, block: int) -> bytes:
    resampler = FrameResampler(rate)
    frames: list[bytes] = []
    for start in range(0, len(signal), block):
        frames.extend(resampler.process(signal[start : start + block].tobytes()))
    for frame in frames:
        assert len(frame) == FRAME_SAMPLES * 2
    return b"".join(frames)


class FrameResamplerTests(unittest.TestCase):
    """USB microphones that only record at 44.1 or 48 kHz still feed 16 kHz frames."""

    def _peak_hz_and_rms(self, pcm: bytes) -> tuple[float, float]:
        import numpy as np

        samples = np.frombuffer(pcm, dtype=np.int16).astype(np.float64)[1600:]  # skip filter warm-up
        spectrum = np.abs(np.fft.rfft(samples))
        peak = float(np.fft.rfftfreq(len(samples), 1 / 16_000)[spectrum.argmax()])
        return peak, float(np.sqrt(np.mean(samples**2)))

    def test_48k_and_44k_keep_pitch_duration_and_level(self) -> None:
        for rate, block in ((48_000, 1536), (44_100, 1411)):
            with self.subTest(rate=rate):
                pcm = _resample(rate, _tone(rate, 1.0, 1000), block)
                self.assertAlmostEqual(len(pcm) / 2 / 16_000, 1.0, delta=0.04)
                peak, rms = self._peak_hz_and_rms(pcm)
                self.assertAlmostEqual(peak, 1000, delta=5)
                self.assertAlmostEqual(rms, 8000 / 2**0.5, delta=300)

    def test_tones_above_8k_do_not_fold_into_speech_band(self) -> None:
        # Without the low-pass filter a 12 kHz tone at 48 kHz would alias to 4 kHz.
        _, rms = self._peak_hz_and_rms(_resample(48_000, _tone(48_000, 1.0, 12_000), 1536))
        self.assertLess(rms, 200)

    def test_block_size_does_not_change_the_output(self) -> None:
        for rate, block in ((48_000, 1536), (44_100, 1411)):
            with self.subTest(rate=rate):
                signal = _tone(rate, 0.5, 440)
                self.assertEqual(_resample(rate, signal, block), _resample(rate, signal, 1000))


class HighPassFilterTests(unittest.TestCase):
    def _rms(self, frequency: float) -> float:
        import numpy as np

        signal = _tone(16_000, 1.0, frequency)
        high_pass = HighPassFilter()
        out = b"".join(
            high_pass.process(signal[start : start + FRAME_SAMPLES].tobytes())
            for start in range(0, len(signal), FRAME_SAMPLES)
        )
        samples = np.frombuffer(out, dtype=np.int16).astype(np.float64)[4000:]
        return float(np.sqrt(np.mean(samples**2)))

    def test_removes_mains_hum_and_keeps_the_voice_band(self) -> None:
        full = 8000 / 2**0.5
        self.assertLess(self._rms(50), full * 0.3)  # 50 Hz hum: about -11 dB
        self.assertGreater(self._rms(300), full * 0.95)  # voice fundamentals pass
        self.assertGreater(self._rms(1000), full * 0.99)


class CaptureRateTests(unittest.TestCase):
    def _sd(self, accepted: set[int], default: float = 44100.0) -> mock.Mock:
        def check(*, samplerate: int, **_: object) -> None:
            if samplerate not in accepted:
                raise Exception("Invalid sample rate")

        sd = mock.Mock()
        sd.check_input_settings.side_effect = check
        sd.query_devices.return_value = {"default_samplerate": default}
        return sd

    def test_prefers_16k_then_48k_then_the_microphone_default(self) -> None:
        self.assertEqual(capture_rate(self._sd({16_000, 48_000}), 3), 16_000)
        # The Pi's USB microphone: 44.1 kHz default, but 48 kHz divides evenly.
        self.assertEqual(capture_rate(self._sd({44_100, 48_000}), 3), 48_000)
        sd = self._sd({44_100})
        self.assertEqual(capture_rate(sd, 3), 44_100)
        sd.query_devices.assert_called_with(3, "input")


class UtteranceAudioLogTests(unittest.TestCase):
    def test_debug_logs_level_and_keeps_the_wav(self) -> None:
        import numpy as np

        loud = np.full(1600, 32767, dtype=np.int16)  # 0.1 s, fully clipped
        wav = wav_bytes(loud.tobytes())
        with tempfile.TemporaryDirectory() as folder:
            target = Path(folder) / "last.wav"
            with mock.patch("vadermind.listening.LAST_UTTERANCE_PATH", target):
                with self.assertLogs("vadermind.listening", level="DEBUG") as logs:
                    log_utterance_audio(wav)
                self.assertEqual(target.read_bytes(), wav)
        self.assertIn("pico 100 %", logs.output[0])
        self.assertIn("saturadas 1600", logs.output[0])


@unittest.skipUnless(VAD_MODEL.is_file(), "modelo Silero no presente")
class SileroVadTests(unittest.TestCase):
    def test_silence_scores_low(self) -> None:
        vad = SileroVad(VAD_MODEL)
        probabilities = [vad.speech_probability(b"\x00\x00" * FRAME_SAMPLES) for _ in range(10)]
        self.assertLess(max(probabilities), 0.2)

    def test_recorded_speech_scores_high(self) -> None:
        recording = PROJECT_ROOT / "evaluations" / "stt_recordings" / "frase_05.wav"
        if not recording.is_file():
            self.skipTest("grabacion de benchmark no presente")
        with wave.open(str(recording)) as handle:
            pcm = handle.readframes(handle.getnframes())
        vad = SileroVad(VAD_MODEL)
        size = FRAME_SAMPLES * 2
        probabilities = [
            vad.speech_probability(pcm[start : start + size])
            for start in range(0, len(pcm) - size + 1, size)
        ]
        self.assertGreater(max(probabilities), 0.8)


if __name__ == "__main__":
    unittest.main()
