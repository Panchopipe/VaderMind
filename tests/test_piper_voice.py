"""Tests for the local Piper text-to-speech adapter."""

from __future__ import annotations

from pathlib import Path
from subprocess import CompletedProcess
from unittest.mock import patch
import os
import unittest

from vadermind.config import load_settings
from vadermind.voice_output import PiperTextToSpeech, VoiceError


class PiperVoiceTests(unittest.TestCase):
    """Verify command construction and failures without playing real audio."""

    def _settings(self):
        previous = dict(os.environ)
        try:
            os.environ["PIPER_EXECUTABLE"] = "fake-piper"
            os.environ["PIPER_MODEL_PATH"] = "voice.onnx"
            os.environ["PIPER_CONFIG_PATH"] = "voice.onnx.json"
            os.environ["VADER_SAVE_LAST_AUDIO"] = "false"
            return load_settings(dotenv_path=Path("missing.env"))
        finally:
            os.environ.clear()
            os.environ.update(previous)

    @patch("vadermind.voice_output._play_wav_file")
    @patch("vadermind.voice_output.subprocess.run")
    def test_speak_generates_and_plays_wav(self, run_mock, play_mock) -> None:
        """The adapter should pass text to Piper and play a valid output file."""

        def create_output(command, **kwargs):
            output_path = Path(command[command.index("--output_file") + 1])
            output_path.write_bytes(b"RIFF" + (b"\x00" * 64))
            return CompletedProcess(command, 0, "", "")

        run_mock.side_effect = create_output
        PiperTextToSpeech(self._settings()).speak("Hola, Francisco.")

        command = run_mock.call_args.args[0]
        self.assertIn("--model", command)
        self.assertEqual(run_mock.call_args.kwargs["input"], "Hola, Francisco.\n")
        play_mock.assert_called_once()

    @patch("vadermind.voice_output._play_wav_file")
    @patch("vadermind.voice_output.subprocess.run")
    def test_speak_launches_relative_executable_by_absolute_path(
        self, run_mock, play_mock
    ) -> None:
        """A relative PIPER_EXECUTABLE should not depend on cwd lookup at launch."""

        def create_output(command, **kwargs):
            output_path = Path(command[command.index("--output_file") + 1])
            output_path.write_bytes(b"RIFF" + (b"\x00" * 64))
            return CompletedProcess(command, 0, "", "")

        run_mock.side_effect = create_output
        relative_executable = "tests/test_piper_voice.py"
        settings = self._settings()
        settings.piper_executable = relative_executable

        PiperTextToSpeech(settings).speak("Prueba")

        launched = run_mock.call_args.args[0][0]
        self.assertTrue(Path(launched).is_absolute())
        self.assertEqual(Path(launched), Path(relative_executable).absolute())

    @patch("vadermind.voice_output.subprocess.run")
    def test_speak_keeps_bare_executable_name_for_path_lookup(self, run_mock) -> None:
        """A bare command name should still be resolved through PATH."""

        run_mock.return_value = CompletedProcess([], 2, "", "bad model")

        with self.assertRaises(VoiceError):
            PiperTextToSpeech(self._settings()).speak("Prueba")

        self.assertEqual(run_mock.call_args.args[0][0], "fake-piper")

    @patch("vadermind.voice_output.subprocess.run")
    def test_speak_reports_piper_process_error(self, run_mock) -> None:
        """A non-zero Piper exit should become a readable VoiceError."""

        run_mock.return_value = CompletedProcess([], 2, "", "bad model")

        with self.assertRaises(VoiceError) as context:
            PiperTextToSpeech(self._settings()).speak("Prueba")

        self.assertIn("codigo 2", str(context.exception))


if __name__ == "__main__":
    unittest.main()
