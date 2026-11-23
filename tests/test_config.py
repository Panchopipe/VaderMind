"""Tests for configuration loading and validation."""

from __future__ import annotations

from pathlib import Path
import os
import unittest

from vadermind.config import ConfigurationError, load_settings, validate_settings


class ConfigTests(unittest.TestCase):
    """Cover the small configuration layer."""

    def test_load_settings_reads_dotenv_file(self) -> None:
        """The loader should pull values from a local .env file."""

        dotenv_path = Path(__file__).resolve().parent / ".test.env"
        dotenv_path.write_text(
            "ANTHROPIC_API_KEY=sk-ant-test\n"
            "ANTHROPIC_MODEL=claude-haiku-4-5\n"
            "VADER_LANGUAGE=es\n"
            "VADER_PERSONA_MODE=chileno\n",
            encoding="utf-8",
        )

        previous = dict(os.environ)
        try:
            os.environ.pop("ANTHROPIC_API_KEY", None)
            settings = load_settings(dotenv_path=dotenv_path)
        finally:
            dotenv_path.unlink(missing_ok=True)
            os.environ.clear()
            os.environ.update(previous)

        self.assertEqual(settings.anthropic_api_key, "sk-ant-test")
        self.assertEqual(settings.anthropic_model, "claude-haiku-4-5")
        self.assertEqual(settings.llm_provider, "anthropic")
        self.assertEqual(settings.vader_fast_model, "anthropic/claude-haiku-4.5")
        self.assertEqual(settings.vader_quality_model, "anthropic/claude-haiku-4.5")
        self.assertEqual(settings.vader_quality_max_output_tokens, 1024)
        self.assertEqual(settings.openrouter_max_retries, 1)
        self.assertEqual(settings.ai_gateway_max_retries, 1)
        self.assertTrue(settings.ai_gateway_require_zdr)
        self.assertEqual(settings.openai_tts_model, "gpt-4o-mini-tts")
        self.assertEqual(settings.openai_tts_voice, "onyx")
        self.assertEqual(settings.elevenlabs_voice_id, "")
        self.assertEqual(settings.elevenlabs_model_id, "eleven_flash_v2_5")
        self.assertEqual(settings.elevenlabs_output_format, "pcm_24000")
        self.assertEqual(settings.tts_provider, "openai")
        self.assertFalse(settings.tts_streaming)
        self.assertTrue(settings.save_last_audio)
        self.assertEqual(settings.piper_executable, "piper")
        self.assertIn("es_CL-vader-medium.onnx", settings.piper_model_path)
        self.assertEqual(settings.piper_length_scale, 1.0)
        self.assertIsNone(settings.spotify_client_id)
        self.assertEqual(settings.spotify_redirect_uri, "http://127.0.0.1:8765/callback")
        self.assertEqual(settings.spotify_token_cache_path, ".spotify_token.json")
        self.assertEqual(
            settings.spotify_playlist_aliases_path, ".spotify_playlist_aliases.json"
        )
        self.assertEqual(settings.spotify_market, "CL")
        self.assertIsNone(settings.spotify_device_name)
        self.assertIsNone(settings.lg_tv_host)
        self.assertIsNone(settings.lg_tv_mac)
        self.assertEqual(settings.lg_tv_broadcast_address, "255.255.255.255")
        self.assertEqual(settings.lg_tv_client_key_path, ".lg_tv_client_key")
        self.assertEqual(settings.lg_tv_schedules_path, ".lg_tv_schedules.json")
        self.assertEqual(settings.lg_tv_timeout_seconds, 5.0)
        self.assertEqual(settings.lg_tv_wake_timeout_seconds, 40.0)
        self.assertIsNone(settings.spotify_tv_device_name)
        self.assertEqual(settings.language, "es")
        self.assertEqual(settings.persona_mode, "chileno")

    def test_validate_settings_requires_api_key(self) -> None:
        """A missing API key should fail with a clear Spanish error."""

        previous = dict(os.environ)
        try:
            os.environ.pop("ANTHROPIC_API_KEY", None)
            settings = load_settings(dotenv_path=Path("missing.env"))
        finally:
            os.environ.clear()
            os.environ.update(previous)

        with self.assertRaises(ConfigurationError) as context:
            validate_settings(settings)

        self.assertIn("ANTHROPIC_API_KEY", str(context.exception))

    def test_validate_settings_requires_openrouter_key_when_selected(self) -> None:
        previous = dict(os.environ)
        try:
            os.environ["LLM_PROVIDER"] = "openrouter"
            os.environ.pop("OPENROUTER_API_KEY", None)
            settings = load_settings(dotenv_path=Path("missing.env"))
        finally:
            os.environ.clear()
            os.environ.update(previous)

        with self.assertRaises(ConfigurationError) as context:
            validate_settings(settings)

        self.assertIn("OPENROUTER_API_KEY", str(context.exception))

    def test_validate_settings_requires_vercel_key_when_selected(self) -> None:
        previous = dict(os.environ)
        try:
            os.environ["LLM_PROVIDER"] = "vercel"
            os.environ.pop("AI_GATEWAY_API_KEY", None)
            settings = load_settings(dotenv_path=Path("missing.env"))
        finally:
            os.environ.clear()
            os.environ.update(previous)

        with self.assertRaises(ConfigurationError) as context:
            validate_settings(settings)

        self.assertIn("AI_GATEWAY_API_KEY", str(context.exception))

    def test_hybrid_provider_requires_both_cloud_keys(self) -> None:
        previous = dict(os.environ)
        try:
            os.environ["LLM_PROVIDER"] = "hybrid"
            os.environ["ANTHROPIC_API_KEY"] = "sk-ant-test"
            os.environ.pop("AI_GATEWAY_API_KEY", None)
            settings = load_settings(dotenv_path=Path("missing.env"))
        finally:
            os.environ.clear()
            os.environ.update(previous)

        with self.assertRaises(ConfigurationError) as context:
            validate_settings(settings)

        self.assertIn("AI_GATEWAY_API_KEY", str(context.exception))

    def test_ai_gateway_zdr_can_be_disabled_explicitly(self) -> None:
        previous = dict(os.environ)
        try:
            os.environ["AI_GATEWAY_REQUIRE_ZDR"] = "false"
            settings = load_settings(dotenv_path=Path("missing.env"))
        finally:
            os.environ.clear()
            os.environ.update(previous)

        self.assertFalse(settings.ai_gateway_require_zdr)

    def test_validate_settings_rejects_unknown_persona_mode(self) -> None:
        """An unsupported persona mode should fail with a clear Spanish error."""

        previous = dict(os.environ)
        try:
            os.environ["ANTHROPIC_API_KEY"] = "sk-ant-test"
            os.environ["VADER_PERSONA_MODE"] = "fiestero"
            settings = load_settings(dotenv_path=Path("missing.env"))
        finally:
            os.environ.clear()
            os.environ.update(previous)

        with self.assertRaises(ConfigurationError) as context:
            validate_settings(settings)

        self.assertIn("VADER_PERSONA_MODE", str(context.exception))

    def test_validate_settings_requires_openai_key_for_voice(self) -> None:
        """Voice mode should require an OpenAI key without affecting text mode."""

        previous = dict(os.environ)
        try:
            os.environ["ANTHROPIC_API_KEY"] = "sk-ant-test"
            os.environ.pop("OPENAI_API_KEY", None)
            settings = load_settings(dotenv_path=Path("missing.env"))
        finally:
            os.environ.clear()
            os.environ.update(previous)

        validate_settings(settings, require_voice=False)

        with self.assertRaises(ConfigurationError) as context:
            validate_settings(settings, require_voice=True)

        self.assertIn("OPENAI_API_KEY", str(context.exception))

    def test_validate_settings_requires_elevenlabs_key_for_elevenlabs_voice(self) -> None:
        """ElevenLabs mode should require its own API key."""

        previous = dict(os.environ)
        try:
            os.environ["ANTHROPIC_API_KEY"] = "sk-ant-test"
            os.environ["VADER_TTS_PROVIDER"] = "elevenlabs"
            os.environ.pop("ELEVENLABS_API_KEY", None)
            settings = load_settings(dotenv_path=Path("missing.env"))
        finally:
            os.environ.clear()
            os.environ.update(previous)

        with self.assertRaises(ConfigurationError) as context:
            validate_settings(settings, require_voice=True)

        self.assertIn("ELEVENLABS_API_KEY", str(context.exception))

    def test_validate_settings_requires_elevenlabs_voice_id(self) -> None:
        """ElevenLabs has no default voice: the user picks one from their account."""

        previous = dict(os.environ)
        try:
            os.environ["ANTHROPIC_API_KEY"] = "sk-ant-test"
            os.environ["VADER_TTS_PROVIDER"] = "elevenlabs"
            os.environ["ELEVENLABS_API_KEY"] = "sk_test"
            os.environ.pop("ELEVENLABS_VOICE_ID", None)
            settings = load_settings(dotenv_path=Path("missing.env"))
        finally:
            os.environ.clear()
            os.environ.update(previous)

        with self.assertRaises(ConfigurationError) as context:
            validate_settings(settings, require_voice=True)

        self.assertIn("ELEVENLABS_VOICE_ID", str(context.exception))

    def test_validate_settings_rejects_unknown_tts_provider(self) -> None:
        """Unsupported voice providers should fail before runtime."""

        previous = dict(os.environ)
        try:
            os.environ["ANTHROPIC_API_KEY"] = "sk-ant-test"
            os.environ["VADER_TTS_PROVIDER"] = "unknown"
            settings = load_settings(dotenv_path=Path("missing.env"))
        finally:
            os.environ.clear()
            os.environ.update(previous)

        with self.assertRaises(ConfigurationError) as context:
            validate_settings(settings)

        self.assertIn("VADER_TTS_PROVIDER", str(context.exception))

    def test_validate_settings_accepts_local_piper_files(self) -> None:
        """Piper voice mode should work without a cloud TTS API key."""

        test_dir = Path(__file__).resolve().parent / ".piper-test"
        test_dir.mkdir(exist_ok=True)
        executable = test_dir / "piper-test.exe"
        model = test_dir / "voice.onnx"
        config = test_dir / "voice.onnx.json"
        for path in (executable, model, config):
            path.write_bytes(b"test")

        previous = dict(os.environ)
        try:
            os.environ["ANTHROPIC_API_KEY"] = "sk-ant-test"
            os.environ["VADER_TTS_PROVIDER"] = "piper"
            os.environ["PIPER_EXECUTABLE"] = str(executable)
            os.environ["PIPER_MODEL_PATH"] = str(model)
            os.environ["PIPER_CONFIG_PATH"] = str(config)
            os.environ.pop("OPENAI_API_KEY", None)
            os.environ.pop("ELEVENLABS_API_KEY", None)
            settings = load_settings(dotenv_path=Path("missing.env"))
            validate_settings(settings, require_voice=True)
        finally:
            os.environ.clear()
            os.environ.update(previous)
            for path in (executable, model, config):
                path.unlink(missing_ok=True)
            test_dir.rmdir()

    def test_validate_settings_reports_missing_piper_model(self) -> None:
        """A missing local model should produce a focused configuration error."""

        previous = dict(os.environ)
        try:
            os.environ["ANTHROPIC_API_KEY"] = "sk-ant-test"
            os.environ["VADER_TTS_PROVIDER"] = "piper"
            os.environ["PIPER_EXECUTABLE"] = str(Path(__file__))
            os.environ["PIPER_MODEL_PATH"] = "missing-voice.onnx"
            os.environ["PIPER_CONFIG_PATH"] = "missing-voice.onnx.json"
            settings = load_settings(dotenv_path=Path("missing.env"))
        finally:
            os.environ.clear()
            os.environ.update(previous)

        with self.assertRaises(ConfigurationError) as context:
            validate_settings(settings, require_voice=True)

        self.assertIn("PIPER_MODEL_PATH", str(context.exception))

    def test_validate_settings_requires_spotify_credentials_when_requested(self) -> None:
        """Spotify commands should require their own app credentials."""

        previous = dict(os.environ)
        try:
            os.environ.pop("SPOTIFY_CLIENT_ID", None)
            os.environ.pop("SPOTIFY_CLIENT_SECRET", None)
            settings = load_settings(dotenv_path=Path("missing.env"))
        finally:
            os.environ.clear()
            os.environ.update(previous)

        with self.assertRaises(ConfigurationError) as context:
            validate_settings(settings, require_anthropic=False, require_spotify=True)

        self.assertIn("SPOTIFY_CLIENT_ID", str(context.exception))

    def test_validate_settings_allows_spotify_without_anthropic_when_requested(self) -> None:
        """Spotify setup should not require Claude credentials."""

        previous = dict(os.environ)
        try:
            os.environ.pop("ANTHROPIC_API_KEY", None)
            os.environ["SPOTIFY_CLIENT_ID"] = "spotify-client"
            os.environ["SPOTIFY_CLIENT_SECRET"] = "spotify-secret"
            settings = load_settings(dotenv_path=Path("missing.env"))
        finally:
            os.environ.clear()
            os.environ.update(previous)

        validate_settings(settings, require_anthropic=False, require_spotify=True)

    def test_load_and_validate_lg_tv_settings(self) -> None:
        previous = dict(os.environ)
        try:
            os.environ["LG_TV_HOST"] = "192.0.2.162"
            os.environ["LG_TV_MAC"] = "00:00:5e:00:53:01"
            os.environ["LG_TV_BROADCAST_ADDRESS"] = "192.0.2.255"
            os.environ["LG_TV_CLIENT_KEY_PATH"] = ".test-lg-key"
            os.environ["LG_TV_TIMEOUT_SECONDS"] = "7"
            os.environ["LG_TV_WAKE_TIMEOUT_SECONDS"] = "18"
            settings = load_settings(dotenv_path=Path("missing.env"))
        finally:
            os.environ.clear()
            os.environ.update(previous)

        validate_settings(settings, require_anthropic=False)
        self.assertEqual(settings.lg_tv_host, "192.0.2.162")
        self.assertEqual(settings.lg_tv_mac, "00:00:5e:00:53:01")
        self.assertEqual(settings.lg_tv_broadcast_address, "192.0.2.255")
        self.assertEqual(settings.lg_tv_client_key_path, ".test-lg-key")
        self.assertEqual(settings.lg_tv_timeout_seconds, 7.0)
        self.assertEqual(settings.lg_tv_wake_timeout_seconds, 18.0)

    def test_validate_settings_rejects_invalid_lg_tv_mac(self) -> None:
        previous = dict(os.environ)
        try:
            os.environ["LG_TV_HOST"] = "192.0.2.162"
            os.environ["LG_TV_MAC"] = "not-a-mac"
            settings = load_settings(dotenv_path=Path("missing.env"))
        finally:
            os.environ.clear()
            os.environ.update(previous)

        with self.assertRaises(ConfigurationError) as context:
            validate_settings(settings, require_anthropic=False)

        self.assertIn("LG_TV_MAC", str(context.exception))

    def test_validate_settings_rejects_invalid_lg_tv_broadcast(self) -> None:
        previous = dict(os.environ)
        try:
            os.environ["LG_TV_BROADCAST_ADDRESS"] = "not-an-ip"
            settings = load_settings(dotenv_path=Path("missing.env"))
        finally:
            os.environ.clear()
            os.environ.update(previous)

        with self.assertRaises(ConfigurationError) as context:
            validate_settings(settings, require_anthropic=False)

        self.assertIn("LG_TV_BROADCAST_ADDRESS", str(context.exception))


if __name__ == "__main__":
    unittest.main()
