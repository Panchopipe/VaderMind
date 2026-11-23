"""Configuration loading and validation helpers."""

from __future__ import annotations

from dataclasses import dataclass
import ipaddress
from pathlib import Path
import os
import re
import shutil


class ConfigurationError(RuntimeError):
    """Raised when the application configuration is missing or invalid."""


@dataclass(slots=True)
class Settings:
    """Holds runtime settings loaded from environment variables."""

    anthropic_api_key: str | None
    anthropic_model: str
    anthropic_max_output_tokens: int
    llm_provider: str
    openrouter_api_key: str | None
    ai_gateway_api_key: str | None
    vader_fast_model: str
    vader_quality_model: str
    vader_quality_max_output_tokens: int
    openrouter_timeout_seconds: float
    openrouter_max_retries: int
    ai_gateway_timeout_seconds: float
    ai_gateway_max_retries: int
    ai_gateway_require_zdr: bool
    openai_api_key: str | None
    openai_tts_model: str
    openai_tts_voice: str
    elevenlabs_api_key: str | None
    elevenlabs_voice_id: str
    elevenlabs_model_id: str
    elevenlabs_output_format: str
    elevenlabs_stability: float
    elevenlabs_similarity_boost: float
    elevenlabs_style: float
    elevenlabs_use_speaker_boost: bool
    tts_provider: str
    tts_streaming: bool
    save_last_audio: bool
    piper_executable: str
    piper_model_path: str
    piper_config_path: str
    piper_length_scale: float
    piper_noise_scale: float
    piper_noise_w: float
    piper_sentence_silence: float
    piper_timeout_seconds: float
    spotify_client_id: str | None
    spotify_client_secret: str | None
    spotify_redirect_uri: str
    spotify_token_cache_path: str
    spotify_playlist_aliases_path: str
    spotify_market: str
    spotify_device_name: str | None
    language: str
    persona_mode: str
    lg_tv_host: str | None = None
    lg_tv_mac: str | None = None
    lg_tv_broadcast_address: str = "255.255.255.255"
    lg_tv_client_key_path: str = ".lg_tv_client_key"
    lg_tv_schedules_path: str = ".lg_tv_schedules.json"
    lg_tv_timeout_seconds: float = 5.0
    lg_tv_wake_timeout_seconds: float = 40.0
    spotify_tv_device_name: str | None = None
    stt_model: str = "openai/gpt-4o-transcribe"
    stt_fallback_model: str = "spacexai/grok-stt"
    stt_timeout_seconds: float = 8.0
    stt_prompt: str = "default"
    mic_device: str = ""
    vad_model_path: str = "models/vad/silero_vad.onnx"
    listen_silence_ms: int = 800
    listen_max_seconds: float = 15.0
    listen_cues: bool = True
    listen_start_sound: str = ""
    listen_end_sound: str = ""
    listen_first_sound: str = ""
    stt_speculative_ms: int = 192
    stt_streaming: bool = True
    stt_streaming_model: str = "spacexai/grok-stt"
    piper_persistent: bool = True
    wake_model_path: str = "models/vosk/vosk-model-small-es-0.42"
    wake_phrase: str = "oye vader"
    wake_min_confidence: float = 0.5
    wake_variants: tuple[str, ...] = ("oye bei der",)
    follow_up_seconds: float = 10.0
    allow_shutdown: bool = False
    lights: str = "none"
    lights_spi_device: str = "/dev/spidev0.0"
    lights_brightness: float = 0.10
    lights_eyes: int = 2
    lights_center_index: int = 0
    debug: bool = False
    profile_name: str | None = None


# What changes from one home to another. With a profile active these come only
# from profiles/<name>/perfil.env (or the fixed files inside that folder), never
# from .env, so a value of Pancho's home cannot leak into Pablo's.
PROFILE_KEYS = (
    "SPOTIFY_DEVICE_NAME",
    "SPOTIFY_TV_DEVICE_NAME",
    "LG_TV_HOST",
    "LG_TV_MAC",
    "LG_TV_BROADCAST_ADDRESS",
    "VADER_PERSONA_MODE",
)
# State files of each home: always inside the profile folder, with these names.
PROFILE_STATE_FILES = {
    "SPOTIFY_TOKEN_CACHE_PATH": "spotify_token.json",
    "SPOTIFY_PLAYLIST_ALIASES_PATH": "spotify_playlist_aliases.json",
    "LG_TV_CLIENT_KEY_PATH": "lg_tv_client_key",
    "LG_TV_SCHEDULES_PATH": "lg_tv_schedules.json",
}


def read_env_file(path: Path) -> dict[str, str]:
    """Read KEY=value lines from a small .env-style file (missing file: empty)."""

    if not path.exists():
        return {}
    values: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def load_dotenv_file(dotenv_path: Path, skip: frozenset[str] = frozenset()) -> None:
    """Load a small .env file without requiring external dependencies."""

    for key, value in read_env_file(dotenv_path).items():
        if key not in skip:
            os.environ.setdefault(key, value)


def apply_profile_environment(profile_dir: Path) -> None:
    """Put one home's values in the environment, replacing any earlier ones."""

    values = read_env_file(profile_dir / "perfil.env")
    for key in PROFILE_KEYS:
        if values.get(key):
            os.environ[key] = values[key]
        else:
            os.environ.pop(key, None)
    for key, file_name in PROFILE_STATE_FILES.items():
        os.environ[key] = str(profile_dir / file_name)


def _parse_bool(value: str | None, default: bool = False) -> bool:
    """Parse a boolean-like environment variable."""

    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _parse_float(value: str | None, default: float, variable_name: str) -> float:
    """Parse a floating-point environment variable."""

    if value is None:
        return default
    try:
        return float(value)
    except ValueError as error:
        raise ConfigurationError(f"La variable {variable_name} debe ser un numero.") from error


def _parse_int(value: str | None, default: int, variable_name: str) -> int:
    if value is None:
        return default
    try:
        return int(value)
    except ValueError as error:
        raise ConfigurationError(f"La variable {variable_name} debe ser un numero entero.") from error


def load_settings(
    dotenv_path: Path | None = None, *, profile_dir: Path | None = None
) -> Settings:
    """Load environment settings from the current process and optional .env file.

    With profile_dir, the home-specific values come from that profile folder
    and the same keys in .env are ignored.
    """

    if dotenv_path is None:
        dotenv_path = Path.cwd() / ".env"

    skip: frozenset[str] = frozenset()
    if profile_dir is not None:
        apply_profile_environment(profile_dir)
        skip = frozenset(PROFILE_KEYS) | frozenset(PROFILE_STATE_FILES)
    load_dotenv_file(dotenv_path, skip=skip)

    max_output_tokens = os.getenv("ANTHROPIC_MAX_OUTPUT_TOKENS", "250")
    try:
        parsed_tokens = int(max_output_tokens)
    except ValueError as error:
        raise ConfigurationError(
            "La variable ANTHROPIC_MAX_OUTPUT_TOKENS debe ser un numero entero."
        ) from error

    # Reasoning models spend hidden tokens from this same budget before answering.
    quality_output_tokens = os.getenv("VADER_QUALITY_MAX_OUTPUT_TOKENS", "1024")
    try:
        parsed_quality_tokens = int(quality_output_tokens)
    except ValueError as error:
        raise ConfigurationError(
            "La variable VADER_QUALITY_MAX_OUTPUT_TOKENS debe ser un numero entero."
        ) from error

    return Settings(
        anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"),
        anthropic_model=os.getenv("ANTHROPIC_MODEL", "claude-haiku-4-5"),
        anthropic_max_output_tokens=parsed_tokens,
        llm_provider=os.getenv("LLM_PROVIDER", "anthropic").strip().lower(),
        openrouter_api_key=os.getenv("OPENROUTER_API_KEY"),
        ai_gateway_api_key=os.getenv("AI_GATEWAY_API_KEY"),
        vader_fast_model=os.getenv(
            "VADER_FAST_MODEL", "anthropic/claude-haiku-4.5"
        ).strip(),
        vader_quality_model=os.getenv(
            "VADER_QUALITY_MODEL", "anthropic/claude-haiku-4.5"
        ).strip(),
        vader_quality_max_output_tokens=parsed_quality_tokens,
        openrouter_timeout_seconds=_parse_float(
            os.getenv("OPENROUTER_TIMEOUT_SECONDS"),
            default=30.0,
            variable_name="OPENROUTER_TIMEOUT_SECONDS",
        ),
        openrouter_max_retries=_parse_int(
            os.getenv("OPENROUTER_MAX_RETRIES"),
            default=1,
            variable_name="OPENROUTER_MAX_RETRIES",
        ),
        ai_gateway_timeout_seconds=_parse_float(
            os.getenv("AI_GATEWAY_TIMEOUT_SECONDS"),
            default=30.0,
            variable_name="AI_GATEWAY_TIMEOUT_SECONDS",
        ),
        ai_gateway_max_retries=_parse_int(
            os.getenv("AI_GATEWAY_MAX_RETRIES"),
            default=1,
            variable_name="AI_GATEWAY_MAX_RETRIES",
        ),
        ai_gateway_require_zdr=_parse_bool(
            os.getenv("AI_GATEWAY_REQUIRE_ZDR"), default=True
        ),
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        openai_tts_model=os.getenv("OPENAI_TTS_MODEL", "gpt-4o-mini-tts"),
        openai_tts_voice=os.getenv("OPENAI_TTS_VOICE", "onyx"),
        elevenlabs_api_key=os.getenv("ELEVENLABS_API_KEY"),
        elevenlabs_voice_id=(os.getenv("ELEVENLABS_VOICE_ID") or "").strip(),
        elevenlabs_model_id=os.getenv("ELEVENLABS_MODEL_ID", "eleven_flash_v2_5"),
        elevenlabs_output_format=os.getenv("ELEVENLABS_OUTPUT_FORMAT", "pcm_24000"),
        elevenlabs_stability=_parse_float(
            os.getenv("ELEVENLABS_STABILITY"), default=0.45, variable_name="ELEVENLABS_STABILITY"
        ),
        elevenlabs_similarity_boost=_parse_float(
            os.getenv("ELEVENLABS_SIMILARITY_BOOST"),
            default=0.85,
            variable_name="ELEVENLABS_SIMILARITY_BOOST",
        ),
        elevenlabs_style=_parse_float(
            os.getenv("ELEVENLABS_STYLE"), default=0.25, variable_name="ELEVENLABS_STYLE"
        ),
        elevenlabs_use_speaker_boost=_parse_bool(
            os.getenv("ELEVENLABS_USE_SPEAKER_BOOST"), default=True
        ),
        tts_provider=os.getenv("VADER_TTS_PROVIDER", "openai").strip().lower(),
        tts_streaming=_parse_bool(os.getenv("VADER_TTS_STREAMING"), default=False),
        save_last_audio=_parse_bool(os.getenv("VADER_SAVE_LAST_AUDIO"), default=True),
        piper_executable=os.getenv("PIPER_EXECUTABLE", "piper").strip(),
        piper_model_path=os.getenv(
            "PIPER_MODEL_PATH",
            "models/piper/es_CL-vader-medium/es_CL-vader-medium.onnx",
        ).strip(),
        piper_config_path=os.getenv(
            "PIPER_CONFIG_PATH",
            "models/piper/es_CL-vader-medium/es_CL-vader-medium.onnx.json",
        ).strip(),
        piper_length_scale=_parse_float(
            os.getenv("PIPER_LENGTH_SCALE"), default=1.0, variable_name="PIPER_LENGTH_SCALE"
        ),
        piper_noise_scale=_parse_float(
            os.getenv("PIPER_NOISE_SCALE"), default=0.667, variable_name="PIPER_NOISE_SCALE"
        ),
        piper_noise_w=_parse_float(
            os.getenv("PIPER_NOISE_W"), default=0.8, variable_name="PIPER_NOISE_W"
        ),
        piper_sentence_silence=_parse_float(
            os.getenv("PIPER_SENTENCE_SILENCE"),
            default=0.2,
            variable_name="PIPER_SENTENCE_SILENCE",
        ),
        piper_timeout_seconds=_parse_float(
            os.getenv("PIPER_TIMEOUT_SECONDS"),
            default=60.0,
            variable_name="PIPER_TIMEOUT_SECONDS",
        ),
        spotify_client_id=os.getenv("SPOTIFY_CLIENT_ID"),
        spotify_client_secret=os.getenv("SPOTIFY_CLIENT_SECRET"),
        spotify_redirect_uri=os.getenv(
            "SPOTIFY_REDIRECT_URI", "http://127.0.0.1:8765/callback"
        ),
        spotify_token_cache_path=os.getenv("SPOTIFY_TOKEN_CACHE_PATH", ".spotify_token.json"),
        spotify_playlist_aliases_path=os.getenv(
            "SPOTIFY_PLAYLIST_ALIASES_PATH", ".spotify_playlist_aliases.json"
        ),
        spotify_market=os.getenv("SPOTIFY_MARKET", "CL").strip().upper(),
        spotify_device_name=os.getenv("SPOTIFY_DEVICE_NAME"),
        language=os.getenv("VADER_LANGUAGE", "es"),
        persona_mode=os.getenv("VADER_PERSONA_MODE", "canon").strip().lower(),
        lg_tv_host=(os.getenv("LG_TV_HOST") or "").strip() or None,
        lg_tv_mac=(os.getenv("LG_TV_MAC") or "").strip() or None,
        lg_tv_broadcast_address=os.getenv(
            "LG_TV_BROADCAST_ADDRESS", "255.255.255.255"
        ).strip(),
        lg_tv_client_key_path=os.getenv(
            "LG_TV_CLIENT_KEY_PATH", ".lg_tv_client_key"
        ).strip(),
        lg_tv_schedules_path=os.getenv(
            "LG_TV_SCHEDULES_PATH", ".lg_tv_schedules.json"
        ).strip(),
        lg_tv_timeout_seconds=_parse_float(
            os.getenv("LG_TV_TIMEOUT_SECONDS"),
            default=5.0,
            variable_name="LG_TV_TIMEOUT_SECONDS",
        ),
        lg_tv_wake_timeout_seconds=_parse_float(
            os.getenv("LG_TV_WAKE_TIMEOUT_SECONDS"),
            default=40.0,
            variable_name="LG_TV_WAKE_TIMEOUT_SECONDS",
        ),
        spotify_tv_device_name=(
            (os.getenv("SPOTIFY_TV_DEVICE_NAME") or "").strip() or None
        ),
        stt_model=os.getenv("VADER_STT_MODEL", "openai/gpt-4o-transcribe").strip(),
        stt_fallback_model=os.getenv(
            "VADER_STT_FALLBACK_MODEL", "spacexai/grok-stt"
        ).strip(),
        stt_timeout_seconds=_parse_float(
            os.getenv("VADER_STT_TIMEOUT_SECONDS"),
            default=8.0,
            variable_name="VADER_STT_TIMEOUT_SECONDS",
        ),
        stt_prompt=os.getenv("VADER_STT_PROMPT", "default").strip(),
        mic_device=(os.getenv("VADER_MIC_DEVICE") or "").strip(),
        vad_model_path=os.getenv(
            "VADER_VAD_MODEL_PATH", "models/vad/silero_vad.onnx"
        ).strip(),
        listen_silence_ms=_parse_int(
            os.getenv("VADER_LISTEN_SILENCE_MS"),
            default=800,
            variable_name="VADER_LISTEN_SILENCE_MS",
        ),
        listen_max_seconds=_parse_float(
            os.getenv("VADER_LISTEN_MAX_SECONDS"),
            default=15.0,
            variable_name="VADER_LISTEN_MAX_SECONDS",
        ),
        listen_cues=_parse_bool(os.getenv("VADER_LISTEN_CUES"), default=True),
        listen_start_sound=(os.getenv("VADER_LISTEN_START_SOUND") or "").strip(),
        listen_end_sound=(os.getenv("VADER_LISTEN_END_SOUND") or "").strip(),
        listen_first_sound=(os.getenv("VADER_LISTEN_FIRST_SOUND") or "").strip(),
        stt_speculative_ms=_parse_int(
            os.getenv("VADER_STT_SPECULATIVE_MS"),
            default=192,
            variable_name="VADER_STT_SPECULATIVE_MS",
        ),
        stt_streaming=_parse_bool(os.getenv("VADER_STT_STREAMING"), default=True),
        stt_streaming_model=(
            os.getenv("VADER_STT_STREAMING_MODEL") or "spacexai/grok-stt"
        ).strip(),
        piper_persistent=_parse_bool(os.getenv("PIPER_PERSISTENT"), default=True),
        wake_model_path=(
            os.getenv("VADER_WAKE_MODEL_PATH") or "models/vosk/vosk-model-small-es-0.42"
        ).strip(),
        wake_phrase=(os.getenv("VADER_WAKE_PHRASE") or "oye vader").strip(),
        wake_variants=tuple(
            variant.strip()
            for variant in (os.getenv("VADER_WAKE_VARIANTS", "oye bei der")).split(",")
            if variant.strip()
        ),
        wake_min_confidence=_parse_float(
            os.getenv("VADER_WAKE_MIN_CONFIDENCE"),
            default=0.5,
            variable_name="VADER_WAKE_MIN_CONFIDENCE",
        ),
        follow_up_seconds=_parse_float(
            os.getenv("VADER_FOLLOW_UP_SECONDS"),
            default=10.0,
            variable_name="VADER_FOLLOW_UP_SECONDS",
        ),
        allow_shutdown=_parse_bool(os.getenv("VADER_ALLOW_SHUTDOWN"), default=False),
        lights=(os.getenv("VADER_LIGHTS") or "none").strip().lower(),
        lights_spi_device=(os.getenv("VADER_LIGHTS_SPI_DEVICE") or "/dev/spidev0.0").strip(),
        lights_brightness=_parse_float(
            os.getenv("VADER_LIGHTS_BRIGHTNESS"),
            default=0.10,
            variable_name="VADER_LIGHTS_BRIGHTNESS",
        ),
        lights_eyes=_parse_int(
            os.getenv("VADER_LIGHTS_EYES"), default=2, variable_name="VADER_LIGHTS_EYES"
        ),
        lights_center_index=_parse_int(
            os.getenv("VADER_LIGHTS_CENTER_INDEX"),
            default=0,
            variable_name="VADER_LIGHTS_CENTER_INDEX",
        ),
        debug=_parse_bool(os.getenv("VADER_DEBUG"), default=False),
    )


def validate_settings(
    settings: Settings,
    require_voice: bool = False,
    require_anthropic: bool = True,
    require_spotify: bool = False,
) -> None:
    """Validate the minimum configuration needed to run the app."""

    if settings.llm_provider not in {"anthropic", "openrouter", "vercel", "hybrid"}:
        raise ConfigurationError(
            "LLM_PROVIDER debe ser 'anthropic', 'openrouter', 'vercel' o 'hybrid'."
        )

    if require_anthropic:
        if settings.llm_provider in {"anthropic", "hybrid"} and not settings.anthropic_api_key:
            raise ConfigurationError(
                "Falta ANTHROPIC_API_KEY. Crea un archivo .env usando .env.example y pega ahi tu clave de Claude."
            )
        if settings.llm_provider == "openrouter" and not settings.openrouter_api_key:
            raise ConfigurationError(
                "Falta OPENROUTER_API_KEY para usar LLM_PROVIDER=openrouter."
            )
        if settings.llm_provider in {"vercel", "hybrid"} and not settings.ai_gateway_api_key:
            raise ConfigurationError(
                "Falta AI_GATEWAY_API_KEY para usar LLM_PROVIDER=vercel o hybrid."
            )

    if settings.openrouter_timeout_seconds <= 0:
        raise ConfigurationError("OPENROUTER_TIMEOUT_SECONDS debe ser mayor que cero.")
    if settings.openrouter_max_retries not in {0, 1}:
        raise ConfigurationError("OPENROUTER_MAX_RETRIES debe ser 0 o 1.")
    if settings.ai_gateway_timeout_seconds <= 0:
        raise ConfigurationError("AI_GATEWAY_TIMEOUT_SECONDS debe ser mayor que cero.")
    if settings.ai_gateway_max_retries not in {0, 1}:
        raise ConfigurationError("AI_GATEWAY_MAX_RETRIES debe ser 0 o 1.")
    if not settings.vader_fast_model or not settings.vader_quality_model:
        raise ConfigurationError("VADER_FAST_MODEL y VADER_QUALITY_MODEL no pueden estar vacios.")
    if settings.vader_quality_max_output_tokens <= 0:
        raise ConfigurationError("VADER_QUALITY_MAX_OUTPUT_TOKENS debe ser mayor que cero.")

    if settings.language not in {"es", "en"}:
        raise ConfigurationError("VADER_LANGUAGE debe ser 'es' o 'en'.")

    if settings.persona_mode not in {"canon", "chileno"}:
        raise ConfigurationError("VADER_PERSONA_MODE debe ser 'canon' o 'chileno'.")

    if settings.lg_tv_timeout_seconds <= 0:
        raise ConfigurationError("LG_TV_TIMEOUT_SECONDS debe ser mayor que cero.")
    if settings.lg_tv_wake_timeout_seconds <= 0:
        raise ConfigurationError("LG_TV_WAKE_TIMEOUT_SECONDS debe ser mayor que cero.")
    if settings.lg_tv_host and not settings.lg_tv_client_key_path:
        raise ConfigurationError("LG_TV_CLIENT_KEY_PATH no puede estar vacio.")
    if settings.lg_tv_host and not settings.lg_tv_schedules_path:
        raise ConfigurationError("LG_TV_SCHEDULES_PATH no puede estar vacio.")
    try:
        broadcast_address = ipaddress.ip_address(settings.lg_tv_broadcast_address)
    except ValueError as error:
        raise ConfigurationError(
            "LG_TV_BROADCAST_ADDRESS debe ser una direccion IPv4 valida."
        ) from error
    if broadcast_address.version != 4:
        raise ConfigurationError(
            "LG_TV_BROADCAST_ADDRESS debe ser una direccion IPv4 valida."
        )
    if settings.lg_tv_mac and not re.fullmatch(
        r"(?:[0-9A-Fa-f]{2}[:-]){5}[0-9A-Fa-f]{2}", settings.lg_tv_mac
    ):
        raise ConfigurationError(
            "LG_TV_MAC debe tener seis pares hexadecimales separados por ':' o '-'."
        )

    if settings.tts_provider not in {"openai", "elevenlabs", "piper"}:
        raise ConfigurationError(
            "VADER_TTS_PROVIDER debe ser 'openai', 'elevenlabs' o 'piper'."
        )

    if require_voice:
        if settings.tts_provider == "openai" and not settings.openai_api_key:
            raise ConfigurationError(
                "Falta OPENAI_API_KEY. Agrega tu clave de OpenAI al archivo .env para usar --voice."
            )
        if settings.tts_provider == "elevenlabs" and not settings.elevenlabs_api_key:
            raise ConfigurationError(
                "Falta ELEVENLABS_API_KEY. Agrega tu clave de ElevenLabs al archivo .env para usar --voice."
            )
        if settings.tts_provider == "elevenlabs" and not settings.elevenlabs_voice_id:
            raise ConfigurationError(
                "Falta ELEVENLABS_VOICE_ID. Agrega al archivo .env el ID de una voz de tu cuenta ElevenLabs."
            )
        if settings.tts_provider == "piper":
            executable_path = Path(settings.piper_executable)
            executable_exists = executable_path.is_file() or shutil.which(
                settings.piper_executable
            )
            if not executable_exists:
                raise ConfigurationError(
                    "No se encontro Piper. Revisa PIPER_EXECUTABLE en el archivo .env."
                )
            if not Path(settings.piper_model_path).is_file():
                raise ConfigurationError(
                    "No se encontro el modelo Piper. Revisa PIPER_MODEL_PATH en el archivo .env."
                )
            if not Path(settings.piper_config_path).is_file():
                raise ConfigurationError(
                    "No se encontro la configuracion Piper. Revisa PIPER_CONFIG_PATH en el archivo .env."
                )
            if settings.piper_timeout_seconds <= 0:
                raise ConfigurationError("PIPER_TIMEOUT_SECONDS debe ser mayor que cero.")

    if require_spotify:
        if not settings.spotify_client_id:
            raise ConfigurationError(
                "Falta SPOTIFY_CLIENT_ID. Crea una app en Spotify Developer Dashboard y agrega el Client ID al .env."
            )
        if not settings.spotify_client_secret:
            raise ConfigurationError(
                "Falta SPOTIFY_CLIENT_SECRET. Agrega el Client Secret de tu app de Spotify al .env."
            )
        if not settings.spotify_redirect_uri:
            raise ConfigurationError("SPOTIFY_REDIRECT_URI no puede estar vacio.")
