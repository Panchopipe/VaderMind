"""Speech-to-text entrypoints for VaderMind."""

from __future__ import annotations

import base64
from dataclasses import dataclass, field
from pathlib import Path
import time

import httpx

from vadermind.config import Settings
from vadermind.contracts import SpeechToText
from vadermind.warmup import LONG_KEEPALIVE, warm_connection


VERCEL_TRANSCRIPTION_URL = "https://ai-gateway.vercel.sh/v4/ai/transcription-model"
DEFAULT_TRANSCRIPTION_MODEL = "openai/gpt-4o-mini-transcribe"
# Context that nudges the transcriber toward Chilean speech and Vader's name.
# On the Pi, far speech turned "weón, ¿cachai" into "Juan, ¿qué sé" and
# "Vader" into "David". Used when VADER_STT_PROMPT is "default". It names no
# capability: with muffled audio a mention of Spotify made the model invent
# "Ya po, Spotify, pon la canción al tiro".
CHILEAN_STT_PROMPT = (
    "Conversación en español de Chile con Vader, Darth Vader. Se usan chilenismos: "
    "weón, cachai, po, ya po, al tiro, pucha, vai, hai, querís, sabís, la pega, fome, "
    "bacán."
)


class VoiceInputNotReady(RuntimeError):
    """Raised when voice input is invoked before the CLI supports it."""


class TranscriptionError(RuntimeError):
    """Raised when audio cannot be turned into text."""


class PlaceholderSpeechToText(SpeechToText):
    """Temporary implementation that keeps the contract visible and explicit."""

    def transcribe(self, audio_input: object) -> str:
        """Explain that speech input is planned but not active yet."""

        raise VoiceInputNotReady(
            "La entrada por voz todavía no está implementada. Esta versión se enfoca en chat por texto y voz de salida."
        )


@dataclass(slots=True, frozen=True)
class TranscriptionResult:
    """Text plus the measurements needed to compare transcription models."""

    text: str
    model: str
    seconds: float
    cost_usd: float | None = None
    language: str | None = None
    warnings: tuple[str, ...] = field(default_factory=tuple)


def _read_audio(audio_input: object) -> bytes:
    if isinstance(audio_input, (bytes, bytearray)):
        return bytes(audio_input)
    if isinstance(audio_input, (str, Path)):
        return Path(audio_input).read_bytes()
    raise TranscriptionError("El audio debe ser bytes WAV o la ruta a un archivo WAV.")


class VercelSpeechToText(SpeechToText):
    """Transcribe WAV audio through Vercel AI Gateway's transcription endpoint."""

    def __init__(
        self,
        api_key: str,
        *,
        model: str = DEFAULT_TRANSCRIPTION_MODEL,
        language: str = "es",
        prompt: str = "",
        require_zdr: bool = False,
        timeout_seconds: float = 30.0,
        url: str = VERCEL_TRANSCRIPTION_URL,
        http_client: httpx.Client | None = None,
    ) -> None:
        if not api_key:
            raise TranscriptionError("Falta AI_GATEWAY_API_KEY para transcribir.")
        self.api_key = api_key
        self.model = model
        self.language = language
        self.prompt = prompt
        self.require_zdr = require_zdr
        self.url = url
        self._owns_client = http_client is None
        self._client = http_client or httpx.Client(
            timeout=timeout_seconds, limits=LONG_KEEPALIVE
        )

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def warm(self) -> None:
        """Open the connection ahead of the request so the turn skips the handshake."""

        warm_connection(self._client, self.url)

    def __enter__(self) -> "VercelSpeechToText":
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()

    def transcribe(self, audio_input: object) -> str:
        """Return only the transcribed text, as the SpeechToText contract requires."""

        return self.transcribe_detailed(audio_input).text

    def transcribe_detailed(self, audio_input: object) -> TranscriptionResult:
        """Transcribe and keep latency, cost and provider warnings."""

        audio = _read_audio(audio_input)
        # Vercel only accepts zeroDataRetention on Pro/Enterprise plans.
        gateway: dict[str, object] = {"disallowPromptTraining": True}
        if self.require_zdr:
            gateway["zeroDataRetention"] = True
        provider_options: dict[str, object] = {"gateway": gateway}
        provider = self.model.split("/", 1)[0]
        options: dict[str, object] = {}
        if provider == "openai" and self.language:
            options["language"] = self.language
        if self.prompt:
            options["prompt"] = self.prompt
        if options:
            provider_options[provider] = options
        body: dict[str, object] = {
            "audio": base64.b64encode(audio).decode("ascii"),
            "mediaType": "audio/wav",
            "providerOptions": provider_options,
        }

        started = time.perf_counter()
        try:
            response = self._client.post(
                self.url,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "ai-gateway-protocol-version": "0.0.1",
                    "ai-transcription-model-specification-version": "4",
                    "ai-model-id": self.model,
                    "Content-Type": "application/json",
                },
                json=body,
            )
        except httpx.HTTPError as error:
            raise TranscriptionError(f"No se pudo contactar la transcripcion: {error}") from error
        seconds = time.perf_counter() - started

        if response.status_code != 200:
            raise TranscriptionError(
                f"La transcripcion con {self.model} fallo con HTTP {response.status_code}: "
                f"{response.text[:300]}"
            )
        try:
            payload = response.json()
        except ValueError as error:
            raise TranscriptionError("La transcripcion devolvio una respuesta no JSON.") from error

        routing = (payload.get("providerMetadata") or {}).get("gateway") or {}
        cost = routing.get("cost")
        return TranscriptionResult(
            text=str(payload.get("text") or "").strip(),
            model=self.model,
            seconds=seconds,
            cost_usd=float(cost) if cost is not None else None,
            language=payload.get("language"),
            warnings=tuple(str(item) for item in payload.get("warnings") or ()),
        )


class FallbackSpeechToText(SpeechToText):
    """Use the primary transcriber and switch to the backup only when it fails."""

    def __init__(self, primary: VercelSpeechToText, fallback: VercelSpeechToText | None) -> None:
        self.primary = primary
        self.fallback = fallback
        self.last_primary_error: str | None = None

    def close(self) -> None:
        self.primary.close()
        if self.fallback is not None:
            self.fallback.close()

    def warm(self) -> None:
        self.primary.warm()

    def transcribe(self, audio_input: object) -> str:
        return self.transcribe_detailed(audio_input).text

    def transcribe_detailed(self, audio_input: object) -> TranscriptionResult:
        audio = _read_audio(audio_input)
        self.last_primary_error = None
        try:
            return self.primary.transcribe_detailed(audio)
        except TranscriptionError as error:
            if self.fallback is None:
                raise
            self.last_primary_error = str(error)
        return self.fallback.transcribe_detailed(audio)


def build_speech_to_text(settings: Settings) -> FallbackSpeechToText:
    """Create the configured transcriber: primary model plus optional backup."""

    def create(model: str) -> VercelSpeechToText:
        return VercelSpeechToText(
            settings.ai_gateway_api_key or "",
            model=model,
            language=settings.language,
            prompt=CHILEAN_STT_PROMPT if settings.stt_prompt == "default" else settings.stt_prompt,
            require_zdr=settings.ai_gateway_require_zdr,
            timeout_seconds=settings.stt_timeout_seconds,
        )

    fallback_model = settings.stt_fallback_model
    return FallbackSpeechToText(
        create(settings.stt_model),
        create(fallback_model) if fallback_model and fallback_model != settings.stt_model else None,
    )
