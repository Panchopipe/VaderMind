"""Privacy-constrained OpenRouter client and provider-neutral response types."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import time
from typing import Any, Iterable

import httpx


OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
RECOVERABLE_STATUS_CODES = {408, 409, 429, 500, 502, 503, 504}


class OpenRouterError(RuntimeError):
    """Raised when OpenRouter cannot return a usable completion."""


@dataclass(slots=True)
class ModelToolCall:
    """Normalized function request returned by a language model."""

    id: str
    name: str
    arguments: dict[str, object]
    raw_arguments: str = ""


@dataclass(slots=True)
class ModelUsage:
    """Token and cost information for one model request."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cost_usd: float | None = None


@dataclass(slots=True)
class ModelResponse:
    """Normalized text, tools, usage, routing, and latency metadata."""

    text: str
    tool_calls: list[ModelToolCall] = field(default_factory=list)
    usage: ModelUsage = field(default_factory=ModelUsage)
    requested_model: str = ""
    effective_model: str = ""
    provider: str | None = None
    time_to_first_token_seconds: float | None = None
    total_seconds: float = 0.0
    attempts: int = 1


@dataclass(slots=True, frozen=True)
class ZdrEndpoint:
    """A privacy-compatible provider endpoint reported by OpenRouter."""

    model_id: str
    provider_name: str
    tag: str
    context_length: int
    uptime_last_1d: float
    latency_p50_seconds: float | None
    supported_parameters: tuple[str, ...]


def _json_object(value: Any) -> dict[str, object]:
    return value if isinstance(value, dict) else {}


def _parse_tool_arguments(raw_arguments: str) -> dict[str, object]:
    if not raw_arguments.strip():
        return {}
    try:
        parsed = json.loads(raw_arguments)
    except json.JSONDecodeError as error:
        raise OpenRouterError(
            f"El modelo devolvio argumentos de tool invalidos: {raw_arguments!r}."
        ) from error
    if not isinstance(parsed, dict):
        raise OpenRouterError("Los argumentos de una tool deben ser un objeto JSON.")
    return parsed


class OpenRouterClient:
    """Small OpenAI-compatible client with mandatory ZDR privacy controls."""

    def __init__(
        self,
        api_key: str,
        *,
        timeout_seconds: float = 30.0,
        max_retries: int = 1,
        base_url: str = OPENROUTER_BASE_URL,
        http_client: httpx.Client | None = None,
    ) -> None:
        if not api_key:
            raise OpenRouterError("Falta OPENROUTER_API_KEY.")
        if timeout_seconds <= 0:
            raise OpenRouterError("El timeout de OpenRouter debe ser mayor que cero.")
        if max_retries not in {0, 1}:
            raise OpenRouterError("OpenRouter admite cero o un reintento automatico.")

        self.api_key = api_key
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.base_url = base_url.rstrip("/")
        self._owns_client = http_client is None
        self._client = http_client or httpx.Client(timeout=timeout_seconds)

    def close(self) -> None:
        """Close the internally managed HTTP client."""

        if self._owns_client:
            self._client.close()

    def __enter__(self) -> "OpenRouterClient":
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()

    @property
    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "X-Title": "VaderMind model evaluation",
        }

    @staticmethod
    def provider_preferences(
        *,
        pinned_provider: str | None = None,
        allow_fallbacks: bool = True,
    ) -> dict[str, object]:
        """Build routing preferences that cannot weaken the privacy policy."""

        preferences: dict[str, object] = {
            "data_collection": "deny",
            "zdr": True,
            "require_parameters": True,
            "allow_fallbacks": allow_fallbacks,
        }
        if pinned_provider:
            preferences["order"] = [pinned_provider]
        else:
            preferences["sort"] = "latency"
        return preferences

    def list_zdr_endpoints(self) -> list[ZdrEndpoint]:
        """Return currently available Zero Data Retention endpoints."""

        try:
            response = self._client.get(
                f"{self.base_url}/endpoints/zdr",
                headers=self._headers,
                timeout=self.timeout_seconds,
            )
            response.raise_for_status()
            payload = response.json()
        except (httpx.HTTPError, ValueError) as error:
            raise OpenRouterError("No pude consultar los endpoints ZDR de OpenRouter.") from error

        endpoints: list[ZdrEndpoint] = []
        for raw in _json_object(payload).get("data", []):
            if not isinstance(raw, dict):
                continue
            latency = _json_object(raw.get("latency_last_30m"))
            supported = raw.get("supported_parameters", [])
            endpoints.append(
                ZdrEndpoint(
                    model_id=str(raw.get("model_id", "")),
                    provider_name=str(raw.get("provider_name", "")),
                    tag=str(raw.get("tag", "")),
                    context_length=int(raw.get("context_length") or 0),
                    uptime_last_1d=float(raw.get("uptime_last_1d") or 0.0),
                    latency_p50_seconds=(
                        float(latency["p50"]) if latency.get("p50") is not None else None
                    ),
                    supported_parameters=tuple(str(item) for item in supported),
                )
            )
        return endpoints

    @staticmethod
    def select_benchmark_endpoint(
        model_id: str,
        endpoints: Iterable[ZdrEndpoint],
        *,
        minimum_context: int = 32_768,
        minimum_uptime: float = 99.0,
        require_tools: bool = True,
    ) -> ZdrEndpoint | None:
        """Select the lowest-latency eligible endpoint for a controlled benchmark."""

        eligible = []
        for endpoint in endpoints:
            if endpoint.model_id != model_id:
                continue
            if endpoint.context_length < minimum_context:
                continue
            if endpoint.uptime_last_1d < minimum_uptime:
                continue
            if require_tools and "tools" not in endpoint.supported_parameters:
                continue
            if not endpoint.tag:
                continue
            eligible.append(endpoint)

        return min(
            eligible,
            key=lambda endpoint: (
                endpoint.latency_p50_seconds is None,
                endpoint.latency_p50_seconds or float("inf"),
                endpoint.tag,
            ),
            default=None,
        )

    def complete(
        self,
        *,
        model: str,
        messages: list[dict[str, object]],
        tools: list[dict[str, object]] | None = None,
        max_tokens: int = 250,
        pinned_provider: str | None = None,
        allow_fallbacks: bool = True,
    ) -> ModelResponse:
        """Run a streaming completion and measure user-visible latency."""

        payload: dict[str, object] = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "stream": True,
            "stream_options": {"include_usage": True},
            "provider": self.provider_preferences(
                pinned_provider=pinned_provider,
                allow_fallbacks=allow_fallbacks,
            ),
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"

        last_error: Exception | None = None
        for attempt in range(1, self.max_retries + 2):
            try:
                return self._stream_completion(payload, requested_model=model, attempts=attempt)
            except httpx.HTTPStatusError as error:
                last_error = error
                if error.response.status_code not in RECOVERABLE_STATUS_CODES:
                    break
            except (httpx.TimeoutException, httpx.ConnectError) as error:
                last_error = error
            except OpenRouterError:
                raise

            if attempt <= self.max_retries:
                continue
            break

        if isinstance(last_error, httpx.HTTPStatusError):
            status = last_error.response.status_code
            raise OpenRouterError(f"OpenRouter devolvio HTTP {status}.") from last_error
        raise OpenRouterError("No pude comunicarme con OpenRouter.") from last_error

    def _stream_completion(
        self,
        payload: dict[str, object],
        *,
        requested_model: str,
        attempts: int,
    ) -> ModelResponse:
        start = time.perf_counter()
        first_token: float | None = None
        text_parts: list[str] = []
        tool_buffers: dict[int, dict[str, str]] = {}
        usage = ModelUsage()
        effective_model = requested_model
        provider: str | None = None
        received_event = False

        with self._client.stream(
            "POST",
            f"{self.base_url}/chat/completions",
            headers=self._headers,
            json=payload,
            timeout=self.timeout_seconds,
        ) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if not data or data == "[DONE]":
                    continue
                received_event = True
                try:
                    event = json.loads(data)
                except json.JSONDecodeError as error:
                    raise OpenRouterError("OpenRouter devolvio un evento SSE invalido.") from error
                if not isinstance(event, dict):
                    continue
                effective_model = str(event.get("model") or effective_model)
                provider_value = event.get("provider")
                if provider_value:
                    provider = str(provider_value)
                self._update_usage(usage, event.get("usage"))
                choices = event.get("choices", [])
                if not isinstance(choices, list) or not choices:
                    continue
                delta = _json_object(_json_object(choices[0]).get("delta"))
                content = delta.get("content")
                if isinstance(content, str) and content:
                    if first_token is None:
                        first_token = time.perf_counter() - start
                    text_parts.append(content)
                raw_tool_calls = delta.get("tool_calls", [])
                if isinstance(raw_tool_calls, list):
                    for raw_call in raw_tool_calls:
                        if not isinstance(raw_call, dict):
                            continue
                        if first_token is None:
                            first_token = time.perf_counter() - start
                        index = int(raw_call.get("index") or 0)
                        buffer = tool_buffers.setdefault(
                            index, {"id": "", "name": "", "arguments": ""}
                        )
                        if raw_call.get("id"):
                            buffer["id"] = str(raw_call["id"])
                        function = _json_object(raw_call.get("function"))
                        if function.get("name"):
                            buffer["name"] += str(function["name"])
                        if function.get("arguments"):
                            buffer["arguments"] += str(function["arguments"])

        if not received_event:
            raise OpenRouterError("OpenRouter respondio sin eventos utilizables.")

        tool_calls = [
            ModelToolCall(
                id=buffer["id"] or f"tool-call-{index}",
                name=buffer["name"],
                arguments=_parse_tool_arguments(buffer["arguments"]),
                raw_arguments=buffer["arguments"],
            )
            for index, buffer in sorted(tool_buffers.items())
            if buffer["name"]
        ]
        text = "".join(text_parts).strip()
        if not text and not tool_calls:
            raise OpenRouterError("OpenRouter respondio sin texto ni tool calls utilizables.")

        return ModelResponse(
            text=text,
            tool_calls=tool_calls,
            usage=usage,
            requested_model=requested_model,
            effective_model=effective_model,
            provider=provider,
            time_to_first_token_seconds=first_token,
            total_seconds=time.perf_counter() - start,
            attempts=attempts,
        )

    @staticmethod
    def _update_usage(usage: ModelUsage, raw_usage: object) -> None:
        if not isinstance(raw_usage, dict):
            return
        usage.prompt_tokens = int(raw_usage.get("prompt_tokens") or usage.prompt_tokens)
        usage.completion_tokens = int(
            raw_usage.get("completion_tokens") or usage.completion_tokens
        )
        usage.total_tokens = int(
            raw_usage.get("total_tokens")
            or usage.prompt_tokens + usage.completion_tokens
        )
        raw_cost = raw_usage.get("cost")
        if raw_cost is not None:
            usage.cost_usd = float(raw_cost)


def tool_call_to_message(tool_call: ModelToolCall) -> dict[str, object]:
    """Convert a normalized tool call back to Chat Completions wire format."""

    return {
        "id": tool_call.id,
        "type": "function",
        "function": {
            "name": tool_call.name,
            "arguments": tool_call.raw_arguments or json.dumps(tool_call.arguments),
        },
    }
