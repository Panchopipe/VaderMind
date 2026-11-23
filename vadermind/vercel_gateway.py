"""Privacy-constrained Vercel AI Gateway client for VaderMind."""

from __future__ import annotations

from dataclasses import dataclass
import json
import time
from typing import Any, Iterable

import httpx

from vadermind.openrouter import (
    ModelResponse,
    ModelToolCall,
    ModelUsage,
    OpenRouterError,
)
from vadermind.warmup import LONG_KEEPALIVE, warm_connection


VERCEL_AI_GATEWAY_BASE_URL = "https://ai-gateway.vercel.sh/v1"
RECOVERABLE_STATUS_CODES = {408, 409, 429, 500, 502, 503, 504}


class VercelGatewayError(OpenRouterError):
    """Raised when Vercel AI Gateway cannot return a usable completion."""


@dataclass(slots=True, frozen=True)
class VercelEndpoint:
    """A model endpoint reported by Vercel's provider catalog."""

    model_id: str
    provider_name: str
    context_length: int
    uptime_last_1d: float
    latency_p50_seconds: float | None
    supported_parameters: tuple[str, ...]
    has_zdr: bool
    has_no_training: bool
    status: int


def _json_object(value: Any) -> dict[str, object]:
    return value if isinstance(value, dict) else {}


def _parse_tool_arguments(raw_arguments: str) -> dict[str, object]:
    if not raw_arguments.strip():
        return {}
    try:
        parsed = json.loads(raw_arguments)
    except json.JSONDecodeError as error:
        raise VercelGatewayError(
            f"El modelo devolvio argumentos de tool invalidos: {raw_arguments!r}."
        ) from error
    if not isinstance(parsed, dict):
        raise VercelGatewayError("Los argumentos de una tool deben ser un objeto JSON.")
    return parsed


class VercelGatewayClient:
    """Small OpenAI-compatible client with mandatory ZDR/no-training routing."""

    def __init__(
        self,
        api_key: str,
        *,
        timeout_seconds: float = 30.0,
        max_retries: int = 1,
        require_zdr: bool = True,
        base_url: str = VERCEL_AI_GATEWAY_BASE_URL,
        http_client: httpx.Client | None = None,
    ) -> None:
        if not api_key:
            raise VercelGatewayError("Falta AI_GATEWAY_API_KEY.")
        if timeout_seconds <= 0:
            raise VercelGatewayError("El timeout de Vercel debe ser mayor que cero.")
        if max_retries not in {0, 1}:
            raise VercelGatewayError("Vercel admite cero o un reintento automatico.")

        self.api_key = api_key
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.require_zdr = require_zdr
        self.base_url = base_url.rstrip("/")
        self._owns_client = http_client is None
        self._client = http_client or httpx.Client(
            timeout=timeout_seconds, limits=LONG_KEEPALIVE
        )

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def warm(self) -> None:
        """Open the connection ahead of the request so the turn skips the handshake."""

        warm_connection(self._client, self.base_url)

    def __enter__(self) -> "VercelGatewayClient":
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

    def provider_options(
        self,
        *,
        pinned_provider: str | None = None,
        allow_fallbacks: bool = True,
    ) -> dict[str, object]:
        """Build routing options that always enforce the agreed privacy policy."""

        gateway: dict[str, object] = {
            "disallowPromptTraining": True,
        }
        if self.require_zdr:
            gateway["zeroDataRetention"] = True
        if pinned_provider:
            if allow_fallbacks:
                gateway["order"] = [pinned_provider]
            else:
                gateway["only"] = [pinned_provider]
        else:
            gateway["sort"] = "ttft"
        return {"gateway": gateway}

    def list_model_endpoints(self, model_id: str) -> list[VercelEndpoint]:
        """Return provider endpoints for one model from Vercel's live catalog."""

        try:
            response = self._client.get(
                f"{self.base_url}/models/{model_id}/endpoints",
                headers=self._headers,
                timeout=self.timeout_seconds,
            )
            response.raise_for_status()
            payload = response.json()
        except (httpx.HTTPError, ValueError) as error:
            raise VercelGatewayError(
                f"No pude consultar los endpoints de Vercel para {model_id}."
            ) from error

        data = _json_object(_json_object(payload).get("data"))
        raw_endpoints = data.get("endpoints", [])
        endpoints: list[VercelEndpoint] = []
        if not isinstance(raw_endpoints, list):
            return endpoints
        for raw in raw_endpoints:
            if not isinstance(raw, dict):
                continue
            latency = _json_object(raw.get("latency_last_1h"))
            latency_ms = latency.get("p50")
            supported = raw.get("supported_parameters", [])
            endpoints.append(
                VercelEndpoint(
                    model_id=model_id,
                    provider_name=str(raw.get("provider_name", "")),
                    context_length=int(raw.get("context_length") or 0),
                    uptime_last_1d=float(raw.get("uptime_last_1d") or 0.0),
                    latency_p50_seconds=(
                        float(latency_ms) / 1000.0 if latency_ms is not None else None
                    ),
                    supported_parameters=tuple(str(item) for item in supported),
                    has_zdr=bool(raw.get("has_zdr")),
                    has_no_training=bool(raw.get("has_no_training")),
                    status=int(raw.get("status", -1)),
                )
            )
        return endpoints

    @staticmethod
    def select_benchmark_endpoint(
        model_id: str,
        endpoints: Iterable[VercelEndpoint],
        *,
        minimum_context: int = 32_768,
        minimum_uptime: float = 99.0,
        require_tools: bool = True,
        require_zdr: bool = True,
    ) -> VercelEndpoint | None:
        """Select the lowest-TTFT eligible ZDR/no-training endpoint."""

        eligible = []
        for endpoint in endpoints:
            if endpoint.model_id != model_id or endpoint.status != 0:
                continue
            if (require_zdr and not endpoint.has_zdr) or not endpoint.has_no_training:
                continue
            if endpoint.context_length < minimum_context:
                continue
            if endpoint.uptime_last_1d < minimum_uptime:
                continue
            if require_tools and "tools" not in endpoint.supported_parameters:
                continue
            if not endpoint.provider_name:
                continue
            eligible.append(endpoint)

        return min(
            eligible,
            key=lambda endpoint: (
                endpoint.latency_p50_seconds is None,
                endpoint.latency_p50_seconds or float("inf"),
                endpoint.provider_name,
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
            "providerOptions": self.provider_options(
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
                result = self._stream_completion(
                    payload, requested_model=model, attempts=attempt
                )
                if result.provider is None and pinned_provider and not allow_fallbacks:
                    result.provider = pinned_provider
                return result
            except httpx.HTTPStatusError as error:
                last_error = error
                if error.response.status_code not in RECOVERABLE_STATUS_CODES:
                    break
            except (httpx.TimeoutException, httpx.ConnectError) as error:
                last_error = error
            except VercelGatewayError:
                raise

            if attempt > self.max_retries:
                break

        if isinstance(last_error, httpx.HTTPStatusError):
            status = last_error.response.status_code
            detail = self._safe_error_detail(last_error.response)
            suffix = f": {detail}" if detail else "."
            raise VercelGatewayError(f"Vercel AI Gateway devolvio HTTP {status}{suffix}") from last_error
        raise VercelGatewayError("No pude comunicarme con Vercel AI Gateway.") from last_error

    @staticmethod
    def _safe_error_detail(response: httpx.Response) -> str:
        try:
            error = _json_object(_json_object(response.json()).get("error"))
            return str(error.get("message") or "").strip()
        except ValueError:
            return ""

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
            if response.is_error:
                response.read()
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
                    raise VercelGatewayError("Vercel devolvio un evento SSE invalido.") from error
                if not isinstance(event, dict):
                    continue
                if isinstance(event.get("error"), dict):
                    message = str(_json_object(event["error"]).get("message") or "error desconocido")
                    raise VercelGatewayError(f"Vercel AI Gateway rechazo la solicitud: {message}")
                effective_model = str(event.get("model") or effective_model)
                provider = self._update_gateway_metadata(usage, event, provider)
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
            raise VercelGatewayError("Vercel respondio sin eventos utilizables.")

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
            raise VercelGatewayError("Vercel respondio sin texto ni tool calls utilizables.")

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
        usage.prompt_tokens = int(
            raw_usage.get("prompt_tokens")
            or raw_usage.get("input_tokens")
            or usage.prompt_tokens
        )
        usage.completion_tokens = int(
            raw_usage.get("completion_tokens")
            or raw_usage.get("output_tokens")
            or usage.completion_tokens
        )
        usage.total_tokens = int(
            raw_usage.get("total_tokens")
            or usage.prompt_tokens + usage.completion_tokens
        )

    @staticmethod
    def _update_gateway_metadata(
        usage: ModelUsage,
        event: dict[str, object],
        current_provider: str | None,
    ) -> str | None:
        metadata = _json_object(
            event.get("provider_metadata") or event.get("providerMetadata")
        )
        gateway = _json_object(metadata.get("gateway"))
        routing = _json_object(gateway.get("routing"))
        provider = (
            event.get("provider")
            or routing.get("finalProvider")
            or gateway.get("provider")
            or current_provider
        )
        if gateway.get("cost") is not None:
            usage.cost_usd = float(gateway["cost"])
        return str(provider) if provider else None
