"""Tests for privacy, streaming, retries, and normalization in OpenRouter."""

from __future__ import annotations

import json
import unittest

import httpx

from vadermind.openrouter import OpenRouterClient, OpenRouterError, ZdrEndpoint


def _sse(*events: dict[str, object]) -> bytes:
    lines = [f"data: {json.dumps(event)}\n\n" for event in events]
    lines.append("data: [DONE]\n\n")
    return "".join(lines).encode("utf-8")


def _text_response(text: str = "Hola") -> bytes:
    return _sse(
        {
            "model": "openai/gpt-5.6-luna",
            "provider": "Azure",
            "choices": [{"delta": {"content": text}}],
        },
        {
            "choices": [],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 2,
                "total_tokens": 12,
                "cost": 0.00001,
            },
        },
    )


class OpenRouterClientTests(unittest.TestCase):
    def test_streaming_response_enforces_privacy_and_records_usage(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            payload = json.loads(request.content)
            self.assertEqual(payload["provider"]["data_collection"], "deny")
            self.assertTrue(payload["provider"]["zdr"])
            self.assertTrue(payload["provider"]["require_parameters"])
            self.assertEqual(payload["provider"]["order"], ["azure"])
            self.assertFalse(payload["provider"]["allow_fallbacks"])
            return httpx.Response(200, content=_text_response())

        http_client = httpx.Client(transport=httpx.MockTransport(handler))
        client = OpenRouterClient("test-key", http_client=http_client)

        response = client.complete(
            model="openai/gpt-5.6-luna",
            messages=[{"role": "user", "content": "Hola"}],
            pinned_provider="azure",
            allow_fallbacks=False,
        )

        self.assertEqual(response.text, "Hola")
        self.assertEqual(response.provider, "Azure")
        self.assertEqual(response.usage.total_tokens, 12)
        self.assertEqual(response.usage.cost_usd, 0.00001)
        self.assertIsNotNone(response.time_to_first_token_seconds)

    def test_tool_call_chunks_are_combined_and_parsed(self) -> None:
        content = _sse(
            {
                "choices": [
                    {
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "call-1",
                                    "function": {
                                        "name": "spotify_play",
                                        "arguments": '{"query":',
                                    },
                                }
                            ]
                        }
                    }
                ]
            },
            {
                "choices": [
                    {
                        "delta": {
                            "tool_calls": [
                                {"index": 0, "function": {"arguments": '"Imperial March"}'}}
                            ]
                        }
                    }
                ]
            },
        )

        client = OpenRouterClient(
            "test-key",
            http_client=httpx.Client(
                transport=httpx.MockTransport(lambda _request: httpx.Response(200, content=content))
            ),
        )
        response = client.complete(
            model="test/model",
            messages=[{"role": "user", "content": "Pon musica"}],
            tools=[{"type": "function", "function": {"name": "spotify_play"}}],
        )

        self.assertEqual(response.tool_calls[0].name, "spotify_play")
        self.assertEqual(response.tool_calls[0].arguments, {"query": "Imperial March"})

    def test_retries_once_after_rate_limit(self) -> None:
        attempts = 0

        def handler(_request: httpx.Request) -> httpx.Response:
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                return httpx.Response(429, text="rate limited")
            return httpx.Response(200, content=_text_response("Listo"))

        client = OpenRouterClient(
            "test-key",
            max_retries=1,
            http_client=httpx.Client(transport=httpx.MockTransport(handler)),
        )
        response = client.complete(
            model="test/model",
            messages=[{"role": "user", "content": "Hola"}],
        )

        self.assertEqual(response.text, "Listo")
        self.assertEqual(response.attempts, 2)

    def test_invalid_tool_arguments_fail_closed(self) -> None:
        content = _sse(
            {
                "choices": [
                    {
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "call-1",
                                    "function": {"name": "unsafe", "arguments": "not-json"},
                                }
                            ]
                        }
                    }
                ]
            }
        )
        client = OpenRouterClient(
            "test-key",
            http_client=httpx.Client(
                transport=httpx.MockTransport(lambda _request: httpx.Response(200, content=content))
            ),
        )

        with self.assertRaises(OpenRouterError):
            client.complete(
                model="test/model",
                messages=[{"role": "user", "content": "Hola"}],
            )

    def test_selects_fastest_eligible_zdr_endpoint(self) -> None:
        endpoints = [
            ZdrEndpoint("test/model", "Slow", "slow", 65_536, 99.9, 0.9, ("tools",)),
            ZdrEndpoint("test/model", "Fast", "fast", 65_536, 99.5, 0.4, ("tools",)),
            ZdrEndpoint("test/model", "Unstable", "unstable", 65_536, 98.0, 0.1, ("tools",)),
        ]

        selected = OpenRouterClient.select_benchmark_endpoint("test/model", endpoints)

        self.assertIsNotNone(selected)
        self.assertEqual(selected.tag, "fast")


if __name__ == "__main__":
    unittest.main()
