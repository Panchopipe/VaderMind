"""Tests for Vercel AI Gateway privacy, routing, streaming, and errors."""

from __future__ import annotations

import json
import unittest

import httpx

from vadermind.vercel_gateway import (
    VercelEndpoint,
    VercelGatewayClient,
    VercelGatewayError,
)


def _sse(*events: dict[str, object]) -> bytes:
    lines = [f"data: {json.dumps(event)}\n\n" for event in events]
    lines.append("data: [DONE]\n\n")
    return "".join(lines).encode("utf-8")


class VercelGatewayClientTests(unittest.TestCase):
    def test_non_zdr_evaluation_still_disables_prompt_training(self) -> None:
        captured: dict[str, object] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured.update(json.loads(request.content))
            return httpx.Response(
                200,
                content=_sse({"choices": [{"delta": {"content": "OK"}}]}),
            )

        client = VercelGatewayClient(
            "test-key",
            require_zdr=False,
            http_client=httpx.Client(transport=httpx.MockTransport(handler)),
        )
        client.complete(
            model="test/model",
            messages=[{"role": "user", "content": "Hola"}],
        )

        gateway = captured["providerOptions"]["gateway"]
        self.assertTrue(gateway["disallowPromptTraining"])
        self.assertNotIn("zeroDataRetention", gateway)

    def test_managed_server_search_tool_is_forwarded_unchanged(self) -> None:
        captured: dict[str, object] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured.update(json.loads(request.content))
            return httpx.Response(
                200,
                content=_sse({"choices": [{"delta": {"content": "OK"}}]}),
            )

        client = VercelGatewayClient(
            "test-key",
            require_zdr=False,
            http_client=httpx.Client(transport=httpx.MockTransport(handler)),
        )
        tool = {
            "type": "vercel:perplexity_search",
            "config": {"query": "dato actual", "max_results": 4},
        }

        client.complete(
            model="google/gemini-3.7-flash",
            messages=[{"role": "user", "content": "dato actual"}],
            tools=[tool],
        )

        self.assertEqual(captured["tools"], [tool])
        self.assertEqual(captured["tool_choice"], "auto")

    def test_endpoint_catalog_is_normalized(self) -> None:
        payload = {
            "data": {
                "endpoints": [
                    {
                        "provider_name": "vertex",
                        "context_length": 1_000_000,
                        "has_zdr": True,
                        "has_no_training": True,
                        "status": 0,
                        "uptime_last_1d": 99.99,
                        "latency_last_1h": {"p50": 850.0},
                        "supported_parameters": ["max_tokens", "tools"],
                    }
                ]
            }
        }
        client = VercelGatewayClient(
            "test-key",
            http_client=httpx.Client(
                transport=httpx.MockTransport(
                    lambda _request: httpx.Response(200, json=payload)
                )
            ),
        )

        endpoints = client.list_model_endpoints("google/test-model")

        self.assertEqual(len(endpoints), 1)
        self.assertEqual(endpoints[0].provider_name, "vertex")
        self.assertEqual(endpoints[0].latency_p50_seconds, 0.85)
        self.assertTrue(endpoints[0].has_zdr)

    def test_streaming_enforces_zdr_no_training_and_provider_pin(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            payload = json.loads(request.content)
            gateway = payload["providerOptions"]["gateway"]
            self.assertTrue(gateway["zeroDataRetention"])
            self.assertTrue(gateway["disallowPromptTraining"])
            self.assertEqual(gateway["only"], ["vertex"])
            self.assertNotIn("sort", gateway)
            return httpx.Response(
                200,
                content=_sse(
                    {
                        "model": "google/gemini-3.5-flash-lite",
                        "choices": [{"delta": {"content": "Hola"}}],
                    },
                    {
                        "choices": [],
                        "usage": {
                            "prompt_tokens": 10,
                            "completion_tokens": 2,
                            "total_tokens": 12,
                        },
                        "provider_metadata": {
                            "gateway": {
                                "cost": 0.00001,
                                "routing": {"finalProvider": "vertex"},
                            }
                        },
                    },
                ),
            )

        client = VercelGatewayClient(
            "test-key",
            http_client=httpx.Client(transport=httpx.MockTransport(handler)),
        )
        response = client.complete(
            model="google/gemini-3.5-flash-lite",
            messages=[{"role": "user", "content": "Hola"}],
            pinned_provider="vertex",
            allow_fallbacks=False,
        )

        self.assertEqual(response.text, "Hola")
        self.assertEqual(response.provider, "vertex")
        self.assertEqual(response.usage.total_tokens, 12)
        self.assertEqual(response.usage.cost_usd, 0.00001)

    def test_endpoint_catalog_requires_zdr_no_training_tools_and_uptime(self) -> None:
        endpoints = [
            VercelEndpoint("test/model", "fast", 65_536, 99.9, 0.2, ("tools",), True, True, 0),
            VercelEndpoint("test/model", "no-zdr", 65_536, 100.0, 0.1, ("tools",), False, True, 0),
            VercelEndpoint("test/model", "unstable", 65_536, 98.0, 0.05, ("tools",), True, True, 0),
        ]

        selected = VercelGatewayClient.select_benchmark_endpoint("test/model", endpoints)

        self.assertIsNotNone(selected)
        self.assertEqual(selected.provider_name, "fast")

    def test_account_verification_error_is_reported_without_credentials(self) -> None:
        client = VercelGatewayClient(
            "secret-key",
            http_client=httpx.Client(
                transport=httpx.MockTransport(
                    lambda _request: httpx.Response(
                        403,
                        json={
                            "error": {
                                "message": "A valid credit card is required.",
                                "type": "customer_verification_required",
                            }
                        },
                    )
                )
            ),
        )

        with self.assertRaises(VercelGatewayError) as context:
            client.complete(
                model="test/model",
                messages=[{"role": "user", "content": "Hola"}],
            )

        self.assertIn("HTTP 403", str(context.exception))
        self.assertIn("credit card", str(context.exception))
        self.assertNotIn("secret-key", str(context.exception))


if __name__ == "__main__":
    unittest.main()
