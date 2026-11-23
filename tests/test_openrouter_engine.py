"""Tests for opt-in tiered model routing without real network calls."""

from __future__ import annotations

from types import SimpleNamespace
import unittest

from vadermind.brain import (
    BrainError,
    HybridChatEngine,
    OpenRouterChatEngine,
    VercelGatewayChatEngine,
)
from vadermind.contracts import ModelTurn, SessionState
from vadermind.openrouter import ModelResponse, ModelToolCall, OpenRouterError
from vadermind.orchestrator import _build_chat_engine


class FakeOpenRouterClient:
    def __init__(self, fail_first: bool = False) -> None:
        self.fail_first = fail_first
        self.models: list[str] = []
        self.tools: list[object] | None = None
        self.messages: list[object] | None = None
        self.max_tokens: list[object] = []

    def complete(self, *, model: str, **kwargs: object) -> ModelResponse:
        self.models.append(model)
        self.max_tokens.append(kwargs.get("max_tokens"))
        self.tools = kwargs.get("tools")
        self.messages = kwargs.get("messages")
        if self.fail_first and len(self.models) == 1:
            raise OpenRouterError("fallo temporal")
        return ModelResponse(text=f"Respuesta de {model}", requested_model=model)


def _settings() -> SimpleNamespace:
    return SimpleNamespace(
        openrouter_api_key="test",
        openrouter_timeout_seconds=30.0,
        openrouter_max_retries=1,
        ai_gateway_api_key="test",
        ai_gateway_timeout_seconds=30.0,
        ai_gateway_max_retries=1,
        ai_gateway_require_zdr=False,
        anthropic_api_key="test",
        anthropic_model="claude-sonnet-5",
        vader_fast_model="test/fast",
        vader_quality_model="test/quality",
        vader_quality_max_output_tokens=1024,
        anthropic_max_output_tokens=250,
        persona_mode="canon",
        spotify_client_id=None,
        spotify_client_secret=None,
    )


class OpenRouterEngineTests(unittest.TestCase):
    def test_normal_conversation_uses_fast_tier(self) -> None:
        client = FakeOpenRouterClient()
        engine = OpenRouterChatEngine(_settings(), client=client)

        answer = engine.respond("Hola Vader", SessionState(language="es"))

        self.assertIn("test/fast", answer)
        self.assertEqual(client.models, ["test/fast"])
        self.assertEqual(engine.last_selected_tier, "fast")

    def test_explicit_research_uses_quality_tier_directly(self) -> None:
        client = FakeOpenRouterClient()
        engine = OpenRouterChatEngine(_settings(), client=client)

        engine.respond("Investiga y compara estas opciones", SessionState(language="es"))

        self.assertEqual(client.models, ["test/quality"])
        self.assertEqual(engine.last_selected_tier, "quality")

    def test_fast_failure_retries_once_with_quality_model(self) -> None:
        client = FakeOpenRouterClient(fail_first=True)
        engine = OpenRouterChatEngine(_settings(), client=client)

        answer = engine.respond("Hola Vader", SessionState(language="es"))

        self.assertIn("test/quality", answer)
        self.assertEqual(client.models, ["test/fast", "test/quality"])
        self.assertEqual(engine.last_selected_tier, "quality_fallback")
        # The reasoning model needs its own budget or its visible answer is truncated.
        self.assertEqual(client.max_tokens, [250, 1024])

    def test_quality_tier_uses_its_own_output_budget(self) -> None:
        client = FakeOpenRouterClient()
        engine = OpenRouterChatEngine(_settings(), client=client)

        engine.respond("Explícame paso a paso cómo hacer café", SessionState(language="es"))
        engine.respond("Hola Vader", SessionState(language="es"))

        self.assertEqual(client.models, ["test/quality", "test/fast"])
        self.assertEqual(client.max_tokens, [1024, 250])

    def test_everyday_mentions_of_today_stay_on_fast_tier(self) -> None:
        client = FakeOpenRouterClient()
        engine = OpenRouterChatEngine(_settings(), client=client)

        engine.respond("¿Qué me recomiendas cenar hoy?", SessionState(language="es"))

        self.assertEqual(client.models, ["test/fast"])

    def test_tool_aware_turn_returns_normalized_call(self) -> None:
        client = FakeOpenRouterClient()
        engine = OpenRouterChatEngine(_settings(), client=client)
        client.complete = lambda **kwargs: ModelResponse(
            text="",
            tool_calls=[
                ModelToolCall(
                    id="call-1",
                    name="spotify_play",
                    arguments={"query": "Jazz", "library_only": True},
                )
            ],
        )

        turn = engine.respond_with_tools("pon mi jazz", SessionState(language="es"))

        self.assertEqual(turn.tool_name, "spotify_play")
        self.assertEqual(turn.tool_args["query"], "Jazz")

    def test_vercel_engine_reuses_tiered_contract(self) -> None:
        client = FakeOpenRouterClient()
        engine = VercelGatewayChatEngine(_settings(), client=client)

        answer = engine.respond("Hola Vader", SessionState(language="es"))

        self.assertIn("test/fast", answer)
        self.assertEqual(client.models, ["test/fast"])
        self.assertEqual(engine.provider_label, "Vercel AI Gateway")

    def test_vercel_model_receives_safe_capability_router_without_spotify(self) -> None:
        client = FakeOpenRouterClient()
        engine = VercelGatewayChatEngine(_settings(), client=client)

        turn = engine.respond_with_tools(
            "que competencias hay hoy en los Juegos Olimpicos",
            SessionState(language="es"),
        )

        names = {
            str(item["function"]["name"])
            for item in client.tools or []
        }
        # "hoy" no longer selects the quality tier; web search is disabled.
        self.assertEqual(turn.text, "Respuesta de test/fast")
        self.assertEqual(client.models, ["test/fast"])
        self.assertIn("information_time", names)
        self.assertIn("information_weather", names)
        self.assertNotIn("information_deferred", names)
        self.assertIn("information_sports_query", names)
        self.assertIn("information_web_query", names)
        self.assertIn("external_action_blocked", names)
        self.assertNotIn("spotify_play", names)
        self.assertIn("information_web_query", str(client.messages))

    def test_vercel_web_search_forces_quality_model_and_managed_search(self) -> None:
        client = FakeOpenRouterClient()
        client.complete = lambda *, model, **kwargs: (
            client.models.append(model)
            or setattr(client, "tools", kwargs.get("tools"))
            or ModelResponse(
                text="Dato verificado.\n\nFuentes consultadas:\n- https://example.com",
                requested_model=model,
            )
        )
        engine = VercelGatewayChatEngine(_settings(), client=client)

        turn = engine.respond_with_web_search(
            "dato vigente", SessionState(language="es")
        )

        self.assertEqual(client.models, ["test/quality"])
        self.assertEqual(engine.last_selected_tier, "web_quality")
        self.assertEqual(turn.tool_name, "information_web_result")
        self.assertEqual(client.tools[0]["type"], "vercel:perplexity_search")
        self.assertEqual(client.tools[0]["config"]["max_results"], 4)

    def test_vercel_web_search_rejects_answer_without_source_url(self) -> None:
        engine = VercelGatewayChatEngine(
            _settings(), client=FakeOpenRouterClient()
        )

        with self.assertRaises(BrainError):
            engine.respond_with_web_search(
                "dato vigente", SessionState(language="es")
            )


class FakeHybridEngine:
    def __init__(self, *, fail_chat: bool = False) -> None:
        self.fail_chat = fail_chat
        self.calls: list[str] = []

    def respond(self, user_text: str, session_state: SessionState) -> str:
        self.calls.append("chat")
        if self.fail_chat:
            raise BrainError("fallo temporal")
        return "respuesta"

    def respond_with_tools(
        self, user_text: str, session_state: SessionState
    ) -> ModelTurn:
        self.calls.append("spotify")
        return ModelTurn(text="", tool_name="spotify_pause", tool_args={})

    def respond_with_information_tools(
        self, user_text: str, session_state: SessionState
    ) -> ModelTurn:
        self.calls.append("web")
        return ModelTurn(text="informacion", tool_name="information_web_result")

    def respond_with_web_search(
        self, user_text: str, session_state: SessionState
    ) -> ModelTurn:
        self.calls.append("web_search")
        return ModelTurn(text="informacion", tool_name="information_web_result")

    def respond_with_sports_search(
        self,
        user_text: str,
        sports_query: dict[str, object],
        session_state: SessionState,
    ) -> ModelTurn:
        self.calls.append("sports")
        return ModelTurn(text="resultado", tool_name="information_web_result")


class HybridEngineTests(unittest.TestCase):
    def test_factory_builds_hybrid_engine(self) -> None:
        settings = _settings()
        settings.llm_provider = "hybrid"

        engine = _build_chat_engine(settings)

        self.assertIsInstance(engine, HybridChatEngine)

    def test_routes_normal_chat_and_spotify_to_vercel(self) -> None:
        fast = FakeHybridEngine()
        quality = FakeHybridEngine()
        engine = HybridChatEngine(_settings(), fast_engine=fast, quality_engine=quality)

        self.assertEqual(engine.respond("Hola Vader", SessionState(language="es")), "respuesta")
        engine.respond_with_tools("pausa Spotify", SessionState(language="es"))

        self.assertEqual(fast.calls, ["chat", "spotify"])
        self.assertEqual(quality.calls, [])
        self.assertEqual(engine.last_selected_tier, "router_vercel")

    def test_routes_complex_and_web_requests_to_anthropic(self) -> None:
        fast = FakeHybridEngine()
        quality = FakeHybridEngine()
        engine = HybridChatEngine(_settings(), fast_engine=fast, quality_engine=quality)

        engine.respond("Analiza estas opciones", SessionState(language="es"))
        engine.respond_with_information_tools("Noticias de hoy", SessionState(language="es"))
        engine.respond_with_web_search("Dato vigente", SessionState(language="es"))
        engine.respond_with_sports_search(
            "ultimo resultado",
            {"intent": "latest_result"},
            SessionState(language="es"),
        )

        self.assertEqual(fast.calls, [])
        self.assertEqual(quality.calls, ["chat", "web", "web_search", "sports"])

    def test_normal_chat_failure_falls_back_to_anthropic(self) -> None:
        fast = FakeHybridEngine(fail_chat=True)
        quality = FakeHybridEngine()
        engine = HybridChatEngine(_settings(), fast_engine=fast, quality_engine=quality)

        answer = engine.respond("Hola Vader", SessionState(language="es"))

        self.assertEqual(answer, "respuesta")
        self.assertEqual(fast.calls, ["chat"])
        self.assertEqual(quality.calls, ["chat"])
        self.assertEqual(engine.last_selected_tier, "quality_fallback_anthropic")


if __name__ == "__main__":
    unittest.main()
