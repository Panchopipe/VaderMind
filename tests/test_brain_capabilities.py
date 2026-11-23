"""Tests for truthful capability context passed to Claude."""

from __future__ import annotations

from types import SimpleNamespace
import unittest

from vadermind.brain import BrainError, ClaudeChatEngine, build_runtime_system_prompt
from vadermind.contracts import SessionState


class FakeAnthropicMessages:
    def __init__(self, response: object) -> None:
        self.response = response
        self.request: dict[str, object] = {}
        self.requests: list[dict[str, object]] = []

    def create(self, **kwargs: object) -> object:
        self.request = kwargs
        self.requests.append(kwargs)
        if isinstance(self.response, list):
            return self.response.pop(0)
        return self.response


class FakeAnthropicClient:
    def __init__(self, response: object) -> None:
        self.messages = FakeAnthropicMessages(response)


class BrainCapabilityPromptTests(unittest.TestCase):
    """Ensure Claude does not contradict real tool results."""

    def test_spotify_enabled_prompt_trusts_tool_confirmations(self) -> None:
        settings = SimpleNamespace(
            persona_mode="canon",
            spotify_client_id="configured",
            spotify_client_secret="configured",
        )

        prompt = build_runtime_system_prompt(settings, "es")

        self.assertIn("integracion real y autorizada con Spotify", prompt)
        self.assertIn("nunca afirmes que Vader carece de acceso a Spotify", prompt)
        self.assertIn("Nunca respondas con 'Hecho'", prompt)
        self.assertIn("sin una tool call", prompt)

    def test_lg_tv_enabled_prompt_describes_real_local_control(self) -> None:
        settings = SimpleNamespace(
            persona_mode="canon",
            spotify_client_id=None,
            spotify_client_secret=None,
            lg_tv_host="192.0.2.162",
        )

        prompt = build_runtime_system_prompt(settings, "es")

        self.assertIn("integracion local y autorizada", prompt)
        self.assertIn("television LG webOS", prompt)
        self.assertIn("Nunca confirmes una accion", prompt)

    def test_claude_returns_structured_spotify_tool_call(self) -> None:
        response = SimpleNamespace(
            content=[
                SimpleNamespace(
                    type="tool_use",
                    name="spotify_play",
                    input={
                        "query": "Jazz",
                        "item_type": "playlist",
                        "library_only": True,
                        "shuffle": True,
                    },
                )
            ]
        )
        settings = SimpleNamespace(
            anthropic_api_key="test",
            anthropic_model="test-model",
            anthropic_max_output_tokens=250,
            persona_mode="canon",
            spotify_client_id="configured",
            spotify_client_secret="configured",
            lg_tv_host="192.0.2.162",
        )
        engine = ClaudeChatEngine(settings)
        engine._client = FakeAnthropicClient(response)
        state = SessionState(
            language="es",
            messages=[{"role": "user", "content": "pon algo de mi jazz pero mezclado"}],
        )

        turn = engine.respond_with_tools("pon algo de mi jazz pero mezclado", state)

        self.assertEqual(turn.tool_name, "spotify_play")
        self.assertEqual(turn.tool_args["query"], "Jazz")
        request = engine._client.messages.request
        self.assertIn("tools", request)
        self.assertEqual(len(request["messages"]), 1)
        tool_names = {str(tool.get("name")) for tool in request["tools"]}
        self.assertIn("spotify_play", tool_names)
        self.assertIn("information_time", tool_names)
        self.assertIn("information_weather", tool_names)
        self.assertNotIn("information_deferred", tool_names)
        self.assertIn("information_sports_query", tool_names)
        self.assertIn("information_web_query", tool_names)
        self.assertIn("external_action_blocked", tool_names)
        self.assertIn("lg_tv_power", tool_names)
        self.assertIn("lg_tv_launch_app", tool_names)
        self.assertIn("information_web_query", str(request["system"]))

    def test_information_turn_exposes_read_only_tools_without_spotify(self) -> None:
        response = SimpleNamespace(
            content=[
                SimpleNamespace(
                    type="tool_use",
                    name="information_weather",
                    input={"location": "Santiago", "when": "tomorrow"},
                )
            ]
        )
        settings = SimpleNamespace(
            anthropic_api_key="test",
            anthropic_model="test-model",
            anthropic_max_output_tokens=250,
            persona_mode="canon",
            spotify_client_id="configured",
            spotify_client_secret="configured",
        )
        engine = ClaudeChatEngine(settings)
        engine._client = FakeAnthropicClient(response)

        turn = engine.respond_with_information_tools(
            "que clima hara manana en Santiago", SessionState(language="es")
        )

        self.assertEqual(turn.tool_name, "information_weather")
        tool_names = {
            str(tool.get("name")) for tool in engine._client.messages.request["tools"]
        }
        self.assertIn("information_weather", tool_names)
        self.assertIn("web_search", tool_names)
        self.assertNotIn("spotify_play", tool_names)
        self.assertEqual(engine._client.messages.request["max_tokens"], 500)

    def test_web_search_answer_preserves_source_urls(self) -> None:
        citation = SimpleNamespace(title="Tienda", url="https://example.com/producto")
        response = SimpleNamespace(
            content=[
                SimpleNamespace(type="server_tool_use", name="web_search"),
                SimpleNamespace(type="web_search_tool_result", content=[]),
                SimpleNamespace(
                    type="text",
                    text="El precio observado es 100.",
                    citations=[citation],
                ),
            ]
        )
        settings = SimpleNamespace(
            anthropic_api_key="test",
            anthropic_model="test-model",
            anthropic_max_output_tokens=250,
            persona_mode="canon",
            spotify_client_id="configured",
            spotify_client_secret="configured",
        )
        engine = ClaudeChatEngine(settings)
        engine._client = FakeAnthropicClient(response)

        turn = engine.respond_with_information_tools(
            "cuanto cuesta el producto", SessionState(language="es")
        )

        self.assertEqual(turn.tool_name, "information_web_result")
        self.assertIn("https://example.com/producto", turn.text)

    def test_model_routed_web_query_forces_a_real_read_only_search(self) -> None:
        citation = SimpleNamespace(
            title="Dato actual",
            url="https://example.com/dato-actual",
        )
        response = SimpleNamespace(
            content=[
                SimpleNamespace(type="server_tool_use", name="web_search"),
                SimpleNamespace(type="web_search_tool_result", content=[]),
                SimpleNamespace(
                    type="text",
                    text="El dato vigente es 42.",
                    citations=[citation],
                ),
            ]
        )
        settings = SimpleNamespace(
            anthropic_api_key="test",
            anthropic_model="test-model",
            anthropic_max_output_tokens=250,
            persona_mode="canon",
            spotify_client_id="configured",
            spotify_client_secret="configured",
        )
        engine = ClaudeChatEngine(settings)
        engine._client = FakeAnthropicClient(response)

        turn = engine.respond_with_web_search(
            "necesito el valor vigente", SessionState(language="es")
        )

        self.assertEqual(turn.tool_name, "information_web_result")
        self.assertIn("https://example.com/dato-actual", turn.text)
        request = engine._client.messages.request
        self.assertEqual(
            request["tool_choice"], {"type": "tool", "name": "web_search"}
        )
        self.assertEqual(
            {tool.get("name") for tool in request["tools"]}, {"web_search"}
        )

    def test_current_day_web_query_retries_a_past_only_answer(self) -> None:
        citation = SimpleNamespace(
            title="Calendario oficial",
            url="https://example.com/calendario",
        )

        def web_response(text: str) -> object:
            return SimpleNamespace(
                content=[
                    SimpleNamespace(type="server_tool_use", name="web_search"),
                    SimpleNamespace(type="web_search_tool_result", content=[]),
                    SimpleNamespace(type="text", text=text, citations=[citation]),
                ]
            )

        settings = SimpleNamespace(
            anthropic_api_key="test",
            anthropic_model="test-model",
            anthropic_max_output_tokens=250,
            persona_mode="canon",
            spotify_client_id="configured",
            spotify_client_secret="configured",
        )
        engine = ClaudeChatEngine(settings)
        engine._client = FakeAnthropicClient(
            [
                web_response("Los Juegos se disputaron en febrero de 2026."),
                web_response("No hay Juegos Olimpicos en curso hoy."),
            ]
        )

        turn = engine.respond_with_web_search(
            "que competencias se disputan hoy en los Juegos Olimpicos?",
            SessionState(language="es"),
        )

        self.assertIn("No hay Juegos Olimpicos en curso hoy", turn.text)
        self.assertEqual(len(engine._client.messages.requests), 2)
        self.assertIn(
            "Una fecha o edicion pasada por si sola no responde",
            engine._client.messages.requests[1]["system"],
        )

    def test_current_day_web_query_fails_closed_after_two_past_answers(self) -> None:
        response = SimpleNamespace(
            content=[
                SimpleNamespace(type="server_tool_use", name="web_search"),
                SimpleNamespace(type="web_search_tool_result", content=[]),
                SimpleNamespace(
                    type="text",
                    text="El torneo se disputo en febrero de 2026.",
                    citations=[],
                ),
            ]
        )
        settings = SimpleNamespace(
            anthropic_api_key="test",
            anthropic_model="test-model",
            anthropic_max_output_tokens=250,
            persona_mode="canon",
            spotify_client_id="configured",
            spotify_client_secret="configured",
        )
        engine = ClaudeChatEngine(settings)
        engine._client = FakeAnthropicClient([response, response])

        with self.assertRaisesRegex(BrainError, "correspondiente a hoy"):
            engine.respond_with_web_search(
                "que ocurre hoy en el torneo?", SessionState(language="es")
            )

    def test_sports_search_is_restricted_to_score_sources(self) -> None:
        citation = SimpleNamespace(
            title="Campeonato Chileno",
            url="https://www.campeonatochileno.cl/match/example",
        )
        response = SimpleNamespace(
            content=[
                SimpleNamespace(type="server_tool_use", name="web_search"),
                SimpleNamespace(type="web_search_tool_result", content=[]),
                SimpleNamespace(
                    type="text",
                    text="Universidad de Chile gano 4-2.",
                    citations=[citation],
                ),
            ]
        )
        settings = SimpleNamespace(
            anthropic_api_key="test",
            anthropic_model="test-model",
            anthropic_max_output_tokens=250,
            persona_mode="canon",
            spotify_client_id="configured",
            spotify_client_secret="configured",
        )
        engine = ClaudeChatEngine(settings)
        engine._client = FakeAnthropicClient(response)

        engine.respond_with_sports_search(
            "como le fue a la U de Chile en su ultimo partido",
            {
                "intent": "latest_result",
                "sport": "football",
                "participant": "Universidad de Chile",
                "country": "Chile",
            },
            SessionState(language="es"),
        )

        web_tool = next(
            tool
            for tool in engine._client.messages.request["tools"]
            if tool.get("name") == "web_search"
        )
        self.assertIn("campeonatochileno.cl", web_tool["allowed_domains"])
        self.assertNotIn("sofascore.com", web_tool["allowed_domains"])
        fetch_tool = next(
            tool
            for tool in engine._client.messages.request["tools"]
            if tool.get("name") == "web_fetch"
        )
        self.assertEqual(fetch_tool["max_uses"], 1)
        self.assertEqual(
            fetch_tool["allowed_domains"], web_tool["allowed_domains"]
        )

    def test_tennis_schedule_uses_official_tournament_sources(self) -> None:
        citation = SimpleNamespace(
            title="US Open schedule",
            url="https://www.usopen.org/en_US/scores/schedule/",
        )
        response = SimpleNamespace(
            content=[
                SimpleNamespace(type="server_tool_use", name="web_search"),
                SimpleNamespace(type="web_search_tool_result", content=[]),
                SimpleNamespace(
                    type="text",
                    text=(
                        "Cuadro masculino: Carlos Alcaraz vs Jannik Sinner.\n"
                        "Cuadro femenino: Coco Gauff vs Aryna Sabalenka.\n"
                        "Resumen: Alcaraz contra Sinner y Gauff contra Sabalenka.\n"
                        "El Imperio observa con interes."
                    ),
                    citations=[citation],
                ),
            ]
        )
        settings = SimpleNamespace(
            anthropic_api_key="test",
            anthropic_model="test-model",
            anthropic_max_output_tokens=250,
            persona_mode="canon",
            spotify_client_id="configured",
            spotify_client_secret="configured",
        )
        engine = ClaudeChatEngine(settings)
        engine._client = FakeAnthropicClient(response)

        turn = engine.respond_with_sports_search(
            "quienes juegan los cuartos de final hoy en el US Open?",
            {
                "intent": "schedule",
                "sport": "tennis",
                "event": "US Open",
                "competition": "US Open",
                "date_reference": "hoy, cuartos de final",
            },
            SessionState(language="es"),
        )

        self.assertEqual(turn.tool_name, "information_web_result")
        web_tool = next(
            tool
            for tool in engine._client.messages.request["tools"]
            if tool.get("name") == "web_search"
        )
        self.assertIn("usopen.org", web_tool["allowed_domains"])
        self.assertNotIn("spotify_play", str(engine._client.messages.request["tools"]))
        self.assertEqual(len(engine._client.messages.requests), 1)

    def test_malformed_tennis_schedule_is_retried_with_one_matchup_per_line(self) -> None:
        citation = SimpleNamespace(
            title="US Open schedule",
            url="https://www.usopen.org/en_US/scores/schedule/",
        )
        malformed = SimpleNamespace(
            content=[
                SimpleNamespace(type="server_tool_use", name="web_search"),
                SimpleNamespace(type="web_search_tool_result", content=[]),
                SimpleNamespace(
                    type="text",
                    text=(
                        "Aryna Sabalenka vs Linda Noskova; "
                        "Jessica Pegula vs Emma Navarro."
                    ),
                    citations=[citation],
                ),
            ]
        )
        corrected = SimpleNamespace(
            content=[
                SimpleNamespace(type="server_tool_use", name="web_search"),
                SimpleNamespace(type="web_search_tool_result", content=[]),
                SimpleNamespace(
                    type="text",
                    text=(
                        "Cuartos de final de hoy:\n"
                        "- WTA: Aryna Sabalenka vs Linda Noskova\n"
                        "- WTA: Jessica Pegula vs Emma Navarro"
                    ),
                    citations=[citation],
                ),
            ]
        )
        settings = SimpleNamespace(
            anthropic_api_key="test",
            anthropic_model="test-model",
            anthropic_max_output_tokens=250,
            persona_mode="canon",
            spotify_client_id="configured",
            spotify_client_secret="configured",
        )
        engine = ClaudeChatEngine(settings)
        engine._client = FakeAnthropicClient([malformed, corrected])

        turn = engine.respond_with_sports_search(
            "quienes juegan los cuartos de final hoy en el US Open?",
            {
                "intent": "schedule",
                "sport": "tennis",
                "event": "US Open",
                "date_reference": "hoy, cuartos de final",
            },
            SessionState(language="es"),
        )

        self.assertIn("Sabalenka vs Linda Noskova", turn.text)
        self.assertIn("Pegula vs Emma Navarro", turn.text)
        self.assertEqual(len(engine._client.messages.requests), 2)
        self.assertFalse(
            ClaudeChatEngine._sports_answer_has_required_fact(
                "Linda Noskova — 11:30 a.m.\nBen Shelton — 7 p.m.",
                {
                    "intent": "schedule",
                    "sport": "tennis",
                    "competition": "US Open",
                },
            )
        )

    def test_multisport_schedule_accepts_named_events_with_verified_times(self) -> None:
        citation = SimpleNamespace(
            title="Olympic schedule",
            url="https://olympics.com/en/olympic-games/olympic-schedule",
        )
        response = SimpleNamespace(
            content=[
                SimpleNamespace(type="server_tool_use", name="web_search"),
                SimpleNamespace(type="web_search_tool_result", content=[]),
                SimpleNamespace(
                    type="text",
                    text=(
                        "- Atletismo: 100 metros masculinos — 10:30\n"
                        "- Natacion: 200 metros libres femeninos — 12:00\n"
                        "El Imperio observa la jornada."
                    ),
                    citations=[citation],
                ),
            ]
        )
        settings = SimpleNamespace(
            anthropic_api_key="test",
            anthropic_model="test-model",
            anthropic_max_output_tokens=250,
            persona_mode="canon",
            spotify_client_id="configured",
            spotify_client_secret="configured",
        )
        engine = ClaudeChatEngine(settings)
        engine._client = FakeAnthropicClient(response)

        turn = engine.respond_with_sports_search(
            "que competencias hay hoy en los Juegos Olimpicos",
            {
                "intent": "schedule",
                "sport": "other",
                "event": "Juegos Olimpicos",
                "competition": "Juegos Olimpicos",
                "date_reference": "hoy",
            },
            SessionState(language="es"),
        )

        self.assertIn("100 metros masculinos", turn.text)
        self.assertEqual(len(engine._client.messages.requests), 1)
        self.assertTrue(
            ClaudeChatEngine._sports_answer_has_required_fact(
                "No hay Juegos Olímpicos en curso hoy.",
                {
                    "intent": "schedule",
                    "sport": "other",
                    "competition": "Juegos Olimpicos",
                },
            )
        )

    def test_sports_question_is_interpreted_before_any_web_search(self) -> None:
        response = SimpleNamespace(
            content=[
                SimpleNamespace(
                    type="tool_use",
                    name="information_sports_query",
                    input={
                        "intent": "event_result",
                        "sport": "american_football",
                        "event": "Super Bowl",
                        "competition": "NFL",
                        "date_reference": "most recent",
                    },
                )
            ]
        )
        settings = SimpleNamespace(
            anthropic_api_key="test",
            anthropic_model="test-model",
            anthropic_max_output_tokens=250,
            persona_mode="canon",
            spotify_client_id="configured",
            spotify_client_secret="configured",
        )
        engine = ClaudeChatEngine(settings)
        engine._client = FakeAnthropicClient(response)

        turn = engine.respond_with_information_tools(
            "quien gano el Super Bowl", SessionState(language="es")
        )

        self.assertEqual(turn.tool_name, "information_sports_query")
        self.assertEqual(turn.tool_args["sport"], "american_football")
        tool_names = {
            tool.get("name") for tool in engine._client.messages.request["tools"]
        }
        self.assertEqual(tool_names, {"information_sports_query"})
        self.assertEqual(
            engine._client.messages.request["tool_choice"],
            {"type": "tool", "name": "information_sports_query"},
        )

    def test_paused_web_search_continues_until_final_answer(self) -> None:
        paused = SimpleNamespace(
            stop_reason="pause_turn",
            content=[SimpleNamespace(type="server_tool_use", name="web_search")],
        )
        citation = SimpleNamespace(
            title="Campeonato Chileno",
            url="https://www.campeonatochileno.cl/match/example",
        )
        completed = SimpleNamespace(
            stop_reason="end_turn",
            content=[
                SimpleNamespace(type="web_fetch_tool_result", content={}),
                SimpleNamespace(
                    type="text",
                    text="Universidad de Chile vencio 4-2 a Coquimbo Unido.",
                    citations=[citation],
                ),
            ],
        )
        settings = SimpleNamespace(
            anthropic_api_key="test",
            anthropic_model="test-model",
            anthropic_max_output_tokens=250,
            persona_mode="canon",
            spotify_client_id="configured",
            spotify_client_secret="configured",
        )
        engine = ClaudeChatEngine(settings)
        engine._client = FakeAnthropicClient([paused, completed])

        turn = engine.respond_with_sports_search(
            "como le fue a la U de Chile en su ultimo partido",
            {
                "intent": "latest_result",
                "sport": "football",
                "participant": "Universidad de Chile",
                "country": "Chile",
            },
            SessionState(language="es"),
        )

        self.assertEqual(turn.tool_name, "information_web_result")
        self.assertIn("4-2", turn.text)
        self.assertEqual(len(engine._client.messages.requests), 2)
        continued_messages = engine._client.messages.requests[1]["messages"]
        self.assertEqual(continued_messages[-1]["role"], "assistant")
        self.assertNotIn("spotify_play", str(engine._client.messages.request["tools"]))

    def test_vague_sports_answer_is_retried_before_it_can_be_spoken(self) -> None:
        citation = SimpleNamespace(
            title="Argentina XV cayo sobre la hora 28-27 ante Chile XV",
            url="https://www.espn.cl/rugby/nota/example",
        )
        vague = SimpleNamespace(
            stop_reason="end_turn",
            content=[
                SimpleNamespace(type="server_tool_use", name="web_search"),
                SimpleNamespace(type="web_search_tool_result", content=[]),
                SimpleNamespace(
                    type="text",
                    text="He encontrado referencias a un partido reciente.",
                    citations=[citation],
                ),
            ],
        )
        corrected = SimpleNamespace(
            stop_reason="end_turn",
            content=[
                SimpleNamespace(type="server_tool_use", name="web_search"),
                SimpleNamespace(type="web_search_tool_result", content=[]),
                SimpleNamespace(
                    type="text",
                    text="Chile XV vencio 28-27 a Argentina XV.",
                    citations=[citation],
                ),
            ],
        )
        settings = SimpleNamespace(
            anthropic_api_key="test",
            anthropic_model="test-model",
            anthropic_max_output_tokens=250,
            persona_mode="canon",
            spotify_client_id="configured",
            spotify_client_secret="configured",
        )
        engine = ClaudeChatEngine(settings)
        engine._client = FakeAnthropicClient([vague, corrected])

        turn = engine.respond_with_sports_search(
            "cual fue el resultado del ultimo partido de rugby de Chile XV",
            {
                "intent": "latest_result",
                "sport": "rugby",
                "participant": "Chile XV",
                "competition": "Americas Rugby Championship",
                "country": "Chile",
                "date_reference": "ultimo partido",
            },
            SessionState(language="es"),
        )

        self.assertIn("Chile XV vencio 28-27 a Argentina XV", turn.text)
        self.assertNotIn("He encontrado referencias", turn.text)
        self.assertEqual(len(engine._client.messages.requests), 2)


if __name__ == "__main__":
    unittest.main()
