"""Tests for the main orchestration layer."""

from __future__ import annotations

import threading
import time
from types import SimpleNamespace
import unittest

from vadermind.contracts import ModelTurn, SessionState
from vadermind.orchestrator import VaderMindApp


class FakeEngine:
    """Small deterministic engine for local tests."""

    def __init__(self) -> None:
        """Track whether the fake engine was called."""

        self.calls = 0

    def respond(self, user_text: str, session_state: SessionState) -> str:
        """Return a predictable reply that depends on memory."""

        self.calls += 1
        return f"Recibido: {user_text}. Historial={len(session_state.messages)}"


class FakeTTS:
    """No-op test double."""

    def __init__(self, events: list[str] | None = None) -> None:
        self.calls: list[str] = []
        self.events = events

    def speak(self, text: str) -> None:
        """Keep the interface compatible without side effects."""

        self.calls.append(text)
        if self.events is not None:
            self.events.append(f"speak:{text}")


class FakeToolExecutor:
    """Deterministic tool executor for orchestration tests."""

    def __init__(self, events: list[str] | None = None) -> None:
        """Track the last executed tool call."""

        self.calls: list[tuple[str, dict[str, object]]] = []
        self.events = events

    def execute(self, tool_name: str, args: dict[str, object]) -> dict[str, object]:
        """Return a successful fake Spotify result."""

        self.calls.append((tool_name, args))
        if self.events is not None:
            self.events.append(f"execute:{tool_name}")
        if tool_name == "spotify.list_playlists":
            return {
                "ok": True,
                "tool_name": tool_name,
                "message": "Tus playlists visibles son: entrenamiento, trabajo.",
            }
        if tool_name == "spotify.search":
            return {
                "ok": True,
                "tool_name": tool_name,
                "message": "Encontre dos resultados.",
                "candidates": [
                    {"type": "album", "uri": "spotify:album:1", "name": "Uno", "subtitle": "Artista"},
                    {"type": "album", "uri": "spotify:album:2", "name": "Dos", "subtitle": "Artista"},
                ],
            }
        if tool_name == "spotify.delete_playlist" and not args.get("confirmed"):
            return {
                "ok": False,
                "confirmation_required": True,
                "tool_name": tool_name,
                "message": "Confirma la eliminacion.",
                "pending_action": {
                    "name": tool_name,
                    "args": {"confirmed": True, "uri": "spotify:playlist:1", "name": "Prueba"},
                },
            }
        result = {
            "ok": True,
            "tool_name": tool_name,
            "message": f"Reproduciendo {args.get('query')} en FRANCISCO.",
        }
        if tool_name == "spotify.play":
            result["explanation"] = "Elegi un artista exacto y use su contexto."
        return result


class FakeModelToolExecutor(FakeToolExecutor):
    supports_model_tools = True


class FakeLgTvExecutor(FakeToolExecutor):
    supports_model_tools = True
    supports_lg_tv_tools = True

    def execute(self, tool_name: str, args: dict[str, object]) -> dict[str, object]:
        self.calls.append((tool_name, args))
        return {
            "ok": True,
            "tool_name": tool_name,
            "message": "Volumen de la television ajustado a 21.",
        }


class FailingToolExecutor(FakeToolExecutor):
    def execute(self, tool_name: str, args: dict[str, object]) -> dict[str, object]:
        self.calls.append((tool_name, args))
        return {
            "ok": False,
            "tool_name": tool_name,
            "message": "Spotify no respondio.",
        }


class FakeToolAwareEngine(FakeEngine):
    def __init__(self, turn: ModelTurn) -> None:
        super().__init__()
        self.turn = turn
        self.tool_turns = 0

    def respond_with_tools(
        self, user_text: str, session_state: SessionState
    ) -> ModelTurn:
        self.tool_turns += 1
        return self.turn


class FakeInformationEngine(FakeEngine):
    def __init__(
        self, turn: ModelTurn, sports_search_turn: ModelTurn | None = None
    ) -> None:
        super().__init__()
        self.turn = turn
        self.sports_search_turn = sports_search_turn or ModelTurn(
            text="No deberia usarse", tool_name="information_web_result"
        )
        self.information_turns = 0
        self.sports_search_turns = 0

    def respond_with_information_tools(
        self, user_text: str, session_state: SessionState
    ) -> ModelTurn:
        self.information_turns += 1
        return self.turn

    def respond_with_sports_search(
        self,
        user_text: str,
        sports_query: dict[str, object],
        session_state: SessionState,
    ) -> ModelTurn:
        self.sports_search_turns += 1
        return self.sports_search_turn


class FakeCapabilityRouterEngine(FakeInformationEngine):
    def __init__(
        self,
        router_turn: ModelTurn,
        information_turn: ModelTurn | None = None,
        sports_search_turn: ModelTurn | None = None,
    ) -> None:
        super().__init__(
            information_turn or ModelTurn(text="No deberia usarse"),
            sports_search_turn,
        )
        self.router_turn = router_turn
        self.tool_turns = 0
        self.web_search_turns = 0

    def respond_with_tools(
        self, user_text: str, session_state: SessionState
    ) -> ModelTurn:
        self.tool_turns += 1
        return self.router_turn

    def respond_with_web_search(
        self, user_text: str, session_state: SessionState
    ) -> ModelTurn:
        self.web_search_turns += 1
        return self.turn


class FakeInformationExecutor:
    supports_information_tools = True

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, object]]] = []

    def execute(self, tool_name: str, args: dict[str, object]) -> dict[str, object]:
        self.calls.append((tool_name, args))
        if tool_name in {
            "information.sports_match_page",
            "information.latest_universidad_de_chile_match",
            "information.latest_chilean_team_match",
        }:
            if args.get("team_slug") == "universidad-catolica":
                return {
                    "ok": True,
                    "tool_name": tool_name,
                    "message": (
                        "Universidad Catolica vencio 1-0 a Everton.\n\n"
                        "Fuente: Campeonato Chileno — https://www.campeonatochileno.cl/"
                        "match/everton-universidad-catolica-2026-09-05/"
                    ),
                    "spoken_message": "Universidad Catolica vencio 1-0 a Everton.",
                }
            source_url = str(
                args.get(
                    "url",
                    "https://www.campeonatochileno.cl/match/"
                    "universidad-de-chile-coquimbo-unido-2026-09-05/",
                )
            )
            return {
                "ok": True,
                "tool_name": tool_name,
                "message": (
                    "Universidad de Chile vencio 4-2 a Coquimbo Unido.\n\n"
                    f"Fuente: Campeonato Chileno — {source_url}"
                ),
                "spoken_message": "Universidad de Chile vencio 4-2 a Coquimbo Unido.",
            }
        return {
            "ok": True,
            "tool_name": tool_name,
            "message": "Ahora en Santiago: 18 grados.\n\nFuente: Open-Meteo — https://open-meteo.com/",
            "spoken_message": "Ahora en Santiago: 18 grados.",
        }


class OrchestratorTests(unittest.TestCase):
    """Verify that the CLI app updates state correctly."""

    def test_process_turn_updates_memory_and_returns_text(self) -> None:
        """Processing a turn should append user and assistant messages."""

        session_state = SessionState(language="es")
        app = VaderMindApp(engine=FakeEngine(), tts=FakeTTS(), session_state=session_state)

        answer = app.process_turn("Hola Vader")

        self.assertIn("Hola Vader", answer)
        self.assertEqual(session_state.total_turns, 1)
        self.assertEqual(len(session_state.messages), 2)
        self.assertEqual(session_state.messages[0]["role"], "user")
        self.assertEqual(session_state.messages[1]["role"], "assistant")

    def test_process_turn_rejects_empty_input(self) -> None:
        """Blank input should fail early."""

        session_state = SessionState(language="es")
        app = VaderMindApp(engine=FakeEngine(), tts=FakeTTS(), session_state=session_state)

        with self.assertRaises(ValueError):
            app.process_turn("   ")

    def test_process_turn_routes_clear_spotify_command_to_tool(self) -> None:
        """A Spotify command should run the tool instead of the chat engine."""

        session_state = SessionState(language="es")
        engine = FakeEngine()
        tool_executor = FakeToolExecutor()
        app = VaderMindApp(
            engine=engine,
            tts=FakeTTS(),
            session_state=session_state,
            tool_executor=tool_executor,
        )

        answer = app.process_turn("Vader, pon Imperial March en Spotify")

        self.assertEqual(engine.calls, 0)
        self.assertEqual(len(tool_executor.calls), 1)
        self.assertEqual(tool_executor.calls[0][0], "spotify.play")
        self.assertEqual(tool_executor.calls[0][1]["query"], "Imperial March")
        self.assertIn("Hecho", answer)
        self.assertEqual(session_state.total_turns, 1)

    def test_process_turn_routes_spotify_playlist_list_to_tool(self) -> None:
        """A playlist list request should run the Spotify playlist tool."""

        session_state = SessionState(language="es")
        engine = FakeEngine()
        tool_executor = FakeToolExecutor()
        app = VaderMindApp(
            engine=engine,
            tts=FakeTTS(),
            session_state=session_state,
            tool_executor=tool_executor,
        )

        answer = app.process_turn("Vader, muestra mis playlists de Spotify")

        self.assertEqual(engine.calls, 0)
        self.assertEqual(tool_executor.calls[0][0], "spotify.list_playlists")
        self.assertIn("playlists visibles", answer)

    def test_followup_plays_second_previous_search_result(self) -> None:
        """An ordinal follow-up should resolve against results saved in session context."""

        state = SessionState(language="es")
        executor = FakeToolExecutor()
        app = VaderMindApp(FakeEngine(), FakeTTS(), state, executor)

        app.process_turn("busca el album Discovery en Spotify")
        answer = app.process_turn("ahora pon la segunda")

        self.assertEqual(executor.calls[-1][0], "spotify.play_item")
        self.assertEqual(executor.calls[-1][1]["item"]["name"], "Dos")
        self.assertIn("Hecho", answer)

    def test_explain_last_spotify_choice_is_local_and_truthful(self) -> None:
        """Why-followups should use recorded tool facts instead of asking Claude."""

        state = SessionState(language="es")
        engine = FakeEngine()
        executor = FakeToolExecutor()
        app = VaderMindApp(engine, FakeTTS(), state, executor)

        app.process_turn("pon Juan Gabriel en Spotify")
        answer = app.process_turn("por que pusiste esa cancion?")

        self.assertEqual(engine.calls, 0)
        self.assertEqual(len(executor.calls), 1)
        self.assertEqual(answer, "Elegi un artista exacto y use su contexto.")

    def test_playlist_delete_requires_confirmation(self) -> None:
        """Playlist removal should execute only after a separate confirmation turn."""

        state = SessionState(language="es")
        executor = FakeToolExecutor()
        app = VaderMindApp(FakeEngine(), FakeTTS(), state, executor)

        prompt = app.process_turn("elimina mi playlist Prueba")
        self.assertEqual(len(executor.calls), 1)
        self.assertIn("Confirma", prompt)

        answer = app.process_turn("si, eliminar")
        self.assertEqual(len(executor.calls), 2)
        self.assertTrue(executor.calls[-1][1]["confirmed"])
        self.assertIn("Hecho", answer)

    def test_flexible_spotify_language_is_interpreted_by_model(self) -> None:
        """Non-fast playback language should use the model's structured decision."""

        state = SessionState(language="es")
        engine = FakeToolAwareEngine(
            ModelTurn(
                tool_name="spotify_play",
                tool_args={
                    "query": "Jazz",
                    "item_type": "playlist",
                    "library_only": True,
                    "shuffle": True,
                },
            )
        )
        executor = FakeModelToolExecutor()
        app = VaderMindApp(engine, FakeTTS(), state, executor)

        answer = app.process_turn("quiero escuchar la lista de jazz que tengo, pero revuelta")

        self.assertEqual(engine.tool_turns, 1)
        self.assertEqual(executor.calls[0][0], "spotify.play_playlist")
        self.assertEqual(
            executor.calls[0][1], {"query": "Jazz", "shuffle": True}
        )
        self.assertIn("Hecho", answer)

    def test_model_routes_lg_tv_through_its_local_executor(self) -> None:
        state = SessionState(language="es")
        engine = FakeToolAwareEngine(
            ModelTurn(
                tool_name="lg_tv_volume",
                tool_args={"action": "set", "volume": 21},
            )
        )
        executor = FakeLgTvExecutor()
        app = VaderMindApp(engine, FakeTTS(), state, executor)

        answer = app.process_turn("deja la tele a un volumen comodo, como 21")

        self.assertEqual(engine.tool_turns, 1)
        self.assertEqual(
            executor.calls,
            [("lg_tv.set_volume", {"volume": 21})],
        )
        self.assertIn("Hecho", answer)

    def test_explicit_lg_tv_command_uses_fast_local_route(self) -> None:
        state = SessionState(language="es")
        engine = FakeToolAwareEngine(ModelTurn(text="No deberia usarse"))
        executor = FakeLgTvExecutor()
        app = VaderMindApp(engine, FakeTTS(), state, executor)

        answer = app.process_turn("pon el volumen de la tele en 21")

        self.assertEqual(engine.tool_turns, 0)
        self.assertEqual(
            executor.calls,
            [("lg_tv.set_volume", {"volume": 21})],
        )
        self.assertIn("Hecho", answer)

    def test_model_routes_second_layer_lg_tv_action(self) -> None:
        state = SessionState(language="es")
        engine = FakeToolAwareEngine(
            ModelTurn(
                tool_name="lg_tv_navigation",
                tool_args={"action": "home"},
            )
        )
        executor = FakeLgTvExecutor()
        app = VaderMindApp(engine, FakeTTS(), state, executor)

        answer = app.process_turn("quiero volver a la pantalla principal del televisor")

        self.assertEqual(
            executor.calls,
            [("lg_tv.navigate", {"action": "home"})],
        )
        self.assertIn("Hecho", answer)

    def test_explicit_tv_message_uses_fast_local_route(self) -> None:
        state = SessionState(language="es")
        engine = FakeToolAwareEngine(ModelTurn(text="No deberia usarse"))
        executor = FakeLgTvExecutor()
        app = VaderMindApp(engine, FakeTTS(), state, executor)

        answer = app.process_turn("muestra en la tele el mensaje La cena esta lista")

        self.assertEqual(engine.tool_turns, 0)
        self.assertEqual(
            executor.calls,
            [("lg_tv.show_message", {"message": "La cena esta lista"})],
        )
        self.assertIn("Hecho", answer)

    def test_model_text_response_is_reused_without_second_call(self) -> None:
        state = SessionState(language="es")
        engine = FakeToolAwareEngine(ModelTurn(text="Conversacion normal."))
        app = VaderMindApp(engine, FakeTTS(), state, FakeModelToolExecutor())

        answer = app.process_turn("como estas hoy?")

        self.assertEqual(answer, "Conversacion normal.")
        self.assertEqual(engine.tool_turns, 1)
        self.assertEqual(engine.calls, 0)

    def test_model_router_defers_an_unlisted_sport_without_searching(self) -> None:
        engine = FakeCapabilityRouterEngine(
            ModelTurn(
                tool_name="information_deferred",
                tool_args={
                    "query": "cruces de hoy de Premier Padel Paris",
                    "category": "sports",
                },
            )
        )
        spotify = FakeToolExecutor()
        information = FakeInformationExecutor()
        app = VaderMindApp(
            engine,
            FakeTTS(),
            SessionState(language="es"),
            spotify,
            information_executor=information,
        )

        answer = app.process_turn(
            "dime los cruces de hoy del torneo Premier Padel de Paris"
        )

        self.assertIn("deshabilitada por ahora", answer)
        self.assertIn("hora y el clima", answer)
        self.assertEqual(engine.tool_turns, 1)
        self.assertEqual(engine.information_turns, 0)
        self.assertEqual(engine.sports_search_turns, 0)
        self.assertEqual(engine.web_search_turns, 0)
        self.assertEqual(spotify.calls, [])
        self.assertEqual(information.calls, [])

    def test_model_router_disables_sports_even_with_legacy_keywords(self) -> None:
        engine = FakeCapabilityRouterEngine(
            ModelTurn(
                tool_name="information_deferred",
                tool_args={
                    "query": "competencias de hoy en los Juegos Olimpicos",
                    "category": "sports",
                },
            )
        )
        app = VaderMindApp(
            engine,
            FakeTTS(),
            SessionState(language="es"),
            FakeToolExecutor(),
            information_executor=FakeInformationExecutor(),
        )

        answer = app.process_turn("quienes juegan hoy en el US Open?")

        self.assertIn("deshabilitada por ahora", answer)
        self.assertEqual(engine.tool_turns, 0)
        self.assertEqual(engine.information_turns, 0)
        self.assertEqual(engine.sports_search_turns, 0)
        self.assertEqual(engine.web_search_turns, 0)

    def test_model_router_defers_general_current_information(self) -> None:
        engine = FakeCapabilityRouterEngine(
            ModelTurn(
                tool_name="information_deferred",
                tool_args={
                    "query": "valor vigente del abono mensual RED en Santiago",
                    "category": "current_fact",
                },
            )
        )
        spotify = FakeToolExecutor()
        information = FakeInformationExecutor()
        app = VaderMindApp(
            engine,
            FakeTTS(),
            SessionState(language="es"),
            spotify,
            information_executor=information,
        )

        answer = app.process_turn(
            "necesito el valor vigente del abono mensual RED en Santiago"
        )

        self.assertIn("deshabilitada por ahora", answer)
        self.assertEqual(engine.tool_turns, 1)
        self.assertEqual(engine.information_turns, 0)
        self.assertEqual(engine.web_search_turns, 0)
        self.assertEqual(spotify.calls, [])
        self.assertEqual(information.calls, [])

    def test_model_router_keeps_time_and_weather_enabled(self) -> None:
        for tool_name, tool_args, expected_internal_name in (
            (
                "information_time",
                {"timezone": "America/Santiago", "location": "Santiago"},
                "information.time",
            ),
            (
                "information_weather",
                {"location": "Santiago", "when": "today"},
                "information.weather",
            ),
        ):
            with self.subTest(tool_name=tool_name):
                engine = FakeCapabilityRouterEngine(
                    ModelTurn(tool_name=tool_name, tool_args=tool_args)
                )
                information = FakeInformationExecutor()
                app = VaderMindApp(
                    engine,
                    FakeTTS(),
                    SessionState(language="es"),
                    FakeToolExecutor(),
                    information_executor=information,
                )

                answer = app.process_turn("consulta permitida")

                self.assertIn("18 grados", answer)
                self.assertEqual(information.calls[0][0], expected_internal_name)
                self.assertEqual(engine.web_search_turns, 0)
                self.assertEqual(engine.sports_search_turns, 0)

    def test_time_place_is_kept_only_if_the_user_named_it(self) -> None:
        for user_text, expected_args in (
            ("¿Y en Madrid?", {"location": "Madrid"}),
            ("¿Qué hora es?", {}),  # the model guessed a city: local clock instead
        ):
            with self.subTest(user_text=user_text):
                engine = FakeCapabilityRouterEngine(
                    ModelTurn(tool_name="information_time", tool_args={"location": "Madrid"})
                )
                information = FakeInformationExecutor()
                app = VaderMindApp(
                    engine,
                    FakeTTS(),
                    SessionState(language="es"),
                    FakeToolExecutor(),
                    information_executor=information,
                )
                app.process_turn(user_text)
                self.assertEqual(information.calls[0], ("information.time", expected_args))

    def test_model_router_blocks_unlisted_external_action_wording(self) -> None:
        engine = FakeCapabilityRouterEngine(
            ModelTurn(
                tool_name="external_action_blocked",
                tool_args={"action": "purchase"},
            )
        )
        spotify = FakeToolExecutor()
        information = FakeInformationExecutor()
        app = VaderMindApp(
            engine,
            FakeTTS(),
            SessionState(language="es"),
            spotify,
            information_executor=information,
        )

        answer = app.process_turn("encarga por la web dos entradas para mañana")

        self.assertIn("no puedo comprar", answer)
        self.assertEqual(engine.tool_turns, 1)
        self.assertEqual(engine.information_turns, 0)
        self.assertEqual(engine.sports_search_turns, 0)
        self.assertEqual(spotify.calls, [])
        self.assertEqual(information.calls, [])

    def test_unknown_model_tool_is_rejected_before_executor(self) -> None:
        state = SessionState(language="es")
        engine = FakeToolAwareEngine(
            ModelTurn(tool_name="spotify_destroy_account", tool_args={})
        )
        executor = FakeModelToolExecutor()
        app = VaderMindApp(engine, FakeTTS(), state, executor)

        answer = app.process_turn("haz algo no permitido")

        self.assertIn("no permitida", answer)
        self.assertEqual(executor.calls, [])

    def test_voice_acknowledgement_happens_before_spotify_playback(self) -> None:
        """Piper should finish the short acknowledgement before Spotify executes."""

        events: list[str] = []
        tts = FakeTTS(events)
        executor = FakeToolExecutor(events)
        state = SessionState(language="es")
        app = VaderMindApp(
            FakeEngine(), tts, state, executor, voice_enabled=True
        )

        answer = app.process_turn("pon Juan Gabriel en Spotify")

        self.assertEqual(
            events,
            [
                "speak:Reproduciendo Juan Gabriel en Spotify.",
                "execute:spotify.play",
            ],
        )
        self.assertEqual(answer, "Reproduciendo Juan Gabriel en Spotify.")
        self.assertTrue(app.skip_post_turn_speech)
        self.assertIn("Hecho", state.messages[-1]["content"])

    def test_spotify_lookups_overlap_the_confirmation_but_action_waits(self) -> None:
        """Prefetch runs while Vader speaks; execute still starts after speech."""

        events: list[str] = []
        speaking = threading.Event()
        prefetched_while_speaking: list[bool] = []

        class SlowTTS(FakeTTS):
            def speak(self, text: str) -> None:
                speaking.set()
                time.sleep(0.05)
                super().speak(text)

        class PrefetchingExecutor(FakeToolExecutor):
            def prefetch(self, tool_name: str, args: dict[str, object]) -> None:
                prefetched_while_speaking.append(speaking.wait(1.0))
                events.append(f"prefetch:{tool_name}")

        app = VaderMindApp(
            FakeEngine(),
            SlowTTS(events),
            SessionState(language="es"),
            PrefetchingExecutor(events),
            voice_enabled=True,
        )
        app.process_turn("pon Juan Gabriel en Spotify")

        self.assertEqual(prefetched_while_speaking, [True])
        self.assertEqual(events[-1], "execute:spotify.play")
        self.assertLess(events.index("prefetch:spotify.play"), events.index("execute:spotify.play"))
        self.assertLess(
            events.index("speak:Reproduciendo Juan Gabriel en Spotify."),
            events.index("execute:spotify.play"),
        )

    def test_failed_prefetch_does_not_break_the_command(self) -> None:
        class BrokenPrefetchExecutor(FakeToolExecutor):
            def prefetch(self, tool_name: str, args: dict[str, object]) -> None:
                raise RuntimeError("red caida")

        executor = BrokenPrefetchExecutor()
        app = VaderMindApp(
            FakeEngine(), FakeTTS(), SessionState(language="es"), executor, voice_enabled=True
        )
        app.process_turn("pon Juan Gabriel en Spotify")
        self.assertEqual(executor.calls[0][0], "spotify.play")

    def test_voice_tv_spotify_only_speaks_before_the_long_action(self) -> None:
        events: list[str] = []
        tts = FakeTTS(events)
        executor = FakeToolExecutor(events)
        state = SessionState(language="es")
        app = VaderMindApp(
            FakeEngine(), tts, state, executor, voice_enabled=True
        )

        answer = app.process_turn("pon Dua Lipa en Spotify en la tele")

        self.assertEqual(
            events,
            [
                "speak:Preparando Dua Lipa en la television.",
                "execute:media.spotify_on_tv",
            ],
        )
        self.assertEqual(answer, "Preparando Dua Lipa en la television.")
        self.assertTrue(app.skip_post_turn_speech)
        self.assertIn("Hecho", state.messages[-1]["content"])

    def test_voice_pause_is_successful_and_completely_silent(self) -> None:
        tts = FakeTTS()
        executor = FakeToolExecutor()
        app = VaderMindApp(
            FakeEngine(),
            tts,
            SessionState(language="es"),
            executor,
            voice_enabled=True,
        )

        answer = app.process_turn("pausa la musica")

        self.assertEqual(answer, "")
        self.assertEqual(tts.calls, [])
        self.assertEqual(executor.calls[0][0], "spotify.pause")
        self.assertTrue(app.skip_post_turn_speech)

    def test_voice_resume_phrase_executes_after_brief_acknowledgement(self) -> None:
        """The reported phrase must execute locally instead of becoming model text."""

        events: list[str] = []
        engine = FakeToolAwareEngine(
            ModelTurn(text="Hecho. Reanudando reproduccion en FRANCISCO.")
        )
        executor = FakeModelToolExecutor(events)
        app = VaderMindApp(
            engine,
            FakeTTS(events),
            SessionState(language="es"),
            executor,
            voice_enabled=True,
        )

        answer = app.process_turn("reanuda la musica")

        self.assertEqual(events, ["speak:Reanudando.", "execute:spotify.resume"])
        self.assertEqual(executor.calls[0][0], "spotify.resume")
        self.assertEqual(engine.tool_turns, 0)
        self.assertEqual(answer, "Reanudando.")
        self.assertTrue(app.skip_post_turn_speech)

    def test_spotify_read_query_is_spoken_after_execution(self) -> None:
        tts = FakeTTS()
        executor = FakeToolExecutor()
        app = VaderMindApp(
            FakeEngine(),
            tts,
            SessionState(language="es"),
            executor,
            voice_enabled=True,
        )

        answer = app.process_turn("que esta sonando")

        self.assertNotEqual(answer, "")
        self.assertEqual(tts.calls, [])
        self.assertFalse(app.skip_post_turn_speech)

    def test_spotify_failure_is_not_suppressed_after_pre_acknowledgement(self) -> None:
        tts = FakeTTS()
        app = VaderMindApp(
            FakeEngine(),
            tts,
            SessionState(language="es"),
            FailingToolExecutor(),
            voice_enabled=True,
        )

        answer = app.process_turn("siguiente cancion")

        self.assertEqual(tts.calls, ["Hecho."])
        self.assertIn("Spotify no respondio", answer)
        self.assertFalse(app.skip_post_turn_speech)

    def test_weather_turn_uses_only_information_executor(self) -> None:
        engine = FakeInformationEngine(
            ModelTurn(
                tool_name="information_weather",
                tool_args={"location": "Santiago", "when": "today"},
            )
        )
        information = FakeInformationExecutor()
        spotify = FakeToolExecutor()
        app = VaderMindApp(
            engine,
            FakeTTS(),
            SessionState(language="es"),
            spotify,
            information_executor=information,
            voice_enabled=True,
        )

        answer = app.process_turn("que clima hace en Santiago")

        self.assertIn("Open-Meteo", answer)
        self.assertEqual(information.calls[0][0], "information.weather")
        self.assertEqual(spotify.calls, [])
        self.assertEqual(app.spoken_answer, "Ahora en Santiago: 18 grados.")

    def test_web_result_is_returned_without_exposing_spotify(self) -> None:
        engine = FakeInformationEngine(
            ModelTurn(
                text="El producto cuesta aproximadamente 100.\n\nFuentes consultadas:\n- Tienda: https://example.com",
                tool_name="information_web_result",
            )
        )
        spotify = FakeToolExecutor()
        app = VaderMindApp(
            engine,
            FakeTTS(),
            SessionState(language="es"),
            spotify,
            information_executor=FakeInformationExecutor(),
            expanded_web_enabled=True,
        )

        answer = app.process_turn("cuanto cuesta ese producto")

        self.assertIn("example.com", answer)
        self.assertEqual(spotify.calls, [])
        self.assertEqual(app.spoken_answer, "El producto cuesta aproximadamente 100.")

    def test_sports_result_removes_scorers_and_personal_commentary(self) -> None:
        engine = FakeInformationEngine(
            ModelTurn(
                tool_name="information_sports_query",
                tool_args={
                    "intent": "latest_result",
                    "sport": "basketball",
                    "participant": "Los Angeles Lakers",
                    "competition": "NBA",
                    "country": "Estados Unidos",
                    "date_reference": "ultimo partido",
                },
            ),
            ModelTurn(
                text=(
                    "He encontrado referencias a un partido reciente."
                    "Lakers vencio 112-106 a Boston. Boston marco primero y Lakers "
                    "empato al final. La determinacion fue admirable. Tambien puedo poner "
                    "musica para celebrarlo.\n\n"
                    "Fuentes consultadas:\n- Marcador: "
                    "https://www.nba.com/game/example"
                ),
                tool_name="information_web_result",
            ),
        )
        spotify = FakeToolExecutor()
        app = VaderMindApp(
            engine,
            FakeTTS(),
            SessionState(language="es"),
            spotify,
            information_executor=FakeInformationExecutor(),
            expanded_web_enabled=True,
        )

        answer = app.process_turn(
            "como le fue a los Lakers en su ultimo partido"
        )

        self.assertIn("Lakers vencio 112-106 a Boston.", answer)
        self.assertNotIn("He encontrado referencias", answer)
        self.assertIn("nba.com", answer)
        self.assertNotIn("marco primero", answer)
        self.assertNotIn("determinacion", answer)
        self.assertNotIn("musica", answer)
        self.assertEqual(
            app.spoken_answer, "Lakers vencio 112-106 a Boston."
        )
        self.assertEqual(engine.information_turns, 1)
        self.assertEqual(engine.sports_search_turns, 1)
        self.assertEqual(spotify.calls, [])

    def test_us_open_schedule_keeps_all_matchups_and_drops_persona_commentary(self) -> None:
        engine = FakeInformationEngine(
            ModelTurn(
                tool_name="information_sports_query",
                tool_args={
                    "intent": "schedule",
                    "sport": "tennis",
                    "event": "US Open",
                    "competition": "US Open",
                    "date_reference": "hoy, cuartos de final",
                },
            ),
            ModelTurn(
                text=(
                    "Cuadro masculino: Carlos Alcaraz vs Jannik Sinner.\n"
                    "Cuadro femenino: Coco Gauff vs Aryna Sabalenka.\n"
                    "ATP cuartos: Carlos Alcaraz vs Jannik Sinner.\n"
                    "WTA cuartos: Coco Gauff vs Aryna Sabalenka.\n"
                    "Arthur Ashe Stadium comienza a las 11:30 con Gauff vs.\n"
                    "Resumen: Alcaraz contra Sinner y Gauff contra Sabalenka.\n"
                    "Una jornada digna del poder del lado oscuro. Tambien puedo poner "
                    "musica para acompañarla.\n\n"
                    "Fuentes consultadas:\n- US Open: "
                    "https://www.usopen.org/en_US/scores/schedule/"
                ),
                tool_name="information_web_result",
            ),
        )
        spotify = FakeToolExecutor()
        app = VaderMindApp(
            engine,
            FakeTTS(),
            SessionState(language="es"),
            spotify,
            information_executor=FakeInformationExecutor(),
            expanded_web_enabled=True,
        )

        answer = app.process_turn(
            "quienes juegan los cuartos de final hoy en el US Open?"
        )

        self.assertIn("Carlos Alcaraz vs Jannik Sinner", answer)
        self.assertIn("Coco Gauff vs Aryna Sabalenka", answer)
        self.assertEqual(answer.count("Carlos Alcaraz vs Jannik Sinner"), 1)
        self.assertEqual(answer.count("Coco Gauff vs Aryna Sabalenka"), 1)
        self.assertNotIn("lado oscuro", answer)
        self.assertNotIn("Arthur Ashe Stadium", answer)
        self.assertNotIn("Resumen", answer)
        self.assertNotIn("musica", answer)
        self.assertIn("usopen.org", answer)
        self.assertEqual(engine.calls, 0)
        self.assertEqual(engine.information_turns, 1)
        self.assertEqual(engine.sports_search_turns, 1)
        self.assertEqual(spotify.calls, [])
        self.assertIn("Carlos Alcaraz vs Jannik Sinner", app.spoken_answer)
        self.assertIn("Coco Gauff vs Aryna Sabalenka", app.spoken_answer)
        self.assertNotIn("usopen.org", app.spoken_answer)

    def test_u_catolica_does_not_reuse_u_de_chile_result(self) -> None:
        engine = FakeInformationEngine(
            ModelTurn(
                tool_name="information_sports_query",
                tool_args={
                    "intent": "latest_result",
                    "sport": "football",
                    "participant": "Universidad Catolica",
                    "competition": "Liga de Primera Chile",
                    "country": "Chile",
                    "date_reference": "ultimo partido",
                },
            )
        )
        information = FakeInformationExecutor()
        spotify = FakeToolExecutor()
        app = VaderMindApp(
            engine,
            FakeTTS(),
            SessionState(language="es"),
            spotify,
            information_executor=information,
            expanded_web_enabled=True,
        )

        answer = app.process_turn(
            "y como le fue a la U catolica en su ultimo partido?"
        )

        self.assertIn("Universidad Catolica vencio 1-0 a Everton", answer)
        self.assertNotIn("Coquimbo Unido", answer)
        self.assertEqual(
            information.calls[0][1]["team_slug"], "universidad-catolica"
        )
        self.assertEqual(engine.information_turns, 1)
        self.assertEqual(engine.sports_search_turns, 0)
        self.assertEqual(spotify.calls, [])

    def test_athletics_result_removes_unsolicited_historical_trivia(self) -> None:
        engine = FakeInformationEngine(
            ModelTurn(
                tool_name="information_sports_query",
                tool_args={
                    "intent": "event_result",
                    "sport": "athletics",
                    "event": "100 metros masculinos",
                    "competition": "Campeonato Mundial de Atletismo 2025",
                    "date_reference": "ultima edicion",
                },
            ),
            ModelTurn(
                text=(
                    "Oblique Seville gano los 100 metros en 9.77 segundos, "
                    "convirtiendose en el primer campeon jamaicano desde 2015.\n\n"
                    "Fuentes consultadas:\n- World Athletics: "
                    "https://worldathletics.org/example"
                ),
                tool_name="information_web_result",
            ),
        )
        app = VaderMindApp(
            engine,
            FakeTTS(),
            SessionState(language="es"),
            FakeToolExecutor(),
            information_executor=FakeInformationExecutor(),
            expanded_web_enabled=True,
        )

        answer = app.process_turn(
            "quien gano los 100 metros masculinos del mundial de atletismo"
        )

        self.assertIn("Oblique Seville gano los 100 metros en 9.77 segundos.", answer)
        self.assertNotIn("primer campeon", answer)
        self.assertIn("worldathletics.org", answer)

    def test_sports_result_strips_reference_wrapper_but_keeps_score(self) -> None:
        answer = VaderMindApp._concise_sports_answer(
            "He encontrado una referencia reciente que indica que Argentina XV "
            "cayo sobre la hora 28-27 ante Chile XV hace 2 dias.\n\n"
            "Fuentes consultadas:\n- ESPN: https://www.espn.cl/rugby/example"
        )

        self.assertTrue(answer.startswith("Argentina XV cayo sobre la hora 28-27"))
        self.assertNotIn("He encontrado", answer)
        self.assertNotIn("hace 2 dias", answer)
        self.assertIn("espn.cl", answer)

    def test_multisport_schedule_keeps_events_and_removes_commentary(self) -> None:
        answer = VaderMindApp._concise_sports_answer(
            "- Atletismo: 100 metros masculinos — 10:30\n"
            "- Natacion: 200 metros libres femeninos — 12:00\n"
            "- Atletismo: 100 metros masculinos — 10:30\n"
            "El Imperio observa la jornada.\n\n"
            "Fuentes consultadas:\n- Olympics: https://olympics.com/example",
            {"intent": "schedule", "sport": "other"},
        )

        self.assertIn("100 metros masculinos", answer)
        self.assertIn("200 metros libres femeninos", answer)
        self.assertEqual(answer.count("100 metros masculinos"), 1)
        self.assertNotIn("Imperio", answer)
        self.assertIn("olympics.com", answer)

        no_events = VaderMindApp._concise_sports_answer(
            "No hay Juegos Olimpicos en curso hoy. La proxima edicion sera en 2028. "
            "El Imperio no espera.\n\n"
            "Fuentes consultadas:\n- Olympics: https://olympics.com/example",
            {"intent": "schedule", "sport": "other"},
        )
        self.assertIn("No hay Juegos Olimpicos en curso hoy.", no_events)
        self.assertNotIn("proxima edicion", no_events)
        self.assertNotIn("Imperio", no_events)

    def test_latest_champions_winner_uses_sports_path_and_drops_persona_opinion(self) -> None:
        engine = FakeInformationEngine(
            ModelTurn(
                tool_name="information_sports_query",
                tool_args={
                    "intent": "event_result",
                    "sport": "football",
                    "event": "UEFA Champions League",
                    "competition": "UEFA Champions League",
                    "country": "Europa",
                    "date_reference": "ultimo campeon",
                },
            ),
            ModelTurn(
                text=(
                    "Paris Saint-Germain fue el campeon de la Champions League 2025 "
                    "tras vencer 4-3 al Arsenal en la final disputada en Budapest. "
                    "Una victoria inevitable. El campeon demostro el dominio que "
                    "caracteriza a los verdaderamente poderosos.\n\n"
                    "Fuentes consultadas:\n- UEFA: https://www.uefa.com/example"
                ),
                tool_name="information_web_result",
            ),
        )
        spotify = FakeToolExecutor()
        app = VaderMindApp(
            engine,
            FakeTTS(),
            SessionState(language="es"),
            spotify,
            information_executor=FakeInformationExecutor(),
            expanded_web_enabled=True,
        )

        answer = app.process_turn(
            "quien fue el ultimo ganado de la champions league"
        )

        self.assertIn("Paris Saint-Germain fue el campeon", answer)
        self.assertNotIn("victoria inevitable", answer)
        self.assertNotIn("verdaderamente poderosos", answer)
        self.assertIn("uefa.com", answer)
        self.assertEqual(
            app.spoken_answer,
            "Paris Saint-Germain fue el campeon de la Champions League 2025 tras vencer 4-3 al Arsenal.",
        )
        self.assertEqual(engine.calls, 0)
        self.assertEqual(engine.information_turns, 1)
        self.assertEqual(spotify.calls, [])

    def test_external_purchase_is_rejected_without_any_tool(self) -> None:
        engine = FakeInformationEngine(ModelTurn(text="No deberia usarse"))
        information = FakeInformationExecutor()
        spotify = FakeToolExecutor()
        app = VaderMindApp(
            engine,
            FakeTTS(),
            SessionState(language="es"),
            spotify,
            information_executor=information,
        )

        answer = app.process_turn("compra el segundo")

        self.assertIn("no puedo comprar", answer)
        self.assertEqual(information.calls, [])
        self.assertEqual(spotify.calls, [])
        self.assertEqual(engine.information_turns, 0)



class TvWarmUpTests(unittest.TestCase):
    def test_wake_warm_up_targets_only_the_tv_executor(self) -> None:
        from vadermind.orchestrator import _tv_warm_up
        from vadermind.tools.base import RoutedToolsExecutor

        warmed: list[str] = []

        class WarmTv(FakeLgTvExecutor):
            def warm(self) -> None:
                warmed.append("tv")

        app = SimpleNamespace(
            tool_executor=RoutedToolsExecutor({"lg_tv.": WarmTv(), "spotify.": FakeToolExecutor()})
        )
        warm = _tv_warm_up(app)
        self.assertIsNotNone(warm)
        warm()
        self.assertEqual(warmed, ["tv"])

    def test_no_tv_means_no_wake_warm_up(self) -> None:
        from vadermind.orchestrator import _tv_warm_up

        self.assertIsNone(_tv_warm_up(SimpleNamespace(tool_executor=FakeToolExecutor())))


if __name__ == "__main__":
    unittest.main()
