"""CLI orchestration for VaderMind."""

from __future__ import annotations

import argparse
from datetime import date
import logging
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import threading
from typing import Callable

from vadermind.brain import (
    BrainError,
    ClaudeChatEngine,
    HybridChatEngine,
    OpenRouterChatEngine,
    VercelGatewayChatEngine,
)
from vadermind.config import ConfigurationError, load_settings, validate_settings
from vadermind.conversation import append_assistant_message, append_user_message
from vadermind.contracts import ChatEngine, SessionState, TextToSpeech, ToolExecutor
from vadermind.hands_free import HandsFreeSession
from vadermind.lights import ERROR, ESCUCHANDO, PENSANDO, REPOSO, Lights, NullLights
from vadermind.lights import build_lights, speech_envelope
from vadermind.listening import ListeningError, VoiceListener, build_voice_listener
from vadermind.response_text import clean_assistant_text
from vadermind.telemetry import STARTUP, TURN_TIMELINE, configure_logging, timed_operation
from vadermind.tools.base import NoToolsExecutor, RoutedToolsExecutor
from vadermind.tools.catalog import (
    resolve_capability_tool_call,
    resolve_information_tool_call,
)
from vadermind.tools.information import (
    chilean_team_slug,
    InformationError,
    InformationToolsExecutor,
    is_disallowed_external_action_request,
    is_read_only_information_request,
    is_sports_result_request,
    is_time_or_weather_request,
    is_universidad_de_chile_request,
    place_mentioned,
)
from vadermind.tools.lg_tv import LgTvToolsExecutor
from vadermind.tools.media import MediaToolsExecutor
from vadermind.tools.router import ToolCall, parse_tool_call
from vadermind.tools.spotify import SpotifyClient, SpotifyError, SpotifyOAuth, SpotifyToolsExecutor
from vadermind.network_wait import wait_for_network
from vadermind.profiles import ProfileChoice, profile_after_network, remember_profile, resolve_profile
from vadermind.warmup import in_background
from vadermind.voice_output import (
    ElevenLabsTextToSpeech,
    OpenAITextToSpeech,
    PiperTextToSpeech,
    SilentTextToSpeech,
    VoiceError,
    add_speech_observer,
)
from vadermind.wake_word import build_wake_word, wait_for_wake_word

LOGGER = logging.getLogger("vadermind.cli")
EXIT_COMMANDS = {"salir", "exit", "quit"}
FAST_LOCAL_TOOL_NAMES = {
    "spotify.pause",
    "spotify.resume",
    "spotify.next",
    "spotify.previous",
    "spotify.current",
    "spotify.volume",
    "spotify.shuffle",
    "spotify.repeat",
    "spotify.save_current",
    "spotify.list_playlists",
    "spotify.play_reference",
    "spotify.play_result",
    "spotify.play_similar",
    "spotify.explain_last",
    "spotify.confirm",
    "spotify.cancel",
    "lg_tv.status",
    "lg_tv.power_on",
    "lg_tv.power_off",
    "lg_tv.set_volume",
    "lg_tv.volume_up",
    "lg_tv.volume_down",
    "lg_tv.set_mute",
    "lg_tv.launch_app",
    "lg_tv.select_input",
    "lg_tv.playback",
    "lg_tv.navigate",
    "lg_tv.channel_up",
    "lg_tv.channel_down",
    "lg_tv.select_channel",
    "lg_tv.screen_on",
    "lg_tv.screen_off",
    "lg_tv.close_app",
    "lg_tv.show_message",
    "lg_tv.schedule_power_off",
    "lg_tv.schedule_message",
    "lg_tv.cancel_timers",
    "lg_tv.timer_status",
    "media.spotify_on_tv",
}


def _safe_print(text: str = "") -> None:
    """Print text even when the Windows console cannot encode some characters."""

    encoding = sys.stdout.encoding or "utf-8"
    safe_text = text.encode(encoding, errors="replace").decode(encoding)
    print(safe_text)


class VaderMindApp:
    """Coordinates session state, text generation, and optional voice output."""

    def __init__(
        self,
        engine: ChatEngine,
        tts: TextToSpeech,
        session_state: SessionState,
        tool_executor: ToolExecutor | None = None,
        information_executor: ToolExecutor | None = None,
        voice_enabled: bool = False,
        expanded_web_enabled: bool = False,
    ) -> None:
        """Store the collaborators used by the interactive loop."""

        self.engine = engine
        self.tts = tts
        self.session_state = session_state
        self.tool_executor = tool_executor
        self.information_executor = information_executor
        self.voice_enabled = voice_enabled
        self.expanded_web_enabled = expanded_web_enabled
        self.skip_post_turn_speech = False
        self.spoken_answer: str | None = None

    def process_turn(self, user_text: str) -> str:
        """Process a single user turn and update session memory."""

        cleaned_text = user_text.strip()
        if not cleaned_text:
            raise ValueError("No recibi ningun texto para procesar.")

        self.skip_post_turn_speech = False
        self.spoken_answer = None
        append_user_message(self.session_state, cleaned_text)
        self._warm_information_provider(cleaned_text)

        if is_disallowed_external_action_request(cleaned_text):
            answer = (
                "Puedo consultar informacion publica y mostrarte fuentes, pero no puedo "
                "comprar, reservar, iniciar sesion, crear cuentas ni ejecutar acciones en internet."
            )
            append_assistant_message(self.session_state, answer)
            return answer

        model_tool_method = getattr(self.engine, "respond_with_tools", None)
        generic_model_tools = bool(
            self.tool_executor
            and getattr(self.tool_executor, "supports_model_tools", False)
        )
        lg_tv_model_enabled = bool(
            self.tool_executor
            and getattr(self.tool_executor, "supports_lg_tv_tools", False)
        )
        spotify_model_enabled = bool(
            self.tool_executor
            and (
                getattr(self.tool_executor, "supports_spotify_tools", False)
                or (generic_model_tools and not lg_tv_model_enabled)
            )
        )
        information_model_enabled = bool(
            self.information_executor
            and getattr(self.information_executor, "supports_information_tools", False)
        )
        model_router_enabled = callable(model_tool_method) and (
            spotify_model_enabled or lg_tv_model_enabled or information_model_enabled
        )
        information_context = isinstance(
            self.session_state.tool_context.get("last_information_result"), dict
        )
        information_request = is_read_only_information_request(
            cleaned_text, information_context
        )
        if (
            information_request
            and not is_time_or_weather_request(cleaned_text)
            and not self.expanded_web_enabled
        ):
            answer = self._deferred_information_answer()
            append_assistant_message(self.session_state, answer)
            return answer
        if (
            not model_router_enabled
            and information_request
        ):
            answer = self._process_information_turn(cleaned_text)
            append_assistant_message(self.session_state, answer)
            return answer

        rule_call = parse_tool_call(cleaned_text)
        tool_call = rule_call
        if (
            model_router_enabled
            and rule_call
            and rule_call.name not in FAST_LOCAL_TOOL_NAMES
        ):
            tool_call = None
        if tool_call and self.tool_executor:
            resolved_call = self._resolve_contextual_tool_call(tool_call.name, tool_call.args)
            if isinstance(resolved_call, dict) and "ok" in resolved_call:
                answer = self._answer_from_tool_result(resolved_call)
                append_assistant_message(self.session_state, answer)
                return answer
            tool_name, tool_args = resolved_call
            answer, memory_answer = self._execute_external_tool(
                tool_name, tool_args, timing_name=tool_call.name
            )
            append_assistant_message(self.session_state, memory_answer)
            return answer

        if model_router_enabled:
            self.session_state.tool_context.pop("pending_action", None)
            with timed_operation("enrutamiento_capacidades_modelo"):
                model_turn = model_tool_method(cleaned_text, self.session_state)
            LOGGER.debug(
                "Enrutador de capacidades: tool=%s args=%s",
                model_turn.tool_name or "conversation",
                model_turn.tool_args,
            )
            if model_turn.tool_name:
                try:
                    model_call = resolve_capability_tool_call(
                        model_turn.tool_name, model_turn.tool_args
                    )
                except ValueError as error:
                    answer = f"No pude completar la solicitud. {error}"
                    append_assistant_message(self.session_state, answer)
                    return answer
                if model_call.name == "external.blocked":
                    answer = (
                        "Puedo consultar informacion publica y mostrarte fuentes, pero no "
                        "puedo comprar, reservar, iniciar sesion, crear cuentas ni ejecutar "
                        "acciones en internet."
                    )
                    append_assistant_message(self.session_state, answer)
                    return answer
                if model_call.name == "information.deferred":
                    answer = self._deferred_information_answer()
                    append_assistant_message(self.session_state, answer)
                    return answer
                if model_call.name.startswith("information."):
                    answer = self._process_classified_information_call(
                        cleaned_text, model_call
                    )
                    append_assistant_message(self.session_state, answer)
                    return answer
                if model_call.name.startswith("lg_tv.") and not lg_tv_model_enabled:
                    answer = "La television LG no esta configurada en esta sesion."
                    append_assistant_message(self.session_state, answer)
                    return answer
                if model_call.name.startswith("spotify.") and not spotify_model_enabled:
                    answer = "Spotify no esta configurado en esta sesion."
                    append_assistant_message(self.session_state, answer)
                    return answer
                resolved_call = self._resolve_contextual_tool_call(
                    model_call.name, model_call.args
                )
                if isinstance(resolved_call, dict) and "ok" in resolved_call:
                    answer = self._answer_from_tool_result(resolved_call)
                    append_assistant_message(self.session_state, answer)
                    return answer
                tool_name, tool_args = resolved_call
                answer, memory_answer = self._execute_external_tool(
                    tool_name, tool_args, timing_name=tool_name
                )
                append_assistant_message(self.session_state, memory_answer)
                return answer

            cleaned_answer = clean_assistant_text(model_turn.text)
            append_assistant_message(self.session_state, cleaned_answer)
            return cleaned_answer

        self.session_state.tool_context.pop("pending_action", None)

        with timed_operation("respuesta_modelo"):
            answer = self.engine.respond(cleaned_text, self.session_state)

        cleaned_answer = clean_assistant_text(answer)
        append_assistant_message(self.session_state, cleaned_answer)
        return cleaned_answer

    def _warm_information_provider(self, user_text: str) -> None:
        """Open the weather connection in parallel with the model's decision."""

        warm = getattr(self.information_executor, "warm", None)
        if callable(warm) and is_time_or_weather_request(user_text):
            in_background(warm)

    def _process_information_turn(self, user_text: str) -> str:
        """Run an isolated turn where only read-only information tools are available."""

        if (
            not self.expanded_web_enabled
            and not is_time_or_weather_request(user_text)
        ):
            return self._deferred_information_answer()
        information_method = getattr(self.engine, "respond_with_information_tools", None)
        if not self.information_executor or not callable(information_method):
            return "La consulta de informacion actual no esta disponible en esta configuracion."

        with timed_operation("consulta_informacion_modelo"):
            model_turn = information_method(user_text, self.session_state)

        if model_turn.tool_name == "information_web_result":
            answer = clean_assistant_text(model_turn.text)
            if is_universidad_de_chile_request(user_text):
                official_url_match = re.search(
                    r"https://(?:www\.)?campeonatochileno\.cl/match/[^\s)]+",
                    answer,
                )
                if official_url_match:
                    official_url = official_url_match.group(0).rstrip(".,;:")
                    try:
                        result = self.information_executor.execute(
                            "information.sports_match_page", {"url": official_url}
                        )
                    except InformationError:
                        pass
                    else:
                        if result.get("ok"):
                            answer = str(result.get("message", answer))
                            self.spoken_answer = str(
                                result.get("spoken_message", "")
                            ).strip() or self._information_spoken_text(answer)
            if is_sports_result_request(user_text):
                answer = self._concise_sports_answer(answer)
            if not self.spoken_answer:
                self.spoken_answer = self._information_spoken_text(answer)
            self.session_state.tool_context["last_information_result"] = {
                "kind": "web",
                "answer": answer,
            }
            return answer

        if not model_turn.tool_name:
            answer = clean_assistant_text(model_turn.text)
            self.spoken_answer = self._information_spoken_text(answer)
            return answer

        try:
            tool_call = resolve_information_tool_call(
                model_turn.tool_name, model_turn.tool_args
            )
        except ValueError as error:
            return f"No pude completar la consulta. {error}"

        return self._process_classified_information_call(user_text, tool_call)

    def _process_classified_information_call(
        self, user_text: str, tool_call: ToolCall
    ) -> str:
        """Execute a locally validated information decision from either model."""

        if (
            not self.expanded_web_enabled
            and tool_call.name not in {"information.time", "information.weather"}
        ):
            return self._deferred_information_answer()
        if not self.information_executor:
            return "La consulta de informacion actual no esta disponible en esta configuracion."
        if tool_call.name == "information.web_query":
            web_search_method = getattr(self.engine, "respond_with_web_search", None)
            if not callable(web_search_method):
                return self._process_information_turn(user_text)
            with timed_operation("consulta_web_clasificada"):
                search_turn = web_search_method(user_text, self.session_state)
            if search_turn.tool_name != "information_web_result":
                return "No pude verificar esa informacion con una fuente actual."
            answer = clean_assistant_text(search_turn.text)
            self.spoken_answer = self._information_spoken_text(answer)
            self.session_state.tool_context["last_information_result"] = {
                "kind": "web",
                "query": tool_call.args,
                "answer": answer,
            }
            return answer
        if tool_call.name == "information.sports_query":
            return self._process_structured_sports_query(user_text, tool_call.args)

        args = tool_call.args
        if (
            tool_call.name == "information.time"
            and "location" in args
            and "timezone" not in args
            and not place_mentioned(str(args["location"]), user_text)
        ):
            # A place the user never said is the model guessing: use the local clock.
            args = {}
        try:
            with timed_operation(f"tool_{tool_call.name}"):
                result = self.information_executor.execute(tool_call.name, args)
        except InformationError as error:
            return f"No pude completar la consulta. {error}"

        if not result.get("ok"):
            return f"No pude completar la consulta. {result.get('message', 'El proveedor no respondio.')}"
        answer = str(result.get("message", "La consulta no entrego una respuesta."))
        spoken = str(result.get("spoken_message", "")).strip()
        self.spoken_answer = spoken or self._information_spoken_text(answer)
        self.session_state.tool_context["last_information_result"] = {
            "kind": tool_call.name,
            "args": args,
            "result": result,
        }
        return answer

    def _process_structured_sports_query(
        self, user_text: str, sports_query: dict[str, object]
    ) -> str:
        """Use a verified local provider when possible, otherwise a sports-only search."""

        if not self.expanded_web_enabled:
            return self._deferred_information_answer()
        participant = str(sports_query.get("participant", "")).strip()
        team_slug = chilean_team_slug(participant) if participant else None
        country = str(sports_query.get("country", "")).casefold()
        competition = str(sports_query.get("competition", "")).casefold()
        known_chilean_slugs = {
            "universidad-de-chile",
            "universidad-catolica",
            "colo-colo",
        }
        is_chilean_football = (
            sports_query.get("sport") == "football"
            and (
                "chile" in country
                or "chilen" in competition
                or team_slug in known_chilean_slugs
            )
        )
        if (
            self.information_executor
            and sports_query.get("intent") == "latest_result"
            and is_chilean_football
            and team_slug
        ):
            try:
                with timed_operation("consulta_ultimo_partido_chileno"):
                    result = self.information_executor.execute(
                        "information.latest_chilean_team_match",
                        {
                            "team_slug": team_slug,
                            "on_or_before": date.today().isoformat(),
                        },
                    )
            except InformationError:
                pass
            else:
                if result.get("ok"):
                    answer = str(result.get("message", ""))
                    self.spoken_answer = str(
                        result.get("spoken_message", "")
                    ).strip() or self._information_spoken_text(answer)
                    self.session_state.tool_context["last_information_result"] = {
                        "kind": "information.sports_query",
                        "query": sports_query,
                        "result": result,
                    }
                    return answer

        search_method = getattr(self.engine, "respond_with_sports_search", None)
        if not callable(search_method):
            return (
                "Pude interpretar la consulta deportiva, pero la busqueda web "
                "no esta disponible en esta configuracion."
            )
        with timed_operation("consulta_deportiva_web"):
            search_turn = search_method(
                user_text, sports_query, self.session_state
            )
        if search_turn.tool_name != "information_web_result":
            return "No pude verificar el resultado deportivo con una fuente actual."

        cleaned_sports_text = clean_assistant_text(search_turn.text)
        answer = self._concise_sports_answer(cleaned_sports_text, sports_query)
        self.spoken_answer = self._information_spoken_text(answer)
        self.session_state.tool_context["last_information_result"] = {
            "kind": "information.sports_query",
            "query": sports_query,
            "answer": answer,
        }
        return answer

    @staticmethod
    def _deferred_information_answer() -> str:
        """Return the fixed boundary for information capabilities postponed by the user."""

        return (
            "Esa capacidad web esta deshabilitada por ahora. En este momento solo "
            "puedo consultar la hora y el clima."
        )

    @staticmethod
    def _information_spoken_text(answer: str) -> str:
        """Keep source URLs visible in text without making Piper read them aloud."""

        for marker in ("\n\nFuentes consultadas:", "\n\nFuente:"):
            if marker in answer:
                return answer.split(marker, 1)[0].strip()
        return answer

    @staticmethod
    def _concise_sports_answer(
        answer: str, sports_query: dict[str, object] | None = None
    ) -> str:
        """Keep sports facts concise without dropping requested schedule entries."""

        source_marker = next(
            (
                marker
                for marker in ("\n\nFuentes consultadas:", "\n\nFuente:")
                if marker in answer
            ),
            "",
        )
        body, separator, sources = (
            answer.partition(source_marker) if source_marker else (answer, "", "")
        )
        intent = str((sports_query or {}).get("intent", ""))
        if intent == "schedule":
            matchup_pattern = re.compile(
                r"[a-záéíóúñ][^\n]{1,80}\s(?:vs\.?|contra|ante)\s"
                r"[^\n]{1,80}[a-záéíóúñ]",
                flags=re.I,
            )
            matchup_marker = re.compile(r"(?:\bvs\.?|\bcontra\b|\bante\b)", re.I)
            event_time = re.compile(
                r"\b(?:[01]?\d|2[0-3]):[0-5]\d\b|"
                r"\b\d{1,2}\s*(?:a\.?\s*m\.?|p\.?\s*m\.?|h(?:rs?)?|horas)\b|"
                r"\bhorario por confirmar\b",
                re.I,
            )
            allows_solo_events = str((sports_query or {}).get("sport", "")) in {
                "athletics",
                "motorsport",
                "other",
            }
            schedule_units: list[str] = []
            for line in (line.strip() for line in body.splitlines() if line.strip()):
                parts = re.split(
                    r"(?<=[.!?])\s*(?=[A-ZÁÉÍÓÚÑ])", line
                )
                for part in parts:
                    markers = list(matchup_marker.finditer(part))
                    if len(markers) == 1 and matchup_pattern.search(part):
                        schedule_units.append(part.strip())
                    elif (
                        not markers
                        and allows_solo_events
                        and re.search(r"\s[–—]\s", part)
                        and event_time.search(part)
                        and re.search(r"[a-záéíóúñ]{3}", part, flags=re.I)
                    ):
                        schedule_units.append(part.strip())
            if schedule_units:
                unique_units: list[str] = []
                seen_schedule_items: set[str] = set()
                for unit in schedule_units:
                    marker = matchup_marker.search(unit)

                    def participant_key(value: str) -> str:
                        without_seed = re.sub(
                            r"\s*(?:\[\d+\]|\(\d+\))\s*", " ", value
                        )
                        return re.sub(
                            r"[^a-z0-9áéíóúñ]+", " ", without_seed.casefold()
                        ).strip()

                    if marker:
                        left = unit[: marker.start()].rsplit(":", 1)[-1]
                        right = re.split(
                            r"\s[–—]\s|,\s*(?:no antes|a las|desde las|\d{1,2}:\d{2})",
                            unit[marker.end() :],
                            maxsplit=1,
                            flags=re.I,
                        )[0]
                        item_key = (
                            f"match|{participant_key(left)}|{participant_key(right)}"
                        )
                    else:
                        event_name = re.split(r"\s[–—]\s", unit, maxsplit=1)[0]
                        item_key = f"event|{participant_key(event_name)}"
                    if item_key in seen_schedule_items:
                        continue
                    seen_schedule_items.add(item_key)
                    unique_units.append(unit)
                factual_body = "\n".join(unique_units[:16])
                factual_body = re.sub(
                    r"^(?:he encontrado|encontre|encontré)\b[^:.\n]*[:.]\s*",
                    "",
                    factual_body,
                    flags=re.I,
                )
            else:
                fallback_sentences = [
                    sentence.strip()
                    for sentence in re.split(
                        r"(?<=[.!?])\s*(?=[A-ZÁÉÍÓÚÑ])", body.strip()
                    )
                    if sentence.strip()
                ]
                factual_body = fallback_sentences[0] if fallback_sentences else body.strip()
            if separator and sources.strip():
                return f"{factual_body}{source_marker}{sources}"
            return factual_body

        sentences = [
            sentence.strip()
            for sentence in re.split(
                r"(?<=[.!?])\s*(?=[A-ZÁÉÍÓÚÑ])", body.strip()
            )
            if sentence.strip()
        ]
        score_pattern = r"\b\d{1,3}\s*[-:\u2013]\s*\d{1,3}\b"
        result_pattern = (
            r"\b(?:gano|ganó|vencio|venció|derroto|derrotó|perdio|perdió|"
            r"empato|empató|cayo|cayó|campeon|campeón|campeona)\b|\bse impuso\b"
        )
        first_sentence = next(
            (sentence for sentence in sentences if re.search(score_pattern, sentence)),
            next(
                (
                    sentence
                    for sentence in sentences
                    if re.search(result_pattern, sentence, flags=re.I)
                ),
                sentences[0] if sentences else body.strip(),
            ),
        )
        first_sentence = re.sub(
            r"^(?:he encontrado|encontre|encontré)\s+(?:una\s+)?"
            r"(?:referencia|referencias)\s+.*?\b(?:que indica(?:n)? que|"
            r"segun (?:la|las) cual(?:es)?|según (?:la|las) cual(?:es)?)\s+",
            "",
            first_sentence,
            flags=re.I,
        )
        first_sentence = re.sub(
            r"\s+hace\s+\d+\s+d[ií]as(?=[.]?$)", "", first_sentence, flags=re.I
        )
        first_sentence = re.sub(
            r"\s+en\s+la\s+final\s+disputada\b.*?[.]?$",
            ".",
            first_sentence,
            flags=re.I,
        )
        first_sentence = re.sub(
            r",\s+(?:convirtiendose|convirtiéndose|siendo|lo que|mostrando|demostrando)\b.*?[.]?$",
            ".",
            first_sentence,
            flags=re.I,
        )
        if separator and sources.strip():
            return f"{first_sentence}{source_marker}{sources}"
        return first_sentence

    def _execute_external_tool(
        self,
        tool_name: str,
        tool_args: dict[str, object],
        *,
        timing_name: str,
    ) -> tuple[str, str]:
        """Execute one validated local integration without crossing namespaces."""

        if tool_name.startswith(("spotify.", "media.")):
            return self._execute_spotify_tool(
                tool_name, tool_args, timing_name=timing_name
            )
        assert self.tool_executor is not None
        with timed_operation(f"tool_{timing_name}"):
            tool_result = self.tool_executor.execute(tool_name, tool_args)
        answer = self._answer_from_tool_result(tool_result)
        return answer, answer

    def _execute_spotify_tool(
        self,
        tool_name: str,
        tool_args: dict[str, object],
        *,
        timing_name: str,
    ) -> tuple[str, str]:
        """Acknowledge Spotify and cross-device media actions before execution."""

        assert self.tool_executor is not None
        voice_ack = self._spotify_voice_ack(tool_name, tool_args)
        acknowledgement_spoken = False
        if self.voice_enabled and voice_ack:
            # Read-only lookups (search, device list) overlap the spoken
            # confirmation; nothing changes in Spotify until execute() below.
            prefetch = getattr(self.tool_executor, "prefetch", None)
            warmup = (
                threading.Thread(
                    target=self._safe_prefetch,
                    args=(prefetch, tool_name, dict(tool_args)),
                    daemon=True,
                )
                if prefetch is not None
                else None
            )
            if warmup:
                warmup.start()
            try:
                with timed_operation("salida_voz_previa_spotify"):
                    self.tts.speak(voice_ack)
                acknowledgement_spoken = True
            except VoiceError as error:
                LOGGER.warning("Spotify pre-action voice failed: %s", error)
            finally:
                if warmup:
                    with timed_operation("espera_prefetch_spotify"):
                        warmup.join()

        with timed_operation(f"tool_{timing_name}"):
            tool_result = self.tool_executor.execute(tool_name, tool_args)
        self._remember_tool_result(tool_result)
        full_answer = self._answer_from_tool_result(tool_result)

        if not self.voice_enabled or not tool_result.get("ok"):
            return full_answer, full_answer

        if tool_name == "spotify.pause":
            self.skip_post_turn_speech = True
            return "", full_answer

        if acknowledgement_spoken and voice_ack:
            self.skip_post_turn_speech = True
            return voice_ack, full_answer

        return full_answer, full_answer

    @staticmethod
    def _safe_prefetch(prefetch, tool_name: str, tool_args: dict[str, object]) -> None:
        """A failed warm-up must never break the command; execute retries it."""

        try:
            with timed_operation("prefetch_spotify"):
                prefetch(tool_name, tool_args)
        except Exception as error:  # noqa: BLE001 - background thread boundary
            LOGGER.debug("Spotify prefetch failed: %s", error)

    @staticmethod
    def _spotify_voice_ack(
        tool_name: str, tool_args: dict[str, object]
    ) -> str | None:
        """Return the short phrase spoken before a Spotify mutation."""

        if tool_name == "spotify.pause":
            return ""
        if tool_name == "media.spotify_on_tv":
            query = str(tool_args.get("query", "")).strip()
            return (
                f"Preparando {query} en la television."
                if query
                else "Preparando Spotify en la television."
            )
        if tool_name in {"spotify.play", "spotify.play_playlist"}:
            query = str(tool_args.get("query", "")).strip()
            return f"Reproduciendo {query} en Spotify." if query else "Reproduciendo en Spotify."
        if tool_name == "spotify.play_item":
            item = tool_args.get("item")
            name = str(item.get("name", "")).strip() if isinstance(item, dict) else ""
            return f"Reproduciendo {name} en Spotify." if name else "Reproduciendo en Spotify."
        if tool_name == "spotify.play_reference":
            return "Reproduciendo en Spotify."
        if tool_name in {"spotify.next", "spotify.previous"}:
            return "Hecho."
        if tool_name == "spotify.resume":
            return "Reanudando."
        if tool_name in {"spotify.volume", "spotify.shuffle", "spotify.repeat"}:
            return "Hecho."
        if tool_name == "spotify.queue":
            return "Agregando a la cola."
        if tool_name in {"spotify.save", "spotify.save_current"}:
            return "Guardando."
        if tool_name == "spotify.play_similar":
            return "Buscando algo similar."
        if tool_name == "spotify.create_playlist":
            return "Creando playlist."
        if tool_name == "spotify.update_playlist":
            return "Actualizando playlist."
        if tool_name in {"spotify.add_to_playlist", "spotify.remove_from_playlist"}:
            return "Actualizando playlist."
        if tool_name == "spotify.delete_playlist" and tool_args.get("confirmed"):
            return "Hecho."
        return None

    def _answer_from_tool_result(self, tool_result: dict[str, object]) -> str:
        """Convert structured tool output into a short spoken answer."""

        message = str(tool_result.get("message", "La herramienta no entrego respuesta."))
        if tool_result.get("direct_answer"):
            return message
        if tool_result.get("confirmation_required"):
            return message
        if tool_result.get("ok"):
            return f"Hecho. {message}"
        return f"No pude completar la orden. {message}"

    def _resolve_contextual_tool_call(
        self, tool_name: str, args: dict[str, object]
    ) -> tuple[str, dict[str, object]] | dict[str, object]:
        """Resolve confirmations and ordinal references using session context."""

        pending = self.session_state.tool_context.get("pending_action")
        if tool_name == "spotify.cancel":
            self.session_state.tool_context.pop("pending_action", None)
            return {"ok": True, "message": "Operacion cancelada."}
        if tool_name == "spotify.confirm":
            if not isinstance(pending, dict):
                return {"ok": False, "message": "No hay ninguna operacion pendiente de confirmacion."}
            self.session_state.tool_context.pop("pending_action", None)
            action_name = str(pending.get("name", ""))
            action_args = pending.get("args", {})
            if not action_name or not isinstance(action_args, dict):
                return {"ok": False, "message": "La confirmacion pendiente no es valida."}
            return action_name, action_args

        if tool_name == "spotify.explain_last":
            last_result = self.session_state.tool_context.get("last_spotify_result")
            if not isinstance(last_result, dict):
                return {
                    "ok": False,
                    "direct_answer": True,
                    "message": "No tengo una accion anterior de Spotify que pueda explicar.",
                }
            explanation = str(last_result.get("explanation", "")).strip()
            if explanation:
                return {
                    "ok": True,
                    "direct_answer": True,
                    "message": explanation,
                }
            return {
                "ok": True,
                "direct_answer": True,
                "message": str(
                    last_result.get(
                        "message", "La ultima accion de Spotify no incluyo mas detalles."
                    )
                ),
            }

        self.session_state.tool_context.pop("pending_action", None)
        if tool_name == "spotify.play_result":
            candidates = self.session_state.tool_context.get("spotify_results")
            index = int(args.get("index", -1))
            if not isinstance(candidates, list) or not candidates:
                return {"ok": False, "message": "No tengo una busqueda anterior para elegir un resultado."}
            if index < 0 or index >= len(candidates):
                return {"ok": False, "message": f"La busqueda anterior solo tiene {len(candidates)} resultados."}
            item = candidates[index]
            if not isinstance(item, dict):
                return {"ok": False, "message": "Ese resultado anterior ya no es valido."}
            return "spotify.play_item", {"item": item}
        return tool_name, args

    def _remember_tool_result(self, tool_result: dict[str, object]) -> None:
        """Store only the structured context needed by later Spotify commands."""

        if str(tool_result.get("tool_name", "")).startswith("spotify."):
            self.session_state.tool_context["last_spotify_result"] = dict(tool_result)
        candidates = tool_result.get("candidates")
        if isinstance(candidates, list):
            self.session_state.tool_context["spotify_results"] = candidates
        pending_action = tool_result.get("pending_action")
        if tool_result.get("confirmation_required") and isinstance(pending_action, dict):
            self.session_state.tool_context["pending_action"] = pending_action


def build_parser() -> argparse.ArgumentParser:
    """Create the command-line parser."""

    parser = argparse.ArgumentParser(description="Habla con Vader desde la terminal.")
    parser.add_argument(
        "--voice",
        action="store_true",
        help="Reproduce las respuestas con voz usando el proveedor TTS configurado.",
    )
    parser.add_argument(
        "--listen",
        action="store_true",
        help="Enter sin texto graba tu voz; escribir sigue funcionando igual.",
    )
    parser.add_argument(
        "--manos-libres",
        action="store_true",
        help='Sin teclado: espera "Oye Vader", conversa y enciende las luces (implica --voice --listen).',
    )
    parser.add_argument(
        "--esperar-red",
        action="store_true",
        help=(
            "Con --manos-libres: carga todo y espera la red (network-online.target) "
            "justo antes de las pupilas tenues; lo usa el servicio de arranque."
        ),
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Muestra logs detallados para diagnostico.",
    )
    parser.add_argument(
        "--mode",
        choices=("canon", "chileno"),
        help="Selecciona la personalidad activa de Vader para esta sesion.",
    )
    parser.add_argument(
        "--perfil",
        metavar="NOMBRE",
        help=(
            "Usa el perfil profiles/NOMBRE (por ejemplo pablo) en vez de elegirlo "
            "por la red Wi-Fi; sirve para --spotify-auth de otro hogar."
        ),
    )
    parser.add_argument(
        "--spotify-auth",
        action="store_true",
        help="Autoriza la cuenta de Spotify y guarda un token local.",
    )
    parser.add_argument(
        "--spotify-devices",
        action="store_true",
        help="Lista los dispositivos Spotify Connect disponibles.",
    )
    parser.add_argument(
        "--spotify-play",
        metavar="BUSQUEDA",
        help="Busca y reproduce una cancion o playlist en Spotify.",
    )
    parser.add_argument(
        "--spotify-playlists",
        action="store_true",
        help="Lista tus playlists visibles de Spotify.",
    )
    parser.add_argument(
        "--spotify-playlist",
        metavar="NOMBRE",
        help="Busca una playlist tuya por nombre y la reproduce.",
    )
    parser.add_argument(
        "--spotify-item-type",
        choices=("auto", "track", "artist", "playlist", "album", "show", "episode"),
        default="auto",
        help="Tipo de resultado que debe buscar --spotify-play.",
    )
    return parser


def _build_tts(settings, voice_enabled: bool) -> TextToSpeech:
    """Build the requested text-to-speech implementation."""

    if not voice_enabled:
        return SilentTextToSpeech()
    if settings.tts_provider == "elevenlabs":
        return ElevenLabsTextToSpeech(settings)
    if settings.tts_provider == "piper":
        return PiperTextToSpeech(settings)
    return OpenAITextToSpeech(settings)


def _build_tool_executor(settings) -> ToolExecutor:
    """Build the tool executor available to the conversational loop."""

    executors: dict[str, ToolExecutor] = {}
    spotify_executor: SpotifyToolsExecutor | None = None
    lg_tv_executor: LgTvToolsExecutor | None = None
    if settings.spotify_client_id and settings.spotify_client_secret:
        spotify_executor = SpotifyToolsExecutor(settings)
        executors["spotify."] = spotify_executor
    if settings.lg_tv_host:
        lg_tv_executor = LgTvToolsExecutor(settings)
        executors["lg_tv."] = lg_tv_executor
    if spotify_executor is not None and lg_tv_executor is not None:
        executors["media."] = MediaToolsExecutor(
            settings,
            tv=lg_tv_executor.client,
            spotify=spotify_executor.client,
        )
    if not executors:
        return NoToolsExecutor()
    return RoutedToolsExecutor(executors)


def _build_information_executor() -> ToolExecutor:
    """Build the executor whose network surface is restricted to public GET requests."""

    return InformationToolsExecutor(
        geocoding_cache_path=Path.cwd() / ".runtime_cache" / "geocoding_cache.json"
    )


def _network_warm_ups(app: "VaderMindApp", listener: VoiceListener | None) -> list[Callable[[], None]]:
    """Warm-ups that open connections (gateway, tools): useless without network."""

    tasks = [
        getattr(target, "warm", None)
        for target in (app.engine, app.tool_executor, listener.transcriber if listener else None)
    ]
    return [task for task in tasks if callable(task)]


def _warm_up_session(
    app: "VaderMindApp", listener: VoiceListener | None, *, network: bool = True
) -> None:
    """Prepare Piper and the network before the first turn, without blocking the CLI.

    With network=False the connections wait for the network (`--esperar-red`):
    Piper still warms up now. Once listening, the gateway connections are
    refreshed again as soon as the user starts talking, so they are open when
    the transcript is ready.
    """

    tasks = [getattr(app.tts, "warm", None)]
    if network:
        tasks += _network_warm_ups(app, listener)
    in_background(*(task for task in tasks if callable(task)))
    if listener is not None:
        network = [
            task
            for task in (getattr(listener.transcriber, "warm", None), getattr(app.engine, "warm", None))
            if callable(task)
        ]
        listener.on_speech_start = lambda: in_background(*network)


def _build_chat_engine(settings) -> ChatEngine:
    """Build the configured LLM provider without changing the default provider."""

    if settings.llm_provider == "openrouter":
        return OpenRouterChatEngine(settings)
    if settings.llm_provider == "vercel":
        return VercelGatewayChatEngine(settings)
    if settings.llm_provider == "hybrid":
        return HybridChatEngine(settings)
    return ClaudeChatEngine(settings)


def _run_spotify_auth(settings) -> int:
    """Authorize Spotify with a one-time browser flow."""

    try:
        result = SpotifyOAuth(settings).authorize_with_local_callback()
    except SpotifyError as error:
        _safe_print(f"VaderMind > {error}")
        return 1

    _safe_print(f"VaderMind > {result.message}")
    return 0 if result.authorized else 1


def _run_spotify_devices(settings) -> int:
    """Print available Spotify Connect devices."""

    try:
        devices = SpotifyClient(settings).get_devices()
    except SpotifyError as error:
        _safe_print(f"VaderMind > {error}")
        return 1

    if not devices:
        _safe_print("VaderMind > No hay dispositivos Spotify disponibles.")
        return 0

    _safe_print("Dispositivos Spotify disponibles:")
    for device in devices:
        active = "activo" if device.get("is_active") else "inactivo"
        _safe_print(f"- {device.get('name')} ({device.get('type')}, {active})")
    return 0


def _run_spotify_play(settings, query: str, item_type: str, voice_enabled: bool) -> int:
    """Run the first Spotify playback command from the CLI."""

    try:
        tts = _build_tts(settings, voice_enabled=voice_enabled)
        result = SpotifyToolsExecutor(settings).execute(
            "spotify.play",
            {"query": query, "item_type": item_type},
        )
    except (SpotifyError, VoiceError) as error:
        _safe_print(f"VaderMind > {error}")
        return 1

    message = str(result.get("message", "Spotify no devolvio un mensaje."))
    _safe_print(f"Vader > {message}")
    if voice_enabled:
        try:
            with timed_operation("salida_voz"):
                tts.speak(message)
        except VoiceError as error:
            LOGGER.warning("Audio playback failed: %s", error)
            _safe_print(f"VaderMind > Aviso: {error}")
    return 0 if result.get("ok") else 1


def _run_spotify_playlists(settings) -> int:
    """Print the user's Spotify playlists."""

    result = SpotifyToolsExecutor(settings).execute(
        "spotify.list_playlists",
        {"limit": 10},
    )
    _safe_print(f"Vader > {result.get('message', 'Spotify no devolvio un mensaje.')}")
    return 0 if result.get("ok") else 1


def _run_spotify_playlist(settings, query: str, voice_enabled: bool) -> int:
    """Play one of the user's Spotify playlists by name."""

    try:
        tts = _build_tts(settings, voice_enabled=voice_enabled)
        result = SpotifyToolsExecutor(settings).execute(
            "spotify.play_playlist",
            {"query": query},
        )
    except VoiceError as error:
        _safe_print(f"VaderMind > {error}")
        return 1

    message = str(result.get("message", "Spotify no devolvio un mensaje."))
    _safe_print(f"Vader > {message}")
    if voice_enabled:
        try:
            with timed_operation("salida_voz"):
                tts.speak(message)
        except VoiceError as error:
            LOGGER.warning("Audio playback failed: %s", error)
            _safe_print(f"VaderMind > Aviso: {error}")
    return 0 if result.get("ok") else 1


def run_cli() -> int:
    """Run the interactive VaderMind session."""

    STARTUP.mark("imports")
    args = build_parser().parse_args()
    try:
        profile = resolve_profile(args.perfil)
    except ConfigurationError as error:
        _safe_print(f"Error de configuracion: {error}")
        return 1
    settings = load_settings(profile_dir=profile.directory)
    settings.profile_name = profile.name
    if profile.name:
        _safe_print(f"Perfil: {profile.describe()}.")
    if args.debug:
        settings.debug = True
    if args.mode:
        settings.persona_mode = args.mode
    if args.manos_libres:
        args.voice = args.listen = True
    configure_logging(settings.debug)

    if args.spotify_auth:
        try:
            validate_settings(settings, require_anthropic=False, require_spotify=True)
        except ConfigurationError as error:
            _safe_print(f"Error de configuracion: {error}")
            return 1
        return _run_spotify_auth(settings)

    if args.spotify_devices:
        try:
            validate_settings(settings, require_anthropic=False, require_spotify=True)
        except ConfigurationError as error:
            _safe_print(f"Error de configuracion: {error}")
            return 1
        return _run_spotify_devices(settings)

    if args.spotify_play:
        try:
            validate_settings(
                settings,
                require_voice=args.voice,
                require_anthropic=False,
                require_spotify=True,
            )
        except ConfigurationError as error:
            _safe_print(f"Error de configuracion: {error}")
            return 1
        return _run_spotify_play(
            settings,
            query=args.spotify_play,
            item_type=args.spotify_item_type,
            voice_enabled=args.voice,
        )

    if args.spotify_playlists:
        try:
            validate_settings(settings, require_anthropic=False, require_spotify=True)
        except ConfigurationError as error:
            _safe_print(f"Error de configuracion: {error}")
            return 1
        return _run_spotify_playlists(settings)

    if args.spotify_playlist:
        try:
            validate_settings(
                settings,
                require_voice=args.voice,
                require_anthropic=False,
                require_spotify=True,
            )
        except ConfigurationError as error:
            _safe_print(f"Error de configuracion: {error}")
            return 1
        return _run_spotify_playlist(
            settings,
            query=args.spotify_playlist,
            voice_enabled=args.voice,
        )

    try:
        validate_settings(settings, require_voice=args.voice)
        STARTUP.mark("config")
        engine = _build_chat_engine(settings)
        tts = _build_tts(settings, voice_enabled=args.voice)
        STARTUP.mark("motor_voz")
    except (ConfigurationError, BrainError, VoiceError) as error:
        _safe_print(f"Error de configuracion: {error}")
        return 1

    listener: VoiceListener | None = None
    if args.listen:
        try:
            listener = build_voice_listener(settings)
            STARTUP.mark("escucha")
        except ListeningError as error:
            _safe_print(f"Error de configuracion de voz: {error}")
            return 1

    session_state = SessionState(language=settings.language)
    app = VaderMindApp(
        engine=engine,
        tts=tts,
        session_state=session_state,
        tool_executor=_build_tool_executor(settings),
        information_executor=_build_information_executor(),
        voice_enabled=args.voice,
    )
    STARTUP.mark("herramientas")
    wait_network = args.manos_libres and args.esperar_red
    _warm_up_session(app, listener, network=not wait_network)

    lights = _build_session_lights(settings)
    STARTUP.mark("luces")
    try:
        if args.manos_libres and listener is not None:
            return _run_hands_free(
                settings, app, listener, lights, wait_network=wait_network, profile=profile
            )
        return _run_keyboard_loop(args, settings, app, listener, lights)
    finally:
        lights.close()


def _build_session_lights(settings) -> Lights:
    """Lights from VADER_LIGHTS; while Vader speaks, the ring follows his voice."""

    lights = build_lights(settings)
    if not isinstance(lights, NullLights):
        add_speech_observer(lambda path: lights.speak(speech_envelope(path)))
    return lights


def _power_off() -> bool:
    """Shut the Raspberry down safely (needs passwordless sudo, as on Pi OS)."""

    if sys.platform != "linux" or not shutil.which("sudo"):
        return False
    result = subprocess.run(
        ["sudo", "-n", "shutdown", "-h", "now"], capture_output=True, check=False
    )
    if result.returncode != 0:
        LOGGER.warning("No se pudo apagar: %s", result.stderr.decode(errors="replace").strip())
    return result.returncode == 0


def _tv_warm_up(app: "VaderMindApp") -> Callable[[], None] | None:
    """The LG TV's read-only warm-up, if the TV is configured.

    Reconnecting to webOS takes about 4 s (measured on the Pi). Run on "Oye
    Vader", it overlaps the user's speech, transcription and the model, so a
    TV command finds the connection open. It only checks the port and asks
    the power state: it never changes the TV and never sends Wake-on-LAN.
    """

    executors = getattr(app.tool_executor, "executors", {})
    warm = getattr(executors.get("lg_tv."), "warm", None)
    return warm if callable(warm) else None


REEXEC_MARK = "VADER_PERFIL_REINICIADO"


def _restart_if_other_home(
    profile: ProfileChoice,
    app: "VaderMindApp",
    listener: VoiceListener | None,
    lights: Lights,
    *,
    execv: Callable[[str, list[str]], object] = os.execv,
) -> None:
    """Start again with the right home's profile if the Wi-Fi says it changed.

    At a cold boot Vader loads while the Wi-Fi connects, so it starts with the
    last home's profile. If the network turns out to be another home's, it
    starts over in this same process (same PID for systemd) with that profile:
    Spotify, the TV and its timers are then built only once, for the right
    home. This costs one extra load, only the first time in another home.
    """

    new_profile = profile_after_network(profile)
    if new_profile is None or os.environ.get(REEXEC_MARK):
        return
    _safe_print(f"Red de otro hogar: cambio al perfil {new_profile} y reinicio.")
    remember_profile(Path.cwd(), new_profile)
    for resource in (app.tts, listener, lights):
        close = getattr(resource, "close", None)
        if callable(close):
            try:
                close()
            except Exception as error:  # a stuck resource must not block the restart
                LOGGER.warning("No pude cerrar %s antes de reiniciar: %s", type(resource).__name__, error)
    os.environ[REEXEC_MARK] = "1"
    sys.stdout.flush()
    execv(sys.executable, [sys.executable, *sys.argv])


def _run_hands_free(
    settings,
    app: "VaderMindApp",
    listener: VoiceListener,
    lights: Lights,
    *,
    wait_network: bool = False,
    profile: ProfileChoice | None = None,
) -> int:
    try:
        detector = build_wake_word(settings)
        STARTUP.mark("vosk")
    except ListeningError as error:
        _safe_print(f"Error de configuracion de voz: {error}")
        return 1
    if wait_network:
        # Everything is loaded: the pupils only go dim once the Wi-Fi is up too.
        wait_for_network()
        STARTUP.mark("red")
        if profile is not None:
            _restart_if_other_home(profile, app, listener, lights)
        in_background(*_network_warm_ups(app, listener))
    _safe_print("VaderMind esta listo en modo manos libres.")
    _safe_print(f"Modo de personalidad: {settings.persona_mode}.")
    _safe_print(f"Microfono: {listener.device_name}. Di \"{settings.wake_phrase}\". Ctrl+C termina.")
    listener.on_recorded = lambda: lights.set_state(PENSANDO)

    def stop_service(*_args: object) -> None:
        # `systemctl stop` sends SIGTERM: leave through `finally` so the eyes turn off.
        raise SystemExit(0)

    signal.signal(signal.SIGTERM, stop_service)
    wake_timing: dict[str, float] = {}  # when "Oye Vader" ended and was recognized
    tv_warm = _tv_warm_up(app)
    session = HandsFreeSession(
        app,
        listener,
        lights,
        lambda: wait_for_wake_word(detector, device=listener.device, timing=wake_timing),
        wake_marks=lambda: dict(wake_timing),
        on_idle=STARTUP.log_ready,
        on_wake=(lambda: in_background(tv_warm)) if tv_warm else None,
        follow_up_seconds=settings.follow_up_seconds,
        allow_shutdown=settings.allow_shutdown,
        power_off=_power_off,
        # The engines read settings.persona_mode on every request, so this is enough.
        set_mode=lambda mode: setattr(settings, "persona_mode", mode),
        write=_safe_print,
    )
    try:
        return session.run()
    except KeyboardInterrupt:
        _safe_print("\nVaderMind finalizado.")
        return 0


def _run_keyboard_loop(
    args, settings, app: "VaderMindApp", listener: VoiceListener | None, lights: Lights
) -> int:
    if listener is not None:
        listener.on_recorded = lambda: lights.set_state(PENSANDO)
    lights.set_state(REPOSO)
    _safe_print("VaderMind esta listo.")
    _safe_print(f"Modo de personalidad: {settings.persona_mode}.")
    if listener:
        _safe_print(f"Microfono: {listener.device_name}.")
        _safe_print("Presiona Enter para hablar o escribe tu mensaje. Usa 'salir' para terminar.")
    else:
        _safe_print("Escribe tu mensaje. Usa 'salir' para terminar.")

    while True:
        TURN_TIMELINE.reset()
        try:
            user_text = input("\nTu > ").strip()
        except (EOFError, KeyboardInterrupt):
            _safe_print("\nVaderMind finalizado.")
            return 0

        if not user_text and listener:
            _safe_print("(escuchando...)")
            lights.set_state(ESCUCHANDO)
            try:
                heard = listener.listen()
            except ListeningError as error:
                _safe_print(f"VaderMind > {error}")
                lights.set_state(ERROR)
                continue
            except KeyboardInterrupt:
                _safe_print("\nVaderMind finalizado.")
                return 0
            if heard.notice:
                _safe_print(f"VaderMind > {heard.notice}")
            if heard.text is None:
                lights.set_state(REPOSO if heard.reason == "sin_voz" else ERROR)
                continue
            user_text = heard.text
            _safe_print(f"Tu (voz) > {user_text}")

        if not user_text:
            _safe_print("Vader > Necesito una orden clara.")
            continue

        if user_text.lower() in EXIT_COMMANDS:
            _safe_print("Vader > Puedes retirarte. La sesion ha terminado.")
            return 0

        lights.set_state(PENSANDO)
        try:
            answer = app.process_turn(user_text)
        except ValueError as error:
            _safe_print(f"VaderMind > {error}")
            lights.set_state(ERROR)
            continue
        except BrainError as error:
            _safe_print(f"Vader > {error}")
            lights.set_state(ERROR)
            continue

        if answer:
            _safe_print(f"Vader > {answer}")

        if args.voice and answer and not app.skip_post_turn_speech:
            try:
                with timed_operation("salida_voz"):
                    app.tts.speak(app.spoken_answer or answer)
            except VoiceError as error:
                LOGGER.warning("Audio playback failed: %s", error)
                _safe_print(f"VaderMind > Aviso: {error}")
                lights.set_state(ERROR)
                continue
        lights.set_state(REPOSO)

        latency = TURN_TIMELINE.summary()
        if latency:
            LOGGER.debug("Latencia de voz: %s", latency)
