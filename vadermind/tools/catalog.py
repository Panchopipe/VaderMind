"""Provider-neutral model tools and strict mapping to internal actions."""

from __future__ import annotations

from typing import Any

from vadermind.tools.router import ToolCall


def _function_tool(
    name: str,
    description: str,
    properties: dict[str, object],
    required: list[str] | None = None,
) -> dict[str, object]:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required or [],
                "additionalProperties": False,
            },
        },
    }


def spotify_model_tools() -> list[dict[str, object]]:
    """Return the Spotify tools exposed to a language model."""

    text = {"type": "string"}
    return [
        _function_tool(
            "spotify_play",
            "Reproduce musica, un artista, album, playlist, podcast o episodio. "
            "Usa esta tool si la intencion final es poner, tocar, escuchar o reproducir, "
            "aunque la frase tambien diga busca o encuentra antes de pedir que lo pongas. "
            "Usa library_only=true solamente si el usuario pide su playlist/lista "
            "guardada o personal por nombre. Para 'una playlist de X', usa playlist "
            "con library_only=false. Conserva solo el nombre o busqueda, sin palabras "
            "como Spotify, aleatorio, playlist o lista.",
            {
                "query": text,
                "item_type": {
                    "type": "string",
                    "enum": ["auto", "track", "artist", "playlist", "album", "show", "episode"],
                },
                "shuffle": {"type": "boolean"},
                "library_only": {
                    "type": "boolean",
                    "description": "True solo para una playlist guardada o personal.",
                },
            },
            ["query"],
        ),
        _function_tool(
            "spotify_control",
            "Controla el reproductor actual de Spotify. Debes usar esta herramienta ante "
            "cualquier solicitud de pausar, reanudar, saltar, consultar, cambiar volumen, "
            "activar aleatorio o repeticion; nunca confirmes esas acciones solo con texto.",
            {
                "action": {
                    "type": "string",
                    "enum": [
                        "pause", "resume", "next", "previous", "current",
                        "shuffle_on", "shuffle_off", "repeat_track",
                        "repeat_context", "repeat_off", "set_volume",
                    ],
                },
                "volume_percent": {"type": "integer", "minimum": 0, "maximum": 100},
            },
            ["action"],
        ),
        _function_tool(
            "spotify_queue",
            "Agrega una cancion o episodio a la cola actual.",
            {
                "query": text,
                "item_type": {"type": "string", "enum": ["track", "episode"]},
            },
            ["query"],
        ),
        _function_tool(
            "spotify_save",
            "Guarda o marca Me gusta en la cancion actual o en una cancion buscada.",
            {"current": {"type": "boolean"}, "query": text},
        ),
        _function_tool(
            "spotify_browse_results",
            "Muestra opciones de un tipo especifico sin reproducirlas; usala solamente "
            "si el usuario quiere ver o elegir opciones y no pide reproducir una. Permite "
            "elegir despues por numero.",
            {
                "query": text,
                "item_type": {
                    "type": "string",
                    "enum": ["track", "artist", "playlist", "album", "show", "episode"],
                },
            },
            ["query", "item_type"],
        ),
        _function_tool(
            "spotify_playlist_manage",
            "Crea, renombra, cambia, elimina o modifica una playlist del usuario. "
            "Eliminar siempre iniciara una confirmacion separada en el codigo.",
            {
                "action": {
                    "type": "string",
                    "enum": ["create", "rename", "set_visibility", "set_description", "delete", "add_item", "remove_item"],
                },
                "playlist": text,
                "new_name": text,
                "description": text,
                "public": {"type": "boolean"},
                "item_query": text,
            },
            ["action", "playlist"],
        ),
        _function_tool(
            "spotify_play_reference",
            "Reproduce directamente una URL o URI de Spotify.",
            {"reference": text, "shuffle": {"type": "boolean"}},
            ["reference"],
        ),
        _function_tool(
            "spotify_followup",
            "Resuelve un seguimiento musical usando el contexto anterior.",
            {
                "action": {"type": "string", "enum": ["similar", "play_result"]},
                "index": {"type": "integer", "minimum": 1, "maximum": 10},
            },
            ["action"],
        ),
    ]


def lg_tv_model_tools() -> list[dict[str, object]]:
    """Return the bounded LG TV capabilities exposed to a language model."""

    return [
        _function_tool(
            "lg_tv_status",
            "Consulta el estado real de la television LG: encendido, aplicacion, "
            "canal, pantalla, salida de sonido, volumen y silencio. Usala para "
            "preguntas sobre el estado del televisor. Elige el detail mas especifico "
            "para responder solamente lo que el usuario pregunto; usa full solo si "
            "solicita el estado general.",
            {
                "detail": {
                    "type": "string",
                    "enum": [
                        "full",
                        "power",
                        "app",
                        "channel",
                        "screen",
                        "sound",
                        "volume",
                        "mute",
                    ],
                }
            },
        ),
        _function_tool(
            "lg_tv_power",
            "Enciende o apaga la television LG. Debes usar esta tool en vez de "
            "confirmar la accion solo con texto.",
            {"action": {"type": "string", "enum": ["on", "off"]}},
            ["action"],
        ),
        _function_tool(
            "lg_tv_volume",
            "Sube, baja o fija el volumen de la television LG. Para set debes "
            "incluir un volumen entero entre 0 y 100.",
            {
                "action": {"type": "string", "enum": ["up", "down", "set"]},
                "volume": {"type": "integer", "minimum": 0, "maximum": 100},
            },
            ["action"],
        ),
        _function_tool(
            "lg_tv_mute",
            "Activa o quita explicitamente el silencio de la television LG.",
            {"muted": {"type": "boolean"}},
            ["muted"],
        ),
        _function_tool(
            "lg_tv_launch_app",
            "Abre una aplicacion que ya esta instalada en la television LG. "
            "Conserva solamente su nombre, por ejemplo YouTube, Netflix o Spotify.",
            {"app_name": {"type": "string", "maxLength": 80}},
            ["app_name"],
        ),
        _function_tool(
            "lg_tv_select_input",
            "Cambia la television LG a una entrada HDMI fisica permitida.",
            {
                "input": {
                    "type": "string",
                    "enum": ["HDMI 1", "HDMI 2", "HDMI 3", "HDMI 4"],
                }
            },
            ["input"],
        ),
        _function_tool(
            "lg_tv_playback",
            "Controla la reproduccion de la aplicacion activa en la television LG. "
            "No la uses para controlar Spotify en otro dispositivo.",
            {
                "action": {
                    "type": "string",
                    "enum": ["play", "pause", "stop", "rewind", "fast_forward"],
                }
            },
            ["action"],
        ),
        _function_tool(
            "lg_tv_navigation",
            "Opera controles seguros del mando LG: inicio, volver, flechas, aceptar "
            "o cerrar la aplicacion activa.",
            {
                "action": {
                    "type": "string",
                    "enum": [
                        "home",
                        "back",
                        "up",
                        "down",
                        "left",
                        "right",
                        "ok",
                        "close_app",
                    ],
                }
            },
            ["action"],
        ),
        _function_tool(
            "lg_tv_channel",
            "Sube, baja o selecciona un canal publicado por la television LG. "
            "Para select conserva solamente el numero o nombre solicitado en query.",
            {
                "action": {
                    "type": "string",
                    "enum": ["up", "down", "select"],
                },
                "query": {"type": "string", "maxLength": 80},
            },
            ["action"],
        ),
        _function_tool(
            "lg_tv_screen",
            "Enciende o apaga solo la pantalla manteniendo la television funcionando.",
            {"on": {"type": "boolean"}},
            ["on"],
        ),
        _function_tool(
            "lg_tv_show_message",
            "Muestra en la television LG un mensaje breve pedido por el usuario.",
            {"message": {"type": "string", "maxLength": 160}},
            ["message"],
        ),
        _function_tool(
            "lg_tv_timer",
            "Programa, consulta o cancela temporizadores locales de la television. "
            "Solo puede apagar la TV o mostrar un mensaje entre 1 minuto y 24 horas.",
            {
                "action": {
                    "type": "string",
                    "enum": ["power_off", "message", "cancel", "status"],
                },
                "minutes": {"type": "integer", "minimum": 1, "maximum": 1440},
                "message": {"type": "string", "maxLength": 160},
                "scope": {
                    "type": "string",
                    "enum": ["all", "power_off", "messages"],
                },
            },
            ["action"],
        ),
    ]


def media_model_tools() -> list[dict[str, object]]:
    """Return orchestration tools that require both Spotify and the LG TV."""

    return [
        _function_tool(
            "spotify_play_on_tv",
            "Enciende la television si hace falta, abre Spotify, transfiere alli la "
            "reproduccion y confirma el dispositivo y contenido. Usala cuando el "
            "usuario pida musica de Spotify especificamente en la television. Usa "
            "library_only=true si pide su playlist o lista personal.",
            {
                "query": {"type": "string", "maxLength": 200},
                "item_type": {
                    "type": "string",
                    "enum": [
                        "auto",
                        "track",
                        "artist",
                        "playlist",
                        "album",
                        "show",
                        "episode",
                    ],
                },
                "shuffle": {"type": "boolean"},
                "library_only": {"type": "boolean"},
            },
            ["query"],
        )
    ]


def anthropic_spotify_tools() -> list[dict[str, object]]:
    """Convert OpenAI-compatible definitions to Anthropic's input_schema form."""

    converted: list[dict[str, object]] = []
    for item in spotify_model_tools():
        function = item["function"]
        assert isinstance(function, dict)
        converted.append(
            {
                "name": function["name"],
                "description": function["description"],
                "input_schema": function["parameters"],
            }
        )
    return converted


def information_model_tools() -> list[dict[str, object]]:
    """Return the complete experimental read-only catalog kept for future work."""

    text = {"type": "string"}
    return [
        _function_tool(
            "information_time",
            "Consulta la hora actual. Si el usuario nombra una ciudad o pais, convierte "
            "el lugar a una zona IANA valida, por ejemplo America/Santiago o Asia/Tokyo. "
            "Si no nombra ningun lugar, omite timezone y location: se usa la hora local "
            "del equipo. Nunca inventes un lugar.",
            {"timezone": text, "location": text},
            [],
        ),
        _function_tool(
            "information_weather",
            "Consulta condiciones actuales o el pronostico de hoy o manana mediante "
            "Open-Meteo. Usa current para 'ahora' o 'que temperatura hace'; today para "
            "'como estara hoy'; y tomorrow para manana.",
            {
                "location": text,
                "when": {"type": "string", "enum": ["current", "today", "tomorrow"]},
            },
            ["location", "when"],
        ),
        _function_tool(
            "information_sports_query",
            "Interpreta una consulta deportiva antes de buscar, incluidas competencias, "
            "pruebas, rondas, calendarios y eventos multideportivos como los Juegos "
            "Olimpicos. Identifica el deporte, la intencion, participantes, evento, "
            "competicion, pais y referencia temporal. "
            "Infiere la competicion y el pais cuando el equipo o evento sea inequivoco. "
            "Omite participante, rival, evento o pais si no corresponden; usa unspecified "
            "solo en los campos obligatorios que realmente no se puedan determinar. No "
            "inventes datos ni resultados: esta tool solo describe que se debe consultar.",
            {
                "intent": {
                    "type": "string",
                    "enum": [
                        "latest_result",
                        "live_result",
                        "event_result",
                        "schedule",
                        "standings",
                    ],
                },
                "sport": {
                    "type": "string",
                    "enum": [
                        "football",
                        "basketball",
                        "tennis",
                        "athletics",
                        "american_football",
                        "motorsport",
                        "rugby",
                        "other",
                    ],
                },
                "participant": text,
                "opponent": text,
                "event": text,
                "competition": text,
                "country": text,
                "date_reference": text,
            },
            ["intent", "sport", "competition", "date_reference"],
        ),
        _function_tool(
            "information_web_query",
            "Interpreta una solicitud que necesita informacion publica y vigente de "
            "internet, como precios, noticias o hechos actuales. Esta tool solo describe "
            "la consulta; la busqueda se ejecutara despues mediante Vercel AI Gateway. "
            "No la uses para comprar, reservar, iniciar sesion, enviar formularios ni "
            "publicar contenido.",
            {
                "query": text,
                "category": {
                    "type": "string",
                    "enum": ["price", "news", "current_fact", "other"],
                },
                "location": text,
                "date_reference": text,
            },
            ["query", "category"],
        ),
    ]


def active_information_model_tools() -> list[dict[str, object]]:
    """Return the information capabilities enabled in the current product stage."""

    enabled_names = {
        "information_time",
        "information_weather",
        "information_sports_query",
        "information_web_query",
    }
    return [
        item
        for item in information_model_tools()
        if isinstance(item.get("function"), dict)
        and item["function"].get("name") in enabled_names
    ]


def capability_model_tools(
    *, include_spotify: bool, include_lg_tv: bool = False
) -> list[dict[str, object]]:
    """Return safe intent tools used by the primary model router."""

    text = {"type": "string"}
    tools = [
        *active_information_model_tools(),
        _function_tool(
            "external_action_blocked",
            "Clasifica una solicitud que pretende ejecutar una accion en internet fuera "
            "de las integraciones autorizadas, como comprar, reservar, iniciar sesion, "
            "crear una cuenta, enviar un formulario o publicar contenido. Esta tool no "
            "ejecuta la accion; permite que el codigo la rechace de forma segura.",
            {
                "action": {
                    "type": "string",
                    "enum": [
                        "purchase",
                        "booking",
                        "login",
                        "account",
                        "form",
                        "publish",
                        "other",
                    ],
                }
            },
            ["action"],
        ),
    ]
    integrations: list[dict[str, object]] = []
    if include_spotify:
        integrations.extend(spotify_model_tools())
    if include_lg_tv:
        integrations.extend(lg_tv_model_tools())
    if include_spotify and include_lg_tv:
        integrations.extend(media_model_tools())
    return [*integrations, *tools]


def anthropic_capability_tools(
    *, include_spotify: bool, include_lg_tv: bool = False
) -> list[dict[str, object]]:
    """Convert the safe routing catalog to Anthropic's tool format."""

    converted: list[dict[str, object]] = []
    for item in capability_model_tools(
        include_spotify=include_spotify, include_lg_tv=include_lg_tv
    ):
        function = item["function"]
        assert isinstance(function, dict)
        converted.append(
            {
                "name": function["name"],
                "description": function["description"],
                "input_schema": function["parameters"],
            }
        )
    return converted


def anthropic_information_tools(
    web_allowed_domains: list[str] | None = None,
) -> list[dict[str, object]]:
    """Return read-only client tools plus Anthropic's server-side web search."""

    converted: list[dict[str, object]] = []
    for item in information_model_tools():
        function = item["function"]
        assert isinstance(function, dict)
        converted.append(
            {
                "name": function["name"],
                "description": function["description"],
                "input_schema": function["parameters"],
            }
        )
    web_tool: dict[str, object] = {
        "type": "web_search_20250305",
        "name": "web_search",
        "max_uses": 3,
        "user_location": {
            "type": "approximate",
            "city": "Santiago",
            "region": "Region Metropolitana",
            "country": "CL",
            "timezone": "America/Santiago",
        },
    }
    if web_allowed_domains:
        web_tool["allowed_domains"] = web_allowed_domains
    converted.append(web_tool)
    if web_allowed_domains:
        converted.append(
            {
                "type": "web_fetch_20250910",
                "name": "web_fetch",
                "max_uses": 1,
                "allowed_domains": web_allowed_domains,
                "citations": {"enabled": True},
                "max_content_tokens": 20000,
            }
        )
    return converted


def resolve_information_tool_call(name: str, args: dict[str, object]) -> ToolCall:
    """Validate a model request and map only read-only information calls."""

    if name == "information_time":
        # No timezone and no place means the device's local time. A place
        # without a zone is kept: the executor looks its zone up, and the
        # orchestrator drops it first unless the user actually named it.
        result: dict[str, object] = {}
        timezone = args.get("timezone")
        if isinstance(timezone, str) and timezone.strip():
            result["timezone"] = timezone.strip()
        location = args.get("location")
        if isinstance(location, str) and location.strip():
            result["location"] = location.strip()
        return ToolCall("information.time", result)
    if name == "information_weather":
        return ToolCall(
            "information.weather",
            {
                "location": _required_text(args, "location"),
                "when": _enum(args, "when", {"current", "today", "tomorrow"}, "current"),
            },
        )
    if name == "information_sports_query":
        result = {
            "intent": _enum(
                args,
                "intent",
                {
                    "latest_result",
                    "live_result",
                    "event_result",
                    "schedule",
                    "standings",
                },
            ),
            "sport": _enum(
                args,
                "sport",
                {
                    "football",
                    "basketball",
                    "tennis",
                    "athletics",
                    "american_football",
                    "motorsport",
                    "rugby",
                    "other",
                },
            ),
            "competition": _required_text(args, "competition"),
            "date_reference": _required_text(args, "date_reference"),
        }
        for key in (
            "participant",
            "opponent",
            "event",
            "country",
        ):
            value = args.get(key)
            if (
                isinstance(value, str)
                and value.strip()
                and value.strip().casefold()
                not in {"unspecified", "unknown", "no especificado", "ninguno"}
            ):
                result[key] = value.strip()
        if (
            "participant" not in result
            and "event" not in result
            and str(result["competition"]).casefold() == "unspecified"
        ):
            raise ValueError(
                "La consulta deportiva necesita un participante, evento o competicion."
            )
        return ToolCall("information.sports_query", result)
    if name == "information_web_query":
        result = {
            "query": _required_text(args, "query"),
            "category": _enum(
                args,
                "category",
                {"price", "news", "current_fact", "other"},
            ),
        }
        for key in ("location", "date_reference"):
            value = args.get(key)
            if isinstance(value, str) and value.strip():
                result[key] = value.strip()
        return ToolCall("information.web_query", result)
    raise ValueError(f"El modelo solicito una herramienta informativa no permitida: {name}.")


def resolve_capability_tool_call(name: str, args: dict[str, object]) -> ToolCall:
    """Validate one decision from the model-first capability router."""

    if name == "information_deferred":
        category = str(args.get("category", "other")).strip().casefold()
        if category not in {"sports", "price", "news", "current_fact", "other"}:
            raise ValueError("La categoria de informacion diferida no es valida.")
        query = args.get("query")
        if not isinstance(query, str) or not query.strip():
            query = "consulta deportiva" if category == "sports" else "consulta web"
        return ToolCall(
            "information.deferred",
            {"query": query.strip(), "category": category},
        )
    if name.startswith("information_"):
        return resolve_information_tool_call(name, args)
    if name == "external_action_blocked":
        return ToolCall(
            "external.blocked",
            {
                "action": _enum(
                    args,
                    "action",
                    {
                        "purchase",
                        "booking",
                        "login",
                        "account",
                        "form",
                        "publish",
                        "other",
                    },
                )
            },
        )
    if name.startswith("lg_tv_"):
        return resolve_lg_tv_tool_call(name, args)
    if name == "spotify_play_on_tv":
        query = _required_text(args, "query")
        if len(query) > 200:
            raise ValueError("La busqueda de Spotify es demasiado larga.")
        item_type = _enum(
            args,
            "item_type",
            {
                "auto",
                "track",
                "artist",
                "playlist",
                "album",
                "show",
                "episode",
            },
            "auto",
        )
        library_only = _boolean(args, "library_only", False)
        if library_only and item_type not in {"auto", "playlist"}:
            raise ValueError("library_only solo es valido para playlists personales.")
        return ToolCall(
            "media.spotify_on_tv",
            {
                "query": query,
                "item_type": item_type,
                "shuffle": _boolean(args, "shuffle", False),
                "library_only": library_only,
            },
        )
    return resolve_model_tool_call(name, args)


def resolve_lg_tv_tool_call(name: str, args: dict[str, object]) -> ToolCall:
    """Validate a model request and map only allowlisted LG TV operations."""

    if name == "lg_tv_status":
        detail = _enum(
            args,
            "detail",
            {"full", "power", "app", "channel", "screen", "sound", "volume", "mute"},
            "full",
        )
        return ToolCall("lg_tv.status", {"detail": detail})
    if name == "lg_tv_power":
        action = _enum(args, "action", {"on", "off"})
        return ToolCall(f"lg_tv.power_{action}", {})
    if name == "lg_tv_volume":
        action = _enum(args, "action", {"up", "down", "set"})
        if action == "up":
            return ToolCall("lg_tv.volume_up", {})
        if action == "down":
            return ToolCall("lg_tv.volume_down", {})
        volume = args.get("volume")
        if (
            not isinstance(volume, int)
            or isinstance(volume, bool)
            or not 0 <= volume <= 100
        ):
            raise ValueError("El volumen debe ser un entero entre 0 y 100.")
        return ToolCall("lg_tv.set_volume", {"volume": volume})
    if name == "lg_tv_mute":
        muted = args.get("muted")
        if not isinstance(muted, bool):
            raise ValueError("El estado de silencio debe ser verdadero o falso.")
        return ToolCall("lg_tv.set_mute", {"muted": muted})
    if name == "lg_tv_launch_app":
        app_name = _required_text(args, "app_name")
        if len(app_name) > 80:
            raise ValueError("El nombre de la aplicacion es demasiado largo.")
        return ToolCall("lg_tv.launch_app", {"app_name": app_name})
    if name == "lg_tv_select_input":
        input_name = _enum(
            args,
            "input",
            {"HDMI 1", "HDMI 2", "HDMI 3", "HDMI 4"},
        )
        return ToolCall("lg_tv.select_input", {"input": input_name})
    if name == "lg_tv_playback":
        action = _enum(
            args,
            "action",
            {"play", "pause", "stop", "rewind", "fast_forward"},
        )
        return ToolCall("lg_tv.playback", {"action": action})
    if name == "lg_tv_navigation":
        action = _enum(
            args,
            "action",
            {"home", "back", "up", "down", "left", "right", "ok", "close_app"},
        )
        if action == "close_app":
            return ToolCall("lg_tv.close_app", {})
        return ToolCall("lg_tv.navigate", {"action": action})
    if name == "lg_tv_channel":
        action = _enum(args, "action", {"up", "down", "select"})
        if action == "select":
            query = _required_text(args, "query")
            if len(query) > 80:
                raise ValueError("El nombre o numero de canal es demasiado largo.")
            return ToolCall("lg_tv.select_channel", {"query": query})
        return ToolCall(f"lg_tv.channel_{action}", {})
    if name == "lg_tv_screen":
        on = args.get("on")
        if not isinstance(on, bool):
            raise ValueError("El estado de la pantalla debe ser verdadero o falso.")
        return ToolCall(f"lg_tv.screen_{'on' if on else 'off'}", {})
    if name == "lg_tv_show_message":
        message = " ".join(_required_text(args, "message").split())
        if len(message) > 160:
            raise ValueError("El mensaje para la television es demasiado largo.")
        return ToolCall("lg_tv.show_message", {"message": message})
    if name == "lg_tv_timer":
        action = _enum(args, "action", {"power_off", "message", "cancel", "status"})
        if action == "cancel":
            scope = _enum(
                args,
                "scope",
                {"all", "power_off", "messages"},
                "all",
            )
            return ToolCall("lg_tv.cancel_timers", {"scope": scope})
        if action == "status":
            return ToolCall("lg_tv.timer_status", {})
        minutes = args.get("minutes")
        if (
            not isinstance(minutes, int)
            or isinstance(minutes, bool)
            or not 1 <= minutes <= 1440
        ):
            raise ValueError("El temporizador debe estar entre 1 minuto y 24 horas.")
        if action == "power_off":
            return ToolCall("lg_tv.schedule_power_off", {"minutes": minutes})
        message = " ".join(_required_text(args, "message").split())
        if len(message) > 160:
            raise ValueError("El mensaje para la television es demasiado largo.")
        return ToolCall(
            "lg_tv.schedule_message", {"minutes": minutes, "message": message}
        )
    raise ValueError(f"El modelo solicito una herramienta LG TV no permitida: {name}.")


def resolve_model_tool_call(name: str, args: dict[str, object]) -> ToolCall:
    """Validate one model request and map it to an internal executor call."""

    if name == "spotify_play":
        query = _required_text(args, "query")
        item_type = _enum(args, "item_type", {"auto", "track", "artist", "playlist", "album", "show", "episode"}, "auto")
        shuffle = _boolean(args, "shuffle", False)
        library_only = _boolean(args, "library_only", False)
        if library_only:
            if item_type not in {"auto", "playlist"}:
                raise ValueError("library_only solo es valido para playlists.")
            return ToolCall("spotify.play_playlist", {"query": query, "shuffle": shuffle})
        return ToolCall(
            "spotify.play",
            {"query": query, "item_type": item_type, "shuffle": shuffle},
        )

    if name == "spotify_control":
        action = _enum(
            args,
            "action",
            {"pause", "resume", "next", "previous", "current", "shuffle_on", "shuffle_off", "repeat_track", "repeat_context", "repeat_off", "set_volume"},
        )
        simple = {
            "pause": ToolCall("spotify.pause", {}),
            "resume": ToolCall("spotify.resume", {}),
            "next": ToolCall("spotify.next", {}),
            "previous": ToolCall("spotify.previous", {}),
            "current": ToolCall("spotify.current", {}),
            "shuffle_on": ToolCall("spotify.shuffle", {"state": True}),
            "shuffle_off": ToolCall("spotify.shuffle", {"state": False}),
            "repeat_track": ToolCall("spotify.repeat", {"state": "track"}),
            "repeat_context": ToolCall("spotify.repeat", {"state": "context"}),
            "repeat_off": ToolCall("spotify.repeat", {"state": "off"}),
        }
        if action == "set_volume":
            volume = args.get("volume_percent")
            if not isinstance(volume, int) or isinstance(volume, bool) or not 0 <= volume <= 100:
                raise ValueError("El volumen debe ser un entero entre 0 y 100.")
            return ToolCall("spotify.volume", {"volume_percent": volume})
        return simple[action]

    if name == "spotify_queue":
        return ToolCall(
            "spotify.queue",
            {
                "query": _required_text(args, "query"),
                "item_type": _enum(args, "item_type", {"track", "episode"}, "track"),
            },
        )
    if name == "spotify_save":
        if _boolean(args, "current", not bool(str(args.get("query", "")).strip())):
            return ToolCall("spotify.save_current", {})
        return ToolCall(
            "spotify.save",
            {"query": _required_text(args, "query"), "item_type": "track"},
        )
    if name == "spotify_browse_results":
        return ToolCall(
            "spotify.search",
            {
                "query": _required_text(args, "query"),
                "item_type": _enum(args, "item_type", {"track", "artist", "playlist", "album", "show", "episode"}),
            },
        )
    if name == "spotify_play_reference":
        return ToolCall(
            "spotify.play_reference",
            {
                "reference": _required_text(args, "reference"),
                "shuffle": _boolean(args, "shuffle", False),
            },
        )
    if name == "spotify_followup":
        action = _enum(args, "action", {"similar", "play_result"})
        if action == "similar":
            return ToolCall("spotify.play_similar", {})
        index = args.get("index")
        if not isinstance(index, int) or isinstance(index, bool) or not 1 <= index <= 10:
            raise ValueError("El resultado debe ser un numero entre 1 y 10.")
        return ToolCall("spotify.play_result", {"index": index - 1})
    if name == "spotify_playlist_manage":
        return _resolve_playlist_management(args)
    raise ValueError(f"El modelo solicito una herramienta no permitida: {name}.")


def _resolve_playlist_management(args: dict[str, object]) -> ToolCall:
    action = _enum(
        args,
        "action",
        {"create", "rename", "set_visibility", "set_description", "delete", "add_item", "remove_item"},
    )
    playlist = _required_text(args, "playlist")
    if action == "create":
        return ToolCall("spotify.create_playlist", {"name": playlist, "public": _boolean(args, "public", False)})
    if action == "rename":
        return ToolCall("spotify.update_playlist", {"query": playlist, "name": _required_text(args, "new_name")})
    if action == "set_visibility":
        return ToolCall("spotify.update_playlist", {"query": playlist, "public": _boolean(args, "public")})
    if action == "set_description":
        return ToolCall("spotify.update_playlist", {"query": playlist, "description": _required_text(args, "description")})
    if action == "delete":
        return ToolCall("spotify.delete_playlist", {"query": playlist})
    internal = "spotify.add_to_playlist" if action == "add_item" else "spotify.remove_from_playlist"
    return ToolCall(internal, {"playlist_query": playlist, "item_query": _required_text(args, "item_query")})


def _required_text(args: dict[str, object], key: str) -> str:
    value = args.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Falta un valor de texto valido para {key}.")
    return value.strip()


def _boolean(args: dict[str, object], key: str, default: bool | None = None) -> bool:
    value: Any = args.get(key, default)
    if not isinstance(value, bool):
        raise ValueError(f"{key} debe ser verdadero o falso.")
    return value


def _enum(
    args: dict[str, object],
    key: str,
    allowed: set[str],
    default: str | None = None,
) -> str:
    value = args.get(key, default)
    if not isinstance(value, str) or value not in allowed:
        raise ValueError(f"Valor invalido para {key}.")
    return value
