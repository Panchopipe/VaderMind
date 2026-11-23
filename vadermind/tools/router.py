"""Rule-based Spanish router for explicit VaderMind tool commands."""

from __future__ import annotations

from dataclasses import dataclass
import re
import unicodedata


@dataclass(frozen=True, slots=True)
class ToolCall:
    """A parsed request to run a tool."""

    name: str
    args: dict[str, object]


_PREFIX = r"^(?:oye\s+)?(?:vader[:,]?\s+)?"
_ENDING = r"[.!?]*$"
_TV = r"(?:televisi[oó]n|televisor|tele|tv)"


def parse_tool_call(user_text: str) -> ToolCall | None:
    """Return a tool call for an explicit supported local integration command."""

    text = " ".join(user_text.strip().split()).rstrip("\\/").strip()
    if not text:
        return None

    text, shuffle_after_play = _extract_shuffle_modifier(text)

    reference_match = re.search(
        r"(https?://open\.spotify\.com/(?:playlist|album|track|episode|show)/[A-Za-z0-9]+(?:\?[^\s]+)?|spotify:(?:playlist|album|track|episode|show):[A-Za-z0-9]+)",
        text,
        re.IGNORECASE,
    )
    if reference_match and re.match(
        _PREFIX + r"(?:pon(?:me)?|reproduce|toca)\b", text, re.IGNORECASE
    ):
        return ToolCall(
            "spotify.play_reference",
            {
                "reference": reference_match.group(1),
                "shuffle": shuffle_after_play,
            },
        )

    normalized = _normalize(text)

    if _matches(
        text,
        rf"(?:cancela|anula)\s+(?:el\s+)?apagado\s+(?:programado\s+)?(?:de\s+)?(?:la\s+|el\s+)?{_TV}",
    ):
        return ToolCall("lg_tv.cancel_timers", {"scope": "power_off"})
    if _matches(
        text,
        rf"(?:cancela|anula)\s+(?:los\s+)?temporizadores\s+(?:de\s+)?(?:la\s+|el\s+)?{_TV}",
    ):
        return ToolCall("lg_tv.cancel_timers", {"scope": "all"})
    if _matches(
        text,
        rf"(?:cu[aá]nto|cuanto)\s+falta\s+para\s+(?:que\s+)?(?:se\s+)?apague\s+(?:la\s+|el\s+)?{_TV}",
    ) or _matches(
        text,
        rf"(?:qu[eé]|que)\s+temporizadores\s+(?:hay|tiene)\s+(?:la\s+|el\s+)?{_TV}",
    ):
        return ToolCall("lg_tv.timer_status", {})
    timer_match = _capture(
        text,
        rf"apaga\s+(?:la\s+|el\s+)?{_TV}\s+(?:en|dentro\s+de)\s+(\d{{1,4}})\s+minutos?",
    )
    if timer_match:
        return ToolCall(
            "lg_tv.schedule_power_off", {"minutes": int(timer_match.group(1))}
        )
    message_timer_match = _capture(
        text,
        rf"(?:muestra|pon)\s+(?:en\s+)?(?:la\s+|el\s+)?{_TV}\s+(?:el\s+mensaje\s+)?(.+?)\s+(?:en|dentro\s+de)\s+(\d{{1,4}})\s+minutos?",
    )
    if message_timer_match:
        return ToolCall(
            "lg_tv.schedule_message",
            {
                "message": _clean(message_timer_match.group(1)),
                "minutes": int(message_timer_match.group(2)),
            },
        )

    for spotify_tv_pattern in (
        rf"(?:pon(?:me)?|reproduce|toca)\s+(.+?)\s+(?:en|por|de)\s+spotify\s+(?:en|por)\s+(?:la\s+|el\s+)?{_TV}",
        rf"(?:pon(?:me)?|reproduce|toca)\s+(?:en|por)\s+spotify\s+(.+?)\s+(?:en|por)\s+(?:la\s+|el\s+)?{_TV}",
    ):
        spotify_tv_match = _capture(text, spotify_tv_pattern)
        if spotify_tv_match:
            raw_query = _clean(spotify_tv_match.group(1))
            playlist_match = re.match(
                r"^(?:(?:mi|la|una)\s+)?(?:playlist|lista)\s+(?:de\s+)?(.+)$",
                raw_query,
                re.IGNORECASE,
            )
            return ToolCall(
                "media.spotify_on_tv",
                {
                    "query": _clean_playlist(
                        playlist_match.group(1) if playlist_match else raw_query
                    ),
                    "item_type": "playlist" if playlist_match else "auto",
                    "library_only": bool(
                        playlist_match
                        and re.match(r"^(?:mi|la)\s+", raw_query, re.IGNORECASE)
                    ),
                    "shuffle": shuffle_after_play,
                },
            )

    if _matches(
        text,
        r"(?:qu[eé]|que|cu[aá]l|cual)\s+canal\s+(?:estoy\s+viendo|tengo\s+puesto)",
    ) or _matches(
        text,
        rf"(?:qu[eé]|que|cu[aá]l|cual)\s+canal\s+(?:est[aá]\s+viendo|tiene)\s+(?:la\s+|el\s+)?{_TV}",
    ):
        return ToolCall("lg_tv.status", {"detail": "channel"})
    if _matches(
        text,
        rf"(?:qu[eé]|que|cu[aá]l|cual)\s+(?:aplicaci[oó]n|app)\s+(?:est[aá]\s+usando|tiene\s+abierta)\s+(?:la\s+|el\s+)?{_TV}",
    ):
        return ToolCall("lg_tv.status", {"detail": "app"})
    if _matches(
        text,
        rf"(?:qu[eé]|que|cu[aá]l|cual)\s+(?:salida\s+de\s+sonido|salida\s+de\s+audio)\s+(?:est[aá]\s+usando|tiene)\s+(?:la\s+|el\s+)?{_TV}",
    ):
        return ToolCall("lg_tv.status", {"detail": "sound"})
    if _matches(
        text,
        rf"(?:qu[eé]|que)\s+volumen\s+tiene\s+(?:la\s+|el\s+)?{_TV}",
    ):
        return ToolCall("lg_tv.status", {"detail": "volume"})
    if _matches(
        text,
        rf"(?:la\s+|el\s+)?{_TV}\s+est[aá]\s+(?:en\s+silencio|silenciada|silenciado)",
    ):
        return ToolCall("lg_tv.status", {"detail": "mute"})
    if _matches(
        text,
        rf"(?:la\s+|el\s+)?{_TV}\s+est[aá]\s+(?:apagada|apagado)\s+(?:o|u|en)\s+(?:encendida|encendido)",
    ) or _matches(
        text,
        rf"(?:la\s+|el\s+)?{_TV}\s+est[aá]\s+(?:encendida|encendido|apagada|apagado)",
    ):
        return ToolCall("lg_tv.status", {"detail": "power"})
    if _matches(
        text,
        rf"la\s+pantalla\s+(?:de\s+)?(?:la\s+|el\s+)?{_TV}\s+est[aá]\s+(?:encendida|encendido|apagada|apagado)",
    ) or _matches(
        text,
        rf"(?:la\s+|el\s+)?{_TV}\s+tiene\s+la\s+pantalla\s+(?:encendida|apagada)",
    ):
        return ToolCall("lg_tv.status", {"detail": "screen"})
    if _matches(
        text,
        rf"(?:(?:cu[aá]l|cual)\s+es\s+el\s+estado|dime\s+el\s+estado|estado)\s+(?:de\s+)?(?:la\s+|el\s+)?{_TV}",
    ):
        return ToolCall("lg_tv.status", {"detail": "full"})
    if _matches(
        text,
        rf"(?:apaga|desactiva)\s+(?:solo\s+)?la\s+pantalla\s+(?:de\s+)?(?:la\s+|el\s+)?{_TV}",
    ):
        return ToolCall("lg_tv.screen_off", {})
    if _matches(
        text,
        rf"(?:enciende|activa)\s+(?:solo\s+)?la\s+pantalla\s+(?:de\s+)?(?:la\s+|el\s+)?{_TV}",
    ):
        return ToolCall("lg_tv.screen_on", {})
    if _matches(text, rf"(?:enciende|prende)\s+(?:la\s+|el\s+)?{_TV}"):
        return ToolCall("lg_tv.power_on", {})
    if _matches(text, rf"apaga\s+(?:la\s+|el\s+)?{_TV}"):
        return ToolCall("lg_tv.power_off", {})
    if _matches(
        text,
        rf"(?:silencia|mutea|pon\s+en\s+silencio)\s+(?:la\s+|el\s+)?{_TV}",
    ):
        return ToolCall("lg_tv.set_mute", {"muted": True})
    if _matches(
        text,
        rf"(?:quita|desactiva)\s+(?:el\s+)?silencio\s+(?:de\s+)?(?:la\s+|el\s+)?{_TV}",
    ):
        return ToolCall("lg_tv.set_mute", {"muted": False})
    match = _capture(
        text,
        rf"(?:pon|ajusta|cambia)\s+(?:el\s+)?volumen\s+(?:de\s+)?(?:la\s+|el\s+)?{_TV}\s+(?:a|al|en)\s+(\d{{1,3}})(?:\s*%)?",
    )
    if match:
        return ToolCall("lg_tv.set_volume", {"volume": int(match.group(1))})
    if _matches(
        text,
        rf"(?:sube|aumenta)\s+(?:el\s+)?volumen\s+(?:de\s+)?(?:la\s+|el\s+)?{_TV}",
    ):
        return ToolCall("lg_tv.volume_up", {})
    if _matches(
        text,
        rf"(?:baja|reduce)\s+(?:el\s+)?volumen\s+(?:de\s+)?(?:la\s+|el\s+)?{_TV}",
    ):
        return ToolCall("lg_tv.volume_down", {})
    if _matches(
        text,
        rf"pausa(?:\s+la\s+reproducci[oó]n|\s+el\s+video|\s+lo\s+que\s+estoy\s+viendo)?\s+(?:en\s+|de\s+)?(?:la\s+|el\s+)?{_TV}",
    ):
        return ToolCall("lg_tv.playback", {"action": "pause"})
    if _matches(
        text,
        rf"(?:reanuda|contin[uú]a|reproduce)(?:\s+la\s+reproducci[oó]n|\s+el\s+video)?\s+(?:en\s+|de\s+)?(?:la\s+|el\s+)?{_TV}",
    ):
        return ToolCall("lg_tv.playback", {"action": "play"})
    if _matches(
        text,
        rf"(?:det[eé]n|para)\s+(?:la\s+reproducci[oó]n|el\s+video)\s+(?:en\s+|de\s+)?(?:la\s+|el\s+)?{_TV}",
    ):
        return ToolCall("lg_tv.playback", {"action": "stop"})
    if _matches(
        text,
        rf"(?:retrocede|rebobina)\s+(?:la\s+reproducci[oó]n\s+)?(?:en\s+|de\s+)?(?:la\s+|el\s+)?{_TV}",
    ):
        return ToolCall("lg_tv.playback", {"action": "rewind"})
    if _matches(
        text,
        rf"adelanta\s+(?:la\s+reproducci[oó]n\s+)?(?:en\s+|de\s+)?(?:la\s+|el\s+)?{_TV}",
    ):
        return ToolCall("lg_tv.playback", {"action": "fast_forward"})
    if _matches(
        text,
        rf"(?:sube|pasa|cambia)\s+(?:un\s+canal|al\s+canal\s+siguiente)\s+(?:en\s+|de\s+)?(?:la\s+|el\s+)?{_TV}",
    ):
        return ToolCall("lg_tv.channel_up", {})
    if _matches(
        text,
        rf"(?:baja|pasa|cambia)\s+(?:un\s+canal|al\s+canal\s+anterior)\s+(?:en\s+|de\s+)?(?:la\s+|el\s+)?{_TV}",
    ):
        return ToolCall("lg_tv.channel_down", {})
    channel_match = _capture(
        text,
        rf"(?:pon|sintoniza|cambia)\s+(?:(?:la\s+|el\s+)?{_TV}\s+)?(?:al|a|en)?\s*(?:el\s+)?canal\s+(.+?)(?:\s+en\s+(?:la\s+|el\s+)?{_TV})?",
    )
    if channel_match:
        return ToolCall(
            "lg_tv.select_channel", {"query": _clean(channel_match.group(1))}
        )
    if _matches(
        text,
        rf"(?:ve|vuelve|abre)\s+(?:al\s+)?inicio\s+(?:de\s+)?(?:la\s+|el\s+)?{_TV}",
    ):
        return ToolCall("lg_tv.navigate", {"action": "home"})
    if _matches(
        text,
        rf"(?:vuelve|ve)\s+atr[aá]s\s+(?:en\s+|de\s+)?(?:la\s+|el\s+)?{_TV}",
    ):
        return ToolCall("lg_tv.navigate", {"action": "back"})
    direction_match = _capture(
        text,
        rf"(?:mu[eé]vete|mueve\s+la\s+selecci[oó]n|ve)\s+(arriba|abajo|a\s+la\s+izquierda|a\s+la\s+derecha)\s+(?:en\s+|de\s+)?(?:la\s+|el\s+)?{_TV}",
    )
    if direction_match:
        action = {
            "arriba": "up",
            "abajo": "down",
            "a la izquierda": "left",
            "a la derecha": "right",
        }[_normalize(direction_match.group(1))]
        return ToolCall("lg_tv.navigate", {"action": action})
    if _matches(
        text,
        rf"(?:presiona|pulsa|aprieta)\s+(?:ok|aceptar|enter)\s+(?:en\s+|de\s+)?(?:la\s+|el\s+)?{_TV}",
    ):
        return ToolCall("lg_tv.navigate", {"action": "ok"})
    if _matches(
        text,
        rf"cierra\s+(?:la\s+)?aplicaci[oó]n\s+(?:actual\s+)?(?:de\s+|en\s+)?(?:la\s+|el\s+)?{_TV}",
    ):
        return ToolCall("lg_tv.close_app", {})
    message_match = _capture(
        text,
        rf"(?:muestra|pon)\s+(?:en\s+)?(?:la\s+|el\s+)?{_TV}\s+(?:el\s+mensaje\s+)?(.+)",
    )
    if message_match:
        return ToolCall(
            "lg_tv.show_message", {"message": _clean(message_match.group(1))}
        )
    match = _capture(
        text,
        rf"(?:cambia|pon)\s+(?:la\s+|el\s+)?{_TV}\s+(?:en|a)\s+(?:la\s+entrada\s+)?hdmi\s*([1-4])",
    )
    if match:
        return ToolCall("lg_tv.select_input", {"input": f"HDMI {match.group(1)}"})
    match = _capture(
        text,
        rf"(?:abre|inicia|lanza)\s+(.+?)\s+en\s+(?:la\s+|el\s+)?{_TV}",
    )
    if match:
        return ToolCall("lg_tv.launch_app", {"app_name": _clean(match.group(1))})

    simple_commands: tuple[tuple[set[str], str, dict[str, object]], ...] = (
        ({"pausa", "pausa spotify", "pausa la musica", "deten la musica"}, "spotify.pause", {}),
        (
            {
                "reanuda",
                "continua",
                "sigue",
                "reanuda spotify",
                "reanuda la musica",
                "reanuda la cancion",
                "reanuda la reproduccion",
                "continua la musica",
                "continua la cancion",
                "continua la reproduccion",
                "sigue con la musica",
                "vuelve a reproducir",
            },
            "spotify.resume",
            {},
        ),
        ({"siguiente", "siguiente cancion", "pon la siguiente cancion", "pon siguiente cancion", "salta esta cancion", "salta a la siguiente", "pasa a la siguiente"}, "spotify.next", {}),
        ({"anterior", "cancion anterior", "pon la cancion anterior", "pon la anterior", "vuelve a la anterior", "pasa a la anterior"}, "spotify.previous", {}),
        ({"que esta sonando", "que cancion esta sonando", "que estoy escuchando", "que suena"}, "spotify.current", {}),
        ({"activa aleatorio", "activa el modo aleatorio", "mezcla la musica", "shuffle on"}, "spotify.shuffle", {"state": True}),
        ({"desactiva aleatorio", "desactiva el modo aleatorio", "quita el aleatorio", "shuffle off"}, "spotify.shuffle", {"state": False}),
        ({"repite esta cancion", "repite la cancion", "repetir cancion"}, "spotify.repeat", {"state": "track"}),
        ({"repite la playlist", "repite el album", "repetir contexto"}, "spotify.repeat", {"state": "context"}),
        ({"desactiva repeticion", "quita la repeticion", "no repitas", "repeticion desactivada"}, "spotify.repeat", {"state": "off"}),
        ({"dale me gusta", "me gusta esta cancion", "guarda esta cancion", "guarda lo que suena"}, "spotify.save_current", {}),
        ({"pon algo parecido", "pon algo parecido a esto", "pon algo parecido a lo anterior", "reproduce algo similar"}, "spotify.play_similar", {}),
        ({"si eliminar", "si, eliminar", "confirmo", "confirmar eliminacion"}, "spotify.confirm", {}),
        ({"no eliminar", "cancelar", "cancela", "cancela la eliminacion"}, "spotify.cancel", {}),
    )
    without_prefix = _strip_prefix(normalized)
    if re.fullmatch(
        r"(?:por que|porque)\s+(?:pusiste|reprodujiste|elegiste)\s+.+",
        without_prefix,
    ):
        return ToolCall("spotify.explain_last", {})
    for phrases, name, args in simple_commands:
        if without_prefix in phrases:
            return ToolCall(name, dict(args))

    match = _capture(text, r"(?:pon|ajusta|cambia)\s+(?:el\s+)?volumen\s+(?:a|al|en)\s+(\d{1,3})(?:\s*%)?")
    if match:
        return ToolCall("spotify.volume", {"volume_percent": int(match.group(1))})
    if _matches(text, r"(?:sube|aumenta)\s+(?:el\s+)?volumen"):
        return ToolCall("spotify.volume", {"delta": 10})
    if _matches(text, r"(?:baja|reduce)\s+(?:el\s+)?volumen"):
        return ToolCall("spotify.volume", {"delta": -10})

    if _matches(text, r"(?:muestra|lista|lee|dime|ver)\s+(?:mis\s+)?playlists(?:\s+(?:de|en)\s+spotify)?") or _matches(text, r"(?:cuales|cu[aá]les)\s+son\s+(?:mis\s+)?playlists(?:\s+(?:de|en)\s+spotify)?") or _matches(text, r"spotify[:,]?\s+(?:mis\s+)?playlists"):
        return ToolCall("spotify.list_playlists", {"limit": 10})

    match = _capture(text, r"crea(?:me)?\s+(?:una\s+)?playlist(?:\s+(?:llamada|de nombre))?\s+(.+?)")
    if match:
        return ToolCall("spotify.create_playlist", {"name": _clean(match.group(1)), "public": False})
    match = _capture(text, r"(?:elimina|borra|quita)\s+(?:mi\s+|la\s+)?playlist\s+(.+?)")
    if match:
        return ToolCall("spotify.delete_playlist", {"query": _clean(match.group(1))})
    match = _capture(text, r"(?:renombra|cambia\s+el\s+nombre\s+de)\s+(?:mi\s+|la\s+)?playlist\s+(.+?)\s+(?:a|como)\s+(.+?)")
    if match:
        return ToolCall("spotify.update_playlist", {"query": _clean(match.group(1)), "name": _clean(match.group(2))})
    match = _capture(text, r"(?:haz|pon)\s+(?:mi\s+|la\s+)?playlist\s+(.+?)\s+(publica|p[uú]blica|privada)")
    if match:
        return ToolCall("spotify.update_playlist", {"query": _clean(match.group(1)), "public": _normalize(match.group(2)) == "publica"})
    match = _capture(text, r"(?:cambia|pon)\s+la\s+descripcion\s+de\s+(?:mi\s+|la\s+)?playlist\s+(.+?)\s+(?:a|por)\s+(.+?)")
    if match:
        return ToolCall("spotify.update_playlist", {"query": _clean(match.group(1)), "description": _clean(match.group(2))})
    match = _capture(text, r"(?:agrega|a[nñ]ade)\s+(.+?)\s+(?:a|en)\s+(?:mi\s+|la\s+)?playlist\s+(.+?)")
    if match:
        return ToolCall("spotify.add_to_playlist", {"item_query": _clean(match.group(1)), "playlist_query": _clean(match.group(2))})
    match = _capture(text, r"(?:quita|elimina|saca)\s+(.+?)\s+de\s+(?:mi\s+|la\s+)?playlist\s+(.+?)")
    if match:
        return ToolCall("spotify.remove_from_playlist", {"item_query": _clean(match.group(1)), "playlist_query": _clean(match.group(2))})

    match = _capture(text, r"(?:agrega|a[nñ]ade|pon)\s+(?:(?:el|la)\s+)?(?:(cancion|canci[oó]n|tema|episodio)\s+)?(.+?)\s+(?:a|en)\s+la\s+cola")
    if match:
        item_type = "episode" if _normalize(match.group(1) or "") == "episodio" else "track"
        return ToolCall("spotify.queue", {"query": _clean(match.group(2)), "item_type": item_type})

    match = _capture(text, r"(?:dale\s+me\s+gusta\s+a|guarda)\s+(?:la\s+cancion\s+|el\s+tema\s+)?(.+?)")
    if match:
        return ToolCall("spotify.save", {"query": _clean(match.group(1)), "item_type": "track"})

    ordinal = _parse_ordinal_followup(text)
    if ordinal is not None:
        return ToolCall("spotify.play_result", {"index": ordinal})

    match = _capture(text, r"(?:busca|encuentra)\s+(?:el\s+|un\s+|una\s+)?(album|[aá]lbum|podcast|episodio)\s+(.+?)(?:\s+en\s+spotify)?")
    if match:
        return ToolCall("spotify.search", {"query": _clean(match.group(2)), "item_type": _content_type(match.group(1))})
    match = _capture(text, r"(?:pon(?:me)?|reproduce|toca)\s+(?:el\s+|un\s+|una\s+)?(album|[aá]lbum|artista|podcast|episodio)\s+(.+?)(?:\s+(?:en|por)\s+spotify)?")
    if match:
        return ToolCall("spotify.play", {"query": _clean(match.group(2)), "item_type": _content_type(match.group(1))})

    # "Una playlist de X" asks Spotify's catalog for a playlist. In contrast,
    # "mi/la lista X" asks for a playlist saved in the user's library.
    match = _capture(
        text,
        r"(?:pon(?:me)?|reproduce|toca)\s+una\s+(?:playlist|lista)\s+(?:de\s+)?(.+?)(?:\s+(?:en|por)\s+spotify)?",
    )
    if match:
        return ToolCall(
            "spotify.play",
            {"query": _clean_playlist(match.group(1)), "item_type": "playlist"},
        )

    playlist_body = (
        r"(?:pon(?:me)?|reproduce|toca)\s+"
        r"(?:(?:mi|la)\s+)?(?:playlist|lista)\s+(.+?)"
        r"(?:\s+(?:en|por)\s+spotify)?"
    )
    match = _capture(text, playlist_body)
    if match:
        return ToolCall(
            "spotify.play_playlist",
            {"query": _clean_playlist(match.group(1)), "shuffle": shuffle_after_play},
        )
    match = _capture(text, r"(?:pon(?:me)?|reproduce|toca)\s+(?:en|por)\s+spotify\s+(?:(?:mi|la)\s+)?(?:playlist|lista)\s+(.+?)")
    if match:
        return ToolCall(
            "spotify.play_playlist",
            {"query": _clean_playlist(match.group(1)), "shuffle": shuffle_after_play},
        )
    match = _capture(
        text,
        r"(?:en|por)\s+spotify[:,]?\s+" + playlist_body,
    )
    if match:
        return ToolCall(
            "spotify.play_playlist",
            {"query": _clean_playlist(match.group(1)), "shuffle": shuffle_after_play},
        )

    for pattern in (
        r"(?:pon(?:me)?|reproduce|toca)\s+(.+?)\s+(?:en|por)\s+spotify",
        r"(?:pon(?:me)?|reproduce|toca)\s+(?:en|por)\s+spotify\s+(.+?)",
        r"(?:en|por)\s+spotify[:,]?\s+(?:pon(?:me)?|reproduce|toca)\s+(.+?)",
        r"spotify[:,]?\s+(.+?)",
    ):
        match = _capture(text, pattern)
        if match:
            return ToolCall("spotify.play", {"query": _clean(match.group(1)), "item_type": "auto"})
    return None


def _parse_ordinal_followup(text: str) -> int | None:
    normalized = _strip_prefix(_normalize(text))
    match = re.match(r"^(?:ahora\s+)?(?:pon|reproduce|toca)\s+(?:la|el)\s+(primera|primero|segunda|segundo|tercera|tercero|cuarta|cuarto|quinta|quinto|sexta|sexto|septima|septimo|octava|octavo|novena|noveno|decima|decimo|\d+)$", normalized)
    if not match:
        return None
    words = {"primera": 1, "primero": 1, "segunda": 2, "segundo": 2, "tercera": 3, "tercero": 3, "cuarta": 4, "cuarto": 4, "quinta": 5, "quinto": 5, "sexta": 6, "sexto": 6, "septima": 7, "septimo": 7, "octava": 8, "octavo": 8, "novena": 9, "noveno": 9, "decima": 10, "decimo": 10}
    token = match.group(1)
    return (int(token) if token.isdigit() else words[token]) - 1


def _content_type(value: str) -> str:
    return {
        "album": "album",
        "artista": "artist",
        "podcast": "show",
        "episodio": "episode",
    }[_normalize(value)]


def _matches(text: str, body: str) -> bool:
    return re.match(_PREFIX + body + _ENDING, text, re.IGNORECASE) is not None


def _capture(text: str, body: str) -> re.Match[str] | None:
    return re.match(_PREFIX + body + _ENDING, text, re.IGNORECASE)


def _clean_playlist(query: str) -> str:
    cleaned = _clean(query)
    cleaned = re.sub(r"^(?:llamada|llamado|de nombre|de)\s+", "", cleaned, flags=re.IGNORECASE)
    return _clean(cleaned)


def _clean(query: str) -> str:
    cleaned = query.strip().strip("\"'` ")
    cleaned = re.sub(r"\s*,?\s*(?:por favor|please)[.!?]*$", "", cleaned, flags=re.IGNORECASE)
    return cleaned.strip().strip("\"'` ")


def _strip_prefix(text: str) -> str:
    return re.sub(r"^(?:oye\s+)?(?:vader[:,]?\s+)?", "", text).strip().rstrip(".!?")


def _normalize(text: str) -> str:
    decomposed = unicodedata.normalize("NFKD", text.casefold())
    without_accents = "".join(char for char in decomposed if not unicodedata.combining(char))
    return " ".join(without_accents.split())


def _extract_shuffle_modifier(text: str) -> tuple[str, bool]:
    """Remove a trailing shuffle instruction without polluting the item name."""

    trailing_pattern = (
        r"(?:\s*[,;.]\s*|\s+)"
        r"(?:(?:y\s+)?(?:pon(?:lo|la)?|ponga(?:lo|la)?|activa(?:lo|la)?)\s+)?"
        r"(?:en\s+(?:modo\s+)?aleatorio|modo\s+aleatorio)"
        r"[.!?\\/]*$"
    )
    before_spotify_pattern = (
        r"(?:\s*[,;.]\s*|\s+)"
        r"(?:(?:y\s+)?(?:pon(?:lo|la)?|ponga(?:lo|la)?|activa(?:lo|la)?)\s+)?"
        r"(?:en\s+(?:modo\s+)?aleatorio|modo\s+aleatorio)"
        r"(?=\s+(?:en|por)\s+spotify[.!?\\/]*$)"
    )
    for pattern in (trailing_pattern, before_spotify_pattern):
        if not re.search(pattern, text, re.IGNORECASE):
            continue
        cleaned = re.sub(pattern, "", text, flags=re.IGNORECASE).strip()
        if _strip_prefix(_normalize(cleaned)) in {"activa", "activa el"}:
            return text, False
        return cleaned.rstrip(".,;!? "), True
    return text, False
