"""Read-only information tools for time and weather."""

from __future__ import annotations

from datetime import date, datetime
from html import unescape
import json
from pathlib import Path
import re
import threading
import unicodedata
from urllib.parse import unquote, urlparse
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import httpx

from vadermind.warmup import LONG_KEEPALIVE, warm_connection


class InformationError(RuntimeError):
    """Raised when a read-only information provider cannot answer safely."""


READ_ONLY_INFORMATION_MARKERS = (
    "a cuanto esta",
    "averigua",
    "busca en internet",
    "como va el partido",
    "cuando juega",
    "cuando es el partido",
    "consulta en internet",
    "clima",
    "cotizacion",
    "cuanto cuesta",
    "dime la hora en",
    "hora en",
    "en que posicion",
    "informacion actual",
    "marcador",
    "noticia",
    "precio",
    "pronostico",
    "proximo partido",
    "proxima carrera",
    "resultado del partido",
    "resultado en vivo",
    "temperatura",
    "como estara el tiempo",
    "como esta el tiempo",
    "tiempo hace",
    "tiempo hara",
    "tiempo hoy",
    "tiempo manana",
    "llovera",
    "hara calor",
    "hara frio",
    "que hora",
    "quien gano",
    "quien va ganando",
    "tabla de posiciones",
    "ultima hora",
    "ultimo ganador",
    "ultimo ganado",
    "ultimo campeon",
    "campeon vigente",
    "champions league",
)

FOLLOWUP_MARKERS = (
    "y ahora",
    "y manana",
    "y mañana",
    "y el segundo",
    "y la segunda",
    "como va",
    "cuanto cuesta el segundo",
    "cuánto cuesta el segundo",
)

TIME_OR_WEATHER_MARKERS = (
    "dime la hora",
    "hora en",
    "que hora",
    "clima",
    "pronostico",
    "temperatura",
    "como estara el tiempo",
    "como esta el tiempo",
    "tiempo hace",
    "tiempo hara",
    "tiempo hoy",
    "tiempo manana",
    "llovera",
    "hara calor",
    "hara frio",
)

DISALLOWED_EXTERNAL_ACTION_PATTERNS = (
    r"^(?:(?:por favor|puedes|podrias)\s+|(?:quiero|necesito)\s+que\s+)?(?:compra|comprar|comprame|comprarme|compres|adquiere|adquirir|adquieras|reserva|reservar|reserve|reserves|contrata|contratar|contrates)\b",
    r"^(?:(?:por favor|puedes|podrias)\s+|(?:quiero|necesito)\s+que\s+)?(?:inicia|iniciar|abre|haz|hacer)\s+(?:un\s+)?(?:login|sesion)\b",
    r"^(?:(?:por favor|puedes|podrias)\s+|(?:quiero|necesito)\s+que\s+)?(?:crea|crear|registrar|registrame)\s+(?:una\s+)?cuenta\b",
    r"^(?:(?:por favor|puedes|podrias)\s+|(?:quiero|necesito)\s+que\s+)?(?:agrega|agregar|anade|anadir|mete)\b.*\bcarrito\b",
    r"^(?:(?:por favor|puedes|podrias)\s+|(?:quiero|necesito)\s+que\s+)?(?:publica|publicar|envia|enviar)\b.*\b(?:comentario|formulario|mensaje|correo)\b",
)


def _normalize(text: str) -> str:
    decomposed = unicodedata.normalize("NFD", text.casefold())
    without_accents = "".join(char for char in decomposed if unicodedata.category(char) != "Mn")
    return re.sub(r"\s+", " ", without_accents).strip(" .?!¡¿")


def place_mentioned(location: str, text: str) -> bool:
    """True if the user's own words name this place ("¿y en Madrid?" names Madrid)."""

    place = _normalize(location)
    return bool(place) and re.search(rf"\b{re.escape(place)}\b", _normalize(text)) is not None


def is_read_only_information_request(text: str, has_information_context: bool = False) -> bool:
    """Identify requests that need fresh public information."""

    normalized = _normalize(text)
    if is_sports_result_request(text):
        return True
    if any(marker in normalized for marker in READ_ONLY_INFORMATION_MARKERS):
        return True
    return has_information_context and any(
        _normalize(marker) in normalized for marker in FOLLOWUP_MARKERS
    )


def is_time_or_weather_request(text: str) -> bool:
    """Recognize the only information capabilities enabled in the current stage."""

    normalized = _normalize(text)
    return any(_normalize(marker) in normalized for marker in TIME_OR_WEATHER_MARKERS)


def is_disallowed_external_action_request(text: str) -> bool:
    """Reject explicit attempts to mutate websites or accounts."""

    normalized = _normalize(text)
    return any(re.search(pattern, normalized) for pattern in DISALLOWED_EXTERNAL_ACTION_PATTERNS)


def is_sports_result_request(text: str) -> bool:
    """Identify sports results, schedules, rounds, and standings questions."""

    normalized = _normalize(text)
    if any(
        marker in normalized
        for marker in (
            "como va el partido",
            "como le fue",
            "como salio",
            "marcador",
            "resultado del partido",
            "resultado en vivo",
            "ultimo partido",
            "quien gano el partido",
            "quien va ganando",
            "cuando juega",
            "proximo partido",
            "proxima carrera",
            "tabla de posiciones",
            "ultimo ganador",
            "ultimo ganado",
            "ultimo campeon",
            "campeon vigente",
        )
    ):
        return True
    sports_subject = any(
        marker in normalized
        for marker in (
            "partido",
            "equipo",
            "seleccion",
            "liga",
            "campeonato",
            "futbol",
            "basquetbol",
            "basketball",
            "nba",
            "tenis",
            "atp",
            "wta",
            "us open",
            "roland garros",
            "wimbledon",
            "australian open",
            "copa davis",
            "atletismo",
            "100 metros",
            "200 metros",
            "maraton",
            "super bowl",
            "nfl",
            "formula 1",
            "gran premio",
            "rugby",
            "champions league",
            "uefa champions",
            "copa libertadores",
            "copa sudamericana",
        )
    )
    outcome_question = any(
        marker in normalized
        for marker in (
            "gano",
            "perdio",
            "empato",
            "cuando juega",
            "cuando es",
            "jugo ayer",
            "resultado",
            "quien gano",
            "como va",
            "en que posicion",
            "ultimo ganador",
            "ultimo ganado",
            "ultimo campeon",
            "campeon vigente",
            "quien juega",
            "quienes juegan",
            "partidos de hoy",
            "octavos de final",
            "cuartos de final",
            "semifinal",
            "ronda de",
        )
    )
    return sports_subject and outcome_question


def is_universidad_de_chile_request(text: str) -> bool:
    """Resolve common Chilean names for Club Universidad de Chile."""

    return latest_chilean_team_slug(text) == "universidad-de-chile"


def chilean_team_slug(name: str) -> str | None:
    """Normalize a model-interpreted Chilean club name for official sitemap lookup."""

    candidate = _normalize(name)
    candidate = re.sub(r"^(?:el\s+equipo|el\s+club|equipo|club)\s+", "", candidate)
    aliases = {
        "la u": "universidad-de-chile",
        "la u de chile": "universidad-de-chile",
        "u de chile": "universidad-de-chile",
        "universidad de chile": "universidad-de-chile",
        "la u catolica": "universidad-catolica",
        "u catolica": "universidad-catolica",
        "la uc": "universidad-catolica",
        "uc": "universidad-catolica",
        "universidad catolica": "universidad-catolica",
        "colo colo": "colo-colo",
    }
    if candidate in aliases:
        return aliases[candidate]

    candidate = re.sub(r"^(?:el|la)\s+", "", candidate).strip()
    slug = re.sub(r"[^a-z0-9]+", "-", candidate).strip("-")
    return slug if len(slug) >= 3 else None


def latest_chilean_team_slug(text: str) -> str | None:
    """Extract the requested team from a natural-language latest-match question."""

    if not is_sports_result_request(text):
        return None
    normalized = _normalize(text)
    patterns = (
        r"(?:como le fue|como salio)\s+(?:a|al)\s+(.+?)\s+en\s+su\s+ultimo\s+partido",
        r"(?:resultado\s+del\s+)?ultimo\s+partido\s+de\s+(.+)$",
    )
    candidate = ""
    for pattern in patterns:
        match = re.search(pattern, normalized)
        if match:
            candidate = match.group(1).strip()
            break
    if not candidate:
        return None

    return chilean_team_slug(candidate)


class InformationToolsExecutor:
    """Execute only allowlisted GET requests to public information providers."""

    supports_information_tools = True
    _ALLOWED_HOSTS = {"geocoding-api.open-meteo.com", "api.open-meteo.com"}

    _GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"
    _FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

    def __init__(
        self,
        client: httpx.Client | None = None,
        geocoding_cache_path: Path | None = None,
    ) -> None:
        self._client = client or httpx.Client(
            timeout=15.0, follow_redirects=False, limits=LONG_KEEPALIVE
        )
        self._campeonato_match_urls: tuple[str, ...] | None = None
        # A city's coordinates never change: remember them and skip a round trip
        # to Open-Meteo (about 1 s on a fresh connection from Chile).
        self._geocoding_cache_path = geocoding_cache_path
        self._geocoding_cache: dict[str, dict[str, object]] = self._load_geocoding_cache()
        self._geocoding_lock = threading.Lock()

    def warm(self) -> None:
        """Open the forecast connection while the model is still deciding."""

        warm_connection(self._client, self._FORECAST_URL)

    def execute(self, tool_name: str, args: dict[str, object]) -> dict[str, object]:
        if tool_name == "information.time":
            return self.current_time(
                timezone=str(args.get("timezone", "")),
                location=str(args.get("location", "")),
            )
        if tool_name == "information.weather":
            return self.weather(
                location=str(args.get("location", "")),
                when=str(args.get("when", "today")),
            )
        if tool_name == "information.sports_match_page":
            return self.sports_match_page(str(args.get("url", "")))
        if tool_name == "information.latest_universidad_de_chile_match":
            return self.latest_universidad_de_chile_match(
                str(args.get("on_or_before", ""))
            )
        if tool_name == "information.latest_chilean_team_match":
            return self.latest_chilean_team_match(
                team_slug=str(args.get("team_slug", "")),
                on_or_before=str(args.get("on_or_before", "")),
            )
        raise InformationError(f"Herramienta informativa no permitida: {tool_name}.")

    def current_time(self, timezone: str = "", location: str = "") -> dict[str, object]:
        timezone = timezone.strip()
        if not timezone and location.strip():
            # The model named the place but forgot its zone ("¿y en Madrid?" on the
            # Pi answered Chile's time): ask the geocoder, already used for weather.
            timezone = self._timezone_for(location)
        if not timezone:
            # No place named: the device's own clock, set per household.
            now = datetime.now().astimezone()
            message = f"Son las {now:%H:%M}."
        else:
            try:
                now = datetime.now(ZoneInfo(timezone))
            except ZoneInfoNotFoundError as error:
                raise InformationError(f"No reconozco la zona horaria {timezone}.") from error
            label = location.strip() or timezone.rsplit("/", 1)[-1].replace("_", " ")
            message = f"En {label} son las {now:%H:%M}."
        return {
            "ok": True,
            "tool_name": "information.time",
            "message": message,
            "spoken_message": message,
            "observed_at": now.isoformat(),
        }

    def weather(self, location: str, when: str = "today") -> dict[str, object]:
        location = location.strip()
        if not location:
            raise InformationError("Falta la ciudad o lugar para consultar el clima.")
        if when not in {"current", "today", "tomorrow"}:
            raise InformationError("La consulta del clima solo admite ahora, hoy o manana.")

        place = self._geocode(location)

        forecast = self._get_json(
            self._FORECAST_URL,
            {
                "latitude": place.get("latitude"),
                "longitude": place.get("longitude"),
                "current": "temperature_2m,apparent_temperature,weather_code,wind_speed_10m",
                "daily": "weather_code,temperature_2m_max,temperature_2m_min,precipitation_probability_max",
                "timezone": "auto",
                "forecast_days": 2,
            },
        )
        place_name = location
        if when in {"today", "tomorrow"}:
            daily = forecast.get("daily")
            if not isinstance(daily, dict):
                raise InformationError("Open-Meteo no entrego el pronostico diario esperado.")
            index = 1 if when == "tomorrow" else 0
            code = self._daily_value(daily, "weather_code", index)
            maximum = self._daily_value(daily, "temperature_2m_max", index)
            minimum = self._daily_value(daily, "temperature_2m_min", index)
            day = "Manana" if when == "tomorrow" else "Hoy"
            message = (
                f"{day} en {place_name}: maxima {maximum:.0f} grados, "
                f"minima {minimum:.0f} grados y {self._weather_description(code)}."
            )
        else:
            current = forecast.get("current")
            if not isinstance(current, dict):
                raise InformationError("Open-Meteo no entrego las condiciones actuales esperadas.")
            temperature = self._number(current.get("temperature_2m"), "temperatura")
            code = self._number(current.get("weather_code"), "estado del tiempo")
            message = (
                f"Ahora en {place_name}: {temperature:.0f} grados y "
                f"{self._weather_description(code)}."
            )

        return {
            "ok": True,
            "tool_name": "information.weather",
            "message": f"{message}\n\nFuente: Open-Meteo — https://open-meteo.com/",
            "spoken_message": message,
            "source": {"title": "Open-Meteo", "url": "https://open-meteo.com/"},
            "location": place_name,
        }

    def sports_match_page(self, url: str) -> dict[str, object]:
        """Read the final score from an allowlisted Campeonato Chileno match page."""

        if not self._is_campeonato_url(url, path_prefix="/match/"):
            raise InformationError("La pagina del partido no pertenece al proveedor autorizado.")
        html = self._get_campeonato_text(url)
        return self._parse_campeonato_match(html, url)

    def latest_universidad_de_chile_match(
        self, on_or_before: str = ""
    ) -> dict[str, object]:
        """Backward-compatible wrapper for the original team-specific lookup."""

        return self.latest_chilean_team_match("universidad-de-chile", on_or_before)

    def latest_chilean_team_match(
        self, team_slug: str, on_or_before: str = ""
    ) -> dict[str, object]:
        """Find a team's latest completed senior men's match in the official sitemap."""

        team_slug = team_slug.strip().casefold()
        if not re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", team_slug):
            raise InformationError("El nombre normalizado del equipo no es valido.")

        try:
            cutoff = date.fromisoformat(on_or_before) if on_or_before else date.today()
        except ValueError as error:
            raise InformationError("La fecha de referencia del partido no es valida.") from error

        candidates: dict[str, date] = {}
        excluded_categories = ("femenino", "futsal", "sub-", "juvenil")
        for url in self._official_match_urls():
            lowered = _normalize(unquote(url))
            if team_slug not in lowered or any(
                marker in lowered for marker in excluded_categories
            ):
                continue
            match_date = re.search(r"-(20\d{2}-\d{2}-\d{2})/?$", url)
            if not match_date:
                continue
            played_on = date.fromisoformat(match_date.group(1))
            if played_on <= cutoff:
                candidates[url] = played_on

        for url, _played_on in sorted(
            candidates.items(), key=lambda item: item[1], reverse=True
        ):
            try:
                return self.sports_match_page(url)
            except InformationError:
                # A fixture dated today may not be over yet; try the prior one.
                continue
        raise InformationError(
            "No encontre un partido terminado de ese equipo en el sitio oficial."
        )

    def _official_match_urls(self) -> tuple[str, ...]:
        """Load the official match index once per Vader session."""

        if self._campeonato_match_urls is not None:
            return self._campeonato_match_urls

        index_url = "https://www.campeonatochileno.cl/wp-sitemap.xml"
        index = self._get_campeonato_text(index_url)
        sitemap_urls = [
            unescape(url)
            for url in re.findall(r"<loc>(.*?)</loc>", index, flags=re.I)
            if re.search(r"wp-sitemap-posts-anwp_match-\d+\.xml$", unescape(url))
            and self._is_campeonato_url(
                unescape(url), path_prefix="/wp-sitemap-posts-anwp_match-"
            )
        ]
        if not sitemap_urls:
            raise InformationError("El sitio oficial no publico su indice de partidos.")

        match_urls: list[str] = []
        for sitemap_url in sitemap_urls:
            sitemap = self._get_campeonato_text(sitemap_url)
            match_urls.extend(
                unescape(url)
                for url in re.findall(r"<loc>(.*?)</loc>", sitemap, flags=re.I)
                if self._is_campeonato_url(unescape(url), path_prefix="/match/")
            )
        self._campeonato_match_urls = tuple(match_urls)
        return self._campeonato_match_urls

    def _parse_campeonato_match(self, html: str, url: str) -> dict[str, object]:
        """Parse one official scoreboard after its URL has been validated."""

        clubs = [
            self._clean_html_text(value)
            for value in re.findall(
                r'match-scoreboard__club-title[^>]*>\s*([^<]+)', html, flags=re.I
            )
        ]
        scores = [
            int(value)
            for value in re.findall(
                r'match-scoreboard__score-number[^>]*>\s*(\d+)', html, flags=re.I
            )
        ]
        if len(clubs) < 2 or len(scores) < 2 or "Tiempo completo" not in html:
            raise InformationError("La pagina oficial no contiene un marcador final reconocible.")

        home, away = clubs[0], clubs[1]
        home_score, away_score = scores[0], scores[1]
        if home_score == away_score:
            message = f"{home} y {away} empataron {home_score}-{away_score}."
        elif home_score > away_score:
            message = f"{home} vencio {home_score}-{away_score} a {away}."
        else:
            message = f"{away} vencio {away_score}-{home_score} a {home}."
        return {
            "ok": True,
            "tool_name": "information.sports_match_page",
            "message": f"{message}\n\nFuente: Campeonato Chileno — {url}",
            "spoken_message": message,
            "source": {"title": "Campeonato Chileno", "url": url},
        }

    def _get_campeonato_text(self, url: str) -> str:
        if not self._is_campeonato_url(url):
            raise InformationError("La pagina solicitada no pertenece al proveedor autorizado.")
        try:
            response = self._client.get(url, params={})
            response.raise_for_status()
            return response.text
        except httpx.HTTPError as error:
            raise InformationError("No pude leer el sitio oficial del campeonato.") from error

    @staticmethod
    def _is_campeonato_url(url: str, path_prefix: str = "/") -> bool:
        parsed = urlparse(url)
        return (
            parsed.scheme == "https"
            and parsed.hostname in {"campeonatochileno.cl", "www.campeonatochileno.cl"}
            and parsed.path.startswith(path_prefix)
        )

    def _timezone_for(self, location: str) -> str:
        timezone = self._geocode(location, need_timezone=True).get("timezone")
        if not isinstance(timezone, str) or not timezone:
            raise InformationError(f"No encontre la zona horaria de {location}.")
        return timezone

    def _geocode(self, location: str, need_timezone: bool = False) -> dict[str, object]:
        key = " ".join(location.lower().split())
        with self._geocoding_lock:
            cached = self._geocoding_cache.get(key)
        if cached is not None and (not need_timezone or "timezone" in cached):
            return cached
        geocoding = self._get_json(
            self._GEOCODING_URL,
            {"name": location, "count": 1, "language": "es", "format": "json"},
        )
        results = geocoding.get("results")
        if not isinstance(results, list) or not results:
            raise InformationError(f"No encontre una ubicacion llamada {location}.")
        place = results[0]
        if not isinstance(place, dict):
            raise InformationError("Open-Meteo devolvio una ubicacion invalida.")
        entry: dict[str, object] = {
            "latitude": place.get("latitude"),
            "longitude": place.get("longitude"),
        }
        if not all(isinstance(value, (int, float)) for value in entry.values()):
            return entry  # let the forecast request report it; never cache it
        if isinstance(place.get("timezone"), str) and place["timezone"]:
            entry["timezone"] = place["timezone"]
        with self._geocoding_lock:
            self._geocoding_cache[key] = entry
            self._save_geocoding_cache()
        return entry

    def _load_geocoding_cache(self) -> dict[str, dict[str, object]]:
        path = self._geocoding_cache_path
        if path is None or not path.is_file():
            return {}
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}
        if not isinstance(data, dict):
            return {}
        return {
            str(key): {
                "latitude": value["latitude"],
                "longitude": value["longitude"],
                **({"timezone": value["timezone"]} if isinstance(value.get("timezone"), str) else {}),
            }
            for key, value in data.items()
            if isinstance(value, dict)
            and isinstance(value.get("latitude"), (int, float))
            and isinstance(value.get("longitude"), (int, float))
        }

    def _save_geocoding_cache(self) -> None:
        path = self._geocoding_cache_path
        if path is None:
            return
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                json.dumps(self._geocoding_cache, ensure_ascii=False, indent=1),
                encoding="utf-8",
            )
        except OSError:
            pass  # the cache only saves time; losing it is harmless

    def _get_json(self, url: str, params: dict[str, object]) -> dict[str, object]:
        parsed = urlparse(url)
        if parsed.scheme != "https" or parsed.hostname not in self._ALLOWED_HOSTS:
            raise InformationError("El proveedor solicitado no esta autorizado.")
        try:
            response = self._client.get(url, params=params)
            response.raise_for_status()
            data = response.json()
        except (httpx.HTTPError, ValueError) as error:
            raise InformationError("No pude consultar Open-Meteo en este momento.") from error
        if not isinstance(data, dict):
            raise InformationError("Open-Meteo devolvio una respuesta invalida.")
        return data

    @staticmethod
    def _clean_html_text(value: str) -> str:
        return re.sub(r"\s+", " ", unescape(value)).strip()

    @classmethod
    def _daily_value(cls, daily: dict[str, object], key: str, index: int) -> float:
        values = daily.get(key)
        if not isinstance(values, list) or len(values) <= index:
            raise InformationError(f"Falta el dato meteorologico {key}.")
        return cls._number(values[index], key)

    @staticmethod
    def _number(value: object, label: str) -> float:
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise InformationError(f"El valor de {label} no es valido.")
        return float(value)

    @staticmethod
    def _weather_description(code: float) -> str:
        numeric = int(code)
        if numeric == 0:
            return "cielo despejado"
        if numeric in {1, 2, 3}:
            return "cielo parcialmente nublado"
        if numeric in {45, 48}:
            return "niebla"
        if numeric in {51, 53, 55, 56, 57}:
            return "llovizna"
        if numeric in {61, 63, 65, 66, 67, 80, 81, 82}:
            return "lluvia"
        if numeric in {71, 73, 75, 77, 85, 86}:
            return "nieve"
        if numeric in {95, 96, 99}:
            return "tormenta"
        return "condiciones variables"
