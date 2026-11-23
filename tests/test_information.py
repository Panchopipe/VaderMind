"""Tests for the read-only information capability."""

from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from vadermind.tools.catalog import resolve_information_tool_call
from vadermind.tools.information import (
    InformationError,
    InformationToolsExecutor,
    is_disallowed_external_action_request,
    is_read_only_information_request,
    is_time_or_weather_request,
    is_sports_result_request,
    is_universidad_de_chile_request,
    latest_chilean_team_slug,
    place_mentioned,
)


class FakeResponse:
    def __init__(self, data: dict[str, object], text: str = "") -> None:
        self._data = data
        self.text = text

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, object]:
        return self._data


class FakeHttpClient:
    def __init__(self, responses: list[dict[str, object]], texts: list[str] | None = None) -> None:
        self.responses = responses
        self.texts = texts or []
        self.calls: list[tuple[str, dict[str, object]]] = []

    def get(self, url: str, params: dict[str, object]) -> FakeResponse:
        self.calls.append((url, params))
        data = self.responses.pop(0) if self.responses else {}
        text = self.texts.pop(0) if self.texts else ""
        return FakeResponse(data, text)


class InformationToolsTests(unittest.TestCase):
    def test_information_intent_and_external_actions_are_separate(self) -> None:
        self.assertTrue(is_read_only_information_request("que clima hara manana en Santiago"))
        self.assertTrue(is_read_only_information_request("como estara el tiempo hoy en Santiago"))
        self.assertTrue(
            is_read_only_information_request("como le fue a la U de Chile en su ultimo partido")
        )
        self.assertTrue(is_read_only_information_request("cuanto cuesta una PlayStation 5"))
        self.assertTrue(is_time_or_weather_request("que hora es en Santiago"))
        self.assertTrue(is_time_or_weather_request("como estara el clima manana"))
        self.assertFalse(
            is_time_or_weather_request("como le fue a la U en su ultimo partido")
        )
        self.assertTrue(is_disallowed_external_action_request("compra el segundo"))
        self.assertTrue(is_disallowed_external_action_request("puedes comprarme el segundo"))
        self.assertTrue(is_disallowed_external_action_request("quiero que reserves una mesa"))
        self.assertTrue(is_disallowed_external_action_request("hacer login con una cuenta ficticia"))
        self.assertTrue(is_disallowed_external_action_request("crear una cuenta nueva"))
        self.assertTrue(is_disallowed_external_action_request("inicia sesion en esa tienda"))
        self.assertFalse(is_disallowed_external_action_request("como puedo comprar de forma segura"))
        self.assertTrue(is_sports_result_request("cual fue el resultado del partido de ayer"))
        self.assertTrue(
            is_sports_result_request("como le fue a la U de Chile en su ultimo partido")
        )
        self.assertTrue(
            is_universidad_de_chile_request("como le fue a la U de Chile en su ultimo partido")
        )
        self.assertTrue(is_sports_result_request("quien gano el ultimo Super Bowl"))
        self.assertTrue(
            is_sports_result_request(
                "quien gano los 100 metros masculinos del mundial de atletismo"
            )
        )
        self.assertTrue(is_sports_result_request("cuando juega el proximo partido"))
        us_open_schedule = "quienes juegan los cuartos de final hoy en el US Open?"
        self.assertTrue(is_sports_result_request(us_open_schedule))
        self.assertTrue(is_read_only_information_request(us_open_schedule))
        self.assertTrue(
            is_read_only_information_request(
                "quien fue el ultimo ganado de la champions league"
            )
        )
        self.assertTrue(
            is_sports_result_request(
                "quien fue el ultimo ganado de la champions league"
            )
        )
        self.assertFalse(
            is_universidad_de_chile_request(
                "y como le fue a la U catolica en su ultimo partido"
            )
        )
        self.assertEqual(
            latest_chilean_team_slug(
                "y como le fue a la U catolica en su ultimo partido"
            ),
            "universidad-catolica",
        )
        self.assertEqual(
            latest_chilean_team_slug("como le fue a Colo-Colo en su ultimo partido"),
            "colo-colo",
        )

    def test_time_without_a_place_uses_the_device_clock(self) -> None:
        # The model once picked Europe/Madrid for "¿qué hora es?" on the Pi.
        executor = InformationToolsExecutor(client=FakeHttpClient([]))  # type: ignore[arg-type]

        result = executor.execute("information.time", {})

        self.assertTrue(result["ok"])
        self.assertRegex(result["message"], r"^Son las \d\d:\d\d\.$")
        self.assertEqual(resolve_information_tool_call("information_time", {}).args, {})

    def test_place_without_zone_looks_the_zone_up(self) -> None:
        # On the Pi, "¿y en Madrid?" came back as Chile's time: the model sent
        # the city without its IANA zone and the city was dropped.
        self.assertEqual(
            resolve_information_tool_call("information_time", {"location": "Madrid"}).args,
            {"location": "Madrid"},
        )
        client = FakeHttpClient(
            [{"results": [{"latitude": 40.4, "longitude": -3.7, "timezone": "Europe/Madrid"}]}]
        )
        with tempfile.TemporaryDirectory() as folder:
            cache = Path(folder) / "geo.json"
            executor = InformationToolsExecutor(client=client, geocoding_cache_path=cache)  # type: ignore[arg-type]
            result = executor.execute("information.time", {"location": "Madrid"})
            self.assertRegex(result["message"], r"^En Madrid son las \d\d:\d\d\.$")
            self.assertEqual(len(client.calls), 1)
            self.assertIn("geocoding", client.calls[0][0])
            # Second time it comes from the cache, zone included.
            again = InformationToolsExecutor(client=client, geocoding_cache_path=cache)  # type: ignore[arg-type]
            again.execute("information.time", {"location": "Madrid"})
            self.assertEqual(len(client.calls), 1)

    def test_cached_place_without_zone_is_refreshed_for_time(self) -> None:
        client = FakeHttpClient(
            [{"results": [{"latitude": -33.4, "longitude": -70.6, "timezone": "America/Santiago"}]}]
        )
        with tempfile.TemporaryDirectory() as folder:
            cache = Path(folder) / "geo.json"
            cache.write_text('{"santiago": {"latitude": -33.4, "longitude": -70.6}}', encoding="utf-8")
            executor = InformationToolsExecutor(client=client, geocoding_cache_path=cache)  # type: ignore[arg-type]
            result = executor.execute("information.time", {"location": "Santiago"})
        self.assertIn("En Santiago son las", result["message"])
        self.assertEqual(len(client.calls), 1)

    def test_unknown_place_for_time_is_reported(self) -> None:
        executor = InformationToolsExecutor(client=FakeHttpClient([{"results": []}]))  # type: ignore[arg-type]
        with self.assertRaises(InformationError):
            executor.execute("information.time", {"location": "Narnia"})

    def test_place_mentioned(self) -> None:
        self.assertTrue(place_mentioned("Madrid", "¿Y en Madrid?"))
        self.assertTrue(place_mentioned("Ciudad de México", "y en ciudad de mexico"))
        self.assertFalse(place_mentioned("Madrid", "¿Qué hora es?"))
        self.assertFalse(place_mentioned("Roma", "romance"))

    def test_time_lookup_does_not_make_a_network_request(self) -> None:
        client = FakeHttpClient([])
        executor = InformationToolsExecutor(client=client)  # type: ignore[arg-type]

        result = executor.execute(
            "information.time", {"timezone": "Asia/Tokyo", "location": "Tokio"}
        )

        self.assertTrue(result["ok"])
        self.assertIn("Tokio", result["message"])
        self.assertNotIn("Asia/Tokyo", result["message"])
        self.assertNotIn("2026", result["message"])
        self.assertEqual(client.calls, [])

    def test_weather_uses_only_allowlisted_get_endpoints(self) -> None:
        client = FakeHttpClient(
            [
                {
                    "results": [
                        {
                            "name": "Santiago",
                            "admin1": "Region Metropolitana",
                            "country": "Chile",
                            "latitude": -33.45,
                            "longitude": -70.67,
                        }
                    ]
                },
                {
                    "current": {
                        "temperature_2m": 18.0,
                        "apparent_temperature": 17.0,
                        "weather_code": 1,
                        "wind_speed_10m": 8.0,
                    },
                    "daily": {
                        "weather_code": [1, 61],
                        "temperature_2m_max": [22.0, 19.0],
                        "temperature_2m_min": [9.0, 8.0],
                        "precipitation_probability_max": [10, 70],
                    },
                },
            ]
        )
        executor = InformationToolsExecutor(client=client)  # type: ignore[arg-type]

        result = executor.execute(
            "information.weather", {"location": "Santiago", "when": "tomorrow"}
        )

        self.assertTrue(result["ok"])
        self.assertIn("maxima 19 grados", result["message"])
        self.assertIn("Open-Meteo", result["message"])
        self.assertEqual(len(client.calls), 2)
        self.assertTrue(all(url.startswith("https://") for url, _ in client.calls))

    def test_today_weather_reports_daily_maximum_instead_of_model_guess(self) -> None:
        client = FakeHttpClient(
            [
                {
                    "results": [
                        {
                            "name": "Santiago",
                            "country": "Chile",
                            "latitude": -33.45,
                            "longitude": -70.67,
                        }
                    ]
                },
                {
                    "current": {
                        "temperature_2m": 12.0,
                        "apparent_temperature": 11.0,
                        "weather_code": 2,
                        "wind_speed_10m": 4.0,
                    },
                    "daily": {
                        "weather_code": [2, 1],
                        "temperature_2m_max": [18.6, 19.8],
                        "temperature_2m_min": [6.2, 7.8],
                        "precipitation_probability_max": [0, 17],
                    },
                },
            ]
        )
        executor = InformationToolsExecutor(client=client)  # type: ignore[arg-type]

        result = executor.execute(
            "information.weather", {"location": "Santiago", "when": "today"}
        )

        self.assertIn("maxima 19 grados", result["message"])
        self.assertNotIn("28", result["message"])

    def test_official_chilean_match_page_yields_final_score(self) -> None:
        html = """
        <div class="match-scoreboard__club-title">Universidad de Chile</div>
        <span class="match-scoreboard__score-number">4</span>
        <span class="match-scoreboard__score-number">2</span>
        <div class="match-scoreboard__club-title">Coquimbo Unido</div>
        <span>Tiempo completo</span>
        """
        executor = InformationToolsExecutor(
            client=FakeHttpClient([], texts=[html])  # type: ignore[arg-type]
        )

        result = executor.execute(
            "information.sports_match_page",
            {
                "url": "https://www.campeonatochileno.cl/match/universidad-de-chile-coquimbo-unido-2026-09-05/"
            },
        )

        self.assertIn("Universidad de Chile vencio 4-2 a Coquimbo Unido", result["message"])

    def test_latest_u_de_chile_match_excludes_future_and_youth_fixtures(self) -> None:
        index = """
        <loc>https://www.campeonatochileno.cl/wp-sitemap-posts-anwp_match-1.xml</loc>
        """
        matches = """
        <loc>https://www.campeonatochileno.cl/match/u-de-chile-sub-20-rival-sub-20-2026-09-07/</loc>
        <loc>https://www.campeonatochileno.cl/match/universidad-de-chile-rival-2026-09-13/</loc>
        <loc>https://www.campeonatochileno.cl/match/universidad-de-chile-coquimbo-unido-2026-09-05/</loc>
        <loc>https://www.campeonatochileno.cl/match/everton-universidad-catolica-2026-09-05/</loc>
        <loc>https://www.campeonatochileno.cl/match/huachipato-universidad-de-chile-2026-08-30/</loc>
        """
        latest_html = """
        <div class="match-scoreboard__club-title">Universidad de Chile</div>
        <span class="match-scoreboard__score-number">4</span>
        <span class="match-scoreboard__score-number">2</span>
        <div class="match-scoreboard__club-title">Coquimbo Unido</div>
        <span>Tiempo completo</span>
        """
        catolica_html = """
        <div class="match-scoreboard__club-title">Everton</div>
        <span class="match-scoreboard__score-number">0</span>
        <span class="match-scoreboard__score-number">1</span>
        <div class="match-scoreboard__club-title">Universidad Catolica</div>
        <span>Tiempo completo</span>
        """
        client = FakeHttpClient([], texts=[index, matches, latest_html, catolica_html])
        executor = InformationToolsExecutor(client=client)  # type: ignore[arg-type]

        result = executor.execute(
            "information.latest_universidad_de_chile_match",
            {"on_or_before": "2026-09-07"},
        )

        self.assertIn("Universidad de Chile vencio 4-2 a Coquimbo Unido", result["message"])
        self.assertEqual(len(client.calls), 3)
        self.assertIn("2026-09-05", client.calls[-1][0])

        second = executor.execute(
            "information.latest_chilean_team_match",
            {"team_slug": "universidad-catolica", "on_or_before": "2026-09-07"},
        )

        self.assertIn("Universidad Catolica vencio 1-0 a Everton", second["message"])
        self.assertEqual(len(client.calls), 4)


if __name__ == "__main__":
    unittest.main()
