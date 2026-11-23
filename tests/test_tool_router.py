"""Tests for the simple tool command router."""

from __future__ import annotations

import unittest

from vadermind.tools.router import parse_tool_call


class ToolRouterTests(unittest.TestCase):
    """Verify explicit user commands map to tool calls."""

    def test_parse_spotify_command_at_end(self) -> None:
        """A natural Spotify command should produce a spotify.play tool call."""

        call = parse_tool_call("Vader, pon Imperial March en Spotify")

        self.assertIsNotNone(call)
        assert call is not None
        self.assertEqual(call.name, "spotify.play")
        self.assertEqual(call.args["query"], "Imperial March")

    def test_parse_spotify_command_at_start(self) -> None:
        """The router should also understand 'pon en Spotify X'."""

        call = parse_tool_call("pon en spotify Daft Punk por favor")

        self.assertIsNotNone(call)
        assert call is not None
        self.assertEqual(call.args["query"], "Daft Punk")

    def test_parse_spotify_playlist_list_request(self) -> None:
        """The router should understand requests to list user playlists."""

        call = parse_tool_call("Vader, muestra mis playlists de Spotify")

        self.assertIsNotNone(call)
        assert call is not None
        self.assertEqual(call.name, "spotify.list_playlists")
        self.assertEqual(call.args["limit"], 10)

    def test_parse_spotify_user_playlist_request(self) -> None:
        """The router should detect requests for a user's named playlist."""

        call = parse_tool_call("pon mi playlist de entrenamiento en Spotify")

        self.assertIsNotNone(call)
        assert call is not None
        self.assertEqual(call.name, "spotify.play_playlist")
        self.assertEqual(call.args["query"], "entrenamiento")

    def test_parse_lista_synonym_and_shuffle_modifier(self) -> None:
        """The wording from the real failure should target a saved playlist."""

        plain = parse_tool_call("pon la lista New UK Garage en spotify")
        shuffled = parse_tool_call(
            "pon la lista New UK Garage en spotify en modo aleatorio"
        )

        assert plain is not None and shuffled is not None
        self.assertEqual(plain.name, "spotify.play_playlist")
        self.assertEqual(plain.args, {"query": "New UK Garage", "shuffle": False})
        self.assertEqual(shuffled.name, "spotify.play_playlist")
        self.assertEqual(
            shuffled.args, {"query": "New UK Garage", "shuffle": True}
        )

    def test_parse_real_playlist_phrasings_without_name_pollution(self) -> None:
        """Shuffle, quotes, punctuation, and Spotify-first wording stay outside the name."""

        cases = {
            "pon mi lista de jazz en spotify en aleatorio": (
                "spotify.play_playlist",
                {"query": "jazz", "shuffle": True},
            ),
            "pon mi playlist de jazz en aleatorio en spotify": (
                "spotify.play_playlist",
                {"query": "jazz", "shuffle": True},
            ),
            "pon mi lista jazz en spotify en aleatorio\\": (
                "spotify.play_playlist",
                {"query": "jazz", "shuffle": True},
            ),
            "en spotify, pon mi lista llamada jazz. ponga en modo aleatorio": (
                "spotify.play_playlist",
                {"query": "jazz", "shuffle": True},
            ),
            'pon mi lista llamada "Jazz" en spotify': (
                "spotify.play_playlist",
                {"query": "Jazz", "shuffle": False},
            ),
        }
        for phrase, expected in cases.items():
            with self.subTest(phrase=phrase):
                call = parse_tool_call(phrase)
                self.assertIsNotNone(call)
                assert call is not None
                self.assertEqual((call.name, call.args), expected)

    def test_catalog_playlist_and_expanded_next_command(self) -> None:
        """A requested catalog playlist must not degrade into an automatic track search."""

        playlist = parse_tool_call("pon una playlist de juan gabriel en spotify")
        next_track = parse_tool_call("pon la siguiente cancion")

        assert playlist is not None and next_track is not None
        self.assertEqual(
            (playlist.name, playlist.args),
            ("spotify.play", {"query": "juan gabriel", "item_type": "playlist"}),
        )
        self.assertEqual(next_track.name, "spotify.next")

    def test_parse_explanation_of_last_spotify_choice(self) -> None:
        call = parse_tool_call("por que pusiste esa cancion?")

        self.assertIsNotNone(call)
        assert call is not None
        self.assertEqual(call.name, "spotify.explain_last")

    def test_parse_direct_spotify_playlist_link(self) -> None:
        """A Spotify link should bypass catalog discovery and preserve shuffle."""

        call = parse_tool_call(
            "pon https://open.spotify.com/playlist/37i9dQZF1DX6MNYaY0PcFh?si=test "
            "en modo aleatorio"
        )

        self.assertIsNotNone(call)
        assert call is not None
        self.assertEqual(call.name, "spotify.play_reference")
        self.assertEqual(call.args["shuffle"], True)
        self.assertIn("37i9dQZF1DX6MNYaY0PcFh", str(call.args["reference"]))

    def test_parse_non_tool_message_returns_none(self) -> None:
        """Ordinary conversation should continue to the language model."""

        self.assertIsNone(parse_tool_call("Hola Vader, como estas?"))

    def test_parse_extended_playback_controls(self) -> None:
        """Playback, volume, shuffle, repeat, and status commands should route locally."""

        cases = {
            "pausa la musica": ("spotify.pause", {}),
            "reanuda": ("spotify.resume", {}),
            "reanuda la musica": ("spotify.resume", {}),
            "reanuda la canción": ("spotify.resume", {}),
            "continúa la reproducción": ("spotify.resume", {}),
            "siguiente cancion": ("spotify.next", {}),
            "vuelve a la anterior": ("spotify.previous", {}),
            "pon el volumen al 35%": ("spotify.volume", {"volume_percent": 35}),
            "activa el modo aleatorio": ("spotify.shuffle", {"state": True}),
            "repite esta cancion": ("spotify.repeat", {"state": "track"}),
            "que esta sonando": ("spotify.current", {}),
        }
        for phrase, expected in cases.items():
            with self.subTest(phrase=phrase):
                call = parse_tool_call(phrase)
                self.assertIsNotNone(call)
                assert call is not None
                self.assertEqual((call.name, call.args), expected)

    def test_parse_queue_save_and_typed_search(self) -> None:
        """Queue, library, and catalog types should preserve their arguments."""

        queue = parse_tool_call("agrega el episodio Marte a la cola")
        album = parse_tool_call("busca el album Discovery en Spotify")
        save = parse_tool_call("guarda esta cancion")

        assert queue is not None and album is not None and save is not None
        self.assertEqual(queue.name, "spotify.queue")
        self.assertEqual(queue.args, {"query": "Marte", "item_type": "episode"})
        self.assertEqual(album.name, "spotify.search")
        self.assertEqual(album.args, {"query": "Discovery", "item_type": "album"})
        self.assertEqual(save.name, "spotify.save_current")

    def test_parse_playlist_management(self) -> None:
        """Create, rename, edit, add, remove, and delete should be distinct actions."""

        cases = {
            "crea una playlist llamada Prueba": "spotify.create_playlist",
            "renombra mi playlist Prueba a Nueva": "spotify.update_playlist",
            "haz mi playlist Nueva privada": "spotify.update_playlist",
            "agrega One a mi playlist Nueva": "spotify.add_to_playlist",
            "quita One de mi playlist Nueva": "spotify.remove_from_playlist",
            "elimina mi playlist Nueva": "spotify.delete_playlist",
        }
        for phrase, expected_name in cases.items():
            with self.subTest(phrase=phrase):
                call = parse_tool_call(phrase)
                self.assertIsNotNone(call)
                assert call is not None
                self.assertEqual(call.name, expected_name)

    def test_parse_contextual_commands(self) -> None:
        """Indirect recommendations and ordinal follow-ups should become context calls."""

        similar = parse_tool_call("pon algo parecido a lo anterior")
        second = parse_tool_call("ahora pon la segunda")

        assert similar is not None and second is not None
        self.assertEqual(similar.name, "spotify.play_similar")
        self.assertEqual(second.name, "spotify.play_result")
        self.assertEqual(second.args["index"], 1)

    def test_parse_explicit_lg_tv_commands(self) -> None:
        cases = {
            "enciende la tele": ("lg_tv.power_on", {}),
            "apaga el televisor": ("lg_tv.power_off", {}),
            "sube el volumen de la television": ("lg_tv.volume_up", {}),
            "baja el volumen de la tele": ("lg_tv.volume_down", {}),
            "pon el volumen de la tele en 22": ("lg_tv.set_volume", {"volume": 22}),
            "silencia la tv": ("lg_tv.set_mute", {"muted": True}),
            "quita el silencio de la television": ("lg_tv.set_mute", {"muted": False}),
            "abre YouTube en la tele": ("lg_tv.launch_app", {"app_name": "YouTube"}),
            "cambia la tele a HDMI 2": ("lg_tv.select_input", {"input": "HDMI 2"}),
            "cual es el estado de la television": ("lg_tv.status", {"detail": "full"}),
            "la tele esta apagada o encendida": ("lg_tv.status", {"detail": "power"}),
            "la tele esta apagado en encendida": ("lg_tv.status", {"detail": "power"}),
            "que canal esta viendo la tele": ("lg_tv.status", {"detail": "channel"}),
            "que canal estoy viendo": ("lg_tv.status", {"detail": "channel"}),
            "que aplicacion esta usando la tele": ("lg_tv.status", {"detail": "app"}),
            "que salida de sonido tiene la tele": ("lg_tv.status", {"detail": "sound"}),
            "que volumen tiene la tele": ("lg_tv.status", {"detail": "volume"}),
            "la tele esta silenciada": ("lg_tv.status", {"detail": "mute"}),
            "la tele esta encendida": ("lg_tv.status", {"detail": "power"}),
            "la pantalla de la tele esta encendida": ("lg_tv.status", {"detail": "screen"}),
            "apaga solo la pantalla de la tele": ("lg_tv.screen_off", {}),
            "enciende la pantalla de la tele": ("lg_tv.screen_on", {}),
            "pausa el video en la tele": ("lg_tv.playback", {"action": "pause"}),
            "reanuda la reproduccion en la tele": ("lg_tv.playback", {"action": "play"}),
            "deten el video en la tele": ("lg_tv.playback", {"action": "stop"}),
            "retrocede en la tele": ("lg_tv.playback", {"action": "rewind"}),
            "adelanta en la tele": ("lg_tv.playback", {"action": "fast_forward"}),
            "sube un canal en la tele": ("lg_tv.channel_up", {}),
            "baja un canal en la tele": ("lg_tv.channel_down", {}),
            "pon el canal 13-2 en la tele": (
                "lg_tv.select_channel",
                {"query": "13-2"},
            ),
            "sintoniza la tele al canal T13": (
                "lg_tv.select_channel",
                {"query": "T13"},
            ),
            "ve al inicio de la tele": ("lg_tv.navigate", {"action": "home"}),
            "vuelve atras en la tele": ("lg_tv.navigate", {"action": "back"}),
            "muevete a la derecha en la tele": ("lg_tv.navigate", {"action": "right"}),
            "presiona ok en la tele": ("lg_tv.navigate", {"action": "ok"}),
            "cierra la aplicacion actual de la tele": ("lg_tv.close_app", {}),
            "muestra en la tele el mensaje Hola Francisco": (
                "lg_tv.show_message",
                {"message": "Hola Francisco"},
            ),
            "apaga la tele en 30 minutos": (
                "lg_tv.schedule_power_off",
                {"minutes": 30},
            ),
            "cuanto falta para que se apague la tele": (
                "lg_tv.timer_status",
                {},
            ),
            "cancela el apagado programado de la tele": (
                "lg_tv.cancel_timers",
                {"scope": "power_off"},
            ),
            "muestra en la tele el mensaje La cena esta lista en 10 minutos": (
                "lg_tv.schedule_message",
                {"message": "La cena esta lista", "minutes": 10},
            ),
            "pon mi playlist Timba en Spotify en la tele": (
                "media.spotify_on_tv",
                {
                    "query": "Timba",
                    "item_type": "playlist",
                    "library_only": True,
                    "shuffle": False,
                },
            ),
        }
        for phrase, expected in cases.items():
            with self.subTest(phrase=phrase):
                call = parse_tool_call(phrase)
                self.assertIsNotNone(call)
                assert call is not None
                self.assertEqual((call.name, call.args), expected)


if __name__ == "__main__":
    unittest.main()
