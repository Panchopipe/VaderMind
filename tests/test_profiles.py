import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest import mock

from vadermind.config import ConfigurationError, load_settings
from vadermind.profiles import (
    LAST_PROFILE_FILE,
    ProfileChoice,
    current_ssid,
    parse_netsh_ssid,
    parse_nmcli_ssid,
    profile_after_network,
    resolve_profile,
)


def write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


class ProfileTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self._env = mock.patch.dict(os.environ, {}, clear=True)
        self._env.start()
        write(self.root / "profiles/pancho/perfil.env", "VADER_WIFI_SSIDS=RedPancho\nLG_TV_HOST=192.0.2.167\n")
        write(
            self.root / "profiles/pablo/perfil.env",
            "VADER_WIFI_SSIDS=RedPablo, Pablo 5G\nVADER_PERSONA_MODE=chileno\n",
        )

    def tearDown(self) -> None:
        self._env.stop()
        self._tmp.cleanup()

    def resolve(self, ssid: str | None, requested: str | None = None) -> ProfileChoice:
        return resolve_profile(requested, root=self.root, ssid_lookup=lambda: ssid)


class ResolveProfileTests(ProfileTestCase):
    def test_network_picks_the_home_and_is_remembered(self) -> None:
        choice = self.resolve("RedPablo")
        self.assertEqual(choice.name, "pablo")
        self.assertEqual(choice.reason, "red RedPablo")
        self.assertEqual((self.root / LAST_PROFILE_FILE).read_text().strip(), "pablo")
        self.assertEqual(self.resolve("Pablo 5G").name, "pablo")
        self.assertEqual(self.resolve("RedPancho").name, "pancho")

    def test_unknown_network_keeps_the_last_home(self) -> None:
        self.resolve("RedPablo")
        choice = self.resolve("Pancho 2")
        self.assertEqual(choice.name, "pablo")
        self.assertIn("desconocida", choice.reason)
        self.assertEqual(self.resolve(None).name, "pablo")

    def test_first_start_without_network_is_pancho(self) -> None:
        choice = self.resolve(None)
        self.assertEqual((choice.name, choice.reason), ("pancho", "por defecto"))

    def test_ssid_match_is_exact(self) -> None:
        self.assertEqual(self.resolve("redpablo").name, "pancho")

    def test_command_line_forces_and_is_not_remembered(self) -> None:
        choice = self.resolve("RedPancho", requested="Pablo")
        self.assertEqual((choice.name, choice.forced), ("pablo", True))
        self.assertFalse((self.root / LAST_PROFILE_FILE).exists())

    def test_env_file_can_force_a_profile(self) -> None:
        write(self.root / ".env", "VADER_PROFILE=pablo\n")
        choice = self.resolve("RedPancho")
        self.assertEqual((choice.name, choice.reason, choice.forced), ("pablo", "VADER_PROFILE", True))

    def test_unknown_profile_is_an_error(self) -> None:
        with self.assertRaises(ConfigurationError):
            self.resolve("RedPancho", requested="inexistente")

    def test_no_profiles_folder_keeps_the_old_behaviour(self) -> None:
        with tempfile.TemporaryDirectory() as empty:
            choice = resolve_profile(root=Path(empty), ssid_lookup=lambda: "RedPancho")
        self.assertIsNone(choice.name)
        self.assertIsNone(choice.directory)


class AfterNetworkTests(ProfileTestCase):
    def test_switches_when_the_network_is_another_home(self) -> None:
        started = self.resolve(None)  # cold boot: Wi-Fi not up yet
        self.assertEqual(started.name, "pancho")
        self.assertEqual(
            profile_after_network(started, root=self.root, ssid_lookup=lambda: "RedPablo"),
            "pablo",
        )

    def test_same_home_or_unknown_network_keeps_going(self) -> None:
        started = self.resolve(None)
        for ssid in ("RedPancho", "Pancho 2", None):
            self.assertIsNone(profile_after_network(started, root=self.root, ssid_lookup=lambda: ssid))

    def test_forced_profile_never_switches(self) -> None:
        forced = self.resolve(None, requested="pancho")
        self.assertIsNone(
            profile_after_network(forced, root=self.root, ssid_lookup=lambda: "RedPablo")
        )


class SettingsWithProfileTests(ProfileTestCase):
    def test_home_values_come_only_from_the_profile(self) -> None:
        write(
            self.root / ".env",
            "LG_TV_HOST=192.0.2.167\nLG_TV_MAC=AA:BB:CC:DD:EE:FF\nVADER_PERSONA_MODE=canon\n"
            "SPOTIFY_TOKEN_CACHE_PATH=.spotify_token.json\nSPOTIFY_CLIENT_ID=comun\n",
        )
        pablo = self.root / "profiles/pablo"
        settings = load_settings(self.root / ".env", profile_dir=pablo)
        self.assertIsNone(settings.lg_tv_host)  # Pancho's TV does not leak into Pablo's home
        self.assertIsNone(settings.lg_tv_mac)
        self.assertEqual(settings.persona_mode, "chileno")
        self.assertEqual(settings.spotify_client_id, "comun")  # the Spotify app is shared
        self.assertEqual(Path(settings.spotify_token_cache_path), pablo / "spotify_token.json")
        self.assertEqual(Path(settings.lg_tv_schedules_path), pablo / "lg_tv_schedules.json")

    def test_pancho_profile_keeps_his_tv(self) -> None:
        pancho = self.root / "profiles/pancho"
        settings = load_settings(self.root / ".env", profile_dir=pancho)
        self.assertEqual(settings.lg_tv_host, "192.0.2.167")
        self.assertEqual(settings.persona_mode, "canon")
        self.assertEqual(Path(settings.lg_tv_client_key_path), pancho / "lg_tv_client_key")

    def test_without_profile_env_is_used_as_before(self) -> None:
        write(self.root / ".env", "LG_TV_HOST=192.0.2.167\n")
        settings = load_settings(self.root / ".env")
        self.assertEqual(settings.lg_tv_host, "192.0.2.167")
        self.assertEqual(settings.spotify_token_cache_path, ".spotify_token.json")


class SsidParsingTests(unittest.TestCase):
    def test_nmcli_active_line(self) -> None:
        output = "no:Vecino\nyes:RedPablo\nno:RedPablo\n"
        self.assertEqual(parse_nmcli_ssid(output), "RedPablo")

    def test_nmcli_escaped_colon_and_no_active(self) -> None:
        self.assertEqual(parse_nmcli_ssid("yes:Casa\\:2\n"), "Casa:2")
        self.assertIsNone(parse_nmcli_ssid("no:RedPancho\n"))

    def test_netsh_ignores_bssid(self) -> None:
        output = (
            "    Nombre                 : Wi-Fi\n"
            "    Estado                 : conectado\n"
            "    SSID                   : RedPablo\n"
            "    BSSID                  : 00:00:5e:00:53:02\n"
        )
        self.assertEqual(parse_netsh_ssid(output), "RedPablo")
        self.assertIsNone(parse_netsh_ssid("    Estado : desconectado\n"))

    def test_current_ssid_uses_nmcli_on_linux(self) -> None:
        run = mock.Mock(return_value=subprocess.CompletedProcess([], 0, stdout=b"yes:RedPancho\n"))
        self.assertEqual(current_ssid(run=run, platform="linux"), "RedPancho")
        self.assertEqual(run.call_args.args[0][0], "nmcli")

    def test_current_ssid_survives_a_failing_command(self) -> None:
        run = mock.Mock(side_effect=OSError("sin nmcli"))
        self.assertIsNone(current_ssid(run=run, platform="linux"))
        self.assertIsNone(current_ssid(run=run, platform="darwin"))


class RestartTests(ProfileTestCase):
    def test_restart_closes_resources_and_execs_once(self) -> None:
        from vadermind import orchestrator

        started = self.resolve(None)
        tts, listener, lights = mock.Mock(), mock.Mock(), mock.Mock()
        app = mock.Mock(tts=tts)
        execv = mock.Mock()
        with mock.patch.object(orchestrator, "profile_after_network", return_value="pablo"), \
                mock.patch.object(orchestrator.Path, "cwd", return_value=self.root):
            orchestrator._restart_if_other_home(started, app, listener, lights, execv=execv)
            execv.assert_called_once()
            for resource in (tts, listener, lights):
                resource.close.assert_called_once()
            self.assertEqual((self.root / LAST_PROFILE_FILE).read_text().strip(), "pablo")
            # The restarted process never restarts again.
            orchestrator._restart_if_other_home(started, app, listener, lights, execv=execv)
            execv.assert_called_once()

    def test_same_home_does_not_restart(self) -> None:
        from vadermind import orchestrator

        execv = mock.Mock()
        with mock.patch.object(orchestrator, "profile_after_network", return_value=None):
            orchestrator._restart_if_other_home(
                self.resolve(None), mock.Mock(), None, mock.Mock(), execv=execv
            )
        execv.assert_not_called()


if __name__ == "__main__":
    unittest.main()
