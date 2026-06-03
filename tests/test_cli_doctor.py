from __future__ import annotations

import json
import os
import unittest
from pathlib import Path
from unittest.mock import patch

from click.testing import CliRunner

from openshard.cli.main import cli

_NO_KEYS = {"OPENROUTER_API_KEY": "", "ANTHROPIC_API_KEY": "", "OPENAI_API_KEY": "",
            "OPENSHARD_CONFIG": ""}
_CONFIG_REL = Path(".openshard") / "config.yml"


def _runner():
    return CliRunner()


def _write_config(text: str) -> None:
    _CONFIG_REL.parent.mkdir(parents=True, exist_ok=True)
    _CONFIG_REL.write_text(text, encoding="utf-8")


class TestDoctorJson(unittest.TestCase):
    def test_no_config_reports_not_found(self):
        with patch.dict(os.environ, _NO_KEYS, clear=False):
            with _runner().isolated_filesystem():
                result = _runner().invoke(cli, ["doctor", "--json"])
                self.assertEqual(result.exit_code, 0, result.output)
                state = json.loads(result.output)
                self.assertFalse(state["config_found"])
                self.assertIn("git_repo", state)
                self.assertTrue(any("openshard init" in s for s in state["next_steps"]))

    def test_config_present_relative_display_path(self):
        with patch.dict(os.environ, _NO_KEYS, clear=False):
            with _runner().isolated_filesystem():
                _write_config(
                    "model_tiers: []\nonboarding:\n  mode: local_only\n  provider: skip\n")
                result = _runner().invoke(cli, ["doctor", "--json"])
                state = json.loads(result.output)
                self.assertTrue(state["config_found"])
                self.assertEqual(state["config_path_display"], ".openshard/config.yml")
                self.assertNotIn(":", state["config_path_display"])
                self.assertEqual(state["mode"], "local_only")

    def test_malformed_config_degrades(self):
        with patch.dict(os.environ, _NO_KEYS, clear=False):
            with _runner().isolated_filesystem():
                _write_config("model_tiers: [unclosed\n  : : :\n")
                result = _runner().invoke(cli, ["doctor", "--json"])
                self.assertEqual(result.exit_code, 0, result.output)
                state = json.loads(result.output)
                self.assertFalse(state["config_valid"])
                self.assertTrue(any("could not be parsed" in w for w in state["warnings"]))

    def test_api_key_present_reflects_env_and_no_value_leak(self):
        env = {**_NO_KEYS, "ANTHROPIC_API_KEY": "sk-super-secret-123"}
        with patch.dict(os.environ, env, clear=False):
            with _runner().isolated_filesystem():
                result = _runner().invoke(cli, ["doctor", "--json"])
                self.assertNotIn("sk-super-secret-123", result.output)

    def test_git_repo_true_with_dot_git(self):
        with patch.dict(os.environ, _NO_KEYS, clear=False):
            with _runner().isolated_filesystem():
                Path(".git").mkdir()
                result = _runner().invoke(cli, ["doctor", "--json"])
                state = json.loads(result.output)
                self.assertTrue(state["git_repo"])


class TestDoctorHuman(unittest.TestCase):
    def test_human_runs_and_redacts(self):
        env = {**_NO_KEYS, "OPENAI_API_KEY": "sk-leak-me"}
        with patch.dict(os.environ, env, clear=False):
            with _runner().isolated_filesystem():
                result = _runner().invoke(cli, ["doctor"])
                self.assertEqual(result.exit_code, 0, result.output)
                self.assertIn("OpenShard Doctor", result.output)
                self.assertNotIn("sk-leak-me", result.output)
                self.assertIn("openai", result.output)


if __name__ == "__main__":
    unittest.main()
