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


class TestConfigShow(unittest.TestCase):
    def test_no_file_shows_defaults(self):
        with patch.dict(os.environ, _NO_KEYS, clear=False):
            with _runner().isolated_filesystem():
                result = _runner().invoke(cli, ["config", "show"])
                self.assertEqual(result.exit_code, 0, result.output)
                self.assertIn("model_tiers", result.output)

    def test_json_valid_and_redacted(self):
        with patch.dict(os.environ, _NO_KEYS, clear=False):
            with _runner().isolated_filesystem():
                _write_config(
                    "model_tiers: []\napi_key: SEKRET\nproviders:\n  - token: NESTED_T\n")
                result = _runner().invoke(cli, ["config", "show", "--json"])
                self.assertEqual(result.exit_code, 0, result.output)
                data = json.loads(result.output)
                self.assertEqual(data["api_key"], "***REDACTED***")
                self.assertEqual(data["providers"][0]["token"], "***REDACTED***")
                self.assertNotIn("SEKRET", result.output)
                self.assertNotIn("NESTED_T", result.output)

    def test_human_redacts(self):
        with patch.dict(os.environ, _NO_KEYS, clear=False):
            with _runner().isolated_filesystem():
                _write_config("api_key: SEKRET\nworkflow: auto\n")
                result = _runner().invoke(cli, ["config", "show"])
                self.assertIn("***REDACTED***", result.output)
                self.assertNotIn("SEKRET", result.output)

    def test_malformed_config_warns_and_shows_defaults(self):
        with patch.dict(os.environ, _NO_KEYS, clear=False):
            with _runner().isolated_filesystem():
                _write_config("model_tiers: [unclosed\n : : :\n")
                result = _runner().invoke(cli, ["config", "show"])
                self.assertEqual(result.exit_code, 0, result.output)
                self.assertIn("could not be parsed", result.output)
                self.assertIn("model_tiers", result.output)

    def test_partial_config_renders(self):
        with patch.dict(os.environ, _NO_KEYS, clear=False):
            with _runner().isolated_filesystem():
                _write_config("workflow: auto\n")
                result = _runner().invoke(cli, ["config", "show"])
                self.assertEqual(result.exit_code, 0, result.output)
                self.assertIn("workflow", result.output)


if __name__ == "__main__":
    unittest.main()
