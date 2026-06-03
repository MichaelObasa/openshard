from __future__ import annotations

import json
import os
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml
from click.testing import CliRunner

from openshard.cli.main import cli

_NO_KEYS = {"OPENROUTER_API_KEY": "", "ANTHROPIC_API_KEY": "", "OPENAI_API_KEY": "",
            "OPENSHARD_CONFIG": ""}
_CONFIG_REL = Path(".openshard") / "config.yml"


def _runner():
    return CliRunner()


class TestInitJsonDiscovery(unittest.TestCase):
    def test_json_emits_options_and_state_writes_nothing(self):
        with patch.dict(os.environ, _NO_KEYS, clear=False):
            with _runner().isolated_filesystem():
                result = _runner().invoke(cli, ["init", "--json"])
                self.assertEqual(result.exit_code, 0)
                payload = json.loads(result.output)
                self.assertIn("options", payload)
                self.assertIn("state", payload)
                self.assertIn("modes", payload["options"])
                self.assertFalse(_CONFIG_REL.exists())


class TestInitYes(unittest.TestCase):
    def test_yes_writes_and_preserves_model_tiers(self):
        with patch.dict(os.environ, _NO_KEYS, clear=False):
            with _runner().isolated_filesystem():
                result = _runner().invoke(
                    cli, ["init", "--yes", "--mode", "local_only", "--provider", "skip"])
                self.assertEqual(result.exit_code, 0, result.output)
                self.assertTrue(_CONFIG_REL.exists())
                data = yaml.safe_load(_CONFIG_REL.read_text(encoding="utf-8"))
                self.assertIn("onboarding", data)
                self.assertIn("model_tiers", data)
                self.assertEqual(data["onboarding"]["mode"], "local_only")
                self.assertEqual(data["onboarding"]["provider"], "skip")

    def test_yes_no_key_defaults_local_only_skip_balanced_human(self):
        with patch.dict(os.environ, _NO_KEYS, clear=False):
            with _runner().isolated_filesystem():
                result = _runner().invoke(cli, ["init", "--yes"])
                self.assertEqual(result.exit_code, 0, result.output)
                data = yaml.safe_load(_CONFIG_REL.read_text(encoding="utf-8"))
                ob = data["onboarding"]
                self.assertEqual(ob["mode"], "local_only")
                self.assertEqual(ob["provider"], "skip")
                self.assertEqual(ob["model_mode"], "balanced")
                self.assertEqual(ob["output_mode"], "human")

    def test_yes_with_key_defaults_native_and_provider(self):
        env = {**_NO_KEYS, "OPENROUTER_API_KEY": "or-key"}
        with patch.dict(os.environ, env, clear=False):
            with _runner().isolated_filesystem():
                result = _runner().invoke(cli, ["init", "--yes"])
                self.assertEqual(result.exit_code, 0, result.output)
                data = yaml.safe_load(_CONFIG_REL.read_text(encoding="utf-8"))
                self.assertEqual(data["onboarding"]["mode"], "native")
                self.assertEqual(data["onboarding"]["provider"], "openrouter")

    def test_yes_json_emits_state(self):
        with patch.dict(os.environ, _NO_KEYS, clear=False):
            with _runner().isolated_filesystem():
                result = _runner().invoke(
                    cli, ["init", "--yes", "--json", "--mode", "local_only"])
                self.assertEqual(result.exit_code, 0, result.output)
                state = json.loads(result.output)
                self.assertEqual(state["mode"], "local_only")
                self.assertIn("warnings", state)

    def test_invalid_choice_rejected(self):
        with patch.dict(os.environ, _NO_KEYS, clear=False):
            with _runner().isolated_filesystem():
                result = _runner().invoke(cli, ["init", "--yes", "--mode", "bogus"])
                self.assertNotEqual(result.exit_code, 0)


class TestInitInteractive(unittest.TestCase):
    def test_interactive_writes_file(self):
        # Four selection prompts (each takes the default on empty line) + confirm.
        stdin = "\n\n\n\n"
        with patch.dict(os.environ, _NO_KEYS, clear=False):
            with _runner().isolated_filesystem():
                result = _runner().invoke(cli, ["init"], input=stdin)
                self.assertEqual(result.exit_code, 0, result.output)
                self.assertTrue(_CONFIG_REL.exists())

    def test_overwrite_declined_leaves_file_unchanged(self):
        with patch.dict(os.environ, _NO_KEYS, clear=False):
            with _runner().isolated_filesystem():
                _CONFIG_REL.parent.mkdir(parents=True)
                original = "model_tiers:\n  - name: x\n    model: m\n    max_tokens: 1\nworkflow: auto\n"
                _CONFIG_REL.write_text(original, encoding="utf-8")
                # All four selections via flags -> only the confirm prompt remains; decline.
                result = _runner().invoke(
                    cli,
                    ["init", "--mode", "local_only", "--provider", "skip",
                     "--model-mode", "balanced", "--output-mode", "human"],
                    input="n\n",
                )
                self.assertEqual(result.exit_code, 0, result.output)
                self.assertEqual(_CONFIG_REL.read_text(encoding="utf-8"), original)


class TestInitOverwriteForce(unittest.TestCase):
    def test_force_preserves_custom_model_tiers(self):
        with patch.dict(os.environ, _NO_KEYS, clear=False):
            with _runner().isolated_filesystem():
                _CONFIG_REL.parent.mkdir(parents=True)
                _CONFIG_REL.write_text(
                    "model_tiers:\n  - name: custom\n    model: my/model\n    max_tokens: 42\n"
                    "planning_model: my/model\n",
                    encoding="utf-8",
                )
                result = _runner().invoke(
                    cli, ["init", "--yes", "--force", "--mode", "local_only"])
                self.assertEqual(result.exit_code, 0, result.output)
                data = yaml.safe_load(_CONFIG_REL.read_text(encoding="utf-8"))
                self.assertEqual(data["model_tiers"][0]["name"], "custom")
                self.assertEqual(data["model_tiers"][0]["max_tokens"], 42)
                self.assertIn("onboarding", data)


class TestInitWarnings(unittest.TestCase):
    def test_free_mode_anthropic_warns(self):
        env = {**_NO_KEYS, "ANTHROPIC_API_KEY": "sk"}
        with patch.dict(os.environ, env, clear=False):
            with _runner().isolated_filesystem():
                result = _runner().invoke(
                    cli, ["init", "--yes", "--json", "--provider", "anthropic",
                          "--model-mode", "free"])
                state = json.loads(result.output)
                self.assertTrue(any("free-model mode" in w for w in state["warnings"]))


if __name__ == "__main__":
    unittest.main()
