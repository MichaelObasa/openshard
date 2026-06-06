"""Tests for first-run UX: load_config() defaults and env-var API key injection."""
from __future__ import annotations

import os
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml
from click.testing import CliRunner

from openshard.cli.main import cli
from openshard.config.settings import load_config

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_NO_API_KEYS = {
    "OPENROUTER_API_KEY": "",
    "ANTHROPIC_API_KEY": "",
    "OPENAI_API_KEY": "",
    "OPENSHARD_CONFIG": "",
}

_REQUIRED_KEYS = {"model_tiers", "planning_model", "execution_model", "workflow", "approval_mode"}


def _runner() -> CliRunner:
    return CliRunner()


# ---------------------------------------------------------------------------
# 1. load_config() returns defaults when no file exists (never raises)
# ---------------------------------------------------------------------------


class TestLoadConfigNoFile(unittest.TestCase):

    def test_load_config_returns_defaults_when_no_file(self):
        """In an isolated dir with no config, load_config() must return a dict, not raise."""
        with patch.dict(os.environ, _NO_API_KEYS, clear=False):
            with _runner().isolated_filesystem():
                result = load_config()
                self.assertIsInstance(result, dict)

    def test_load_config_default_has_required_keys(self):
        """Returned default dict has the keys the pipeline expects."""
        with patch.dict(os.environ, _NO_API_KEYS, clear=False):
            with _runner().isolated_filesystem():
                cfg = load_config()
                for key in _REQUIRED_KEYS:
                    self.assertIn(key, cfg, f"Missing required key: {key!r}")
                self.assertIsInstance(cfg["model_tiers"], list)
                self.assertGreater(len(cfg["model_tiers"]), 0)


# ---------------------------------------------------------------------------
# 2. Env-var API key injection into default config
# ---------------------------------------------------------------------------


class TestLoadConfigApiKeyInjection(unittest.TestCase):

    def test_load_config_injects_openrouter_key_from_env(self):
        """When OPENROUTER_API_KEY is set, the default config contains it."""
        env = {**_NO_API_KEYS, "OPENROUTER_API_KEY": "test-openrouter-key"}
        with patch.dict(os.environ, env, clear=False):
            with _runner().isolated_filesystem():
                cfg = load_config()
                self.assertEqual(cfg.get("openrouter_api_key"), "test-openrouter-key")

    def test_load_config_injects_anthropic_key_from_env(self):
        """When ANTHROPIC_API_KEY is set, the default config contains it."""
        env = {**_NO_API_KEYS, "ANTHROPIC_API_KEY": "test-anthropic-key"}
        with patch.dict(os.environ, env, clear=False):
            with _runner().isolated_filesystem():
                cfg = load_config()
                self.assertEqual(cfg.get("anthropic_api_key"), "test-anthropic-key")

    def test_load_config_injects_openai_key_from_env(self):
        """When OPENAI_API_KEY is set, the default config contains it."""
        env = {**_NO_API_KEYS, "OPENAI_API_KEY": "test-openai-key"}
        with patch.dict(os.environ, env, clear=False):
            with _runner().isolated_filesystem():
                cfg = load_config()
                self.assertEqual(cfg.get("openai_api_key"), "test-openai-key")

    def test_load_config_env_key_not_injected_when_absent(self):
        """Unset env vars do not appear as keys (or are None/empty) in the default config."""
        with patch.dict(os.environ, _NO_API_KEYS, clear=False):
            with _runner().isolated_filesystem():
                cfg = load_config()
                # Key should not be present or should be falsy when env var is empty/unset
                self.assertFalse(cfg.get("openrouter_api_key"))
                self.assertFalse(cfg.get("anthropic_api_key"))
                self.assertFalse(cfg.get("openai_api_key"))


# ---------------------------------------------------------------------------
# 3. Existing config file is still read normally
# ---------------------------------------------------------------------------


class TestLoadConfigFileStillWorks(unittest.TestCase):

    def test_load_config_file_still_works_when_present(self):
        """An existing .openshard/config.yml is read normally — injection path not taken."""
        env = {**_NO_API_KEYS, "OPENROUTER_API_KEY": "should-not-appear"}
        with patch.dict(os.environ, env, clear=False):
            with _runner().isolated_filesystem():
                config_dir = Path(".openshard")
                config_dir.mkdir(parents=True)
                config_file = config_dir / "config.yml"
                file_data = {
                    "approval_mode": "ask",
                    "model_tiers": [],
                    "workflow": "direct",
                }
                config_file.write_text(
                    yaml.safe_dump(file_data, sort_keys=False), encoding="utf-8"
                )
                cfg = load_config()
                # Config values from file are present
                self.assertEqual(cfg["approval_mode"], "ask")
                self.assertEqual(cfg["workflow"], "direct")
                # API key is NOT injected when reading from a real file
                self.assertNotIn("openrouter_api_key", cfg)


# ---------------------------------------------------------------------------
# 4. openshard init --help still exits 0 (init command still works)
# ---------------------------------------------------------------------------


class TestInitCommandStillWorks(unittest.TestCase):

    def test_init_command_still_exits_zero(self):
        """openshard init --help still exits 0."""
        result = _runner().invoke(cli, ["init", "--help"])
        self.assertEqual(result.exit_code, 0, result.output)


if __name__ == "__main__":
    unittest.main()
