"""Tests for first-run UX: load_config() defaults, env-var injection, provider detection."""
from __future__ import annotations

import os
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml
from click.testing import CliRunner

from openshard.cli.main import cli
from openshard.config.settings import detect_provider, is_agent_environment, load_config

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


# ---------------------------------------------------------------------------
# 5. detect_provider: auto-detects from available API keys
# ---------------------------------------------------------------------------

_NO_KEYS = {
    "OPENROUTER_API_KEY": "",
    "ANTHROPIC_API_KEY": "",
    "OPENAI_API_KEY": "",
    "CI": "",
    "GITHUB_ACTIONS": "",
    "GITLAB_CI": "",
    "NO_COLOR": "",
    "OPENSHARD_AGENT": "",
    "OPENSHARD_CONFIG": "",
}


class TestDetectProvider(unittest.TestCase):

    def test_returns_openrouter_when_key_set(self):
        with patch.dict(os.environ, {**_NO_KEYS, "OPENROUTER_API_KEY": "key"}, clear=True):
            self.assertEqual(detect_provider(), "openrouter")

    def test_returns_anthropic_without_openrouter(self):
        with patch.dict(os.environ, {**_NO_KEYS, "ANTHROPIC_API_KEY": "ant"}, clear=True):
            self.assertEqual(detect_provider(), "anthropic")

    def test_returns_openai_when_only_openai_set(self):
        with patch.dict(os.environ, {**_NO_KEYS, "OPENAI_API_KEY": "oai"}, clear=True):
            self.assertEqual(detect_provider(), "openai")

    def test_openrouter_wins_over_anthropic(self):
        env = {**_NO_KEYS, "OPENROUTER_API_KEY": "or", "ANTHROPIC_API_KEY": "ant"}
        with patch.dict(os.environ, env, clear=True):
            self.assertEqual(detect_provider(), "openrouter")

    def test_raises_when_no_key_set(self):
        with patch.dict(os.environ, _NO_KEYS, clear=True):
            with self.assertRaises(ValueError) as cm:
                detect_provider()
            msg = str(cm.exception)
            self.assertIn("OPENROUTER_API_KEY", msg)
            self.assertIn("ANTHROPIC_API_KEY", msg)
            self.assertIn("OPENAI_API_KEY", msg)
            self.assertIn("openshard init", msg)


# ---------------------------------------------------------------------------
# 6. is_agent_environment: CI/agent env var detection
# ---------------------------------------------------------------------------


class TestIsAgentEnvironment(unittest.TestCase):

    def test_false_by_default(self):
        with patch.dict(os.environ, _NO_KEYS, clear=True):
            self.assertFalse(is_agent_environment())

    def test_true_for_ci(self):
        with patch.dict(os.environ, {**_NO_KEYS, "CI": "true"}, clear=True):
            self.assertTrue(is_agent_environment())

    def test_true_for_github_actions(self):
        with patch.dict(os.environ, {**_NO_KEYS, "GITHUB_ACTIONS": "true"}, clear=True):
            self.assertTrue(is_agent_environment())

    def test_true_for_openshard_agent(self):
        with patch.dict(os.environ, {**_NO_KEYS, "OPENSHARD_AGENT": "1"}, clear=True):
            self.assertTrue(is_agent_environment())

    def test_true_for_no_color(self):
        with patch.dict(os.environ, {**_NO_KEYS, "NO_COLOR": "1"}, clear=True):
            self.assertTrue(is_agent_environment())

    def test_never_raises(self):
        with patch.dict(os.environ, _NO_KEYS, clear=True):
            try:
                result = is_agent_environment()
                self.assertIsInstance(result, bool)
            except Exception as exc:  # noqa: BLE001
                self.fail(f"is_agent_environment() raised unexpectedly: {exc}")


# ---------------------------------------------------------------------------
# 7. load_config: agent mode injection
# ---------------------------------------------------------------------------


class TestLoadConfigAgentMode(unittest.TestCase):

    def test_sets_agent_json_in_ci_without_config(self):
        with _runner().isolated_filesystem():
            with patch.dict(os.environ, {**_NO_KEYS, "CI": "true"}, clear=True):
                cfg = load_config()
        self.assertEqual(cfg.get("output_mode"), "agent_json")

    def test_does_not_override_explicit_config_file(self):
        with _runner().isolated_filesystem():
            Path(".openshard").mkdir()
            Path(".openshard/config.yml").write_text(
                yaml.safe_dump({"output_mode": "human"}), encoding="utf-8"
            )
            with patch.dict(os.environ, {**_NO_KEYS, "CI": "true"}, clear=True):
                cfg = load_config()
        self.assertEqual(cfg.get("output_mode"), "human")


# ---------------------------------------------------------------------------
# 8. run command: clean error message when no API key
# ---------------------------------------------------------------------------


class TestRunNoKeyCleanError(unittest.TestCase):

    def test_no_traceback_on_missing_key(self):
        runner = CliRunner()
        with runner.isolated_filesystem():
            with patch.dict(os.environ, _NO_KEYS, clear=True):
                result = runner.invoke(cli, ["run", "say hello"])
        self.assertNotIn("Traceback", result.output)
        self.assertNotIn("RuntimeError", result.output)

    def test_error_mentions_all_three_providers(self):
        runner = CliRunner()
        with runner.isolated_filesystem():
            with patch.dict(os.environ, _NO_KEYS, clear=True):
                result = runner.invoke(cli, ["run", "say hello"])
        self.assertIn("OPENROUTER_API_KEY", result.output)
        self.assertIn("ANTHROPIC_API_KEY", result.output)
        self.assertIn("OPENAI_API_KEY", result.output)

    def test_error_suggests_init(self):
        runner = CliRunner()
        with runner.isolated_filesystem():
            with patch.dict(os.environ, _NO_KEYS, clear=True):
                result = runner.invoke(cli, ["run", "say hello"])
        self.assertIn("openshard init", result.output)

    def test_exits_nonzero(self):
        runner = CliRunner()
        with runner.isolated_filesystem():
            with patch.dict(os.environ, _NO_KEYS, clear=True):
                result = runner.invoke(cli, ["run", "say hello"])
        self.assertNotEqual(result.exit_code, 0)


# ---------------------------------------------------------------------------
# 9. openshard env command
# ---------------------------------------------------------------------------


class TestEnvCommand(unittest.TestCase):

    def _run(self, extra_env: dict | None = None) -> object:
        env = dict(_NO_KEYS)
        if extra_env:
            env.update(extra_env)
        runner = CliRunner()
        with runner.isolated_filesystem():
            with patch.dict(os.environ, env, clear=True):
                return runner.invoke(cli, ["env"])

    def test_exits_zero(self):
        result = self._run()
        self.assertEqual(result.exit_code, 0, msg=result.output)

    def test_shows_agent_no_by_default(self):
        result = self._run()
        self.assertIn("no", result.output)

    def test_shows_agent_yes_with_ci(self):
        result = self._run({"CI": "true"})
        self.assertIn("yes", result.output)
        self.assertIn("CI", result.output)

    def test_never_prints_key_value(self):
        result = self._run({"OPENROUTER_API_KEY": "sk-super-secret-key-xyz"})
        self.assertNotIn("sk-super-secret-key-xyz", result.output)

    def test_shows_provider_name_when_key_present(self):
        result = self._run({"OPENROUTER_API_KEY": "any-key"})
        self.assertIn("openrouter", result.output)

    def test_shows_not_set_when_no_key(self):
        result = self._run()
        self.assertIn("not set", result.output)


if __name__ == "__main__":
    unittest.main()
