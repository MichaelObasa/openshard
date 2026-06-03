from __future__ import annotations

import os
import unittest
from pathlib import Path
from unittest.mock import patch

from openshard.config import onboarding as ob

_NO_KEYS = {"OPENROUTER_API_KEY": "", "ANTHROPIC_API_KEY": "", "OPENAI_API_KEY": ""}


class TestRedact(unittest.TestCase):
    def test_masks_secret_keys(self):
        out = ob.redact({"api_key": "abc", "token": "t", "secret": "s", "password": "p"})
        self.assertEqual(out["api_key"], "***REDACTED***")
        self.assertEqual(out["token"], "***REDACTED***")
        self.assertEqual(out["secret"], "***REDACTED***")
        self.assertEqual(out["password"], "***REDACTED***")

    def test_leaves_non_secret_values(self):
        out = ob.redact({"workflow": "auto", "cost": 0.1})
        self.assertEqual(out, {"workflow": "auto", "cost": 0.1})

    def test_recurses_into_nested(self):
        out = ob.redact({"provider": {"api_key": "x", "name": "openrouter"}})
        self.assertEqual(out["provider"]["api_key"], "***REDACTED***")
        self.assertEqual(out["provider"]["name"], "openrouter")

    def test_recurses_into_lists(self):
        out = ob.redact([{"token": "x"}, {"name": "y"}])
        self.assertEqual(out[0]["token"], "***REDACTED***")
        self.assertEqual(out[1]["name"], "y")

    def test_leaves_empty_and_none(self):
        out = ob.redact({"api_key": "", "token": None})
        self.assertEqual(out["api_key"], "")
        self.assertIsNone(out["token"])

    def test_is_secret_key(self):
        self.assertTrue(ob.is_secret_key("OPENROUTER_API_KEY"))
        self.assertTrue(ob.is_secret_key("access_token"))
        self.assertFalse(ob.is_secret_key("model"))


class TestOptionsCatalog(unittest.TestCase):
    def test_shape(self):
        cat = ob.options_catalog()
        for k in ("modes", "providers", "model_modes", "output_modes",
                  "implemented_providers", "safety_notes"):
            self.assertIn(k, cat)
        self.assertTrue(all("key" in o and "label" in o for o in cat["modes"]))
        self.assertEqual(sorted(cat["implemented_providers"]),
                         ["anthropic", "openai", "openrouter"])


class TestApiKeyPresent(unittest.TestCase):
    def test_booleans_only_and_reflects_env(self):
        with patch.dict(os.environ, {**_NO_KEYS, "ANTHROPIC_API_KEY": "sk-xyz"}, clear=False):
            present = ob.api_key_present()
        self.assertEqual(present, {"openrouter": False, "anthropic": True, "openai": False})
        self.assertTrue(all(isinstance(v, bool) for v in present.values()))

    def test_default_provider_prefers_openrouter(self):
        with patch.dict(os.environ, {**_NO_KEYS, "OPENROUTER_API_KEY": "k",
                                     "ANTHROPIC_API_KEY": "k"}, clear=False):
            self.assertEqual(ob.default_provider(), "openrouter")

    def test_default_provider_skip_when_none(self):
        with patch.dict(os.environ, _NO_KEYS, clear=False):
            self.assertEqual(ob.default_provider(), "skip")


class TestDetectGitRepo(unittest.TestCase):
    def test_true_when_dot_git_present(self):
        import tempfile

        with tempfile.TemporaryDirectory() as td:
            (Path(td) / ".git").mkdir()
            self.assertTrue(ob.detect_git_repo(Path(td)))

    def test_false_when_no_dot_git_in_chain(self):
        # The home dir may itself be a git repo, so force every .git probe to miss
        # and confirm the parent walk completes and returns False.
        import tempfile

        real_exists = Path.exists

        def fake_exists(self):
            if self.name == ".git":
                return False
            return real_exists(self)

        with tempfile.TemporaryDirectory() as td:
            with patch.object(Path, "exists", fake_exists):
                self.assertFalse(ob.detect_git_repo(Path(td)))


class TestBuildState(unittest.TestCase):
    def _state(self, onboarding, env=None, config_found=True, config_valid=True, path=None):
        with patch.dict(os.environ, {**_NO_KEYS, **(env or {})}, clear=False):
            return ob.build_state(
                version="9.9.9",
                config_found=config_found,
                config_path=path,
                config_valid=config_valid,
                onboarding=onboarding,
            )

    def test_all_shared_keys_present(self):
        st = self._state({})
        for k in ("openshard_version", "config_found", "config_path_display",
                  "config_valid", "mode", "provider", "model_mode",
                  "api_key_present", "output_mode", "warnings", "next_steps"):
            self.assertIn(k, st)

    def test_no_config_adds_next_step(self):
        st = self._state({}, config_found=False)
        self.assertTrue(any("openshard init" in s for s in st["next_steps"]))

    def test_malformed_adds_warning(self):
        st = self._state({}, config_valid=False)
        self.assertTrue(any("could not be parsed" in w for w in st["warnings"]))

    def test_implemented_provider_missing_key_warns(self):
        st = self._state({"provider": "anthropic"})
        self.assertTrue(any("ANTHROPIC_API_KEY is not set" in w for w in st["warnings"]))
        self.assertFalse(st["api_key_present"])

    def test_implemented_provider_with_key_no_warning(self):
        st = self._state({"provider": "anthropic"}, env={"ANTHROPIC_API_KEY": "sk"})
        self.assertFalse(any("ANTHROPIC_API_KEY is not set" in w for w in st["warnings"]))
        self.assertTrue(st["api_key_present"])

    def test_via_openrouter_provider_warns_when_no_openrouter_key(self):
        st = self._state({"provider": "glm"})
        self.assertTrue(any("reached via OpenRouter" in w for w in st["warnings"]))

    def test_free_mode_non_openrouter_warns(self):
        st = self._state({"provider": "anthropic", "model_mode": "free"},
                         env={"ANTHROPIC_API_KEY": "sk"})
        self.assertTrue(any("free-model mode" in w for w in st["warnings"]))

    def test_no_key_local_only_next_step(self):
        st = self._state({"mode": "local_only"})
        self.assertTrue(any("Local-only mode" in s for s in st["next_steps"]))
        self.assertFalse(any("limited to local-only mode until" in w for w in st["warnings"]))

    def test_no_key_native_mode_warns(self):
        st = self._state({"mode": "native"})
        self.assertTrue(any("limited to local-only" in w for w in st["warnings"]))

    def test_display_path_is_relative(self):
        import tempfile

        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / ".openshard" / "config.yml"
            p.parent.mkdir(parents=True)
            p.write_text("{}", encoding="utf-8")
            st = ob.build_state(version="1", config_found=True, config_path=p,
                                config_valid=True, onboarding={}, cwd=Path(td))
        self.assertEqual(st["config_path_display"], ".openshard/config.yml")
        self.assertNotIn(":", st["config_path_display"] or "")


if __name__ == "__main__":
    unittest.main()
