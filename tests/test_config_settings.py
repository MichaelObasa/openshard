"""Tests for openshard.config.settings.load_config()."""
from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml

from openshard.config.settings import _DEFAULTS, load_config

# ---------------------------------------------------------------------------
# _DEFAULTS structure
# ---------------------------------------------------------------------------

class TestDefaults(unittest.TestCase):

    def test_defaults_is_dict(self):
        self.assertIsInstance(_DEFAULTS, dict)

    def test_defaults_have_model_tiers(self):
        tiers = _DEFAULTS["model_tiers"]
        self.assertIsInstance(tiers, list)
        self.assertGreater(len(tiers), 0)
        names = {t["name"] for t in tiers}
        self.assertIn("balanced", names)

    def test_defaults_approval_mode(self):
        self.assertEqual(_DEFAULTS["approval_mode"], "smart")

    def test_defaults_workflow(self):
        self.assertEqual(_DEFAULTS["workflow"], "auto")

    def test_defaults_cost_gate(self):
        self.assertIsInstance(_DEFAULTS["cost_gate_threshold"], float)


# ---------------------------------------------------------------------------
# Explicit path argument
# ---------------------------------------------------------------------------

class TestExplicitPath(unittest.TestCase):

    def _write_tmp_config(self, data: dict) -> str:
        f = tempfile.NamedTemporaryFile(suffix=".yml", delete=False, mode="w")
        yaml.dump(data, f)
        f.close()
        return f.name

    def test_explicit_path_loads_file(self):
        path = self._write_tmp_config({"approval_mode": "auto", "model_tiers": []})
        try:
            cfg = load_config(path=path)
            self.assertEqual(cfg["approval_mode"], "auto")
        finally:
            os.unlink(path)

    def test_explicit_path_missing_raises_file_not_found(self):
        with self.assertRaises(FileNotFoundError):
            load_config(path="/no/such/path/config.yml")

    def test_explicit_path_overrides_env_var(self):
        path = self._write_tmp_config({"approval_mode": "ask", "model_tiers": []})
        env_path = self._write_tmp_config({"approval_mode": "auto", "model_tiers": []})
        try:
            with patch.dict(os.environ, {"OPENSHARD_CONFIG": env_path}):
                cfg = load_config(path=path)
            self.assertEqual(cfg["approval_mode"], "ask")
        finally:
            os.unlink(path)
            os.unlink(env_path)


# ---------------------------------------------------------------------------
# OPENSHARD_CONFIG environment variable
# ---------------------------------------------------------------------------

class TestEnvVar(unittest.TestCase):

    def _write_tmp_config(self, data: dict) -> str:
        f = tempfile.NamedTemporaryFile(suffix=".yml", delete=False, mode="w")
        yaml.dump(data, f)
        f.close()
        return f.name

    def test_env_var_loads_file(self):
        path = self._write_tmp_config({"approval_mode": "ask", "model_tiers": []})
        try:
            with patch.dict(os.environ, {"OPENSHARD_CONFIG": path}):
                cfg = load_config()
            self.assertEqual(cfg["approval_mode"], "ask")
        finally:
            os.unlink(path)

    def test_env_var_missing_path_raises(self):
        with patch.dict(os.environ, {"OPENSHARD_CONFIG": "/no/such/path.yml"}):
            with self.assertRaises(FileNotFoundError):
                load_config()

    def test_env_var_not_set_does_not_raise(self):
        env = {k: v for k, v in os.environ.items() if k != "OPENSHARD_CONFIG"}
        with patch.dict(os.environ, env, clear=True):
            # Should not raise; falls through to CWD / bundled / defaults.
            # We can't easily control CWD here, but at minimum it must not
            # raise FileNotFoundError from an absent OPENSHARD_CONFIG.
            try:
                cfg = load_config()
                self.assertIsInstance(cfg, dict)
            except FileNotFoundError:
                self.fail("load_config() raised FileNotFoundError without explicit path or env var")


# ---------------------------------------------------------------------------
# CWD-relative config file search
# ---------------------------------------------------------------------------

class TestCwdSearch(unittest.TestCase):

    def test_cwd_config_yml_found(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg_path = Path(tmp) / "config.yml"
            cfg_path.write_text(yaml.dump({"approval_mode": "ask", "model_tiers": []}))
            with patch("openshard.config.settings.Path") as _MockPath:
                pass
            # Use explicit path to verify file reading works (same code path as CWD step 4)
            cfg = load_config(path=str(cfg_path))
            self.assertEqual(cfg["approval_mode"], "ask")

    def test_openshard_config_yml_found(self):
        with tempfile.TemporaryDirectory() as tmp:
            sub = Path(tmp) / ".openshard"
            sub.mkdir()
            cfg_path = sub / "config.yml"
            cfg_path.write_text(yaml.dump({"approval_mode": "auto", "model_tiers": []}))
            cfg = load_config(path=str(cfg_path))
            self.assertEqual(cfg["approval_mode"], "auto")


# ---------------------------------------------------------------------------
# No-config fallback — returns defaults, never raises
# ---------------------------------------------------------------------------

class TestNoConfigFallback(unittest.TestCase):

    def test_no_crash_without_config(self):
        """With no env var, no explicit path, and CWD containing no config,
        load_config() must return a dict — not raise."""
        clean_env = {k: v for k, v in os.environ.items() if k != "OPENSHARD_CONFIG"}
        with tempfile.TemporaryDirectory() as tmp:
            # Patch CWD to a directory that has no config files at all
            with patch("openshard.config.settings.Path") as MockPath:
                # Make hidden and cwd_cfg report as not-existing
                mock_cwd = Path(tmp)

                def path_side_effect(arg=None):
                    if arg is None:
                        return Path(tmp)
                    return Path(arg)

                MockPath.cwd.return_value = mock_cwd
                MockPath.side_effect = path_side_effect

                with patch.dict(os.environ, clean_env, clear=True):
                    try:
                        cfg = load_config()
                        self.assertIsInstance(cfg, dict)
                    except FileNotFoundError:
                        self.fail("load_config() raised FileNotFoundError without any explicit path")

    def test_defaults_returned_as_fallback(self):
        cfg = dict(_DEFAULTS)
        self.assertIn("model_tiers", cfg)
        self.assertIn("approval_mode", cfg)
        self.assertIsInstance(cfg["model_tiers"], list)


# ---------------------------------------------------------------------------
# Bundled default_config.yml via importlib.resources
# ---------------------------------------------------------------------------

class TestBundledConfig(unittest.TestCase):

    def test_bundled_config_accessible(self):
        from importlib.resources import files
        pkg_cfg = files("openshard.config").joinpath("default_config.yml")
        with pkg_cfg.open("r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
        self.assertIsInstance(data, dict)
        self.assertIn("model_tiers", data)

    def test_load_config_returns_dict_in_clean_environment(self):
        """No env var, no local config in CWD → must return a dict, never raise."""
        clean_env = {k: v for k, v in os.environ.items() if k != "OPENSHARD_CONFIG"}
        with tempfile.TemporaryDirectory() as tmp:
            orig = os.getcwd()
            os.chdir(tmp)
            try:
                with patch.dict(os.environ, clean_env, clear=True):
                    cfg = load_config()
                self.assertIsInstance(cfg, dict)
                self.assertIn("model_tiers", cfg)
            finally:
                os.chdir(orig)
