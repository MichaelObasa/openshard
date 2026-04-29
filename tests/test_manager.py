from __future__ import annotations

import time
import unittest
from unittest.mock import MagicMock, patch

from openshard.providers.base import ModelInfo, ProviderAuthError, ProviderError
from openshard.providers.manager import ProviderManager, UnifiedInventory


def _fresh_cache(provider: str, rows: list[dict] | None = None) -> dict:
    return {
        "cached_at": time.time(),
        "models": {provider: rows or []},
    }


def _model_row(**kwargs) -> dict:
    base = {
        "id": "m1", "name": "Model 1", "pricing": {},
        "context_window": None, "max_output_tokens": None,
        "supports_vision": False, "supports_tools": False,
    }
    base.update(kwargs)
    return base


_NO_KEYS = {"OPENROUTER_API_KEY": "", "ANTHROPIC_API_KEY": "", "OPENAI_API_KEY": ""}


class TestProviderManager(unittest.TestCase):

    def test_no_providers_configured(self):
        with patch.dict("os.environ", _NO_KEYS):
            manager = ProviderManager()
        self.assertEqual(manager.providers, {})
        with patch("openshard.providers.manager.load_cache", return_value=None):
            inventory = manager.get_inventory()
        self.assertIsInstance(inventory, UnifiedInventory)
        self.assertEqual(inventory.models, [])
        self.assertEqual(inventory.provider_count, 0)

    def test_single_provider_detected(self):
        mock_model = ModelInfo(id="r1", name="Router 1", pricing={})
        mock_provider = MagicMock()
        mock_provider.list_models.return_value = [mock_model]

        env = {**_NO_KEYS, "OPENROUTER_API_KEY": "test-key"}
        with patch.dict("os.environ", env), \
             patch("openshard.providers.manager._make_provider", return_value=mock_provider):
            manager = ProviderManager()

        self.assertIn("openrouter", manager.providers)
        self.assertEqual(len(manager.providers), 1)

        with patch("openshard.providers.manager.load_cache", return_value=None), \
             patch("openshard.providers.manager.save_cache"):
            inventory = manager.get_inventory()

        self.assertEqual(len(inventory.models), 1)
        self.assertEqual(inventory.models[0].provider, "openrouter")
        self.assertEqual(inventory.models[0].model.id, "r1")
        self.assertEqual(inventory.provider_count, 1)

    def test_auth_error_on_init_skipped(self):
        env = {**_NO_KEYS, "OPENROUTER_API_KEY": "bad-key"}
        with patch.dict("os.environ", env), \
             patch("openshard.providers.manager._make_provider",
                   side_effect=ProviderAuthError("invalid key")):
            manager = ProviderManager()

        self.assertNotIn("openrouter", manager.providers)
        self.assertEqual(manager.providers, {})

    def test_list_models_error_skipped(self):
        mock_or = MagicMock()
        mock_or.list_models.side_effect = ProviderError("timeout")

        mock_an = MagicMock()
        mock_an.list_models.return_value = [
            ModelInfo(id="claude-x", name="Claude X", pricing={})
        ]

        def make_provider(name: str, key: str):
            return mock_or if name == "openrouter" else mock_an

        env = {**_NO_KEYS, "OPENROUTER_API_KEY": "k1", "ANTHROPIC_API_KEY": "k2"}
        with patch.dict("os.environ", env), \
             patch("openshard.providers.manager._make_provider", side_effect=make_provider):
            manager = ProviderManager()

        with patch("openshard.providers.manager.load_cache", return_value=None), \
             patch("openshard.providers.manager.save_cache"):
            inventory = manager.get_inventory()

        provider_names = {e.provider for e in inventory.models}
        self.assertNotIn("openrouter", provider_names)
        self.assertIn("anthropic", provider_names)

    def test_cache_used_when_fresh(self):
        cache = _fresh_cache("openrouter", [_model_row(id="cached-model", name="Cached")])
        mock_provider = MagicMock()

        env = {**_NO_KEYS, "OPENROUTER_API_KEY": "key"}
        with patch.dict("os.environ", env), \
             patch("openshard.providers.manager._make_provider", return_value=mock_provider):
            manager = ProviderManager()

        with patch("openshard.providers.manager.load_cache", return_value=cache):
            inventory = manager.get_inventory(refresh=False)

        mock_provider.list_models.assert_not_called()
        self.assertEqual(len(inventory.models), 1)
        self.assertEqual(inventory.models[0].model.id, "cached-model")

    def test_refresh_bypasses_cache(self):
        cache = _fresh_cache("openrouter", [_model_row(id="cached-model", name="Cached")])
        live_model = ModelInfo(id="live-model", name="Live", pricing={})
        mock_provider = MagicMock()
        mock_provider.list_models.return_value = [live_model]

        env = {**_NO_KEYS, "OPENROUTER_API_KEY": "key"}
        with patch.dict("os.environ", env), \
             patch("openshard.providers.manager._make_provider", return_value=mock_provider):
            manager = ProviderManager()

        with patch("openshard.providers.manager.load_cache", return_value=cache), \
             patch("openshard.providers.manager.save_cache"):
            inventory = manager.get_inventory(refresh=True)

        mock_provider.list_models.assert_called_once()
        self.assertEqual(inventory.models[0].model.id, "live-model")

    def test_cached_models_rehydrate_to_modelinfo(self):
        row = _model_row(
            id="claude-3", name="Claude 3",
            pricing={"prompt": "0.01"},
            context_window=200000, max_output_tokens=4096,
            supports_vision=True, supports_tools=True,
        )
        cache = _fresh_cache("anthropic", [row])
        mock_provider = MagicMock()

        env = {**_NO_KEYS, "ANTHROPIC_API_KEY": "key"}
        with patch.dict("os.environ", env), \
             patch("openshard.providers.manager._make_provider", return_value=mock_provider):
            manager = ProviderManager()

        with patch("openshard.providers.manager.load_cache", return_value=cache):
            inventory = manager.get_inventory()

        self.assertEqual(len(inventory.models), 1)
        m = inventory.models[0].model
        self.assertIsInstance(m, ModelInfo)
        self.assertEqual(m.id, "claude-3")
        self.assertEqual(m.context_window, 200000)
        self.assertEqual(m.max_output_tokens, 4096)
        self.assertTrue(m.supports_vision)
        self.assertTrue(m.supports_tools)
