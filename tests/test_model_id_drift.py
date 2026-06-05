"""Drift-prevention tests: hardcoded model IDs must stay registry-backed.

These tests fail if config tier defaults, routing constants, the friendly-name
mapping, or cost baselines reference a model ID that is not in the registry.
The registry is the single source of truth for model existence (see
openshard.models.registry.is_known_model).

The pricing snapshot keeps a few deliberate legacy IDs; those are tracked
explicitly via LEGACY_PRICING_IDS_ALLOWED_UNTIL_METADATA_V2 rather than ignored.
"""

from __future__ import annotations

import unittest

import yaml

from openshard.config.settings import _DEFAULTS
from openshard.cost.baseline import (
    BASELINE_MODELS,
    FRONTIER_BASELINE_MODEL,
    FULL_COMPARISON_MODELS,
)
from openshard.history.shard_contract import _MODEL_FRIENDLY_NAMES
from openshard.models.registry import is_known_model, registry_ids
from openshard.providers.openrouter import MODEL_PRICING
from openshard.routing.engine import (
    MODEL_CHEAP,
    MODEL_COMPLEX,
    MODEL_ESCALATE,
    MODEL_MAIN,
    MODEL_STRONG,
    MODEL_VISUAL,
)

# Legacy IDs kept in the pricing snapshot on purpose. Consolidation onto the
# registry is deferred to feat/registry-metadata-v2. Mirrors the comment above
# MODEL_PRICING in openshard/providers/openrouter.py.
LEGACY_PRICING_IDS_ALLOWED_UNTIL_METADATA_V2: frozenset[str] = frozenset(
    {
        "anthropic/claude-opus-4.6",
        "anthropic/claude-haiku-4.5-20251001",
        "openai/gpt-4o",
        "openai/gpt-4o-mini",
    }
)


def _normalize_slug(model_id: str) -> str:
    """Collapse a model slug for alias-tolerant comparison.

    Strips the optional provider prefix and removes "." and "-" so that
    intentional hyphen/dot aliases (claude-opus-4-7 vs claude-opus-4.7)
    compare equal, while genuinely different IDs do not.
    """
    tail = model_id.lower().strip().split("/", 1)[-1]
    return tail.replace("-", "").replace(".", "")


class TestConfigTierDrift(unittest.TestCase):
    def _config_model_ids(self, config: dict) -> list[str]:
        ids: list[str] = []
        for tier in config.get("model_tiers", []):
            model = tier.get("model")
            if model:
                ids.append(model)
        for key in ("planning_model", "execution_model", "fixer_model"):
            value = config.get(key)
            if value:
                ids.append(value)
        return ids

    def test_defaults_reference_known_models(self) -> None:
        for model_id in self._config_model_ids(_DEFAULTS):
            with self.subTest(model_id=model_id):
                self.assertTrue(
                    is_known_model(model_id),
                    f"Config default references unknown model: {model_id}",
                )

    def test_bundled_default_config_references_known_models(self) -> None:
        from importlib.resources import files

        pkg_cfg = files("openshard.config").joinpath("default_config.yml")
        with pkg_cfg.open("r", encoding="utf-8") as fh:
            config = yaml.safe_load(fh) or {}
        ids = self._config_model_ids(config)
        self.assertTrue(ids, "Bundled default_config.yml has no model IDs to check")
        for model_id in ids:
            with self.subTest(model_id=model_id):
                self.assertTrue(
                    is_known_model(model_id),
                    f"Bundled default_config.yml references unknown model: {model_id}",
                )


class TestRoutingConstantDrift(unittest.TestCase):
    def test_all_routing_constants_in_registry(self) -> None:
        ids = registry_ids()
        for model_id in (
            MODEL_CHEAP,
            MODEL_MAIN,
            MODEL_STRONG,
            MODEL_ESCALATE,
            MODEL_VISUAL,
            MODEL_COMPLEX,
        ):
            with self.subTest(model_id=model_id):
                self.assertIn(
                    model_id, ids, f"Routing constant references unknown model: {model_id}"
                )


class TestFriendlyNameDrift(unittest.TestCase):
    def test_every_friendly_key_maps_to_known_model(self) -> None:
        known = {_normalize_slug(i) for i in registry_ids()}
        for key in _MODEL_FRIENDLY_NAMES:
            with self.subTest(key=key):
                self.assertIn(
                    _normalize_slug(key),
                    known,
                    f"Friendly-name key references unknown model: {key}",
                )


class TestCostBaselineDrift(unittest.TestCase):
    def test_baseline_models_known(self) -> None:
        for _label, model_id in BASELINE_MODELS:
            with self.subTest(model_id=model_id):
                self.assertTrue(is_known_model(model_id), model_id)

    def test_frontier_baseline_known(self) -> None:
        self.assertTrue(is_known_model(FRONTIER_BASELINE_MODEL), FRONTIER_BASELINE_MODEL)

    def test_full_comparison_models_known(self) -> None:
        for _label, model_id in FULL_COMPARISON_MODELS:
            with self.subTest(model_id=model_id):
                self.assertTrue(is_known_model(model_id), model_id)


class TestPricingDrift(unittest.TestCase):
    def test_pricing_keys_known_or_explicitly_allowed(self) -> None:
        for model_id in MODEL_PRICING:
            with self.subTest(model_id=model_id):
                allowed = (
                    is_known_model(model_id)
                    or model_id in LEGACY_PRICING_IDS_ALLOWED_UNTIL_METADATA_V2
                )
                self.assertTrue(
                    allowed,
                    f"Pricing key is neither registry-known nor allowlisted: {model_id}. "
                    "Add it to the registry or to "
                    "LEGACY_PRICING_IDS_ALLOWED_UNTIL_METADATA_V2.",
                )

    def test_allowlist_has_no_stale_entries(self) -> None:
        # Every allowlisted legacy ID should still be present in the pricing
        # snapshot. If one is gone, drop it from the allowlist too.
        for model_id in LEGACY_PRICING_IDS_ALLOWED_UNTIL_METADATA_V2:
            with self.subTest(model_id=model_id):
                self.assertIn(model_id, MODEL_PRICING, model_id)


if __name__ == "__main__":
    unittest.main()
