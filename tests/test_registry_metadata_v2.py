"""Metadata v2 completeness and safety tests.

Every ModelEntry must carry valid metadata v2 fields, and pricing must stay
deferred (empty/unknown) in this branch. Authoritative runtime pricing remains
MODEL_PRICING in openshard/providers/openrouter.py; these tests pin the contract
that the registry's pricing scaffolding is intentionally unpopulated in v2.
"""

from __future__ import annotations

import unittest

from openshard.models.registry import (
    METADATA_VERSION,
    PRICING_SOURCES,
    RISK_LEVELS,
    SOURCE_VALUES,
    StaticPricing,
    _REGISTRY,
)


class TestMetadataV2Fields(unittest.TestCase):
    def test_every_entry_has_metadata_version_2(self) -> None:
        self.assertEqual(METADATA_VERSION, "2")
        for entry in _REGISTRY:
            with self.subTest(model_id=entry.id):
                self.assertEqual(entry.metadata_version, "2")

    def test_every_entry_source_is_valid(self) -> None:
        for entry in _REGISTRY:
            with self.subTest(model_id=entry.id):
                self.assertIn(entry.source, SOURCE_VALUES)

    def test_every_entry_risk_level_is_valid(self) -> None:
        for entry in _REGISTRY:
            with self.subTest(model_id=entry.id):
                self.assertIn(entry.risk_level, RISK_LEVELS)

    def test_recommended_and_avoid_are_string_tuples(self) -> None:
        for entry in _REGISTRY:
            with self.subTest(model_id=entry.id):
                self.assertIsInstance(entry.recommended_for, tuple)
                self.assertIsInstance(entry.avoid_for, tuple)
                for token in (*entry.recommended_for, *entry.avoid_for):
                    self.assertIsInstance(token, str)

    def test_recommended_for_and_avoid_for_disjoint(self) -> None:
        for entry in _REGISTRY:
            with self.subTest(model_id=entry.id):
                overlap = set(entry.recommended_for) & set(entry.avoid_for)
                self.assertEqual(
                    overlap,
                    set(),
                    f"{entry.id} has tokens in both recommended_for and avoid_for: {overlap}",
                )


class TestMetadataV2Pricing(unittest.TestCase):
    def test_pricing_is_static_pricing_instance(self) -> None:
        for entry in _REGISTRY:
            with self.subTest(model_id=entry.id):
                self.assertIsInstance(entry.pricing, StaticPricing)

    def test_pricing_source_is_valid(self) -> None:
        for entry in _REGISTRY:
            with self.subTest(model_id=entry.id):
                self.assertIn(entry.pricing.pricing_source, PRICING_SOURCES)

    def test_pricing_costs_are_none_or_float(self) -> None:
        for entry in _REGISTRY:
            with self.subTest(model_id=entry.id):
                for cost in (
                    entry.pricing.input_cost_per_mtok,
                    entry.pricing.output_cost_per_mtok,
                ):
                    self.assertTrue(
                        cost is None or isinstance(cost, float),
                        f"{entry.id} pricing cost must be None or float, got {cost!r}",
                    )

    def test_pricing_deferred_all_unknown_in_v2(self) -> None:
        # Pricing is deliberately deferred in metadata v2: every entry's pricing
        # stays empty/unknown. A future live-pricing branch will change this.
        for entry in _REGISTRY:
            with self.subTest(model_id=entry.id):
                self.assertIsNone(entry.pricing.input_cost_per_mtok)
                self.assertIsNone(entry.pricing.output_cost_per_mtok)
                self.assertEqual(entry.pricing.pricing_source, "unknown")
                self.assertIsNone(entry.pricing.last_verified_at)


if __name__ == "__main__":
    unittest.main()
