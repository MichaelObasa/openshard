from __future__ import annotations

import unittest

from openshard.providers.base import ModelInfo
from openshard.providers.manager import InventoryEntry
from openshard.scoring.requirements import TaskRequirements
from openshard.scoring.scorer import score_model, select_candidate


def _entry(
    id="m1",
    context_window=None,
    supports_vision=False,
    supports_tools=False,
    pricing=None,
) -> InventoryEntry:
    return InventoryEntry(
        provider="openrouter",
        model=ModelInfo(
            id=id,
            name=id,
            pricing=pricing or {},
            context_window=context_window,
            supports_vision=supports_vision,
            supports_tools=supports_tools,
        ),
    )


class TestScoreModel(unittest.TestCase):

    def test_base_score(self):
        entry = _entry("basic")
        req = TaskRequirements()
        self.assertEqual(score_model(entry, req), 10.0)

    def test_vision_bonus_when_needed(self):
        entry = _entry("vision", supports_vision=True)
        req = TaskRequirements(needs_vision=True)
        self.assertEqual(score_model(entry, req), 12.0)

    def test_no_vision_bonus_when_not_needed(self):
        entry = _entry("vision", supports_vision=True)
        req = TaskRequirements(needs_vision=False)
        self.assertEqual(score_model(entry, req), 10.0)

    def test_security_claude_bonus(self):
        entry = _entry("anthropic/claude-sonnet-4.6")
        req = TaskRequirements(security_sensitive=True)
        self.assertEqual(score_model(entry, req), 13.0)

    def test_security_bonus_not_applied_to_non_claude(self):
        entry = _entry("openai/gpt-4o")
        req = TaskRequirements(security_sensitive=True)
        self.assertEqual(score_model(entry, req), 10.0)

    def test_context_window_100k_bonus(self):
        entry = _entry("big", context_window=128_000)
        req = TaskRequirements()
        self.assertEqual(score_model(entry, req), 11.0)

    def test_context_window_200k_bonus(self):
        entry = _entry("huge", context_window=200_000)
        req = TaskRequirements()
        self.assertEqual(score_model(entry, req), 12.0)

    def test_cheap_cost_bonus(self):
        entry = _entry("cheap", pricing={"prompt": "0.20"})
        req = TaskRequirements()
        self.assertEqual(score_model(entry, req), 13.0)  # <= 1.0 (+2) and <= 0.25 (+1)

    def test_expensive_model_no_cost_bonus(self):
        entry = _entry("pricey", pricing={"prompt": "5.00"})
        req = TaskRequirements()
        self.assertEqual(score_model(entry, req), 10.0)


    def test_alias_id_penalised(self):
        entry = _entry("~anthropic/claude-opus-latest")
        req = TaskRequirements()
        self.assertEqual(score_model(entry, req), 9.0)

    def test_versioned_id_beats_alias_when_otherwise_equal(self):
        versioned = _entry("anthropic/claude-opus-4.7")
        alias = _entry("~anthropic/claude-opus-latest")
        req = TaskRequirements()
        self.assertGreater(score_model(versioned, req), score_model(alias, req))

    def test_non_alias_id_unaffected(self):
        entry = _entry("anthropic/claude-opus-4.7")
        req = TaskRequirements()
        self.assertEqual(score_model(entry, req), 10.0)


class TestSelectCandidate(unittest.TestCase):

    def test_empty_list_returns_none(self):
        req = TaskRequirements()
        self.assertIsNone(select_candidate([], req))

    def test_returns_top_scorer(self):
        cheap = _entry("cheap", pricing={"prompt": "0.10"})
        pricey = _entry("pricey", pricing={"prompt": "5.00"})
        req = TaskRequirements()
        result = select_candidate([cheap, pricey], req)
        self.assertEqual(result.model.id, "cheap")

    def test_returns_none_when_all_filtered(self):
        entry = _entry("no-vision", supports_vision=False)
        req = TaskRequirements(needs_vision=True)
        self.assertIsNone(select_candidate([entry], req))
