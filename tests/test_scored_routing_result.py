from __future__ import annotations

import unittest

from openshard.providers.base import ModelInfo
from openshard.providers.manager import InventoryEntry
from openshard.scoring.requirements import TaskRequirements
from openshard.scoring.scorer import select_with_info


def _make_entry(model_id: str, provider: str = "openrouter", **kwargs) -> InventoryEntry:
    return InventoryEntry(
        provider=provider,
        model=ModelInfo(
            id=model_id,
            name=model_id,
            pricing=kwargs.get("pricing", {}),
            context_window=kwargs.get("context_window"),
            max_output_tokens=None,
            supports_vision=kwargs.get("supports_vision", False),
            supports_tools=kwargs.get("supports_tools", False),
        ),
    )


class TestSelectWithInfo(unittest.TestCase):

    def test_select_with_info_returns_winner(self):
        entry = _make_entry("openrouter/fast-model")
        reqs = TaskRequirements()
        result = select_with_info([entry], reqs, "standard")
        self.assertFalse(result.used_fallback)
        self.assertEqual(result.selected_model, "openrouter/fast-model")
        self.assertEqual(result.candidate_count, 1)

    def test_select_with_info_fallback_when_empty(self):
        entry = _make_entry("openrouter/no-vision", supports_vision=False)
        reqs = TaskRequirements(needs_vision=True)
        result = select_with_info([entry], reqs, "visual")
        self.assertTrue(result.used_fallback)
        self.assertIsNone(result.selected_model)
        self.assertIsNone(result.selected_provider)
        self.assertEqual(result.candidate_count, 0)

    def test_select_with_info_category_preserved(self):
        entry = _make_entry("openrouter/fast-model")
        reqs = TaskRequirements()
        result = select_with_info([entry], reqs, "security")
        self.assertEqual(result.category, "security")

    def test_select_with_info_provider_preserved(self):
        entry = _make_entry("claude-sonnet", provider="anthropic")
        reqs = TaskRequirements()
        result = select_with_info([entry], reqs, "standard")
        self.assertEqual(result.selected_provider, "anthropic")
        self.assertEqual(result.selected_model, "claude-sonnet")


if __name__ == "__main__":
    unittest.main()
