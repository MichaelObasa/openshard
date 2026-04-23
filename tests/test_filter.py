from __future__ import annotations

import unittest

from openshard.providers.base import ModelInfo
from openshard.providers.manager import InventoryEntry
from openshard.scoring.filter import filter_inventory
from openshard.scoring.requirements import TaskRequirements


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


class TestFilterInventory(unittest.TestCase):

    def test_vision_filter_excludes_non_vision(self):
        entries = [_entry("no-vision"), _entry("vision", supports_vision=True)]
        req = TaskRequirements(needs_vision=True)
        result = filter_inventory(entries, req)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].model.id, "vision")

    def test_vision_filter_keeps_all_when_not_required(self):
        entries = [_entry("no-vision"), _entry("vision", supports_vision=True)]
        req = TaskRequirements(needs_vision=False)
        result = filter_inventory(entries, req)
        self.assertEqual(len(result), 2)

    def test_tools_filter_excludes_no_tools(self):
        entries = [_entry("no-tools"), _entry("tools", supports_tools=True)]
        req = TaskRequirements(needs_tools=True)
        result = filter_inventory(entries, req)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].model.id, "tools")

    def test_tools_filter_keeps_all_when_not_required(self):
        entries = [_entry("no-tools"), _entry("tools", supports_tools=True)]
        req = TaskRequirements(needs_tools=False)
        result = filter_inventory(entries, req)
        self.assertEqual(len(result), 2)

    def test_context_window_filter_excludes_small(self):
        entries = [_entry("small", context_window=4000), _entry("large", context_window=128000)]
        req = TaskRequirements(min_context_window=8000)
        result = filter_inventory(entries, req)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].model.id, "large")

    def test_context_window_keeps_none_window(self):
        entries = [_entry("unknown"), _entry("large", context_window=128000)]
        req = TaskRequirements(min_context_window=8000)
        result = filter_inventory(entries, req)
        self.assertEqual(len(result), 2)

    def test_cost_filter_excludes_expensive(self):
        entries = [
            _entry("cheap", pricing={"prompt": "0.20"}),
            _entry("expensive", pricing={"prompt": "5.00"}),
        ]
        req = TaskRequirements(preferred_max_cost_per_m=1.0)
        result = filter_inventory(entries, req)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].model.id, "cheap")

    def test_cost_filter_keeps_missing_pricing(self):
        entries = [_entry("no-price"), _entry("expensive", pricing={"prompt": "5.00"})]
        req = TaskRequirements(preferred_max_cost_per_m=1.0)
        result = filter_inventory(entries, req)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].model.id, "no-price")

    def test_cost_filter_keeps_unparseable_pricing(self):
        entries = [_entry("weird", pricing={"prompt": "n/a"})]
        req = TaskRequirements(preferred_max_cost_per_m=1.0)
        result = filter_inventory(entries, req)
        self.assertEqual(len(result), 1)

    def test_no_requirements_returns_all(self):
        entries = [_entry("a"), _entry("b"), _entry("c")]
        req = TaskRequirements()
        result = filter_inventory(entries, req)
        self.assertEqual(len(result), 3)
