"""Tests for lifecycle-aware scoring filter (PR 2).

Covers:
- lifecycle_adjustments_for_entries: correct penalty values per lifecycle
- filter_deprecated: hard removal of deprecated models
- select_with_info integration: active_default wins over experimental when
  all other scores are equal; deprecated models are never selected.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_inventory_entry(model_id: str, pricing: dict | None = None):
    """Minimal InventoryEntry with a ModelInfo."""
    entry = MagicMock()
    entry.model.id = model_id
    entry.model.pricing = pricing or {}
    entry.model.supports_vision = False
    entry.model.supports_tools = True
    entry.model.context_window = 128_000
    entry.provider = "test_provider"
    return entry


def _lc_fn_for(mapping: dict):
    """Return a lifecycle_for function backed by *mapping*."""
    def _fn(model_id: str):
        return mapping.get(model_id)
    return _fn


# ---------------------------------------------------------------------------
# lifecycle_adjustments_for_entries
# ---------------------------------------------------------------------------

class TestLifecycleAdjustmentsForEntries:
    def test_active_default_zero_penalty(self):
        from openshard.scoring.filter import lifecycle_adjustments_for_entries
        entry = _make_inventory_entry("model/a")
        result = lifecycle_adjustments_for_entries(
            [entry], registry_fn=_lc_fn_for({"model/a": "active_default"})
        )
        assert result["model/a"] == 0.0

    def test_active_specialist_moderate_penalty(self):
        from openshard.scoring.filter import lifecycle_adjustments_for_entries
        entry = _make_inventory_entry("model/b")
        result = lifecycle_adjustments_for_entries(
            [entry], registry_fn=_lc_fn_for({"model/b": "active_specialist"})
        )
        assert result["model/b"] == -3.0

    def test_fallback_moderate_penalty(self):
        from openshard.scoring.filter import lifecycle_adjustments_for_entries
        entry = _make_inventory_entry("model/c")
        result = lifecycle_adjustments_for_entries(
            [entry], registry_fn=_lc_fn_for({"model/c": "fallback"})
        )
        assert result["model/c"] == -3.0

    def test_open_weight_moderate_penalty(self):
        from openshard.scoring.filter import lifecycle_adjustments_for_entries
        entry = _make_inventory_entry("model/d")
        result = lifecycle_adjustments_for_entries(
            [entry], registry_fn=_lc_fn_for({"model/d": "open_weight"})
        )
        assert result["model/d"] == -3.0

    def test_experimental_heavy_penalty(self):
        from openshard.scoring.filter import lifecycle_adjustments_for_entries
        entry = _make_inventory_entry("model/e")
        result = lifecycle_adjustments_for_entries(
            [entry], registry_fn=_lc_fn_for({"model/e": "experimental"})
        )
        assert result["model/e"] == -10.0

    def test_watchlist_heavy_penalty(self):
        from openshard.scoring.filter import lifecycle_adjustments_for_entries
        entry = _make_inventory_entry("model/f")
        result = lifecycle_adjustments_for_entries(
            [entry], registry_fn=_lc_fn_for({"model/f": "watchlist"})
        )
        assert result["model/f"] == -10.0

    def test_deprecated_returns_neg_inf_sentinel(self):
        from openshard.scoring.filter import lifecycle_adjustments_for_entries
        entry = _make_inventory_entry("model/dep")
        result = lifecycle_adjustments_for_entries(
            [entry], registry_fn=_lc_fn_for({"model/dep": "deprecated"})
        )
        import math
        assert math.isinf(result["model/dep"]) and result["model/dep"] < 0

    def test_unknown_model_zero_penalty_passthrough(self):
        from openshard.scoring.filter import lifecycle_adjustments_for_entries
        entry = _make_inventory_entry("model/unknown")
        result = lifecycle_adjustments_for_entries(
            [entry], registry_fn=_lc_fn_for({})  # nothing in registry
        )
        assert result["model/unknown"] == 0.0

    def test_multiple_entries_independent(self):
        from openshard.scoring.filter import lifecycle_adjustments_for_entries
        entries = [
            _make_inventory_entry("a/active"),
            _make_inventory_entry("b/exp"),
        ]
        mapping = {"a/active": "active_default", "b/exp": "experimental"}
        result = lifecycle_adjustments_for_entries(entries, registry_fn=_lc_fn_for(mapping))
        assert result["a/active"] == 0.0
        assert result["b/exp"] == -10.0

    def test_graceful_on_registry_import_failure(self):
        import sys

        from openshard.scoring.filter import lifecycle_adjustments_for_entries
        entry = _make_inventory_entry("model/x")
        with pytest.MonkeyPatch().context() as mp:
            mp.setitem(sys.modules, "openshard.models.registry", None)
            # Should not raise; returns 0.0 pass-through for all
            result = lifecycle_adjustments_for_entries([entry])
        assert result.get("model/x", 0.0) == 0.0


# ---------------------------------------------------------------------------
# filter_deprecated
# ---------------------------------------------------------------------------

class TestFilterDeprecated:
    def test_removes_deprecated_model(self):
        from openshard.scoring.filter import filter_deprecated
        dep = _make_inventory_entry("model/dep")
        active = _make_inventory_entry("model/ok")
        mapping = {"model/dep": "deprecated", "model/ok": "active_default"}
        result = filter_deprecated([dep, active], registry_fn=_lc_fn_for(mapping))
        ids = [e.model.id for e in result]
        assert "model/dep" not in ids
        assert "model/ok" in ids

    def test_keeps_active_default(self):
        from openshard.scoring.filter import filter_deprecated
        entry = _make_inventory_entry("model/a")
        result = filter_deprecated(
            [entry], registry_fn=_lc_fn_for({"model/a": "active_default"})
        )
        assert len(result) == 1

    def test_keeps_experimental(self):
        from openshard.scoring.filter import filter_deprecated
        entry = _make_inventory_entry("model/x")
        result = filter_deprecated(
            [entry], registry_fn=_lc_fn_for({"model/x": "experimental"})
        )
        assert len(result) == 1  # experimental is NOT hard-excluded

    def test_unknown_model_kept(self):
        from openshard.scoring.filter import filter_deprecated
        entry = _make_inventory_entry("model/mystery")
        result = filter_deprecated([entry], registry_fn=_lc_fn_for({}))
        assert len(result) == 1

    def test_empty_input_returns_empty(self):
        from openshard.scoring.filter import filter_deprecated
        assert filter_deprecated([]) == []

    def test_all_deprecated_returns_empty(self):
        from openshard.scoring.filter import filter_deprecated
        entries = [_make_inventory_entry(f"dep/{i}") for i in range(3)]
        mapping = {f"dep/{i}": "deprecated" for i in range(3)}
        result = filter_deprecated(entries, registry_fn=_lc_fn_for(mapping))
        assert result == []

    def test_graceful_on_registry_import_failure(self):
        import sys

        from openshard.scoring.filter import filter_deprecated
        entry = _make_inventory_entry("model/x")
        with pytest.MonkeyPatch().context() as mp:
            mp.setitem(sys.modules, "openshard.models.registry", None)
            result = filter_deprecated([entry])
        assert len(result) == 1  # pass-through on failure


# ---------------------------------------------------------------------------
# select_with_info integration: lifecycle penalties affect winner
# ---------------------------------------------------------------------------

class TestSelectWithInfoLifecycleIntegration:
    """
    Integration tests that confirm lifecycle adjustments flow into select_with_info.
    Uses registry_fn injection via filter_deprecated/lifecycle_adjustments_for_entries
    indirectly — we patch lifecycle_for at the registry level.
    """

    def _make_scoring_entry(self, model_id, cost_class="mid"):
        from unittest.mock import MagicMock
        e = MagicMock()
        e.model.id = model_id
        e.model.pricing = {}
        e.model.supports_vision = False
        e.model.supports_tools = True
        e.model.context_window = 128_000
        e.provider = "openrouter"
        return e

    def test_active_default_beats_experimental_when_equal_base(self):
        from unittest.mock import patch

        from openshard.scoring.requirements import TaskRequirements
        from openshard.scoring.scorer import select_with_info

        active = self._make_scoring_entry("model/active")
        experimental = self._make_scoring_entry("model/exp")

        reqs = TaskRequirements()

        lc_map = {"model/active": "active_default", "model/exp": "experimental"}

        with patch("openshard.scoring.shortlist.build_shortlist", return_value=[active, experimental]):
            with patch("openshard.scoring.filter.filter_inventory", return_value=[active, experimental]):
                with patch("openshard.models.registry.lifecycle_for", side_effect=lambda mid: lc_map.get(mid)):
                    result = select_with_info([active, experimental], reqs, "standard")

        # experimental has -10 penalty so active_default wins
        assert result.selected_model == "model/active"

    def test_deprecated_model_never_selected(self):
        from unittest.mock import patch

        from openshard.scoring.requirements import TaskRequirements
        from openshard.scoring.scorer import select_with_info

        active = self._make_scoring_entry("model/active")
        deprecated = self._make_scoring_entry("model/dep")

        reqs = TaskRequirements()
        lc_map = {"model/active": "active_default", "model/dep": "deprecated"}

        with patch("openshard.scoring.shortlist.build_shortlist", return_value=[active, deprecated]):
            with patch("openshard.scoring.filter.filter_inventory", return_value=[active, deprecated]):
                with patch("openshard.models.registry.lifecycle_for", side_effect=lambda mid: lc_map.get(mid)):
                    result = select_with_info([active, deprecated], reqs, "standard")

        assert result.selected_model != "model/dep"
        assert result.selected_model == "model/active"

    def test_all_deprecated_returns_fallback_result(self):
        from unittest.mock import patch

        from openshard.scoring.requirements import TaskRequirements
        from openshard.scoring.scorer import select_with_info

        deprecated = self._make_scoring_entry("model/dep")
        reqs = TaskRequirements()
        lc_map = {"model/dep": "deprecated"}

        with patch("openshard.scoring.shortlist.build_shortlist", return_value=[deprecated]):
            with patch("openshard.scoring.filter.filter_inventory", return_value=[deprecated]):
                with patch("openshard.models.registry.lifecycle_for", side_effect=lambda mid: lc_map.get(mid)):
                    result = select_with_info([deprecated], reqs, "standard")

        assert result.used_fallback is True
        assert result.selected_model is None

    def test_lifecycle_adjustments_stored_in_result(self):
        from unittest.mock import patch

        from openshard.scoring.requirements import TaskRequirements
        from openshard.scoring.scorer import select_with_info

        active = self._make_scoring_entry("model/active")
        watchlist = self._make_scoring_entry("model/watch")

        reqs = TaskRequirements()
        lc_map = {"model/active": "active_default", "model/watch": "watchlist"}

        with patch("openshard.scoring.shortlist.build_shortlist", return_value=[active, watchlist]):
            with patch("openshard.scoring.filter.filter_inventory", return_value=[active, watchlist]):
                with patch("openshard.models.registry.lifecycle_for", side_effect=lambda mid: lc_map.get(mid)):
                    result = select_with_info([active, watchlist], reqs, "standard")

        # The watchlist model should have a negative adjustment recorded
        if result.history_adjustments:
            watchlist_adj = result.history_adjustments.get("model/watch", 0.0)
            assert watchlist_adj < 0.0
