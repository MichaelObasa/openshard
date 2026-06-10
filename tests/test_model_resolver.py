"""Tests for openshard.routing.model_resolver.

Validates that resolve_routing_model correctly queries the registry,
falls back to hardcoded constants when the registry yields nothing, and
that module-level constants are valid non-empty strings.
"""
from __future__ import annotations

from unittest.mock import patch

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _clear_cache():
    """Clear lru_cache on resolver functions between tests."""
    from openshard.routing import model_resolver
    model_resolver.resolve_routing_model.cache_clear()
    model_resolver.resolution_source.cache_clear()


def _make_entry(model_id: str, lifecycle: str, cost_class: str = "mid", tier: str = "mid", roles=()):
    """Return a minimal ModelEntry-like object."""
    from openshard.models.registry import ModelEntry
    return ModelEntry(
        id=model_id,
        display_name=model_id,
        provider="test",
        tier=tier,
        cost_class=cost_class,
        lifecycle=lifecycle,
        roles=roles,
    )


# ---------------------------------------------------------------------------
# Module-level constant sanity checks
# ---------------------------------------------------------------------------

class TestModuleLevelConstants:
    def test_all_constants_are_non_empty_strings(self):
        from openshard.routing.model_resolver import (
            MODEL_CHEAP,
            MODEL_COMPLEX,
            MODEL_ESCALATE,
            MODEL_MAIN,
            MODEL_STRONG,
            MODEL_VISUAL,
        )
        for name, val in [
            ("MODEL_CHEAP", MODEL_CHEAP),
            ("MODEL_MAIN", MODEL_MAIN),
            ("MODEL_STRONG", MODEL_STRONG),
            ("MODEL_ESCALATE", MODEL_ESCALATE),
            ("MODEL_VISUAL", MODEL_VISUAL),
            ("MODEL_COMPLEX", MODEL_COMPLEX),
        ]:
            assert isinstance(val, str) and val, f"{name} must be a non-empty string"

    def test_escalation_chain_has_two_entries(self):
        from openshard.routing.model_resolver import ESCALATION_CHAIN
        assert len(ESCALATION_CHAIN) == 2

    def test_escalation_chain_entries_are_non_empty_strings(self):
        from openshard.routing.model_resolver import ESCALATION_CHAIN
        for item in ESCALATION_CHAIN:
            assert isinstance(item, str) and item

    def test_escalation_chain_strong_before_escalate(self):
        from openshard.routing.model_resolver import ESCALATION_CHAIN, MODEL_ESCALATE, MODEL_STRONG
        assert ESCALATION_CHAIN[0] == MODEL_STRONG
        assert ESCALATION_CHAIN[1] == MODEL_ESCALATE


# ---------------------------------------------------------------------------
# Registry lookup behaviour
# ---------------------------------------------------------------------------

class TestResolveRoutingModel:
    def setup_method(self):
        _clear_cache()

    def teardown_method(self):
        _clear_cache()

    def test_returns_registry_entry_when_eligible_candidate_exists(self):
        cheap_entry = _make_entry(
            "test/cheap-model", lifecycle="active_default",
            cost_class="cheap", tier="cheap", roles=("cheap_control",),
        )
        from openshard.routing import model_resolver as _mr
        with patch("openshard.models.registry.models_by_lifecycle", return_value=[cheap_entry]):
            _mr.resolve_routing_model.cache_clear()
            result = _mr.resolve_routing_model("cheap")
        assert result == "test/cheap-model"

    def test_fallback_when_no_eligible_registry_entry(self):
        from openshard.routing import model_resolver as _mr
        with patch("openshard.models.registry.models_by_lifecycle", return_value=[]):
            _mr.resolve_routing_model.cache_clear()
            result = _mr.resolve_routing_model("cheap")
        assert result == _mr._FALLBACKS["cheap"]

    def test_fallback_when_registry_import_fails(self):
        import sys

        from openshard.routing import model_resolver as _mr
        _mr.resolve_routing_model.cache_clear()
        with patch.dict(sys.modules, {"openshard.models.registry": None}):
            result = _mr.resolve_routing_model("main")
        assert result == _mr._FALLBACKS["main"]

    def test_unknown_role_returns_main_fallback(self):
        from openshard.routing import model_resolver as _mr
        with patch("openshard.models.registry.models_by_lifecycle", return_value=[]):
            _mr.resolve_routing_model.cache_clear()
            result = _mr.resolve_routing_model("nonexistent_role")
        assert result == _mr._FALLBACKS["main"]

    def test_deterministic_with_multiple_candidates(self):
        entries = [
            _make_entry("b/model", lifecycle="active_default", cost_class="cheap", tier="cheap"),
            _make_entry("a/model", lifecycle="active_default", cost_class="cheap", tier="cheap"),
        ]
        from openshard.routing import model_resolver as _mr
        with patch("openshard.models.registry.models_by_lifecycle", return_value=entries):
            _mr.resolve_routing_model.cache_clear()
            result1 = _mr.resolve_routing_model("cheap")
            _mr.resolve_routing_model.cache_clear()
            with patch("openshard.models.registry.models_by_lifecycle", return_value=entries):
                result2 = _mr.resolve_routing_model("cheap")
        assert result1 == result2

    def test_roles_hint_tiebreaker_prefers_matching_role(self):
        # Two equally valid cheap models — one has the hint role, one doesn't
        with_hint = _make_entry(
            "z/with-hint", lifecycle="active_default",
            cost_class="cheap", tier="cheap", roles=("cheap_control",),
        )
        without_hint = _make_entry(
            "a/without-hint", lifecycle="active_default",
            cost_class="cheap", tier="cheap", roles=("summariser",),
        )
        from openshard.routing import model_resolver as _mr
        with patch("openshard.models.registry.models_by_lifecycle", return_value=[without_hint, with_hint]):
            _mr.resolve_routing_model.cache_clear()
            result = _mr.resolve_routing_model("cheap")
        assert result == "z/with-hint"  # hint match wins over alphabetical

    def test_deprecated_model_never_returned(self):
        deprecated = _make_entry(
            "x/deprecated", lifecycle="deprecated",
            cost_class="cheap", tier="cheap",
        )
        active = _make_entry(
            "y/active", lifecycle="active_default",
            cost_class="cheap", tier="cheap",
        )
        from openshard.routing import model_resolver as _mr

        def _side_effect(lifecycle):
            if lifecycle == "active_default":
                return [active]
            if lifecycle == "deprecated":
                return [deprecated]
            return []

        with patch("openshard.models.registry.models_by_lifecycle", side_effect=_side_effect):
            _mr.resolve_routing_model.cache_clear()
            result = _mr.resolve_routing_model("cheap")
        assert result != "x/deprecated"

    def test_visual_searches_active_specialist(self):
        # kimi-k2.5 is active_specialist — visual role should still find it
        visual_entry = _make_entry(
            "moonshotai/kimi-k2.5", lifecycle="active_specialist",
            cost_class="mid", tier="mid", roles=("visual", "multimodal"),
        )
        from openshard.routing import model_resolver as _mr

        def _side_effect(lifecycle):
            if lifecycle == "active_specialist":
                return [visual_entry]
            return []

        with patch("openshard.models.registry.models_by_lifecycle", side_effect=_side_effect):
            _mr.resolve_routing_model.cache_clear()
            result = _mr.resolve_routing_model("visual")
        assert result == "moonshotai/kimi-k2.5"

    def test_escalate_searches_active_specialist(self):
        # Escalation models are active_specialist by design
        esc_entry = _make_entry(
            "anthropic/claude-opus-4.8", lifecycle="active_specialist",
            cost_class="expensive", tier="frontier", roles=("escalation",),
        )
        from openshard.routing import model_resolver as _mr

        def _side_effect(lifecycle):
            if lifecycle == "active_specialist":
                return [esc_entry]
            return []

        with patch("openshard.models.registry.models_by_lifecycle", side_effect=_side_effect):
            _mr.resolve_routing_model.cache_clear()
            result = _mr.resolve_routing_model("escalate")
        assert result == "anthropic/claude-opus-4.8"


# ---------------------------------------------------------------------------
# resolution_source
# ---------------------------------------------------------------------------

class TestResolutionSource:
    def setup_method(self):
        _clear_cache()

    def teardown_method(self):
        _clear_cache()

    def test_registry_source_when_different_from_fallback(self):
        from openshard.routing import model_resolver as _mr
        non_fallback = _make_entry(
            "registry/model", lifecycle="active_default",
            cost_class="cheap", tier="cheap",
        )
        with patch("openshard.models.registry.models_by_lifecycle", return_value=[non_fallback]):
            _mr.resolve_routing_model.cache_clear()
            _mr.resolution_source.cache_clear()
            src = _mr.resolution_source("cheap")
        assert src == "registry"

    def test_hardcoded_source_when_matches_fallback(self):
        from openshard.routing import model_resolver as _mr
        # Return exactly the fallback model from registry — source should still be "hardcoded"
        # because the resolved value matches the fallback string
        fallback_entry = _make_entry(
            _mr._FALLBACKS["cheap"], lifecycle="active_default",
            cost_class="cheap", tier="cheap",
        )
        with patch("openshard.models.registry.models_by_lifecycle", return_value=[fallback_entry]):
            _mr.resolve_routing_model.cache_clear()
            _mr.resolution_source.cache_clear()
            src = _mr.resolution_source("cheap")
        # fallback value == registry value → "hardcoded" is correct label
        assert src == "hardcoded"

    def test_hardcoded_source_when_no_registry_match(self):
        from openshard.routing import model_resolver as _mr
        with patch("openshard.models.registry.models_by_lifecycle", return_value=[]):
            _mr.resolve_routing_model.cache_clear()
            _mr.resolution_source.cache_clear()
            src = _mr.resolution_source("strong")
        assert src == "hardcoded"


# ---------------------------------------------------------------------------
# engine.py imports resolver constants (smoke test)
# ---------------------------------------------------------------------------

class TestEngineImportsResolver:
    def test_engine_model_constants_come_from_resolver(self):
        from openshard.routing import engine
        from openshard.routing.model_resolver import (
            ESCALATION_CHAIN,
            MODEL_CHEAP,
            MODEL_COMPLEX,
            MODEL_ESCALATE,
            MODEL_MAIN,
            MODEL_STRONG,
            MODEL_VISUAL,
        )
        assert engine.MODEL_CHEAP == MODEL_CHEAP
        assert engine.MODEL_MAIN == MODEL_MAIN
        assert engine.MODEL_STRONG == MODEL_STRONG
        assert engine.MODEL_ESCALATE == MODEL_ESCALATE
        assert engine.MODEL_VISUAL == MODEL_VISUAL
        assert engine.MODEL_COMPLEX == MODEL_COMPLEX
        assert engine.ESCALATION_CHAIN == ESCALATION_CHAIN
