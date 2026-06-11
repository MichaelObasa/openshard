"""Provider-aware dispatch enforcement v1.

Tests that resolve_routing_model_for_context() only selects models the user
can actually call given their API keys and executor, and that the result is
recorded correctly in RoutingTruth.
"""
from __future__ import annotations

from openshard.models.registry import models_by_lifecycle
from openshard.routing.model_resolver import (
    resolve_routing_model,
    resolve_routing_model_for_context,
)
from openshard.routing.provider_availability import (
    build_routable_pool,
    detect_provider_availability,
)

ANTHROPIC_ONLY = {"ANTHROPIC_API_KEY": "k"}
OPENAI_ONLY = {"OPENAI_API_KEY": "k"}
OPENROUTER_ONLY = {"OPENROUTER_API_KEY": "k"}
ALL_KEYS = {"OPENROUTER_API_KEY": "k", "ANTHROPIC_API_KEY": "k", "OPENAI_API_KEY": "k"}
NO_KEYS: dict[str, str] = {}


def _pool(env: dict, executor: str | None = None):
    return build_routable_pool(detect_provider_availability(env), executor=executor)


def _vendor(model_id: str) -> str:
    return model_id.lstrip("~").split("/", 1)[0]


# ---------------------------------------------------------------------------
# Core enforcement: provider-constrained selection
# ---------------------------------------------------------------------------

class TestAnthropicOnlyEnforcement:
    def test_main_role_selects_anthropic_model(self):
        result = resolve_routing_model_for_context("main", _pool(ANTHROPIC_ONLY))
        assert result.model is not None
        assert _vendor(result.model) == "anthropic", result.model

    def test_strong_role_selects_anthropic_model(self):
        result = resolve_routing_model_for_context("strong", _pool(ANTHROPIC_ONLY))
        assert result.model is not None
        assert _vendor(result.model) == "anthropic", result.model

    def test_cheap_role_selects_anthropic_model(self):
        result = resolve_routing_model_for_context("cheap", _pool(ANTHROPIC_ONLY))
        assert result.model is not None
        assert _vendor(result.model) == "anthropic", result.model

    def test_visual_specialist_role_returns_pool_model(self):
        # visual role is active_specialist; pool is active_default only.
        # Enforcement falls back to any pool model rather than failing.
        result = resolve_routing_model_for_context("visual", _pool(ANTHROPIC_ONLY))
        assert result.model is not None
        assert _vendor(result.model) == "anthropic", result.model

    def test_enforcement_applied_flag_set_when_model_differs(self):
        unconstrained = resolve_routing_model("main")
        result = resolve_routing_model_for_context("main", _pool(ANTHROPIC_ONLY))
        if result.model != unconstrained:
            assert result.enforcement_applied is True
            assert result.rejected_model == unconstrained


class TestOpenAIOnlyEnforcement:
    def test_main_role_selects_openai_model(self):
        result = resolve_routing_model_for_context("main", _pool(OPENAI_ONLY))
        assert result.model is not None
        assert _vendor(result.model) == "openai", result.model

    def test_strong_role_selects_openai_model(self):
        result = resolve_routing_model_for_context("strong", _pool(OPENAI_ONLY))
        assert result.model is not None
        assert _vendor(result.model) == "openai", result.model


class TestOpenRouterBroadRouting:
    def test_openrouter_model_matches_unconstrained_for_main(self):
        # OpenRouter pool covers all active_default models, so the context-aware
        # resolver should select the same model as the unconstrained resolver.
        result = resolve_routing_model_for_context("main", _pool(OPENROUTER_ONLY))
        assert result.model == resolve_routing_model("main")

    def test_openrouter_model_matches_unconstrained_for_strong(self):
        result = resolve_routing_model_for_context("strong", _pool(OPENROUTER_ONLY))
        assert result.model == resolve_routing_model("strong")

    def test_openrouter_model_matches_unconstrained_for_cheap(self):
        result = resolve_routing_model_for_context("cheap", _pool(OPENROUTER_ONLY))
        assert result.model == resolve_routing_model("cheap")

    def test_openrouter_enforcement_not_applied(self):
        result = resolve_routing_model_for_context("main", _pool(OPENROUTER_ONLY))
        assert result.enforcement_applied is False
        assert result.rejected_model is None


class TestMultiKeyRouting:
    def test_multi_key_pool_contains_both_vendors(self):
        pool = _pool({"ANTHROPIC_API_KEY": "k", "OPENAI_API_KEY": "k"})
        vendors = {_vendor(m.id) for m in pool.routable}
        assert "anthropic" in vendors
        assert "openai" in vendors

    def test_multi_key_resolves_a_model(self):
        result = resolve_routing_model_for_context(
            "main", _pool({"ANTHROPIC_API_KEY": "k", "OPENAI_API_KEY": "k"})
        )
        assert result.model is not None


class TestNoKeyEnforcement:
    def test_no_key_returns_no_eligible_model(self):
        result = resolve_routing_model_for_context("main", _pool(NO_KEYS))
        assert result.model is None
        assert result.source == "no_eligible_model"

    def test_no_key_routable_pool_size_is_zero(self):
        result = resolve_routing_model_for_context("main", _pool(NO_KEYS))
        assert result.routable_pool_size == 0

    def test_no_key_enforcement_applied_true(self):
        result = resolve_routing_model_for_context("main", _pool(NO_KEYS))
        assert result.enforcement_applied is True


class TestOpenCodeExecutorConstraint:
    def test_opencode_without_openrouter_returns_no_eligible_model(self):
        # opencode executor requires openrouter; without it the pool is empty.
        result = resolve_routing_model_for_context(
            "main", _pool(ANTHROPIC_ONLY, executor="opencode")
        )
        assert result.model is None
        assert result.source == "no_eligible_model"

    def test_opencode_with_openrouter_resolves_model(self):
        result = resolve_routing_model_for_context(
            "main", _pool(OPENROUTER_ONLY, executor="opencode")
        )
        assert result.model is not None


# ---------------------------------------------------------------------------
# Specific model exclusion requirements
# ---------------------------------------------------------------------------

class TestMythosNeverSelected:
    def test_mythos_not_in_routable_pool(self):
        # claude-mythos-5 is access_restricted; must never appear in any pool.
        for env in (ANTHROPIC_ONLY, OPENROUTER_ONLY, ALL_KEYS):
            pool = _pool(env)
            ids = {m.id for m in pool.routable}
            assert "anthropic/claude-mythos-5" not in ids, f"mythos in pool for {env}"

    def test_mythos_not_selected_by_any_role(self):
        for role in ("cheap", "main", "strong", "visual", "complex", "escalate"):
            for env in (ANTHROPIC_ONLY, OPENROUTER_ONLY, ALL_KEYS):
                result = resolve_routing_model_for_context(role, _pool(env))
                assert result.model != "anthropic/claude-mythos-5"


class TestFableNotDefaultRoutable:
    def test_fable_not_in_active_default_lifecycle(self):
        active_default_ids = {e.id for e in models_by_lifecycle("active_default")}
        fable_ids = {e.id for e in models_by_lifecycle("active_specialist")
                     if "fable" in e.id}
        # Fable must be in specialist, never in default.
        assert not fable_ids.issubset(active_default_ids)

    def test_fable_not_in_routable_pool(self):
        for env in (ANTHROPIC_ONLY, OPENROUTER_ONLY, ALL_KEYS):
            pool = _pool(env)
            fable_in_pool = [m.id for m in pool.routable if "fable" in m.id]
            assert fable_in_pool == [], f"fable found in pool for {env}: {fable_in_pool}"

    def test_fable_not_selected_as_main_role(self):
        for env in (ANTHROPIC_ONLY, OPENROUTER_ONLY, ALL_KEYS):
            result = resolve_routing_model_for_context("main", _pool(env))
            assert result.model is None or "fable" not in result.model


class TestUserBlockedModelNotSelected:
    def test_blocked_model_excluded_from_pool(self):
        pool = build_routable_pool(
            detect_provider_availability(OPENROUTER_ONLY),
            blocked_model_ids=frozenset({"anthropic/claude-sonnet-4.6"}),
        )
        ids = {m.id for m in pool.routable}
        assert "anthropic/claude-sonnet-4.6" not in ids

    def test_blocked_model_not_selected(self):
        strong_model = resolve_routing_model("strong")
        pool = build_routable_pool(
            detect_provider_availability(OPENROUTER_ONLY),
            blocked_model_ids=frozenset({strong_model}),
        )
        result = resolve_routing_model_for_context("strong", pool)
        assert result.model != strong_model


# ---------------------------------------------------------------------------
# Backward compatibility: original resolver unchanged without context
# ---------------------------------------------------------------------------

class TestExistingResolverUnchanged:
    def test_resolve_routing_model_unchanged(self):
        # The original @cache resolver must continue to return the same value
        # regardless of any provider context being used elsewhere.
        from openshard.routing.model_resolver import MODEL_MAIN, resolve_routing_model
        assert resolve_routing_model("main") == MODEL_MAIN

    def test_no_provider_context_uses_unconstrained_model(self):
        # Without a pool, the original resolver picks freely across all vendors.
        model = resolve_routing_model("main")
        assert isinstance(model, str) and "/" in model


# ---------------------------------------------------------------------------
# ProviderAwareResolution structure
# ---------------------------------------------------------------------------

class TestResolutionStructure:
    def test_selected_model_matches_model(self):
        result = resolve_routing_model_for_context("main", _pool(ANTHROPIC_ONLY))
        assert result.selected_model == result.model

    def test_source_is_routable_pool_when_model_found(self):
        result = resolve_routing_model_for_context("main", _pool(ANTHROPIC_ONLY))
        assert result.source == "routable_pool"

    def test_rejected_model_none_when_not_enforced(self):
        result = resolve_routing_model_for_context("main", _pool(OPENROUTER_ONLY))
        assert result.rejected_model is None

    def test_routable_pool_size_positive(self):
        result = resolve_routing_model_for_context("main", _pool(ANTHROPIC_ONLY))
        assert result.routable_pool_size > 0


# ---------------------------------------------------------------------------
# RoutingTruth records enforcement output
# ---------------------------------------------------------------------------

class TestShardRoutingTruthRecordsEnforcement:
    def test_enforcement_fields_populated_when_present(self):
        from openshard.history.routing_truth import build_routing_truth

        entry = {
            "execution_model": "anthropic/claude-sonnet-4.6",
            "provider_enforcement": {
                "applied": True,
                "source": "routable_pool",
                "selected_model": "anthropic/claude-sonnet-4.6",
                "rejected_model": "z-ai/glm-5.1",
                "routable_pool_size": 5,
            },
        }
        truth = build_routing_truth(entry)
        assert truth.provider_enforcement_applied is True
        assert truth.provider_enforcement_source == "routable_pool"
        assert truth.provider_enforcement_selected_model == "anthropic/claude-sonnet-4.6"
        assert truth.provider_enforcement_rejected_model == "z-ai/glm-5.1"
        assert truth.provider_enforcement_routable_size == 5

    def test_enforcement_fields_safe_defaults_when_absent(self):
        from openshard.history.routing_truth import build_routing_truth

        truth = build_routing_truth({"execution_model": "anthropic/claude-sonnet-4.6"})
        assert truth.provider_enforcement_applied is False
        assert truth.provider_enforcement_source is None
        assert truth.provider_enforcement_selected_model is None
        assert truth.provider_enforcement_rejected_model is None
        assert truth.provider_enforcement_routable_size == 0


# ---------------------------------------------------------------------------
# Policy integration: blocked model is never selected by resolver
# ---------------------------------------------------------------------------

class TestPolicyWithEnforcement:
    def test_blocked_model_never_selected(self):
        """A model in blocked_models should never come out of resolve_routing_model_for_context."""
        from openshard.routing.model_policy import ModelPolicyConfig

        avail = detect_provider_availability(OPENROUTER_ONLY)
        # Find what the resolver would select for 'cheap' without policy.
        base_pool = build_routable_pool(avail)
        base_result = resolve_routing_model_for_context("cheap", base_pool)
        if base_result.model is None:
            return  # nothing to block

        blocked_id = base_result.model
        policy = ModelPolicyConfig(blocked_models=frozenset({blocked_id}))
        policy_pool = build_routable_pool(avail, policy=policy)

        result = resolve_routing_model_for_context("cheap", policy_pool)
        assert result.model != blocked_id, (
            f"{blocked_id} should be blocked by policy but was selected"
        )
