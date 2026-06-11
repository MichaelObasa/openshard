"""Provider-aware routing eligibility v1 (known -> available -> routable).

Covers: Anthropic-only, OpenAI-only, multi-key, OpenRouter-wide, and no-key
environments; restricted/lifecycle exclusions; executor constraints;
unsupported direct-provider env vars; user blocklist; determinism; and the
Shard metadata shape. All env access goes through an injected mapping.
"""
from __future__ import annotations

from openshard.models.registry import all_models, models_by_lifecycle
from openshard.routing.provider_availability import (
    ELIGIBILITY_VERSION,
    EXECUTOR_CONSTRAINTS,
    RESTRICTED_MODEL_IDS,
    build_available_pool,
    build_routable_pool,
    detect_provider_availability,
    routing_constraints_metadata,
)

ANTHROPIC_ONLY = {"ANTHROPIC_API_KEY": "k"}
OPENAI_ONLY = {"OPENAI_API_KEY": "k"}
MULTI_DIRECT = {"ANTHROPIC_API_KEY": "k", "OPENAI_API_KEY": "k"}
OPENROUTER_ONLY = {"OPENROUTER_API_KEY": "k"}
ALL_KEYS = {"OPENROUTER_API_KEY": "k", "ANTHROPIC_API_KEY": "k", "OPENAI_API_KEY": "k"}


def _vendor(model_id: str) -> str:
    return model_id.lstrip("~").split("/", 1)[0]


# ---------------------------------------------------------------------------
# Provider detection
# ---------------------------------------------------------------------------

class TestProviderDetection:
    def test_anthropic_only(self):
        avail = detect_provider_availability(ANTHROPIC_ONLY)
        assert avail.detected == ("anthropic",)
        assert avail.anthropic and not avail.openrouter and not avail.openai

    def test_no_keys_detects_nothing_and_never_raises(self):
        avail = detect_provider_availability({})
        assert avail.detected == ()

    def test_priority_order_openrouter_first(self):
        avail = detect_provider_availability(ALL_KEYS)
        assert avail.detected == ("openrouter", "anthropic", "openai")

    def test_empty_string_key_does_not_count(self):
        avail = detect_provider_availability({"ANTHROPIC_API_KEY": ""})
        assert avail.detected == ()

    def test_unsupported_direct_provider_env_var_ignored(self):
        # No DeepSeek client exists yet: its key must not register a provider.
        avail = detect_provider_availability({"DEEPSEEK_API_KEY": "k"})
        assert avail.detected == ()


# ---------------------------------------------------------------------------
# Available pool (known -> available)
# ---------------------------------------------------------------------------

class TestAvailablePool:
    def test_anthropic_only_exactly_anthropic_models(self):
        pool = build_available_pool(detect_provider_availability(ANTHROPIC_ONLY))
        available_ids = {ma.entry.id for ma in pool if ma.available}
        expected = {
            e.id for e in all_models()
            if _vendor(e.id) == "anthropic" and e.id not in RESTRICTED_MODEL_IDS
        }
        assert available_ids == expected
        assert all(ma.via == ("anthropic",) for ma in pool if ma.available)

    def test_anthropic_only_includes_tilde_fallback_alias(self):
        pool = build_available_pool(detect_provider_availability(ANTHROPIC_ONLY))
        by_id = {ma.entry.id: ma for ma in pool}
        alias = next(i for i in by_id if i.startswith("~anthropic/"))
        assert by_id[alias].available

    def test_openai_only_exactly_openai_models(self):
        pool = build_available_pool(detect_provider_availability(OPENAI_ONLY))
        available_ids = {ma.entry.id for ma in pool if ma.available}
        assert available_ids == {
            e.id for e in all_models() if _vendor(e.id) == "openai"
        }

    def test_multi_direct_keys_union_no_third_party_vendors(self):
        pool = build_available_pool(detect_provider_availability(MULTI_DIRECT))
        for ma in pool:
            vendor = _vendor(ma.entry.id)
            if ma.entry.id in RESTRICTED_MODEL_IDS:
                assert not ma.available
            elif vendor in ("anthropic", "openai"):
                assert ma.available and ma.via == (vendor,)
            else:
                assert not ma.available and ma.reason == "no_api_key"

    def test_openrouter_wide_all_but_restricted(self):
        pool = build_available_pool(detect_provider_availability(OPENROUTER_ONLY))
        unavailable = {ma.entry.id: ma.reason for ma in pool if not ma.available}
        assert unavailable == {mid: "access_restricted" for mid in RESTRICTED_MODEL_IDS}
        assert all(
            "openrouter" in ma.via for ma in pool if ma.available
        )

    def test_all_keys_via_lists_every_route(self):
        pool = build_available_pool(detect_provider_availability(ALL_KEYS))
        by_id = {ma.entry.id: ma for ma in pool}
        sonnet = by_id["anthropic/claude-sonnet-4.6"]
        assert sonnet.via == ("openrouter", "anthropic")

    def test_no_keys_all_unavailable_with_reason(self):
        pool = build_available_pool(detect_provider_availability({}))
        assert len(pool) == len(all_models())
        assert all(not ma.available for ma in pool)
        assert all(
            ma.reason in ("no_api_key", "access_restricted") for ma in pool
        )


# ---------------------------------------------------------------------------
# Routable pool (available -> routable)
# ---------------------------------------------------------------------------

class TestRoutablePool:
    def test_openrouter_wide_routable_is_full_active_default(self):
        pool = build_routable_pool(detect_provider_availability(OPENROUTER_ONLY))
        assert [e.id for e in pool.routable] == sorted(
            e.id for e in models_by_lifecycle("active_default")
        )

    def test_anthropic_only_routable_is_anthropic_active_default(self):
        pool = build_routable_pool(detect_provider_availability(ANTHROPIC_ONLY))
        expected = sorted(
            e.id for e in models_by_lifecycle("active_default")
            if _vendor(e.id) == "anthropic"
        )
        assert [e.id for e in pool.routable] == expected
        assert expected  # the chain must keep valid anthropic candidates

    def test_openai_only_routable_is_openai_active_default(self):
        pool = build_routable_pool(detect_provider_availability(OPENAI_ONLY))
        assert [e.id for e in pool.routable] == sorted(
            e.id for e in models_by_lifecycle("active_default")
            if _vendor(e.id) == "openai"
        )

    def test_mythos5_never_routable_even_with_all_keys(self):
        pool = build_routable_pool(detect_provider_availability(ALL_KEYS))
        assert "anthropic/claude-mythos-5" not in {e.id for e in pool.routable}
        assert ("anthropic/claude-mythos-5", "access_restricted") in pool.excluded

    def test_non_default_lifecycles_never_routable(self):
        pool = build_routable_pool(detect_provider_availability(OPENROUTER_ONLY))
        excluded = dict(pool.excluded)
        for lifecycle in (
            "active_specialist", "fallback", "open_weight",
            "experimental", "watchlist", "deprecated",
        ):
            for entry in models_by_lifecycle(lifecycle):
                assert entry.id not in {e.id for e in pool.routable}
                assert excluded[entry.id] in (
                    f"lifecycle:{lifecycle}", "access_restricted"
                )

    def test_opencode_executor_requires_openrouter(self):
        pool = build_routable_pool(
            detect_provider_availability(ANTHROPIC_ONLY), executor="opencode"
        )
        assert pool.routable == ()
        assert any(r == "executor_constraint" for _i, r in pool.excluded)

    def test_opencode_executor_with_openrouter_unaffected(self):
        pool = build_routable_pool(
            detect_provider_availability(OPENROUTER_ONLY), executor="opencode"
        )
        assert len(pool.routable) == len(models_by_lifecycle("active_default"))

    def test_user_blocklist_excludes_with_reason(self):
        pool = build_routable_pool(
            detect_provider_availability(OPENROUTER_ONLY),
            blocked_model_ids=frozenset({"z-ai/glm-5.1"}),
        )
        assert "z-ai/glm-5.1" not in {e.id for e in pool.routable}
        assert ("z-ai/glm-5.1", "user_blocked") in pool.excluded

    def test_no_keys_routable_empty_no_exception(self):
        pool = build_routable_pool(detect_provider_availability({}))
        assert pool.routable == ()
        assert len(pool.excluded) == len(all_models())

    def test_deterministic_output(self):
        a = build_routable_pool(detect_provider_availability(ALL_KEYS))
        b = build_routable_pool(detect_provider_availability(ALL_KEYS))
        assert [e.id for e in a.routable] == [e.id for e in b.routable]
        assert a.excluded == b.excluded

    def test_known_executors_all_have_constraints(self):
        assert set(EXECUTOR_CONSTRAINTS) == {"native", "opencode", "direct", "staged"}


# ---------------------------------------------------------------------------
# Shard metadata shape
# ---------------------------------------------------------------------------

class TestRoutingConstraintsMetadata:
    def test_metadata_shape_and_counts(self):
        pool = build_routable_pool(
            detect_provider_availability(OPENROUTER_ONLY), executor="native"
        )
        meta = routing_constraints_metadata(pool)
        assert meta["executor"] == "native"
        assert meta["eligibility_version"] == ELIGIBILITY_VERSION
        assert meta["routable_pool_size"] == len(pool.routable)
        assert meta["excluded_counts"]["access_restricted"] == 1
        non_default = len(all_models()) - len(models_by_lifecycle("active_default")) - 1
        assert meta["excluded_counts"]["lifecycle"] == non_default


# ---------------------------------------------------------------------------
# Model policy integration smoke test
# ---------------------------------------------------------------------------

class TestPolicyIntegrationSmoke:
    def test_default_policy_same_as_no_policy(self):
        """Default ModelPolicyConfig must not change the routable pool."""
        from openshard.routing.model_policy import ModelPolicyConfig

        avail = detect_provider_availability(OPENROUTER_ONLY)
        without_policy = build_routable_pool(avail)
        with_policy = build_routable_pool(avail, policy=ModelPolicyConfig())
        assert set(m.id for m in with_policy.routable) == set(
            m.id for m in without_policy.routable
        )

    def test_metadata_never_contains_model_lists_or_keys(self):
        pool = build_routable_pool(detect_provider_availability(ALL_KEYS))
        meta = routing_constraints_metadata(pool)
        assert set(meta) == {
            "executor", "routable_pool_size", "excluded_counts",
            "eligibility_version",
        }

    def test_routing_truth_records_eligibility_fields(self):
        from openshard.history.routing_truth import build_routing_truth
        rt = build_routing_truth({
            "available_providers": ["openrouter"],
            "routing_constraints": {"routable_pool_size": 15},
        })
        assert rt.available_providers == ["openrouter"]
        assert rt.routing_constraints == {"routable_pool_size": 15}

    def test_routing_truth_legacy_defaults(self):
        from openshard.history.routing_truth import build_routing_truth
        rt = build_routing_truth({})
        assert rt.available_providers == []
        assert rt.routing_constraints is None
