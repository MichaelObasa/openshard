from __future__ import annotations

import json
import unittest
from types import SimpleNamespace

from openshard.native.dispatch import (
    TierCandidate,
    _ROLE_DEFAULT_TIER,
    _ROLE_VALID_TIERS,
    _TIER_CANDIDATES,
    _TIER_MODEL_MAP,
    _UNKNOWN_TIER_FALLBACK,
    get_tier_candidate,
    resolve_role,
    resolve_tier,
)
from openshard.native.context import build_native_tier_dispatch_receipt
from openshard.routing.engine import MODEL_CHEAP, MODEL_MAIN, MODEL_STRONG


# ---------------------------------------------------------------------------
# TierCandidate structure
# ---------------------------------------------------------------------------

class TestTierCandidates(unittest.TestCase):
    """_TIER_CANDIDATES is the single source of truth for all 4 tiers."""

    def test_all_four_tiers_present(self):
        expected = {
            "frontier-reasoning-model",
            "balanced-coding-model",
            "low-cost-coding-model",
            "independent-validator-model",
        }
        self.assertEqual(set(_TIER_CANDIDATES), expected)

    def test_each_candidate_has_preferred(self):
        for tier, tc in _TIER_CANDIDATES.items():
            with self.subTest(tier=tier):
                self.assertIsInstance(tc.preferred, str)
                self.assertTrue(tc.preferred, f"{tier} has empty preferred")

    def test_each_candidate_has_fallbacks(self):
        for tier, tc in _TIER_CANDIDATES.items():
            with self.subTest(tier=tier):
                self.assertIsInstance(tc.fallbacks, list)
                self.assertTrue(tc.fallbacks, f"{tier} has no fallbacks")

    def test_each_candidate_has_provider(self):
        for tier, tc in _TIER_CANDIDATES.items():
            with self.subTest(tier=tier):
                self.assertTrue(tc.provider, f"{tier} has empty provider")

    def test_each_candidate_has_reason(self):
        for tier, tc in _TIER_CANDIDATES.items():
            with self.subTest(tier=tier):
                self.assertTrue(tc.reason, f"{tier} has empty reason")

    def test_frontier_preferred_is_model_strong(self):
        self.assertEqual(_TIER_CANDIDATES["frontier-reasoning-model"].preferred, MODEL_STRONG)

    def test_balanced_preferred_is_model_main(self):
        self.assertEqual(_TIER_CANDIDATES["balanced-coding-model"].preferred, MODEL_MAIN)

    def test_low_cost_preferred_is_model_cheap(self):
        self.assertEqual(_TIER_CANDIDATES["low-cost-coding-model"].preferred, MODEL_CHEAP)

    def test_validator_preferred_is_model_strong(self):
        self.assertEqual(_TIER_CANDIDATES["independent-validator-model"].preferred, MODEL_STRONG)

    def test_tier_model_map_derived_from_candidates(self):
        for tier, tc in _TIER_CANDIDATES.items():
            with self.subTest(tier=tier):
                self.assertEqual(_TIER_MODEL_MAP[tier], tc.preferred)

    def test_tier_model_map_has_same_keys(self):
        self.assertEqual(set(_TIER_MODEL_MAP), set(_TIER_CANDIDATES))

    def test_get_tier_candidate_known(self):
        tc = get_tier_candidate("frontier-reasoning-model")
        self.assertIsInstance(tc, TierCandidate)
        self.assertEqual(tc.preferred, MODEL_STRONG)

    def test_get_tier_candidate_unknown_returns_none(self):
        self.assertIsNone(get_tier_candidate("nonexistent-tier"))

    def test_get_tier_candidate_all_known_tiers(self):
        for tier in _TIER_CANDIDATES:
            with self.subTest(tier=tier):
                self.assertIsNotNone(get_tier_candidate(tier))


# ---------------------------------------------------------------------------
# resolve_tier — strengthened behavior
# ---------------------------------------------------------------------------

class TestResolveTierStrengthened(unittest.TestCase):

    def test_known_tiers_resolve_without_fallback(self):
        for tier, tc in _TIER_CANDIDATES.items():
            with self.subTest(tier=tier):
                model, fb, reason = resolve_tier(tier)
                self.assertEqual(model, tc.preferred)
                self.assertFalse(fb)
                self.assertEqual(reason, "")

    def test_unknown_tier_returns_safe_fallback_not_none(self):
        model, fb, reason = resolve_tier("totally-unknown-tier")
        self.assertIsNotNone(model, "unknown tier must not return None")
        self.assertTrue(fb)
        self.assertIn("totally-unknown-tier", reason)

    def test_unknown_tier_falls_back_to_balanced(self):
        fallback_model = _TIER_CANDIDATES[_UNKNOWN_TIER_FALLBACK].preferred
        model, fb, reason = resolve_tier("made-up-tier")
        self.assertEqual(model, fallback_model)

    def test_none_tier_returns_none_model(self):
        model, fb, reason = resolve_tier(None)
        self.assertIsNone(model)
        self.assertTrue(fb)

    def test_empty_tier_returns_none_model(self):
        model, fb, reason = resolve_tier("")
        self.assertIsNone(model)
        self.assertTrue(fb)

    def test_blocked_preferred_uses_first_fallback(self):
        blocked = {MODEL_STRONG}
        model, fb, reason = resolve_tier("frontier-reasoning-model", blocked_models=blocked)
        self.assertNotEqual(model, MODEL_STRONG)
        self.assertTrue(fb)
        self.assertIn("frontier-reasoning-model", reason)
        # Should fall to the declared fallback for frontier tier
        expected_fallback = _TIER_CANDIDATES["frontier-reasoning-model"].fallbacks[0]
        self.assertEqual(model, expected_fallback)

    def test_unblocked_preferred_ignores_blocked_set(self):
        blocked = {"some/other-model"}
        model, fb, reason = resolve_tier("balanced-coding-model", blocked_models=blocked)
        self.assertEqual(model, MODEL_MAIN)
        self.assertFalse(fb)

    def test_blocked_preferred_records_reason(self):
        blocked = {MODEL_STRONG}
        _model, fb, reason = resolve_tier("frontier-reasoning-model", blocked_models=blocked)
        self.assertTrue(fb)
        self.assertIn("blocked", reason)

    def test_all_models_blocked_returns_none_or_fallback(self):
        # Block everything possible — resolve_tier should not crash
        all_models = {MODEL_STRONG, MODEL_MAIN, MODEL_CHEAP}
        model, fb, reason = resolve_tier("frontier-reasoning-model", blocked_models=all_models)
        # Either None (all blocked) or a model, but must not raise
        self.assertTrue(fb)
        self.assertIn("blocked", reason)

    def test_empty_blocked_set_behaves_like_none(self):
        m1, fb1, _ = resolve_tier("balanced-coding-model", blocked_models=None)
        m2, fb2, _ = resolve_tier("balanced-coding-model", blocked_models=set())
        self.assertEqual(m1, m2)
        self.assertEqual(fb1, fb2)


# ---------------------------------------------------------------------------
# Role mapping constants
# ---------------------------------------------------------------------------

class TestRoleMapping(unittest.TestCase):

    def test_role_default_tier_has_three_roles(self):
        self.assertIn("planner", _ROLE_DEFAULT_TIER)
        self.assertIn("executor", _ROLE_DEFAULT_TIER)
        self.assertIn("validator", _ROLE_DEFAULT_TIER)

    def test_planner_default_is_frontier(self):
        self.assertEqual(_ROLE_DEFAULT_TIER["planner"], "frontier-reasoning-model")

    def test_executor_default_is_balanced(self):
        self.assertEqual(_ROLE_DEFAULT_TIER["executor"], "balanced-coding-model")

    def test_validator_default_is_independent(self):
        self.assertEqual(_ROLE_DEFAULT_TIER["validator"], "independent-validator-model")

    def test_valid_tiers_covers_all_roles(self):
        for role in _ROLE_DEFAULT_TIER:
            with self.subTest(role=role):
                self.assertIn(role, _ROLE_VALID_TIERS)

    def test_default_tier_is_within_valid_tiers(self):
        for role, default in _ROLE_DEFAULT_TIER.items():
            with self.subTest(role=role):
                self.assertIn(default, _ROLE_VALID_TIERS[role])

    def test_context_default_role_tiers_matches_dispatch(self):
        from openshard.native import context as ctx
        self.assertIs(ctx._DEFAULT_ROLE_TIERS, _ROLE_DEFAULT_TIER)

    def test_context_valid_tiers_by_role_matches_dispatch(self):
        from openshard.native import context as ctx
        self.assertIs(ctx._VALID_TIERS_BY_ROLE, _ROLE_VALID_TIERS)


# ---------------------------------------------------------------------------
# resolve_role
# ---------------------------------------------------------------------------

class TestResolveRole(unittest.TestCase):

    def test_planner_resolves_to_frontier_model(self):
        model, tier, fb, reason = resolve_role("planner")
        self.assertEqual(model, MODEL_STRONG)
        self.assertEqual(tier, "frontier-reasoning-model")
        self.assertFalse(fb)

    def test_executor_resolves_to_balanced_model(self):
        model, tier, fb, reason = resolve_role("executor")
        self.assertEqual(model, MODEL_MAIN)
        self.assertEqual(tier, "balanced-coding-model")
        self.assertFalse(fb)

    def test_validator_resolves_to_strong_model(self):
        model, tier, fb, reason = resolve_role("validator")
        self.assertEqual(model, MODEL_STRONG)
        self.assertEqual(tier, "independent-validator-model")
        self.assertFalse(fb)

    def test_unknown_role_returns_none_with_reason(self):
        model, tier, fb, reason = resolve_role("oracle")
        self.assertIsNone(model)
        self.assertEqual(tier, "")
        self.assertTrue(fb)
        self.assertIn("oracle", reason)

    def test_none_role_returns_none(self):
        model, tier, fb, reason = resolve_role(None)
        self.assertIsNone(model)
        self.assertTrue(fb)

    def test_empty_role_returns_none(self):
        model, tier, fb, reason = resolve_role("")
        self.assertIsNone(model)
        self.assertTrue(fb)

    def test_blocked_planner_default_falls_to_alt_tier(self):
        # Block MODEL_STRONG so frontier-reasoning-model can't use its preferred.
        blocked = {MODEL_STRONG}
        model, tier, fb, reason = resolve_role("planner", blocked_models=blocked)
        self.assertNotEqual(model, MODEL_STRONG)
        self.assertTrue(fb)
        # Should find an alt valid tier for planner (balanced)
        self.assertIsNotNone(model)

    def test_blocked_executor_default_falls_to_alt_tier(self):
        blocked = {MODEL_MAIN}
        model, tier, fb, reason = resolve_role("executor", blocked_models=blocked)
        self.assertNotEqual(model, MODEL_MAIN)
        self.assertTrue(fb)
        self.assertIsNotNone(model)

    def test_fallback_reason_recorded_when_blocked(self):
        blocked = {MODEL_STRONG}
        _model, _tier, fb, reason = resolve_role("planner", blocked_models=blocked)
        self.assertTrue(fb)
        self.assertTrue(reason, "fallback reason should be non-empty when fallback occurs")


# ---------------------------------------------------------------------------
# Receipt resolves models clearly
# ---------------------------------------------------------------------------

class TestReceiptShowsResolvedModels(unittest.TestCase):

    def _rr(self, planner_tier="", executor_tier="", validator_tier=""):
        return SimpleNamespace(
            planner_tier=planner_tier,
            executor_tier=executor_tier,
            validator_tier=validator_tier,
        )

    def test_receipt_shows_planner_model_from_frontier_tier(self):
        rr = self._rr(
            planner_tier="frontier-reasoning-model",
            executor_tier="balanced-coding-model",
            validator_tier="independent-validator-model",
        )
        r = build_native_tier_dispatch_receipt(
            experimental_tier_dispatch=True,
            routing_receipt=rr,
            applied=True,
        )
        self.assertEqual(r.planner_model, MODEL_STRONG)
        self.assertEqual(r.executor_model, MODEL_MAIN)
        self.assertEqual(r.validator_model, MODEL_STRONG)

    def test_receipt_fallback_reason_set_when_unknown_tier(self):
        rr = self._rr(planner_tier="no-such-tier", executor_tier="balanced-coding-model",
                      validator_tier="independent-validator-model")
        r = build_native_tier_dispatch_receipt(
            experimental_tier_dispatch=True,
            routing_receipt=rr,
            applied=True,
        )
        self.assertTrue(r.fallback_used)
        self.assertTrue(r.fallback_reason or r.warnings,
                        "fallback reason or warning should be recorded")

    def test_receipt_unknown_tier_does_not_crash(self):
        rr = self._rr(planner_tier="fantasy-tier", executor_tier="fantasy-tier",
                      validator_tier="fantasy-tier")
        # Should not raise
        r = build_native_tier_dispatch_receipt(
            experimental_tier_dispatch=True,
            routing_receipt=rr,
            applied=True,
        )
        self.assertTrue(r.enabled)


# ---------------------------------------------------------------------------
# Old history records render safely
# ---------------------------------------------------------------------------

class TestOldHistoryRenderSafe(unittest.TestCase):

    def setUp(self):
        from openshard.cli.run_output import _render_tier_dispatch_block
        self._render = _render_tier_dispatch_block

    def _old_tdr(self):
        return {
            "enabled": True,
            "applied": True,
            "tier_source": "category_fallback",
            "planner_tier": "frontier-reasoning-model",
            "planner_model": MODEL_STRONG,
            "executor_tier": "balanced-coding-model",
            "executor_model": MODEL_MAIN,
            "validator_tier": "independent-validator-model",
            "validator_model": MODEL_STRONG,
            "fallback_used": False,
            "fallback_reason": "",
            "warnings": [],
            # executor_model_actual, validator_dispatch_status absent (old format)
        }

    def test_old_dict_renders_at_more_without_crash(self):
        lines = self._render(self._old_tdr(), "more")
        self.assertTrue(len(lines) >= 1)

    def test_old_dict_renders_at_full_without_crash(self):
        lines = self._render(self._old_tdr(), "full")
        self.assertTrue(len(lines) >= 1)

    def test_old_dict_renders_at_default_without_crash(self):
        lines = self._render(self._old_tdr(), "default")
        # default returns empty — just confirm no crash
        self.assertIsInstance(lines, list)

    def test_old_dict_model_plan_visible_at_more(self):
        lines = self._render(self._old_tdr(), "more")
        joined = "\n".join(lines)
        self.assertIn("Model plan", joined)
        self.assertIn("Planning:", joined)

    def test_disabled_old_record_returns_empty(self):
        tdr = {"enabled": False}
        self.assertEqual(self._render(tdr, "more"), [])
        self.assertEqual(self._render(tdr, "full"), [])


# ---------------------------------------------------------------------------
# End-to-end: each known tier resolves to a model candidate
# ---------------------------------------------------------------------------

class TestEachTierResolvesToCandidate(unittest.TestCase):

    def test_all_known_tiers_resolve(self):
        for tier in _TIER_CANDIDATES:
            with self.subTest(tier=tier):
                model, fb, reason = resolve_tier(tier)
                self.assertIsNotNone(model, f"{tier} resolved to None")
                self.assertFalse(fb, f"{tier} unexpectedly used fallback")

    def test_all_roles_resolve(self):
        for role in _ROLE_DEFAULT_TIER:
            with self.subTest(role=role):
                model, tier, fb, reason = resolve_role(role)
                self.assertIsNotNone(model, f"role {role} resolved to None model")
                self.assertFalse(fb, f"role {role} unexpectedly used fallback")
                self.assertTrue(tier, f"role {role} returned empty tier")

    def test_tier_candidate_asdict_json_roundtrip(self):
        from dataclasses import asdict
        tc = _TIER_CANDIDATES["frontier-reasoning-model"]
        d = asdict(tc)
        raw = json.dumps(d)
        loaded = json.loads(raw)
        self.assertEqual(loaded["preferred"], MODEL_STRONG)
        self.assertIn(MODEL_MAIN, loaded["fallbacks"])
        self.assertEqual(loaded["provider"], "anthropic")


if __name__ == "__main__":
    unittest.main()
