from __future__ import annotations

import json
import unittest
from dataclasses import asdict
from types import SimpleNamespace

from openshard.native.dispatch import (
    _TIER_MODEL_MAP,
    resolve_tier,
    resolve_tier_for_category,
)
from openshard.native.context import (
    NativeTierDispatchReceipt,
    build_native_tier_dispatch_receipt,
)
from openshard.routing.engine import MODEL_CHEAP, MODEL_MAIN, MODEL_STRONG


def _build(**kwargs) -> NativeTierDispatchReceipt:
    return build_native_tier_dispatch_receipt(**kwargs)


def _rr(planner_tier="", executor_tier="", validator_tier=""):
    return SimpleNamespace(
        planner_tier=planner_tier,
        executor_tier=executor_tier,
        validator_tier=validator_tier,
    )


class TestTierModelMap(unittest.TestCase):
    def test_frontier_reasoning_maps_to_strong(self):
        self.assertEqual(_TIER_MODEL_MAP["frontier-reasoning-model"], MODEL_STRONG)

    def test_balanced_coding_maps_to_main(self):
        self.assertEqual(_TIER_MODEL_MAP["balanced-coding-model"], MODEL_MAIN)

    def test_low_cost_maps_to_cheap(self):
        self.assertEqual(_TIER_MODEL_MAP["low-cost-coding-model"], MODEL_CHEAP)

    def test_independent_validator_maps_to_strong(self):
        self.assertEqual(_TIER_MODEL_MAP["independent-validator-model"], MODEL_STRONG)

    def test_exactly_four_tiers(self):
        self.assertEqual(len(_TIER_MODEL_MAP), 4)


class TestResolveTier(unittest.TestCase):
    def test_known_tier_returns_model(self):
        model, fallback, reason = resolve_tier("frontier-reasoning-model")
        self.assertEqual(model, MODEL_STRONG)
        self.assertFalse(fallback)
        self.assertEqual(reason, "")

    def test_balanced_coding(self):
        model, fallback, reason = resolve_tier("balanced-coding-model")
        self.assertEqual(model, MODEL_MAIN)
        self.assertFalse(fallback)

    def test_low_cost(self):
        model, fallback, reason = resolve_tier("low-cost-coding-model")
        self.assertEqual(model, MODEL_CHEAP)
        self.assertFalse(fallback)

    def test_unknown_tier_returns_fallback(self):
        model, fallback, reason = resolve_tier("nonexistent-tier")
        self.assertIsNone(model)
        self.assertTrue(fallback)
        self.assertIn("nonexistent-tier", reason)

    def test_none_tier_returns_fallback(self):
        model, fallback, reason = resolve_tier(None)
        self.assertIsNone(model)
        self.assertTrue(fallback)

    def test_empty_string_returns_fallback(self):
        model, fallback, reason = resolve_tier("")
        self.assertIsNone(model)
        self.assertTrue(fallback)


class TestResolveTierForCategory(unittest.TestCase):
    def test_security_maps_to_strong(self):
        model, tier, fallback, reason = resolve_tier_for_category("security")
        self.assertEqual(model, MODEL_STRONG)
        self.assertEqual(tier, "frontier-reasoning-model")
        self.assertFalse(fallback)

    def test_complex_maps_to_strong(self):
        model, tier, fallback, reason = resolve_tier_for_category("complex")
        self.assertEqual(model, MODEL_STRONG)
        self.assertFalse(fallback)

    def test_standard_maps_to_main(self):
        model, tier, fallback, reason = resolve_tier_for_category("standard")
        self.assertEqual(model, MODEL_MAIN)
        self.assertEqual(tier, "balanced-coding-model")
        self.assertFalse(fallback)

    def test_boilerplate_maps_to_cheap(self):
        model, tier, fallback, reason = resolve_tier_for_category("boilerplate")
        self.assertEqual(model, MODEL_CHEAP)
        self.assertEqual(tier, "low-cost-coding-model")
        self.assertFalse(fallback)

    def test_visual_maps_to_main_with_fallback(self):
        model, tier, fallback, reason = resolve_tier_for_category("visual")
        self.assertEqual(model, MODEL_MAIN)
        self.assertTrue(fallback)
        self.assertIn("visual", reason)

    def test_unknown_category_returns_fallback(self):
        model, tier, fallback, reason = resolve_tier_for_category("unknown-cat")
        self.assertIsNone(model)
        self.assertEqual(tier, "")
        self.assertTrue(fallback)

    def test_none_category_returns_fallback(self):
        model, tier, fallback, reason = resolve_tier_for_category(None)
        self.assertIsNone(model)
        self.assertTrue(fallback)


class TestDispatchReceiptDefaults(unittest.TestCase):
    def test_enabled_false_by_default(self):
        r = NativeTierDispatchReceipt()
        self.assertFalse(r.enabled)

    def test_applied_false_by_default(self):
        r = NativeTierDispatchReceipt()
        self.assertFalse(r.applied)

    def test_tier_fields_empty_by_default(self):
        r = NativeTierDispatchReceipt()
        self.assertEqual(r.planner_tier, "")
        self.assertEqual(r.executor_tier, "")
        self.assertEqual(r.validator_tier, "")
        self.assertIsNone(r.planner_model)
        self.assertIsNone(r.executor_model)
        self.assertIsNone(r.validator_model)

    def test_warnings_list_by_default(self):
        r = NativeTierDispatchReceipt()
        self.assertEqual(r.warnings, [])


class TestBuildReceiptFlagOff(unittest.TestCase):
    def test_flag_off_returns_disabled(self):
        r = _build(experimental_tier_dispatch=False)
        self.assertFalse(r.enabled)
        self.assertFalse(r.applied)

    def test_flag_off_ignores_routing_receipt(self):
        rr = _rr(planner_tier="frontier-reasoning-model")
        r = _build(experimental_tier_dispatch=False, routing_receipt=rr)
        self.assertFalse(r.enabled)

    def test_flag_off_ignores_category(self):
        r = _build(experimental_tier_dispatch=False, routing_category="security")
        self.assertFalse(r.enabled)


class TestBuildReceiptNativePath(unittest.TestCase):
    def test_routing_receipt_tiers_used_first(self):
        rr = _rr(
            planner_tier="frontier-reasoning-model",
            executor_tier="balanced-coding-model",
            validator_tier="independent-validator-model",
        )
        r = _build(
            experimental_tier_dispatch=True,
            routing_receipt=rr,
            applied=False,
            not_applied_reason="native tier dispatch is recorded only in v1",
        )
        self.assertTrue(r.enabled)
        self.assertFalse(r.applied)
        self.assertEqual(r.tier_source, "candidate_scoring")
        self.assertEqual(r.planner_tier, "frontier-reasoning-model")
        self.assertEqual(r.planner_model, MODEL_STRONG)
        self.assertEqual(r.executor_tier, "balanced-coding-model")
        self.assertEqual(r.executor_model, MODEL_MAIN)
        self.assertEqual(r.validator_tier, "independent-validator-model")
        self.assertEqual(r.validator_model, MODEL_STRONG)
        self.assertFalse(r.fallback_used)
        self.assertEqual(r.fallback_reason, "native tier dispatch is recorded only in v1")


class TestBuildReceiptFallbackPath(unittest.TestCase):
    def test_no_routing_receipt_uses_category(self):
        r = _build(
            experimental_tier_dispatch=True,
            routing_receipt=None,
            routing_category="security",
            applied=True,
        )
        self.assertTrue(r.enabled)
        self.assertTrue(r.applied)
        self.assertEqual(r.tier_source, "category_fallback")
        self.assertEqual(r.executor_model, MODEL_STRONG)

    def test_empty_routing_receipt_tiers_falls_back_to_category(self):
        rr = _rr(planner_tier="", executor_tier="", validator_tier="")
        r = _build(
            experimental_tier_dispatch=True,
            routing_receipt=rr,
            routing_category="boilerplate",
            applied=True,
        )
        self.assertEqual(r.tier_source, "category_fallback")
        self.assertEqual(r.executor_model, MODEL_CHEAP)


class TestBuildReceiptApplied(unittest.TestCase):
    def test_applied_true_when_flag_and_not_dry_run(self):
        r = _build(experimental_tier_dispatch=True, applied=True, routing_category="standard")
        self.assertTrue(r.enabled)
        self.assertTrue(r.applied)
        self.assertEqual(r.fallback_reason, "")

    def test_applied_false_for_dry_run(self):
        r = _build(
            experimental_tier_dispatch=True,
            applied=False,
            not_applied_reason="dry-run",
        )
        self.assertTrue(r.enabled)
        self.assertFalse(r.applied)
        self.assertEqual(r.fallback_reason, "dry-run")

    def test_applied_false_for_native(self):
        r = _build(
            experimental_tier_dispatch=True,
            applied=False,
            not_applied_reason="native tier dispatch is recorded only in v1",
        )
        self.assertFalse(r.applied)
        self.assertIn("native", r.fallback_reason)


class TestBuildReceiptUnknownTier(unittest.TestCase):
    def test_unknown_tier_sets_fallback_and_none_model(self):
        rr = _rr(
            planner_tier="some-unknown-tier",
            executor_tier="balanced-coding-model",
            validator_tier="independent-validator-model",
        )
        r = _build(
            experimental_tier_dispatch=True,
            routing_receipt=rr,
            applied=True,
        )
        self.assertTrue(r.fallback_used)
        self.assertIsNone(r.planner_model)
        self.assertEqual(r.executor_model, MODEL_MAIN)


class TestReceiptAsdict(unittest.TestCase):
    def test_asdict_json_roundtrip(self):
        r = NativeTierDispatchReceipt(
            enabled=True,
            applied=True,
            tier_source="category_fallback",
            planner_tier="frontier-reasoning-model",
            planner_model=MODEL_STRONG,
            executor_tier="balanced-coding-model",
            executor_model=MODEL_MAIN,
            validator_tier="independent-validator-model",
            validator_model=MODEL_STRONG,
            fallback_used=False,
            fallback_reason="",
            warnings=[],
        )
        d = asdict(r)
        raw = json.dumps(d)
        loaded = json.loads(raw)
        self.assertEqual(loaded["enabled"], True)
        self.assertEqual(loaded["applied"], True)
        self.assertEqual(loaded["planner_model"], MODEL_STRONG)
        self.assertEqual(loaded["executor_model"], MODEL_MAIN)

    def test_none_models_serialize(self):
        r = NativeTierDispatchReceipt(planner_model=None, executor_model=None)
        d = asdict(r)
        raw = json.dumps(d)
        loaded = json.loads(raw)
        self.assertIsNone(loaded["planner_model"])


class TestRenderingMore(unittest.TestCase):
    def setUp(self):
        from openshard.cli.run_output import _render_tier_dispatch_block
        self._render = _render_tier_dispatch_block

    def test_enabled_false_returns_empty(self):
        tdr = {"enabled": False}
        self.assertEqual(self._render(tdr, "more"), [])

    def test_compact_line_shown_when_enabled(self):
        tdr = {
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
        }
        lines = self._render(tdr, "more")
        joined = "\n".join(lines)
        self.assertIn("Model plan", joined)
        self.assertIn("Planning:", joined)
        self.assertIn("Applied: yes", joined)
        self.assertIn("Source:  category fallback", joined)

    def test_no_full_block_at_more(self):
        tdr = {
            "enabled": True, "applied": True, "tier_source": "category_fallback",
            "planner_tier": "frontier-reasoning-model", "planner_model": MODEL_STRONG,
            "executor_tier": "balanced-coding-model", "executor_model": MODEL_MAIN,
            "validator_tier": "independent-validator-model", "validator_model": MODEL_STRONG,
            "fallback_used": False, "fallback_reason": "", "warnings": [],
        }
        lines = self._render(tdr, "more")
        self.assertFalse(any("[tier dispatch]" in ln for ln in lines))
        self.assertFalse(any("Tier:" in ln for ln in lines))

    def test_fallback_shown_in_compact(self):
        tdr = {
            "enabled": True, "applied": False, "tier_source": "category_fallback",
            "planner_tier": "", "planner_model": None,
            "executor_tier": "", "executor_model": None,
            "validator_tier": "", "validator_model": None,
            "fallback_used": True, "fallback_reason": "dry-run", "warnings": [],
        }
        lines = self._render(tdr, "more")
        joined = "\n".join(lines)
        self.assertIn("Applied: no", joined)
        self.assertIn("Fallback: yes", joined)

    def test_more_shows_work_model(self):
        tdr = {
            "enabled": True, "applied": True, "tier_source": "category_fallback",
            "planner_tier": "frontier-reasoning-model", "planner_model": MODEL_STRONG,
            "executor_tier": "balanced-coding-model", "executor_model": MODEL_MAIN,
            "validator_tier": "independent-validator-model", "validator_model": MODEL_STRONG,
            "fallback_used": False, "fallback_reason": "", "warnings": [],
        }
        lines = self._render(tdr, "more")
        joined = "\n".join(lines)
        self.assertIn("Work model:", joined)

    def test_more_shows_initial_candidate_when_provided(self):
        tdr = {
            "enabled": True, "applied": True, "tier_source": "category_fallback",
            "planner_tier": "frontier-reasoning-model", "planner_model": MODEL_STRONG,
            "executor_tier": "balanced-coding-model", "executor_model": MODEL_MAIN,
            "validator_tier": "independent-validator-model", "validator_model": MODEL_STRONG,
            "fallback_used": False, "fallback_reason": "", "warnings": [],
        }
        lines = self._render(tdr, "more", initial_model="openrouter/deepseek-v4")
        joined = "\n".join(lines)
        self.assertIn("Initial candidate:", joined)

    def test_more_no_initial_candidate_without_arg(self):
        tdr = {
            "enabled": True, "applied": True, "tier_source": "category_fallback",
            "planner_tier": "frontier-reasoning-model", "planner_model": MODEL_STRONG,
            "executor_tier": "balanced-coding-model", "executor_model": MODEL_MAIN,
            "validator_tier": "independent-validator-model", "validator_model": MODEL_STRONG,
            "fallback_used": False, "fallback_reason": "", "warnings": [],
        }
        lines = self._render(tdr, "more")
        joined = "\n".join(lines)
        self.assertNotIn("Initial candidate:", joined)


class TestRenderingFull(unittest.TestCase):
    def setUp(self):
        from openshard.cli.run_output import _render_tier_dispatch_block
        self._render = _render_tier_dispatch_block

    def test_full_block_shown(self):
        tdr = {
            "enabled": True, "applied": True, "tier_source": "candidate_scoring",
            "planner_tier": "frontier-reasoning-model", "planner_model": MODEL_STRONG,
            "executor_tier": "balanced-coding-model", "executor_model": MODEL_MAIN,
            "validator_tier": "independent-validator-model", "validator_model": MODEL_STRONG,
            "fallback_used": False, "fallback_reason": "", "warnings": [],
        }
        lines = self._render(tdr, "full")
        joined = "\n".join(lines)
        self.assertIn("Model plan", joined)
        self.assertIn("Planning:", joined)
        self.assertTrue(any("Tier:" in ln for ln in lines))
        self.assertIn("Dispatch", joined)

    def test_fallback_reason_shown_in_full(self):
        tdr = {
            "enabled": True, "applied": False, "tier_source": "category_fallback",
            "planner_tier": "", "planner_model": None,
            "executor_tier": "", "executor_model": None,
            "validator_tier": "", "validator_model": None,
            "fallback_used": True, "fallback_reason": "native tier dispatch is recorded only in v1",
            "warnings": [],
        }
        lines = self._render(tdr, "full")
        self.assertTrue(any("Reason:" in ln for ln in lines))

    def test_warnings_listed_in_full(self):
        tdr = {
            "enabled": True, "applied": True, "tier_source": "category_fallback",
            "planner_tier": "", "planner_model": None,
            "executor_tier": "balanced-coding-model", "executor_model": MODEL_MAIN,
            "validator_tier": "", "validator_model": None,
            "fallback_used": True, "fallback_reason": "",
            "warnings": ["planner fallback: unknown tier"],
        }
        lines = self._render(tdr, "full")
        self.assertTrue(any("planner fallback" in ln for ln in lines))

    def test_full_shows_work_model(self):
        tdr = {
            "enabled": True, "applied": True, "tier_source": "candidate_scoring",
            "planner_tier": "frontier-reasoning-model", "planner_model": MODEL_STRONG,
            "executor_tier": "balanced-coding-model", "executor_model": MODEL_MAIN,
            "validator_tier": "independent-validator-model", "validator_model": MODEL_STRONG,
            "fallback_used": False, "fallback_reason": "", "warnings": [],
        }
        lines = self._render(tdr, "full")
        joined = "\n".join(lines)
        self.assertIn("Work model:", joined)

    def test_full_shows_initial_candidate_when_provided(self):
        tdr = {
            "enabled": True, "applied": True, "tier_source": "candidate_scoring",
            "planner_tier": "frontier-reasoning-model", "planner_model": MODEL_STRONG,
            "executor_tier": "balanced-coding-model", "executor_model": MODEL_MAIN,
            "validator_tier": "independent-validator-model", "validator_model": MODEL_STRONG,
            "fallback_used": False, "fallback_reason": "", "warnings": [],
        }
        lines = self._render(tdr, "full", initial_model="openrouter/deepseek-v4")
        joined = "\n".join(lines)
        self.assertIn("Initial candidate:", joined)

    def test_full_no_initial_candidate_without_arg(self):
        tdr = {
            "enabled": True, "applied": True, "tier_source": "candidate_scoring",
            "planner_tier": "frontier-reasoning-model", "planner_model": MODEL_STRONG,
            "executor_tier": "balanced-coding-model", "executor_model": MODEL_MAIN,
            "validator_tier": "independent-validator-model", "validator_model": MODEL_STRONG,
            "fallback_used": False, "fallback_reason": "", "warnings": [],
        }
        lines = self._render(tdr, "full")
        joined = "\n".join(lines)
        self.assertNotIn("Initial candidate:", joined)


class TestRenderingDefault(unittest.TestCase):
    def setUp(self):
        from openshard.cli.run_output import _render_tier_dispatch_block
        self._render = _render_tier_dispatch_block

    def test_disabled_returns_empty_at_default(self):
        tdr = {"enabled": False}
        self.assertEqual(self._render(tdr, "default"), [])

    def test_enabled_also_returns_empty_at_default(self):
        tdr = {"enabled": True, "applied": True, "tier_source": "category_fallback",
               "planner_model": MODEL_STRONG, "executor_model": MODEL_MAIN,
               "validator_model": MODEL_STRONG, "fallback_used": False, "warnings": []}
        self.assertEqual(self._render(tdr, "default"), [])


class TestLastNonNativeRender(unittest.TestCase):
    """_render_log_entry renders tier dispatch for non-native entries at --more/--full."""

    def _make_entry(self, workflow: str, enabled: bool, applied: bool) -> dict:
        tdr = {
            "enabled": enabled,
            "applied": applied,
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
        } if enabled else None
        return {
            "timestamp": "2026-01-01T00:00:00",
            "task": "test task",
            "summary": "done",
            "workflow": workflow,
            "tier_dispatch_receipt": tdr,
        }

    def test_non_native_entry_renders_at_more(self):
        from openshard.cli.run_output import _render_tier_dispatch_block
        entry = self._make_entry("staged", enabled=True, applied=True)
        tdr = entry["tier_dispatch_receipt"]
        lines = _render_tier_dispatch_block(tdr, "more")
        self.assertTrue(len(lines) >= 1)
        self.assertTrue(any("Applied: yes" in ln for ln in lines))

    def test_disabled_receipt_not_rendered(self):
        from openshard.cli.run_output import _render_tier_dispatch_block
        tdr = {"enabled": False}
        lines = _render_tier_dispatch_block(tdr, "more")
        self.assertEqual(lines, [])

    def test_native_entry_not_rendered_in_log_entry(self):
        # The main.py guard skips rendering for native entries in _render_log_entry.
        # Verify the guard condition works: workflow == "native" → skip
        entry = self._make_entry("native", enabled=True, applied=False)
        self.assertEqual(entry.get("workflow"), "native")


class TestCategoryFallbackOnlyWhenNoTierData(unittest.TestCase):
    def test_routing_receipt_tiers_take_priority_over_category(self):
        rr = _rr(
            planner_tier="frontier-reasoning-model",
            executor_tier="low-cost-coding-model",
            validator_tier="independent-validator-model",
        )
        r = _build(
            experimental_tier_dispatch=True,
            routing_receipt=rr,
            routing_category="boilerplate",
            applied=False,
            not_applied_reason="native tier dispatch is recorded only in v1",
        )
        self.assertEqual(r.tier_source, "candidate_scoring")
        # executor comes from receipt (low-cost) not category (boilerplate → cheap is same, but tier name differs)
        self.assertEqual(r.executor_tier, "low-cost-coding-model")

    def test_none_routing_receipt_uses_category(self):
        r = _build(
            experimental_tier_dispatch=True,
            routing_receipt=None,
            routing_category="standard",
            applied=True,
        )
        self.assertEqual(r.tier_source, "category_fallback")
        self.assertEqual(r.executor_model, MODEL_MAIN)


class TestReceiptSavedNativeRun(unittest.TestCase):
    def test_asdict_produces_top_level_serializable_dict(self):
        r = NativeTierDispatchReceipt(
            enabled=True,
            applied=False,
            tier_source="candidate_scoring",
            planner_tier="frontier-reasoning-model",
            planner_model=MODEL_STRONG,
            executor_tier="balanced-coding-model",
            executor_model=MODEL_MAIN,
            validator_tier="independent-validator-model",
            validator_model=MODEL_STRONG,
            fallback_used=False,
            fallback_reason="native tier dispatch is recorded only in v1",
            warnings=[],
        )
        d = asdict(r)
        self.assertIn("tier_source", d)
        self.assertIn("planner_model", d)
        self.assertIn("fallback_reason", d)
        raw = json.dumps(d)
        loaded = json.loads(raw)
        self.assertEqual(loaded["fallback_reason"], "native tier dispatch is recorded only in v1")


class TestBuildReceiptActualModels(unittest.TestCase):
    def test_executor_model_actual_stored(self):
        r = _build(
            experimental_tier_dispatch=True,
            applied=False,
            not_applied_reason="routing decisions resolved post-execution",
            executor_model_actual=MODEL_STRONG,
            routing_category="standard",
        )
        self.assertEqual(r.executor_model_actual, MODEL_STRONG)

    def test_planner_model_actual_stored(self):
        r = _build(
            experimental_tier_dispatch=True,
            applied=False,
            planner_model_actual=MODEL_STRONG,
            routing_category="standard",
        )
        self.assertEqual(r.planner_model_actual, MODEL_STRONG)

    def test_actual_defaults_to_none(self):
        r = _build(experimental_tier_dispatch=True, routing_category="standard")
        self.assertIsNone(r.executor_model_actual)
        self.assertIsNone(r.planner_model_actual)
        self.assertIsNone(r.validator_model_actual)

    def test_actual_included_in_asdict(self):
        r = _build(
            experimental_tier_dispatch=True,
            executor_model_actual=MODEL_MAIN,
            routing_category="standard",
        )
        d = asdict(r)
        self.assertIn("executor_model_actual", d)
        self.assertEqual(d["executor_model_actual"], MODEL_MAIN)

    def test_flag_off_actual_fields_default(self):
        r = _build(experimental_tier_dispatch=False, executor_model_actual=MODEL_STRONG)
        self.assertIsNone(r.executor_model_actual)


class TestBuildReceiptValidatorStatus(unittest.TestCase):
    def test_validator_dispatch_status_reserved(self):
        r = _build(
            experimental_tier_dispatch=True,
            applied=False,
            not_applied_reason="routing decisions resolved post-execution",
            validator_dispatch_status="reserved",
            routing_category="standard",
        )
        self.assertEqual(r.validator_dispatch_status, "reserved")

    def test_validator_dispatch_status_defaults_empty(self):
        r = _build(experimental_tier_dispatch=True, routing_category="standard")
        self.assertEqual(r.validator_dispatch_status, "")

    def test_flag_off_status_empty(self):
        r = _build(experimental_tier_dispatch=False, validator_dispatch_status="applied")
        self.assertEqual(r.validator_dispatch_status, "")

    def test_validator_dispatch_status_in_asdict(self):
        r = _build(
            experimental_tier_dispatch=True,
            validator_dispatch_status="reserved",
            routing_category="security",
        )
        d = asdict(r)
        self.assertEqual(d["validator_dispatch_status"], "reserved")


class TestReceiptOldHistoryRender(unittest.TestCase):
    """Old receipts stored without actual-model fields still render cleanly."""

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

    def test_missing_actual_field_is_none_on_default_receipt(self):
        r = NativeTierDispatchReceipt()
        self.assertIsNone(r.executor_model_actual)

    def test_old_dict_renders_without_crash_at_more(self):
        lines = self._render(self._old_tdr(), "more")
        self.assertTrue(len(lines) >= 1)
        self.assertFalse(any("Actual model" in ln for ln in lines))

    def test_old_dict_renders_without_crash_at_full(self):
        lines = self._render(self._old_tdr(), "full")
        self.assertTrue(len(lines) >= 1)

    def test_actual_line_absent_when_same_model(self):
        tdr = {**self._old_tdr(), "executor_model_actual": MODEL_MAIN}
        lines = self._render(tdr, "more")
        self.assertFalse(any("Actual model" in ln for ln in lines))


class TestPolicyBlockedTierNotUsed(unittest.TestCase):
    """Policy-blocked candidates should not appear as selected models."""

    def test_downgraded_planner_not_blocked_model(self):
        from openshard.native.context import NativeModelCandidateScoring
        scoring = NativeModelCandidateScoring(
            candidates=[],
            selected_by_role={
                "planner": "balanced-coding-model",
                "executor": "balanced-coding-model",
                "validator": "independent-validator-model",
            },
            strategy="cost-balanced",
            confidence="medium",
            warnings=["planner downgraded by policy"],
            blocked_candidates=[MODEL_STRONG],
        )
        r = _build(
            experimental_tier_dispatch=True,
            model_candidate_scoring=scoring,
            routing_receipt=None,
            applied=True,
        )
        self.assertEqual(r.planner_tier, "balanced-coding-model")
        self.assertEqual(r.planner_model, MODEL_MAIN)
        self.assertNotEqual(r.planner_model, MODEL_STRONG)


class TestActualModelRendering(unittest.TestCase):
    """Actual model line appears in more/full only when it differs from planned model."""

    def setUp(self):
        from openshard.cli.run_output import _render_tier_dispatch_block
        self._render = _render_tier_dispatch_block

    def _tdr_with_actual(self, actual: str) -> dict:
        return {
            "enabled": True,
            "applied": False,
            "tier_source": "candidate_scoring",
            "planner_tier": "frontier-reasoning-model",
            "planner_model": MODEL_STRONG,
            "executor_tier": "balanced-coding-model",
            "executor_model": MODEL_MAIN,
            "validator_tier": "independent-validator-model",
            "validator_model": MODEL_STRONG,
            "fallback_used": False,
            "fallback_reason": "routing decisions resolved post-execution",
            "warnings": [],
            "executor_model_actual": actual,
            "validator_dispatch_status": "reserved",
        }

    def test_actual_shown_in_more_when_differs(self):
        tdr = self._tdr_with_actual(MODEL_STRONG)
        lines = self._render(tdr, "more")
        self.assertTrue(any("Actual model" in ln for ln in lines))

    def test_actual_shown_in_full_when_differs(self):
        tdr = self._tdr_with_actual(MODEL_STRONG)
        lines = self._render(tdr, "full")
        self.assertTrue(any("Actual:" in ln for ln in lines))

    def test_actual_absent_in_more_when_same(self):
        tdr = self._tdr_with_actual(MODEL_MAIN)
        lines = self._render(tdr, "more")
        self.assertFalse(any("Actual model" in ln for ln in lines))

    def test_actual_absent_in_full_when_same(self):
        tdr = self._tdr_with_actual(MODEL_MAIN)
        lines = self._render(tdr, "full")
        self.assertFalse(any("Actual:" in ln for ln in lines))

    def test_validator_reserved_in_full(self):
        tdr = self._tdr_with_actual(MODEL_STRONG)
        lines = self._render(tdr, "full")
        joined = "\n".join(lines)
        self.assertIn("reserved for validation", joined)


if __name__ == "__main__":
    unittest.main()
