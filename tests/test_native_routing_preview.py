from __future__ import annotations

import json
import unittest
from dataclasses import asdict
from types import SimpleNamespace

import click
from click.testing import CliRunner

from openshard.cli.run_output import _native_meta_from_entry, _render_native_demo_block
from openshard.native.context import (
    NativeModelCandidateScoring,
    NativeModelPolicyReceipt,
    NativeModelRoleDecision,
    NativeModelSelectionDecision,
    NativeRoutingPreview,
    NativeRunTrustScore,
    build_native_routing_preview,
)


def _ns(**kwargs: object) -> SimpleNamespace:
    return SimpleNamespace(**kwargs)


def _build(**kwargs: object) -> NativeRoutingPreview:
    return build_native_routing_preview(**kwargs)


def _render_entry(entry: dict, detail: str = "more") -> str:
    from openshard.cli.main import _render_log_entry

    @click.command()
    def cmd():
        _render_log_entry(entry, detail)

    return CliRunner().invoke(cmd).output


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

class TestNativeRoutingPreviewDefaults(unittest.TestCase):

    def setUp(self) -> None:
        self.preview = NativeRoutingPreview()

    def test_strategy_default(self):
        self.assertEqual(self.preview.strategy, "cost-balanced")

    def test_policy_mode_default(self):
        self.assertEqual(self.preview.policy_mode, "auto")

    def test_planner_tier_default(self):
        self.assertEqual(self.preview.planner_tier, "unknown")

    def test_executor_tier_default(self):
        self.assertEqual(self.preview.executor_tier, "unknown")

    def test_validator_tier_default(self):
        self.assertEqual(self.preview.validator_tier, "unknown")

    def test_risk_level_default(self):
        self.assertEqual(self.preview.risk_level, "unknown")

    def test_confidence_default(self):
        self.assertEqual(self.preview.confidence, "medium")

    def test_blocked_count_default(self):
        self.assertEqual(self.preview.blocked_count, 0)

    def test_policy_affected_default(self):
        self.assertFalse(self.preview.policy_affected)

    def test_trust_level_default(self):
        self.assertEqual(self.preview.trust_level, "unknown")

    def test_summary_default(self):
        self.assertEqual(self.preview.summary, "")

    def test_warnings_default(self):
        self.assertEqual(self.preview.warnings, [])


# ---------------------------------------------------------------------------
# Build – all None inputs
# ---------------------------------------------------------------------------

class TestBuildNativeRoutingPreviewAllNone(unittest.TestCase):

    def setUp(self) -> None:
        self.preview = _build()

    def test_no_crash(self):
        self.assertIsInstance(self.preview, NativeRoutingPreview)

    def test_strategy_fallback(self):
        self.assertEqual(self.preview.strategy, "cost-balanced")

    def test_policy_mode_fallback(self):
        self.assertEqual(self.preview.policy_mode, "auto")

    def test_tiers_are_unknown(self):
        self.assertEqual(self.preview.planner_tier, "unknown")
        self.assertEqual(self.preview.executor_tier, "unknown")
        self.assertEqual(self.preview.validator_tier, "unknown")

    def test_blocked_zero(self):
        self.assertEqual(self.preview.blocked_count, 0)

    def test_policy_changed_false(self):
        self.assertFalse(self.preview.policy_affected)

    def test_trust_unknown(self):
        self.assertEqual(self.preview.trust_level, "unknown")


# ---------------------------------------------------------------------------
# Build – tier resolution from selected_by_role
# ---------------------------------------------------------------------------

class TestBuildNativeRoutingPreviewTierResolution(unittest.TestCase):

    def test_tiers_from_selected_by_role(self):
        mcs = NativeModelCandidateScoring(
            selected_by_role={"planner": "frontier", "executor": "fast", "validator": "low-cost"},
            strategy="frontier-heavy",
            confidence="high",
        )
        preview = _build(model_candidate_scoring=mcs)
        self.assertEqual(preview.planner_tier, "frontier")
        self.assertEqual(preview.executor_tier, "fast")
        self.assertEqual(preview.validator_tier, "low-cost")

    def test_tiers_fallback_to_selection_decision(self):
        msd = NativeModelSelectionDecision(
            roles=[
                NativeModelRoleDecision(role="planner", model_tier="fast"),
                NativeModelRoleDecision(role="executor", model_tier="frontier"),
            ]
        )
        preview = _build(model_selection_decision=msd)
        self.assertEqual(preview.planner_tier, "fast")
        self.assertEqual(preview.executor_tier, "frontier")
        self.assertEqual(preview.validator_tier, "unknown")

    def test_scoring_takes_precedence_over_decision(self):
        mcs = NativeModelCandidateScoring(
            selected_by_role={"planner": "frontier"},
        )
        msd = NativeModelSelectionDecision(
            roles=[NativeModelRoleDecision(role="planner", model_tier="fast")]
        )
        preview = _build(model_candidate_scoring=mcs, model_selection_decision=msd)
        self.assertEqual(preview.planner_tier, "frontier")

    def test_scoring_dict_input(self):
        mcs = {"selected_by_role": {"planner": "fast", "executor": "frontier", "validator": "fast"}, "strategy": "cost-balanced", "confidence": "medium"}
        preview = _build(model_candidate_scoring=mcs)
        self.assertEqual(preview.planner_tier, "fast")
        self.assertEqual(preview.executor_tier, "frontier")


# ---------------------------------------------------------------------------
# Build – strategy and confidence resolution
# ---------------------------------------------------------------------------

class TestBuildNativeRoutingPreviewStrategyAndConfidence(unittest.TestCase):

    def test_strategy_from_scoring(self):
        mcs = NativeModelCandidateScoring(strategy="frontier-heavy", confidence="high")
        preview = _build(model_candidate_scoring=mcs)
        self.assertEqual(preview.strategy, "frontier-heavy")

    def test_strategy_fallback_to_decision(self):
        msd = NativeModelSelectionDecision(strategy="cheapest-safe", confidence="low")
        preview = _build(model_selection_decision=msd)
        self.assertEqual(preview.strategy, "cheapest-safe")

    def test_confidence_from_scoring(self):
        mcs = NativeModelCandidateScoring(confidence="high")
        preview = _build(model_candidate_scoring=mcs)
        self.assertEqual(preview.confidence, "high")

    def test_risk_level_from_decision(self):
        msd = NativeModelSelectionDecision(risk_level="high")
        preview = _build(model_selection_decision=msd)
        self.assertEqual(preview.risk_level, "high")


# ---------------------------------------------------------------------------
# Build – policy receipt fields
# ---------------------------------------------------------------------------

class TestBuildNativeRoutingPreviewPolicyFields(unittest.TestCase):

    def test_policy_mode_from_receipt(self):
        mpr = NativeModelPolicyReceipt(mode="open-source-only")
        preview = _build(model_policy_receipt=mpr)
        self.assertEqual(preview.policy_mode, "open-source-only")

    def test_policy_affected_true(self):
        mpr = NativeModelPolicyReceipt(affected_selection=True)
        preview = _build(model_policy_receipt=mpr)
        self.assertTrue(preview.policy_affected)

    def test_blocked_count_from_receipt(self):
        mpr = NativeModelPolicyReceipt(blocked_count=3)
        preview = _build(model_policy_receipt=mpr)
        self.assertEqual(preview.blocked_count, 3)

    def test_receipt_dict_input(self):
        mpr = {"mode": "cheapest-safe", "affected_selection": True, "blocked_count": 1}
        preview = _build(model_policy_receipt=mpr)
        self.assertEqual(preview.policy_mode, "cheapest-safe")
        self.assertTrue(preview.policy_affected)
        self.assertEqual(preview.blocked_count, 1)


# ---------------------------------------------------------------------------
# Build – trust level
# ---------------------------------------------------------------------------

class TestBuildNativeRoutingPreviewTrustLevel(unittest.TestCase):

    def test_trust_level_from_run_trust_score(self):
        rts = NativeRunTrustScore(level="strong")
        preview = _build(run_trust_score=rts)
        self.assertEqual(preview.trust_level, "strong")

    def test_trust_level_dict_input(self):
        rts = {"level": "fair", "score": 50, "factors": [], "warnings": [], "blockers": []}
        preview = _build(run_trust_score=rts)
        self.assertEqual(preview.trust_level, "fair")


# ---------------------------------------------------------------------------
# Build – summary string
# ---------------------------------------------------------------------------

class TestBuildNativeRoutingPreviewSummary(unittest.TestCase):

    def test_summary_contains_strategy(self):
        mcs = NativeModelCandidateScoring(strategy="frontier-heavy", selected_by_role={"planner": "frontier", "executor": "frontier", "validator": "fast"})
        preview = _build(model_candidate_scoring=mcs)
        self.assertIn("frontier-heavy", preview.summary)

    def test_summary_contains_tier_names(self):
        mcs = NativeModelCandidateScoring(selected_by_role={"planner": "frontier", "executor": "fast", "validator": "low-cost"})
        preview = _build(model_candidate_scoring=mcs)
        self.assertIn("frontier", preview.summary)
        self.assertIn("fast", preview.summary)

    def test_summary_contains_policy_mode(self):
        mpr = NativeModelPolicyReceipt(mode="open-source-only")
        preview = _build(model_policy_receipt=mpr)
        self.assertIn("open-source-only", preview.summary)


# ---------------------------------------------------------------------------
# Build – warnings
# ---------------------------------------------------------------------------

class TestBuildNativeRoutingPreviewWarnings(unittest.TestCase):

    def test_warnings_from_candidate_scoring(self):
        mcs = NativeModelCandidateScoring(warnings=["low confidence due to task ambiguity"])
        preview = _build(model_candidate_scoring=mcs)
        self.assertEqual(len(preview.warnings), 1)

    def test_no_warnings_when_none(self):
        preview = _build()
        self.assertEqual(preview.warnings, [])


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------

class TestNativeRoutingPreviewAsdict(unittest.TestCase):

    def test_asdict_is_json_serializable(self):
        mcs = NativeModelCandidateScoring(
            selected_by_role={"planner": "frontier", "executor": "fast", "validator": "fast"},
            strategy="frontier-heavy",
            confidence="high",
            warnings=["w1"],
        )
        mpr = NativeModelPolicyReceipt(mode="auto", affected_selection=False, blocked_count=0)
        preview = _build(model_candidate_scoring=mcs, model_policy_receipt=mpr)
        try:
            json.dumps(asdict(preview))
        except (TypeError, ValueError) as exc:
            self.fail(f"asdict(preview) is not JSON-serializable: {exc}")

    def test_asdict_round_trips_fields(self):
        preview = NativeRoutingPreview(
            strategy="cheapest-safe",
            policy_mode="cheapest-safe",
            planner_tier="fast",
            executor_tier="fast",
            validator_tier="fast",
            risk_level="low",
            confidence="high",
            blocked_count=2,
            policy_affected=True,
            trust_level="good",
            summary="cheapest-safe | planner=fast executor=fast validator=fast | policy=cheapest-safe",
            warnings=["w"],
        )
        d = asdict(preview)
        self.assertEqual(d["strategy"], "cheapest-safe")
        self.assertEqual(d["blocked_count"], 2)
        self.assertTrue(d["policy_affected"])
        self.assertEqual(d["warnings"], ["w"])


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

class TestNativeRoutingPreviewRendering(unittest.TestCase):

    def _meta_with_preview(self, **preview_kwargs):
        preview = NativeRoutingPreview(
            strategy="frontier-heavy",
            policy_mode="auto",
            planner_tier="frontier",
            executor_tier="frontier",
            validator_tier="fast",
            risk_level="high",
            confidence="high",
            blocked_count=0,
            policy_affected=False,
            trust_level="good",
            summary="frontier-heavy | planner=frontier executor=frontier validator=fast | policy=auto",
            warnings=["internal warning"],
            **preview_kwargs,
        )
        return _ns(
            routing_preview=preview,
            repo_context_summary=None, observation=None, plan=None,
            write_path="pipeline", verification_loop=None,
            verification_command_summary=None, diff_review=None,
            final_report=None, native_loop_steps=[], native_loop_trace=_ns(events=[]),
            native_backend="builtin", native_backend_available=True,
            native_backend_notes=[], native_backend_proof=None,
            read_search_findings=[], patch_proposal=None,
            command_policy_preview=None, context_packet=None,
            file_context=None, context_quality_score=None,
            context_quality_advisory=None, change_budget=None,
            change_budget_preview=None, change_budget_soft_gate=None,
            approval_request=None, approval_receipt=None,
            verification_plan=None, clarification_request=None,
            context_usage_summary=None, failure_memory=None,
            osn_loop=None, deepagents_adapter=None,
            validation_contract=None, context_provenance=None,
            run_trust_score=None, model_selection_decision=None,
            model_candidate_scoring=None, model_policy=None,
            model_policy_receipt=None,
        )

    def test_routing_preview_appears_in_more(self):
        meta = self._meta_with_preview()
        lines = _render_native_demo_block(meta, detail="more")
        combined = "\n".join(lines)
        self.assertIn("routing preview:", combined)
        self.assertIn("frontier-heavy", combined)

    def test_routing_preview_appears_in_full(self):
        meta = self._meta_with_preview()
        lines = _render_native_demo_block(meta, detail="full")
        combined = "\n".join(lines)
        self.assertIn("[routing preview]", combined)
        self.assertIn("strategy:", combined)
        self.assertIn("policy_mode:", combined)
        self.assertIn("planner:", combined)
        self.assertIn("executor:", combined)
        self.assertIn("validator:", combined)

    def test_routing_preview_hidden_in_default(self):
        meta = self._meta_with_preview()
        lines = _render_native_demo_block(meta, detail="default")
        combined = "\n".join(lines)
        self.assertNotIn("routing preview", combined)

    def test_warnings_count_not_raw_text_in_more(self):
        meta = self._meta_with_preview()
        lines = _render_native_demo_block(meta, detail="more")
        combined = "\n".join(lines)
        self.assertNotIn("internal warning", combined)

    def test_warnings_count_not_raw_text_in_full(self):
        meta = self._meta_with_preview()
        lines = _render_native_demo_block(meta, detail="full")
        combined = "\n".join(lines)
        self.assertNotIn("internal warning", combined)
        self.assertIn("warnings:", combined)

    def test_full_shows_warnings_count(self):
        meta = self._meta_with_preview()
        lines = _render_native_demo_block(meta, detail="full")
        combined = "\n".join(lines)
        self.assertIn("warnings:                1", combined)


# ---------------------------------------------------------------------------
# _native_meta_from_entry round-trip
# ---------------------------------------------------------------------------

class TestNativeMetaFromEntryRoutingPreview(unittest.TestCase):

    def _entry_with_preview(self) -> dict:
        return {
            "workflow": "native",
            "executor": "native",
            "routing_preview": {
                "strategy": "cheapest-safe",
                "policy_mode": "cheapest-safe",
                "planner_tier": "fast",
                "executor_tier": "fast",
                "validator_tier": "fast",
                "risk_level": "low",
                "confidence": "high",
                "blocked_count": 1,
                "policy_affected": True,
                "trust_level": "good",
                "summary": "cheapest-safe | planner=fast executor=fast validator=fast | policy=cheapest-safe",
                "warnings": [],
            },
        }

    def test_round_trips_routing_preview(self):
        entry = self._entry_with_preview()
        meta = _native_meta_from_entry(entry)
        self.assertIsNotNone(meta)
        rp = getattr(meta, "routing_preview", None)
        self.assertIsNotNone(rp)
        strategy = rp.get("strategy") if isinstance(rp, dict) else getattr(rp, "strategy", None)
        self.assertEqual(strategy, "cheapest-safe")

    def test_old_entry_without_routing_preview_does_not_crash(self):
        entry = {
            "workflow": "native",
            "executor": "native",
        }
        meta = _native_meta_from_entry(entry)
        self.assertIsNotNone(meta)
        rp = getattr(meta, "routing_preview", None)
        self.assertIsNone(rp)

    def test_old_entry_renders_safely_in_more(self):
        entry = {
            "workflow": "native",
            "executor": "native",
        }
        meta = _native_meta_from_entry(entry)
        # Should not raise
        try:
            _render_native_demo_block(meta, detail="more")
        except Exception as exc:
            self.fail(f"rendering crashed on old entry: {exc}")

    def test_saved_entry_routing_preview_renders_in_more(self):
        entry = self._entry_with_preview()
        meta = _native_meta_from_entry(entry)
        lines = _render_native_demo_block(meta, detail="more")
        combined = "\n".join(lines)
        self.assertIn("routing preview:", combined)
        self.assertIn("cheapest-safe", combined)

    def test_saved_entry_routing_preview_renders_in_full(self):
        entry = self._entry_with_preview()
        meta = _native_meta_from_entry(entry)
        lines = _render_native_demo_block(meta, detail="full")
        combined = "\n".join(lines)
        self.assertIn("[routing preview]", combined)
        self.assertIn("planner:", combined)

    def test_saved_entry_routing_preview_hidden_in_default(self):
        entry = self._entry_with_preview()
        meta = _native_meta_from_entry(entry)
        lines = _render_native_demo_block(meta, detail="default")
        combined = "\n".join(lines)
        self.assertNotIn("routing preview", combined)
