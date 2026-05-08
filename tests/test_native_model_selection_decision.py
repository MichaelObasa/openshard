from __future__ import annotations

import json
import unittest
from dataclasses import asdict
from types import SimpleNamespace

from openshard.native.context import (
    NativeModelCandidateScoring,
    NativeModelRoleDecision,
    NativeModelSelectionDecision,
    build_native_model_selection_decision,
    render_native_model_selection_decision,
    sync_native_model_selection_decision_with_candidate_scoring,
)


def _build(**kwargs) -> NativeModelSelectionDecision:
    return build_native_model_selection_decision(**kwargs)


def _ns(**kwargs):
    return SimpleNamespace(**kwargs)


class TestNativeModelRoleDecisionDefaults(unittest.TestCase):
    def test_defaults(self):
        r = NativeModelRoleDecision()
        self.assertEqual(r.role, "")
        self.assertEqual(r.model_tier, "")
        self.assertEqual(r.cost_tier, "")
        self.assertEqual(r.reason, "")


class TestNativeModelSelectionDecisionDefaults(unittest.TestCase):
    def test_defaults(self):
        d = NativeModelSelectionDecision()
        self.assertEqual(d.strategy, "cost-balanced")
        self.assertEqual(d.task_type, "unknown")
        self.assertEqual(d.risk_level, "unknown")
        self.assertEqual(d.roles, [])
        self.assertEqual(d.warnings, [])
        self.assertEqual(d.fallback_reason, "")
        self.assertEqual(d.confidence, "medium")


class TestNativeModelSelectionDecisionJSONSerializable(unittest.TestCase):
    def test_asdict_json_roundtrip(self):
        d = NativeModelSelectionDecision(
            strategy="frontier-heavy",
            task_type="feature",
            risk_level="high",
            roles=[
                NativeModelRoleDecision(role="planner", model_tier="frontier-reasoning-model", cost_tier="high", reason="high risk"),
                NativeModelRoleDecision(role="executor", model_tier="frontier-reasoning-model", cost_tier="high", reason="high risk exec"),
                NativeModelRoleDecision(role="validator", model_tier="independent-validator-model", cost_tier="medium", reason="independent"),
            ],
            warnings=["run trust score is weak"],
            fallback_reason="",
            confidence="high",
        )
        raw = json.dumps(asdict(d))
        parsed = json.loads(raw)
        self.assertEqual(parsed["strategy"], "frontier-heavy")
        self.assertEqual(parsed["confidence"], "high")
        self.assertEqual(len(parsed["roles"]), 3)
        self.assertEqual(parsed["roles"][0]["role"], "planner")


class TestNativeModelSelectionDecisionStrategyRules(unittest.TestCase):
    def test_high_risk_produces_frontier_heavy(self):
        vp = _ns(risk_level="high", task_type="feature")
        result = _build(verification_plan=vp)
        self.assertEqual(result.strategy, "frontier-heavy")

    def test_weak_validation_produces_frontier_heavy(self):
        vc = _ns(strength="weak", risk_level="medium")
        result = _build(validation_contract=vc)
        self.assertEqual(result.strategy, "frontier-heavy")

    def test_weak_trust_produces_frontier_heavy(self):
        rts = _ns(level="weak")
        result = _build(run_trust_score=rts)
        self.assertEqual(result.strategy, "frontier-heavy")

    def test_weak_context_quality_produces_context_cautious(self):
        cqs = _ns(level="weak")
        vp = _ns(risk_level="medium", task_type="feature")
        vc = _ns(strength="fair", risk_level="medium")
        rts = _ns(level="good")
        result = _build(
            context_quality_score=cqs,
            verification_plan=vp,
            validation_contract=vc,
            run_trust_score=rts,
        )
        self.assertEqual(result.strategy, "context-cautious")

    def test_context_gaps_produce_context_cautious(self):
        cp = _ns(has_gaps=True, warnings=[])
        vp = _ns(risk_level="medium", task_type="feature")
        vc = _ns(strength="fair", risk_level="medium")
        rts = _ns(level="good")
        cqs = _ns(level="good")
        result = _build(
            context_provenance=cp,
            verification_plan=vp,
            validation_contract=vc,
            run_trust_score=rts,
            context_quality_score=cqs,
        )
        self.assertEqual(result.strategy, "context-cautious")

    def test_low_risk_fair_validation_produces_cost_balanced(self):
        vp = _ns(risk_level="low", task_type="feature")
        vc = _ns(strength="fair", risk_level="low")
        result = _build(verification_plan=vp, validation_contract=vc)
        self.assertEqual(result.strategy, "cost-balanced")

    def test_medium_risk_strong_validation_produces_cost_balanced(self):
        vp = _ns(risk_level="medium", task_type="feature")
        vc = _ns(strength="strong", risk_level="medium")
        result = _build(verification_plan=vp, validation_contract=vc)
        self.assertEqual(result.strategy, "cost-balanced")

    def test_all_none_inputs_produces_cost_balanced_with_fallback(self):
        result = _build()
        self.assertEqual(result.strategy, "cost-balanced")
        self.assertNotEqual(result.fallback_reason, "")


class TestNativeModelSelectionDecisionRoles(unittest.TestCase):
    def test_always_has_three_roles(self):
        result = _build()
        self.assertEqual(len(result.roles), 3)

    def test_roles_always_has_planner_executor_validator(self):
        result = _build()
        names = {r.role for r in result.roles}
        self.assertEqual(names, {"planner", "executor", "validator"})

    def test_docs_task_produces_low_cost_executor(self):
        vp = _ns(risk_level="low", task_type="docs")
        vc = _ns(strength="fair", risk_level="low")
        result = _build(verification_plan=vp, validation_contract=vc)
        executor = next(r for r in result.roles if r.role == "executor")
        self.assertEqual(executor.model_tier, "low-cost-coding-model")

    def test_test_task_produces_low_cost_executor(self):
        vp = _ns(risk_level="low", task_type="test")
        vc = _ns(strength="fair", risk_level="low")
        result = _build(verification_plan=vp, validation_contract=vc)
        executor = next(r for r in result.roles if r.role == "executor")
        self.assertEqual(executor.model_tier, "low-cost-coding-model")

    def test_config_task_produces_low_cost_executor(self):
        vp = _ns(risk_level="low", task_type="config")
        vc = _ns(strength="fair", risk_level="low")
        result = _build(verification_plan=vp, validation_contract=vc)
        executor = next(r for r in result.roles if r.role == "executor")
        self.assertEqual(executor.model_tier, "low-cost-coding-model")

    def test_frontier_heavy_produces_frontier_executor(self):
        vp = _ns(risk_level="high", task_type="feature")
        result = _build(verification_plan=vp)
        executor = next(r for r in result.roles if r.role == "executor")
        self.assertEqual(executor.model_tier, "frontier-reasoning-model")

    def test_planner_always_frontier(self):
        for risk in ("low", "medium", "high"):
            vp = _ns(risk_level=risk, task_type="feature")
            result = _build(verification_plan=vp)
            planner = next(r for r in result.roles if r.role == "planner")
            self.assertEqual(planner.model_tier, "frontier-reasoning-model")

    def test_validator_always_independent(self):
        for risk in ("low", "medium", "high"):
            vp = _ns(risk_level=risk, task_type="feature")
            result = _build(verification_plan=vp)
            validator = next(r for r in result.roles if r.role == "validator")
            self.assertEqual(validator.model_tier, "independent-validator-model")


class TestNativeModelSelectionDecisionWarnings(unittest.TestCase):
    def test_weak_context_quality_generates_warning(self):
        cqs = _ns(level="weak")
        result = _build(context_quality_score=cqs)
        self.assertIn("context quality is weak", result.warnings)

    def test_weak_validation_contract_generates_warning(self):
        vc = _ns(strength="weak", risk_level="medium")
        result = _build(validation_contract=vc)
        self.assertIn("validation contract is weak — signals may be unreliable", result.warnings)

    def test_weak_run_trust_generates_warning(self):
        rts = _ns(level="weak")
        result = _build(run_trust_score=rts)
        self.assertIn("run trust score is weak", result.warnings)

    def test_failure_memory_lessons_generate_warning(self):
        fm = _ns(has_lessons=True, lessons=[])
        result = _build(failure_memory=fm)
        self.assertIn("failure memory has lessons that may affect model reliability", result.warnings)

    def test_no_warnings_when_all_clean(self):
        vp = _ns(risk_level="medium", task_type="feature")
        vc = _ns(strength="strong", risk_level="medium")
        cqs = _ns(level="good")
        cp = _ns(has_gaps=False, warnings=[])
        rts = _ns(level="good")
        fm = _ns(has_lessons=False, lessons=[])
        result = _build(
            verification_plan=vp,
            validation_contract=vc,
            context_quality_score=cqs,
            context_provenance=cp,
            run_trust_score=rts,
            failure_memory=fm,
        )
        self.assertEqual(result.warnings, [])


class TestNativeModelSelectionDecisionRenderer(unittest.TestCase):
    def _msd(self, strategy="cost-balanced", task_type="feature", risk_level="medium",
             confidence="high", fallback_reason="") -> NativeModelSelectionDecision:
        return NativeModelSelectionDecision(
            strategy=strategy,
            task_type=task_type,
            risk_level=risk_level,
            roles=[
                NativeModelRoleDecision(role="planner", model_tier="frontier-reasoning-model", cost_tier="high", reason="planning"),
                NativeModelRoleDecision(role="executor", model_tier="balanced-coding-model", cost_tier="medium", reason="balanced"),
                NativeModelRoleDecision(role="validator", model_tier="independent-validator-model", cost_tier="medium", reason="independent"),
            ],
            warnings=[],
            fallback_reason=fallback_reason,
            confidence=confidence,
        )

    def test_none_returns_empty_string(self):
        self.assertEqual(render_native_model_selection_decision(None), "")

    def test_unknown_detail_returns_empty_string(self):
        msd = self._msd()
        self.assertEqual(render_native_model_selection_decision(msd, detail="default"), "")

    def test_compact_contains_strategy(self):
        msd = self._msd(strategy="cost-balanced")
        out = render_native_model_selection_decision(msd, detail="compact")
        self.assertIn("cost-balanced", out)

    def test_compact_contains_confidence(self):
        msd = self._msd(confidence="high")
        out = render_native_model_selection_decision(msd, detail="compact")
        self.assertIn("confidence=high", out)

    def test_compact_contains_planner_and_executor(self):
        msd = self._msd()
        out = render_native_model_selection_decision(msd, detail="compact")
        self.assertIn("planner=frontier-reasoning-model", out)
        self.assertIn("executor=balanced-coding-model", out)

    def test_full_contains_header(self):
        msd = self._msd()
        out = render_native_model_selection_decision(msd, detail="full")
        self.assertIn("[model selection]", out)

    def test_full_contains_all_role_names(self):
        msd = self._msd()
        out = render_native_model_selection_decision(msd, detail="full")
        self.assertIn("planner", out)
        self.assertIn("executor", out)
        self.assertIn("validator", out)

    def test_full_contains_role_tiers(self):
        msd = self._msd()
        out = render_native_model_selection_decision(msd, detail="full")
        self.assertIn("frontier-reasoning-model", out)
        self.assertIn("independent-validator-model", out)

    def test_fallback_reason_shown_in_full_when_present(self):
        msd = self._msd(fallback_reason="insufficient signal — defaulting to cost-balanced")
        out = render_native_model_selection_decision(msd, detail="full")
        self.assertIn("fallback:", out)
        self.assertIn("insufficient signal", out)

    def test_fallback_reason_absent_in_compact(self):
        msd = self._msd(fallback_reason="insufficient signal — defaulting to cost-balanced")
        out = render_native_model_selection_decision(msd, detail="compact")
        self.assertNotIn("insufficient signal", out)
        self.assertNotIn("fallback", out)


class TestSyncNativeModelSelectionDecision(unittest.TestCase):
    def _decision(self, **kwargs) -> NativeModelSelectionDecision:
        roles = [
            NativeModelRoleDecision(role="planner", model_tier="frontier-reasoning-model", cost_tier="high", reason="default"),
            NativeModelRoleDecision(role="executor", model_tier="frontier-reasoning-model", cost_tier="high", reason="default"),
            NativeModelRoleDecision(role="validator", model_tier="independent-validator-model", cost_tier="low", reason="default"),
        ]
        return NativeModelSelectionDecision(
            strategy=kwargs.get("strategy", "cost-balanced"),
            confidence=kwargs.get("confidence", "medium"),
            roles=kwargs.get("roles", roles),
            warnings=kwargs.get("warnings", []),
            fallback_reason=kwargs.get("fallback_reason", ""),
        )

    def _scoring(self, selected_by_role: dict) -> NativeModelCandidateScoring:
        return NativeModelCandidateScoring(selected_by_role=selected_by_role)

    def test_no_candidate_scoring_returns_original(self):
        d = self._decision()
        result = sync_native_model_selection_decision_with_candidate_scoring(d, None)
        self.assertIs(result, d)

    def test_no_selection_decision_returns_none(self):
        s = self._scoring({"executor": "balanced-coding-model"})
        result = sync_native_model_selection_decision_with_candidate_scoring(None, s)
        self.assertIsNone(result)

    def test_selected_by_role_updates_executor_tier(self):
        d = self._decision()
        s = self._scoring({"executor": "balanced-coding-model"})
        result = sync_native_model_selection_decision_with_candidate_scoring(d, s)
        executor = next(r for r in result.roles if r.role == "executor")
        self.assertEqual(executor.model_tier, "balanced-coding-model")

    def test_selected_by_role_updates_planner_tier(self):
        d = self._decision()
        s = self._scoring({"planner": "balanced-coding-model"})
        result = sync_native_model_selection_decision_with_candidate_scoring(d, s)
        planner = next(r for r in result.roles if r.role == "planner")
        self.assertEqual(planner.model_tier, "balanced-coding-model")

    def test_missing_role_is_added(self):
        d = self._decision(roles=[
            NativeModelRoleDecision(role="planner", model_tier="frontier-reasoning-model"),
        ])
        s = self._scoring({"reviewer": "balanced-coding-model"})
        result = sync_native_model_selection_decision_with_candidate_scoring(d, s)
        roles = {r.role: r for r in result.roles}
        self.assertIn("reviewer", roles)
        self.assertEqual(roles["reviewer"].model_tier, "balanced-coding-model")

    def test_warning_added_when_changed(self):
        d = self._decision()
        s = self._scoring({"executor": "balanced-coding-model"})
        result = sync_native_model_selection_decision_with_candidate_scoring(d, s)
        self.assertIn("model selection adjusted by policy enforcement", result.warnings)

    def test_warning_not_duplicated_when_already_present(self):
        d = self._decision(warnings=["model selection adjusted by policy enforcement"])
        s = self._scoring({"executor": "balanced-coding-model"})
        result = sync_native_model_selection_decision_with_candidate_scoring(d, s)
        self.assertEqual(result.warnings.count("model selection adjusted by policy enforcement"), 1)

    def test_no_warning_when_unchanged(self):
        d = self._decision()
        s = self._scoring({"executor": "frontier-reasoning-model"})
        result = sync_native_model_selection_decision_with_candidate_scoring(d, s)
        self.assertEqual(result.warnings, [])

    def test_fallback_reason_set_when_changed_and_empty(self):
        d = self._decision()
        s = self._scoring({"executor": "balanced-coding-model"})
        result = sync_native_model_selection_decision_with_candidate_scoring(d, s)
        self.assertEqual(result.fallback_reason, "candidate scoring applied model policy constraints")

    def test_fallback_reason_not_overwritten_when_already_set(self):
        d = self._decision(fallback_reason="existing reason")
        s = self._scoring({"executor": "balanced-coding-model"})
        result = sync_native_model_selection_decision_with_candidate_scoring(d, s)
        self.assertEqual(result.fallback_reason, "existing reason")

    def test_original_strategy_confidence_preserved(self):
        d = self._decision(strategy="frontier-heavy", confidence="high")
        s = self._scoring({"executor": "balanced-coding-model"})
        result = sync_native_model_selection_decision_with_candidate_scoring(d, s)
        self.assertEqual(result.strategy, "frontier-heavy")
        self.assertEqual(result.confidence, "high")

    def test_does_not_mutate_original(self):
        d = self._decision()
        original_tier = next(r for r in d.roles if r.role == "executor").model_tier
        s = self._scoring({"executor": "balanced-coding-model"})
        sync_native_model_selection_decision_with_candidate_scoring(d, s)
        self.assertEqual(next(r for r in d.roles if r.role == "executor").model_tier, original_tier)

    def test_empty_selected_by_role_returns_original(self):
        d = self._decision()
        s = self._scoring({})
        result = sync_native_model_selection_decision_with_candidate_scoring(d, s)
        self.assertIs(result, d)


if __name__ == "__main__":
    unittest.main()
