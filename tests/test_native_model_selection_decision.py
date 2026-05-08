from __future__ import annotations

import json
import unittest
from dataclasses import asdict
from types import SimpleNamespace

from openshard.native.context import (
    NativeModelRoleDecision,
    NativeModelSelectionDecision,
    build_native_model_selection_decision,
    render_native_model_selection_decision,
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


if __name__ == "__main__":
    unittest.main()
