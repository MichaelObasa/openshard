from __future__ import annotations

import json
import unittest
from dataclasses import asdict
from types import SimpleNamespace

from openshard.native.context import (
    NativeModelCandidateScore,
    NativeModelCandidateScoring,
    build_native_model_candidate_scoring,
    render_native_model_candidate_scoring,
)


def _ns(**kwargs: object) -> SimpleNamespace:
    return SimpleNamespace(**kwargs)


def _build(**kwargs: object) -> NativeModelCandidateScoring:
    return build_native_model_candidate_scoring(**kwargs)


class TestNativeModelCandidateScoreDefaults(unittest.TestCase):
    def test_field_defaults(self) -> None:
        s = NativeModelCandidateScore()
        self.assertEqual(s.role, "")
        self.assertEqual(s.candidate, "")
        self.assertEqual(s.score, 0)
        self.assertEqual(s.capability_score, 0)
        self.assertEqual(s.risk_fit_score, 0)
        self.assertEqual(s.context_fit_score, 0)
        self.assertEqual(s.cost_score, 0)
        self.assertEqual(s.verification_fit_score, 0)
        self.assertEqual(s.penalty, 0)
        self.assertEqual(s.reason, "")

    def test_asdict_json_roundtrip(self) -> None:
        s = NativeModelCandidateScore(
            role="executor",
            candidate="balanced-coding-model",
            score=72,
            capability_score=15,
            cost_score=10,
            reason="capability+15, cost+10",
        )
        d = asdict(s)
        raw = json.dumps(d)
        parsed = json.loads(raw)
        self.assertEqual(parsed["role"], "executor")
        self.assertEqual(parsed["score"], 72)
        self.assertEqual(parsed["reason"], "capability+15, cost+10")


class TestNativeModelCandidateScoringDefaults(unittest.TestCase):
    def test_field_defaults(self) -> None:
        sc = NativeModelCandidateScoring()
        self.assertEqual(sc.candidates, [])
        self.assertEqual(sc.selected_by_role, {})
        self.assertEqual(sc.strategy, "cost-balanced")
        self.assertEqual(sc.confidence, "medium")
        self.assertEqual(sc.warnings, [])

    def test_asdict_json_roundtrip(self) -> None:
        sc = NativeModelCandidateScoring(
            strategy="frontier-heavy",
            confidence="high",
            selected_by_role={"planner": "frontier-reasoning-model"},
            warnings=["validation contract weak"],
        )
        d = asdict(sc)
        raw = json.dumps(d)
        parsed = json.loads(raw)
        self.assertEqual(parsed["strategy"], "frontier-heavy")
        self.assertEqual(parsed["confidence"], "high")
        self.assertEqual(parsed["selected_by_role"]["planner"], "frontier-reasoning-model")
        self.assertEqual(len(parsed["warnings"]), 1)


class TestBuildNativeModelCandidateScoringAllNone(unittest.TestCase):
    def test_all_none_returns_valid_object(self) -> None:
        sc = _build()
        self.assertIsInstance(sc, NativeModelCandidateScoring)
        self.assertIsInstance(sc.candidates, list)
        self.assertIsInstance(sc.selected_by_role, dict)
        self.assertIsInstance(sc.warnings, list)

    def test_all_none_produces_default_three_roles(self) -> None:
        sc = _build()
        roles = {c.role for c in sc.candidates}
        self.assertIn("planner", roles)
        self.assertIn("executor", roles)
        self.assertIn("validator", roles)

    def test_all_none_selected_by_role_defaults(self) -> None:
        sc = _build()
        self.assertEqual(sc.selected_by_role.get("planner"), "frontier-reasoning-model")
        self.assertEqual(sc.selected_by_role.get("executor"), "balanced-coding-model")
        self.assertEqual(sc.selected_by_role.get("validator"), "independent-validator-model")

    def test_all_none_scores_clamped(self) -> None:
        sc = _build()
        for c in sc.candidates:
            self.assertGreaterEqual(c.score, 0)
            self.assertLessEqual(c.score, 100)

    def test_all_none_strategy_and_confidence_defaults(self) -> None:
        sc = _build()
        self.assertEqual(sc.strategy, "cost-balanced")
        self.assertEqual(sc.confidence, "medium")

    def test_all_none_warning_for_missing_decision(self) -> None:
        sc = _build()
        self.assertIn("model selection decision missing", sc.warnings)


class TestBuildNativeModelCandidateScoringScoreRules(unittest.TestCase):
    def _score_for(self, sc: NativeModelCandidateScoring, role: str, candidate: str) -> int:
        for c in sc.candidates:
            if c.role == role and c.candidate == candidate:
                return c.score
        raise KeyError(f"{role}/{candidate} not in candidates")

    def test_high_risk_boosts_frontier_executor(self) -> None:
        vp = _ns(risk_level="high", task_type="feature", verification_commands=[])
        sc = _build(verification_plan=vp)
        frontier = self._score_for(sc, "executor", "frontier-reasoning-model")
        low_cost = self._score_for(sc, "executor", "low-cost-coding-model")
        self.assertGreater(frontier, low_cost)

    def test_low_risk_boosts_low_cost_executor(self) -> None:
        vp = _ns(risk_level="low", task_type="feature", verification_commands=[])
        sc = _build(verification_plan=vp)
        low_cost = self._score_for(sc, "executor", "low-cost-coding-model")
        frontier = self._score_for(sc, "executor", "frontier-reasoning-model")
        self.assertGreater(low_cost, frontier)

    def test_docs_task_boosts_low_cost_capability(self) -> None:
        vp = _ns(risk_level="low", task_type="docs", verification_commands=[])
        sc = _build(verification_plan=vp)
        for c in sc.candidates:
            if c.role == "executor" and c.candidate == "low-cost-coding-model":
                self.assertGreater(c.capability_score, 0)
                return
        self.fail("low-cost-coding-model executor not found")

    def test_weak_trust_penalizes_low_cost_executor(self) -> None:
        rts = _ns(level="weak")
        vp = _ns(risk_level="medium", task_type="feature", verification_commands=[])
        sc_weak = _build(run_trust_score=rts, verification_plan=vp)
        sc_none = _build(verification_plan=vp)
        weak_score = self._score_for(sc_weak, "executor", "low-cost-coding-model")
        base_score = self._score_for(sc_none, "executor", "low-cost-coding-model")
        self.assertLessEqual(weak_score, base_score)

    def test_truncated_context_penalizes_low_cost(self) -> None:
        cus = _ns(any_truncated=True)
        sc_trunc = _build(context_usage_summary=cus)
        sc_none = _build()
        trunc_score = self._score_for(sc_trunc, "executor", "low-cost-coding-model")
        base_score = self._score_for(sc_none, "executor", "low-cost-coding-model")
        self.assertLess(trunc_score, base_score)

    def test_validator_candidate_boosted_for_validator_role(self) -> None:
        sc = _build()
        validator_score = self._score_for(sc, "validator", "independent-validator-model")
        low_cost_val = self._score_for(sc, "validator", "low-cost-coding-model")
        self.assertGreater(validator_score, low_cost_val)

    def test_weak_validation_contract_adds_warning(self) -> None:
        vc = _ns(strength="weak", risk_level="medium")
        sc = _build(validation_contract=vc)
        self.assertIn("validation contract weak", sc.warnings)

    def test_context_gaps_add_warning(self) -> None:
        cp = _ns(has_gaps=True)
        sc = _build(context_provenance=cp)
        self.assertIn("context gaps may affect candidate scoring", sc.warnings)

    def test_weak_trust_adds_warning(self) -> None:
        rts = _ns(level="weak")
        sc = _build(run_trust_score=rts)
        self.assertIn("low trust run may reduce model selection confidence", sc.warnings)

    def test_model_selection_decision_populates_selected_by_role(self) -> None:
        role1 = _ns(role="planner", model_tier="frontier-reasoning-model", cost_tier="high", reason="r")
        role2 = _ns(role="executor", model_tier="balanced-coding-model", cost_tier="medium", reason="r")
        role3 = _ns(role="validator", model_tier="independent-validator-model", cost_tier="medium", reason="r")
        msd = _ns(strategy="cost-balanced", confidence="high", roles=[role1, role2, role3], warnings=[])
        sc = _build(model_selection_decision=msd)
        self.assertEqual(sc.selected_by_role.get("planner"), "frontier-reasoning-model")
        self.assertEqual(sc.selected_by_role.get("executor"), "balanced-coding-model")
        self.assertEqual(sc.selected_by_role.get("validator"), "independent-validator-model")
        self.assertNotIn("model selection decision missing", sc.warnings)

    def test_model_selection_decision_strategy_confidence_copied(self) -> None:
        msd = _ns(strategy="frontier-heavy", confidence="high", roles=[], warnings=[])
        sc = _build(model_selection_decision=msd)
        self.assertEqual(sc.strategy, "frontier-heavy")
        self.assertEqual(sc.confidence, "high")

    def test_context_quality_weak_boosts_frontier(self) -> None:
        cqs = _ns(level="weak")
        sc = _build(context_quality_score=cqs)
        for c in sc.candidates:
            if c.role == "planner" and c.candidate == "frontier-reasoning-model":
                self.assertGreater(c.context_fit_score, 0)
                return
        self.fail("planner/frontier-reasoning-model not found")

    def test_has_lessons_penalizes_low_cost(self) -> None:
        fm = _ns(has_lessons=True)
        sc_lessons = _build(failure_memory=fm)
        sc_none = _build()
        for role in ("planner", "executor", "validator"):
            lc_lessons = self._score_for(sc_lessons, role, "low-cost-coding-model")
            lc_none = self._score_for(sc_none, role, "low-cost-coding-model")
            self.assertLessEqual(lc_lessons, lc_none, f"role={role}")

    def test_verification_commands_boost_balanced_executor(self) -> None:
        vp_with = _ns(risk_level="medium", task_type="feature", verification_commands=["pytest"])
        vp_without = _ns(risk_level="medium", task_type="feature", verification_commands=[])
        sc_with = _build(verification_plan=vp_with)
        sc_without = _build(verification_plan=vp_without)
        with_score = self._score_for(sc_with, "executor", "balanced-coding-model")
        without_score = self._score_for(sc_without, "executor", "balanced-coding-model")
        self.assertGreaterEqual(with_score, without_score)

    def test_no_verification_commands_penalizes_low_cost_executor(self) -> None:
        vp_no = _ns(risk_level="medium", task_type="feature", verification_commands=[])
        sc = _build(verification_plan=vp_no)
        for c in sc.candidates:
            if c.role == "executor" and c.candidate == "low-cost-coding-model":
                self.assertLess(c.verification_fit_score, 0)
                return
        self.fail("executor/low-cost-coding-model not found")

    def test_all_scores_clamped_0_100_under_worst_inputs(self) -> None:
        vp = _ns(risk_level="high", task_type="feature", verification_commands=[])
        vc = _ns(strength="weak", risk_level="high")
        rts = _ns(level="weak")
        cqs = _ns(level="weak")
        cp = _ns(has_gaps=True)
        cus = _ns(any_truncated=True)
        fm = _ns(has_lessons=True)
        sc = _build(
            verification_plan=vp,
            validation_contract=vc,
            run_trust_score=rts,
            context_quality_score=cqs,
            context_provenance=cp,
            context_usage_summary=cus,
            failure_memory=fm,
        )
        for c in sc.candidates:
            self.assertGreaterEqual(c.score, 0, f"{c.role}/{c.candidate}")
            self.assertLessEqual(c.score, 100, f"{c.role}/{c.candidate}")

    def test_dict_roles_on_model_selection_decision_handled(self) -> None:
        msd = _ns(
            strategy="cost-balanced",
            confidence="medium",
            roles=[
                {"role": "planner", "model_tier": "frontier-reasoning-model", "cost_tier": "high", "reason": "r"},
                {"role": "executor", "model_tier": "balanced-coding-model", "cost_tier": "medium", "reason": "r"},
                {"role": "validator", "model_tier": "independent-validator-model", "cost_tier": "medium", "reason": "r"},
            ],
            warnings=[],
        )
        sc = _build(model_selection_decision=msd)
        self.assertEqual(sc.selected_by_role.get("planner"), "frontier-reasoning-model")


class TestRenderNativeModelCandidateScoring(unittest.TestCase):
    def test_none_returns_empty_string(self) -> None:
        self.assertEqual(render_native_model_candidate_scoring(None), "")

    def test_compact_contains_strategy(self) -> None:
        sc = NativeModelCandidateScoring(strategy="frontier-heavy", confidence="high")
        out = render_native_model_candidate_scoring(sc, detail="compact")
        self.assertIn("frontier-heavy", out)
        self.assertIn("high", out)

    def test_compact_contains_role_count(self) -> None:
        sc = NativeModelCandidateScoring(
            selected_by_role={"planner": "frontier-reasoning-model", "executor": "balanced-coding-model"},
            strategy="cost-balanced",
            confidence="medium",
        )
        out = render_native_model_candidate_scoring(sc, detail="compact")
        self.assertIn("2 roles", out)

    def test_compact_does_not_expose_raw_warnings(self) -> None:
        sc = NativeModelCandidateScoring(
            strategy="cost-balanced",
            confidence="medium",
            warnings=["some secret warning text"],
        )
        out = render_native_model_candidate_scoring(sc, detail="compact")
        self.assertNotIn("some secret warning text", out)

    def test_full_contains_model_candidates_header(self) -> None:
        sc = _build()
        out = render_native_model_candidate_scoring(sc, detail="full")
        self.assertIn("[model candidates]", out)

    def test_full_contains_selected_roles(self) -> None:
        sc = _build()
        out = render_native_model_candidate_scoring(sc, detail="full")
        self.assertIn("planner", out)
        self.assertIn("executor", out)
        self.assertIn("validator", out)

    def test_full_contains_score_lines(self) -> None:
        sc = _build()
        out = render_native_model_candidate_scoring(sc, detail="full")
        self.assertIn("scores:", out)
        self.assertIn("frontier-reasoning-model", out)

    def test_full_contains_warnings_count(self) -> None:
        sc = _build()
        out = render_native_model_candidate_scoring(sc, detail="full")
        self.assertIn("warnings:", out)

    def test_full_strategy_and_confidence_lines(self) -> None:
        msd = _ns(strategy="context-cautious", confidence="low", roles=[], warnings=[])
        sc = _build(model_selection_decision=msd)
        out = render_native_model_candidate_scoring(sc, detail="full")
        self.assertIn("strategy: context-cautious", out)
        self.assertIn("confidence: low", out)

    def test_full_with_dict_candidates_no_crash(self) -> None:
        sc = NativeModelCandidateScoring(
            strategy="cost-balanced",
            confidence="medium",
            selected_by_role={"executor": "balanced-coding-model"},
            candidates=[  # type: ignore[list-item]
                {"role": "executor", "candidate": "balanced-coding-model", "score": 75,
                 "capability_score": 15, "risk_fit_score": 0, "context_fit_score": 0,
                 "cost_score": 10, "verification_fit_score": 0, "penalty": 0, "reason": "cap+15"},
            ],
        )
        try:
            out = render_native_model_candidate_scoring(sc, detail="full")
        except Exception as exc:
            self.fail(f"render raised {exc}")
        self.assertIn("scores:", out)
        self.assertIn("executor/balanced-coding-model: 75", out)

    def test_asdict_json_roundtrip_full_scoring(self) -> None:
        sc = _build()
        d = asdict(sc)
        raw = json.dumps(d)
        parsed = json.loads(raw)
        self.assertIsInstance(parsed["candidates"], list)
        self.assertIsInstance(parsed["selected_by_role"], dict)
        self.assertIsInstance(parsed["warnings"], list)


if __name__ == "__main__":
    unittest.main()
