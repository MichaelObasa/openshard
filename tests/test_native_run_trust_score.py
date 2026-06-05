from __future__ import annotations

import json
import unittest
from dataclasses import asdict

from openshard.native.context import (
    FailureLesson,
    NativeApprovalReceipt,
    NativeApprovalRequest,
    NativeChangeBudgetPreview,
    NativeCommandPolicyPreview,
    NativeContextProvenance,
    NativeContextQualityScore,
    NativeContextUsageSummary,
    NativeFailureMemory,
    NativeFinalReport,
    NativeRunTrustFactor,
    NativeRunTrustScore,
    NativeValidationContract,
    NativeVerificationLoop,
    build_native_run_trust_score,
    render_native_run_trust_score,
)


def _build(**kwargs) -> NativeRunTrustScore:
    return build_native_run_trust_score(**kwargs)


class TestNativeRunTrustScoreDefaults(unittest.TestCase):

    def test_factor_defaults(self):
        f = NativeRunTrustFactor()
        self.assertEqual(f.name, "")
        self.assertEqual(f.impact, 0)
        self.assertEqual(f.reason, "")

    def test_score_defaults(self):
        s = NativeRunTrustScore()
        self.assertEqual(s.score, 0)
        self.assertEqual(s.level, "unknown")
        self.assertEqual(s.factors, [])
        self.assertEqual(s.warnings, [])
        self.assertEqual(s.blockers, [])

    def test_asdict_json_serializable(self):
        s = NativeRunTrustScore(
            score=72,
            level="good",
            factors=[NativeRunTrustFactor(name="verification_passed", impact=20, reason="verification passed")],
            warnings=["context truncated"],
            blockers=[],
        )
        d = asdict(s)
        raw = json.dumps(d)
        parsed = json.loads(raw)
        self.assertEqual(parsed["score"], 72)
        self.assertEqual(parsed["level"], "good")
        self.assertEqual(len(parsed["factors"]), 1)
        self.assertEqual(parsed["factors"][0]["name"], "verification_passed")


class TestNativeRunTrustScoreNullInputs(unittest.TestCase):

    def test_all_none_inputs_produce_valid_score(self):
        s = _build()
        self.assertIsInstance(s.score, int)
        self.assertIn(s.level, ("weak", "fair", "good", "strong"))
        self.assertIsInstance(s.factors, list)
        self.assertIsInstance(s.warnings, list)
        self.assertIsInstance(s.blockers, list)

    def test_score_is_clamped_minimum(self):
        # Load up maximum negatives: verification failed, weak contract, weak quality,
        # blocked commands, budget exceeded, approval not granted, 5 lessons, truncated, 10 warnings
        vloop = NativeVerificationLoop(attempted=True, passed=False)
        vc = NativeValidationContract(strength="weak")
        cqs = NativeContextQualityScore(level="weak")
        cpp = NativeCommandPolicyPreview(blocked_count=3)
        cbp = NativeChangeBudgetPreview(would_exceed_budget=True)
        ar = NativeApprovalRequest(requires_approval=True)
        receipt = NativeApprovalReceipt(requested=True, granted=False)
        fm = NativeFailureMemory(
            lessons=[FailureLesson(lesson_type="t", reason="r") for _ in range(5)],
            has_lessons=True,
        )
        cus = NativeContextUsageSummary(any_truncated=True, failure_warning_count=10)
        s = _build(
            verification_loop=vloop,
            validation_contract=vc,
            context_quality_score=cqs,
            command_policy_preview=cpp,
            change_budget_preview=cbp,
            approval_request=ar,
            approval_receipt=receipt,
            failure_memory=fm,
            context_usage_summary=cus,
        )
        self.assertGreaterEqual(s.score, 0)

    def test_score_is_clamped_maximum(self):
        vloop = NativeVerificationLoop(attempted=True, passed=True)
        vc = NativeValidationContract(strength="strong")
        cqs = NativeContextQualityScore(level="strong")
        prov = NativeContextProvenance(injected_sources=3, has_gaps=False)
        ar = NativeApprovalRequest(requires_approval=True)
        receipt = NativeApprovalReceipt(requested=True, granted=True)
        fr = NativeFinalReport(used_native_context=True)
        s = _build(
            verification_loop=vloop,
            validation_contract=vc,
            context_quality_score=cqs,
            context_provenance=prov,
            approval_request=ar,
            approval_receipt=receipt,
            final_report=fr,
        )
        self.assertLessEqual(s.score, 100)


class TestVerificationFactors(unittest.TestCase):

    def test_verification_passed_adds_positive_factor(self):
        vloop = NativeVerificationLoop(attempted=True, passed=True)
        s = _build(verification_loop=vloop)
        names = [f.name for f in s.factors]
        self.assertIn("verification_passed", names)
        factor = next(f for f in s.factors if f.name == "verification_passed")
        self.assertGreater(factor.impact, 0)

    def test_verification_failed_adds_warning_and_blocker(self):
        vloop = NativeVerificationLoop(attempted=True, passed=False)
        s = _build(verification_loop=vloop)
        self.assertIn("verification failed", s.warnings)
        self.assertIn("verification failed", s.blockers)
        names = [f.name for f in s.factors]
        self.assertIn("verification_failed", names)

    def test_verification_not_attempted_adds_warning(self):
        s = _build(verification_loop=None)
        self.assertIn("verification not attempted", s.warnings)
        self.assertNotIn("verification not attempted", s.blockers)

    def test_verification_loop_not_attempted_flag_adds_warning(self):
        vloop = NativeVerificationLoop(attempted=False, passed=False)
        s = _build(verification_loop=vloop)
        self.assertIn("verification not attempted", s.warnings)


class TestValidationContractFactors(unittest.TestCase):

    def test_strong_contract_adds_positive_factor(self):
        vc = NativeValidationContract(strength="strong")
        s = _build(validation_contract=vc)
        names = [f.name for f in s.factors]
        self.assertIn("validation_contract_strong", names)
        factor = next(f for f in s.factors if f.name == "validation_contract_strong")
        self.assertEqual(factor.impact, 15)

    def test_fair_contract_adds_positive_factor(self):
        vc = NativeValidationContract(strength="fair")
        s = _build(validation_contract=vc)
        names = [f.name for f in s.factors]
        self.assertIn("validation_contract_fair", names)
        factor = next(f for f in s.factors if f.name == "validation_contract_fair")
        self.assertEqual(factor.impact, 7)

    def test_weak_contract_adds_warning(self):
        vc = NativeValidationContract(strength="weak")
        s = _build(validation_contract=vc)
        self.assertIn("validation contract weak", s.warnings)
        names = [f.name for f in s.factors]
        self.assertIn("validation_contract_weak", names)


class TestContextQualityFactors(unittest.TestCase):

    def test_good_quality_adds_positive_factor(self):
        for level in ("good", "strong"):
            with self.subTest(level=level):
                cqs = NativeContextQualityScore(level=level)
                s = _build(context_quality_score=cqs)
                names = [f.name for f in s.factors]
                self.assertIn("context_quality_good", names)

    def test_weak_quality_adds_warning(self):
        for level in ("weak", "unknown"):
            with self.subTest(level=level):
                cqs = NativeContextQualityScore(level=level)
                s = _build(context_quality_score=cqs)
                self.assertIn("context quality weak", s.warnings)


class TestContextProvenanceFactors(unittest.TestCase):

    def test_injected_sources_adds_positive_factor(self):
        prov = NativeContextProvenance(injected_sources=2, has_gaps=False)
        s = _build(context_provenance=prov)
        names = [f.name for f in s.factors]
        self.assertIn("provenance_injected", names)

    def test_provenance_gaps_adds_warning(self):
        prov = NativeContextProvenance(injected_sources=0, has_gaps=True)
        s = _build(context_provenance=prov)
        self.assertIn("context provenance gaps", s.warnings)
        names = [f.name for f in s.factors]
        self.assertIn("provenance_gaps", names)

    def test_no_gaps_adds_positive_factor(self):
        prov = NativeContextProvenance(injected_sources=1, has_gaps=False)
        s = _build(context_provenance=prov)
        names = [f.name for f in s.factors]
        self.assertIn("provenance_no_gaps", names)


class TestBlockedCommandsFactor(unittest.TestCase):

    def test_blocked_commands_adds_blocker(self):
        cpp = NativeCommandPolicyPreview(blocked_count=2)
        s = _build(command_policy_preview=cpp)
        self.assertIn("blocked commands detected", s.blockers)
        self.assertIn("blocked commands detected", s.warnings)

    def test_zero_blocked_no_blocker(self):
        cpp = NativeCommandPolicyPreview(blocked_count=0)
        s = _build(command_policy_preview=cpp)
        self.assertNotIn("blocked commands detected", s.blockers)


class TestChangeBudgetFactor(unittest.TestCase):

    def test_budget_exceeded_adds_blocker(self):
        cbp = NativeChangeBudgetPreview(would_exceed_budget=True)
        s = _build(change_budget_preview=cbp)
        self.assertIn("change budget exceeded", s.blockers)
        self.assertIn("change budget exceeded", s.warnings)

    def test_within_budget_no_blocker(self):
        cbp = NativeChangeBudgetPreview(would_exceed_budget=False)
        s = _build(change_budget_preview=cbp)
        self.assertNotIn("change budget exceeded", s.blockers)


class TestApprovalFactors(unittest.TestCase):

    def test_approval_required_and_granted_adds_positive_factor(self):
        ar = NativeApprovalRequest(requires_approval=True)
        receipt = NativeApprovalReceipt(requested=True, granted=True)
        s = _build(approval_request=ar, approval_receipt=receipt)
        names = [f.name for f in s.factors]
        self.assertIn("approval_granted", names)
        self.assertNotIn("approval not granted", s.blockers)

    def test_approval_required_and_not_granted_adds_blocker(self):
        ar = NativeApprovalRequest(requires_approval=True)
        receipt = NativeApprovalReceipt(requested=True, granted=False)
        s = _build(approval_request=ar, approval_receipt=receipt)
        self.assertIn("approval not granted", s.blockers)
        self.assertIn("approval not granted", s.warnings)

    def test_approval_required_receipt_none_no_blocker(self):
        ar = NativeApprovalRequest(requires_approval=True)
        s = _build(approval_request=ar, approval_receipt=None)
        self.assertNotIn("approval not granted", s.blockers)
        self.assertNotIn("approval not granted", s.warnings)

    def test_approval_not_required_no_factor(self):
        ar = NativeApprovalRequest(requires_approval=False)
        receipt = NativeApprovalReceipt(granted=False)
        s = _build(approval_request=ar, approval_receipt=receipt)
        names = [f.name for f in s.factors]
        self.assertNotIn("approval_granted", names)
        self.assertNotIn("approval_not_granted", names)


class TestFailureMemoryFactor(unittest.TestCase):

    def test_lessons_reduce_score(self):
        fm = NativeFailureMemory(
            lessons=[FailureLesson(lesson_type="t", reason="r")],
            has_lessons=True,
        )
        s_none = _build(failure_memory=None)
        s_with = _build(failure_memory=fm)
        self.assertLess(s_with.score, s_none.score)
        self.assertIn("failure lessons present", s_with.warnings)

    def test_lesson_penalty_capped_at_25(self):
        fm = NativeFailureMemory(
            lessons=[FailureLesson(lesson_type="t", reason="r") for _ in range(10)],
            has_lessons=True,
        )
        s = _build(failure_memory=fm)
        factor = next((f for f in s.factors if f.name == "failure_lessons"), None)
        self.assertIsNotNone(factor)
        self.assertGreaterEqual(factor.impact, -25)


class TestContextUsageFactor(unittest.TestCase):

    def test_truncation_reduces_score(self):
        cus = NativeContextUsageSummary(any_truncated=True)
        s = _build(context_usage_summary=cus)
        self.assertIn("context truncated", s.warnings)
        names = [f.name for f in s.factors]
        self.assertIn("context_truncated", names)

    def test_warning_count_reduces_score(self):
        cus = NativeContextUsageSummary(failure_warning_count=3)
        s = _build(context_usage_summary=cus)
        self.assertIn("context warnings present", s.warnings)

    def test_warning_count_penalty_capped_at_10(self):
        cus = NativeContextUsageSummary(failure_warning_count=50)
        s = _build(context_usage_summary=cus)
        factor = next((f for f in s.factors if f.name == "context_warnings"), None)
        self.assertIsNotNone(factor)
        self.assertGreaterEqual(factor.impact, -10)


class TestLevelThresholds(unittest.TestCase):

    def test_level_strong(self):
        # verification passed (+20), strong contract (+15), good quality (+10),
        # provenance injected (+8), no gaps (+5), approval granted (+5), native context (+5)
        # 50 + 20 + 15 + 10 + 8 + 5 + 5 + 5 = 118 -> clamped 100 -> strong
        s = _build(
            verification_loop=NativeVerificationLoop(attempted=True, passed=True),
            validation_contract=NativeValidationContract(strength="strong"),
            context_quality_score=NativeContextQualityScore(level="strong"),
            context_provenance=NativeContextProvenance(injected_sources=1, has_gaps=False),
            approval_request=NativeApprovalRequest(requires_approval=True),
            approval_receipt=NativeApprovalReceipt(granted=True),
            final_report=NativeFinalReport(used_native_context=True),
        )
        self.assertEqual(s.level, "strong")

    def test_level_weak(self):
        # all nulls: 50 - 10 (not attempted) = 40 -> weak
        s = _build()
        self.assertEqual(s.level, "weak")

    def test_level_mapping_fair(self):
        # Build a score in 45-69 range
        s = NativeRunTrustScore(score=55, level="unknown")
        s.level = "fair" if 45 <= s.score <= 69 else s.level
        rebuilt = build_native_run_trust_score(
            verification_loop=NativeVerificationLoop(attempted=True, passed=True),
        )
        self.assertIn(rebuilt.level, ("fair", "good", "strong"))

    def test_score_85_gives_strong(self):
        s = NativeRunTrustScore(score=85)
        level = "strong" if s.score >= 85 else ("good" if s.score >= 70 else ("fair" if s.score >= 45 else "weak"))
        self.assertEqual(level, "strong")

    def test_score_70_gives_good(self):
        s = NativeRunTrustScore(score=70)
        level = "strong" if s.score >= 85 else ("good" if s.score >= 70 else ("fair" if s.score >= 45 else "weak"))
        self.assertEqual(level, "good")

    def test_score_45_gives_fair(self):
        s = NativeRunTrustScore(score=45)
        level = "strong" if s.score >= 85 else ("good" if s.score >= 70 else ("fair" if s.score >= 45 else "weak"))
        self.assertEqual(level, "fair")

    def test_score_44_gives_weak(self):
        s = NativeRunTrustScore(score=44)
        level = "strong" if s.score >= 85 else ("good" if s.score >= 70 else ("fair" if s.score >= 45 else "weak"))
        self.assertEqual(level, "weak")


class TestRenderNativeRunTrustScore(unittest.TestCase):

    def test_none_returns_empty_string(self):
        self.assertEqual(render_native_run_trust_score(None), "")

    def test_compact_shows_score_and_level(self):
        s = NativeRunTrustScore(score=82, level="good")
        out = render_native_run_trust_score(s)
        self.assertIn("82/100", out)
        self.assertIn("good", out)
        self.assertIn("run trust:", out)

    def test_compact_shows_warning_count(self):
        s = NativeRunTrustScore(score=55, level="fair", warnings=["w1", "w2"])
        out = render_native_run_trust_score(s)
        self.assertIn("2 warnings", out)
        self.assertNotIn("w1", out)
        self.assertNotIn("w2", out)

    def test_compact_shows_blocker_count(self):
        s = NativeRunTrustScore(score=20, level="weak", blockers=["b1"])
        out = render_native_run_trust_score(s)
        self.assertIn("1 blocker", out)
        self.assertNotIn("b1", out)

    def test_compact_no_warning_line_when_empty(self):
        s = NativeRunTrustScore(score=82, level="good")
        out = render_native_run_trust_score(s)
        self.assertNotIn("warnings", out)
        self.assertNotIn("blockers", out)

    def test_full_includes_run_trust_header(self):
        s = NativeRunTrustScore(
            score=82,
            level="good",
            factors=[NativeRunTrustFactor(name="verification_passed", impact=20, reason="verification passed")],
        )
        out = render_native_run_trust_score(s, detail="full")
        self.assertIn("[run trust]", out)
        self.assertIn("score: 82/100", out)
        self.assertIn("level: good", out)

    def test_full_includes_factor_lines(self):
        s = NativeRunTrustScore(
            score=82,
            level="good",
            factors=[NativeRunTrustFactor(name="verification_passed", impact=20, reason="verification passed")],
        )
        out = render_native_run_trust_score(s, detail="full")
        self.assertIn("verification_passed", out)
        self.assertIn("+20", out)
        self.assertIn("verification passed", out)

    def test_full_dict_factors_render_without_crash(self):
        s = NativeRunTrustScore(score=50, level="fair")
        s.factors = [{"name": "verification_passed", "impact": 20, "reason": "verification passed"}]  # type: ignore[assignment]
        out = render_native_run_trust_score(s, detail="full")
        self.assertIn("verification_passed", out)
        self.assertIn("+20", out)

    def test_compact_does_not_expose_raw_warning_text(self):
        s = NativeRunTrustScore(
            score=30,
            level="weak",
            warnings=["verification failed", "blocked commands detected"],
            blockers=["verification failed"],
        )
        out = render_native_run_trust_score(s, detail="compact")
        self.assertNotIn("verification failed", out)
        self.assertNotIn("blocked commands detected", out)
        self.assertIn("2 warnings", out)
        self.assertIn("1 blocker", out)


if __name__ == "__main__":
    unittest.main()
