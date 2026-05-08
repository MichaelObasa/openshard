from __future__ import annotations

import json
import unittest
from dataclasses import asdict

from openshard.native.context import (
    NativeChangeBudget,
    NativeChangeBudgetPreview,
    NativeChangeBudgetSoftGate,
    NativeClarificationRequest,
    NativeContextQualityScore,
    NativePlan,
    NativeValidationContract,
    NativeVerificationPlan,
    build_native_validation_contract,
    render_native_validation_contract,
)


def _build(
    task: str = "add a login feature",
    plan: NativePlan | None = None,
    verification_plan: NativeVerificationPlan | None = None,
    change_budget: NativeChangeBudget | None = None,
    change_budget_preview: NativeChangeBudgetPreview | None = None,
    change_budget_soft_gate: NativeChangeBudgetSoftGate | None = None,
    clarification_request: NativeClarificationRequest | None = None,
    context_quality_score: NativeContextQualityScore | None = None,
) -> NativeValidationContract:
    return build_native_validation_contract(
        task=task,
        plan=plan,
        verification_plan=verification_plan,
        change_budget=change_budget,
        change_budget_preview=change_budget_preview,
        change_budget_soft_gate=change_budget_soft_gate,
        clarification_request=clarification_request,
        context_quality_score=context_quality_score,
    )


def _vplan(**kwargs) -> NativeVerificationPlan:
    defaults: dict = dict(
        task_type="feature",
        risk_level="low",
        likely_files_or_folders=[],
        allowed_commands=[],
        blocked_commands=[],
        suggested_verification_commands=[],
        approval_rules=[],
        success_criteria=[],
        failure_handling="halt and report",
        clarification_needed=[],
        warnings=[],
    )
    defaults.update(kwargs)
    return NativeVerificationPlan(**defaults)


class TestNativeValidationContractDataclass(unittest.TestCase):

    def test_dataclass_defaults(self):
        vc = NativeValidationContract()
        self.assertEqual(vc.intent, "")
        self.assertEqual(vc.risk_level, "unknown")
        self.assertEqual(vc.expected_change_scope, "unknown")
        self.assertEqual(vc.acceptance_checks, [])
        self.assertEqual(vc.verification_commands, [])
        self.assertFalse(vc.approval_expected)
        self.assertEqual(vc.strength, "weak")
        self.assertEqual(vc.warnings, [])

    def test_json_serializable_via_asdict(self):
        vc = NativeValidationContract(
            intent="add feature",
            risk_level="low",
            expected_change_scope="3 files expected",
            acceptance_checks=["verification passed"],
            verification_commands=["pytest"],
            approval_expected=False,
            strength="strong",
            warnings=[],
        )
        d = asdict(vc)
        json.dumps(d)
        self.assertEqual(d["intent"], "add feature")
        self.assertEqual(d["strength"], "strong")
        self.assertIsInstance(d["acceptance_checks"], list)
        self.assertIsInstance(d["verification_commands"], list)
        self.assertIsInstance(d["warnings"], list)


class TestBuildNativeValidationContract(unittest.TestCase):

    def test_all_none_inputs_returns_valid_weak_contract(self):
        vc = _build(task="", plan=None, verification_plan=None)
        self.assertIsInstance(vc, NativeValidationContract)
        self.assertEqual(vc.strength, "weak")
        self.assertEqual(vc.intent, "")

    def test_intent_stripped(self):
        vc = _build(task="  add feature  ")
        self.assertEqual(vc.intent, "add feature")

    def test_intent_capped_at_120_chars(self):
        long_task = "x" * 200
        vc = _build(task=long_task)
        self.assertEqual(len(vc.intent), 120)
        self.assertEqual(vc.intent, "x" * 120)

    def test_intent_exact_120_chars_preserved(self):
        task = "a" * 120
        vc = _build(task=task)
        self.assertEqual(vc.intent, task)

    def test_intent_shorter_than_120_preserved(self):
        vc = _build(task="short task")
        self.assertEqual(vc.intent, "short task")

    def test_risk_level_from_plan(self):
        plan = NativePlan(risk="high")
        vc = _build(plan=plan)
        self.assertEqual(vc.risk_level, "high")

    def test_risk_level_from_verification_plan_when_no_plan(self):
        vp = _vplan(risk_level="medium")
        vc = _build(plan=None, verification_plan=vp)
        self.assertEqual(vc.risk_level, "medium")

    def test_risk_level_plan_takes_precedence_over_verification_plan(self):
        plan = NativePlan(risk="low")
        vp = _vplan(risk_level="high")
        vc = _build(plan=plan, verification_plan=vp)
        self.assertEqual(vc.risk_level, "low")

    def test_risk_level_unknown_when_no_inputs(self):
        vc = _build(plan=None, verification_plan=None)
        self.assertEqual(vc.risk_level, "unknown")

    def test_expected_change_scope_from_change_budget(self):
        budget = NativeChangeBudget(max_files=3)
        vc = _build(change_budget=budget)
        self.assertEqual(vc.expected_change_scope, "3 files expected")

    def test_expected_change_scope_from_budget_preview_when_no_budget(self):
        preview = NativeChangeBudgetPreview(proposed_files=2)
        vc = _build(change_budget=None, change_budget_preview=preview)
        self.assertEqual(vc.expected_change_scope, "2 files proposed")

    def test_expected_change_scope_budget_takes_precedence_over_preview(self):
        budget = NativeChangeBudget(max_files=5)
        preview = NativeChangeBudgetPreview(proposed_files=2)
        vc = _build(change_budget=budget, change_budget_preview=preview)
        self.assertIn("5 files", vc.expected_change_scope)

    def test_expected_change_scope_unknown_when_no_budget_or_preview(self):
        vc = _build(change_budget=None, change_budget_preview=None)
        self.assertEqual(vc.expected_change_scope, "unknown")

    def test_acceptance_checks_from_success_criteria(self):
        vp = _vplan(success_criteria=["verification passed", "files within budget"])
        vc = _build(verification_plan=vp)
        self.assertIn("verification passed", vc.acceptance_checks)
        self.assertIn("files within budget", vc.acceptance_checks)

    def test_acceptance_checks_appends_plan_steps(self):
        plan = NativePlan(suggested_steps=["step one", "step two"])
        vc = _build(plan=plan, verification_plan=None)
        self.assertIn("step one", vc.acceptance_checks)
        self.assertIn("step two", vc.acceptance_checks)

    def test_acceptance_checks_plan_steps_capped_at_3(self):
        plan = NativePlan(suggested_steps=["s1", "s2", "s3", "s4", "s5"])
        vc = _build(plan=plan, verification_plan=None)
        plan_steps_in_checks = [c for c in vc.acceptance_checks if c in ["s1", "s2", "s3", "s4", "s5"]]
        self.assertLessEqual(len(plan_steps_in_checks), 3)

    def test_acceptance_checks_deduplicated(self):
        vp = _vplan(success_criteria=["verification passed"])
        plan = NativePlan(suggested_steps=["verification passed", "step two"])
        vc = _build(plan=plan, verification_plan=vp)
        self.assertEqual(vc.acceptance_checks.count("verification passed"), 1)

    def test_verification_commands_copied_from_verification_plan(self):
        vp = _vplan(suggested_verification_commands=["pytest", "npm test"])
        vc = _build(verification_plan=vp)
        self.assertEqual(vc.verification_commands, ["pytest", "npm test"])

    def test_verification_commands_capped_at_5(self):
        vp = _vplan(suggested_verification_commands=["c1", "c2", "c3", "c4", "c5", "c6"])
        vc = _build(verification_plan=vp)
        self.assertEqual(len(vc.verification_commands), 5)
        self.assertNotIn("c6", vc.verification_commands)

    def test_verification_commands_empty_when_no_verification_plan(self):
        vc = _build(verification_plan=None)
        self.assertEqual(vc.verification_commands, [])

    def test_approval_expected_false_by_default(self):
        vc = _build()
        self.assertFalse(vc.approval_expected)

    def test_approval_expected_true_when_soft_gate_requires_approval(self):
        gate = NativeChangeBudgetSoftGate(requires_approval=True)
        vc = _build(change_budget_soft_gate=gate)
        self.assertTrue(vc.approval_expected)

    def test_approval_expected_true_when_budget_preview_would_exceed(self):
        preview = NativeChangeBudgetPreview(would_exceed_budget=True)
        vc = _build(change_budget_preview=preview)
        self.assertTrue(vc.approval_expected)

    def test_approval_expected_false_when_soft_gate_does_not_require(self):
        gate = NativeChangeBudgetSoftGate(requires_approval=False)
        vc = _build(change_budget_soft_gate=gate)
        self.assertFalse(vc.approval_expected)

    def test_strength_strong_when_checks_and_commands_and_no_clarification(self):
        vp = _vplan(
            success_criteria=["verification passed"],
            suggested_verification_commands=["pytest"],
        )
        cr = NativeClarificationRequest(needed=False)
        vc = _build(verification_plan=vp, clarification_request=cr)
        self.assertEqual(vc.strength, "strong")

    def test_strength_strong_when_clarification_is_none(self):
        vp = _vplan(
            success_criteria=["verification passed"],
            suggested_verification_commands=["pytest"],
        )
        vc = _build(verification_plan=vp, clarification_request=None)
        self.assertEqual(vc.strength, "strong")

    def test_strength_fair_when_checks_but_no_commands(self):
        vp = _vplan(
            success_criteria=["verification passed"],
            suggested_verification_commands=[],
        )
        cr = NativeClarificationRequest(needed=False)
        vc = _build(verification_plan=vp, clarification_request=cr)
        self.assertEqual(vc.strength, "fair")

    def test_strength_fair_when_checks_and_clarification_needed(self):
        vp = _vplan(
            success_criteria=["verification passed"],
            suggested_verification_commands=["pytest"],
        )
        cr = NativeClarificationRequest(needed=True)
        vc = _build(verification_plan=vp, clarification_request=cr)
        self.assertEqual(vc.strength, "fair")

    def test_strength_weak_when_no_checks(self):
        vp = _vplan(success_criteria=[], suggested_verification_commands=["pytest"])
        cr = NativeClarificationRequest(needed=False)
        vc = _build(verification_plan=vp, plan=None, clarification_request=cr)
        self.assertEqual(vc.strength, "weak")

    def test_strength_weak_all_none_inputs(self):
        vc = _build(plan=None, verification_plan=None, clarification_request=None)
        self.assertEqual(vc.strength, "weak")

    def test_warnings_include_clarification_message_when_needed(self):
        cr = NativeClarificationRequest(needed=True)
        vc = _build(clarification_request=cr)
        self.assertIn("clarification needed before proceeding", vc.warnings)

    def test_warnings_include_no_checks_when_none_derived(self):
        vc = _build(plan=None, verification_plan=None)
        self.assertIn("no acceptance checks derived from plan", vc.warnings)

    def test_warnings_include_no_commands_when_missing(self):
        vc = _build(verification_plan=None)
        self.assertIn("no verification commands available", vc.warnings)

    def test_no_clarification_warning_when_not_needed(self):
        cr = NativeClarificationRequest(needed=False)
        vc = _build(clarification_request=cr)
        self.assertNotIn("clarification needed before proceeding", vc.warnings)

    def test_no_checks_warning_absent_when_checks_present(self):
        vp = _vplan(success_criteria=["verification passed"])
        vc = _build(verification_plan=vp)
        self.assertNotIn("no acceptance checks derived from plan", vc.warnings)

    def test_no_commands_warning_absent_when_commands_present(self):
        vp = _vplan(suggested_verification_commands=["pytest"])
        vc = _build(verification_plan=vp)
        self.assertNotIn("no verification commands available", vc.warnings)


class TestRenderNativeValidationContract(unittest.TestCase):

    def test_render_none_returns_empty_string(self):
        self.assertEqual(render_native_validation_contract(None), "")

    def test_render_includes_header(self):
        vc = NativeValidationContract()
        out = render_native_validation_contract(vc)
        self.assertIn("[validation contract]", out)

    def test_render_includes_intent(self):
        vc = NativeValidationContract(intent="add login feature")
        out = render_native_validation_contract(vc)
        self.assertIn("intent: add login feature", out)

    def test_render_includes_risk(self):
        vc = NativeValidationContract(risk_level="high")
        out = render_native_validation_contract(vc)
        self.assertIn("risk: high", out)

    def test_render_includes_scope(self):
        vc = NativeValidationContract(expected_change_scope="3 files expected")
        out = render_native_validation_contract(vc)
        self.assertIn("scope: 3 files expected", out)

    def test_render_includes_strength(self):
        vc = NativeValidationContract(strength="strong")
        out = render_native_validation_contract(vc)
        self.assertIn("strength: strong", out)

    def test_render_approval_expected_yes(self):
        vc = NativeValidationContract(approval_expected=True)
        out = render_native_validation_contract(vc)
        self.assertIn("approval expected: yes", out)

    def test_render_approval_expected_no(self):
        vc = NativeValidationContract(approval_expected=False)
        out = render_native_validation_contract(vc)
        self.assertIn("approval expected: no", out)

    def test_render_includes_acceptance_checks_section(self):
        vc = NativeValidationContract(acceptance_checks=["verification passed", "files within budget"])
        out = render_native_validation_contract(vc)
        self.assertIn("acceptance checks:", out)
        self.assertIn("  - verification passed", out)
        self.assertIn("  - files within budget", out)

    def test_render_omits_acceptance_checks_section_when_empty(self):
        vc = NativeValidationContract(acceptance_checks=[])
        out = render_native_validation_contract(vc)
        self.assertNotIn("acceptance checks:", out)

    def test_render_includes_verification_section(self):
        vc = NativeValidationContract(verification_commands=["pytest", "npm test"])
        out = render_native_validation_contract(vc)
        self.assertIn("verification:", out)
        self.assertIn("  - pytest", out)
        self.assertIn("  - npm test", out)

    def test_render_omits_verification_section_when_empty(self):
        vc = NativeValidationContract(verification_commands=[])
        out = render_native_validation_contract(vc)
        self.assertNotIn("verification:", out)

    def test_render_does_not_include_warnings(self):
        vc = NativeValidationContract(warnings=["UNIQUE_WARNING_NOT_IN_OUTPUT"])
        out = render_native_validation_contract(vc)
        self.assertNotIn("UNIQUE_WARNING_NOT_IN_OUTPUT", out)

    def test_render_no_raw_unsafe_content(self):
        vc = NativeValidationContract(
            intent="safe task",
            warnings=["RAW_UNSAFE_CONTENT"],
        )
        out = render_native_validation_contract(vc)
        self.assertNotIn("RAW_UNSAFE_CONTENT", out)


if __name__ == "__main__":
    unittest.main()
