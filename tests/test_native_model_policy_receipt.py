from __future__ import annotations

import json
import unittest
from dataclasses import asdict

from openshard.native.context import (
    NativeModelCandidateScoring,
    NativeModelPolicyReceipt,
    NativeModelRoleDecision,
    NativeModelSelectionDecision,
    build_native_model_policy,
    build_native_model_policy_receipt,
)


class TestNativeModelPolicyReceiptDefaults(unittest.TestCase):
    def setUp(self):
        self.receipt = NativeModelPolicyReceipt()

    def test_default_active(self):
        self.assertFalse(self.receipt.active)

    def test_default_mode(self):
        self.assertEqual(self.receipt.mode, "auto")

    def test_default_affected_selection(self):
        self.assertFalse(self.receipt.affected_selection)

    def test_default_blocked_count(self):
        self.assertEqual(self.receipt.blocked_count, 0)

    def test_default_changed_roles(self):
        self.assertEqual(self.receipt.changed_roles, [])

    def test_default_warnings_count(self):
        self.assertEqual(self.receipt.warnings_count, 0)

    def test_default_summary(self):
        self.assertEqual(self.receipt.summary, "")


class TestBuildNativeModelPolicyReceipt(unittest.TestCase):
    def test_all_none_inputs_no_crash(self):
        receipt = build_native_model_policy_receipt()
        self.assertFalse(receipt.active)
        self.assertEqual(receipt.summary, "policy inactive")

    def test_auto_policy_inactive(self):
        receipt = build_native_model_policy_receipt(
            model_policy=build_native_model_policy("auto"),
        )
        self.assertFalse(receipt.active)
        self.assertEqual(receipt.summary, "policy inactive")

    def test_non_auto_policy_active(self):
        receipt = build_native_model_policy_receipt(
            model_policy=build_native_model_policy("open-source-only"),
        )
        self.assertTrue(receipt.active)

    def test_blocked_count_from_candidate_scoring(self):
        mcs = NativeModelCandidateScoring(blocked_candidates=["a", "b", "c"])
        receipt = build_native_model_policy_receipt(
            model_policy=build_native_model_policy("open-source-only"),
            model_candidate_scoring=mcs,
        )
        self.assertEqual(receipt.blocked_count, 3)

    def test_changed_roles_detects_tier_change(self):
        before = NativeModelSelectionDecision(
            roles=[NativeModelRoleDecision(role="planner", model_tier="fast")]
        )
        after = NativeModelSelectionDecision(
            roles=[NativeModelRoleDecision(role="planner", model_tier="frontier")]
        )
        receipt = build_native_model_policy_receipt(
            model_policy=build_native_model_policy("open-source-only"),
            model_selection_decision_before=before,
            model_selection_decision_after=after,
        )
        self.assertEqual(receipt.changed_roles, ["planner"])

    def test_no_changed_roles_when_same(self):
        decision = NativeModelSelectionDecision(
            roles=[NativeModelRoleDecision(role="executor", model_tier="standard")]
        )
        receipt = build_native_model_policy_receipt(
            model_policy=build_native_model_policy("open-source-only"),
            model_selection_decision_before=decision,
            model_selection_decision_after=decision,
        )
        self.assertEqual(receipt.changed_roles, [])

    def test_affected_selection_true_when_blocked(self):
        mcs = NativeModelCandidateScoring(blocked_candidates=["x"])
        receipt = build_native_model_policy_receipt(
            model_policy=build_native_model_policy("open-source-only"),
            model_candidate_scoring=mcs,
        )
        self.assertTrue(receipt.affected_selection)

    def test_affected_selection_true_when_roles_changed(self):
        before = NativeModelSelectionDecision(
            roles=[NativeModelRoleDecision(role="executor", model_tier="fast")]
        )
        after = NativeModelSelectionDecision(
            roles=[NativeModelRoleDecision(role="executor", model_tier="standard")]
        )
        receipt = build_native_model_policy_receipt(
            model_policy=build_native_model_policy("cheapest-safe"),
            model_selection_decision_before=before,
            model_selection_decision_after=after,
        )
        self.assertTrue(receipt.affected_selection)

    def test_affected_selection_false_when_nothing_changed(self):
        receipt = build_native_model_policy_receipt(
            model_policy=build_native_model_policy("open-source-only"),
        )
        self.assertFalse(receipt.affected_selection)

    def test_summary_no_selection_changes(self):
        receipt = build_native_model_policy_receipt(
            model_policy=build_native_model_policy("open-source-only"),
        )
        self.assertEqual(receipt.summary, "policy active: no selection changes")

    def test_summary_blocked_and_changed(self):
        mcs = NativeModelCandidateScoring(blocked_candidates=["a", "b", "c"])
        before = NativeModelSelectionDecision(
            roles=[NativeModelRoleDecision(role="executor", model_tier="fast")]
        )
        after = NativeModelSelectionDecision(
            roles=[NativeModelRoleDecision(role="executor", model_tier="standard")]
        )
        receipt = build_native_model_policy_receipt(
            model_policy=build_native_model_policy("open-source-only"),
            model_selection_decision_before=before,
            model_selection_decision_after=after,
            model_candidate_scoring=mcs,
        )
        self.assertEqual(
            receipt.summary,
            "policy active: blocked 3 candidates and changed 1 role",
        )

    def test_summary_only_blocked(self):
        mcs = NativeModelCandidateScoring(blocked_candidates=["a"])
        receipt = build_native_model_policy_receipt(
            model_policy=build_native_model_policy("open-source-only"),
            model_candidate_scoring=mcs,
        )
        self.assertEqual(receipt.summary, "policy active: blocked 1 candidate")

    def test_summary_only_changed_roles(self):
        before = NativeModelSelectionDecision(
            roles=[NativeModelRoleDecision(role="planner", model_tier="fast")]
        )
        after = NativeModelSelectionDecision(
            roles=[NativeModelRoleDecision(role="planner", model_tier="frontier")]
        )
        receipt = build_native_model_policy_receipt(
            model_policy=build_native_model_policy("open-source-only"),
            model_selection_decision_before=before,
            model_selection_decision_after=after,
        )
        self.assertEqual(receipt.summary, "policy active: changed 1 role")

    def test_asdict_json_serializable(self):
        mcs = NativeModelCandidateScoring(blocked_candidates=["a", "b"])
        receipt = build_native_model_policy_receipt(
            model_policy=build_native_model_policy("open-source-only"),
            model_candidate_scoring=mcs,
        )
        try:
            json.dumps(asdict(receipt))
        except (TypeError, ValueError) as exc:
            self.fail(f"asdict(receipt) is not JSON-serializable: {exc}")

    def test_dict_policy_input(self):
        policy_dict = asdict(build_native_model_policy("cheapest-safe"))
        receipt = build_native_model_policy_receipt(model_policy=policy_dict)
        self.assertTrue(receipt.active)
        self.assertEqual(receipt.mode, "cheapest-safe")

    def test_warnings_count_combined(self):
        policy = build_native_model_policy("custom")
        mcs = NativeModelCandidateScoring(warnings=["w1", "w2"])
        receipt = build_native_model_policy_receipt(
            model_policy=policy,
            model_candidate_scoring=mcs,
        )
        self.assertEqual(receipt.warnings_count, len(policy.warnings) + 2)


if __name__ == "__main__":
    unittest.main()
