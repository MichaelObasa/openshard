from __future__ import annotations

import unittest

from openshard.policy.decision import (
    PolicyDecision,
    make_allow,
    make_ask,
    make_deny,
    resolve_policy_decisions,
)


class TestPolicyDecisionFactory(unittest.TestCase):

    def test_make_allow_fields(self):
        d = make_allow("read_file", resource="auth.py", reason="allowed", source="path_policy")
        self.assertEqual(d.decision, "allow")
        self.assertEqual(d.action, "read_file")
        self.assertEqual(d.resource, "auth.py")
        self.assertFalse(d.approval_required)
        self.assertIsNone(d.approval_granted)
        self.assertTrue(d.decision_id)

    def test_make_ask_fields(self):
        d = make_ask("write_file", resource="secrets.py", reason="requires approval", source="approval_gate")
        self.assertEqual(d.decision, "ask")
        self.assertTrue(d.approval_required)
        self.assertIsNone(d.approval_granted)
        self.assertEqual(d.resource, "secrets.py")

    def test_make_deny_fields_with_severity(self):
        d = make_deny("write_file", resource=".env", reason="protected path", source="path_policy", severity="critical")
        self.assertEqual(d.decision, "deny")
        self.assertEqual(d.severity, "critical")
        self.assertFalse(d.approval_required)

    def test_created_at_is_iso_timestamp(self):
        d = make_allow("act")
        self.assertIsNotNone(d.created_at)
        self.assertRegex(d.created_at, r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z")

    def test_reason_truncated_to_200_chars(self):
        d = make_allow("act", reason="x" * 300)
        self.assertEqual(len(d.reason), 200)

    def test_resource_truncated_to_200_chars(self):
        d = make_allow("act", resource="r" * 300)
        self.assertEqual(len(d.resource), 200)

    def test_reason_newlines_replaced(self):
        d = make_allow("act", reason="line1\nline2\r\nline3")
        self.assertNotIn("\n", d.reason)
        self.assertNotIn("\r", d.reason)

    def test_resource_newlines_replaced(self):
        d = make_deny("act", resource="path\nwith\nnewlines")
        self.assertNotIn("\n", d.resource)


class TestResolvePolicy(unittest.TestCase):

    def test_deny_beats_allow(self):
        decisions = [make_allow("a"), make_deny("b")]
        self.assertEqual(resolve_policy_decisions(decisions).decision, "deny")

    def test_deny_beats_ask_beats_allow(self):
        decisions = [make_allow("a"), make_ask("b"), make_deny("c")]
        self.assertEqual(resolve_policy_decisions(decisions).decision, "deny")

        ask_vs_allow = [make_allow("a"), make_ask("b")]
        self.assertEqual(resolve_policy_decisions(ask_vs_allow).decision, "ask")

    def test_not_applicable_is_ignored(self):
        na = PolicyDecision(decision_id="x", action="a", resource=None, decision="not_applicable")
        decisions = [na, make_allow("b")]
        self.assertEqual(resolve_policy_decisions(decisions).decision, "allow")

    def test_all_not_applicable_returns_not_applicable(self):
        na = PolicyDecision(decision_id="x", action="a", resource=None, decision="not_applicable")
        result = resolve_policy_decisions([na])
        self.assertEqual(result.decision, "not_applicable")

    def test_empty_list_returns_not_applicable(self):
        result = resolve_policy_decisions([])
        self.assertEqual(result.decision, "not_applicable")

    def test_multiple_denies_highest_severity_wins(self):
        decisions = [
            make_deny("a", severity="low"),
            make_deny("b", severity="critical"),
            make_deny("c", severity="medium"),
        ]
        result = resolve_policy_decisions(decisions)
        self.assertEqual(result.decision, "deny")
        self.assertEqual(result.severity, "critical")

    def test_allow_wins_when_all_allow(self):
        decisions = [make_allow("a"), make_allow("b")]
        self.assertEqual(resolve_policy_decisions(decisions).decision, "allow")

    def test_none_severity_is_lowest(self):
        decisions = [make_deny("a", severity=None), make_deny("b", severity="info")]
        result = resolve_policy_decisions(decisions)
        self.assertEqual(result.severity, "info")


class TestResolveTieBreak(unittest.TestCase):

    def test_tie_break_by_decision_id_lexicographic(self):
        d_aaa = make_deny("a", severity="high")
        d_bbb = make_deny("b", severity="high")
        object.__setattr__(d_aaa, "decision_id", "aaa") if False else None
        d_aaa = PolicyDecision(
            decision_id="aaa", action="a", resource=None, decision="deny", severity="high"
        )
        d_bbb = PolicyDecision(
            decision_id="bbb", action="b", resource=None, decision="deny", severity="high"
        )
        result = resolve_policy_decisions([d_bbb, d_aaa])
        self.assertEqual(result.decision_id, "aaa")

    def test_result_stable_regardless_of_input_order(self):
        d1 = PolicyDecision(decision_id="zzz", action="a", resource=None, decision="deny", severity="high")
        d2 = PolicyDecision(decision_id="aaa", action="b", resource=None, decision="deny", severity="high")
        d3 = PolicyDecision(decision_id="mmm", action="c", resource=None, decision="deny", severity="high")

        r1 = resolve_policy_decisions([d1, d2, d3])
        r2 = resolve_policy_decisions([d3, d1, d2])
        r3 = resolve_policy_decisions([d2, d3, d1])
        self.assertEqual(r1.decision_id, "aaa")
        self.assertEqual(r2.decision_id, "aaa")
        self.assertEqual(r3.decision_id, "aaa")


if __name__ == "__main__":
    unittest.main()
