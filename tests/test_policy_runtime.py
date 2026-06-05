from __future__ import annotations

import json
import unittest

from openshard.policy.runtime import (
    _dedup_decisions,
    build_runtime_policy_decisions,
)


def _approval_request(requires_approval: bool = True, action: str = "write") -> dict:
    return {
        "source": "approval_gate",
        "requires_approval": requires_approval,
        "reason": "budget exceeded",
        "action": action,
        "proposed_files": 3,
        "budget_max_files": 2,
        "prompt": "",
        "warnings": [],
    }


def _approval_receipt(granted: bool, action: str = "write", reason: str = "") -> dict:
    return {
        "source": "approval_gate",
        "requested": True,
        "granted": granted,
        "action": action,
        "reason": reason,
    }


def _secret_scan_result(findings_count: int = 1) -> dict:
    findings = [
        {
            "kind": "anthropic_key",
            "path": "config.py",
            "line": 10,
            "redacted": "sk-ant-...XXXX",
            "severity": "Critical",
            "fingerprint": "abc123def456",
        }
    ] * findings_count
    return {
        "scanned_files_count": 5,
        "findings": findings,
        "blocked": False,
        "summary": f"{findings_count} finding(s)",
    }


class TestApprovalDecisions(unittest.TestCase):

    def test_approval_request_without_receipt_records_ask(self):
        result = build_runtime_policy_decisions(
            approval_request=_approval_request(),
        )
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["decision"], "ask")
        self.assertEqual(result[0]["source"], "approval_gate")
        self.assertTrue(result[0]["approval_required"])
        self.assertIsNone(result[0]["approval_granted"])

    def test_granted_approval_receipt_records_allow(self):
        result = build_runtime_policy_decisions(
            approval_request=_approval_request(),
            approval_receipt=_approval_receipt(granted=True, reason="user approved"),
        )
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["decision"], "allow")
        self.assertEqual(result[0]["source"], "approval_gate")
        self.assertTrue(result[0]["approval_required"])
        self.assertTrue(result[0]["approval_granted"])
        self.assertEqual(result[0]["reason"], "user approved")

    def test_denied_approval_receipt_records_deny(self):
        result = build_runtime_policy_decisions(
            approval_request=_approval_request(),
            approval_receipt=_approval_receipt(granted=False, reason="user denied"),
        )
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["decision"], "deny")
        self.assertEqual(result[0]["source"], "approval_gate")
        self.assertEqual(result[0]["reason"], "user denied")

    def test_denied_approval_has_severity_high(self):
        result = build_runtime_policy_decisions(
            approval_request=_approval_request(),
            approval_receipt=_approval_receipt(granted=False),
        )
        self.assertEqual(result[0]["severity"], "high")
        self.assertFalse(result[0]["approval_granted"])

    def test_no_approval_decision_when_requires_approval_false(self):
        result = build_runtime_policy_decisions(
            approval_request=_approval_request(requires_approval=False),
            approval_receipt=_approval_receipt(granted=True),
        )
        approval_decisions = [d for d in result if d.get("source") == "approval_gate"]
        self.assertEqual(len(approval_decisions), 0)


class TestReadonlyDecision(unittest.TestCase):

    def test_readonly_records_allow_decision(self):
        result = build_runtime_policy_decisions(readonly=True)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["decision"], "allow")
        self.assertEqual(result[0]["action"], "read_only_review")
        self.assertEqual(result[0]["source"], "path_policy")
        self.assertFalse(result[0]["approval_required"])

    def test_readonly_suppressed_when_approval_required(self):
        result = build_runtime_policy_decisions(
            approval_request=_approval_request(requires_approval=True),
            readonly=True,
        )
        ro_decisions = [d for d in result if d.get("action") == "read_only_review"]
        self.assertEqual(len(ro_decisions), 0)

    def test_no_readonly_decision_when_false(self):
        result = build_runtime_policy_decisions(readonly=False)
        self.assertEqual(result, [])

    def test_no_readonly_decision_when_none(self):
        result = build_runtime_policy_decisions(readonly=None)
        self.assertEqual(result, [])


class TestSecretScanDecision(unittest.TestCase):

    def test_secret_scan_findings_records_not_applicable(self):
        result = build_runtime_policy_decisions(
            secret_scan_result=_secret_scan_result(findings_count=1),
        )
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["decision"], "not_applicable")
        self.assertEqual(result[0]["action"], "secret_scan_review")
        self.assertEqual(result[0]["source"], "secret_scan")
        self.assertEqual(result[0]["severity"], "high")
        self.assertFalse(result[0]["approval_required"])

    def test_secret_scan_decision_excludes_raw_secret(self):
        result = build_runtime_policy_decisions(
            secret_scan_result=_secret_scan_result(findings_count=2),
        )
        serialised = json.dumps(result)
        self.assertNotIn("sk-ant-", serialised)
        self.assertNotIn("redacted", serialised)
        self.assertNotIn("fingerprint", serialised)
        self.assertNotIn("abc123", serialised)

    def test_no_secret_scan_decision_when_empty_findings(self):
        result = build_runtime_policy_decisions(
            secret_scan_result={"scanned_files_count": 3, "findings": [], "blocked": False, "summary": "clean"},
        )
        self.assertEqual(result, [])

    def test_no_secret_scan_decision_when_result_none(self):
        result = build_runtime_policy_decisions(secret_scan_result=None)
        self.assertEqual(result, [])


class TestValidatorPolicyDecision(unittest.TestCase):

    def test_validator_skipped_records_not_applicable(self):
        result = build_runtime_policy_decisions(
            validator_policy={"run": False, "reason": "read-only task"},
        )
        vp = [d for d in result if d.get("source") == "validator"]
        self.assertEqual(len(vp), 1)
        self.assertEqual(vp[0]["decision"], "not_applicable")
        self.assertIn("read-only task", vp[0]["reason"])

    def test_validator_ran_records_allow(self):
        result = build_runtime_policy_decisions(
            validator_policy={"run": True, "reason": "security category"},
        )
        vp = [d for d in result if d.get("source") == "validator"]
        self.assertEqual(len(vp), 1)
        self.assertEqual(vp[0]["decision"], "allow")

    def test_no_validator_decision_when_none(self):
        result = build_runtime_policy_decisions(validator_policy=None)
        self.assertEqual(result, [])


class TestDeduplication(unittest.TestCase):

    def test_dedup_prevents_duplicate_decisions(self):
        existing = [
            {"source": "approval_gate", "action": "write", "decision": "allow", "reason": "approval granted"}
        ]
        new = [
            {"source": "approval_gate", "action": "write", "decision": "allow", "reason": "approval granted"},
            {"source": "secret_scan", "action": "secret_scan_review", "decision": "not_applicable", "reason": "warning"},
        ]
        result = _dedup_decisions(existing, new)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["source"], "secret_scan")

    def test_dedup_empty_existing_keeps_all_new(self):
        new = [
            {"source": "approval_gate", "action": "write", "decision": "allow", "reason": "ok"},
            {"source": "validator", "action": "validator_review", "decision": "not_applicable", "reason": "skipped"},
        ]
        result = _dedup_decisions([], new)
        self.assertEqual(len(result), 2)


class TestSafety(unittest.TestCase):

    def test_helper_returns_json_safe_dicts(self):
        result = build_runtime_policy_decisions(
            approval_request=_approval_request(),
            approval_receipt=_approval_receipt(granted=True, reason="ok"),
            secret_scan_result=_secret_scan_result(),
            validator_policy={"run": False, "reason": "dry run"},
            readonly=False,
        )
        serialised = json.dumps(result)
        self.assertIsInstance(serialised, str)

    def test_invalid_metadata_ignored_safely(self):
        self.assertEqual(build_runtime_policy_decisions(), [])
        self.assertEqual(build_runtime_policy_decisions(approval_request="not-a-dict"), [])
        self.assertEqual(build_runtime_policy_decisions(approval_receipt=42), [])
        self.assertEqual(build_runtime_policy_decisions(secret_scan_result=[]), [])
        self.assertEqual(build_runtime_policy_decisions(validator_policy="oops"), [])

    def test_existing_policy_decisions_preserved_on_append(self):
        from openshard.history.shard_contract import build_shard_receipt

        pre_existing = [
            {
                "decision_id": "pre-existing-1",
                "action": "custom_check",
                "resource": None,
                "decision": "allow",
                "reason": "custom policy",
            }
        ]
        runtime_pds = build_runtime_policy_decisions(
            approval_request=_approval_request(),
            approval_receipt=_approval_receipt(granted=True),
        )

        combined = pre_existing + _dedup_decisions(pre_existing, runtime_pds)
        entry = {
            "task": "test",
            "status": "completed",
            "timestamp": "2026-01-01T00:00:00Z",
            "policy_decisions": combined,
        }
        receipt = build_shard_receipt(entry)
        decision_ids = [d["decision_id"] for d in receipt.policy_decisions]
        self.assertIn("pre-existing-1", decision_ids)
        self.assertEqual(len(receipt.policy_decisions), 2)


class TestReceiptRendering(unittest.TestCase):

    def _make_entry_with_runtime_pds(self) -> dict:
        pds = build_runtime_policy_decisions(
            approval_request=_approval_request(),
            approval_receipt=_approval_receipt(granted=False, reason="budget exceeded"),
        )
        return {
            "task": "deploy infra",
            "status": "completed",
            "timestamp": "2026-01-01T00:00:00Z",
            "policy_decisions": pds,
        }

    def test_full_receipt_renders_runtime_policy_decisions(self):
        from openshard.history.shard_contract import build_shard_receipt, render_full_shard_receipt

        entry = self._make_entry_with_runtime_pds()
        receipt = build_shard_receipt(entry)
        out = render_full_shard_receipt(receipt)
        self.assertIn("POLICY DECISIONS", out)
        self.assertIn("deny", out)
        self.assertIn("budget exceeded", out)

    def test_compact_receipt_unchanged_with_runtime_policy_decisions(self):
        from openshard.history.shard_contract import (
            build_shard_receipt,
            render_compact_shard_receipt,
        )

        entry = self._make_entry_with_runtime_pds()
        receipt = build_shard_receipt(entry)
        out = render_compact_shard_receipt(receipt)
        self.assertNotIn("POLICY DECISIONS", out)

    def test_old_receipt_without_policy_decisions_renders_safely(self):
        from openshard.history.shard_contract import build_shard_receipt, render_full_shard_receipt

        entry = {
            "task": "old task",
            "status": "completed",
            "timestamp": "2025-01-01T00:00:00Z",
        }
        receipt = build_shard_receipt(entry)
        out = render_full_shard_receipt(receipt)
        self.assertNotIn("POLICY DECISIONS", out)


if __name__ == "__main__":
    unittest.main()
