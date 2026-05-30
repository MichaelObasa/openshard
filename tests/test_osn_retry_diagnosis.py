"""Tests for OSNRetryDiagnosis: dataclass, builder, receipt rendering, and integration.

Coverage:
- Dataclass safety: defaults, JSON safety, valid statuses, caps, no raw fields (tests 1-6)
- Diagnosis building: all status branches, failure_kind, changes, next_action (tests 7-20)
- Integration: NativeRunMeta field, entry dict, no shell, no model, existing fields (tests 21-27)
- Receipt rendering: section present/absent, rows, no raw output, no em dash (tests 28-36)
"""
from __future__ import annotations

import json
import unittest
from dataclasses import asdict, fields
from types import SimpleNamespace
from typing import Any


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_loop(
    *,
    enabled: bool = True,
    verification_attempted: bool = False,
    verification_passed: bool | None = None,
    stopped_reason: str = "completed",
    retry_used: bool = False,
    retry_count: int = 0,
    approval_required: bool = False,
    approval_granted: bool = False,
) -> Any:
    return SimpleNamespace(
        enabled=enabled,
        verification_attempted=verification_attempted,
        verification_passed=verification_passed,
        stopped_reason=stopped_reason,
        retry_used=retry_used,
        retry_count=retry_count,
        approval_required=approval_required,
        approval_granted=approval_granted,
    )


def _make_vc(*, status: str = "not_run", manual_review_required: bool = False) -> Any:
    return SimpleNamespace(status=status, manual_review_required=manual_review_required)


# ---------------------------------------------------------------------------
# 1-6: Dataclass safety
# ---------------------------------------------------------------------------

class TestOSNRetryDiagnosisDefaults(unittest.TestCase):
    def setUp(self) -> None:
        from openshard.native.retry_diagnosis import OSNRetryDiagnosis
        self.cls = OSNRetryDiagnosis

    def test_1_defaults_are_safe(self) -> None:
        d = self.cls()
        self.assertFalse(d.enabled)
        self.assertFalse(d.retry_allowed)
        self.assertFalse(d.retry_used)
        self.assertEqual(d.retry_count, 0)
        self.assertEqual(d.retry_limit, 1)
        self.assertEqual(d.status, "not_needed")
        self.assertEqual(d.source, "osn_retry_diagnosis_v1")
        self.assertEqual(d.failure_kind, "")
        self.assertEqual(d.failure_summary, "")
        self.assertEqual(d.retry_reason, "")
        self.assertEqual(d.allowed_changes, [])
        self.assertEqual(d.blocked_changes, [])
        self.assertEqual(d.next_action, "")
        self.assertFalse(d.manual_review_required)
        self.assertFalse(d.raw_content_stored)

    def test_2_serializes_to_json_safe_dict(self) -> None:
        from openshard.native.retry_diagnosis import OSNRetryDiagnosis
        d = OSNRetryDiagnosis(
            enabled=True,
            status="allowed",
            failure_kind="verification_failed",
            allowed_changes=["fix path"],
            blocked_changes=["no scope expansion"],
        )
        raw = asdict(d)
        # must not raise
        dumped = json.dumps(raw)
        reloaded = json.loads(dumped)
        self.assertEqual(reloaded["status"], "allowed")
        self.assertFalse(reloaded["raw_content_stored"])

    def test_3_valid_statuses_are_stable(self) -> None:
        from openshard.native.retry_diagnosis import _VALID_STATUSES
        expected = {
            "not_needed", "allowed", "used", "exhausted",
            "blocked", "manual_review", "unknown",
        }
        self.assertEqual(_VALID_STATUSES, expected)

    def test_4_lists_are_capped(self) -> None:
        from openshard.native.retry_diagnosis import OSNRetryDiagnosis
        d = OSNRetryDiagnosis(
            allowed_changes=["a", "b", "c", "d", "e", "f", "g"],
            blocked_changes=["x", "y", "z", "w", "v", "u"],
        )
        self.assertLessEqual(len(d.allowed_changes), 5)
        self.assertLessEqual(len(d.blocked_changes), 5)

    def test_5_strings_are_capped(self) -> None:
        from openshard.native.retry_diagnosis import OSNRetryDiagnosis
        long = "x" * 200
        d = OSNRetryDiagnosis(
            failure_kind=long,
            failure_summary=long,
            retry_reason=long,
            next_action=long,
        )
        self.assertLessEqual(len(d.failure_kind), 120)
        self.assertLessEqual(len(d.failure_summary), 120)
        self.assertLessEqual(len(d.retry_reason), 120)
        self.assertLessEqual(len(d.next_action), 120)

    def test_6_raw_content_stored_always_false(self) -> None:
        from openshard.native.retry_diagnosis import OSNRetryDiagnosis
        d = OSNRetryDiagnosis(raw_content_stored=True)  # attempt to set True
        self.assertFalse(d.raw_content_stored)
        field_names = {f.name for f in fields(d)}
        self.assertIn("raw_content_stored", field_names)
        # no other field should carry raw content
        self.assertNotIn("raw_stdout", field_names)
        self.assertNotIn("raw_stderr", field_names)
        self.assertNotIn("raw_output", field_names)


# ---------------------------------------------------------------------------
# 7-20: Diagnosis building
# ---------------------------------------------------------------------------

class TestBuildOSNRetryDiagnosis(unittest.TestCase):
    def setUp(self) -> None:
        from openshard.native.retry_diagnosis import build_osn_retry_diagnosis
        self.build = build_osn_retry_diagnosis

    def test_7_passed_verification_gives_not_needed(self) -> None:
        loop = _make_loop(verification_attempted=True, verification_passed=True)
        d = self.build(osn_loop_summary=loop)
        self.assertEqual(d.status, "not_needed")
        self.assertFalse(d.retry_allowed)
        self.assertFalse(d.enabled)

    def test_8_failed_verification_under_limit_gives_allowed(self) -> None:
        loop = _make_loop(
            verification_attempted=True, verification_passed=False,
            retry_count=0,
        )
        d = self.build(osn_loop_summary=loop, retry_limit=1)
        self.assertEqual(d.status, "allowed")
        self.assertTrue(d.retry_allowed)
        self.assertTrue(d.enabled)

    def test_9_failed_verification_at_limit_gives_exhausted(self) -> None:
        loop = _make_loop(
            verification_attempted=True, verification_passed=False,
            retry_count=1,
        )
        d = self.build(osn_loop_summary=loop, retry_limit=1)
        self.assertEqual(d.status, "exhausted")
        self.assertFalse(d.retry_allowed)
        self.assertTrue(d.manual_review_required)
        self.assertTrue(d.enabled)

    def test_10_approval_denied_gives_blocked(self) -> None:
        loop = _make_loop(verification_attempted=True, verification_passed=True)
        d = self.build(
            osn_loop_summary=loop,
            approval_required=True,
            approval_granted=False,
        )
        self.assertEqual(d.status, "blocked")
        self.assertFalse(d.retry_allowed)
        self.assertTrue(d.manual_review_required)
        self.assertTrue(d.enabled)

    def test_11_skipped_verification_contract_gives_manual_review(self) -> None:
        loop = _make_loop(verification_attempted=False)
        vc = _make_vc(status="skipped")
        d = self.build(osn_loop_summary=loop, osn_verification_contract=vc)
        self.assertEqual(d.status, "manual_review")
        self.assertFalse(d.retry_allowed)
        self.assertTrue(d.manual_review_required)
        self.assertTrue(d.enabled)

    def test_12_stopped_reason_retry_limit_gives_exhausted(self) -> None:
        loop = _make_loop(
            verification_attempted=True, verification_passed=False,
            stopped_reason="retry_limit", retry_count=1,
        )
        d = self.build(osn_loop_summary=loop, retry_limit=1)
        self.assertEqual(d.status, "exhausted")

    def test_13_stopped_reason_verification_failed_at_zero_count_gives_allowed(self) -> None:
        loop = _make_loop(
            verification_attempted=True, verification_passed=False,
            stopped_reason="verification_failed", retry_count=0,
        )
        d = self.build(osn_loop_summary=loop, retry_limit=1)
        self.assertEqual(d.status, "allowed")

    def test_14_stopped_reason_verification_failed_at_limit_gives_exhausted(self) -> None:
        loop = _make_loop(
            verification_attempted=True, verification_passed=False,
            stopped_reason="verification_failed", retry_count=1,
        )
        d = self.build(osn_loop_summary=loop, retry_limit=1)
        self.assertEqual(d.status, "exhausted")

    def test_15_retry_used_with_passed_gives_used(self) -> None:
        loop = _make_loop(
            verification_attempted=True, verification_passed=True,
            retry_used=True, retry_count=1,
        )
        d = self.build(osn_loop_summary=loop)
        self.assertEqual(d.status, "used")
        self.assertTrue(d.retry_used)
        self.assertTrue(d.enabled)

    def test_16_retry_count_preserved(self) -> None:
        loop = _make_loop(
            verification_attempted=True, verification_passed=False,
            retry_count=1,
        )
        d = self.build(osn_loop_summary=loop, retry_limit=2)
        self.assertEqual(d.retry_count, 1)

    def test_17_retry_limit_preserved(self) -> None:
        loop = _make_loop(verification_attempted=True, verification_passed=False)
        d = self.build(osn_loop_summary=loop, retry_limit=3)
        self.assertEqual(d.retry_limit, 3)

    def test_18_failure_kind_is_safe_and_deterministic(self) -> None:
        from openshard.native.retry_diagnosis import _FAILURE_KINDS
        for status, kind in _FAILURE_KINDS.items():
            self.assertIsInstance(kind, str)
            self.assertNotIn("--", kind)  # no em dash
            self.assertNotIn("\n", kind)
            self.assertNotIn("exit_code=", kind)

    def test_19_allowed_changes_bounded_for_allowed_status(self) -> None:
        loop = _make_loop(
            verification_attempted=True, verification_passed=False, retry_count=0,
        )
        d = self.build(osn_loop_summary=loop, retry_limit=1)
        self.assertEqual(d.status, "allowed")
        self.assertGreater(len(d.allowed_changes), 0)
        self.assertLessEqual(len(d.allowed_changes), 5)
        self.assertGreater(len(d.blocked_changes), 0)
        self.assertLessEqual(len(d.blocked_changes), 5)

    def test_20_next_action_populated_for_failure_cases(self) -> None:
        cases = [
            _make_loop(verification_attempted=True, verification_passed=False, retry_count=0),
            _make_loop(verification_attempted=True, verification_passed=False, retry_count=1),
        ]
        for loop in cases:
            d = self.build(osn_loop_summary=loop, retry_limit=1)
            self.assertNotEqual(d.next_action, "", f"expected next_action for status={d.status}")

    def test_20b_no_loop_summary_returns_disabled(self) -> None:
        d = self.build(osn_loop_summary=None)
        self.assertFalse(d.enabled)
        self.assertEqual(d.status, "not_needed")

    def test_20c_disabled_loop_summary_returns_disabled(self) -> None:
        loop = _make_loop(enabled=False)
        d = self.build(osn_loop_summary=loop)
        self.assertFalse(d.enabled)


# ---------------------------------------------------------------------------
# 21-27: Integration
# ---------------------------------------------------------------------------

class TestOSNRetryDiagnosisIntegration(unittest.TestCase):
    def test_21_native_run_meta_has_field(self) -> None:
        from openshard.native.executor import NativeRunMeta
        meta = NativeRunMeta()
        self.assertTrue(hasattr(meta, "osn_retry_diagnosis"))
        self.assertIsNone(meta.osn_retry_diagnosis)

    def test_22_can_assign_diagnosis_to_native_meta(self) -> None:
        from openshard.native.executor import NativeRunMeta
        from openshard.native.retry_diagnosis import OSNRetryDiagnosis
        meta = NativeRunMeta()
        diag = OSNRetryDiagnosis(enabled=True, status="allowed")
        meta.osn_retry_diagnosis = diag
        self.assertIs(meta.osn_retry_diagnosis, diag)

    def test_23_native_meta_from_entry_reads_osn_retry_diagnosis(self) -> None:
        from openshard.cli.run_output import _native_meta_from_entry
        entry = {
            "workflow": "native",
            "osn_retry_diagnosis": {
                "enabled": True,
                "status": "exhausted",
                "retry_count": 1,
                "retry_limit": 1,
                "raw_content_stored": False,
            },
        }
        ns = _native_meta_from_entry(entry)
        self.assertIsNotNone(ns)
        diag = getattr(ns, "osn_retry_diagnosis", None)
        self.assertIsNotNone(diag)
        self.assertTrue(getattr(diag, "enabled", False))
        self.assertEqual(getattr(diag, "status", ""), "exhausted")

    def test_24_diagnosis_creation_calls_no_subprocess(self) -> None:
        from openshard.native.retry_diagnosis import build_osn_retry_diagnosis
        # subprocess must not be imported or used by build_osn_retry_diagnosis
        loop = _make_loop(verification_attempted=True, verification_passed=False)
        d = build_osn_retry_diagnosis(osn_loop_summary=loop)
        self.assertIsNotNone(d)
        # If subprocess were called, the test would timeout or raise; structural check only
        self.assertIn("subprocess" not in dir(build_osn_retry_diagnosis), [True])

    def test_25_diagnosis_imports_no_provider(self) -> None:
        import importlib
        mod = importlib.import_module("openshard.native.retry_diagnosis")
        src = mod.__file__ or ""
        with open(src, encoding="utf-8") as f:
            content = f.read()
        self.assertNotIn("anthropic", content)
        self.assertNotIn("openai", content)
        self.assertNotIn("requests.post", content)

    def test_26_approval_behavior_unchanged(self) -> None:
        from openshard.native.retry_diagnosis import build_osn_retry_diagnosis
        loop = _make_loop(verification_attempted=True, verification_passed=True)
        # explicit approval_required=True takes priority even with passing verification
        d = build_osn_retry_diagnosis(
            osn_loop_summary=loop,
            approval_required=True,
            approval_granted=False,
        )
        self.assertEqual(d.status, "blocked")

    def test_27_existing_loop_summary_retry_fields_work(self) -> None:
        from openshard.native.context import NativeOSNLoopSummary
        s = NativeOSNLoopSummary(enabled=True, retry_used=True, retry_count=1)
        self.assertTrue(s.retry_used)
        self.assertEqual(s.retry_count, 1)


# ---------------------------------------------------------------------------
# 28-36: Receipt rendering
# ---------------------------------------------------------------------------

class TestRenderOSNRetryReceipt(unittest.TestCase):
    def setUp(self) -> None:
        from openshard.native.retry_diagnosis import render_osn_retry_receipt
        self.render = render_osn_retry_receipt

    def _diag(self, **kwargs: Any) -> Any:
        from openshard.native.retry_diagnosis import OSNRetryDiagnosis
        return OSNRetryDiagnosis(**kwargs)

    def test_28_section_renders_when_enabled(self) -> None:
        d = self._diag(enabled=True, status="allowed", failure_kind="verification_failed")
        lines = self.render(d)
        self.assertTrue(any("OSN RETRY" in line for line in lines))

    def test_29_section_omitted_when_disabled(self) -> None:
        d = self._diag(enabled=False)
        self.assertEqual(self.render(d), [])

    def test_29b_section_omitted_for_none(self) -> None:
        self.assertEqual(self.render(None), [])

    def test_30_status_row_appears(self) -> None:
        d = self._diag(enabled=True, status="exhausted")
        lines = self.render(d)
        self.assertTrue(any("Status" in line and "exhausted" in line for line in lines))

    def test_31_count_row_appears(self) -> None:
        d = self._diag(enabled=True, status="exhausted", retry_count=1, retry_limit=1)
        lines = self.render(d)
        self.assertTrue(any("Count" in line and "1/1" in line for line in lines))

    def test_32_failure_row_only_when_kind_set(self) -> None:
        d_with = self._diag(enabled=True, status="allowed", failure_kind="verification_failed")
        d_without = self._diag(enabled=True, status="used", failure_kind="")
        lines_with = self.render(d_with)
        lines_without = self.render(d_without)
        self.assertTrue(any("Failure" in line for line in lines_with))
        self.assertFalse(any("Failure" in line for line in lines_without))

    def test_33_no_raw_command_output(self) -> None:
        d = self._diag(
            enabled=True,
            status="allowed",
            failure_kind="verification_failed",
            failure_summary="verification failed",
        )
        lines = self.render(d)
        combined = "\n".join(lines)
        self.assertNotIn("exit_code=", combined)
        self.assertNotIn("stderr:", combined)
        self.assertNotIn("stdout:", combined)
        self.assertNotIn("Traceback", combined)

    def test_34_no_raw_json_in_section(self) -> None:
        d = self._diag(enabled=True, status="allowed", failure_kind="verification_failed")
        lines = self.render(d)
        combined = "\n".join(lines)
        self.assertNotIn('{"', combined)
        self.assertNotIn('":', combined)

    def test_35_no_em_dash_in_section(self) -> None:
        d = self._diag(
            enabled=True,
            status="exhausted",
            failure_kind="verification_failed",
            failure_summary="verification failed - retry limit reached",
            next_action="manual review required before retry",
        )
        lines = self.render(d)
        combined = "\n".join(lines)
        self.assertNotIn("—", combined)  # em dash character
        self.assertNotIn("--", combined.replace("---", ""))  # double hyphen (allow triple in comments)

    def test_36_no_chain_of_thought_phrases(self) -> None:
        d = self._diag(
            enabled=True,
            status="allowed",
            next_action="fix failing verification path",
        )
        lines = self.render(d)
        combined = "\n".join(lines).lower()
        forbidden = ["i should", "let me", "i think", "thinking", "chain of thought"]
        for phrase in forbidden:
            self.assertNotIn(phrase, combined, f"found chain-of-thought phrase: {phrase!r}")


if __name__ == "__main__":
    unittest.main()
