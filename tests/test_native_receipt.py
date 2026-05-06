from __future__ import annotations

import unittest
from types import SimpleNamespace

from openshard.cli.run_output import _render_native_receipt, _native_meta_from_entry


def _ns(**kwargs):
    return SimpleNamespace(**kwargs)


def _report(**kwargs):
    defaults = dict(
        used_native_context=True,
        observed_tools=[],
        selected_skills=[],
        plan_intent=None,
        plan_risk=None,
        evidence_items=0,
        snippet_files=0,
        verification_attempted=False,
        verification_passed=False,
        verification_retried=False,
        diff_files=[],
        added_lines=0,
        removed_lines=0,
        warnings=[],
    )
    defaults.update(kwargs)
    return _ns(**defaults)


def _meta(**kwargs):
    defaults = dict(
        final_report=None,
        diff_review=None,
        approval_receipt=None,
    )
    defaults.update(kwargs)
    return _ns(**defaults)


class TestRenderNativeReceipt(unittest.TestCase):

    def test_receipt_saved_always_present(self):
        out = _render_native_receipt(_meta())
        self.assertIn("Receipt saved", out)

    def test_ends_with_period(self):
        out = _render_native_receipt(_meta())
        self.assertTrue(out.endswith("."))

    def test_no_files_changed_omitted(self):
        meta = _meta(final_report=_report(diff_files=[]))
        out = _render_native_receipt(meta)
        self.assertNotIn("file", out)

    def test_one_file_changed(self):
        meta = _meta(final_report=_report(diff_files=["a.py"]))
        out = _render_native_receipt(meta)
        self.assertIn("1 file changed", out)

    def test_plural_files_changed(self):
        meta = _meta(final_report=_report(diff_files=["a.py", "b.py", "c.py"]))
        out = _render_native_receipt(meta)
        self.assertIn("3 files changed", out)

    def test_files_from_diff_review_fallback(self):
        diff_review = _ns(changed_files=["x.py", "y.py"], has_diff=True)
        meta = _meta(final_report=None, diff_review=diff_review)
        out = _render_native_receipt(meta)
        self.assertIn("2 files changed", out)

    def test_verification_passed(self):
        meta = _meta(
            final_report=_report(
                diff_files=["a.py"],
                verification_attempted=True,
                verification_passed=True,
            )
        )
        out = _render_native_receipt(meta)
        self.assertIn("Verification passed", out)
        self.assertNotIn("Verification failed", out)

    def test_verification_failed(self):
        meta = _meta(
            final_report=_report(
                diff_files=["a.py"],
                verification_attempted=True,
                verification_passed=False,
            )
        )
        out = _render_native_receipt(meta)
        self.assertIn("Verification failed", out)
        self.assertNotIn("Verification passed", out)

    def test_verification_not_attempted_omitted(self):
        meta = _meta(final_report=_report(verification_attempted=False))
        out = _render_native_receipt(meta)
        self.assertNotIn("Verification", out)

    def test_no_risky_writes_default(self):
        out = _render_native_receipt(_meta(approval_receipt=None))
        self.assertIn("No risky writes", out)
        self.assertNotIn("Write approved", out)

    def test_write_approved_when_granted(self):
        receipt = _ns(source="change_budget_soft_gate", granted=True)
        meta = _meta(approval_receipt=receipt)
        out = _render_native_receipt(meta)
        self.assertIn("Write approved", out)
        self.assertNotIn("No risky writes", out)

    def test_approval_not_granted_shows_no_risky_writes(self):
        receipt = _ns(source="change_budget_soft_gate", granted=False)
        meta = _meta(approval_receipt=receipt)
        out = _render_native_receipt(meta)
        self.assertIn("No risky writes", out)

    def test_full_receipt_format(self):
        meta = _meta(
            final_report=_report(
                diff_files=["a.py", "b.py"],
                verification_attempted=True,
                verification_passed=True,
            )
        )
        out = _render_native_receipt(meta)
        self.assertEqual(
            out,
            "2 files changed. Verification passed. No risky writes. Receipt saved.",
        )

    def test_receipt_from_entry_via_native_meta_from_entry(self):
        entry = {
            "workflow": "native",
            "executor": "native",
            "diff_review": {
                "has_diff": True,
                "changed_files": ["x.py"],
                "added_lines": 5,
                "removed_lines": 2,
                "output_chars": 100,
                "truncated": False,
            },
            "final_report": {
                "used_native_context": True,
                "observed_tools": [],
                "selected_skills": [],
                "plan_intent": None,
                "plan_risk": None,
                "evidence_items": 0,
                "snippet_files": 0,
                "verification_attempted": True,
                "verification_passed": True,
                "verification_retried": False,
                "diff_files": ["x.py"],
                "added_lines": 5,
                "removed_lines": 2,
                "warnings": [],
            },
        }
        nm = _native_meta_from_entry(entry)
        out = _render_native_receipt(nm)
        self.assertIn("1 file changed", out)
        self.assertIn("Verification passed", out)
        self.assertIn("Receipt saved", out)

    def test_non_native_entry_returns_none_meta(self):
        entry = {"workflow": "standard", "task": "do a thing"}
        nm = _native_meta_from_entry(entry)
        self.assertIsNone(nm)
