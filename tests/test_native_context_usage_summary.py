from __future__ import annotations

import unittest
from dataclasses import asdict
from types import SimpleNamespace

from openshard.native.context import (
    NativeContextPacket,
    NativeContextUsageSummary,
    NativeDiffReview,
    NativeEvidence,
    NativeFileContext,
    NativeFileSnippet,
    NativeFinalReport,
    NativeObservation,
    NativePlan,
    NativeVerificationLoop,
    build_native_context_usage_summary,
    render_native_context_usage_summary,
)
from openshard.cli.run_output import _native_meta_from_entry, _render_native_demo_block


def _build(**kwargs) -> NativeContextUsageSummary:
    defaults = dict(
        repo_context_summary=None,
        file_context=None,
        context_packet=None,
        evidence=None,
        observation=None,
        plan=None,
        context_quality_score=None,
        final_report=None,
        diff_review=None,
        verification_loop=None,
        total_chars=0,
    )
    defaults.update(kwargs)
    return build_native_context_usage_summary(**defaults)


def _meta(**kwargs):
    defaults = dict(
        repo_context_summary=None,
        observation=None,
        plan=None,
        verification_plan=None,
        clarification_request=None,
        context_usage_summary=None,
        write_path="pipeline",
        verification_loop=None,
        verification_command_summary=None,
        diff_review=None,
        final_report=None,
        native_loop_steps=[],
        native_loop_trace=[],
        native_backend=None,
        native_backend_available=True,
        native_backend_notes=[],
        native_backend_proof=None,
        read_search_findings=[],
        patch_proposal=None,
        command_policy_preview=None,
        context_packet=None,
        file_context=None,
        context_quality_score=None,
        context_quality_advisory=None,
        change_budget=None,
        change_budget_preview=None,
        change_budget_soft_gate=None,
        approval_request=None,
        approval_receipt=None,
    )
    defaults.update(kwargs)
    return SimpleNamespace(**defaults)


def _render_block(context_usage_summary=None) -> str:
    meta = _meta(context_usage_summary=context_usage_summary)
    return "\n".join(_render_native_demo_block(meta))


class TestBuildNativeContextUsageSummary(unittest.TestCase):

    def test_all_none_inputs_gives_zero_defaults(self):
        s = _build()
        self.assertFalse(s.repo_summary_included)
        self.assertEqual(s.selected_files_count, 0)
        self.assertEqual(s.compact_paths_count, 0)
        self.assertEqual(s.evidence_items_count, 0)
        self.assertEqual(s.snippet_count, 0)
        self.assertEqual(s.failure_warning_count, 0)
        self.assertFalse(s.any_truncated)
        self.assertEqual(s.truncated_components, [])
        self.assertEqual(s.total_chars, 0)
        self.assertFalse(s.compacted)

    def test_repo_summary_included_when_non_none(self):
        s = _build(repo_context_summary=object())
        self.assertTrue(s.repo_summary_included)

    def test_repo_summary_not_included_when_none(self):
        s = _build(repo_context_summary=None)
        self.assertFalse(s.repo_summary_included)

    def test_selected_files_count_from_file_context(self):
        fc = NativeFileContext(files_read=7)
        s = _build(file_context=fc)
        self.assertEqual(s.selected_files_count, 7)

    def test_selected_files_count_zero_when_file_context_none(self):
        s = _build(file_context=None)
        self.assertEqual(s.selected_files_count, 0)

    def test_compact_paths_count_from_context_packet(self):
        cp = NativeContextPacket(compact_paths=["a", "b", "c"])
        s = _build(context_packet=cp)
        self.assertEqual(s.compact_paths_count, 3)

    def test_compact_paths_count_zero_when_packet_none(self):
        s = _build(context_packet=None)
        self.assertEqual(s.compact_paths_count, 0)

    def test_evidence_items_count_from_evidence(self):
        ev = NativeEvidence(search_results=["r1", "r2"])
        s = _build(evidence=ev)
        self.assertEqual(s.evidence_items_count, 2)

    def test_snippet_count_from_evidence(self):
        ev = NativeEvidence(file_snippets=[NativeFileSnippet("x.py"), NativeFileSnippet("y.py")])
        s = _build(evidence=ev)
        self.assertEqual(s.snippet_count, 2)

    def test_evidence_counts_zero_when_evidence_none(self):
        s = _build(evidence=None)
        self.assertEqual(s.evidence_items_count, 0)
        self.assertEqual(s.snippet_count, 0)

    def test_failure_warning_count_aggregates_across_components(self):
        obs = NativeObservation(warnings=["w1", "w2"])
        plan = NativePlan(warnings=["w3"])
        cp = NativeContextPacket(warnings=["w4"])
        s = _build(observation=obs, plan=plan, context_packet=cp)
        self.assertEqual(s.failure_warning_count, 4)

    def test_failure_warning_count_zero_when_no_warnings(self):
        obs = NativeObservation(warnings=[])
        s = _build(observation=obs)
        self.assertEqual(s.failure_warning_count, 0)

    def test_failure_warning_count_aggregates_final_report(self):
        fr = NativeFinalReport(warnings=["x", "y"])
        s = _build(final_report=fr)
        self.assertEqual(s.failure_warning_count, 2)

    def test_truncated_components_includes_evidence(self):
        ev = NativeEvidence(truncated=True)
        s = _build(evidence=ev)
        self.assertIn("evidence", s.truncated_components)
        self.assertTrue(s.any_truncated)

    def test_truncated_components_includes_diff_review(self):
        dr = NativeDiffReview(truncated=True)
        s = _build(diff_review=dr)
        self.assertIn("diff_review", s.truncated_components)
        self.assertTrue(s.any_truncated)

    def test_truncated_components_includes_verification_loop(self):
        vl = NativeVerificationLoop(truncated=True)
        s = _build(verification_loop=vl)
        self.assertIn("verification_loop", s.truncated_components)
        self.assertTrue(s.any_truncated)

    def test_truncated_components_includes_file_context(self):
        fc = NativeFileContext(truncated=True)
        s = _build(file_context=fc)
        self.assertIn("file_context", s.truncated_components)
        self.assertTrue(s.any_truncated)

    def test_multiple_truncated_sources(self):
        ev = NativeEvidence(truncated=True)
        fc = NativeFileContext(truncated=True)
        s = _build(evidence=ev, file_context=fc)
        self.assertIn("evidence", s.truncated_components)
        self.assertIn("file_context", s.truncated_components)
        self.assertEqual(len(s.truncated_components), 2)
        self.assertTrue(s.any_truncated)

    def test_no_truncated_components_when_none_truncated(self):
        ev = NativeEvidence(truncated=False)
        fc = NativeFileContext(truncated=False)
        s = _build(evidence=ev, file_context=fc)
        self.assertFalse(s.any_truncated)
        self.assertEqual(s.truncated_components, [])

    def test_total_chars_passed_through(self):
        s = _build(total_chars=1234)
        self.assertEqual(s.total_chars, 1234)

    def test_compacted_true_when_any_truncated(self):
        ev = NativeEvidence(truncated=True)
        s = _build(evidence=ev)
        self.assertTrue(s.compacted)

    def test_compacted_true_when_paths_at_cap(self):
        cp = NativeContextPacket(compact_paths=["p"] * 8)
        s = _build(context_packet=cp)
        self.assertTrue(s.compacted)

    def test_compacted_false_when_nothing_capped(self):
        cp = NativeContextPacket(compact_paths=["p"] * 3)
        s = _build(context_packet=cp)
        self.assertFalse(s.compacted)

    def test_compacted_false_when_all_none(self):
        s = _build()
        self.assertFalse(s.compacted)


class TestRenderNativeContextUsageSummary(unittest.TestCase):

    def test_none_returns_empty_string(self):
        self.assertEqual(render_native_context_usage_summary(None), "")

    def test_header_present(self):
        s = NativeContextUsageSummary()
        out = render_native_context_usage_summary(s)
        self.assertIn("[context usage summary]", out)

    def test_repo_summary_yes(self):
        s = NativeContextUsageSummary(repo_summary_included=True)
        out = render_native_context_usage_summary(s)
        self.assertIn("repo summary: yes", out)

    def test_repo_summary_no(self):
        s = NativeContextUsageSummary(repo_summary_included=False)
        out = render_native_context_usage_summary(s)
        self.assertIn("repo summary: no", out)

    def test_truncated_yes_with_components(self):
        s = NativeContextUsageSummary(any_truncated=True, truncated_components=["evidence"])
        out = render_native_context_usage_summary(s)
        self.assertIn("truncated: yes (evidence)", out)

    def test_truncated_no(self):
        s = NativeContextUsageSummary(any_truncated=False)
        out = render_native_context_usage_summary(s)
        self.assertIn("truncated: no", out)

    def test_compacted_yes(self):
        s = NativeContextUsageSummary(compacted=True)
        out = render_native_context_usage_summary(s)
        self.assertIn("compacted: yes", out)

    def test_compacted_no(self):
        s = NativeContextUsageSummary(compacted=False)
        out = render_native_context_usage_summary(s)
        self.assertIn("compacted: no", out)

    def test_total_chars(self):
        s = NativeContextUsageSummary(total_chars=500)
        out = render_native_context_usage_summary(s)
        self.assertIn("total chars: 500", out)

    def test_evidence_line(self):
        s = NativeContextUsageSummary(evidence_items_count=3, snippet_count=2)
        out = render_native_context_usage_summary(s)
        self.assertIn("evidence: 3 items, 2 snippets", out)


class TestNativeContextUsageSummarySerializes(unittest.TestCase):

    def test_asdict_produces_dict(self):
        s = NativeContextUsageSummary(
            repo_summary_included=True,
            selected_files_count=3,
            compact_paths_count=5,
            evidence_items_count=2,
            snippet_count=1,
            failure_warning_count=4,
            any_truncated=True,
            truncated_components=["evidence"],
            total_chars=2418,
            compacted=True,
        )
        d = asdict(s)
        self.assertIsInstance(d, dict)
        self.assertEqual(d["total_chars"], 2418)
        self.assertEqual(d["evidence_items_count"], 2)
        self.assertTrue(d["repo_summary_included"])
        self.assertTrue(d["compacted"])

    def test_truncated_components_serializes_as_list(self):
        s = NativeContextUsageSummary(truncated_components=["evidence", "file_context"])
        d = asdict(s)
        self.assertIsInstance(d["truncated_components"], list)
        self.assertEqual(d["truncated_components"], ["evidence", "file_context"])

    def test_asdict_has_all_expected_keys(self):
        s = NativeContextUsageSummary()
        d = asdict(s)
        expected_keys = {
            "repo_summary_included", "selected_files_count", "compact_paths_count",
            "evidence_items_count", "snippet_count", "failure_warning_count",
            "any_truncated", "truncated_components", "total_chars", "compacted",
        }
        self.assertEqual(set(d.keys()), expected_keys)

    def test_roundtrip_via_native_meta_from_entry(self):
        s = NativeContextUsageSummary(total_chars=100, selected_files_count=3, compacted=False)
        entry = {
            "workflow": "native",
            "context_usage_summary": asdict(s),
        }
        reconstructed = _native_meta_from_entry(entry)
        self.assertIsNotNone(reconstructed)
        cus = getattr(reconstructed, "context_usage_summary", None)
        self.assertIsNotNone(cus)
        self.assertEqual(getattr(cus, "total_chars", None), 100)
        self.assertEqual(getattr(cus, "selected_files_count", None), 3)
        self.assertFalse(getattr(cus, "compacted", True))


class TestContextUsageSummaryInDemoBlock(unittest.TestCase):

    def test_context_summary_line_appears(self):
        s = NativeContextUsageSummary(total_chars=100)
        out = _render_block(context_usage_summary=s)
        self.assertIn("context summary:", out)

    def test_context_summary_includes_chars(self):
        s = NativeContextUsageSummary(total_chars=2418)
        out = _render_block(context_usage_summary=s)
        self.assertIn("2418 chars", out)

    def test_context_summary_includes_files_paths_evidence_snippets(self):
        s = NativeContextUsageSummary(
            selected_files_count=3,
            compact_paths_count=5,
            evidence_items_count=2,
            snippet_count=1,
        )
        out = _render_block(context_usage_summary=s)
        self.assertIn("3 files", out)
        self.assertIn("5 paths", out)
        self.assertIn("2 evidence", out)
        self.assertIn("1 snippets", out)

    def test_truncated_suffix_appears_when_any_truncated(self):
        s = NativeContextUsageSummary(any_truncated=True, truncated_components=["evidence"])
        out = _render_block(context_usage_summary=s)
        self.assertIn("truncated", out)

    def test_truncated_suffix_absent_when_not_truncated(self):
        s = NativeContextUsageSummary(any_truncated=False)
        lines = [ln for ln in _render_block(context_usage_summary=s).splitlines() if "context summary:" in ln]
        self.assertTrue(lines)
        self.assertNotIn("truncated", lines[0])

    def test_warnings_suffix_appears(self):
        s = NativeContextUsageSummary(failure_warning_count=4)
        out = _render_block(context_usage_summary=s)
        self.assertIn("4 warnings", out)

    def test_no_context_summary_line_when_none(self):
        out = _render_block(context_usage_summary=None)
        self.assertNotIn("context summary:", out)

    def test_old_saved_run_without_key_renders_safely(self):
        entry = {"workflow": "native"}
        reconstructed = _native_meta_from_entry(entry)
        self.assertIsNotNone(reconstructed)
        cus = getattr(reconstructed, "context_usage_summary", None)
        self.assertIsNone(cus)
        out = "\n".join(_render_native_demo_block(reconstructed))
        self.assertNotIn("context summary:", out)
