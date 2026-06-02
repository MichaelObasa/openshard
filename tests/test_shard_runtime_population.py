"""Tests for shard runtime population helpers and receipt wiring.

Covers:
  - _populate_context_usage_metadata (pipeline helper)
  - _populate_execution_span_metadata (pipeline helper)
  - build_shard_receipt reading execution_spans + context usage fields
  - render_full_shard_receipt showing/hiding new sections
  - backward compatibility for old entries
"""
from __future__ import annotations

import unittest

from openshard.history.shard_contract import (
    ExecutionSpan,
    build_shard_receipt,
    render_compact_shard_receipt,
    render_full_shard_receipt,
)
from openshard.run.pipeline import (
    _populate_context_usage_metadata,
    _populate_execution_span_metadata,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _minimal_entry() -> dict:
    return {
        "task": "add feature",
        "timestamp": "2026-04-13T06:24:08.695472Z",
        "execution_model": "anthropic/claude-sonnet-4-6",
        "summary": "Feature implemented successfully.",
    }


def _entry_with_context_usage(total: int, injected: int) -> dict:
    e = _minimal_entry()
    e["context_files_considered_count"] = total
    e["context_files_injected_count"] = injected
    e["context_utilisation_ratio"] = round(injected / total, 4) if total > 0 else None
    return e


def _entry_with_spans(spans: list[dict]) -> dict:
    e = _minimal_entry()
    e["execution_spans"] = spans
    return e


def _one_span(phase: str = "plan", status: str = "completed", summary: str = "") -> dict:
    return {
        "span_id": f"phase-0-{phase}",
        "name": phase,
        "kind": "phase",
        "started_at": None,
        "duration_ms": None,
        "status": status,
        "error_class": None,
        "summary": summary or None,
    }


# ---------------------------------------------------------------------------
# _populate_context_usage_metadata
# ---------------------------------------------------------------------------

class TestPopulateContextUsageMetadata(unittest.TestCase):

    def test_promotes_total_and_injected(self):
        meta = {"context_provenance": {"total_items": 12, "injected_sources": 4}}
        _populate_context_usage_metadata(meta)
        self.assertEqual(meta["context_files_considered_count"], 12)
        self.assertEqual(meta["context_files_injected_count"], 4)

    def test_ratio_correct(self):
        meta = {"context_provenance": {"total_items": 10, "injected_sources": 3}}
        _populate_context_usage_metadata(meta)
        self.assertAlmostEqual(meta["context_utilisation_ratio"], 0.3, places=4)

    def test_ratio_none_when_total_zero(self):
        meta = {"context_provenance": {"total_items": 0, "injected_sources": 0}}
        _populate_context_usage_metadata(meta)
        self.assertIsNone(meta["context_utilisation_ratio"])

    def test_ratio_zero_when_injected_zero(self):
        meta = {"context_provenance": {"total_items": 5, "injected_sources": 0}}
        _populate_context_usage_metadata(meta)
        self.assertEqual(meta["context_utilisation_ratio"], 0.0)

    def test_noop_when_none(self):
        # Must not raise
        _populate_context_usage_metadata(None)

    def test_noop_when_context_provenance_missing(self):
        meta: dict = {}
        _populate_context_usage_metadata(meta)
        self.assertNotIn("context_files_considered_count", meta)

    def test_noop_when_context_provenance_not_dict(self):
        meta = {"context_provenance": "bad"}
        _populate_context_usage_metadata(meta)
        self.assertNotIn("context_files_considered_count", meta)

    def test_noop_when_counts_not_int(self):
        meta = {"context_provenance": {"total_items": "bad", "injected_sources": None}}
        _populate_context_usage_metadata(meta)
        self.assertNotIn("context_files_considered_count", meta)

    def test_does_not_overwrite_unrelated_keys(self):
        meta = {
            "context_provenance": {"total_items": 5, "injected_sources": 2},
            "other_key": "keep_me",
        }
        _populate_context_usage_metadata(meta)
        self.assertEqual(meta["other_key"], "keep_me")

    def test_exception_in_provenance_does_not_raise(self):
        # Simulate malformed nested object that could trigger unexpected errors.
        meta = {"context_provenance": {"total_items": [], "injected_sources": {}}}
        try:
            _populate_context_usage_metadata(meta)
        except Exception as exc:  # noqa: BLE001
            self.fail(f"Helper raised unexpectedly: {exc}")


# ---------------------------------------------------------------------------
# _populate_execution_span_metadata
# ---------------------------------------------------------------------------

class TestPopulateExecutionSpanMetadata(unittest.TestCase):

    def test_converts_loop_trace_to_spans(self):
        meta = {
            "native_loop_trace": [
                {"phase": "plan", "status": "completed", "summary": "", "metadata": {}},
                {"phase": "generation", "status": "completed", "summary": "", "metadata": {}},
            ]
        }
        _populate_execution_span_metadata(meta)
        spans = meta.get("execution_spans", [])
        self.assertEqual(len(spans), 2)
        self.assertEqual(spans[0]["name"], "plan")
        self.assertEqual(spans[0]["kind"], "phase")
        self.assertEqual(spans[0]["span_id"], "phase-0-plan")
        self.assertEqual(spans[1]["name"], "generation")

    def test_status_preserved(self):
        meta = {
            "native_loop_trace": [
                {"phase": "verification", "status": "failed", "summary": "", "metadata": {}},
            ]
        }
        _populate_execution_span_metadata(meta)
        self.assertEqual(meta["execution_spans"][0]["status"], "failed")

    def test_status_defaults_to_completed_when_missing(self):
        meta = {"native_loop_trace": [{"phase": "plan", "metadata": {}}]}
        _populate_execution_span_metadata(meta)
        self.assertEqual(meta["execution_spans"][0]["status"], "completed")

    def test_timestamps_always_none(self):
        meta = {"native_loop_trace": [{"phase": "plan", "status": "completed", "metadata": {}}]}
        _populate_execution_span_metadata(meta)
        span = meta["execution_spans"][0]
        self.assertIsNone(span["started_at"])
        self.assertIsNone(span["duration_ms"])

    def test_blank_phase_skipped(self):
        meta = {
            "native_loop_trace": [
                {"phase": "", "status": "completed", "metadata": {}},
                {"phase": "plan", "status": "completed", "metadata": {}},
            ]
        }
        _populate_execution_span_metadata(meta)
        spans = meta.get("execution_spans", [])
        self.assertEqual(len(spans), 1)
        self.assertEqual(spans[0]["name"], "plan")

    def test_non_dict_event_skipped(self):
        meta = {"native_loop_trace": ["bad", {"phase": "plan", "status": "completed", "metadata": {}}]}
        _populate_execution_span_metadata(meta)
        spans = meta.get("execution_spans", [])
        self.assertEqual(len(spans), 1)

    def test_long_summary_capped_at_200(self):
        long_summary = "x" * 300
        meta = {
            "native_loop_trace": [
                {"phase": "plan", "status": "completed", "summary": long_summary, "metadata": {}},
            ]
        }
        _populate_execution_span_metadata(meta)
        self.assertEqual(len(meta["execution_spans"][0]["summary"]), 200)

    def test_empty_summary_stored_as_none(self):
        meta = {
            "native_loop_trace": [
                {"phase": "plan", "status": "completed", "summary": "", "metadata": {}},
            ]
        }
        _populate_execution_span_metadata(meta)
        self.assertIsNone(meta["execution_spans"][0]["summary"])

    def test_noop_when_none(self):
        _populate_execution_span_metadata(None)

    def test_noop_when_trace_missing(self):
        meta: dict = {}
        _populate_execution_span_metadata(meta)
        self.assertNotIn("execution_spans", meta)

    def test_noop_when_trace_empty_list(self):
        meta = {"native_loop_trace": []}
        _populate_execution_span_metadata(meta)
        self.assertNotIn("execution_spans", meta)

    def test_noop_when_all_phases_blank(self):
        meta = {"native_loop_trace": [{"phase": "", "metadata": {}}]}
        _populate_execution_span_metadata(meta)
        self.assertNotIn("execution_spans", meta)

    def test_exception_does_not_raise(self):
        meta = {"native_loop_trace": None}
        try:
            _populate_execution_span_metadata(meta)
        except Exception as exc:  # noqa: BLE001
            self.fail(f"Helper raised unexpectedly: {exc}")


# ---------------------------------------------------------------------------
# build_shard_receipt — context usage fields
# ---------------------------------------------------------------------------

class TestBuildShardReceiptContextUsage(unittest.TestCase):

    def test_context_fields_populated_from_entry(self):
        receipt = build_shard_receipt(_entry_with_context_usage(12, 4))
        self.assertEqual(receipt.context_files_considered_count, 12)
        self.assertEqual(receipt.context_files_injected_count, 4)
        self.assertAlmostEqual(receipt.context_utilisation_ratio, 0.3333, places=3)

    def test_ratio_none_when_total_zero(self):
        receipt = build_shard_receipt(_entry_with_context_usage(0, 0))
        self.assertIsNone(receipt.context_utilisation_ratio)

    def test_ratio_zero_when_injected_zero(self):
        receipt = build_shard_receipt(_entry_with_context_usage(5, 0))
        self.assertEqual(receipt.context_utilisation_ratio, 0.0)

    def test_context_fields_none_when_missing(self):
        receipt = build_shard_receipt(_minimal_entry())
        self.assertIsNone(receipt.context_files_considered_count)
        self.assertIsNone(receipt.context_files_injected_count)
        self.assertIsNone(receipt.context_utilisation_ratio)

    def test_non_int_considered_coerced_to_none(self):
        e = _minimal_entry()
        e["context_files_considered_count"] = "bad"
        receipt = build_shard_receipt(e)
        self.assertIsNone(receipt.context_files_considered_count)

    def test_bool_ratio_coerced_to_none(self):
        e = _minimal_entry()
        e["context_utilisation_ratio"] = True
        receipt = build_shard_receipt(e)
        self.assertIsNone(receipt.context_utilisation_ratio)


# ---------------------------------------------------------------------------
# build_shard_receipt — execution spans
# ---------------------------------------------------------------------------

class TestBuildShardReceiptExecutionSpans(unittest.TestCase):

    def test_spans_populated_from_entry(self):
        e = _entry_with_spans([_one_span("plan", "completed")])
        receipt = build_shard_receipt(e)
        self.assertEqual(len(receipt.execution_spans), 1)
        span = receipt.execution_spans[0]
        self.assertIsInstance(span, ExecutionSpan)
        self.assertEqual(span.name, "plan")
        self.assertEqual(span.kind, "phase")
        self.assertEqual(span.span_id, "phase-0-plan")
        self.assertEqual(span.status, "completed")

    def test_span_timestamps_remain_none(self):
        e = _entry_with_spans([_one_span("plan")])
        receipt = build_shard_receipt(e)
        span = receipt.execution_spans[0]
        self.assertIsNone(span.started_at)
        self.assertIsNone(span.duration_ms)

    def test_span_missing_span_id_skipped(self):
        e = _entry_with_spans([{"name": "plan", "kind": "phase"}])
        receipt = build_shard_receipt(e)
        self.assertEqual(receipt.execution_spans, [])

    def test_span_missing_name_skipped(self):
        e = _entry_with_spans([{"span_id": "s1", "kind": "phase"}])
        receipt = build_shard_receipt(e)
        self.assertEqual(receipt.execution_spans, [])

    def test_non_dict_span_skipped(self):
        e = _entry_with_spans(["bad", _one_span("plan")])
        receipt = build_shard_receipt(e)
        self.assertEqual(len(receipt.execution_spans), 1)

    def test_spans_empty_when_missing(self):
        receipt = build_shard_receipt(_minimal_entry())
        self.assertEqual(receipt.execution_spans, [])


# ---------------------------------------------------------------------------
# render_full_shard_receipt — section visibility
# ---------------------------------------------------------------------------

class TestReceiptSectionVisibility(unittest.TestCase):

    def test_context_usage_section_shown_when_populated(self):
        receipt = build_shard_receipt(_entry_with_context_usage(10, 3))
        out = render_full_shard_receipt(receipt)
        self.assertIn("CONTEXT USAGE", out)
        self.assertIn("Considered", out)
        self.assertIn("Injected", out)
        self.assertIn("Utilisation", out)

    def test_context_usage_section_absent_for_old_entry(self):
        receipt = build_shard_receipt(_minimal_entry())
        out = render_full_shard_receipt(receipt)
        self.assertNotIn("CONTEXT USAGE", out)

    def test_execution_spans_section_shown_when_populated(self):
        e = _entry_with_spans([_one_span("plan"), _one_span("generation")])
        receipt = build_shard_receipt(e)
        out = render_full_shard_receipt(receipt)
        self.assertIn("EXECUTION SPANS", out)
        self.assertIn("plan", out)
        self.assertIn("generation", out)

    def test_execution_spans_section_absent_for_old_entry(self):
        receipt = build_shard_receipt(_minimal_entry())
        out = render_full_shard_receipt(receipt)
        self.assertNotIn("EXECUTION SPANS", out)

    def test_compact_receipt_does_not_crash_with_new_fields(self):
        receipt = build_shard_receipt(_entry_with_context_usage(8, 2))
        out = render_compact_shard_receipt(receipt)
        self.assertIsInstance(out, str)
        self.assertGreater(len(out), 0)


# ---------------------------------------------------------------------------
# Backward compatibility
# ---------------------------------------------------------------------------

class TestBackwardCompatibility(unittest.TestCase):

    def test_old_entry_full_receipt_no_crash(self):
        receipt = build_shard_receipt({"task": "old task", "timestamp": "2024-01-01T00:00:00Z"})
        out = render_full_shard_receipt(receipt)
        self.assertIsInstance(out, str)
        self.assertIn("TASK", out)

    def test_old_entry_compact_receipt_no_crash(self):
        receipt = build_shard_receipt({"task": "old task", "timestamp": "2024-01-01T00:00:00Z"})
        out = render_compact_shard_receipt(receipt)
        self.assertIsInstance(out, str)
        self.assertGreater(len(out), 0)

    def test_old_entry_context_usage_absent(self):
        receipt = build_shard_receipt(_minimal_entry())
        out = render_full_shard_receipt(receipt)
        self.assertNotIn("CONTEXT USAGE", out)

    def test_old_entry_execution_spans_absent(self):
        receipt = build_shard_receipt(_minimal_entry())
        out = render_full_shard_receipt(receipt)
        self.assertNotIn("EXECUTION SPANS", out)

    def test_old_entry_context_fields_none(self):
        receipt = build_shard_receipt(_minimal_entry())
        self.assertIsNone(receipt.context_files_considered_count)
        self.assertIsNone(receipt.context_files_injected_count)
        self.assertIsNone(receipt.context_utilisation_ratio)

    def test_old_entry_execution_spans_empty(self):
        receipt = build_shard_receipt(_minimal_entry())
        self.assertEqual(receipt.execution_spans, [])
