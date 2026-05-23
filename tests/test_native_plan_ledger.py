from __future__ import annotations

import unittest
from dataclasses import asdict
from types import SimpleNamespace

import click
from click.testing import CliRunner

from openshard.native.context import (
    NativePlanItem,
    NativePlanLedger,
    build_native_plan_ledger,
    render_native_plan_ledger,
    update_native_plan_ledger_status,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _render_block(native_meta: SimpleNamespace, detail: str = "more") -> str:
    from openshard.cli.main import _render_log_entry

    entry = {
        "task": "add feature",
        "workflow": "native",
        "executor": "native",
        "write_path": "pipeline",
        "plan_ledger": asdict(native_meta.plan_ledger) if native_meta.plan_ledger else None,
    }

    @click.command()
    def cmd():
        _render_log_entry(entry, detail)

    return CliRunner().invoke(cmd).output


def _ledger_with_all_passed() -> NativePlanLedger:
    ledger = build_native_plan_ledger("refactor the auth module")
    for title_fragment in ["Understand task", "Generate patch", "Write patch", "Run verification", "Record receipt"]:
        ledger = update_native_plan_ledger_status(ledger, title_fragment, "passed")
    return ledger


# ---------------------------------------------------------------------------
# Dataclass defaults
# ---------------------------------------------------------------------------

class TestNativePlanItemDefaults(unittest.TestCase):

    def test_default_status_is_pending(self):
        item = NativePlanItem()
        self.assertEqual(item.status, "pending")

    def test_raw_content_stored_always_false(self):
        item = NativePlanItem()
        self.assertFalse(item.raw_content_stored)

    def test_raw_content_stored_reset_on_post_init(self):
        item = NativePlanItem(raw_content_stored=True)
        self.assertFalse(item.raw_content_stored)

    def test_default_title_empty(self):
        item = NativePlanItem()
        self.assertEqual(item.title, "")

    def test_default_evidence_empty(self):
        item = NativePlanItem()
        self.assertEqual(item.evidence, "")


class TestNativePlanLedgerDefaults(unittest.TestCase):

    def test_enabled_by_default(self):
        ledger = NativePlanLedger()
        self.assertTrue(ledger.enabled)

    def test_items_empty_by_default(self):
        ledger = NativePlanLedger()
        self.assertEqual(ledger.items, [])

    def test_counts_zero_by_default(self):
        ledger = NativePlanLedger()
        self.assertEqual(ledger.completed_count, 0)
        self.assertEqual(ledger.failed_count, 0)
        self.assertEqual(ledger.pending_count, 0)

    def test_raw_content_stored_always_false(self):
        ledger = NativePlanLedger()
        self.assertFalse(ledger.raw_content_stored)

    def test_raw_content_stored_reset_on_post_init(self):
        ledger = NativePlanLedger(raw_content_stored=True)
        self.assertFalse(ledger.raw_content_stored)


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------

class TestBuildNativePlanLedger(unittest.TestCase):

    def test_builds_five_items(self):
        ledger = build_native_plan_ledger("add feature")
        self.assertEqual(len(ledger.items), 5)

    def test_all_items_start_pending(self):
        ledger = build_native_plan_ledger("add feature")
        for item in ledger.items:
            self.assertEqual(item.status, "pending")

    def test_pending_count_equals_five(self):
        ledger = build_native_plan_ledger("add feature")
        self.assertEqual(ledger.pending_count, 5)

    def test_completed_and_failed_counts_zero(self):
        ledger = build_native_plan_ledger("add feature")
        self.assertEqual(ledger.completed_count, 0)
        self.assertEqual(ledger.failed_count, 0)

    def test_raw_content_stored_false_on_ledger(self):
        ledger = build_native_plan_ledger("add feature")
        self.assertFalse(ledger.raw_content_stored)

    def test_raw_content_stored_false_on_all_items(self):
        ledger = build_native_plan_ledger("add feature")
        for item in ledger.items:
            self.assertFalse(item.raw_content_stored)

    def test_task_text_not_stored_verbatim_when_long(self):
        long_task = "x" * 500
        ledger = build_native_plan_ledger(long_task)
        serialized = str(asdict(ledger))
        self.assertNotIn(long_task, serialized)

    def test_planned_files_evidence_on_write_item(self):
        ledger = build_native_plan_ledger("add feature", planned_files=["a.py", "b.py"])
        write_item = next(it for it in ledger.items if "Write patch" in it.title)
        self.assertEqual(write_item.evidence, "files=2")

    def test_no_planned_files_no_evidence(self):
        ledger = build_native_plan_ledger("add feature")
        write_item = next(it for it in ledger.items if "Write patch" in it.title)
        self.assertEqual(write_item.evidence, "")

    def test_items_indexed_sequentially(self):
        ledger = build_native_plan_ledger("add feature")
        for i, item in enumerate(ledger.items):
            self.assertEqual(item.index, i)

    def test_ledger_is_json_serializable(self):
        import json
        ledger = build_native_plan_ledger("add feature")
        json.dumps(asdict(ledger))  # must not raise


# ---------------------------------------------------------------------------
# Updater
# ---------------------------------------------------------------------------

class TestUpdateNativePlanLedgerStatus(unittest.TestCase):

    def setUp(self):
        self.ledger = build_native_plan_ledger("add feature")

    def test_updates_matching_item_to_passed(self):
        ledger = update_native_plan_ledger_status(self.ledger, "Understand task", "passed")
        item = next(it for it in ledger.items if "Understand" in it.title)
        self.assertEqual(item.status, "passed")

    def test_completed_count_increments_on_passed(self):
        ledger = update_native_plan_ledger_status(self.ledger, "Generate patch", "passed")
        self.assertEqual(ledger.completed_count, 1)

    def test_failed_count_increments_on_failed(self):
        ledger = update_native_plan_ledger_status(self.ledger, "Run verification", "failed")
        self.assertEqual(ledger.failed_count, 1)

    def test_pending_count_decrements_when_resolved(self):
        ledger = update_native_plan_ledger_status(self.ledger, "Understand task", "passed")
        self.assertEqual(ledger.pending_count, 4)

    def test_unknown_title_no_crash(self):
        ledger = update_native_plan_ledger_status(self.ledger, "nonexistent step", "passed")
        self.assertEqual(ledger.completed_count, 0)

    def test_unknown_status_no_crash_and_no_change(self):
        ledger = update_native_plan_ledger_status(self.ledger, "Generate patch", "invalid_status")
        for item in ledger.items:
            self.assertEqual(item.status, "pending")

    def test_evidence_stored_when_provided(self):
        ledger = update_native_plan_ledger_status(
            self.ledger, "Generate patch", "passed", evidence="files=3"
        )
        item = next(it for it in ledger.items if "Generate" in it.title)
        self.assertEqual(item.evidence, "files=3")

    def test_raw_content_stored_remains_false_after_updates(self):
        ledger = update_native_plan_ledger_status(self.ledger, "Understand task", "passed")
        self.assertFalse(ledger.raw_content_stored)
        for item in ledger.items:
            self.assertFalse(item.raw_content_stored)

    def test_all_passed_counts_correct(self):
        ledger = _ledger_with_all_passed()
        self.assertEqual(ledger.completed_count, 5)
        self.assertEqual(ledger.failed_count, 0)
        self.assertEqual(ledger.pending_count, 0)

    def test_case_insensitive_title_match(self):
        ledger = update_native_plan_ledger_status(self.ledger, "understand task", "passed")
        item = next(it for it in ledger.items if "Understand" in it.title)
        self.assertEqual(item.status, "passed")

    def test_skipped_status_accepted(self):
        ledger = update_native_plan_ledger_status(self.ledger, "Run verification", "skipped")
        item = next(it for it in ledger.items if "verification" in it.title.lower())
        self.assertEqual(item.status, "skipped")


# ---------------------------------------------------------------------------
# Renderer
# ---------------------------------------------------------------------------

class TestRenderNativePlanLedger(unittest.TestCase):

    def test_none_returns_empty_string(self):
        self.assertEqual(render_native_plan_ledger(None), "")

    def test_disabled_ledger_returns_empty_string(self):
        ledger = NativePlanLedger(enabled=False)
        self.assertEqual(render_native_plan_ledger(ledger), "")

    def test_empty_items_returns_empty_string(self):
        ledger = NativePlanLedger(enabled=True, items=[])
        self.assertEqual(render_native_plan_ledger(ledger), "")

    def test_compact_returns_single_line(self):
        ledger = _ledger_with_all_passed()
        rendered = render_native_plan_ledger(ledger, detail="compact")
        self.assertNotIn("\n", rendered)
        self.assertIn("plan ledger:", rendered)

    def test_compact_shows_pass_count(self):
        ledger = _ledger_with_all_passed()
        rendered = render_native_plan_ledger(ledger, detail="compact")
        self.assertIn("5/5 passed", rendered)

    def test_compact_shows_failed_count(self):
        ledger = build_native_plan_ledger("x")
        ledger = update_native_plan_ledger_status(ledger, "Run verification", "failed")
        rendered = render_native_plan_ledger(ledger, detail="compact")
        self.assertIn("1 failed", rendered)

    def test_full_returns_multiline(self):
        ledger = _ledger_with_all_passed()
        rendered = render_native_plan_ledger(ledger, detail="full")
        self.assertIn("\n", rendered)

    def test_full_shows_numbered_items(self):
        ledger = _ledger_with_all_passed()
        rendered = render_native_plan_ledger(ledger, detail="full")
        self.assertIn("1.", rendered)
        self.assertIn("5.", rendered)

    def test_full_shows_item_titles(self):
        ledger = _ledger_with_all_passed()
        rendered = render_native_plan_ledger(ledger, detail="full")
        self.assertIn("Understand task and repo context", rendered)
        self.assertIn("Record receipt", rendered)

    def test_full_shows_evidence(self):
        ledger = build_native_plan_ledger("x")
        ledger = update_native_plan_ledger_status(ledger, "Generate patch", "passed", evidence="files=2")
        rendered = render_native_plan_ledger(ledger, detail="full")
        self.assertIn("files=2", rendered)

    def test_no_raw_content_in_output(self):
        ledger = build_native_plan_ledger("x")
        ledger = update_native_plan_ledger_status(ledger, "Generate patch", "passed", evidence="files=1")
        rendered = render_native_plan_ledger(ledger, detail="full")
        self.assertNotIn("secret content", rendered)


# ---------------------------------------------------------------------------
# Run output compatibility
# ---------------------------------------------------------------------------

class TestRunOutputCompatibility(unittest.TestCase):

    def test_old_entry_without_plan_ledger_no_crash(self):
        from openshard.cli.run_output import _native_meta_from_entry
        entry = {"task": "old run", "workflow": "native", "executor": "native"}
        meta = _native_meta_from_entry(entry)
        self.assertIsNotNone(meta)
        self.assertIsNone(getattr(meta, "plan_ledger", None))

    def test_entry_with_plan_ledger_deserializes(self):
        from openshard.cli.run_output import _native_meta_from_entry
        ledger = _ledger_with_all_passed()
        entry = {
            "task": "native run",
            "workflow": "native",
            "executor": "native",
            "plan_ledger": asdict(ledger),
        }
        meta = _native_meta_from_entry(entry)
        self.assertIsNotNone(meta)
        self.assertIsNotNone(meta.plan_ledger)

    def test_default_detail_does_not_show_plan_ledger(self):
        ledger = _ledger_with_all_passed()
        meta = SimpleNamespace(plan_ledger=ledger)
        out = _render_block(meta, detail="default")
        self.assertNotIn("plan ledger", out)

    def test_more_shows_compact_plan_ledger(self):
        ledger = _ledger_with_all_passed()
        meta = SimpleNamespace(plan_ledger=ledger)
        out = _render_block(meta, detail="full")
        self.assertIn("plan ledger", out)
        self.assertIn("5/5 passed", out)

    def test_full_shows_full_plan_ledger(self):
        ledger = _ledger_with_all_passed()
        meta = SimpleNamespace(plan_ledger=ledger)
        out = _render_block(meta, detail="full")
        self.assertIn("plan ledger", out)
        self.assertIn("Understand task and repo context", out)
        self.assertIn("Record receipt", out)

    def test_none_plan_ledger_no_ledger_section(self):
        meta = SimpleNamespace(plan_ledger=None)
        out = _render_block(meta, detail="more")
        self.assertNotIn("plan ledger", out)

    def test_more_shows_compact_not_items(self):
        ledger = _ledger_with_all_passed()
        meta = SimpleNamespace(plan_ledger=ledger)
        out = _render_block(meta, detail="more")
        self.assertNotIn("Understand task and repo context", out)
        self.assertNotIn("Record receipt", out)


# ---------------------------------------------------------------------------
# Pipeline integration (unit-level, no live API)
# ---------------------------------------------------------------------------

class TestPipelineIntegration(unittest.TestCase):

    def test_build_then_update_chain(self):
        ledger = build_native_plan_ledger("fix auth")
        ledger = update_native_plan_ledger_status(ledger, "Understand task", "passed", evidence="context prepared")
        ledger = update_native_plan_ledger_status(ledger, "Generate patch", "passed", evidence="files=2")
        ledger = update_native_plan_ledger_status(ledger, "Write patch", "passed", evidence="sandbox write complete")
        ledger = update_native_plan_ledger_status(ledger, "Run verification", "passed")
        ledger = update_native_plan_ledger_status(ledger, "Record receipt", "passed")
        self.assertEqual(ledger.completed_count, 5)
        self.assertEqual(ledger.failed_count, 0)
        self.assertEqual(ledger.pending_count, 0)

    def test_asdict_is_json_serializable(self):
        import json
        ledger = _ledger_with_all_passed()
        json.dumps(asdict(ledger))  # must not raise

    def test_no_raw_file_content_in_evidence(self):
        ledger = build_native_plan_ledger("x", planned_files=["src/foo.py"])
        ledger = update_native_plan_ledger_status(ledger, "Generate patch", "passed", evidence="files=1")
        serialized = str(asdict(ledger))
        self.assertNotIn("src/foo.py", serialized)

    def test_verification_failed_after_retry_marks_failed(self):
        ledger = build_native_plan_ledger("x")
        ledger = update_native_plan_ledger_status(ledger, "Run verification", "failed")
        item = next(it for it in ledger.items if "verification" in it.title.lower())
        self.assertEqual(item.status, "failed")
        self.assertEqual(ledger.failed_count, 1)

    def test_verification_skipped_marks_skipped(self):
        ledger = build_native_plan_ledger("x")
        ledger = update_native_plan_ledger_status(ledger, "Run verification", "skipped")
        item = next(it for it in ledger.items if "verification" in it.title.lower())
        self.assertEqual(item.status, "skipped")
        self.assertEqual(ledger.failed_count, 0)


if __name__ == "__main__":
    unittest.main()
