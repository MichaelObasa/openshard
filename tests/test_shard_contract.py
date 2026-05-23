from __future__ import annotations

import unittest

import click
from click.testing import CliRunner

from openshard.history.shard_contract import (
    ShardFinding,
    ShardReceipt,
    _extract_findings,
    _result_display,
    build_live_run_receipt,
    build_shard_receipt,
    render_compact_shard_receipt,
    render_full_shard_receipt,
)


def _minimal_entry() -> dict:
    return {
        "task": "add feature",
        "timestamp": "2026-04-13T06:24:08.695472Z",
        "execution_model": "anthropic/claude-sonnet-4-6",
        "summary": "Feature implemented successfully.",
    }


def _rich_native_entry() -> dict:
    return {
        "task": "add JWT auth with test coverage",
        "timestamp": "2026-04-13T06:24:08.695472Z",
        "workflow": "native",
        "executor": "native",
        "execution_profile": "native_deep",
        "execution_model": "anthropic/claude-sonnet-4-6",
        "routing_selected_model": "anthropic/claude-sonnet-4-6",
        "routing_selected_provider": "anthropic",
        "estimated_cost": 0.0234,
        "verification_attempted": True,
        "verification_passed": True,
        "files_created": 2,
        "files_updated": 1,
        "files_deleted": 0,
        "duration_seconds": 12.5,
        "summary": "Implemented JWT authentication with full test suite.",
        "files_detail": [
            {"path": "src/auth.py", "change_type": "create", "summary": "created"},
            {"path": "tests/test_auth.py", "change_type": "create", "summary": "created"},
            {"path": "pyproject.toml", "change_type": "update", "summary": "updated"},
        ],
        "form_factor": {
            "risk_level": "medium",
            "read_only": False,
            "internal_form_factor": "native_deep",
            "public_mode": "Deep Run",
            "reason": "multi-file changes",
            "confidence": "high",
            "context_quality": "good",
            "warnings": [],
        },
        "write_path": "pipeline",
        "diff_review": {
            "has_diff": True,
            "changed_files": ["src/auth.py", "tests/test_auth.py", "pyproject.toml"],
            "added_lines": 34,
            "removed_lines": 8,
        },
    }


def _render_last(entry: dict, detail: str = "default") -> str:
    from openshard.cli.main import _render_log_entry

    @click.command()
    def cmd():
        _render_log_entry(entry, detail)

    return CliRunner().invoke(cmd).output


class TestBuildShardReceiptMinimal(unittest.TestCase):
    """Test 1: Minimal entry produces a valid ShardReceipt without crashing."""

    def test_produces_shard_receipt(self):
        receipt = build_shard_receipt(_minimal_entry())
        self.assertIsInstance(receipt, ShardReceipt)

    def test_shard_id_present(self):
        receipt = build_shard_receipt(_minimal_entry())
        self.assertRegex(receipt.shard_id, r"shard-\d{8}-\d{4}")

    def test_agent_openshard_for_non_native(self):
        receipt = build_shard_receipt(_minimal_entry())
        self.assertEqual(receipt.agent, "OpenShard")

    def test_result_from_summary(self):
        receipt = build_shard_receipt(_minimal_entry())
        self.assertIn("Feature implemented", receipt.result)


class TestBuildShardReceiptRichNative(unittest.TestCase):
    """Test 2: Rich native entry maps all fields correctly."""

    def setUp(self):
        self.receipt = build_shard_receipt(_rich_native_entry(), index=5)

    def test_agent_openshard_native(self):
        self.assertEqual(self.receipt.agent, "OpenShard Native")

    def test_strategy_plan_plus_execute(self):
        self.assertEqual(self.receipt.strategy, "Plan + Execute")

    def test_model_display_auto_prefix(self):
        self.assertEqual(self.receipt.model_display, "Auto → Claude Sonnet 4.6")

    def test_risk_medium(self):
        self.assertEqual(self.receipt.risk, "Medium")

    def test_sandbox_off_pipeline(self):
        self.assertEqual(self.receipt.sandbox, "Off")

    def test_files_changed_count(self):
        self.assertEqual(self.receipt.files_changed, 3)

    def test_checks_passed(self):
        self.assertEqual(self.receipt.checks_display, "1/1 passed")

    def test_status_passed(self):
        self.assertEqual(self.receipt.status, "Passed")

    def test_approval_not_required(self):
        self.assertEqual(self.receipt.approval, "Not required")

    def test_cost_display(self):
        self.assertEqual(self.receipt.cost_display, "$0.0234")

    def test_shard_id_uses_index(self):
        self.assertEqual(self.receipt.shard_id, "shard-20260413-0006")

    def test_files_touched(self):
        self.assertIn("src/auth.py", self.receipt.files_touched)
        self.assertIn("pyproject.toml", self.receipt.files_touched)

    def test_diff_added_removed(self):
        self.assertEqual(self.receipt.diff_added, 34)
        self.assertEqual(self.receipt.diff_removed, 8)


class TestBuildShardReceiptMissingFields(unittest.TestCase):
    """Test 3: Entry with only task+timestamp does not crash; missing fields are Not recorded."""

    def setUp(self):
        self.receipt = build_shard_receipt({"task": "do something", "timestamp": "2026-04-13T00:00:00Z"})

    def test_no_crash(self):
        self.assertIsInstance(self.receipt, ShardReceipt)

    def test_strategy_not_recorded(self):
        self.assertEqual(self.receipt.strategy, "Not recorded")

    def test_model_not_recorded(self):
        self.assertEqual(self.receipt.model_display, "Not recorded")

    def test_risk_not_recorded(self):
        self.assertEqual(self.receipt.risk, "Not recorded")

    def test_sandbox_not_recorded(self):
        self.assertEqual(self.receipt.sandbox, "Not recorded")

    def test_checks_not_recorded(self):
        self.assertEqual(self.receipt.checks_display, "Not recorded")

    def test_approval_not_recorded(self):
        self.assertEqual(self.receipt.approval, "Not recorded")

    def test_cost_not_recorded(self):
        self.assertEqual(self.receipt.cost_display, "Not recorded")

    def test_empty_lists(self):
        self.assertEqual(self.receipt.files_touched, [])
        self.assertEqual(self.receipt.model_stages, [])


class TestShardIdFormat(unittest.TestCase):
    """Test 4: Shard ID format is shard-YYYYMMDD-NNNN and is sortable."""

    def test_format_matches_pattern(self):
        receipt = build_shard_receipt({"timestamp": "2026-04-13T06:24:08Z"}, index=0)
        self.assertRegex(receipt.shard_id, r"^shard-\d{8}-\d{4}$")

    def test_date_from_timestamp(self):
        receipt = build_shard_receipt({"timestamp": "2026-04-13T06:24:08Z"}, index=0)
        self.assertTrue(receipt.shard_id.startswith("shard-20260413-"))

    def test_index_zero_gives_0001(self):
        receipt = build_shard_receipt({"timestamp": "2026-04-13T00:00:00Z"}, index=0)
        self.assertEqual(receipt.shard_id, "shard-20260413-0001")

    def test_index_nine_gives_0010(self):
        receipt = build_shard_receipt({"timestamp": "2026-04-13T00:00:00Z"}, index=9)
        self.assertEqual(receipt.shard_id, "shard-20260413-0010")

    def test_no_index_gives_0001(self):
        receipt = build_shard_receipt({"timestamp": "2026-04-13T00:00:00Z"})
        self.assertEqual(receipt.shard_id, "shard-20260413-0001")

    def test_sortable(self):
        r1 = build_shard_receipt({"timestamp": "2026-04-13T00:00:00Z"}, index=0)
        r2 = build_shard_receipt({"timestamp": "2026-04-14T00:00:00Z"}, index=0)
        self.assertLess(r1.shard_id, r2.shard_id)

    def test_bad_timestamp_uses_today(self):
        receipt = build_shard_receipt({"timestamp": "not-a-date"})
        self.assertRegex(receipt.shard_id, r"^shard-\d{8}-\d{4}$")


class TestCompactReceiptFields(unittest.TestCase):
    """Test 5: render_compact_shard_receipt includes all required fields."""

    def setUp(self):
        self.receipt = build_shard_receipt(_rich_native_entry(), index=0)
        self.out = render_compact_shard_receipt(self.receipt)

    def test_receipt_header(self):
        self.assertIn("RECEIPT", self.out)

    def test_task_field(self):
        self.assertIn("Task", self.out)

    def test_agent_field(self):
        self.assertIn("Executor", self.out)

    def test_strategy_field_absent(self):
        self.assertNotIn("Strategy", self.out)

    def test_model_field(self):
        self.assertIn("Model", self.out)

    def test_risk_field(self):
        self.assertIn("Risk", self.out)

    def test_sandbox_field(self):
        self.assertIn("Sandbox", self.out)

    def test_changed_field(self):
        self.assertIn("Changed", self.out)

    def test_checks_field(self):
        self.assertIn("Checks", self.out)

    def test_approval_field(self):
        self.assertIn("Approval", self.out)

    def test_cost_field(self):
        self.assertIn("Cost", self.out)

    def test_result_field(self):
        self.assertIn("Result", self.out)

    def test_no_internal_terms(self):
        for term in ("candidate", "routing advisory", "verification loop", "plan ledger"):
            self.assertNotIn(term, self.out.lower())


class TestFullReceiptSections(unittest.TestCase):
    """Test 6: render_full_shard_receipt includes all required sections."""

    def setUp(self):
        self.receipt = build_shard_receipt(_rich_native_entry(), index=0)
        self.out = render_full_shard_receipt(self.receipt)

    def test_shard_header(self):
        self.assertIn("SHARD", self.out)

    def test_task_section(self):
        self.assertIn("TASK", self.out)

    def test_execution_section(self):
        self.assertIn("EXECUTION", self.out)

    def test_context_section(self):
        self.assertIn("CONTEXT", self.out)

    def test_policy_section(self):
        self.assertIn("POLICY", self.out)

    def test_checks_section(self):
        self.assertIn("CHECKS", self.out)

    def test_changes_section(self):
        self.assertIn("CHANGES", self.out)

    def test_cost_section(self):
        self.assertIn("COST", self.out)

    def test_receipt_section(self):
        self.assertIn("RECEIPT", self.out)

    def test_shard_id_in_receipt_section(self):
        self.assertIn("Shard ID", self.out)

    def test_created_in_receipt_section(self):
        self.assertIn("Created", self.out)


class TestFailedRunReceipt(unittest.TestCase):
    """Test 7 + 11: Failed run shows Status Failed and Checks 0/1 passed."""

    def setUp(self):
        entry = {
            "task": "deploy to production",
            "timestamp": "2026-04-13T06:24:08Z",
            "workflow": "native",
            "executor": "native",
            "verification_attempted": True,
            "verification_passed": False,
            "summary": "Deployment failed: tests did not pass.",
        }
        self.receipt = build_shard_receipt(entry, index=0)

    def test_status_failed(self):
        self.assertEqual(self.receipt.status, "Failed")

    def test_checks_zero_of_one(self):
        self.assertEqual(self.receipt.checks_display, "0/1 passed")

    def test_result_not_empty(self):
        self.assertNotEqual(self.receipt.result, "Not recorded")

    def test_compact_receipt_no_crash(self):
        out = render_compact_shard_receipt(self.receipt)
        self.assertIn("RECEIPT", out)

    def test_full_receipt_no_crash(self):
        out = render_full_shard_receipt(self.receipt)
        self.assertIn("SHARD", out)


class TestOldEntryWithoutNativeFields(unittest.TestCase):
    """Test 8: Old run entries without native fields render with Not recorded."""

    def setUp(self):
        entry = {
            "task": "old run from v0.1",
            "timestamp": "2025-01-01T00:00:00Z",
            "execution_model": "anthropic/claude-3-opus",
            "summary": "Completed.",
        }
        self.receipt = build_shard_receipt(entry, index=0)
        self.compact = render_compact_shard_receipt(self.receipt)

    def test_strategy_not_recorded(self):
        self.assertIn("Not recorded", self.compact)

    def test_no_crash(self):
        self.assertIn("RECEIPT", self.compact)

    def test_agent_openshard(self):
        self.assertEqual(self.receipt.agent, "OpenShard")


class TestLastMoreIncludesShardSections(unittest.TestCase):
    """Test 9: _render_log_entry with detail='more' includes SHARD sections before [native]."""

    def _native_entry(self) -> dict:
        return {
            "task": "add feature",
            "workflow": "native",
            "executor": "native",
            "repo_context_summary": {
                "likely_stack_markers": ["python"],
                "test_markers": ["pytest"],
                "total_files": 30,
                "top_level_dirs": ["openshard"],
                "package_files": ["pyproject.toml"],
                "truncated": False,
            },
            "observation": {
                "dirty_diff_present": False,
                "search_matches_count": 3,
                "observed_tools": [],
                "verification_available": True,
                "warnings": [],
            },
            "plan": {"intent": "implementation", "risk": "medium", "suggested_steps": [], "warnings": []},
            "write_path": "pipeline",
            "verification_loop": {
                "attempted": True,
                "passed": True,
                "retried": True,
                "exit_code": 0,
                "output_chars": 200,
                "truncated": False,
            },
            "diff_review": {
                "has_diff": True,
                "changed_files": ["a.py", "b.py"],
                "added_lines": 34,
                "removed_lines": 8,
                "output_chars": 500,
                "truncated": False,
            },
            "final_report": {
                "used_native_context": True,
                "observed_tools": [],
                "selected_skills": ["repo mapping", "safe file editing"],
                "plan_intent": "implementation",
                "plan_risk": "medium",
                "evidence_items": 3,
                "snippet_files": 2,
                "verification_attempted": True,
                "verification_passed": True,
                "verification_retried": True,
                "diff_files": ["a.py", "b.py"],
                "added_lines": 34,
                "removed_lines": 8,
                "warnings": [],
            },
        }

    def test_shard_header_present(self):
        out = _render_last(self._native_entry(), detail="more")
        self.assertIn("SHARD —", out)

    def test_execution_section_present(self):
        out = _render_last(self._native_entry(), detail="more")
        self.assertIn("EXECUTION", out)

    def test_policy_section_present(self):
        out = _render_last(self._native_entry(), detail="more")
        self.assertIn("POLICY", out)

    def test_receipt_section_present(self):
        out = _render_last(self._native_entry(), detail="more")
        self.assertIn("RECEIPT —", out)

    def test_shard_appears_before_native_block(self):
        out = _render_last(self._native_entry(), detail="more")
        shard_idx = out.index("SHARD —")
        native_idx = out.find("[native]")
        if native_idx != -1:
            self.assertLess(shard_idx, native_idx)


class TestReadOnlySandboxNotFalsified(unittest.TestCase):
    """Test 12: Read-only run does not falsely show Sandbox On just from workspace_path."""

    def test_read_only_form_factor_gives_not_required(self):
        entry = {
            "task": "review the codebase",
            "timestamp": "2026-04-13T06:00:00Z",
            "workflow": "native",
            "executor": "native",
            "form_factor": {
                "risk_level": "low",
                "read_only": True,
                "internal_form_factor": "native_light",
                "public_mode": "Ask",
                "reason": "read-only",
                "confidence": "high",
                "context_quality": "good",
                "warnings": [],
            },
            "workspace_path": "/tmp/some-workspace",
        }
        receipt = build_shard_receipt(entry)
        self.assertEqual(receipt.sandbox, "Off")
        self.assertNotEqual(receipt.sandbox, "On")

    def test_workspace_path_alone_does_not_imply_on(self):
        entry = {
            "task": "check something",
            "timestamp": "2026-04-13T06:00:00Z",
            "workspace_path": "/tmp/ws",
        }
        receipt = build_shard_receipt(entry)
        self.assertNotEqual(receipt.sandbox, "On")


class TestModelDisplayWithoutRouting(unittest.TestCase):
    """Test 13: execution_model-only entry does not get Auto → prefix."""

    def test_no_auto_prefix_without_routing(self):
        entry = {
            "task": "fix bug",
            "timestamp": "2026-04-13T06:24:08Z",
            "execution_model": "anthropic/claude-sonnet-4.6",
        }
        receipt = build_shard_receipt(entry)
        self.assertFalse(receipt.model_display.startswith("Auto →"))
        self.assertEqual(receipt.model_display, "Claude Sonnet 4.6")


class TestModelDisplayWithRouting(unittest.TestCase):
    """Test 14: Entry with routing_selected_model gets Auto → prefix."""

    def test_auto_prefix_with_routing(self):
        entry = {
            "task": "fix bug",
            "timestamp": "2026-04-13T06:24:08Z",
            "execution_model": "anthropic/claude-sonnet-4.6",
            "routing_selected_model": "anthropic/claude-opus-4-7",
        }
        receipt = build_shard_receipt(entry)
        self.assertEqual(receipt.model_display, "Auto → Claude Opus 4.7")


# ---------------------------------------------------------------------------
# New tests for visual polish and model display helper
# ---------------------------------------------------------------------------

from openshard.history.shard_contract import _display_model_name  # noqa: E402


class TestDisplayModelName(unittest.TestCase):
    """_display_model_name converts provider/model slugs to friendly names."""

    def test_deepseek_v4_pro(self):
        self.assertEqual(_display_model_name("deepseek/deepseek-v4-pro"), "DeepSeek V4 Pro")

    def test_deepseek_v4_flash(self):
        self.assertEqual(_display_model_name("deepseek/deepseek-v4-flash"), "DeepSeek V4 Flash")

    def test_claude_sonnet_dot(self):
        self.assertEqual(_display_model_name("anthropic/claude-sonnet-4.6"), "Claude Sonnet 4.6")

    def test_claude_sonnet_hyphen(self):
        self.assertEqual(_display_model_name("anthropic/claude-sonnet-4-6"), "Claude Sonnet 4.6")

    def test_claude_opus(self):
        self.assertEqual(_display_model_name("anthropic/claude-opus-4.7"), "Claude Opus 4.7")

    def test_gpt_55(self):
        self.assertEqual(_display_model_name("openai/gpt-5.5"), "GPT-5.5")

    def test_gpt_55_thinking(self):
        self.assertEqual(_display_model_name("openai/gpt-5.5-thinking"), "GPT-5.5 Thinking")

    def test_glm(self):
        self.assertEqual(_display_model_name("z-ai/glm-5.1"), "GLM-5.1")

    def test_qwen(self):
        self.assertEqual(_display_model_name("qwen/qwen-3.6-plus"), "Qwen 3.6 Plus")

    def test_gemini(self):
        self.assertEqual(_display_model_name("google/gemini-3.1-pro"), "Gemini 3.1 Pro")

    def test_empty_string_passthrough(self):
        self.assertEqual(_display_model_name(""), "")

    def test_unknown_slug_fallback_no_crash(self):
        result = _display_model_name("unknown/my-model-v2")
        self.assertIsInstance(result, str)
        self.assertGreater(len(result), 0)


class TestCompactReceiptVisualStyle(unittest.TestCase):
    """Compact receipt uses separator lines and indented rows."""

    def setUp(self):
        self.receipt = build_shard_receipt(_rich_native_entry(), index=0)
        self.out = render_compact_shard_receipt(self.receipt)

    def test_has_separator_lines(self):
        self.assertIn("━", self.out)

    def test_starts_with_separator(self):
        self.assertTrue(self.out.startswith("━"))

    def test_ends_with_separator(self):
        self.assertTrue(self.out.rstrip("\n").endswith("━"))

    def test_receipt_header_indented(self):
        self.assertIn("  RECEIPT —", self.out)

    def test_rows_are_indented(self):
        # Every data row should start with two spaces
        data_rows = [
            ln for ln in self.out.splitlines()
            if ln.strip() and not ln.startswith("━") and "RECEIPT —" not in ln
        ]
        for row in data_rows:
            self.assertTrue(row.startswith("  "), f"Row not indented: {row!r}")

    def test_no_raw_provider_slash_in_output(self):
        # Friendly-name entries should not leak raw slugs like "anthropic/..."
        self.assertNotIn("anthropic/", self.out)

    def test_single_model_uses_model_label(self):
        self.assertIn("  Model", self.out)
        self.assertNotIn("  Models", self.out)

    def test_friendly_model_name_in_receipt(self):
        self.assertIn("Claude Sonnet 4.6", self.out)


class TestCompactReceiptMultiStageModels(unittest.TestCase):
    """Multi-stage entries use 'Models' label with friendly names."""

    def _two_model_entry(self) -> dict:
        return {
            "task": "big refactor",
            "timestamp": "2026-04-13T06:00:00Z",
            "stage_runs": [
                {"stage_type": "planning", "model": "anthropic/claude-sonnet-4-6", "duration": 5.0, "cost": 0.01},
                {"stage_type": "implementation", "model": "deepseek/deepseek-v4-pro", "duration": 10.0, "cost": 0.02},
            ],
        }

    def _same_model_entry(self) -> dict:
        return {
            "task": "simple task",
            "timestamp": "2026-04-13T06:00:00Z",
            "stage_runs": [
                {"stage_type": "planning", "model": "anthropic/claude-sonnet-4-6", "duration": 2.0, "cost": 0.005},
                {"stage_type": "implementation", "model": "anthropic/claude-sonnet-4-6", "duration": 3.0, "cost": 0.01},
            ],
        }

    def test_two_distinct_models_uses_models_label(self):
        receipt = build_shard_receipt(self._two_model_entry())
        out = render_compact_shard_receipt(receipt)
        self.assertIn("  Models", out)

    def test_two_distinct_models_shows_arrow(self):
        receipt = build_shard_receipt(self._two_model_entry())
        out = render_compact_shard_receipt(receipt)
        self.assertIn("→", out)

    def test_two_distinct_models_friendly_names(self):
        receipt = build_shard_receipt(self._two_model_entry())
        out = render_compact_shard_receipt(receipt)
        self.assertIn("Claude Sonnet 4.6", out)
        self.assertIn("DeepSeek V4 Pro", out)

    def test_same_model_twice_uses_model_label(self):
        receipt = build_shard_receipt(self._same_model_entry())
        out = render_compact_shard_receipt(receipt)
        self.assertIn("  Model", out)
        self.assertNotIn("  Models", out)

    def test_stage_labels_are_friendly_in_full_receipt(self):
        receipt = build_shard_receipt(self._two_model_entry())
        out = render_full_shard_receipt(receipt)
        self.assertIn("Planning", out)
        self.assertIn("Execution", out)

    def test_raw_stage_type_not_in_full_receipt(self):
        receipt = build_shard_receipt(self._two_model_entry())
        out = render_full_shard_receipt(receipt)
        self.assertNotIn("implementation", out)

    def test_no_raw_slugs_in_full_receipt(self):
        receipt = build_shard_receipt(self._two_model_entry())
        out = render_full_shard_receipt(receipt)
        self.assertNotIn("anthropic/", out)
        self.assertNotIn("deepseek/", out)


class TestFullReceiptVisualStyle(unittest.TestCase):
    """Full SHARD receipt uses the same separator style as the compact receipt."""

    def setUp(self):
        self.receipt = build_shard_receipt(_rich_native_entry(), index=0)
        self.out = render_full_shard_receipt(self.receipt)

    def test_has_separator_lines(self):
        self.assertIn("━", self.out)

    def test_shard_header_indented(self):
        self.assertIn("  SHARD —", self.out)

    def test_receipt_footer_section_indented(self):
        self.assertIn("  RECEIPT", self.out)


class TestMoreViewHasSeparatorStyle(unittest.TestCase):
    """openshard last --more uses the same bordered receipt style as default."""

    def _native_entry(self) -> dict:
        return {
            "task": "add feature",
            "timestamp": "2026-04-13T06:24:08Z",
            "workflow": "native",
            "executor": "native",
            "write_path": "pipeline",
            "final_report": {
                "used_native_context": True,
                "observed_tools": [],
                "selected_skills": [],
                "plan_intent": "implementation",
                "plan_risk": "medium",
                "evidence_items": 0,
                "snippet_files": 0,
                "verification_attempted": True,
                "verification_passed": True,
                "verification_retried": False,
                "diff_files": [],
                "added_lines": 0,
                "removed_lines": 0,
                "warnings": [],
            },
        }

    def test_more_output_has_separator(self):
        out = _render_last(self._native_entry(), detail="more")
        self.assertIn("━", out)

    def test_more_output_has_receipt_block(self):
        out = _render_last(self._native_entry(), detail="more")
        self.assertIn("  RECEIPT —", out)

    def test_more_output_has_shard_block(self):
        out = _render_last(self._native_entry(), detail="more")
        self.assertIn("  SHARD —", out)


class TestReceiptAppearsBeforeTimeFoooter(unittest.TestCase):
    """Compact RECEIPT appears before the Time/Cost footer in openshard last."""

    def _native_entry(self) -> dict:
        return {
            "task": "add feature",
            "timestamp": "2026-04-13T06:24:08Z",
            "workflow": "native",
            "executor": "native",
            "write_path": "pipeline",
            "duration_seconds": 12.5,
            "estimated_cost": 0.0234,
            "summary": "Done.",
        }

    def test_receipt_before_time_footer(self):
        out = _render_last(self._native_entry(), detail="default")
        receipt_idx = out.index("RECEIPT —")
        time_idx = out.index("Time:")
        self.assertLess(receipt_idx, time_idx)


class TestShardFindings(unittest.TestCase):
    """Agent Notes / Findings v0: contract, extraction, and rendering."""

    # ------------------------------------------------------------------ #
    # Test 1: Structured finding dicts convert to ShardFinding objects    #
    # ------------------------------------------------------------------ #
    def test_structured_finding_dicts_convert(self):
        entry = {
            "findings": [
                {"severity": "Critical", "message": "Database public IP enabled", "path": "main.tf", "line": 42},
                {"severity": "High", "message": "Bucket ACLs enabled"},
            ]
        }
        findings = _extract_findings(entry)
        self.assertEqual(len(findings), 2)
        self.assertIsInstance(findings[0], ShardFinding)
        self.assertEqual(findings[0].severity, "Critical")
        self.assertEqual(findings[0].message, "Database public IP enabled")
        self.assertEqual(findings[0].path, "main.tf")
        self.assertEqual(findings[0].line, 42)
        self.assertEqual(findings[1].severity, "High")
        self.assertIsNone(findings[1].path)

    # ------------------------------------------------------------------ #
    # Test 2: Missing findings does not crash                              #
    # ------------------------------------------------------------------ #
    def test_missing_findings_no_crash(self):
        entry = {"task": "add feature", "timestamp": "2026-04-13T00:00:00Z"}
        findings = _extract_findings(entry)
        self.assertEqual(findings, [])

    # ------------------------------------------------------------------ #
    # Test 3: final_report.findings renders in full SHARD receipt         #
    # ------------------------------------------------------------------ #
    def test_final_report_findings_render_in_full_receipt(self):
        entry = {
            "task": "IaC review",
            "timestamp": "2026-05-01T00:00:00Z",
            "final_report": {
                "findings": [
                    {"severity": "High", "message": "Bucket ACLs enabled"},
                ],
                "warnings": [],
            },
        }
        receipt = build_shard_receipt(entry)
        out = render_full_shard_receipt(receipt)
        self.assertIn("FINDINGS", out)
        self.assertIn("Bucket ACLs enabled", out)

    # ------------------------------------------------------------------ #
    # Test 4: final_report.warnings render as Note findings               #
    # ------------------------------------------------------------------ #
    def test_final_report_warnings_render_as_note(self):
        entry = {
            "task": "review",
            "timestamp": "2026-05-01T00:00:00Z",
            "final_report": {
                "warnings": ["Missing lifecycle rules on storage buckets"],
            },
        }
        findings = _extract_findings(entry)
        self.assertEqual(len(findings), 1)
        self.assertEqual(findings[0].severity, "Note")
        self.assertEqual(findings[0].message, "Missing lifecycle rules on storage buckets")

    # ------------------------------------------------------------------ #
    # Test 5: Findings appear after CHECKS and before CHANGES             #
    # ------------------------------------------------------------------ #
    def test_findings_section_position_in_full_receipt(self):
        entry = {
            "task": "IaC review",
            "timestamp": "2026-05-01T00:00:00Z",
            "findings": [{"severity": "Medium", "message": "Missing tags"}],
        }
        receipt = build_shard_receipt(entry)
        out = render_full_shard_receipt(receipt)
        checks_idx = out.index("CHECKS")
        findings_idx = out.index("FINDINGS")
        changes_idx = out.index("CHANGES")
        self.assertLess(checks_idx, findings_idx)
        self.assertLess(findings_idx, changes_idx)

    # ------------------------------------------------------------------ #
    # Test 6: Findings grouped in severity order (Critical before High)   #
    # ------------------------------------------------------------------ #
    def test_findings_grouped_by_severity_order(self):
        entry = {
            "task": "review",
            "timestamp": "2026-05-01T00:00:00Z",
            "findings": [
                {"severity": "Medium", "message": "Missing tags"},
                {"severity": "Critical", "message": "Public IP enabled"},
                {"severity": "High", "message": "Bucket ACLs"},
            ],
        }
        receipt = build_shard_receipt(entry)
        out = render_full_shard_receipt(receipt)
        critical_idx = out.index("CRITICAL")
        high_idx = out.index("HIGH")
        medium_idx = out.index("MEDIUM")
        self.assertLess(critical_idx, high_idx)
        self.assertLess(high_idx, medium_idx)

    # ------------------------------------------------------------------ #
    # Test 7: Severity icons — Critical ✖, High ⚠, Medium ~              #
    # ------------------------------------------------------------------ #
    def test_severity_icons_present(self):
        entry = {
            "task": "review",
            "timestamp": "2026-05-01T00:00:00Z",
            "findings": [
                {"severity": "Critical", "message": "Public IP"},
                {"severity": "High", "message": "ACLs"},
                {"severity": "Medium", "message": "Tags"},
            ],
        }
        receipt = build_shard_receipt(entry)
        out = render_full_shard_receipt(receipt)
        self.assertIn("✖", out)
        self.assertIn("⚠", out)
        self.assertIn("~", out)

    # ------------------------------------------------------------------ #
    # Test 8: No fake findings rendered when entry has none               #
    # ------------------------------------------------------------------ #
    def test_no_fake_findings_when_entry_has_none(self):
        receipt = build_shard_receipt(_minimal_entry())
        out = render_full_shard_receipt(receipt)
        self.assertIn("FINDINGS", out)
        self.assertIn("No structured findings recorded.", out)
        self.assertNotIn("✖", out)
        self.assertNotIn("⚠", out)

    # ------------------------------------------------------------------ #
    # Test 9: Raw internal terms not introduced in findings output        #
    # ------------------------------------------------------------------ #
    def test_no_internal_terms_in_findings_output(self):
        entry = {
            "task": "review",
            "timestamp": "2026-05-01T00:00:00Z",
            "findings": [{"severity": "High", "message": "Open port found"}],
        }
        receipt = build_shard_receipt(entry)
        out = render_full_shard_receipt(receipt)
        forbidden = (
            "candidate",
            "routing advisory",
            "verification loop",
            "plan ledger",
            "internal form factor",
        )
        for term in forbidden:
            self.assertNotIn(term, out.lower(), f"Found forbidden term: {term!r}")

    # ------------------------------------------------------------------ #
    # Test 10: Compact receipt renders findings grouped by severity       #
    # ------------------------------------------------------------------ #
    def test_compact_receipt_renders_with_findings(self):
        entry = {
            "task": "IaC audit",
            "timestamp": "2026-05-01T00:00:00Z",
            "findings": [
                {"severity": "Critical", "message": "Public IP enabled"},
                {"severity": "High", "message": "Bucket ACLs"},
                {"severity": "High", "message": "Logging disabled"},
            ],
        }
        receipt = build_shard_receipt(entry)
        out = render_compact_shard_receipt(receipt)
        self.assertIn("RECEIPT", out)
        self.assertIn("FINDINGS", out)
        self.assertIn("Public IP enabled", out)
        self.assertIn("Bucket ACLs", out)
        # Critical (higher priority) must appear before High in the sorted output
        self.assertLess(out.index("Public IP enabled"), out.index("Bucket ACLs"))

    # ------------------------------------------------------------------ #
    # Test 11: Full receipt uses polished separator style                 #
    # ------------------------------------------------------------------ #
    def test_full_receipt_separator_style_unchanged(self):
        receipt = build_shard_receipt(_rich_native_entry(), index=0)
        out = render_full_shard_receipt(receipt)
        self.assertIn("━", out)
        self.assertIn("  SHARD —", out)
        self.assertIn("  RECEIPT", out)

    # ------------------------------------------------------------------ #
    # Test 12: Old run-history entries still render                       #
    # ------------------------------------------------------------------ #
    def test_old_entries_render_without_crash(self):
        old_entry = {
            "task": "legacy run from v0.1",
            "timestamp": "2025-01-01T00:00:00Z",
            "execution_model": "anthropic/claude-3-opus",
            "summary": "Completed.",
        }
        receipt = build_shard_receipt(old_entry, index=0)
        compact = render_compact_shard_receipt(receipt)
        full = render_full_shard_receipt(receipt)
        self.assertIn("RECEIPT", compact)
        self.assertIn("SHARD", full)
        self.assertIn("No structured findings recorded.", full)
        self.assertEqual(receipt.findings, [])
        self.assertEqual(receipt.agent_notes, [])

    # ------------------------------------------------------------------ #
    # Bonus: agent_notes preserved separately from findings               #
    # ------------------------------------------------------------------ #
    def test_agent_notes_preserved_on_receipt(self):
        entry = {
            "task": "review",
            "timestamp": "2026-05-01T00:00:00Z",
            "agent_notes": ["Observed elevated network activity", "Terraform plan generated successfully"],
        }
        receipt = build_shard_receipt(entry)
        self.assertEqual(receipt.agent_notes, ["Observed elevated network activity", "Terraform plan generated successfully"])
        self.assertEqual(len(receipt.findings), 2)
        self.assertTrue(all(f.severity == "Note" for f in receipt.findings))

    def test_plan_warnings_render_as_note_not_medium(self):
        entry = {
            "task": "review",
            "timestamp": "2026-05-01T00:00:00Z",
            "plan": {"intent": "implementation", "risk": "medium", "suggested_steps": [], "warnings": ["Large change scope detected"]},
        }
        findings = _extract_findings(entry)
        self.assertEqual(len(findings), 1)
        self.assertEqual(findings[0].severity, "Note")


class TestContextProvenancePolish(unittest.TestCase):
    def test_minimal_entry_context_not_recorded(self):
        receipt = build_shard_receipt(_minimal_entry())
        out = render_full_shard_receipt(receipt)
        self.assertIn("CONTEXT", out)
        self.assertIsNone(receipt.repo)
        self.assertIsNone(receipt.branch)
        self.assertIsNone(receipt.context_quality)
        self.assertIsNone(receipt.files_read_count)
        self.assertEqual(receipt.inspected_files, [])
        self.assertIn("Not recorded", out)

    def test_context_quality_level_mapping(self):
        for level, expected in [("good", "Good"), ("strong", "Good"), ("fair", "Partial"), ("weak", "Weak")]:
            entry = dict(_minimal_entry())
            entry["context_quality_score"] = {"level": level}
            receipt = build_shard_receipt(entry)
            self.assertEqual(receipt.context_quality, expected, f"level={level!r}")

    def test_context_quality_unknown_gives_not_recorded(self):
        for level in ("unknown", "", None):
            entry = dict(_minimal_entry())
            entry["context_quality_score"] = {"level": level}
            receipt = build_shard_receipt(entry)
            out = render_full_shard_receipt(receipt)
            self.assertIsNone(receipt.context_quality, f"level={level!r}")
            quality_line = next(ln for ln in out.splitlines() if "Quality" in ln)
            self.assertIn("Not recorded", quality_line)

    def test_inspected_files_from_file_context_paths(self):
        entry = dict(_minimal_entry())
        entry["file_context"] = {"files_read": 2, "paths": ["openshard/cli/main.py", "tests/test_shard_contract.py"]}
        receipt = build_shard_receipt(entry)
        out = render_full_shard_receipt(receipt)
        self.assertIn("INSPECTED FILES", out)
        self.assertIn("openshard/cli/main.py", out)
        self.assertIn("tests/test_shard_contract.py", out)

    def test_inspected_files_capped_at_10(self):
        entry = dict(_minimal_entry())
        paths = [f"file_{i}.py" for i in range(15)]
        entry["file_context"] = {"files_read": 15, "paths": paths}
        receipt = build_shard_receipt(entry)
        out = render_full_shard_receipt(receipt)
        self.assertIn("(+5 more)", out)
        self.assertIn("file_0.py", out)
        self.assertNotIn("file_10.py", out)

    def test_diff_review_changed_files_not_in_inspected_files(self):
        entry = dict(_minimal_entry())
        entry["diff_review"] = {"changed_files": ["a.py", "b.py"]}
        receipt = build_shard_receipt(entry)
        out = render_full_shard_receipt(receipt)
        self.assertEqual(receipt.inspected_files, [])
        inspected_idx = out.index("INSPECTED FILES")
        policy_idx = out.index("POLICY")
        inspected_block = out[inspected_idx:policy_idx]
        self.assertNotIn("a.py", inspected_block)
        self.assertIn("Not recorded", inspected_block)

    def test_diff_review_used_as_touched_fallback(self):
        entry = dict(_minimal_entry())
        entry["diff_review"] = {"changed_files": ["x.py", "y.py"]}
        receipt = build_shard_receipt(entry)
        self.assertEqual(receipt.files_touched, ["x.py", "y.py"])
        out = render_full_shard_receipt(receipt)
        touched_line = next(ln for ln in out.splitlines() if "Touched" in ln)
        self.assertIn("2 files", touched_line)

    def test_touched_renders_as_count(self):
        receipt = build_shard_receipt(_rich_native_entry())
        out = render_full_shard_receipt(receipt)
        touched_line = next(ln for ln in out.splitlines() if "Touched" in ln)
        self.assertIn("3 files", touched_line)
        self.assertNotIn(",", touched_line)

    def test_repo_set_from_workspace_path(self):
        entry = dict(_minimal_entry())
        entry["workspace_path"] = r"C:\Users\Michael\openshard"
        receipt = build_shard_receipt(entry)
        self.assertEqual(receipt.repo, "openshard")
        out = render_full_shard_receipt(receipt)
        self.assertIn("openshard", out)

    def test_git_state_from_observation(self):
        for dirty, expected in [(True, "Changes pending"), (False, "Clean")]:
            entry = dict(_minimal_entry())
            entry["observation"] = {"dirty_diff_present": dirty}
            receipt = build_shard_receipt(entry)
            self.assertEqual(receipt.git_state, expected, f"dirty={dirty!r}")
            out = render_full_shard_receipt(receipt)
            git_state_line = next(ln for ln in out.splitlines() if "Git state" in ln)
            self.assertIn(expected, git_state_line)

    def test_git_state_absent_gives_not_recorded(self):
        receipt = build_shard_receipt(_minimal_entry())
        out = render_full_shard_receipt(receipt)
        git_state_line = next(ln for ln in out.splitlines() if "Git state" in ln)
        self.assertIn("Not recorded", git_state_line)

    def test_no_internal_terms_in_context_rendering(self):
        entry = dict(_rich_native_entry())
        entry["workspace_path"] = r"C:\Users\Michael\openshard"
        entry["context_quality_score"] = {"level": "good"}
        entry["observation"] = {"dirty_diff_present": True}
        entry["file_context"] = {"files_read": 3, "paths": ["a.py", "b.py", "c.py"]}
        receipt = build_shard_receipt(entry)
        out = render_full_shard_receipt(receipt)
        forbidden = (
            "routing advisory",
            "plan ledger",
            "context_quality_score",
            "dirty_diff_present",
            "snippet_files",
            "file_context",
            "native context packet",
        )
        for term in forbidden:
            self.assertNotIn(term, out.lower(), f"Found forbidden term: {term!r}")

    def test_compact_receipt_unchanged(self):
        receipt = build_shard_receipt(_rich_native_entry())
        out = render_compact_shard_receipt(receipt)
        self.assertNotIn("CONTEXT", out)
        self.assertNotIn("INSPECTED FILES", out)
        self.assertIn("RECEIPT", out)

    def test_findings_and_inspected_files_coexist(self):
        entry = dict(_rich_native_entry())
        entry["findings"] = [{"severity": "High", "message": "Exposed credentials"}]
        entry["file_context"] = {"files_read": 1, "paths": ["secrets.py"]}
        receipt = build_shard_receipt(entry)
        out = render_full_shard_receipt(receipt)
        self.assertIn("FINDINGS", out)
        self.assertIn("Exposed credentials", out)
        self.assertIn("CONTEXT", out)
        self.assertIn("INSPECTED FILES", out)
        self.assertIn("secrets.py", out)

    def test_section_order_in_full_receipt(self):
        entry = dict(_rich_native_entry())
        entry["workspace_path"] = r"C:\Users\Michael\openshard"
        entry["context_quality_score"] = {"level": "good"}
        entry["observation"] = {"dirty_diff_present": True}
        entry["file_context"] = {"files_read": 2, "paths": ["a.py", "b.py"]}
        receipt = build_shard_receipt(entry)
        out = render_full_shard_receipt(receipt)
        idx = {s: out.index(s) for s in ["CONTEXT", "INSPECTED FILES", "POLICY", "CHECKS", "FINDINGS", "CHANGES"]}
        self.assertLess(idx["CONTEXT"], idx["INSPECTED FILES"])
        self.assertLess(idx["INSPECTED FILES"], idx["POLICY"])
        self.assertLess(idx["POLICY"], idx["CHECKS"])
        self.assertLess(idx["CHECKS"], idx["FINDINGS"])
        self.assertLess(idx["FINDINGS"], idx["CHANGES"])


class TestCompactReceiptFindingsGrouped(unittest.TestCase):
    """Compact receipt shows findings under FINDINGS section, sorted by severity."""

    def test_findings_shown_in_findings_section(self):
        entry = dict(_rich_native_entry())
        entry["findings"] = [
            {"severity": "High", "message": "Exposed port 22"},
            {"severity": "Critical", "message": "Root access enabled"},
        ]
        receipt = build_shard_receipt(entry)
        out = render_compact_shard_receipt(receipt)
        self.assertIn("FINDINGS", out)
        self.assertIn("Root access enabled", out)
        self.assertIn("Exposed port 22", out)
        # Critical must appear before High (sorted by severity priority)
        self.assertLess(out.index("Root access enabled"), out.index("Exposed port 22"))

    def test_no_findings_row_when_empty(self):
        receipt = build_shard_receipt(_minimal_entry())
        out = render_compact_shard_receipt(receipt)
        self.assertNotIn("Findings", out)
        self.assertNotIn("No structured findings", out)


class TestCompactReceiptWarningLine(unittest.TestCase):
    """Compact receipt shows a warning line only for explicit blocker keywords."""

    def test_warning_line_shown_for_do_not_run_note(self):
        entry = dict(_rich_native_entry())
        entry["agent_notes"] = ["DO NOT RUN terraform apply — destructive changes detected"]
        receipt = build_shard_receipt(entry)
        out = render_compact_shard_receipt(receipt)
        self.assertIn("DO NOT RUN", out)

    def test_warning_line_absent_for_normal_note(self):
        entry = dict(_rich_native_entry())
        entry["agent_notes"] = ["Review recommended before applying changes"]
        receipt = build_shard_receipt(entry)
        out = render_compact_shard_receipt(receipt)
        # Normal note appears in Findings section (as Note-severity finding) but
        # must NOT appear as a standalone warning block after the Result row.
        lines = out.split("\n")
        result_idx = next((i for i, ln in enumerate(lines) if "Result" in ln), None)
        self.assertIsNotNone(result_idx, "Result row should be in compact receipt")
        after_result = "\n".join(lines[result_idx + 1:])
        self.assertNotIn("Review recommended", after_result)


class TestCompactReceiptExactFormat(unittest.TestCase):
    """Golden-format tests: exact separator width, field set, and FINDINGS section."""

    _SEP = "━" * 40

    def _make_receipt(self, **kwargs) -> ShardReceipt:
        defaults = dict(
            shard_id="shard-20260519-0042",
            created_at="2026-05-19T00:00:00Z",
            task_short="Review Terraform networking change",
            task_full="Review Terraform networking change",
            agent="OpenShard Native",
            strategy="Not recorded",
            model_display="claude-sonnet-4-5",
            risk="High",
            sandbox="On",
            files_changed=0,
            checks_display="3/3 passed",
            approval="Required → Granted",
            cost_display="$0.0041",
            result="3 risks flagged",
            status="Passed",
            duration_seconds=None,
        )
        defaults.update(kwargs)
        return ShardReceipt(**defaults)

    def test_separator_width_is_40(self):
        out = render_compact_shard_receipt(self._make_receipt())
        sep_char = "━"
        for line in out.splitlines():
            stripped = line.strip()
            if stripped and all(c == sep_char for c in stripped):
                self.assertEqual(len(stripped), 40, f"Separator has wrong width: {stripped!r}")

    def test_no_rich_box_title(self):
        out = render_compact_shard_receipt(self._make_receipt())
        self.assertNotIn("╭─ OpenShard Receipt", out)
        self.assertNotIn("╰", out)

    def test_strategy_absent(self):
        out = render_compact_shard_receipt(self._make_receipt())
        self.assertNotIn("Strategy", out)

    def test_all_required_fields_present(self):
        out = render_compact_shard_receipt(self._make_receipt())
        for field in ("Task", "Executor", "Model", "Risk", "Sandbox",
                      "Changed", "Checks", "Approval", "Cost", "Result"):
            self.assertIn(field, out, f"Required field {field!r} missing from compact receipt")

    def test_findings_section_rendered_when_present(self):
        findings = [
            ShardFinding(severity="High", message="Route table removes NAT gateway fallback"),
            ShardFinding(severity="High", message="Security group opens 443 to 0.0.0.0/0"),
        ]
        out = render_compact_shard_receipt(self._make_receipt(findings=findings))
        self.assertIn("FINDINGS", out)
        self.assertIn("Route table removes NAT gateway fallback", out)
        self.assertIn("Security group opens 443 to 0.0.0.0/0", out)

    def test_findings_use_warning_icon(self):
        findings = [ShardFinding(severity="High", message="some risk")]
        out = render_compact_shard_receipt(self._make_receipt(findings=findings))
        self.assertTrue("⚠" in out or "!" in out)

    def test_findings_section_absent_when_empty(self):
        out = render_compact_shard_receipt(self._make_receipt(findings=[]))
        self.assertNotIn("FINDINGS", out)

    def test_golden_snapshot(self):
        """Full string comparison against the canonical sample receipt."""
        findings = [
            ShardFinding(severity="High", message="Route table removes NAT gateway fallback — connectivity risk"),
            ShardFinding(severity="High", message="Security group opens 443 to 0.0.0.0/0 — scope to known CIDRs"),
            ShardFinding(severity="High", message="Subnet CIDR overlaps existing range in variables.tf"),
        ]
        out = render_compact_shard_receipt(self._make_receipt(findings=findings))
        sep = self._SEP
        expected = "\n".join([
            sep,
            "  RECEIPT — shard-20260519-0042",
            sep,
            "  Task        Review Terraform networking change",
            "  Executor    OpenShard Native",
            "  Model       claude-sonnet-4-5",
            "  Risk        High",
            "  Sandbox     On",
            "  Changed     0 files",
            "  Checks      3/3 passed",
            "  Approval    Required → Granted",
            "  Cost        $0.0041",
            "  Result      3 risks flagged",
            sep,
            "  FINDINGS",
            "  ⚠  Route table removes NAT gateway fallback — connectivity risk",
            "  ⚠  Security group opens 443 to 0.0.0.0/0 — scope to known CIDRs",
            "  ⚠  Subnet CIDR overlaps existing range in variables.tf",
            sep,
        ])
        self.assertEqual(out, expected)


class TestBuildLiveRunReceipt(unittest.TestCase):
    """Tests for the public build_live_run_receipt helper."""

    def _build(self, **kwargs):
        defaults = dict(
            task="Review Terraform networking change",
            run_id="2026-05-19T00:42:00Z",
            run_index=41,
            agent="OpenShard Native",
            stage_runs=[],
            routing_model="anthropic/claude-sonnet-4-6",
            risk="High",
            sandbox="On",
            files_changed=0,
            verification_attempted=True,
            verification_passed=True,
            approval="Required → Granted",
            estimated_cost=0.0041,
            result_summary="3 risks flagged",
        )
        defaults.update(kwargs)
        return build_live_run_receipt(**defaults)

    def test_shard_id_uses_run_index(self):
        r = self._build(run_id="2026-05-19T00:00:00Z", run_index=41)
        self.assertEqual(r.shard_id, "shard-20260519-0042")

    def test_checks_passed(self):
        r = self._build(verification_attempted=True, verification_passed=True)
        self.assertEqual(r.checks_display, "1/1 passed")

    def test_checks_failed(self):
        r = self._build(verification_attempted=True, verification_passed=False)
        self.assertEqual(r.checks_display, "0/1 passed")

    def test_checks_not_run(self):
        r = self._build(verification_attempted=False, verification_passed=None)
        self.assertEqual(r.checks_display, "Not run")

    def test_cost_display(self):
        r = self._build(estimated_cost=0.0041)
        self.assertEqual(r.cost_display, "$0.0041")

    def test_cost_not_recorded_when_none(self):
        r = self._build(estimated_cost=None)
        self.assertEqual(r.cost_display, "Not recorded")

    def test_renders_without_rich_box(self):
        r = self._build()
        out = render_compact_shard_receipt(r)
        self.assertNotIn("╭─ OpenShard Receipt", out)
        self.assertIn("RECEIPT", out)


class TestResultDisplay(unittest.TestCase):
    """Tests for _result_display: short, complete result lines, no mid-sentence truncation."""

    def test_short_summary_returned_as_is(self):
        self.assertEqual(_result_display("11 risks flagged."), "11 risks flagged.")

    def test_short_summary_no_period_returned_as_is(self):
        self.assertEqual(_result_display("Review completed"), "Review completed")

    def test_empty_summary_returns_not_recorded(self):
        self.assertEqual(_result_display(""), "Not recorded")

    def test_none_equivalent_whitespace_returns_not_recorded(self):
        self.assertEqual(_result_display("   "), "Not recorded")

    def test_first_sentence_used_when_fits(self):
        summary = "Review completed. 3 files changed in the network module."
        result = _result_display(summary)
        self.assertEqual(result, "Review completed.")

    def test_semicolon_sentence_boundary(self):
        summary = "Review completed; 1 file changed. Additional details follow."
        result = _result_display(summary)
        self.assertIn("Review completed", result)
        self.assertNotIn("Additional details", result)

    def test_long_summary_no_ellipsis(self):
        long = (
            "Produced a comprehensive Terraform hardening review document covering "
            "security/compliance posture, operability, and risk findings."
        )
        result = _result_display(long)
        self.assertFalse(result.endswith("…"), f"Result ends with ellipsis: {result!r}")
        self.assertFalse(result.endswith("..."), f"Result ends with dots: {result!r}")

    def test_long_summary_does_not_end_with_connective(self):
        long = (
            "Produced a comprehensive Terraform hardening review document covering "
            "security/compliance posture, operability, and risk findings."
        )
        result = _result_display(long)
        lower = result.lower().rstrip()
        for bad_end in (" and", " or", " with", " for", " covering", " including", " a", " the"):
            self.assertFalse(
                lower.endswith(bad_end),
                f"Result ends with connective {bad_end!r}: {result!r}",
            )

    def test_result_within_length_limit(self):
        long = "A" * 200
        result = _result_display(long)
        self.assertLessEqual(len(result), 65)

    def test_multiline_summary_uses_first_line_only(self):
        summary = "Review completed.\nLine two is ignored."
        result = _result_display(summary)
        self.assertNotIn("Line two", result)

    def test_result_not_empty_for_long_summary(self):
        long = "word " * 30
        result = _result_display(long)
        self.assertTrue(len(result) > 0)

    def test_compact_receipt_result_row_no_ellipsis(self):
        """Result row in the rendered compact receipt never ends with …"""
        long_summary = (
            "Produced a comprehensive Terraform hardening review document covering "
            "security/compliance posture, 2am operability, and risk findings."
        )
        entry = {
            "task": "audit",
            "timestamp": "2026-05-22T00:00:00Z",
            "summary": long_summary,
        }
        receipt = build_shard_receipt(entry)
        out = render_compact_shard_receipt(receipt)
        result_line = next((ln for ln in out.splitlines() if "Result" in ln), None)
        self.assertIsNotNone(result_line, "Result row missing from compact receipt")
        self.assertFalse(result_line.rstrip().endswith("…"), f"Result row ends with ellipsis: {result_line!r}")
        self.assertFalse(result_line.rstrip().endswith("..."), f"Result row ends with dots: {result_line!r}")

    def test_compact_receipt_rich_box_absent(self):
        receipt = build_shard_receipt({"task": "t", "timestamp": "2026-05-22T00:00:00Z"})
        out = render_compact_shard_receipt(receipt)
        self.assertNotIn("╭─ OpenShard Receipt", out)

    def test_compact_receipt_findings_absent_when_none(self):
        entry = {"task": "t", "timestamp": "2026-05-22T00:00:00Z"}
        receipt = build_shard_receipt(entry)
        out = render_compact_shard_receipt(receipt)
        self.assertNotIn("FINDINGS", out)

    def test_compact_receipt_findings_present_when_structured(self):
        entry = {
            "task": "IaC audit",
            "timestamp": "2026-05-22T00:00:00Z",
            "findings": [{"severity": "High", "message": "Open port 22 detected"}],
        }
        receipt = build_shard_receipt(entry)
        out = render_compact_shard_receipt(receipt)
        self.assertIn("FINDINGS", out)
        self.assertIn("Open port 22 detected", out)


class TestBuildShardReceiptStageRunsFallback(unittest.TestCase):
    """build_shard_receipt falls back to stage_runs for model and cost when top-level fields are absent."""

    def _entry_with_stage_runs(self, model="anthropic/claude-sonnet-4-6", cost=0.0123):
        return {
            "task": "review this codebase",
            "timestamp": "2026-05-22T00:00:00Z",
            "stage_runs": [
                {"stage_type": "implementation", "model": model, "cost": cost, "duration": 5.0},
            ],
        }

    def test_model_from_stage_runs_when_no_top_level_model(self):
        receipt = build_shard_receipt(self._entry_with_stage_runs())
        self.assertNotEqual(receipt.model_display, "Not recorded")
        self.assertIn("Claude Sonnet 4.6", receipt.model_display)

    def test_cost_from_stage_runs_when_no_estimated_cost(self):
        receipt = build_shard_receipt(self._entry_with_stage_runs(cost=0.0123))
        self.assertNotEqual(receipt.cost_display, "Not recorded")
        self.assertIn("0.0123", receipt.cost_display)

    def test_cost_summed_across_multiple_stage_runs(self):
        entry = {
            "task": "review",
            "timestamp": "2026-05-22T00:00:00Z",
            "stage_runs": [
                {"stage_type": "planning", "model": "anthropic/claude-sonnet-4-6", "cost": 0.01, "duration": 2.0},
                {"stage_type": "implementation", "model": "anthropic/claude-sonnet-4-6", "cost": 0.02, "duration": 3.0},
            ],
        }
        receipt = build_shard_receipt(entry)
        self.assertIn("0.0300", receipt.cost_display)

    def test_model_not_recorded_when_stage_runs_empty(self):
        entry = {"task": "t", "timestamp": "2026-05-22T00:00:00Z", "stage_runs": []}
        receipt = build_shard_receipt(entry)
        self.assertEqual(receipt.model_display, "Not recorded")

    def test_is_review_task_result_when_no_findings(self):
        entry = {
            "task": "review iac",
            "timestamp": "2026-05-22T00:00:00Z",
            "is_review_task": True,
        }
        receipt = build_shard_receipt(entry)
        self.assertEqual(receipt.result, "Review completed.")

    def test_findings_count_result_when_findings_present(self):
        entry = {
            "task": "review iac",
            "timestamp": "2026-05-22T00:00:00Z",
            "is_review_task": True,
            "findings": [
                {"severity": "Critical", "message": "Public bucket"},
                {"severity": "High", "message": "Wildcard IAM"},
            ],
        }
        receipt = build_shard_receipt(entry)
        self.assertIn("2 issues", receipt.result)


# ---------------------------------------------------------------------------
# Run timeline storage and rendering
# ---------------------------------------------------------------------------

def _tl_event(event: str, label: str, kind: str = "run", status: str = "completed") -> dict:
    return {"event": event, "label": label, "kind": kind, "status": status}


class TestRunTimeline(unittest.TestCase):

    def test_build_receipt_populates_timeline(self):
        entry = {**_minimal_entry(), "run_timeline": [_tl_event("run_started", "Started run")]}
        receipt = build_shard_receipt(entry)
        self.assertEqual(len(receipt.run_timeline), 1)
        self.assertEqual(receipt.run_timeline[0]["label"], "Started run")

    def test_build_receipt_empty_when_key_absent(self):
        receipt = build_shard_receipt(_minimal_entry())
        self.assertEqual(receipt.run_timeline, [])

    def test_build_receipt_drops_invalid_events(self):
        entry = {**_minimal_entry(), "run_timeline": [
            _tl_event("x", "Good event"),
            "not a dict",
            {"event": "y"},          # missing label
            {"label": ""},            # empty label
        ]}
        receipt = build_shard_receipt(entry)
        self.assertEqual(len(receipt.run_timeline), 1)
        self.assertEqual(receipt.run_timeline[0]["label"], "Good event")

    def test_render_full_shows_timeline_section(self):
        entry = {**_minimal_entry(), "run_timeline": [_tl_event("repo_scanned", "Scanned repo", "scan")]}
        receipt = build_shard_receipt(entry)
        out = render_full_shard_receipt(receipt)
        self.assertIn("TIMELINE", out)
        self.assertIn("Scanned repo", out)

    def test_render_full_receipt_saved_from_stored_timeline(self):
        entry = {**_minimal_entry(), "run_timeline": [
            _tl_event("repo_scanned", "Scanned repo", "scan"),
            _tl_event("receipt_saved", "Saved Shard receipt", "receipt"),
        ]}
        receipt = build_shard_receipt(entry)
        out = render_full_shard_receipt(receipt)
        self.assertIn("Saved Shard receipt", out)
        self.assertEqual(out.count("Saved Shard receipt"), 1)

    def test_render_full_no_timeline_section_when_empty(self):
        receipt = build_shard_receipt(_minimal_entry())
        out = render_full_shard_receipt(receipt)
        self.assertNotIn("TIMELINE", out)

    def test_render_full_failed_event_gets_x_symbol(self):
        entry = {**_minimal_entry(), "run_timeline": [
            _tl_event("model_response_received", "Model request failed", "model", "failed"),
        ]}
        receipt = build_shard_receipt(entry)
        out = render_full_shard_receipt(receipt)
        self.assertTrue("✖" in out or "x" in out)
        self.assertIn("Model request failed", out)

    def test_build_live_receipt_accepts_timeline(self):
        tl = [_tl_event("run_started", "Started run")]
        receipt = build_live_run_receipt(
            task="test task",
            run_id="2026-05-23T00:00:00Z",
            run_index=None,
            agent="OpenShard",
            stage_runs=[],
            routing_model=None,
            risk="Not recorded",
            sandbox="Not recorded",
            files_changed=0,
            verification_attempted=None,
            verification_passed=None,
            approval="Not recorded",
            estimated_cost=None,
            result_summary="Done.",
            run_timeline=tl,
        )
        self.assertEqual(len(receipt.run_timeline), 1)
        self.assertEqual(receipt.run_timeline[0]["label"], "Started run")


class TestRepoFromEntryFields(unittest.TestCase):
    """repo_name field takes priority; workspace_path is a folder-name fallback."""

    def test_repo_name_field_used_when_present(self):
        entry = dict(_minimal_entry())
        entry["repo_name"] = "harbourdocs-infra"
        receipt = build_shard_receipt(entry)
        self.assertEqual(receipt.repo, "harbourdocs-infra")

    def test_workspace_path_falls_back_to_folder_name(self):
        entry = dict(_minimal_entry())
        entry["workspace_path"] = "/home/user/openshard"
        receipt = build_shard_receipt(entry)
        self.assertEqual(receipt.repo, "openshard")

    def test_workspace_path_windows_style_folder_name(self):
        entry = dict(_minimal_entry())
        entry["workspace_path"] = r"C:\Users\Michael\harbourdocs-infra"
        receipt = build_shard_receipt(entry)
        self.assertEqual(receipt.repo, "harbourdocs-infra")

    def test_repo_name_takes_priority_over_workspace_path(self):
        entry = dict(_minimal_entry())
        entry["repo_name"] = "my-repo"
        entry["workspace_path"] = "/home/user/other-folder"
        receipt = build_shard_receipt(entry)
        self.assertEqual(receipt.repo, "my-repo")

    def test_no_fields_gives_none(self):
        receipt = build_shard_receipt(_minimal_entry())
        self.assertIsNone(receipt.repo)

    def test_full_path_never_leaked_into_repo(self):
        entry = dict(_minimal_entry())
        entry["workspace_path"] = "/home/user/my-project"
        receipt = build_shard_receipt(entry)
        self.assertNotIn("/", receipt.repo or "")
        self.assertNotIn("\\", receipt.repo or "")


class TestBranchFromEntry(unittest.TestCase):
    """branch is read from the git_branch key in the log entry."""

    def test_branch_read_from_entry(self):
        entry = dict(_minimal_entry())
        entry["git_branch"] = "main"
        receipt = build_shard_receipt(entry)
        self.assertEqual(receipt.branch, "main")

    def test_branch_feature_branch(self):
        entry = dict(_minimal_entry())
        entry["git_branch"] = "feat/context-inspected-files-accuracy"
        receipt = build_shard_receipt(entry)
        self.assertEqual(receipt.branch, "feat/context-inspected-files-accuracy")

    def test_branch_absent_gives_none(self):
        receipt = build_shard_receipt(_minimal_entry())
        self.assertIsNone(receipt.branch)

    def test_branch_appears_in_render(self):
        entry = dict(_minimal_entry())
        entry["git_branch"] = "feat/my-feature"
        receipt = build_shard_receipt(entry)
        out = render_full_shard_receipt(receipt)
        self.assertIn("feat/my-feature", out)

    def test_branch_absent_shows_not_recorded(self):
        receipt = build_shard_receipt(_minimal_entry())
        out = render_full_shard_receipt(receipt)
        branch_line = next(ln for ln in out.splitlines() if "Branch" in ln)
        self.assertIn("Not recorded", branch_line)


class TestGitStateFallback(unittest.TestCase):
    """git_dirty from the log entry is used when observation data is absent."""

    def test_git_dirty_true_gives_changes_pending(self):
        entry = dict(_minimal_entry())
        entry["git_dirty"] = True
        receipt = build_shard_receipt(entry)
        self.assertEqual(receipt.git_state, "Changes pending")

    def test_git_dirty_false_gives_clean(self):
        entry = dict(_minimal_entry())
        entry["git_dirty"] = False
        receipt = build_shard_receipt(entry)
        self.assertEqual(receipt.git_state, "Clean")

    def test_neither_source_gives_none(self):
        receipt = build_shard_receipt(_minimal_entry())
        self.assertIsNone(receipt.git_state)

    def test_observation_takes_priority_over_git_dirty(self):
        entry = dict(_minimal_entry())
        entry["observation"] = {"dirty_diff_present": False}
        entry["git_dirty"] = True
        receipt = build_shard_receipt(entry)
        self.assertEqual(receipt.git_state, "Clean")


class TestFilesReferencedFromFindings(unittest.TestCase):
    """files_referenced is derived from finding.path values in the log entry."""

    def test_finding_paths_populate_files_referenced(self):
        entry = dict(_minimal_entry())
        entry["findings"] = [
            {"severity": "Critical", "message": "root IAM", "path": "iam.tf"},
            {"severity": "High", "message": "no KMS", "path": "secrets.tf"},
        ]
        receipt = build_shard_receipt(entry)
        self.assertEqual(receipt.files_referenced, ["iam.tf", "secrets.tf"])

    def test_files_referenced_sorted(self):
        entry = dict(_minimal_entry())
        entry["findings"] = [
            {"severity": "High", "message": "b", "path": "z.tf"},
            {"severity": "High", "message": "a", "path": "a.tf"},
        ]
        receipt = build_shard_receipt(entry)
        self.assertEqual(receipt.files_referenced, ["a.tf", "z.tf"])

    def test_duplicate_paths_deduplicated(self):
        entry = dict(_minimal_entry())
        entry["findings"] = [
            {"severity": "High", "message": "x", "path": "iam.tf"},
            {"severity": "Medium", "message": "y", "path": "iam.tf"},
        ]
        receipt = build_shard_receipt(entry)
        self.assertEqual(receipt.files_referenced, ["iam.tf"])

    def test_findings_without_path_excluded(self):
        entry = dict(_minimal_entry())
        entry["findings"] = [
            {"severity": "Note", "message": "no path finding"},
        ]
        receipt = build_shard_receipt(entry)
        self.assertEqual(receipt.files_referenced, [])

    def test_inspected_files_unaffected_by_findings(self):
        entry = dict(_minimal_entry())
        entry["findings"] = [{"severity": "High", "message": "x", "path": "iam.tf"}]
        entry["file_context"] = {"files_read": 1, "paths": ["demo-task.md"]}
        receipt = build_shard_receipt(entry)
        self.assertEqual(receipt.inspected_files, ["demo-task.md"])
        self.assertEqual(receipt.files_referenced, ["iam.tf"])

    def test_default_files_referenced_is_empty(self):
        receipt = build_shard_receipt(_minimal_entry())
        self.assertEqual(receipt.files_referenced, [])


class TestFilesWithFindingsRender(unittest.TestCase):
    """FILES WITH FINDINGS section appears only when findings have file paths."""

    def test_section_present_when_files_referenced_non_empty(self):
        entry = dict(_minimal_entry())
        entry["findings"] = [{"severity": "High", "message": "x", "path": "iam.tf"}]
        receipt = build_shard_receipt(entry)
        out = render_full_shard_receipt(receipt)
        self.assertIn("FILES WITH FINDINGS", out)
        self.assertIn("iam.tf", out)

    def test_section_absent_when_no_finding_paths(self):
        entry = dict(_minimal_entry())
        entry["findings"] = [{"severity": "Note", "message": "general note"}]
        receipt = build_shard_receipt(entry)
        out = render_full_shard_receipt(receipt)
        self.assertNotIn("FILES WITH FINDINGS", out)

    def test_section_absent_when_no_findings(self):
        out = render_full_shard_receipt(build_shard_receipt(_minimal_entry()))
        self.assertNotIn("FILES WITH FINDINGS", out)

    def test_inspected_files_always_present(self):
        entry = dict(_minimal_entry())
        entry["findings"] = [{"severity": "High", "message": "x", "path": "iam.tf"}]
        out = render_full_shard_receipt(build_shard_receipt(entry))
        self.assertIn("INSPECTED FILES", out)

    def test_both_sections_simultaneously(self):
        entry = dict(_minimal_entry())
        entry["file_context"] = {"files_read": 1, "paths": ["demo-task.md"]}
        entry["findings"] = [{"severity": "Critical", "message": "root", "path": "iam.tf"}]
        receipt = build_shard_receipt(entry)
        out = render_full_shard_receipt(receipt)
        self.assertIn("INSPECTED FILES", out)
        self.assertIn("demo-task.md", out)
        self.assertIn("FILES WITH FINDINGS", out)
        self.assertIn("iam.tf", out)
        inspected_idx = out.index("INSPECTED FILES")
        findings_file_idx = out.index("FILES WITH FINDINGS")
        policy_idx = out.index("POLICY")
        self.assertLess(inspected_idx, findings_file_idx)
        self.assertLess(findings_file_idx, policy_idx)

    def test_files_not_swapped_between_sections(self):
        entry = dict(_minimal_entry())
        entry["file_context"] = {"files_read": 1, "paths": ["demo-task.md"]}
        entry["findings"] = [{"severity": "Critical", "message": "x", "path": "iam.tf"}]
        out = render_full_shard_receipt(build_shard_receipt(entry))
        inspected_block_end = out.index("FILES WITH FINDINGS")
        inspected_block = out[out.index("INSPECTED FILES"):inspected_block_end]
        self.assertIn("demo-task.md", inspected_block)
        self.assertNotIn("iam.tf", inspected_block)

    def test_cap_at_10_with_overflow(self):
        entry = dict(_minimal_entry())
        entry["findings"] = [
            {"severity": "High", "message": f"issue {i}", "path": f"file_{i:02d}.tf"}
            for i in range(12)
        ]
        receipt = build_shard_receipt(entry)
        out = render_full_shard_receipt(receipt)
        # Isolate the FILES WITH FINDINGS block only
        block_start = out.index("FILES WITH FINDINGS")
        block_end = out.index("POLICY")
        block = out[block_start:block_end]
        self.assertIn("(+2 more)", block)
        self.assertIn("file_00.tf", block)
        self.assertNotIn("file_10.tf", block)
