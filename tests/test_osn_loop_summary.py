"""Tests for the pipeline-level OSN loop summary (NativeOSNLoopStep / NativeOSNLoopSummary).

Distinct from test_osn_loop.py which tests the tool-level OSNLoopStep / OSNLoopMeta.
"""
from __future__ import annotations

import json
import unittest
from dataclasses import asdict
from unittest.mock import MagicMock, patch

from openshard.native.context import (
    NativeOSNLoopStep,
    NativeOSNLoopSummary,
    OSNLoopMeta,
)
from openshard.native.executor import (
    _LOOP_ALLOWED_TOOLS,
    NativeAgentExecutor,
    _is_safe_osn_tool_name,
)
from openshard.native.osn_loop_recorder import OsnLoopRecorder, should_enable_osn_recorder


def _make_executor(native_loop: str | None = "experimental"):
    fake_gen = MagicMock()
    fake_gen.generate.return_value = MagicMock(usage=None, files=[], summary="ok", notes=[])
    fake_gen.model = "mock-model"
    fake_gen.fixer_model = "mock-fixer"
    with patch("openshard.native.executor.ExecutionGenerator", return_value=fake_gen):
        executor = NativeAgentExecutor(provider=MagicMock(), native_loop=native_loop)
    return executor, fake_gen


# ---------------------------------------------------------------------------
# NativeOSNLoopStep dataclass
# ---------------------------------------------------------------------------

class TestNativeOSNLoopStep(unittest.TestCase):

    def test_defaults(self):
        step = NativeOSNLoopStep()
        self.assertEqual(step.step_index, 0)
        self.assertEqual(step.step_name, "")
        self.assertEqual(step.status, "pending")
        self.assertEqual(step.tool_name, "")
        self.assertEqual(step.reason, "")
        self.assertEqual(step.result_summary, "")
        self.assertFalse(step.context_injected)
        self.assertFalse(step.approval_required)
        self.assertEqual(step.verification_status, "")
        self.assertEqual(step.warnings, [])

    def test_serializes_via_asdict(self):
        step = NativeOSNLoopStep(
            step_index=1, step_name="preflight", status="passed",
            result_summary="repo_map_ok=True"
        )
        d = asdict(step)
        self.assertIsInstance(d, dict)
        self.assertEqual(d["step_name"], "preflight")
        self.assertEqual(d["status"], "passed")
        json.dumps(d)  # must not raise


# ---------------------------------------------------------------------------
# NativeOSNLoopSummary dataclass
# ---------------------------------------------------------------------------

class TestNativeOSNLoopSummary(unittest.TestCase):

    def test_defaults(self):
        summary = NativeOSNLoopSummary()
        self.assertFalse(summary.enabled)
        self.assertEqual(summary.mode, "")
        self.assertEqual(summary.max_steps, 11)
        self.assertEqual(summary.steps_taken, 0)
        self.assertFalse(summary.completed)
        self.assertEqual(summary.stopped_reason, "")
        self.assertEqual(summary.verification_status, "")
        self.assertFalse(summary.retry_used)
        self.assertFalse(summary.approval_required)
        self.assertFalse(summary.approval_granted)
        self.assertEqual(summary.warnings, [])
        self.assertEqual(summary.steps, [])

    def test_serializes_via_asdict(self):
        summary = NativeOSNLoopSummary(enabled=True, mode="experimental", steps_taken=3)
        d = asdict(summary)
        json.dumps(d)  # must not raise


# ---------------------------------------------------------------------------
# should_enable_osn_recorder
# ---------------------------------------------------------------------------

class TestShouldEnableOsnRecorder(unittest.TestCase):

    def test_experimental_returns_true(self):
        self.assertTrue(should_enable_osn_recorder("experimental"))

    def test_none_returns_false(self):
        self.assertFalse(should_enable_osn_recorder(None))

    def test_disabled_values_return_false(self):
        for val in ["off", "none", "disabled", "false", "0", "no", ""]:
            with self.subTest(val=val):
                self.assertFalse(should_enable_osn_recorder(val))

    def test_other_strings_return_false(self):
        for val in ["auto", "bounded", "true", "1", "yes", "on"]:
            with self.subTest(val=val):
                self.assertFalse(should_enable_osn_recorder(val))


# ---------------------------------------------------------------------------
# OsnLoopRecorder basics
# ---------------------------------------------------------------------------

class TestOsnLoopRecorderBasic(unittest.TestCase):

    def test_initial_summary(self):
        rec = OsnLoopRecorder()
        s = rec.summary
        self.assertTrue(s.enabled)
        self.assertEqual(s.mode, "experimental")
        self.assertEqual(s.max_steps, 11)
        self.assertEqual(s.steps_taken, 0)
        self.assertFalse(s.completed)

    def test_record_step_appends(self):
        rec = OsnLoopRecorder()
        rec.record_step("preflight", "passed", result_summary="ok")
        self.assertEqual(len(rec.summary.steps), 1)
        self.assertEqual(rec.summary.steps_taken, 1)
        self.assertEqual(rec.summary.steps[0].step_name, "preflight")
        self.assertEqual(rec.summary.steps[0].status, "passed")

    def test_result_summary_capped_at_120_chars(self):
        rec = OsnLoopRecorder()
        long_summary = "x" * 200
        rec.record_step("preflight", "passed", result_summary=long_summary)
        self.assertLessEqual(len(rec.summary.steps[0].result_summary), 120)

    def test_step_index_increments(self):
        rec = OsnLoopRecorder()
        rec.record_step("preflight", "passed")
        rec.record_step("observe", "passed")
        self.assertEqual(rec.summary.steps[0].step_index, 0)
        self.assertEqual(rec.summary.steps[1].step_index, 1)

    def test_complete_sets_fields(self):
        rec = OsnLoopRecorder()
        rec.record_step("final_receipt", "passed")
        rec.complete(
            stopped_reason="completed",
            verification_status="passed",
            retry_used=True,
            approval_granted=True,
        )
        s = rec.summary
        self.assertTrue(s.completed)
        self.assertEqual(s.stopped_reason, "completed")
        self.assertEqual(s.verification_status, "passed")
        self.assertTrue(s.retry_used)
        self.assertTrue(s.approval_granted)

    def test_approval_required_propagates_to_summary(self):
        rec = OsnLoopRecorder()
        rec.record_step("approval", "passed", approval_required=True)
        self.assertTrue(rec.summary.approval_required)

    def test_approval_required_false_does_not_set_summary(self):
        rec = OsnLoopRecorder()
        rec.record_step("preflight", "passed", approval_required=False)
        self.assertFalse(rec.summary.approval_required)


# ---------------------------------------------------------------------------
# max_steps enforcement
# ---------------------------------------------------------------------------

class TestMaxStepsEnforcement(unittest.TestCase):

    def _fill_recorder(self, recorder: OsnLoopRecorder, n: int):
        for i in range(n):
            recorder.record_step(f"step_{i}", "passed")

    def test_steps_never_exceed_max_steps(self):
        rec = OsnLoopRecorder()
        for i in range(20):
            rec.record_step(f"step_{i}", "passed")
        self.assertLessEqual(len(rec.summary.steps), rec.summary.max_steps)

    def test_steps_taken_equals_len_steps(self):
        rec = OsnLoopRecorder()
        for i in range(15):
            rec.record_step(f"step_{i}", "passed")
        self.assertEqual(rec.summary.steps_taken, len(rec.summary.steps))

    def test_last_slot_blocked_when_non_final_receipt(self):
        rec = OsnLoopRecorder()
        # Fill all but the last slot (max_steps - 1 = 10 steps)
        self._fill_recorder(rec, rec.summary.max_steps - 1)
        # Recording a non-final_receipt step at the last slot → blocked marker
        rec.record_step("extra_step", "passed")
        last = rec.summary.steps[-1]
        self.assertEqual(last.step_name, "max_steps_exceeded")
        self.assertEqual(last.status, "blocked")

    def test_final_receipt_fills_last_slot_normally(self):
        rec = OsnLoopRecorder()
        self._fill_recorder(rec, rec.summary.max_steps - 1)
        rec.record_step("final_receipt", "passed")
        last = rec.summary.steps[-1]
        self.assertEqual(last.step_name, "final_receipt")
        self.assertEqual(last.status, "passed")

    def test_warning_added_once(self):
        rec = OsnLoopRecorder()
        self._fill_recorder(rec, rec.summary.max_steps - 1)
        for _ in range(5):
            rec.record_step("overflow", "passed")
        self.assertEqual(rec.summary.warnings.count("max steps reached"), 1)

    def test_repeated_calls_after_max_do_not_append(self):
        rec = OsnLoopRecorder()
        self._fill_recorder(rec, rec.summary.max_steps)
        count_before = len(rec.summary.steps)
        rec.record_step("overflow_1", "passed")
        rec.record_step("overflow_2", "passed")
        self.assertEqual(len(rec.summary.steps), count_before)

    def test_complete_stopped_reason_is_max_steps_when_no_final_receipt(self):
        rec = OsnLoopRecorder()
        self._fill_recorder(rec, rec.summary.max_steps - 1)
        rec.record_step("overflow", "passed")  # triggers blocked marker
        rec.complete(stopped_reason="completed")
        self.assertFalse(rec.summary.completed)
        self.assertEqual(rec.summary.stopped_reason, "max_steps")

    def test_complete_stopped_reason_is_completed_when_final_receipt_present(self):
        rec = OsnLoopRecorder()
        self._fill_recorder(rec, rec.summary.max_steps - 1)
        rec.record_step("final_receipt", "passed")
        rec.complete(stopped_reason="completed")
        self.assertTrue(rec.summary.completed)
        self.assertEqual(rec.summary.stopped_reason, "completed")


# ---------------------------------------------------------------------------
# No raw content in metadata
# ---------------------------------------------------------------------------

class TestNoRawContentPolicy(unittest.TestCase):

    def test_result_summary_does_not_store_raw_content(self):
        rec = OsnLoopRecorder()
        raw_diff = "diff --git a/foo.py b/foo.py\n" + "+" * 5000
        rec.record_step("generate_patch", "passed", result_summary=raw_diff)
        self.assertLessEqual(len(rec.summary.steps[0].result_summary), 120)

    def test_no_write_file_tool_name(self):
        rec = OsnLoopRecorder()
        rec.record_step("safe_write", "passed", tool_name="")
        for step in rec.summary.steps:
            self.assertNotEqual(step.tool_name, "write_file")

    def test_no_run_command_tool_name(self):
        rec = OsnLoopRecorder()
        rec.record_step("verify", "passed", tool_name="")
        for step in rec.summary.steps:
            self.assertNotEqual(step.tool_name, "run_command")

    def test_serialization_roundtrip(self):
        rec = OsnLoopRecorder()
        rec.record_step("preflight", "passed", result_summary="repo_map_ok=True")
        rec.record_step("observe", "passed", result_summary="dirty=False", context_injected=True)
        rec.record_step("final_receipt", "passed")
        rec.complete(stopped_reason="completed", verification_status="passed")
        d = asdict(rec.summary)
        dumped = json.dumps(d)
        loaded = json.loads(dumped)
        self.assertEqual(loaded["steps_taken"], rec.summary.steps_taken)
        self.assertEqual(loaded["mode"], "experimental")


# ---------------------------------------------------------------------------
# Public methods on NativeAgentExecutor
# ---------------------------------------------------------------------------

class TestPublicMethodsOnExecutor(unittest.TestCase):

    def test_record_osn_loop_step_noop_when_no_recorder(self):
        executor, _ = _make_executor(native_loop=None)
        # Should not raise
        executor.record_osn_loop_step("preflight", "passed")
        self.assertIsNone(executor.native_meta.osn_loop_summary)

    def test_record_osn_loop_step_appends_when_recorder_present(self):
        executor, _ = _make_executor(native_loop="experimental")
        executor.record_osn_loop_step("preflight", "passed", result_summary="ok")
        # complete_osn_loop hasn't been called yet, but recorder has the step
        self.assertIsNotNone(executor._osn_recorder)
        self.assertEqual(len(executor._osn_recorder.summary.steps), 1)

    def test_complete_osn_loop_attaches_summary(self):
        executor, _ = _make_executor(native_loop="experimental")
        executor.record_osn_loop_step("final_receipt", "passed")
        executor.complete_osn_loop(stopped_reason="completed", verification_status="passed")
        self.assertIsNotNone(executor.native_meta.osn_loop_summary)
        self.assertTrue(executor.native_meta.osn_loop_summary.enabled)
        self.assertTrue(executor.native_meta.osn_loop_summary.completed)

    def test_complete_osn_loop_noop_when_no_recorder(self):
        executor, _ = _make_executor(native_loop=None)
        executor.complete_osn_loop(stopped_reason="completed")
        self.assertIsNone(executor.native_meta.osn_loop_summary)


# ---------------------------------------------------------------------------
# Default workflow does not create recorder
# ---------------------------------------------------------------------------

class TestDefaultWorkflowNoSummary(unittest.TestCase):

    def test_no_recorder_when_native_loop_none(self):
        executor, _ = _make_executor(native_loop=None)
        self.assertIsNone(executor._osn_recorder)

    def test_osn_loop_summary_none_when_native_loop_none(self):
        executor, _ = _make_executor(native_loop=None)
        executor.generate("fix bug")
        self.assertIsNone(executor.native_meta.osn_loop_summary)

    def test_recorder_created_for_experimental(self):
        executor, _ = _make_executor(native_loop="experimental")
        self.assertIsNotNone(executor._osn_recorder)

    def test_no_recorder_for_disabled_values(self):
        for val in ["off", "none", "disabled", "false", "0", "no", ""]:
            with self.subTest(val=val):
                executor, _ = _make_executor(native_loop=val)
                self.assertIsNone(executor._osn_recorder)


# ---------------------------------------------------------------------------
# Existing tool-level OSNLoopMeta is untouched
# ---------------------------------------------------------------------------

class TestExistingOSNLoopUntouched(unittest.TestCase):

    def test_osn_loop_field_is_still_osn_loop_meta_type(self):
        executor, _ = _make_executor(native_loop="experimental")
        # Before any run, osn_loop is None (not yet set)
        self.assertIsNone(executor.native_meta.osn_loop)
        # After generate, osn_loop_summary is set but osn_loop may still be None
        # (it's only set during _run_experimental_loop which requires a real runner)
        executor.generate("task")
        if executor.native_meta.osn_loop is not None:
            self.assertIsInstance(executor.native_meta.osn_loop, OSNLoopMeta)

    def test_osn_loop_summary_is_separate_field(self):
        executor, _ = _make_executor(native_loop="experimental")
        executor.generate("task")
        # Both fields can coexist independently
        summary = executor.native_meta.osn_loop_summary
        loop = executor.native_meta.osn_loop
        if summary is not None:
            self.assertIsInstance(summary, NativeOSNLoopSummary)
        if loop is not None:
            self.assertIsInstance(loop, OSNLoopMeta)


# ---------------------------------------------------------------------------
# generate() early phase instrumentation (integration-level)
# ---------------------------------------------------------------------------

class TestGenerateInstrumentsEarlyPhases(unittest.TestCase):

    def test_preflight_step_recorded(self):
        executor, _ = _make_executor(native_loop="experimental")
        executor.generate("fix bug")
        rec = executor._osn_recorder
        self.assertIsNotNone(rec)
        names = [s.step_name for s in rec.summary.steps]
        self.assertIn("preflight", names)

    def test_observe_step_recorded(self):
        executor, _ = _make_executor(native_loop="experimental")
        executor.generate("fix bug")
        names = [s.step_name for s in executor._osn_recorder.summary.steps]
        self.assertIn("observe", names)

    def test_gather_context_step_recorded(self):
        executor, _ = _make_executor(native_loop="experimental")
        executor.generate("fix bug")
        names = [s.step_name for s in executor._osn_recorder.summary.steps]
        self.assertIn("gather_context", names)

    def test_plan_update_step_recorded(self):
        executor, _ = _make_executor(native_loop="experimental")
        executor.generate("fix bug")
        names = [s.step_name for s in executor._osn_recorder.summary.steps]
        self.assertIn("plan_update", names)

    def test_generate_patch_step_recorded(self):
        executor, _ = _make_executor(native_loop="experimental")
        executor.generate("fix bug")
        names = [s.step_name for s in executor._osn_recorder.summary.steps]
        self.assertIn("generate_patch", names)

    def test_no_steps_recorded_without_experimental(self):
        executor, _ = _make_executor(native_loop=None)
        executor.generate("fix bug")
        self.assertIsNone(executor._osn_recorder)
        self.assertIsNone(executor.native_meta.osn_loop_summary)

    def test_no_raw_content_in_any_step_after_generate(self):
        executor, _ = _make_executor(native_loop="experimental")
        executor.generate("fix bug")
        rec = executor._osn_recorder
        if rec is not None:
            for step in rec.summary.steps:
                self.assertLessEqual(
                    len(step.result_summary), 120,
                    f"result_summary too long on step {step.step_name!r}",
                )


# ---------------------------------------------------------------------------
# _is_safe_osn_tool_name and _LOOP_ALLOWED_TOOLS
# ---------------------------------------------------------------------------

class TestIsSafeOsnToolName(unittest.TestCase):

    def test_allowed_tools_are_safe(self):
        for name in _LOOP_ALLOWED_TOOLS:
            with self.subTest(name=name):
                self.assertTrue(_is_safe_osn_tool_name(name))

    def test_write_file_is_not_safe(self):
        self.assertFalse(_is_safe_osn_tool_name("write_file"))

    def test_run_command_is_not_safe(self):
        self.assertFalse(_is_safe_osn_tool_name("run_command"))

    def test_run_verification_is_not_safe(self):
        self.assertFalse(_is_safe_osn_tool_name("run_verification"))

    def test_empty_string_is_not_safe(self):
        self.assertFalse(_is_safe_osn_tool_name(""))

    def test_unknown_tool_is_not_safe(self):
        self.assertFalse(_is_safe_osn_tool_name("arbitrary_tool"))

    def test_allowlist_contains_exactly_four_tools(self):
        self.assertEqual(len(_LOOP_ALLOWED_TOOLS), 4)

    def test_known_safe_tools_in_allowlist(self):
        expected = {"list_files", "read_file", "search_repo", "get_git_diff"}
        self.assertTrue(expected.issubset(_LOOP_ALLOWED_TOOLS))


# ---------------------------------------------------------------------------
# verify / retry_once / final_receipt steps at the executor level
# ---------------------------------------------------------------------------

class TestVerifyRetryReceiptIntegration(unittest.TestCase):

    def _make(self):
        return _make_executor(native_loop="experimental")

    def test_verify_step_recorded(self):
        executor, _ = self._make()
        executor.record_osn_loop_step("verify", "passed", verification_status="passed")
        names = [s.step_name for s in executor._osn_recorder.summary.steps]
        self.assertIn("verify", names)

    def test_verify_step_verification_status_propagated(self):
        executor, _ = self._make()
        executor.record_osn_loop_step("verify", "passed", verification_status="passed")
        step = next(s for s in executor._osn_recorder.summary.steps if s.step_name == "verify")
        self.assertEqual(step.verification_status, "passed")

    def test_verify_step_failed_status(self):
        executor, _ = self._make()
        executor.record_osn_loop_step("verify", "failed", verification_status="failed")
        step = next(s for s in executor._osn_recorder.summary.steps if s.step_name == "verify")
        self.assertEqual(step.status, "failed")
        self.assertEqual(step.verification_status, "failed")

    def test_retry_once_step_recorded(self):
        executor, _ = self._make()
        executor.record_osn_loop_step("retry_once", "running")
        names = [s.step_name for s in executor._osn_recorder.summary.steps]
        self.assertIn("retry_once", names)

    def test_final_receipt_step_recorded(self):
        executor, _ = self._make()
        executor.record_osn_loop_step("final_receipt", "passed")
        names = [s.step_name for s in executor._osn_recorder.summary.steps]
        self.assertIn("final_receipt", names)

    def test_all_three_steps_in_sequence(self):
        executor, _ = self._make()
        executor.record_osn_loop_step("verify", "passed", verification_status="passed")
        executor.record_osn_loop_step("retry_once", "running")
        executor.record_osn_loop_step("final_receipt", "passed")
        names = [s.step_name for s in executor._osn_recorder.summary.steps]
        self.assertIn("verify", names)
        self.assertIn("retry_once", names)
        self.assertIn("final_receipt", names)
        # order is preserved
        self.assertLess(names.index("verify"), names.index("retry_once"))
        self.assertLess(names.index("retry_once"), names.index("final_receipt"))

    def test_complete_osn_loop_after_verify_sets_verification_status(self):
        executor, _ = self._make()
        executor.record_osn_loop_step("verify", "passed", verification_status="passed")
        executor.record_osn_loop_step("final_receipt", "passed")
        executor.complete_osn_loop(stopped_reason="completed", verification_status="passed")
        self.assertEqual(executor.native_meta.osn_loop_summary.verification_status, "passed")

    def test_complete_osn_loop_after_retry_sets_retry_used(self):
        executor, _ = self._make()
        executor.record_osn_loop_step("retry_once", "running")
        executor.record_osn_loop_step("final_receipt", "passed")
        executor.complete_osn_loop(stopped_reason="completed", retry_used=True)
        self.assertTrue(executor.native_meta.osn_loop_summary.retry_used)

    def test_final_receipt_allows_complete_true(self):
        executor, _ = self._make()
        executor.record_osn_loop_step("final_receipt", "passed")
        executor.complete_osn_loop(stopped_reason="completed")
        self.assertTrue(executor.native_meta.osn_loop_summary.completed)

    def test_no_recorder_noop_for_all_steps(self):
        executor, _ = _make_executor(native_loop=None)
        executor.record_osn_loop_step("verify", "passed", verification_status="passed")
        executor.record_osn_loop_step("retry_once", "running")
        executor.record_osn_loop_step("final_receipt", "passed")
        executor.complete_osn_loop(stopped_reason="completed")
        self.assertIsNone(executor.native_meta.osn_loop_summary)


# ---------------------------------------------------------------------------
# Hardening v1 - new counter fields and normalized stop reasons
# ---------------------------------------------------------------------------

class TestNativeOSNLoopStepHardening(unittest.TestCase):

    def test_raw_content_stored_always_false(self):
        step = NativeOSNLoopStep(step_name="verify", status="passed")
        self.assertFalse(step.raw_content_stored)

    def test_raw_content_stored_cannot_be_set_true(self):
        step = NativeOSNLoopStep(step_name="verify", status="passed", raw_content_stored=True)
        self.assertFalse(step.raw_content_stored)

    def test_target_label_field_exists(self):
        step = NativeOSNLoopStep(step_name="tool", status="passed", target_label="src/auth.py")
        self.assertEqual(step.target_label, "src/auth.py")

    def test_target_label_capped_at_40(self):
        long_label = "a" * 60
        step = NativeOSNLoopStep(step_name="tool", status="passed", target_label=long_label)
        self.assertLessEqual(len(step.target_label), 40)

    def test_blocked_reason_field_exists(self):
        step = NativeOSNLoopStep(step_name="tool", status="blocked", blocked_reason="approval required")
        self.assertEqual(step.blocked_reason, "approval required")

    def test_step_serializes_new_fields_json_safe(self):
        step = NativeOSNLoopStep(
            step_name="tool",
            status="blocked",
            target_label="src/auth.py",
            blocked_reason="approval required",
        )
        d = asdict(step)
        self.assertIn("target_label", d)
        self.assertIn("blocked_reason", d)
        self.assertIn("raw_content_stored", d)
        self.assertFalse(d["raw_content_stored"])
        json.dumps(d)


class TestNativeOSNLoopSummaryCounterFields(unittest.TestCase):

    def test_new_fields_default_clean(self):
        summary = NativeOSNLoopSummary()
        self.assertEqual(summary.loop_id, "")
        self.assertEqual(summary.attempted_steps, 0)
        self.assertEqual(summary.completed_steps, 0)
        self.assertEqual(summary.failed_steps, 0)
        self.assertEqual(summary.blocked_steps, 0)
        self.assertEqual(summary.tool_calls_attempted, 0)
        self.assertEqual(summary.tool_calls_completed, 0)
        self.assertEqual(summary.tool_calls_blocked, 0)
        self.assertFalse(summary.verification_attempted)
        self.assertIsNone(summary.verification_passed)
        self.assertEqual(summary.retry_count, 0)
        self.assertEqual(summary.final_status, "")

    def test_summary_new_fields_json_safe(self):
        summary = NativeOSNLoopSummary(
            enabled=True, mode="experimental",
            attempted_steps=2, completed_steps=1, blocked_steps=1,
            loop_id="test-uuid",
        )
        d = asdict(summary)
        json.dumps(d)
        self.assertIn("loop_id", d)
        self.assertIn("attempted_steps", d)


class TestOsnLoopRecorderCounters(unittest.TestCase):

    def test_loop_id_set_on_init(self):
        rec = OsnLoopRecorder()
        self.assertNotEqual(rec.summary.loop_id, "")
        self.assertIsInstance(rec.summary.loop_id, str)

    def test_increments_completed_steps(self):
        rec = OsnLoopRecorder()
        rec.record_step("preflight", "passed")
        self.assertEqual(rec.summary.completed_steps, 1)
        self.assertEqual(rec.summary.attempted_steps, 1)
        self.assertEqual(rec.summary.failed_steps, 0)
        self.assertEqual(rec.summary.blocked_steps, 0)

    def test_increments_failed_steps(self):
        rec = OsnLoopRecorder()
        rec.record_step("verify", "failed")
        self.assertEqual(rec.summary.failed_steps, 1)
        self.assertEqual(rec.summary.completed_steps, 0)

    def test_increments_blocked_steps(self):
        rec = OsnLoopRecorder()
        rec.record_step("approval", "blocked")
        self.assertEqual(rec.summary.blocked_steps, 1)
        self.assertEqual(rec.summary.completed_steps, 0)

    def test_tool_call_counters_tracked(self):
        rec = OsnLoopRecorder()
        rec.record_step("read", "passed", tool_name="read_file")
        self.assertEqual(rec.summary.tool_calls_attempted, 1)
        self.assertEqual(rec.summary.tool_calls_completed, 1)
        self.assertEqual(rec.summary.tool_calls_blocked, 0)

    def test_tool_call_blocked_counter(self):
        rec = OsnLoopRecorder()
        rec.record_step("tool", "blocked", tool_name="run_command")
        self.assertEqual(rec.summary.tool_calls_blocked, 1)
        self.assertEqual(rec.summary.tool_calls_attempted, 1)
        self.assertEqual(rec.summary.tool_calls_completed, 0)

    def test_multiple_steps_accumulate(self):
        rec = OsnLoopRecorder()
        rec.record_step("preflight", "passed")
        rec.record_step("read", "passed", tool_name="read_file")
        rec.record_step("tool", "blocked", tool_name="run_command")
        self.assertEqual(rec.summary.attempted_steps, 3)
        self.assertEqual(rec.summary.completed_steps, 2)
        self.assertEqual(rec.summary.blocked_steps, 1)
        self.assertEqual(rec.summary.tool_calls_attempted, 2)
        self.assertEqual(rec.summary.tool_calls_completed, 1)
        self.assertEqual(rec.summary.tool_calls_blocked, 1)

    def test_complete_sets_verification_attempted_when_status_present(self):
        rec = OsnLoopRecorder()
        rec.complete(stopped_reason="completed", verification_status="passed")
        self.assertTrue(rec.summary.verification_attempted)

    def test_complete_sets_verification_passed_true(self):
        rec = OsnLoopRecorder()
        rec.complete(stopped_reason="completed", verification_status="passed")
        self.assertTrue(rec.summary.verification_passed)

    def test_complete_sets_verification_passed_false(self):
        rec = OsnLoopRecorder()
        rec.complete(stopped_reason="completed", verification_status="failed")
        self.assertFalse(rec.summary.verification_passed)

    def test_complete_verification_not_attempted_when_no_status(self):
        rec = OsnLoopRecorder()
        rec.complete(stopped_reason="completed")
        self.assertFalse(rec.summary.verification_attempted)
        self.assertIsNone(rec.summary.verification_passed)

    def test_complete_sets_final_status_completed(self):
        rec = OsnLoopRecorder()
        rec.record_step("final_receipt", "passed")
        rec.complete(stopped_reason="completed")
        self.assertEqual(rec.summary.final_status, "completed")

    def test_complete_sets_final_status_to_stopped_reason_when_not_complete(self):
        rec = OsnLoopRecorder()
        rec.complete(stopped_reason="tool_error")
        self.assertEqual(rec.summary.final_status, "tool_error")

    def test_complete_normalizes_stop_reason(self):
        rec = OsnLoopRecorder()
        rec.complete(stopped_reason="complete")
        self.assertEqual(rec.summary.stopped_reason, "completed")

    def test_complete_retry_limit_enforced(self):
        from openshard.native.context import _MAX_RETRY_COUNT
        rec = OsnLoopRecorder()
        rec.complete(stopped_reason="completed", retry_count=_MAX_RETRY_COUNT + 1)
        self.assertEqual(rec.summary.stopped_reason, "retry_limit")

    def test_complete_retry_count_stored(self):
        rec = OsnLoopRecorder()
        rec.complete(stopped_reason="completed", retry_count=1)
        self.assertEqual(rec.summary.retry_count, 1)

    def test_repeated_blocked_tool_warning_added(self):
        from openshard.native.context import _MAX_REPEATED_BLOCKED_TOOL
        rec = OsnLoopRecorder()
        for _ in range(_MAX_REPEATED_BLOCKED_TOOL):
            rec.record_step("tool", "blocked", tool_name="run_command")
        self.assertIn("repeated_blocked_tool_limit", rec.summary.warnings)

    def test_step_events_cap_enforced(self):
        from openshard.native.context import _MAX_STEP_EVENTS_RECORDED
        rec = OsnLoopRecorder()
        # Bypass max_steps by temporarily raising it
        rec._summary.max_steps = _MAX_STEP_EVENTS_RECORDED + 10
        for i in range(_MAX_STEP_EVENTS_RECORDED + 5):
            rec.record_step(f"step_{i}", "passed")
        self.assertLessEqual(len(rec.summary.steps), _MAX_STEP_EVENTS_RECORDED)
        self.assertIn("step_events_cap_reached", rec.summary.warnings)


class TestNormalizeOsnStopReason(unittest.TestCase):

    def setUp(self):
        from openshard.native.context import _VALID_OSN_STOP_REASONS, normalize_osn_stop_reason
        self._normalize = normalize_osn_stop_reason
        self._valid = _VALID_OSN_STOP_REASONS

    def test_none_returns_unknown(self):
        self.assertEqual(self._normalize(None), "unknown")

    def test_empty_string_returns_unknown(self):
        self.assertEqual(self._normalize(""), "unknown")

    def test_whitespace_returns_unknown(self):
        self.assertEqual(self._normalize("   "), "unknown")

    def test_alias_complete_to_completed(self):
        self.assertEqual(self._normalize("complete"), "completed")

    def test_alias_consecutive_empty_to_empty_response_limit(self):
        self.assertEqual(self._normalize("consecutive_empty"), "empty_response_limit")

    def test_alias_max_steps_reached_to_max_steps(self):
        self.assertEqual(self._normalize("max steps reached"), "max_steps")

    def test_mixed_case_alias(self):
        self.assertEqual(self._normalize("Complete"), "completed")
        self.assertEqual(self._normalize("COMPLETE"), "completed")

    def test_unknown_value_returns_unknown(self):
        self.assertEqual(self._normalize("some_garbage_value"), "unknown")

    def test_all_valid_values_are_stable(self):
        for reason in self._valid:
            self.assertEqual(self._normalize(reason), reason)


if __name__ == "__main__":
    unittest.main()
