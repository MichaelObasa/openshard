"""Bounded OSN native loop v0 — focused tests.

Covers requirements not addressed by test_osn_loop.py / test_osn_loop_summary.py:
  - approval step recorded via build_approval_receipt (once, correct status)
  - retry_once recorded with final "passed"/"failed" status, never "running"
  - native step events written to native_steps.jsonl for loop stages
  - tool/search provenance written for context-gathering in experimental loop
  - unsafe write paths rejected (security / safe write path)
  - approval gate triggers for risky write (native_loop integration)
"""
from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from openshard.native.executor import NativeAgentExecutor
from openshard.native.tools import NativeToolResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_executor(native_loop: str | None = "experimental"):
    fake_gen = MagicMock()
    fake_gen.generate.return_value = MagicMock(usage=None, files=[], summary="ok", notes=[])
    fake_gen.model = "mock-model"
    fake_gen.fixer_model = "mock-fixer"
    with patch("openshard.native.executor.ExecutionGenerator", return_value=fake_gen):
        executor = NativeAgentExecutor(provider=MagicMock(), native_loop=native_loop)
    return executor


def _make_runner(output: str = "content"):
    runner = MagicMock()
    runner.run.return_value = NativeToolResult(tool_name="read_file", ok=True, output=output)
    runner.trace_entry.return_value = {
        "tool": "read_file", "ok": True, "approved": False,
        "output_chars": len(output), "error": None,
    }
    return runner


# ---------------------------------------------------------------------------
# Approval step — recorded via build_approval_receipt (osn_loop)
# ---------------------------------------------------------------------------

class TestOsnLoopApprovalRecording(unittest.TestCase):
    """Approval step is emitted into the OSN loop recorder when approval is granted."""

    def test_build_approval_receipt_records_approval_step(self):
        executor = _make_executor()
        executor.build_approval_receipt(granted=True)
        names = [s.step_name for s in executor._osn_recorder.summary.steps]
        self.assertIn("approval", names)

    def test_approval_step_status_is_passed_when_granted(self):
        executor = _make_executor()
        executor.build_approval_receipt(granted=True)
        step = next(s for s in executor._osn_recorder.summary.steps if s.step_name == "approval")
        self.assertEqual(step.status, "passed")

    def test_approval_step_status_is_blocked_when_denied(self):
        executor = _make_executor()
        executor.build_approval_receipt(granted=False)
        step = next(s for s in executor._osn_recorder.summary.steps if s.step_name == "approval")
        self.assertEqual(step.status, "blocked")

    def test_approval_step_approval_required_true(self):
        executor = _make_executor()
        executor.build_approval_receipt(granted=True)
        step = next(s for s in executor._osn_recorder.summary.steps if s.step_name == "approval")
        self.assertTrue(step.approval_required)

    def test_approval_step_recorded_exactly_once(self):
        executor = _make_executor()
        executor.build_approval_receipt(granted=True)
        approval_steps = [s for s in executor._osn_recorder.summary.steps if s.step_name == "approval"]
        self.assertEqual(len(approval_steps), 1)

    def test_no_approval_step_without_build_approval_receipt(self):
        # If approval is not triggered, no approval step in summary
        executor = _make_executor()
        # Don't call build_approval_receipt
        executor.record_osn_loop_step("preflight", "passed")
        names = [s.step_name for s in executor._osn_recorder.summary.steps]
        self.assertNotIn("approval", names)

    def test_approval_step_not_recorded_when_no_osn_recorder(self):
        # Non-experimental mode — no recorder, no crash
        executor = _make_executor(native_loop=None)
        executor.build_approval_receipt(granted=True)
        self.assertIsNone(executor._osn_recorder)
        self.assertIsNone(executor.native_meta.osn_loop_summary)


# ---------------------------------------------------------------------------
# retry_once status — must be "passed" or "failed", never "running" (osn_loop)
# ---------------------------------------------------------------------------

class TestOsnLoopRetryOnceStatus(unittest.TestCase):
    """retry_once step carries the final verification status from the second run."""

    def test_retry_once_passed_status_recorded(self):
        executor = _make_executor()
        executor.record_osn_loop_step(
            "retry_once", "passed", verification_status="passed",
        )
        step = next(s for s in executor._osn_recorder.summary.steps if s.step_name == "retry_once")
        self.assertEqual(step.status, "passed")

    def test_retry_once_failed_status_recorded(self):
        executor = _make_executor()
        executor.record_osn_loop_step(
            "retry_once", "failed", verification_status="failed",
        )
        step = next(s for s in executor._osn_recorder.summary.steps if s.step_name == "retry_once")
        self.assertEqual(step.status, "failed")
        self.assertEqual(step.verification_status, "failed")

    def test_retry_once_verification_status_propagated_for_passed(self):
        executor = _make_executor()
        executor.record_osn_loop_step("retry_once", "passed", verification_status="passed")
        step = next(s for s in executor._osn_recorder.summary.steps if s.step_name == "retry_once")
        self.assertEqual(step.verification_status, "passed")

    def test_retry_once_not_present_when_not_recorded(self):
        executor = _make_executor()
        executor.record_osn_loop_step("verify", "passed", verification_status="passed")
        executor.record_osn_loop_step("final_receipt", "passed")
        names = [s.step_name for s in executor._osn_recorder.summary.steps]
        self.assertNotIn("retry_once", names)

    def test_retry_once_appears_after_verify_in_sequence(self):
        executor = _make_executor()
        executor.record_osn_loop_step("verify", "failed", verification_status="failed")
        executor.record_osn_loop_step("retry_once", "passed", verification_status="passed")
        executor.record_osn_loop_step("final_receipt", "passed")
        names = [s.step_name for s in executor._osn_recorder.summary.steps]
        self.assertLess(names.index("verify"), names.index("retry_once"))

    def test_complete_after_retry_sets_retry_used_and_verification_status(self):
        executor = _make_executor()
        executor.record_osn_loop_step("verify", "failed", verification_status="failed")
        executor.record_osn_loop_step("retry_once", "passed", verification_status="passed")
        executor.record_osn_loop_step("final_receipt", "passed")
        executor.complete_osn_loop(
            stopped_reason="completed", retry_used=True, verification_status="passed",
        )
        summary = executor.native_meta.osn_loop_summary
        self.assertTrue(summary.retry_used)
        self.assertEqual(summary.verification_status, "passed")


# ---------------------------------------------------------------------------
# Native step events written for loop stages (native_step + osn_loop)
# ---------------------------------------------------------------------------

class TestNativeStepEventsOsnLoop(unittest.TestCase):
    """record_osn_loop_step writes NativeStepEvent to native_steps.jsonl."""

    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self._orig_cwd = os.getcwd()
        os.chdir(self._tmpdir.name)

    def tearDown(self):
        os.chdir(self._orig_cwd)
        self._tmpdir.cleanup()

    def test_step_event_written_to_jsonl(self):
        executor = _make_executor()
        executor._run_id = "test-run-id-001"
        executor.record_osn_loop_step("preflight", "passed", result_summary="repo_map_ok=True")
        log_path = Path(".openshard") / "native_steps.jsonl"
        self.assertTrue(log_path.exists(), "native_steps.jsonl should be created")

    def test_step_event_contains_correct_step_name(self):
        import json as _json
        executor = _make_executor()
        executor._run_id = "test-run-id-002"
        executor.record_osn_loop_step("observe", "passed")
        log_path = Path(".openshard") / "native_steps.jsonl"
        events = [_json.loads(line) for line in log_path.read_text().splitlines() if line.strip()]
        names = [e["step_name"] for e in events]
        self.assertIn("observe", names)

    def test_step_event_raw_content_stored_is_false(self):
        import json as _json
        executor = _make_executor()
        executor._run_id = "test-run-id-003"
        executor.record_osn_loop_step("gather_context", "passed", result_summary="findings=3")
        log_path = Path(".openshard") / "native_steps.jsonl"
        for line in log_path.read_text().splitlines():
            if not line.strip():
                continue
            evt = _json.loads(line)
            self.assertFalse(evt.get("raw_content_stored", True),
                             "raw_content_stored must be False")

    def test_step_event_summary_capped_at_120_chars(self):
        import json as _json
        executor = _make_executor()
        executor._run_id = "test-run-id-004"
        long_summary = "x" * 300
        executor.record_osn_loop_step("plan_update", "passed", result_summary=long_summary)
        log_path = Path(".openshard") / "native_steps.jsonl"
        for line in log_path.read_text().splitlines():
            if not line.strip():
                continue
            evt = _json.loads(line)
            if evt.get("step_name") == "plan_update":
                self.assertLessEqual(len(evt.get("summary", "")), 120)

    def test_no_event_written_without_run_id(self):
        executor = _make_executor()
        # _run_id defaults to "" which causes early return in log_step_event
        executor.record_osn_loop_step("preflight", "passed")
        log_path = Path(".openshard") / "native_steps.jsonl"
        self.assertFalse(log_path.exists(), "No file should be created without a run_id")


# ---------------------------------------------------------------------------
# Tool/search provenance in experimental loop (tool_search + osn_loop)
# ---------------------------------------------------------------------------

class TestToolSearchProvenanceOsnLoop(unittest.TestCase):
    """_record_tool_search_event is called for each tool step in the experimental loop."""

    def test_tool_search_events_appended_during_experimental_loop(self):
        executor = _make_executor()
        executor._runner = _make_runner()
        executor.native_meta.read_search_findings = ["file:src/foo.py"]
        executor.native_meta.evidence = None
        executor._run_experimental_loop("add auth feature")
        self.assertGreater(len(executor.native_meta.tool_search_events), 0)

    def test_tool_search_event_has_tool_name(self):
        executor = _make_executor()
        executor._runner = _make_runner()
        executor.native_meta.read_search_findings = ["file:src/foo.py"]
        executor.native_meta.evidence = None
        executor._run_experimental_loop("some task")
        for ev in executor.native_meta.tool_search_events:
            self.assertNotEqual(ev.tool_name, "", "tool_name must not be empty")

    def test_tool_search_event_has_result_quality_field(self):
        executor = _make_executor()
        executor._runner = _make_runner()
        executor.native_meta.read_search_findings = ["file:src/foo.py"]
        executor.native_meta.evidence = None
        executor._run_experimental_loop("some task")
        for ev in executor.native_meta.tool_search_events:
            self.assertIn(ev.result_quality, {"unknown", "empty", "weak", "useful"},
                          f"unexpected result_quality: {ev.result_quality!r}")

    def test_tool_search_event_no_raw_output_stored(self):
        executor = _make_executor()
        runner = _make_runner(output="SECRET FILE CONTENT")
        executor._runner = runner
        executor.native_meta.read_search_findings = ["file:src/foo.py"]
        executor.native_meta.evidence = None
        executor._run_experimental_loop("some task")
        for ev in executor.native_meta.tool_search_events:
            # Events store metadata only — no raw output field
            self.assertFalse(hasattr(ev, "raw_output"),
                             "tool search events must not carry raw output")

    def test_no_tool_search_event_for_skipped_disallowed_tool(self):
        executor = _make_executor()
        runner = _make_runner()
        executor._runner = runner
        with patch.object(executor, "_build_loop_step_queue", return_value=[
            ("write_file", "out.py", "should_be_skipped"),
        ]):
            executor._run_experimental_loop("some task")
        # write_file is skipped — no tool call happens, so no provenance event
        for ev in executor.native_meta.tool_search_events:
            self.assertNotEqual(ev.tool_name, "write_file")
        runner.run.assert_not_called()

    def test_context_injected_true_when_output_nonempty(self):
        executor = _make_executor()
        executor._runner = _make_runner(output="meaningful content")
        executor.native_meta.read_search_findings = ["file:src/foo.py"]
        executor.native_meta.evidence = None
        executor._run_experimental_loop("some task")
        self.assertTrue(
            any(ev.context_injected for ev in executor.native_meta.tool_search_events),
            "At least one event should have context_injected=True for non-empty output",
        )

    def test_context_injected_false_when_output_empty(self):
        executor = _make_executor()
        runner = MagicMock()
        runner.run.return_value = NativeToolResult(tool_name="read_file", ok=True, output="")
        runner.trace_entry.return_value = {"tool": "read_file", "ok": True, "approved": False, "output_chars": 0, "error": None}
        executor._runner = runner
        executor.native_meta.read_search_findings = ["file:src/empty.py"]
        executor.native_meta.evidence = None
        executor._run_experimental_loop("some task")
        for ev in executor.native_meta.tool_search_events:
            if ev.tool_name == "read_file":
                self.assertFalse(ev.context_injected,
                                 "context_injected must be False for empty output")


# ---------------------------------------------------------------------------
# Write safety — unsafe paths rejected (security + native_loop)
# ---------------------------------------------------------------------------

class TestNativeLoopWriteSecurity(unittest.TestCase):
    """_write_files rejects unsafe paths without writing any file."""

    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()

    def tearDown(self):
        self._tmpdir.cleanup()

    def _write_files(self, files):
        from openshard.run.pipeline import _write_files as _wf
        from unittest.mock import patch
        # Suppress click.echo output during tests
        with patch("openshard.run.pipeline.click"):
            _wf(files, Path(self._tmpdir.name))

    def _changed_file(self, path: str, content: str = "data", change_type: str = "create"):
        from openshard.execution.generator import ChangedFile
        return ChangedFile(path=path, content=content, change_type=change_type, summary="")

    def test_safe_relative_path_written(self):
        f = self._changed_file("src/module.py", "print('hello')")
        self._write_files([f])
        target = Path(self._tmpdir.name) / "src" / "module.py"
        self.assertTrue(target.exists())
        self.assertEqual(target.read_text(), "print('hello')")

    def test_traversal_path_rejected(self):
        marker = "openshard_ci_escape_marker.txt"
        escaped = Path(self._tmpdir.name).parent / marker
        escaped.unlink(missing_ok=True)
        try:
            f = self._changed_file(f"../{marker}", "evil")
            self._write_files([f])
            self.assertFalse(escaped.exists(), "Traversal path must not be written outside workspace")
        finally:
            escaped.unlink(missing_ok=True)

    def test_absolute_path_rejected(self):
        f = self._changed_file("/tmp/injected.py", "evil")
        self._write_files([f])
        self.assertFalse(Path("/tmp/injected.py").exists())

    def test_safe_path_not_rejected_when_mixed_with_unsafe(self):
        safe = self._changed_file("src/ok.py", "safe")
        unsafe = self._changed_file("../../escape.py", "bad")
        self._write_files([safe, unsafe])
        self.assertTrue((Path(self._tmpdir.name) / "src" / "ok.py").exists())


# ---------------------------------------------------------------------------
# Approval gate triggers for risky write (native_loop + security)
# ---------------------------------------------------------------------------

class TestNativeLoopApprovalGate(unittest.TestCase):
    """GateEvaluator correctly signals when approval is required."""

    def _gate(self, mode: str = "auto", risky: list | None = None, threshold: float = 1.0):
        from openshard.execution.gates import GateEvaluator
        return GateEvaluator(approval_mode=mode, risky_paths=risky or [], cost_threshold=threshold)

    def test_ask_mode_always_requires_file_write_approval(self):
        gate = self._gate(mode="ask")
        dec = gate.check_file_write(["src/foo.py"])
        self.assertTrue(dec.required)

    def test_auto_mode_no_approval_for_non_risky_path(self):
        gate = self._gate(mode="auto")
        dec = gate.check_file_write(["src/foo.py"])
        self.assertFalse(dec.required)

    def test_smart_mode_triggers_for_risky_path(self):
        gate = self._gate(mode="smart", risky=["src/secret.py"])
        dec = gate.check_file_write(["src/secret.py"])
        self.assertTrue(dec.required)

    def test_smart_mode_no_approval_for_safe_path(self):
        gate = self._gate(mode="smart", risky=["src/secret.py"])
        dec = gate.check_file_write(["src/safe.py"])
        self.assertFalse(dec.required)

    def test_check_risky_paths_triggers_on_match(self):
        gate = self._gate(mode="smart", risky=["config/prod.yaml"])
        dec = gate.check_risky_paths(["config/prod.yaml"])
        self.assertTrue(dec.required)

    def test_check_risky_paths_no_trigger_without_risky_config(self):
        gate = self._gate(mode="auto")
        dec = gate.check_risky_paths(["config/prod.yaml"])
        self.assertFalse(dec.required)


# ---------------------------------------------------------------------------
# Hardening v1 - bounds constants and run_command safety
# ---------------------------------------------------------------------------

class TestBoundsConstants(unittest.TestCase):

    def test_max_step_events_recorded_constant_exists(self):
        from openshard.native.context import _MAX_STEP_EVENTS_RECORDED
        self.assertIsInstance(_MAX_STEP_EVENTS_RECORDED, int)
        self.assertGreater(_MAX_STEP_EVENTS_RECORDED, 0)

    def test_max_repeated_blocked_tool_constant_exists(self):
        from openshard.native.context import _MAX_REPEATED_BLOCKED_TOOL
        self.assertIsInstance(_MAX_REPEATED_BLOCKED_TOOL, int)
        self.assertGreater(_MAX_REPEATED_BLOCKED_TOOL, 0)

    def test_max_retry_count_constant_exists(self):
        from openshard.native.context import _MAX_RETRY_COUNT
        self.assertIsInstance(_MAX_RETRY_COUNT, int)
        self.assertGreaterEqual(_MAX_RETRY_COUNT, 1)

    def test_run_command_not_in_loop_allowed_tools(self):
        from openshard.native.executor import _LOOP_ALLOWED_TOOLS
        self.assertNotIn("run_command", _LOOP_ALLOWED_TOOLS)

    def test_write_file_not_in_loop_allowed_tools(self):
        from openshard.native.executor import _LOOP_ALLOWED_TOOLS
        self.assertNotIn("write_file", _LOOP_ALLOWED_TOOLS)

    def test_valid_stop_reasons_contains_expected_values(self):
        from openshard.native.context import _VALID_OSN_STOP_REASONS
        required = {
            "completed", "max_steps", "no_steps", "blocked_tool", "approval_required",
            "verification_failed", "tool_error", "empty_response_limit", "retry_limit",
            "policy_denied", "unknown",
        }
        self.assertTrue(required.issubset(_VALID_OSN_STOP_REASONS))


if __name__ == "__main__":
    unittest.main()
