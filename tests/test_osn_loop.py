from __future__ import annotations

import json
import unittest
from dataclasses import asdict
from unittest.mock import MagicMock

from openshard.native.context import (
    OSNLoopMeta,
    OSNLoopStep,
    build_osn_loop_meta,
    render_osn_loop,
    render_osn_loop_context,
)
from openshard.native.executor import (
    NativeAgentExecutor,
    _LOOP_ALLOWED_TOOLS,
    _MAX_OSN_LOOP_STEPS,
    _MAX_OSN_QUEUE_CAP,
)
from openshard.native.tools import NativeToolResult


def _make_executor(native_loop: str | None = "experimental"):
    fake_gen = MagicMock()
    fake_gen.generate.return_value = MagicMock(usage=None, files=[], summary="ok", notes=[])
    fake_gen.model = "mock-model"
    fake_gen.fixer_model = "mock-fixer"
    from unittest.mock import patch
    with patch("openshard.native.executor.ExecutionGenerator", return_value=fake_gen):
        executor = NativeAgentExecutor(provider=MagicMock(), native_loop=native_loop)
    return executor, fake_gen


def _make_runner_mock(outputs: list[str] | None = None):
    runner = MagicMock()
    results = []
    for out in (outputs or []):
        r = NativeToolResult(tool_name="read_file", ok=True, output=out)
        results.append(r)
    runner.run.side_effect = results if results else [
        NativeToolResult(tool_name="read_file", ok=True, output="some content")
    ]
    runner.trace_entry.return_value = {"tool": "read_file", "ok": True, "approved": False, "output_chars": 4, "error": None}
    return runner


class TestOSNLoopStep(unittest.TestCase):

    def test_defaults(self):
        step = OSNLoopStep()
        self.assertEqual(step.step_index, 0)
        self.assertEqual(step.tool_name, "")
        self.assertEqual(step.target_label, "")
        self.assertEqual(step.reason, "")
        self.assertFalse(step.ok)
        self.assertEqual(step.output_chars, 0)
        self.assertFalse(step.empty)
        self.assertFalse(step.skipped)

    def test_skipped_step(self):
        step = OSNLoopStep(step_index=2, tool_name="write_file", skipped=True)
        self.assertTrue(step.skipped)
        self.assertFalse(step.ok)

    def test_serializes_via_asdict(self):
        step = OSNLoopStep(step_index=1, tool_name="read_file", target_label="src/foo.py", ok=True, output_chars=100)
        d = asdict(step)
        self.assertIsInstance(d, dict)
        self.assertEqual(d["tool_name"], "read_file")
        self.assertEqual(d["target_label"], "src/foo.py")
        json.dumps(d)


class TestOSNLoopMeta(unittest.TestCase):

    def test_defaults(self):
        meta = OSNLoopMeta()
        self.assertFalse(meta.enabled)
        self.assertEqual(meta.steps_run, 0)
        self.assertEqual(meta.terminated_reason, "not_run")
        self.assertEqual(meta.steps, [])
        self.assertEqual(meta.paths_surfaced, [])
        self.assertFalse(meta.truncated)

    def test_paths_surfaced_only_read_file_ok_nonempty(self):
        steps = [
            OSNLoopStep(step_index=0, tool_name="read_file", target_label="src/foo.py", ok=True, empty=False),
            OSNLoopStep(step_index=1, tool_name="search_repo", target_label="task_keyword_search", ok=True, empty=False),
            OSNLoopStep(step_index=2, tool_name="read_file", target_label="src/bar.py", ok=False),
            OSNLoopStep(step_index=3, tool_name="read_file", target_label="src/baz.py", ok=True, empty=True),
            OSNLoopStep(step_index=4, tool_name="read_file", target_label="src/qux.py", ok=True, empty=False, skipped=False),
        ]
        meta = build_osn_loop_meta(
            steps_run=4,
            steps_queued=5,
            max_steps=5,
            consecutive_empty=0,
            terminated_reason="complete",
            steps=steps,
            warnings=[],
        )
        self.assertIn("src/foo.py", meta.paths_surfaced)
        self.assertNotIn("task_keyword_search", meta.paths_surfaced)
        self.assertNotIn("src/bar.py", meta.paths_surfaced)
        self.assertNotIn("src/baz.py", meta.paths_surfaced)
        self.assertIn("src/qux.py", meta.paths_surfaced)

    def test_paths_surfaced_capped_at_5(self):
        steps = [
            OSNLoopStep(step_index=i, tool_name="read_file", target_label=f"src/f{i}.py", ok=True, empty=False)
            for i in range(8)
        ]
        meta = build_osn_loop_meta(
            steps_run=8, steps_queued=8, max_steps=8,
            consecutive_empty=0, terminated_reason="complete",
            steps=steps, warnings=[],
        )
        self.assertLessEqual(len(meta.paths_surfaced), 5)

    def test_truncated_flag_set_when_queued_exceeds_max(self):
        meta = build_osn_loop_meta(
            steps_run=5, steps_queued=9, max_steps=5,
            consecutive_empty=0, terminated_reason="max_steps",
            steps=[], warnings=[],
        )
        self.assertTrue(meta.truncated)

    def test_truncated_false_when_within_max(self):
        meta = build_osn_loop_meta(
            steps_run=3, steps_queued=3, max_steps=5,
            consecutive_empty=0, terminated_reason="complete",
            steps=[], warnings=[],
        )
        self.assertFalse(meta.truncated)

    def test_serializes_via_asdict(self):
        meta = OSNLoopMeta(enabled=True, steps_run=2, steps=[OSNLoopStep(step_index=0)])
        d = asdict(meta)
        json.dumps(d)


class TestBuildOSNLoopMeta(unittest.TestCase):

    def test_complete_run(self):
        steps = [OSNLoopStep(step_index=0, tool_name="read_file", target_label="foo.py", ok=True)]
        meta = build_osn_loop_meta(
            steps_run=1, steps_queued=1, max_steps=5,
            consecutive_empty=0, terminated_reason="complete",
            steps=steps, warnings=[],
        )
        self.assertTrue(meta.enabled)
        self.assertEqual(meta.terminated_reason, "complete")
        self.assertEqual(meta.steps_run, 1)

    def test_max_steps_termination(self):
        meta = build_osn_loop_meta(
            steps_run=5, steps_queued=10, max_steps=5,
            consecutive_empty=0, terminated_reason="max_steps",
            steps=[], warnings=[],
        )
        self.assertEqual(meta.terminated_reason, "max_steps")
        self.assertTrue(meta.truncated)

    def test_consecutive_empty_termination(self):
        meta = build_osn_loop_meta(
            steps_run=3, steps_queued=5, max_steps=5,
            consecutive_empty=2, terminated_reason="consecutive_empty",
            steps=[], warnings=[],
        )
        self.assertEqual(meta.terminated_reason, "consecutive_empty")

    def test_no_steps(self):
        meta = build_osn_loop_meta(
            steps_run=0, steps_queued=0, max_steps=5,
            consecutive_empty=0, terminated_reason="no_steps",
            steps=[], warnings=[],
        )
        self.assertEqual(meta.terminated_reason, "no_steps")
        self.assertEqual(meta.paths_surfaced, [])

    def test_no_paths_when_all_failed(self):
        steps = [OSNLoopStep(step_index=0, tool_name="read_file", target_label="foo.py", ok=False)]
        meta = build_osn_loop_meta(
            steps_run=1, steps_queued=1, max_steps=5,
            consecutive_empty=0, terminated_reason="complete",
            steps=steps, warnings=[],
        )
        self.assertEqual(meta.paths_surfaced, [])


class TestRenderOSNLoopContext(unittest.TestCase):

    def test_none_returns_empty(self):
        self.assertEqual(render_osn_loop_context(None), "")

    def test_disabled_returns_empty(self):
        meta = OSNLoopMeta(enabled=False)
        self.assertEqual(render_osn_loop_context(meta), "")

    def test_basic_render_has_header(self):
        meta = OSNLoopMeta(
            enabled=True, steps_run=3, max_steps=5, terminated_reason="complete",
            steps=[OSNLoopStep(tool_name="read_file", ok=True)],
            paths_surfaced=["src/foo.py"],
        )
        result = render_osn_loop_context(meta)
        self.assertIn("[osn loop]", result)
        self.assertIn("steps:", result)
        self.assertIn("reason:", result)

    def test_render_contains_permitted_content_only(self):
        meta = OSNLoopMeta(
            enabled=True, steps_run=2, max_steps=5, terminated_reason="complete",
            steps=[
                OSNLoopStep(tool_name="read_file", target_label="src/foo.py", ok=True),
                OSNLoopStep(tool_name="search_repo", target_label="task_keyword_search", ok=True),
            ],
            paths_surfaced=["src/foo.py"],
        )
        result = render_osn_loop_context(meta)
        self.assertNotIn("raw content", result)
        for line in result.splitlines():
            self.assertFalse(line.startswith("/"), f"absolute path leaked: {line!r}")
            self.assertFalse(line.startswith("C:\\"), f"Windows path leaked: {line!r}")

    def test_render_paths_capped_at_5(self):
        paths = [f"src/f{i}.py" for i in range(8)]
        meta = OSNLoopMeta(enabled=True, steps_run=5, max_steps=5, terminated_reason="complete", paths_surfaced=paths)
        result = render_osn_loop_context(meta)
        path_line = next((ln for ln in result.splitlines() if ln.startswith("paths:")), "")
        shown = [p.strip() for p in path_line.replace("paths:", "").split(",") if p.strip()]
        self.assertLessEqual(len(shown), 5)

    def test_no_paths_line_when_no_surfaced_paths(self):
        meta = OSNLoopMeta(enabled=True, steps_run=1, max_steps=5, terminated_reason="complete", paths_surfaced=[])
        result = render_osn_loop_context(meta)
        self.assertNotIn("paths:", result)

    def test_no_per_step_details_in_context(self):
        meta = OSNLoopMeta(
            enabled=True, steps_run=2, max_steps=5, terminated_reason="complete",
            steps=[
                OSNLoopStep(step_index=0, tool_name="read_file", target_label="src/foo.py", ok=True, output_chars=500),
                OSNLoopStep(step_index=1, tool_name="read_file", target_label="src/bar.py", ok=True, output_chars=200),
            ],
        )
        result = render_osn_loop_context(meta)
        self.assertNotIn("output_chars", result)
        self.assertNotIn("[0]", result)
        self.assertNotIn("[1]", result)


class TestRenderOSNLoop(unittest.TestCase):

    def test_none_returns_empty(self):
        self.assertEqual(render_osn_loop(None), "")

    def test_disabled_returns_empty(self):
        self.assertEqual(render_osn_loop(OSNLoopMeta(enabled=False)), "")

    def test_basic_summary(self):
        meta = OSNLoopMeta(enabled=True, steps_run=3, max_steps=5, terminated_reason="complete")
        result = render_osn_loop(meta)
        self.assertIn("3/5", result)
        self.assertIn("complete", result)

    def test_full_detail_shows_steps(self):
        meta = OSNLoopMeta(
            enabled=True, steps_run=2, max_steps=5, terminated_reason="complete",
            steps=[
                OSNLoopStep(step_index=0, tool_name="read_file", target_label="foo.py", ok=True, output_chars=100),
                OSNLoopStep(step_index=1, tool_name="search_repo", target_label="task_keyword_search", ok=True),
            ],
        )
        result = render_osn_loop(meta, detail="full")
        self.assertIn("read_file", result)
        self.assertIn("search_repo", result)

    def test_full_detail_caps_at_8_steps(self):
        steps = [OSNLoopStep(step_index=i, tool_name="read_file", target_label=f"f{i}.py", ok=True) for i in range(12)]
        meta = OSNLoopMeta(enabled=True, steps_run=12, max_steps=12, terminated_reason="max_steps", steps=steps)
        result = render_osn_loop(meta, detail="full")
        self.assertLessEqual(result.count("read_file"), 8)


class TestIsSafeRepoPath(unittest.TestCase):

    def _check(self, path: str, expected: bool):
        result = NativeAgentExecutor._is_safe_repo_path(path)
        self.assertEqual(result, expected, f"_is_safe_repo_path({path!r}) expected {expected}, got {result}")

    def test_posix_absolute_rejected(self):
        self._check("/etc/passwd", False)

    def test_windows_drive_backslash_rejected(self):
        self._check("C:\\Users\\foo", False)

    def test_windows_drive_forward_slash_rejected(self):
        self._check("C:/Users/foo", False)

    def test_home_relative_rejected(self):
        self._check("~/.ssh/id_rsa", False)

    def test_posix_traversal_rejected(self):
        self._check("../../evil.py", False)

    def test_backslash_traversal_rejected(self):
        self._check("..\\evil.py", False)

    def test_mixed_separator_traversal_rejected(self):
        self._check("src\\..\\evil.py", False)

    def test_deep_traversal_rejected(self):
        self._check("a/b/../../../evil.py", False)

    def test_simple_file_accepted(self):
        self._check("foo.py", True)

    def test_nested_path_accepted(self):
        self._check("src/foo.py", True)

    def test_test_path_accepted(self):
        self._check("tests/test_foo.py", True)

    def test_empty_rejected(self):
        self._check("", False)


class TestSanitizeTargetLabel(unittest.TestCase):

    def test_read_file_safe_path_kept(self):
        result = NativeAgentExecutor._sanitize_target_label("read_file", "src/foo.py")
        self.assertEqual(result, "src/foo.py")

    def test_read_file_backslash_normalized(self):
        result = NativeAgentExecutor._sanitize_target_label("read_file", "src\\foo.py")
        self.assertEqual(result, "src/foo.py")

    def test_read_file_absolute_rejected(self):
        result = NativeAgentExecutor._sanitize_target_label("read_file", "/etc/passwd")
        self.assertEqual(result, "")

    def test_read_file_windows_path_rejected(self):
        result = NativeAgentExecutor._sanitize_target_label("read_file", "C:\\Users\\foo")
        self.assertEqual(result, "")

    def test_read_file_traversal_rejected(self):
        result = NativeAgentExecutor._sanitize_target_label("read_file", "../../evil.py")
        self.assertEqual(result, "")

    def test_search_repo_always_generic(self):
        result = NativeAgentExecutor._sanitize_target_label("search_repo", "add user authentication feature")
        self.assertEqual(result, "task_keyword_search")

    def test_search_repo_with_any_query_always_generic(self):
        result = NativeAgentExecutor._sanitize_target_label("search_repo", "raw task text with lots of detail")
        self.assertEqual(result, "task_keyword_search")

    def test_truncated_at_200(self):
        long_path = "src/" + "a" * 300 + ".py"
        result = NativeAgentExecutor._sanitize_target_label("read_file", long_path)
        self.assertLessEqual(len(result), 200)

    def test_unknown_tool_returns_empty(self):
        result = NativeAgentExecutor._sanitize_target_label("write_file", "output.txt")
        self.assertEqual(result, "")


class TestBuildLoopStepQueue(unittest.TestCase):

    def test_empty_findings_and_evidence_returns_empty(self):
        executor, _ = _make_executor()
        executor.native_meta.read_search_findings = []
        executor.native_meta.evidence = None
        queue = executor._build_loop_step_queue("some task")
        read_file_steps = [q for q in queue if q[0] == "read_file" and q[2] != "task_keyword"]
        self.assertEqual(read_file_steps, [])

    def test_file_finding_queued_as_read_file(self):
        executor, _ = _make_executor()
        executor.native_meta.read_search_findings = ["file:src/foo.py"]
        executor.native_meta.evidence = None
        queue = executor._build_loop_step_queue("some task")
        self.assertTrue(any(t == "read_file" and "src/foo.py" in r for t, r, _ in queue))

    def test_test_marker_queued_as_read_file(self):
        executor, _ = _make_executor()
        executor.native_meta.read_search_findings = ["test-marker:tests/test_foo.py"]
        executor.native_meta.evidence = None
        queue = executor._build_loop_step_queue("some task")
        self.assertTrue(any(t == "read_file" and "tests/test_foo.py" in r for t, r, _ in queue))

    def test_absolute_path_rejected(self):
        executor, _ = _make_executor()
        executor.native_meta.read_search_findings = ["file:/etc/passwd"]
        executor.native_meta.evidence = None
        queue = executor._build_loop_step_queue("some task")
        for tool, raw, _ in queue:
            self.assertFalse(raw.startswith("/"), f"Absolute path leaked: {raw!r}")

    def test_windows_path_rejected(self):
        executor, _ = _make_executor()
        executor.native_meta.read_search_findings = ["file:C:\\Users\\foo"]
        executor.native_meta.evidence = None
        queue = executor._build_loop_step_queue("some task")
        for tool, raw, _ in queue:
            if tool == "read_file":
                self.assertNotIn("C:", raw, f"Windows path leaked: {raw!r}")

    def test_backslash_traversal_rejected(self):
        executor, _ = _make_executor()
        executor.native_meta.read_search_findings = ["file:..\\evil.py"]
        executor.native_meta.evidence = None
        queue = executor._build_loop_step_queue("some task")
        for tool, raw, _ in queue:
            if tool == "read_file":
                self.assertNotIn("evil.py", raw)

    def test_home_path_rejected(self):
        executor, _ = _make_executor()
        executor.native_meta.read_search_findings = ["file:~/secret.py"]
        executor.native_meta.evidence = None
        queue = executor._build_loop_step_queue("some task")
        for tool, raw, _ in queue:
            if tool == "read_file":
                self.assertNotIn("~/", raw)

    def test_dedup_same_path_not_queued_twice(self):
        executor, _ = _make_executor()
        executor.native_meta.read_search_findings = ["file:src/foo.py", "file:src/foo.py"]
        executor.native_meta.evidence = None
        queue = executor._build_loop_step_queue("some task")
        read_file_targets = [r for t, r, _ in queue if t == "read_file"]
        self.assertEqual(len(read_file_targets), len(set(read_file_targets)))

    def test_task_keyword_produces_search_repo(self):
        executor, _ = _make_executor()
        executor.native_meta.read_search_findings = []
        executor.native_meta.evidence = None
        queue = executor._build_loop_step_queue("add authentication feature")
        self.assertTrue(any(t == "search_repo" for t, _, _ in queue))

    def test_queue_never_exceeds_cap(self):
        executor, _ = _make_executor()
        executor.native_meta.read_search_findings = [f"file:src/f{i}.py" for i in range(20)]
        executor.native_meta.evidence = None
        queue = executor._build_loop_step_queue("some task")
        self.assertLessEqual(len(queue), _MAX_OSN_QUEUE_CAP)

    def test_no_blocked_tools_in_queue(self):
        executor, _ = _make_executor()
        executor.native_meta.read_search_findings = ["file:src/foo.py"]
        executor.native_meta.evidence = None
        queue = executor._build_loop_step_queue("some task")
        for tool, _, _ in queue:
            self.assertIn(tool, _LOOP_ALLOWED_TOOLS, f"Blocked tool in queue: {tool!r}")


class TestRunExperimentalLoop(unittest.TestCase):

    def test_no_crash_when_runner_is_none(self):
        executor, _ = _make_executor()
        executor._runner = None
        executor.native_meta.read_search_findings = ["file:src/foo.py"]
        executor._run_experimental_loop("some task")
        self.assertIsNone(executor.native_meta.osn_loop)

    def test_write_file_never_called(self):
        executor, _ = _make_executor()
        runner = MagicMock()
        runner.run.return_value = NativeToolResult(tool_name="read_file", ok=True, output="content")
        runner.trace_entry.return_value = {}
        executor._runner = runner
        executor.native_meta.read_search_findings = ["file:src/foo.py"]
        executor._run_experimental_loop("some task")
        for call in runner.run.call_args_list:
            args = call[0]
            if args:
                self.assertNotEqual(args[0].tool_name, "write_file")

    def test_run_command_never_called(self):
        executor, _ = _make_executor()
        runner = MagicMock()
        runner.run.return_value = NativeToolResult(tool_name="read_file", ok=True, output="content")
        runner.trace_entry.return_value = {}
        executor._runner = runner
        executor.native_meta.read_search_findings = ["file:src/foo.py"]
        executor._run_experimental_loop("some task")
        for call in runner.run.call_args_list:
            args = call[0]
            if args:
                self.assertNotEqual(args[0].tool_name, "run_command")

    def test_run_verification_never_called(self):
        executor, _ = _make_executor()
        runner = MagicMock()
        runner.run.return_value = NativeToolResult(tool_name="read_file", ok=True, output="content")
        runner.trace_entry.return_value = {}
        executor._runner = runner
        executor.native_meta.read_search_findings = ["file:src/foo.py"]
        executor._run_experimental_loop("some task")
        for call in runner.run.call_args_list:
            args = call[0]
            if args:
                self.assertNotEqual(args[0].tool_name, "run_verification")

    def test_max_steps_enforced(self):
        executor, _ = _make_executor()
        runner = MagicMock()
        runner.run.return_value = NativeToolResult(tool_name="read_file", ok=True, output="content")
        runner.trace_entry.return_value = {}
        executor._runner = runner
        executor.native_meta.read_search_findings = [f"file:src/f{i}.py" for i in range(20)]
        executor.native_meta.evidence = None
        executor._run_experimental_loop("some task")
        self.assertLessEqual(runner.run.call_count, _MAX_OSN_LOOP_STEPS)

    def test_consecutive_empty_triggers_early_exit(self):
        executor, _ = _make_executor()
        runner = MagicMock()
        runner.run.side_effect = [
            NativeToolResult(tool_name="read_file", ok=True, output=""),
            NativeToolResult(tool_name="read_file", ok=True, output=""),
            NativeToolResult(tool_name="read_file", ok=True, output="content"),
        ]
        runner.trace_entry.return_value = {}
        executor._runner = runner
        executor.native_meta.read_search_findings = [f"file:src/f{i}.py" for i in range(5)]
        executor._run_experimental_loop("some task")
        meta = executor.native_meta.osn_loop
        self.assertEqual(meta.terminated_reason, "consecutive_empty")
        self.assertLessEqual(runner.run.call_count, 3)

    def test_osn_loop_meta_populated(self):
        executor, _ = _make_executor()
        runner = MagicMock()
        runner.run.return_value = NativeToolResult(tool_name="read_file", ok=True, output="content")
        runner.trace_entry.return_value = {}
        executor._runner = runner
        executor.native_meta.read_search_findings = ["file:src/foo.py"]
        executor._run_experimental_loop("some task")
        meta = executor.native_meta.osn_loop
        self.assertIsNotNone(meta)
        self.assertTrue(meta.enabled)

    def test_osn_loop_in_trace_phases(self):
        executor, _ = _make_executor()
        runner = MagicMock()
        runner.run.return_value = NativeToolResult(tool_name="read_file", ok=True, output="content")
        runner.trace_entry.return_value = {}
        executor._runner = runner
        executor.native_meta.read_search_findings = ["file:src/foo.py"]
        executor._run_experimental_loop("some task")
        self.assertIn("osn_loop", executor.native_meta.native_loop_trace.phases())

    def test_tool_trace_extended(self):
        executor, _ = _make_executor()
        runner = MagicMock()
        runner.run.return_value = NativeToolResult(tool_name="read_file", ok=True, output="content")
        runner.trace_entry.return_value = {"tool": "read_file", "ok": True, "approved": False, "output_chars": 7, "error": None}
        executor._runner = runner
        executor.native_meta.read_search_findings = ["file:src/foo.py"]
        initial_trace_len = len(executor.native_meta.tool_trace)
        executor._run_experimental_loop("some task")
        self.assertGreater(len(executor.native_meta.tool_trace), initial_trace_len)

    def test_terminated_reason_valid_values(self):
        valid = {"not_run", "complete", "max_steps", "consecutive_empty", "no_steps"}
        executor, _ = _make_executor()
        runner = MagicMock()
        runner.run.return_value = NativeToolResult(tool_name="read_file", ok=True, output="content")
        runner.trace_entry.return_value = {}
        executor._runner = runner
        executor.native_meta.read_search_findings = ["file:src/foo.py"]
        executor._run_experimental_loop("some task")
        self.assertIn(executor.native_meta.osn_loop.terminated_reason, valid)

    def test_no_steps_queued_gives_no_steps_reason(self):
        executor, _ = _make_executor()
        runner = MagicMock()
        runner.trace_entry.return_value = {}
        executor._runner = runner
        executor.native_meta.read_search_findings = []
        executor.native_meta.evidence = None
        executor._run_experimental_loop("the")
        meta = executor.native_meta.osn_loop
        self.assertIsNotNone(meta)
        self.assertEqual(meta.terminated_reason, "no_steps")
        runner.run.assert_not_called()

    def test_blocked_tool_skipped_not_executed(self):
        executor, _ = _make_executor()
        runner = MagicMock()
        runner.run.return_value = NativeToolResult(tool_name="read_file", ok=True, output="content")
        runner.trace_entry.return_value = {}
        executor._runner = runner

        from unittest.mock import patch
        with patch.object(executor, "_build_loop_step_queue", return_value=[
            ("write_file", "output.txt", "test"),
            ("read_file", "src/foo.py", "read_search_finding"),
        ]):
            executor._run_experimental_loop("some task")

        for call in runner.run.call_args_list:
            args = call[0]
            if args:
                self.assertNotEqual(args[0].tool_name, "write_file")

        meta = executor.native_meta.osn_loop
        skipped = [s for s in meta.steps if s.skipped]
        self.assertTrue(any(s.tool_name == "write_file" for s in skipped))


class TestOSNLoopSerializationRoundtrip(unittest.TestCase):

    def test_osn_loop_meta_json_serializable(self):
        from openshard.native.executor import NativeRunMeta
        meta = NativeRunMeta()
        meta.osn_loop = OSNLoopMeta(
            enabled=True, steps_run=2, steps_queued=2, max_steps=5,
            consecutive_empty=0, terminated_reason="complete",
            steps=[OSNLoopStep(step_index=0, tool_name="read_file", target_label="src/foo.py", ok=True, output_chars=100)],
            paths_surfaced=["src/foo.py"],
        )
        d = asdict(meta)
        json.dumps(d)

    def test_native_run_meta_osn_loop_defaults_none(self):
        from openshard.native.executor import NativeRunMeta
        meta = NativeRunMeta()
        self.assertIsNone(meta.osn_loop)
        d = asdict(meta)
        self.assertIsNone(d["osn_loop"])


if __name__ == "__main__":
    unittest.main()
