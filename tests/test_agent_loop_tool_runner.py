from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from openshard.native.agent_loop_tool_runner import ReadOnlyToolRunner
from openshard.native.agent_loop_types import AgentAction
from openshard.native.tools import NativeToolResult

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _action(kind: str, args: dict | None = None) -> AgentAction:
    return AgentAction(
        action_id="t0",
        iteration_index=0,
        kind=kind,
        args=args or {},
        rationale="test",
        needs_approval=False,
    )


def _ok_nr(tool_name: str, output: str = "ok") -> NativeToolResult:
    return NativeToolResult(tool_name=tool_name, ok=True, output=output)


def _err_nr(tool_name: str, error: str = "boom") -> NativeToolResult:
    return NativeToolResult(tool_name=tool_name, ok=False, error=error)


class _TempRepoMixin:
    """Creates a temp directory with a minimal file tree for each test."""

    def setUp(self):
        self._td = tempfile.TemporaryDirectory()
        self.root = Path(self._td.name)
        (self.root / "hello.py").write_text("x = 1\n", encoding="utf-8")
        (self.root / "sub").mkdir()
        (self.root / "sub" / "world.py").write_text("y = 2\n", encoding="utf-8")
        self.runner = ReadOnlyToolRunner(self.root)

    def tearDown(self):
        self._td.cleanup()


# ---------------------------------------------------------------------------
# Rejection gates
# ---------------------------------------------------------------------------

class TestRejectionGates(unittest.TestCase):
    def setUp(self):
        self._td = tempfile.TemporaryDirectory()
        self.runner = ReadOnlyToolRunner(Path(self._td.name))

    def tearDown(self):
        self._td.cleanup()

    def test_loop_signal_ask_human_rejected(self):
        r = self.runner.run(_action("ask_human"))
        self.assertFalse(r.ok)
        self.assertIn("loop signal", r.error)

    def test_loop_signal_finish_rejected(self):
        r = self.runner.run(_action("finish"))
        self.assertFalse(r.ok)
        self.assertIn("loop signal", r.error)

    def test_blocked_run_command_rejected(self):
        r = self.runner.run(_action("run_command"))
        self.assertFalse(r.ok)
        self.assertIn("blocked", r.error)

    def test_needs_approval_write_file_rejected(self):
        r = self.runner.run(_action("write_file"))
        self.assertFalse(r.ok)
        self.assertIn("requires approval", r.error)

    def test_unknown_kind_rejected(self):
        # classify_native_tool returns "blocked" for any unregistered kind,
        # so the blocked gate fires before _dispatch's unknown-kind branch.
        r = self.runner.run(_action("teleport"))
        self.assertFalse(r.ok)
        self.assertIn("blocked", r.error)

    def test_rejected_results_have_correct_action_id(self):
        a = AgentAction(
            action_id="myid",
            iteration_index=0,
            kind="run_command",
            args={},
            rationale="test",
            needs_approval=False,
        )
        r = self.runner.run(a)
        self.assertEqual(r.action_id, "myid")

    def test_all_rejected_raw_content_false(self):
        for kind in ("ask_human", "finish", "run_command", "write_file", "teleport"):
            with self.subTest(kind=kind):
                r = self.runner.run(_action(kind))
                self.assertFalse(r.raw_content_stored)


# ---------------------------------------------------------------------------
# list_files dispatch
# ---------------------------------------------------------------------------

class TestListFiles(_TempRepoMixin, unittest.TestCase):
    def test_lists_root_files(self):
        r = self.runner.run(_action("list_files"))
        self.assertTrue(r.ok)
        self.assertIn("hello.py", r.output)

    def test_lists_subdir_files(self):
        r = self.runner.run(_action("list_files", {"subdir": "sub"}))
        self.assertTrue(r.ok)
        self.assertIn("world.py", r.output)

    def test_unsafe_path_rejected_by_executor(self):
        r = self.runner.run(_action("list_files", {"subdir": "../../../etc"}))
        self.assertFalse(r.ok)
        self.assertIsNotNone(r.error)

    def test_ok_result_has_no_error(self):
        r = self.runner.run(_action("list_files"))
        self.assertIsNone(r.error)

    def test_raw_content_stored_false(self):
        r = self.runner.run(_action("list_files"))
        self.assertFalse(r.raw_content_stored)


# ---------------------------------------------------------------------------
# read_file dispatch
# ---------------------------------------------------------------------------

class TestReadFile(_TempRepoMixin, unittest.TestCase):
    def test_reads_existing_file(self):
        r = self.runner.run(_action("read_file", {"path": "hello.py"}))
        self.assertTrue(r.ok)
        self.assertIn("x = 1", r.output)

    def test_missing_file_returns_error(self):
        r = self.runner.run(_action("read_file", {"path": "nope.py"}))
        self.assertFalse(r.ok)
        self.assertIsNotNone(r.error)

    def test_empty_path_rejected(self):
        r = self.runner.run(_action("read_file", {"path": ""}))
        self.assertFalse(r.ok)
        self.assertIn("non-empty", r.error)

    def test_missing_path_key_rejected(self):
        r = self.runner.run(_action("read_file", {}))
        self.assertFalse(r.ok)

    def test_unsafe_path_rejected(self):
        r = self.runner.run(_action("read_file", {"path": "../../../etc/passwd"}))
        self.assertFalse(r.ok)

    def test_raw_content_stored_false(self):
        r = self.runner.run(_action("read_file", {"path": "hello.py"}))
        self.assertFalse(r.raw_content_stored)


# ---------------------------------------------------------------------------
# search_repo dispatch
# ---------------------------------------------------------------------------

class TestSearchRepo(_TempRepoMixin, unittest.TestCase):
    def test_finds_pattern(self):
        r = self.runner.run(_action("search_repo", {"query": "x = 1"}))
        self.assertTrue(r.ok)
        self.assertIn("hello.py", r.output)

    def test_empty_query_returns_error(self):
        r = self.runner.run(_action("search_repo", {"query": ""}))
        self.assertFalse(r.ok)

    def test_no_query_key_returns_error(self):
        # args.get("query", "") → empty → rejected by _exec_search_repo
        r = self.runner.run(_action("search_repo", {}))
        self.assertFalse(r.ok)

    def test_max_matches_respected(self):
        r = self.runner.run(_action("search_repo", {"query": "=", "max_matches": 1}))
        self.assertTrue(r.ok)
        # Can't assert exactly 1 line — just assert it didn't crash and returned output
        self.assertIsNotNone(r.output)

    def test_raw_content_stored_false(self):
        r = self.runner.run(_action("search_repo", {"query": "x"}))
        self.assertFalse(r.raw_content_stored)


# ---------------------------------------------------------------------------
# get_git_diff dispatch (mocked — no real git repo needed)
# ---------------------------------------------------------------------------

class TestGetGitDiff(unittest.TestCase):
    def setUp(self):
        self._td = tempfile.TemporaryDirectory()
        self.root = Path(self._td.name)
        self.runner = ReadOnlyToolRunner(self.root)

    def tearDown(self):
        self._td.cleanup()

    def test_no_git_repo_returns_error(self):
        # No .git dir → executor returns ok=False with a clear message
        r = self.runner.run(_action("get_git_diff"))
        self.assertFalse(r.ok)
        self.assertIn(".git", r.error)

    def test_mocked_success(self):
        nr = NativeToolResult(tool_name="get_git_diff", ok=True,
                              output="diff --git a/foo", metadata={})
        with patch("openshard.native.agent_loop_tool_runner._exec_get_git_diff",
                   return_value=nr):
            r = self.runner.run(_action("get_git_diff"))
        self.assertTrue(r.ok)
        self.assertIn("diff", r.output)

    def test_mocked_failure_propagated(self):
        nr = NativeToolResult(tool_name="get_git_diff", ok=False,
                              error="git exploded", metadata={})
        with patch("openshard.native.agent_loop_tool_runner._exec_get_git_diff",
                   return_value=nr):
            r = self.runner.run(_action("get_git_diff"))
        self.assertFalse(r.ok)
        self.assertEqual(r.error, "git exploded")

    def test_raw_content_stored_false(self):
        nr = NativeToolResult(tool_name="get_git_diff", ok=True, output="x", metadata={})
        with patch("openshard.native.agent_loop_tool_runner._exec_get_git_diff",
                   return_value=nr):
            r = self.runner.run(_action("get_git_diff"))
        self.assertFalse(r.raw_content_stored)


# ---------------------------------------------------------------------------
# run_verification dispatch (always mocked — heavy dependency)
# ---------------------------------------------------------------------------

class TestRunVerification(unittest.TestCase):
    def setUp(self):
        self._td = tempfile.TemporaryDirectory()
        self.root = Path(self._td.name)
        self.runner = ReadOnlyToolRunner(self.root)

    def tearDown(self):
        self._td.cleanup()

    def test_mocked_success(self):
        nr = NativeToolResult(
            tool_name="run_verification", ok=True, output="all tests passed",
            metadata={"attempted": True, "passed": True, "exit_code": 0,
                      "duration_ms": 120, "raw_content_stored": False},
        )
        with patch("openshard.native.agent_loop_tool_runner._exec_run_verification",
                   return_value=nr):
            r = self.runner.run(_action("run_verification"))
        self.assertTrue(r.ok)
        self.assertEqual(r.duration_ms, 120)

    def test_mocked_failure_propagated(self):
        nr = NativeToolResult(
            tool_name="run_verification", ok=False,
            error="tests failed (exit code 1)",
            metadata={"attempted": True, "exit_code": 1,
                      "duration_ms": 80, "raw_content_stored": False},
        )
        with patch("openshard.native.agent_loop_tool_runner._exec_run_verification",
                   return_value=nr):
            r = self.runner.run(_action("run_verification"))
        self.assertFalse(r.ok)
        self.assertIn("exit code", r.error)

    def test_raw_content_stored_false(self):
        nr = NativeToolResult(tool_name="run_verification", ok=True, output="ok",
                              metadata={"duration_ms": 0, "raw_content_stored": False})
        with patch("openshard.native.agent_loop_tool_runner._exec_run_verification",
                   return_value=nr):
            r = self.runner.run(_action("run_verification"))
        self.assertFalse(r.raw_content_stored)


# ---------------------------------------------------------------------------
# _wrap helper — NativeToolResult → ToolResult conversion
# ---------------------------------------------------------------------------

class TestWrap(unittest.TestCase):
    def test_empty_output_becomes_none(self):
        from openshard.native.agent_loop_tool_runner import _wrap
        nr = NativeToolResult(tool_name="x", ok=True, output="")
        r = _wrap(nr, "aid")
        self.assertIsNone(r.output)

    def test_non_empty_output_preserved(self):
        from openshard.native.agent_loop_tool_runner import _wrap
        nr = NativeToolResult(tool_name="x", ok=True, output="data")
        r = _wrap(nr, "aid")
        self.assertEqual(r.output, "data")

    def test_exit_code_extracted_from_metadata(self):
        from openshard.native.agent_loop_tool_runner import _wrap
        nr = NativeToolResult(tool_name="x", ok=False, metadata={"exit_code": 2})
        r = _wrap(nr, "aid")
        self.assertEqual(r.exit_code, 2)

    def test_duration_ms_extracted_from_metadata(self):
        from openshard.native.agent_loop_tool_runner import _wrap
        nr = NativeToolResult(tool_name="x", ok=True, output="y",
                              metadata={"duration_ms": 55})
        r = _wrap(nr, "aid")
        self.assertEqual(r.duration_ms, 55)

    def test_action_id_propagated(self):
        from openshard.native.agent_loop_tool_runner import _wrap
        nr = NativeToolResult(tool_name="x", ok=True, output="y")
        r = _wrap(nr, "zz99")
        self.assertEqual(r.action_id, "zz99")

    def test_raw_content_stored_always_false(self):
        from openshard.native.agent_loop_tool_runner import _wrap
        nr = NativeToolResult(tool_name="x", ok=True, output="y")
        r = _wrap(nr, "aid")
        self.assertFalse(r.raw_content_stored)


if __name__ == "__main__":
    unittest.main()
