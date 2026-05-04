from __future__ import annotations

import subprocess
import tempfile
import unittest
from pathlib import Path

from openshard.native.tool_runner import NativeToolRunner
from openshard.native.tools import NativeToolCall


def _run_git(root: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=root, check=True, capture_output=True, text=True)


def _make_runner(tmp_path: Path) -> NativeToolRunner:
    return NativeToolRunner(repo_root=tmp_path)


class TestNativeToolRunnerListFiles(unittest.TestCase):
    def test_run_list_files_ok(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "foo.py").write_text("x = 1")
            runner = _make_runner(root)
            call = NativeToolCall(tool_name="list_files", args={"subdir": "."})
            result = runner.run(call)
        self.assertTrue(result.ok)
        self.assertIn("foo.py", result.output)
        self.assertIsNone(result.error)

    def test_run_list_files_default_subdir(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "bar.py").write_text("")
            runner = _make_runner(root)
            call = NativeToolCall(tool_name="list_files", args={})
            result = runner.run(call)
        self.assertTrue(result.ok)
        self.assertIn("bar.py", result.output)


class TestNativeToolRunnerReadFile(unittest.TestCase):
    def test_run_read_file_ok(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "hello.py").write_text("print('hello')")
            runner = _make_runner(root)
            call = NativeToolCall(tool_name="read_file", args={"path": "hello.py"})
            result = runner.run(call)
        self.assertTrue(result.ok)
        self.assertIn("print('hello')", result.output)
        self.assertIsNone(result.error)


class TestNativeToolRunnerBlockedAndUnknown(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self._runner = _make_runner(Path(self._tmp.name))

    def tearDown(self):
        self._tmp.cleanup()

    def test_unknown_tool_returns_error(self):
        call = NativeToolCall(tool_name="no_such_tool", args={})
        result = self._runner.run(call)
        self.assertFalse(result.ok)
        self.assertIsNotNone(result.error)

    def test_blocked_run_command_returns_error(self):
        call = NativeToolCall(tool_name="run_command", args={"cmd": "ls"})
        result = self._runner.run(call)
        self.assertFalse(result.ok)
        self.assertIsNotNone(result.error)

    def test_write_file_without_approval_returns_error(self):
        call = NativeToolCall(tool_name="write_file", args={}, approved=False)
        result = self._runner.run(call)
        self.assertFalse(result.ok)
        self.assertIsNotNone(result.error)

    def test_write_file_with_approval_still_returns_error(self):
        call = NativeToolCall(tool_name="write_file", args={}, approved=True)
        result = self._runner.run(call)
        self.assertFalse(result.ok)
        self.assertIsNotNone(result.error)

    def test_malformed_args_does_not_crash(self):
        call = NativeToolCall(tool_name="list_files", args=None)  # type: ignore[arg-type]
        result = self._runner.run(call)
        self.assertIsInstance(result.ok, bool)


class TestNativeToolRunnerSearchRepo(unittest.TestCase):
    def test_run_search_repo_ok(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "greet.py").write_text("def hello():\n    pass\n")
            runner = _make_runner(root)
            call = NativeToolCall(tool_name="search_repo", args={"query": "hello"})
            result = runner.run(call)
        self.assertTrue(result.ok)
        self.assertIn("greet.py", result.output)
        self.assertIsNone(result.error)

    def test_run_search_repo_empty_query_returns_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            runner = _make_runner(root)
            call = NativeToolCall(tool_name="search_repo", args={"query": ""})
            result = runner.run(call)
        self.assertFalse(result.ok)
        self.assertIsNotNone(result.error)

    def test_run_search_repo_missing_query_key_returns_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            runner = _make_runner(root)
            call = NativeToolCall(tool_name="search_repo", args={})
            result = runner.run(call)
        self.assertFalse(result.ok)

    def test_run_search_repo_respects_max_matches(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "many.py").write_text("\n".join(["hit"] * 30))
            runner = _make_runner(root)
            call = NativeToolCall(
                tool_name="search_repo",
                args={"query": "hit", "max_matches": 5},
            )
            result = runner.run(call)
        self.assertTrue(result.ok)
        self.assertLessEqual(result.metadata["matches"], 5)
        self.assertTrue(result.metadata["truncated"])

    def test_trace_entry_search_repo_no_full_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "sample.py").write_text("find me here\n")
            runner = _make_runner(root)
            call = NativeToolCall(tool_name="search_repo", args={"query": "find me"})
            result = runner.run(call)
            entry = runner.trace_entry(call, result)
        self.assertIn("tool", entry)
        self.assertIn("ok", entry)
        self.assertIn("output_chars", entry)
        self.assertNotIn("output", entry)
        self.assertEqual(entry["output_chars"], len(result.output))


class TestNativeToolRunnerTraceEntry(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self._root = Path(self._tmp.name)
        (self._root / "sample.py").write_text("x = 1")
        self._runner = _make_runner(self._root)

    def tearDown(self):
        self._tmp.cleanup()

    def test_trace_entry_structure(self):
        call = NativeToolCall(tool_name="list_files", args={})
        result = self._runner.run(call)
        entry = self._runner.trace_entry(call, result)
        self.assertIn("tool", entry)
        self.assertIn("ok", entry)
        self.assertIn("approved", entry)
        self.assertIn("output_chars", entry)
        self.assertIn("error", entry)

    def test_trace_entry_no_full_output(self):
        call = NativeToolCall(tool_name="read_file", args={"path": "sample.py"})
        result = self._runner.run(call)
        entry = self._runner.trace_entry(call, result)
        self.assertNotIn("output", entry)

    def test_trace_entry_output_chars_matches_output_length(self):
        call = NativeToolCall(tool_name="list_files", args={})
        result = self._runner.run(call)
        entry = self._runner.trace_entry(call, result)
        self.assertEqual(entry["output_chars"], len(result.output))

    def test_trace_entry_ok_false_for_blocked(self):
        call = NativeToolCall(tool_name="run_command", args={})
        result = self._runner.run(call)
        entry = self._runner.trace_entry(call, result)
        self.assertFalse(entry["ok"])
        self.assertIsNotNone(entry["error"])

    def test_trace_entry_approved_reflects_call(self):
        call = NativeToolCall(tool_name="list_files", args={}, approved=True)
        result = self._runner.run(call)
        entry = self._runner.trace_entry(call, result)
        self.assertTrue(entry["approved"])


class TestNativeToolRunnerGitDiff(unittest.TestCase):
    def test_run_get_git_diff_ok(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _run_git(root, "init")
            (root / "change.txt").write_text("original\n")
            _run_git(root, "add", "change.txt")
            (root / "change.txt").write_text("modified\n")
            runner = _make_runner(root)
            call = NativeToolCall(tool_name="get_git_diff", args={})
            result = runner.run(call)
        self.assertTrue(result.ok)
        self.assertIn("change.txt", result.output)

    def test_run_get_git_diff_non_git_repo_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            runner = _make_runner(root)
            call = NativeToolCall(tool_name="get_git_diff", args={})
            result = runner.run(call)
        self.assertFalse(result.ok)

    def test_trace_entry_get_git_diff_no_full_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _run_git(root, "init")
            (root / "t.txt").write_text("a\n")
            _run_git(root, "add", "t.txt")
            (root / "t.txt").write_text("b\n")
            runner = _make_runner(root)
            call = NativeToolCall(tool_name="get_git_diff", args={})
            result = runner.run(call)
            entry = runner.trace_entry(call, result)
        self.assertIn("output_chars", entry)
        self.assertNotIn("output", entry)
        self.assertEqual(entry["output_chars"], len(result.output))

    def test_run_get_git_diff_respects_limit(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _run_git(root, "init")
            (root / "big.txt").write_text("before\n")
            _run_git(root, "add", "big.txt")
            (root / "big.txt").write_text("after\n" * 1000)
            runner = _make_runner(root)
            call = NativeToolCall(tool_name="get_git_diff", args={"limit": 200})
            result = runner.run(call)
        self.assertTrue(result.ok)
        self.assertTrue(result.metadata["truncated"])


if __name__ == "__main__":
    unittest.main()
