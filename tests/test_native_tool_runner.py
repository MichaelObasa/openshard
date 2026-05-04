from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from openshard.native.tool_runner import NativeToolRunner
from openshard.native.tools import NativeToolCall


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


if __name__ == "__main__":
    unittest.main()
