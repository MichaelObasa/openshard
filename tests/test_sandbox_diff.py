from __future__ import annotations

import json
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from click.testing import CliRunner

from openshard.cli.main import cli
from openshard.native.sandbox_diff import (
    SandboxDiffResult,
    get_sandbox_diff,
    redact_diff_text,
)

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _make_entry(**kwargs) -> dict:
    return {
        "task": "test task",
        "timestamp": "2025-01-01T00:00:00Z",
        "execution_model": "anthropic/claude-sonnet-4.6",
        "executor": "native",
        "workflow": "native",
        "duration_seconds": 1.0,
        "files_created": 0,
        "files_updated": 0,
        "files_deleted": 0,
        "retry_triggered": False,
        "verification_attempted": False,
        "verification_passed": None,
        "workspace_path": None,
        "summary": "",
        "files_detail": [],
        "sandbox": None,
        **kwargs,
    }


def _make_sandbox_meta(sandbox_type: str = "worktree", worktree_path: str | None = None) -> dict:
    return {
        "sandbox_enabled": True,
        "sandbox_type": sandbox_type,
        "worktree_path": worktree_path,
        "worktree_branch": "osn/run-test",
        "fallback_reason": None,
    }


def _write_runs(entries: list[dict]) -> Path:
    log_dir = Path(".openshard")
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "runs.jsonl"
    with log_path.open("w", encoding="utf-8") as fh:
        for entry in entries:
            fh.write(json.dumps(entry) + "\n")
    return log_path


def _make_proc(stdout: str = "", stderr: str = "", returncode: int = 0):
    m = MagicMock()
    m.stdout = stdout
    m.stderr = stderr
    m.returncode = returncode
    return m


# ---------------------------------------------------------------------------
# TestSandboxDiffResult
# ---------------------------------------------------------------------------

class TestSandboxDiffResult(unittest.TestCase):

    def test_defaults_correct(self):
        r = SandboxDiffResult()
        self.assertFalse(r.available)
        self.assertEqual(r.sandbox_path, "")
        self.assertEqual(r.files_changed, [])
        self.assertEqual(r.diff_text, "")
        self.assertEqual(r.stat_text, "")
        self.assertEqual(r.reason, "")
        self.assertFalse(r.raw_content_stored)

    def test_raw_content_stored_forced_false(self):
        r = SandboxDiffResult(raw_content_stored=True)
        self.assertFalse(r.raw_content_stored)

    def test_list_defaults_are_independent(self):
        r1 = SandboxDiffResult()
        r2 = SandboxDiffResult()
        r1.files_changed.append("foo.py")
        self.assertEqual(r2.files_changed, [])


# ---------------------------------------------------------------------------
# TestRedactDiffText
# ---------------------------------------------------------------------------

class TestRedactDiffText(unittest.TestCase):

    def test_normal_lines_unchanged(self):
        text = "+def foo():\n-    pass\n context line"
        result = redact_diff_text(text)
        self.assertIn("+def foo():", result)
        self.assertIn("-    pass", result)
        self.assertIn(" context line", result)

    def test_api_key_line_redacted(self):
        text = "+api_key = 'abc123'\n other line"
        result = redact_diff_text(text)
        self.assertIn("[redacted possible secret line]", result)
        self.assertIn("other line", result)

    def test_token_line_redacted(self):
        text = "+token = 'xyz'\n normal"
        result = redact_diff_text(text)
        lines = result.splitlines()
        self.assertEqual(lines[0], "[redacted possible secret line]")

    def test_password_line_redacted(self):
        text = " password: supersecret"
        result = redact_diff_text(text)
        self.assertIn("[redacted possible secret line]", result)

    def test_case_insensitive_redaction(self):
        text = "+API_KEY = 'value'\n+Token: bearer xyz\n plain"
        result = redact_diff_text(text)
        lines = result.splitlines()
        self.assertEqual(lines[0], "[redacted possible secret line]")
        self.assertEqual(lines[1], "[redacted possible secret line]")
        self.assertEqual(lines[2], " plain")

    def test_original_secret_value_not_leaked(self):
        secret = "supersecretvalue99"
        text = f"+api_key = '{secret}'"
        result = redact_diff_text(text)
        self.assertNotIn(secret, result)


# ---------------------------------------------------------------------------
# TestGetSandboxDiff
# ---------------------------------------------------------------------------

class TestGetSandboxDiff(unittest.TestCase):

    def test_stat_text_returned(self):
        stat_out = " foo.py | 3 +++\n 1 file changed"
        with patch("subprocess.run", side_effect=[
            _make_proc(stdout=stat_out),
            _make_proc(stdout="foo.py\n"),
            _make_proc(stdout=""),
        ]):
            result = get_sandbox_diff(Path("/fake/sandbox"))
        self.assertTrue(result.available)
        self.assertEqual(result.stat_text, stat_out.strip())

    def test_full_diff_only_called_when_full_true(self):
        with patch("subprocess.run", side_effect=[
            _make_proc(stdout=" foo.py | 1 +"),
            _make_proc(stdout="foo.py\n"),
            _make_proc(stdout=""),
        ]) as mock_run:
            get_sandbox_diff(Path("/fake/sandbox"), full=False)
        calls = [c.args[0] for c in mock_run.call_args_list]
        self.assertNotIn(["git", "diff"], calls)

    def test_full_diff_called_when_full_true(self):
        with patch("subprocess.run", side_effect=[
            _make_proc(stdout=" foo.py | 1 +"),
            _make_proc(stdout="foo.py\n"),
            _make_proc(stdout=""),
            _make_proc(stdout="+added line\n"),
        ]) as mock_run:
            result = get_sandbox_diff(Path("/fake/sandbox"), full=True)
        calls = [c.args[0] for c in mock_run.call_args_list]
        self.assertIn(["git", "diff"], calls)
        self.assertNotEqual(result.diff_text, "")

    def test_untracked_files_in_files_changed(self):
        with patch("subprocess.run", side_effect=[
            _make_proc(stdout=" existing.py | 2 ++"),
            _make_proc(stdout="existing.py\n"),
            _make_proc(stdout="new_file.py\n"),
        ]):
            result = get_sandbox_diff(Path("/fake/sandbox"))
        self.assertIn("new_file.py", result.files_changed)
        self.assertIn("existing.py", result.files_changed)

    def test_git_failure_returns_available_false(self):
        with patch("subprocess.run", return_value=_make_proc(returncode=1, stderr="not a git repo")):
            result = get_sandbox_diff(Path("/fake/sandbox"))
        self.assertFalse(result.available)
        self.assertIn("not a git repo", result.reason)

    def test_git_failure_with_empty_stderr_has_fallback_reason(self):
        with patch("subprocess.run", return_value=_make_proc(returncode=1, stderr="")):
            result = get_sandbox_diff(Path("/fake/sandbox"))
        self.assertFalse(result.available)
        self.assertNotEqual(result.reason, "")

    def test_diff_text_is_redacted(self):
        secret = "topsecrettoken"
        diff_out = f"+token = '{secret}'\n+normal line\n"
        with patch("subprocess.run", side_effect=[
            _make_proc(stdout=" foo.py | 1 +"),
            _make_proc(stdout="foo.py\n"),
            _make_proc(stdout=""),
            _make_proc(stdout=diff_out),
        ]):
            result = get_sandbox_diff(Path("/fake/sandbox"), full=True)
        self.assertNotIn(secret, result.diff_text)
        self.assertIn("[redacted possible secret line]", result.diff_text)

    def test_raw_content_stored_always_false(self):
        with patch("subprocess.run", side_effect=[
            _make_proc(stdout=" foo.py | 1 +"),
            _make_proc(stdout="foo.py\n"),
            _make_proc(stdout=""),
            _make_proc(stdout="+line\n"),
        ]):
            result = get_sandbox_diff(Path("/fake/sandbox"), full=True)
        self.assertFalse(result.raw_content_stored)

    def test_exception_returns_available_false(self):
        with patch("subprocess.run", side_effect=FileNotFoundError("git not found")):
            result = get_sandbox_diff(Path("/fake/sandbox"))
        self.assertFalse(result.available)
        self.assertIn("git not found", result.reason)

    def test_no_duplicate_files_in_changed(self):
        with patch("subprocess.run", side_effect=[
            _make_proc(stdout=" foo.py | 1 +"),
            _make_proc(stdout="foo.py\n"),
            _make_proc(stdout="foo.py\n"),
        ]):
            result = get_sandbox_diff(Path("/fake/sandbox"))
        self.assertEqual(result.files_changed.count("foo.py"), 1)


# ---------------------------------------------------------------------------
# TestDiffLastCLI
# ---------------------------------------------------------------------------

class TestDiffLastCLI(unittest.TestCase):

    def test_no_history_message(self):
        runner = CliRunner()
        with runner.isolated_filesystem():
            result = runner.invoke(cli, ["diff-last"])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("No run history found", result.output)

    def test_non_native_run_message(self):
        runner = CliRunner()
        with runner.isolated_filesystem():
            _write_runs([_make_entry(executor="direct")])
            result = runner.invoke(cli, ["diff-last"])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("not a native run", result.output)

    def test_no_sandbox_metadata_message(self):
        runner = CliRunner()
        with runner.isolated_filesystem():
            _write_runs([_make_entry(sandbox=None)])
            result = runner.invoke(cli, ["diff-last"])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("no sandbox path", result.output)

    def test_missing_sandbox_path_message(self):
        runner = CliRunner()
        with runner.isolated_filesystem():
            _write_runs([_make_entry(
                sandbox=_make_sandbox_meta(worktree_path="/nonexistent/path/xyz")
            )])
            result = runner.invoke(cli, ["diff-last"])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("no sandbox path", result.output)

    def test_shows_sandbox_path_and_files(self):
        runner = CliRunner()
        with runner.isolated_filesystem():
            sandbox = Path("wt")
            sandbox.mkdir()
            _write_runs([_make_entry(
                sandbox=_make_sandbox_meta(worktree_path=str(sandbox.resolve()))
            )])
            mock_result = SandboxDiffResult(
                available=True,
                sandbox_path=str(sandbox.resolve()),
                files_changed=["src/foo.py", "src/bar.py"],
                stat_text=" src/foo.py | 3 +++\n 1 file changed",
            )
            with patch("openshard.native.sandbox_diff.get_sandbox_diff", return_value=mock_result):
                result = runner.invoke(cli, ["diff-last"])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("Sandbox:", result.output)
        self.assertIn("src/foo.py", result.output)
        self.assertIn("src/bar.py", result.output)
        self.assertIn("Changed files (2)", result.output)

    def test_shows_stat_not_full_diff_by_default(self):
        runner = CliRunner()
        with runner.isolated_filesystem():
            sandbox = Path("wt")
            sandbox.mkdir()
            _write_runs([_make_entry(
                sandbox=_make_sandbox_meta(worktree_path=str(sandbox.resolve()))
            )])
            mock_result = SandboxDiffResult(
                available=True,
                sandbox_path=str(sandbox.resolve()),
                files_changed=["a.py"],
                stat_text=" a.py | 5 +++++",
                diff_text="+full diff content here",
            )
            with patch("openshard.native.sandbox_diff.get_sandbox_diff", return_value=mock_result):
                result = runner.invoke(cli, ["diff-last"])
        self.assertIn("Diff stat:", result.output)
        self.assertIn(" a.py | 5 +++++", result.output)
        self.assertNotIn("full diff content here", result.output)
        self.assertNotIn("Diff:", result.output)

    def test_full_flag_shows_full_diff(self):
        runner = CliRunner()
        with runner.isolated_filesystem():
            sandbox = Path("wt")
            sandbox.mkdir()
            _write_runs([_make_entry(
                sandbox=_make_sandbox_meta(worktree_path=str(sandbox.resolve()))
            )])
            mock_result = SandboxDiffResult(
                available=True,
                sandbox_path=str(sandbox.resolve()),
                files_changed=["a.py"],
                stat_text=" a.py | 1 +",
                diff_text="+added line",
            )
            with patch("openshard.native.sandbox_diff.get_sandbox_diff", return_value=mock_result):
                result = runner.invoke(cli, ["diff-last", "--full"])
        self.assertIn("Diff:", result.output)
        self.assertIn("+added line", result.output)

    def test_full_flag_redacts_secrets(self):
        runner = CliRunner()
        with runner.isolated_filesystem():
            sandbox = Path("wt")
            sandbox.mkdir()
            _write_runs([_make_entry(
                sandbox=_make_sandbox_meta(worktree_path=str(sandbox.resolve()))
            )])
            mock_result = SandboxDiffResult(
                available=True,
                sandbox_path=str(sandbox.resolve()),
                files_changed=["cfg.py"],
                stat_text=" cfg.py | 1 +",
                diff_text="[redacted possible secret line]",
            )
            with patch("openshard.native.sandbox_diff.get_sandbox_diff", return_value=mock_result):
                result = runner.invoke(cli, ["diff-last", "--full"])
        self.assertIn("[redacted possible secret line]", result.output)

    def test_raw_secret_not_in_output(self):
        runner = CliRunner()
        with runner.isolated_filesystem():
            sandbox = Path("wt")
            sandbox.mkdir()
            _write_runs([_make_entry(
                sandbox=_make_sandbox_meta(worktree_path=str(sandbox.resolve()))
            )])
            secret = "rawsecretvalue123"
            mock_result = SandboxDiffResult(
                available=True,
                sandbox_path=str(sandbox.resolve()),
                files_changed=["cfg.py"],
                stat_text=" cfg.py | 1 +",
                diff_text="[redacted possible secret line]",
            )
            with patch("openshard.native.sandbox_diff.get_sandbox_diff", return_value=mock_result):
                result = runner.invoke(cli, ["diff-last", "--full"])
        self.assertNotIn(secret, result.output)

    def test_unavailable_diff_shows_reason(self):
        runner = CliRunner()
        with runner.isolated_filesystem():
            sandbox = Path("wt")
            sandbox.mkdir()
            _write_runs([_make_entry(
                sandbox=_make_sandbox_meta(worktree_path=str(sandbox.resolve()))
            )])
            mock_result = SandboxDiffResult(
                available=False,
                sandbox_path=str(sandbox.resolve()),
                reason="not a git repository",
            )
            with patch("openshard.native.sandbox_diff.get_sandbox_diff", return_value=mock_result):
                result = runner.invoke(cli, ["diff-last"])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("No sandbox diff available", result.output)
        self.assertIn("not a git repository", result.output)


if __name__ == "__main__":
    unittest.main()
