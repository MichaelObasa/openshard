from __future__ import annotations

import json
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from click.testing import CliRunner

from openshard.cli.main import cli
from openshard.native.sandbox_apply import (
    SandboxApplyResult,
    apply_sandbox_changes,
    extract_sandbox_path_from_entry,
    filter_sandbox_changed_files,
    list_sandbox_changed_files,
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


def _read_entries(log_path: Path) -> list[dict]:
    return [
        json.loads(ln)
        for ln in log_path.read_text(encoding="utf-8").splitlines()
        if ln.strip()
    ]


def _write_workspace_files(workspace: Path, rel_paths: list[str]) -> None:
    for rel in rel_paths:
        dest = workspace / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(f"content of {rel}", encoding="utf-8")


# ---------------------------------------------------------------------------
# TestExtractSandboxPath
# ---------------------------------------------------------------------------

class TestExtractSandboxPath(unittest.TestCase):

    def test_missing_sandbox_returns_empty(self):
        entry = _make_entry(sandbox=None)
        self.assertEqual(extract_sandbox_path_from_entry(entry), "")

    def test_non_dict_sandbox_returns_empty(self):
        entry = _make_entry(sandbox="invalid")
        self.assertEqual(extract_sandbox_path_from_entry(entry), "")

    def test_temp_sandbox_returns_empty(self):
        entry = _make_entry(sandbox=_make_sandbox_meta(sandbox_type="temp"))
        self.assertEqual(extract_sandbox_path_from_entry(entry), "")

    def test_none_type_sandbox_returns_empty(self):
        entry = _make_entry(sandbox=_make_sandbox_meta(sandbox_type="none"))
        self.assertEqual(extract_sandbox_path_from_entry(entry), "")

    def test_worktree_no_path_returns_empty(self):
        entry = _make_entry(sandbox=_make_sandbox_meta(sandbox_type="worktree", worktree_path=None))
        self.assertEqual(extract_sandbox_path_from_entry(entry), "")

    def test_worktree_nonexistent_path_returns_empty(self):
        entry = _make_entry(sandbox=_make_sandbox_meta(
            sandbox_type="worktree",
            worktree_path="/nonexistent/path/xyz123",
        ))
        self.assertEqual(extract_sandbox_path_from_entry(entry), "")

    def test_worktree_existing_path_returns_path(self):
        runner = CliRunner()
        with runner.isolated_filesystem():
            wt = Path("wt").resolve()
            wt.mkdir()
            entry = _make_entry(sandbox=_make_sandbox_meta(
                sandbox_type="worktree",
                worktree_path=str(wt),
            ))
            self.assertEqual(extract_sandbox_path_from_entry(entry), str(wt))

    def test_old_run_no_sandbox_field_returns_empty(self):
        entry = {"task": "old", "timestamp": "2025-01-01T00:00:00Z", "executor": "native"}
        self.assertEqual(extract_sandbox_path_from_entry(entry), "")


# ---------------------------------------------------------------------------
# TestListSandboxChangedFiles
# ---------------------------------------------------------------------------

class TestListSandboxChangedFiles(unittest.TestCase):

    def _make_completed_process(self, stdout: str = "", returncode: int = 0):
        m = MagicMock()
        m.stdout = stdout
        m.returncode = returncode
        return m

    def test_git_returns_modified_files(self):
        runner = CliRunner()
        with runner.isolated_filesystem():
            sandbox = Path("sandbox")
            sandbox.mkdir()
            diff_proc = self._make_completed_process("src/foo.py\n")
            ls_proc = self._make_completed_process("")
            with patch("subprocess.run", side_effect=[diff_proc, ls_proc]):
                result = list_sandbox_changed_files(Path.cwd(), sandbox)
            self.assertEqual(result, ["src/foo.py"])

    def test_git_worktree_includes_untracked_files(self):
        runner = CliRunner()
        with runner.isolated_filesystem():
            sandbox = Path("sandbox")
            sandbox.mkdir()
            diff_proc = self._make_completed_process("")
            ls_proc = self._make_completed_process("src/new_file.py\n")
            with patch("subprocess.run", side_effect=[diff_proc, ls_proc]):
                result = list_sandbox_changed_files(Path.cwd(), sandbox)
            self.assertEqual(result, ["src/new_file.py"])

    def test_git_worktree_combines_modified_and_untracked(self):
        runner = CliRunner()
        with runner.isolated_filesystem():
            sandbox = Path("sandbox")
            sandbox.mkdir()
            diff_proc = self._make_completed_process("src/existing.py\n")
            ls_proc = self._make_completed_process("src/new.py\n")
            with patch("subprocess.run", side_effect=[diff_proc, ls_proc]):
                result = list_sandbox_changed_files(Path.cwd(), sandbox)
            self.assertIn("src/existing.py", result)
            self.assertIn("src/new.py", result)
            self.assertEqual(len(result), 2)

    def test_no_duplicates_in_union(self):
        runner = CliRunner()
        with runner.isolated_filesystem():
            sandbox = Path("sandbox")
            sandbox.mkdir()
            diff_proc = self._make_completed_process("src/foo.py\n")
            ls_proc = self._make_completed_process("src/foo.py\n")
            with patch("subprocess.run", side_effect=[diff_proc, ls_proc]):
                result = list_sandbox_changed_files(Path.cwd(), sandbox)
            self.assertEqual(result.count("src/foo.py"), 1)

    def test_fallback_walk_when_git_fails(self):
        runner = CliRunner()
        with runner.isolated_filesystem():
            sandbox = Path("sandbox")
            sandbox.mkdir()
            (sandbox / "src").mkdir()
            (sandbox / "src" / "foo.py").write_text("x")
            fail_proc = self._make_completed_process(returncode=1)
            with patch("subprocess.run", side_effect=[fail_proc, fail_proc]):
                result = list_sandbox_changed_files(Path.cwd(), sandbox)
            self.assertIn("src/foo.py", result)

    def test_fallback_walk_when_git_raises(self):
        runner = CliRunner()
        with runner.isolated_filesystem():
            sandbox = Path("sandbox")
            sandbox.mkdir()
            (sandbox / "file.py").write_text("x")
            with patch("subprocess.run", side_effect=FileNotFoundError):
                result = list_sandbox_changed_files(Path.cwd(), sandbox)
            self.assertIn("file.py", result)

    def test_ignores_dot_git(self):
        runner = CliRunner()
        with runner.isolated_filesystem():
            sandbox = Path("sandbox")
            sandbox.mkdir()
            git_dir = sandbox / ".git"
            git_dir.mkdir()
            (git_dir / "config").write_text("x")
            (sandbox / "real.py").write_text("x")
            fail_proc = self._make_completed_process(returncode=1)
            with patch("subprocess.run", side_effect=[fail_proc, fail_proc]):
                result = list_sandbox_changed_files(Path.cwd(), sandbox)
            self.assertNotIn(".git/config", result)
            self.assertIn("real.py", result)

    def test_ignores_dot_openshard(self):
        runner = CliRunner()
        with runner.isolated_filesystem():
            sandbox = Path("sandbox")
            sandbox.mkdir()
            osdir = sandbox / ".openshard"
            osdir.mkdir()
            (osdir / "runs.jsonl").write_text("{}")
            (sandbox / "ok.py").write_text("x")
            fail_proc = self._make_completed_process(returncode=1)
            with patch("subprocess.run", side_effect=[fail_proc, fail_proc]):
                result = list_sandbox_changed_files(Path.cwd(), sandbox)
            self.assertFalse(any(".openshard" in f for f in result))
            self.assertIn("ok.py", result)

    def test_ignores_pycache(self):
        runner = CliRunner()
        with runner.isolated_filesystem():
            sandbox = Path("sandbox")
            sandbox.mkdir()
            pc = sandbox / "__pycache__"
            pc.mkdir()
            (pc / "foo.pyc").write_bytes(b"x")
            (sandbox / "real.py").write_text("x")
            fail_proc = self._make_completed_process(returncode=1)
            with patch("subprocess.run", side_effect=[fail_proc, fail_proc]):
                result = list_sandbox_changed_files(Path.cwd(), sandbox)
            self.assertFalse(any("__pycache__" in f for f in result))
            self.assertIn("real.py", result)


# ---------------------------------------------------------------------------
# TestApplySandboxChanges
# ---------------------------------------------------------------------------

class TestApplySandboxChanges(unittest.TestCase):

    def test_applies_file(self):
        runner = CliRunner()
        with runner.isolated_filesystem():
            repo = Path("repo")
            repo.mkdir()
            sandbox = Path("sandbox")
            sandbox.mkdir()
            _write_workspace_files(sandbox, ["src/foo.py"])
            with patch("openshard.native.sandbox_apply.list_sandbox_changed_files", return_value=["src/foo.py"]):
                result = apply_sandbox_changes(repo, sandbox)
            self.assertTrue(result.applied)
            self.assertIn("src/foo.py", result.files_applied)
            self.assertTrue((repo / "src" / "foo.py").exists())

    def test_creates_parent_directories(self):
        runner = CliRunner()
        with runner.isolated_filesystem():
            repo = Path("repo")
            repo.mkdir()
            sandbox = Path("sandbox")
            sandbox.mkdir()
            _write_workspace_files(sandbox, ["deep/nested/file.py"])
            with patch("openshard.native.sandbox_apply.list_sandbox_changed_files", return_value=["deep/nested/file.py"]):
                apply_sandbox_changes(repo, sandbox)
            self.assertTrue((repo / "deep" / "nested" / "file.py").exists())

    def test_skips_missing_source(self):
        runner = CliRunner()
        with runner.isolated_filesystem():
            repo = Path("repo")
            repo.mkdir()
            sandbox = Path("sandbox")
            sandbox.mkdir()
            with patch("openshard.native.sandbox_apply.list_sandbox_changed_files", return_value=["missing.py"]):
                result = apply_sandbox_changes(repo, sandbox)
            self.assertFalse(result.applied)
            self.assertTrue(any("missing.py" in s for s in result.files_skipped))

    def test_rejects_traversal_path(self):
        runner = CliRunner()
        with runner.isolated_filesystem():
            repo = Path("repo")
            repo.mkdir()
            sandbox = Path("sandbox")
            sandbox.mkdir()
            with patch("openshard.native.sandbox_apply.list_sandbox_changed_files", return_value=["../escape.py"]):
                result = apply_sandbox_changes(repo, sandbox)
            self.assertTrue(any("escape.py" in s for s in result.files_skipped))
            self.assertFalse((repo.parent / "escape.py").exists())

    def test_rejects_openshard_protected_file(self):
        runner = CliRunner()
        with runner.isolated_filesystem():
            repo = Path("repo")
            repo.mkdir()
            sandbox = Path("sandbox")
            sandbox.mkdir()
            _write_workspace_files(sandbox, [".openshard/failure_memory.jsonl"])
            with patch("openshard.native.sandbox_apply.list_sandbox_changed_files", return_value=[".openshard/failure_memory.jsonl"]):
                result = apply_sandbox_changes(repo, sandbox)
            self.assertTrue(any("failure_memory" in s for s in result.files_skipped))

    def test_no_deletions(self):
        runner = CliRunner()
        with runner.isolated_filesystem():
            repo = Path("repo")
            repo.mkdir()
            target = repo / "existing.py"
            target.write_text("original")
            sandbox = Path("sandbox")
            sandbox.mkdir()
            # file listed by git diff (delete) but not in sandbox
            with patch("openshard.native.sandbox_apply.list_sandbox_changed_files", return_value=["existing.py"]):
                apply_sandbox_changes(repo, sandbox)
            # file should still exist (skipped, not deleted)
            self.assertTrue(target.exists())

    def test_raw_content_stored_always_false(self):
        runner = CliRunner()
        with runner.isolated_filesystem():
            repo = Path("repo")
            repo.mkdir()
            sandbox = Path("sandbox")
            sandbox.mkdir()
            _write_workspace_files(sandbox, ["a.py"])
            with patch("openshard.native.sandbox_apply.list_sandbox_changed_files", return_value=["a.py"]):
                result = apply_sandbox_changes(repo, sandbox)
            self.assertFalse(result.raw_content_stored)

    def test_binary_content_preserved(self):
        runner = CliRunner()
        with runner.isolated_filesystem():
            repo = Path("repo")
            repo.mkdir()
            sandbox = Path("sandbox")
            sandbox.mkdir()
            binary = b"\x00\x01\x02\xff\xfe"
            (sandbox / "data.bin").write_bytes(binary)
            with patch("openshard.native.sandbox_apply.list_sandbox_changed_files", return_value=["data.bin"]):
                apply_sandbox_changes(repo, sandbox)
            self.assertEqual((repo / "data.bin").read_bytes(), binary)

    def test_empty_file_list_sets_reason(self):
        runner = CliRunner()
        with runner.isolated_filesystem():
            repo = Path("repo")
            repo.mkdir()
            sandbox = Path("sandbox")
            sandbox.mkdir()
            with patch("openshard.native.sandbox_apply.list_sandbox_changed_files", return_value=[]):
                result = apply_sandbox_changes(repo, sandbox)
            self.assertFalse(result.applied)
            self.assertTrue(result.reason)

    def test_mixed_safe_and_unsafe_paths(self):
        runner = CliRunner()
        with runner.isolated_filesystem():
            repo = Path("repo")
            repo.mkdir()
            sandbox = Path("sandbox")
            sandbox.mkdir()
            _write_workspace_files(sandbox, ["safe.py"])
            with patch("openshard.native.sandbox_apply.list_sandbox_changed_files", return_value=["safe.py", "../bad.py"]):
                result = apply_sandbox_changes(repo, sandbox)
            self.assertIn("safe.py", result.files_applied)
            self.assertTrue(any("bad.py" in s for s in result.files_skipped))

    def test_result_dataclass_defaults(self):
        r = SandboxApplyResult()
        self.assertFalse(r.applied)
        self.assertEqual(r.sandbox_path, "")
        self.assertEqual(r.files_applied, [])
        self.assertEqual(r.files_skipped, [])
        self.assertEqual(r.reason, "")
        self.assertFalse(r.raw_content_stored)


# ---------------------------------------------------------------------------
# TestApplyLastCLI
# ---------------------------------------------------------------------------

class TestApplyLastCLI(unittest.TestCase):

    def test_no_history(self):
        runner = CliRunner()
        with runner.isolated_filesystem():
            result = runner.invoke(cli, ["apply-last"])
            self.assertEqual(result.exit_code, 0)
            self.assertIn("No run history", result.output)

    def test_non_native_run(self):
        runner = CliRunner()
        with runner.isolated_filesystem():
            _write_runs([_make_entry(executor="direct")])
            result = runner.invoke(cli, ["apply-last"])
            self.assertEqual(result.exit_code, 0)
            self.assertIn("not a native run", result.output)

    def test_no_sandbox_path(self):
        runner = CliRunner()
        with runner.isolated_filesystem():
            _write_runs([_make_entry(sandbox=None)])
            result = runner.invoke(cli, ["apply-last"])
            self.assertEqual(result.exit_code, 0)
            self.assertIn("no sandbox path", result.output)

    def test_no_sandbox_path_temp_type(self):
        runner = CliRunner()
        with runner.isolated_filesystem():
            _write_runs([_make_entry(sandbox=_make_sandbox_meta(sandbox_type="temp"))])
            result = runner.invoke(cli, ["apply-last"])
            self.assertEqual(result.exit_code, 0)
            self.assertIn("no sandbox path", result.output)

    def test_dry_run_shows_files_no_copy(self):
        runner = CliRunner()
        with runner.isolated_filesystem():
            sandbox = Path("wt")
            sandbox.mkdir()
            _write_workspace_files(sandbox, ["src/foo.py"])
            _write_runs([_make_entry(
                workspace_path=str(sandbox),
                sandbox=_make_sandbox_meta(sandbox_type="worktree", worktree_path=str(sandbox.resolve())),
            )])
            with patch("openshard.native.sandbox_apply.list_sandbox_changed_files", return_value=["src/foo.py"]):
                result = runner.invoke(cli, ["apply-last", "--dry-run"])
            self.assertEqual(result.exit_code, 0)
            self.assertIn("src/foo.py", result.output)
            self.assertFalse(Path("src/foo.py").exists())

    def test_normal_apply_copies_files(self):
        runner = CliRunner()
        with runner.isolated_filesystem():
            sandbox = Path("wt")
            sandbox.mkdir()
            _write_workspace_files(sandbox, ["src/bar.py"])
            _write_runs([_make_entry(
                workspace_path=str(sandbox),
                sandbox=_make_sandbox_meta(sandbox_type="worktree", worktree_path=str(sandbox.resolve())),
            )])
            with patch("openshard.native.sandbox_apply.list_sandbox_changed_files", return_value=["src/bar.py"]):
                result = runner.invoke(cli, ["apply-last"])
            self.assertEqual(result.exit_code, 0)
            self.assertIn("Applied", result.output)
            self.assertTrue(Path("src/bar.py").exists())

    def test_output_shows_file_list(self):
        runner = CliRunner()
        with runner.isolated_filesystem():
            sandbox = Path("wt")
            sandbox.mkdir()
            _write_workspace_files(sandbox, ["a.py", "b.py"])
            _write_runs([_make_entry(
                workspace_path=str(sandbox),
                sandbox=_make_sandbox_meta(sandbox_type="worktree", worktree_path=str(sandbox.resolve())),
            )])
            with patch("openshard.native.sandbox_apply.list_sandbox_changed_files", return_value=["a.py", "b.py"]):
                result = runner.invoke(cli, ["apply-last"])
            self.assertIn("a.py", result.output)
            self.assertIn("b.py", result.output)

    def test_skipped_file_reported(self):
        runner = CliRunner()
        with runner.isolated_filesystem():
            sandbox = Path("wt")
            sandbox.mkdir()
            _write_runs([_make_entry(
                workspace_path=str(sandbox),
                sandbox=_make_sandbox_meta(sandbox_type="worktree", worktree_path=str(sandbox.resolve())),
            )])
            with patch("openshard.native.sandbox_apply.list_sandbox_changed_files", return_value=["../bad.py"]):
                result = runner.invoke(cli, ["apply-last"])
            self.assertIn("skipped", result.output.lower())

    def test_output_has_no_raw_file_content(self):
        runner = CliRunner()
        with runner.isolated_filesystem():
            sandbox = Path("wt")
            sandbox.mkdir()
            secret = "SUPER_SECRET_TOKEN_12345"
            (sandbox / "secret.py").write_text(secret)
            _write_runs([_make_entry(
                workspace_path=str(sandbox),
                sandbox=_make_sandbox_meta(sandbox_type="worktree", worktree_path=str(sandbox.resolve())),
            )])
            with patch("openshard.native.sandbox_apply.list_sandbox_changed_files", return_value=["secret.py"]):
                result = runner.invoke(cli, ["apply-last"])
            self.assertNotIn(secret, result.output)

    def test_preserves_all_run_entries(self):
        runner = CliRunner()
        with runner.isolated_filesystem():
            sandbox = Path("wt")
            sandbox.mkdir()
            _write_workspace_files(sandbox, ["x.py"])
            log_path = _write_runs([
                _make_entry(task="first"),
                _make_entry(task="second"),
                _make_entry(
                    task="third",
                    workspace_path=str(sandbox),
                    sandbox=_make_sandbox_meta(sandbox_type="worktree", worktree_path=str(sandbox.resolve())),
                ),
            ])
            with patch("openshard.native.sandbox_apply.list_sandbox_changed_files", return_value=["x.py"]):
                runner.invoke(cli, ["apply-last"])
            entries = _read_entries(log_path)
            self.assertEqual(len(entries), 3)


# ---------------------------------------------------------------------------
# TestFilterSandboxChangedFiles
# ---------------------------------------------------------------------------

class TestFilterSandboxChangedFiles(unittest.TestCase):

    def test_no_include_or_exclude_returns_deduped_original_order(self):
        files = ["b.py", "a.py", "b.py", "c.py"]
        self.assertEqual(filter_sandbox_changed_files(files), ["b.py", "a.py", "c.py"])

    def test_include_only_selected_file(self):
        result = filter_sandbox_changed_files(["a.py", "b.py", "c.py"], include=["a.py"])
        self.assertEqual(result, ["a.py"])

    def test_include_multiple_files_preserves_order(self):
        result = filter_sandbox_changed_files(["a.py", "b.py", "c.py"], include=["c.py", "a.py"])
        self.assertEqual(result, ["a.py", "c.py"])

    def test_exclude_removes_file(self):
        result = filter_sandbox_changed_files(["a.py", "b.py", "c.py"], exclude=["b.py"])
        self.assertEqual(result, ["a.py", "c.py"])

    def test_include_then_exclude(self):
        result = filter_sandbox_changed_files(
            ["a.py", "b.py", "c.py"],
            include=["a.py", "b.py"],
            exclude=["b.py"],
        )
        self.assertEqual(result, ["a.py"])

    def test_unknown_include_returns_empty(self):
        result = filter_sandbox_changed_files(["a.py", "b.py"], include=["missing.py"])
        self.assertEqual(result, [])

    def test_exclude_nonexistent_is_noop(self):
        result = filter_sandbox_changed_files(["a.py", "b.py"], exclude=["missing.py"])
        self.assertEqual(result, ["a.py", "b.py"])

    def test_exclude_all_leaves_empty(self):
        result = filter_sandbox_changed_files(["a.py", "b.py"], exclude=["a.py", "b.py"])
        self.assertEqual(result, [])

    def test_empty_files_returns_empty(self):
        self.assertEqual(filter_sandbox_changed_files([], include=["a.py"]), [])

    def test_normalizes_backslashes(self):
        result = filter_sandbox_changed_files(
            ["src\\foo.py"],
            include=["src/foo.py"],
        )
        self.assertEqual(result, ["src/foo.py"])


# ---------------------------------------------------------------------------
# TestApplySandboxChangesSelective
# ---------------------------------------------------------------------------

class TestApplySandboxChangesSelective(unittest.TestCase):

    def test_apply_include_only_one_file(self):
        runner = CliRunner()
        with runner.isolated_filesystem():
            repo = Path("repo")
            repo.mkdir()
            sandbox = Path("sandbox")
            sandbox.mkdir()
            _write_workspace_files(sandbox, ["a.py", "b.py"])
            with patch("openshard.native.sandbox_apply.list_sandbox_changed_files", return_value=["a.py", "b.py"]):
                result = apply_sandbox_changes(repo, sandbox, include=["a.py"])
            self.assertIn("a.py", result.files_applied)
            self.assertNotIn("b.py", result.files_applied)
            self.assertTrue((repo / "a.py").exists())
            self.assertFalse((repo / "b.py").exists())

    def test_apply_exclude_skips_file(self):
        runner = CliRunner()
        with runner.isolated_filesystem():
            repo = Path("repo")
            repo.mkdir()
            sandbox = Path("sandbox")
            sandbox.mkdir()
            _write_workspace_files(sandbox, ["a.py", "b.py"])
            with patch("openshard.native.sandbox_apply.list_sandbox_changed_files", return_value=["a.py", "b.py"]):
                result = apply_sandbox_changes(repo, sandbox, exclude=["b.py"])
            self.assertIn("a.py", result.files_applied)
            self.assertNotIn("b.py", result.files_applied)

    def test_apply_include_unknown_returns_reason(self):
        runner = CliRunner()
        with runner.isolated_filesystem():
            repo = Path("repo")
            repo.mkdir()
            sandbox = Path("sandbox")
            sandbox.mkdir()
            with patch("openshard.native.sandbox_apply.list_sandbox_changed_files", return_value=["a.py"]):
                result = apply_sandbox_changes(repo, sandbox, include=["missing.py"])
            self.assertFalse(result.applied)
            self.assertIn("selection", result.reason)

    def test_apply_include_and_exclude_results_empty(self):
        runner = CliRunner()
        with runner.isolated_filesystem():
            repo = Path("repo")
            repo.mkdir()
            sandbox = Path("sandbox")
            sandbox.mkdir()
            _write_workspace_files(sandbox, ["a.py"])
            with patch("openshard.native.sandbox_apply.list_sandbox_changed_files", return_value=["a.py"]):
                result = apply_sandbox_changes(repo, sandbox, include=["a.py"], exclude=["a.py"])
            self.assertFalse(result.applied)
            self.assertTrue(result.reason)

    def test_apply_selection_still_rejects_traversal(self):
        runner = CliRunner()
        with runner.isolated_filesystem():
            repo = Path("repo")
            repo.mkdir()
            sandbox = Path("sandbox")
            sandbox.mkdir()
            with patch("openshard.native.sandbox_apply.list_sandbox_changed_files", return_value=["../escape.py"]):
                result = apply_sandbox_changes(repo, sandbox, include=["../escape.py"])
            self.assertFalse(result.applied)
            self.assertTrue(any("escape.py" in s for s in result.files_skipped))

    def test_apply_selection_still_rejects_openshard_file(self):
        runner = CliRunner()
        with runner.isolated_filesystem():
            repo = Path("repo")
            repo.mkdir()
            sandbox = Path("sandbox")
            sandbox.mkdir()
            _write_workspace_files(sandbox, [".openshard/runs.jsonl"])
            with patch("openshard.native.sandbox_apply.list_sandbox_changed_files", return_value=[".openshard/runs.jsonl"]):
                result = apply_sandbox_changes(repo, sandbox, include=[".openshard/runs.jsonl"])
            self.assertFalse(result.applied)

    def test_binary_content_preserved_with_include(self):
        runner = CliRunner()
        with runner.isolated_filesystem():
            repo = Path("repo")
            repo.mkdir()
            sandbox = Path("sandbox")
            sandbox.mkdir()
            binary = b"\x00\x01\x02\xff\xfe"
            (sandbox / "data.bin").write_bytes(binary)
            with patch("openshard.native.sandbox_apply.list_sandbox_changed_files", return_value=["data.bin", "other.py"]):
                apply_sandbox_changes(repo, sandbox, include=["data.bin"])
            self.assertEqual((repo / "data.bin").read_bytes(), binary)
            self.assertFalse((repo / "other.py").exists())


# ---------------------------------------------------------------------------
# TestApplyLastCLISelective
# ---------------------------------------------------------------------------

class TestApplyLastCLISelective(unittest.TestCase):

    def _make_sandbox_entry(self, sandbox: Path) -> dict:
        return _make_entry(
            workspace_path=str(sandbox),
            sandbox=_make_sandbox_meta(sandbox_type="worktree", worktree_path=str(sandbox.resolve())),
        )

    def test_dry_run_file_only_shows_selected_file(self):
        runner = CliRunner()
        with runner.isolated_filesystem():
            sandbox = Path("wt")
            sandbox.mkdir()
            _write_sandbox_files(sandbox, ["a.py", "b.py"])
            _write_runs([self._make_sandbox_entry(sandbox)])
            with patch("openshard.native.sandbox_apply.list_sandbox_changed_files", return_value=["a.py", "b.py"]):
                result = runner.invoke(cli, ["apply-last", "--dry-run", "--file", "a.py"])
            self.assertEqual(result.exit_code, 0)
            self.assertIn("a.py", result.output)
            self.assertNotIn("b.py", result.output)

    def test_dry_run_exclude_hides_file(self):
        runner = CliRunner()
        with runner.isolated_filesystem():
            sandbox = Path("wt")
            sandbox.mkdir()
            _write_sandbox_files(sandbox, ["a.py", "b.py"])
            _write_runs([self._make_sandbox_entry(sandbox)])
            with patch("openshard.native.sandbox_apply.list_sandbox_changed_files", return_value=["a.py", "b.py"]):
                result = runner.invoke(cli, ["apply-last", "--dry-run", "--exclude", "b.py"])
            self.assertEqual(result.exit_code, 0)
            self.assertIn("a.py", result.output)
            self.assertNotIn("b.py", result.output)

    def test_apply_file_only_copies_selected_file(self):
        runner = CliRunner()
        with runner.isolated_filesystem():
            sandbox = Path("wt")
            sandbox.mkdir()
            _write_sandbox_files(sandbox, ["a.py", "b.py"])
            _write_runs([self._make_sandbox_entry(sandbox)])
            with patch("openshard.native.sandbox_apply.list_sandbox_changed_files", return_value=["a.py", "b.py"]):
                result = runner.invoke(cli, ["apply-last", "--file", "a.py"])
            self.assertEqual(result.exit_code, 0)
            self.assertTrue(Path("a.py").exists())
            self.assertFalse(Path("b.py").exists())

    def test_apply_exclude_does_not_copy_excluded_file(self):
        runner = CliRunner()
        with runner.isolated_filesystem():
            sandbox = Path("wt")
            sandbox.mkdir()
            _write_sandbox_files(sandbox, ["a.py", "b.py"])
            _write_runs([self._make_sandbox_entry(sandbox)])
            with patch("openshard.native.sandbox_apply.list_sandbox_changed_files", return_value=["a.py", "b.py"]):
                result = runner.invoke(cli, ["apply-last", "--exclude", "b.py"])
            self.assertEqual(result.exit_code, 0)
            self.assertTrue(Path("a.py").exists())
            self.assertFalse(Path("b.py").exists())

    def test_unknown_file_selection_prints_no_match(self):
        runner = CliRunner()
        with runner.isolated_filesystem():
            sandbox = Path("wt")
            sandbox.mkdir()
            _write_sandbox_files(sandbox, ["a.py"])
            _write_runs([self._make_sandbox_entry(sandbox)])
            with patch("openshard.native.sandbox_apply.list_sandbox_changed_files", return_value=["a.py"]):
                result = runner.invoke(cli, ["apply-last", "--file", "missing.py"])
            self.assertEqual(result.exit_code, 0)
            self.assertIn("No sandbox changes matched", result.output)

    def test_output_has_no_raw_content_with_file_selection(self):
        runner = CliRunner()
        with runner.isolated_filesystem():
            sandbox = Path("wt")
            sandbox.mkdir()
            secret = "SUPER_SECRET_TOKEN_SELECTIVE_99"
            (sandbox / "secret.py").write_text(secret)
            _write_runs([self._make_sandbox_entry(sandbox)])
            with patch("openshard.native.sandbox_apply.list_sandbox_changed_files", return_value=["secret.py"]):
                result = runner.invoke(cli, ["apply-last", "--file", "secret.py"])
            self.assertNotIn(secret, result.output)


def _write_sandbox_files(sandbox: Path, rel_paths: list[str]) -> None:
    for rel in rel_paths:
        dest = sandbox / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(f"content of {rel}", encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
