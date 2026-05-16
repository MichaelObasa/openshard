from __future__ import annotations

import json
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from click.testing import CliRunner

from openshard.cli.main import cli
from openshard.native.sandbox_apply import (
    extract_candidate_sandbox_path_from_entry,
    get_candidate_records_from_entry,
)
from openshard.native.sandbox_diff import SandboxDiffResult


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


def _make_sandbox_meta(worktree_path: str | None = None) -> dict:
    return {
        "sandbox_enabled": True,
        "sandbox_type": "worktree",
        "worktree_path": worktree_path,
        "worktree_branch": "osn/run-test",
        "fallback_reason": None,
    }


def _make_candidate(
    index: int,
    sandbox_path: str = "",
    status: str = "passed",
    selected: bool = False,
    files: list[str] | None = None,
    exit_code: int | None = 0,
) -> dict:
    return {
        "candidate_index": index,
        "model": "mock-model",
        "sandbox_path": sandbox_path,
        "files_written": files or [],
        "verification_status": status,
        "exit_code": exit_code,
        "output_chars": 0,
        "selected": selected,
        "selection_reason": "first_passed" if selected else "",
        "raw_content_stored": False,
    }


def _make_candidate_summary(candidates: list[dict]) -> dict:
    selected_index = next(
        (c["candidate_index"] for c in candidates if c.get("selected")), None
    )
    return {
        "enabled": True,
        "requested_count": len(candidates),
        "completed_count": len(candidates),
        "selected_index": selected_index,
        "selection_reason": "first_passed",
        "candidates": candidates,
        "raw_content_stored": False,
    }


def _write_runs(entries: list[dict]) -> Path:
    log_dir = Path(".openshard")
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "runs.jsonl"
    with log_path.open("w", encoding="utf-8") as fh:
        for e in entries:
            fh.write(json.dumps(e) + "\n")
    return log_path


def _write_sandbox_files(sandbox: Path, rel_paths: list[str]) -> None:
    for rel in rel_paths:
        dest = sandbox / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(f"content of {rel}", encoding="utf-8")


# ---------------------------------------------------------------------------
# TestGetCandidateRecords
# ---------------------------------------------------------------------------

class TestGetCandidateRecords(unittest.TestCase):

    def test_no_candidate_summary_returns_empty(self):
        entry = _make_entry()
        self.assertEqual(get_candidate_records_from_entry(entry), [])

    def test_candidate_summary_none_returns_empty(self):
        entry = _make_entry(candidate_summary=None)
        self.assertEqual(get_candidate_records_from_entry(entry), [])

    def test_candidate_summary_empty_dict_returns_empty(self):
        entry = _make_entry(candidate_summary={})
        self.assertEqual(get_candidate_records_from_entry(entry), [])

    def test_missing_candidates_key_returns_empty(self):
        entry = _make_entry(candidate_summary={"enabled": True})
        self.assertEqual(get_candidate_records_from_entry(entry), [])

    def test_candidates_none_returns_empty(self):
        entry = _make_entry(candidate_summary={"candidates": None})
        self.assertEqual(get_candidate_records_from_entry(entry), [])

    def test_returns_candidate_list_from_dict(self):
        cs = _make_candidate_summary([
            _make_candidate(1, selected=True),
            _make_candidate(2),
        ])
        entry = _make_entry(candidate_summary=cs)
        records = get_candidate_records_from_entry(entry)
        self.assertEqual(len(records), 2)

    def test_dict_candidates_are_dicts(self):
        cs = _make_candidate_summary([_make_candidate(1)])
        entry = _make_entry(candidate_summary=cs)
        records = get_candidate_records_from_entry(entry)
        self.assertIsInstance(records[0], dict)

    def test_normalizes_simplenamespace_candidates(self):
        candidate_ns = types.SimpleNamespace(
            candidate_index=1,
            model="m",
            sandbox_path="/tmp/c1",
            files_written=["a.py"],
            verification_status="passed",
            exit_code=0,
            output_chars=0,
            selected=True,
            selection_reason="first_passed",
            raw_content_stored=False,
        )
        cs_ns = types.SimpleNamespace(
            enabled=True,
            requested_count=1,
            completed_count=1,
            selected_index=1,
            selection_reason="first_passed",
            candidates=[candidate_ns],
            raw_content_stored=False,
        )
        entry = _make_entry(candidate_summary=cs_ns)
        records = get_candidate_records_from_entry(entry)
        self.assertEqual(len(records), 1)
        self.assertIsInstance(records[0], dict)
        self.assertEqual(records[0]["candidate_index"], 1)

    def test_simplenamespace_summary_with_dict_candidates(self):
        cs_ns = types.SimpleNamespace(
            enabled=True,
            candidates=[_make_candidate(1), _make_candidate(2)],
        )
        entry = _make_entry(candidate_summary=cs_ns)
        records = get_candidate_records_from_entry(entry)
        self.assertEqual(len(records), 2)


# ---------------------------------------------------------------------------
# TestExtractCandidateSandboxPath
# ---------------------------------------------------------------------------

class TestExtractCandidateSandboxPath(unittest.TestCase):

    def test_no_candidates_returns_empty(self):
        entry = _make_entry()
        self.assertEqual(extract_candidate_sandbox_path_from_entry(entry, 1), "")

    def test_index_not_found_returns_empty(self):
        cs = _make_candidate_summary([_make_candidate(1), _make_candidate(2)])
        entry = _make_entry(candidate_summary=cs)
        self.assertEqual(extract_candidate_sandbox_path_from_entry(entry, 3), "")

    def test_missing_sandbox_path_returns_empty(self):
        cs = _make_candidate_summary([_make_candidate(1, sandbox_path="")])
        entry = _make_entry(candidate_summary=cs)
        self.assertEqual(extract_candidate_sandbox_path_from_entry(entry, 1), "")

    def test_nonexistent_sandbox_path_returns_empty(self):
        cs = _make_candidate_summary([_make_candidate(1, sandbox_path="/nonexistent/path/xyz")])
        entry = _make_entry(candidate_summary=cs)
        self.assertEqual(extract_candidate_sandbox_path_from_entry(entry, 1), "")

    def test_existing_sandbox_path_returned(self):
        runner = CliRunner()
        with runner.isolated_filesystem():
            sandbox = Path("c1").resolve()
            sandbox.mkdir()
            cs = _make_candidate_summary([_make_candidate(1, sandbox_path=str(sandbox))])
            entry = _make_entry(candidate_summary=cs)
            result = extract_candidate_sandbox_path_from_entry(entry, 1)
            self.assertEqual(result, str(sandbox))

    def test_one_based_indexing(self):
        runner = CliRunner()
        with runner.isolated_filesystem():
            c1 = Path("c1").resolve()
            c2 = Path("c2").resolve()
            c1.mkdir()
            c2.mkdir()
            cs = _make_candidate_summary([
                _make_candidate(1, sandbox_path=str(c1), selected=True),
                _make_candidate(2, sandbox_path=str(c2)),
            ])
            entry = _make_entry(candidate_summary=cs)
            self.assertEqual(extract_candidate_sandbox_path_from_entry(entry, 2), str(c2))

    def test_wrong_index_skips_candidate(self):
        runner = CliRunner()
        with runner.isolated_filesystem():
            c1 = Path("c1").resolve()
            c2 = Path("c2").resolve()
            c1.mkdir()
            c2.mkdir()
            cs = _make_candidate_summary([
                _make_candidate(1, sandbox_path="/nonexistent/bad"),
                _make_candidate(2, sandbox_path=str(c2)),
            ])
            entry = _make_entry(candidate_summary=cs)
            # Index 1 has nonexistent path, should return ""
            self.assertEqual(extract_candidate_sandbox_path_from_entry(entry, 1), "")
            # Index 2 has valid path
            self.assertEqual(extract_candidate_sandbox_path_from_entry(entry, 2), str(c2))


# ---------------------------------------------------------------------------
# TestCandidatesLastCLI
# ---------------------------------------------------------------------------

class TestCandidatesLastCLI(unittest.TestCase):

    def test_no_history(self):
        runner = CliRunner()
        with runner.isolated_filesystem():
            result = runner.invoke(cli, ["candidates-last"])
            self.assertEqual(result.exit_code, 0)
            self.assertIn("No run history found", result.output)

    def test_non_native_run(self):
        runner = CliRunner()
        with runner.isolated_filesystem():
            _write_runs([_make_entry(executor="direct")])
            result = runner.invoke(cli, ["candidates-last"])
            self.assertEqual(result.exit_code, 0)
            self.assertIn("not a native run", result.output)

    def test_no_candidate_summary(self):
        runner = CliRunner()
        with runner.isolated_filesystem():
            _write_runs([_make_entry()])
            result = runner.invoke(cli, ["candidates-last"])
            self.assertEqual(result.exit_code, 0)
            self.assertIn("No candidate summary available", result.output)

    def test_shows_table_header(self):
        runner = CliRunner()
        with runner.isolated_filesystem():
            cs = _make_candidate_summary([_make_candidate(1, selected=True)])
            _write_runs([_make_entry(candidate_summary=cs)])
            result = runner.invoke(cli, ["candidates-last"])
            self.assertEqual(result.exit_code, 0)
            self.assertIn("Candidate", result.output)
            self.assertIn("Status", result.output)
            self.assertIn("Selected", result.output)
            self.assertIn("Files", result.output)
            self.assertIn("Exit", result.output)

    def test_shows_candidate_row_indexes(self):
        runner = CliRunner()
        with runner.isolated_filesystem():
            cs = _make_candidate_summary([
                _make_candidate(1, selected=True),
                _make_candidate(2, status="failed"),
            ])
            _write_runs([_make_entry(candidate_summary=cs)])
            result = runner.invoke(cli, ["candidates-last"])
            self.assertEqual(result.exit_code, 0)
            self.assertIn("1", result.output)
            self.assertIn("2", result.output)

    def test_shows_file_count_not_names(self):
        runner = CliRunner()
        with runner.isolated_filesystem():
            cs = _make_candidate_summary([
                _make_candidate(1, selected=True, files=["a.py", "b.py"]),
            ])
            _write_runs([_make_entry(candidate_summary=cs)])
            result = runner.invoke(cli, ["candidates-last"])
            self.assertEqual(result.exit_code, 0)
            self.assertIn("2", result.output)
            self.assertNotIn("a.py", result.output)

    def test_marks_selected_yes_no(self):
        runner = CliRunner()
        with runner.isolated_filesystem():
            cs = _make_candidate_summary([
                _make_candidate(1, selected=True),
                _make_candidate(2, status="failed"),
            ])
            _write_runs([_make_entry(candidate_summary=cs)])
            result = runner.invoke(cli, ["candidates-last"])
            self.assertEqual(result.exit_code, 0)
            self.assertIn("yes", result.output)
            self.assertIn("no", result.output)

    def test_shows_verification_status(self):
        runner = CliRunner()
        with runner.isolated_filesystem():
            cs = _make_candidate_summary([
                _make_candidate(1, status="passed", selected=True),
                _make_candidate(2, status="failed"),
            ])
            _write_runs([_make_entry(candidate_summary=cs)])
            result = runner.invoke(cli, ["candidates-last"])
            self.assertEqual(result.exit_code, 0)
            self.assertIn("passed", result.output)
            self.assertIn("failed", result.output)

    def test_shows_exit_code(self):
        runner = CliRunner()
        with runner.isolated_filesystem():
            cs = _make_candidate_summary([_make_candidate(1, exit_code=1)])
            _write_runs([_make_entry(candidate_summary=cs)])
            result = runner.invoke(cli, ["candidates-last"])
            self.assertEqual(result.exit_code, 0)
            self.assertIn("1", result.output)

    def test_exit_code_none_shows_dash(self):
        runner = CliRunner()
        with runner.isolated_filesystem():
            cs = _make_candidate_summary([_make_candidate(1, exit_code=None)])
            _write_runs([_make_entry(candidate_summary=cs)])
            result = runner.invoke(cli, ["candidates-last"])
            self.assertEqual(result.exit_code, 0)
            self.assertIn("-", result.output)


# ---------------------------------------------------------------------------
# TestDiffLastCandidateCLI
# ---------------------------------------------------------------------------

class TestDiffLastCandidateCLI(unittest.TestCase):

    def _mock_diff_result(self, sandbox_path: str, files: list[str] | None = None) -> SandboxDiffResult:
        return SandboxDiffResult(
            available=True,
            sandbox_path=sandbox_path,
            files_changed=files or ["src/foo.py"],
            stat_text=" src/foo.py | 1 +\n 1 file changed",
        )

    def test_no_candidate_flag_unchanged(self):
        runner = CliRunner()
        with runner.isolated_filesystem():
            sandbox = Path("wt").resolve()
            sandbox.mkdir()
            _write_runs([_make_entry(
                sandbox=_make_sandbox_meta(worktree_path=str(sandbox))
            )])
            mock_result = self._mock_diff_result(str(sandbox))
            with patch("openshard.native.sandbox_diff.get_sandbox_diff", return_value=mock_result):
                result = runner.invoke(cli, ["diff-last"])
            self.assertEqual(result.exit_code, 0)
            self.assertIn("Sandbox:", result.output)

    def test_candidate_no_candidate_summary(self):
        runner = CliRunner()
        with runner.isolated_filesystem():
            _write_runs([_make_entry()])
            result = runner.invoke(cli, ["diff-last", "--candidate", "1"])
            self.assertEqual(result.exit_code, 0)
            self.assertIn("Candidate 1 has no sandbox path to diff", result.output)

    def test_candidate_index_not_found(self):
        runner = CliRunner()
        with runner.isolated_filesystem():
            cs = _make_candidate_summary([_make_candidate(1), _make_candidate(2)])
            _write_runs([_make_entry(candidate_summary=cs)])
            result = runner.invoke(cli, ["diff-last", "--candidate", "3"])
            self.assertEqual(result.exit_code, 0)
            self.assertIn("Candidate 3 has no sandbox path to diff", result.output)

    def test_candidate_nonexistent_path(self):
        runner = CliRunner()
        with runner.isolated_filesystem():
            cs = _make_candidate_summary([
                _make_candidate(1, sandbox_path="/nonexistent/xyz")
            ])
            _write_runs([_make_entry(candidate_summary=cs)])
            result = runner.invoke(cli, ["diff-last", "--candidate", "1"])
            self.assertEqual(result.exit_code, 0)
            self.assertIn("Candidate 1 has no sandbox path to diff", result.output)

    def test_candidate_valid_shows_diff(self):
        runner = CliRunner()
        with runner.isolated_filesystem():
            sandbox = Path("c1").resolve()
            sandbox.mkdir()
            cs = _make_candidate_summary([
                _make_candidate(1, sandbox_path=str(sandbox), selected=True),
            ])
            _write_runs([_make_entry(candidate_summary=cs)])
            mock_result = self._mock_diff_result(str(sandbox))
            with patch("openshard.native.sandbox_diff.get_sandbox_diff", return_value=mock_result):
                result = runner.invoke(cli, ["diff-last", "--candidate", "1"])
            self.assertEqual(result.exit_code, 0)
            self.assertIn("Sandbox:", result.output)

    def test_candidate_full_flag_passed(self):
        runner = CliRunner()
        with runner.isolated_filesystem():
            sandbox = Path("c1").resolve()
            sandbox.mkdir()
            cs = _make_candidate_summary([
                _make_candidate(1, sandbox_path=str(sandbox), selected=True),
            ])
            _write_runs([_make_entry(candidate_summary=cs)])
            mock_result = self._mock_diff_result(str(sandbox))
            with patch("openshard.native.sandbox_diff.get_sandbox_diff", return_value=mock_result) as mock_fn:
                runner.invoke(cli, ["diff-last", "--candidate", "1", "--full"])
            _, kwargs = mock_fn.call_args
            self.assertTrue(kwargs.get("full"))

    def test_stat_mode_does_not_call_full(self):
        runner = CliRunner()
        with runner.isolated_filesystem():
            sandbox = Path("c1").resolve()
            sandbox.mkdir()
            cs = _make_candidate_summary([
                _make_candidate(1, sandbox_path=str(sandbox), selected=True),
            ])
            _write_runs([_make_entry(candidate_summary=cs)])
            mock_result = self._mock_diff_result(str(sandbox))
            with patch("openshard.native.sandbox_diff.get_sandbox_diff", return_value=mock_result) as mock_fn:
                runner.invoke(cli, ["diff-last", "--candidate", "1"])
            _, kwargs = mock_fn.call_args
            self.assertFalse(kwargs.get("full"))

    def test_no_candidate_message_unchanged(self):
        runner = CliRunner()
        with runner.isolated_filesystem():
            _write_runs([_make_entry(sandbox=None)])
            result = runner.invoke(cli, ["diff-last"])
            self.assertEqual(result.exit_code, 0)
            self.assertIn("no sandbox path", result.output)


# ---------------------------------------------------------------------------
# TestApplyLastCandidateCLI
# ---------------------------------------------------------------------------

class TestApplyLastCandidateCLI(unittest.TestCase):

    def test_no_candidate_flag_unchanged(self):
        runner = CliRunner()
        with runner.isolated_filesystem():
            sandbox = Path("wt").resolve()
            sandbox.mkdir()
            _write_sandbox_files(sandbox, ["src/a.py"])
            _write_runs([_make_entry(
                sandbox=_make_sandbox_meta(worktree_path=str(sandbox))
            )])
            with patch("openshard.native.sandbox_apply.list_sandbox_changed_files", return_value=["src/a.py"]):
                result = runner.invoke(cli, ["apply-last"])
            self.assertEqual(result.exit_code, 0)
            self.assertIn("Applied", result.output)

    def test_candidate_no_candidate_summary(self):
        runner = CliRunner()
        with runner.isolated_filesystem():
            _write_runs([_make_entry()])
            result = runner.invoke(cli, ["apply-last", "--candidate", "1"])
            self.assertEqual(result.exit_code, 0)
            self.assertIn("Candidate 1 has no sandbox path to apply", result.output)

    def test_candidate_index_not_found(self):
        runner = CliRunner()
        with runner.isolated_filesystem():
            cs = _make_candidate_summary([_make_candidate(1), _make_candidate(2)])
            _write_runs([_make_entry(candidate_summary=cs)])
            result = runner.invoke(cli, ["apply-last", "--candidate", "3"])
            self.assertEqual(result.exit_code, 0)
            self.assertIn("Candidate 3 has no sandbox path to apply", result.output)

    def test_candidate_nonexistent_path(self):
        runner = CliRunner()
        with runner.isolated_filesystem():
            cs = _make_candidate_summary([
                _make_candidate(2, sandbox_path="/nonexistent/xyz")
            ])
            _write_runs([_make_entry(candidate_summary=cs)])
            result = runner.invoke(cli, ["apply-last", "--candidate", "2"])
            self.assertEqual(result.exit_code, 0)
            self.assertIn("Candidate 2 has no sandbox path to apply", result.output)

    def test_candidate_dry_run(self):
        runner = CliRunner()
        with runner.isolated_filesystem():
            sandbox = Path("c1").resolve()
            sandbox.mkdir()
            _write_sandbox_files(sandbox, ["src/a.py"])
            cs = _make_candidate_summary([
                _make_candidate(1, sandbox_path=str(sandbox), selected=True, files=["src/a.py"]),
            ])
            _write_runs([_make_entry(candidate_summary=cs)])
            with patch("openshard.native.sandbox_apply.list_sandbox_changed_files", return_value=["src/a.py"]):
                result = runner.invoke(cli, ["apply-last", "--candidate", "1", "--dry-run"])
            self.assertEqual(result.exit_code, 0)
            self.assertIn("Would apply", result.output)
            self.assertIn("src/a.py", result.output)
            self.assertFalse(Path("src/a.py").exists())

    def test_candidate_applies_files(self):
        runner = CliRunner()
        with runner.isolated_filesystem():
            sandbox = Path("c1").resolve()
            sandbox.mkdir()
            _write_sandbox_files(sandbox, ["src/b.py"])
            cs = _make_candidate_summary([
                _make_candidate(1, sandbox_path=str(sandbox), selected=True, files=["src/b.py"]),
            ])
            _write_runs([_make_entry(candidate_summary=cs)])
            with patch("openshard.native.sandbox_apply.list_sandbox_changed_files", return_value=["src/b.py"]):
                result = runner.invoke(cli, ["apply-last", "--candidate", "1"])
            self.assertEqual(result.exit_code, 0)
            self.assertIn("Applied", result.output)
            self.assertTrue(Path("src/b.py").exists())

    def test_candidate_2_can_be_applied_even_if_candidate_1_was_selected(self):
        runner = CliRunner()
        with runner.isolated_filesystem():
            c1 = Path("c1").resolve()
            c2 = Path("c2").resolve()
            c1.mkdir()
            c2.mkdir()
            _write_sandbox_files(c2, ["src/c.py"])
            cs = _make_candidate_summary([
                _make_candidate(1, sandbox_path=str(c1), selected=True, files=["src/a.py"]),
                _make_candidate(2, sandbox_path=str(c2), status="failed", files=["src/c.py"]),
            ])
            _write_runs([_make_entry(candidate_summary=cs)])
            with patch("openshard.native.sandbox_apply.list_sandbox_changed_files", return_value=["src/c.py"]):
                result = runner.invoke(cli, ["apply-last", "--candidate", "2"])
            self.assertEqual(result.exit_code, 0)
            self.assertIn("Applied", result.output)
            self.assertTrue(Path("src/c.py").exists())

    def test_candidate_with_file_filter(self):
        runner = CliRunner()
        with runner.isolated_filesystem():
            sandbox = Path("c1").resolve()
            sandbox.mkdir()
            _write_sandbox_files(sandbox, ["a.py", "b.py"])
            cs = _make_candidate_summary([
                _make_candidate(1, sandbox_path=str(sandbox), selected=True, files=["a.py", "b.py"]),
            ])
            _write_runs([_make_entry(candidate_summary=cs)])
            with patch("openshard.native.sandbox_apply.list_sandbox_changed_files", return_value=["a.py", "b.py"]):
                result = runner.invoke(cli, ["apply-last", "--candidate", "1", "--file", "a.py"])
            self.assertEqual(result.exit_code, 0)
            self.assertTrue(Path("a.py").exists())
            self.assertFalse(Path("b.py").exists())

    def test_candidate_with_exclude(self):
        runner = CliRunner()
        with runner.isolated_filesystem():
            sandbox = Path("c1").resolve()
            sandbox.mkdir()
            _write_sandbox_files(sandbox, ["a.py", "b.py"])
            cs = _make_candidate_summary([
                _make_candidate(1, sandbox_path=str(sandbox), selected=True, files=["a.py", "b.py"]),
            ])
            _write_runs([_make_entry(candidate_summary=cs)])
            with patch("openshard.native.sandbox_apply.list_sandbox_changed_files", return_value=["a.py", "b.py"]):
                result = runner.invoke(cli, ["apply-last", "--candidate", "1", "--exclude", "b.py"])
            self.assertEqual(result.exit_code, 0)
            self.assertTrue(Path("a.py").exists())
            self.assertFalse(Path("b.py").exists())

    def test_candidate_dry_run_with_file_filter(self):
        runner = CliRunner()
        with runner.isolated_filesystem():
            sandbox = Path("c1").resolve()
            sandbox.mkdir()
            _write_sandbox_files(sandbox, ["a.py", "b.py"])
            cs = _make_candidate_summary([
                _make_candidate(1, sandbox_path=str(sandbox), selected=True, files=["a.py", "b.py"]),
            ])
            _write_runs([_make_entry(candidate_summary=cs)])
            with patch("openshard.native.sandbox_apply.list_sandbox_changed_files", return_value=["a.py", "b.py"]):
                result = runner.invoke(cli, ["apply-last", "--candidate", "1", "--dry-run", "--file", "a.py"])
            self.assertEqual(result.exit_code, 0)
            self.assertIn("Would apply", result.output)
            self.assertIn("a.py", result.output)
            self.assertNotIn("b.py", result.output)
            self.assertFalse(Path("a.py").exists())

    def test_unsafe_file_paths_skipped(self):
        runner = CliRunner()
        with runner.isolated_filesystem():
            sandbox = Path("c1").resolve()
            sandbox.mkdir()
            cs = _make_candidate_summary([
                _make_candidate(1, sandbox_path=str(sandbox), selected=True),
            ])
            _write_runs([_make_entry(candidate_summary=cs)])
            with patch("openshard.native.sandbox_apply.list_sandbox_changed_files", return_value=["../bad.py"]):
                result = runner.invoke(cli, ["apply-last", "--candidate", "1"])
            self.assertIn("skipped", result.output.lower())

    def test_output_does_not_print_raw_file_content(self):
        runner = CliRunner()
        with runner.isolated_filesystem():
            sandbox = Path("c1").resolve()
            sandbox.mkdir()
            secret = "SUPER_SECRET_CANDIDATE_TOKEN_99"
            (sandbox / "secret.py").write_text(secret)
            cs = _make_candidate_summary([
                _make_candidate(1, sandbox_path=str(sandbox), selected=True, files=["secret.py"]),
            ])
            _write_runs([_make_entry(candidate_summary=cs)])
            with patch("openshard.native.sandbox_apply.list_sandbox_changed_files", return_value=["secret.py"]):
                result = runner.invoke(cli, ["apply-last", "--candidate", "1"])
            self.assertNotIn(secret, result.output)

    def test_no_candidate_message_unchanged(self):
        runner = CliRunner()
        with runner.isolated_filesystem():
            _write_runs([_make_entry(sandbox=None)])
            result = runner.invoke(cli, ["apply-last"])
            self.assertEqual(result.exit_code, 0)
            self.assertIn("no sandbox path", result.output)


if __name__ == "__main__":
    unittest.main()
