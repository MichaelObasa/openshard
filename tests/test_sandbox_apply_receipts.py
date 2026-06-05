from __future__ import annotations

import json
import unittest
from pathlib import Path
from unittest.mock import patch

from click.testing import CliRunner

from openshard.cli.main import cli
from openshard.history.sandbox_apply_receipts import (
    SandboxApplyReceipt,
    _dict_to_receipt,
    _receipt_to_dict,
    load_sandbox_apply_receipts,
    log_sandbox_apply_receipt,
    recent_sandbox_apply_receipts,
)
from openshard.security.paths import UnsafePathError, resolve_safe_repo_path

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


def _write_runs(entries: list[dict]) -> None:
    log_dir = Path(".openshard")
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "runs.jsonl"
    with log_path.open("w", encoding="utf-8") as fh:
        for entry in entries:
            fh.write(json.dumps(entry) + "\n")


# ---------------------------------------------------------------------------
# TestDataclassDefaults
# ---------------------------------------------------------------------------

class TestDataclassDefaults(unittest.TestCase):

    def test_receipt_id_non_empty(self):
        r = SandboxApplyReceipt()
        self.assertTrue(r.receipt_id)

    def test_receipt_id_is_uuid_format(self):
        r = SandboxApplyReceipt()
        self.assertRegex(
            r.receipt_id,
            r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
        )

    def test_timestamp_format(self):
        r = SandboxApplyReceipt()
        self.assertRegex(r.timestamp, r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")

    def test_raw_content_stored_always_false(self):
        r = SandboxApplyReceipt(raw_content_stored=True)
        self.assertFalse(r.raw_content_stored)

    def test_applied_count_derived_from_files_applied(self):
        r = SandboxApplyReceipt(files_applied=["a.py", "b.py"])
        self.assertEqual(r.applied_count, 2)

    def test_skipped_count_derived_from_files_skipped(self):
        r = SandboxApplyReceipt(files_skipped=["x.py"])
        self.assertEqual(r.skipped_count, 1)

    def test_zero_counts_when_lists_empty(self):
        r = SandboxApplyReceipt()
        self.assertEqual(r.applied_count, 0)
        self.assertEqual(r.skipped_count, 0)

    def test_receipt_ids_unique(self):
        a = SandboxApplyReceipt()
        b = SandboxApplyReceipt()
        self.assertNotEqual(a.receipt_id, b.receipt_id)


# ---------------------------------------------------------------------------
# TestSerialization
# ---------------------------------------------------------------------------

class TestSerialization(unittest.TestCase):

    def test_receipt_to_dict_has_all_keys(self):
        r = SandboxApplyReceipt(source_run_id="run-1", sandbox_path="/tmp/wt")
        d = _receipt_to_dict(r)
        for key in (
            "schema_version", "receipt_id", "timestamp", "source_run_id",
            "sandbox_path", "applied", "files_applied", "files_skipped",
            "applied_count", "skipped_count", "dry_run", "reason",
            "raw_content_stored",
        ):
            self.assertIn(key, d)

    def test_round_trip_preserves_fields(self):
        r = SandboxApplyReceipt(
            source_run_id="run-abc",
            sandbox_path="/tmp/sandbox",
            applied=True,
            files_applied=["foo.py"],
            files_skipped=["bar.py"],
            dry_run=False,
            reason="ok",
        )
        r2 = _dict_to_receipt(_receipt_to_dict(r))
        self.assertEqual(r2.source_run_id, r.source_run_id)
        self.assertEqual(r2.sandbox_path, r.sandbox_path)
        self.assertEqual(r2.applied, r.applied)
        self.assertEqual(r2.files_applied, r.files_applied)
        self.assertEqual(r2.files_skipped, r.files_skipped)
        self.assertEqual(r2.dry_run, r.dry_run)
        self.assertEqual(r2.reason, r.reason)

    def test_raw_content_stored_forced_false_on_load(self):
        d = _receipt_to_dict(SandboxApplyReceipt())
        d["raw_content_stored"] = True
        r = _dict_to_receipt(d)
        self.assertFalse(r.raw_content_stored)

    def test_dict_is_json_serializable(self):
        d = _receipt_to_dict(SandboxApplyReceipt(files_applied=["a.py"]))
        json.dumps(d)  # must not raise

    def test_dict_raw_content_stored_always_false(self):
        d = _receipt_to_dict(SandboxApplyReceipt())
        self.assertFalse(d["raw_content_stored"])


# ---------------------------------------------------------------------------
# TestLoggingAndLoading
# ---------------------------------------------------------------------------

class TestLoggingAndLoading(unittest.TestCase):

    def test_missing_file_returns_empty(self):
        runner = CliRunner()
        with runner.isolated_filesystem():
            self.assertEqual(load_sandbox_apply_receipts(), [])

    def test_three_writes_produce_three_lines(self):
        runner = CliRunner()
        with runner.isolated_filesystem():
            for _ in range(3):
                log_sandbox_apply_receipt(SandboxApplyReceipt())
            receipts_path = Path(".openshard") / "sandbox_apply_receipts.jsonl"
            lines = [ln for ln in receipts_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
            self.assertEqual(len(lines), 3)

    def test_load_returns_all_receipts(self):
        runner = CliRunner()
        with runner.isolated_filesystem():
            for _ in range(3):
                log_sandbox_apply_receipt(SandboxApplyReceipt())
            receipts = load_sandbox_apply_receipts()
            self.assertEqual(len(receipts), 3)

    def test_malformed_lines_skipped(self):
        runner = CliRunner()
        with runner.isolated_filesystem():
            receipts_dir = Path(".openshard")
            receipts_dir.mkdir(parents=True, exist_ok=True)
            receipts_path = receipts_dir / "sandbox_apply_receipts.jsonl"
            log_sandbox_apply_receipt(SandboxApplyReceipt(source_run_id="good"))
            with receipts_path.open("a", encoding="utf-8") as fh:
                fh.write("not-json\n")
                fh.write("\n")
            receipts = load_sandbox_apply_receipts()
            self.assertEqual(len(receipts), 1)
            self.assertEqual(receipts[0].source_run_id, "good")

    def test_recent_returns_last_n(self):
        runner = CliRunner()
        with runner.isolated_filesystem():
            for i in range(5):
                log_sandbox_apply_receipt(SandboxApplyReceipt(source_run_id=f"run-{i}"))
            recent = recent_sandbox_apply_receipts(limit=2)
            self.assertEqual(len(recent), 2)
            self.assertEqual(recent[-1].source_run_id, "run-4")

    def test_directory_auto_created(self):
        runner = CliRunner()
        with runner.isolated_filesystem():
            self.assertFalse(Path(".openshard").exists())
            log_sandbox_apply_receipt(SandboxApplyReceipt())
            self.assertTrue(Path(".openshard").exists())
            self.assertTrue((Path(".openshard") / "sandbox_apply_receipts.jsonl").exists())

    def test_recent_all_when_fewer_than_limit(self):
        runner = CliRunner()
        with runner.isolated_filesystem():
            log_sandbox_apply_receipt(SandboxApplyReceipt())
            log_sandbox_apply_receipt(SandboxApplyReceipt())
            recent = recent_sandbox_apply_receipts(limit=10)
            self.assertEqual(len(recent), 2)


# ---------------------------------------------------------------------------
# TestPathSafety
# ---------------------------------------------------------------------------

class TestPathSafety(unittest.TestCase):

    def test_sandbox_apply_receipts_jsonl_rejected(self, tmp_path=None):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            with self.assertRaises(UnsafePathError):
                resolve_safe_repo_path(root, ".openshard/sandbox_apply_receipts.jsonl")

    def test_runs_jsonl_still_rejected(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            with self.assertRaises(UnsafePathError):
                resolve_safe_repo_path(root, ".openshard/runs.jsonl")

    def test_failure_memory_jsonl_still_rejected(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            with self.assertRaises(UnsafePathError):
                resolve_safe_repo_path(root, ".openshard/failure_memory.jsonl")

    def test_error_message_names_the_path(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            with self.assertRaises(UnsafePathError) as cm:
                resolve_safe_repo_path(root, ".openshard/sandbox_apply_receipts.jsonl")
            self.assertIn("sandbox_apply_receipts.jsonl", str(cm.exception))


# ---------------------------------------------------------------------------
# TestApplyLastIntegration
# ---------------------------------------------------------------------------

class TestApplyLastIntegration(unittest.TestCase):

    def _make_setup(self, runner: CliRunner, sandbox: Path, files: list[str]) -> None:
        sandbox.mkdir(parents=True, exist_ok=True)
        for rel in files:
            dest = sandbox / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(f"content of {rel}", encoding="utf-8")
        _write_runs([_make_entry(
            sandbox=_make_sandbox_meta(worktree_path=str(sandbox.resolve())),
        )])

    def test_dry_run_logs_dry_run_receipt(self):
        runner = CliRunner()
        logged = []
        with runner.isolated_filesystem():
            sandbox = Path("wt")
            self._make_setup(runner, sandbox, ["a.py"])
            with patch("openshard.native.sandbox_apply.list_sandbox_changed_files", return_value=["a.py"]):
                with patch("openshard.cli.main.log_sandbox_apply_receipt", side_effect=logged.append) as mock_log:
                    runner.invoke(cli, ["apply-last", "--dry-run"])
                    self.assertEqual(mock_log.call_count, 1)
                    receipt: SandboxApplyReceipt = logged[0]
                    self.assertTrue(receipt.dry_run)
                    self.assertFalse(receipt.applied)

    def test_normal_apply_logs_non_dry_run_receipt(self):
        runner = CliRunner()
        logged = []
        with runner.isolated_filesystem():
            sandbox = Path("wt")
            self._make_setup(runner, sandbox, ["b.py"])
            with patch("openshard.native.sandbox_apply.list_sandbox_changed_files", return_value=["b.py"]):
                with patch("openshard.cli.main.log_sandbox_apply_receipt", side_effect=logged.append) as mock_log:
                    runner.invoke(cli, ["apply-last"])
                    self.assertEqual(mock_log.call_count, 1)
                    receipt: SandboxApplyReceipt = logged[0]
                    self.assertFalse(receipt.dry_run)

    def test_no_changes_logs_receipt_with_reason(self):
        runner = CliRunner()
        logged = []
        with runner.isolated_filesystem():
            sandbox = Path("wt")
            sandbox.mkdir()
            _write_runs([_make_entry(
                sandbox=_make_sandbox_meta(worktree_path=str(sandbox.resolve())),
            )])
            with patch("openshard.native.sandbox_apply.list_sandbox_changed_files", return_value=[]):
                with patch("openshard.cli.main.log_sandbox_apply_receipt", side_effect=logged.append) as mock_log:
                    runner.invoke(cli, ["apply-last"])
                    self.assertEqual(mock_log.call_count, 1)
                    receipt: SandboxApplyReceipt = logged[0]
                    self.assertIn("No sandbox changes", receipt.reason)

    def test_receipt_source_run_id_matches_timestamp(self):
        runner = CliRunner()
        logged = []
        with runner.isolated_filesystem():
            sandbox = Path("wt")
            self._make_setup(runner, sandbox, ["c.py"])
            with patch("openshard.native.sandbox_apply.list_sandbox_changed_files", return_value=["c.py"]):
                with patch("openshard.cli.main.log_sandbox_apply_receipt", side_effect=logged.append):
                    runner.invoke(cli, ["apply-last"])
                    self.assertEqual(logged[0].source_run_id, "2025-01-01T00:00:00Z")

    def test_receipt_sandbox_path_set(self):
        runner = CliRunner()
        logged = []
        with runner.isolated_filesystem():
            sandbox = Path("wt")
            self._make_setup(runner, sandbox, ["d.py"])
            with patch("openshard.native.sandbox_apply.list_sandbox_changed_files", return_value=["d.py"]):
                with patch("openshard.cli.main.log_sandbox_apply_receipt", side_effect=logged.append):
                    runner.invoke(cli, ["apply-last"])
                    self.assertTrue(logged[0].sandbox_path)

    def test_receipt_files_applied_correct(self):
        runner = CliRunner()
        logged = []
        with runner.isolated_filesystem():
            sandbox = Path("wt")
            self._make_setup(runner, sandbox, ["e.py"])
            with patch("openshard.native.sandbox_apply.list_sandbox_changed_files", return_value=["e.py"]):
                with patch("openshard.cli.main.log_sandbox_apply_receipt", side_effect=logged.append):
                    runner.invoke(cli, ["apply-last"])
                    self.assertIn("e.py", logged[0].files_applied)

    def test_receipt_never_contains_raw_content(self):
        runner = CliRunner()
        logged = []
        with runner.isolated_filesystem():
            sandbox = Path("wt")
            self._make_setup(runner, sandbox, ["f.py"])
            with patch("openshard.native.sandbox_apply.list_sandbox_changed_files", return_value=["f.py"]):
                with patch("openshard.cli.main.log_sandbox_apply_receipt", side_effect=logged.append):
                    runner.invoke(cli, ["apply-last"])
                    receipt = logged[0]
                    self.assertFalse(receipt.raw_content_stored)
                    d = _receipt_to_dict(receipt)
                    self.assertFalse(d["raw_content_stored"])
                    self.assertNotIn("content of f.py", json.dumps(d))


# ---------------------------------------------------------------------------
# TestApplyReceiptsCommand
# ---------------------------------------------------------------------------

class TestApplyReceiptsCommand(unittest.TestCase):

    def test_no_receipts_message(self):
        runner = CliRunner()
        with runner.isolated_filesystem():
            result = runner.invoke(cli, ["apply-receipts"])
            self.assertEqual(result.exit_code, 0)
            self.assertIn("No sandbox apply receipts recorded yet", result.output)

    def test_table_headers_present(self):
        runner = CliRunner()
        with runner.isolated_filesystem():
            log_sandbox_apply_receipt(SandboxApplyReceipt(sandbox_path="/tmp/wt"))
            result = runner.invoke(cli, ["apply-receipts"])
            self.assertIn("Time", result.output)
            self.assertIn("Applied", result.output)
            self.assertIn("Skipped", result.output)
            self.assertIn("Dry Run", result.output)
            self.assertIn("Sandbox", result.output)

    def test_last_limits_output(self):
        runner = CliRunner()
        with runner.isolated_filesystem():
            for i in range(5):
                log_sandbox_apply_receipt(SandboxApplyReceipt(sandbox_path=f"/tmp/wt{i}"))
            result = runner.invoke(cli, ["apply-receipts", "--last", "2"])
            self.assertEqual(result.exit_code, 0)
            data_lines = [
                ln for ln in result.output.splitlines()
                if "/tmp/wt" in ln
            ]
            self.assertEqual(len(data_lines), 2)

    def test_dry_run_column_shows_yes(self):
        runner = CliRunner()
        with runner.isolated_filesystem():
            log_sandbox_apply_receipt(SandboxApplyReceipt(dry_run=True, sandbox_path="/tmp/wt"))
            result = runner.invoke(cli, ["apply-receipts"])
            self.assertIn("yes", result.output)

    def test_non_dry_run_column_shows_no(self):
        runner = CliRunner()
        with runner.isolated_filesystem():
            log_sandbox_apply_receipt(SandboxApplyReceipt(dry_run=False, sandbox_path="/tmp/wt"))
            result = runner.invoke(cli, ["apply-receipts"])
            self.assertIn("no", result.output)

    def test_applied_skipped_counts_visible(self):
        runner = CliRunner()
        with runner.isolated_filesystem():
            log_sandbox_apply_receipt(SandboxApplyReceipt(
                files_applied=["a.py", "b.py"],
                files_skipped=["c.py"],
                sandbox_path="/tmp/wt",
            ))
            result = runner.invoke(cli, ["apply-receipts"])
            self.assertIn("2", result.output)
            self.assertIn("1", result.output)

    def test_sandbox_path_visible(self):
        runner = CliRunner()
        with runner.isolated_filesystem():
            log_sandbox_apply_receipt(SandboxApplyReceipt(sandbox_path="/tmp/my-sandbox"))
            result = runner.invoke(cli, ["apply-receipts"])
            self.assertIn("/tmp/my-sandbox", result.output)


if __name__ == "__main__":
    unittest.main()
