from __future__ import annotations

import json
import unittest
from pathlib import Path

from click.testing import CliRunner

from openshard.cli.main import cli
from openshard.history.feedback import (
    ALLOWED_OUTCOMES,
    FeedbackRecord,
    build_feedback_record,
    load_feedback_records,
    log_feedback_record,
)
from openshard.history.shard_contract import _make_shard_id


def _make_entry(**kwargs) -> dict:
    return {
        "task": "test task",
        "timestamp": "2025-01-01T00:00:00Z",
        "execution_model": "gpt-4o",
        "duration_seconds": 1.0,
        "files_created": 0,
        "files_updated": 0,
        "files_deleted": 0,
        "retry_triggered": False,
        "verification_attempted": False,
        "verification_passed": None,
        "workspace_path": None,
        "summary": "",
        **kwargs,
    }


def _write_runs(tmp_path: Path, entries: list[dict]) -> Path:
    log_dir = tmp_path / ".openshard"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "runs.jsonl"
    with log_path.open("w", encoding="utf-8") as fh:
        for entry in entries:
            fh.write(json.dumps(entry) + "\n")
    return log_path


class TestFeedbackRecordBuilding(unittest.TestCase):

    def test_record_built_for_last_run(self):
        entry = _make_entry(task="Fix the login bug", timestamp="2025-06-01T10:00:00Z")
        record = build_feedback_record(entry, run_index=4, outcome="accepted", note="")
        self.assertIsInstance(record, FeedbackRecord)
        self.assertEqual(record.outcome, "accepted")
        self.assertEqual(record.shard_id, _make_shard_id("2025-06-01T10:00:00Z", 4))
        self.assertEqual(record.task_short, "Fix the login bug")
        self.assertEqual(record.run_timestamp, "2025-06-01T10:00:00Z")
        self.assertEqual(record.source, "cli")
        self.assertEqual(record.schema_version, 1)

    def test_all_valid_outcomes_build_successfully(self):
        entry = _make_entry()
        for outcome in ALLOWED_OUTCOMES:
            record = build_feedback_record(entry, run_index=0, outcome=outcome, note="")
            self.assertEqual(record.outcome, outcome)

    def test_invalid_outcome_rejected_by_cli(self):
        runner = CliRunner()
        with runner.isolated_filesystem():
            _write_runs(Path("."), [_make_entry()])
            result = runner.invoke(cli, ["feedback", "--outcome", "bogus"])
            self.assertNotEqual(result.exit_code, 0)

    def test_note_optional_stored_as_empty_string(self):
        entry = _make_entry()
        record = build_feedback_record(entry, run_index=0, outcome="useful", note="")
        self.assertEqual(record.note, "")


class TestFeedbackStorage(unittest.TestCase):

    def test_feedback_writes_append_only_jsonl(self, tmp_path=None):
        runner = CliRunner()
        with runner.isolated_filesystem() as td:
            cwd = Path(td)
            entry = _make_entry()
            r1 = build_feedback_record(entry, 0, "accepted", "first")
            r2 = build_feedback_record(entry, 0, "rejected", "second")
            log_feedback_record(r1, cwd=cwd)
            log_feedback_record(r2, cwd=cwd)
            fb_path = cwd / ".openshard" / "feedback.jsonl"
            lines = [ln for ln in fb_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
            self.assertEqual(len(lines), 2)
            d1 = json.loads(lines[0])
            d2 = json.loads(lines[1])
            self.assertEqual(d1["outcome"], "accepted")
            self.assertEqual(d2["outcome"], "rejected")

    def test_feedback_id_stable_format(self):
        entry = _make_entry()
        record = build_feedback_record(entry, 0, "useful", "")
        self.assertRegex(record.feedback_id, r"^fb-\d{8}-\d{6}-\d{6}$")

    def test_feedback_record_references_shard_id(self):
        runner = CliRunner()
        with runner.isolated_filesystem() as td:
            cwd = Path(td)
            entry = _make_entry(timestamp="2025-03-15T09:30:00Z")
            record = build_feedback_record(entry, run_index=2, outcome="partial", note="")
            log_feedback_record(record, cwd=cwd)
            records = load_feedback_records(cwd=cwd)
            self.assertEqual(len(records), 1)
            expected_shard_id = _make_shard_id("2025-03-15T09:30:00Z", 2)
            self.assertEqual(records[0].shard_id, expected_shard_id)


class TestFeedbackCLI(unittest.TestCase):

    def test_missing_last_run_shows_clear_error(self):
        runner = CliRunner()
        with runner.isolated_filesystem():
            result = runner.invoke(cli, ["feedback", "--outcome", "accepted"])
            self.assertNotEqual(result.exit_code, 0)
            self.assertIn("No run history found", result.output)

    def test_cli_saves_feedback_for_last_run(self):
        runner = CliRunner()
        with runner.isolated_filesystem() as td:
            cwd = Path(td)
            log_path = _write_runs(cwd, [_make_entry(task="first"), _make_entry(task="second")])
            result = runner.invoke(cli, ["feedback", "--outcome", "accepted"])
            self.assertEqual(result.exit_code, 0)
            entries = [json.loads(ln) for ln in log_path.read_text().splitlines() if ln.strip()]
            self.assertEqual(entries[-1]["developer_feedback"]["outcome"], "accepted")

    def test_cli_output_shows_feedback_saved_outcome(self):
        runner = CliRunner()
        with runner.isolated_filesystem() as td:
            _write_runs(Path(td), [_make_entry()])
            result = runner.invoke(cli, ["feedback", "--outcome", "rejected", "--reason", "missed the issue"])
            self.assertEqual(result.exit_code, 0)
            self.assertIn("Feedback recorded: rejected", result.output)

    def test_no_external_network_behaviour(self):
        import inspect

        import openshard.history.feedback as fb_module
        source = inspect.getsource(fb_module)
        for net_import in ("socket", "urllib", "requests", "httpx", "http.client"):
            self.assertNotIn(net_import, source, f"feedback.py must not import {net_import}")

    def test_old_run_history_still_works(self):
        runner = CliRunner()
        with runner.isolated_filesystem() as td:
            cwd = Path(td)
            entry = _make_entry(task="old task")
            _write_runs(cwd, [entry])
            record = build_feedback_record(entry, 0, "accepted", "all good")
            log_feedback_record(record, cwd=cwd)
            result = runner.invoke(cli, ["last", "--more"])
            self.assertEqual(result.exit_code, 0)

    def test_two_feedback_calls_overwrite_developer_feedback(self):
        runner = CliRunner()
        with runner.isolated_filesystem() as td:
            cwd = Path(td)
            log_path = _write_runs(cwd, [_make_entry()])
            r1 = runner.invoke(cli, ["feedback", "--outcome", "accepted"])
            r2 = runner.invoke(cli, ["feedback", "--outcome", "rejected"])
            self.assertEqual(r1.exit_code, 0)
            self.assertEqual(r2.exit_code, 0)
            entries = [json.loads(ln) for ln in log_path.read_text().splitlines() if ln.strip()]
            self.assertEqual(entries[-1]["developer_feedback"]["outcome"], "rejected")

    def test_outcome_path_does_not_mutate_runs_jsonl(self):
        runner = CliRunner()
        with runner.isolated_filesystem() as td:
            cwd = Path(td)
            _write_runs(cwd, [_make_entry(task="original")])
            runner.invoke(cli, ["feedback", "--outcome", "accepted"])
            log_path = cwd / ".openshard" / "runs.jsonl"
            entries = [json.loads(ln) for ln in log_path.read_text().splitlines() if ln.strip()]
            self.assertEqual(len(entries), 1)
            self.assertNotIn("feedback", entries[0])

    def test_receipt_shows_feedback_section_on_last_more(self):
        runner = CliRunner()
        with runner.isolated_filesystem() as td:
            cwd = Path(td)
            _write_runs(cwd, [_make_entry(task="deploy fix")])
            record = build_feedback_record(_make_entry(task="deploy fix"), 0, "accepted", "clean result")
            log_feedback_record(record, cwd=cwd)
            result = runner.invoke(cli, ["last", "--more"])
            self.assertEqual(result.exit_code, 0)
            self.assertIn("FEEDBACK", result.output)
            self.assertIn("accepted", result.output)
            self.assertIn("clean result", result.output)

    def test_outcome_feedback_works(self):
        runner = CliRunner()
        with runner.isolated_filesystem():
            _write_runs(Path("."), [_make_entry()])
            result = runner.invoke(cli, ["feedback", "--outcome", "accepted"])
            self.assertEqual(result.exit_code, 0)
            self.assertIn("Feedback recorded", result.output)


if __name__ == "__main__":
    unittest.main()
