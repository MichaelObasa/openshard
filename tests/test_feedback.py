from __future__ import annotations

import json
import unittest
from pathlib import Path

from click.testing import CliRunner

from openshard.cli.main import cli


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


class TestFeedbackCommand(unittest.TestCase):

    def test_feedback_adds_outcome_to_last_entry(self):
        runner = CliRunner()
        with runner.isolated_filesystem():
            log_path = _write_runs([_make_entry(task="first"), _make_entry(task="second")])
            result = runner.invoke(cli, ["feedback", "--outcome", "accepted"])
            self.assertEqual(result.exit_code, 0)
            entries = _read_entries(log_path)
            self.assertEqual(entries[-1]["developer_feedback"]["outcome"], "accepted")

    def test_feedback_adds_reason_to_last_entry(self):
        runner = CliRunner()
        with runner.isolated_filesystem():
            log_path = _write_runs([_make_entry()])
            runner.invoke(cli, ["feedback", "--outcome", "partial", "--reason", "GLM was good enough"])
            entries = _read_entries(log_path)
            self.assertEqual(entries[-1]["developer_feedback"]["reason"], "GLM was good enough")

    def test_feedback_reason_is_none_when_omitted(self):
        runner = CliRunner()
        with runner.isolated_filesystem():
            log_path = _write_runs([_make_entry()])
            runner.invoke(cli, ["feedback", "--outcome", "rejected"])
            entries = _read_entries(log_path)
            self.assertIsNone(entries[-1]["developer_feedback"]["reason"])

    def test_feedback_stores_schema_version(self):
        runner = CliRunner()
        with runner.isolated_filesystem():
            log_path = _write_runs([_make_entry()])
            runner.invoke(cli, ["feedback", "--outcome", "accepted"])
            entries = _read_entries(log_path)
            self.assertEqual(entries[-1]["developer_feedback"]["schema_version"], 1)

    def test_feedback_creates_recorded_at_timestamp(self):
        runner = CliRunner()
        with runner.isolated_filesystem():
            log_path = _write_runs([_make_entry()])
            runner.invoke(cli, ["feedback", "--outcome", "accepted"])
            entries = _read_entries(log_path)
            recorded_at = entries[-1]["developer_feedback"]["recorded_at"]
            self.assertIsNotNone(recorded_at)
            self.assertGreater(len(recorded_at), 10)

    def test_feedback_only_updates_last_entry(self):
        runner = CliRunner()
        with runner.isolated_filesystem():
            log_path = _write_runs([
                _make_entry(task="first"),
                _make_entry(task="second"),
                _make_entry(task="third"),
            ])
            runner.invoke(cli, ["feedback", "--outcome", "accepted"])
            entries = _read_entries(log_path)
            self.assertNotIn("developer_feedback", entries[0])
            self.assertNotIn("developer_feedback", entries[1])
            self.assertIn("developer_feedback", entries[2])

    def test_feedback_preserves_entry_count(self):
        runner = CliRunner()
        with runner.isolated_filesystem():
            log_path = _write_runs([_make_entry(task="a"), _make_entry(task="b")])
            runner.invoke(cli, ["feedback", "--outcome", "accepted"])
            entries = _read_entries(log_path)
            self.assertEqual(len(entries), 2)

    def test_invalid_outcome_rejected(self):
        runner = CliRunner()
        with runner.isolated_filesystem():
            _write_runs([_make_entry()])
            result = runner.invoke(cli, ["feedback", "--outcome", "excellent"])
            self.assertNotEqual(result.exit_code, 0)

    def test_no_file_shows_clear_message(self):
        runner = CliRunner()
        with runner.isolated_filesystem():
            result = runner.invoke(cli, ["feedback", "--outcome", "accepted"])
            self.assertNotEqual(result.exit_code, 0)
            self.assertIn("No run history found", result.output)

    def test_empty_file_shows_error(self):
        runner = CliRunner()
        with runner.isolated_filesystem():
            Path(".openshard").mkdir()
            Path(".openshard/runs.jsonl").write_text("", encoding="utf-8")
            result = runner.invoke(cli, ["feedback", "--outcome", "accepted"])
            self.assertNotEqual(result.exit_code, 0)
            self.assertIn("No run history found", result.output)

    def test_success_exit_code(self):
        runner = CliRunner()
        with runner.isolated_filesystem():
            _write_runs([_make_entry()])
            result = runner.invoke(cli, ["feedback", "--outcome", "accepted"])
            self.assertEqual(result.exit_code, 0)

    def test_success_echo_confirms_outcome(self):
        runner = CliRunner()
        with runner.isolated_filesystem():
            _write_runs([_make_entry()])
            result = runner.invoke(cli, ["feedback", "--outcome", "accepted"])
            self.assertIn("Feedback recorded: accepted", result.output)

    def test_outcome_stored_lowercase(self):
        runner = CliRunner()
        with runner.isolated_filesystem():
            log_path = _write_runs([_make_entry()])
            runner.invoke(cli, ["feedback", "--outcome", "ACCEPTED"])
            entries = _read_entries(log_path)
            self.assertEqual(entries[-1]["developer_feedback"]["outcome"], "accepted")


class TestFeedbackCorrectionFields(unittest.TestCase):

    def test_accepted_outcome_stored(self):
        runner = CliRunner()
        with runner.isolated_filesystem():
            log_path = _write_runs([_make_entry()])
            result = runner.invoke(cli, ["feedback", "--outcome", "accepted"])
            self.assertEqual(result.exit_code, 0)
            entries = _read_entries(log_path)
            self.assertEqual(entries[-1]["developer_feedback"]["outcome"], "accepted")

    def test_edited_flag_and_reason_stored(self):
        runner = CliRunner()
        with runner.isolated_filesystem():
            log_path = _write_runs([_make_entry()])
            result = runner.invoke(cli, ["feedback", "--outcome", "partial", "--edited", "--reason", "wrong file targeted"])
            self.assertEqual(result.exit_code, 0)
            entries = _read_entries(log_path)
            df = entries[-1]["developer_feedback"]
            self.assertIs(df["edited"], True)
            self.assertEqual(df["reason"], "wrong file targeted")

    def test_retried_with_reason_stored(self):
        runner = CliRunner()
        with runner.isolated_filesystem():
            log_path = _write_runs([_make_entry()])
            result = runner.invoke(
                cli,
                ["feedback", "--outcome", "retried", "--reason", "failed tests missed fixture"],
            )
            self.assertEqual(result.exit_code, 0)
            entries = _read_entries(log_path)
            df = entries[-1]["developer_feedback"]
            self.assertEqual(df["outcome"], "retried")
            self.assertEqual(df["reason"], "failed tests missed fixture")

    def test_invalid_outcome_rejected(self):
        runner = CliRunner()
        with runner.isolated_filesystem():
            _write_runs([_make_entry()])
            result = runner.invoke(cli, ["feedback", "--outcome", "bogus"])
            self.assertNotEqual(result.exit_code, 0)

    def test_free_text_reason_accepted(self):
        runner = CliRunner()
        with runner.isolated_filesystem():
            _write_runs([_make_entry()])
            result = runner.invoke(cli, ["feedback", "--outcome", "accepted", "--reason", "any free text is valid"])
            self.assertEqual(result.exit_code, 0)

    def test_no_args_fails_with_missing_outcome_error(self):
        runner = CliRunner()
        with runner.isolated_filesystem():
            _write_runs([_make_entry()])
            result = runner.invoke(cli, ["feedback"])
            self.assertNotEqual(result.exit_code, 0)

    def test_outcome_case_insensitive(self):
        runner = CliRunner()
        with runner.isolated_filesystem():
            log_path = _write_runs([_make_entry()])
            runner.invoke(cli, ["feedback", "--outcome", "REJECTED"])
            entries = _read_entries(log_path)
            self.assertEqual(entries[-1]["developer_feedback"]["outcome"], "rejected")

    def test_old_entry_without_new_fields_renders_cleanly(self):
        runner = CliRunner()
        with runner.isolated_filesystem():
            _write_runs([_make_entry(feedback={"schema_version": 1, "rating": "good", "note": "", "created_at": "2025-01-01T00:00:00Z"})])
            result = runner.invoke(cli, ["last", "--more"])
            self.assertEqual(result.exit_code, 0)
            self.assertIn("Rating: good", result.output)


if __name__ == "__main__":
    unittest.main()
