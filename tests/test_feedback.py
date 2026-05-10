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

    def test_feedback_adds_rating_to_last_entry(self):
        runner = CliRunner()
        with runner.isolated_filesystem():
            log_path = _write_runs([_make_entry(task="first"), _make_entry(task="second")])
            result = runner.invoke(cli, ["feedback", "--rating", "good"])
            self.assertEqual(result.exit_code, 0)
            entries = _read_entries(log_path)
            self.assertEqual(entries[-1]["feedback"]["rating"], "good")

    def test_feedback_adds_note_to_last_entry(self):
        runner = CliRunner()
        with runner.isolated_filesystem():
            log_path = _write_runs([_make_entry()])
            runner.invoke(cli, ["feedback", "--rating", "mixed", "--note", "GLM was good enough"])
            entries = _read_entries(log_path)
            self.assertEqual(entries[-1]["feedback"]["note"], "GLM was good enough")

    def test_feedback_stores_empty_string_note_when_omitted(self):
        runner = CliRunner()
        with runner.isolated_filesystem():
            log_path = _write_runs([_make_entry()])
            runner.invoke(cli, ["feedback", "--rating", "bad"])
            entries = _read_entries(log_path)
            self.assertEqual(entries[-1]["feedback"]["note"], "")

    def test_feedback_stores_schema_version(self):
        runner = CliRunner()
        with runner.isolated_filesystem():
            log_path = _write_runs([_make_entry()])
            runner.invoke(cli, ["feedback", "--rating", "good"])
            entries = _read_entries(log_path)
            self.assertEqual(entries[-1]["feedback"]["schema_version"], 1)

    def test_feedback_creates_created_at_timestamp(self):
        runner = CliRunner()
        with runner.isolated_filesystem():
            log_path = _write_runs([_make_entry()])
            runner.invoke(cli, ["feedback", "--rating", "good"])
            entries = _read_entries(log_path)
            created_at = entries[-1]["feedback"]["created_at"]
            self.assertTrue(created_at.endswith("Z"), created_at)

    def test_feedback_only_updates_last_entry(self):
        runner = CliRunner()
        with runner.isolated_filesystem():
            log_path = _write_runs([
                _make_entry(task="first"),
                _make_entry(task="second"),
                _make_entry(task="third"),
            ])
            runner.invoke(cli, ["feedback", "--rating", "good"])
            entries = _read_entries(log_path)
            self.assertNotIn("feedback", entries[0])
            self.assertNotIn("feedback", entries[1])
            self.assertIn("feedback", entries[2])

    def test_feedback_preserves_entry_count(self):
        runner = CliRunner()
        with runner.isolated_filesystem():
            log_path = _write_runs([_make_entry(task="a"), _make_entry(task="b")])
            runner.invoke(cli, ["feedback", "--rating", "good"])
            entries = _read_entries(log_path)
            self.assertEqual(len(entries), 2)

    def test_invalid_rating_rejected(self):
        runner = CliRunner()
        with runner.isolated_filesystem():
            _write_runs([_make_entry()])
            result = runner.invoke(cli, ["feedback", "--rating", "excellent"])
            self.assertNotEqual(result.exit_code, 0)

    def test_no_file_shows_clear_message(self):
        runner = CliRunner()
        with runner.isolated_filesystem():
            result = runner.invoke(cli, ["feedback", "--rating", "good"])
            self.assertEqual(result.exit_code, 0)
            self.assertIn("No run history found", result.output)

    def test_empty_file_shows_no_runs_message(self):
        runner = CliRunner()
        with runner.isolated_filesystem():
            Path(".openshard").mkdir()
            Path(".openshard/runs.jsonl").write_text("", encoding="utf-8")
            result = runner.invoke(cli, ["feedback", "--rating", "good"])
            self.assertEqual(result.exit_code, 0)
            self.assertIn("No runs recorded yet", result.output)

    def test_success_exit_code(self):
        runner = CliRunner()
        with runner.isolated_filesystem():
            _write_runs([_make_entry()])
            result = runner.invoke(cli, ["feedback", "--rating", "good"])
            self.assertEqual(result.exit_code, 0)

    def test_success_echo_confirms_rating(self):
        runner = CliRunner()
        with runner.isolated_filesystem():
            _write_runs([_make_entry()])
            result = runner.invoke(cli, ["feedback", "--rating", "good"])
            self.assertIn("Feedback recorded: good", result.output)

    def test_rating_stored_lowercase(self):
        runner = CliRunner()
        with runner.isolated_filesystem():
            log_path = _write_runs([_make_entry()])
            runner.invoke(cli, ["feedback", "--rating", "GOOD"])
            entries = _read_entries(log_path)
            self.assertEqual(entries[-1]["feedback"]["rating"], "good")


if __name__ == "__main__":
    unittest.main()
