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


def _parse_jsonl(text: str) -> list[dict]:
    return [json.loads(ln) for ln in text.splitlines() if ln.strip()]


class TestExportRunsCommand(unittest.TestCase):

    # 1. No run history file gives clear message and exit code 0.
    def test_no_history_file(self):
        runner = CliRunner()
        with runner.isolated_filesystem():
            result = runner.invoke(cli, ["export-runs"])
            self.assertEqual(result.exit_code, 0)
            self.assertIn("No run history found", result.output)

    # 2. Empty run file gives clear message and exit code 0.
    def test_empty_run_file(self):
        runner = CliRunner()
        with runner.isolated_filesystem():
            _write_runs([])
            # Overwrite with truly empty content
            Path(".openshard/runs.jsonl").write_text("", encoding="utf-8")
            result = runner.invoke(cli, ["export-runs"])
            self.assertEqual(result.exit_code, 0)
            self.assertIn("No runs recorded", result.output)

    # 3. Stdout export emits valid JSONL.
    def test_stdout_export_valid_jsonl(self):
        runner = CliRunner()
        with runner.isolated_filesystem():
            _write_runs([_make_entry(task="first"), _make_entry(task="second")])
            result = runner.invoke(cli, ["export-runs"])
            self.assertEqual(result.exit_code, 0)
            rows = _parse_jsonl(result.output)
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0]["task"], "first")
            self.assertEqual(rows[1]["task"], "second")

    # 4. Output file export writes JSONL.
    def test_output_file_export(self):
        runner = CliRunner()
        with runner.isolated_filesystem():
            _write_runs([_make_entry(task="mytask")])
            result = runner.invoke(cli, ["export-runs", "--output", "out.jsonl"])
            self.assertEqual(result.exit_code, 0)
            rows = _parse_jsonl(Path("out.jsonl").read_text(encoding="utf-8"))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["task"], "mytask")

    # 5. Export does not mutate .openshard/runs.jsonl.
    def test_does_not_mutate_runs_jsonl(self):
        runner = CliRunner()
        with runner.isolated_filesystem():
            log_path = _write_runs([_make_entry(task="unchanged")])
            before = log_path.read_bytes()
            runner.invoke(cli, ["export-runs"])
            after = log_path.read_bytes()
            self.assertEqual(before, after)

    # 6. Invalid JSON lines are skipped.
    def test_invalid_json_lines_skipped(self):
        runner = CliRunner()
        with runner.isolated_filesystem():
            log_dir = Path(".openshard")
            log_dir.mkdir(parents=True, exist_ok=True)
            log_path = log_dir / "runs.jsonl"
            with log_path.open("w", encoding="utf-8") as fh:
                fh.write(json.dumps(_make_entry(task="valid1")) + "\n")
                fh.write("{{not valid json}}\n")
                fh.write(json.dumps(_make_entry(task="valid2")) + "\n")
            result = runner.invoke(cli, ["export-runs"])
            self.assertEqual(result.exit_code, 0)
            rows = _parse_jsonl(result.output)
            self.assertEqual(len(rows), 2)
            tasks = [r["task"] for r in rows]
            self.assertIn("valid1", tasks)
            self.assertIn("valid2", tasks)

    # 7. --limit 1 exports only the newest entry.
    def test_limit_one(self):
        runner = CliRunner()
        with runner.isolated_filesystem():
            _write_runs([_make_entry(task="old"), _make_entry(task="new")])
            result = runner.invoke(cli, ["export-runs", "--limit", "1"])
            self.assertEqual(result.exit_code, 0)
            rows = _parse_jsonl(result.output)
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["task"], "new")

    # 8. Feedback fields are flattened correctly.
    def test_feedback_fields_flattened(self):
        runner = CliRunner()
        with runner.isolated_filesystem():
            entry = _make_entry(feedback={"schema_version": 1, "rating": "good", "note": "nice", "created_at": "2025-01-01T00:00:00Z"})
            _write_runs([entry])
            result = runner.invoke(cli, ["export-runs"])
            self.assertEqual(result.exit_code, 0)
            row = _parse_jsonl(result.output)[0]
            self.assertEqual(row["feedback_rating"], "good")
            self.assertEqual(row["feedback_note"], "nice")

    # 9. Tier dispatch fields are flattened correctly.
    def test_tier_dispatch_fields_flattened(self):
        runner = CliRunner()
        with runner.isolated_filesystem():
            tdr = {
                "enabled": True,
                "applied": True,
                "executor_model": "anthropic/claude-sonnet-4.6",
                "planner_model": "anthropic/claude-sonnet-4.6",
            }
            entry = _make_entry(tier_dispatch_receipt=tdr)
            _write_runs([entry])
            result = runner.invoke(cli, ["export-runs"])
            self.assertEqual(result.exit_code, 0)
            row = _parse_jsonl(result.output)[0]
            self.assertTrue(row["tier_dispatch_enabled"])
            self.assertTrue(row["tier_dispatch_applied"])
            self.assertEqual(row["tier_dispatch_work_model"], "anthropic/claude-sonnet-4.6")

    # 10. Notes are excluded by default.
    def test_notes_excluded_by_default(self):
        runner = CliRunner()
        with runner.isolated_filesystem():
            _write_runs([_make_entry(notes=["a note"])])
            result = runner.invoke(cli, ["export-runs"])
            self.assertEqual(result.exit_code, 0)
            row = _parse_jsonl(result.output)[0]
            self.assertNotIn("notes", row)

    # 11. Notes are included with --with-notes.
    def test_notes_included_with_flag(self):
        runner = CliRunner()
        with runner.isolated_filesystem():
            _write_runs([_make_entry(notes=["first note", "second note"])])
            result = runner.invoke(cli, ["export-runs", "--with-notes"])
            self.assertEqual(result.exit_code, 0)
            row = _parse_jsonl(result.output)[0]
            self.assertIn("notes", row)
            self.assertIsInstance(row["notes"], list)
            self.assertEqual(row["notes"], ["first note", "second note"])

    # 12. Read-only entries display execution_mode_label as Ask.
    def test_readonly_execution_mode_label(self):
        runner = CliRunner()
        with runner.isolated_filesystem():
            entry = _make_entry(routing_rationale="read-only analysis")
            _write_runs([entry])
            result = runner.invoke(cli, ["export-runs"])
            self.assertEqual(result.exit_code, 0)
            row = _parse_jsonl(result.output)[0]
            self.assertEqual(row["execution_mode_label"], "Ask")

    # 13. native_light displays execution_mode_label as Run.
    def test_native_light_execution_mode_label(self):
        runner = CliRunner()
        with runner.isolated_filesystem():
            _write_runs([_make_entry(execution_profile="native_light")])
            result = runner.invoke(cli, ["export-runs"])
            self.assertEqual(result.exit_code, 0)
            row = _parse_jsonl(result.output)[0]
            self.assertEqual(row["execution_mode_label"], "Run")

    # 14. native_deep displays execution_mode_label as Deep Run.
    def test_native_deep_execution_mode_label(self):
        runner = CliRunner()
        with runner.isolated_filesystem():
            _write_runs([_make_entry(execution_profile="native_deep")])
            result = runner.invoke(cli, ["export-runs"])
            self.assertEqual(result.exit_code, 0)
            row = _parse_jsonl(result.output)[0]
            self.assertEqual(row["execution_mode_label"], "Deep Run")

    # 15. native_swarm displays execution_mode_label as Deep Run.
    def test_native_swarm_execution_mode_label(self):
        runner = CliRunner()
        with runner.isolated_filesystem():
            _write_runs([_make_entry(execution_profile="native_swarm")])
            result = runner.invoke(cli, ["export-runs"])
            self.assertEqual(result.exit_code, 0)
            row = _parse_jsonl(result.output)[0]
            self.assertEqual(row["execution_mode_label"], "Deep Run")

    # 16. Missing fields export as null, not fake values.
    def test_missing_fields_are_null(self):
        runner = CliRunner()
        with runner.isolated_filesystem():
            # Minimal entry with no optional fields
            _write_runs([{"task": "sparse", "timestamp": "2025-01-01T00:00:00Z"}])
            result = runner.invoke(cli, ["export-runs"])
            self.assertEqual(result.exit_code, 0)
            row = _parse_jsonl(result.output)[0]
            self.assertIsNone(row["execution_model"])
            self.assertIsNone(row["routing_category"])
            self.assertIsNone(row["planning_model"])
            self.assertIsNone(row["analysis_model"])
            self.assertIsNone(row["total_cost_usd"])
            self.assertIsNone(row["feedback_rating"])
            self.assertIsNone(row["tier_dispatch_enabled"])
            self.assertIsNone(row["tier_dispatch_work_model"])


if __name__ == "__main__":
    unittest.main()
