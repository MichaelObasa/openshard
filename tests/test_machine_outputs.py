from __future__ import annotations

import json
import unittest
from pathlib import Path

from click.testing import CliRunner

from openshard.cli.main import cli


def _make_entry(**kwargs) -> dict:
    return {
        "schema_version": "1.1",
        "task": "Add tests for the auth module",
        "timestamp": "2026-01-01T00:00:00Z",
        "execution_model": "gpt-4o",
        "duration_seconds": 1.0,
        "files_created": 1,
        "files_updated": 0,
        "files_deleted": 0,
        "verification_attempted": True,
        "verification_passed": True,
        "summary": "done",
        **kwargs,
    }


def _write_runs(entries: list[dict]) -> None:
    log_dir = Path(".openshard")
    log_dir.mkdir(parents=True, exist_ok=True)
    with (log_dir / "runs.jsonl").open("w", encoding="utf-8") as fh:
        for entry in entries:
            fh.write(json.dumps(entry) + "\n")


def _assert_no_unsafe(test: unittest.TestCase, text: str) -> None:
    """No absolute local paths and no obvious secret tokens in JSON output."""
    for needle in ("C:\\", "C:/", "/Users/", "/home/", "sk-", "AKIA"):
        test.assertNotIn(needle, text, msg=f"unsafe substring {needle!r} leaked: {text}")


class _Base(unittest.TestCase):
    def setUp(self) -> None:
        self.runner = CliRunner()


class TestLastJson(_Base):
    def test_success_emits_valid_envelope(self):
        with self.runner.isolated_filesystem():
            _write_runs([_make_entry()])
            result = self.runner.invoke(cli, ["last", "--json"])
        self.assertEqual(result.exit_code, 0, msg=result.output)
        data = json.loads(result.output)
        self.assertEqual(data["schema_version"], "1")
        self.assertEqual(data["command"], "last")
        self.assertEqual(data["status"], "ok")
        self.assertTrue(data["shard_id"])
        self.assertIsInstance(data["warnings"], list)
        self.assertIsInstance(data["run"], dict)
        self.assertEqual(data["run"]["task"], "Add tests for the auth module")
        _assert_no_unsafe(self, result.output)

    def test_no_run_is_clean_not_found(self):
        with self.runner.isolated_filesystem():
            result = self.runner.invoke(cli, ["last", "--json"])
        self.assertEqual(result.exit_code, 0, msg=result.output)
        data = json.loads(result.output)
        self.assertEqual(data["status"], "not_found")
        self.assertIsNone(data["shard_id"])
        self.assertIsNone(data["run"])
        self.assertEqual(data["warnings"], [])

    def test_human_output_unchanged(self):
        with self.runner.isolated_filesystem():
            _write_runs([_make_entry()])
            result = self.runner.invoke(cli, ["last"])
        self.assertEqual(result.exit_code, 0, msg=result.output)
        # Human path is not JSON.
        with self.assertRaises(json.JSONDecodeError):
            json.loads(result.output)

    def test_timeline_included_when_present(self):
        with self.runner.isolated_filesystem():
            _write_runs([_make_entry(run_timeline=[
                {"event": "repo_scanned", "label": "Scanned repo",
                 "kind": "scan", "status": "completed"},
                {"event": "receipt_saved", "label": "Saved Shard receipt",
                 "kind": "receipt", "status": "completed"},
            ])])
            result = self.runner.invoke(cli, ["last", "--json"])
        self.assertEqual(result.exit_code, 0, msg=result.output)
        data = json.loads(result.output)
        timeline = data["run"]["timeline"]
        self.assertIsInstance(timeline, list)
        events = {e["event"] for e in timeline}
        self.assertIn("repo_scanned", events)
        self.assertIn("receipt_saved", events)
        for e in timeline:
            self.assertIn("label", e)
            self.assertIn("kind", e)
            self.assertIn("status", e)
        _assert_no_unsafe(self, result.output)

    def test_timeline_empty_for_old_receipt(self):
        with self.runner.isolated_filesystem():
            _write_runs([_make_entry()])  # no run_timeline key
            result = self.runner.invoke(cli, ["last", "--json"])
        self.assertEqual(result.exit_code, 0, msg=result.output)
        data = json.loads(result.output)
        self.assertEqual(data["run"]["timeline"], [])

    def test_timeline_strips_unsafe_values(self):
        with self.runner.isolated_filesystem():
            _write_runs([_make_entry(run_timeline=[
                {"event": "x", "label": "did work",
                 "kind": "run", "status": "completed",
                 "target": r"C:\Users\alice\.env",
                 "detail": "token=supersecretleak1234",
                 "metadata": {"k": "sk-ABCDEFGHIJKLMNOP1234"}},
            ])])
            result = self.runner.invoke(cli, ["last", "--json"])
        self.assertEqual(result.exit_code, 0, msg=result.output)
        # Absolute path + secret-like tokens must not appear anywhere.
        _assert_no_unsafe(self, result.output)
        self.assertNotIn("token=supersecretleak", result.output)
        data = json.loads(result.output)
        row = data["run"]["timeline"][0]
        self.assertNotIn("target", row)
        self.assertNotIn("detail", row)


class TestReflectLastJson(_Base):
    def test_success_emits_valid_envelope(self):
        with self.runner.isolated_filesystem():
            _write_runs([_make_entry()])
            result = self.runner.invoke(cli, ["reflect", "last", "--json"])
        self.assertEqual(result.exit_code, 0, msg=result.output)
        data = json.loads(result.output)
        self.assertEqual(data["schema_version"], "1")
        self.assertEqual(data["command"], "reflect last")
        self.assertEqual(data["status"], "ok")
        self.assertTrue(data["shard_id"])
        self.assertIsInstance(data["warnings"], list)
        reflection = data["reflection"]
        self.assertIsInstance(reflection, dict)
        self.assertIn("score", reflection)
        self.assertIn("level", reflection)
        self.assertEqual(reflection["version"], "reflector_v0")
        # Top-level warnings mirror the reflection warnings.
        self.assertEqual(data["warnings"], reflection["warnings"])
        _assert_no_unsafe(self, result.output)

    def test_no_run_is_clean_not_found(self):
        with self.runner.isolated_filesystem():
            result = self.runner.invoke(cli, ["reflect", "last", "--json"])
        self.assertEqual(result.exit_code, 0, msg=result.output)
        data = json.loads(result.output)
        self.assertEqual(data["status"], "not_found")
        self.assertIsNone(data["shard_id"])
        self.assertIsNone(data["reflection"])


class TestPrCommentJson(_Base):
    def test_success_emits_valid_envelope(self):
        with self.runner.isolated_filesystem():
            _write_runs([_make_entry()])
            result = self.runner.invoke(cli, ["pr", "comment", "--json"])
        self.assertEqual(result.exit_code, 0, msg=result.output)
        data = json.loads(result.output)
        self.assertEqual(data["schema_version"], "1")
        self.assertEqual(data["command"], "pr comment")
        self.assertEqual(data["status"], "ok")
        self.assertTrue(data["shard_id"])
        self.assertIsInstance(data["warnings"], list)
        summary = data["summary"]
        self.assertIsInstance(summary, dict)
        self.assertEqual(summary["source"], "github_pr_comment_v1")
        self.assertFalse(summary["raw_content_stored"])
        self.assertNotIn("written", data)
        _assert_no_unsafe(self, result.output)

    def test_no_run_is_clean_not_found(self):
        with self.runner.isolated_filesystem():
            result = self.runner.invoke(cli, ["pr", "comment", "--json"])
        self.assertEqual(result.exit_code, 0, msg=result.output)
        data = json.loads(result.output)
        self.assertEqual(data["status"], "not_found")
        self.assertIsNone(data["summary"])

    def test_json_with_output_writes_file_and_prints_json(self):
        with self.runner.isolated_filesystem():
            _write_runs([_make_entry()])
            result = self.runner.invoke(
                cli, ["pr", "comment", "--json", "--output", "out/comment.json"]
            )
            self.assertEqual(result.exit_code, 0, msg=result.output)
            # stdout is still pure JSON (no human confirmation line).
            data = json.loads(result.output)
            self.assertTrue(data["written"])
            self.assertEqual(data["output_path_display"], "out/comment.json")
            self.assertNotIn("PR comment written", result.output)
            # File holds valid JSON too.
            written = Path("out/comment.json").read_text(encoding="utf-8")
            file_data = json.loads(written)
            self.assertEqual(file_data["command"], "pr comment")
            self.assertEqual(file_data["status"], "ok")
        _assert_no_unsafe(self, result.output)

    def test_human_output_unchanged_with_output_flag(self):
        with self.runner.isolated_filesystem():
            _write_runs([_make_entry()])
            result = self.runner.invoke(cli, ["pr", "comment", "--output", "comment.md"])
            self.assertEqual(result.exit_code, 0, msg=result.output)
            self.assertIn("PR comment written to", result.output)
            self.assertTrue(Path("comment.md").exists())


if __name__ == "__main__":
    unittest.main()
