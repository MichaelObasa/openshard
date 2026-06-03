from __future__ import annotations

import json
import os
import unittest
from pathlib import Path
from unittest.mock import patch

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
    for needle in ("C:\\", "C:/", "/Users/", "/home/", "sk-", "AKIA"):
        test.assertNotIn(needle, text, msg=f"unsafe substring {needle!r} leaked: {text}")


class _Base(unittest.TestCase):
    def setUp(self) -> None:
        self.runner = CliRunner()


class TestStepSummary(_Base):
    def test_written_when_env_set(self):
        with self.runner.isolated_filesystem():
            _write_runs([_make_entry()])
            with patch.dict(os.environ, {"GITHUB_STEP_SUMMARY": "summary.md"}, clear=False):
                result = self.runner.invoke(cli, ["pr", "comment", "--github-step-summary"])
            self.assertEqual(result.exit_code, 0, msg=result.output)
            content = Path("summary.md").read_text(encoding="utf-8")
            self.assertIn("## OpenShard Run Summary", content)
            _assert_no_unsafe(self, content)

    def test_graceful_when_env_missing(self):
        with self.runner.isolated_filesystem():
            _write_runs([_make_entry()])
            env = {k: v for k, v in os.environ.items() if k != "GITHUB_STEP_SUMMARY"}
            with patch.dict(os.environ, env, clear=True):
                result = self.runner.invoke(cli, ["pr", "comment", "--github-step-summary"])
            self.assertEqual(result.exit_code, 0, msg=result.output)
            # Human markdown still produced; nothing else written.
            self.assertIn("## OpenShard Run Summary", result.output)
            self.assertFalse(Path("summary.md").exists())


class TestGithubOutput(_Base):
    def test_written_when_env_set(self):
        with self.runner.isolated_filesystem():
            _write_runs([_make_entry()])
            with patch.dict(os.environ, {"GITHUB_OUTPUT": "gh_out.txt"}, clear=False):
                result = self.runner.invoke(cli, ["pr", "comment", "--github-output"])
            self.assertEqual(result.exit_code, 0, msg=result.output)
            content = Path("gh_out.txt").read_text(encoding="utf-8")
            self.assertIn("openshard_available=true", content)
            self.assertIn("openshard_status=ok", content)
            self.assertIn("openshard_shard_id=", content)
            self.assertIn("openshard_manual_review_required=", content)
            # Each output stays on a single line.
            for line in content.splitlines():
                self.assertIn("=", line)
            _assert_no_unsafe(self, content)

    def test_graceful_when_env_missing(self):
        with self.runner.isolated_filesystem():
            _write_runs([_make_entry()])
            env = {k: v for k, v in os.environ.items() if k != "GITHUB_OUTPUT"}
            with patch.dict(os.environ, env, clear=True):
                result = self.runner.invoke(cli, ["pr", "comment", "--github-output"])
            self.assertEqual(result.exit_code, 0, msg=result.output)
            self.assertFalse(Path("gh_out.txt").exists())

    def test_manual_review_true(self):
        with self.runner.isolated_filesystem():
            _write_runs([_make_entry(error_class="VerificationError")])
            with patch.dict(os.environ, {"GITHUB_OUTPUT": "gh_out.txt"}, clear=False):
                result = self.runner.invoke(cli, ["pr", "comment", "--github-output"])
            self.assertEqual(result.exit_code, 0, msg=result.output)
            content = Path("gh_out.txt").read_text(encoding="utf-8")
            self.assertIn("openshard_manual_review_required=true", content)

    def test_human_output_uses_comment_path_key(self):
        with self.runner.isolated_filesystem():
            _write_runs([_make_entry()])
            with patch.dict(os.environ, {"GITHUB_OUTPUT": "gh_out.txt"}, clear=False):
                result = self.runner.invoke(
                    cli, ["pr", "comment", "--output", "comment.md", "--github-output"]
                )
            self.assertEqual(result.exit_code, 0, msg=result.output)
            content = Path("gh_out.txt").read_text(encoding="utf-8")
            self.assertIn("openshard_comment_path=comment.md", content)
            self.assertNotIn("openshard_output_path=", content)


class TestJsonMode(_Base):
    def test_json_with_both_flags_env_set(self):
        with self.runner.isolated_filesystem():
            _write_runs([_make_entry()])
            with patch.dict(
                os.environ,
                {"GITHUB_STEP_SUMMARY": "summary.md", "GITHUB_OUTPUT": "gh_out.txt"},
                clear=False,
            ):
                result = self.runner.invoke(
                    cli, ["pr", "comment", "--json", "--github-step-summary", "--github-output"]
                )
            self.assertEqual(result.exit_code, 0, msg=result.output)
            data = json.loads(result.output)  # stdout must be valid JSON only
            self.assertEqual(data["status"], "ok")
            self.assertTrue(data["github_step_summary_available"])
            self.assertTrue(data["github_step_summary_written"])
            self.assertTrue(data["github_output_available"])
            self.assertTrue(data["github_output_written"])
            self.assertTrue(Path("summary.md").exists())
            self.assertTrue(Path("gh_out.txt").exists())
            _assert_no_unsafe(self, result.output)
            _assert_no_unsafe(self, Path("gh_out.txt").read_text(encoding="utf-8"))

    def test_json_with_both_flags_env_missing(self):
        with self.runner.isolated_filesystem():
            _write_runs([_make_entry()])
            env = {
                k: v for k, v in os.environ.items()
                if k not in ("GITHUB_STEP_SUMMARY", "GITHUB_OUTPUT")
            }
            with patch.dict(os.environ, env, clear=True):
                result = self.runner.invoke(
                    cli, ["pr", "comment", "--json", "--github-step-summary", "--github-output"]
                )
            self.assertEqual(result.exit_code, 0, msg=result.output)
            data = json.loads(result.output)
            self.assertFalse(data["github_step_summary_available"])
            self.assertFalse(data["github_step_summary_written"])
            self.assertFalse(data["github_output_available"])
            self.assertFalse(data["github_output_written"])

    def test_json_output_uses_output_path_key(self):
        with self.runner.isolated_filesystem():
            _write_runs([_make_entry()])
            with patch.dict(os.environ, {"GITHUB_OUTPUT": "gh_out.txt"}, clear=False):
                result = self.runner.invoke(
                    cli, ["pr", "comment", "--json", "--output", "out.json", "--github-output"]
                )
            self.assertEqual(result.exit_code, 0, msg=result.output)
            data = json.loads(result.output)
            self.assertTrue(data["written"])
            content = Path("gh_out.txt").read_text(encoding="utf-8")
            self.assertIn("openshard_output_path=out.json", content)
            self.assertNotIn("openshard_comment_path=", content)


class TestNotFound(_Base):
    def test_not_found_with_both_flags_json(self):
        with self.runner.isolated_filesystem():
            with patch.dict(
                os.environ,
                {"GITHUB_STEP_SUMMARY": "summary.md", "GITHUB_OUTPUT": "gh_out.txt"},
                clear=False,
            ):
                result = self.runner.invoke(
                    cli, ["pr", "comment", "--json", "--github-step-summary", "--github-output"]
                )
            self.assertEqual(result.exit_code, 0, msg=result.output)
            data = json.loads(result.output)
            self.assertEqual(data["status"], "not_found")
            self.assertTrue(data["github_output_written"])
            content = Path("gh_out.txt").read_text(encoding="utf-8")
            self.assertIn("openshard_available=false", content)
            self.assertIn("openshard_status=not_found", content)
            self.assertNotIn("openshard_shard_id=", content)

    def test_not_found_human_graceful(self):
        with self.runner.isolated_filesystem():
            env = {
                k: v for k, v in os.environ.items()
                if k not in ("GITHUB_STEP_SUMMARY", "GITHUB_OUTPUT")
            }
            with patch.dict(os.environ, env, clear=True):
                result = self.runner.invoke(
                    cli, ["pr", "comment", "--github-step-summary", "--github-output"]
                )
            self.assertEqual(result.exit_code, 0, msg=result.output)
            self.assertIn("No run history found", result.output)


if __name__ == "__main__":
    unittest.main()
