"""CLI tests for `openshard repo plan` (repo-aware Plan Mode v1)."""
from __future__ import annotations

import json
import unittest
from pathlib import Path
from unittest.mock import patch

from click.testing import CliRunner

import openshard.analysis.repo_map as rm
from openshard.analysis.repo_map import GitInfo
from openshard.cli.main import cli


def _git(branch="main", head="a" * 40, dirty=False, is_git=True) -> GitInfo:
    return GitInfo(branch=branch, head_commit=head, dirty=dirty, is_git=is_git)


def _assert_no_unsafe(test: unittest.TestCase, text: str) -> None:
    for needle in ("C:\\", "C:/", "/Users/", "/home/", "sk-", "AKIA"):
        test.assertNotIn(needle, text, msg=f"unsafe substring {needle!r} leaked: {text}")


class _Base(unittest.TestCase):
    def setUp(self) -> None:
        self.runner = CliRunner()


class TestRepoPlanJson(_Base):
    def test_valid_envelope_and_cache_miss_then_hit(self):
        with self.runner.isolated_filesystem():
            Path("pyproject.toml").write_text("[project]\nname='x'\n", encoding="utf-8")
            Path("test_x.py").write_text("def test_x():\n    assert True\n", encoding="utf-8")
            with patch.object(rm, "collect_git_info", return_value=_git(dirty=False)):
                first = self.runner.invoke(cli, ["repo", "plan", "add tests", "--json"])
                second = self.runner.invoke(cli, ["repo", "plan", "add tests", "--json"])

        self.assertEqual(first.exit_code, 0, msg=first.output)
        d1 = json.loads(first.output)
        self.assertEqual(d1["schema_version"], "1")
        self.assertEqual(d1["command"], "repo plan")
        self.assertEqual(d1["status"], "ok")
        self.assertIsNone(d1["shard_id"])
        self.assertIsInstance(d1["warnings"], list)
        self.assertEqual(d1["task"], "add tests")
        ctx = d1["repo_context"]
        for key in (
            "languages", "frameworks", "package_managers", "test_commands",
            "important_files", "risky_areas", "git_dirty", "cache_hit",
        ):
            self.assertIn(key, ctx)
        self.assertIn("python", ctx["languages"])
        self.assertFalse(ctx["cache_hit"])
        self.assertIsInstance(d1["plan_steps"], list)
        self.assertTrue(d1["plan_steps"])
        self.assertIsInstance(d1["safety_notes"], list)
        _assert_no_unsafe(self, first.output)

        d2 = json.loads(second.output)
        self.assertTrue(d2["repo_context"]["cache_hit"])

    def test_dirty_tree_rebuilds_with_warning(self):
        with self.runner.isolated_filesystem():
            Path("pyproject.toml").write_text("[project]\nname='x'\n", encoding="utf-8")
            with patch.object(rm, "collect_git_info", return_value=_git(dirty=True)):
                first = self.runner.invoke(cli, ["repo", "plan", "t", "--json"])
                second = self.runner.invoke(cli, ["repo", "plan", "t", "--json"])

        d1 = json.loads(first.output)
        d2 = json.loads(second.output)
        self.assertFalse(d1["repo_context"]["cache_hit"])
        self.assertFalse(d2["repo_context"]["cache_hit"])
        self.assertTrue(any("dirty" in w.lower() for w in d1["warnings"]))
        self.assertTrue(d1["repo_context"]["git_dirty"])

    def test_refresh_rebuilds_after_hit(self):
        with self.runner.isolated_filesystem():
            Path("pyproject.toml").write_text("[project]\nname='x'\n", encoding="utf-8")
            with patch.object(rm, "collect_git_info", return_value=_git(dirty=False)):
                self.runner.invoke(cli, ["repo", "plan", "t", "--json"])
                hit = self.runner.invoke(cli, ["repo", "plan", "t", "--json"])
                refreshed = self.runner.invoke(cli, ["repo", "plan", "t", "--json", "--refresh"])

        self.assertTrue(json.loads(hit.output)["repo_context"]["cache_hit"])
        self.assertFalse(json.loads(refreshed.output)["repo_context"]["cache_hit"])

    def test_non_git_repo_does_not_crash(self):
        with self.runner.isolated_filesystem():
            Path("main.py").write_text("x = 1\n", encoding="utf-8")
            with patch.object(rm, "collect_git_info", return_value=_git(is_git=False, branch=None, head=None)):
                result = self.runner.invoke(cli, ["repo", "plan", "t", "--json"])

        self.assertEqual(result.exit_code, 0, msg=result.output)
        d = json.loads(result.output)
        self.assertEqual(d["status"], "ok")
        self.assertTrue(any("not a git repository" in n.lower() for n in d["safety_notes"]))

    def test_task_sanitised_in_json(self):
        with self.runner.isolated_filesystem():
            Path("pyproject.toml").write_text("[project]\nname='x'\n", encoding="utf-8")
            with patch.object(rm, "collect_git_info", return_value=_git(dirty=False)):
                result = self.runner.invoke(
                    cli,
                    ["repo", "plan", r"edit C:\Users\Michael\app.py with sk-deadbeef0123456789", "--json"],
                )

        d = json.loads(result.output)
        _assert_no_unsafe(self, result.output)
        self.assertNotIn("C:\\", d["task"])
        self.assertNotIn("sk-", d["task"])
        self.assertIn("edit", d["task"])


class TestRepoPlanHuman(_Base):
    def test_human_output_shows_sections(self):
        with self.runner.isolated_filesystem():
            Path("pyproject.toml").write_text("[project]\nname='x'\n", encoding="utf-8")
            with patch.object(rm, "collect_git_info", return_value=_git(dirty=False)):
                result = self.runner.invoke(cli, ["repo", "plan", "add tests for auth"])

        self.assertEqual(result.exit_code, 0, msg=result.output)
        out = result.output
        self.assertIn("OpenShard Plan", out)
        self.assertIn("Task:", out)
        self.assertIn("add tests for auth", out)
        self.assertIn("Repo context:", out)
        self.assertIn("Suggested plan:", out)
        self.assertIn("Safety notes:", out)
        self.assertIn("Suggested files to inspect", out)
        _assert_no_unsafe(self, out)

    def test_human_task_sanitised(self):
        with self.runner.isolated_filesystem():
            Path("pyproject.toml").write_text("[project]\nname='x'\n", encoding="utf-8")
            with patch.object(rm, "collect_git_info", return_value=_git(dirty=False)):
                result = self.runner.invoke(cli, ["repo", "plan", r"see /home/michael/.env"])

        _assert_no_unsafe(self, result.output)
        self.assertIn("<path>", result.output)


if __name__ == "__main__":
    unittest.main()
