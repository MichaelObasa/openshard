from __future__ import annotations

import json
import unittest
from pathlib import Path
from unittest.mock import patch

from click.testing import CliRunner

from openshard.analysis import repo_map as rm
from openshard.analysis.repo_map import GitInfo
from openshard.cli.main import cli


def _git(dirty: bool = False, is_git: bool = True) -> GitInfo:
    return GitInfo(branch="main", head_commit="a" * 40, dirty=dirty, is_git=is_git)


def _assert_no_unsafe(test: unittest.TestCase, text: str) -> None:
    for needle in ("C:\\", "C:/", "/Users/", "/home/", "sk-", "AKIA"):
        test.assertNotIn(needle, text, msg=f"unsafe substring {needle!r} leaked: {text}")


class _Base(unittest.TestCase):
    def setUp(self) -> None:
        self.runner = CliRunner()


class TestRepoMapJson(_Base):

    def test_valid_envelope_and_cache_miss_then_hit(self):
        with self.runner.isolated_filesystem():
            Path("main.py").write_text("x = 1\n", encoding="utf-8")
            with patch.object(rm, "collect_git_info", return_value=_git(dirty=False)):
                first = self.runner.invoke(cli, ["repo", "map", "--json"])
                second = self.runner.invoke(cli, ["repo", "map", "--json"])

        self.assertEqual(first.exit_code, 0, msg=first.output)
        d1 = json.loads(first.output)
        self.assertEqual(d1["schema_version"], "1")
        self.assertEqual(d1["command"], "repo map")
        self.assertEqual(d1["status"], "ok")
        self.assertFalse(d1["cache_hit"])
        self.assertTrue(d1["cache_path_display"].startswith(".openshard/cache/repo-"))
        self.assertIsInstance(d1["repo_map"], dict)
        self.assertEqual(d1["repo_map"]["source"], "repo_map_v1")
        _assert_no_unsafe(self, first.output)

        d2 = json.loads(second.output)
        self.assertTrue(d2["cache_hit"])

    def test_dirty_tree_always_rebuilds(self):
        with self.runner.isolated_filesystem():
            Path("main.py").write_text("x = 1\n", encoding="utf-8")
            with patch.object(rm, "collect_git_info", return_value=_git(dirty=True)):
                first = self.runner.invoke(cli, ["repo", "map", "--json"])
                second = self.runner.invoke(cli, ["repo", "map", "--json"])

        d1 = json.loads(first.output)
        d2 = json.loads(second.output)
        self.assertFalse(d1["cache_hit"])
        self.assertFalse(d2["cache_hit"])
        self.assertTrue(any("dirty git tree" in w for w in d2["warnings"]))

    def test_refresh_rebuilds_after_hit(self):
        with self.runner.isolated_filesystem():
            Path("main.py").write_text("x = 1\n", encoding="utf-8")
            with patch.object(rm, "collect_git_info", return_value=_git(dirty=False)):
                self.runner.invoke(cli, ["repo", "map", "--json"])
                hit = self.runner.invoke(cli, ["repo", "map", "--json"])
                refreshed = self.runner.invoke(cli, ["repo", "map", "--json", "--refresh"])

        self.assertTrue(json.loads(hit.output)["cache_hit"])
        self.assertFalse(json.loads(refreshed.output)["cache_hit"])

    def test_non_git_repo_does_not_crash(self):
        with self.runner.isolated_filesystem():
            Path("main.py").write_text("x = 1\n", encoding="utf-8")
            with patch.object(rm, "collect_git_info", return_value=_git(is_git=False)):
                result = self.runner.invoke(cli, ["repo", "map", "--json"])
        self.assertEqual(result.exit_code, 0, msg=result.output)
        data = json.loads(result.output)
        self.assertEqual(data["status"], "ok")
        self.assertTrue(any("not a git repository" in w for w in data["warnings"]))

    def test_output_writes_valid_json_file(self):
        with self.runner.isolated_filesystem():
            Path("main.py").write_text("x = 1\n", encoding="utf-8")
            with patch.object(rm, "collect_git_info", return_value=_git(dirty=False)):
                result = self.runner.invoke(
                    cli, ["repo", "map", "--json", "--output", "out.json"]
                )
            written = json.loads(Path("out.json").read_text(encoding="utf-8"))
        self.assertEqual(result.exit_code, 0, msg=result.output)
        self.assertEqual(written["source"], "repo_map_v1")
        self.assertEqual(json.loads(result.output)["output_path_display"], "out.json")


class TestRepoMapHuman(_Base):

    def test_human_output_shows_stack_and_cache(self):
        with self.runner.isolated_filesystem():
            Path("main.py").write_text("x = 1\n", encoding="utf-8")
            Path("pyproject.toml").write_text("pytest\n", encoding="utf-8")
            with patch.object(rm, "collect_git_info", return_value=_git(dirty=False)):
                result = self.runner.invoke(cli, ["repo", "map"])
        self.assertEqual(result.exit_code, 0, msg=result.output)
        self.assertIn("Repo map", result.output)
        self.assertIn("python", result.output)
        self.assertIn(".openshard/cache/repo-", result.output)
        _assert_no_unsafe(self, result.output)


if __name__ == "__main__":
    unittest.main()
