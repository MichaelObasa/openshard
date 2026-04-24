from __future__ import annotations

import unittest

import click
from click.testing import CliRunner

from openshard.analysis.repo import RepoFacts
from openshard.cli.main import _render_repo_summary


def _render(facts: RepoFacts) -> str:
    @click.command()
    def cmd():
        _render_repo_summary(facts)

    return CliRunner().invoke(cmd).output


def _make_facts(**kwargs) -> RepoFacts:
    defaults = dict(
        languages=[],
        package_files=[],
        framework=None,
        test_command=None,
        risky_paths=[],
        changed_files=[],
    )
    defaults.update(kwargs)
    return RepoFacts(**defaults)


class TestRenderRepoSummary(unittest.TestCase):

    def test_repo_header_always_shown(self):
        out = _render(_make_facts())
        self.assertIn("Repo", out)

    def test_shows_languages(self):
        out = _render(_make_facts(languages=["javascript", "python"]))
        self.assertIn("Languages: javascript, python", out)

    def test_no_languages_line_when_empty(self):
        out = _render(_make_facts(languages=[]))
        self.assertNotIn("Languages:", out)

    def test_shows_packages(self):
        out = _render(_make_facts(package_files=["pyproject.toml", "requirements.txt"]))
        self.assertIn("Packages: pyproject.toml, requirements.txt", out)

    def test_no_packages_line_when_empty(self):
        out = _render(_make_facts(package_files=[]))
        self.assertNotIn("Packages:", out)

    def test_shows_framework_when_set(self):
        out = _render(_make_facts(framework="flask"))
        self.assertIn("Framework: flask", out)

    def test_no_framework_line_when_none(self):
        out = _render(_make_facts(framework=None))
        self.assertNotIn("Framework:", out)

    def test_shows_test_command_when_set(self):
        out = _render(_make_facts(test_command="python -m pytest"))
        self.assertIn("Tests: python -m pytest", out)

    def test_no_test_line_when_none(self):
        out = _render(_make_facts(test_command=None))
        self.assertNotIn("Tests:", out)

    def test_risky_paths_shows_count_and_sample(self):
        out = _render(_make_facts(risky_paths=["auth.py", "config.py"]))
        self.assertIn("Risky: 2 paths", out)
        self.assertIn("auth.py", out)
        self.assertIn("config.py", out)

    def test_risky_paths_truncates_beyond_three(self):
        paths = ["auth.py", "config.py", "env.py", "security.py"]
        out = _render(_make_facts(risky_paths=paths))
        self.assertIn("Risky: 4 paths", out)
        self.assertIn("+ 1 more", out)
        self.assertNotIn("security.py", out)

    def test_risky_paths_no_truncation_for_exactly_three(self):
        paths = ["auth.py", "config.py", "env.py"]
        out = _render(_make_facts(risky_paths=paths))
        self.assertIn("Risky: 3 paths", out)
        self.assertNotIn("more", out)

    def test_no_risky_line_when_empty(self):
        out = _render(_make_facts(risky_paths=[]))
        self.assertNotIn("Risky:", out)

    def test_changed_files_shows_count_and_sample(self):
        out = _render(_make_facts(changed_files=["main.py", "utils.py"]))
        self.assertIn("Changed: 2 files", out)
        self.assertIn("main.py", out)
        self.assertIn("utils.py", out)

    def test_changed_files_truncates_beyond_three(self):
        files = ["a.py", "b.py", "c.py", "d.py"]
        out = _render(_make_facts(changed_files=files))
        self.assertIn("Changed: 4 files", out)
        self.assertIn("+ 1 more", out)
        self.assertNotIn("d.py", out)

    def test_no_changed_line_when_empty(self):
        out = _render(_make_facts(changed_files=[]))
        self.assertNotIn("Changed:", out)


if __name__ == "__main__":
    unittest.main()
