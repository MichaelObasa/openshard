from __future__ import annotations

import unittest

from openshard.analysis.repo import RepoFacts
from openshard.execution.generator import (
    ChangedFile,
    _build_repo_context,
    check_stack_mismatch,
)


def _facts(**kwargs) -> RepoFacts:
    defaults = dict(
        languages=["python"],
        package_files=["pyproject.toml"],
        framework=None,
        test_command="python -m pytest",
        risky_paths=[],
        changed_files=[],
    )
    defaults.update(kwargs)
    return RepoFacts(**defaults)


def _file(path: str) -> ChangedFile:
    return ChangedFile(path=path, change_type="create", content="", summary="")


class TestCheckStackMismatch(unittest.TestCase):

    def test_python_files_allowed_in_python_repo(self):
        files = [_file("src/utils.py"), _file("tests/test_utils.py")]
        self.assertEqual(check_stack_mismatch(files, _facts()), [])

    def test_typescript_rejected_in_python_repo(self):
        files = [_file("src/index.ts")]
        result = check_stack_mismatch(files, _facts())
        self.assertIn("src/index.ts", result)

    def test_tsx_rejected_in_python_repo(self):
        files = [_file("components/Button.tsx")]
        result = check_stack_mismatch(files, _facts())
        self.assertIn("components/Button.tsx", result)

    def test_javascript_rejected_in_python_repo(self):
        files = [_file("app.js")]
        result = check_stack_mismatch(files, _facts())
        self.assertIn("app.js", result)

    def test_markdown_docs_always_allowed(self):
        files = [_file("README.md"), _file("docs/guide.md")]
        self.assertEqual(check_stack_mismatch(files, _facts()), [])

    def test_rst_docs_allowed(self):
        self.assertEqual(check_stack_mismatch([_file("docs/index.rst")], _facts()), [])

    def test_yaml_config_allowed(self):
        files = [_file(".github/workflows/ci.yml"), _file("config.yaml")]
        self.assertEqual(check_stack_mismatch(files, _facts()), [])

    def test_json_config_allowed(self):
        self.assertEqual(check_stack_mismatch([_file("schema.json")], _facts()), [])

    def test_toml_config_allowed(self):
        self.assertEqual(check_stack_mismatch([_file("pyproject.toml")], _facts()), [])

    def test_shell_script_allowed(self):
        self.assertEqual(check_stack_mismatch([_file("scripts/build.sh")], _facts()), [])

    def test_html_allowed(self):
        self.assertEqual(check_stack_mismatch([_file("templates/index.html")], _facts()), [])

    def test_css_allowed(self):
        self.assertEqual(check_stack_mismatch([_file("static/style.css")], _facts()), [])

    def test_no_extension_allowed(self):
        self.assertEqual(check_stack_mismatch([_file("Makefile"), _file("Dockerfile")], _facts()), [])

    def test_empty_languages_allows_everything(self):
        facts = _facts(languages=[])
        files = [_file("src/index.ts"), _file("app.js")]
        self.assertEqual(check_stack_mismatch(files, facts), [])

    def test_empty_files_returns_empty(self):
        self.assertEqual(check_stack_mismatch([], _facts()), [])

    def test_mixed_files_only_mismatches_returned(self):
        files = [_file("main.py"), _file("README.md"), _file("src/index.ts")]
        result = check_stack_mismatch(files, _facts())
        self.assertNotIn("main.py", result)
        self.assertNotIn("README.md", result)
        self.assertIn("src/index.ts", result)

    def test_typescript_repo_allows_ts_files(self):
        facts = _facts(languages=["typescript"])
        files = [_file("src/app.ts"), _file("components/Nav.tsx")]
        self.assertEqual(check_stack_mismatch(files, facts), [])

    def test_typescript_repo_rejects_python(self):
        facts = _facts(languages=["typescript"])
        files = [_file("tests/test_app.py")]
        result = check_stack_mismatch(files, facts)
        self.assertIn("tests/test_app.py", result)

    def test_multi_language_repo_allows_both(self):
        facts = _facts(languages=["python", "javascript"])
        files = [_file("backend/app.py"), _file("frontend/app.js")]
        self.assertEqual(check_stack_mismatch(files, facts), [])

    def test_rust_file_rejected_in_python_repo(self):
        files = [_file("src/main.rs")]
        result = check_stack_mismatch(files, _facts())
        self.assertIn("src/main.rs", result)


class TestBuildRepoContext(unittest.TestCase):

    def test_includes_languages(self):
        ctx = _build_repo_context(_facts(languages=["python"]))
        self.assertIn("python", ctx)

    def test_includes_test_command(self):
        ctx = _build_repo_context(_facts(test_command="python -m pytest"))
        self.assertIn("python -m pytest", ctx)

    def test_includes_framework(self):
        ctx = _build_repo_context(_facts(framework="django"))
        self.assertIn("django", ctx)

    def test_includes_package_files(self):
        ctx = _build_repo_context(_facts(package_files=["pyproject.toml", "requirements.txt"]))
        self.assertIn("pyproject.toml", ctx)

    def test_empty_facts_returns_empty_string(self):
        facts = _facts(languages=[], package_files=[], framework=None, test_command=None)
        self.assertEqual(_build_repo_context(facts), "")

    def test_context_contains_stack_instruction(self):
        ctx = _build_repo_context(_facts())
        self.assertIn("match this repo", ctx.lower())


if __name__ == "__main__":
    unittest.main()
