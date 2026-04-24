from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from openshard.analysis.repo import (
    RepoFacts,
    _detect_changed_files,
    _detect_framework,
    _detect_languages,
    _detect_package_files,
    _detect_risky_paths,
    _detect_test_command,
    analyze_repo,
)


class TestDetectLanguages(unittest.TestCase):

    def test_python_detected(self):
        with tempfile.TemporaryDirectory() as d:
            Path(d, "main.py").touch()
            langs = _detect_languages(Path(d))
        self.assertIn("python", langs)

    def test_multi_language(self):
        with tempfile.TemporaryDirectory() as d:
            Path(d, "app.py").touch()
            Path(d, "index.ts").touch()
            langs = _detect_languages(Path(d))
        self.assertIn("python", langs)
        self.assertIn("typescript", langs)

    def test_unknown_extension_ignored(self):
        with tempfile.TemporaryDirectory() as d:
            Path(d, "readme.txt").touch()
            langs = _detect_languages(Path(d))
        self.assertEqual(langs, [])

    def test_git_dir_ignored(self):
        with tempfile.TemporaryDirectory() as d:
            git_dir = Path(d, ".git")
            git_dir.mkdir()
            Path(git_dir, "hook.py").touch()
            langs = _detect_languages(Path(d))
        self.assertEqual(langs, [])

    def test_result_is_sorted(self):
        with tempfile.TemporaryDirectory() as d:
            Path(d, "app.py").touch()
            Path(d, "app.go").touch()
            langs = _detect_languages(Path(d))
        self.assertEqual(langs, sorted(langs))


class TestDetectPackageFiles(unittest.TestCase):

    def test_requirements_txt(self):
        with tempfile.TemporaryDirectory() as d:
            Path(d, "requirements.txt").touch()
            files = _detect_package_files(Path(d))
        self.assertIn("requirements.txt", files)

    def test_multiple_package_files(self):
        with tempfile.TemporaryDirectory() as d:
            Path(d, "package.json").write_text("{}")
            Path(d, "pyproject.toml").touch()
            files = _detect_package_files(Path(d))
        self.assertIn("package.json", files)
        self.assertIn("pyproject.toml", files)

    def test_no_package_files(self):
        with tempfile.TemporaryDirectory() as d:
            files = _detect_package_files(Path(d))
        self.assertEqual(files, [])

    def test_unknown_file_not_included(self):
        with tempfile.TemporaryDirectory() as d:
            Path(d, "random.txt").touch()
            files = _detect_package_files(Path(d))
        self.assertEqual(files, [])


class TestDetectFramework(unittest.TestCase):

    def test_react_detected(self):
        with tempfile.TemporaryDirectory() as d:
            pkg = {"dependencies": {"react": "^18.0.0", "react-dom": "^18.0.0"}}
            Path(d, "package.json").write_text(json.dumps(pkg))
            fw = _detect_framework(Path(d), ["package.json"])
        self.assertEqual(fw, "react")

    def test_next_detected_before_react(self):
        with tempfile.TemporaryDirectory() as d:
            pkg = {"dependencies": {"next": "^14.0.0", "react": "^18.0.0"}}
            Path(d, "package.json").write_text(json.dumps(pkg))
            fw = _detect_framework(Path(d), ["package.json"])
        self.assertEqual(fw, "next")

    def test_django_detected(self):
        with tempfile.TemporaryDirectory() as d:
            Path(d, "requirements.txt").write_text("django==4.2\npsycopg2==2.9\n")
            fw = _detect_framework(Path(d), ["requirements.txt"])
        self.assertEqual(fw, "django")

    def test_fastapi_detected(self):
        with tempfile.TemporaryDirectory() as d:
            Path(d, "requirements.txt").write_text("fastapi==0.111.0\nuvicorn\n")
            fw = _detect_framework(Path(d), ["requirements.txt"])
        self.assertEqual(fw, "fastapi")

    def test_flask_detected(self):
        with tempfile.TemporaryDirectory() as d:
            Path(d, "requirements.txt").write_text("flask==3.0\n")
            fw = _detect_framework(Path(d), ["requirements.txt"])
        self.assertEqual(fw, "flask")

    def test_no_framework(self):
        with tempfile.TemporaryDirectory() as d:
            fw = _detect_framework(Path(d), [])
        self.assertIsNone(fw)

    def test_no_framework_empty_package_json(self):
        with tempfile.TemporaryDirectory() as d:
            Path(d, "package.json").write_text(json.dumps({"name": "my-app"}))
            fw = _detect_framework(Path(d), ["package.json"])
        self.assertIsNone(fw)


class TestDetectTestCommand(unittest.TestCase):

    def test_npm_test_script(self):
        with tempfile.TemporaryDirectory() as d:
            pkg = {"scripts": {"test": "jest"}}
            Path(d, "package.json").write_text(json.dumps(pkg))
            cmd = _detect_test_command(Path(d), ["package.json"])
        self.assertEqual(cmd, "npm test")

    def test_no_npm_test_when_script_absent(self):
        with tempfile.TemporaryDirectory() as d:
            pkg = {"scripts": {"build": "webpack"}}
            Path(d, "package.json").write_text(json.dumps(pkg))
            cmd = _detect_test_command(Path(d), ["package.json"])
        self.assertIsNone(cmd)

    def test_pytest_from_pyproject(self):
        with tempfile.TemporaryDirectory() as d:
            Path(d, "pyproject.toml").write_text("[tool.pytest.ini_options]\n")
            cmd = _detect_test_command(Path(d), ["pyproject.toml"])
        self.assertEqual(cmd, "python -m pytest")

    def test_pytest_from_requirements(self):
        with tempfile.TemporaryDirectory() as d:
            Path(d, "requirements.txt").write_text("pytest==8.0\nrequests\n")
            cmd = _detect_test_command(Path(d), ["requirements.txt"])
        self.assertEqual(cmd, "python -m pytest")

    def test_pytest_fallback_from_test_files(self):
        with tempfile.TemporaryDirectory() as d:
            Path(d, "requirements.txt").write_text("requests\n")
            Path(d, "test_app.py").touch()
            cmd = _detect_test_command(Path(d), ["requirements.txt"])
        self.assertEqual(cmd, "python -m pytest")

    def test_cargo_test(self):
        with tempfile.TemporaryDirectory() as d:
            Path(d, "Cargo.toml").write_text('[package]\nname = "foo"\n')
            cmd = _detect_test_command(Path(d), ["Cargo.toml"])
        self.assertEqual(cmd, "cargo test")

    def test_go_test(self):
        with tempfile.TemporaryDirectory() as d:
            Path(d, "go.mod").write_text("module example.com/foo\n")
            cmd = _detect_test_command(Path(d), ["go.mod"])
        self.assertEqual(cmd, "go test ./...")

    def test_no_test_command(self):
        with tempfile.TemporaryDirectory() as d:
            cmd = _detect_test_command(Path(d), [])
        self.assertIsNone(cmd)


class TestDetectRiskyPaths(unittest.TestCase):

    def test_auth_file_flagged(self):
        with tempfile.TemporaryDirectory() as d:
            Path(d, "auth.py").touch()
            risky = _detect_risky_paths(Path(d))
        self.assertTrue(any("auth" in r for r in risky))

    def test_env_file_flagged(self):
        with tempfile.TemporaryDirectory() as d:
            Path(d, ".env").touch()
            risky = _detect_risky_paths(Path(d))
        self.assertTrue(any(".env" in r for r in risky))

    def test_migrations_folder_flagged(self):
        with tempfile.TemporaryDirectory() as d:
            Path(d, "migrations").mkdir()
            risky = _detect_risky_paths(Path(d))
        self.assertTrue(any("migration" in r for r in risky))

    def test_payments_file_flagged(self):
        with tempfile.TemporaryDirectory() as d:
            Path(d, "payments.py").touch()
            risky = _detect_risky_paths(Path(d))
        self.assertTrue(any("payment" in r for r in risky))

    def test_security_dir_flagged(self):
        with tempfile.TemporaryDirectory() as d:
            Path(d, "security").mkdir()
            risky = _detect_risky_paths(Path(d))
        self.assertTrue(any("security" in r for r in risky))

    def test_safe_file_not_flagged(self):
        with tempfile.TemporaryDirectory() as d:
            Path(d, "main.py").touch()
            risky = _detect_risky_paths(Path(d))
        self.assertFalse(any("main" in r for r in risky))

    def test_no_risky_paths_for_clean_repo(self):
        with tempfile.TemporaryDirectory() as d:
            Path(d, "main.py").touch()
            Path(d, "utils.py").touch()
            risky = _detect_risky_paths(Path(d))
        self.assertEqual(risky, [])

    def test_venv_dir_not_flagged(self):
        with tempfile.TemporaryDirectory() as d:
            venv_dir = Path(d, "venv")
            venv_dir.mkdir()
            risky = _detect_risky_paths(Path(d))
        self.assertFalse(any("venv" in r for r in risky))


class TestDetectChangedFiles(unittest.TestCase):

    def test_returns_changed_files(self):
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "foo.py\nbar.py\n"
        with patch("openshard.analysis.repo.subprocess.run", return_value=mock_result):
            files = _detect_changed_files(Path("."))
        self.assertEqual(files, ["foo.py", "bar.py"])

    def test_returns_empty_on_nonzero_exit(self):
        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stdout = ""
        with patch("openshard.analysis.repo.subprocess.run", return_value=mock_result):
            files = _detect_changed_files(Path("."))
        self.assertEqual(files, [])

    def test_returns_empty_on_exception(self):
        with patch("openshard.analysis.repo.subprocess.run", side_effect=FileNotFoundError):
            files = _detect_changed_files(Path("."))
        self.assertEqual(files, [])

    def test_filters_empty_lines(self):
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "foo.py\n\nbar.py\n"
        with patch("openshard.analysis.repo.subprocess.run", return_value=mock_result):
            files = _detect_changed_files(Path("."))
        self.assertEqual(files, ["foo.py", "bar.py"])


class TestAnalyzeRepo(unittest.TestCase):

    def test_returns_repo_facts(self):
        mock_git = MagicMock()
        mock_git.returncode = 1
        mock_git.stdout = ""
        with tempfile.TemporaryDirectory() as d:
            Path(d, "main.py").touch()
            Path(d, "requirements.txt").write_text("flask\n")
            Path(d, "auth.py").touch()
            with patch("openshard.analysis.repo.subprocess.run", return_value=mock_git):
                facts = analyze_repo(d)
        self.assertIsInstance(facts, RepoFacts)
        self.assertIn("python", facts.languages)
        self.assertIn("requirements.txt", facts.package_files)
        self.assertEqual(facts.framework, "flask")
        self.assertTrue(any("auth" in r for r in facts.risky_paths))
        self.assertEqual(facts.changed_files, [])

    def test_accepts_path_object(self):
        mock_git = MagicMock()
        mock_git.returncode = 1
        mock_git.stdout = ""
        with tempfile.TemporaryDirectory() as d:
            with patch("openshard.analysis.repo.subprocess.run", return_value=mock_git):
                facts = analyze_repo(Path(d))
        self.assertIsInstance(facts, RepoFacts)
