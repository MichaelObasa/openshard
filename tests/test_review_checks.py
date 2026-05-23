from __future__ import annotations

import subprocess
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch


def _run(path: Path | None = None) -> list[dict]:
    from openshard.review.checks import run_review_checks
    return run_review_checks(path or Path("/fake/repo"))


def _proc(returncode: int = 0, stdout: str = "", stderr: str = "") -> MagicMock:
    p = MagicMock()
    p.returncode = returncode
    p.stdout = stdout
    p.stderr = stderr
    return p


class TestNoTfFiles(unittest.TestCase):

    def test_empty_when_no_tf_files(self):
        with patch("openshard.review.checks.Path.glob", return_value=iter([])):
            result = _run()
        self.assertEqual(result, [])


class TestTerraformNotInstalled(unittest.TestCase):

    def setUp(self):
        patcher_glob = patch("openshard.review.checks.Path.glob", return_value=iter([Path("main.tf")]))
        patcher_which = patch("openshard.review.checks.shutil.which", return_value=None)
        self.mock_glob = patcher_glob.start()
        self.mock_which = patcher_which.start()
        self.addCleanup(patcher_glob.stop)
        self.addCleanup(patcher_which.stop)

    def _checks(self) -> list[dict]:
        from openshard.review.checks import run_review_checks
        return run_review_checks(Path("/fake/repo"))

    def test_fmt_skipped(self):
        checks = self._checks()
        fmt = next(c for c in checks if c["name"] == "terraform fmt")
        self.assertEqual(fmt["status"], "skipped")
        self.assertEqual(fmt["reason"], "terraform not installed")

    def test_validate_skipped(self):
        checks = self._checks()
        val = next(c for c in checks if c["name"] == "terraform validate")
        self.assertEqual(val["status"], "skipped")
        self.assertEqual(val["reason"], "terraform not installed")

    def test_tflint_skipped(self):
        checks = self._checks()
        tflint = next(c for c in checks if c["name"] == "tflint")
        self.assertEqual(tflint["status"], "skipped")
        self.assertEqual(tflint["reason"], "tflint not installed")

    def test_no_huge_output_stored(self):
        checks = self._checks()
        for c in checks:
            self.assertNotIn("stdout", c)
            self.assertNotIn("stderr", c)


class TestTerraformFmtPassed(unittest.TestCase):

    def test_fmt_passed(self):
        with (
            patch("openshard.review.checks.Path.glob", return_value=iter([Path("main.tf")])),
            patch("openshard.review.checks.shutil.which", side_effect=lambda x: "/usr/bin/terraform" if x == "terraform" else None),
            patch("openshard.review.checks.subprocess.run", return_value=_proc(0, "", "")),
            patch("openshard.review.checks.Path.is_dir", return_value=False),
        ):
            from openshard.review.checks import run_review_checks
            checks = run_review_checks(Path("/fake/repo"))
        fmt = next(c for c in checks if c["name"] == "terraform fmt")
        self.assertEqual(fmt["status"], "passed")
        self.assertEqual(fmt["summary"], "formatting is clean")
        self.assertEqual(fmt["returncode"], 0)


class TestTerraformFmtFailed(unittest.TestCase):

    def test_fmt_failed_with_short_summary(self):
        with (
            patch("openshard.review.checks.Path.glob", return_value=iter([Path("main.tf")])),
            patch("openshard.review.checks.shutil.which", side_effect=lambda x: "/usr/bin/terraform" if x == "terraform" else None),
            patch("openshard.review.checks.subprocess.run", return_value=_proc(1, "main.tf\n", "")),
            patch("openshard.review.checks.Path.is_dir", return_value=False),
        ):
            from openshard.review.checks import run_review_checks
            checks = run_review_checks(Path("/fake/repo"))
        fmt = next(c for c in checks if c["name"] == "terraform fmt")
        self.assertEqual(fmt["status"], "failed")
        self.assertEqual(fmt["summary"], "main.tf")
        self.assertEqual(fmt["returncode"], 1)

    def test_fmt_summary_capped_at_120_chars(self):
        long_line = "X" * 200
        with (
            patch("openshard.review.checks.Path.glob", return_value=iter([Path("main.tf")])),
            patch("openshard.review.checks.shutil.which", side_effect=lambda x: "/usr/bin/terraform" if x == "terraform" else None),
            patch("openshard.review.checks.subprocess.run", return_value=_proc(1, long_line, "")),
            patch("openshard.review.checks.Path.is_dir", return_value=False),
        ):
            from openshard.review.checks import run_review_checks
            checks = run_review_checks(Path("/fake/repo"))
        fmt = next(c for c in checks if c["name"] == "terraform fmt")
        self.assertLessEqual(len(fmt["summary"]), 120)


class TestTerraformValidate(unittest.TestCase):

    def test_validate_skipped_no_terraform_dir(self):
        with (
            patch("openshard.review.checks.Path.glob", return_value=iter([Path("main.tf")])),
            patch("openshard.review.checks.shutil.which", side_effect=lambda x: "/usr/bin/terraform" if x == "terraform" else None),
            patch("openshard.review.checks.subprocess.run", return_value=_proc(0, "", "")),
            patch("openshard.review.checks.Path.is_dir", return_value=False),
        ):
            from openshard.review.checks import run_review_checks
            checks = run_review_checks(Path("/fake/repo"))
        val = next(c for c in checks if c["name"] == "terraform validate")
        self.assertEqual(val["status"], "skipped")
        self.assertEqual(val["reason"], "terraform init required")

    def test_validate_runs_when_terraform_dir_exists(self):
        with (
            patch("openshard.review.checks.Path.glob", return_value=iter([Path("main.tf")])),
            patch("openshard.review.checks.shutil.which", side_effect=lambda x: "/usr/bin/terraform" if x == "terraform" else None),
            patch("openshard.review.checks.subprocess.run", return_value=_proc(0, "Success!", "")),
            patch("openshard.review.checks.Path.is_dir", return_value=True),
        ):
            from openshard.review.checks import run_review_checks
            checks = run_review_checks(Path("/fake/repo"))
        val = next(c for c in checks if c["name"] == "terraform validate")
        self.assertEqual(val["status"], "passed")
        self.assertEqual(val["summary"], "Success!")


class TestTflint(unittest.TestCase):

    def test_tflint_skipped_when_not_installed(self):
        with (
            patch("openshard.review.checks.Path.glob", return_value=iter([Path("main.tf")])),
            patch("openshard.review.checks.shutil.which", return_value=None),
        ):
            from openshard.review.checks import run_review_checks
            checks = run_review_checks(Path("/fake/repo"))
        tflint = next(c for c in checks if c["name"] == "tflint")
        self.assertEqual(tflint["status"], "skipped")
        self.assertEqual(tflint["reason"], "tflint not installed")

    def test_tflint_passed_when_installed(self):
        with (
            patch("openshard.review.checks.Path.glob", return_value=iter([Path("main.tf")])),
            patch("openshard.review.checks.shutil.which", side_effect=lambda x: "/usr/bin/tflint" if x == "tflint" else None),
            patch("openshard.review.checks.subprocess.run", return_value=_proc(0, "", "")),
            patch("openshard.review.checks.Path.is_dir", return_value=False),
        ):
            from openshard.review.checks import run_review_checks
            checks = run_review_checks(Path("/fake/repo"))
        tflint = next(c for c in checks if c["name"] == "tflint")
        self.assertEqual(tflint["status"], "passed")
        self.assertEqual(tflint["summary"], "no issues found")


class TestSubprocessError(unittest.TestCase):

    def test_subprocess_error_yields_skipped(self):
        with (
            patch("openshard.review.checks.Path.glob", return_value=iter([Path("main.tf")])),
            patch("openshard.review.checks.shutil.which", side_effect=lambda x: "/usr/bin/terraform" if x == "terraform" else None),
            patch("openshard.review.checks.subprocess.run", side_effect=subprocess.TimeoutExpired(cmd="terraform", timeout=30)),
            patch("openshard.review.checks.Path.is_dir", return_value=False),
        ):
            from openshard.review.checks import run_review_checks
            checks = run_review_checks(Path("/fake/repo"))
        fmt = next(c for c in checks if c["name"] == "terraform fmt")
        self.assertEqual(fmt["status"], "skipped")
        self.assertEqual(fmt["reason"], "check errored")


class TestSummaryFallback(unittest.TestCase):

    def test_uses_stderr_when_stdout_empty(self):
        with (
            patch("openshard.review.checks.Path.glob", return_value=iter([Path("main.tf")])),
            patch("openshard.review.checks.shutil.which", side_effect=lambda x: "/usr/bin/terraform" if x == "terraform" else None),
            patch("openshard.review.checks.subprocess.run", return_value=_proc(1, "", "Error from stderr")),
            patch("openshard.review.checks.Path.is_dir", return_value=False),
        ):
            from openshard.review.checks import run_review_checks
            checks = run_review_checks(Path("/fake/repo"))
        fmt = next(c for c in checks if c["name"] == "terraform fmt")
        self.assertEqual(fmt["status"], "failed")
        self.assertEqual(fmt["summary"], "Error from stderr")

    def test_uses_fallback_when_both_empty(self):
        with (
            patch("openshard.review.checks.Path.glob", return_value=iter([Path("main.tf")])),
            patch("openshard.review.checks.shutil.which", side_effect=lambda x: "/usr/bin/terraform" if x == "terraform" else None),
            patch("openshard.review.checks.subprocess.run", return_value=_proc(1, "", "")),
            patch("openshard.review.checks.Path.is_dir", return_value=False),
        ):
            from openshard.review.checks import run_review_checks
            checks = run_review_checks(Path("/fake/repo"))
        fmt = next(c for c in checks if c["name"] == "terraform fmt")
        self.assertEqual(fmt["status"], "failed")
        self.assertNotEqual(fmt["summary"], "")


class TestNoFilesModified(unittest.TestCase):

    def test_run_commands_are_read_only(self):
        from openshard.review.checks import run_review_checks
        calls: list[list[str]] = []

        def capture_run(args, **kwargs):
            calls.append(args)
            return _proc(0, "", "")

        with (
            patch("openshard.review.checks.Path.glob", return_value=iter([Path("main.tf")])),
            patch("openshard.review.checks.shutil.which", side_effect=lambda x: f"/usr/bin/{x}"),
            patch("openshard.review.checks.subprocess.run", side_effect=capture_run),
            patch("openshard.review.checks.Path.is_dir", return_value=False),
        ):
            run_review_checks(Path("/fake/repo"))

        forbidden = {"init", "apply", "plan", "destroy", "get", "install"}
        for call_args in calls:
            for arg in call_args:
                self.assertNotIn(arg.lower(), forbidden, f"Forbidden command arg: {arg}")
