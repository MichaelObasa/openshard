from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from openshard.cli.main import _run_verification


class TestRunVerificationOutput(unittest.TestCase):

    def _patch(self, cmd, returncode=0, stdout=""):
        detect = patch("openshard.run.pipeline._detect_command", return_value=cmd)
        proc = MagicMock()
        proc.returncode = returncode
        proc.stdout = stdout
        run = patch("subprocess.run", return_value=proc)
        return detect, run

    def test_no_command_prints_not_detected(self):
        detect, _ = self._patch(None)
        with detect, patch("click.echo") as mock_echo:
            result = _run_verification(Path("/tmp"))
        mock_echo.assert_called_once_with("  [verify] no test command detected")
        self.assertEqual(result, 0)

    def test_no_command_capture_is_silent(self):
        detect, _ = self._patch(None)
        with detect, patch("click.echo") as mock_echo:
            result = _run_verification(Path("/tmp"), capture=True)
        mock_echo.assert_not_called()
        self.assertEqual(result, (0, ""))

    def test_command_pass_shows_command_and_status(self):
        detect, run = self._patch(["pytest"], returncode=0)
        with detect, run, patch("click.echo") as mock_echo:
            result = _run_verification(Path("/tmp"))
        calls = [c.args[0] for c in mock_echo.call_args_list]
        self.assertIn("  [verify] running: pytest", calls)
        self.assertIn("  [verify] passed", calls)
        self.assertEqual(result, 0)

    def test_command_fail_shows_command_and_status(self):
        detect, run = self._patch(["pytest"], returncode=1)
        with detect, run, patch("click.echo") as mock_echo:
            result = _run_verification(Path("/tmp"))
        calls = [c.args[0] for c in mock_echo.call_args_list]
        self.assertIn("  [verify] running: pytest", calls)
        self.assertIn("  [verify] failed (exit code 1)", calls)
        self.assertEqual(result, 1)

    def test_custom_label_used_in_output(self):
        detect, run = self._patch(["npm", "test"], returncode=0)
        with detect, run, patch("click.echo") as mock_echo:
            _run_verification(Path("/tmp"), label="[retry/opus]")
        calls = [c.args[0] for c in mock_echo.call_args_list]
        self.assertTrue(any("[retry/opus]" in c for c in calls))

    def test_capture_mode_silent_returns_code_and_output(self):
        detect, run = self._patch(["pytest"], returncode=0, stdout="1 passed")
        with detect, run, patch("click.echo") as mock_echo:
            result = _run_verification(Path("/tmp"), capture=True)
        mock_echo.assert_not_called()
        self.assertEqual(result, (0, "1 passed"))
