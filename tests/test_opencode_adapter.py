from __future__ import annotations

import json
import subprocess
import unittest
from dataclasses import asdict
from pathlib import Path
from unittest.mock import MagicMock, patch

from openshard.execution.opencode_adapter import (
    OpenCodeAvailability,
    build_opencode_command,
    detect_opencode,
    get_opencode_install_guidance,
    run_opencode_task,
)


class TestDetectOpencode(unittest.TestCase):
    def test_unavailable_when_not_on_path(self):
        with patch("shutil.which", return_value=None):
            with patch("openshard.execution.opencode_adapter.sys") as mock_sys:
                mock_sys.platform = "linux"
                result = detect_opencode()
        self.assertFalse(result.available)
        self.assertIsNone(result.path)

    def test_available_when_on_path(self):
        with patch("shutil.which", return_value="/usr/local/bin/opencode"):
            result = detect_opencode()
        self.assertTrue(result.available)
        self.assertEqual(result.path, "/usr/local/bin/opencode")

    def test_unavailable_includes_guidance(self):
        with patch("shutil.which", return_value=None):
            with patch("openshard.execution.opencode_adapter.sys") as mock_sys:
                mock_sys.platform = "linux"
                result = detect_opencode()
        self.assertGreater(len(result.install_guidance), 0)

    def test_available_has_no_reason(self):
        with patch("shutil.which", return_value="/usr/bin/opencode"):
            result = detect_opencode()
        self.assertIsNone(result.reason)

    def test_unavailable_has_reason(self):
        with patch("shutil.which", return_value=None):
            with patch("openshard.execution.opencode_adapter.sys") as mock_sys:
                mock_sys.platform = "linux"
                result = detect_opencode()
        self.assertIsNotNone(result.reason)
        self.assertIn("PATH", result.reason)

    def test_never_raises_when_missing(self):
        with patch("shutil.which", return_value=None):
            with patch("openshard.execution.opencode_adapter.sys") as mock_sys:
                mock_sys.platform = "linux"
                result = detect_opencode()
        self.assertIsInstance(result, OpenCodeAvailability)


class TestInstallGuidance(unittest.TestCase):
    def test_includes_npm_option(self):
        guidance = get_opencode_install_guidance()
        self.assertTrue(any("npm" in g for g in guidance))

    def test_includes_windows_option(self):
        guidance = get_opencode_install_guidance(platform="win32")
        self.assertTrue(any("scoop" in g or "choco" in g for g in guidance))

    def test_windows_platform_returns_list(self):
        guidance = get_opencode_install_guidance(platform="win32")
        self.assertIsInstance(guidance, list)
        self.assertGreater(len(guidance), 0)

    def test_linux_platform_includes_curl(self):
        guidance = get_opencode_install_guidance(platform="linux")
        self.assertTrue(any("curl" in g for g in guidance))

    def test_darwin_platform_includes_brew(self):
        guidance = get_opencode_install_guidance(platform="darwin")
        self.assertTrue(any("brew" in g for g in guidance))


class TestAdapterDoctor(unittest.TestCase):
    def test_output_says_detected_when_available(self):
        from click.testing import CliRunner
        from openshard.cli.main import cli

        with patch("openshard.execution.opencode_adapter.shutil.which", return_value="/usr/bin/opencode"):
            runner = CliRunner()
            result = runner.invoke(cli, ["adapters", "doctor"])
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertIn("detected", result.output.lower())

    def test_output_says_not_installed_when_missing(self):
        from click.testing import CliRunner
        from openshard.cli.main import cli

        with patch("openshard.execution.opencode_adapter.shutil.which", return_value=None):
            with patch("openshard.execution.opencode_adapter.sys") as mock_sys:
                mock_sys.platform = "linux"
                runner = CliRunner()
                result = runner.invoke(cli, ["adapters", "doctor"])
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertIn("not installed", result.output.lower())

    def test_output_includes_install_options_when_missing(self):
        from click.testing import CliRunner
        from openshard.cli.main import cli

        with patch("openshard.execution.opencode_adapter.shutil.which", return_value=None):
            with patch("openshard.execution.opencode_adapter.sys") as mock_sys:
                mock_sys.platform = "linux"
                runner = CliRunner()
                result = runner.invoke(cli, ["adapters", "doctor"])
        self.assertIn("npm", result.output)

    def test_doctor_never_calls_subprocess(self):
        from click.testing import CliRunner
        from openshard.cli.main import cli

        with patch("subprocess.run") as mock_sub:
            with patch("openshard.execution.opencode_adapter.shutil.which", return_value=None):
                with patch("openshard.execution.opencode_adapter.sys") as mock_sys:
                    mock_sys.platform = "linux"
                    runner = CliRunner()
                    runner.invoke(cli, ["adapters", "doctor"])
        mock_sub.assert_not_called()

    def test_output_shows_path_when_detected(self):
        from click.testing import CliRunner
        from openshard.cli.main import cli

        with patch("openshard.execution.opencode_adapter.shutil.which", return_value="/usr/local/bin/opencode"):
            runner = CliRunner()
            result = runner.invoke(cli, ["adapters", "doctor"])
        self.assertIn("/usr/local/bin/opencode", result.output)


class TestBuildOpenCodeCommand(unittest.TestCase):
    def test_returns_list(self):
        cmd = build_opencode_command("fix auth tests", Path("/repo"))
        self.assertIsInstance(cmd, list)

    def test_task_is_single_argument(self):
        task = "fix auth tests and update the docs"
        cmd = build_opencode_command(task, Path("/repo"))
        self.assertIn(task, cmd)
        # task must appear as one element, not split across multiple
        self.assertEqual(cmd.count(task), 1)

    def test_starts_with_opencode_run(self):
        cmd = build_opencode_command("any task", Path("/repo"))
        self.assertEqual(cmd[0], "opencode")
        self.assertEqual(cmd[1], "run")

    def test_not_a_shell_string(self):
        cmd = build_opencode_command("fix tests", Path("/repo"))
        self.assertNotIsInstance(cmd, str)


class TestRunOpenCodeTask(unittest.TestCase):
    def _make_proc(self, returncode=0, stdout="done", stderr=""):
        mock = MagicMock()
        mock.returncode = returncode
        mock.stdout = stdout
        mock.stderr = stderr
        return mock

    def test_captures_exit_code(self):
        with patch("shutil.which", return_value="/usr/bin/opencode"):
            with patch("subprocess.run", return_value=self._make_proc(returncode=0)):
                result = run_opencode_task("fix tests", Path("/repo"))
        self.assertEqual(result.exit_code, 0)

    def test_captures_nonzero_exit_code(self):
        with patch("shutil.which", return_value="/usr/bin/opencode"):
            with patch("subprocess.run", return_value=self._make_proc(returncode=1, stderr="err")):
                result = run_opencode_task("fix tests", Path("/repo"))
        self.assertEqual(result.exit_code, 1)

    def test_truncates_stdout(self):
        long_output = "x" * 5000
        with patch("shutil.which", return_value="/usr/bin/opencode"):
            with patch("subprocess.run", return_value=self._make_proc(stdout=long_output)):
                result = run_opencode_task("fix tests", Path("/repo"))
        self.assertLessEqual(len(result.stdout_summary), 1000)

    def test_truncates_stderr(self):
        long_err = "e" * 5000
        with patch("shutil.which", return_value="/usr/bin/opencode"):
            with patch("subprocess.run", return_value=self._make_proc(returncode=1, stderr=long_err)):
                result = run_opencode_task("fix tests", Path("/repo"))
        self.assertLessEqual(len(result.stderr_summary), 1000)

    def test_handles_timeout_safely(self):
        with patch("shutil.which", return_value="/usr/bin/opencode"):
            with patch("subprocess.run", side_effect=subprocess.TimeoutExpired("opencode", 300)):
                result = run_opencode_task("fix tests", Path("/repo"), timeout_s=300)
        self.assertEqual(result.exit_code, -1)
        self.assertIn("Timed out", result.stderr_summary)

    def test_unavailable_returns_127_without_raising(self):
        with patch("shutil.which", return_value=None):
            with patch("openshard.execution.opencode_adapter.sys") as mock_sys:
                mock_sys.platform = "linux"
                result = run_opencode_task("fix tests", Path("/repo"))
        self.assertEqual(result.exit_code, 127)

    def test_unavailable_does_not_call_subprocess(self):
        with patch("shutil.which", return_value=None):
            with patch("openshard.execution.opencode_adapter.sys") as mock_sys:
                mock_sys.platform = "linux"
                with patch("subprocess.run") as mock_sub:
                    run_opencode_task("fix tests", Path("/repo"))
        mock_sub.assert_not_called()

    def test_result_is_json_safe(self):
        with patch("shutil.which", return_value="/usr/bin/opencode"):
            with patch("subprocess.run", return_value=self._make_proc(stdout="done")):
                result = run_opencode_task("fix tests", Path("/repo"))
        data = asdict(result)
        serialized = json.dumps(data)
        self.assertIn("exit_code", serialized)

    def test_command_field_is_list(self):
        with patch("shutil.which", return_value="/usr/bin/opencode"):
            with patch("subprocess.run", return_value=self._make_proc()):
                result = run_opencode_task("fix tests", Path("/repo"))
        self.assertIsInstance(result.command, list)

    def test_executor_field_is_opencode(self):
        with patch("shutil.which", return_value="/usr/bin/opencode"):
            with patch("subprocess.run", return_value=self._make_proc()):
                result = run_opencode_task("fix tests", Path("/repo"))
        self.assertEqual(result.executor, "OpenCode")


class TestShardReceiptLabel(unittest.TestCase):
    def test_opencode_workflow_shows_opencode_label(self):
        from openshard.history.shard_contract import build_shard_receipt

        entry = {
            "workflow": "opencode",
            "task": "fix auth",
            "timestamp": "2026-01-01T00:00:00",
        }
        receipt = build_shard_receipt(entry)
        self.assertEqual(receipt.agent, "OpenCode")

    def test_native_workflow_shows_native_label(self):
        from openshard.history.shard_contract import build_shard_receipt

        entry = {
            "workflow": "native",
            "task": "fix auth",
            "timestamp": "2026-01-01T00:00:00",
        }
        receipt = build_shard_receipt(entry)
        self.assertEqual(receipt.agent, "OpenShard Native")

    def test_direct_workflow_shows_openshard_label(self):
        from openshard.history.shard_contract import build_shard_receipt

        entry = {
            "workflow": "direct",
            "task": "fix auth",
            "timestamp": "2026-01-01T00:00:00",
        }
        receipt = build_shard_receipt(entry)
        self.assertEqual(receipt.agent, "OpenShard")
