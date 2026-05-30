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

    def test_adapter_field_also_triggers_opencode_label(self):
        from openshard.history.shard_contract import build_shard_receipt

        entry = {
            "adapter": "opencode",
            "task": "fix auth",
            "timestamp": "2026-01-01T00:00:00",
        }
        receipt = build_shard_receipt(entry)
        self.assertEqual(receipt.agent, "OpenCode")


class TestAdapterMetaCapture(unittest.TestCase):
    """Tests for adapter_meta populated by OpenCodeExecutor.generate()."""

    def _make_proc(self, returncode=0, stdout="Task complete.\nOpenCode done."):
        mock = MagicMock()
        mock.returncode = returncode
        mock.stdout = stdout
        return mock

    def _make_snapshot(self):
        return {}

    def test_adapter_meta_populated_on_success(self):
        from openshard.execution.opencode_executor import OpenCodeExecutor

        with patch("openshard.execution.opencode_executor._resolve_opencode_binary", return_value="/usr/bin/opencode"):
            with patch("openshard.execution.opencode_executor._snapshot", return_value=self._make_snapshot()):
                with patch("openshard.execution.opencode_executor._classify_changes", return_value=[]):
                    with patch("subprocess.run", return_value=self._make_proc()):
                        with patch("openshard.execution.opencode_executor.load_config", return_value={
                            "model_tiers": [{"name": "balanced", "model": "claude-sonnet-4-6"}]
                        }):
                            executor = OpenCodeExecutor()
                            result = executor.generate("fix tests", workspace=Path("/repo"))

        self.assertIsNotNone(result.adapter_meta)
        meta = result.adapter_meta
        self.assertEqual(meta["adapter"], "opencode")
        self.assertTrue(meta["adapter_available"])
        self.assertEqual(meta["adapter_exit_code"], 0)

    def test_adapter_meta_has_duration_ms(self):
        from openshard.execution.opencode_executor import OpenCodeExecutor

        with patch("openshard.execution.opencode_executor._resolve_opencode_binary", return_value="/usr/bin/opencode"):
            with patch("openshard.execution.opencode_executor._snapshot", return_value=self._make_snapshot()):
                with patch("openshard.execution.opencode_executor._classify_changes", return_value=[]):
                    with patch("subprocess.run", return_value=self._make_proc()):
                        with patch("openshard.execution.opencode_executor.load_config", return_value={
                            "model_tiers": [{"name": "balanced", "model": "claude-sonnet-4-6"}]
                        }):
                            executor = OpenCodeExecutor()
                            result = executor.generate("fix tests", workspace=Path("/repo"))

        self.assertIsInstance(result.adapter_meta["adapter_duration_ms"], int)

    def test_adapter_command_is_list(self):
        from openshard.execution.opencode_executor import OpenCodeExecutor

        with patch("openshard.execution.opencode_executor._resolve_opencode_binary", return_value="/usr/bin/opencode"):
            with patch("openshard.execution.opencode_executor._snapshot", return_value=self._make_snapshot()):
                with patch("openshard.execution.opencode_executor._classify_changes", return_value=[]):
                    with patch("subprocess.run", return_value=self._make_proc()):
                        with patch("openshard.execution.opencode_executor.load_config", return_value={
                            "model_tiers": [{"name": "balanced", "model": "claude-sonnet-4-6"}]
                        }):
                            executor = OpenCodeExecutor()
                            result = executor.generate("fix tests", workspace=Path("/repo"))

        self.assertIsInstance(result.adapter_meta["adapter_command"], list)

    def test_stderr_summary_is_none(self):
        from openshard.execution.opencode_executor import OpenCodeExecutor

        with patch("openshard.execution.opencode_executor._resolve_opencode_binary", return_value="/usr/bin/opencode"):
            with patch("openshard.execution.opencode_executor._snapshot", return_value=self._make_snapshot()):
                with patch("openshard.execution.opencode_executor._classify_changes", return_value=[]):
                    with patch("subprocess.run", return_value=self._make_proc()):
                        with patch("openshard.execution.opencode_executor.load_config", return_value={
                            "model_tiers": [{"name": "balanced", "model": "claude-sonnet-4-6"}]
                        }):
                            executor = OpenCodeExecutor()
                            result = executor.generate("fix tests", workspace=Path("/repo"))

        self.assertIsNone(result.adapter_meta["adapter_stderr_summary"])


class TestAdapterShardReceipt(unittest.TestCase):
    """Tests for adapter metadata fields in ShardReceipt and receipt rendering."""

    def _opencode_entry(self) -> dict:
        return {
            "adapter": "opencode",
            "adapter_available": True,
            "adapter_command": ["opencode", "run", "--model", "openrouter/claude-sonnet-4-6", "fix tests"],
            "adapter_exit_code": 0,
            "adapter_stdout_summary": "OpenCode done.",
            "adapter_stderr_summary": None,
            "adapter_duration_ms": 1240,
            "task": "fix tests",
            "timestamp": "2026-01-01T00:00:00Z",
        }

    def test_old_receipt_without_adapter_renders_safely(self):
        from openshard.history.shard_contract import build_shard_receipt, render_full_shard_receipt

        entry = {"task": "fix tests", "timestamp": "2026-01-01T00:00:00Z"}
        receipt = build_shard_receipt(entry)
        rendered = render_full_shard_receipt(receipt)
        self.assertNotIn("ADAPTER", rendered)

    def test_shard_receipt_parses_adapter_fields(self):
        from openshard.history.shard_contract import build_shard_receipt

        receipt = build_shard_receipt(self._opencode_entry())
        self.assertEqual(receipt.adapter, "opencode")
        self.assertTrue(receipt.adapter_available)
        self.assertEqual(receipt.adapter_exit_code, 0)
        self.assertEqual(receipt.adapter_duration_ms, 1240)
        self.assertEqual(receipt.adapter_stdout_summary, "OpenCode done.")
        self.assertIsNone(receipt.adapter_stderr_summary)
        self.assertIsInstance(receipt.adapter_command, list)

    def test_full_receipt_renders_adapter_section_when_present(self):
        from openshard.history.shard_contract import build_shard_receipt, render_full_shard_receipt

        receipt = build_shard_receipt(self._opencode_entry())
        rendered = render_full_shard_receipt(receipt)
        self.assertIn("ADAPTER", rendered)
        self.assertIn("opencode", rendered)

    def test_full_receipt_omits_adapter_section_when_absent(self):
        from openshard.history.shard_contract import build_shard_receipt, render_full_shard_receipt

        entry = {"task": "fix tests", "timestamp": "2026-01-01T00:00:00Z"}
        receipt = build_shard_receipt(entry)
        rendered = render_full_shard_receipt(receipt)
        self.assertNotIn("ADAPTER", rendered)

    def test_stdout_summary_is_safe_length(self):
        from openshard.history.shard_contract import build_shard_receipt

        entry = self._opencode_entry()
        entry["adapter_stdout_summary"] = "x" * 500
        receipt = build_shard_receipt(entry)
        self.assertLessEqual(len(receipt.adapter_stdout_summary), 1000)

    def test_adapter_command_is_list_not_string(self):
        from openshard.history.shard_contract import build_shard_receipt

        receipt = build_shard_receipt(self._opencode_entry())
        self.assertIsInstance(receipt.adapter_command, list)

    def test_missing_availability_represented_safely(self):
        from openshard.history.shard_contract import build_shard_receipt

        entry = self._opencode_entry()
        entry["adapter_available"] = False
        receipt = build_shard_receipt(entry)
        self.assertFalse(receipt.adapter_available)

    def test_compact_receipt_does_not_include_adapter_section(self):
        from openshard.history.shard_contract import build_shard_receipt, render_compact_shard_receipt

        receipt = build_shard_receipt(self._opencode_entry())
        rendered = render_compact_shard_receipt(receipt)
        self.assertNotIn("ADAPTER", rendered)

    def test_compact_receipt_renders_without_crashing(self):
        from openshard.history.shard_contract import build_shard_receipt, render_compact_shard_receipt

        receipt = build_shard_receipt(self._opencode_entry())
        rendered = render_compact_shard_receipt(receipt)
        self.assertIsInstance(rendered, str)
        self.assertGreater(len(rendered), 0)

    def test_command_display_in_receipt_is_truncated(self):
        from openshard.history.shard_contract import build_shard_receipt, render_full_shard_receipt

        entry = self._opencode_entry()
        entry["adapter_command"] = ["opencode", "run", "--model", "openrouter/x", "a very long task " * 20]
        receipt = build_shard_receipt(entry)
        rendered = render_full_shard_receipt(receipt)
        # Command row should not dump the full long task
        self.assertIn("opencode run", rendered)
        for line in rendered.splitlines():
            if "Command" in line:
                self.assertLessEqual(len(line), 200)
                break
