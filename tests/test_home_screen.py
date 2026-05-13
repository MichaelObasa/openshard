from __future__ import annotations

import io
import json
import unittest
from pathlib import Path
from unittest.mock import patch

from click.testing import CliRunner
from rich.console import Console

from openshard.cli.main import cli


def _write_runs(entries: list[dict]) -> None:
    log_dir = Path(".openshard")
    log_dir.mkdir(parents=True, exist_ok=True)
    with (log_dir / "runs.jsonl").open("w", encoding="utf-8") as fh:
        for entry in entries:
            fh.write(json.dumps(entry) + "\n")


class TestHomeScreen(unittest.TestCase):
    def test_no_args_renders_home_screen(self):
        result = CliRunner().invoke(cli, [])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("OpenShard", result.output)
        self.assertIn("The control layer for AI coding agents.", result.output)

    def test_help_still_renders_click_help(self):
        result = CliRunner().invoke(cli, ["--help"])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("Usage:", result.output)
        self.assertIn("Commands:", result.output)

    def test_existing_safe_subcommands_still_dispatch(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["demo-run"])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("Task:", result.output)

        result = runner.invoke(cli, ["export-runs", "--help"])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("Export run history", result.output)

    def test_no_receipts_empty_state(self):
        runner = CliRunner()
        with runner.isolated_filesystem():
            result = runner.invoke(cli, [])
            self.assertEqual(result.exit_code, 0)
            self.assertIn("No recent receipts yet.", result.output)
            self.assertFalse((Path(".openshard") / "runs.jsonl").exists())

    def test_recent_receipts_render(self):
        runner = CliRunner()
        with runner.isolated_filesystem():
            _write_runs([
                {
                    "task": "first task",
                    "workflow": "Run",
                    "verification_attempted": False,
                    "estimated_cost": 0.01,
                },
                {
                    "task": "second task",
                    "workflow": "Deep Run",
                    "verification_passed": True,
                    "estimated_cost": 0.02,
                },
            ])
            result = runner.invoke(cli, [])
            self.assertEqual(result.exit_code, 0)
            self.assertIn("Recent receipts", result.output)
            self.assertIn("second task", result.output)
            self.assertIn("verified", result.output)

    def test_repo_analysis_failure_does_not_crash(self):
        runner = CliRunner()
        with runner.isolated_filesystem():
            with patch("openshard.cli.ui.home.analyze_repo", side_effect=RuntimeError("boom")):
                result = runner.invoke(cli, [])
            self.assertEqual(result.exit_code, 0)
            self.assertIn("repo analysis unavailable", result.output)

    def test_home_output_includes_control_labels(self):
        result = CliRunner().invoke(cli, [])
        self.assertEqual(result.exit_code, 0)
        for label in ("Route", "Risk", "Verify", "Cost", "Receipt"):
            self.assertIn(label, result.output)

    def test_home_does_not_call_providers_or_models(self):
        runner = CliRunner()
        with runner.isolated_filesystem():
            with (
                patch("openshard.cli.main.PlanGenerator") as plan_generator,
                patch("openshard.cli.main.RunPipeline") as run_pipeline,
                patch("openshard.providers.manager.ProviderManager") as provider_manager,
            ):
                result = runner.invoke(cli, [])
            self.assertEqual(result.exit_code, 0)
            plan_generator.assert_not_called()
            run_pipeline.assert_not_called()
            provider_manager.assert_not_called()

    def test_malformed_run_history_is_ignored(self):
        runner = CliRunner()
        with runner.isolated_filesystem():
            log_dir = Path(".openshard")
            log_dir.mkdir()
            (log_dir / "runs.jsonl").write_text(
                "{bad json\n"
                + json.dumps({"task": "good receipt", "verification_attempted": False})
                + "\n",
                encoding="utf-8",
            )
            result = runner.invoke(cli, [])
            self.assertEqual(result.exit_code, 0)
            self.assertIn("good receipt", result.output)


    def test_recent_receipts_capped_at_three(self):
        runner = CliRunner()
        with runner.isolated_filesystem():
            _write_runs([
                {"task": "task one", "workflow": "Run", "verification_attempted": False},
                {"task": "task two", "workflow": "Run", "verification_attempted": False},
                {"task": "task three", "workflow": "Run", "verification_attempted": False},
                {"task": "task four", "workflow": "Run", "verification_attempted": False},
                {"task": "task five", "workflow": "Run", "verification_attempted": False},
            ])
            result = runner.invoke(cli, [])
            self.assertEqual(result.exit_code, 0)
            self.assertIn("task five", result.output)
            self.assertIn("task four", result.output)
            self.assertIn("task three", result.output)
            self.assertNotIn("task two", result.output)
            self.assertNotIn("task one", result.output)

    def test_git_failure_does_not_crash(self):
        runner = CliRunner()
        with runner.isolated_filesystem():
            with patch("openshard.cli.ui.home.subprocess.run", side_effect=OSError("no git")):
                result = runner.invoke(cli, [])
            self.assertEqual(result.exit_code, 0)
            self.assertIn("Git unavailable", result.output)

    def test_small_width_does_not_crash(self):
        runner = CliRunner()
        with runner.isolated_filesystem():
            with patch("openshard.cli.ui.home.make_console") as mock_make:
                mock_make.side_effect = lambda: Console(
                    file=io.StringIO(), width=40, force_terminal=False, color_system=None
                )
                result = runner.invoke(cli, [])
            self.assertEqual(result.exit_code, 0)

    def test_no_fake_free_mode_shown(self):
        runner = CliRunner()
        with runner.isolated_filesystem():
            result = runner.invoke(cli, [])
            self.assertEqual(result.exit_code, 0)
            self.assertNotIn("free mode", result.output.lower())

    def test_unconfigured_shows_not_configured(self):
        runner = CliRunner()
        with runner.isolated_filesystem():
            with patch("openshard.cli.ui.home.load_config", return_value={}):
                result = runner.invoke(cli, [])
            self.assertEqual(result.exit_code, 0)
            self.assertIn("Not configured", result.output)


if __name__ == "__main__":
    unittest.main()
