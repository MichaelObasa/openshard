"""CLI tests for openshard packs run."""
from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

from click.testing import CliRunner

from openshard.cli.main import cli


def _mock_pipeline(exit_code: int = 0):
    mock_result = MagicMock()
    mock_result.exit_code = exit_code
    mock_pl = MagicMock()
    mock_pl.run.return_value = mock_result
    return mock_pl


def _invoke_run(args: list[str]):
    """Invoke packs run with RunPipeline and load_config mocked out."""
    runner = CliRunner()
    mock_pl = _mock_pipeline()
    with (
        patch("openshard.cli.main.load_config", return_value={}),
        patch("openshard.cli.main.RunPipeline") as MockPipeline,
    ):
        MockPipeline.return_value = mock_pl
        result = runner.invoke(cli, ["packs", "run"] + args)
    return result, MockPipeline, mock_pl


class TestPacksRunBasic(unittest.TestCase):

    def test_unknown_pack_exits_nonzero(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["packs", "run", "does-not-exist"])
        self.assertNotEqual(result.exit_code, 0)

    def test_unknown_pack_shows_pack_id_in_error(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["packs", "run", "does-not-exist"])
        self.assertIn("does-not-exist", result.output)

    def test_unknown_pack_shows_available_packs(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["packs", "run", "does-not-exist"])
        self.assertIn("repo-explanation", result.output)

    def test_valid_pack_exits_zero(self):
        result, _, _ = _invoke_run(["repo-explanation"])
        self.assertEqual(result.exit_code, 0, msg=result.output)

    def test_shows_pack_title_in_output(self):
        result, _, _ = _invoke_run(["repo-explanation"])
        self.assertIn("Explain this repo", result.output)

    def test_shows_category_in_output(self):
        result, _, _ = _invoke_run(["repo-explanation"])
        self.assertIn("repo_review", result.output)

    def test_shows_safety_notes_in_output(self):
        result, _, _ = _invoke_run(["repo-explanation"])
        self.assertIn("Read-only", result.output)

    def test_help_exits_zero(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["packs", "run", "--help"])
        self.assertEqual(result.exit_code, 0)

    def test_help_shows_pack_id_arg(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["packs", "run", "--help"])
        self.assertIn("PACK_ID", result.output)


class TestPacksRunWorkflow(unittest.TestCase):

    def test_native_pack_passes_native_workflow(self):
        _, MockPipeline, _ = _invoke_run(["production-iac-hardening"])
        _, kwargs = MockPipeline.call_args
        self.assertEqual(kwargs.get("workflow"), "native")

    def test_repo_explanation_passes_no_workflow(self):
        _, MockPipeline, _ = _invoke_run(["repo-explanation"])
        _, kwargs = MockPipeline.call_args
        self.assertIsNone(kwargs.get("workflow"))

    def test_workflow_flag_overrides_pack_default(self):
        _, MockPipeline, _ = _invoke_run(["production-iac-hardening", "--workflow", "direct"])
        _, kwargs = MockPipeline.call_args
        self.assertEqual(kwargs.get("workflow"), "direct")

    def test_workflow_flag_shown_in_output(self):
        result, _, _ = _invoke_run(["production-iac-hardening"])
        self.assertIn("native", result.output)

    def test_workflow_override_shown_in_output(self):
        result, _, _ = _invoke_run(["production-iac-hardening", "--workflow", "direct"])
        self.assertIn("direct", result.output)


class TestPacksRunTask(unittest.TestCase):

    def test_pack_prompt_passed_as_task(self):
        _, _, mock_pl = _invoke_run(["repo-explanation"])
        task_arg = mock_pl.run.call_args[0][0]
        self.assertIn("Explain this repo", task_arg)

    def test_context_appended_to_task(self):
        _, _, mock_pl = _invoke_run(["repo-explanation", "--context", "focus on the auth module"])
        task_arg = mock_pl.run.call_args[0][0]
        self.assertIn("focus on the auth module", task_arg)

    def test_context_appended_after_prompt(self):
        _, _, mock_pl = _invoke_run(["repo-explanation", "--context", "focus on auth"])
        task_arg = mock_pl.run.call_args[0][0]
        prompt_pos = task_arg.index("Explain this repo")
        context_pos = task_arg.index("focus on auth")
        self.assertGreater(context_pos, prompt_pos)

    def test_execution_suffix_included_for_review_pack(self):
        _, _, mock_pl = _invoke_run(["production-iac-hardening"])
        task_arg = mock_pl.run.call_args[0][0]
        self.assertIn("STRUCTURED_FINDINGS", task_arg)

    def test_no_execution_suffix_for_repo_explanation(self):
        _, _, mock_pl = _invoke_run(["repo-explanation"])
        task_arg = mock_pl.run.call_args[0][0]
        self.assertNotIn("STRUCTURED_FINDINGS", task_arg)


class TestPacksRunFlags(unittest.TestCase):

    def test_write_flag_forwarded(self):
        _, MockPipeline, _ = _invoke_run(["repo-explanation", "--write"])
        _, kwargs = MockPipeline.call_args
        self.assertTrue(kwargs.get("write"))

    def test_dry_run_flag_forwarded(self):
        _, MockPipeline, _ = _invoke_run(["repo-explanation", "--dry-run"])
        _, kwargs = MockPipeline.call_args
        self.assertTrue(kwargs.get("dry_run"))

    def test_verify_flag_forwarded(self):
        _, MockPipeline, _ = _invoke_run(["repo-explanation", "--verify"])
        _, kwargs = MockPipeline.call_args
        self.assertTrue(kwargs.get("verify"))

    def test_no_flags_defaults_to_no_write(self):
        _, MockPipeline, _ = _invoke_run(["repo-explanation"])
        _, kwargs = MockPipeline.call_args
        self.assertFalse(kwargs.get("write"))


class TestPacksRegressions(unittest.TestCase):
    """Ensure existing packs subcommands still work."""

    def test_packs_list_exits_zero(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["packs", "list"])
        self.assertEqual(result.exit_code, 0)

    def test_packs_list_shows_repo_explanation(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["packs", "list"])
        self.assertIn("repo-explanation", result.output)

    def test_packs_show_exits_zero(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["packs", "show", "repo-explanation"])
        self.assertEqual(result.exit_code, 0)

    def test_packs_prompt_exits_zero(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["packs", "prompt", "repo-explanation"])
        self.assertEqual(result.exit_code, 0)

    def test_packs_no_subcommand_shows_help(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["packs"])
        self.assertIn("run", result.output)


if __name__ == "__main__":
    unittest.main()
