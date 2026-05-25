from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

from click.testing import CliRunner

from openshard.analysis.repo import RepoFacts
from openshard.cli.main import cli
from openshard.execution.generator import ChangedFile
from openshard.providers.base import ModelInfo
from openshard.providers.manager import InventoryEntry
from openshard.routing.engine import is_readonly_task, has_inline_readonly_instruction, looks_like_review_task

_DEFAULT_CONFIG = {"approval_mode": "smart"}

_SAFE_REPO = RepoFacts(
    languages=["python"], package_files=[], framework=None,
    test_command=None, risky_paths=[], changed_files=[],
)


def _empty_plan():
    from openshard.verification.plan import VerificationPlan
    return VerificationPlan(commands=[])


def _fake_result_no_files():
    r = MagicMock()
    r.usage = None
    r.files = []
    r.summary = "This file handles the CLI entry point for OpenShard."
    r.notes = []
    return r


def _fake_result_with_files():
    """Generator returns file changes — should be discarded for read-only tasks."""
    r = MagicMock()
    r.usage = None
    r.files = [
        ChangedFile(
            path="openshard/cli/main.py",
            change_type="update",
            content="# Implemented the complete OpenShard CLI...\n",
            summary="Implemented CLI",
        )
    ]
    r.summary = "Implemented the complete OpenShard CLI."
    r.notes = []
    return r


def _make_generator_mock(result=None):
    g = MagicMock()
    g.generate.return_value = result if result is not None else _fake_result_no_files()
    g.model = "mock-model"
    g.fixer_model = "mock-fixer"
    return g


def _make_manager_mock():
    m = MagicMock()
    inv = MagicMock()
    inv.models = []
    m.get_inventory.return_value = inv
    m.providers = {"openrouter": MagicMock()}
    return m


def _invoke(task: str, *, extra_args: tuple = (), result=None):
    gen_mock = _make_generator_mock(result)
    with patch("openshard.run.pipeline.ExecutionGenerator", return_value=gen_mock), \
         patch("openshard.run.pipeline.NativeAgentExecutor", return_value=MagicMock()), \
         patch("openshard.run.pipeline.ProviderManager", return_value=_make_manager_mock()), \
         patch("openshard.cli.main.load_config", return_value=_DEFAULT_CONFIG), \
         patch("openshard.run.pipeline.analyze_repo", return_value=_SAFE_REPO), \
         patch("openshard.run.pipeline.build_verification_plan", return_value=_empty_plan()), \
         patch("openshard.run.pipeline._log_run"), \
         patch("openshard.run.pipeline._build_explicit_file_context", return_value=""):
        runner = CliRunner()
        cli_result = runner.invoke(cli, ["run"] + list(extra_args) + [task])
    return cli_result, gen_mock


# ---------------------------------------------------------------------------
# Unit tests for is_readonly_task()
# ---------------------------------------------------------------------------

class TestIsReadonlyTask(unittest.TestCase):

    # --- read-only → True ---------------------------------------------------

    def test_what_does(self):
        self.assertTrue(is_readonly_task("what does openshard/cli/main.py do?"))

    def test_what_is(self):
        self.assertTrue(is_readonly_task("what is the purpose of the routing engine?"))

    def test_what_are(self):
        self.assertTrue(is_readonly_task("what are the available commands?"))

    def test_explain(self):
        self.assertTrue(is_readonly_task("explain the pipeline execution flow"))

    def test_summarise(self):
        self.assertTrue(is_readonly_task("summarise the routing module"))

    def test_summarize(self):
        self.assertTrue(is_readonly_task("summarize openshard/routing/engine.py"))

    def test_describe(self):
        self.assertTrue(is_readonly_task("describe what this file does"))

    def test_walk_me_through(self):
        self.assertTrue(is_readonly_task("walk me through the native executor"))

    def test_where_is(self):
        self.assertTrue(is_readonly_task("where is the write guard implemented?"))

    def test_show_me(self):
        self.assertTrue(is_readonly_task("show me the generator interface"))

    def test_how_does(self):
        self.assertTrue(is_readonly_task("how does the staged workflow select a model?"))

    def test_tell_me_about(self):
        self.assertTrue(is_readonly_task("tell me about the verification plan"))

    def test_list_the(self):
        self.assertTrue(is_readonly_task("list the available skills"))

    def test_find_where(self):
        self.assertTrue(is_readonly_task("find where file writes are applied"))

    def test_case_insensitive(self):
        self.assertTrue(is_readonly_task("What Does openshard/cli/main.py Do?"))

    def test_leading_whitespace(self):
        self.assertTrue(is_readonly_task("  explain the pipeline"))

    # --- write → False ------------------------------------------------------

    def test_fix(self):
        self.assertFalse(is_readonly_task("fix the bug in the generator"))

    def test_add(self):
        self.assertFalse(is_readonly_task("add a helper function"))

    def test_update(self):
        self.assertFalse(is_readonly_task("update README.md"))

    def test_implement(self):
        self.assertFalse(is_readonly_task("implement the read-only guard"))

    def test_refactor(self):
        self.assertFalse(is_readonly_task("refactor the routing engine"))

    def test_create(self):
        self.assertFalse(is_readonly_task("create a new CLI command"))

    def test_remove(self):
        self.assertFalse(is_readonly_task("remove the deprecated executor flag"))

    def test_delete(self):
        self.assertFalse(is_readonly_task("delete unused imports"))

    def test_migrate(self):
        self.assertFalse(is_readonly_task("migrate to the new API"))

    # --- edge cases ---------------------------------------------------------

    def test_find_and_fix_is_not_readonly(self):
        self.assertFalse(is_readonly_task("find and fix the null pointer"))

    def test_find_and_remove_is_not_readonly(self):
        self.assertFalse(is_readonly_task("find and remove the dead code"))

    def test_what_does_create_do_is_readonly(self):
        # starts with "what does", create() is the object not the verb
        self.assertTrue(is_readonly_task("what does create() do?"))

    def test_plain_task_is_not_readonly(self):
        self.assertFalse(is_readonly_task("the generator parses JSON"))

    def test_empty_string(self):
        self.assertFalse(is_readonly_task(""))


# ---------------------------------------------------------------------------
# Pipeline integration tests
# ---------------------------------------------------------------------------

class TestReadonlyPipelineIntegration(unittest.TestCase):

    def test_readonly_task_still_calls_generate(self):
        """Provider call runs so the model can produce an explanation."""
        _, gen_mock = _invoke("what does openshard/cli/main.py do?")
        gen_mock.generate.assert_called_once()

    def test_readonly_task_injects_readonly_instruction(self):
        """skills_context passed to generate contains the read-only instruction."""
        _, gen_mock = _invoke("what does openshard/cli/main.py do?")
        gen_mock.generate.assert_called_once()
        ctx = gen_mock.generate.call_args[1].get("skills_context", "")
        self.assertIn("[IMPORTANT]", ctx)
        self.assertIn("read-only", ctx)
        self.assertIn("files` field", ctx)

    def test_readonly_task_discards_returned_files(self):
        """Files returned by the model are not written for a read-only task."""
        result, _ = _invoke(
            "what does openshard/cli/main.py do?",
            result=_fake_result_with_files(),
        )
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertNotIn("[written]", result.output)

    def test_readonly_task_shows_no_file_count(self):
        """'Files:' line must not appear in output for read-only tasks."""
        result, _ = _invoke(
            "explain the pipeline",
            result=_fake_result_with_files(),
        )
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertNotIn("Files:", result.output)

    def test_readonly_task_shows_ignored_warning(self):
        """A notice is printed when generated file changes are discarded."""
        result, _ = _invoke(
            "what does openshard/cli/main.py do?",
            result=_fake_result_with_files(),
        )
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertIn("Read-only task", result.output)
        self.assertIn("discarded", result.output)

    def test_dry_run_readonly_does_not_call_generate(self):
        """Dry-run must not call the provider even for a read-only task."""
        _, gen_mock = _invoke(
            "what does openshard/cli/main.py do?",
            extra_args=("--dry-run",),
        )
        gen_mock.generate.assert_not_called()

    def test_dry_run_readonly_does_not_inject_instruction(self):
        """Read-only instruction must not be injected when dry-run skips generation."""
        _, gen_mock = _invoke(
            "explain the CLI",
            extra_args=("--dry-run",),
        )
        gen_mock.generate.assert_not_called()

    def test_write_task_is_not_affected(self):
        """A write task still calls generate and does not inject the read-only instruction."""
        _, gen_mock = _invoke(
            "update README.md with a description",
            result=_fake_result_no_files(),
        )
        gen_mock.generate.assert_called_once()
        ctx = gen_mock.generate.call_args[1].get("skills_context", "")
        self.assertNotIn("[IMPORTANT]", ctx)

    def test_write_task_with_files_is_not_discarded(self):
        """Files returned by generate for a write task must not be stripped."""
        result, gen_mock = _invoke(
            "add a helper function to utils.py",
            result=_fake_result_with_files(),
        )
        gen_mock.generate.assert_called_once()
        self.assertNotIn("Read-only task", result.output)


# ---------------------------------------------------------------------------
# Output label tests
# ---------------------------------------------------------------------------

class TestReadonlyOutputLabels(unittest.TestCase):

    def test_readonly_routing_label_no_feature_implementation(self):
        """[routing] line must not say 'feature implementation' for read-only tasks."""
        result, _ = _invoke("what does openshard/cli/main.py do?", extra_args=("--full",))
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertNotIn("feature implementation", result.output)

    def test_readonly_routing_label_says_readonly_analysis(self):
        """[routing] line must contain 'read-only analysis' for read-only tasks."""
        result, _ = _invoke("what does openshard/cli/main.py do?", extra_args=("--full",))
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertIn("read-only analysis", result.output)

    def test_write_task_routing_label_unchanged(self):
        """Write task routing must still say 'standard feature implementation'."""
        # Task must route to standard category (no security/visual/complex/boilerplate keywords).
        result, _ = _invoke("implement the new pagination feature", extra_args=("--full",))
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertIn("standard feature implementation", result.output)

    def test_readonly_more_shows_ask_mode(self):
        """--more output must show 'Mode: Ask' for read-only tasks."""
        result, _ = _invoke("what does openshard/cli/main.py do?", extra_args=("--more",))
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertIn("Mode: Ask", result.output)

    def test_dry_run_readonly_exits_cleanly(self):
        """Dry-run of a read-only task must complete without error."""
        result, _ = _invoke(
            "what does openshard/cli/main.py do?",
            extra_args=("--dry-run",),
        )
        self.assertEqual(result.exit_code, 0, result.output)


# ---------------------------------------------------------------------------
# Helpers for fast-path tests (tier dispatch needs a real model entry)
# ---------------------------------------------------------------------------

def _make_entry(model_id: str) -> InventoryEntry:
    return InventoryEntry(
        provider="openrouter",
        model=ModelInfo(
            id=model_id, name=model_id, pricing={"prompt": "0.0000005"},
            context_window=None, max_output_tokens=None,
            supports_vision=False, supports_tools=False,
        ),
    )


_ENTRY = _make_entry("openrouter/test-model")


def _make_manager_mock_with_entry():
    m = MagicMock()
    inv = MagicMock()
    inv.models = [_ENTRY]
    m.get_inventory.return_value = inv
    m.providers = {"openrouter": MagicMock()}
    return m


# ---------------------------------------------------------------------------
# Workflow fast path: read-only tasks default to direct (no stage_runs)
# ---------------------------------------------------------------------------

class TestReadonlyFastPath(unittest.TestCase):

    def _run_with_tier_dispatch(self, task: str, extra_args: list[str] | None = None):
        gen_mock = _make_generator_mock()
        log_mock = MagicMock()
        args = ["run"] + (extra_args or []) + [task]
        with patch("openshard.run.pipeline.ExecutionGenerator", return_value=gen_mock), \
             patch("openshard.run.pipeline.NativeAgentExecutor", return_value=MagicMock()), \
             patch("openshard.run.pipeline.ProviderManager", return_value=_make_manager_mock_with_entry()), \
             patch("openshard.cli.main.load_config", return_value=_DEFAULT_CONFIG), \
             patch("openshard.run.pipeline.analyze_repo", return_value=_SAFE_REPO), \
             patch("openshard.run.pipeline.build_verification_plan", return_value=_empty_plan()), \
             patch("openshard.run.pipeline._log_run", log_mock), \
             patch("openshard.run.pipeline._build_explicit_file_context", return_value=""):
            cli_result = CliRunner().invoke(cli, args)
        return cli_result, log_mock

    def test_simple_readonly_no_planning_stage(self):
        """Standard category read-only task must not print a Planning stage."""
        result, _ = _invoke("explain the pipeline execution flow")
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertNotIn("Planning", result.output)

    def test_readonly_security_category_no_planning_stage(self):
        """Security category read-only task must still skip Planning (readonly wins)."""
        result, _ = _invoke("explain the auth token flow")
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertNotIn("Planning", result.output)

    def test_readonly_mode_ask_label_unchanged(self):
        """Mode: Ask must still appear for read-only tasks after fast-path change."""
        result, _ = _invoke("what does openshard/cli/main.py do?", extra_args=("--more",))
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertIn("Mode: Ask", result.output)

    def test_write_task_workflow_unaffected(self):
        """Write tasks are not affected by the read-only fast path."""
        _, gen_mock = _invoke("implement auth token refresh")
        gen_mock.generate.assert_called_once()

    def test_readonly_validator_policy_shown_as_skipped(self):
        """With --full --experimental-tier-dispatch, validator shows skipped with
        'read-only task' reason — proves policy is evaluated, not merely absent."""
        result, _ = self._run_with_tier_dispatch(
            "explain the auth token flow",
            extra_args=["--full", "--experimental-tier-dispatch"],
        )
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertIn("Validator", result.output)
        self.assertIn("skipped", result.output.lower())
        self.assertIn("read-only", result.output.lower())

    def test_simple_readonly_no_planning_stage_runs_logged(self):
        """Logged stage_runs must contain no planning entry for simple read-only tasks.
        Uses real select_workflow so the fast path is exercised end-to-end."""
        _, log_mock = self._run_with_tier_dispatch(
            "explain the pipeline execution flow",
            extra_args=["--experimental-tier-dispatch"],
        )
        log_mock.assert_called_once()
        stage_runs = log_mock.call_args.kwargs.get("stage_runs") or []
        stage_types = [sr.stage.stage_type for sr in stage_runs]
        self.assertNotIn("planning", stage_types)

    def test_direct_ask_execution_reason_clear(self):
        """Execution Reason: line uses 'read-only task — direct analysis' for Ask mode."""
        result, _ = _invoke("what does main.py do?", extra_args=("--more",))
        self.assertEqual(result.exit_code, 0, result.output)
        reason_line = next(
            (ln for ln in result.output.splitlines() if "  Reason:" in ln), None
        )
        self.assertIsNotNone(reason_line, f"No Reason: line in output:\n{result.output}")
        self.assertIn("read-only task", reason_line)
        self.assertIn("direct analysis", reason_line)

    def test_write_task_execution_reason_not_overridden(self):
        """Staged write tasks keep the profile reason, not the read-only override."""
        result, _ = _invoke("implement auth token refresh", extra_args=("--more",))
        reason_line = next(
            (ln for ln in result.output.splitlines() if "  Reason:" in ln), None
        )
        if reason_line:
            self.assertNotIn("read-only task", reason_line)

    def test_direct_ask_model_line_uses_executor_model(self):
        """Stage table Ask row reflects executor model, not routing candidate."""
        with patch("openshard.native.dispatch.resolve_tier_for_category",
                   return_value=("z-ai/glm-5.1", "executor-tier", False, "")), \
             patch("openshard.native.dispatch.resolve_tier",
                   return_value=("z-ai/glm-5.1", False, "")):
            result, _ = self._run_with_tier_dispatch(
                "what does main.py do?",
                extra_args=["--more", "--experimental-tier-dispatch"],
            )
        self.assertEqual(result.exit_code, 0, result.output)
        # Stage table Ask row must show executor model (GLM-5.1), not routing candidate
        ask_rows = [ln for ln in result.output.splitlines() if "ask" in ln.lower() and "GLM" in ln]
        self.assertTrue(ask_rows, f"No Ask row with GLM model found\n{result.output}")
        # routing candidate is "openrouter/test-model" — must not appear in Ask row
        self.assertFalse(
            any("test-model" in ln for ln in ask_rows),
            f"Routing candidate leaked into Ask row: {ask_rows}",
        )

    def test_direct_ask_model_line_consistent_with_ask_plan(self):
        """Stage table Ask row and Model plan Ask: entry show the same model."""
        with patch("openshard.native.dispatch.resolve_tier_for_category",
                   return_value=("z-ai/glm-5.1", "executor-tier", False, "")), \
             patch("openshard.native.dispatch.resolve_tier",
                   return_value=("z-ai/glm-5.1", False, "")):
            result, _ = self._run_with_tier_dispatch(
                "what does main.py do?",
                extra_args=["--more", "--experimental-tier-dispatch"],
            )
        self.assertEqual(result.exit_code, 0, result.output)
        lines = result.output.splitlines()
        ask_stage_rows = [ln.strip() for ln in lines if ln.strip().startswith("Ask") and "ask" in ln.lower()]
        ask_plan_lines = [ln.strip() for ln in lines if ln.strip().startswith("Ask:")]
        if ask_stage_rows and ask_plan_lines:
            # Both should reference the same model label
            ask_plan_model = ask_plan_lines[0].split("Ask:", 1)[-1].strip()
            self.assertTrue(
                any(ask_plan_model in ln for ln in ask_stage_rows),
                f"Stage Ask row {ask_stage_rows!r} does not match plan Ask: model {ask_plan_model!r}",
            )


# ---------------------------------------------------------------------------
# SF task (STRUCTURED_FINDINGS) pipeline integration tests
# ---------------------------------------------------------------------------

_SF_TASK = (
    "Review this code for security issues.\n\n"
    'STRUCTURED_FINDINGS: [{"severity": "Critical", "message": "test"}]'
)


class TestSfTaskPipelineIntegration(unittest.TestCase):
    """Tasks containing STRUCTURED_FINDINGS: are treated as review tasks regardless of prefix."""

    def test_sf_task_injects_review_instruction(self):
        """skills_context must contain the STRUCTURED_FINDINGS review instruction."""
        _, gen_mock = _invoke(_SF_TASK, extra_args=("--workflow", "direct"))
        gen_mock.generate.assert_called_once()
        ctx = gen_mock.generate.call_args[1].get("skills_context", "")
        self.assertIn("[IMPORTANT]", ctx)
        self.assertIn("STRUCTURED_FINDINGS", ctx)
        self.assertIn("files` array must be empty", ctx)

    def test_sf_task_uses_review_max_tokens(self):
        """max_tokens must be 8192 for SF tasks, not the 65536 OpenRouter default."""
        _, gen_mock = _invoke(_SF_TASK, extra_args=("--workflow", "direct"))
        gen_mock.generate.assert_called_once()
        max_tokens = gen_mock.generate.call_args[1].get("max_tokens")
        self.assertIsNotNone(max_tokens, "max_tokens was not passed to generate()")
        self.assertEqual(max_tokens, 8192)
        self.assertNotEqual(max_tokens, 65536)

    def test_normal_task_uses_standard_max_tokens(self):
        """max_tokens must be 16384 for normal write tasks."""
        _, gen_mock = _invoke("implement a new feature", extra_args=("--workflow", "direct"))
        gen_mock.generate.assert_called_once()
        max_tokens = gen_mock.generate.call_args[1].get("max_tokens")
        self.assertIsNotNone(max_tokens, "max_tokens was not passed to generate()")
        self.assertEqual(max_tokens, 16384)
        self.assertNotEqual(max_tokens, 65536)

    def test_sf_task_discards_returned_files(self):
        """Files returned by the model must be discarded for SF tasks (even when not _readonly_task)."""
        result, _ = _invoke(
            _SF_TASK,
            extra_args=("--workflow", "direct"),
            result=_fake_result_with_files(),
        )
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertNotIn("[written]", result.output)
        self.assertIn("Read-only task", result.output)
        self.assertIn("discarded", result.output)

    def test_production_iac_hardening_max_tokens_not_65536(self):
        """The production-iac-hardening full task must request 8192 max_tokens, not 65536."""
        from openshard.workflow_packs.packs import get_pack
        p = get_pack("production-iac-hardening")
        full_task = p.prompt + p.execution_prompt_suffix
        _, gen_mock = _invoke(full_task, extra_args=("--workflow", "direct"))
        gen_mock.generate.assert_called_once()
        max_tokens = gen_mock.generate.call_args[1].get("max_tokens")
        self.assertIsNotNone(max_tokens, "max_tokens was not passed to generate()")
        self.assertLessEqual(max_tokens, 16384, f"max_tokens={max_tokens} exceeds 16384 cap")
        self.assertEqual(max_tokens, 8192)


# ---------------------------------------------------------------------------
# Unit tests for has_inline_readonly_instruction()
# ---------------------------------------------------------------------------

class TestHasInlineReadonlyInstruction(unittest.TestCase):

    def test_do_not_apply_changes(self):
        self.assertTrue(has_inline_readonly_instruction("Review this repo. Do not apply changes."))

    def test_do_not_make_changes(self):
        self.assertTrue(has_inline_readonly_instruction("Audit the config. Do not make changes."))

    def test_review_only(self):
        self.assertTrue(has_inline_readonly_instruction("This is a review only task."))

    def test_do_not_modify(self):
        self.assertTrue(has_inline_readonly_instruction("Assess the code. Do not modify anything."))

    def test_without_making_changes(self):
        self.assertTrue(has_inline_readonly_instruction("Check the module without making changes."))

    def test_without_modifying_files(self):
        self.assertTrue(has_inline_readonly_instruction("Inspect the repo without modifying files."))

    def test_do_not_change_files(self):
        self.assertTrue(has_inline_readonly_instruction("Evaluate this. Do not change files."))

    def test_no_file_changes(self):
        self.assertTrue(has_inline_readonly_instruction("No file changes should be made."))

    def test_read_only_hyphen(self):
        self.assertTrue(has_inline_readonly_instruction("This is a read-only assessment."))

    def test_read_only_space(self):
        self.assertTrue(has_inline_readonly_instruction("Treat this as read only."))

    def test_case_insensitive(self):
        self.assertTrue(has_inline_readonly_instruction("DO NOT APPLY CHANGES to any file."))

    def test_normal_write_task_is_false(self):
        self.assertFalse(has_inline_readonly_instruction("implement pagination"))

    def test_empty_string_is_false(self):
        self.assertFalse(has_inline_readonly_instruction(""))

    def test_explanation_without_phrase_is_false(self):
        self.assertFalse(has_inline_readonly_instruction("explain the routing logic"))


# ---------------------------------------------------------------------------
# Unit tests for looks_like_review_task()
# ---------------------------------------------------------------------------

class TestLooksLikeReviewTask(unittest.TestCase):

    def test_terraform_production_readiness(self):
        task = "Review this Terraform repo for production readiness. Do not apply changes."
        self.assertTrue(looks_like_review_task(task))

    def test_security_audit(self):
        self.assertTrue(looks_like_review_task("Security audit of the API layer."))

    def test_assess(self):
        self.assertTrue(looks_like_review_task("Assess the networking configuration."))

    def test_iac_hardening(self):
        self.assertTrue(looks_like_review_task("Review IaC hardening for the cluster."))

    def test_explanation_task_is_false(self):
        self.assertFalse(looks_like_review_task("explain this module, do not write files"))

    def test_summarise_task_is_false(self):
        self.assertFalse(looks_like_review_task("summarise this code, do not change files"))

    def test_empty_string_is_false(self):
        self.assertFalse(looks_like_review_task(""))


# ---------------------------------------------------------------------------
# Pipeline integration: inline read-only instruction enforcement
# ---------------------------------------------------------------------------

_VALIDATION_PROMPT = (
    "Review this Terraform repo for production readiness. "
    "Focus on security, deletion protection, networking risk, secrets, and 2am operability. "
    "Do not apply changes."
)


def _fake_result_with_report():
    """Generator returns a markdown report file — must be discarded on read-only runs."""
    r = MagicMock()
    r.usage = None
    r.files = [
        ChangedFile(
            path="TERRAFORM_PRODUCTION_READINESS_REVIEW.md",
            change_type="create",
            content="# Review\n\n## Critical\n- Missing deletion protection\n",
            summary="Terraform production readiness review",
        )
    ]
    r.summary = "Production readiness review complete."
    r.notes = []
    return r


class TestInlineReadonlyPipelineIntegration(unittest.TestCase):

    def test_do_not_apply_discards_files(self):
        """Prompt with 'Do not apply changes' must discard model-generated files."""
        result, _ = _invoke(
            _VALIDATION_PROMPT,
            result=_fake_result_with_report(),
        )
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertIn("Read-only task", result.output)
        self.assertIn("discarded", result.output)

    def test_do_not_apply_changed_count_zero(self):
        """Receipt must show 0 changed files for 'Do not apply changes' prompt."""
        result, _ = _invoke(
            _VALIDATION_PROMPT,
            result=_fake_result_with_report(),
        )
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertNotIn("Files: 1 created", result.output)

    def test_no_report_file_created_on_disk(self):
        """The markdown report file must not be written to disk for read-only runs."""
        import os
        import tempfile

        report_name = "TERRAFORM_PRODUCTION_READINESS_REVIEW.md"
        gen_mock = _make_generator_mock(_fake_result_with_report())

        with tempfile.TemporaryDirectory() as tmpdir:
            original_cwd = os.getcwd()
            try:
                os.chdir(tmpdir)
                with patch("openshard.run.pipeline.ExecutionGenerator", return_value=gen_mock), \
                     patch("openshard.run.pipeline.NativeAgentExecutor", return_value=MagicMock()), \
                     patch("openshard.run.pipeline.ProviderManager", return_value=_make_manager_mock()), \
                     patch("openshard.cli.main.load_config", return_value=_DEFAULT_CONFIG), \
                     patch("openshard.run.pipeline.analyze_repo", return_value=_SAFE_REPO), \
                     patch("openshard.run.pipeline.build_verification_plan", return_value=_empty_plan()), \
                     patch("openshard.run.pipeline._log_run"), \
                     patch("openshard.run.pipeline._build_explicit_file_context", return_value=""):
                    runner = CliRunner()
                    cli_result = runner.invoke(cli, ["run", _VALIDATION_PROMPT])
                self.assertEqual(cli_result.exit_code, 0, cli_result.output)
                self.assertFalse(
                    os.path.exists(os.path.join(tmpdir, report_name)),
                    f"{report_name} was created on disk despite read-only instruction",
                )
            finally:
                os.chdir(original_cwd)

    def test_explain_with_do_not_write_is_readonly(self):
        """Explanation task with 'do not write files' is also treated as read-only."""
        result, _ = _invoke(
            "Explain this repo architecture. Do not write files.",
            result=_fake_result_with_files(),
        )
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertIn("Read-only task", result.output)
        self.assertIn("discarded", result.output)
