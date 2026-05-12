from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

from click.testing import CliRunner

from openshard.analysis.repo import RepoFacts
from openshard.cli.main import cli
from openshard.execution.generator import ChangedFile
from openshard.providers.base import ModelInfo
from openshard.providers.manager import InventoryEntry
from openshard.routing.engine import is_readonly_task

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
        """With tier dispatch, Model: line reflects executor model, not routing candidate."""
        with patch("openshard.native.dispatch.resolve_tier_for_category",
                   return_value=("z-ai/glm-5.1", "executor-tier", False, "")), \
             patch("openshard.native.dispatch.resolve_tier",
                   return_value=("z-ai/glm-5.1", False, "")):
            result, _ = self._run_with_tier_dispatch(
                "what does main.py do?",
                extra_args=["--more", "--experimental-tier-dispatch"],
            )
        self.assertEqual(result.exit_code, 0, result.output)
        model_lines = [ln for ln in result.output.splitlines() if ln.startswith("Model:")]
        self.assertTrue(model_lines, f"No Model: line found\n{result.output}")
        # routing candidate is "openrouter/test-model" (from _make_manager_mock_with_entry)
        # executor model is "z-ai/glm-5.1" — Model: line must show the executor, not routing
        self.assertNotIn("test-model", model_lines[0])

    def test_direct_ask_model_line_consistent_with_ask_plan(self):
        """Model: line and Model plan Ask: line show the same model for direct Ask."""
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
        model_lines = [ln.strip() for ln in lines if ln.startswith("Model:")]
        ask_lines = [ln.strip() for ln in lines if ln.strip().startswith("Ask:")]
        if model_lines and ask_lines:
            # Both should reference the same model label
            ask_model = ask_lines[0].split("Ask:", 1)[-1].strip()
            self.assertTrue(
                any(ask_model in ln for ln in model_lines),
                f"Model: {model_lines!r} does not match Ask: model {ask_model!r}",
            )
