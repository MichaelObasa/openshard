from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

from click.testing import CliRunner

from openshard.analysis.repo import RepoFacts
from openshard.cli.main import cli
from openshard.execution.generator import ChangedFile
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
