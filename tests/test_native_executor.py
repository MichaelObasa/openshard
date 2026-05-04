from __future__ import annotations

import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from click.testing import CliRunner

from openshard.analysis.repo import RepoFacts
from openshard.cli.main import cli
from openshard.native.context import (
    CompactRunState,
    NativeContextBudget,
    NativeEvidence,
    NativeObservation,
    NativePlan,
    build_compact_run_state,
    build_initial_context_budget,
    render_native_evidence,
    render_native_observation,
    render_native_plan,
)
from openshard.native.executor import NativeAgentExecutor, NativeRunMeta, _build_native_plan, _extract_search_query
from openshard.native.repo_context import NativeRepoContextSummary
from openshard.native.skills import NativeSkill, NativeSkillMatch
from openshard.native.tools import NativeToolCall

_DEFAULT_CONFIG = {"approval_mode": "smart"}

_PYTHON_REPO = RepoFacts(
    languages=["python"], package_files=[], framework=None,
    test_command=None, risky_paths=[], changed_files=[],
)


def _fake_result():
    r = MagicMock()
    r.usage = None
    r.files = []
    r.summary = "done"
    r.notes = []
    return r


def _make_generator_mock():
    g = MagicMock()
    g.generate.return_value = _fake_result()
    g.model = "mock-model"
    g.fixer_model = "mock-fixer"
    return g


def _make_native_mock():
    g = MagicMock(spec=NativeAgentExecutor)
    g.generate.return_value = _fake_result()
    g.model = "mock-model"
    g.fixer_model = "mock-fixer"
    g.native_meta = NativeRunMeta()
    return g


def _make_manager_mock():
    m = MagicMock()
    inv = MagicMock()
    inv.models = []
    m.get_inventory.return_value = inv
    m.providers = {"openrouter": MagicMock()}
    return m


class TestNativeRunMeta(unittest.TestCase):
    """NativeRunMeta default values."""

    def test_defaults(self):
        meta = NativeRunMeta()
        self.assertEqual(meta.workflow, "native")
        self.assertEqual(meta.executor, "native")
        self.assertEqual(meta.execution_depth, "fast")
        self.assertEqual(meta.selected_skills, [])
        self.assertIsNone(meta.context_budget)
        self.assertIsNone(meta.context_state)
        self.assertEqual(meta.context_warnings, [])
        self.assertEqual(meta.tool_trace, [])
        self.assertIsNone(meta.repo_context_summary)


class TestNativeContextBudget(unittest.TestCase):
    """Unit tests for NativeContextBudget, CompactRunState, and their builders."""

    def test_budget_defaults(self):
        budget = NativeContextBudget()
        self.assertIsNone(budget.context_window)
        self.assertEqual(budget.estimated_tokens_used, 0)
        self.assertIsNone(budget.estimated_tokens_remaining)
        self.assertEqual(budget.files_loaded, 0)
        self.assertEqual(budget.skills_loaded, 0)
        self.assertFalse(budget.repo_map_built)
        self.assertEqual(budget.warnings, [])

    def test_build_initial_budget_no_window(self):
        budget = build_initial_context_budget()
        self.assertIsNone(budget.context_window)
        self.assertIsNone(budget.estimated_tokens_remaining)
        self.assertEqual(budget.estimated_tokens_used, 0)

    def test_build_initial_budget_with_window(self):
        budget = build_initial_context_budget(context_window=8000)
        self.assertEqual(budget.context_window, 8000)
        self.assertEqual(budget.estimated_tokens_remaining, 8000)
        self.assertEqual(budget.estimated_tokens_used, 0)

    def test_compact_run_state_defaults(self):
        state = CompactRunState()
        self.assertEqual(state.task_goal, "")
        self.assertEqual(state.repo_facts_summary, "")
        self.assertEqual(state.files_touched, [])
        self.assertIsNone(state.verification_result)
        self.assertEqual(state.blockers, [])
        self.assertEqual(state.next_step, "")

    def test_build_compact_run_state_defaults(self):
        state = build_compact_run_state("fix the bug")
        self.assertEqual(state.task_goal, "fix the bug")
        self.assertEqual(state.repo_facts_summary, "")
        self.assertEqual(state.files_touched, [])
        self.assertIsNone(state.verification_result)
        self.assertEqual(state.blockers, [])
        self.assertEqual(state.next_step, "")

    def test_build_compact_run_state_with_values(self):
        state = build_compact_run_state(
            task_goal="refactor auth",
            repo_facts_summary="Python, Django",
            files_touched=["auth.py", "views.py"],
            verification_result="passed",
            blockers=["missing test"],
            next_step="add coverage",
        )
        self.assertEqual(state.task_goal, "refactor auth")
        self.assertEqual(state.repo_facts_summary, "Python, Django")
        self.assertEqual(state.files_touched, ["auth.py", "views.py"])
        self.assertEqual(state.verification_result, "passed")
        self.assertEqual(state.blockers, ["missing test"])
        self.assertEqual(state.next_step, "add coverage")

    def test_build_compact_run_state_none_lists_default_to_empty(self):
        state = build_compact_run_state("task", files_touched=None, blockers=None)
        self.assertEqual(state.files_touched, [])
        self.assertEqual(state.blockers, [])


class TestNativeAgentExecutor(unittest.TestCase):
    """NativeAgentExecutor unit tests — no real provider or API calls."""

    def _make_executor(self):
        fake_gen = _make_generator_mock()
        with patch("openshard.native.executor.ExecutionGenerator", return_value=fake_gen):
            executor = NativeAgentExecutor(provider=MagicMock())
        return executor, fake_gen

    def test_native_meta_defaults(self):
        executor, _ = self._make_executor()
        self.assertIsInstance(executor.native_meta, NativeRunMeta)
        self.assertEqual(executor.native_meta.workflow, "native")
        self.assertEqual(executor.native_meta.executor, "native")
        self.assertEqual(executor.native_meta.execution_depth, "fast")
        self.assertEqual(executor.native_meta.selected_skills, [])
        self.assertIsNone(executor.native_meta.context_budget)
        self.assertIsNone(executor.native_meta.context_state)
        self.assertEqual(executor.native_meta.context_warnings, [])
        self.assertEqual(executor.native_meta.tool_trace, [])

    def test_model_attributes_forwarded(self):
        executor, fake_gen = self._make_executor()
        self.assertEqual(executor.model, fake_gen.model)
        self.assertEqual(executor.fixer_model, fake_gen.fixer_model)

    def test_generate_delegates_to_inner_generator(self):
        executor, fake_gen = self._make_executor()
        result = executor.generate("fix the bug", model="some-model")
        fake_gen.generate.assert_called_once()
        _, kwargs = fake_gen.generate.call_args
        self.assertIn("[native plan]", kwargs["skills_context"])
        self.assertIs(result, fake_gen.generate.return_value)

    def test_generate_passes_repo_facts_and_skills(self):
        executor, fake_gen = self._make_executor()
        executor.generate("task", repo_facts=_PYTHON_REPO, skills_context="hint")
        _, kwargs = fake_gen.generate.call_args
        self.assertIn("[native plan]", kwargs["skills_context"])
        self.assertIn("hint", kwargs["skills_context"])

    def test_generate_updates_selected_skills(self):
        executor, _ = self._make_executor()
        fake_skill = NativeSkill(
            name="test-discovery",
            description="test",
            categories=["testing"],
            triggers=["test"],
        )
        fake_match = NativeSkillMatch(skill=fake_skill, reason="task matches: test", score=1.0)
        with patch(
            "openshard.native.executor.match_builtin_skills",
            return_value=[fake_match],
        ):
            executor.generate("run the tests")
        self.assertEqual(executor.native_meta.selected_skills, ["test-discovery"])


class TestNativeAgentExecutorRunTool(unittest.TestCase):
    """Tests for NativeAgentExecutor.run_tool() and tool_trace population."""

    def _make_executor_with_repo(self, tmp_path: Path):
        fake_gen = _make_generator_mock()
        with patch("openshard.native.executor.ExecutionGenerator", return_value=fake_gen):
            executor = NativeAgentExecutor(provider=MagicMock(), repo_root=tmp_path)
        return executor

    def _make_executor_no_repo(self):
        fake_gen = _make_generator_mock()
        with patch("openshard.native.executor.ExecutionGenerator", return_value=fake_gen):
            executor = NativeAgentExecutor(provider=MagicMock())
        return executor

    def test_run_tool_appends_trace_entry(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "a.py").write_text("x = 1")
            executor = self._make_executor_with_repo(root)
            call = NativeToolCall(tool_name="list_files", args={})
            executor.run_tool(call)
        self.assertEqual(len(executor.native_meta.tool_trace), 1)

    def test_run_tool_trace_has_no_output_key(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "b.py").write_text("y = 2")
            executor = self._make_executor_with_repo(root)
            call = NativeToolCall(tool_name="list_files", args={})
            executor.run_tool(call)
        entry = executor.native_meta.tool_trace[0]
        self.assertNotIn("output", entry)

    def test_run_tool_trace_entry_structure(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            executor = self._make_executor_with_repo(root)
            call = NativeToolCall(tool_name="list_files", args={})
            executor.run_tool(call)
        entry = executor.native_meta.tool_trace[0]
        self.assertIn("tool", entry)
        self.assertIn("ok", entry)
        self.assertIn("approved", entry)
        self.assertIn("output_chars", entry)
        self.assertIn("error", entry)

    def test_run_tool_no_repo_root_returns_error_and_appends_trace(self):
        executor = self._make_executor_no_repo()
        call = NativeToolCall(tool_name="list_files", args={})
        result = executor.run_tool(call)
        self.assertFalse(result.ok)
        self.assertIsNotNone(result.error)
        self.assertEqual(len(executor.native_meta.tool_trace), 1)
        self.assertFalse(executor.native_meta.tool_trace[0]["ok"])

    def test_generate_leaves_tool_trace_unchanged(self):
        fake_gen = _make_generator_mock()
        with patch("openshard.native.executor.ExecutionGenerator", return_value=fake_gen):
            executor = NativeAgentExecutor(provider=MagicMock())
        executor.generate("do something")
        self.assertEqual(executor.native_meta.tool_trace, [])

    def test_preflight_stores_repo_context_summary_when_repo_root_set(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "main.py").write_text("x = 1")
            (root / "utils.py").write_text("y = 2")
            fake_gen = _make_generator_mock()
            with patch("openshard.native.executor.ExecutionGenerator", return_value=fake_gen):
                executor = NativeAgentExecutor(provider=MagicMock(), repo_root=root)
            executor.generate("fix bug")
        self.assertIsInstance(executor.native_meta.repo_context_summary, NativeRepoContextSummary)
        self.assertGreater(executor.native_meta.repo_context_summary.total_files, 0)
        self.assertTrue(executor.native_meta.context_budget.repo_map_built)

    def test_preflight_not_called_without_repo_root(self):
        fake_gen = _make_generator_mock()
        with patch("openshard.native.executor.ExecutionGenerator", return_value=fake_gen):
            executor = NativeAgentExecutor(provider=MagicMock())
        executor.generate("fix bug")
        self.assertIsNone(executor.native_meta.repo_context_summary)
        self.assertIsNone(executor.native_meta.context_budget)

    def test_multiple_run_tool_calls_accumulate_trace(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "c.py").write_text("z = 3")
            executor = self._make_executor_with_repo(root)
            executor.run_tool(NativeToolCall(tool_name="list_files", args={}))
            executor.run_tool(NativeToolCall(tool_name="run_command", args={}))
        self.assertEqual(len(executor.native_meta.tool_trace), 2)


class TestNativePreflight(unittest.TestCase):
    """Tests for NativeAgentExecutor._run_preflight() via generate()."""

    def _make_executor_with_repo(self, tmp_path: Path):
        fake_gen = _make_generator_mock()
        with patch("openshard.native.executor.ExecutionGenerator", return_value=fake_gen):
            executor = NativeAgentExecutor(provider=MagicMock(), repo_root=tmp_path)
        return executor, fake_gen

    def _make_executor_no_repo(self):
        fake_gen = _make_generator_mock()
        with patch("openshard.native.executor.ExecutionGenerator", return_value=fake_gen):
            executor = NativeAgentExecutor(provider=MagicMock())
        return executor, fake_gen

    def test_generate_with_repo_root_runs_preflight(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "main.py").write_text("x = 1")
            executor, fake_gen = self._make_executor_with_repo(root)
            executor.generate("fix the bug")
        self.assertGreaterEqual(len(executor.native_meta.tool_trace), 1)
        self.assertEqual(executor.native_meta.tool_trace[0]["tool"], "list_files")

    def test_preflight_records_list_files_in_trace(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            executor, _ = self._make_executor_with_repo(root)
            executor.generate("fix the bug")
        entry = executor.native_meta.tool_trace[0]
        self.assertIn("tool", entry)
        self.assertIn("ok", entry)
        self.assertIn("approved", entry)
        self.assertIn("output_chars", entry)
        self.assertIn("error", entry)

    def test_preflight_does_not_store_full_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "a.py").write_text("a = 1")
            executor, _ = self._make_executor_with_repo(root)
            executor.generate("fix the bug")
        self.assertNotIn("output", executor.native_meta.tool_trace[0])

    def test_context_budget_updated_after_preflight(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            executor, _ = self._make_executor_with_repo(root)
            executor.generate("fix the bug")
        self.assertIsNotNone(executor.native_meta.context_budget)
        self.assertTrue(executor.native_meta.context_budget.repo_map_built)

    def test_context_budget_files_loaded_not_incremented(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "b.py").write_text("b = 2")
            executor, _ = self._make_executor_with_repo(root)
            executor.generate("fix the bug")
        self.assertEqual(executor.native_meta.context_budget.files_loaded, 0)

    def test_generate_without_repo_root_still_works(self):
        executor, fake_gen = self._make_executor_no_repo()
        result = executor.generate("fix the bug")
        self.assertIs(result, fake_gen.generate.return_value)
        self.assertEqual(executor.native_meta.tool_trace, [])
        self.assertIsNone(executor.native_meta.context_budget)


class TestNativeContextInjection(unittest.TestCase):
    """Tests for repo context summary injection into ExecutionGenerator.generate()."""

    def _make_executor_with_repo(self, tmp_path: Path):
        fake_gen = _make_generator_mock()
        with patch("openshard.native.executor.ExecutionGenerator", return_value=fake_gen):
            executor = NativeAgentExecutor(provider=MagicMock(), repo_root=tmp_path)
        return executor, fake_gen

    def _make_executor_no_repo(self):
        fake_gen = _make_generator_mock()
        with patch("openshard.native.executor.ExecutionGenerator", return_value=fake_gen):
            executor = NativeAgentExecutor(provider=MagicMock())
        return executor, fake_gen

    def test_generate_with_repo_root_passes_rendered_context_to_generator(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "main.py").write_text("x = 1")
            executor, fake_gen = self._make_executor_with_repo(root)
            executor.generate("fix the bug")
        _, kwargs = fake_gen.generate.call_args
        self.assertIn("[repo context]", kwargs["skills_context"])

    def test_generate_combines_repo_context_with_existing_skills_context(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "main.py").write_text("x = 1")
            executor, fake_gen = self._make_executor_with_repo(root)
            executor.generate("fix the bug", skills_context="hint")
        _, kwargs = fake_gen.generate.call_args
        self.assertIn("[repo context]", kwargs["skills_context"])
        self.assertIn("hint", kwargs["skills_context"])

    def test_generate_without_repo_root_plan_and_skills_context_present(self):
        executor, fake_gen = self._make_executor_no_repo()
        executor.generate("fix the bug", skills_context="hint")
        _, kwargs = fake_gen.generate.call_args
        self.assertIn("[native plan]", kwargs["skills_context"])
        self.assertIn("hint", kwargs["skills_context"])

    def test_generate_with_repo_root_updates_estimated_tokens(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "main.py").write_text("x = 1")
            executor, _ = self._make_executor_with_repo(root)
            executor.generate("fix the bug")
        self.assertGreater(executor.native_meta.context_budget.estimated_tokens_used, 0)

    def test_generate_without_repo_root_no_context_budget(self):
        executor, _ = self._make_executor_no_repo()
        executor.generate("fix the bug")
        self.assertIsNone(executor.native_meta.context_budget)


class TestNativeWorkflowIntegration(unittest.TestCase):
    """Integration tests via CLI runner — verify routing and log metadata."""

    def _run(self, args, native_mock=None, generator_mock=None):
        native_mock = native_mock or _make_native_mock()
        generator_mock = generator_mock or _make_generator_mock()
        manager_mock = _make_manager_mock()
        logged = {}

        def _capture_log(*a, **kw):
            logged.update(kw.get("extra_metadata") or {})

        with patch("openshard.run.pipeline.NativeAgentExecutor", return_value=native_mock), \
             patch("openshard.run.pipeline.ExecutionGenerator", return_value=generator_mock), \
             patch("openshard.run.pipeline.ProviderManager", return_value=manager_mock), \
             patch("openshard.cli.main.load_config", return_value=_DEFAULT_CONFIG), \
             patch("openshard.run.pipeline.analyze_repo", return_value=_PYTHON_REPO), \
             patch("openshard.run.pipeline._log_run", side_effect=_capture_log):
            runner = CliRunner()
            result = runner.invoke(cli, ["run"] + args)
        return result, native_mock, logged

    def test_native_workflow_selects_native_executor(self):
        """--workflow native instantiates NativeAgentExecutor, not ExecutionGenerator."""
        native_mock = _make_native_mock()
        generator_mock = _make_generator_mock()

        with patch("openshard.run.pipeline.NativeAgentExecutor", return_value=native_mock) as native_cls, \
             patch("openshard.run.pipeline.ExecutionGenerator", return_value=generator_mock) as gen_cls, \
             patch("openshard.run.pipeline.ProviderManager", return_value=_make_manager_mock()), \
             patch("openshard.cli.main.load_config", return_value=_DEFAULT_CONFIG), \
             patch("openshard.run.pipeline.analyze_repo", return_value=_PYTHON_REPO), \
             patch("openshard.run.pipeline._log_run"):
            runner = CliRunner()
            runner.invoke(cli, ["run", "--workflow", "native", "add a feature"])

        native_cls.assert_called_once()
        gen_cls.assert_not_called()

    def test_native_log_contains_metadata_fields(self):
        result, _, logged = self._run(["--workflow", "native", "add a feature"])
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertEqual(logged.get("workflow"), "native")
        self.assertEqual(logged.get("executor"), "native")
        self.assertEqual(logged.get("execution_depth"), "fast")
        self.assertEqual(logged.get("selected_skills"), [])
        self.assertIsNone(logged.get("context_budget"))
        self.assertIsNone(logged.get("context_state"))
        self.assertEqual(logged.get("context_warnings"), [])
        self.assertEqual(logged.get("tool_trace"), [])
        self.assertIn("repo_context_summary", logged)

    def test_native_log_repo_context_summary_is_none_without_repo_root(self):
        result, _, logged = self._run(["--workflow", "native", "add a feature"])
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertIsNone(logged.get("repo_context_summary"))

    def test_native_log_contains_evidence_key(self):
        result, _, logged = self._run(["--workflow", "native", "add a feature"])
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertIn("evidence", logged)
        self.assertIsNone(logged.get("evidence"))

    def test_native_dry_run(self):
        """--workflow native --dry-run exits 0 without writing files."""
        result, native_mock, _ = self._run(["--workflow", "native", "--dry-run", "add a feature"])
        self.assertEqual(result.exit_code, 0, result.output)

    def test_direct_workflow_unchanged(self):
        """--workflow direct still instantiates ExecutionGenerator."""
        generator_mock = _make_generator_mock()
        with patch("openshard.run.pipeline.NativeAgentExecutor") as native_cls, \
             patch("openshard.run.pipeline.ExecutionGenerator", return_value=generator_mock) as gen_cls, \
             patch("openshard.run.pipeline.ProviderManager", return_value=_make_manager_mock()), \
             patch("openshard.cli.main.load_config", return_value=_DEFAULT_CONFIG), \
             patch("openshard.run.pipeline.analyze_repo", return_value=_PYTHON_REPO), \
             patch("openshard.run.pipeline._log_run"):
            runner = CliRunner()
            runner.invoke(cli, ["run", "--workflow", "direct", "add a feature"])

        gen_cls.assert_called_once()
        native_cls.assert_not_called()

    def test_staged_workflow_unchanged(self):
        """--workflow staged still uses ExecutionGenerator."""
        generator_mock = _make_generator_mock()
        with patch("openshard.run.pipeline.NativeAgentExecutor") as native_cls, \
             patch("openshard.run.pipeline.ExecutionGenerator", return_value=generator_mock) as gen_cls, \
             patch("openshard.run.pipeline.ProviderManager", return_value=_make_manager_mock()), \
             patch("openshard.cli.main.load_config", return_value=_DEFAULT_CONFIG), \
             patch("openshard.run.pipeline.analyze_repo", return_value=_PYTHON_REPO), \
             patch("openshard.run.pipeline._log_run"):
            runner = CliRunner()
            runner.invoke(cli, ["run", "--workflow", "staged", "add a feature"])

        gen_cls.assert_called_once()
        native_cls.assert_not_called()

    def test_opencode_workflow_unchanged(self):
        """--workflow opencode still uses OpenCodeExecutor."""
        with patch("openshard.run.pipeline.NativeAgentExecutor") as native_cls, \
             patch("openshard.run.pipeline.ExecutionGenerator") as gen_cls, \
             patch("openshard.run.pipeline.OpenCodeExecutor") as oc_cls, \
             patch("openshard.run.pipeline.ProviderManager", return_value=_make_manager_mock()), \
             patch("openshard.cli.main.load_config", return_value=_DEFAULT_CONFIG), \
             patch("openshard.run.pipeline.analyze_repo", return_value=_PYTHON_REPO), \
             patch("openshard.run.pipeline._log_run"):
            oc_mock = _make_generator_mock()
            oc_cls.return_value = oc_mock
            runner = CliRunner()
            runner.invoke(cli, ["run", "--workflow", "opencode", "add a feature"])

        oc_cls.assert_called_once()
        native_cls.assert_not_called()
        gen_cls.assert_not_called()

    def test_claude_code_still_rejected(self):
        """--workflow claude-code must still raise a ClickException."""
        with patch("openshard.cli.main.load_config", return_value=_DEFAULT_CONFIG):
            runner = CliRunner()
            result = runner.invoke(cli, ["run", "--workflow", "claude-code", "task"])
        self.assertNotEqual(result.exit_code, 0)
        self.assertIn("not yet available", result.output)

    def test_non_native_log_has_no_native_metadata(self):
        """--workflow direct produces no native metadata in the log entry."""
        generator_mock = _make_generator_mock()
        logged = {}

        def _capture_log(*a, **kw):
            logged.update(kw.get("extra_metadata") or {})

        with patch("openshard.run.pipeline.ExecutionGenerator", return_value=generator_mock), \
             patch("openshard.run.pipeline.ProviderManager", return_value=_make_manager_mock()), \
             patch("openshard.cli.main.load_config", return_value=_DEFAULT_CONFIG), \
             patch("openshard.run.pipeline.analyze_repo", return_value=_PYTHON_REPO), \
             patch("openshard.run.pipeline._log_run", side_effect=_capture_log):
            runner = CliRunner()
            runner.invoke(cli, ["run", "--workflow", "direct", "add a feature"])

        self.assertNotIn("workflow", logged)
        self.assertNotIn("execution_depth", logged)


class TestNativeObservePhase(unittest.TestCase):
    """Tests for NativeAgentExecutor._run_observe_phase()."""

    def _run_git(self, root: Path, *args: str) -> None:
        subprocess.run(["git", *args], cwd=root, check=True, capture_output=True, text=True)

    def _make_executor_with_repo(self, tmp_path: Path):
        fake_gen = _make_generator_mock()
        with patch("openshard.native.executor.ExecutionGenerator", return_value=fake_gen):
            executor = NativeAgentExecutor(provider=MagicMock(), repo_root=tmp_path)
        return executor

    def _make_executor_no_repo(self):
        fake_gen = _make_generator_mock()
        with patch("openshard.native.executor.ExecutionGenerator", return_value=fake_gen):
            executor = NativeAgentExecutor(provider=MagicMock())
        return executor

    def test_observe_returns_early_without_runner(self):
        executor = self._make_executor_no_repo()
        executor._run_observe_phase("fix the bug")
        self.assertEqual(executor.native_meta.tool_trace, [])
        self.assertIsNone(executor.native_meta.observation)

    def test_observe_records_get_git_diff_trace_entry(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            executor = self._make_executor_with_repo(root)
            executor._run_observe_phase("fix the bug")
        tools = [e["tool"] for e in executor.native_meta.tool_trace]
        self.assertIn("get_git_diff", tools)

    def test_observe_get_git_diff_trace_has_no_raw_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            executor = self._make_executor_with_repo(root)
            executor._run_observe_phase("fix the bug")
        diff_entry = next(e for e in executor.native_meta.tool_trace if e["tool"] == "get_git_diff")
        self.assertNotIn("output", diff_entry)
        self.assertIn("output_chars", diff_entry)

    def test_observe_runs_search_repo_for_where_task(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "config.py").write_text("DATABASE_URL = 'sqlite:///db'\n")
            executor = self._make_executor_with_repo(root)
            executor._run_observe_phase("where is the database config")
        tools = [e["tool"] for e in executor.native_meta.tool_trace]
        self.assertIn("search_repo", tools)

    def test_observe_runs_search_repo_for_find_task(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "auth.py").write_text("def login(): pass\n")
            executor = self._make_executor_with_repo(root)
            executor._run_observe_phase("find the login function")
        tools = [e["tool"] for e in executor.native_meta.tool_trace]
        self.assertIn("search_repo", tools)

    def test_observe_runs_search_repo_for_search_task(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            executor = self._make_executor_with_repo(root)
            executor._run_observe_phase("search for payment handler")
        tools = [e["tool"] for e in executor.native_meta.tool_trace]
        self.assertIn("search_repo", tools)

    def test_observe_runs_search_repo_for_locate_task(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            executor = self._make_executor_with_repo(root)
            executor._run_observe_phase("locate error handling code")
        tools = [e["tool"] for e in executor.native_meta.tool_trace]
        self.assertIn("search_repo", tools)

    def test_observe_does_not_run_search_repo_for_non_search_task(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            executor = self._make_executor_with_repo(root)
            executor._run_observe_phase("fix the bug in auth.py")
        tools = [e["tool"] for e in executor.native_meta.tool_trace]
        self.assertNotIn("search_repo", tools)

    def test_observe_does_not_run_search_repo_for_refactor_task(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            executor = self._make_executor_with_repo(root)
            executor._run_observe_phase("refactor the database module")
        tools = [e["tool"] for e in executor.native_meta.tool_trace]
        self.assertNotIn("search_repo", tools)

    def test_observe_does_not_run_verification(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            executor = self._make_executor_with_repo(root)
            executor._run_observe_phase("fix the bug")
        tools = [e["tool"] for e in executor.native_meta.tool_trace]
        self.assertNotIn("run_verification", tools)

    def test_observe_stores_native_observation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            executor = self._make_executor_with_repo(root)
            executor._run_observe_phase("fix the bug")
        self.assertIsInstance(executor.native_meta.observation, NativeObservation)

    def test_observe_observation_observed_tools_contains_get_git_diff(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            executor = self._make_executor_with_repo(root)
            executor._run_observe_phase("fix the bug")
        self.assertIn("get_git_diff", executor.native_meta.observation.observed_tools)

    def test_observe_dirty_diff_present_true_when_diff_has_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._run_git(root, "init")
            self._run_git(root, "config", "user.email", "test@test.com")
            self._run_git(root, "config", "user.name", "Test")
            (root / "hello.txt").write_text("original\n")
            self._run_git(root, "add", "hello.txt")
            (root / "hello.txt").write_text("modified\n")
            executor = self._make_executor_with_repo(root)
            executor._run_observe_phase("fix the bug")
        self.assertTrue(executor.native_meta.observation.dirty_diff_present)

    def test_observe_dirty_diff_present_false_for_non_git_repo(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            executor = self._make_executor_with_repo(root)
            executor._run_observe_phase("fix the bug")
        self.assertFalse(executor.native_meta.observation.dirty_diff_present)

    def test_observe_search_matches_count_reflects_actual_matches(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "main.py").write_text("def login(): pass\ndef logout(): pass\n")
            executor = self._make_executor_with_repo(root)
            # "find login" → query = "login", which matches the file content
            executor._run_observe_phase("find login")
        self.assertGreater(executor.native_meta.observation.search_matches_count, 0)

    def test_observe_search_matches_count_zero_for_non_search_task(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            executor = self._make_executor_with_repo(root)
            executor._run_observe_phase("fix the bug")
        self.assertEqual(executor.native_meta.observation.search_matches_count, 0)

    def test_generate_still_delegates_to_execution_generator(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fake_gen = _make_generator_mock()
            with patch("openshard.native.executor.ExecutionGenerator", return_value=fake_gen):
                executor = NativeAgentExecutor(provider=MagicMock(), repo_root=root)
            result = executor.generate("fix the bug")
        fake_gen.generate.assert_called_once()
        self.assertIs(result, fake_gen.generate.return_value)

    def test_generate_sets_observation_after_observe_phase(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "main.py").write_text("x = 1")
            fake_gen = _make_generator_mock()
            with patch("openshard.native.executor.ExecutionGenerator", return_value=fake_gen):
                executor = NativeAgentExecutor(provider=MagicMock(), repo_root=root)
            executor.generate("fix the bug")
        self.assertIsInstance(executor.native_meta.observation, NativeObservation)

    def test_generate_observation_is_none_without_repo_root(self):
        fake_gen = _make_generator_mock()
        with patch("openshard.native.executor.ExecutionGenerator", return_value=fake_gen):
            executor = NativeAgentExecutor(provider=MagicMock())
        executor.generate("fix the bug")
        self.assertIsNone(executor.native_meta.observation)

    def test_observe_search_repo_trace_has_no_raw_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "auth.py").write_text("def login(): pass\n")
            executor = self._make_executor_with_repo(root)
            executor._run_observe_phase("find the login function")
        search_entry = next(e for e in executor.native_meta.tool_trace if e["tool"] == "search_repo")
        self.assertNotIn("output", search_entry)
        self.assertIn("output_chars", search_entry)


class TestExtractSearchQuery(unittest.TestCase):
    """Unit tests for the _extract_search_query() helper."""

    def test_returns_none_for_empty_string(self):
        self.assertIsNone(_extract_search_query(""))

    def test_returns_none_when_only_stop_and_trigger_words(self):
        self.assertIsNone(_extract_search_query("find the a"))

    def test_returns_none_when_remaining_words_too_short(self):
        self.assertIsNone(_extract_search_query("find is db"))

    def test_basic_extraction(self):
        result = _extract_search_query("find the login function")
        self.assertIsNotNone(result)
        self.assertIn("login", result)

    def test_trigger_word_removed(self):
        result = _extract_search_query("where is the config file")
        self.assertIsNotNone(result)
        self.assertNotIn("where", result.split())

    def test_stop_words_removed(self):
        result = _extract_search_query("search for the payment handler")
        self.assertIsNotNone(result)
        self.assertNotIn("the", result.split())
        self.assertNotIn("for", result.split())

    def test_takes_at_most_three_words(self):
        result = _extract_search_query("find login authentication middleware service")
        self.assertIsNotNone(result)
        self.assertLessEqual(len(result.split()), 3)

    def test_minimum_word_length_filter(self):
        result = _extract_search_query("find db api config")
        self.assertIsNotNone(result)
        self.assertNotIn("db", result.split())
        self.assertIn("api", result.split())

    def test_case_insensitive(self):
        result = _extract_search_query("Find THE Config File")
        self.assertIsNotNone(result)
        self.assertEqual(result, result.lower())

    def test_punctuation_stripped(self):
        result = _extract_search_query("where is config?")
        self.assertIsNotNone(result)
        self.assertNotIn("config?", result.split())
        self.assertIn("config", result.split())

    def test_punctuation_stripped_various(self):
        result = _extract_search_query("find (login) function.")
        self.assertIsNotNone(result)
        for word in result.split():
            self.assertFalse(any(c in word for c in ".,:;!?()[]{}\"'`"))


class TestRenderNativeObservation(unittest.TestCase):
    """Unit tests for render_native_observation()."""

    def _obs(self, **kwargs) -> NativeObservation:
        defaults = dict(
            observed_tools=[],
            dirty_diff_present=False,
            search_matches_count=0,
            verification_available=False,
            warnings=[],
        )
        defaults.update(kwargs)
        return NativeObservation(**defaults)

    def test_starts_with_observation_header(self):
        result = render_native_observation(self._obs())
        self.assertEqual(result.splitlines()[0], "[observation]")

    def test_includes_tools(self):
        result = render_native_observation(self._obs(observed_tools=["get_git_diff"]))
        self.assertIn("tools: get_git_diff", result)

    def test_no_tools_line_when_empty(self):
        result = render_native_observation(self._obs(observed_tools=[]))
        self.assertNotIn("tools:", result)

    def test_dirty_diff_yes(self):
        result = render_native_observation(self._obs(dirty_diff_present=True))
        self.assertIn("dirty diff: yes", result)

    def test_dirty_diff_no(self):
        result = render_native_observation(self._obs(dirty_diff_present=False))
        self.assertIn("dirty diff: no", result)

    def test_search_matches(self):
        result = render_native_observation(self._obs(search_matches_count=3))
        self.assertIn("search matches: 3", result)

    def test_verification_available_yes(self):
        result = render_native_observation(self._obs(verification_available=True))
        self.assertIn("verification available: yes", result)

    def test_verification_available_no(self):
        result = render_native_observation(self._obs(verification_available=False))
        self.assertIn("verification available: no", result)

    def test_warnings_capped_at_3(self):
        obs = self._obs(warnings=["w1", "w2", "w3", "w4", "w5"])
        result = render_native_observation(obs)
        self.assertIn("w1", result)
        self.assertIn("w3", result)
        self.assertNotIn("w4", result)
        self.assertNotIn("w5", result)

    def test_no_warnings_line_when_empty(self):
        result = render_native_observation(self._obs(warnings=[]))
        self.assertNotIn("warnings:", result)

    def test_bounded_by_limit(self):
        obs = self._obs(observed_tools=["get_git_diff", "search_repo"], warnings=["x" * 200])
        result = render_native_observation(obs, limit=50)
        self.assertLessEqual(len(result), 50)

    def test_truncation_marker_appears(self):
        obs = self._obs(observed_tools=["get_git_diff"], warnings=["x" * 200])
        result = render_native_observation(obs, limit=50)
        self.assertIn("[truncated]", result)

    def test_no_truncation_marker_when_within_limit(self):
        result = render_native_observation(self._obs())
        self.assertNotIn("[truncated]", result)

    def test_empty_observation_renders_minimal_block(self):
        result = render_native_observation(self._obs())
        self.assertIn("[observation]", result)
        self.assertIn("dirty diff: no", result)
        self.assertIn("search matches: 0", result)


class TestNativeObservationInjection(unittest.TestCase):
    """Tests that render_native_observation output is injected into generate()."""

    def _make_executor_with_repo(self, tmp_path: Path):
        fake_gen = _make_generator_mock()
        with patch("openshard.native.executor.ExecutionGenerator", return_value=fake_gen):
            executor = NativeAgentExecutor(provider=MagicMock(), repo_root=tmp_path)
        return executor, fake_gen

    def _make_executor_no_repo(self):
        fake_gen = _make_generator_mock()
        with patch("openshard.native.executor.ExecutionGenerator", return_value=fake_gen):
            executor = NativeAgentExecutor(provider=MagicMock())
        return executor, fake_gen

    def test_observation_injected_with_repo_root(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "main.py").write_text("x = 1")
            executor, fake_gen = self._make_executor_with_repo(root)
            executor.generate("fix the bug")
        _, kwargs = fake_gen.generate.call_args
        self.assertIn("[observation]", kwargs["skills_context"])

    def test_context_order_repo_then_observation_then_skills(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "main.py").write_text("x = 1")
            executor, fake_gen = self._make_executor_with_repo(root)
            executor.generate("fix the bug", skills_context="user-skills")
        _, kwargs = fake_gen.generate.call_args
        ctx = kwargs["skills_context"]
        repo_pos = ctx.index("[repo context]")
        obs_pos = ctx.index("[observation]")
        skills_pos = ctx.index("user-skills")
        self.assertLess(repo_pos, obs_pos)
        self.assertLess(obs_pos, skills_pos)

    def test_no_observation_without_repo_root(self):
        executor, fake_gen = self._make_executor_no_repo()
        executor.generate("fix the bug", skills_context="hint")
        _, kwargs = fake_gen.generate.call_args
        self.assertNotIn("[observation]", kwargs["skills_context"])

    def test_raw_diff_not_in_context(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "main.py").write_text("x = 1")
            executor, fake_gen = self._make_executor_with_repo(root)
            executor.generate("fix the bug")
        _, kwargs = fake_gen.generate.call_args
        ctx = kwargs["skills_context"]
        self.assertNotIn("diff --git", ctx)
        self.assertNotIn("@@", ctx)

    def test_raw_search_not_in_context(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "config.py").write_text("DATABASE_URL = 'sqlite:///db'\n")
            executor, fake_gen = self._make_executor_with_repo(root)
            executor.generate("where is the database config")
        _, kwargs = fake_gen.generate.call_args
        ctx = kwargs["skills_context"]
        self.assertNotIn("DATABASE_URL", ctx)

    def test_context_budget_populated_with_observation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "main.py").write_text("x = 1")
            executor, _ = self._make_executor_with_repo(root)
            executor.generate("fix the bug")
        self.assertIsNotNone(executor.native_meta.context_budget)
        self.assertGreater(executor.native_meta.context_budget.estimated_tokens_used, 0)
        self.assertIsNotNone(executor.native_meta.observation)


class TestRenderNativeEvidence(unittest.TestCase):
    """Unit tests for render_native_evidence()."""

    def _ev(self, **kwargs) -> NativeEvidence:
        defaults = dict(search_results=[], truncated=False)
        defaults.update(kwargs)
        return NativeEvidence(**defaults)

    def test_starts_with_evidence_header(self):
        result = render_native_evidence(self._ev())
        self.assertEqual(result.splitlines()[0], "[evidence]")

    def test_includes_search_results(self):
        result = render_native_evidence(self._ev(search_results=["src/auth.py:12: def login_user(*)"]))
        self.assertIn("- src/auth.py:12: def login_user(*)", result)
        self.assertIn("search:", result)

    def test_caps_search_results_at_3(self):
        items = [f"file.py:{i}: line" for i in range(5)]
        result = render_native_evidence(self._ev(search_results=items))
        self.assertIn("file.py:0:", result)
        self.assertIn("file.py:2:", result)
        self.assertNotIn("file.py:3:", result)
        self.assertNotIn("file.py:4:", result)

    def test_truncated_flag_adds_marker(self):
        result = render_native_evidence(self._ev(truncated=True))
        self.assertIn("[truncated]", result)

    def test_no_truncated_marker_when_false(self):
        result = render_native_evidence(self._ev(truncated=False))
        self.assertNotIn("[truncated]", result)

    def test_bounded_by_limit(self):
        items = ["x" * 100 for _ in range(3)]
        result = render_native_evidence(self._ev(search_results=items), limit=50)
        self.assertLessEqual(len(result), 50)

    def test_truncation_suffix_on_char_overflow(self):
        items = ["x" * 100 for _ in range(3)]
        result = render_native_evidence(self._ev(search_results=items), limit=50)
        self.assertIn("[truncated]", result)

    def test_empty_evidence_renders_header_only(self):
        result = render_native_evidence(self._ev())
        self.assertIn("[evidence]", result)
        self.assertNotIn("search:", result)


class TestNativeEvidenceObservePhase(unittest.TestCase):
    """Tests for evidence population in _run_observe_phase()."""

    def _make_executor_with_repo(self, tmp_path: Path):
        fake_gen = _make_generator_mock()
        with patch("openshard.native.executor.ExecutionGenerator", return_value=fake_gen):
            executor = NativeAgentExecutor(provider=MagicMock(), repo_root=tmp_path)
        return executor

    def _make_executor_no_repo(self):
        fake_gen = _make_generator_mock()
        with patch("openshard.native.executor.ExecutionGenerator", return_value=fake_gen):
            executor = NativeAgentExecutor(provider=MagicMock())
        return executor

    def test_observe_stores_evidence_for_search_task(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "auth.py").write_text("def login_user(): pass\n")
            executor = self._make_executor_with_repo(root)
            executor._run_observe_phase("where is the login function")
        self.assertIsInstance(executor.native_meta.evidence, NativeEvidence)

    def test_observe_evidence_none_for_non_search_task(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            executor = self._make_executor_with_repo(root)
            executor._run_observe_phase("fix the bug")
        self.assertIsNone(executor.native_meta.evidence)

    def test_observe_evidence_has_at_most_3_lines(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for i in range(10):
                (root / f"file{i}.py").write_text(f"def func_{i}(): pass\n")
            executor = self._make_executor_with_repo(root)
            executor._run_observe_phase("where is the func")
        if executor.native_meta.evidence is not None:
            self.assertLessEqual(len(executor.native_meta.evidence.search_results), 3)

    def test_observe_evidence_does_not_include_diff_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "auth.py").write_text("def login(): pass\n")
            executor = self._make_executor_with_repo(root)
            executor._run_observe_phase("where is the login")
        if executor.native_meta.evidence is not None:
            for line in executor.native_meta.evidence.search_results:
                self.assertNotIn("diff --git", line)
                self.assertNotIn("@@", line)

    def test_observe_evidence_truncated_when_more_than_3_lines(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for i in range(10):
                (root / f"mod{i}.py").write_text(f"def target_func_{i}(): pass\n")
            executor = self._make_executor_with_repo(root)
            executor._run_observe_phase("where is the target func")
        if executor.native_meta.evidence is not None and executor.native_meta.observation is not None:
            if executor.native_meta.observation.search_matches_count > 3:
                self.assertTrue(executor.native_meta.evidence.truncated)


class TestNativeEvidenceInjection(unittest.TestCase):
    """Tests that evidence is injected into generate() context."""

    def _make_executor_with_repo(self, tmp_path: Path):
        fake_gen = _make_generator_mock()
        with patch("openshard.native.executor.ExecutionGenerator", return_value=fake_gen):
            executor = NativeAgentExecutor(provider=MagicMock(), repo_root=tmp_path)
        return executor, fake_gen

    def _make_executor_no_repo(self):
        fake_gen = _make_generator_mock()
        with patch("openshard.native.executor.ExecutionGenerator", return_value=fake_gen):
            executor = NativeAgentExecutor(provider=MagicMock())
        return executor, fake_gen

    def test_evidence_injected_for_search_task(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "auth.py").write_text("def login_user(): pass\n")
            executor, fake_gen = self._make_executor_with_repo(root)
            executor.generate("where is the login function")
        _, kwargs = fake_gen.generate.call_args
        self.assertIn("[evidence]", kwargs["skills_context"])

    def test_evidence_not_injected_for_non_search_task(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "main.py").write_text("x = 1")
            executor, fake_gen = self._make_executor_with_repo(root)
            executor.generate("fix the bug")
        _, kwargs = fake_gen.generate.call_args
        self.assertNotIn("[evidence]", kwargs["skills_context"])

    def test_context_order_repo_observation_evidence_skills(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "auth.py").write_text("def login_user(): pass\n")
            executor, fake_gen = self._make_executor_with_repo(root)
            executor.generate("where is the login function", skills_context="user-skills")
        _, kwargs = fake_gen.generate.call_args
        ctx = kwargs["skills_context"]
        repo_pos = ctx.index("[repo context]")
        obs_pos = ctx.index("[observation]")
        ev_pos = ctx.index("[evidence]")
        skills_pos = ctx.index("user-skills")
        self.assertLess(repo_pos, obs_pos)
        self.assertLess(obs_pos, ev_pos)
        self.assertLess(ev_pos, skills_pos)

    def test_no_evidence_without_repo_root(self):
        executor, fake_gen = self._make_executor_no_repo()
        executor.generate("where is the login", skills_context="hint")
        _, kwargs = fake_gen.generate.call_args
        self.assertNotIn("[evidence]", kwargs["skills_context"])

    def test_token_estimate_includes_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "auth.py").write_text("def login_user(): pass\n")
            executor, _ = self._make_executor_with_repo(root)
            executor.generate("where is the login function")
        self.assertIsNotNone(executor.native_meta.context_budget)
        self.assertGreater(executor.native_meta.context_budget.estimated_tokens_used, 0)
        self.assertIsNotNone(executor.native_meta.evidence)


class TestRenderNativePlan(unittest.TestCase):
    """Tests for render_native_plan()."""

    def test_renders_header(self):
        plan = NativePlan()
        self.assertIn("[native plan]", render_native_plan(plan))

    def test_renders_intent(self):
        plan = NativePlan(intent="search")
        self.assertIn("intent: search", render_native_plan(plan))

    def test_renders_risk(self):
        plan = NativePlan(risk="medium")
        self.assertIn("risk: medium", render_native_plan(plan))

    def test_renders_suggested_steps(self):
        plan = NativePlan(suggested_steps=["step one", "step two"])
        rendered = render_native_plan(plan)
        self.assertIn("suggested steps:", rendered)
        self.assertIn("- step one", rendered)
        self.assertIn("- step two", rendered)

    def test_caps_steps_at_five(self):
        plan = NativePlan(suggested_steps=[f"step {i}" for i in range(10)])
        rendered = render_native_plan(plan)
        self.assertIn("- step 4", rendered)
        self.assertNotIn("- step 5", rendered)

    def test_renders_warnings(self):
        plan = NativePlan(warnings=["watch out"])
        rendered = render_native_plan(plan)
        self.assertIn("warnings:", rendered)
        self.assertIn("- watch out", rendered)

    def test_caps_warnings_at_three(self):
        plan = NativePlan(warnings=[f"warn {i}" for i in range(5)])
        rendered = render_native_plan(plan)
        self.assertIn("- warn 2", rendered)
        self.assertNotIn("- warn 3", rendered)

    def test_respects_limit(self):
        plan = NativePlan(suggested_steps=["x" * 200] * 5)
        rendered = render_native_plan(plan, limit=100)
        self.assertLessEqual(len(rendered), 100)

    def test_truncation_marker_when_limit_exceeded(self):
        plan = NativePlan(suggested_steps=["x" * 200] * 5)
        rendered = render_native_plan(plan, limit=100)
        self.assertIn("[truncated]", rendered)

    def test_no_truncation_when_within_limit(self):
        plan = NativePlan(intent="standard", risk="low")
        rendered = render_native_plan(plan, limit=600)
        self.assertNotIn("[truncated]", rendered)


class TestBuildNativePlan(unittest.TestCase):
    """Tests for _build_native_plan()."""

    def test_search_intent(self):
        for word in ("search", "find", "where", "locate"):
            with self.subTest(word=word):
                plan = _build_native_plan(f"please {word} the module")
                self.assertEqual(plan.intent, "search")

    def test_debug_intent(self):
        for word in ("debug", "fix", "error", "fail", "verify", "test"):
            with self.subTest(word=word):
                plan = _build_native_plan(f"please {word} this")
                self.assertEqual(plan.intent, "debug")

    def test_refactor_intent(self):
        plan = _build_native_plan("refactor the auth module")
        self.assertEqual(plan.intent, "refactor")

    def test_implementation_intent(self):
        for word in ("add", "create", "implement", "build"):
            with self.subTest(word=word):
                plan = _build_native_plan(f"please {word} a feature")
                self.assertEqual(plan.intent, "implementation")

    def test_standard_intent_default(self):
        plan = _build_native_plan("do something useful")
        self.assertEqual(plan.intent, "standard")

    def test_word_boundary_matching(self):
        # "searchable" should NOT match "search"
        plan = _build_native_plan("make it searchable")
        self.assertNotEqual(plan.intent, "search")

    def test_security_skill_sets_medium_risk(self):
        plan = _build_native_plan("update the config", selected_skills=["security-sensitive-change"])
        self.assertEqual(plan.risk, "medium")

    def test_security_skill_adds_warning(self):
        plan = _build_native_plan("update the config", selected_skills=["security-sensitive-change"])
        self.assertIn("security-sensitive task", plan.warnings)

    def test_dirty_diff_sets_medium_risk(self):
        obs = NativeObservation(dirty_diff_present=True)
        plan = _build_native_plan("update the config", observation=obs)
        self.assertEqual(plan.risk, "medium")

    def test_dirty_diff_adds_warning(self):
        obs = NativeObservation(dirty_diff_present=True)
        plan = _build_native_plan("update the config", observation=obs)
        self.assertIn("dirty working tree detected", plan.warnings)

    def test_clean_diff_is_low_risk(self):
        obs = NativeObservation(dirty_diff_present=False)
        plan = _build_native_plan("update the config", observation=obs)
        self.assertEqual(plan.risk, "low")

    def test_evidence_prepends_step(self):
        ev = NativeEvidence(search_results=["some result"])
        plan = _build_native_plan("find the thing", evidence=ev)
        self.assertEqual(plan.suggested_steps[0], "use bounded search evidence to choose files")

    def test_no_evidence_no_prepend(self):
        plan = _build_native_plan("find the thing", evidence=None)
        self.assertNotIn("use bounded search evidence to choose files", plan.suggested_steps)

    def test_debug_intent_appends_verification_step(self):
        plan = _build_native_plan("fix the bug")
        self.assertIn("run verification after changes", plan.suggested_steps)

    def test_non_debug_no_verification_step(self):
        plan = _build_native_plan("add a feature")
        self.assertNotIn("run verification after changes", plan.suggested_steps)

    def test_dirty_diff_appends_avoid_overwrite_step(self):
        obs = NativeObservation(dirty_diff_present=True)
        plan = _build_native_plan("update config", observation=obs)
        self.assertIn("avoid overwriting existing user changes", plan.suggested_steps)

    def test_always_includes_base_steps(self):
        plan = _build_native_plan("do something")
        self.assertIn("inspect relevant files", plan.suggested_steps)
        self.assertIn("make the smallest safe change", plan.suggested_steps)
        self.assertIn("review the diff", plan.suggested_steps)


class TestNativePlanInjection(unittest.TestCase):
    """Tests that native plan is injected into generate() context."""

    def _make_executor_with_repo(self, tmp_path: Path):
        fake_gen = _make_generator_mock()
        with patch("openshard.native.executor.ExecutionGenerator", return_value=fake_gen):
            executor = NativeAgentExecutor(provider=MagicMock(), repo_root=tmp_path)
        return executor, fake_gen

    def _make_executor_no_repo(self):
        fake_gen = _make_generator_mock()
        with patch("openshard.native.executor.ExecutionGenerator", return_value=fake_gen):
            executor = NativeAgentExecutor(provider=MagicMock())
        return executor, fake_gen

    def test_plan_injected_with_repo_root(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "main.py").write_text("x = 1\n")
            executor, fake_gen = self._make_executor_with_repo(root)
            executor.generate("fix the bug")
        _, kwargs = fake_gen.generate.call_args
        self.assertIn("[native plan]", kwargs["skills_context"])

    def test_context_order_repo_observation_evidence_plan_skills(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "auth.py").write_text("def login_user(): pass\n")
            executor, fake_gen = self._make_executor_with_repo(root)
            executor.generate("where is the login function", skills_context="user-skills")
        _, kwargs = fake_gen.generate.call_args
        ctx = kwargs["skills_context"]
        repo_pos = ctx.index("[repo context]")
        obs_pos = ctx.index("[observation]")
        ev_pos = ctx.index("[evidence]")
        plan_pos = ctx.index("[native plan]")
        skills_pos = ctx.index("user-skills")
        self.assertLess(repo_pos, obs_pos)
        self.assertLess(obs_pos, ev_pos)
        self.assertLess(ev_pos, plan_pos)
        self.assertLess(plan_pos, skills_pos)

    def test_no_repo_root_plan_present_others_absent(self):
        executor, fake_gen = self._make_executor_no_repo()
        executor.generate("fix the bug", skills_context="hint")
        _, kwargs = fake_gen.generate.call_args
        ctx = kwargs["skills_context"]
        self.assertIn("[native plan]", ctx)
        self.assertNotIn("[repo context]", ctx)
        self.assertNotIn("[observation]", ctx)
        self.assertNotIn("[evidence]", ctx)

    def test_plan_stored_on_native_meta(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "main.py").write_text("x = 1\n")
            executor, _ = self._make_executor_with_repo(root)
            executor.generate("fix the bug")
        self.assertIsNotNone(executor.native_meta.plan)


class TestNativePlanMetaSerialization(unittest.TestCase):
    """Tests that plan is serialized correctly in native_meta."""

    def _make_executor_with_repo(self, tmp_path: Path):
        fake_gen = _make_generator_mock()
        with patch("openshard.native.executor.ExecutionGenerator", return_value=fake_gen):
            executor = NativeAgentExecutor(provider=MagicMock(), repo_root=tmp_path)
        return executor, fake_gen

    def test_plan_has_expected_fields(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "main.py").write_text("x = 1\n")
            executor, _ = self._make_executor_with_repo(root)
            executor.generate("fix the bug")
        plan = executor.native_meta.plan
        self.assertIsNotNone(plan)
        self.assertIsInstance(plan.intent, str)
        self.assertIsInstance(plan.risk, str)
        self.assertIsInstance(plan.suggested_steps, list)
        self.assertIsInstance(plan.warnings, list)

    def test_plan_fields_contain_no_raw_tool_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "main.py").write_text("x = 1\n")
            executor, _ = self._make_executor_with_repo(root)
            executor.generate("fix the bug")
        plan = executor.native_meta.plan
        for step in plan.suggested_steps:
            self.assertNotIn("diff --git", step)
            self.assertNotIn("@@", step)
        for warning in plan.warnings:
            self.assertNotIn("diff --git", warning)

    def test_asdict_produces_expected_keys(self):
        from dataclasses import asdict
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "main.py").write_text("x = 1\n")
            executor, _ = self._make_executor_with_repo(root)
            executor.generate("add a feature")
        d = asdict(executor.native_meta.plan)
        self.assertIn("intent", d)
        self.assertIn("risk", d)
        self.assertIn("suggested_steps", d)
        self.assertIn("warnings", d)
