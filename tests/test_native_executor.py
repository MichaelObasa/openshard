from __future__ import annotations

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
    build_compact_run_state,
    build_initial_context_budget,
)
from openshard.native.executor import NativeAgentExecutor, NativeRunMeta
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
        fake_gen.generate.assert_called_once_with(
            "fix the bug", model="some-model", repo_facts=None, skills_context=""
        )
        self.assertIs(result, fake_gen.generate.return_value)

    def test_generate_passes_repo_facts_and_skills(self):
        executor, fake_gen = self._make_executor()
        executor.generate("task", repo_facts=_PYTHON_REPO, skills_context="hint")
        fake_gen.generate.assert_called_once_with(
            "task", model=None, repo_facts=_PYTHON_REPO, skills_context="hint"
        )

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
