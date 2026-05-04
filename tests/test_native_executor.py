from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

from click.testing import CliRunner

from openshard.analysis.repo import RepoFacts
from openshard.cli.main import cli
from openshard.native.executor import NativeAgentExecutor, NativeRunMeta

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
        self.assertEqual(meta.tool_trace, [])


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
        self.assertEqual(logged.get("tool_trace"), [])

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
