from __future__ import annotations

import unittest
from dataclasses import asdict
from unittest.mock import MagicMock, patch

from click.testing import CliRunner

from openshard.analysis.repo import RepoFacts
from openshard.cli.main import cli
from openshard.native.context import OSNLoopMeta, OSNLoopStep

_DEFAULT_CONFIG = {"approval_mode": "smart"}

_PYTHON_REPO = RepoFacts(
    languages=["python"], package_files=[], framework=None,
    test_command=None, risky_paths=[], changed_files=[],
)


def _make_native_mock(osn_loop=None):
    from openshard.native.executor import NativeRunMeta
    from openshard.native.context import NativeChangeBudgetSoftGate, NativeApprovalRequest
    g = MagicMock()
    g.generate.return_value = MagicMock(usage=None, files=[], summary="done", notes=[])
    g.model = "mock-model"
    g.fixer_model = "mock-fixer"
    meta = NativeRunMeta()
    meta.osn_loop = osn_loop
    g.native_meta = meta
    g.build_change_budget_soft_gate.return_value = NativeChangeBudgetSoftGate(
        requires_approval=False, reason="", action="allow"
    )
    g.build_budget_gate_approval_request.return_value = NativeApprovalRequest(
        source="change_budget_soft_gate",
        requires_approval=False,
        reason="",
        action="allow",
    )
    return g


def _make_manager_mock():
    m = MagicMock()
    inv = MagicMock()
    inv.models = []
    m.get_inventory.return_value = inv
    m.providers = {"openrouter": MagicMock()}
    return m


class TestNativeLoopCLIFlag(unittest.TestCase):
    """CLI validation and wiring for --native-loop experimental."""

    def _invoke(self, args):
        with patch("openshard.cli.main.load_config", return_value=_DEFAULT_CONFIG):
            runner = CliRunner()
            return runner.invoke(cli, ["run"] + args)

    def _invoke_with_pipeline_mock(self, args, native_mock=None):
        if native_mock is None:
            native_mock = _make_native_mock()
        logged = {}

        def _capture_log(*a, **kw):
            logged.update(kw.get("extra_metadata") or {})

        with patch("openshard.run.pipeline.NativeAgentExecutor", return_value=native_mock) as native_cls, \
             patch("openshard.run.pipeline.ProviderManager", return_value=_make_manager_mock()), \
             patch("openshard.cli.main.load_config", return_value=_DEFAULT_CONFIG), \
             patch("openshard.run.pipeline.analyze_repo", return_value=_PYTHON_REPO), \
             patch("openshard.run.pipeline._log_run", side_effect=_capture_log):
            runner = CliRunner()
            result = runner.invoke(cli, ["run"] + args)
        return result, native_mock, native_cls, logged

    # --- flag validation ---

    def test_native_loop_without_native_workflow_exits_nonzero(self):
        result = self._invoke(["--native-loop", "experimental", "fix the bug"])
        self.assertNotEqual(result.exit_code, 0)

    def test_native_loop_without_native_workflow_error_message(self):
        result = self._invoke(["--native-loop", "experimental", "fix the bug"])
        self.assertIn("--native-loop experimental requires --workflow native", result.output)

    def test_native_loop_without_native_workflow_pipeline_not_constructed(self):
        with patch("openshard.run.pipeline.RunPipeline") as pipeline_cls, \
             patch("openshard.cli.main.load_config", return_value=_DEFAULT_CONFIG):
            runner = CliRunner()
            runner.invoke(cli, ["run", "--native-loop", "experimental", "fix the bug"])
        pipeline_cls.assert_not_called()

    def test_native_loop_with_wrong_workflow_direct_pipeline_not_constructed(self):
        with patch("openshard.run.pipeline.RunPipeline") as pipeline_cls, \
             patch("openshard.cli.main.load_config", return_value=_DEFAULT_CONFIG):
            runner = CliRunner()
            runner.invoke(cli, ["run", "--workflow", "direct", "--native-loop", "experimental", "fix the bug"])
        pipeline_cls.assert_not_called()

    def test_native_loop_with_native_workflow_no_error(self):
        result, _, _, _ = self._invoke_with_pipeline_mock(
            ["--workflow", "native", "--native-loop", "experimental", "fix the bug"]
        )
        self.assertEqual(result.exit_code, 0, result.output)

    def test_native_loop_not_passed_no_error(self):
        result, _, _, _ = self._invoke_with_pipeline_mock(
            ["--workflow", "native", "fix the bug"]
        )
        self.assertEqual(result.exit_code, 0, result.output)

    # --- wiring: native_loop passes through to NativeAgentExecutor ---

    def test_native_loop_experimental_passed_to_executor(self):
        result, _, native_cls, _ = self._invoke_with_pipeline_mock(
            ["--workflow", "native", "--native-loop", "experimental", "fix the bug"]
        )
        self.assertEqual(result.exit_code, 0, result.output)
        call_kwargs = native_cls.call_args[1] if native_cls.call_args else {}
        self.assertEqual(call_kwargs.get("native_loop"), "experimental")

    def test_native_loop_absent_passes_none_to_executor(self):
        result, _, native_cls, _ = self._invoke_with_pipeline_mock(
            ["--workflow", "native", "fix the bug"]
        )
        self.assertEqual(result.exit_code, 0, result.output)
        call_kwargs = native_cls.call_args[1] if native_cls.call_args else {}
        self.assertIsNone(call_kwargs.get("native_loop"))

    # --- serialization: osn_loop key in logged run ---

    def test_log_contains_osn_loop_key_when_none(self):
        result, _, _, logged = self._invoke_with_pipeline_mock(
            ["--workflow", "native", "fix the bug"]
        )
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertIn("osn_loop", logged)
        self.assertIsNone(logged["osn_loop"])

    def test_log_contains_osn_loop_dict_when_loop_ran(self):
        loop = OSNLoopMeta(
            enabled=True, steps_run=2, steps_queued=3, max_steps=5,
            terminated_reason="complete",
            steps=[
                OSNLoopStep(step_index=0, tool_name="read_file", target_label="src/foo.py",
                            reason="read_search_finding", ok=True, output_chars=80),
                OSNLoopStep(step_index=1, tool_name="search_repo", target_label="task_keyword_search",
                            reason="task_keyword", ok=True, output_chars=120),
            ],
            paths_surfaced=["src/foo.py"],
        )
        native_mock = _make_native_mock(osn_loop=loop)
        result, _, _, logged = self._invoke_with_pipeline_mock(
            ["--workflow", "native", "--native-loop", "experimental", "fix the bug"],
            native_mock=native_mock,
        )
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertIn("osn_loop", logged)
        self.assertIsNotNone(logged["osn_loop"])
        self.assertTrue(logged["osn_loop"]["enabled"])
        self.assertEqual(logged["osn_loop"]["steps_run"], 2)
        self.assertEqual(logged["osn_loop"]["terminated_reason"], "complete")
        self.assertEqual(logged["osn_loop"]["paths_surfaced"], ["src/foo.py"])


class TestNativeLoopRenderingOldRuns(unittest.TestCase):
    """_render_native_demo_block handles old log entries without osn_loop key."""

    def _entry_without_osn_loop(self):
        return {
            "task": "fix the bug",
            "workflow": "native",
            "executor": "native",
            "execution_depth": "fast",
            "selected_skills": [],
            "tool_trace": [],
            "native_loop_steps": ["plan"],
            "native_loop_trace": {"events": []},
            "read_search_findings": [],
        }

    def _entry_with_osn_loop_none(self):
        e = self._entry_without_osn_loop()
        e["osn_loop"] = None
        return e

    def _entry_with_osn_loop(self, loop_dict):
        e = self._entry_without_osn_loop()
        e["osn_loop"] = loop_dict
        return e

    def _render(self, entry):
        from openshard.cli.run_output import _render_native_demo_block, _native_meta_from_entry
        meta = _native_meta_from_entry(entry)
        lines = _render_native_demo_block(meta, detail="default")
        return "\n".join(lines)

    def test_old_entry_no_osn_loop_key_no_crash(self):
        entry = self._entry_without_osn_loop()
        rendered = self._render(entry)
        self.assertIsInstance(rendered, str)

    def test_old_entry_no_osn_loop_key_no_osn_section(self):
        entry = self._entry_without_osn_loop()
        rendered = self._render(entry)
        self.assertNotIn("osn loop", rendered)

    def test_entry_with_osn_loop_none_no_crash(self):
        entry = self._entry_with_osn_loop_none()
        rendered = self._render(entry)
        self.assertIsInstance(rendered, str)

    def test_entry_with_osn_loop_none_no_osn_section(self):
        entry = self._entry_with_osn_loop_none()
        rendered = self._render(entry)
        self.assertNotIn("osn loop", rendered)

    def test_entry_with_osn_loop_dict_renders_section(self):
        loop_dict = asdict(OSNLoopMeta(
            enabled=True, steps_run=3, steps_queued=4, max_steps=5,
            terminated_reason="complete",
            steps=[
                OSNLoopStep(step_index=0, tool_name="read_file", target_label="src/a.py",
                            reason="read_search_finding", ok=True, output_chars=100),
            ],
            paths_surfaced=["src/a.py"],
        ))
        entry = self._entry_with_osn_loop(loop_dict)
        rendered = self._render(entry)
        self.assertIn("osn loop", rendered)

    def test_round_trip_json_serializable(self):
        import json
        loop = OSNLoopMeta(
            enabled=True, steps_run=1, steps_queued=2, max_steps=5,
            terminated_reason="complete",
            steps=[
                OSNLoopStep(step_index=0, tool_name="list_files", target_label="src/",
                            reason="task_keyword", ok=True, output_chars=40),
            ],
            paths_surfaced=[],
        )
        from openshard.native.executor import NativeRunMeta
        meta = NativeRunMeta()
        meta.osn_loop = loop
        d = asdict(meta)
        serialized = json.dumps(d)
        parsed = json.loads(serialized)
        self.assertIn("osn_loop", parsed)
        self.assertIsNotNone(parsed["osn_loop"])
        self.assertTrue(parsed["osn_loop"]["enabled"])

    def test_round_trip_reconstruct_and_render_no_crash(self):
        loop_dict = asdict(OSNLoopMeta(
            enabled=True, steps_run=2, steps_queued=3, max_steps=5,
            terminated_reason="complete",
            steps=[
                OSNLoopStep(step_index=0, tool_name="read_file", target_label="src/b.py",
                            reason="read_search_finding", ok=True, output_chars=200),
                OSNLoopStep(step_index=1, tool_name="search_repo", target_label="task_keyword_search",
                            reason="task_keyword", ok=True, output_chars=80),
            ],
            paths_surfaced=["src/b.py"],
        ))
        entry = self._entry_with_osn_loop(loop_dict)
        rendered = self._render(entry)
        self.assertIsInstance(rendered, str)
        self.assertIn("osn loop", rendered)
