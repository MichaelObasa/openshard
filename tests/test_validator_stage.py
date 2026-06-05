from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

from click.testing import CliRunner

from openshard.analysis.repo import RepoFacts
from openshard.cli.main import cli
from openshard.execution.stages import VALIDATOR_SYSTEM, _parse_verdict, run_validator_stage
from openshard.providers.base import ModelInfo
from openshard.providers.manager import InventoryEntry
from openshard.routing.engine import MODEL_MAIN, MODEL_STRONG
from openshard.routing.workflow_selector import WorkflowDecision

_DEFAULT_CONFIG = {"approval_mode": "smart"}
_PYTHON_REPO = RepoFacts(
    languages=["python"], package_files=[], framework=None,
    test_command=None, risky_paths=[], changed_files=[],
)


def _make_entry(model_id: str) -> InventoryEntry:
    return InventoryEntry(
        provider="openrouter",
        model=ModelInfo(
            id=model_id, name=model_id, pricing={"prompt": "0.0000005"},
            context_window=None, max_output_tokens=None,
            supports_vision=False, supports_tools=False,
        ),
    )


def _make_generator_mock(summary: str = "done"):
    g = MagicMock()
    result = MagicMock()
    result.usage = None
    result.files = []
    result.summary = summary
    result.notes = []
    g.generate.return_value = result
    g.model = "mock-model"
    g.fixer_model = "mock-fixer"
    client = MagicMock()
    val_response = MagicMock()
    val_response.content = "verdict: pass\nsummary: looks good"
    val_response.usage = None
    client.execute.return_value = val_response
    g.client = client
    return g


def _make_manager_mock(entries):
    m = MagicMock()
    inv = MagicMock()
    inv.models = entries
    m.get_inventory.return_value = inv
    m.providers = {"openrouter": MagicMock()}
    return m


_ENTRY = _make_entry("openrouter/test-model")


class TestParseVerdict(unittest.TestCase):

    def test_pass_verdict_parsed(self):
        result = _parse_verdict("verdict: pass\nsummary: all good")
        self.assertEqual(result["verdict"], "pass")
        self.assertEqual(result["summary"], "all good")

    def test_warn_verdict_parsed(self):
        result = _parse_verdict("verdict: warn\nsummary: minor issue")
        self.assertEqual(result["verdict"], "warn")
        self.assertEqual(result["summary"], "minor issue")

    def test_fail_verdict_parsed(self):
        result = _parse_verdict("verdict: fail\nsummary: task not complete")
        self.assertEqual(result["verdict"], "fail")

    def test_unknown_verdict_defaults_to_pass(self):
        result = _parse_verdict("verdict: unknown\nsummary: x")
        self.assertEqual(result["verdict"], "pass")

    def test_missing_verdict_line_defaults_to_pass(self):
        result = _parse_verdict("no structured output here")
        self.assertEqual(result["verdict"], "pass")

    def test_summary_truncated_at_300(self):
        long = "x" * 400
        result = _parse_verdict(f"verdict: pass\nsummary: {long}")
        self.assertLessEqual(len(result["summary"]), 300)

    def test_case_insensitive_verdict(self):
        result = _parse_verdict("VERDICT: FAIL\nSUMMARY: bad")
        self.assertEqual(result["verdict"], "fail")

    def test_model_key_not_in_parse_result(self):
        result = _parse_verdict("verdict: pass\nsummary: ok")
        self.assertNotIn("model", result)


class TestRunValidatorStage(unittest.TestCase):

    def _make_client(self, content: str):
        client = MagicMock()
        resp = MagicMock()
        resp.content = content
        resp.usage = None
        client.execute.return_value = resp
        return client

    def test_calls_client_execute_with_correct_model(self):
        client = self._make_client("verdict: pass\nsummary: ok")
        result, _ = run_validator_stage(client, "do X", "did X", model="test-model")
        client.execute.assert_called_once()
        call_kwargs = client.execute.call_args.kwargs
        self.assertEqual(call_kwargs["model"], "test-model")

    def test_uses_validator_system_prompt(self):
        client = self._make_client("verdict: pass\nsummary: ok")
        run_validator_stage(client, "do X", "did X", model="test-model")
        call_kwargs = client.execute.call_args.kwargs
        self.assertEqual(call_kwargs["system"], VALIDATOR_SYSTEM)

    def test_task_and_summary_in_prompt(self):
        client = self._make_client("verdict: pass\nsummary: ok")
        run_validator_stage(client, "do X", "did X", model="test-model")
        prompt = client.execute.call_args.kwargs["prompt"]
        self.assertIn("do X", prompt)
        self.assertIn("did X", prompt)

    def test_notes_included_in_prompt(self):
        client = self._make_client("verdict: pass\nsummary: ok")
        run_validator_stage(client, "do X", "did X", notes=["note one"], model="test-model")
        prompt = client.execute.call_args.kwargs["prompt"]
        self.assertIn("note one", prompt)

    def test_returns_verdict_dict_with_model(self):
        client = self._make_client("verdict: warn\nsummary: partially done")
        result, usage = run_validator_stage(client, "do X", "did X", model="my-model")
        self.assertEqual(result["verdict"], "warn")
        self.assertEqual(result["summary"], "partially done")
        self.assertEqual(result["model"], "my-model")
        self.assertIsNone(usage)

    def test_only_first_three_notes_used(self):
        client = self._make_client("verdict: pass\nsummary: ok")
        many_notes = [f"note {i}" for i in range(10)]
        run_validator_stage(client, "task", "summary", notes=many_notes, model="m")
        prompt = client.execute.call_args.kwargs["prompt"]
        self.assertIn("note 0", prompt)
        self.assertIn("note 2", prompt)
        self.assertNotIn("note 3", prompt)


class TestValidatorPipelineIntegration(unittest.TestCase):
    """Verify validator execution integrates correctly with pipeline staged runs."""

    def _run_staged(self, extra_args: list[str], generator=None):
        if generator is None:
            generator = _make_generator_mock()
        manager = _make_manager_mock([_ENTRY])
        log_mock = MagicMock()
        with patch("openshard.run.pipeline.ProviderManager", return_value=manager), \
             patch("openshard.run.pipeline.ExecutionGenerator", return_value=generator), \
             patch("openshard.cli.main.load_config", return_value=_DEFAULT_CONFIG), \
             patch("openshard.run.pipeline.analyze_repo", return_value=_PYTHON_REPO), \
             patch("openshard.run.pipeline.run_planning_stage",
                   return_value=("mock plan", None)), \
             patch("openshard.run.pipeline.select_workflow",
                   return_value=WorkflowDecision("staged", "test forced")), \
             patch("openshard.run.pipeline._log_run", log_mock):
            runner = CliRunner()
            result = runner.invoke(cli, ["run", "implement a feature"] + extra_args)
        return generator, log_mock, result

    def test_validator_does_not_run_when_flag_off(self):
        generator, log_mock, _ = self._run_staged([])
        log_mock.assert_called_once()
        extra = log_mock.call_args.kwargs.get("extra_metadata") or {}
        self.assertNotIn("validator_result", extra)

    def test_validator_does_not_call_client_when_flag_off(self):
        generator, _, _ = self._run_staged([])
        generator.client.execute.assert_not_called()

    def test_validator_runs_when_dispatch_flag_on(self):
        generator, log_mock, _ = self._run_staged(["--experimental-tier-dispatch"])
        generator.client.execute.assert_called_once()

    def test_validator_receives_validator_model(self):
        generator, _, _ = self._run_staged(["--experimental-tier-dispatch"])
        call_kwargs = generator.client.execute.call_args.kwargs
        self.assertEqual(call_kwargs["model"], MODEL_STRONG)

    def test_validator_result_stored_in_run_history(self):
        generator, log_mock, _ = self._run_staged(["--experimental-tier-dispatch"])
        log_mock.assert_called_once()
        extra = log_mock.call_args.kwargs.get("extra_metadata") or {}
        self.assertIn("validator_result", extra)
        vr = extra["validator_result"]
        self.assertIn("verdict", vr)
        self.assertIn("summary", vr)
        self.assertIn("model", vr)

    def test_validator_verdict_is_pass(self):
        generator, log_mock, _ = self._run_staged(["--experimental-tier-dispatch"])
        extra = log_mock.call_args.kwargs.get("extra_metadata") or {}
        vr = extra["validator_result"]
        self.assertEqual(vr["verdict"], "pass")

    def test_validator_model_is_recorded(self):
        generator, log_mock, _ = self._run_staged(["--experimental-tier-dispatch"])
        extra = log_mock.call_args.kwargs.get("extra_metadata") or {}
        vr = extra["validator_result"]
        self.assertEqual(vr["model"], MODEL_STRONG)

    def test_validator_adds_stage_run(self):
        generator, log_mock, _ = self._run_staged(["--experimental-tier-dispatch"])
        stage_runs = log_mock.call_args.kwargs.get("stage_runs", [])
        val_runs = [sr for sr in stage_runs if sr.stage.stage_type == "validation"]
        self.assertEqual(len(val_runs), 1)
        self.assertEqual(val_runs[0].model, MODEL_STRONG)
        self.assertIn("pass", val_runs[0].summary)

    def test_stage_runs_total_three_with_dispatch(self):
        """planning + implementation + validation = 3 stage_runs."""
        generator, log_mock, _ = self._run_staged(["--experimental-tier-dispatch"])
        stage_runs = log_mock.call_args.kwargs.get("stage_runs", [])
        self.assertEqual(len(stage_runs), 3)
        types = {sr.stage.stage_type for sr in stage_runs}
        self.assertIn("planning", types)
        self.assertIn("implementation", types)
        self.assertIn("validation", types)

    def test_validator_error_does_not_crash_run(self):
        generator = _make_generator_mock()
        generator.client.execute.side_effect = RuntimeError("network error")
        _, log_mock, cli_result = self._run_staged(
            ["--experimental-tier-dispatch"], generator=generator
        )
        self.assertEqual(cli_result.exit_code, 0)
        extra = log_mock.call_args.kwargs.get("extra_metadata") or {}
        vr = extra.get("validator_result")
        self.assertIsNotNone(vr)
        self.assertEqual(vr["verdict"], "error")
        self.assertIn("network error", vr["summary"])

    def test_dry_run_does_not_call_validator(self):
        generator, _, _ = self._run_staged(
            ["--experimental-tier-dispatch", "--dry-run"]
        )
        generator.client.execute.assert_not_called()

    def test_dry_run_no_validator_result_in_history(self):
        generator, log_mock, _ = self._run_staged(
            ["--experimental-tier-dispatch", "--dry-run"]
        )
        extra = log_mock.call_args.kwargs.get("extra_metadata") or {}
        self.assertNotIn("validator_result", extra)


class TestValidatorLastRendering(unittest.TestCase):
    """Verify openshard last --more/--full shows validator verdict from history."""

    def _render(self, entry: dict, detail: str) -> str:
        import click

        from openshard.cli.main import _render_log_entry

        @click.command()
        def cmd():
            _render_log_entry(entry, detail)

        return CliRunner().invoke(cmd).output

    def _entry_with_validator(self, verdict: str = "pass", summary: str = "looks good") -> dict:
        tdr = {
            "enabled": True, "applied": True, "tier_source": "category_fallback",
            "planner_tier": "frontier-reasoning-model", "planner_model": MODEL_STRONG,
            "executor_tier": "balanced-coding-model", "executor_model": MODEL_MAIN,
            "validator_tier": "independent-validator-model", "validator_model": MODEL_STRONG,
            "fallback_used": False, "fallback_reason": "", "warnings": [],
        }
        return {
            "timestamp": "2026-01-01T00:00:00",
            "task": "test task",
            "summary": "done",
            "tier_dispatch_receipt": tdr,
            "validator_result": {
                "verdict": verdict,
                "summary": summary,
                "model": MODEL_STRONG,
            },
        }

    def _entry_without_validator(self) -> dict:
        tdr = {
            "enabled": True, "applied": True, "tier_source": "category_fallback",
            "planner_tier": "frontier-reasoning-model", "planner_model": MODEL_STRONG,
            "executor_tier": "balanced-coding-model", "executor_model": MODEL_MAIN,
            "validator_tier": "independent-validator-model", "validator_model": MODEL_STRONG,
            "fallback_used": False, "fallback_reason": "", "warnings": [],
        }
        return {
            "timestamp": "2026-01-01T00:00:00",
            "task": "test task",
            "summary": "done",
            "tier_dispatch_receipt": tdr,
        }

    def test_more_shows_validator_verdict(self):
        out = self._render(self._entry_with_validator("pass"), "full")
        self.assertIn("pass", out)

    def test_more_shows_warn_verdict(self):
        out = self._render(self._entry_with_validator("warn"), "full")
        self.assertIn("warn", out)

    def test_more_shows_fail_verdict(self):
        out = self._render(self._entry_with_validator("fail"), "full")
        self.assertIn("fail", out)

    def test_more_shows_reserved_when_no_validator_result(self):
        out = self._render(self._entry_without_validator(), "full")
        self.assertIn("reserved", out)

    def test_full_shows_verdict(self):
        out = self._render(self._entry_with_validator("warn", "minor problem"), "full")
        self.assertIn("warn", out)

    def test_full_shows_summary(self):
        out = self._render(self._entry_with_validator("warn", "minor problem"), "full")
        self.assertIn("minor problem", out)

    def test_full_shows_reserved_when_no_validator_result(self):
        out = self._render(self._entry_without_validator(), "full")
        self.assertIn("reserved for validation", out)

    def test_default_detail_does_not_render_validator_block(self):
        out = self._render(self._entry_with_validator("pass"), "default")
        self.assertNotIn("Validator:", out)


class TestValidatorReadonlyNoRegression(unittest.TestCase):
    """Validator must not interfere with readonly task protection."""

    def _run_readonly(self, extra_args: list[str]):
        generator = _make_generator_mock()
        generator.generate.return_value.files = [
            MagicMock(change_type="create", path="foo.py", summary="x")
        ]
        manager = _make_manager_mock([_ENTRY])
        log_mock = MagicMock()
        with patch("openshard.run.pipeline.ProviderManager", return_value=manager), \
             patch("openshard.run.pipeline.ExecutionGenerator", return_value=generator), \
             patch("openshard.cli.main.load_config", return_value=_DEFAULT_CONFIG), \
             patch("openshard.run.pipeline.analyze_repo", return_value=_PYTHON_REPO), \
             patch("openshard.run.pipeline._log_run", log_mock):
            runner = CliRunner()
            result = runner.invoke(
                cli, ["run", "what does this code do?"] + extra_args
            )
        return log_mock, result

    def test_readonly_files_discarded_even_with_dispatch(self):
        log_mock, _ = self._run_readonly(["--experimental-tier-dispatch"])
        log_mock.assert_called_once()
        files = log_mock.call_args.kwargs.get("files", [])
        self.assertEqual(len(files), 0)


if __name__ == "__main__":
    unittest.main()
