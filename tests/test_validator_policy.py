from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

import click
from click.testing import CliRunner

from openshard.analysis.repo import RepoFacts
from openshard.cli.main import cli, _render_log_entry
from openshard.cli.run_output import _render_tier_dispatch_block
from openshard.providers.base import ModelInfo
from openshard.providers.manager import InventoryEntry
from openshard.routing.engine import MODEL_MAIN, MODEL_STRONG
from openshard.routing.workflow_selector import WorkflowDecision
from openshard.run.validator_policy import ValidatorPolicyDecision, should_run_validator

_DEFAULT_CONFIG = {"approval_mode": "smart"}

_PASS_ARGS = dict(
    has_validator_model=True,
    dry_run=False,
    can_dispatch=True,
    tier_dispatch_applied=True,
    readonly_task=False,
    routing_category="standard",
    execution_profile="native_light",
    workflow="direct",
    risky_paths_count=0,
    verification_attempted=False,
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
_PYTHON_REPO = RepoFacts(
    languages=["python"], package_files=[], framework=None,
    test_command=None, risky_paths=[], changed_files=[],
)
_RISKY_REPO = RepoFacts(
    languages=["python"], package_files=[], framework=None,
    test_command=None, risky_paths=["auth/login.py"], changed_files=[],
)


def _render(entry: dict, detail: str) -> str:
    @click.command()
    def cmd():
        _render_log_entry(entry, detail)

    return CliRunner().invoke(cmd).output


def _tdr_base() -> dict:
    return {
        "enabled": True, "applied": True, "tier_source": "category_fallback",
        "planner_tier": "frontier-reasoning-model", "planner_model": MODEL_STRONG,
        "executor_tier": "balanced-coding-model", "executor_model": MODEL_MAIN,
        "validator_tier": "independent-validator-model", "validator_model": MODEL_STRONG,
        "fallback_used": False, "fallback_reason": "", "warnings": [],
    }


class TestShouldRunValidatorUnit(unittest.TestCase):

    def _call(self, **overrides) -> ValidatorPolicyDecision:
        return should_run_validator(**{**_PASS_ARGS, **overrides})

    # --- skip conditions ---

    def test_no_validator_model_skips(self):
        dec = self._call(has_validator_model=False)
        self.assertFalse(dec.run)
        self.assertIn("validator model", dec.reason)

    def test_dry_run_skips(self):
        dec = self._call(dry_run=True)
        self.assertFalse(dec.run)
        self.assertIn("dry run", dec.reason)

    def test_dispatch_not_enabled_skips(self):
        dec = self._call(can_dispatch=False)
        self.assertFalse(dec.run)
        self.assertIn("not enabled", dec.reason)

    def test_dispatch_not_applied_skips(self):
        dec = self._call(tier_dispatch_applied=False)
        self.assertFalse(dec.run)
        self.assertIn("not applied", dec.reason)

    def test_readonly_task_skips(self):
        dec = self._call(readonly_task=True)
        self.assertFalse(dec.run)
        self.assertIn("read-only", dec.reason)

    def test_readonly_is_absolute_risky_paths_do_not_override(self):
        dec = self._call(readonly_task=True, risky_paths_count=3)
        self.assertFalse(dec.run)
        self.assertIn("read-only", dec.reason)

    def test_readonly_is_absolute_security_category_does_not_override(self):
        dec = self._call(readonly_task=True, routing_category="security")
        self.assertFalse(dec.run)
        self.assertIn("read-only", dec.reason)

    def test_boilerplate_direct_no_signals_skips(self):
        dec = self._call(routing_category="boilerplate", execution_profile="native_light",
                         risky_paths_count=0, verification_attempted=False, workflow="direct")
        self.assertFalse(dec.run)

    def test_default_simple_task_skips(self):
        dec = self._call()
        self.assertFalse(dec.run)
        self.assertIn("safe", dec.reason)

    def test_skip_reason_is_non_empty_string(self):
        dec = self._call(readonly_task=True)
        self.assertIsInstance(dec.reason, str)
        self.assertGreater(len(dec.reason), 0)

    # --- run conditions ---

    def test_security_category_runs(self):
        dec = self._call(routing_category="security")
        self.assertTrue(dec.run)
        self.assertIn("security", dec.reason)

    def test_complex_category_runs(self):
        dec = self._call(routing_category="complex")
        self.assertTrue(dec.run)
        self.assertIn("complex", dec.reason)

    def test_native_deep_runs(self):
        dec = self._call(execution_profile="native_deep")
        self.assertTrue(dec.run)
        self.assertIn("deep", dec.reason)

    def test_native_swarm_runs(self):
        dec = self._call(execution_profile="native_swarm")
        self.assertTrue(dec.run)
        self.assertIn("swarm", dec.reason)

    def test_risky_paths_runs(self):
        dec = self._call(risky_paths_count=2)
        self.assertTrue(dec.run)
        self.assertIn("risky", dec.reason)

    def test_verification_attempted_runs(self):
        dec = self._call(verification_attempted=True)
        self.assertTrue(dec.run)
        self.assertIn("verification", dec.reason)

    def test_staged_workflow_runs(self):
        dec = self._call(workflow="staged")
        self.assertTrue(dec.run)
        self.assertIn("staged", dec.reason)

    def test_decision_is_frozen_dataclass(self):
        dec = self._call()
        with self.assertRaises(Exception):
            dec.run = True  # type: ignore[misc]


class TestValidatorPolicyPipelineIntegration(unittest.TestCase):
    """Verify policy decisions are stored in run history."""

    def _run(self, task: str, extra_args: list[str], repo: RepoFacts = _PYTHON_REPO,
             generator=None, workflow: str = "staged"):
        if generator is None:
            generator = _make_generator_mock()
        manager = _make_manager_mock([_ENTRY])
        log_mock = MagicMock()
        with patch("openshard.run.pipeline.ProviderManager", return_value=manager), \
             patch("openshard.run.pipeline.ExecutionGenerator", return_value=generator), \
             patch("openshard.cli.main.load_config", return_value=_DEFAULT_CONFIG), \
             patch("openshard.run.pipeline.analyze_repo", return_value=repo), \
             patch("openshard.run.pipeline.run_planning_stage",
                   return_value=("mock plan", None)), \
             patch("openshard.run.pipeline.select_workflow",
                   return_value=WorkflowDecision(workflow, "test")), \
             patch("openshard.run.pipeline._log_run", log_mock):
            runner = CliRunner()
            runner.invoke(cli, ["run", task] + extra_args)
        return log_mock

    def _extra(self, log_mock) -> dict:
        log_mock.assert_called_once()
        return log_mock.call_args.kwargs.get("extra_metadata") or {}

    def test_readonly_task_stores_policy_skipped(self):
        log_mock = self._run(
            "what does this function do?",
            ["--experimental-tier-dispatch"],
        )
        extra = self._extra(log_mock)
        self.assertIn("validator_policy", extra)
        vp = extra["validator_policy"]
        self.assertFalse(vp["run"])
        self.assertIn("read-only", vp["reason"])

    def test_readonly_task_does_not_call_validator_client(self):
        generator = _make_generator_mock()
        self._run(
            "what does this function do?",
            ["--experimental-tier-dispatch"],
            generator=generator,
        )
        generator.client.execute.assert_not_called()

    def test_readonly_risky_paths_still_skips_validator(self):
        log_mock = self._run(
            "explain this code",
            ["--experimental-tier-dispatch"],
            repo=_RISKY_REPO,
        )
        extra = self._extra(log_mock)
        vp = extra.get("validator_policy", {})
        self.assertFalse(vp.get("run"))
        self.assertIn("read-only", vp.get("reason", ""))

    def test_staged_write_task_stores_policy_run(self):
        log_mock = self._run(
            "implement a feature",
            ["--experimental-tier-dispatch"],
            workflow="staged",
        )
        extra = self._extra(log_mock)
        self.assertIn("validator_policy", extra)
        vp = extra["validator_policy"]
        self.assertTrue(vp["run"])

    def test_policy_stored_even_when_dry_run(self):
        log_mock = self._run(
            "implement a feature",
            ["--experimental-tier-dispatch", "--dry-run"],
            workflow="staged",
        )
        extra = self._extra(log_mock)
        self.assertIn("validator_policy", extra)
        vp = extra["validator_policy"]
        self.assertFalse(vp["run"])

    def test_no_policy_stored_without_dispatch_flag(self):
        log_mock = self._run("implement a feature", [])
        extra = self._extra(log_mock)
        self.assertNotIn("validator_policy", extra)

    def test_cost_fields_unaffected_when_validator_skipped(self):
        log_mock = self._run(
            "what does this function do?",
            ["--experimental-tier-dispatch"],
        )
        extra = self._extra(log_mock)
        self.assertNotIn("validator_result", extra)


class TestValidatorPolicyRendering(unittest.TestCase):
    """Verify skip reason surfaces correctly in --more/--full output."""

    def _entry_skipped(self, reason: str = "read-only task") -> dict:
        return {
            "timestamp": "2026-01-01T00:00:00",
            "task": "what does this do?",
            "summary": "analysis done",
            "tier_dispatch_receipt": _tdr_base(),
            "validator_policy": {"run": False, "reason": reason},
        }

    def _entry_skipped_no_tdr(self, reason: str = "simple/safe task") -> dict:
        return {
            "timestamp": "2026-01-01T00:00:00",
            "task": "implement a feature",
            "summary": "done",
            "validator_policy": {"run": False, "reason": reason},
        }

    def test_more_shows_skipped_reason_in_model_plan(self):
        out = _render(self._entry_skipped("read-only task"), "more")
        self.assertIn("skipped", out)
        self.assertIn("read-only task", out)

    def test_full_shows_skipped_reason_in_model_plan(self):
        out = _render(self._entry_skipped("read-only task"), "full")
        self.assertIn("Skipped", out)
        self.assertIn("read-only task", out)

    def test_more_skipped_inside_model_plan_not_standalone(self):
        out = _render(self._entry_skipped("read-only task"), "more")
        self.assertNotIn("\nValidator: skipped", out)

    def test_standalone_skipped_line_when_no_tdr(self):
        out = _render(self._entry_skipped_no_tdr("simple/safe task"), "more")
        self.assertIn("Validator: skipped", out)
        self.assertIn("simple/safe task", out)

    def test_default_detail_does_not_show_skipped(self):
        out = _render(self._entry_skipped("read-only task"), "default")
        self.assertNotIn("skipped", out)

    def test_reserved_still_shown_when_no_policy(self):
        entry = {
            "timestamp": "2026-01-01T00:00:00",
            "task": "test",
            "summary": "done",
            "tier_dispatch_receipt": _tdr_base(),
        }
        out = _render(entry, "more")
        self.assertIn("reserved", out)

    def test_verdict_shown_when_validator_ran(self):
        entry = {
            "timestamp": "2026-01-01T00:00:00",
            "task": "test",
            "summary": "done",
            "tier_dispatch_receipt": _tdr_base(),
            "validator_result": {"verdict": "pass", "summary": "looks good", "model": MODEL_STRONG},
            "validator_policy": {"run": True, "reason": "staged write task"},
        }
        out = _render(entry, "more")
        self.assertIn("pass", out)
        self.assertNotIn("skipped", out)


class TestRenderTierDispatchPolicyTypes(unittest.TestCase):
    """Regression: _render_tier_dispatch_block must not crash when validator_policy
    is a ValidatorPolicyDecision dataclass (live output path) instead of a dict
    (stored history path). Both must render identically."""

    _TDR = _tdr_base()

    def _lines_more(self, policy) -> list[str]:
        return _render_tier_dispatch_block(self._TDR, "more", validator_policy=policy)

    def _lines_full(self, policy) -> list[str]:
        return _render_tier_dispatch_block(self._TDR, "full", validator_policy=policy)

    def test_dict_policy_more_shows_skipped(self):
        lines = self._lines_more({"run": False, "reason": "read-only task"})
        combined = "\n".join(lines)
        self.assertIn("skipped", combined)
        self.assertIn("read-only task", combined)

    def test_dataclass_policy_more_shows_skipped(self):
        lines = self._lines_more(ValidatorPolicyDecision(run=False, reason="read-only task"))
        combined = "\n".join(lines)
        self.assertIn("skipped", combined)
        self.assertIn("read-only task", combined)

    def test_dict_policy_full_shows_skipped(self):
        lines = self._lines_full({"run": False, "reason": "read-only task"})
        combined = "\n".join(lines)
        self.assertIn("Skipped", combined)
        self.assertIn("read-only task", combined)

    def test_dataclass_policy_full_shows_skipped(self):
        lines = self._lines_full(ValidatorPolicyDecision(run=False, reason="read-only task"))
        combined = "\n".join(lines)
        self.assertIn("Skipped", combined)
        self.assertIn("read-only task", combined)

    def test_dict_and_dataclass_more_output_identical(self):
        dict_lines = self._lines_more({"run": False, "reason": "read-only task"})
        dc_lines = self._lines_more(ValidatorPolicyDecision(run=False, reason="read-only task"))
        self.assertEqual(dict_lines, dc_lines)

    def test_dict_and_dataclass_full_output_identical(self):
        dict_lines = self._lines_full({"run": False, "reason": "read-only task"})
        dc_lines = self._lines_full(ValidatorPolicyDecision(run=False, reason="read-only task"))
        self.assertEqual(dict_lines, dc_lines)

    def test_dataclass_run_true_shows_reserved(self):
        lines = self._lines_more(ValidatorPolicyDecision(run=True, reason="staged write task"))
        combined = "\n".join(lines)
        self.assertIn("reserved", combined)
        self.assertNotIn("skipped", combined)


if __name__ == "__main__":
    unittest.main()
