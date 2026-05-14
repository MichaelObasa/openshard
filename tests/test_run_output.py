"""Tests for build_stage_displays() and the post-run stage summary panel.

Most tests are pure unit tests against build_stage_displays() — no CLI
invocation needed. A single integration smoke test checks that openshard last
rendering is unaffected.
"""
from __future__ import annotations

import unittest
from types import SimpleNamespace

from openshard.cli.run_output import build_stage_displays
from openshard.execution.stages import Stage, StageRun
from openshard.routing.engine import RoutingDecision


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _rd(model: str = "anthropic/claude-sonnet-4.6", rationale: str = "standard feature implementation", category: str = "standard") -> RoutingDecision:
    return RoutingDecision(model=model, category=category, rationale=rationale)


def _sr(stage_type: str, model: str = "anthropic/claude-sonnet-4.6", duration: float = 1.0) -> StageRun:
    return StageRun(
        stage=Stage(stage_type=stage_type, description=""),
        model=model,
        duration=duration,
        cost=None,
        summary="",
    )


def _vp(run: bool = False, reason: str = "not configured") -> SimpleNamespace:
    return SimpleNamespace(run=run, reason=reason)


def _stages(
    stage_runs=None,
    routing_decision=None,
    verification_attempted=False,
    verification_passed=None,
    readonly_task=False,
    validator_policy=None,
    final_files=None,
):
    return build_stage_displays(
        stage_runs=stage_runs or [],
        routing_decision=routing_decision or _rd(),
        verification_attempted=verification_attempted,
        verification_passed=verification_passed,
        readonly_task=readonly_task,
        validator_policy=validator_policy,
        final_files=final_files or [],
    )


def _names(stages) -> list[str]:
    return [s.name for s in stages]


def _by_name(stages, name: str):
    return next((s for s in stages if s.name == name), None)


# ---------------------------------------------------------------------------
# Route stage
# ---------------------------------------------------------------------------

class TestRouteStage(unittest.TestCase):
    def test_route_always_first(self):
        stages = _stages()
        self.assertEqual(stages[0].name, "Route")

    def test_route_status_is_routed(self):
        stages = _stages()
        self.assertEqual(stages[0].status, "routed")

    def test_route_contains_model_label(self):
        stages = _stages(routing_decision=_rd(model="anthropic/claude-sonnet-4.6"))
        self.assertIn("Sonnet 4.6", stages[0].detail)

    def test_route_contains_rationale(self):
        stages = _stages(routing_decision=_rd(rationale="standard feature implementation"))
        self.assertIn("standard coding", stages[0].detail)

    def test_route_absent_when_no_routing_decision(self):
        stages = build_stage_displays(
            stage_runs=[],
            routing_decision=None,
            verification_attempted=False,
            verification_passed=None,
            readonly_task=False,
            validator_policy=None,
            final_files=[],
        )
        self.assertNotIn("Route", _names(stages))


# ---------------------------------------------------------------------------
# Plan stage
# ---------------------------------------------------------------------------

class TestPlanStage(unittest.TestCase):
    def test_plan_shown_when_planning_stage_ran(self):
        stages = _stages(stage_runs=[_sr("planning")])
        self.assertIn("Plan", _names(stages))

    def test_plan_status_passed_when_ran(self):
        stages = _stages(stage_runs=[_sr("planning")])
        plan = _by_name(stages, "Plan")
        self.assertEqual(plan.status, "passed")

    def test_plan_contains_model_label(self):
        stages = _stages(stage_runs=[_sr("planning", model="anthropic/claude-sonnet-4.6")])
        plan = _by_name(stages, "Plan")
        self.assertIn("Sonnet 4.6", plan.detail)

    def test_plan_contains_duration(self):
        stages = _stages(stage_runs=[_sr("planning", duration=2.5)])
        plan = _by_name(stages, "Plan")
        self.assertIn("2.5s", plan.detail)

    def test_plan_skipped_for_readonly(self):
        stages = _stages(readonly_task=True)
        plan = _by_name(stages, "Plan")
        self.assertIsNotNone(plan)
        self.assertEqual(plan.status, "skipped")
        self.assertIn("read-only", plan.detail)

    def test_plan_omitted_for_direct_executor(self):
        # No planning StageRun and not readonly → Plan row absent
        stages = _stages(stage_runs=[_sr("implementation")], readonly_task=False)
        self.assertNotIn("Plan", _names(stages))


# ---------------------------------------------------------------------------
# Work stage (non-readonly)
# ---------------------------------------------------------------------------

class TestWorkStage(unittest.TestCase):
    def test_work_shown_for_write_task(self):
        stages = _stages(stage_runs=[_sr("implementation")], readonly_task=False)
        self.assertIn("Work", _names(stages))

    def test_work_status_passed(self):
        stages = _stages(stage_runs=[_sr("implementation")], readonly_task=False)
        work = _by_name(stages, "Work")
        self.assertEqual(work.status, "passed")

    def test_work_uses_stage_run_model_not_routing(self):
        # StageRun model differs from routing decision model
        stages = _stages(
            stage_runs=[_sr("implementation", model="deepseek/deepseek-v4-flash")],
            routing_decision=_rd(model="anthropic/claude-sonnet-4.6"),
            readonly_task=False,
        )
        work = _by_name(stages, "Work")
        self.assertIn("DeepSeek V4 Flash", work.detail)
        self.assertNotIn("Sonnet 4.6", work.detail)

    def test_work_falls_back_to_routing_model_when_no_stage_run(self):
        stages = _stages(
            stage_runs=[],
            routing_decision=_rd(model="anthropic/claude-sonnet-4.6"),
            readonly_task=False,
        )
        work = _by_name(stages, "Work")
        self.assertIsNotNone(work)
        self.assertIn("Sonnet 4.6", work.detail)


# ---------------------------------------------------------------------------
# Ask stage (readonly)
# ---------------------------------------------------------------------------

class TestAskStage(unittest.TestCase):
    def test_ask_shown_for_readonly(self):
        stages = _stages(stage_runs=[_sr("implementation")], readonly_task=True)
        self.assertIn("Ask", _names(stages))

    def test_no_work_passed_for_readonly(self):
        stages = _stages(stage_runs=[_sr("implementation")], readonly_task=True)
        work = _by_name(stages, "Work")
        # Either absent or not "passed" — hard rule: no "Work passed" for readonly
        self.assertTrue(work is None or work.status != "passed")

    def test_ask_status_is_ask(self):
        stages = _stages(stage_runs=[_sr("implementation")], readonly_task=True)
        ask = _by_name(stages, "Ask")
        self.assertEqual(ask.status, "ask")

    def test_readonly_layout_order(self):
        # Route → Plan skipped → Ask → Verify → Receipt
        stages = _stages(
            stage_runs=[_sr("implementation")],
            readonly_task=True,
        )
        names = _names(stages)
        self.assertIn("Route", names)
        self.assertIn("Ask", names)
        self.assertNotIn("Work", names)
        route_i = names.index("Route")
        ask_i = names.index("Ask")
        self.assertLess(route_i, ask_i)


# ---------------------------------------------------------------------------
# Verify stage
# ---------------------------------------------------------------------------

class TestVerifyStage(unittest.TestCase):
    def test_verify_passed(self):
        stages = _stages(verification_attempted=True, verification_passed=True)
        verify = _by_name(stages, "Verify")
        self.assertEqual(verify.status, "passed")

    def test_verify_failed(self):
        stages = _stages(verification_attempted=True, verification_passed=False)
        verify = _by_name(stages, "Verify")
        self.assertEqual(verify.status, "failed")

    def test_verify_skipped_not_configured(self):
        stages = _stages(verification_attempted=False, validator_policy=_vp(run=False, reason=""))
        verify = _by_name(stages, "Verify")
        self.assertEqual(verify.status, "skipped")
        self.assertIn("not configured", verify.detail)

    def test_verify_skipped_readonly(self):
        stages = _stages(readonly_task=True, verification_attempted=False)
        verify = _by_name(stages, "Verify")
        self.assertEqual(verify.status, "skipped")
        self.assertIn("read-only", verify.detail)

    def test_verify_always_present(self):
        stages = _stages()
        self.assertIn("Verify", _names(stages))


# ---------------------------------------------------------------------------
# Receipt stage
# ---------------------------------------------------------------------------

class TestReceiptStage(unittest.TestCase):
    def test_receipt_always_last(self):
        stages = _stages()
        self.assertEqual(stages[-1].name, "Receipt")

    def test_receipt_status_saved(self):
        stages = _stages()
        self.assertEqual(stages[-1].status, "saved")

    def test_receipt_file_count_singular(self):
        files = [SimpleNamespace(path="a.py", change_type="create", summary="")]
        stages = _stages(final_files=files)
        self.assertIn("1 file changed", stages[-1].detail)

    def test_receipt_file_count_plural(self):
        files = [
            SimpleNamespace(path="a.py", change_type="create", summary=""),
            SimpleNamespace(path="b.py", change_type="update", summary=""),
            SimpleNamespace(path="c.py", change_type="delete", summary=""),
        ]
        stages = _stages(final_files=files)
        self.assertIn("3 files changed", stages[-1].detail)

    def test_receipt_no_files(self):
        stages = _stages(final_files=[])
        self.assertIn("no files changed", stages[-1].detail)


# ---------------------------------------------------------------------------
# Integration: openshard last rendering unaffected
# ---------------------------------------------------------------------------

class TestOldEntriesUnaffected(unittest.TestCase):
    def test_last_rendering_still_contains_receipt_saved(self):
        from openshard.cli.run_output import _render_native_receipt
        from types import SimpleNamespace as NS

        meta = NS(
            final_report=NS(
                diff_files=["a.py"],
                verification_attempted=True,
                verification_passed=True,
                approval_receipt=None,
            ),
            diff_review=None,
            approval_receipt=None,
        )
        line = _render_native_receipt(meta)
        self.assertIn("Receipt saved", line)

    def test_last_rendering_model_line_preserved(self):
        from openshard.cli.run_output import _build_model_line
        rd = _rd(model="anthropic/claude-sonnet-4.6")
        line = _build_model_line(rd, [], model="anthropic/claude-sonnet-4.6")
        self.assertIsNotNone(line)
        self.assertIn("Model:", line)
