"""Tests for build_stage_displays(), the post-run stage summary panel, and
terminal-safety helpers.

Most tests are pure unit tests against build_stage_displays() - no CLI
invocation needed. A single integration smoke test checks that openshard last
rendering is unaffected.
"""
from __future__ import annotations

import sys
import types as _types
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from openshard.cli.run_output import _safe_console_text, build_stage_displays, render_run_timeline
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


# ---------------------------------------------------------------------------
# render_run_timeline()
# ---------------------------------------------------------------------------

def _tl_ev(event: str, label: str, kind: str = "run", status: str = "completed") -> dict:
    return {"event": event, "label": label, "kind": kind, "status": status}


class TestRenderRunTimeline(unittest.TestCase):

    def test_empty_timeline_shows_receipt_saved(self):
        text = "\n".join(render_run_timeline(None))
        self.assertIn("Saved Shard receipt", text)

    def test_events_appear_in_order(self):
        tl = [
            _tl_ev("repo_scanned", "Scanned repo", "scan"),
            _tl_ev("model_selected", "Routed to Claude", "route"),
        ]
        text = "\n".join(render_run_timeline(tl))
        repo_pos = text.index("Scanned repo")
        model_pos = text.index("Routed to Claude")
        receipt_pos = text.index("Saved Shard receipt")
        self.assertLess(repo_pos, model_pos)
        self.assertLess(model_pos, receipt_pos)

    def test_task_header_shown(self):
        lines = render_run_timeline([], task="production IaC review")
        self.assertEqual(lines[0], "Running production IaC review")

    def test_no_task_header_when_task_empty(self):
        lines = render_run_timeline([], task="")
        self.assertFalse(any(ln.startswith("Running") for ln in lines))

    def test_completed_event_gets_checkmark(self):
        text = "\n".join(render_run_timeline([_tl_ev("x", "Did something")]))
        self.assertTrue("✓" in text or "+" in text)

    def test_failed_event_gets_x_symbol(self):
        text = "\n".join(render_run_timeline([_tl_ev("x", "Model failed", "model", "failed")]))
        self.assertTrue("✖" in text or "x  Model failed" in text)

    def test_receipt_saved_is_last_non_blank(self):
        lines = render_run_timeline([_tl_ev("run_started", "Started run")])
        non_blank = [ln for ln in lines if ln.strip()]
        self.assertIn("Saved Shard receipt", non_blank[-1])

    def test_receipt_saved_not_duplicated(self):
        tl = [_tl_ev("receipt_saved", "Saved Shard receipt", "receipt")]
        text = "\n".join(render_run_timeline(tl))
        self.assertEqual(text.count("Saved Shard receipt"), 1)

    def test_run_started_skipped_in_live_feed(self):
        tl = [_tl_ev("run_started", "Started run")]
        lines = render_run_timeline(tl)
        self.assertFalse(any("Started run" in ln for ln in lines))

    def test_run_label_overrides_task_header(self):
        lines = render_run_timeline(
            [],
            task="Review and harden this deliberately flawed Terraform codebase...",
            run_label="production IaC review",
        )
        self.assertEqual(lines[0], "Running production IaC review")
        self.assertFalse(any("Terraform" in ln for ln in lines))

    def test_run_label_fallback_to_task(self):
        lines = render_run_timeline([], task="some task", run_label="")
        self.assertEqual(lines[0], "Running some task")

    def test_timeline_scan_before_route(self):
        tl = [
            _tl_ev("repo_scanned", "Scanned repo", "scan"),
            _tl_ev("model_selected", "Routed to Claude", "route"),
        ]
        text = "\n".join(render_run_timeline(tl))
        self.assertLess(text.index("Scanned repo"), text.index("Routed to Claude"))


class TestTimelineWordingGuards(unittest.TestCase):

    def _rendered(self) -> str:
        return "\n".join(render_run_timeline([
            _tl_ev("repo_scanned", "Scanned repo", "scan"),
            _tl_ev("model_selected", "Routed to Claude Sonnet 4.6", "route"),
        ]))

    def test_no_chain_of_thought(self):
        self.assertNotIn("chain of thought", self._rendered().lower())

    def test_no_reasoning_trace(self):
        self.assertNotIn("reasoning trace", self._rendered().lower())

    def test_no_internal_thoughts(self):
        self.assertNotIn("internal thoughts", self._rendered().lower())

    def test_no_structured_findings_in_labels(self):
        self.assertNotIn("STRUCTURED_FINDINGS", self._rendered())

    def test_no_raw_json_in_event_lines(self):
        for ln in self._rendered().splitlines():
            stripped = ln.strip()
            if stripped:
                self.assertFalse(stripped.startswith("{"), f"JSON leak: {stripped!r}")
                self.assertFalse(stripped.startswith("["), f"JSON leak: {stripped!r}")

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


# ---------------------------------------------------------------------------
# Receipt panel rendering
# ---------------------------------------------------------------------------

class TestReceiptPanel(unittest.TestCase):
    """Tests for render_receipt_panel and render_receipt_panel_plain."""

    def _render_rich(self, stages, mode_label=None, cost_str=None) -> str:
        import click
        from click.testing import CliRunner
        from openshard.cli.ui.console import make_console
        from openshard.cli.ui.run_screen import render_receipt_panel

        @click.command()
        def cmd():
            render_receipt_panel(stages, mode_label, cost_str, make_console())

        return CliRunner().invoke(cmd).output

    def _render_plain(self, stages, mode_label=None, cost_str=None) -> str:
        import click
        from click.testing import CliRunner
        from openshard.cli.ui.run_screen import render_receipt_panel_plain

        @click.command()
        def cmd():
            render_receipt_panel_plain(stages, mode_label, cost_str)

        return CliRunner().invoke(cmd).output

    def _ask_stages(self):
        return _stages(readonly_task=True, verification_attempted=False)

    def _write_stages(self, n_files=2):
        files = [
            SimpleNamespace(path=f"f{i}.py", change_type="create", summary="")
            for i in range(n_files)
        ]
        return _stages(
            stage_runs=[_sr("implementation")],
            readonly_task=False,
            verification_attempted=True,
            verification_passed=True,
            final_files=files,
        )

    def test_panel_title_present(self):
        out = self._render_rich(self._ask_stages(), mode_label="Ask")
        self.assertIn("OpenShard Receipt", out)

    def test_ask_run_mode_row(self):
        out = self._render_rich(self._ask_stages(), mode_label="Ask")
        self.assertIn("Mode", out)
        self.assertIn("Ask", out)

    def test_ask_run_no_work_row(self):
        out = self._render_rich(self._ask_stages(), mode_label="Ask")
        self.assertNotIn("Work", out)

    def test_ask_run_verify_skipped(self):
        out = self._render_rich(self._ask_stages(), mode_label="Ask")
        self.assertIn("Verify", out)
        self.assertIn("skipped", out)

    def test_ask_run_receipt_saved(self):
        out = self._render_rich(self._ask_stages(), mode_label="Ask")
        self.assertIn("Receipt", out)
        self.assertIn("saved", out)

    def test_write_run_work_row_present(self):
        out = self._render_rich(self._write_stages(), mode_label="Run")
        self.assertIn("Work", out)
        self.assertIn("passed", out)

    def test_write_run_work_shows_files_changed(self):
        out = self._render_rich(self._write_stages(n_files=2), mode_label="Run")
        self.assertIn("2 files changed", out)

    def test_cost_row_present_when_provided(self):
        out = self._render_rich(self._ask_stages(), mode_label="Ask", cost_str="$0.0020")
        self.assertIn("Cost", out)
        self.assertIn("$0.0020", out)

    def test_no_cost_row_when_absent(self):
        out = self._render_rich(self._ask_stages(), mode_label="Ask", cost_str=None)
        self.assertNotIn("Cost", out)

    def test_no_color_plain_title(self):
        out = self._render_plain(self._ask_stages(), mode_label="Ask")
        self.assertIn("OpenShard Receipt", out)

    def test_no_color_plain_receipt_saved(self):
        out = self._render_plain(self._ask_stages(), mode_label="Ask")
        self.assertIn("Receipt", out)
        self.assertIn("saved", out)

    def test_no_color_no_work_for_readonly(self):
        out = self._render_plain(self._ask_stages(), mode_label="Ask")
        self.assertNotIn("Work", out)

    def test_no_color_write_shows_files(self):
        out = self._render_plain(self._write_stages(n_files=3), mode_label="Run")
        self.assertIn("Work", out)
        self.assertIn("3 files changed", out)


# ---------------------------------------------------------------------------
# _safe_console_text — Windows CP1252 safety
# ---------------------------------------------------------------------------

class TestSafeConsoleText(unittest.TestCase):

    def _mock_stdout(self, encoding: str):
        mock_stdout = _types.SimpleNamespace(encoding=encoding)
        return patch.object(sys, "stdout", mock_stdout)

    def test_utf8_preserves_arrow(self):
        with self._mock_stdout("utf-8"):
            result = _safe_console_text("A → B")
        self.assertEqual(result, "A → B")

    def test_cp1252_does_not_raise(self):
        with self._mock_stdout("cp1252"):
            result = _safe_console_text("A → B")
        self.assertIsInstance(result, str)

    def test_cp1252_arrow_becomes_ascii(self):
        with self._mock_stdout("cp1252"):
            result = _safe_console_text("A → B")
        self.assertEqual(result, "A -> B")

    def test_ascii_em_dash_becomes_hyphen(self):
        # — is in CP1252 (0x97) so it encodes fine there; use ascii to test substitution
        with self._mock_stdout("ascii"):
            result = _safe_console_text("foo — bar")
        self.assertEqual(result, "foo - bar")

    def test_ascii_en_dash_becomes_hyphen(self):
        # – is in CP1252 (0x96) so it encodes fine there; use ascii to test substitution
        with self._mock_stdout("ascii"):
            result = _safe_console_text("foo – bar")
        self.assertEqual(result, "foo - bar")

    def test_cp1252_checkmark_becomes_ok(self):
        with self._mock_stdout("cp1252"):
            result = _safe_console_text("✓ done")
        self.assertEqual(result, "OK done")

    def test_cp1252_cross_becomes_x(self):
        with self._mock_stdout("cp1252"):
            result = _safe_console_text("✖ failed")
        self.assertEqual(result, "X failed")

    def test_cp1252_warning_becomes_bang(self):
        with self._mock_stdout("cp1252"):
            result = _safe_console_text("⚠ caution")
        self.assertEqual(result, "! caution")

    def test_ascii_passthrough(self):
        text = "plain ASCII text"
        with self._mock_stdout("ascii"):
            result = _safe_console_text(text)
        self.assertEqual(result, text)

    def test_empty_string(self):
        with self._mock_stdout("cp1252"):
            result = _safe_console_text("")
        self.assertEqual(result, "")

    def test_receipt_model_line_cp1252(self):
        """The receipt model line that caused the original crash now renders legibly."""
        receipt_line = "  Routing       Auto → DeepSeek V4 Pro"
        with self._mock_stdout("cp1252"):
            result = _safe_console_text(receipt_line)
        self.assertEqual(result, "  Routing       Auto -> DeepSeek V4 Pro")
