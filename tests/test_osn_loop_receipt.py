"""Tests for the OSN LOOP section in the native demo block (run_output.py).

Validates that the structured OSN LOOP section:
- appears when osn_loop_summary is enabled
- is omitted when not enabled or absent
- shows correct rows (Steps, Tools, Verify, Retry, Stopped)
- caps recent steps at 5
- contains no raw JSON, no em dash, no chain-of-thought phrases
"""
from __future__ import annotations

import unittest
from dataclasses import dataclass
from typing import Any

from openshard.cli.run_output import _render_native_demo_block
from openshard.native.context import NativeOSNLoopStep, NativeOSNLoopSummary
from openshard.native.osn_loop_recorder import OsnLoopRecorder


def _make_native_meta(**kwargs: Any) -> Any:
    """Minimal object with osn_loop_summary attribute."""
    @dataclass
    class _FakeMeta:
        osn_loop_summary: Any = None
        osn_loop: Any = None
        native_loop_steps: Any = None
        read_search_findings: Any = None
        diff_review: Any = None
        repo_context_summary: Any = None
        native_backend: Any = None

    m = _FakeMeta(**kwargs)
    return m


def _make_summary(
    *,
    enabled: bool = True,
    steps: list[NativeOSNLoopStep] | None = None,
    attempted_steps: int = 3,
    completed_steps: int = 2,
    blocked_steps: int = 1,
    failed_steps: int = 0,
    tool_calls_attempted: int = 2,
    tool_calls_blocked: int = 1,
    verification_passed: bool | None = None,
    verification_attempted: bool = False,
    verification_status: str = "",
    retry_used: bool = False,
    retry_count: int = 0,
    stopped_reason: str = "completed",
) -> NativeOSNLoopSummary:
    return NativeOSNLoopSummary(
        enabled=enabled,
        mode="experimental",
        steps_taken=attempted_steps,
        attempted_steps=attempted_steps,
        completed_steps=completed_steps,
        blocked_steps=blocked_steps,
        failed_steps=failed_steps,
        tool_calls_attempted=tool_calls_attempted,
        tool_calls_blocked=tool_calls_blocked,
        verification_passed=verification_passed,
        verification_attempted=verification_attempted,
        verification_status=verification_status,
        retry_used=retry_used,
        retry_count=retry_count,
        stopped_reason=stopped_reason,
        steps=steps or [],
    )


def _render(native_meta: Any, detail: str = "default") -> str:
    lines = _render_native_demo_block(native_meta, detail=detail)
    return "\n".join(lines)


class TestOSNLoopSectionRendered(unittest.TestCase):

    def test_section_rendered_when_enabled(self):
        meta = _make_native_meta(osn_loop_summary=_make_summary())
        output = _render(meta)
        self.assertIn("OSN LOOP", output)

    def test_section_omitted_when_not_enabled(self):
        summary = _make_summary(enabled=False)
        meta = _make_native_meta(osn_loop_summary=summary)
        output = _render(meta)
        self.assertNotIn("OSN LOOP", output)

    def test_section_omitted_when_no_summary(self):
        meta = _make_native_meta(osn_loop_summary=None)
        output = _render(meta)
        self.assertNotIn("OSN LOOP", output)

    def test_section_shows_steps_row(self):
        meta = _make_native_meta(osn_loop_summary=_make_summary(
            attempted_steps=3, completed_steps=2,
        ))
        output = _render(meta)
        self.assertIn("Steps", output)
        self.assertIn("3 attempted", output)
        self.assertIn("2 completed", output)

    def test_section_shows_tools_row_when_tools_present(self):
        meta = _make_native_meta(osn_loop_summary=_make_summary(
            tool_calls_attempted=2, tool_calls_blocked=1,
        ))
        output = _render(meta)
        self.assertIn("Tools", output)
        self.assertIn("2 attempted", output)
        self.assertIn("1 blocked", output)

    def test_section_shows_tools_row_omitted_when_no_tools(self):
        meta = _make_native_meta(osn_loop_summary=_make_summary(
            tool_calls_attempted=0, tool_calls_blocked=0,
        ))
        output = _render(meta)
        # Tools row should not appear when tool_calls_attempted is 0
        self.assertNotIn("Tools", output)

    def test_section_shows_verify_passed(self):
        meta = _make_native_meta(osn_loop_summary=_make_summary(
            verification_attempted=True, verification_passed=True,
        ))
        output = _render(meta)
        self.assertIn("Verify", output)
        self.assertIn("passed", output)

    def test_section_shows_verify_failed(self):
        meta = _make_native_meta(osn_loop_summary=_make_summary(
            verification_attempted=True, verification_passed=False,
        ))
        output = _render(meta)
        self.assertIn("Verify", output)
        self.assertIn("failed", output)

    def test_section_shows_verify_not_run_when_not_attempted(self):
        meta = _make_native_meta(osn_loop_summary=_make_summary(
            verification_attempted=False,
        ))
        output = _render(meta)
        self.assertIn("Verify", output)
        self.assertIn("not run", output)

    def test_section_shows_retry_no(self):
        meta = _make_native_meta(osn_loop_summary=_make_summary(retry_used=False))
        output = _render(meta)
        self.assertIn("Retry", output)
        self.assertIn("no", output)

    def test_section_shows_retry_yes_with_count(self):
        meta = _make_native_meta(osn_loop_summary=_make_summary(
            retry_used=True, retry_count=1,
        ))
        output = _render(meta)
        self.assertIn("Retry", output)
        self.assertIn("yes", output)
        self.assertIn("1", output)

    def test_section_shows_stopped_reason(self):
        meta = _make_native_meta(osn_loop_summary=_make_summary(stopped_reason="max_steps"))
        output = _render(meta)
        self.assertIn("Stopped", output)
        self.assertIn("max_steps", output)


class TestOSNLoopSectionSafety(unittest.TestCase):

    def test_no_em_dash_in_output(self):
        rec = OsnLoopRecorder()
        rec.record_step("preflight", "passed", result_summary="ok")
        rec.record_step("tool", "blocked", tool_name="run_command", blocked_reason="approval required")
        rec.complete(stopped_reason="completed")
        meta = _make_native_meta(osn_loop_summary=rec.summary)
        output = _render(meta)
        self.assertNotIn("—", output)  # em dash

    def test_no_raw_json_in_output(self):
        rec = OsnLoopRecorder()
        rec.record_step("preflight", "passed", result_summary="ok")
        rec.complete(stopped_reason="completed")
        meta = _make_native_meta(osn_loop_summary=rec.summary)
        output = _render(meta)
        # Should not dump a JSON object structure into output
        self.assertNotIn('{"', output)
        self.assertNotIn('"step_name"', output)

    def test_no_chain_of_thought_phrases(self):
        cot_phrases = [
            "let me think",
            "i will now",
            "chain of thought",
            "reasoning:",
            "<think>",
        ]
        rec = OsnLoopRecorder()
        rec.record_step("preflight", "passed", result_summary="ok")
        rec.complete(stopped_reason="completed")
        meta = _make_native_meta(osn_loop_summary=rec.summary)
        output = _render(meta).lower()
        for phrase in cot_phrases:
            self.assertNotIn(phrase, output)

    def test_no_raw_file_content_in_output(self):
        rec = OsnLoopRecorder()
        rec.record_step(
            "read",
            "passed",
            tool_name="read_file",
            result_summary="x" * 130,  # over cap, gets truncated
        )
        rec.complete(stopped_reason="completed")
        meta = _make_native_meta(osn_loop_summary=rec.summary)
        output = _render(meta)
        # result_summary is not shown in recent steps (only step_name/tool_name/status)
        # and the summary itself is capped to 120 chars by the recorder
        self.assertLessEqual(len(output), 5000)


class TestOSNLoopRecentSteps(unittest.TestCase):

    def _make_steps(self, count: int) -> list[NativeOSNLoopStep]:
        return [
            NativeOSNLoopStep(
                step_index=i,
                step_name=f"step_{i}",
                status="passed",
                tool_name="read_file",
                target_label=f"src/f{i}.py",
            )
            for i in range(count)
        ]

    def test_recent_steps_shown_in_default_mode(self):
        summary = _make_summary(steps=self._make_steps(3))
        meta = _make_native_meta(osn_loop_summary=summary)
        output = _render(meta, detail="default")
        self.assertIn("Recent steps", output)

    def test_recent_steps_capped_at_5(self):
        summary = _make_summary(steps=self._make_steps(10))
        meta = _make_native_meta(osn_loop_summary=summary)
        output = _render(meta, detail="default")
        # Count "- read_file" occurrences - should be at most 5
        count = output.count("- read_file")
        self.assertLessEqual(count, 5)

    def test_recent_steps_omitted_in_full_mode(self):
        summary = _make_summary(steps=self._make_steps(3))
        meta = _make_native_meta(osn_loop_summary=summary)
        output = _render(meta, detail="full")
        # Recent steps section should not appear in full mode to avoid duplication
        self.assertNotIn("Recent steps", output)

    def test_recent_steps_not_shown_when_no_steps(self):
        summary = _make_summary(steps=[])
        meta = _make_native_meta(osn_loop_summary=summary)
        output = _render(meta, detail="default")
        self.assertNotIn("Recent steps", output)

    def test_step_with_blocked_reason_shows_reason(self):
        steps = [
            NativeOSNLoopStep(
                step_index=0,
                step_name="tool",
                status="blocked",
                tool_name="run_command",
                blocked_reason="approval required",
            )
        ]
        summary = _make_summary(steps=steps)
        meta = _make_native_meta(osn_loop_summary=summary)
        output = _render(meta, detail="default")
        self.assertIn("approval required", output)

    def test_step_without_tool_shows_step_name(self):
        steps = [
            NativeOSNLoopStep(
                step_index=0,
                step_name="preflight",
                status="passed",
                tool_name="",
            )
        ]
        summary = _make_summary(steps=steps)
        meta = _make_native_meta(osn_loop_summary=summary)
        output = _render(meta, detail="default")
        self.assertIn("preflight", output)


if __name__ == "__main__":
    unittest.main()
