"""Tests for OSN Progress Memory v1.

Covers:
- OSNProgressMemory dataclass defaults, caps, serialization safety
- build_osn_progress_memory logic from all four source signals
- render_osn_progress_context output rules
- Receipt rendering (OSN PROGRESS section)
- Integration: NativeRunMeta field, pipeline persistence, history entry mapping
- Regression: no em dash, no absolute paths, no raw content, no model calls
"""
from __future__ import annotations

import json
import unittest
from dataclasses import asdict
from typing import Any
from unittest.mock import patch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_obs(
    *,
    enabled: bool = True,
    stack_signals: list[str] | None = None,
    candidate_files: list[str] | None = None,
    suggested_checks: list[str] | None = None,
) -> Any:
    from openshard.native.context import OSNObservationPacket
    return OSNObservationPacket(
        enabled=enabled,
        stack_signals=stack_signals or [],
        candidate_files=candidate_files or [],
        suggested_checks=suggested_checks or [],
    )


def _make_loop_summary(
    *,
    enabled: bool = True,
    verification_attempted: bool = False,
    verification_passed: bool | None = None,
    retry_used: bool = False,
    retry_count: int = 0,
    stopped_reason: str = "completed",
    steps: list[Any] | None = None,
) -> Any:
    from openshard.native.context import NativeOSNLoopSummary
    return NativeOSNLoopSummary(
        enabled=enabled,
        mode="experimental",
        verification_attempted=verification_attempted,
        verification_passed=verification_passed,
        retry_used=retry_used,
        retry_count=retry_count,
        stopped_reason=stopped_reason,
        steps=steps or [],
    )


def _make_loop_step(step_name: str, status: str) -> Any:
    from openshard.native.context import NativeOSNLoopStep
    return NativeOSNLoopStep(step_name=step_name, status=status)


def _make_vc(
    *,
    enabled: bool = True,
    status: str = "not_run",
    missing_checks: list[str] | None = None,
    manual_review_required: bool = False,
) -> Any:
    from openshard.native.verification_contract import OSNVerificationContract
    return OSNVerificationContract(
        enabled=enabled,
        status=status,
        missing_checks=missing_checks or [],
        manual_review_required=manual_review_required,
    )


def _make_rd(
    *,
    enabled: bool = True,
    status: str = "not_needed",
    next_action: str = "",
    manual_review_required: bool = False,
    retry_used: bool = False,
    retry_count: int = 0,
) -> Any:
    from openshard.native.retry_diagnosis import OSNRetryDiagnosis
    return OSNRetryDiagnosis(
        enabled=enabled,
        status=status,
        next_action=next_action,
        manual_review_required=manual_review_required,
        retry_used=retry_used,
        retry_count=retry_count,
    )


def _make_native_meta(**kwargs: Any) -> Any:
    """Minimal namespace-style object for receipt rendering tests."""
    class _FakeMeta:
        def __init__(self, **kw: Any) -> None:
            for k, v in kw.items():
                setattr(self, k, v)
    m = _FakeMeta(**kwargs)
    # set None for any attrs used by the renderer that were not supplied
    for attr in (
        "osn_observation", "osn_progress_memory", "osn_verification_contract",
        "osn_loop_summary", "osn_retry_diagnosis", "osn_loop",
        "context_packet", "repo_context_summary", "observation",
        "diff_review", "native_backend", "native_loop_steps",
        "read_search_findings", "patch_proposal", "file_context",
        "context_quality_score", "context_quality_advisory",
        "change_budget", "change_budget_preview", "change_budget_soft_gate",
        "approval_request", "approval_receipt", "verification_loop",
        "verification_command_summary", "validation_contract",
        "verification_contract_result", "context_provenance",
        "run_trust_score", "model_selection_decision", "model_candidate_scoring",
        "model_policy", "model_policy_receipt", "routing_preview",
        "routing_receipt", "tier_dispatch_receipt", "sandbox",
        "failure_memory_routing_advisory", "plan_ledger", "edit_loop_summary",
        "candidate_summary", "context_usage_summary", "failure_memory",
        "deepagents_adapter",
    ):
        if not hasattr(m, attr):
            setattr(m, attr, None)
    return m


def _render_block(native_meta: Any, detail: str = "default") -> list[str]:
    from openshard.cli.run_output import _render_native_demo_block
    result = _render_native_demo_block(native_meta, detail=detail, entry={})
    if isinstance(result, tuple):
        return result[0]
    return result


# ---------------------------------------------------------------------------
# 1-5: OSNProgressMemory defaults and serialization
# ---------------------------------------------------------------------------

class TestOSNProgressMemoryDefaults(unittest.TestCase):

    def test_defaults_are_safe(self):
        from openshard.native.progress_memory import OSNProgressMemory
        m = OSNProgressMemory()
        self.assertFalse(m.enabled)
        self.assertFalse(m.raw_content_stored)
        self.assertEqual(m.confidence, "low")
        self.assertEqual(m.source, "osn_progress_memory_v1")
        self.assertEqual(m.summary, "")
        self.assertEqual(m.completed, [])
        self.assertEqual(m.current_focus, [])
        self.assertEqual(m.relevant_files, [])
        self.assertEqual(m.unresolved_items, [])
        self.assertEqual(m.blockers, [])
        self.assertEqual(m.next_safe_step, "")

    def test_raw_content_stored_always_false(self):
        from openshard.native.progress_memory import OSNProgressMemory
        m = OSNProgressMemory(raw_content_stored=True)
        self.assertFalse(m.raw_content_stored)

    def test_serializes_via_asdict(self):
        from openshard.native.progress_memory import OSNProgressMemory
        m = OSNProgressMemory(enabled=True, summary="ok", confidence="medium")
        d = asdict(m)
        self.assertIsInstance(d, dict)
        json.dumps(d)  # must not raise

    def test_source_field(self):
        from openshard.native.progress_memory import OSNProgressMemory
        self.assertEqual(OSNProgressMemory().source, "osn_progress_memory_v1")

    def test_valid_confidence_values_stable(self):
        from openshard.native.progress_memory import OSNProgressMemory, _VALID_CONFIDENCE
        for val in ("low", "medium", "high", "unknown"):
            self.assertIn(val, _VALID_CONFIDENCE)
            m = OSNProgressMemory(confidence=val)
            self.assertEqual(m.confidence, val)


# ---------------------------------------------------------------------------
# 6-13: Cap enforcement
# ---------------------------------------------------------------------------

class TestOSNProgressMemoryCaps(unittest.TestCase):

    def test_summary_capped_at_240(self):
        from openshard.native.progress_memory import OSNProgressMemory
        m = OSNProgressMemory(summary="x" * 300)
        self.assertEqual(len(m.summary), 240)

    def test_completed_capped_at_5(self):
        from openshard.native.progress_memory import OSNProgressMemory
        m = OSNProgressMemory(completed=["step"] * 8)
        self.assertEqual(len(m.completed), 5)

    def test_completed_item_capped_at_120(self):
        from openshard.native.progress_memory import OSNProgressMemory
        m = OSNProgressMemory(completed=["x" * 200])
        self.assertEqual(len(m.completed[0]), 120)

    def test_relevant_files_capped_at_8(self):
        from openshard.native.progress_memory import OSNProgressMemory
        m = OSNProgressMemory(relevant_files=["f.py"] * 12)
        self.assertEqual(len(m.relevant_files), 8)

    def test_unresolved_capped_at_5(self):
        from openshard.native.progress_memory import OSNProgressMemory
        m = OSNProgressMemory(unresolved_items=["item"] * 9)
        self.assertEqual(len(m.unresolved_items), 5)

    def test_blockers_capped_at_5(self):
        from openshard.native.progress_memory import OSNProgressMemory
        m = OSNProgressMemory(blockers=["b"] * 9)
        self.assertEqual(len(m.blockers), 5)

    def test_next_safe_step_capped_at_160(self):
        from openshard.native.progress_memory import OSNProgressMemory
        m = OSNProgressMemory(next_safe_step="n" * 200)
        self.assertEqual(len(m.next_safe_step), 160)

    def test_invalid_confidence_normalized_to_unknown(self):
        from openshard.native.progress_memory import OSNProgressMemory
        m = OSNProgressMemory(confidence="bogus_value")
        self.assertEqual(m.confidence, "unknown")


# ---------------------------------------------------------------------------
# 14-35: build_osn_progress_memory logic
# ---------------------------------------------------------------------------

class TestBuildOSNProgressMemory(unittest.TestCase):

    def _build(self, **kwargs: Any) -> Any:
        from openshard.native.progress_memory import build_osn_progress_memory
        return build_osn_progress_memory(**kwargs)

    def test_no_signals_returns_disabled(self):
        m = self._build()
        self.assertFalse(m.enabled)
        self.assertEqual(m.confidence, "unknown")

    def test_observation_only_low_confidence(self):
        obs = _make_obs(stack_signals=["python"])
        m = self._build(osn_observation=obs)
        self.assertTrue(m.enabled)
        self.assertEqual(m.confidence, "low")

    def test_stack_signals_become_current_focus(self):
        obs = _make_obs(stack_signals=["python", "docker"])
        m = self._build(osn_observation=obs)
        self.assertEqual(m.current_focus, ["python", "docker"])

    def test_candidate_files_become_relevant_files(self):
        obs = _make_obs(candidate_files=["openshard/native/executor.py", "openshard/run/pipeline.py"])
        m = self._build(osn_observation=obs)
        self.assertIn("openshard/native/executor.py", m.relevant_files)
        self.assertIn("openshard/run/pipeline.py", m.relevant_files)

    def test_absolute_paths_stripped_from_relevant_files(self):
        obs = _make_obs(candidate_files=["/absolute/path.py", "relative/path.py"])
        m = self._build(osn_observation=obs)
        for f in m.relevant_files:
            self.assertFalse(f.startswith("/"))
        self.assertIn("relative/path.py", m.relevant_files)

    def test_windows_absolute_paths_stripped(self):
        obs = _make_obs(candidate_files=["C:/Users/foo/bar.py", "relative/bar.py"])
        m = self._build(osn_observation=obs)
        for f in m.relevant_files:
            self.assertFalse(f.startswith("C:"))
        self.assertIn("relative/bar.py", m.relevant_files)

    def test_codegraph_paths_excluded(self):
        obs = _make_obs(candidate_files=[".codegraph/index.db", "openshard/native/executor.py"])
        m = self._build(osn_observation=obs)
        for f in m.relevant_files:
            self.assertNotIn(".codegraph", f)
        self.assertIn("openshard/native/executor.py", m.relevant_files)

    def test_passed_steps_become_completed(self):
        steps = [
            _make_loop_step("observe", "passed"),
            _make_loop_step("plan_update", "passed"),
            _make_loop_step("verify", "failed"),
        ]
        loop = _make_loop_summary(steps=steps)
        obs = _make_obs()
        m = self._build(osn_observation=obs, osn_loop_summary=loop)
        self.assertIn("observe", m.completed)
        self.assertIn("plan_update", m.completed)
        self.assertNotIn("verify", m.completed)

    def test_blocked_steps_become_blockers(self):
        steps = [
            _make_loop_step("approval", "blocked"),
            _make_loop_step("verify", "passed"),
        ]
        loop = _make_loop_summary(steps=steps)
        obs = _make_obs()
        m = self._build(osn_observation=obs, osn_loop_summary=loop)
        blocker_str = " ".join(m.blockers)
        self.assertIn("approval", blocker_str)

    def test_retry_status_blocked_adds_blocker(self):
        obs = _make_obs()
        loop = _make_loop_summary()
        rd = _make_rd(status="blocked")
        m = self._build(osn_observation=obs, osn_loop_summary=loop, osn_retry_diagnosis=rd)
        blocker_str = " ".join(m.blockers)
        self.assertIn("blocked", blocker_str)

    def test_retry_status_exhausted_adds_blocker(self):
        obs = _make_obs()
        loop = _make_loop_summary()
        rd = _make_rd(status="exhausted")
        m = self._build(osn_observation=obs, osn_loop_summary=loop, osn_retry_diagnosis=rd)
        blocker_str = " ".join(m.blockers)
        self.assertIn("exhausted", blocker_str)

    def test_retry_manual_review_required_adds_blocker(self):
        obs = _make_obs()
        loop = _make_loop_summary()
        rd = _make_rd(manual_review_required=True)
        m = self._build(osn_observation=obs, osn_loop_summary=loop, osn_retry_diagnosis=rd)
        all_items = m.blockers + m.unresolved_items
        self.assertTrue(any("manual review" in s for s in all_items))

    def test_vc_manual_review_required_adds_blocker(self):
        obs = _make_obs()
        loop = _make_loop_summary()
        vc = _make_vc(manual_review_required=True)
        m = self._build(osn_observation=obs, osn_loop_summary=loop, osn_verification_contract=vc)
        all_items = m.blockers + m.unresolved_items
        self.assertTrue(any("manual review" in s for s in all_items))

    def test_vc_missing_checks_become_unresolved(self):
        obs = _make_obs()
        loop = _make_loop_summary()
        vc = _make_vc(missing_checks=["run pytest", "run ruff"])
        m = self._build(osn_observation=obs, osn_loop_summary=loop, osn_verification_contract=vc)
        self.assertIn("run pytest", m.unresolved_items)
        self.assertIn("run ruff", m.unresolved_items)

    def test_vc_passed_gives_high_confidence(self):
        obs = _make_obs()
        loop = _make_loop_summary(verification_attempted=True, verification_passed=True)
        vc = _make_vc(status="passed")
        m = self._build(osn_observation=obs, osn_loop_summary=loop, osn_verification_contract=vc)
        self.assertEqual(m.confidence, "high")

    def test_verification_attempted_not_passed_gives_medium(self):
        obs = _make_obs()
        loop = _make_loop_summary(verification_attempted=True, verification_passed=False)
        m = self._build(osn_observation=obs, osn_loop_summary=loop)
        self.assertEqual(m.confidence, "medium")

    def test_retry_next_action_becomes_next_safe_step(self):
        obs = _make_obs()
        loop = _make_loop_summary()
        rd = _make_rd(next_action="fix failing verification path")
        m = self._build(osn_observation=obs, osn_loop_summary=loop, osn_retry_diagnosis=rd)
        self.assertEqual(m.next_safe_step, "fix failing verification path")

    def test_suggested_check_fallback_for_next_safe_step(self):
        obs = _make_obs(suggested_checks=["python -m pytest tests/ -v"])
        loop = _make_loop_summary()
        m = self._build(osn_observation=obs, osn_loop_summary=loop)
        self.assertEqual(m.next_safe_step, "python -m pytest tests/ -v")

    def test_retry_next_action_takes_priority_over_suggested_check(self):
        obs = _make_obs(suggested_checks=["python -m pytest tests/ -v"])
        loop = _make_loop_summary()
        rd = _make_rd(next_action="fix failing verification path")
        m = self._build(osn_observation=obs, osn_loop_summary=loop, osn_retry_diagnosis=rd)
        self.assertEqual(m.next_safe_step, "fix failing verification path")

    def test_not_needed_retry_no_noise(self):
        obs = _make_obs()
        loop = _make_loop_summary(verification_attempted=True, verification_passed=True)
        rd = _make_rd(status="not_needed")
        m = self._build(osn_observation=obs, osn_loop_summary=loop, osn_retry_diagnosis=rd)
        blocker_str = " ".join(m.blockers)
        self.assertNotIn("not_needed", blocker_str)

    def test_summary_is_deterministic(self):
        obs = _make_obs(stack_signals=["python"])
        loop = _make_loop_summary(steps=[_make_loop_step("observe", "passed")])
        m1 = self._build(osn_observation=obs, osn_loop_summary=loop)
        m2 = self._build(osn_observation=obs, osn_loop_summary=loop)
        self.assertEqual(m1.summary, m2.summary)

    def test_no_raw_content_stored_in_result(self):
        obs = _make_obs()
        loop = _make_loop_summary()
        m = self._build(osn_observation=obs, osn_loop_summary=loop)
        self.assertFalse(m.raw_content_stored)

    def test_json_roundtrip_stable(self):
        obs = _make_obs(stack_signals=["python"], candidate_files=["openshard/native/executor.py"])
        loop = _make_loop_summary(steps=[_make_loop_step("observe", "passed")])
        vc = _make_vc(status="passed")
        rd = _make_rd(next_action="verify result before merging")
        m = self._build(osn_observation=obs, osn_loop_summary=loop, osn_verification_contract=vc, osn_retry_diagnosis=rd)
        d = asdict(m)
        json.dumps(d)  # must not raise
        self.assertEqual(d["source"], "osn_progress_memory_v1")
        self.assertFalse(d["raw_content_stored"])


# ---------------------------------------------------------------------------
# 36-43: render_osn_progress_context
# ---------------------------------------------------------------------------

class TestRenderOSNProgressContext(unittest.TestCase):

    def _render(self, memory: Any) -> str:
        from openshard.native.progress_memory import render_osn_progress_context
        return render_osn_progress_context(memory)

    def test_none_returns_empty(self):
        self.assertEqual(self._render(None), "")

    def test_disabled_returns_empty(self):
        from openshard.native.progress_memory import OSNProgressMemory
        self.assertEqual(self._render(OSNProgressMemory(enabled=False)), "")

    def test_enabled_renders_header(self):
        from openshard.native.progress_memory import OSNProgressMemory
        m = OSNProgressMemory(enabled=True, summary="test", confidence="low")
        out = self._render(m)
        self.assertTrue(out.startswith("[osn progress]"))

    def test_output_contains_expected_lines(self):
        from openshard.native.progress_memory import OSNProgressMemory
        m = OSNProgressMemory(
            enabled=True,
            summary="repo observed",
            current_focus=["python"],
            relevant_files=["openshard/native/executor.py"],
            unresolved_items=["run tests"],
            next_safe_step="run pytest",
        )
        out = self._render(m)
        self.assertIn("summary: repo observed", out)
        self.assertIn("focus: python", out)
        self.assertIn("files: openshard/native/executor.py", out)
        self.assertIn("unresolved: run tests", out)
        self.assertIn("next: run pytest", out)

    def test_no_em_dash(self):
        from openshard.native.progress_memory import OSNProgressMemory
        m = OSNProgressMemory(enabled=True, summary="test", next_safe_step="do something")
        out = self._render(m)
        self.assertNotIn("—", out)  # em dash
        self.assertNotIn("–", out)  # en dash

    def test_no_raw_json(self):
        from openshard.native.progress_memory import OSNProgressMemory
        m = OSNProgressMemory(enabled=True, summary="test")
        out = self._render(m)
        self.assertNotIn("{", out)
        self.assertNotIn("}", out)

    def test_no_absolute_paths(self):
        from openshard.native.progress_memory import OSNProgressMemory
        # absolute paths should never reach here; test renderer doesn't inject them
        m = OSNProgressMemory(enabled=True, relevant_files=["relative/path.py"])
        out = self._render(m)
        self.assertNotIn("/absolute", out)

    def test_no_chain_of_thought_phrases(self):
        from openshard.native.progress_memory import OSNProgressMemory
        m = OSNProgressMemory(enabled=True, summary="repo observed, verification pending")
        out = self._render(m)
        for phrase in ("let me", "i think", "reasoning", "i will", "i need to"):
            self.assertNotIn(phrase, out.lower())


# ---------------------------------------------------------------------------
# 44-55: Receipt rendering
# ---------------------------------------------------------------------------

class TestOSNProgressMemoryReceiptRendering(unittest.TestCase):

    def _make_pm(self, **kwargs: Any) -> Any:
        from openshard.native.progress_memory import OSNProgressMemory
        return OSNProgressMemory(**kwargs)

    def test_section_absent_when_disabled(self):
        meta = _make_native_meta(osn_progress_memory=self._make_pm(enabled=False))
        lines = _render_block(meta)
        self.assertNotIn("  OSN PROGRESS", lines)

    def test_section_absent_when_missing(self):
        meta = _make_native_meta(osn_progress_memory=None)
        lines = _render_block(meta)
        self.assertNotIn("  OSN PROGRESS", lines)

    def test_section_present_when_enabled(self):
        pm = self._make_pm(enabled=True, confidence="medium", summary="ok")
        meta = _make_native_meta(osn_progress_memory=pm)
        lines = _render_block(meta)
        self.assertIn("  OSN PROGRESS", lines)

    def test_shows_confidence_row(self):
        pm = self._make_pm(enabled=True, confidence="medium")
        meta = _make_native_meta(osn_progress_memory=pm)
        lines = _render_block(meta)
        joined = "\n".join(lines)
        self.assertIn("Confidence", joined)
        self.assertIn("medium", joined)

    def test_shows_summary_row_when_present(self):
        pm = self._make_pm(enabled=True, summary="repo observed, verification pending")
        meta = _make_native_meta(osn_progress_memory=pm)
        lines = _render_block(meta)
        joined = "\n".join(lines)
        self.assertIn("Summary", joined)
        self.assertIn("repo observed", joined)

    def test_shows_files_count(self):
        pm = self._make_pm(enabled=True, relevant_files=["a.py", "b.py", "c.py"])
        meta = _make_native_meta(osn_progress_memory=pm)
        lines = _render_block(meta)
        joined = "\n".join(lines)
        self.assertIn("3 relevant", joined)

    def test_shows_open_count_only_when_nonzero(self):
        pm_zero = self._make_pm(enabled=True, unresolved_items=[])
        meta_zero = _make_native_meta(osn_progress_memory=pm_zero)
        lines_zero = _render_block(meta_zero)
        self.assertFalse(any("Open" in ln for ln in lines_zero))

        pm_has = self._make_pm(enabled=True, unresolved_items=["run tests"])
        meta_has = _make_native_meta(osn_progress_memory=pm_has)
        lines_has = _render_block(meta_has)
        self.assertTrue(any("Open" in ln for ln in lines_has))

    def test_shows_next_step(self):
        pm = self._make_pm(enabled=True, next_safe_step="run focused verification before merge")
        meta = _make_native_meta(osn_progress_memory=pm)
        lines = _render_block(meta)
        joined = "\n".join(lines)
        self.assertIn("run focused verification before merge", joined)

    def test_full_detail_shows_file_list(self):
        pm = self._make_pm(
            enabled=True,
            relevant_files=["openshard/native/executor.py", "openshard/run/pipeline.py"],
        )
        meta = _make_native_meta(osn_progress_memory=pm)
        lines = _render_block(meta, detail="full")
        joined = "\n".join(lines)
        self.assertIn("openshard/native/executor.py", joined)
        self.assertIn("openshard/run/pipeline.py", joined)

    def test_full_detail_caps_files_at_5(self):
        pm = self._make_pm(
            enabled=True,
            relevant_files=[f"file{i}.py" for i in range(8)],
        )
        meta = _make_native_meta(osn_progress_memory=pm)
        lines = _render_block(meta, detail="full")
        file_lines = [ln for ln in lines if ln.strip().startswith("- file")]
        self.assertLessEqual(len(file_lines), 5)

    def test_no_raw_json_in_output(self):
        pm = self._make_pm(enabled=True, summary="ok", confidence="low")
        meta = _make_native_meta(osn_progress_memory=pm)
        joined = "\n".join(_render_block(meta))
        self.assertNotIn("{", joined)
        self.assertNotIn("}", joined)

    def test_no_em_dash_in_output(self):
        pm = self._make_pm(enabled=True, summary="test summary")
        meta = _make_native_meta(osn_progress_memory=pm)
        joined = "\n".join(_render_block(meta))
        self.assertNotIn("—", joined)
        self.assertNotIn("–", joined)

    def test_ordering_observation_before_progress_before_verification(self):
        from openshard.native.context import OSNObservationPacket
        from openshard.native.verification_contract import OSNVerificationContract
        obs = OSNObservationPacket(enabled=True, stack_signals=["python"])
        pm = self._make_pm(enabled=True, confidence="low", summary="snapshot")
        vc = OSNVerificationContract(enabled=True, status="skipped")
        meta = _make_native_meta(osn_observation=obs, osn_progress_memory=pm, osn_verification_contract=vc)
        lines = _render_block(meta)
        obs_idx = next((i for i, ln in enumerate(lines) if "OSN OBSERVATION" in ln), -1)
        pm_idx = next((i for i, ln in enumerate(lines) if "OSN PROGRESS" in ln), -1)
        vc_idx = next((i for i, ln in enumerate(lines) if "OSN VERIFICATION" in ln), -1)
        self.assertGreater(obs_idx, -1, "OSN OBSERVATION not found")
        self.assertGreater(pm_idx, -1, "OSN PROGRESS not found")
        self.assertGreater(vc_idx, -1, "OSN VERIFICATION not found")
        self.assertLess(obs_idx, pm_idx, "OSN OBSERVATION must come before OSN PROGRESS")
        self.assertLess(pm_idx, vc_idx, "OSN PROGRESS must come before OSN VERIFICATION")


# ---------------------------------------------------------------------------
# 56-61: Integration tests
# ---------------------------------------------------------------------------

class TestOSNProgressMemoryIntegration(unittest.TestCase):

    def test_native_run_meta_has_field(self):
        from openshard.native.executor import NativeRunMeta
        meta = NativeRunMeta()
        self.assertIsNone(meta.osn_progress_memory)

    def test_native_run_meta_asdict_succeeds(self):
        from openshard.native.executor import NativeRunMeta
        d = asdict(NativeRunMeta())
        self.assertIn("osn_progress_memory", d)
        self.assertIsNone(d["osn_progress_memory"])

    def test_native_run_meta_with_progress_memory_asdict_succeeds(self):
        from openshard.native.executor import NativeRunMeta
        from openshard.native.progress_memory import OSNProgressMemory
        meta = NativeRunMeta()
        meta.osn_progress_memory = OSNProgressMemory(enabled=True, confidence="low", summary="ok")
        d = asdict(meta)
        self.assertIsNotNone(d["osn_progress_memory"])
        json.dumps(d["osn_progress_memory"])  # must not raise

    def test_native_meta_from_entry_reads_progress_memory(self):
        from openshard.native.progress_memory import OSNProgressMemory
        from openshard.cli.run_output import _native_meta_from_entry
        pm_dict = asdict(OSNProgressMemory(enabled=True, confidence="medium", summary="snapshot"))
        entry = {
            "workflow": "native",
            "osn_progress_memory": pm_dict,
        }
        meta = _native_meta_from_entry(entry)
        self.assertIsNotNone(meta)
        pm = getattr(meta, "osn_progress_memory", None)
        self.assertIsNotNone(pm)
        self.assertEqual(pm.get("enabled") if isinstance(pm, dict) else getattr(pm, "enabled", None), True)

    def test_history_entry_renders_osn_progress(self):
        from openshard.native.progress_memory import OSNProgressMemory
        from openshard.cli.run_output import _native_meta_from_entry
        pm_dict = asdict(OSNProgressMemory(
            enabled=True, confidence="medium", summary="repo observed",
            relevant_files=["openshard/native/executor.py"],
        ))
        entry = {"workflow": "native", "osn_progress_memory": pm_dict}
        native_meta = _native_meta_from_entry(entry)
        self.assertIsNotNone(native_meta)
        lines = _render_block(native_meta)
        self.assertIn("  OSN PROGRESS", lines)

    def test_no_shell_command_executed_during_build(self):
        import subprocess
        from openshard.native.progress_memory import build_osn_progress_memory
        obs = _make_obs(stack_signals=["python"])
        loop = _make_loop_summary()
        with patch.object(subprocess, "run", side_effect=AssertionError("shell call not allowed")):
            with patch.object(subprocess, "Popen", side_effect=AssertionError("shell call not allowed")):
                m = build_osn_progress_memory(osn_observation=obs, osn_loop_summary=loop)
        self.assertTrue(m.enabled)


if __name__ == "__main__":
    unittest.main()
