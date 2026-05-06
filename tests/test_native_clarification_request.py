from __future__ import annotations

import json
import unittest
from dataclasses import asdict
from types import SimpleNamespace

from openshard.native.context import (
    NativeClarificationRequest,
    NativeVerificationPlan,
    build_native_clarification_request,
    build_native_verification_plan,
)
from openshard.cli.run_output import _native_meta_from_entry, _render_native_demo_block


def _vplan(task_type: str = "unknown", clarification_needed: list[str] | None = None) -> NativeVerificationPlan:
    vp = NativeVerificationPlan(task_type=task_type)
    vp.clarification_needed = clarification_needed if clarification_needed is not None else []
    return vp


def _build(task: str = "", task_type: str = "unknown", clarification_needed: list[str] | None = None) -> NativeClarificationRequest:
    vp = _vplan(task_type=task_type, clarification_needed=clarification_needed)
    return build_native_clarification_request(task=task, verification_plan=vp)


def _meta(**kwargs):
    defaults = dict(
        repo_context_summary=None,
        observation=None,
        plan=None,
        verification_plan=None,
        clarification_request=None,
        write_path="pipeline",
        verification_loop=None,
        verification_command_summary=None,
        diff_review=None,
        final_report=None,
        native_loop_steps=[],
        native_loop_trace=[],
        native_backend=None,
        native_backend_available=True,
        native_backend_notes=[],
        native_backend_proof=None,
        read_search_findings=[],
        patch_proposal=None,
        command_policy_preview=None,
        context_packet=None,
        file_context=None,
        context_quality_score=None,
        context_quality_advisory=None,
        change_budget=None,
        change_budget_preview=None,
        change_budget_soft_gate=None,
        approval_request=None,
        approval_receipt=None,
    )
    defaults.update(kwargs)
    return SimpleNamespace(**defaults)


def _render(clarification_request=None) -> str:
    meta = _meta(clarification_request=clarification_request)
    return "\n".join(_render_native_demo_block(meta))


class TestBuildNativeClarificationRequest(unittest.TestCase):

    def test_not_needed_when_task_type_known(self):
        cr = _build(task_type="feature")
        self.assertFalse(cr.needed)

    def test_not_needed_returns_no_question(self):
        cr = _build(task_type="bugfix")
        self.assertIsNone(cr.question)

    def test_not_needed_returns_empty_options(self):
        cr = _build(task_type="refactor")
        self.assertEqual(cr.options, [])

    def test_needed_when_task_type_unknown(self):
        cr = _build(task_type="unknown")
        self.assertTrue(cr.needed)

    def test_needed_sets_question(self):
        cr = _build(task_type="unknown")
        self.assertIsNotNone(cr.question)
        self.assertGreater(len(cr.question), 0)

    def test_needed_sets_options(self):
        cr = _build(task_type="unknown")
        self.assertGreater(len(cr.options), 0)

    def test_allows_custom_true_when_needed(self):
        cr = _build(task_type="unknown")
        self.assertTrue(cr.allows_custom)

    def test_allows_custom_false_when_not_needed(self):
        cr = _build(task_type="feature")
        self.assertFalse(cr.allows_custom)

    def test_task_field_set_when_unknown(self):
        cr = _build(task_type="unknown")
        self.assertEqual(cr.task_field, "task_type")

    def test_task_field_none_when_not_needed(self):
        cr = _build(task_type="docs")
        self.assertIsNone(cr.task_field)

    def test_reason_from_clarification_needed(self):
        cr = _build(task_type="unknown", clarification_needed=["ambiguous task"])
        self.assertEqual(cr.reason, "ambiguous task")

    def test_reason_none_when_clarification_needed_empty_but_unknown(self):
        cr = _build(task_type="unknown", clarification_needed=[])
        self.assertIsNone(cr.reason)

    def test_needed_via_verification_plan_builder(self):
        vp = build_native_verification_plan(
            task="do the thing",
            plan=None,
            change_budget=None,
            read_search_findings=[],
            repo_facts=None,
        )
        cr = build_native_clarification_request(task="do the thing", verification_plan=vp)
        self.assertTrue(cr.needed)

    def test_not_needed_via_verification_plan_builder(self):
        vp = build_native_verification_plan(
            task="fix the crash in auth",
            plan=None,
            change_budget=None,
            read_search_findings=[],
            repo_facts=None,
        )
        cr = build_native_clarification_request(task="fix the crash in auth", verification_plan=vp)
        self.assertFalse(cr.needed)


class TestNativeClarificationRequestSerialization(unittest.TestCase):

    def test_asdict_roundtrip_needed(self):
        cr = _build(task_type="unknown", clarification_needed=["ambiguous"])
        d = asdict(cr)
        self.assertTrue(d["needed"])
        self.assertIsNotNone(d["question"])
        self.assertIsInstance(d["options"], list)
        self.assertTrue(d["allows_custom"])
        self.assertEqual(d["reason"], "ambiguous")
        self.assertEqual(d["task_field"], "task_type")

    def test_asdict_roundtrip_not_needed(self):
        cr = _build(task_type="feature")
        d = asdict(cr)
        self.assertFalse(d["needed"])
        self.assertIsNone(d["question"])
        self.assertEqual(d["options"], [])

    def test_json_serializable_needed(self):
        cr = _build(task_type="unknown")
        json.dumps(asdict(cr))

    def test_json_serializable_not_needed(self):
        cr = _build(task_type="config")
        json.dumps(asdict(cr))

    def test_options_list_preserved(self):
        cr = _build(task_type="unknown")
        d = asdict(cr)
        restored = d["options"]
        self.assertEqual(restored, cr.options)


class TestNativeClarificationRequestReconstruction(unittest.TestCase):

    def _entry(self, clarification_request=None) -> dict:
        return {
            "workflow": "native",
            "executor": "native",
            "clarification_request": clarification_request,
        }

    def test_reconstruct_needed_from_entry(self):
        stored = asdict(_build(task_type="unknown", clarification_needed=["ambiguous"]))
        ns = _native_meta_from_entry(self._entry(clarification_request=stored))
        self.assertTrue(ns.clarification_request.needed)
        self.assertIsNotNone(ns.clarification_request.question)
        self.assertEqual(ns.clarification_request.reason, "ambiguous")

    def test_reconstruct_options_survive(self):
        stored = asdict(_build(task_type="unknown"))
        ns = _native_meta_from_entry(self._entry(clarification_request=stored))
        self.assertIsInstance(ns.clarification_request.options, list)
        self.assertGreater(len(ns.clarification_request.options), 0)

    def test_reconstruct_when_missing_is_none(self):
        ns = _native_meta_from_entry(self._entry(clarification_request=None))
        self.assertIsNone(ns.clarification_request)

    def test_reconstruct_when_key_absent_is_none(self):
        entry = {"workflow": "native", "executor": "native"}
        ns = _native_meta_from_entry(entry)
        self.assertIsNone(ns.clarification_request)

    def test_not_needed_roundtrip(self):
        stored = asdict(_build(task_type="test"))
        ns = _native_meta_from_entry(self._entry(clarification_request=stored))
        self.assertFalse(ns.clarification_request.needed)


class TestNativeClarificationRequestRendering(unittest.TestCase):

    def test_no_render_when_not_needed(self):
        cr = NativeClarificationRequest(needed=False)
        output = _render(clarification_request=cr)
        self.assertNotIn("clarification", output)

    def test_no_render_when_none(self):
        output = _render(clarification_request=None)
        self.assertNotIn("clarification", output)

    def test_renders_needed_label(self):
        cr = _build(task_type="unknown")
        output = _render(clarification_request=cr)
        self.assertIn("clarification: needed", output)

    def test_renders_question_text(self):
        cr = NativeClarificationRequest(
            needed=True,
            question="Which validation rule?",
            options=["opt a", "opt b"],
            allows_custom=False,
        )
        output = _render(clarification_request=cr)
        self.assertIn("Which validation rule?", output)

    def test_renders_option_count_not_raw_list(self):
        cr = NativeClarificationRequest(
            needed=True,
            question="Which rule?",
            options=["a", "b", "c"],
            allows_custom=False,
        )
        output = _render(clarification_request=cr)
        self.assertIn("3 options", output)
        self.assertNotIn('"a"', output)
        self.assertNotIn('"b"', output)

    def test_renders_custom_flag_when_true(self):
        cr = NativeClarificationRequest(
            needed=True,
            question="Which rule?",
            options=["a"],
            allows_custom=True,
        )
        output = _render(clarification_request=cr)
        self.assertIn("custom answer allowed", output)

    def test_no_custom_flag_when_false(self):
        cr = NativeClarificationRequest(
            needed=True,
            question="Which rule?",
            options=["a"],
            allows_custom=False,
        )
        output = _render(clarification_request=cr)
        self.assertNotIn("custom answer", output)

    def test_max_two_clarification_lines(self):
        cr = _build(task_type="unknown")
        meta = _meta(clarification_request=cr)
        all_lines = _render_native_demo_block(meta)
        cr_lines = [ln for ln in all_lines if "clarification" in ln]
        self.assertLessEqual(len(cr_lines), 2)

    def test_singular_option_label(self):
        cr = NativeClarificationRequest(
            needed=True,
            question="Pick one?",
            options=["only this"],
            allows_custom=False,
        )
        output = _render(clarification_request=cr)
        self.assertIn("1 option", output)
        self.assertNotIn("1 options", output)
