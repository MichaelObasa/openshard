from __future__ import annotations

import json
import unittest
from dataclasses import asdict
from types import SimpleNamespace

from openshard.native.context import (
    NativeChangeBudget,
    NativePlan,
    NativeVerificationPlan,
    build_native_verification_plan,
)
from openshard.cli.run_output import _native_meta_from_entry, _render_native_demo_block


def _build(**kwargs) -> NativeVerificationPlan:
    defaults = dict(task="", plan=None, change_budget=None, read_search_findings=[], repo_facts=None)
    defaults.update(kwargs)
    return build_native_verification_plan(**defaults)


def _meta(**kwargs):
    defaults = dict(
        repo_context_summary=None,
        observation=None,
        plan=None,
        verification_plan=None,
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
        clarification_request=None,
    )
    defaults.update(kwargs)
    return SimpleNamespace(**defaults)


class TestBuildNativeVerificationPlan(unittest.TestCase):

    def test_defaults_with_no_inputs(self):
        vp = _build()
        self.assertEqual(vp.task_type, "unknown")
        self.assertEqual(vp.risk_level, "unknown")
        self.assertEqual(vp.likely_files_or_folders, [])
        self.assertGreater(len(vp.clarification_needed), 0)
        self.assertEqual(vp.failure_handling, "halt and report")

    def test_task_type_feature(self):
        self.assertEqual(_build(task="add a login button").task_type, "feature")

    def test_task_type_bugfix(self):
        self.assertEqual(_build(task="fix the crash in auth").task_type, "bugfix")

    def test_task_type_refactor(self):
        self.assertEqual(_build(task="refactor the utils module").task_type, "refactor")

    def test_task_type_test(self):
        self.assertEqual(_build(task="write tests for auth flow").task_type, "test")

    def test_task_type_docs(self):
        self.assertEqual(_build(task="update the readme").task_type, "docs")

    def test_task_type_config(self):
        self.assertEqual(_build(task="update env config settings").task_type, "config")

    def test_task_type_unknown(self):
        self.assertEqual(_build(task="do the thing").task_type, "unknown")

    def test_risk_level_from_plan(self):
        vp = _build(task="", plan=NativePlan(risk="high"))
        self.assertEqual(vp.risk_level, "high")

    def test_risk_level_unknown_when_no_plan(self):
        vp = _build(task="", plan=None)
        self.assertEqual(vp.risk_level, "unknown")

    def test_likely_files_strips_file_prefix(self):
        vp = _build(read_search_findings=["file:src/a.py", "file:src/b.py"])
        self.assertIn("src/a.py", vp.likely_files_or_folders)
        self.assertIn("src/b.py", vp.likely_files_or_folders)

    def test_likely_files_strips_test_marker_prefix(self):
        vp = _build(read_search_findings=["test-marker:tests/test_auth.py"])
        self.assertIn("tests/test_auth.py", vp.likely_files_or_folders)

    def test_likely_files_skips_package_markers(self):
        vp = _build(read_search_findings=["package:requirements.txt", "file:src/a.py"])
        self.assertNotIn("requirements.txt", vp.likely_files_or_folders)
        self.assertIn("src/a.py", vp.likely_files_or_folders)

    def test_likely_files_max_five(self):
        findings = [f"file:src/f{i}.py" for i in range(10)]
        vp = _build(read_search_findings=findings)
        self.assertEqual(len(vp.likely_files_or_folders), 5)

    def test_likely_files_deduped(self):
        vp = _build(read_search_findings=["file:src/a.py", "file:src/a.py", "file:src/b.py"])
        self.assertEqual(vp.likely_files_or_folders.count("src/a.py"), 1)

    def test_suggested_verification_from_repo_facts(self):
        rf = SimpleNamespace(test_command="pytest -x")
        vp = _build(repo_facts=rf)
        self.assertEqual(vp.suggested_verification_commands, ["pytest -x"])

    def test_suggested_verification_empty_without_repo_facts(self):
        vp = _build(repo_facts=None)
        self.assertEqual(vp.suggested_verification_commands, [])

    def test_suggested_verification_empty_when_no_test_command(self):
        rf = SimpleNamespace(test_command=None)
        vp = _build(repo_facts=rf)
        self.assertEqual(vp.suggested_verification_commands, [])

    def test_blocked_commands_present(self):
        vp = _build()
        self.assertIn("rm", vp.blocked_commands)
        self.assertIn("sudo", vp.blocked_commands)

    def test_allowed_commands_present(self):
        vp = _build()
        self.assertIn("pytest", vp.allowed_commands)
        self.assertIn("npm test", vp.allowed_commands)

    def test_approval_rules_from_budget(self):
        budget = NativeChangeBudget(max_files=3, max_change_size="normal", level="good", guidance="")
        vp = _build(change_budget=budget)
        self.assertTrue(any("3" in r for r in vp.approval_rules))

    def test_approval_rules_include_blocked_commands(self):
        vp = _build()
        self.assertTrue(any("blocked" in r for r in vp.approval_rules))

    def test_clarification_needed_empty_when_task_type_known(self):
        vp = _build(task="fix the bug")
        self.assertEqual(vp.clarification_needed, [])

    def test_clarification_needed_populated_when_task_type_unknown(self):
        vp = _build(task="do the thing")
        self.assertGreater(len(vp.clarification_needed), 0)

    def test_failure_handling_default(self):
        vp = _build()
        self.assertEqual(vp.failure_handling, "halt and report")

    def test_success_criteria_non_empty(self):
        vp = _build(task="fix the bug")
        self.assertGreater(len(vp.success_criteria), 0)


class TestNativeVerificationPlanSerialization(unittest.TestCase):

    def test_asdict_roundtrip(self):
        vp = _build(task="add login", plan=NativePlan(risk="low"), read_search_findings=["file:src/a.py"])
        d = asdict(vp)
        self.assertIn("task_type", d)
        self.assertIn("risk_level", d)
        self.assertIn("likely_files_or_folders", d)
        self.assertIn("allowed_commands", d)
        self.assertIn("blocked_commands", d)
        self.assertIn("suggested_verification_commands", d)
        self.assertIn("approval_rules", d)
        self.assertIn("success_criteria", d)
        self.assertIn("failure_handling", d)
        self.assertIn("clarification_needed", d)
        self.assertIn("warnings", d)

    def test_all_fields_are_json_serializable(self):
        vp = _build(task="implement feature", plan=NativePlan(risk="medium"))
        self.assertIsNotNone(json.dumps(asdict(vp)))

    def test_empty_lists_serialize_as_list(self):
        vp = _build(task="implement feature")
        d = asdict(vp)
        self.assertEqual(d["likely_files_or_folders"], [])
        self.assertEqual(d["clarification_needed"], [])


class TestNativeVerificationPlanReconstruction(unittest.TestCase):

    def _round_trip(self, **kwargs) -> SimpleNamespace:
        vp = _build(**kwargs)
        entry = {"workflow": "native", "verification_plan": asdict(vp)}
        ns = _native_meta_from_entry(entry)
        return ns

    def test_reconstruct_from_entry_dict(self):
        ns = self._round_trip(task="add feature", plan=NativePlan(risk="low"))
        self.assertIsNotNone(ns.verification_plan)
        self.assertEqual(ns.verification_plan.task_type, "feature")
        self.assertEqual(ns.verification_plan.risk_level, "low")

    def test_reconstruct_none_when_missing(self):
        entry = {"workflow": "native"}
        ns = _native_meta_from_entry(entry)
        self.assertIsNone(ns.verification_plan)

    def test_reconstruct_nested_lists_survive_roundtrip(self):
        findings = ["file:src/a.py", "file:src/b.py"]
        ns = self._round_trip(read_search_findings=findings)
        self.assertIn("src/a.py", ns.verification_plan.likely_files_or_folders)

    def test_reconstruct_not_native_returns_none(self):
        entry = {"workflow": "standard", "verification_plan": {"task_type": "feature"}}
        ns = _native_meta_from_entry(entry)
        self.assertIsNone(ns)


class TestNativeVerificationPlanRendering(unittest.TestCase):

    def _render(self, vplan=None, change_budget=None):
        meta = _meta(verification_plan=vplan, change_budget=change_budget)
        return "\n".join(_render_native_demo_block(meta))

    def test_render_shows_task_type_and_risk(self):
        vp = SimpleNamespace(
            task_type="feature", risk_level="low",
            likely_files_or_folders=[], suggested_verification_commands=[],
            blocked_commands=["rm"],
        )
        out = self._render(vplan=vp)
        self.assertIn("verification plan: feature, risk=low", out)

    def test_render_shows_verify_command(self):
        vp = SimpleNamespace(
            task_type="feature", risk_level="low",
            likely_files_or_folders=[], suggested_verification_commands=["pytest -x"],
            blocked_commands=[],
        )
        out = self._render(vplan=vp)
        self.assertIn("vplan verify: pytest -x", out)

    def test_render_shows_scope_when_present(self):
        vp = SimpleNamespace(
            task_type="bugfix", risk_level="medium",
            likely_files_or_folders=["src/a.py", "src/b.py"],
            suggested_verification_commands=[],
            blocked_commands=[],
        )
        out = self._render(vplan=vp)
        self.assertIn("vplan scope:", out)
        self.assertIn("src/a.py", out)

    def test_render_omits_scope_when_empty(self):
        vp = SimpleNamespace(
            task_type="unknown", risk_level="unknown",
            likely_files_or_folders=[], suggested_verification_commands=[],
            blocked_commands=[],
        )
        out = self._render(vplan=vp)
        self.assertNotIn("vplan scope:", out)

    def test_render_omits_when_no_verification_plan(self):
        out = self._render(vplan=None)
        self.assertNotIn("verification plan:", out)

    def test_render_shows_policy_when_blocked_commands(self):
        vp = SimpleNamespace(
            task_type="feature", risk_level="low",
            likely_files_or_folders=[], suggested_verification_commands=[],
            blocked_commands=["rm", "curl"],
        )
        out = self._render(vplan=vp)
        self.assertIn("vplan policy: blocked destructive/network commands", out)

    def test_render_no_raw_blocked_list(self):
        vp = SimpleNamespace(
            task_type="feature", risk_level="low",
            likely_files_or_folders=[], suggested_verification_commands=[],
            blocked_commands=["rm", "sudo", "curl"],
        )
        out = self._render(vplan=vp)
        self.assertNotIn("rm", out)
        self.assertNotIn("sudo", out)

    def test_render_max_four_vplan_lines(self):
        vp = SimpleNamespace(
            task_type="feature", risk_level="low",
            likely_files_or_folders=["src/a.py", "src/b.py"],
            suggested_verification_commands=["pytest"],
            blocked_commands=["rm"],
        )
        out = self._render(vplan=vp)
        vplan_lines = [ln for ln in out.splitlines() if "vplan" in ln or "verification plan:" in ln]
        self.assertLessEqual(len(vplan_lines), 4)


if __name__ == "__main__":
    unittest.main()
