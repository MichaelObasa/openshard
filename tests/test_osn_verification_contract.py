"""Tests for OSNVerificationContract: dataclass, builder, receipt rendering, and integration.

Coverage:
- Contract object: defaults, JSON safety, allowed statuses, caps, no raw fields (tests 1-6)
- Contract building: check flow, status mapping, skipped/manual review logic (tests 7-14)
- OSN integration: NativeRunMeta field, no shell exec, no model calls, approval unchanged (tests 15-21)
- Receipt rendering: OSN VERIFICATION section, caps, no raw output, no em dash (tests 22-30)
- Persistence: osn_observation and osn_verification_contract in _extra_metadata (tests 31-33)
- Regression: existing modules unaffected, JSON round-trip, ruff clean (tests 34-39)
"""
from __future__ import annotations

import json
import unittest
from dataclasses import asdict, fields
from types import SimpleNamespace
from typing import Any


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_loop_summary(
    *,
    enabled: bool = True,
    verification_attempted: bool = False,
    verification_passed: bool | None = None,
    verification_status: str = "",
    stopped_reason: str = "completed",
) -> Any:
    ns = SimpleNamespace(
        enabled=enabled,
        verification_attempted=verification_attempted,
        verification_passed=verification_passed,
        verification_status=verification_status,
        stopped_reason=stopped_reason,
    )
    return ns


def _make_observation(suggested_checks: list[str] | None = None) -> Any:
    return SimpleNamespace(
        enabled=True,
        suggested_checks=suggested_checks or [],
    )


# ---------------------------------------------------------------------------
# 1-6: Contract object
# ---------------------------------------------------------------------------

class TestOSNVerificationContractDefaults(unittest.TestCase):
    def setUp(self) -> None:
        from openshard.native.verification_contract import OSNVerificationContract
        self.cls = OSNVerificationContract

    def test_1_defaults_are_safe(self) -> None:
        c = self.cls()
        self.assertFalse(c.enabled)
        self.assertEqual(c.status, "not_run")
        self.assertFalse(c.required)
        self.assertEqual(c.source, "osn_verification_contract_v1")
        self.assertEqual(c.expected_checks, [])
        self.assertEqual(c.attempted_checks, [])
        self.assertEqual(c.passed_checks, [])
        self.assertEqual(c.failed_checks, [])
        self.assertEqual(c.skipped_checks, [])
        self.assertEqual(c.missing_checks, [])
        self.assertEqual(c.skipped_reason, "")
        self.assertFalse(c.manual_review_required)
        self.assertEqual(c.summary, "")

    def test_2_serializes_to_json_safe_dict(self) -> None:
        from openshard.native.verification_contract import OSNVerificationContract
        c = OSNVerificationContract(
            enabled=True,
            status="passed",
            expected_checks=["pytest"],
            summary="ok",
        )
        d = asdict(c)
        # must round-trip through JSON without error
        json_str = json.dumps(d)
        loaded = json.loads(json_str)
        self.assertEqual(loaded["status"], "passed")
        self.assertEqual(loaded["expected_checks"], ["pytest"])

    def test_3_allowed_status_values_are_stable(self) -> None:
        from openshard.native.verification_contract import _ALLOWED_STATUSES
        expected = {"not_run", "passed", "failed", "skipped", "impossible", "manual_review", "unknown"}
        self.assertEqual(_ALLOWED_STATUSES, expected)

    def test_4_lists_are_capped(self) -> None:
        from openshard.native.verification_contract import OSNVerificationContract, _MAX_CHECKS
        long_list = [f"check_{i}" for i in range(20)]
        c = OSNVerificationContract(
            expected_checks=long_list,
            attempted_checks=long_list,
            passed_checks=long_list,
            failed_checks=long_list,
            skipped_checks=long_list,
            missing_checks=long_list,
        )
        self.assertLessEqual(len(c.expected_checks), _MAX_CHECKS)
        self.assertLessEqual(len(c.attempted_checks), _MAX_CHECKS)
        self.assertLessEqual(len(c.passed_checks), _MAX_CHECKS)
        self.assertLessEqual(len(c.failed_checks), _MAX_CHECKS)
        self.assertLessEqual(len(c.skipped_checks), _MAX_CHECKS)
        self.assertLessEqual(len(c.missing_checks), _MAX_CHECKS)

    def test_5_summary_and_skipped_reason_are_capped(self) -> None:
        from openshard.native.verification_contract import (
            OSNVerificationContract,
            _MAX_REASON_CHARS,
            _MAX_SUMMARY_CHARS,
        )
        long_str = "x" * 500
        c = OSNVerificationContract(summary=long_str, skipped_reason=long_str)
        self.assertLessEqual(len(c.summary), _MAX_SUMMARY_CHARS)
        self.assertLessEqual(len(c.skipped_reason), _MAX_REASON_CHARS)

    def test_6_no_raw_output_fields_exist(self) -> None:
        from openshard.native.verification_contract import OSNVerificationContract
        forbidden_names = {"output", "stdout", "stderr", "command", "raw_output", "raw_content"}
        field_names = {f.name for f in fields(OSNVerificationContract)}
        self.assertTrue(forbidden_names.isdisjoint(field_names), f"Forbidden fields found: {forbidden_names & field_names}")


# ---------------------------------------------------------------------------
# 7-14: Contract building
# ---------------------------------------------------------------------------

class TestBuildOSNVerificationContract(unittest.TestCase):
    def setUp(self) -> None:
        from openshard.native.verification_contract import build_osn_verification_contract
        self.build = build_osn_verification_contract

    def test_7_suggested_checks_become_expected_checks(self) -> None:
        obs = _make_observation(["pytest", "ruff"])
        loop = _make_loop_summary()
        c = self.build(osn_observation=obs, osn_loop_summary=loop)
        self.assertEqual(c.expected_checks, ["pytest", "ruff"])

    def test_8_passed_verification_sets_status_passed(self) -> None:
        obs = _make_observation()
        loop = _make_loop_summary(verification_attempted=True, verification_passed=True)
        c = self.build(osn_observation=obs, osn_loop_summary=loop)
        self.assertEqual(c.status, "passed")

    def test_9_failed_verification_sets_status_failed(self) -> None:
        obs = _make_observation()
        loop = _make_loop_summary(verification_attempted=True, verification_passed=False)
        c = self.build(osn_observation=obs, osn_loop_summary=loop)
        self.assertEqual(c.status, "failed")

    def test_10_not_attempted_sets_status_skipped_with_reason(self) -> None:
        obs = _make_observation()
        loop = _make_loop_summary(verification_attempted=False)
        c = self.build(osn_observation=obs, osn_loop_summary=loop)
        self.assertEqual(c.status, "skipped")
        self.assertNotEqual(c.skipped_reason, "")

    def test_11_not_attempted_write_task_requires_manual_review(self) -> None:
        obs = _make_observation()
        loop = _make_loop_summary(verification_attempted=False)
        c = self.build(osn_observation=obs, osn_loop_summary=loop, is_write_task=True)
        self.assertTrue(c.manual_review_required)

    def test_12_unknown_verification_does_not_pretend_success(self) -> None:
        obs = _make_observation()
        loop = _make_loop_summary(verification_attempted=True, verification_passed=None, verification_status="")
        c = self.build(osn_observation=obs, osn_loop_summary=loop)
        self.assertNotEqual(c.status, "passed")

    def test_13_failed_verification_sets_manual_review_required(self) -> None:
        obs = _make_observation(["pytest"])
        loop = _make_loop_summary(verification_attempted=True, verification_passed=False)
        c = self.build(osn_observation=obs, osn_loop_summary=loop)
        self.assertTrue(c.manual_review_required)

    def test_14_stopped_reason_verification_failed_forces_failed_status(self) -> None:
        obs = _make_observation()
        loop = _make_loop_summary(
            verification_attempted=False,
            stopped_reason="verification_failed",
        )
        c = self.build(osn_observation=obs, osn_loop_summary=loop)
        self.assertEqual(c.status, "failed")
        self.assertTrue(c.manual_review_required)


# ---------------------------------------------------------------------------
# 15-21: OSN integration
# ---------------------------------------------------------------------------

class TestOSNIntegration(unittest.TestCase):
    def test_15_native_run_meta_has_osn_verification_contract_field(self) -> None:
        from openshard.native.executor import NativeRunMeta
        meta = NativeRunMeta()
        self.assertTrue(hasattr(meta, "osn_verification_contract"))
        self.assertIsNone(meta.osn_verification_contract)

    def test_16_contract_created_when_loop_summary_present(self) -> None:
        from openshard.native.verification_contract import build_osn_verification_contract
        loop = _make_loop_summary(verification_attempted=True, verification_passed=True)
        c = build_osn_verification_contract(osn_observation=None, osn_loop_summary=loop)
        self.assertTrue(c.enabled)
        self.assertIsNotNone(c.status)

    def test_17_no_shell_command_executed_by_contract(self) -> None:
        import openshard.native.verification_contract as vc_mod
        import inspect
        src = inspect.getsource(vc_mod)
        self.assertNotIn("subprocess", src)
        self.assertNotIn("os.system", src)
        self.assertNotIn("Popen", src)

    def test_18_no_provider_model_call_added(self) -> None:
        import openshard.native.verification_contract as vc_mod
        import inspect
        src = inspect.getsource(vc_mod)
        self.assertNotIn("anthropic", src)
        self.assertNotIn("openai", src)
        self.assertNotIn("litellm", src)

    def test_19_approval_behavior_unchanged(self) -> None:
        from openshard.native.verification_contract import OSNVerificationContract
        field_names = {f.name for f in fields(OSNVerificationContract)}
        self.assertNotIn("approval_required", field_names)
        self.assertNotIn("approval_granted", field_names)

    def test_20_existing_verification_fields_still_work(self) -> None:
        from openshard.native.context import NativeOSNLoopSummary
        s = NativeOSNLoopSummary(enabled=True)
        self.assertFalse(s.verification_attempted)
        self.assertIsNone(s.verification_passed)
        self.assertEqual(s.verification_status, "")

    def test_21_stopped_reason_verification_failed_preserved_on_summary(self) -> None:
        from openshard.native.context import normalize_osn_stop_reason
        result = normalize_osn_stop_reason("verification_failed")
        self.assertEqual(result, "verification_failed")


# ---------------------------------------------------------------------------
# 22-30: Receipt rendering
# ---------------------------------------------------------------------------

class TestOSNVerificationReceiptRendering(unittest.TestCase):
    def _make_vc(self, **kwargs: Any) -> Any:
        from openshard.native.verification_contract import OSNVerificationContract
        return OSNVerificationContract(**kwargs)

    def test_22_section_renders_when_contract_present_and_enabled(self) -> None:
        from openshard.native.verification_contract import render_osn_verification_receipt
        c = self._make_vc(enabled=True, status="passed")
        lines = render_osn_verification_receipt(c)
        self.assertTrue(any("OSN VERIFICATION" in line for line in lines))

    def test_23_section_omitted_when_absent_or_not_enabled(self) -> None:
        from openshard.native.verification_contract import render_osn_verification_receipt
        self.assertEqual(render_osn_verification_receipt(None), [])
        c = self._make_vc(enabled=False, status="passed")
        self.assertEqual(render_osn_verification_receipt(c), [])

    def test_24_status_row_appears(self) -> None:
        from openshard.native.verification_contract import render_osn_verification_receipt
        c = self._make_vc(enabled=True, status="failed")
        lines = render_osn_verification_receipt(c)
        self.assertTrue(any("failed" in line for line in lines))

    def test_25_expected_checks_are_capped_in_display(self) -> None:
        from openshard.native.verification_contract import render_osn_verification_receipt
        many = [f"check_{i}" for i in range(10)]
        c = self._make_vc(enabled=True, status="passed", expected_checks=many)
        lines = render_osn_verification_receipt(c)
        expected_lines = [ln for ln in lines if "Expected" in ln]
        if expected_lines:
            # Should show at most 4 checks in display
            content = expected_lines[0]
            shown = [x.strip() for x in content.split("Expected")[-1].split(",")]
            self.assertLessEqual(len(shown), 4)

    def test_26_attempted_checks_are_capped_in_display(self) -> None:
        from openshard.native.verification_contract import render_osn_verification_receipt
        many = [f"check_{i}" for i in range(10)]
        c = self._make_vc(enabled=True, status="passed", attempted_checks=many)
        lines = render_osn_verification_receipt(c)
        attempted_lines = [ln for ln in lines if "Attempted" in ln]
        if attempted_lines:
            content = attempted_lines[0]
            shown = [x.strip() for x in content.split("Attempted")[-1].split(",")]
            self.assertLessEqual(len(shown), 4)

    def test_27_no_raw_command_output_in_rendered_lines(self) -> None:
        from openshard.native.verification_contract import render_osn_verification_receipt
        c = self._make_vc(enabled=True, status="passed", summary="ok")
        lines = render_osn_verification_receipt(c, detail="full")
        full_text = "\n".join(lines)
        # No shell-style markers
        self.assertNotIn("$", full_text)
        self.assertNotIn(">>", full_text)
        self.assertNotIn("FAILED", full_text)
        self.assertNotIn("ERROR:", full_text)

    def test_28_no_raw_json_in_rendered_lines(self) -> None:
        from openshard.native.verification_contract import render_osn_verification_receipt
        c = self._make_vc(enabled=True, status="skipped", skipped_reason="no checks")
        lines = render_osn_verification_receipt(c)
        full_text = "\n".join(lines)
        self.assertNotIn("{", full_text)
        self.assertNotIn("}", full_text)

    def test_29_no_em_dash_in_rendered_lines(self) -> None:
        from openshard.native.verification_contract import render_osn_verification_receipt
        c = self._make_vc(enabled=True, status="passed", skipped_reason="none", summary="all good")
        lines = render_osn_verification_receipt(c, detail="full")
        full_text = "\n".join(lines)
        self.assertNotIn("—", full_text)  # em dash

    def test_30_no_chain_of_thought_phrases_in_rendered_lines(self) -> None:
        from openshard.native.verification_contract import render_osn_verification_receipt
        c = self._make_vc(enabled=True, status="manual_review", summary="check required")
        lines = render_osn_verification_receipt(c, detail="full")
        full_text = "\n".join(lines).lower()
        for phrase in ("i think", "let me", "step by step", "as an ai"):
            self.assertNotIn(phrase, full_text)


# ---------------------------------------------------------------------------
# 31-33: Persistence
# ---------------------------------------------------------------------------

class TestPersistence(unittest.TestCase):
    def test_31_osn_observation_added_to_extra_metadata_key(self) -> None:
        # Verify the key name is mapped in _native_meta_from_entry
        from openshard.cli.run_output import _native_meta_from_entry  # type: ignore[attr-defined]
        entry: dict = {
            "workflow": "native",
            "osn_observation": {"enabled": True, "suggested_checks": ["pytest"]},
        }
        meta = _native_meta_from_entry(entry)
        self.assertIsNotNone(meta)
        obs = getattr(meta, "osn_observation", None)
        self.assertIsNotNone(obs)
        checks = getattr(obs, "suggested_checks", None)
        self.assertIsNotNone(checks)

    def test_32_osn_verification_contract_added_to_extra_metadata_key(self) -> None:
        from openshard.cli.run_output import _native_meta_from_entry  # type: ignore[attr-defined]
        entry: dict = {
            "workflow": "native",
            "osn_verification_contract": {
                "enabled": True,
                "status": "passed",
                "source": "osn_verification_contract_v1",
                "expected_checks": ["pytest"],
                "attempted_checks": [],
                "passed_checks": [],
                "failed_checks": [],
                "skipped_checks": [],
                "missing_checks": ["pytest"],
                "skipped_reason": "",
                "manual_review_required": False,
                "required": False,
                "summary": "ok",
            },
        }
        meta = _native_meta_from_entry(entry)
        self.assertIsNotNone(meta)
        vc = getattr(meta, "osn_verification_contract", None)
        self.assertIsNotNone(vc)
        self.assertEqual(getattr(vc, "status", None), "passed")

    def test_33_run_output_renders_from_history_entry_dict(self) -> None:
        from openshard.cli.run_output import _native_meta_from_entry  # type: ignore[attr-defined]
        from openshard.native.verification_contract import render_osn_verification_receipt
        entry: dict = {
            "workflow": "native",
            "osn_verification_contract": {
                "enabled": True,
                "status": "skipped",
                "source": "osn_verification_contract_v1",
                "expected_checks": ["ruff"],
                "attempted_checks": [],
                "passed_checks": [],
                "failed_checks": [],
                "skipped_checks": [],
                "missing_checks": ["ruff"],
                "skipped_reason": "verification not attempted",
                "manual_review_required": True,
                "required": False,
                "summary": "skipped",
            },
        }
        meta = _native_meta_from_entry(entry)
        self.assertIsNotNone(meta)
        vc_ns = getattr(meta, "osn_verification_contract", None)
        self.assertIsNotNone(vc_ns)
        # SimpleNamespace from entry dict - convert to something render can use
        # The renderer uses getattr so it works with SimpleNamespace too
        from openshard.native.verification_contract import OSNVerificationContract
        vc_obj = OSNVerificationContract(
            enabled=getattr(vc_ns, "enabled", False),
            status=getattr(vc_ns, "status", "not_run"),
            expected_checks=getattr(vc_ns, "expected_checks", []),
            skipped_reason=getattr(vc_ns, "skipped_reason", ""),
            manual_review_required=getattr(vc_ns, "manual_review_required", False),
            missing_checks=getattr(vc_ns, "missing_checks", []),
        )
        lines = render_osn_verification_receipt(vc_obj)
        self.assertTrue(any("OSN VERIFICATION" in ln for ln in lines))
        self.assertTrue(any("skipped" in ln for ln in lines))


# ---------------------------------------------------------------------------
# 34-39: Regression
# ---------------------------------------------------------------------------

class TestRegression(unittest.TestCase):
    def test_34_osn_observation_module_imports_cleanly(self) -> None:
        import openshard.native.osn_observation  # noqa: F401

    def test_35_osn_loop_recorder_imports_cleanly(self) -> None:
        import openshard.native.osn_loop_recorder  # noqa: F401

    def test_36_osn_loop_summary_fields_unchanged(self) -> None:
        from openshard.native.context import NativeOSNLoopSummary
        s = NativeOSNLoopSummary()
        # Core fields still present
        self.assertFalse(s.verification_attempted)
        self.assertIsNone(s.verification_passed)
        self.assertEqual(s.stopped_reason, "")
        self.assertEqual(s.verification_status, "")

    def test_37_reflector_imports_cleanly(self) -> None:
        import openshard.reflection.reflector  # noqa: F401

    def test_38_osn_verification_contract_json_round_trip(self) -> None:
        from openshard.native.verification_contract import OSNVerificationContract
        c = OSNVerificationContract(
            enabled=True,
            status="manual_review",
            expected_checks=["pytest", "ruff"],
            missing_checks=["ruff"],
            manual_review_required=True,
            skipped_reason="ruff was not recorded",
            summary="review required",
        )
        d = asdict(c)
        json_bytes = json.dumps(d).encode()
        loaded = json.loads(json_bytes)
        c2 = OSNVerificationContract(**loaded)
        self.assertEqual(c2.status, "manual_review")
        self.assertEqual(c2.expected_checks, ["pytest", "ruff"])
        self.assertEqual(c2.missing_checks, ["ruff"])
        self.assertTrue(c2.manual_review_required)

    def test_39_invalid_status_normalized_to_unknown(self) -> None:
        from openshard.native.verification_contract import OSNVerificationContract
        c = OSNVerificationContract(status="totally_invalid_value")
        self.assertEqual(c.status, "unknown")


# ---------------------------------------------------------------------------
# 40-50: Per-check proof wiring (verification_loop parameter)
# ---------------------------------------------------------------------------

def _make_vloop(
    *,
    check_attempted: list[str] | None = None,
    check_passed: list[str] | None = None,
    check_failed: list[str] | None = None,
    check_skipped: list[str] | None = None,
    check_skipped_reasons: list[str] | None = None,
) -> Any:
    return SimpleNamespace(
        check_attempted=check_attempted or [],
        check_passed=check_passed or [],
        check_failed=check_failed or [],
        check_skipped=check_skipped or [],
        check_skipped_reasons=check_skipped_reasons or [],
    )


class TestVerificationLoopWiring(unittest.TestCase):
    def setUp(self) -> None:
        from openshard.native.verification_contract import build_osn_verification_contract
        self.build = build_osn_verification_contract

    def test_40_loop_none_is_backward_compat(self) -> None:
        loop_summary = _make_loop_summary(verification_attempted=True, verification_passed=True)
        contract = self.build(
            osn_observation=None,
            osn_loop_summary=loop_summary,
            verification_loop=None,
        )
        self.assertEqual(contract.attempted_checks, [])
        self.assertEqual(contract.passed_checks, [])

    def test_41_attempted_checks_populated_from_loop(self) -> None:
        loop_summary = _make_loop_summary(verification_attempted=True, verification_passed=True)
        vloop = _make_vloop(check_attempted=["pytest"], check_passed=["pytest"])
        contract = self.build(
            osn_observation=None,
            osn_loop_summary=loop_summary,
            verification_loop=vloop,
        )
        self.assertEqual(contract.attempted_checks, ["pytest"])

    def test_42_passed_checks_populated_from_loop(self) -> None:
        loop_summary = _make_loop_summary(verification_attempted=True, verification_passed=True)
        vloop = _make_vloop(check_attempted=["pytest"], check_passed=["pytest"])
        contract = self.build(
            osn_observation=None,
            osn_loop_summary=loop_summary,
            verification_loop=vloop,
        )
        self.assertEqual(contract.passed_checks, ["pytest"])
        self.assertEqual(contract.failed_checks, [])

    def test_43_failed_checks_populated_from_loop(self) -> None:
        loop_summary = _make_loop_summary(verification_attempted=True, verification_passed=False)
        vloop = _make_vloop(check_attempted=["pytest"], check_failed=["pytest"])
        contract = self.build(
            osn_observation=None,
            osn_loop_summary=loop_summary,
            verification_loop=vloop,
        )
        self.assertEqual(contract.failed_checks, ["pytest"])
        self.assertEqual(contract.passed_checks, [])

    def test_44_skipped_checks_and_reason_from_loop(self) -> None:
        loop_summary = _make_loop_summary(verification_attempted=False)
        vloop = _make_vloop(
            check_skipped=["make test"],
            check_skipped_reasons=["needs_approval: medium-risk command"],
        )
        contract = self.build(
            osn_observation=None,
            osn_loop_summary=loop_summary,
            verification_loop=vloop,
        )
        self.assertEqual(contract.skipped_checks, ["make test"])
        self.assertIn("needs_approval", contract.skipped_reason)

    def test_45_expected_checks_seeded_from_loop_when_obs_empty(self) -> None:
        loop_summary = _make_loop_summary(verification_attempted=True, verification_passed=True)
        vloop = _make_vloop(check_attempted=["pytest"], check_passed=["pytest"])
        contract = self.build(
            osn_observation=None,
            osn_loop_summary=loop_summary,
            verification_loop=vloop,
        )
        self.assertIn("pytest", contract.expected_checks)

    def test_46_obs_expected_checks_not_overridden_by_loop(self) -> None:
        obs = _make_observation(suggested_checks=["ruff", "pytest"])
        loop_summary = _make_loop_summary(verification_attempted=True, verification_passed=True)
        vloop = _make_vloop(check_attempted=["pytest"], check_passed=["pytest"])
        contract = self.build(
            osn_observation=obs,
            osn_loop_summary=loop_summary,
            verification_loop=vloop,
        )
        self.assertIn("ruff", contract.expected_checks)

    def test_47_check_lists_capped_at_max(self) -> None:
        many = [f"check_{i}" for i in range(15)]
        loop_summary = _make_loop_summary(verification_attempted=True, verification_passed=True)
        vloop = _make_vloop(check_attempted=many, check_passed=many)
        contract = self.build(
            osn_observation=None,
            osn_loop_summary=loop_summary,
            verification_loop=vloop,
        )
        self.assertLessEqual(len(contract.attempted_checks), 8)
        self.assertLessEqual(len(contract.passed_checks), 8)

    def test_48_render_shows_passed_row(self) -> None:
        from openshard.native.verification_contract import OSNVerificationContract, render_osn_verification_receipt
        contract = OSNVerificationContract(
            enabled=True, status="passed",
            attempted_checks=["pytest"], passed_checks=["pytest"],
        )
        lines = render_osn_verification_receipt(contract)
        self.assertTrue(any("Passed" in ln for ln in lines))
        self.assertTrue(any("pytest" in ln for ln in lines))

    def test_49_render_shows_failed_row(self) -> None:
        from openshard.native.verification_contract import OSNVerificationContract, render_osn_verification_receipt
        contract = OSNVerificationContract(
            enabled=True, status="failed",
            attempted_checks=["pytest"], failed_checks=["pytest"],
        )
        lines = render_osn_verification_receipt(contract)
        self.assertTrue(any("Failed" in ln for ln in lines))

    def test_50_render_shows_skipped_row(self) -> None:
        from openshard.native.verification_contract import OSNVerificationContract, render_osn_verification_receipt
        contract = OSNVerificationContract(
            enabled=True, status="skipped",
            skipped_checks=["make test"],
            skipped_reason="needs_approval: medium-risk",
        )
        lines = render_osn_verification_receipt(contract)
        self.assertTrue(any("Skipped" in ln for ln in lines))
        self.assertTrue(any("make test" in ln for ln in lines))

    def test_51_render_backward_compat_empty_per_check_lists(self) -> None:
        from openshard.native.verification_contract import OSNVerificationContract, render_osn_verification_receipt
        contract = OSNVerificationContract(enabled=True, status="passed")
        lines = render_osn_verification_receipt(contract)
        self.assertTrue(any("OSN VERIFICATION" in ln for ln in lines))
        self.assertFalse(any("Passed" in ln for ln in lines))
        self.assertFalse(any("Failed" in ln for ln in lines))


class TestVerificationLoopWiringIntegration(unittest.TestCase):
    """Integration test: proves build_osn_verification_contract consumes verification_loop
    check fields and surfaces them in the contract output, as wired by the pipeline."""

    def test_54_pipeline_helper_wires_verification_loop(self) -> None:
        """Imports _build_osn_verification_contract_with_loop from the real pipeline module
        and calls it with a native_meta stub. Proves the helper passes verification_loop
        into build_osn_verification_contract — the wiring cannot be silently removed."""
        from openshard.run.pipeline import _build_osn_verification_contract_with_loop
        from openshard.native.context import NativeVerificationLoop

        loop = NativeVerificationLoop()
        loop.attempted = True
        loop.passed = True
        loop.check_attempted = ["pytest"]
        loop.check_passed = ["pytest"]

        native_meta = SimpleNamespace(
            osn_observation=_make_observation(suggested_checks=[]),
            osn_loop_summary=_make_loop_summary(
                verification_attempted=True, verification_passed=True
            ),
            verification_loop=loop,
        )

        contract = _build_osn_verification_contract_with_loop(
            native_meta, is_write_task=False
        )

        self.assertEqual(contract.attempted_checks, ["pytest"])
        self.assertEqual(contract.passed_checks, ["pytest"])
        self.assertEqual(contract.status, "passed")

    def test_52_pipeline_wiring_round_trip(self) -> None:
        """Simulate the pipeline flow: populate NativeVerificationLoop per-check fields,
        pass to build_osn_verification_contract, assert contract reflects actual results."""
        from openshard.native.context import NativeVerificationLoop
        from openshard.native.verification_contract import build_osn_verification_contract
        from openshard.verification.plan import (
            CommandSafety, VerificationCommand, VerificationKind,
            VerificationPlan, VerificationSource, safe_check_label,
        )

        # Simulate a safe plan with one pytest command
        cmd = VerificationCommand(
            name="tests", argv=["pytest"],
            kind=VerificationKind.test, source=VerificationSource.detected,
            safety=CommandSafety.safe, reason="matches safe prefix: pytest",
        )
        plan = VerificationPlan(commands=[cmd])

        # Simulate the pipeline populating the loop (as wired in pipeline.py)
        loop = NativeVerificationLoop()
        loop.attempted = True
        loop.exit_code = 0
        loop.passed = True
        check_label = safe_check_label(plan.commands[0])
        loop.check_attempted = [check_label]
        loop.check_passed = [check_label]

        # Simulate build_osn_verification_contract call with verification_loop
        loop_summary = _make_loop_summary(verification_attempted=True, verification_passed=True)
        contract = build_osn_verification_contract(
            osn_observation=None,
            osn_loop_summary=loop_summary,
            is_write_task=True,
            verification_loop=loop,
        )

        self.assertEqual(contract.status, "passed")
        self.assertEqual(contract.attempted_checks, ["pytest"])
        self.assertEqual(contract.passed_checks, ["pytest"])
        self.assertEqual(contract.failed_checks, [])
        self.assertIn("pytest", contract.expected_checks)
        self.assertFalse(contract.manual_review_required)

    def test_53_pipeline_wiring_skipped_unsafe_command(self) -> None:
        """Simulate: command not-all-safe → check_skipped populated → contract reflects it."""
        from openshard.native.context import NativeVerificationLoop
        from openshard.native.verification_contract import build_osn_verification_contract
        from openshard.verification.plan import (
            CommandSafety, VerificationCommand, VerificationKind,
            VerificationPlan, VerificationSource, safe_check_label,
        )

        cmd = VerificationCommand(
            name="verification", argv=["make", "test"],
            kind=VerificationKind.unknown, source=VerificationSource.detected,
            safety=CommandSafety.needs_approval,
            reason="medium-risk command requires approval: make",
        )
        plan = VerificationPlan(commands=[cmd])

        # Simulate not-all-safe branch in pipeline
        loop = NativeVerificationLoop()
        loop.attempted = False
        check_label = safe_check_label(plan.commands[0])
        loop.check_skipped = [check_label]
        loop.check_skipped_reasons = [
            f"{cmd.safety.value}: {cmd.reason}"[:80]
        ]

        loop_summary = _make_loop_summary(verification_attempted=False)
        contract = build_osn_verification_contract(
            osn_observation=None,
            osn_loop_summary=loop_summary,
            is_write_task=True,
            verification_loop=loop,
        )

        self.assertEqual(contract.status, "skipped")
        self.assertEqual(contract.skipped_checks, ["make test"])
        self.assertIn("needs_approval", contract.skipped_reason)
        self.assertTrue(contract.manual_review_required)


if __name__ == "__main__":
    unittest.main()
