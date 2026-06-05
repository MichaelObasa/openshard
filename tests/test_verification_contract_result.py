from __future__ import annotations

import json
import unittest
from dataclasses import asdict

from openshard.native.context import (
    NativeContractCheckResult,
    NativeValidationContract,
    NativeVerificationContractResult,
    NativeVerificationLoop,
    build_native_verification_contract_result,
)


def _contract(**kwargs) -> NativeValidationContract:
    defaults = dict(
        intent="add pagination",
        risk_level="medium",
        acceptance_checks=["tests pass", "lint clean"],
        verification_commands=["pytest"],
        strength="strong",
    )
    defaults.update(kwargs)
    return NativeValidationContract(**defaults)


def _loop(**kwargs) -> NativeVerificationLoop:
    defaults = dict(attempted=True, passed=True, exit_code=0, output_chars=500)
    defaults.update(kwargs)
    return NativeVerificationLoop(**defaults)


def _build(**kwargs) -> NativeVerificationContractResult:
    return build_native_verification_contract_result(**kwargs)


class TestNativeVerificationContractResultDefaults(unittest.TestCase):

    def test_check_result_defaults(self):
        c = NativeContractCheckResult()
        self.assertEqual(c.check_id, "")
        self.assertEqual(c.expected_check, "")
        self.assertEqual(c.verification_source, "none")
        self.assertEqual(c.status, "unknown")
        self.assertEqual(c.reason, "")
        self.assertEqual(c.evidence_summary, "")
        self.assertFalse(c.raw_content_stored)

    def test_result_defaults(self):
        r = NativeVerificationContractResult()
        self.assertEqual(r.checks, [])
        self.assertEqual(r.overall_status, "unknown")
        self.assertEqual(r.reason, "")
        self.assertFalse(r.raw_content_stored)

    def test_asdict_json_serializable(self):
        r = NativeVerificationContractResult(
            checks=[NativeContractCheckResult(check_id="check_0", status="passed")],
            overall_status="passed",
            reason="verification suite passed",
        )
        d = asdict(r)
        raw = json.dumps(d)
        parsed = json.loads(raw)
        self.assertEqual(parsed["overall_status"], "passed")
        self.assertEqual(len(parsed["checks"]), 1)
        self.assertEqual(parsed["checks"][0]["check_id"], "check_0")
        self.assertFalse(parsed["raw_content_stored"])


class TestBuildNoContract(unittest.TestCase):

    def test_none_contract_returns_unknown(self):
        r = _build(validation_contract=None, verification_loop=_loop())
        self.assertEqual(r.overall_status, "unknown")
        self.assertEqual(r.checks, [])

    def test_none_contract_reason_set(self):
        r = _build(validation_contract=None, verification_loop=None)
        self.assertIn("no validation contract", r.reason)


class TestBuildNoChecks(unittest.TestCase):

    def test_empty_checks_returns_unknown(self):
        contract = _contract(acceptance_checks=[])
        r = _build(validation_contract=contract, verification_loop=_loop())
        self.assertEqual(r.overall_status, "unknown")
        self.assertEqual(r.checks, [])

    def test_empty_checks_reason_set(self):
        contract = _contract(acceptance_checks=[])
        r = _build(validation_contract=contract, verification_loop=None)
        self.assertIn("no acceptance checks defined", r.reason)


class TestBuildVerificationSkipped(unittest.TestCase):

    def test_none_loop_all_checks_skipped(self):
        r = _build(validation_contract=_contract(), verification_loop=None)
        self.assertEqual(r.overall_status, "skipped")
        for ck in r.checks:
            self.assertEqual(ck.status, "skipped")

    def test_none_loop_reason_not_attempted(self):
        r = _build(validation_contract=_contract(), verification_loop=None)
        self.assertIn("not attempted", r.reason)

    def test_not_attempted_loop_all_checks_skipped(self):
        loop = _loop(attempted=False)
        r = _build(validation_contract=_contract(), verification_loop=loop)
        self.assertEqual(r.overall_status, "skipped")
        for ck in r.checks:
            self.assertEqual(ck.status, "skipped")

    def test_skipped_source_is_none(self):
        r = _build(validation_contract=_contract(), verification_loop=None)
        for ck in r.checks:
            self.assertEqual(ck.verification_source, "none")

    def test_skipped_evidence_is_empty(self):
        r = _build(validation_contract=_contract(), verification_loop=None)
        for ck in r.checks:
            self.assertEqual(ck.evidence_summary, "")


class TestBuildVerificationPassed(unittest.TestCase):

    def test_passed_loop_all_checks_passed(self):
        r = _build(validation_contract=_contract(), verification_loop=_loop(passed=True, exit_code=0, output_chars=300))
        self.assertEqual(r.overall_status, "passed")
        for ck in r.checks:
            self.assertEqual(ck.status, "passed")

    def test_passed_reason_set(self):
        r = _build(validation_contract=_contract(), verification_loop=_loop(passed=True))
        self.assertIn("passed", r.reason)

    def test_passed_source_is_verification_loop(self):
        r = _build(validation_contract=_contract(), verification_loop=_loop(passed=True))
        for ck in r.checks:
            self.assertEqual(ck.verification_source, "verification_loop")

    def test_passed_evidence_contains_exit_code(self):
        r = _build(validation_contract=_contract(), verification_loop=_loop(passed=True, exit_code=0, output_chars=200))
        for ck in r.checks:
            self.assertIn("exit_code=0", ck.evidence_summary)
            self.assertIn("200", ck.evidence_summary)


class TestBuildVerificationFailed(unittest.TestCase):

    def test_failed_loop_all_checks_failed(self):
        r = _build(validation_contract=_contract(), verification_loop=_loop(attempted=True, passed=False, exit_code=1, output_chars=100))
        self.assertEqual(r.overall_status, "failed")
        for ck in r.checks:
            self.assertEqual(ck.status, "failed")

    def test_failed_reason_set(self):
        r = _build(validation_contract=_contract(), verification_loop=_loop(attempted=True, passed=False))
        self.assertIn("failed", r.reason)

    def test_failed_source_is_verification_loop(self):
        r = _build(validation_contract=_contract(), verification_loop=_loop(attempted=True, passed=False))
        for ck in r.checks:
            self.assertEqual(ck.verification_source, "verification_loop")

    def test_failed_evidence_contains_exit_code(self):
        r = _build(
            validation_contract=_contract(),
            verification_loop=_loop(attempted=True, passed=False, exit_code=2, output_chars=50),
        )
        for ck in r.checks:
            self.assertIn("exit_code=2", ck.evidence_summary)
            self.assertIn("50", ck.evidence_summary)


class TestRawContentNeverStored(unittest.TestCase):

    def test_skipped_raw_content_false(self):
        r = _build(validation_contract=_contract(), verification_loop=None)
        self.assertFalse(r.raw_content_stored)
        for ck in r.checks:
            self.assertFalse(ck.raw_content_stored)

    def test_passed_raw_content_false(self):
        r = _build(validation_contract=_contract(), verification_loop=_loop(passed=True))
        self.assertFalse(r.raw_content_stored)
        for ck in r.checks:
            self.assertFalse(ck.raw_content_stored)

    def test_failed_raw_content_false(self):
        r = _build(validation_contract=_contract(), verification_loop=_loop(attempted=True, passed=False))
        self.assertFalse(r.raw_content_stored)
        for ck in r.checks:
            self.assertFalse(ck.raw_content_stored)

    def test_no_contract_raw_content_false(self):
        r = _build(validation_contract=None, verification_loop=_loop())
        self.assertFalse(r.raw_content_stored)


class TestCheckIds(unittest.TestCase):

    def test_check_ids_sequential(self):
        contract = _contract(acceptance_checks=["a", "b", "c"])
        r = _build(validation_contract=contract, verification_loop=_loop(passed=True))
        ids = [ck.check_id for ck in r.checks]
        self.assertEqual(ids, ["check_0", "check_1", "check_2"])

    def test_single_check_id(self):
        contract = _contract(acceptance_checks=["only check"])
        r = _build(validation_contract=contract, verification_loop=_loop(passed=True))
        self.assertEqual(r.checks[0].check_id, "check_0")

    def test_check_count_matches_acceptance_checks(self):
        contract = _contract(acceptance_checks=["a", "b"])
        r = _build(validation_contract=contract, verification_loop=_loop(passed=True))
        self.assertEqual(len(r.checks), 2)


class TestExpectedCheckPreserved(unittest.TestCase):

    def test_expected_check_matches_original(self):
        checks = ["tests pass", "lint clean", "coverage >= 80%"]
        contract = _contract(acceptance_checks=checks)
        r = _build(validation_contract=contract, verification_loop=_loop(passed=True))
        for i, ck in enumerate(r.checks):
            self.assertEqual(ck.expected_check, checks[i])

    def test_expected_check_not_modified(self):
        contract = _contract(acceptance_checks=["  whitespace check  "])
        r = _build(validation_contract=contract, verification_loop=_loop(passed=True))
        self.assertEqual(r.checks[0].expected_check, "  whitespace check  ")


class TestOldRecordsNoLinkage(unittest.TestCase):
    """Entries without verification_contract_result must render without error."""

    def test_native_meta_from_entry_missing_key(self):
        from openshard.cli.run_output import _native_meta_from_entry
        entry = {"workflow": "native", "executor": "native"}
        meta = _native_meta_from_entry(entry)
        self.assertIsNotNone(meta)
        self.assertIsNone(getattr(meta, "verification_contract_result", None))
