"""Tests for openshard.history.provenance — Evidence Provenance v0."""

from __future__ import annotations

import json
import re
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

from click.testing import CliRunner

from openshard.history.provenance import (
    VALID_SOURCE_TYPES,
    VALID_STATUSES,
    ProvenanceRecord,
    _stable_provenance_id,
    build_provenance_from_entry,
    build_provenance_from_evidence_capsules,
    build_provenance_from_review_checks,
    make_provenance_record,
)

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_PROV_ID_RE = re.compile(r"^prov-[0-9a-f]{12}$")


def _rec(**kwargs) -> ProvenanceRecord:
    defaults = dict(
        source_type="verification",
        source_name="review_checks",
        claim="all checks passed",
        status="passed",
    )
    defaults.update(kwargs)
    return make_provenance_record(**defaults)


def _assert_no_unsafe(text: str) -> None:
    for needle in (
        "C:\\", "C:/", "/Users/", "/home/", "/etc/",
        "sk-", "AKIA", "api_key=", "password=", "secret=",
    ):
        assert needle.lower() not in text.lower(), (
            f"unsafe substring {needle!r} leaked in: {text!r}"
        )


def _capsule(
    capsule_id: str = "cap-001",
    kind: str = "secret_scan",
    summary: str = "Potential secret detected",
    source: str | None = "secret_scanner",
    severity: str | None = None,
):
    """Minimal EvidenceCapsule-like object."""
    from openshard.history.shard_contract import EvidenceCapsule
    return EvidenceCapsule(
        capsule_id=capsule_id,
        kind=kind,
        summary=summary,
        source=source,
        severity=severity,
    )


def _check(name: str = "terraform_fmt", status: str = "passed", summary: str = "formatting ok"):
    return {"name": name, "status": status, "summary": summary}


# ---------------------------------------------------------------------------
# TestProvenanceRecordShape
# ---------------------------------------------------------------------------

class TestProvenanceRecordShape(unittest.TestCase):
    def test_raw_content_stored_always_false(self):
        r = _rec()
        self.assertFalse(r.raw_content_stored)

    def test_raw_content_stored_cannot_be_forced_true(self):
        # Directly construct the dataclass with raw_content_stored=True and
        # verify __post_init__ resets it to False.
        r = ProvenanceRecord(
            provenance_id="prov-test",
            source_type="verification",
            source_name="checks",
            claim="ok",
            status="passed",
            raw_content_stored=True,
        )
        self.assertFalse(r.raw_content_stored)

    def test_provenance_id_format(self):
        r = _rec()
        self.assertRegex(r.provenance_id, _PROV_ID_RE)

    def test_json_serializable(self):
        r = _rec()
        json.dumps(asdict(r))  # must not raise

    def test_default_metadata_is_empty_dict(self):
        r = _rec()
        self.assertEqual(r.metadata, {})


# ---------------------------------------------------------------------------
# TestMakeProvenanceRecord
# ---------------------------------------------------------------------------

class TestMakeProvenanceRecord(unittest.TestCase):
    def test_unknown_source_type_coerced_to_unknown(self):
        r = _rec(source_type="invalid_xyz")
        self.assertEqual(r.source_type, "unknown")

    def test_all_valid_source_types_preserved(self):
        for st in VALID_SOURCE_TYPES:
            with self.subTest(source_type=st):
                r = _rec(source_type=st)
                self.assertEqual(r.source_type, st)

    def test_invalid_status_coerced_to_unknown(self):
        for bad in ("ok", "success", "error", "yes", ""):
            with self.subTest(status=bad):
                r = _rec(status=bad)
                self.assertEqual(r.status, "unknown")

    def test_valid_statuses_preserved(self):
        for s in VALID_STATUSES:
            with self.subTest(status=s):
                r = _rec(status=s)
                self.assertEqual(r.status, s)

    def test_sanitize_text_applied_to_claim_removes_secret(self):
        r = _rec(claim="result: sk-ABCDEFGHIJKLMNOPQRSTUVWXYZ1234")
        _assert_no_unsafe(r.claim)

    def test_sanitize_text_applied_to_source_name_drops_absolute_path(self):
        r = _rec(source_name="C:\\Users\\admin\\checks.py")
        _assert_no_unsafe(r.source_name)
        # Falls back to "unknown" since the whole value is an absolute path
        self.assertEqual(r.source_name, "unknown")

    def test_sanitize_text_applied_to_safe_summary_drops_secret(self):
        r = _rec(safe_summary="bearer sk-ABCDEFGHIJKLMNOPQRSTUVWXYZ1234")
        if r.safe_summary is not None:
            _assert_no_unsafe(r.safe_summary)

    def test_metadata_sanitized_removes_secret_value(self):
        r = _rec(metadata={"key": "sk-SECRETSECRETSECRETSECRET"})
        for val in r.metadata.values():
            if isinstance(val, str):
                _assert_no_unsafe(val)

    def test_metadata_nested_dict_dropped(self):
        r = _rec(metadata={"nested": {"a": 1}})
        self.assertNotIn("nested", r.metadata)

    def test_none_safe_summary_stays_none(self):
        r = _rec(safe_summary=None)
        self.assertIsNone(r.safe_summary)


# ---------------------------------------------------------------------------
# TestDeterministicProvenanceIds
# ---------------------------------------------------------------------------

class TestDeterministicProvenanceIds(unittest.TestCase):
    def test_same_input_produces_same_id(self):
        id1 = _stable_provenance_id("verification", "terraform_fmt", 0, run_ref="shard-20260101-0001")
        id2 = _stable_provenance_id("verification", "terraform_fmt", 0, run_ref="shard-20260101-0001")
        self.assertEqual(id1, id2)

    def test_different_index_produces_different_id(self):
        id1 = _stable_provenance_id("verification", "terraform_fmt", 0, run_ref="shard-x")
        id2 = _stable_provenance_id("verification", "terraform_fmt", 1, run_ref="shard-x")
        self.assertNotEqual(id1, id2)

    def test_different_source_name_produces_different_id(self):
        id1 = _stable_provenance_id("verification", "terraform_fmt", 0, run_ref="shard-x")
        id2 = _stable_provenance_id("verification", "terraform_validate", 0, run_ref="shard-x")
        self.assertNotEqual(id1, id2)

    def test_different_check_produces_different_id(self):
        id1 = _stable_provenance_id("verification", "checks", 0, run_ref="shard-x", related_check="fmt")
        id2 = _stable_provenance_id("verification", "checks", 0, run_ref="shard-x", related_check="validate")
        self.assertNotEqual(id1, id2)

    def test_different_run_ref_produces_different_id(self):
        id1 = _stable_provenance_id("verification", "terraform_fmt", 0, run_ref="shard-20260101-0001")
        id2 = _stable_provenance_id("verification", "terraform_fmt", 0, run_ref="shard-20260102-0001")
        self.assertNotEqual(id1, id2)

    def test_same_run_ref_produces_same_ids_across_calls(self):
        entry = {
            "shard_id": "shard-20260101-0001",
            "review_checks": [_check()],
        }
        ids1 = {r.provenance_id for r in build_provenance_from_entry(entry)}
        ids2 = {r.provenance_id for r in build_provenance_from_entry(entry)}
        self.assertEqual(ids1, ids2)

    def test_id_matches_expected_format(self):
        pid = _stable_provenance_id("evidence", "secret_scan", 0, run_ref="shard-x")
        self.assertRegex(pid, _PROV_ID_RE)


# ---------------------------------------------------------------------------
# TestStatusNormalization
# ---------------------------------------------------------------------------

class TestStatusNormalization(unittest.TestCase):
    def test_valid_statuses_preserved(self):
        for s in VALID_STATUSES:
            with self.subTest(status=s):
                r = _rec(status=s)
                self.assertEqual(r.status, s)

    def test_invalid_status_coerced_to_unknown(self):
        for bad in ("ok", "success", "error", "done", "n/a", "YES", ""):
            with self.subTest(status=bad):
                r = _rec(status=bad)
                self.assertEqual(r.status, "unknown")

    def test_evidence_capsule_severity_maps_to_warning(self):
        cap = _capsule(severity="high")
        records = build_provenance_from_evidence_capsules([cap], run_ref="shard-x")
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].status, "warning")

    def test_evidence_capsule_no_severity_maps_to_passed(self):
        cap = _capsule(severity=None)
        records = build_provenance_from_evidence_capsules([cap], run_ref="shard-x")
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].status, "passed")

    def test_evidence_capsule_empty_string_severity_maps_to_passed(self):
        cap = _capsule(severity="")
        records = build_provenance_from_evidence_capsules([cap], run_ref="shard-x")
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].status, "passed")


# ---------------------------------------------------------------------------
# TestBuildProvenanceFromEvidenceCapsules
# ---------------------------------------------------------------------------

class TestBuildProvenanceFromEvidenceCapsules(unittest.TestCase):
    def test_empty_capsules_returns_empty_list(self):
        self.assertEqual(build_provenance_from_evidence_capsules([]), [])

    def test_non_list_returns_empty_list(self):
        self.assertEqual(build_provenance_from_evidence_capsules(None), [])  # type: ignore[arg-type]
        self.assertEqual(build_provenance_from_evidence_capsules("bad"), [])  # type: ignore[arg-type]

    def test_single_capsule_produces_one_record(self):
        records = build_provenance_from_evidence_capsules([_capsule()])
        self.assertEqual(len(records), 1)

    def test_source_type_is_evidence(self):
        records = build_provenance_from_evidence_capsules([_capsule()])
        self.assertEqual(records[0].source_type, "evidence")

    def test_related_event_id_is_capsule_id(self):
        records = build_provenance_from_evidence_capsules([_capsule(capsule_id="cap-xyz")])
        self.assertEqual(records[0].related_event_id, "cap-xyz")

    def test_capsule_with_no_source_uses_kind_as_source_name(self):
        cap = _capsule(source=None, kind="my_scanner")
        records = build_provenance_from_evidence_capsules([cap])
        self.assertEqual(records[0].source_name, "my_scanner")

    def test_no_raw_content_stored(self):
        records = build_provenance_from_evidence_capsules([_capsule(), _capsule(capsule_id="cap-2")])
        for r in records:
            self.assertFalse(r.raw_content_stored)

    def test_ids_are_stable(self):
        caps = [_capsule()]
        ids1 = [r.provenance_id for r in build_provenance_from_evidence_capsules(caps, run_ref="shard-a")]
        ids2 = [r.provenance_id for r in build_provenance_from_evidence_capsules(caps, run_ref="shard-a")]
        self.assertEqual(ids1, ids2)

    def test_dict_capsule_also_accepted(self):
        cap_dict = {
            "capsule_id": "cap-dict-001",
            "kind": "secret_scan",
            "summary": "Potential secret detected and redacted",
            "source": "secret_scanner",
            "severity": None,
        }
        records = build_provenance_from_evidence_capsules([cap_dict])
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].source_type, "evidence")


# ---------------------------------------------------------------------------
# TestBuildProvenanceFromReviewChecks
# ---------------------------------------------------------------------------

class TestBuildProvenanceFromReviewChecks(unittest.TestCase):
    def test_empty_list_returns_empty_list(self):
        self.assertEqual(build_provenance_from_review_checks([]), [])

    def test_non_list_returns_empty_list(self):
        self.assertEqual(build_provenance_from_review_checks(None), [])  # type: ignore[arg-type]
        self.assertEqual(build_provenance_from_review_checks(99), [])  # type: ignore[arg-type]

    def test_passed_check_produces_passed_record(self):
        records = build_provenance_from_review_checks([_check(status="passed")])
        self.assertEqual(records[0].status, "passed")

    def test_failed_check_produces_failed_record(self):
        records = build_provenance_from_review_checks([_check(status="failed")])
        self.assertEqual(records[0].status, "failed")

    def test_skipped_check_produces_skipped_record(self):
        records = build_provenance_from_review_checks([_check(status="skipped")])
        self.assertEqual(records[0].status, "skipped")

    def test_source_type_is_verification_not_ci_check(self):
        records = build_provenance_from_review_checks([_check()])
        self.assertEqual(records[0].source_type, "verification")
        self.assertNotEqual(records[0].source_type, "ci_check")

    def test_check_name_in_related_check(self):
        records = build_provenance_from_review_checks([_check(name="terraform_fmt")])
        self.assertEqual(records[0].related_check, "terraform_fmt")

    def test_related_stage_is_verify(self):
        records = build_provenance_from_review_checks([_check()])
        self.assertEqual(records[0].related_stage, "verify")

    def test_non_dict_items_skipped(self):
        items = [None, "bad", 42, _check(name="ok_check", status="passed")]
        records = build_provenance_from_review_checks(items)
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].related_check, "ok_check")

    def test_no_raw_content_stored(self):
        records = build_provenance_from_review_checks([_check(), _check(name="terraform_validate")])
        for r in records:
            self.assertFalse(r.raw_content_stored)

    def test_ids_are_stable(self):
        checks = [_check()]
        ids1 = [r.provenance_id for r in build_provenance_from_review_checks(checks, run_ref="shard-a")]
        ids2 = [r.provenance_id for r in build_provenance_from_review_checks(checks, run_ref="shard-a")]
        self.assertEqual(ids1, ids2)

    def test_invalid_check_status_coerced_to_unknown(self):
        records = build_provenance_from_review_checks([_check(status="weird_status")])
        self.assertEqual(records[0].status, "unknown")


# ---------------------------------------------------------------------------
# TestBuildProvenanceFromEntry
# ---------------------------------------------------------------------------

class TestBuildProvenanceFromEntry(unittest.TestCase):
    def test_empty_entry_returns_empty_list(self):
        self.assertEqual(build_provenance_from_entry({}), [])

    def test_non_dict_input_returns_empty_list(self):
        for bad in (None, [], "string", 42, True):
            with self.subTest(input=bad):
                self.assertEqual(build_provenance_from_entry(bad), [])

    def test_entry_with_evidence_capsules_produces_evidence_records(self):
        entry = {
            "shard_id": "shard-20260101-0001",
            "evidence_capsules": [
                {"capsule_id": "c1", "kind": "secret_scan",
                 "summary": "Potential secret redacted", "source": "scanner", "severity": None},
            ],
        }
        records = build_provenance_from_entry(entry)
        self.assertTrue(any(r.source_type == "evidence" for r in records))

    def test_entry_with_review_checks_produces_verification_records(self):
        entry = {
            "shard_id": "shard-20260101-0001",
            "review_checks": [_check()],
        }
        records = build_provenance_from_entry(entry)
        self.assertTrue(any(r.source_type == "verification" for r in records))

    def test_entry_with_both_produces_combined_list(self):
        entry = {
            "shard_id": "shard-20260101-0001",
            "evidence_capsules": [
                {"capsule_id": "c1", "kind": "secret_scan",
                 "summary": "secret redacted", "source": "scanner", "severity": None},
            ],
            "review_checks": [_check()],
        }
        records = build_provenance_from_entry(entry)
        types = {r.source_type for r in records}
        self.assertIn("evidence", types)
        self.assertIn("verification", types)

    def test_never_raises_on_garbage_input(self):
        bad_entries = [
            {"evidence_capsules": "not-a-list", "review_checks": 99},
            {"evidence_capsules": [None, None, None]},
            {"review_checks": [None, None, None]},
            {"shard_id": 12345, "evidence_capsules": [{"no_id": True}]},
        ]
        for entry in bad_entries:
            with self.subTest(entry=entry):
                result = build_provenance_from_entry(entry)
                self.assertIsInstance(result, list)

    def test_all_records_json_serializable(self):
        entry = {
            "shard_id": "shard-20260101-0001",
            "evidence_capsules": [
                {"capsule_id": "c1", "kind": "secret_scan",
                 "summary": "secret redacted", "source": "scanner", "severity": None},
            ],
            "review_checks": [_check()],
        }
        records = build_provenance_from_entry(entry)
        json.dumps([asdict(r) for r in records])  # must not raise

    def test_ids_are_stable_across_calls(self):
        entry = {
            "shard_id": "shard-20260101-0001",
            "review_checks": [_check("terraform_fmt"), _check("terraform_validate", "passed", "ok")],
        }
        ids1 = {r.provenance_id for r in build_provenance_from_entry(entry)}
        ids2 = {r.provenance_id for r in build_provenance_from_entry(entry)}
        self.assertEqual(ids1, ids2)

    def test_run_ref_from_shard_id(self):
        entry = {
            "shard_id": "shard-A",
            "review_checks": [_check()],
        }
        entry_b = {
            "shard_id": "shard-B",
            "review_checks": [_check()],
        }
        ids_a = {r.provenance_id for r in build_provenance_from_entry(entry)}
        ids_b = {r.provenance_id for r in build_provenance_from_entry(entry_b)}
        self.assertTrue(ids_a.isdisjoint(ids_b), "same source in different shards must produce different IDs")

    def test_run_ref_fallback_to_timestamp(self):
        entry = {
            "timestamp": "2026-01-01T00:00:00Z",
            "review_checks": [_check()],
        }
        records = build_provenance_from_entry(entry)
        self.assertGreater(len(records), 0)

    def test_run_ref_fallback_to_unknown_run(self):
        entry = {"review_checks": [_check()]}
        records = build_provenance_from_entry(entry)
        self.assertGreater(len(records), 0)

    def test_no_unsafe_in_provenance_output(self):
        entry = {
            "shard_id": "shard-safe",
            "review_checks": [_check()],
            "evidence_capsules": [
                {"capsule_id": "c1", "kind": "secret_scan",
                 "summary": "Potential secret detected and redacted", "source": "scanner", "severity": None},
            ],
        }
        records = build_provenance_from_entry(entry)
        full_json = json.dumps([asdict(r) for r in records])
        _assert_no_unsafe(full_json)


# ---------------------------------------------------------------------------
# TestProvenanceWiredIntoShardReceipt
# ---------------------------------------------------------------------------

class TestProvenanceWiredIntoShardReceipt(unittest.TestCase):
    def test_provenance_field_exists_on_receipt(self):
        from openshard.history.shard_contract import build_shard_receipt
        receipt = build_shard_receipt({})
        self.assertTrue(hasattr(receipt, "provenance"))

    def test_provenance_defaults_to_empty_list_for_old_entry(self):
        from openshard.history.shard_contract import build_shard_receipt
        receipt = build_shard_receipt({"task": "test", "timestamp": "2026-01-01T00:00:00Z"})
        self.assertEqual(receipt.provenance, [])

    def test_evidence_capsules_produce_provenance_records(self):
        from openshard.history.shard_contract import build_shard_receipt
        entry = {
            "shard_id": "shard-20260101-0001",
            "evidence_capsules": [
                {"capsule_id": "c1", "kind": "secret_scan",
                 "summary": "Potential secret detected and redacted",
                 "source": "secret_scanner", "severity": None},
            ],
        }
        receipt = build_shard_receipt(entry)
        self.assertGreater(len(receipt.provenance), 0)
        self.assertTrue(all(r.source_type == "evidence" for r in receipt.provenance))

    def test_review_checks_produce_provenance_records(self):
        from openshard.history.shard_contract import build_shard_receipt
        entry = {
            "shard_id": "shard-20260101-0001",
            "review_checks": [_check()],
        }
        receipt = build_shard_receipt(entry)
        self.assertGreater(len(receipt.provenance), 0)
        self.assertTrue(all(r.source_type == "verification" for r in receipt.provenance))

    def test_never_raises_on_garbage_capsules(self):
        from openshard.history.shard_contract import build_shard_receipt
        # Only garbage evidence_capsules here — review_checks with None items is
        # an existing _format_review_checks limitation outside our scope.
        entry = {"evidence_capsules": "totally-invalid"}
        receipt = build_shard_receipt(entry)
        self.assertIsInstance(receipt.provenance, list)


# ---------------------------------------------------------------------------
# TestProvenanceInLastJson (CLI integration)
# ---------------------------------------------------------------------------

def _write_log(td: str, entries: list[dict]) -> Path:
    log_dir = Path(td) / ".openshard"
    log_dir.mkdir(exist_ok=True)
    log_file = log_dir / "runs.jsonl"
    log_file.write_text(
        "\n".join(json.dumps(e) for e in entries), encoding="utf-8"
    )
    return log_file


def _invoke_last_json(entries: list[dict]):
    from openshard.cli.main import cli
    runner = CliRunner()
    with tempfile.TemporaryDirectory() as td:
        _write_log(td, entries)
        with patch("openshard.cli.main.Path.cwd", return_value=Path(td)):
            result = runner.invoke(cli, ["last", "--json"])
    return result


_BASE_ENTRY = {
    "schema_version": "1.1",
    "task": "Add a helper function",
    "timestamp": "2026-06-01T00:00:00Z",
    "execution_model": "claude-sonnet-4-6",
    "verification_attempted": True,
    "verification_passed": True,
}


class TestProvenanceInLastJson(unittest.TestCase):
    def test_provenance_present_in_last_json_output(self):
        result = _invoke_last_json([_BASE_ENTRY])
        self.assertEqual(result.exit_code, 0)
        data = json.loads(result.output)
        self.assertIn("provenance", data["run"])
        self.assertIsInstance(data["run"]["provenance"], list)

    def test_provenance_empty_for_old_entry_without_checks(self):
        result = _invoke_last_json([_BASE_ENTRY])
        self.assertEqual(result.exit_code, 0)
        data = json.loads(result.output)
        self.assertEqual(data["run"]["provenance"], [])

    def test_provenance_populated_for_entry_with_checks(self):
        entry = {
            **_BASE_ENTRY,
            "shard_id": "shard-20260601-0001",
            "review_checks": [
                {"name": "terraform_fmt", "status": "passed", "summary": "formatting ok"},
            ],
        }
        result = _invoke_last_json([entry])
        self.assertEqual(result.exit_code, 0)
        data = json.loads(result.output)
        self.assertGreater(len(data["run"]["provenance"]), 0)
        rec = data["run"]["provenance"][0]
        self.assertEqual(rec["source_type"], "verification")
        self.assertFalse(rec["raw_content_stored"])

    def test_provenance_records_have_expected_keys(self):
        entry = {
            **_BASE_ENTRY,
            "shard_id": "shard-20260601-0001",
            "review_checks": [_check()],
        }
        result = _invoke_last_json([entry])
        self.assertEqual(result.exit_code, 0)
        data = json.loads(result.output)
        if data["run"]["provenance"]:
            rec = data["run"]["provenance"][0]
            for key in ("provenance_id", "source_type", "source_name", "claim", "status",
                        "raw_content_stored", "metadata"):
                self.assertIn(key, rec, f"key {key!r} missing from provenance record")

    def test_no_unsafe_values_in_provenance_json(self):
        entry = {
            **_BASE_ENTRY,
            "shard_id": "shard-20260601-0001",
            "review_checks": [_check()],
        }
        result = _invoke_last_json([entry])
        self.assertEqual(result.exit_code, 0)
        data = json.loads(result.output)
        _assert_no_unsafe(json.dumps(data["run"]["provenance"]))

    def test_provenance_absent_from_human_output(self):
        from openshard.cli.main import cli
        runner = CliRunner()
        entry = {
            **_BASE_ENTRY,
            "shard_id": "shard-20260601-0001",
            "review_checks": [_check()],
        }
        with tempfile.TemporaryDirectory() as td:
            _write_log(td, [entry])
            with patch("openshard.cli.main.Path.cwd", return_value=Path(td)):
                result = runner.invoke(cli, ["last"])
        self.assertEqual(result.exit_code, 0)
        self.assertNotIn("provenance_id", result.output)
        self.assertNotIn("prov-", result.output)

    def test_last_json_remains_valid_json(self):
        result = _invoke_last_json([_BASE_ENTRY])
        self.assertEqual(result.exit_code, 0)
        # must not raise
        json.loads(result.output)
