"""Tests for the canonical verification signal mapping.

Covers _verification_from_osn_contract (shard_contract) and
verification_status_from_receipt (proof_signals): the full OSN state set is
preserved end-to-end, manual_review and impossible are never collapsed into
unknown, and old or non-native records keep their existing behaviour.
"""
from __future__ import annotations

from openshard.history.proof_signals import verification_status_from_receipt
from openshard.history.shard_contract import build_shard_receipt


def _receipt(osn: dict | None = None, **entry_kwargs):
    entry = dict(entry_kwargs)
    if osn is not None:
        base = {"enabled": True}
        base.update(osn)
        entry["osn_verification_contract"] = base
    return build_shard_receipt(entry, index=0)


def _status(osn: dict | None = None, **entry_kwargs) -> str:
    return verification_status_from_receipt(_receipt(osn, **entry_kwargs))


def test_passed_maps_to_passed():
    assert _status({"status": "passed"}) == "passed"


def test_failed_maps_to_failed():
    assert _status({"status": "failed"}) == "failed"


def test_skipped_maps_to_skipped():
    assert _status({"status": "skipped"}) == "skipped"


def test_manual_review_status_maps_to_manual_review():
    assert _status({"status": "manual_review"}) == "manual_review"


def test_manual_review_required_overrides_skipped():
    # A skipped write task records manual_review_required; surface manual_review.
    assert _status({"status": "skipped", "manual_review_required": True}) == "manual_review"


def test_impossible_with_manual_review_maps_to_manual_review():
    assert _status({"status": "impossible", "manual_review_required": True}) == "manual_review"


def test_impossible_without_manual_review_maps_to_skipped():
    assert _status({"status": "impossible"}) == "skipped"


def test_not_run_maps_to_not_run():
    assert _status({"status": "not_run"}) == "not_run"


def test_unknown_with_manual_review_is_not_collapsed():
    # unknown + needs human intervention should surface as manual_review.
    assert _status({"status": "unknown", "manual_review_required": True}) == "manual_review"


def test_unknown_without_review_stays_unknown():
    assert _status({"status": "unknown"}) == "unknown"


def test_reason_is_carried_onto_receipt():
    receipt = _receipt({"status": "skipped", "skipped_reason": "needs_approval: make"})
    assert "needs_approval" in receipt.verification_reason


def test_returncode_and_duration_carried_onto_receipt():
    receipt = _receipt({"status": "failed", "returncode": 1, "duration_seconds": 2.5})
    assert receipt.verification_returncode == 1
    assert receipt.verification_duration_seconds == 2.5
    assert receipt.verification_raw_output_stored is False


def test_disabled_contract_falls_back_to_booleans():
    # No OSN contract: old boolean logic still applies.
    assert _status(verification_attempted=True, verification_passed=True) == "passed"
    assert _status(verification_attempted=True, verification_passed=False) == "failed"
    assert _status(verification_attempted=False) == "not_run"


def test_old_record_without_contract_is_unknown():
    assert _status(task="legacy run") == "unknown"
