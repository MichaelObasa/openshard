from __future__ import annotations

from openshard.history.failures import (
    CATEGORIES,
    _safe_error_class,
    classify_failure,
    evaluate_failures,
)
from openshard.history.shard_contract import build_shard_receipt


def _classify(entry: dict):
    receipt = build_shard_receipt(entry, index=0)
    return classify_failure(entry, receipt)


def _report(entries: list[dict]):
    pairs = [(e, build_shard_receipt(e, index=i)) for i, e in enumerate(entries)]
    return evaluate_failures(pairs)


# --- one test per category -------------------------------------------------

def test_policy_denied_via_approval():
    c = _classify({"approval_receipt": {"granted": False, "reason": "review"}})
    assert c.category == "policy_denied"
    assert c.confidence == "high"
    assert c.signals["policy_denied"] is True


def test_policy_denied_via_policy_decision():
    c = _classify(
        {
            "verification_attempted": True,
            "verification_passed": True,
            "policy_decisions": [{"decision_id": "d1", "decision": "deny", "action": "write"}],
        }
    )
    assert c.category == "policy_denied"


def test_verification_failed():
    c = _classify({"verification_attempted": True, "verification_passed": False})
    assert c.category == "verification_failed"
    assert c.confidence == "high"
    assert c.signals["verification"] == "failed"


def test_secret_scan_finding():
    c = _classify(
        {
            "verification_attempted": True,
            "verification_passed": True,
            "secret_scan_result": {
                "findings": [{"fingerprint": "fp1", "kind": "aws_key", "severity": "high"}]
            },
        }
    )
    assert c.category == "secret_scan_finding"
    assert c.signals["secret_scan_findings"] == 1


def test_manual_review_required():
    c = _classify(
        {
            "verification_attempted": True,
            "verification_passed": True,
            "osn_verification_contract": {"manual_review_required": True},
        }
    )
    assert c.category == "manual_review_required"
    assert c.signals["manual_review_required"] is True


def test_execution_error():
    c = _classify(
        {
            "verification_attempted": True,
            "verification_passed": True,
            "error_class": "ToolError",
        }
    )
    assert c.category == "execution_error"
    assert c.confidence == "medium"
    assert c.signals["error_class"] == "ToolError"


def test_user_rejected():
    c = _classify(
        {
            "verification_attempted": True,
            "verification_passed": True,
            "developer_feedback": {"outcome": "rejected"},
        }
    )
    assert c.category == "user_rejected"
    assert c.signals["feedback_outcome"] == "rejected"


def test_user_abandoned_is_rejected():
    c = _classify(
        {
            "verification_attempted": True,
            "verification_passed": True,
            "developer_feedback": {"outcome": "abandoned"},
        }
    )
    assert c.category == "user_rejected"


def test_partial_success_via_outcome():
    c = _classify(
        {
            "verification_attempted": True,
            "verification_passed": True,
            "developer_feedback": {"outcome": "partial"},
        }
    )
    assert c.category == "partial_success"
    assert c.confidence == "medium"


def test_partial_success_via_manual_fix():
    c = _classify(
        {
            "verification_attempted": True,
            "verification_passed": True,
            "developer_feedback": {"outcome": "accepted", "manual_fix_required": True},
        }
    )
    assert c.category == "partial_success"


def test_verification_not_run_with_changes():
    c = _classify({"verification_attempted": False, "files_updated": 1})
    assert c.category == "verification_not_run"
    assert c.confidence == "low"
    assert c.signals["verification"] == "not_run"


def test_unknown_failure_via_retried():
    c = _classify({"developer_feedback": {"outcome": "retried"}})
    assert c.category == "unknown_failure"
    assert c.confidence == "low"


def test_no_failure_detected_clean_run():
    c = _classify({"verification_attempted": True, "verification_passed": True})
    assert c.category == "no_failure_detected"
    assert c.confidence == "high"


def test_read_only_not_run_is_no_failure():
    # No checks run but no changes either -> not a failure.
    c = _classify({"verification_attempted": False})
    assert c.category == "no_failure_detected"


# --- priority order --------------------------------------------------------

def test_policy_deny_beats_verification_failed():
    c = _classify(
        {
            "verification_attempted": True,
            "verification_passed": False,
            "policy_decisions": [{"decision_id": "d1", "decision": "deny"}],
        }
    )
    assert c.category == "policy_denied"


def test_verification_failed_beats_secret_finding():
    c = _classify(
        {
            "verification_attempted": True,
            "verification_passed": False,
            "secret_scan_result": {"findings": [{"fingerprint": "fp1", "kind": "token"}]},
        }
    )
    assert c.category == "verification_failed"
    assert c.signals["secret_scan_findings"] == 1


def test_secret_beats_manual_review():
    c = _classify(
        {
            "verification_attempted": True,
            "verification_passed": True,
            "secret_scan_result": {"findings": [{"fingerprint": "fp1", "kind": "token"}]},
            "osn_verification_contract": {"manual_review_required": True},
        }
    )
    assert c.category == "secret_scan_finding"


def test_error_class_beats_feedback():
    c = _classify(
        {
            "verification_attempted": True,
            "verification_passed": True,
            "error_class": "ToolError",
            "developer_feedback": {"outcome": "rejected"},
        }
    )
    assert c.category == "execution_error"


# --- robustness ------------------------------------------------------------

def test_old_receipt_missing_fields_does_not_crash():
    c = _classify({"task": "legacy run"})
    assert c.category in CATEGORIES
    # Unknown verification + no changes -> not a failure.
    assert c.category == "no_failure_detected"


def test_corrupt_feedback_shape_does_not_crash():
    c = _classify({"developer_feedback": "not-a-dict"})
    assert c.category in CATEGORIES


# --- aggregate -------------------------------------------------------------

def test_empty_input_zeroed_report():
    report = evaluate_failures([])
    assert report.runs_checked == 0
    assert report.top_categories == []
    assert report.failures == []
    assert report.recommendations == []
    assert all(v == 0 for v in report.category_counts.values())


def test_aggregate_counts_and_top_categories():
    entries = [
        {"verification_attempted": True, "verification_passed": False},  # verification_failed
        {"verification_attempted": True, "verification_passed": False},  # verification_failed
        {"approval_receipt": {"granted": False}},  # policy_denied
        {"verification_attempted": True, "verification_passed": True},  # no_failure_detected
    ]
    report = _report(entries)
    assert report.runs_checked == 4
    assert report.category_counts["verification_failed"] == 2
    assert report.category_counts["policy_denied"] == 1
    assert report.category_counts["no_failure_detected"] == 1
    # Most common failure first; no_failure_detected excluded from top.
    assert report.top_categories[0] == {"category": "verification_failed", "count": 2}
    assert all(tc["category"] != "no_failure_detected" for tc in report.top_categories)


def test_failures_exclude_no_failure_and_are_most_recent_first():
    entries = [
        {"verification_attempted": True, "verification_passed": True},  # clean (oldest)
        {"verification_attempted": True, "verification_passed": False},  # fail
        {"approval_receipt": {"granted": False}},  # policy_denied (newest)
    ]
    report = _report(entries)
    cats = [fc.category for fc in report.failures]
    assert "no_failure_detected" not in cats
    # Most recent failure first.
    assert cats[0] == "policy_denied"
    assert cats[1] == "verification_failed"


def test_recommendations_present_for_seen_categories():
    report = _report([{"verification_attempted": True, "verification_passed": False}])
    assert any("verification" in r.lower() for r in report.recommendations)


# --- error_class sanitisation ---------------------------------------------

def test_safe_error_class_accepts_short_token():
    assert _safe_error_class("VerificationError") == "VerificationError"
    assert _safe_error_class("  TimeoutError  ") == "TimeoutError"


def test_safe_error_class_rejects_non_string():
    assert _safe_error_class(None) is None
    assert _safe_error_class(123) is None


def test_safe_error_class_rejects_paths():
    assert _safe_error_class("/home/user/secret.env") is None
    assert _safe_error_class("C:\\Users\\Michael\\x") is None


def test_safe_error_class_rejects_long_and_freetext():
    assert _safe_error_class("A" * 200) is None
    assert _safe_error_class("an error happened with the tool") is None


def test_signals_error_class_scrubs_pathlike_value():
    c = _classify(
        {
            "verification_attempted": True,
            "verification_passed": True,
            "error_class": "/home/user/secret.env",
        }
    )
    # Path-like error_class is scrubbed, so it is not an execution_error.
    assert c.signals["error_class"] is None
    assert c.category == "no_failure_detected"
