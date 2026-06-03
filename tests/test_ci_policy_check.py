from __future__ import annotations

from openshard.ci.policy_check import evaluate_ci_check
from openshard.history.shard_contract import build_shard_receipt


def _eval(entry: dict, *, strict: bool = False):
    receipt = build_shard_receipt(entry, index=0)
    return evaluate_ci_check(entry, receipt, strict=strict)


def test_clean_verified_run_passes():
    result = _eval({"verification_attempted": True, "verification_passed": True})
    assert result.status == "pass"
    assert result.exit_code == 0
    assert result.checks["verification"] == "passed"
    assert result.reasons == []
    assert result.warnings == []


def test_verification_failed_fails():
    result = _eval({"verification_attempted": True, "verification_passed": False})
    assert result.status == "fail"
    assert result.exit_code == 1
    assert result.checks["verification"] == "failed"
    assert any("Verification failed" in r for r in result.reasons)


def test_manual_review_via_approval_denied_fails():
    result = _eval({"approval_receipt": {"granted": False, "reason": "needs review"}})
    assert result.status == "fail"
    assert result.exit_code == 1
    assert result.checks["manual_review_required"] is True
    assert any("Manual review required" in r for r in result.reasons)


def test_policy_deny_fails():
    result = _eval(
        {
            "verification_attempted": True,
            "verification_passed": True,
            "policy_decisions": [
                {"decision_id": "d1", "decision": "deny", "action": "write"}
            ],
        }
    )
    assert result.status == "fail"
    assert result.checks["manual_review_required"] is True


def test_verification_not_run_warns():
    result = _eval({"verification_attempted": False})
    assert result.status == "warn"
    assert result.exit_code == 0
    assert result.checks["verification"] == "not_run"


def test_old_receipt_missing_fields_is_unknown_warn():
    result = _eval({"task": "legacy run"})
    assert result.status == "warn"
    assert result.exit_code == 0
    assert result.checks["verification"] == "unknown"


def test_secret_findings_warn_by_default():
    result = _eval(
        {
            "verification_attempted": True,
            "verification_passed": True,
            "secret_scan_result": {
                "findings": [
                    {"fingerprint": "fp1", "kind": "aws_key", "severity": "high"}
                ]
            },
        }
    )
    assert result.status == "warn"
    assert result.exit_code == 0
    assert result.checks["secret_scan_findings"] == 1
    assert any("secret-scan finding" in w for w in result.warnings)


def test_strict_promotes_warn_to_fail():
    result = _eval({"verification_attempted": False}, strict=True)
    assert result.status == "fail"
    assert result.exit_code == 1
    # Promoted warnings become blocking reasons; warnings list is cleared.
    assert result.warnings == []
    assert any("Verification was not run" in r for r in result.reasons)


def test_strict_does_not_change_pass():
    result = _eval(
        {"verification_attempted": True, "verification_passed": True}, strict=True
    )
    assert result.status == "pass"
    assert result.exit_code == 0


def test_failed_verification_takes_priority_over_secret_warn():
    result = _eval(
        {
            "verification_attempted": True,
            "verification_passed": False,
            "secret_scan_result": {
                "findings": [{"fingerprint": "fp1", "kind": "token"}]
            },
        }
    )
    assert result.status == "fail"
    # Secret findings are still reported in checks even when failing.
    assert result.checks["secret_scan_findings"] == 1
