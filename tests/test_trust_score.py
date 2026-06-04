"""Tests for the pure Run Trust Score v1 evaluator (openshard/history/trust_score.py)."""

from __future__ import annotations

import json

from openshard.history.shard_contract import build_shard_receipt
from openshard.history.trust_score import (
    PENALTY_DIRTY_REPO,
    PENALTY_EXECUTION_ERROR,
    PENALTY_FEEDBACK_PARTIAL,
    PENALTY_FEEDBACK_REJECTED,
    PENALTY_FEEDBACK_RETRIED,
    PENALTY_LOW_COMPLETENESS_HIGH,
    PENALTY_MANUAL_REVIEW,
    PENALTY_NO_TIMELINE,
    PENALTY_POLICY_DENIED,
    PENALTY_SECRET_SCAN,
    PENALTY_UNSAFE_INTERACTION,
    PENALTY_VERIFICATION_FAILED,
    PENALTY_VERIFICATION_NOT_RUN,
    RunTrustScore,
    evaluate_trust_score,
    format_human,
    to_payload,
)

# A maximally rich, clean run: all 15 completeness fields present, verification
# passed, no negative signals, a timeline. Scores a perfect 100 / strong and is
# the baseline for delta-based penalty tests.
BASE: dict = {
    "schema_version": "1.1",
    "task": "Add a helper function",
    "timestamp": "2026-06-03T00:00:00Z",
    "execution_model": "claude-opus-4.7",
    "duration_seconds": 2.5,
    "estimated_cost": 0.01,
    "files_updated": 1,
    "verification_attempted": True,
    "verification_passed": True,
    "file_context": {"paths": ["src/a.py"]},
    "context_files_injected_count": 3,
    "policy_decisions": [{"decision_id": "d1", "decision": "allow", "action": "write"}],
    "approval_receipt": {"granted": True},
    "execution_spans": [{"span_id": "s1", "name": "planning", "kind": "phase"}],
    "developer_feedback": {"outcome": "accepted"},
    "run_timeline": [
        {"event": "receipt_saved", "label": "Saved Shard receipt",
         "kind": "receipt", "status": "completed"}
    ],
    "summary": "done",
}


def _score(entry: dict, types: list[str] | None = None) -> RunTrustScore:
    receipt = build_shard_receipt(entry, index=0)
    return evaluate_trust_score(entry, receipt, interaction_event_types=types)


def _codes(ts: RunTrustScore) -> set[str]:
    return {p.code for p in ts.penalties}


def _points(ts: RunTrustScore, code: str) -> int:
    return next(p.points for p in ts.penalties if p.code == code)


# --- baseline & bands ------------------------------------------------------

def test_perfect_run_scores_100_strong():
    ts = _score(BASE)
    assert ts.score == 100
    assert ts.band == "strong"
    assert ts.status == "ok"
    assert ts.penalties == []
    assert ts.signals["verification"] == "passed"
    assert ts.signals["failure_category"] == "no_failure_detected"
    assert ts.signals["completeness_score_percent"] >= 85


def test_band_boundaries():
    # Directly exercise band mapping via crafted scores is covered by penalties;
    # here assert the documented edges hold through real penalties.
    assert _score(BASE).band == "strong"  # 100
    # verification failed (-35) -> 65 -> caution
    assert _score({**BASE, "verification_passed": False}).band == "caution"


# --- verification ----------------------------------------------------------

def test_verification_failed_penalty():
    base = _score(BASE)
    ts = _score({**BASE, "verification_passed": False})
    assert "verification_failed" in _codes(ts)
    assert base.score - ts.score == PENALTY_VERIFICATION_FAILED
    assert ts.signals["verification"] == "failed"


def test_verification_not_run_on_changed_run():
    base = _score(BASE)
    ts = _score({**BASE, "verification_attempted": False, "verification_passed": None})
    assert "verification_not_run" in _codes(ts)
    assert base.score - ts.score == PENALTY_VERIFICATION_NOT_RUN


def test_verification_not_run_skipped_without_changes():
    entry = {**BASE, "verification_attempted": False, "verification_passed": None,
             "files_updated": 0, "files_created": 0, "files_deleted": 0}
    # Remove file_context paths so no change evidence remains.
    entry.pop("file_context", None)
    ts = _score(entry)
    assert "verification_not_run" not in _codes(ts)


# --- policy / approval group cap (no double counting) ----------------------

def test_policy_denied_penalty():
    base = _score(BASE)
    ts = _score({**BASE, "approval_receipt": {"granted": False, "reason": "needs review"}})
    assert "policy_denied" in _codes(ts)
    assert base.score - ts.score == PENALTY_POLICY_DENIED


def test_policy_and_manual_review_do_not_double_count():
    # An approval-denied run also satisfies manual-review-required, but only the
    # stronger policy_denied penalty should apply.
    ts = _score({**BASE, "approval_receipt": {"granted": False, "reason": "needs review"}})
    assert "policy_denied" in _codes(ts)
    assert "manual_review_required" not in _codes(ts)


def test_manual_review_only_penalty():
    base = _score(BASE)
    ts = _score({**BASE, "osn_verification_contract": {"manual_review_required": True}})
    assert "manual_review_required" in _codes(ts)
    assert "policy_denied" not in _codes(ts)
    assert base.score - ts.score == PENALTY_MANUAL_REVIEW


# --- secret scan -----------------------------------------------------------

def test_secret_scan_penalty():
    base = _score(BASE)
    ts = _score({**BASE, "secret_scan_result": {
        "findings": [{"fingerprint": "fp1", "kind": "aws_key", "severity": "high"}]}})
    assert "secret_scan_finding" in _codes(ts)
    assert base.score - ts.score == PENALTY_SECRET_SCAN
    assert ts.signals["secret_scan_findings"] == 1


# --- completeness ----------------------------------------------------------

def test_low_completeness_penalty():
    ts = _score({"task": "x", "timestamp": "2026-06-03T00:00:00Z"})
    assert "low_completeness" in _codes(ts)
    assert _points(ts, "low_completeness") == PENALTY_LOW_COMPLETENESS_HIGH
    assert ts.signals["completeness_score_percent"] < 50


def test_high_completeness_has_no_completeness_penalty():
    ts = _score(BASE)
    assert "low_completeness" not in _codes(ts)


# --- feedback group --------------------------------------------------------

def test_feedback_rejected_penalty():
    base = _score(BASE)
    ts = _score({**BASE, "developer_feedback": {"outcome": "rejected"}})
    assert "feedback_rejected" in _codes(ts)
    assert base.score - ts.score == PENALTY_FEEDBACK_REJECTED


def test_feedback_partial_penalty():
    base = _score(BASE)
    ts = _score({**BASE, "developer_feedback": {"outcome": "partial"}})
    assert "feedback_partial" in _codes(ts)
    assert base.score - ts.score == PENALTY_FEEDBACK_PARTIAL


def test_feedback_manual_fix_is_partial():
    ts = _score({**BASE, "developer_feedback": {"outcome": "accepted", "manual_fix_required": True}})
    assert "feedback_partial" in _codes(ts)


def test_feedback_retried_penalty():
    base = _score(BASE)
    ts = _score({**BASE, "developer_feedback": {"outcome": "retried"}})
    assert "feedback_retried" in _codes(ts)
    assert base.score - ts.score == PENALTY_FEEDBACK_RETRIED


def test_feedback_group_emits_at_most_one():
    ts = _score({**BASE, "developer_feedback": {"outcome": "rejected", "manual_fix_required": True}})
    feedback_codes = {c for c in _codes(ts) if c.startswith("feedback_")}
    assert feedback_codes == {"feedback_rejected"}


# --- execution error -------------------------------------------------------

def test_execution_error_penalty():
    base = _score(BASE)
    ts = _score({**BASE, "error_class": "ToolError"})
    assert "execution_error" in _codes(ts)
    assert base.score - ts.score == PENALTY_EXECUTION_ERROR


def test_execution_error_suppressed_when_verification_failed():
    ts = _score({**BASE, "verification_passed": False, "error_class": "ToolError"})
    assert "verification_failed" in _codes(ts)
    assert "execution_error" not in _codes(ts)


# --- interaction events ----------------------------------------------------

def test_unsafe_interaction_penalty():
    base = _score(BASE)
    ts = _score(BASE, types=["unsafe_command"])
    assert "unsafe_interaction" in _codes(ts)
    assert base.score - ts.score == PENALTY_UNSAFE_INTERACTION
    assert ts.signals["interaction_events"] == 1


def test_unsafe_interaction_capped_once():
    base = _score(BASE)
    ts = _score(BASE, types=["unsafe_command", "wrong_file", "wrong_scope"])
    # Only one unsafe penalty regardless of count.
    assert sum(1 for c in _codes(ts) if c == "unsafe_interaction") == 1
    assert base.score - ts.score == PENALTY_UNSAFE_INTERACTION


def test_benign_interaction_types_are_not_penalised():
    ts = _score(BASE, types=["accepted", "edited"])
    assert "unsafe_interaction" not in _codes(ts)
    assert ts.penalties == []


# --- dirty repo & timeline (weak signals) ----------------------------------

def test_dirty_repo_penalty():
    base = _score(BASE)
    ts = _score({**BASE, "git_dirty": True})
    assert "dirty_repo" in _codes(ts)
    assert base.score - ts.score == PENALTY_DIRTY_REPO


def test_no_timeline_penalty_on_changed_run():
    entry = {**BASE}
    entry.pop("run_timeline", None)
    base = _score(BASE)
    ts = _score(entry)
    assert "no_timeline" in _codes(ts)
    assert base.score - ts.score == PENALTY_NO_TIMELINE


def test_no_timeline_skipped_without_changes():
    entry = {**BASE, "files_updated": 0, "files_created": 0, "files_deleted": 0}
    entry.pop("run_timeline", None)
    entry.pop("file_context", None)
    ts = _score(entry)
    assert "no_timeline" not in _codes(ts)


# --- floor & stacking ------------------------------------------------------

def test_score_floors_at_zero():
    ts = _score(
        {
            **BASE,
            "verification_passed": False,  # -35
            "approval_receipt": {"granted": False},  # -40
            "secret_scan_result": {"findings": [{"fingerprint": "fp", "kind": "k"}]},  # -40
            "developer_feedback": {"outcome": "rejected"},  # -25
            "git_dirty": True,  # -5
        },
        types=["unsafe_command"],  # -25
    )
    assert ts.score == 0
    assert ts.band == "unsafe"


# --- robustness ------------------------------------------------------------

def test_corrupt_receipt_does_not_crash():
    # Garbage-ish entry: build_shard_receipt never raises; evaluator degrades.
    ts = _score({"task": None, "timestamp": None, "developer_feedback": "not-a-dict"})
    assert isinstance(ts.score, int)
    assert 0 <= ts.score <= 100
    assert ts.status == "ok"


def test_failure_category_contributes_zero_points():
    # An execution_error run: the failure category is reported but the only
    # penalty comes from the discrete error signal, not the category itself.
    ts = _score({**BASE, "error_class": "ToolError"})
    assert ts.signals["failure_category"] == "execution_error"
    assert _codes(ts) == {"execution_error"}


# --- serialisation & no-leak ----------------------------------------------

def test_to_payload_shape():
    ts = _score(BASE)
    payload = to_payload(ts)
    assert set(payload) == {"score", "band", "signals", "penalties"}
    assert payload["score"] == 100
    assert isinstance(payload["penalties"], list)
    # JSON-serialisable.
    json.dumps(payload)


def test_no_secret_or_path_leak_in_output():
    entry = {
        **BASE,
        "verification_passed": False,
        "error_class": r"C:\Users\alice\secret.py",  # path-like -> scrubbed to None
        "developer_feedback": {
            "outcome": "rejected",
            "reason": "leaked AKIA1234567890 token at /home/alice/.env",
        },
        "files_updated": 1,
    }
    ts = _score(entry, types=["unsafe_command"])
    blob = json.dumps(to_payload(ts)) + "\n".join(format_human(ts))
    for needle in ("C:\\", "C:/", "/home/", "/Users/", "AKIA", "sk-", ".env", "leaked"):
        assert needle not in blob, f"unsafe substring {needle!r} leaked"


def test_format_human_is_short_and_careful():
    lines = format_human(_score(BASE))
    text = "\n".join(lines)
    assert "Trust Score: 100 / 100 (strong)" in text
    assert "trust heuristic over recorded proof signals" in text
    assert "guaranteed safe" not in text
    assert "certified" not in text
