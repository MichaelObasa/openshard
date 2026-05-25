from __future__ import annotations

import json
from pathlib import Path

from openshard.history.session_signals import (
    RETRY_WORDS,
    _has_retry_words,
    infer_signals_from_session,
    run_inference,
)

_SESSION = "sess-abc"
_RUN = "run-001"
_SHARD = "shard-x"

_REQUIRED_FIELDS = {
    "schema_version",
    "signal_id",
    "session_id",
    "run_id",
    "shard_id",
    "timestamp",
    "source",
    "signal_type",
    "confidence",
    "reason",
    "metadata",
}


def _response_event(run_id: str = _RUN, shard_id: str = _SHARD) -> dict:
    return {
        "event_type": "openshard_response",
        "session_id": _SESSION,
        "run_id": run_id,
        "shard_id": shard_id,
        "summary": "",
        "metadata": {},
    }


def _feedback_event(outcome: str) -> dict:
    return {
        "event_type": "feedback_recorded",
        "session_id": _SESSION,
        "run_id": None,
        "shard_id": None,
        "metadata": {"outcome": outcome},
    }


def _command_event(command: str) -> dict:
    return {
        "event_type": "command_invoked",
        "session_id": _SESSION,
        "run_id": None,
        "shard_id": None,
        "command": command,
        "summary": "",
        "metadata": {},
    }


def _user_message_event(summary: str) -> dict:
    return {
        "event_type": "user_message",
        "session_id": _SESSION,
        "run_id": None,
        "shard_id": None,
        "summary": summary,
        "metadata": {},
    }


# ---------------------------------------------------------------------------
# Explicit feedback signals
# ---------------------------------------------------------------------------


def test_accepted_feedback_produces_accepted_explicit():
    signals = infer_signals_from_session([_feedback_event("accepted")])
    assert len(signals) == 1
    s = signals[0]
    assert s["signal_type"] == "accepted_explicit"
    assert s["confidence"] == "high"
    assert s["source"] == "rules_v1"


def test_partial_feedback_produces_partial_explicit():
    signals = infer_signals_from_session([_feedback_event("partial")])
    assert len(signals) == 1
    assert signals[0]["signal_type"] == "partial_explicit"
    assert signals[0]["confidence"] == "high"


def test_rejected_feedback_produces_rejected_explicit():
    signals = infer_signals_from_session([_feedback_event("rejected")])
    assert len(signals) == 1
    assert signals[0]["signal_type"] == "rejected_explicit"
    assert signals[0]["confidence"] == "high"


def test_accepted_after_run_links_to_run_shard():
    events = [_response_event(), _feedback_event("accepted")]
    signals = infer_signals_from_session(events)
    assert len(signals) == 1
    s = signals[0]
    assert s["signal_type"] == "accepted_explicit"
    assert s["run_id"] == _RUN
    assert s["shard_id"] == _SHARD


def test_partial_feedback_falls_back_to_last_run_response():
    events = [_response_event(), _feedback_event("partial")]
    signals = infer_signals_from_session(events)
    assert signals[0]["run_id"] == _RUN
    assert signals[0]["shard_id"] == _SHARD


def test_rejected_feedback_falls_back_to_last_run_response():
    events = [_response_event(), _feedback_event("rejected")]
    signals = infer_signals_from_session(events)
    assert signals[0]["run_id"] == _RUN
    assert signals[0]["shard_id"] == _SHARD


def test_non_run_response_does_not_erase_run_context():
    # An openshard_response with null run_id (e.g. /last output) must not
    # clear the last_run_response pointer established by the prior real run.
    non_run_response = {
        "event_type": "openshard_response",
        "session_id": _SESSION,
        "run_id": None,
        "shard_id": None,
        "metadata": {},
    }
    events = [_response_event(), non_run_response, _feedback_event("accepted")]
    signals = infer_signals_from_session(events)
    assert signals[0]["run_id"] == _RUN
    assert signals[0]["shard_id"] == _SHARD


# ---------------------------------------------------------------------------
# inspected_result
# ---------------------------------------------------------------------------


def test_last_more_after_run_produces_inspected_result():
    events = [_response_event(), _command_event("/last more")]
    signals = infer_signals_from_session(events)
    assert len(signals) == 1
    s = signals[0]
    assert s["signal_type"] == "inspected_result"
    assert s["confidence"] == "medium"
    assert s["run_id"] == _RUN
    assert s["shard_id"] == _SHARD


def test_last_full_after_run_produces_inspected_result():
    events = [_response_event(), _command_event("/last full")]
    signals = infer_signals_from_session(events)
    assert len(signals) == 1
    assert signals[0]["signal_type"] == "inspected_result"


# ---------------------------------------------------------------------------
# continued_session
# ---------------------------------------------------------------------------


def test_user_message_after_run_produces_continued_session():
    events = [_response_event(), _user_message_event("write a unit test")]
    signals = infer_signals_from_session(events)
    assert len(signals) == 1
    s = signals[0]
    assert s["signal_type"] == "continued_session"
    assert s["confidence"] == "low"
    assert s["run_id"] == _RUN


# ---------------------------------------------------------------------------
# retry_requested
# ---------------------------------------------------------------------------


def test_retry_words_produce_retry_requested():
    events = [_response_event(), _user_message_event("try again please")]
    signals = infer_signals_from_session(events)
    assert len(signals) == 1
    s = signals[0]
    assert s["signal_type"] == "retry_requested"
    assert s["confidence"] == "medium"


def test_each_retry_word_triggers_signal():
    for word in RETRY_WORDS:
        events = [_response_event(), _user_message_event(f"please {word} now")]
        signals = infer_signals_from_session(events)
        assert len(signals) == 1, f"expected signal for word {word!r}"
        assert signals[0]["signal_type"] == "retry_requested"


# ---------------------------------------------------------------------------
# No-signal cases
# ---------------------------------------------------------------------------


def test_no_signals_for_command_only_session():
    events = [
        {"event_type": "session_started", "session_id": _SESSION, "metadata": {}},
        _command_event("/help"),
        _command_event("/last"),
    ]
    signals = infer_signals_from_session(events)
    assert signals == []


def test_user_message_before_any_run_produces_no_signal():
    events = [
        _user_message_event("do something"),
        _command_event("/help"),
    ]
    signals = infer_signals_from_session(events)
    assert signals == []


def test_last_more_before_any_run_produces_no_signal():
    signals = infer_signals_from_session([_command_event("/last more")])
    assert signals == []


# ---------------------------------------------------------------------------
# Schema completeness
# ---------------------------------------------------------------------------


def test_signal_schema_fields_present():
    events = [
        _feedback_event("accepted"),
        _response_event(),
        _command_event("/last more"),
        _user_message_event("continue"),
        _user_message_event("try again"),
    ]
    signals = infer_signals_from_session(events)
    assert len(signals) == 4
    for s in signals:
        missing = _REQUIRED_FIELDS - s.keys()
        assert not missing, f"signal missing fields: {missing}"
        assert s["schema_version"] == 1
        assert s["source"] == "rules_v1"
        assert s["signal_id"]  # non-empty uuid string


# ---------------------------------------------------------------------------
# File I/O via run_inference
# ---------------------------------------------------------------------------


def test_missing_session_events_file_returns_empty(tmp_path: Path):
    signals = run_inference(tmp_path)
    assert signals == []


def test_run_inference_writes_signals_file(tmp_path: Path):
    openshard_dir = tmp_path / ".openshard"
    openshard_dir.mkdir()
    events_file = openshard_dir / "session_events.jsonl"
    events_file.write_text(
        json.dumps(_response_event()) + "\n"
        + json.dumps(_user_message_event("continue")) + "\n",
        encoding="utf-8",
    )

    signals = run_inference(tmp_path)
    assert len(signals) == 1
    assert signals[0]["signal_type"] == "continued_session"

    signals_file = openshard_dir / "session_signals.jsonl"
    assert signals_file.exists()
    lines = [ln for ln in signals_file.read_text(encoding="utf-8").splitlines() if ln.strip()]
    assert len(lines) == 1
    assert json.loads(lines[0])["signal_type"] == "continued_session"


def test_run_inference_overwrites_not_appends(tmp_path: Path):
    openshard_dir = tmp_path / ".openshard"
    openshard_dir.mkdir()
    events_file = openshard_dir / "session_events.jsonl"
    events_file.write_text(
        json.dumps(_response_event()) + "\n"
        + json.dumps(_user_message_event("continue")) + "\n",
        encoding="utf-8",
    )

    run_inference(tmp_path)
    run_inference(tmp_path)

    signals_file = openshard_dir / "session_signals.jsonl"
    lines = [ln for ln in signals_file.read_text(encoding="utf-8").splitlines() if ln.strip()]
    assert len(lines) == 1, "second call must overwrite, not append"


def test_run_inference_produces_valid_jsonl(tmp_path: Path):
    openshard_dir = tmp_path / ".openshard"
    openshard_dir.mkdir()
    events_file = openshard_dir / "session_events.jsonl"
    events_file.write_text(
        json.dumps(_feedback_event("accepted")) + "\n",
        encoding="utf-8",
    )

    run_inference(tmp_path)
    signals_file = openshard_dir / "session_signals.jsonl"
    for line in signals_file.read_text(encoding="utf-8").splitlines():
        if line.strip():
            obj = json.loads(line)
            assert obj["signal_type"] == "accepted_explicit"


# ---------------------------------------------------------------------------
# _has_retry_words helper
# ---------------------------------------------------------------------------


def test_has_retry_words_case_insensitive():
    assert _has_retry_words("Please RETRY this")
    assert _has_retry_words("Can you Try Again")
    assert not _has_retry_words("looks good to me")


# ---------------------------------------------------------------------------
# Fix 5: Session boundary resets last_run_response
# ---------------------------------------------------------------------------

_SESSION_A = "sess-aaa"
_SESSION_B = "sess-bbb"


def _response_event_for(session: str, run_id: str, shard_id: str) -> dict:
    return {
        "event_type": "openshard_response",
        "session_id": session,
        "run_id": run_id,
        "shard_id": shard_id,
        "summary": "",
        "metadata": {},
    }


def _command_event_for(session: str, command: str) -> dict:
    return {
        "event_type": "command_invoked",
        "session_id": session,
        "run_id": None,
        "shard_id": None,
        "command": command,
        "summary": "",
        "metadata": {},
    }


def test_old_session_run_does_not_produce_inspected_result_in_new_session():
    """Old session run must not link to a /last more in a new session."""
    events = [
        _response_event_for(_SESSION_A, run_id="run-old", shard_id="shard-old"),
        _command_event_for(_SESSION_B, "/last more"),
    ]
    signals = infer_signals_from_session(events)
    inspected = [s for s in signals if s["signal_type"] == "inspected_result"]
    assert inspected == [], (
        "inspected_result must not be created when /last more is in a different session "
        f"from the run response; got signals: {signals}"
    )


def test_session_boundary_resets_new_run_links_correctly():
    """After session boundary reset, a new run in session B links /last more correctly."""
    events = [
        _response_event_for(_SESSION_A, run_id="run-old", shard_id="shard-old"),
        _response_event_for(_SESSION_B, run_id="run-new", shard_id="shard-new"),
        _command_event_for(_SESSION_B, "/last more"),
    ]
    signals = infer_signals_from_session(events)
    inspected = [s for s in signals if s["signal_type"] == "inspected_result"]
    assert len(inspected) == 1, f"Expected 1 inspected_result signal, got: {signals}"
    assert inspected[0]["run_id"] == "run-new", (
        f"inspected_result must link to run-new, got: {inspected[0]['run_id']!r}"
    )
    assert inspected[0]["shard_id"] == "shard-new", (
        f"inspected_result must link to shard-new, got: {inspected[0]['shard_id']!r}"
    )


def test_same_session_linkage_unchanged():
    """Within a single session, /last more still links to the session's run (regression guard)."""
    events = [
        _response_event_for(_SESSION_A, run_id="run-x", shard_id="shard-x"),
        _command_event_for(_SESSION_A, "/last more"),
    ]
    signals = infer_signals_from_session(events)
    inspected = [s for s in signals if s["signal_type"] == "inspected_result"]
    assert len(inspected) == 1, f"Expected 1 inspected_result signal, got: {signals}"
    assert inspected[0]["run_id"] == "run-x"
    assert inspected[0]["shard_id"] == "shard-x"
