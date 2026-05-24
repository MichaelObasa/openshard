from __future__ import annotations

import json
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

_SIGNALS_FILE = Path(".openshard") / "session_signals.jsonl"

RETRY_WORDS = frozenset({"retry", "try again", "redo", "again", "fix that"})


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _signal(
    *,
    session_id: str,
    run_id: str | None,
    shard_id: str | None,
    signal_type: str,
    confidence: str,
    reason: str,
    metadata: dict | None = None,
) -> dict:
    return {
        "schema_version": 1,
        "signal_id": str(uuid.uuid4()),
        "session_id": session_id,
        "run_id": run_id,
        "shard_id": shard_id,
        "timestamp": _now_iso(),
        "source": "rules_v1",
        "signal_type": signal_type,
        "confidence": confidence,
        "reason": reason,
        "metadata": metadata or {},
    }


def _has_retry_words(text: str) -> bool:
    low = text.lower()
    return any(word in low for word in RETRY_WORDS)


def infer_signals_from_session(events: list[dict]) -> list[dict]:
    """Derive behavioural signals from a list of session events.

    Pure function — no file I/O. Events must be in chronological order.
    Only infers what is structurally obvious; silence produces no signal.
    """
    signals: list[dict] = []
    # Only updated when openshard_response carries a real run_id or shard_id.
    # Non-run responses (e.g. /last output) must not erase prior run context.
    last_run_response: dict | None = None

    for event in events:
        etype = event.get("event_type", "")
        session_id = event.get("session_id", "")

        if etype == "openshard_response":
            if event.get("run_id") or event.get("shard_id"):
                last_run_response = event

        elif etype == "feedback_recorded":
            outcome = (event.get("metadata") or {}).get("outcome", "")
            _explicit_map = {
                "accepted": ("accepted_explicit", "high"),
                "partial": ("partial_explicit", "high"),
                "rejected": ("rejected_explicit", "high"),
            }
            if outcome in _explicit_map:
                signal_type, confidence = _explicit_map[outcome]
                # Prefer run_id/shard_id on the event itself; fall back to
                # the most recent real run response when feedback events carry nulls.
                run_id = event.get("run_id") or (
                    last_run_response.get("run_id") if last_run_response else None
                )
                shard_id = event.get("shard_id") or (
                    last_run_response.get("shard_id") if last_run_response else None
                )
                signals.append(
                    _signal(
                        session_id=session_id,
                        run_id=run_id,
                        shard_id=shard_id,
                        signal_type=signal_type,
                        confidence=confidence,
                        reason=f"feedback_recorded outcome={outcome}",
                    )
                )

        elif etype == "command_invoked" and last_run_response is not None:
            cmd = event.get("command", "")
            if cmd in {"/last more", "/last full"}:
                signals.append(
                    _signal(
                        session_id=session_id,
                        run_id=last_run_response.get("run_id"),
                        shard_id=last_run_response.get("shard_id"),
                        signal_type="inspected_result",
                        confidence="medium",
                        reason=f"command_invoked {cmd!r} after openshard_response",
                    )
                )

        elif etype == "user_message" and last_run_response is not None:
            summary = event.get("summary", "")
            if _has_retry_words(summary):
                signals.append(
                    _signal(
                        session_id=session_id,
                        run_id=last_run_response.get("run_id"),
                        shard_id=last_run_response.get("shard_id"),
                        signal_type="retry_requested",
                        confidence="medium",
                        reason="user_message contained retry words after openshard_response",
                    )
                )
            else:
                signals.append(
                    _signal(
                        session_id=session_id,
                        run_id=last_run_response.get("run_id"),
                        shard_id=last_run_response.get("shard_id"),
                        signal_type="continued_session",
                        confidence="low",
                        reason="user_message after openshard_response",
                    )
                )

    return signals


def run_inference(base_path: Path | None = None) -> list[dict]:
    """Read session_events.jsonl, infer signals, overwrite session_signals.jsonl.

    Returns the list of inferred signals. If session_events.jsonl is missing
    or unreadable, returns an empty list without raising.
    """
    root = (base_path or Path.cwd()) / ".openshard"
    events_file = root / "session_events.jsonl"
    signals_file = root / "session_signals.jsonl"

    if not events_file.exists():
        return []

    events: list[dict] = []
    try:
        for line in events_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                events.append(json.loads(line))
    except (OSError, json.JSONDecodeError):
        return []

    signals = infer_signals_from_session(events)

    try:
        root.mkdir(parents=True, exist_ok=True)
        with signals_file.open("w", encoding="utf-8") as fh:
            for signal in signals:
                fh.write(json.dumps(signal) + "\n")
    except OSError as exc:
        print(
            f"[session_signals] warning: could not write session_signals.jsonl: {exc}",
            file=sys.stderr,
        )

    return signals
