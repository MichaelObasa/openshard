from __future__ import annotations

import json
from pathlib import Path

_KNOWN_SIGNAL_TYPES = frozenset({
    "retry_requested",
    "partial_explicit",
    "rejected_explicit",
    "accepted_explicit",
    "inspected_result",
    "continued_session",
})

_NEGATIVE_SIGNAL_TYPES = frozenset({
    "retry_requested",
    "partial_explicit",
    "rejected_explicit",
})


def _load_recent_session_signals(path: Path, limit: int = 25) -> list[dict]:
    """Return the last `limit` known-type signal dicts from a JSONL file. Never raises."""
    try:
        if not path.exists():
            return []
        valid: list[dict] = []
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(obj, dict) and obj.get("signal_type") in _KNOWN_SIGNAL_TYPES:
                    valid.append(obj)
        return valid[-limit:]
    except Exception:
        return []


def build_feedback_routing_advisory(signals: list[dict]) -> dict | None:
    """Return a feedback routing advisory dict from a list of session signals, or None.

    Only considers the last 25 known-type signals passed in. Reads no files.
    Returns None when no negative signals are present.
    """
    counts: dict[str, int] = {
        "retry_requested": 0,
        "partial_explicit": 0,
        "rejected_explicit": 0,
    }
    for s in signals:
        t = s.get("signal_type")
        if t in counts:
            counts[t] += 1

    rejected = counts["rejected_explicit"]
    partial = counts["partial_explicit"]
    retry = counts["retry_requested"]

    if rejected > 0 or partial > 0:
        confidence = "medium"
        parts: list[str] = []
        if partial > 0:
            parts.append("partial feedback")
        if rejected > 0:
            parts.append("rejected feedback")
        reason = f"Recent local session signals included {' and '.join(parts)}."
    elif retry > 0:
        confidence = "low"
        reason = "Recent local session signals included retry requests."
    else:
        return None

    return {
        "version": "rules_v1",
        "advisory_only": True,
        "recommendation": "consider_stronger_review",
        "confidence": confidence,
        "reason": reason,
        "signals_considered": counts,
        "signals_window": {
            "source": "session_signals.jsonl",
            "max_recent_signals": 25,
            "signals_read": len(signals),
            "signals_used": sum(counts.values()),
        },
    }
