from __future__ import annotations

from collections import Counter

MIN_EVIDENCE = 3
_ADJ_MIN = -0.5
_ADJ_MAX = 0.3

_ACTION_SIGNALS: dict[str, float] = {
    "accepted": 0.20,
    "partially-accepted": 0.05,
    "rejected": -0.25,
    "retried": -0.20,
    "unknown": 0.0,
}

_SERIOUS_REASONS = frozenset(
    {"hallucinated", "wrong-file", "wrong-scope", "missed-requirement", "failed-tests"}
)
_MILD_REASONS = frozenset(
    {"unclear-output", "bad-style", "unsafe-command", "manual-edit"}
)
# "too-expensive" / "too-slow" are cost/speed signals only — not quality penalties


def _reason_signal(reason: str | None) -> float:
    if reason is None:
        return 0.0
    if reason == "hallucinated":
        return -0.30
    if reason in _SERIOUS_REASONS:
        return -0.25
    if reason in _MILD_REASONS:
        return -0.10
    return 0.0


def _feedback_signal(feedback: dict) -> float:
    """Compute a routing score signal from a single feedback record.

    Free-text note is never read.
    """
    action = feedback.get("action")
    reason = feedback.get("correction_reason")

    if action == "edited":
        if reason in _SERIOUS_REASONS:
            return -0.20
        return -0.10

    base = _ACTION_SIGNALS.get(action, 0.0)
    rsig = _reason_signal(reason)
    return min(base, rsig) if rsig < 0 else base


def compute_feedback_adjustments(runs: list[dict]) -> dict[str, float]:
    """Return per-model score adjustments derived from developer feedback records.

    Models with fewer than MIN_EVIDENCE feedback records are omitted (insufficient signal).
    Adjustments are averaged across all feedback records and clamped to [_ADJ_MIN, _ADJ_MAX].
    Run history is never mutated.
    """
    signals_by_model: dict[str, list[float]] = {}
    for run in runs:
        model = run.get("execution_model") or "unknown"
        fb = run.get("feedback")
        if not isinstance(fb, dict):
            continue
        signals_by_model.setdefault(model, []).append(_feedback_signal(fb))

    result: dict[str, float] = {}
    for model, sigs in signals_by_model.items():
        if len(sigs) < MIN_EVIDENCE:
            continue
        avg = sum(sigs) / len(sigs)
        result[model] = max(_ADJ_MIN, min(_ADJ_MAX, avg))
    return result


def compute_feedback_adjustment_reasons(runs: list[dict]) -> dict[str, str]:
    """Return a short human-readable reason string for each model with a nonzero adjustment."""
    adjustments = compute_feedback_adjustments(runs)
    reasons: dict[str, str] = {}
    for model, adj in adjustments.items():
        if adj == 0.0:
            continue
        fb_runs = [
            r for r in runs
            if r.get("execution_model") == model and isinstance(r.get("feedback"), dict)
        ]
        actions = Counter(
            r["feedback"].get("action")
            for r in fb_runs
            if r["feedback"].get("action")
        )
        correction_reasons = Counter(
            r["feedback"].get("correction_reason")
            for r in fb_runs
            if r["feedback"].get("correction_reason")
        )
        parts: list[str] = []
        if actions["accepted"] and adj > 0:
            parts.append(f"{actions['accepted']} accepted")
        if actions["rejected"]:
            parts.append(f"{actions['rejected']} rejected")
        if actions["retried"]:
            parts.append(f"{actions['retried']} retried")
        top = correction_reasons.most_common(1)
        if top:
            parts.append(top[0][0])
        reasons[model] = "feedback: " + (", ".join(parts) if parts else "signal")
    return reasons
