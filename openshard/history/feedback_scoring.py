from __future__ import annotations

from collections import Counter

MIN_EVIDENCE = 3
_ADJ_MIN = -0.5
_ADJ_MAX = 0.3
_COST_REASON_PENALTY = -0.05  # small penalty for cost/speed flags

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
_COST_SPEED_REASONS = frozenset({"too-expensive", "too-slow"})


def _reason_signal(reason: str | None) -> float:
    if reason is None:
        return 0.0
    if reason == "hallucinated":
        return -0.30
    if reason == "failed-tests":
        return -0.25
    if reason in _SERIOUS_REASONS:
        return -0.25
    if reason in _MILD_REASONS:
        return -0.10
    if reason in _COST_SPEED_REASONS:
        return _COST_REASON_PENALTY
    return 0.0


def _feedback_signal(feedback: dict, verification_passed: bool | None = None) -> float:
    """Compute a routing score signal from a single feedback record.

    Free-text note is never read.
    """
    action = feedback.get("action")
    reason = feedback.get("correction_reason")

    # failed-tests: only penalize when verification failed or run was explicitly rejected/retried
    if reason == "failed-tests":
        qualifies = (verification_passed is False) or (action in ("rejected", "retried"))
        if not qualifies:
            reason = None

    if action == "edited":
        if reason in _SERIOUS_REASONS:
            return -0.20
        return -0.10

    base = _ACTION_SIGNALS.get(action, 0.0)  # type: ignore[arg-type]  # action may be None; treated as missing key
    rsig = _reason_signal(reason)
    return min(base, rsig) if rsig < 0 else base


def _get_event_field(event, field: str):
    """Read a field from either a dict or a dataclass-like event object."""
    if isinstance(event, dict):
        return event.get(field)
    return getattr(event, field, None)


def compute_feedback_adjustments(
    runs: list[dict],
    interaction_events: list | None = None,
) -> dict[str, float]:
    """Return per-model score adjustments derived from developer feedback records.

    Models with fewer than MIN_EVIDENCE feedback records are omitted (insufficient signal).
    Adjustments are averaged across all feedback records and clamped to [_ADJ_MIN, _ADJ_MAX].
    Interaction events supplement runs that lack a feedback field (no double-counting).
    Run history is never mutated.
    """
    run_model_map: dict[str, str] = {
        r.get("timestamp", ""): (r.get("execution_model") or "unknown")
        for r in runs
        if isinstance(r, dict)
    }
    run_ids_with_feedback: set[str] = {
        r.get("timestamp", "")
        for r in runs
        if isinstance(r, dict) and isinstance(r.get("feedback"), dict)
    }

    signals_by_model: dict[str, list[float]] = {}
    for run in runs:
        model = run.get("execution_model") or "unknown"
        fb = run.get("feedback")
        if not isinstance(fb, dict):
            continue
        vp = run.get("verification_passed")
        signals_by_model.setdefault(model, []).append(_feedback_signal(fb, verification_passed=vp))

    if interaction_events:
        for event in interaction_events:
            try:
                run_id = _get_event_field(event, "run_id") or ""
                if run_id in run_ids_with_feedback:
                    continue
                model = run_model_map.get(run_id)
                if not model:
                    continue
                accepted = _get_event_field(event, "accepted")
                reason = _get_event_field(event, "correction_reason")
                if accepted is True:
                    sig = 0.15
                elif accepted is False:
                    sig = -0.20
                else:
                    sig = 0.0
                if reason:
                    rsig = _reason_signal(reason)
                    if rsig < 0:
                        sig = min(sig, rsig)
                signals_by_model.setdefault(model, []).append(sig)
            except Exception:
                continue

    result: dict[str, float] = {}
    for model, sigs in signals_by_model.items():
        if len(sigs) < MIN_EVIDENCE:
            continue
        avg = sum(sigs) / len(sigs)
        result[model] = max(_ADJ_MIN, min(_ADJ_MAX, avg))
    return result


def compute_feedback_adjustment_reasons(
    runs: list[dict],
    interaction_events: list | None = None,
) -> dict[str, str]:
    """Return a short human-readable reason string for each model with a nonzero adjustment."""
    adjustments = compute_feedback_adjustments(runs, interaction_events=interaction_events)
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
