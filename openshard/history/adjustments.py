from __future__ import annotations

from openshard.history.metrics import compute_model_stats

MIN_MODEL_RUNS = 5
MIN_VERIFIED_RUNS = 3

_PASS_RATE_BONUS_THRESHOLD = 0.8
_PASS_RATE_PENALTY_THRESHOLD = 0.3
_PASS_RATE_BONUS = 1.0
_PASS_RATE_PENALTY = -1.0

_RETRY_RATE_PENALTY_THRESHOLD = 0.4
_RETRY_PENALTY = -1.0

_ADJ_MIN = -2.0
_ADJ_MAX = 1.0


def compute_history_adjustments(runs: list[dict]) -> dict[str, float]:
    """Return per-model score adjustments derived from run history.

    Returns an empty dict when history is absent.  Returns 0.0 for models
    with fewer than MIN_MODEL_RUNS total runs (too noisy to trust).
    """
    if not runs:
        return {}

    stats = compute_model_stats(runs)
    adjustments: dict[str, float] = {}

    for model, s in stats.items():
        if s["runs_count"] < MIN_MODEL_RUNS:
            adjustments[model] = 0.0
            continue

        adj = 0.0

        pass_rate = s.get("verification_pass_rate")
        if pass_rate is not None:
            verified_count = _count_verified_runs(runs, model)
            if verified_count >= MIN_VERIFIED_RUNS:
                if pass_rate >= _PASS_RATE_BONUS_THRESHOLD:
                    adj += _PASS_RATE_BONUS
                elif pass_rate <= _PASS_RATE_PENALTY_THRESHOLD:
                    adj += _PASS_RATE_PENALTY

        retry_rate = s.get("retry_rate", 0.0)
        if retry_rate >= _RETRY_RATE_PENALTY_THRESHOLD:
            adj += _RETRY_PENALTY

        adjustments[model] = max(_ADJ_MIN, min(_ADJ_MAX, adj))

    return adjustments


def compute_history_adjustment_reasons(runs: list[dict]) -> dict[str, str]:
    """Return a short human-readable reason string for each model with a non-zero adjustment.

    Models with insufficient history (< MIN_MODEL_RUNS) are omitted entirely.
    """
    if not runs:
        return {}

    stats = compute_model_stats(runs)
    reasons: dict[str, str] = {}

    for model, s in stats.items():
        if s["runs_count"] < MIN_MODEL_RUNS:
            continue

        parts: list[str] = []

        pass_rate = s.get("verification_pass_rate")
        if pass_rate is not None:
            verified_count = _count_verified_runs(runs, model)
            if verified_count >= MIN_VERIFIED_RUNS:
                if pass_rate >= _PASS_RATE_BONUS_THRESHOLD:
                    parts.append("high pass rate")
                elif pass_rate <= _PASS_RATE_PENALTY_THRESHOLD:
                    parts.append("low pass rate")

        retry_rate = s.get("retry_rate", 0.0)
        if retry_rate >= _RETRY_RATE_PENALTY_THRESHOLD:
            parts.append("high retry rate")

        if parts:
            reasons[model] = "; ".join(parts)

    return reasons


def _count_verified_runs(runs: list[dict], model: str) -> int:
    """Count runs for a model where verification_passed is explicitly True or False."""
    return sum(
        1
        for r in runs
        if r.get("execution_model") == model and r.get("verification_passed") is not None
    )
