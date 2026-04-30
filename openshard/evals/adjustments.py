from __future__ import annotations

from openshard.evals.stats import EvalStats

MIN_EVAL_RUNS = 3

_PASS_RATE_BONUS_THRESHOLD = 0.80
_PASS_RATE_PENALTY_THRESHOLD = 0.50
_PASS_RATE_BONUS = 0.5
_PASS_RATE_PENALTY = -0.5
_UNSAFE_PENALTY = -0.3
_TOKEN_BONUS = 0.2
_TOKEN_BONUS_AVG_THRESHOLD = 5000

_ADJ_MIN = -2.0
_ADJ_MAX = 1.0


def compute_eval_adjustments(stats: list[EvalStats]) -> dict[str, float]:
    """Return per-model score adjustments derived from eval stats.

    Only returns models with nonzero adjustments and at least MIN_EVAL_RUNS total runs.
    """
    models: dict[str, list[EvalStats]] = {}
    for s in stats:
        models.setdefault(s.model, []).append(s)

    adjustments: dict[str, float] = {}
    for model, rows in models.items():
        total_runs = sum(r.run_count for r in rows)
        if total_runs < MIN_EVAL_RUNS:
            continue

        total_passes = sum(r.pass_count for r in rows)
        pass_rate = total_passes / total_runs
        total_unsafe = sum(r.unsafe_file_count for r in rows)

        adj = 0.0

        if pass_rate >= _PASS_RATE_BONUS_THRESHOLD:
            adj += _PASS_RATE_BONUS
        elif pass_rate < _PASS_RATE_PENALTY_THRESHOLD:
            adj += _PASS_RATE_PENALTY

        if total_unsafe > 0:
            adj += _UNSAFE_PENALTY

        token_values = [r.avg_total_tokens for r in rows if r.avg_total_tokens is not None]
        if token_values and pass_rate >= _PASS_RATE_BONUS_THRESHOLD:
            avg_tokens = sum(token_values) / len(token_values)
            if avg_tokens < _TOKEN_BONUS_AVG_THRESHOLD:
                adj += _TOKEN_BONUS

        adj = max(_ADJ_MIN, min(_ADJ_MAX, adj))
        if adj != 0.0:
            adjustments[model] = adj

    return adjustments


def compute_eval_adjustment_reasons(stats: list[EvalStats]) -> dict[str, str]:
    """Return human-readable reason strings for models with nonzero eval adjustments."""
    models: dict[str, list[EvalStats]] = {}
    for s in stats:
        models.setdefault(s.model, []).append(s)

    reasons: dict[str, str] = {}
    for model, rows in models.items():
        total_runs = sum(r.run_count for r in rows)
        if total_runs < MIN_EVAL_RUNS:
            continue

        total_passes = sum(r.pass_count for r in rows)
        pass_rate = total_passes / total_runs
        total_unsafe = sum(r.unsafe_file_count for r in rows)

        parts: list[str] = []

        if pass_rate >= _PASS_RATE_BONUS_THRESHOLD:
            parts.append("high eval pass rate")
        elif pass_rate < _PASS_RATE_PENALTY_THRESHOLD:
            parts.append("low eval pass rate")

        if total_unsafe > 0:
            parts.append("unsafe files in evals")

        token_values = [r.avg_total_tokens for r in rows if r.avg_total_tokens is not None]
        if token_values and pass_rate >= _PASS_RATE_BONUS_THRESHOLD:
            avg_tokens = sum(token_values) / len(token_values)
            if avg_tokens < _TOKEN_BONUS_AVG_THRESHOLD:
                parts.append("low token usage")

        if parts:
            reasons[model] = "; ".join(parts)

    return reasons
