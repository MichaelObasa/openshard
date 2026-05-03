from __future__ import annotations

from openshard.evals.stats import CategoryStats, EvalStats

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


MIN_CATEGORY_EVAL_RUNS = 2

_CAT_PASS_RATE_BONUS = 0.3
_CAT_PASS_RATE_PENALTY = -0.3
_CAT_UNSAFE_PENALTY = -0.2
_CAT_COST_BONUS = 0.1
_CAT_COST_BONUS_THRESHOLD = 0.001


def compute_category_eval_adjustments(
    category_stats: list[CategoryStats],
    category: str,
) -> dict[str, float]:
    """Return per-model score overlays using category-filtered eval evidence.

    Only applied when --eval-scoring is enabled. Falls back to no adjustment
    when fewer than MIN_CATEGORY_EVAL_RUNS runs exist for the category.
    """
    adjustments: dict[str, float] = {}
    for s in category_stats:
        if s.category != category:
            continue
        if s.run_count < MIN_CATEGORY_EVAL_RUNS:
            continue

        adj = 0.0

        if s.pass_rate >= _PASS_RATE_BONUS_THRESHOLD:
            adj += _CAT_PASS_RATE_BONUS
        elif s.pass_rate < _PASS_RATE_PENALTY_THRESHOLD:
            adj += _CAT_PASS_RATE_PENALTY

        if s.unsafe_file_count > 0:
            adj += _CAT_UNSAFE_PENALTY

        if (
            s.pass_rate >= _PASS_RATE_BONUS_THRESHOLD
            and s.cost_per_pass is not None
            and s.cost_per_pass <= _CAT_COST_BONUS_THRESHOLD
        ):
            adj += _CAT_COST_BONUS

        adj = max(_ADJ_MIN, min(_ADJ_MAX, adj))
        if adj != 0.0:
            adjustments[s.model] = adjustments.get(s.model, 0.0) + adj

    return adjustments


def compute_category_eval_adjustment_reasons(
    category_stats: list[CategoryStats],
    category: str,
) -> dict[str, str]:
    """Return human-readable reasons for category-specific eval adjustments."""
    reasons: dict[str, str] = {}
    for s in category_stats:
        if s.category != category:
            continue
        if s.run_count < MIN_CATEGORY_EVAL_RUNS:
            continue

        parts: list[str] = []

        if s.pass_rate >= _PASS_RATE_BONUS_THRESHOLD:
            parts.append("high category pass rate")
        elif s.pass_rate < _PASS_RATE_PENALTY_THRESHOLD:
            parts.append("low category pass rate")

        if s.unsafe_file_count > 0:
            parts.append("unsafe files in evals")

        if (
            s.pass_rate >= _PASS_RATE_BONUS_THRESHOLD
            and s.cost_per_pass is not None
            and s.cost_per_pass <= _CAT_COST_BONUS_THRESHOLD
        ):
            parts.append("low cost per pass")

        if parts:
            existing = reasons.get(s.model)
            reasons[s.model] = existing + "; " + "; ".join(parts) if existing else "; ".join(parts)

    return reasons
