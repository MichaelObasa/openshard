from __future__ import annotations

from openshard.providers.openrouter import compute_cost

BASELINE_MODELS: list[tuple[str, str]] = [
    ("GPT-5.5",    "openai/gpt-5.5"),
    ("Sonnet 4.6", "anthropic/claude-sonnet-4.6"),
]


def _multiplier_suffix(actual_cost: float, baseline_cost: float) -> str:
    ratio = baseline_cost / actual_cost
    if ratio >= 10:
        return f" (~{round(ratio)}x higher)"
    return f" (~{ratio:.1f}x higher)"


def format_baseline_line(
    prompt_tokens: int,
    completion_tokens: int,
    actual_cost: float | None = None,
    models: list[tuple[str, str]] | None = None,
) -> str | None:
    """Return a formatted baseline estimate line, or None if it should be omitted.

    Omitted when tokens are both zero/falsy, or no baseline model can be priced.
    Adds a (~Nx higher) suffix per entry when actual_cost > 0.
    """
    if not prompt_tokens and not completion_tokens:
        return None
    if models is None:
        models = BASELINE_MODELS
    parts: list[str] = []
    for label, model_id in models:
        cost = compute_cost(model_id, prompt_tokens, completion_tokens)
        if cost is None:
            continue
        entry = f"{label} ${cost:.3f}"
        if actual_cost and actual_cost > 0:
            entry += _multiplier_suffix(actual_cost, cost)
        parts.append(entry)
    if not parts:
        return None
    return "Baseline estimate: " + ", ".join(parts)
