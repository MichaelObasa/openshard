from __future__ import annotations

from openshard.providers.manager import InventoryEntry
from openshard.scoring.filter import _parse_cost, filter_inventory
from openshard.scoring.requirements import TaskRequirements


def score_model(entry: InventoryEntry, requirements: TaskRequirements) -> float:
    """Return a score for a candidate model. Higher is better."""
    score = 10.0
    m = entry.model

    if requirements.needs_vision and m.supports_vision:
        score += 2.0
    if requirements.needs_tools and m.supports_tools:
        score += 2.0

    if m.context_window is not None:
        if m.context_window >= 100_000:
            score += 1.0
        if m.context_window >= 200_000:
            score += 1.0

    if requirements.security_sensitive and "claude" in m.id.lower():
        score += 3.0

    cost = _parse_cost(m.pricing)
    if cost is not None:
        if cost <= 1.0:
            score += 2.0
        if cost <= 0.25:
            score += 1.0

    return score


def select_candidate(
    entries: list[InventoryEntry],
    requirements: TaskRequirements,
) -> InventoryEntry | None:
    """Filter entries by requirements, score survivors, return top scorer."""
    candidates = filter_inventory(entries, requirements)
    if not candidates:
        return None
    return max(candidates, key=lambda e: score_model(e, requirements))
