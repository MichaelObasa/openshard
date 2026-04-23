from __future__ import annotations

from dataclasses import dataclass

from openshard.providers.manager import InventoryEntry
from openshard.scoring.filter import _parse_cost, filter_inventory
from openshard.scoring.requirements import TaskRequirements


@dataclass
class ScoredRoutingResult:
    category: str
    requirements: TaskRequirements
    candidate_count: int
    selected_model: str | None
    selected_provider: str | None
    used_fallback: bool


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

    # Penalise rolling alias IDs (e.g. ~anthropic/claude-opus-latest).
    # Explicit versioned IDs are preferred for deterministic behaviour.
    if m.id.startswith("~"):
        score -= 1.0

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


def select_with_info(
    entries: list[InventoryEntry],
    requirements: TaskRequirements,
    category: str,
) -> ScoredRoutingResult:
    """Filter and score entries; return a full record of the decision."""
    candidates = filter_inventory(entries, requirements)
    if not candidates:
        return ScoredRoutingResult(
            category=category,
            requirements=requirements,
            candidate_count=0,
            selected_model=None,
            selected_provider=None,
            used_fallback=True,
        )
    winner = max(candidates, key=lambda e: score_model(e, requirements))
    return ScoredRoutingResult(
        category=category,
        requirements=requirements,
        candidate_count=len(candidates),
        selected_model=winner.model.id,
        selected_provider=winner.provider,
        used_fallback=False,
    )
