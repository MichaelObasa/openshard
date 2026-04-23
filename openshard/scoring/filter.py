from __future__ import annotations

from openshard.providers.manager import InventoryEntry
from openshard.scoring.requirements import TaskRequirements


def _parse_cost(pricing: dict) -> float | None:
    raw = pricing.get("prompt", "")
    if not raw:
        return None
    try:
        return float(str(raw).lstrip("$"))
    except (ValueError, TypeError):
        return None


def filter_inventory(
    entries: list[InventoryEntry],
    requirements: TaskRequirements,
) -> list[InventoryEntry]:
    """Return entries that satisfy all hard requirements."""
    result = []
    for entry in entries:
        m = entry.model

        if requirements.needs_vision and not m.supports_vision:
            continue
        if requirements.needs_tools and not m.supports_tools:
            continue
        if (
            requirements.min_context_window is not None
            and m.context_window is not None
            and m.context_window < requirements.min_context_window
        ):
            continue
        if requirements.preferred_max_cost_per_m is not None:
            cost = _parse_cost(m.pricing)
            if cost is not None and cost > requirements.preferred_max_cost_per_m:
                continue

        result.append(entry)
    return result
