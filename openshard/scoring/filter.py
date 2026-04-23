from __future__ import annotations

from openshard.providers.manager import InventoryEntry
from openshard.scoring.requirements import TaskRequirements

_NONCODING_SUBSTRINGS = (
    "embed",
    "tts",
    "whisper",
    "dall-e",
    "stable-diffusion",
    "moderation",
    "transcri",
    "/image-",
    "image",
    "vision",
)


def prefilter_coding(entries: list[InventoryEntry]) -> list[InventoryEntry]:
    """Drop models that are clearly non-coding (embeddings, TTS, image gen, etc.)."""
    result = []
    for entry in entries:
        mid = entry.model.id.lower()
        if any(s in mid for s in _NONCODING_SUBSTRINGS):
            continue
        if mid.endswith("/image"):
            continue
        result.append(entry)
    return result


def _parse_cost(pricing: dict) -> float | None:
    raw = pricing.get("prompt", "")
    if not raw:
        return None
    try:
        val = float(str(raw).lstrip("$"))
    except (ValueError, TypeError):
        return None
    if val <= 0:
        return None
    # Provider APIs return per-token prices; convert to per-million-token.
    return val * 1_000_000


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
            if cost is None or cost > requirements.preferred_max_cost_per_m:
                continue

        result.append(entry)
    return result
