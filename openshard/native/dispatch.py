from __future__ import annotations

from openshard.routing.engine import MODEL_CHEAP, MODEL_MAIN, MODEL_STRONG

_TIER_MODEL_MAP: dict[str, str] = {
    "frontier-reasoning-model":    MODEL_STRONG,
    "balanced-coding-model":       MODEL_MAIN,
    "low-cost-coding-model":       MODEL_CHEAP,
    "independent-validator-model": MODEL_STRONG,
}

_CATEGORY_TIER_MAP: dict[str, str] = {
    "security":    "frontier-reasoning-model",
    "complex":     "frontier-reasoning-model",
    "standard":    "balanced-coding-model",
    "visual":      "balanced-coding-model",
    "boilerplate": "low-cost-coding-model",
}


def resolve_tier(tier_name: str | None) -> tuple[str | None, bool, str]:
    """Resolve a tier name to a model ID.

    Returns (model_id, fallback_used, fallback_reason).
    fallback_used=True when tier_name is unknown, None, or empty.
    """
    if not tier_name:
        return (None, True, "tier name is None or empty")
    model = _TIER_MODEL_MAP.get(tier_name)
    if model is None:
        return (None, True, f"unknown tier {tier_name!r}")
    return (model, False, "")


def resolve_tier_for_category(category: str | None) -> tuple[str | None, str, bool, str]:
    """Map a routing category to a tier then resolve to a model ID.

    Returns (model_id, tier_name, fallback_used, fallback_reason).
    fallback_used=True for unknown categories and the visual approximation.
    """
    if not category:
        return (None, "", True, "category is None or empty")
    tier = _CATEGORY_TIER_MAP.get(category)
    if tier is None:
        return (None, "", True, f"unknown category {category!r}")
    model_id, fb, reason = resolve_tier(tier)
    if category == "visual":
        return (model_id, tier, True, "visual approximated to balanced-coding-model")
    return (model_id, tier, fb, reason)
