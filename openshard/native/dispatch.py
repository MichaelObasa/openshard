from __future__ import annotations

from dataclasses import dataclass, field

from openshard.routing.engine import MODEL_CHEAP, MODEL_MAIN, MODEL_STRONG


@dataclass
class TierCandidate:
    """Tier-to-model resolution entry: preferred model, ordered fallbacks, provider, reason."""

    preferred: str
    fallbacks: list[str] = field(default_factory=list)
    provider: str = ""
    reason: str = ""


# Single source of truth for tier → model candidate information.
_TIER_CANDIDATES: dict[str, TierCandidate] = {
    "frontier-reasoning-model": TierCandidate(
        preferred=MODEL_STRONG,
        fallbacks=[MODEL_MAIN],
        provider="anthropic",
        reason="frontier reasoning for planning and security-sensitive tasks",
    ),
    "balanced-coding-model": TierCandidate(
        preferred=MODEL_MAIN,
        fallbacks=[MODEL_CHEAP],
        provider="z-ai",
        reason="balanced coding for standard execution",
    ),
    "low-cost-coding-model": TierCandidate(
        preferred=MODEL_CHEAP,
        fallbacks=[MODEL_MAIN],
        provider="deepseek",
        reason="low-cost model for boilerplate and simple tasks",
    ),
    "independent-validator-model": TierCandidate(
        preferred=MODEL_STRONG,
        fallbacks=[MODEL_MAIN],
        provider="anthropic",
        reason="independent model for cross-check validation",
    ),
}

# Backward-compatible flat map derived from _TIER_CANDIDATES.
_TIER_MODEL_MAP: dict[str, str] = {
    tier: tc.preferred for tier, tc in _TIER_CANDIDATES.items()
}

# Safe fallback tier when an unknown or fully-blocked tier is requested.
_UNKNOWN_TIER_FALLBACK = "balanced-coding-model"

# Role → default tier (single source of truth for role-tier binding).
_ROLE_DEFAULT_TIER: dict[str, str] = {
    "planner":   "frontier-reasoning-model",
    "executor":  "balanced-coding-model",
    "validator": "independent-validator-model",
}

# Role → ordered list of valid tiers (highest to lowest preference).
_ROLE_VALID_TIERS: dict[str, list[str]] = {
    "planner":   ["frontier-reasoning-model", "balanced-coding-model"],
    "executor":  ["frontier-reasoning-model", "balanced-coding-model", "low-cost-coding-model"],
    "validator": ["independent-validator-model", "frontier-reasoning-model", "balanced-coding-model"],
}

_CATEGORY_TIER_MAP: dict[str, str] = {
    "security":    "frontier-reasoning-model",
    "complex":     "frontier-reasoning-model",
    "standard":    "balanced-coding-model",
    "visual":      "balanced-coding-model",
    "boilerplate": "low-cost-coding-model",
}


def get_tier_candidate(tier_name: str) -> TierCandidate | None:
    """Return the TierCandidate for *tier_name*, or None if unknown."""
    return _TIER_CANDIDATES.get(tier_name)


def resolve_tier(
    tier_name: str | None,
    blocked_models: set[str] | None = None,
) -> tuple[str | None, bool, str]:
    """Resolve a tier name to a model ID, respecting blocked models.

    Returns (model_id, fallback_used, fallback_reason).
    fallback_used=True when tier_name is unknown/None/empty, or when the
    preferred model is blocked and a fallback was substituted.
    """
    if not tier_name:
        return (None, True, "tier name is None or empty")

    tc = _TIER_CANDIDATES.get(tier_name)
    if tc is None:
        # Unknown tier: safe fallback to balanced-coding-model.
        fallback_tc = _TIER_CANDIDATES[_UNKNOWN_TIER_FALLBACK]
        return (
            fallback_tc.preferred,
            True,
            f"unknown tier {tier_name!r}; using {_UNKNOWN_TIER_FALLBACK}",
        )

    blocked = blocked_models or set()

    if tc.preferred not in blocked:
        return (tc.preferred, False, "")

    # Preferred is blocked; try declared fallbacks in order.
    for fb_model in tc.fallbacks:
        if fb_model not in blocked:
            return (
                fb_model,
                True,
                f"preferred model blocked; using fallback for tier {tier_name!r}",
            )

    # All declared fallbacks also blocked; try the safe fallback tier.
    if tier_name != _UNKNOWN_TIER_FALLBACK:
        fallback_tc = _TIER_CANDIDATES[_UNKNOWN_TIER_FALLBACK]
        if fallback_tc.preferred not in blocked:
            return (
                fallback_tc.preferred,
                True,
                f"all models for tier {tier_name!r} blocked; fell back to {_UNKNOWN_TIER_FALLBACK}",
            )

    return (None, True, f"all models for tier {tier_name!r} blocked, no safe fallback available")


def resolve_role(
    role: str | None,
    blocked_models: set[str] | None = None,
) -> tuple[str | None, str, bool, str]:
    """Resolve a role name to (model_id, tier_name, fallback_used, fallback_reason).

    Uses the role's default tier. Falls back through the role's valid-tier list
    if the preferred model is blocked.
    """
    if not role:
        return (None, "", True, "role is None or empty")

    default_tier = _ROLE_DEFAULT_TIER.get(role)
    if default_tier is None:
        return (None, "", True, f"unknown role {role!r}")

    model, fallback_used, reason = resolve_tier(default_tier, blocked_models)
    if not fallback_used:
        return (model, default_tier, False, "")

    # Default tier didn't yield a clean result; try other valid tiers for this role.
    for alt_tier in _ROLE_VALID_TIERS.get(role, []):
        if alt_tier == default_tier:
            continue
        alt_model, alt_fb, _alt_reason = resolve_tier(alt_tier, blocked_models)
        if not alt_fb:
            return (
                alt_model,
                alt_tier,
                True,
                f"default tier {default_tier!r} unavailable; using {alt_tier}",
            )

    # Return whatever resolve_tier gave us (may be a safe fallback).
    return (model, default_tier, True, reason)


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
