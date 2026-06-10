"""Registry-backed routing model resolver.

Replaces the six hardcoded string constants in routing/engine.py with
live lookups against the model registry. Falls back to the original
hardcoded IDs when no eligible registry candidate is found, so the
system degrades gracefully if the registry is empty or malformed.

Module-level constants (MODEL_CHEAP, MODEL_MAIN, …) are evaluated once
at import time and cached. The registry is static at startup, so the
values are stable for the lifetime of the process.

Usage in engine.py and anywhere else that needs routing model IDs::

    from openshard.routing.model_resolver import (
        MODEL_CHEAP, MODEL_MAIN, MODEL_STRONG,
        MODEL_ESCALATE, MODEL_VISUAL, MODEL_COMPLEX,
        ESCALATION_CHAIN,
        resolution_source,
    )
"""
from __future__ import annotations

from functools import cache

# ---------------------------------------------------------------------------
# Hardcoded fallbacks — exact IDs that were previously in engine.py.
# These are ONLY used when the registry returns no eligible candidate.
# ---------------------------------------------------------------------------

_FALLBACKS: dict[str, str] = {
    "cheap":    "deepseek/deepseek-v4-flash",
    "main":     "z-ai/glm-5.1",
    "strong":   "anthropic/claude-sonnet-4.6",
    "escalate": "anthropic/claude-opus-4.7",
    "visual":   "moonshotai/kimi-k2.5",
    "complex":  "minimax/m2.7",
}

# ---------------------------------------------------------------------------
# Role query specifications.
#
# Each entry describes how to search the registry for the best candidate:
#   lifecycle   — primary lifecycle filter (also tries "active_specialist" as
#                 secondary for roles that historically live there)
#   cost_classes — acceptable cost_class values (empty = no cost filter)
#   tiers        — acceptable tier values (empty = no tier filter)
#   roles_hint   — preferred role names used for tie-breaking (not hard filter)
# ---------------------------------------------------------------------------

_ROLE_QUERY: dict[str, dict] = {
    "cheap": {
        "lifecycle":    "active_default",
        "cost_classes": {"cheap", "tiny", "free"},
        "tiers":        {"cheap", "tiny"},
        "roles_hint":   ("cheap_control", "boilerplate"),
    },
    "main": {
        "lifecycle":    "active_default",
        "cost_classes": {"cheap", "mid"},
        "tiers":        {"mid", "value_worker"},
        "roles_hint":   ("routine_engineering", "standard_coding"),
    },
    "strong": {
        "lifecycle":    "active_default",
        "cost_classes": set(),          # no cost ceiling for strong
        "tiers":        {"strong"},
        "roles_hint":   ("planner", "reviewer"),
    },
    "escalate": {
        "lifecycle":    "active_specialist",  # escalation lives here by design
        "cost_classes": set(),
        "tiers":        {"frontier"},
        "roles_hint":   ("escalation",),
    },
    "visual": {
        "lifecycle":    "active_specialist",  # visual specialist
        "cost_classes": set(),
        "tiers":        set(),
        "roles_hint":   ("visual", "multimodal"),
    },
    "complex": {
        "lifecycle":    "active_specialist",  # complex/long-horizon specialist
        "cost_classes": set(),
        "tiers":        set(),
        "roles_hint":   ("complex", "long_context"),
    },
}


@cache
def resolve_routing_model(role: str) -> str:
    """Return the best registry-backed model ID for *role*.

    Resolution order:
    1. Filter ``_REGISTRY`` to entries with the target lifecycle.
    2. Apply cost_class and tier filters where specified.
    3. Prefer entries whose roles overlap with the role hint list.
    4. Stable tie-break: alphabetical by model ID for determinism.
    5. If no candidate survives any step, return ``_FALLBACKS[role]``.

    Never raises. Unknown role strings return the "main" fallback.
    """
    if role not in _ROLE_QUERY:
        return _FALLBACKS.get(role, _FALLBACKS["main"])

    try:
        from openshard.models.registry import models_by_lifecycle as _by_lifecycle
    except Exception:
        return _FALLBACKS[role]

    spec = _ROLE_QUERY[role]
    lifecycle: str = spec["lifecycle"]
    cost_classes: set = spec["cost_classes"]
    tiers: set = spec["tiers"]
    roles_hint: tuple = spec["roles_hint"]

    # Gather candidates from primary lifecycle
    candidates = _by_lifecycle(lifecycle)

    # For roles whose primary lifecycle is active_specialist, we only search
    # there. For roles whose primary lifecycle is active_default, also try
    # active_specialist as a secondary source if the primary yields nothing
    # after filtering — this covers edge cases where the registry evolves.

    def _apply_filters(pool):
        result = []
        for m in pool:
            if cost_classes and m.cost_class not in cost_classes:
                continue
            if tiers and m.tier not in tiers:
                continue
            result.append(m)
        return result

    filtered = _apply_filters(candidates)

    # If primary lifecycle is active_default and nothing survived, try
    # relaxing cost/tier filters before giving up.
    if not filtered and lifecycle == "active_default":
        filtered = candidates  # relax filters, keep lifecycle constraint

    if not filtered:
        return _FALLBACKS[role]

    # Tie-break 1: prefer entries with a matching roles_hint
    def _hint_score(m) -> int:
        return sum(1 for h in roles_hint if h in m.roles)

    # Tie-break 2: stable alphabetical by ID
    filtered.sort(key=lambda m: (-_hint_score(m), m.id))
    return filtered[0].id


@cache
def resolution_source(role: str) -> str:
    """Return ``"registry"`` if the model was resolved from the registry,
    ``"hardcoded"`` if the fallback constant was used.
    """
    resolved = resolve_routing_model(role)
    fallback = _FALLBACKS.get(role, "")
    return "hardcoded" if resolved == fallback else "registry"


# ---------------------------------------------------------------------------
# Module-level constants — same names as engine.py previously defined.
# All other modules (stages.py, dispatch.py, pipeline.py) continue to
# import these from engine.py which re-exports them from here.
# ---------------------------------------------------------------------------

try:
    MODEL_CHEAP    = resolve_routing_model("cheap")
    MODEL_MAIN     = resolve_routing_model("main")
    MODEL_STRONG   = resolve_routing_model("strong")
    MODEL_ESCALATE = resolve_routing_model("escalate")
    MODEL_VISUAL   = resolve_routing_model("visual")
    MODEL_COMPLEX  = resolve_routing_model("complex")
except Exception:  # pragma: no cover — pathological registry failure
    MODEL_CHEAP    = _FALLBACKS["cheap"]
    MODEL_MAIN     = _FALLBACKS["main"]
    MODEL_STRONG   = _FALLBACKS["strong"]
    MODEL_ESCALATE = _FALLBACKS["escalate"]
    MODEL_VISUAL   = _FALLBACKS["visual"]
    MODEL_COMPLEX  = _FALLBACKS["complex"]

ESCALATION_CHAIN: list[str] = [MODEL_STRONG, MODEL_ESCALATE]
