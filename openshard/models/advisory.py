from __future__ import annotations

from dataclasses import dataclass

from openshard.models.registry import CAPABILITY_NAMES, ModelEntry, _CAPABILITY_ATTRS, all_models

_COST_ORDER: dict[str, int] = {
    "free": 0,
    "tiny": 1,
    "cheap": 2,
    "mid": 3,
    "expensive": 4,
    "unknown": 5,
}

_HIGH_RISK_TIERS: frozenset[str] = frozenset({"strong", "frontier"})
_LOW_RISK_COSTS: frozenset[str] = frozenset({"free", "tiny", "cheap"})

__all__ = ["CAPABILITY_NAMES", "ModelAdvisory", "recommend_models"]


@dataclass(frozen=True)
class ModelAdvisory:
    model: ModelEntry
    reasons: tuple[str, ...]


def recommend_models(
    *,
    role: str | None = None,
    risk: str | None = None,
    required_capabilities: tuple[str, ...] = (),
    max_cost_class: str | None = None,
    include_experimental: bool = False,
    limit: int = 5,
) -> list[ModelAdvisory]:
    for cap in required_capabilities:
        if cap not in _CAPABILITY_ATTRS:
            return []

    candidates = all_models()

    if role is not None:
        candidates = [m for m in candidates if role in m.roles]
        if not candidates:
            return []

    if not include_experimental:
        candidates = [m for m in candidates if not m.experimental]

    for cap in required_capabilities:
        attr = _CAPABILITY_ATTRS[cap]
        candidates = [m for m in candidates if getattr(m, attr)]

    if max_cost_class is not None:
        ceiling = _COST_ORDER.get(max_cost_class, 5)
        candidates = [m for m in candidates if _COST_ORDER.get(m.cost_class, 5) <= ceiling]

    scored: list[tuple[tuple, ModelAdvisory]] = []
    for model in candidates:
        score = 0
        reasons: list[str] = []

        if role is not None:
            score += 10
            reasons.append(f"matches role {role}")

        if risk == "high":
            if model.tier in _HIGH_RISK_TIERS:
                score += 8
                reasons.append("suitable for high-risk review")
            if model.supports_reasoning:
                score += 4
                if "suitable for high-risk review" not in reasons:
                    reasons.append("suitable for high-risk review")
        elif risk == "low":
            if model.cost_class in _LOW_RISK_COSTS:
                score += 5
            if model.latency_class == "fast":
                score += 3
        elif risk == "medium":
            if model.cost_class in {"cheap", "mid"}:
                score += 3

        for cap in required_capabilities:
            reasons.append(f"supports {cap}")

        if model.cost_class != "unknown":
            reasons.append(f"cost class {model.cost_class}")

        if not model.experimental:
            reasons.append("non-experimental")

        sort_key = (-score, _COST_ORDER.get(model.cost_class, 5), model.id)
        scored.append((sort_key, ModelAdvisory(model=model, reasons=tuple(reasons))))

    scored.sort(key=lambda x: x[0])
    return [advisory for _, advisory in scored[:limit]]
