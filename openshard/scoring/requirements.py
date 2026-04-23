from __future__ import annotations

from dataclasses import dataclass

from openshard.execution.stages import Stage


@dataclass
class TaskRequirements:
    min_context_window: int | None = None
    needs_vision: bool = False
    needs_tools: bool = False
    complexity: str = "standard"  # "simple" | "standard" | "complex"
    security_sensitive: bool = False
    preferred_max_cost_per_m: float | None = None  # USD per million prompt tokens


def requirements_from_category(category: str) -> TaskRequirements:
    if category == "security":
        return TaskRequirements(security_sensitive=True, complexity="complex", min_context_window=100_000)
    if category == "complex":
        return TaskRequirements(complexity="complex", min_context_window=100_000)
    if category == "visual":
        return TaskRequirements(needs_vision=True)
    if category == "boilerplate":
        return TaskRequirements(complexity="simple", preferred_max_cost_per_m=2.0)
    return TaskRequirements(preferred_max_cost_per_m=1.5)


def requirements_from_stage(stage: Stage) -> TaskRequirements:
    return TaskRequirements(
        min_context_window=100_000 if stage.complexity == "complex" else None,
        needs_vision=False,
        needs_tools=False,
        complexity=stage.complexity,
        security_sensitive=stage.security_sensitive,
    )
