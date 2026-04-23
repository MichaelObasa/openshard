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


def requirements_from_stage(stage: Stage) -> TaskRequirements:
    return TaskRequirements(
        min_context_window=8000 if stage.complexity == "complex" else None,
        needs_vision=False,
        needs_tools=False,
        complexity=stage.complexity,
        security_sensitive=stage.security_sensitive,
    )
