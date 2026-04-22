from __future__ import annotations

from dataclasses import dataclass

from openshard.routing.engine import (
    MODEL_CHEAP, MODEL_MAIN, MODEL_STRONG,
)

# ---------------------------------------------------------------------------
# Planning prompt
# ---------------------------------------------------------------------------

PLANNING_SYSTEM = """\
Analyze the coding task and produce a concise technical implementation plan.
Plain text only — no markdown, no code, no headers.
Write at most 5 bullet points.
Focus on: which files to create or modify, the key logic, and important edge cases.\
"""

# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class Stage:
    stage_type: str        # "planning" | "implementation" | "testing" | "review"
    description: str
    security_sensitive: bool = False
    complexity: str = "standard"   # "simple" | "standard" | "complex"


@dataclass
class StageRun:
    stage: Stage
    model: str
    duration: float
    cost: float | None
    summary: str


# ---------------------------------------------------------------------------
# Splitting and routing
# ---------------------------------------------------------------------------

_SECURITY_KW = {
    "auth", "jwt", "login", "logout", "password", "token", "session",
    "encrypt", "decrypt", "permission", "oauth", "secret", "credential",
    "rbac", "privilege", "access control",
}


def split_task(task: str) -> list[Stage]:
    """Break *task* into logical stages using simple heuristics.

    Returns: [planning, implementation]
    Always starts with a planning stage. The implementation stage is marked
    security_sensitive when relevant keywords are found.
    """
    lower = task.lower()
    security_sensitive = any(kw in lower for kw in _SECURITY_KW)
    impl_complexity = "complex" if security_sensitive else "standard"

    return [
        Stage(
            stage_type="planning",
            description="Analyze task and produce implementation plan",
        ),
        Stage(
            stage_type="implementation",
            description="Execute implementation",
            security_sensitive=security_sensitive,
            complexity=impl_complexity,
        ),
    ]


def route_stage(stage: Stage) -> str:
    """Return the model ID for *stage*.

    Rules:
    - planning  → MODEL_STRONG  (sonnet — careful analysis)
    - review    → MODEL_STRONG
    - testing   → MODEL_CHEAP   (deepseek — mechanical work)
    - implementation, security_sensitive or complex → MODEL_STRONG
    - implementation, simple    → MODEL_CHEAP
    - implementation, standard  → MODEL_MAIN
    """
    if stage.stage_type in ("planning", "review"):
        return MODEL_STRONG
    if stage.stage_type == "testing":
        return MODEL_CHEAP
    # implementation
    if stage.security_sensitive or stage.complexity == "complex":
        return MODEL_STRONG
    if stage.complexity == "simple":
        return MODEL_CHEAP
    return MODEL_MAIN


def should_use_stages(category: str) -> bool:
    """Return True when the task category warrants multi-stage execution."""
    return category in ("security", "complex")


# ---------------------------------------------------------------------------
# Planning call helper
# ---------------------------------------------------------------------------

def run_planning_stage(client, task: str) -> tuple[str, object]:
    """Send a planning request and return ``(plan_text, usage)``.

    Uses MODEL_STRONG (Sonnet) unconditionally — planning is cheap in tokens
    and benefits from careful reasoning.
    """
    response = client.execute(
        model=MODEL_STRONG,
        prompt=f"Task: {task}",
        system=PLANNING_SYSTEM,
    )
    return response.content, response.usage
