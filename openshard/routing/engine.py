from __future__ import annotations

from dataclasses import dataclass


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

MODEL_CHEAP    = "deepseek/deepseek-v3.2"   # low-risk boilerplate
MODEL_MAIN     = "z-ai/glm-5.1"             # standard coding (default worker)
MODEL_STRONG   = "anthropic/claude-sonnet-4.6"  # security / careful reasoning
MODEL_ESCALATE = "anthropic/claude-opus-4.7"    # escalation only
MODEL_VISUAL   = "moonshotai/kimi-k2.5"    # UI / visual / multimodal
MODEL_COMPLEX  = "minimax/m2.7"            # long-horizon / multi-file

# Escalation chain used when verification fails.
# First retry: STRONG; second retry: ESCALATE.
ESCALATION_CHAIN: list[str] = [MODEL_STRONG, MODEL_ESCALATE]


# ---------------------------------------------------------------------------
# Routing decision
# ---------------------------------------------------------------------------

@dataclass
class RoutingDecision:
    model: str
    category: str   # boilerplate | standard | security | visual | complex
    rationale: str  # shown in --more output


# ---------------------------------------------------------------------------
# Keyword sets (substring match on lowercased task)
# ---------------------------------------------------------------------------

_SECURITY_KW = {
    "auth", "login", "logout", "password", "token", "jwt", "session",
    "encrypt", "decrypt", "security", "permission", "role", "rbac",
    "oauth", "api key", "secret", "credential", "privilege",
    "access control", "firewall", "ssl", "tls", "certificate",
}

_VISUAL_KW = {
    "ui", "component", "styling", "css", "layout", "design",
    "visual", "chart", "graph", "dashboard", "react", "vue", "angular",
    "tailwind", "scss", "animation", "image processing", "svg",
}

_COMPLEX_KW = {
    "refactor", "migrate", "migration", "architecture", "restructur",
    "system-wide", "multi-file", "large-scale", "codebase", "throughout",
    "across all", "across the", "every file", "optimization",
    "performance tuning",
}

_BOILERPLATE_KW = {
    "validation", "validate", "format", "formatter", "helper", "utility",
    "util", "boilerplate", "getter", "setter", "sanitize",
    "simple form", "basic", "add simple", "add a simple",
}


# ---------------------------------------------------------------------------
# Classifier
# ---------------------------------------------------------------------------

def _matches(lower: str, kw_set: set[str]) -> bool:
    """True if any keyword in *kw_set* appears as a substring of *lower*."""
    return any(kw in lower for kw in kw_set)


def route(task: str) -> RoutingDecision:
    """Classify *task* and return the appropriate model with rationale.

    Priority order (first match wins):
    1. Security-sensitive  → Sonnet (careful reasoning required)
    2. Visual / frontend   → Kimi K2.5 (multimodal specialist)
    3. Complex / multi-file → MiniMax M2.7 (long-horizon model)
    4. Boilerplate         → DeepSeek V3.2 (cheapest)
    5. Default             → GLM-5.1 (main worker)
    """
    lower = task.lower()
    word_count = len(task.split())

    if _matches(lower, _SECURITY_KW):
        return RoutingDecision(
            model=MODEL_STRONG,
            category="security",
            rationale="security-sensitive code requires careful reasoning",
        )

    if _matches(lower, _VISUAL_KW):
        return RoutingDecision(
            model=MODEL_VISUAL,
            category="visual",
            rationale="UI or visual task routed to multimodal specialist",
        )

    if _matches(lower, _COMPLEX_KW) or word_count > 60:
        return RoutingDecision(
            model=MODEL_COMPLEX,
            category="complex",
            rationale="multi-file or long-horizon task",
        )

    if _matches(lower, _BOILERPLATE_KW):
        return RoutingDecision(
            model=MODEL_CHEAP,
            category="boilerplate",
            rationale="low-risk boilerplate task",
        )

    return RoutingDecision(
        model=MODEL_MAIN,
        category="standard",
        rationale="standard feature implementation",
    )


# ---------------------------------------------------------------------------
# Legacy class (preserved for any existing callers)
# ---------------------------------------------------------------------------

class RoutingEngine:
    """Select the appropriate model for a given task."""

    def select_model(self, task: str) -> str:
        return route(task).model
