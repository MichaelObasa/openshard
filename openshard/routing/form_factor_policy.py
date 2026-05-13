from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from openshard.analysis.repo import RepoFacts

_HIGH_RISK_CATEGORIES = {"security", "auth", "payments", "infra", "migration"}
_MEDIUM_RISK_CATEGORIES = {"complex"}


@dataclass
class ExecutionFormFactorDecision:
    public_mode: Literal["ask", "run", "deep-run", "osn-run"]
    internal_form_factor: Literal[
        "direct", "staged", "native-loop-candidate", "subagent-candidate", "swarm-candidate"
    ]
    reason: str
    confidence: Literal["low", "medium", "high"]
    risk_level: Literal["low", "medium", "high"]
    read_only: bool
    write_requested: bool
    verification_available: bool
    context_quality: str | None = None
    warnings: list[str] = field(default_factory=list)


def _native_loop_enabled(value: str | None) -> bool:
    """Return True only when native_loop is explicitly and positively enabled."""
    if not value:
        return False
    return value.lower() not in {"off", "none", "disabled", "false", "0", "no"}


def _derive_risk_level(
    category: str,
    repo_facts: RepoFacts | None,
    write_requested: bool,
) -> Literal["low", "medium", "high"]:
    if category in _HIGH_RISK_CATEGORIES:
        return "high"
    # Risky paths only escalate when the task is a write task
    if write_requested and repo_facts and repo_facts.risky_paths:
        return "high"
    if category in _MEDIUM_RISK_CATEGORIES:
        return "medium"
    return "low"


def select_form_factor(
    category: str,
    readonly: bool,
    workflow: Literal["direct", "staged"],
    profile_name: str,
    repo_facts: RepoFacts | None,
    write_requested: bool,
    verification_available: bool,
    native_loop: str | None = None,
    experimental_deepagents_run: bool = False,
    context_quality_level: str | None = None,
) -> ExecutionFormFactorDecision:
    """Determine the user-facing execution form factor from existing signals.

    This is a recording/labelling layer only — it does not change execution behaviour.
    """
    risk = _derive_risk_level(category, repo_facts, write_requested)
    warnings: list[str] = []

    if context_quality_level in {"weak", "unknown"}:
        warnings.append("context quality is weak")
    if write_requested and repo_facts and repo_facts.risky_paths:
        warnings.append(f"write touches {len(repo_facts.risky_paths)} risky path(s)")

    def _decision(
        public_mode: Literal["ask", "run", "deep-run", "osn-run"],
        internal_form_factor: Literal[
            "direct", "staged", "native-loop-candidate", "subagent-candidate", "swarm-candidate"
        ],
        reason: str,
        confidence: Literal["low", "medium", "high"],
    ) -> ExecutionFormFactorDecision:
        return ExecutionFormFactorDecision(
            public_mode=public_mode,
            internal_form_factor=internal_form_factor,
            reason=reason,
            confidence=confidence,
            risk_level=risk,
            read_only=readonly,
            write_requested=write_requested,
            verification_available=verification_available,
            context_quality=context_quality_level,
            warnings=warnings,
        )

    if _native_loop_enabled(native_loop):
        return _decision("osn-run", "native-loop-candidate", "explicit native loop requested", "high")

    if readonly:
        return _decision("ask", "direct", "read-only task", "high")

    if experimental_deepagents_run:
        return _decision("deep-run", "subagent-candidate", "experimental deep agents requested", "medium")

    if profile_name == "native_swarm":
        return _decision("deep-run", "swarm-candidate", "swarm profile selected", "medium")

    if risk == "high" or profile_name == "native_deep":
        return _decision("deep-run", "native-loop-candidate", "riskier task may need controlled native loop", "medium")

    if workflow == "staged":
        return _decision("run", "staged", "staged planning selected", "high")

    return _decision("run", "direct", "simple safe task", "high")
