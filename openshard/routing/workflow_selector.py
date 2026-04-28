from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from openshard.analysis.repo import RepoFacts

MIN_CATEGORY_RUNS = 5
RETRY_ESCALATE = 0.4
PASS_RATE_ESCALATE = 0.3
PASS_RATE_DEESCALATE = 0.8
RETRY_DEESCALATE = 0.2

_STAGED_BY_DEFAULT = {"security", "complex"}


@dataclass
class WorkflowHistorySummary:
    retry_rate: float
    verification_pass_rate: float
    sample_count: int


@dataclass
class WorkflowDecision:
    workflow: Literal["direct", "staged"]
    reason: str


def build_workflow_history_summary(
    runs: list[dict], category: str
) -> WorkflowHistorySummary | None:
    """Aggregate run history for a specific task category."""
    filtered = [r for r in runs if r.get("routing_category") == category]
    if not filtered:
        return None
    n = len(filtered)
    retry_count = sum(1 for r in filtered if r.get("retry_triggered"))
    verif_runs = [r for r in filtered if r.get("verification_passed") is not None]
    verif_passed = sum(1 for r in verif_runs if r.get("verification_passed") is True)
    pass_rate = verif_passed / len(verif_runs) if verif_runs else 0.0
    return WorkflowHistorySummary(
        retry_rate=retry_count / n,
        verification_pass_rate=pass_rate,
        sample_count=n,
    )


def select_workflow(
    category: str,
    repo_facts: RepoFacts | None,
    history_summary: WorkflowHistorySummary | None,
    verify_enabled: bool,  # noqa: ARG001 — reserved for future compound rules
) -> WorkflowDecision:
    """Choose between direct and staged workflow based on task signals.

    Deterministic rules only. History signals require MIN_CATEGORY_RUNS samples
    before they influence the decision.
    """
    risky = bool(repo_facts and repo_facts.risky_paths)
    has_history = (
        history_summary is not None and history_summary.sample_count >= MIN_CATEGORY_RUNS
    )

    if category in _STAGED_BY_DEFAULT:
        # De-escalate only when all three gates pass (intentionally conservative)
        if (
            has_history
            and history_summary.verification_pass_rate >= PASS_RATE_DEESCALATE  # type: ignore[union-attr]
            and history_summary.retry_rate < RETRY_DEESCALATE  # type: ignore[union-attr]
            and not risky
        ):
            return WorkflowDecision("direct", "history cleared gates for staged category")
        return WorkflowDecision("staged", "category defaults to staged")

    # Base is direct — check escalation signals
    if risky:
        return WorkflowDecision("staged", "risky paths detected")
    if has_history and history_summary.retry_rate >= RETRY_ESCALATE:  # type: ignore[union-attr]
        return WorkflowDecision("staged", "high retry rate in this category")
    if has_history and history_summary.verification_pass_rate <= PASS_RATE_ESCALATE:  # type: ignore[union-attr]
        return WorkflowDecision("staged", "low verification pass rate")
    return WorkflowDecision("direct", "category defaults to direct")
