from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class NativeContextBudget:
    """Estimated context budget for a native run.

    Token counts are placeholder estimates — not exact tiktoken counts.
    Future branches will wire in real counting.
    """

    context_window: int | None = None
    estimated_tokens_used: int = 0
    estimated_tokens_remaining: int | None = None
    files_loaded: int = 0
    skills_loaded: int = 0
    repo_map_built: bool = False
    warnings: list[str] = field(default_factory=list)


@dataclass
class CompactRunState:
    """Compact snapshot of a native run for future context compaction."""

    task_goal: str = ""
    repo_facts_summary: str = ""
    files_touched: list[str] = field(default_factory=list)
    verification_result: str | None = None
    blockers: list[str] = field(default_factory=list)
    next_step: str = ""


@dataclass
class NativeObservation:
    observed_tools: list[str] = field(default_factory=list)
    dirty_diff_present: bool = False
    search_matches_count: int = 0
    verification_available: bool = False
    warnings: list[str] = field(default_factory=list)


@dataclass
class NativeEvidence:
    search_results: list[str] = field(default_factory=list)
    truncated: bool = False


def build_initial_context_budget(
    context_window: int | None = None,
) -> NativeContextBudget:
    """Create a fresh context budget at the start of a native run."""
    return NativeContextBudget(
        context_window=context_window,
        estimated_tokens_remaining=context_window,
    )


def render_native_observation(
    observation: NativeObservation, *, limit: int = 600
) -> str:
    lines = ["[observation]"]

    if observation.observed_tools:
        lines.append(f"tools: {', '.join(observation.observed_tools)}")

    lines.append(f"dirty diff: {'yes' if observation.dirty_diff_present else 'no'}")
    lines.append(f"search matches: {observation.search_matches_count}")
    lines.append(
        f"verification available: {'yes' if observation.verification_available else 'no'}"
    )

    if observation.warnings:
        shown = observation.warnings[:3]
        lines.append(f"warnings: {', '.join(shown)}")

    rendered = "\n".join(lines)

    suffix = "\n[truncated]"
    if len(rendered) <= limit:
        return rendered
    return (rendered[: max(0, limit - len(suffix))].rstrip() + suffix)[:limit]


def render_native_evidence(evidence: NativeEvidence, *, limit: int = 600) -> str:
    lines = ["[evidence]"]

    if evidence.search_results:
        lines.append("search:")
        for item in evidence.search_results[:3]:
            lines.append(f"- {item}")

    if evidence.truncated:
        lines.append("[truncated]")

    rendered = "\n".join(lines)

    suffix = "\n[truncated]"
    if len(rendered) <= limit:
        return rendered
    return (rendered[: max(0, limit - len(suffix))].rstrip() + suffix)[:limit]


def build_compact_run_state(
    task_goal: str,
    repo_facts_summary: str = "",
    files_touched: list[str] | None = None,
    verification_result: str | None = None,
    blockers: list[str] | None = None,
    next_step: str = "",
) -> CompactRunState:
    """Build a compact snapshot of run state for future compaction."""
    return CompactRunState(
        task_goal=task_goal,
        repo_facts_summary=repo_facts_summary,
        files_touched=files_touched or [],
        verification_result=verification_result,
        blockers=blockers or [],
        next_step=next_step,
    )
