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
class NativeFileSnippet:
    path: str
    lines: list[str] = field(default_factory=list)


@dataclass
class NativeEvidence:
    search_results: list[str] = field(default_factory=list)
    file_snippets: list[NativeFileSnippet] = field(default_factory=list)
    truncated: bool = False


@dataclass
class NativeDiffReview:
    has_diff: bool = False
    changed_files: list[str] = field(default_factory=list)
    added_lines: int = 0
    removed_lines: int = 0
    output_chars: int = 0
    truncated: bool = False


@dataclass
class NativePlan:
    intent: str = "standard"
    risk: str = "low"
    suggested_steps: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


@dataclass
class NativeVerificationLoop:
    attempted: bool = False
    passed: bool = False
    retried: bool = False
    exit_code: int | None = None
    output_chars: int = 0
    truncated: bool = False


@dataclass
class NativeFinalReport:
    used_native_context: bool = False
    observed_tools: list[str] = field(default_factory=list)
    selected_skills: list[str] = field(default_factory=list)
    plan_intent: str | None = None
    plan_risk: str | None = None
    evidence_items: int = 0
    snippet_files: int = 0
    verification_attempted: bool = False
    verification_passed: bool = False
    verification_retried: bool = False
    diff_files: list[str] = field(default_factory=list)
    added_lines: int = 0
    removed_lines: int = 0
    warnings: list[str] = field(default_factory=list)


def build_initial_context_budget(
    context_window: int | None = None,
) -> NativeContextBudget:
    """Create a fresh context budget at the start of a native run."""
    return NativeContextBudget(
        context_window=context_window,
        estimated_tokens_remaining=context_window,
    )


def build_native_final_report(
    *,
    selected_skills: list[str],
    observation: NativeObservation | None,
    evidence: NativeEvidence | None,
    plan: NativePlan | None,
    verification_loop: NativeVerificationLoop | None,
    diff_review: NativeDiffReview | None,
) -> NativeFinalReport:
    warnings: list[str] = []

    if observation is not None:
        warnings.extend(observation.warnings)
        if observation.dirty_diff_present:
            warnings.append("dirty working tree detected")

    if plan is not None:
        warnings.extend(plan.warnings)

    if verification_loop is not None and verification_loop.attempted and not verification_loop.passed:
        warnings.append("verification failed")

    return NativeFinalReport(
        used_native_context=observation is not None or evidence is not None or plan is not None,
        observed_tools=observation.observed_tools if observation is not None else [],
        selected_skills=list(selected_skills),
        plan_intent=plan.intent if plan is not None else None,
        plan_risk=plan.risk if plan is not None else None,
        evidence_items=len(evidence.search_results) if evidence is not None else 0,
        snippet_files=len(evidence.file_snippets) if evidence is not None else 0,
        verification_attempted=verification_loop.attempted if verification_loop is not None else False,
        verification_passed=verification_loop.passed if verification_loop is not None else False,
        verification_retried=verification_loop.retried if verification_loop is not None else False,
        diff_files=diff_review.changed_files if diff_review is not None else [],
        added_lines=diff_review.added_lines if diff_review is not None else 0,
        removed_lines=diff_review.removed_lines if diff_review is not None else 0,
        warnings=sorted(set(warnings)),
    )


def build_native_diff_review(
    diff_output: str,
    *,
    truncated: bool = False,
) -> NativeDiffReview:
    changed_files: set[str] = set()
    added = 0
    removed = 0

    for line in diff_output.splitlines():
        if line.startswith("diff --git "):
            parts = line.split()
            if len(parts) >= 4:
                path = parts[3]
                if path.startswith("b/"):
                    path = path[2:]
                changed_files.add(path)
        elif line.startswith("+++ b/"):
            changed_files.add(line[len("+++ b/"):])
        elif line.startswith("--- a/"):
            changed_files.add(line[len("--- a/"):])
        elif line.startswith("+") and not line.startswith("+++"):
            added += 1
        elif line.startswith("-") and not line.startswith("---"):
            removed += 1

    return NativeDiffReview(
        has_diff=bool(diff_output.strip()),
        changed_files=sorted(changed_files)[:20],
        added_lines=added,
        removed_lines=removed,
        output_chars=len(diff_output),
        truncated=truncated,
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


def render_native_evidence(evidence: NativeEvidence, *, limit: int = 1000) -> str:
    lines = ["[evidence]"]

    if evidence.search_results:
        lines.append("search:")
        for item in evidence.search_results[:3]:
            lines.append(f"- {item}")

    if evidence.truncated:
        lines.append("[truncated]")

    if evidence.file_snippets:
        lines.append("snippets:")
        for snippet in evidence.file_snippets[:2]:
            lines.append(f"{snippet.path}:")
            for line in snippet.lines[:8]:
                lines.append(f"  {line}")

    rendered = "\n".join(lines)

    suffix = "\n[truncated]"
    if len(rendered) <= limit:
        return rendered
    return (rendered[: max(0, limit - len(suffix))].rstrip() + suffix)[:limit]


def render_native_plan(plan: NativePlan, *, limit: int = 600) -> str:
    lines = ["[native plan]"]
    lines.append(f"intent: {plan.intent}")
    lines.append(f"risk: {plan.risk}")

    if plan.suggested_steps:
        lines.append("suggested steps:")
        for step in plan.suggested_steps[:5]:
            lines.append(f"- {step}")

    if plan.warnings:
        lines.append("warnings:")
        for warning in plan.warnings[:3]:
            lines.append(f"- {warning}")

    rendered = "\n".join(lines)
    suffix = "\n[truncated]"
    if len(rendered) <= limit:
        return rendered
    return (rendered[: max(0, limit - len(suffix))].rstrip() + suffix)[:limit]


def render_verification_failure_context(
    output: str,
    *,
    exit_code: int,
    limit: int = 1200,
) -> str:
    header_lines = [
        "[verification failure]",
        f"exit_code: {exit_code}",
        "output:",
    ]
    suffix = "\n[truncated]"
    header = "\n".join(header_lines)
    body_limit = max(0, limit - len(header) - 1 - len(suffix))
    bounded = output
    if len(bounded) > body_limit:
        bounded = bounded[:body_limit].rstrip() + suffix
    return (header + "\n" + bounded)[:limit]


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
