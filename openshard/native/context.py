from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


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
class NativeVerificationCommandSummary:
    attempted: bool = False
    command_count: int = 0
    safe_count: int = 0
    needs_approval_count: int = 0
    blocked_count: int = 0
    passed: bool = False
    retried: bool = False
    warnings: list[str] = field(default_factory=list)


def build_native_verification_command_summary(
    *,
    verification_loop: NativeVerificationLoop | None,
    verification_plan: Any | None,
) -> NativeVerificationCommandSummary:
    summary = NativeVerificationCommandSummary()
    if verification_loop is not None:
        summary.attempted = verification_loop.attempted
        summary.passed = verification_loop.passed
        summary.retried = verification_loop.retried
    if verification_plan is not None and hasattr(verification_plan, "commands"):
        from openshard.verification.plan import CommandSafety
        commands = verification_plan.commands
        summary.command_count = len(commands)
        for cmd in commands:
            if cmd.safety == CommandSafety.safe:
                summary.safe_count += 1
            elif cmd.safety == CommandSafety.needs_approval:
                summary.needs_approval_count += 1
            elif cmd.safety == CommandSafety.blocked:
                summary.blocked_count += 1
    if summary.attempted and summary.command_count == 0:
        summary.warnings.append("verification attempted but no commands found in plan")
    return summary


@dataclass
class NativeCommandPolicyPreview:
    safe_count: int = 0
    needs_approval_count: int = 0
    blocked_count: int = 0
    command_classes: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def build_native_command_policy_preview(
    verification_plan: Any | None,
) -> NativeCommandPolicyPreview:
    if verification_plan is None or not hasattr(verification_plan, "commands"):
        return NativeCommandPolicyPreview()
    safe = needs_approval = blocked = 0
    classes: set[str] = set()
    for cmd in verification_plan.commands:
        safety = getattr(cmd, "safety", None)
        if safety is None:
            continue
        label = str(safety.value) if hasattr(safety, "value") else str(safety)
        if label == "safe":
            safe += 1
        elif label == "needs_approval":
            needs_approval += 1
        elif label == "blocked":
            blocked += 1
        classes.add(label)
    warnings: list[str] = []
    if safe + needs_approval + blocked == 0:
        warnings.append("no commands with safety classification found")
    return NativeCommandPolicyPreview(
        safe_count=safe,
        needs_approval_count=needs_approval,
        blocked_count=blocked,
        command_classes=sorted(classes),
        warnings=warnings,
    )


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


@dataclass
class NativeFileContext:
    files_read: int = 0
    paths: list[str] = field(default_factory=list)
    total_chars: int = 0
    truncated: bool = False
    warnings: list[str] = field(default_factory=list)


@dataclass
class NativePatchProposal:
    file_count: int = 0
    files: list[str] = field(default_factory=list)
    change_types: list[str] = field(default_factory=list)
    summaries: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def build_native_patch_proposal(files: list[Any]) -> NativePatchProposal:
    paths: list[str] = []
    change_types: list[str] = []
    summaries: list[str] = []
    for f in files:
        path = getattr(f, "path", None)
        change_type = getattr(f, "change_type", "update")
        summary = getattr(f, "summary", "")
        if path:
            paths.append(path)
            change_types.append(change_type or "update")
            summaries.append(summary or "")
    return NativePatchProposal(
        file_count=len(paths),
        files=paths,
        change_types=change_types,
        summaries=summaries,
    )


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


@dataclass
class NativeContextPacket:
    task_preview: str = ""
    sources: list[str] = field(default_factory=list)
    repo_stack: list[str] = field(default_factory=list)
    test_marker_count: int = 0
    package_file_count: int = 0
    read_search_count: int = 0
    selected_skills: list[str] = field(default_factory=list)
    backend: str = "builtin"
    backend_available: bool = True
    backend_proof_mode: str = ""
    compact_paths: list[str] = field(default_factory=list)
    file_context_files: int = 0
    warnings: list[str] = field(default_factory=list)


def build_native_context_packet(
    *,
    task: str,
    repo_context_summary: Any | None = None,
    read_search_findings: list[str] | None = None,
    selected_skills: list[str] | None = None,
    native_backend: str = "builtin",
    native_backend_available: bool = True,
    native_backend_proof: dict | None = None,
    file_context_files: int = 0,
) -> NativeContextPacket:
    sources: list[str] = []
    warnings: list[str] = []
    task_preview = (task or "")[:300]
    repo_stack: list[str] = []
    test_marker_count = 0
    package_file_count = 0

    if repo_context_summary is not None:
        sources.append("repo_context")
        repo_stack = list(getattr(repo_context_summary, "likely_stack_markers", []) or [])
        test_marker_count = len(getattr(repo_context_summary, "test_markers", []) or [])
        package_file_count = len(getattr(repo_context_summary, "package_files", []) or [])

    findings = list(read_search_findings or [])
    if findings:
        sources.append("read_search")

    compact_paths: list[str] = []
    for item in findings:
        if len(compact_paths) >= 8:
            warnings.append("context packet paths truncated")
            break
        if item.startswith(("file:", "test-marker:", "package:")):
            compact_paths.append(item)

    skills = list(selected_skills or [])
    if skills:
        sources.append("skills")

    proof_mode = ""
    if native_backend_proof:
        proof_mode = str(native_backend_proof.get("mode", ""))

    if native_backend:
        sources.append("backend")

    return NativeContextPacket(
        task_preview=task_preview,
        sources=sorted(set(sources)),
        repo_stack=repo_stack[:8],
        test_marker_count=test_marker_count,
        package_file_count=package_file_count,
        read_search_count=len(findings),
        selected_skills=skills[:8],
        backend=native_backend,
        backend_available=native_backend_available,
        backend_proof_mode=proof_mode,
        compact_paths=compact_paths,
        file_context_files=file_context_files,
        warnings=warnings,
    )


def render_native_context_packet(packet: NativeContextPacket | None) -> str:
    if packet is None:
        return ""

    lines: list[str] = ["OpenShard Native context packet:"]

    if packet.sources:
        lines.append(f"- sources: {', '.join(packet.sources)}")

    if packet.repo_stack:
        lines.append(f"- repo stack: {', '.join(packet.repo_stack[:8])}")

    if packet.test_marker_count or packet.package_file_count:
        lines.append(
            f"- repo signals: {packet.test_marker_count} test markers, "
            f"{packet.package_file_count} package files"
        )

    if packet.read_search_count:
        lines.append(f"- read/search findings: {packet.read_search_count}")

    file_context_files = getattr(packet, "file_context_files", 0)
    file_context_chars = getattr(packet, "file_context_chars", 0)
    if file_context_files:
        lines.append(
            f"- file context: {file_context_files} files, "
            f"{file_context_chars} chars"
        )

    if packet.selected_skills:
        lines.append(f"- selected skills: {', '.join(packet.selected_skills[:8])}")

    if packet.backend:
        availability = "available" if packet.backend_available else "unavailable"
        lines.append(f"- backend: {packet.backend} ({availability})")

    if packet.backend_proof_mode:
        lines.append(f"- backend proof: {packet.backend_proof_mode}")

    if packet.compact_paths:
        lines.append("- compact paths:")
        for path in packet.compact_paths[:8]:
            lines.append(f"  - {path}")

    if packet.warnings:
        lines.append("- warnings:")
        for warning in packet.warnings[:5]:
            lines.append(f"  - {warning}")

    return "\n".join(lines)


@dataclass
class NativeContextQualityScore:
    score: int = 0
    max_score: int = 100
    level: str = "unknown"  # weak | fair | good | strong
    reasons: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


@dataclass
class NativeContextQualityAdvisory:
    level: str = "unknown"
    recommendation: str = ""
    should_block: bool = False
    warnings: list[str] = field(default_factory=list)


def build_native_context_quality_advisory(
    score: NativeContextQualityScore | None,
) -> NativeContextQualityAdvisory:
    if score is None:
        return NativeContextQualityAdvisory(
            level="unknown",
            recommendation="context quality is unknown",
            should_block=False,
            warnings=["context quality score missing"],
        )

    level = score.level

    if level == "strong":
        recommendation = "context is strong enough for normal generation"
        warnings: list[str] = []
    elif level == "good":
        recommendation = "context is good enough for normal generation"
        warnings = []
    elif level == "fair":
        recommendation = "context is usable but may need cautious generation"
        warnings = ["consider smaller changes if the task is risky"]
    elif level == "weak":
        recommendation = "context is weak; prefer smaller changes or gather more context"
        warnings = ["context packet may be insufficient for confident generation"]
    else:
        recommendation = "context quality is unknown"
        warnings = ["unknown context quality level"]

    return NativeContextQualityAdvisory(
        level=level,
        recommendation=recommendation,
        should_block=False,
        warnings=warnings,
    )


def render_native_context_quality_advisory(
    advisory: NativeContextQualityAdvisory | None,
) -> str:
    if advisory is None:
        return ""

    lines = ["OpenShard Native context advisory:"]
    if advisory.level:
        lines.append(f"- level: {advisory.level}")
    if advisory.recommendation:
        lines.append(f"- recommendation: {advisory.recommendation}")
    lines.append(f"- should block: {str(advisory.should_block).lower()}")

    return "\n".join(lines)


@dataclass
class NativeChangeBudget:
    level: str = "unknown"
    max_files: int = 1
    max_change_size: str = "small"
    guidance: str = ""
    warnings: list[str] = field(default_factory=list)


def build_native_change_budget(
    advisory: NativeContextQualityAdvisory | None,
) -> NativeChangeBudget:
    if advisory is None:
        return NativeChangeBudget(
            level="unknown",
            max_files=1,
            max_change_size="small",
            guidance="prefer the smallest safe change; gather more context if needed",
            warnings=["context advisory missing"],
        )

    level = advisory.level

    if level == "strong":
        return NativeChangeBudget(
            level=level,
            max_files=5,
            max_change_size="normal",
            guidance="normal scoped generation is acceptable",
        )

    if level == "good":
        return NativeChangeBudget(
            level=level,
            max_files=3,
            max_change_size="normal",
            guidance="normal generation is acceptable; avoid unnecessary broad refactors",
        )

    if level == "fair":
        return NativeChangeBudget(
            level=level,
            max_files=2,
            max_change_size="small",
            guidance="prefer a cautious, focused change",
            warnings=["context is usable but not strong"],
        )

    return NativeChangeBudget(
        level=level,
        max_files=1,
        max_change_size="small",
        guidance="prefer the smallest safe change; avoid broad refactors",
        warnings=["context is weak or unknown"],
    )


@dataclass
class NativeChangeBudgetPreview:
    budget_max_files: int = 0
    proposed_files: int = 0
    within_budget: bool = True
    would_exceed_budget: bool = False
    action: str = "allow"
    warnings: list[str] = field(default_factory=list)


def build_native_change_budget_preview(
    *,
    budget: NativeChangeBudget | None,
    proposal: NativePatchProposal | None,
) -> NativeChangeBudgetPreview:
    max_files = budget.max_files if budget is not None else 0
    proposed_files = proposal.file_count if proposal is not None else 0

    if max_files <= 0:
        return NativeChangeBudgetPreview(
            budget_max_files=max_files,
            proposed_files=proposed_files,
            within_budget=True,
            would_exceed_budget=False,
            action="allow",
            warnings=["change budget missing or invalid"],
        )

    would_exceed = proposed_files > max_files

    return NativeChangeBudgetPreview(
        budget_max_files=max_files,
        proposed_files=proposed_files,
        within_budget=not would_exceed,
        would_exceed_budget=would_exceed,
        action="warn" if would_exceed else "allow",
        warnings=(
            [f"proposal has {proposed_files} files but budget allows {max_files}"]
            if would_exceed
            else []
        ),
    )


def render_native_change_budget(budget: NativeChangeBudget | None) -> str:
    if budget is None:
        return ""

    lines = ["OpenShard Native change budget:"]
    lines.append(f"- level: {budget.level}")
    lines.append(f"- max files: {budget.max_files}")
    lines.append(f"- max change size: {budget.max_change_size}")
    if budget.guidance:
        lines.append(f"- guidance: {budget.guidance}")

    return "\n".join(lines)


def build_native_context_quality_score(packet: NativeContextPacket) -> NativeContextQualityScore:
    score = 0
    reasons: list[str] = []
    warnings: list[str] = []

    if "repo_context" in packet.sources:
        score += 20
        reasons.append("repo_context")
    if "read_search" in packet.sources:
        score += 15
        reasons.append("read_search")
    if packet.compact_paths:
        score += 15
        reasons.append("compact_paths")
    if packet.selected_skills:
        score += 15
        reasons.append("selected_skills")
    if packet.backend:
        score += 10
        reasons.append("backend")
    if packet.repo_stack:
        score += 10
        reasons.append("repo_stack")
    if packet.test_marker_count > 0:
        score += 10
        reasons.append("test_markers")
    if packet.file_context_files > 0:
        score += 5
        reasons.append("file_context")

    score = min(score, 100)

    if score >= 80:
        level = "strong"
    elif score >= 60:
        level = "good"
    elif score >= 35:
        level = "fair"
    else:
        level = "weak"
        warnings.append("context packet may be insufficient for generation")

    return NativeContextQualityScore(
        score=score,
        max_score=100,
        level=level,
        reasons=reasons,
        warnings=warnings,
    )
