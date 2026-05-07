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
class OSNLoopStep:
    step_index: int = 0
    tool_name: str = ""
    target_label: str = ""
    reason: str = ""
    ok: bool = False
    output_chars: int = 0
    empty: bool = False
    skipped: bool = False


@dataclass
class OSNLoopMeta:
    enabled: bool = False
    steps_run: int = 0
    steps_queued: int = 0
    max_steps: int = 0
    consecutive_empty: int = 0
    terminated_reason: str = "not_run"
    steps: list[OSNLoopStep] = field(default_factory=list)
    paths_surfaced: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    truncated: bool = False


def build_osn_loop_meta(
    *,
    steps_run: int,
    steps_queued: int,
    max_steps: int,
    consecutive_empty: int,
    terminated_reason: str,
    steps: list[OSNLoopStep],
    warnings: list[str],
) -> OSNLoopMeta:
    paths_surfaced = [
        s.target_label
        for s in steps
        if s.ok and not s.empty and not s.skipped and s.tool_name == "read_file" and s.target_label
    ][:5]
    return OSNLoopMeta(
        enabled=True,
        steps_run=steps_run,
        steps_queued=steps_queued,
        max_steps=max_steps,
        consecutive_empty=consecutive_empty,
        terminated_reason=terminated_reason,
        steps=steps,
        paths_surfaced=paths_surfaced,
        warnings=warnings,
        truncated=steps_queued > max_steps,
    )


def render_osn_loop_context(meta: OSNLoopMeta | None) -> str:
    """Bounded prompt-safe OSN loop context block.

    Emits only when enabled. Contains no raw file content, snippets, diffs,
    or command strings — only counts, tool names, safe repo-relative paths,
    and the termination reason.
    """
    if meta is None or not meta.enabled:
        return ""
    lines = ["[osn loop]"]
    lines.append(f"steps: {meta.steps_run}/{meta.max_steps}  reason: {meta.terminated_reason}")
    tool_names = sorted(
        {s.tool_name for s in meta.steps if not s.skipped and s.tool_name}
    )
    if tool_names:
        lines.append(f"tools: {', '.join(tool_names)}")
    if meta.paths_surfaced:
        lines.append(f"paths: {', '.join(meta.paths_surfaced[:5])}")
    return "\n".join(lines)


def render_osn_loop(meta: OSNLoopMeta | None, *, detail: str = "default") -> str:
    """Human-readable OSN loop summary for openshard last rendering."""
    if meta is None or not meta.enabled:
        return ""
    lines = [
        f"osn loop: {meta.steps_run}/{meta.max_steps} steps"
        f"  paths={len(meta.paths_surfaced)}"
        f"  reason={meta.terminated_reason}"
    ]
    if meta.truncated:
        lines.append("  [queue truncated to max_steps]")
    if meta.warnings:
        for w in meta.warnings[:3]:
            lines.append(f"  warn: {w}")
    if detail == "full" and meta.steps:
        _MAX_RENDER = 8
        lines.append("  steps:")
        for s in meta.steps[:_MAX_RENDER]:
            status = "skip" if s.skipped else ("ok" if s.ok else "fail")
            empty_tag = " [empty]" if s.empty else ""
            lines.append(
                f"    [{s.step_index}] {s.tool_name}({s.target_label!r})"
                f" {status} chars={s.output_chars}{empty_tag}"
            )
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


@dataclass
class NativeChangeBudgetSoftGate:
    requires_approval: bool = False
    reason: str = ""
    action: str = "allow"
    warnings: list[str] = field(default_factory=list)


def build_native_change_budget_soft_gate(
    preview: NativeChangeBudgetPreview | None,
) -> NativeChangeBudgetSoftGate:
    if preview is None:
        return NativeChangeBudgetSoftGate(
            requires_approval=False,
            reason="change budget preview missing",
            action="allow",
            warnings=["change budget preview missing"],
        )
    if preview.would_exceed_budget:
        return NativeChangeBudgetSoftGate(
            requires_approval=True,
            reason="proposal exceeds advisory change budget",
            action="require_approval",
            warnings=list(preview.warnings),
        )
    return NativeChangeBudgetSoftGate(
        requires_approval=False,
        reason="proposal is within advisory change budget",
        action="allow",
    )


@dataclass
class NativeApprovalRequest:
    source: str = ""
    requires_approval: bool = False
    reason: str = ""
    action: str = "allow"
    proposed_files: int = 0
    budget_max_files: int = 0
    prompt: str = ""
    warnings: list[str] = field(default_factory=list)


@dataclass
class NativeApprovalReceipt:
    source: str = ""
    requested: bool = False
    granted: bool = False
    action: str = "allow"
    reason: str = ""


def build_native_approval_receipt(
    request: NativeApprovalRequest | None,
    *,
    granted: bool,
) -> NativeApprovalReceipt:
    if request is None:
        return NativeApprovalReceipt(
            source="",
            requested=False,
            granted=granted,
            action="allow",
            reason="approval request missing",
        )
    return NativeApprovalReceipt(
        source=request.source,
        requested=request.requires_approval,
        granted=granted,
        action=request.action,
        reason=request.reason,
    )


def build_native_budget_gate_approval_request(
    *,
    gate: NativeChangeBudgetSoftGate | None,
    preview: NativeChangeBudgetPreview | None,
) -> NativeApprovalRequest:
    if gate is None:
        return NativeApprovalRequest(
            source="change_budget_soft_gate",
            requires_approval=False,
            reason="change budget soft gate missing",
            action="allow",
            warnings=["change budget soft gate missing"],
        )

    proposed_files = preview.proposed_files if preview is not None else 0
    budget_max_files = preview.budget_max_files if preview is not None else 0

    if not gate.requires_approval:
        return NativeApprovalRequest(
            source="change_budget_soft_gate",
            requires_approval=False,
            reason=gate.reason,
            action=gate.action,
            proposed_files=proposed_files,
            budget_max_files=budget_max_files,
        )

    prompt = (
        "Proposal exceeds advisory change budget: "
        f"{proposed_files} files proposed, budget allows {budget_max_files}. Proceed?"
    )

    return NativeApprovalRequest(
        source="change_budget_soft_gate",
        requires_approval=True,
        reason=gate.reason,
        action=gate.action,
        proposed_files=proposed_files,
        budget_max_files=budget_max_files,
        prompt=prompt,
        warnings=list(gate.warnings),
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


_ALLOWED_VERIFICATION_COMMANDS: list[str] = [
    "pytest",
    "npm test",
    "cargo test",
    "go test",
    "rspec",
    "mvn test",
]

_BLOCKED_EXEC_COMMANDS: list[str] = [
    "curl", "wget", "rm", "sudo", "chmod", "chown",
]

_TASK_TYPE_KEYWORDS: dict[str, list[str]] = {
    "test": ["test", "tests", "spec", "unittest"],
    "bugfix": ["fix", "bug", "patch", "broken", "crash", "error", "issue"],
    "refactor": ["refactor", "clean", "reorganize", "restructure", "rename"],
    "feature": ["add", "implement", "create", "build", "new", "introduce"],
    "docs": ["docs", "document", "readme"],
    "config": ["config", "setting", "env", "configure"],
}

_SUCCESS_CRITERIA_BY_TYPE: dict[str, list[str]] = {
    "test": ["test suite passes", "verification passed"],
    "bugfix": ["verification passed", "target file changed"],
    "refactor": ["verification passed", "files within budget", "no new failures"],
    "feature": ["verification passed", "files within budget"],
    "docs": ["files within budget"],
    "config": ["verification passed", "files within budget"],
    "unknown": ["verification passed", "files within budget"],
}


@dataclass
class NativeVerificationPlan:
    task_type: str = "unknown"
    risk_level: str = "unknown"
    likely_files_or_folders: list[str] = field(default_factory=list)
    allowed_commands: list[str] = field(default_factory=list)
    blocked_commands: list[str] = field(default_factory=list)
    suggested_verification_commands: list[str] = field(default_factory=list)
    approval_rules: list[str] = field(default_factory=list)
    success_criteria: list[str] = field(default_factory=list)
    failure_handling: str = "halt and report"
    clarification_needed: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


@dataclass
class NativeClarificationRequest:
    needed: bool = False
    question: str | None = None
    options: list[str] = field(default_factory=list)
    allows_custom: bool = False
    reason: str | None = None
    task_field: str | None = None


@dataclass
class NativeContextUsageSummary:
    repo_summary_included: bool = False
    selected_files_count: int = 0
    compact_paths_count: int = 0
    evidence_items_count: int = 0
    snippet_count: int = 0
    failure_warning_count: int = 0
    any_truncated: bool = False
    truncated_components: list[str] = field(default_factory=list)
    total_chars: int = 0
    compacted: bool = False


def build_native_context_usage_summary(
    *,
    repo_context_summary: Any | None,
    file_context: Any | None,
    context_packet: Any | None,
    evidence: Any | None,
    observation: Any | None,
    plan: Any | None,
    context_quality_score: Any | None,
    final_report: Any | None,
    diff_review: Any | None,
    verification_loop: Any | None,
    total_chars: int = 0,
) -> NativeContextUsageSummary:
    repo_summary_included = repo_context_summary is not None
    selected_files_count = getattr(file_context, "files_read", 0) if file_context is not None else 0
    compact_paths = getattr(context_packet, "compact_paths", []) or [] if context_packet is not None else []
    compact_paths_count = len(compact_paths)
    evidence_items_count = len(getattr(evidence, "search_results", []) or []) if evidence is not None else 0
    snippet_count = len(getattr(evidence, "file_snippets", []) or []) if evidence is not None else 0

    failure_warning_count = 0
    for src in (observation, plan, context_packet, file_context, context_quality_score, final_report):
        if src is not None:
            failure_warning_count += len(getattr(src, "warnings", []) or [])

    truncated_components: list[str] = []
    for name, obj in (
        ("evidence", evidence),
        ("diff_review", diff_review),
        ("verification_loop", verification_loop),
        ("file_context", file_context),
    ):
        if obj is not None and getattr(obj, "truncated", False):
            truncated_components.append(name)
    any_truncated = bool(truncated_components)
    compacted = any_truncated or compact_paths_count >= 8

    return NativeContextUsageSummary(
        repo_summary_included=repo_summary_included,
        selected_files_count=selected_files_count,
        compact_paths_count=compact_paths_count,
        evidence_items_count=evidence_items_count,
        snippet_count=snippet_count,
        failure_warning_count=failure_warning_count,
        any_truncated=any_truncated,
        truncated_components=truncated_components,
        total_chars=total_chars,
        compacted=compacted,
    )


@dataclass
class FailureLesson:
    lesson_type: str = ""
    reason: str = ""


@dataclass
class NativeFailureMemory:
    lessons: list[FailureLesson] = field(default_factory=list)
    has_lessons: bool = False


def build_native_failure_memory(
    *,
    context_quality_score: Any | None,
    clarification_request: Any | None,
    verification_loop: Any | None,
    command_policy_preview: Any | None,
    approval_request: Any | None,
    approval_receipt: Any | None,
    change_budget_preview: Any | None,
    verification_plan: Any | None,
    context_usage_summary: Any | None,
) -> NativeFailureMemory:
    lessons: list[FailureLesson] = []

    if context_quality_score is not None and getattr(context_quality_score, "level", "") == "weak":
        score = getattr(context_quality_score, "score", 0)
        lessons.append(FailureLesson(
            lesson_type="weak_context",
            reason=f"context quality was weak (score {score}/100)",
        ))

    if clarification_request is not None and getattr(clarification_request, "needed", False):
        lessons.append(FailureLesson(
            lesson_type="unknown_task_type",
            reason="task type was unknown or ambiguous",
        ))

    if (
        verification_loop is not None
        and getattr(verification_loop, "attempted", False)
        and not getattr(verification_loop, "passed", True)
    ):
        lessons.append(FailureLesson(
            lesson_type="failed_verification",
            reason="verification ran but did not pass",
        ))

    if command_policy_preview is not None:
        blocked = getattr(command_policy_preview, "blocked_count", 0)
        if blocked > 0:
            lessons.append(FailureLesson(
                lesson_type="unsafe_command",
                reason=f"{blocked} blocked command(s) detected",
            ))

    if approval_request is not None and getattr(approval_request, "requires_approval", False):
        approval_reason = getattr(approval_request, "reason", "") or "approval was required"
        lessons.append(FailureLesson(
            lesson_type="approval_required",
            reason=approval_reason,
        ))

        if approval_receipt is not None and not getattr(approval_receipt, "granted", True):
            lessons.append(FailureLesson(
                lesson_type="approval_rejected",
                reason="approval was requested but not granted",
            ))

    if change_budget_preview is not None and getattr(change_budget_preview, "would_exceed_budget", False):
        proposed = getattr(change_budget_preview, "proposed_files", 0)
        budget = getattr(change_budget_preview, "budget_max_files", 0)
        lessons.append(FailureLesson(
            lesson_type="patch_too_broad",
            reason=f"{proposed} files proposed but budget allows {budget}",
        ))

    if verification_plan is not None:
        cmds = getattr(verification_plan, "suggested_verification_commands", None) or []
        if not cmds:
            lessons.append(FailureLesson(
                lesson_type="missing_verification",
                reason="no verification commands available",
            ))

    if context_usage_summary is not None and getattr(context_usage_summary, "any_truncated", False):
        components = getattr(context_usage_summary, "truncated_components", []) or []
        component_str = ", ".join(components) if components else "unknown"
        lessons.append(FailureLesson(
            lesson_type="context_truncated",
            reason=f"context was truncated: {component_str}",
        ))

    if context_usage_summary is not None:
        warn_count = getattr(context_usage_summary, "failure_warning_count", 0)
        if warn_count > 0:
            lessons.append(FailureLesson(
                lesson_type="warnings_present",
                reason=f"{warn_count} warning(s) accumulated during run",
            ))

    return NativeFailureMemory(lessons=lessons, has_lessons=bool(lessons))


def render_native_context_usage_summary(meta: NativeContextUsageSummary | None) -> str:
    if meta is None:
        return ""
    lines = ["[context usage summary]"]
    lines.append(f"repo summary: {'yes' if meta.repo_summary_included else 'no'}")
    lines.append(f"files: {meta.selected_files_count}")
    lines.append(f"compact paths: {meta.compact_paths_count}")
    lines.append(f"evidence: {meta.evidence_items_count} items, {meta.snippet_count} snippets")
    lines.append(f"warnings: {meta.failure_warning_count}")
    if meta.any_truncated and meta.truncated_components:
        lines.append(f"truncated: yes ({', '.join(meta.truncated_components)})")
    else:
        lines.append("truncated: no")
    lines.append(f"total chars: {meta.total_chars}")
    lines.append(f"compacted: {'yes' if meta.compacted else 'no'}")
    return "\n".join(lines)


def build_native_verification_plan(
    task: str,
    plan: NativePlan | None,
    change_budget: NativeChangeBudget | None,
    read_search_findings: list[str],
    repo_facts: Any | None,
) -> NativeVerificationPlan:
    task_lower = task.lower()
    task_type = "unknown"
    for t_type, keywords in _TASK_TYPE_KEYWORDS.items():
        if any(kw in task_lower for kw in keywords):
            task_type = t_type
            break

    risk_level = getattr(plan, "risk", "unknown") if plan is not None else "unknown"

    seen: set[str] = set()
    likely: list[str] = []
    for finding in read_search_findings:
        path: str | None = None
        if finding.startswith("file:"):
            path = finding[len("file:"):]
        elif finding.startswith("test-marker:"):
            path = finding[len("test-marker:"):]
        if path and path not in seen:
            seen.add(path)
            likely.append(path)
            if len(likely) >= 5:
                break

    test_cmd = getattr(repo_facts, "test_command", None) if repo_facts is not None else None
    suggested: list[str] = [test_cmd.strip()] if (test_cmd and isinstance(test_cmd, str) and test_cmd.strip()) else []

    approval_rules: list[str] = [
        "blocked commands require approval",
        "shell metacharacters require approval",
    ]
    if change_budget is not None and getattr(change_budget, "max_files", 0):
        approval_rules.insert(0, f"changes exceeding {change_budget.max_files} file(s) require approval")

    success_criteria = list(_SUCCESS_CRITERIA_BY_TYPE.get(task_type, _SUCCESS_CRITERIA_BY_TYPE["unknown"]))

    clarification_needed: list[str] = []
    if task_type == "unknown":
        clarification_needed = ["task type is ambiguous — no recognizable action keyword found"]

    return NativeVerificationPlan(
        task_type=task_type,
        risk_level=risk_level,
        likely_files_or_folders=likely,
        allowed_commands=list(_ALLOWED_VERIFICATION_COMMANDS),
        blocked_commands=list(_BLOCKED_EXEC_COMMANDS),
        suggested_verification_commands=suggested,
        approval_rules=approval_rules,
        success_criteria=success_criteria,
        failure_handling="halt and report",
        clarification_needed=clarification_needed,
    )


def build_native_clarification_request(
    task: str,
    verification_plan: NativeVerificationPlan,
) -> NativeClarificationRequest:
    if verification_plan.task_type != "unknown" and not verification_plan.clarification_needed:
        return NativeClarificationRequest(needed=False)

    reason = verification_plan.clarification_needed[0] if verification_plan.clarification_needed else None
    return NativeClarificationRequest(
        needed=True,
        question="What type of change is this task requesting?",
        options=[
            "Feature (add / implement something new)",
            "Bugfix (fix / patch a problem)",
            "Refactor (clean / reorganize existing code)",
            "Test (add or update tests)",
            "Documentation",
            "Configuration / environment change",
        ],
        allows_custom=True,
        reason=reason,
        task_field="task_type",
    )


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
