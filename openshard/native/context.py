from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from openshard.native.dispatch import (
    _ROLE_DEFAULT_TIER as _DEFAULT_ROLE_TIERS,
)
from openshard.native.dispatch import (
    _ROLE_VALID_TIERS as _VALID_TIERS_BY_ROLE,
)


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
class RetryMetadata:
    retry_attempted: bool = False
    retry_reason: str = ""
    failure_summary: str = ""
    retry_patch_files: list[str] = field(default_factory=list)
    retry_verification_status: str = ""
    raw_content_stored: bool = False

    def __post_init__(self) -> None:
        self.raw_content_stored = False


_MAX_CHECK_NAMES: int = 8


@dataclass
class NativeVerificationLoop:
    attempted: bool = False
    passed: bool = False
    retried: bool = False
    exit_code: int | None = None
    output_chars: int = 0
    truncated: bool = False
    duration_seconds: float | None = None
    retry_metadata: RetryMetadata | None = None
    # Per-check proof wiring (v1: one command per plan)
    check_attempted: list[str] = field(default_factory=list)
    check_passed: list[str] = field(default_factory=list)
    check_failed: list[str] = field(default_factory=list)
    check_skipped: list[str] = field(default_factory=list)
    check_skipped_reasons: list[str] = field(default_factory=list)


@dataclass
class NativeEditLoopAttempt:
    attempt_index: int = 0
    purpose: str = ""              # "initial" | "repair"
    files_written: list[str] = field(default_factory=list)
    verification_status: str = ""  # "passed" | "failed" | "skipped"
    exit_code: int | None = None
    output_chars: int = 0
    raw_content_stored: bool = False

    def __post_init__(self) -> None:
        self.raw_content_stored = False


@dataclass
class NativeEditLoopSummary:
    enabled: bool = True
    max_attempts: int = 2
    attempts: list[NativeEditLoopAttempt] = field(default_factory=list)
    completed: bool = False
    final_status: str = ""         # "passed" | "failed" | "skipped"
    repair_used: bool = False
    raw_content_stored: bool = False

    def __post_init__(self) -> None:
        self.raw_content_stored = False


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
    command_records: list[dict] = field(default_factory=list)


def build_native_command_policy_preview(
    verification_plan: Any | None,
) -> NativeCommandPolicyPreview:
    if verification_plan is None or not hasattr(verification_plan, "commands"):
        return NativeCommandPolicyPreview()
    safe = needs_approval = blocked = 0
    classes: set[str] = set()
    records: list[dict] = []
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
        records.append({
            "command": " ".join(getattr(cmd, "argv", [])),
            "classification": label,
            "decision_reason": getattr(cmd, "reason", ""),
            "raw_content_stored": False,
        })
    warnings: list[str] = []
    if safe + needs_approval + blocked == 0:
        warnings.append("no commands with safety classification found")
    return NativeCommandPolicyPreview(
        safe_count=safe,
        needs_approval_count=needs_approval,
        blocked_count=blocked,
        command_classes=sorted(classes),
        warnings=warnings,
        command_records=records,
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


def render_native_evidence(
    evidence: NativeEvidence,
    *,
    limit: int = 1000,
    max_lines_per_snippet: int = 8,
) -> str:
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
            for line in snippet.lines[:max_lines_per_snippet]:
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


_FAILURE_PATTERNS: list[tuple[str, str]] = [
    ("syntaxerror", "syntax_error"),
    ("importerror", "import_error"),
    ("modulenotfounderror", "import_error"),
    ("assertionerror", "assertion_error"),
    ("typeerror", "type_error"),
    ("nameerror", "name_error"),
    ("attributeerror", "attribute_error"),
]


def build_failure_summary(output: str, exit_code: int) -> str:
    """Structured failure summary — no raw content stored."""
    lo = output.lower()
    failure_type = "test_failure"
    for token, label in _FAILURE_PATTERNS:
        if token in lo:
            failure_type = label
            break
    return (
        f"exit_code={exit_code} "
        f"failure_type={failure_type} "
        f"output_chars={len(output)} "
        f"raw_content_stored=false"
    )


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


_VALID_OSN_STOP_REASONS: frozenset[str] = frozenset({
    "completed",
    "max_steps",
    "no_steps",
    "blocked_tool",
    "approval_required",
    "verification_failed",
    "tool_error",
    "empty_response_limit",
    "retry_limit",
    "policy_denied",
    "unknown",
})

_OSN_STOP_REASON_ALIASES: dict[str, str] = {
    "complete": "completed",
    "consecutive_empty": "empty_response_limit",
    "max steps reached": "max_steps",
}

_MAX_STEP_EVENTS_RECORDED: int = 50
_MAX_REPEATED_BLOCKED_TOOL: int = 3
_MAX_RETRY_COUNT: int = 1


def normalize_osn_stop_reason(raw: str | None) -> str:
    """Map any raw or legacy stop reason string to the canonical allowed set.

    Handles None, empty string, whitespace, and mixed-case legacy values safely.
    """
    if not raw or not raw.strip():
        return "unknown"
    normalized = raw.strip().lower()
    normalized = _OSN_STOP_REASON_ALIASES.get(normalized, normalized)
    return normalized if normalized in _VALID_OSN_STOP_REASONS else "unknown"


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


@dataclass
class NativeOSNLoopStep:
    step_index: int = 0
    step_name: str = ""  # preflight|observe|gather_context|plan_update|generate_patch|budget_check|approval|safe_write|verify|retry_once|final_receipt
    status: str = "pending"  # pending|running|skipped|passed|failed|blocked
    tool_name: str = ""
    target_label: str = ""  # capped at 40 chars — no absolute paths
    reason: str = ""
    result_summary: str = ""  # max 120 chars — no raw content ever
    blocked_reason: str = ""
    context_injected: bool = False
    approval_required: bool = False
    verification_status: str = ""
    warnings: list[str] = field(default_factory=list)
    raw_content_stored: bool = False

    def __post_init__(self) -> None:
        self.raw_content_stored = False  # enforced — never store raw content
        if len(self.target_label) > 40:
            self.target_label = self.target_label[:40]


@dataclass
class NativeOSNLoopSummary:
    enabled: bool = False
    mode: str = ""  # "experimental"
    max_steps: int = 11
    steps_taken: int = 0
    completed: bool = False
    stopped_reason: str = ""
    verification_status: str = ""
    retry_used: bool = False
    approval_required: bool = False
    approval_granted: bool = False
    warnings: list[str] = field(default_factory=list)
    steps: list[NativeOSNLoopStep] = field(default_factory=list)
    # explicit counters - added in osn-loop-hardening-v1
    loop_id: str = ""
    attempted_steps: int = 0
    completed_steps: int = 0
    failed_steps: int = 0
    blocked_steps: int = 0
    tool_calls_attempted: int = 0
    tool_calls_completed: int = 0
    tool_calls_blocked: int = 0
    verification_attempted: bool = False
    verification_passed: bool | None = None
    retry_count: int = 0
    final_status: str = ""


# Caps for OSNObservationPacket fields - enforced in __post_init__
_OBS_MAX_CANDIDATE_FILES: int = 10
_OBS_MAX_CONFIG_FILES: int = 8
_OBS_MAX_TEST_FILES: int = 8
_OBS_MAX_RISKY_MARKERS: int = 6
_OBS_MAX_SUGGESTED_CHECKS: int = 6
_OBS_MAX_SUMMARY_CHARS: int = 200


@dataclass
class OSNObservationPacket:
    """Bounded, safe repo observation packet populated before OSN planning.

    Never stores raw file contents, absolute paths, raw prompts, or command output.
    All lists are capped in __post_init__.
    """
    enabled: bool = False
    repo_root_present: bool = False
    stack_signals: list[str] = field(default_factory=list)
    candidate_files: list[str] = field(default_factory=list)
    config_files: list[str] = field(default_factory=list)
    test_files: list[str] = field(default_factory=list)
    risky_markers: list[str] = field(default_factory=list)
    suggested_checks: list[str] = field(default_factory=list)
    observation_summary: str = ""
    files_considered: int = 0
    files_capped: bool = False
    source: str = "repo_observation_v1"

    def __post_init__(self) -> None:
        self.candidate_files = self.candidate_files[:_OBS_MAX_CANDIDATE_FILES]
        self.config_files = self.config_files[:_OBS_MAX_CONFIG_FILES]
        self.test_files = self.test_files[:_OBS_MAX_TEST_FILES]
        self.risky_markers = self.risky_markers[:_OBS_MAX_RISKY_MARKERS]
        self.suggested_checks = self.suggested_checks[:_OBS_MAX_SUGGESTED_CHECKS]
        if len(self.observation_summary) > _OBS_MAX_SUMMARY_CHARS:
            self.observation_summary = self.observation_summary[:_OBS_MAX_SUMMARY_CHARS]


def render_osn_observation_context(packet: OSNObservationPacket | None) -> str:
    """Prompt-safe compact observation block. No raw content, no file contents, no absolute paths."""
    if packet is None or not packet.enabled:
        return ""
    lines = ["[repo observation]"]
    if packet.stack_signals:
        lines.append(f"stack: {', '.join(packet.stack_signals)}")
    if packet.candidate_files:
        lines.append(f"candidates: {', '.join(packet.candidate_files[:5])}")
    if packet.test_files:
        lines.append(f"tests: {', '.join(packet.test_files[:3])}")
    if packet.suggested_checks:
        lines.append(f"checks: {', '.join(packet.suggested_checks[:3])}")
    return "\n".join(lines)


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
class NativeValidationContract:
    intent: str = ""
    risk_level: str = "unknown"
    expected_change_scope: str = "unknown"
    acceptance_checks: list[str] = field(default_factory=list)
    verification_commands: list[str] = field(default_factory=list)
    approval_expected: bool = False
    strength: str = "weak"  # weak | fair | strong
    warnings: list[str] = field(default_factory=list)


@dataclass
class NativeContractCheckResult:
    check_id: str = ""
    expected_check: str = ""
    verification_source: str = "none"  # "verification_loop" | "none"
    status: str = "unknown"  # passed | failed | skipped | unknown
    reason: str = ""
    evidence_summary: str = ""
    raw_content_stored: bool = False


@dataclass
class NativeVerificationContractResult:
    checks: list[NativeContractCheckResult] = field(default_factory=list)
    overall_status: str = "unknown"  # passed | failed | skipped | unknown
    reason: str = ""
    raw_content_stored: bool = False


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
    if change_budget is not None and getattr(change_budget, "max_files", 0):  # type: ignore[arg-type]  # getattr default is int, not bool
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


def build_native_validation_contract(
    *,
    task: str,
    plan: Any | None,
    verification_plan: Any | None,
    change_budget: Any | None,
    change_budget_preview: Any | None,
    change_budget_soft_gate: Any | None,
    clarification_request: Any | None,
    context_quality_score: Any | None,
) -> NativeValidationContract:
    intent = (task or "").strip()[:120]

    if plan is not None and getattr(plan, "risk", None):
        risk_level = plan.risk
    elif verification_plan is not None and getattr(verification_plan, "risk_level", None):
        risk_level = verification_plan.risk_level
    else:
        risk_level = "unknown"

    if change_budget is not None:
        expected_change_scope = f"{change_budget.max_files} files expected"
    elif change_budget_preview is not None:
        proposed = getattr(change_budget_preview, "proposed_files", 0)
        expected_change_scope = f"{proposed} files proposed"
    else:
        expected_change_scope = "unknown"

    seen: set[str] = set()
    acceptance_checks: list[str] = []
    for item in (getattr(verification_plan, "success_criteria", None) or []):
        if item and item not in seen:
            seen.add(item)
            acceptance_checks.append(item)
    for step in (getattr(plan, "suggested_steps", None) or [])[:3]:
        if step and step not in seen:
            seen.add(step)
            acceptance_checks.append(step)

    verification_commands = list(
        (getattr(verification_plan, "suggested_verification_commands", None) or [])[:5]
    )

    approval_expected = False
    if change_budget_soft_gate is not None and getattr(change_budget_soft_gate, "requires_approval", False):
        approval_expected = True
    if change_budget_preview is not None and getattr(change_budget_preview, "would_exceed_budget", False):
        approval_expected = True

    clarification_needed = (
        clarification_request is not None and getattr(clarification_request, "needed", False)
    )

    if acceptance_checks and verification_commands and not clarification_needed:
        strength = "strong"
    elif acceptance_checks and (not verification_commands or clarification_needed):
        strength = "fair"
    else:
        strength = "weak"

    warnings: list[str] = []
    if clarification_needed:
        warnings.append("clarification needed before proceeding")
    if not acceptance_checks:
        warnings.append("no acceptance checks derived from plan")
    if not verification_commands:
        warnings.append("no verification commands available")

    return NativeValidationContract(
        intent=intent,
        risk_level=risk_level,
        expected_change_scope=expected_change_scope,
        acceptance_checks=acceptance_checks,
        verification_commands=verification_commands,
        approval_expected=approval_expected,
        strength=strength,
        warnings=warnings,
    )


def render_native_validation_contract(contract: NativeValidationContract | None) -> str:
    if contract is None:
        return ""
    lines = ["[validation contract]"]
    lines.append(f"intent: {contract.intent}")
    lines.append(f"risk: {contract.risk_level}")
    lines.append(f"scope: {contract.expected_change_scope}")
    if contract.acceptance_checks:
        lines.append("acceptance checks:")
        for check in contract.acceptance_checks:
            lines.append(f"  - {check}")
    if contract.verification_commands:
        lines.append("verification:")
        for cmd in contract.verification_commands:
            lines.append(f"  - {cmd}")
    lines.append(f"approval expected: {'yes' if contract.approval_expected else 'no'}")
    lines.append(f"strength: {contract.strength}")
    return "\n".join(lines)


def build_native_verification_contract_result(
    *,
    validation_contract: NativeValidationContract | None,
    verification_loop: Any | None,
) -> NativeVerificationContractResult:
    if validation_contract is None:
        return NativeVerificationContractResult(overall_status="unknown", reason="no validation contract")

    checks_text: list[str] = list(getattr(validation_contract, "acceptance_checks", None) or [])
    if not checks_text:
        return NativeVerificationContractResult(overall_status="unknown", reason="no acceptance checks defined")

    v_attempted = verification_loop is not None and getattr(verification_loop, "attempted", False)
    v_passed = v_attempted and getattr(verification_loop, "passed", False)
    exit_code = getattr(verification_loop, "exit_code", None) if verification_loop is not None else None
    output_chars = getattr(verification_loop, "output_chars", 0) if verification_loop is not None else 0

    if not v_attempted:
        status = "skipped"
        reason = "verification not attempted"
        source = "none"
        evidence = ""
        overall = "skipped"
    elif v_passed:
        status = "passed"
        reason = "verification suite passed"
        source = "verification_loop"
        _ec = f"exit_code={exit_code}" if exit_code is not None else "exit_code=0"
        evidence = f"{_ec}, {output_chars} chars output"
        overall = "passed"
    else:
        status = "failed"
        reason = "verification suite failed"
        source = "verification_loop"
        _ec = f"exit_code={exit_code}" if exit_code is not None else "exit_code=nonzero"
        evidence = f"{_ec}, {output_chars} chars output"
        overall = "failed"

    checks = [
        NativeContractCheckResult(
            check_id=f"check_{i}",
            expected_check=text,
            verification_source=source,
            status=status,
            reason=reason,
            evidence_summary=evidence,
            raw_content_stored=False,
        )
        for i, text in enumerate(checks_text)
    ]
    return NativeVerificationContractResult(
        checks=checks,
        overall_status=overall,
        reason=reason,
        raw_content_stored=False,
    )


@dataclass
class NativeContextSource:
    name: str = ""
    used: bool = False
    injected: bool = False
    item_count: int = 0
    summary: str = ""


@dataclass
class NativeContextProvenance:
    sources: list[NativeContextSource] = field(default_factory=list)
    injected_sources: int = 0
    used_sources: int = 0
    total_items: int = 0
    has_gaps: bool = False
    warnings: list[str] = field(default_factory=list)


def build_native_context_provenance(
    *,
    repo_context_summary: Any | None = None,
    observation: Any | None = None,
    evidence: Any | None = None,
    read_search_findings: list[str] | None = None,
    file_context: Any | None = None,
    context_packet: Any | None = None,
    context_quality_score: Any | None = None,
    context_quality_advisory: Any | None = None,
    change_budget: Any | None = None,
    plan: Any | None = None,
    verification_plan: Any | None = None,
    clarification_request: Any | None = None,
    validation_contract: Any | None = None,
    context_usage_summary: Any | None = None,
    osn_loop: Any | None = None,
    skills_context: str | None = None,
    injected_source_names: set[str] | None = None,
) -> NativeContextProvenance:
    _inj = injected_source_names or set()

    def _src(name: str, used: bool, item_count: int, summary: str) -> NativeContextSource:
        return NativeContextSource(
            name=name,
            used=used,
            injected=name in _inj,
            item_count=item_count,
            summary=summary,
        )

    sources: list[NativeContextSource] = []

    # repo_summary
    sources.append(_src(
        "repo_summary",
        repo_context_summary is not None,
        1 if repo_context_summary is not None else 0,
        "1 summary" if repo_context_summary is not None else "",
    ))

    # observation
    _obs_tools = len(getattr(observation, "observed_tools", []) or []) if observation is not None else 0
    sources.append(_src(
        "observation",
        observation is not None,
        _obs_tools,
        f"{_obs_tools} tools" if observation is not None else "",
    ))

    # evidence
    _ev_items = 0
    if evidence is not None:
        _ev_items = (
            len(getattr(evidence, "file_snippets", []) or [])
            + len(getattr(evidence, "search_results", []) or [])
        )
    sources.append(_src(
        "evidence",
        evidence is not None,
        _ev_items,
        f"{_ev_items} items" if evidence is not None else "",
    ))

    # read_search
    _rs = list(read_search_findings) if read_search_findings else []
    sources.append(_src(
        "read_search",
        bool(_rs),
        len(_rs),
        f"{len(_rs)} findings" if _rs else "",
    ))

    # file_context
    _fc_files = getattr(file_context, "files_read", 0) if file_context is not None else 0
    sources.append(_src(
        "file_context",
        file_context is not None,
        _fc_files,
        f"{_fc_files} files" if file_context is not None else "",
    ))

    # context_packet
    _cp_sources = len(getattr(context_packet, "sources", []) or []) if context_packet is not None else 0
    sources.append(_src(
        "context_packet",
        context_packet is not None,
        _cp_sources,
        f"{_cp_sources} sources" if context_packet is not None else "",
    ))

    # context_quality
    _cq_level = getattr(context_quality_score, "level", "") if context_quality_score is not None else ""
    sources.append(_src(
        "context_quality",
        context_quality_score is not None,
        1 if context_quality_score is not None else 0,
        f"level={_cq_level}" if context_quality_score is not None else "",
    ))

    # advisory
    sources.append(_src(
        "advisory",
        context_quality_advisory is not None,
        1 if context_quality_advisory is not None else 0,
        "1 advisory" if context_quality_advisory is not None else "",
    ))

    # change_budget
    sources.append(_src(
        "change_budget",
        change_budget is not None,
        1 if change_budget is not None else 0,
        "1 budget" if change_budget is not None else "",
    ))

    # plan
    _plan_steps = len(getattr(plan, "suggested_steps", []) or []) if plan is not None else 0
    _plan_items = _plan_steps if _plan_steps > 0 else (1 if plan is not None else 0)
    sources.append(_src(
        "plan",
        plan is not None,
        _plan_items,
        f"{_plan_items} steps" if plan is not None else "",
    ))

    # verification_plan
    _vp_cmds = (
        len(getattr(verification_plan, "suggested_verification_commands", []) or [])
        if verification_plan is not None
        else 0
    )
    sources.append(_src(
        "verification_plan",
        verification_plan is not None,
        _vp_cmds,
        f"{_vp_cmds} commands" if verification_plan is not None else "",
    ))

    # clarification_request
    _cr_needed = getattr(clarification_request, "needed", False) if clarification_request is not None else False
    sources.append(_src(
        "clarification_request",
        bool(_cr_needed),
        1 if _cr_needed else 0,
        "1 request" if _cr_needed else "",
    ))

    # validation_contract
    _vc_checks = (
        len(getattr(validation_contract, "acceptance_checks", []) or [])
        if validation_contract is not None
        else 0
    )
    sources.append(_src(
        "validation_contract",
        validation_contract is not None,
        _vc_checks,
        f"{_vc_checks} checks" if validation_contract is not None else "",
    ))

    # context_usage_summary
    sources.append(_src(
        "context_usage_summary",
        context_usage_summary is not None,
        1 if context_usage_summary is not None else 0,
        "1 summary" if context_usage_summary is not None else "",
    ))

    # osn_loop
    _osn_enabled = getattr(osn_loop, "enabled", False) if osn_loop is not None else False
    _osn_steps = getattr(osn_loop, "steps_run", 0) if osn_loop is not None else 0
    sources.append(_src(
        "osn_loop",
        bool(_osn_enabled),
        _osn_steps,
        f"{_osn_steps} steps" if _osn_enabled else "",
    ))

    # skills_context
    _sc_used = bool(skills_context)
    sources.append(_src(
        "skills_context",
        _sc_used,
        1 if _sc_used else 0,
        "1 context" if _sc_used else "",
    ))

    # aggregate counts
    used_sources = sum(1 for s in sources if s.used)
    injected_count = sum(1 for s in sources if s.injected)
    total_items = sum(s.item_count for s in sources)

    # has_gaps
    warnings: list[str] = []
    _cq_level_val = getattr(context_quality_score, "level", "") if context_quality_score is not None else ""
    if _cq_level_val in ("weak", "unknown") and context_quality_score is not None:
        warnings.append("context quality weak")
    _vc_strength = getattr(validation_contract, "strength", "") if validation_contract is not None else ""
    if _vc_strength == "weak":
        warnings.append("validation contract weak")
    if _cr_needed:
        warnings.append("clarification needed")
    if file_context is not None and getattr(file_context, "truncated", False):
        warnings.append("file context truncated")
    if context_usage_summary is not None and getattr(context_usage_summary, "any_truncated", False):
        warnings.append("context truncated")

    has_gaps = bool(warnings)

    return NativeContextProvenance(
        sources=sources,
        injected_sources=injected_count,
        used_sources=used_sources,
        total_items=total_items,
        has_gaps=has_gaps,
        warnings=warnings,
    )


def render_native_context_provenance(provenance: NativeContextProvenance | None) -> str:
    """Audit/display renderer — NOT injected into model prompt."""
    if provenance is None:
        return ""
    parts = [
        f"{provenance.used_sources} sources",
        f"{provenance.injected_sources} injected",
        f"{provenance.total_items} items",
    ]
    lines = ["context provenance: " + ", ".join(parts)]
    if provenance.has_gaps:
        lines.append(f"context provenance gaps: {len(provenance.warnings)} warnings")
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


@dataclass
class NativeRunTrustFactor:
    name: str = ""
    impact: int = 0
    reason: str = ""


@dataclass
class NativeRunTrustScore:
    score: int = 0
    level: str = "unknown"  # weak | fair | good | strong
    factors: list[NativeRunTrustFactor] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    blockers: list[str] = field(default_factory=list)


def build_native_run_trust_score(
    context_quality_score=None,
    validation_contract=None,
    context_provenance=None,
    verification_loop=None,
    command_policy_preview=None,
    change_budget_preview=None,
    change_budget_soft_gate=None,
    approval_request=None,
    approval_receipt=None,
    failure_memory=None,
    context_usage_summary=None,
    final_report=None,
) -> NativeRunTrustScore:
    score = 50
    factors: list[NativeRunTrustFactor] = []
    warnings: list[str] = []
    blockers: list[str] = []

    v_attempted = verification_loop is not None and getattr(verification_loop, "attempted", False)
    v_passed = v_attempted and getattr(verification_loop, "passed", False)
    if v_passed:
        factors.append(NativeRunTrustFactor(name="verification_passed", impact=20, reason="verification passed"))
        score += 20
    elif v_attempted:
        factors.append(NativeRunTrustFactor(name="verification_failed", impact=-30, reason="verification failed"))
        score -= 30
        warnings.append("verification failed")
        blockers.append("verification failed")
    else:
        factors.append(NativeRunTrustFactor(name="verification_not_attempted", impact=-10, reason="verification not attempted"))
        score -= 10
        warnings.append("verification not attempted")

    v_strength = getattr(validation_contract, "strength", None) if validation_contract is not None else None
    if v_strength == "strong":
        factors.append(NativeRunTrustFactor(name="validation_contract_strong", impact=15, reason="strong validation contract"))
        score += 15
    elif v_strength == "fair":
        factors.append(NativeRunTrustFactor(name="validation_contract_fair", impact=7, reason="fair validation contract"))
        score += 7
    elif v_strength == "weak":
        factors.append(NativeRunTrustFactor(name="validation_contract_weak", impact=-15, reason="weak validation contract"))
        score -= 15
        warnings.append("validation contract weak")

    cq_level = getattr(context_quality_score, "level", None) if context_quality_score is not None else None
    if cq_level in ("strong", "good"):
        factors.append(NativeRunTrustFactor(name="context_quality_good", impact=10, reason="good context quality"))
        score += 10
    elif cq_level in ("weak", "unknown"):
        factors.append(NativeRunTrustFactor(name="context_quality_weak", impact=-15, reason="weak context quality"))
        score -= 15
        warnings.append("context quality weak")

    if context_provenance is not None:
        p_injected = getattr(context_provenance, "injected_sources", 0)
        if p_injected > 0:
            factors.append(NativeRunTrustFactor(name="provenance_injected", impact=8, reason="injected context sources"))
            score += 8
        p_has_gaps = getattr(context_provenance, "has_gaps", False)
        if not p_has_gaps:
            factors.append(NativeRunTrustFactor(name="provenance_no_gaps", impact=5, reason="no provenance gaps"))
            score += 5
        else:
            factors.append(NativeRunTrustFactor(name="provenance_gaps", impact=-10, reason="provenance gaps detected"))
            score -= 10
            warnings.append("context provenance gaps")

    if command_policy_preview is not None:
        if getattr(command_policy_preview, "blocked_count", 0) > 0:
            factors.append(NativeRunTrustFactor(name="blocked_commands", impact=-25, reason="blocked commands detected"))
            score -= 25
            warnings.append("blocked commands detected")
            blockers.append("blocked commands detected")

    if change_budget_preview is not None:
        if getattr(change_budget_preview, "would_exceed_budget", False):
            factors.append(NativeRunTrustFactor(name="budget_exceeded", impact=-15, reason="change budget exceeded"))
            score -= 15
            warnings.append("change budget exceeded")
            blockers.append("change budget exceeded")

    req_approval = approval_request is not None and getattr(approval_request, "requires_approval", False)
    if req_approval:
        if approval_receipt is not None:
            if getattr(approval_receipt, "granted", False):
                factors.append(NativeRunTrustFactor(name="approval_granted", impact=5, reason="approval granted"))
                score += 5
            else:
                factors.append(NativeRunTrustFactor(name="approval_not_granted", impact=-30, reason="approval not granted"))
                score -= 30
                warnings.append("approval not granted")
                blockers.append("approval not granted")

    if failure_memory is not None and getattr(failure_memory, "has_lessons", False):
        _lessons = getattr(failure_memory, "lessons", []) or []
        _penalty = min(len(_lessons) * 5, 25)
        if _penalty > 0:
            factors.append(NativeRunTrustFactor(name="failure_lessons", impact=-_penalty, reason="failure lessons present"))
            score -= _penalty
            warnings.append("failure lessons present")

    if context_usage_summary is not None:
        if getattr(context_usage_summary, "any_truncated", False):
            factors.append(NativeRunTrustFactor(name="context_truncated", impact=-8, reason="context truncated"))
            score -= 8
            warnings.append("context truncated")
        _warn_count = getattr(context_usage_summary, "failure_warning_count", 0)
        if _warn_count > 0:
            _warn_penalty = min(_warn_count, 10)
            factors.append(NativeRunTrustFactor(name="context_warnings", impact=-_warn_penalty, reason="context warnings present"))
            score -= _warn_penalty
            warnings.append("context warnings present")

    if final_report is not None and getattr(final_report, "used_native_context", False):
        factors.append(NativeRunTrustFactor(name="used_native_context", impact=5, reason="used native context"))
        score += 5

    score = max(0, min(100, score))

    if score >= 85:
        level = "strong"
    elif score >= 70:
        level = "good"
    elif score >= 45:
        level = "fair"
    else:
        level = "weak"

    return NativeRunTrustScore(
        score=score,
        level=level,
        factors=factors,
        warnings=warnings,
        blockers=blockers,
    )


def render_native_run_trust_score(score: NativeRunTrustScore | None, detail: str = "compact") -> str:
    if score is None:
        return ""
    _score = getattr(score, "score", 0)
    _level = getattr(score, "level", "unknown")
    _warnings = getattr(score, "warnings", []) or []
    _blockers = getattr(score, "blockers", []) or []
    _factors = getattr(score, "factors", []) or []

    if detail == "full":
        lines = [
            "[run trust]",
            f"score: {_score}/100",
            f"level: {_level}",
        ]
        if _factors:
            lines.append("factors:")
            for _f in _factors:
                if isinstance(_f, dict):
                    _fn = _f.get("name", "")
                    _fi = _f.get("impact", 0)
                    _fr = _f.get("reason", "")
                else:
                    _fn = getattr(_f, "name", "")
                    _fi = getattr(_f, "impact", 0)
                    _fr = getattr(_f, "reason", "")
                _sign = "+" if _fi >= 0 else ""
                lines.append(f"  - {_fn} ({_sign}{_fi}): {_fr}")
        lines.append(f"warnings: {len(_warnings)}")
        lines.append(f"blockers: {len(_blockers)}")
        return "\n".join(lines)

    parts = [f"run trust: {_score}/100 {_level}"]
    if _warnings:
        _wc = len(_warnings)
        parts.append(f"run trust warnings: {_wc} {'warning' if _wc == 1 else 'warnings'}")
    if _blockers:
        _bc = len(_blockers)
        parts.append(f"run trust blockers: {_bc} {'blocker' if _bc == 1 else 'blockers'}")
    return "\n".join(parts)


@dataclass
class NativeModelRoleDecision:
    role: str = ""
    model_tier: str = ""
    cost_tier: str = ""
    reason: str = ""


@dataclass
class NativeModelSelectionDecision:
    strategy: str = "cost-balanced"
    task_type: str = "unknown"
    risk_level: str = "unknown"
    roles: list[NativeModelRoleDecision] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    fallback_reason: str = ""
    confidence: str = "medium"


def build_native_model_selection_decision(
    *,
    verification_plan=None,
    validation_contract=None,
    context_quality_score=None,
    context_provenance=None,
    run_trust_score=None,
    change_budget=None,
    failure_memory=None,
) -> NativeModelSelectionDecision:
    vc_strength = getattr(validation_contract, "strength", "") or ""
    vp_risk = getattr(verification_plan, "risk_level", "unknown") or "unknown"
    vc_risk = getattr(validation_contract, "risk_level", "unknown") or "unknown"
    rts_level = getattr(run_trust_score, "level", "unknown") or "unknown"
    cqs_level = getattr(context_quality_score, "level", "unknown") or "unknown"
    cp_has_gaps = getattr(context_provenance, "has_gaps", False)
    task_type = getattr(verification_plan, "task_type", "unknown") or "unknown"
    risk_level = vp_risk if vp_risk not in ("", "unknown") else vc_risk

    fallback_reason = ""

    if vc_strength == "weak" or vp_risk == "high" or rts_level == "weak":
        strategy = "frontier-heavy"
    elif cqs_level == "weak" or cp_has_gaps:
        strategy = "context-cautious"
    elif risk_level in ("low", "medium") and vc_strength in ("fair", "strong"):
        strategy = "cost-balanced"
    else:
        strategy = "cost-balanced"
        fallback_reason = "insufficient signal — defaulting to cost-balanced"

    if strategy == "frontier-heavy":
        planner_reason = "high risk requires strong reasoning model"
        executor_tier = "frontier-reasoning-model"
        executor_cost = "high"
        executor_reason = "high risk or weak signals require frontier execution"
    elif strategy == "context-cautious":
        planner_reason = "planning requires strong reasoning despite context gaps"
        executor_tier = "balanced-coding-model"
        executor_cost = "medium"
        executor_reason = "context gaps suggest balanced model for cautious execution"
    else:
        planner_reason = "planning always uses frontier reasoning"
        if task_type in ("docs", "test", "config"):
            executor_tier = "low-cost-coding-model"
            executor_cost = "low"
            executor_reason = "low-complexity task type allows low-cost model"
        else:
            executor_tier = "balanced-coding-model"
            executor_cost = "medium"
            executor_reason = "balanced cost and capability"

    planner_role = NativeModelRoleDecision(
        role="planner",
        model_tier="frontier-reasoning-model",
        cost_tier="high",
        reason=planner_reason,
    )
    executor_role = NativeModelRoleDecision(
        role="executor",
        model_tier=executor_tier,
        cost_tier=executor_cost,
        reason=executor_reason,
    )
    validator_role = NativeModelRoleDecision(
        role="validator",
        model_tier="independent-validator-model",
        cost_tier="medium",
        reason="independent validation required for all strategies",
    )

    warnings: list[str] = []
    if cqs_level == "weak":
        warnings.append("context quality is weak")
    if vc_strength == "weak":
        warnings.append("validation contract is weak — signals may be unreliable")
    if rts_level == "weak":
        warnings.append("run trust score is weak")
    if getattr(failure_memory, "has_lessons", False):
        warnings.append("failure memory has lessons that may affect model reliability")

    signal_count = sum([
        verification_plan is not None,
        validation_contract is not None,
        context_quality_score is not None,
        context_provenance is not None,
        run_trust_score is not None,
    ])

    if signal_count == 0:
        confidence = "low"
        if not fallback_reason:
            fallback_reason = "insufficient signal — defaulting to cost-balanced"
    elif signal_count <= 2:
        confidence = "medium"
    else:
        frontier_triggers = sum([
            vc_strength == "weak",
            vp_risk == "high",
            rts_level == "weak",
        ])
        if strategy == "cost-balanced" and frontier_triggers == 0:
            confidence = "high"
        elif strategy == "frontier-heavy" and frontier_triggers >= 2:
            confidence = "high"
        else:
            confidence = "medium"

    return NativeModelSelectionDecision(
        strategy=strategy,
        task_type=task_type,
        risk_level=risk_level,
        roles=[planner_role, executor_role, validator_role],
        warnings=warnings,
        fallback_reason=fallback_reason,
        confidence=confidence,
    )


def render_native_model_selection_decision(
    msd: NativeModelSelectionDecision | None,
    detail: str = "compact",
) -> str:
    if msd is None:
        return ""
    if detail not in ("compact", "more", "full"):
        return ""
    _strategy = getattr(msd, "strategy", "unknown")
    _confidence = getattr(msd, "confidence", "unknown")
    _roles = getattr(msd, "roles", []) or []
    _planner_tier = ""
    _executor_tier = ""
    for _r in _roles:
        _rname = _r.get("role", "") if isinstance(_r, dict) else getattr(_r, "role", "")
        _rtier = _r.get("model_tier", "") if isinstance(_r, dict) else getattr(_r, "model_tier", "")
        if _rname == "planner":
            _planner_tier = _rtier
        if _rname == "executor":
            _executor_tier = _rtier
    if detail in ("compact", "more"):
        _warnings = getattr(msd, "warnings", []) or []
        _base = (
            f"model selection: {_strategy}"
            f"  confidence={_confidence}"
            f"  planner={_planner_tier}"
            f"  executor={_executor_tier}"
        )
        if _warnings:
            return f"{_base}  warnings={len(_warnings)}"
        return _base
    lines = [
        "[model selection]",
        f"strategy: {_strategy}",
        f"task_type: {getattr(msd, 'task_type', 'unknown')}",
        f"risk_level: {getattr(msd, 'risk_level', 'unknown')}",
        f"confidence: {_confidence}",
    ]
    _fallback = getattr(msd, "fallback_reason", "")
    if _fallback:
        lines.append(f"fallback: {_fallback}")
    if _roles:
        lines.append("roles:")
        for _r in _roles:
            if isinstance(_r, dict):
                _rn = _r.get("role", "")
                _rt = _r.get("model_tier", "")
                _rc = _r.get("cost_tier", "")
                _rr = _r.get("reason", "")
            else:
                _rn = getattr(_r, "role", "")
                _rt = getattr(_r, "model_tier", "")
                _rc = getattr(_r, "cost_tier", "")
                _rr = getattr(_r, "reason", "")
            lines.append(f"  - {_rn}: {_rt} ({_rc}) — {_rr}")
    _warnings = getattr(msd, "warnings", []) or []
    lines.append(f"warnings: {len(_warnings)}")
    return "\n".join(lines)


@dataclass
class NativeModelCandidateScore:
    role: str = ""
    candidate: str = ""
    score: int = 0
    capability_score: int = 0
    risk_fit_score: int = 0
    context_fit_score: int = 0
    cost_score: int = 0
    verification_fit_score: int = 0
    penalty: int = 0
    reason: str = ""


@dataclass
class NativeModelCandidateScoring:
    candidates: list[NativeModelCandidateScore] = field(default_factory=list)
    selected_by_role: dict[str, str] = field(default_factory=dict)
    strategy: str = "cost-balanced"
    confidence: str = "medium"
    warnings: list[str] = field(default_factory=list)
    blocked_candidates: list[str] = field(default_factory=list)


_CANDIDATE_TIERS = [
    "frontier-reasoning-model",
    "balanced-coding-model",
    "low-cost-coding-model",
    "independent-validator-model",
]



def build_native_model_candidate_scoring(
    *,
    model_selection_decision=None,
    verification_plan=None,
    validation_contract=None,
    context_quality_score=None,
    context_provenance=None,
    run_trust_score=None,
    context_usage_summary=None,
    failure_memory=None,
    model_policy=None,
) -> NativeModelCandidateScoring:
    vc_strength = getattr(validation_contract, "strength", "") or ""
    rts_level = getattr(run_trust_score, "level", "") or ""
    cqs_level = getattr(context_quality_score, "level", "") or ""
    cp_has_gaps = getattr(context_provenance, "has_gaps", False)
    task_type = getattr(verification_plan, "task_type", "") or ""
    risk_level = getattr(verification_plan, "risk_level", "") or ""
    if not risk_level:
        risk_level = getattr(validation_contract, "risk_level", "") or ""
    has_lessons = getattr(failure_memory, "has_lessons", False)
    any_truncated = getattr(context_usage_summary, "any_truncated", False)
    verification_commands = getattr(verification_plan, "verification_commands", []) or []
    has_verification_cmds = len(verification_commands) > 0

    if model_selection_decision is not None:
        strategy = getattr(model_selection_decision, "strategy", "cost-balanced") or "cost-balanced"
        confidence = getattr(model_selection_decision, "confidence", "medium") or "medium"
        msd_roles_raw = getattr(model_selection_decision, "roles", []) or []
        roles: list[str] = []
        for _r in msd_roles_raw:
            if isinstance(_r, dict):
                _rn = _r.get("role", "")
            else:
                _rn = getattr(_r, "role", "")
            if _rn:
                roles.append(_rn)
        if not roles:
            roles = list(_DEFAULT_ROLE_TIERS.keys())
    else:
        strategy = "cost-balanced"
        confidence = "medium"
        roles = list(_DEFAULT_ROLE_TIERS.keys())

    candidates: list[NativeModelCandidateScore] = []
    for role in roles:
        for tier in _CANDIDATE_TIERS:
            cap = 0
            risk_fit = 0
            ctx_fit = 0
            cost = 0
            ver_fit = 0
            pen = 0
            reason_parts: list[str] = []

            # Capability
            if tier == "frontier-reasoning-model":
                if role == "planner":
                    cap = 25
                elif role == "executor" and risk_level == "high":
                    cap = 20
            elif tier == "balanced-coding-model":
                if role == "executor":
                    cap = 15
            elif tier == "low-cost-coding-model":
                if role == "executor" and task_type in ("docs", "test", "config"):
                    cap = 15
            elif tier == "independent-validator-model":
                if role == "validator":
                    cap = 25

            # Risk fit
            if risk_level == "high":
                if tier == "frontier-reasoning-model":
                    risk_fit += 15
                elif tier == "low-cost-coding-model":
                    risk_fit -= 15
                elif tier == "balanced-coding-model":
                    risk_fit += 5
            elif risk_level == "low":
                if tier == "low-cost-coding-model":
                    risk_fit += 10
                elif tier == "balanced-coding-model":
                    risk_fit += 5
                elif tier == "frontier-reasoning-model":
                    risk_fit -= 5
            if vc_strength == "weak" and role == "executor":
                if tier == "low-cost-coding-model":
                    risk_fit -= 10
            if rts_level == "weak" and role == "executor":
                if tier == "low-cost-coding-model":
                    risk_fit -= 5
                elif tier == "balanced-coding-model":
                    risk_fit -= 5

            # Context fit
            if cqs_level == "weak" or cp_has_gaps:
                if tier == "frontier-reasoning-model":
                    ctx_fit += 10
                elif tier == "balanced-coding-model":
                    ctx_fit += 5
            elif cqs_level in ("good", "strong") and not cp_has_gaps:
                if tier == "low-cost-coding-model" and role == "executor":
                    ctx_fit += 5

            # Cost
            if tier == "low-cost-coding-model":
                cost = 20
            elif tier == "balanced-coding-model":
                cost = 10
            elif tier == "frontier-reasoning-model":
                cost = -10
            else:
                cost = 0

            # Verification fit
            if role == "validator" and tier == "independent-validator-model":
                ver_fit += 20
            if role == "executor":
                if has_verification_cmds:
                    if tier in ("balanced-coding-model", "low-cost-coding-model"):
                        ver_fit += 5
                else:
                    if tier == "frontier-reasoning-model":
                        ver_fit += 5
                    elif tier == "low-cost-coding-model":
                        ver_fit -= 10

            # Penalties
            if has_lessons:
                if tier == "low-cost-coding-model":
                    pen += 5
                elif tier == "balanced-coding-model":
                    pen += 3
            if any_truncated and tier == "low-cost-coding-model":
                pen += 10

            raw = 50 + cap + risk_fit + ctx_fit + cost + ver_fit - pen
            final_score = max(0, min(100, raw))

            if cap:
                reason_parts.append(f"capability{'+' if cap >= 0 else ''}{cap}")
            if risk_fit:
                reason_parts.append(f"risk{'+' if risk_fit >= 0 else ''}{risk_fit}")
            if ctx_fit:
                reason_parts.append(f"context{'+' if ctx_fit >= 0 else ''}{ctx_fit}")
            if cost:
                reason_parts.append(f"cost{'+' if cost >= 0 else ''}{cost}")
            if ver_fit:
                reason_parts.append(f"verification{'+' if ver_fit >= 0 else ''}{ver_fit}")
            if pen:
                reason_parts.append(f"penalty-{pen}")

            candidates.append(NativeModelCandidateScore(
                role=role,
                candidate=tier,
                score=final_score,
                capability_score=cap,
                risk_fit_score=risk_fit,
                context_fit_score=ctx_fit,
                cost_score=cost,
                verification_fit_score=ver_fit,
                penalty=pen,
                reason=", ".join(reason_parts),
            ))

    # Policy enforcement: zero out blocked candidates
    blocked_candidates: list[str] = []
    _policy_warnings: list[str] = []
    if model_policy is not None:
        _allow_frontier = getattr(model_policy, "allow_frontier", True)
        _require_open_source = getattr(model_policy, "require_open_source", False)
        _require_local = getattr(model_policy, "require_local", False)
        _disallowed = set(getattr(model_policy, "disallowed_tiers", []) or [])
        _allowed = set(getattr(model_policy, "allowed_tiers", []) or [])
        for _c in candidates:
            _block = False
            if not _allow_frontier and _c.candidate == "frontier-reasoning-model":
                _block = True
                if "frontier models blocked by policy" not in _policy_warnings:
                    _policy_warnings.append("frontier models blocked by policy")
            if _require_open_source and _c.candidate == "frontier-reasoning-model":
                _block = True
                if "frontier models blocked by policy" not in _policy_warnings:
                    _policy_warnings.append("frontier models blocked by policy")
            if _require_local:
                if _c.candidate in ("frontier-reasoning-model", "independent-validator-model"):
                    _block = True
                    if _c.candidate == "frontier-reasoning-model":
                        if "frontier models blocked by policy" not in _policy_warnings:
                            _policy_warnings.append("frontier models blocked by policy")
                    else:
                        if "validator model blocked by local-only policy" not in _policy_warnings:
                            _policy_warnings.append("validator model blocked by local-only policy")
            if _c.candidate in _disallowed:
                _block = True
            if _allowed and _c.candidate not in _allowed:
                _block = True
            if _block:
                _c.score = 0
                _key = f"{_c.role}/{_c.candidate}"
                if _key not in blocked_candidates:
                    blocked_candidates.append(_key)
        if blocked_candidates and "policy restrictions reduced candidate pool" not in _policy_warnings:
            _policy_warnings.append("policy restrictions reduced candidate pool")

    # Base selection from MSD (pre-enforcement fallback)
    selected_by_role: dict[str, str] = {}
    if model_selection_decision is not None:
        msd_roles_raw = getattr(model_selection_decision, "roles", []) or []
        for _r in msd_roles_raw:
            if isinstance(_r, dict):
                _rn = _r.get("role", "")
                _rt = _r.get("model_tier", "")
            else:
                _rn = getattr(_r, "role", "")
                _rt = getattr(_r, "model_tier", "")
            if _rn and _rt:
                selected_by_role[_rn] = _rt
    if not selected_by_role:
        selected_by_role = dict(_DEFAULT_ROLE_TIERS)

    # Re-derive selection from enforcement survivors using role-valid tiers
    if blocked_candidates:
        _new_selected: dict[str, str] = {}
        for _role in roles:
            _valid = _VALID_TIERS_BY_ROLE.get(_role, list(_CANDIDATE_TIERS))
            _role_candidates = [_c for _c in candidates if _c.role == _role and _c.candidate in _valid]
            _surviving = [_c for _c in _role_candidates if _c.score > 0]
            if _surviving:
                _best = max(_surviving, key=lambda _c: _c.score)
                _new_selected[_role] = _best.candidate
            else:
                _new_selected[_role] = selected_by_role.get(_role, _DEFAULT_ROLE_TIERS.get(_role, "balanced-coding-model"))
                if "no candidates survived policy enforcement" not in _policy_warnings:
                    _policy_warnings.append("no candidates survived policy enforcement")
        selected_by_role = _new_selected

    warnings: list[str] = []
    if model_selection_decision is None:
        warnings.append("model selection decision missing")
    if rts_level == "weak":
        warnings.append("low trust run may reduce model selection confidence")
    if cp_has_gaps:
        warnings.append("context gaps may affect candidate scoring")
    if vc_strength == "weak":
        warnings.append("validation contract weak")
    for _pw in _policy_warnings:
        if _pw not in warnings:
            warnings.append(_pw)

    return NativeModelCandidateScoring(
        candidates=candidates,
        selected_by_role=selected_by_role,
        strategy=strategy,
        confidence=confidence,
        warnings=warnings,
        blocked_candidates=blocked_candidates,
    )


def _ns_to_dict(obj: object) -> dict:
    if isinstance(obj, dict):
        return obj
    try:
        return vars(obj)  # type: ignore[arg-type]
    except TypeError:
        return {}


def render_native_model_candidate_scoring(
    scoring: NativeModelCandidateScoring | None,
    detail: str = "compact",
) -> str:
    if scoring is None:
        return ""
    _strategy = getattr(scoring, "strategy", "cost-balanced") or "cost-balanced"
    _confidence = getattr(scoring, "confidence", "medium") or "medium"
    _selected = _ns_to_dict(getattr(scoring, "selected_by_role", {}) or {})
    _candidates = getattr(scoring, "candidates", []) or []
    _role_count = len(_selected) if _selected else len({
        (_c.get("role", "") if isinstance(_c, dict) else getattr(_c, "role", ""))
        for _c in _candidates
    })
    if detail == "full":
        lines = [
            "[model candidates]",
            f"strategy: {_strategy}",
            f"confidence: {_confidence}",
        ]
        if _selected:
            lines.append("selected:")
            for _role, _tier in _selected.items():
                lines.append(f"  - {_role}: {_tier}")
        if _candidates:
            lines.append("scores:")
            for _c in _candidates:
                if isinstance(_c, dict):
                    _cr = _c.get("role", "")
                    _cc = _c.get("candidate", "")
                    _cs = _c.get("score", 0)
                else:
                    _cr = getattr(_c, "role", "")
                    _cc = getattr(_c, "candidate", "")
                    _cs = getattr(_c, "score", 0)
                lines.append(f"  - {_cr}/{_cc}: {_cs}")
        _blocked = getattr(scoring, "blocked_candidates", []) or []
        if _blocked:
            lines.append("blocked:")
            for _b in _blocked:
                lines.append(f"  - {_b}")
        _warnings = getattr(scoring, "warnings", []) or []
        lines.append(f"warnings: {len(_warnings)}")
        return "\n".join(lines)
    _blocked = getattr(scoring, "blocked_candidates", []) or []
    _blocked_count = len(_blocked)
    _base = f"model candidates: {_role_count} roles, strategy={_strategy}, confidence={_confidence}"
    if _blocked_count:
        return f"{_base}, blocked={_blocked_count}"
    return _base


@dataclass
class NativeModelPolicy:
    mode: str = "auto"
    allowed_tiers: list[str] = field(default_factory=list)
    disallowed_tiers: list[str] = field(default_factory=list)
    prefer_low_cost: bool = False
    require_open_source: bool = False
    require_local: bool = False
    allow_frontier: bool = True
    warnings: list[str] = field(default_factory=list)


def build_native_model_policy(mode: str | None) -> NativeModelPolicy:
    """Parse a mode string into a NativeModelPolicy. Enforced in candidate scoring."""
    if mode is None or mode == "auto":
        return NativeModelPolicy()
    if mode == "cheapest-safe":
        return NativeModelPolicy(mode="cheapest-safe", prefer_low_cost=True, allow_frontier=True)
    if mode == "frontier-heavy":
        return NativeModelPolicy(mode="frontier-heavy", allow_frontier=True, prefer_low_cost=False)
    if mode == "open-source-only":
        return NativeModelPolicy(
            mode="open-source-only",
            require_open_source=True,
            allow_frontier=False,
            disallowed_tiers=["frontier-reasoning-model"],
        )
    if mode == "local-only":
        return NativeModelPolicy(
            mode="local-only",
            require_local=True,
            require_open_source=True,
            allow_frontier=False,
            disallowed_tiers=["frontier-reasoning-model"],
        )
    if mode == "custom":
        return NativeModelPolicy(
            mode="custom",
            warnings=["custom model policy is not enforced in v1"],
        )
    return NativeModelPolicy(
        mode="auto",
        warnings=["unknown model policy mode; defaulted to auto"],
    )


@dataclass
class NativeModelPolicyReceipt:
    active: bool = False
    mode: str = "auto"
    affected_selection: bool = False
    blocked_count: int = 0
    changed_roles: list[str] = field(default_factory=list)
    warnings_count: int = 0
    summary: str = ""


def build_native_model_policy_receipt(
    *,
    model_policy=None,
    model_selection_decision_before=None,
    model_selection_decision_after=None,
    model_candidate_scoring=None,
) -> NativeModelPolicyReceipt:
    _mode = "auto"
    if model_policy is not None:
        _mode = (
            model_policy.get("mode", "auto") if isinstance(model_policy, dict)
            else getattr(model_policy, "mode", "auto")
        ) or "auto"
    active = model_policy is not None and _mode != "auto"

    _blocked: list = []
    if model_candidate_scoring is not None:
        _blocked = (
            model_candidate_scoring.get("blocked_candidates", []) if isinstance(model_candidate_scoring, dict)
            else getattr(model_candidate_scoring, "blocked_candidates", [])
        ) or []
    blocked_count = len(_blocked)

    def _role_tiers(decision) -> dict:
        if decision is None:
            return {}
        roles = (
            decision.get("roles", []) if isinstance(decision, dict)
            else getattr(decision, "roles", [])
        ) or []
        result: dict = {}
        for r in roles:
            rname = r.get("role", "") if isinstance(r, dict) else getattr(r, "role", "")
            rtier = r.get("model_tier", "") if isinstance(r, dict) else getattr(r, "model_tier", "")
            if rname:
                result[rname] = rtier
        return result

    before_tiers = _role_tiers(model_selection_decision_before)
    after_tiers = _role_tiers(model_selection_decision_after)
    changed_roles = [r for r, t in after_tiers.items() if before_tiers.get(r) != t]

    affected_selection = blocked_count > 0 or len(changed_roles) > 0

    _mcs_warnings: list = []
    if model_candidate_scoring is not None:
        _mcs_warnings = (
            model_candidate_scoring.get("warnings", []) if isinstance(model_candidate_scoring, dict)
            else getattr(model_candidate_scoring, "warnings", [])
        ) or []
    _mp_warnings: list = []
    if model_policy is not None:
        _mp_warnings = (
            model_policy.get("warnings", []) if isinstance(model_policy, dict)
            else getattr(model_policy, "warnings", [])
        ) or []
    warnings_count = len(_mcs_warnings) + len(_mp_warnings)

    if not active:
        summary = "policy inactive"
    elif not affected_selection:
        summary = "policy active: no selection changes"
    else:
        n_b = blocked_count
        n_c = len(changed_roles)
        if n_b > 0 and n_c > 0:
            summary = (
                f"policy active: blocked {n_b} candidate{'s' if n_b != 1 else ''}"
                f" and changed {n_c} role{'s' if n_c != 1 else ''}"
            )
        elif n_b > 0:
            summary = f"policy active: blocked {n_b} candidate{'s' if n_b != 1 else ''}"
        else:
            summary = f"policy active: changed {n_c} role{'s' if n_c != 1 else ''}"

    return NativeModelPolicyReceipt(
        active=active,
        mode=_mode,
        affected_selection=affected_selection,
        blocked_count=blocked_count,
        changed_roles=changed_roles,
        warnings_count=warnings_count,
        summary=summary,
    )


@dataclass
class NativeRoutingPreview:
    strategy: str = "cost-balanced"
    policy_mode: str = "auto"
    planner_tier: str = "unknown"
    executor_tier: str = "unknown"
    validator_tier: str = "unknown"
    risk_level: str = "unknown"
    confidence: str = "medium"
    blocked_count: int = 0
    policy_affected: bool = False
    trust_level: str = "unknown"
    summary: str = ""
    warnings: list[str] = field(default_factory=list)


def build_native_routing_preview(
    *,
    model_candidate_scoring=None,
    model_selection_decision=None,
    model_policy_receipt=None,
    run_trust_score=None,
) -> NativeRoutingPreview:
    def _get(obj, key, default=None):
        if obj is None:
            return default
        return obj.get(key, default) if isinstance(obj, dict) else getattr(obj, key, default)

    def _tier_from_scoring(role: str) -> str:
        sbr = _get(model_candidate_scoring, "selected_by_role", {}) or {}
        return sbr.get(role, "")

    def _tier_from_decision(role: str) -> str:
        roles = _get(model_selection_decision, "roles", []) or []
        for r in roles:
            rname = r.get("role", "") if isinstance(r, dict) else getattr(r, "role", "")
            if rname == role:
                return r.get("model_tier", "") if isinstance(r, dict) else getattr(r, "model_tier", "")
        return ""

    def _tier(role: str) -> str:
        t = _tier_from_scoring(role)
        return t if t else (_tier_from_decision(role) or "unknown")

    strategy = _get(model_candidate_scoring, "strategy", "") or _get(model_selection_decision, "strategy", "cost-balanced") or "cost-balanced"
    confidence = _get(model_candidate_scoring, "confidence", "") or _get(model_selection_decision, "confidence", "medium") or "medium"
    risk_level = _get(model_selection_decision, "risk_level", "unknown") or "unknown"
    policy_mode = _get(model_policy_receipt, "mode", "auto") or "auto"
    policy_affected = bool(_get(model_policy_receipt, "affected_selection", False))
    blocked_count = int(_get(model_policy_receipt, "blocked_count", 0) or 0)
    trust_level = _get(run_trust_score, "level", "unknown") or "unknown"

    planner_tier = _tier("planner")
    executor_tier = _tier("executor")
    validator_tier = _tier("validator")

    mcs_warnings: list[str] = list(_get(model_candidate_scoring, "warnings", []) or [])
    warnings = mcs_warnings

    summary = (
        f"{strategy} | planner={planner_tier} executor={executor_tier}"
        f" validator={validator_tier} | policy={policy_mode}"
    )

    return NativeRoutingPreview(
        strategy=strategy,
        policy_mode=policy_mode,
        planner_tier=planner_tier,
        executor_tier=executor_tier,
        validator_tier=validator_tier,
        risk_level=risk_level,
        confidence=confidence,
        blocked_count=blocked_count,
        policy_affected=policy_affected,
        trust_level=trust_level,
        summary=summary,
        warnings=warnings,
    )


@dataclass
class NativeRoutingReceipt:
    strategy: str = ""
    planner_tier: str = "unknown"
    executor_tier: str = "unknown"
    validator_tier: str = "unknown"
    policy_mode: str = "auto"
    policy_affected: bool = False
    blocked_count: int = 0
    trust_level: str = "unknown"
    confidence: str = "medium"
    warnings_count: int = 0
    summary: str = ""


def build_native_routing_receipt(
    *,
    routing_preview=None,
    model_policy_receipt=None,
    run_trust_score=None,
) -> NativeRoutingReceipt:
    def _get(obj, key, default=None):
        if obj is None:
            return default
        return obj.get(key, default) if isinstance(obj, dict) else getattr(obj, key, default)

    strategy = _get(routing_preview, "strategy", "") or ""
    planner_tier = _get(routing_preview, "planner_tier", "unknown") or "unknown"
    executor_tier = _get(routing_preview, "executor_tier", "unknown") or "unknown"
    validator_tier = _get(routing_preview, "validator_tier", "unknown") or "unknown"
    policy_mode = _get(routing_preview, "policy_mode", "auto") or "auto"
    blocked_count = int(_get(routing_preview, "blocked_count", 0) or 0)
    confidence = _get(routing_preview, "confidence", "medium") or "medium"
    summary = _get(routing_preview, "summary", "") or ""
    preview_trust = _get(routing_preview, "trust_level", "unknown") or "unknown"

    policy_affected = bool(_get(model_policy_receipt, "affected_selection", False))
    warnings_count = int(_get(model_policy_receipt, "warnings_count", 0) or 0)

    trust_level = _get(run_trust_score, "level", None)
    if not trust_level:
        trust_level = preview_trust

    return NativeRoutingReceipt(
        strategy=strategy,
        planner_tier=planner_tier,
        executor_tier=executor_tier,
        validator_tier=validator_tier,
        policy_mode=policy_mode,
        policy_affected=policy_affected,
        blocked_count=blocked_count,
        trust_level=trust_level,
        confidence=confidence,
        warnings_count=warnings_count,
        summary=summary,
    )


def sync_native_model_selection_decision_with_candidate_scoring(
    model_selection_decision: NativeModelSelectionDecision | None,
    model_candidate_scoring: NativeModelCandidateScoring | None,
    model_policy: NativeModelPolicy | None = None,
) -> NativeModelSelectionDecision | None:
    if model_selection_decision is None or model_candidate_scoring is None:
        return model_selection_decision
    selected = getattr(model_candidate_scoring, "selected_by_role", {}) or {}
    if not selected:
        return model_selection_decision

    import copy
    synced = copy.deepcopy(model_selection_decision)
    changed = False

    for role, tier in selected.items():
        existing = next(
            (
                r for r in synced.roles
                if (r.get("role") if isinstance(r, dict) else getattr(r, "role", "")) == role
            ),
            None,
        )
        if existing is not None:
            current_tier = (
                existing.get("model_tier", "") if isinstance(existing, dict)
                else getattr(existing, "model_tier", "")
            )
            if current_tier != tier:
                if isinstance(existing, dict):
                    existing["model_tier"] = tier
                else:
                    existing.model_tier = tier
                changed = True
        else:
            synced.roles.append(NativeModelRoleDecision(role=role, model_tier=tier))
            changed = True

    _policy_warning = "model selection adjusted by policy enforcement"
    if changed:
        if _policy_warning not in synced.warnings:
            synced.warnings.append(_policy_warning)
        if not synced.fallback_reason:
            synced.fallback_reason = "candidate scoring applied model policy constraints"

    return synced


# ---------------------------------------------------------------------------
# Tier dispatch receipt
# ---------------------------------------------------------------------------

@dataclass
class NativeTierDispatchReceipt:
    enabled: bool = False
    applied: bool = False
    tier_source: str = ""           # "candidate_scoring" | "category_fallback" | ""
    planner_tier: str = ""
    planner_model: str | None = None
    executor_tier: str = ""
    executor_model: str | None = None
    validator_tier: str = ""
    validator_model: str | None = None
    fallback_used: bool = False
    fallback_reason: str = ""
    warnings: list[str] = field(default_factory=list)
    planner_model_actual: str | None = None
    executor_model_actual: str | None = None
    validator_model_actual: str | None = None
    validator_dispatch_status: str = ""   # "applied" | "reserved" | "skipped" | ""


def build_native_tier_dispatch_receipt(
    *,
    routing_receipt=None,
    model_candidate_scoring=None,
    routing_category: str | None = None,
    experimental_tier_dispatch: bool = False,
    applied: bool = False,
    not_applied_reason: str = "",
    planner_model_actual: str | None = None,
    executor_model_actual: str | None = None,
    validator_model_actual: str | None = None,
    validator_dispatch_status: str = "",
) -> NativeTierDispatchReceipt:
    """Build a tier dispatch receipt.

    Priority for tier source:
    1. routing_receipt tiers (planner_tier / executor_tier / validator_tier)
    2. model_candidate_scoring.selected_by_role
    3. routing_category fallback (category_fallback)

    applied=True means the resolved models were actually used for generation.
    not_applied_reason explains why applied=False (e.g. "dry-run", "native tier dispatch is recorded only in v1").
    """
    if not experimental_tier_dispatch:
        return NativeTierDispatchReceipt(enabled=False)

    from openshard.native.dispatch import resolve_tier, resolve_tier_for_category

    def _get(obj, key, default=""):
        if obj is None:
            return default
        return obj.get(key, default) if isinstance(obj, dict) else getattr(obj, key, default)

    # Try routing_receipt tiers first
    p_tier = _get(routing_receipt, "planner_tier", "") or ""
    e_tier = _get(routing_receipt, "executor_tier", "") or ""
    v_tier = _get(routing_receipt, "validator_tier", "") or ""

    # Fall through to candidate scoring selected_by_role
    if not (p_tier or e_tier or v_tier) and model_candidate_scoring is not None:
        selected = _get(model_candidate_scoring, "selected_by_role", {}) or {}
        p_tier = selected.get("planner", "") or ""
        e_tier = selected.get("executor", "") or ""
        v_tier = selected.get("validator", "") or ""

    tier_source: str
    if p_tier or e_tier or v_tier:
        # Selected tier metadata available
        tier_source = "candidate_scoring"
        p_model, p_fb, p_reason = resolve_tier(p_tier or None)
        e_model, e_fb, e_reason = resolve_tier(e_tier or None)
        v_model, v_fb, v_reason = resolve_tier(v_tier or None)
    else:
        # No selected tiers — fall back to routing category
        tier_source = "category_fallback"
        # Planner is always frontier tier
        p_tier = "frontier-reasoning-model"
        p_model, p_fb, p_reason = resolve_tier(p_tier)
        # Executor from category
        e_model, e_tier_resolved, e_fb, e_reason = resolve_tier_for_category(routing_category)
        e_tier = e_tier_resolved or ""
        # Validator is always independent tier
        v_tier = "independent-validator-model"
        v_model, v_fb, v_reason = resolve_tier(v_tier)

    any_fallback = p_fb or e_fb or v_fb
    tier_reasons = [r for r in (p_reason, e_reason, v_reason) if r]
    first_tier_reason = tier_reasons[0] if tier_reasons else ""
    effective_fallback_reason = not_applied_reason or first_tier_reason

    warnings: list[str] = []
    if p_fb and p_reason:
        warnings.append(f"planner fallback: {p_reason}")
    if e_fb and e_reason:
        warnings.append(f"executor fallback: {e_reason}")
    if v_fb and v_reason:
        warnings.append(f"validator fallback: {v_reason}")

    return NativeTierDispatchReceipt(
        enabled=True,
        applied=applied,
        tier_source=tier_source,
        planner_tier=p_tier,
        planner_model=p_model,
        executor_tier=e_tier,
        executor_model=e_model,
        validator_tier=v_tier,
        validator_model=v_model,
        fallback_used=any_fallback,
        fallback_reason=effective_fallback_reason,
        warnings=warnings,
        planner_model_actual=planner_model_actual,
        executor_model_actual=executor_model_actual,
        validator_model_actual=validator_model_actual,
        validator_dispatch_status=validator_dispatch_status,
    )


@dataclass
class NativeSandboxMeta:
    sandbox_enabled: bool = False
    sandbox_type: str = "none"        # "worktree" | "temp" | "none"
    worktree_path: str | None = None
    worktree_branch: str | None = None
    fallback_reason: str | None = None
    git_base_branch: str | None = None
    git_base_commit_hash: str | None = None
    safe_workspace_display_name: str | None = None


_ADVISORY_MODEL_FAILURE_THRESHOLD: int = 2
_ADVISORY_FILE_FREQUENCY_THRESHOLD: int = 2


@dataclass
class NativeModelRetryFailureSummary:
    model: str = ""
    failure_count: int = 0
    failure_types: list[str] = field(default_factory=list)


@dataclass
class NativeFailureMemoryRoutingAdvisory:
    events_scanned: int = 0
    model_retry_summaries: list[NativeModelRetryFailureSummary] = field(default_factory=list)
    hot_file_paths: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    advisory_only: bool = True


def build_native_failure_memory_routing_advisory(
    *, limit: int = 10,
) -> NativeFailureMemoryRoutingAdvisory:
    try:
        from openshard.history.failure_memory import recent_failure_memory
        events = recent_failure_memory(limit=limit)
    except Exception:
        return NativeFailureMemoryRoutingAdvisory(advisory_only=True)

    model_failure_counts: dict[str, int] = {}
    model_failure_types: dict[str, list[str]] = {}
    for evt in events:
        if not getattr(evt, "retry_attempted", False):
            continue
        model = getattr(evt, "model", "") or ""
        if not model:
            continue
        model_failure_counts[model] = model_failure_counts.get(model, 0) + 1
        ftype = getattr(evt, "failure_type", "") or ""
        if ftype:
            model_failure_types.setdefault(model, []).append(ftype)

    file_counts: dict[str, int] = {}
    for evt in events:
        for path in (getattr(evt, "related_file_paths", []) or []):
            if path:
                file_counts[path] = file_counts.get(path, 0) + 1

    model_retry_summaries = [
        NativeModelRetryFailureSummary(
            model=model,
            failure_count=count,
            failure_types=model_failure_types.get(model, []),
        )
        for model, count in model_failure_counts.items()
    ]

    hot_file_paths = [
        path for path, count in file_counts.items()
        if count >= _ADVISORY_FILE_FREQUENCY_THRESHOLD
    ]

    warnings: list[str] = []
    for s in model_retry_summaries:
        if s.failure_count >= _ADVISORY_MODEL_FAILURE_THRESHOLD:
            warnings.append(
                f"model {s.model!r} has {s.failure_count} repeated retry failures"
            )
    for path in hot_file_paths:
        warnings.append(
            f"file {path!r} appears in {file_counts[path]} recent failure events"
        )

    return NativeFailureMemoryRoutingAdvisory(
        events_scanned=len(events),
        model_retry_summaries=model_retry_summaries,
        hot_file_paths=hot_file_paths,
        warnings=warnings,
        advisory_only=True,
    )


def render_native_failure_memory_routing_advisory(
    advisory: NativeFailureMemoryRoutingAdvisory | dict | None,
    detail: str = "compact",
) -> str:
    if advisory is None:
        return ""
    _is_dict = isinstance(advisory, dict)
    _warnings = (advisory.get("warnings", []) if _is_dict else getattr(advisory, "warnings", [])) or []  # type: ignore[union-attr]  # narrowed by _is_dict flag
    _scanned = advisory.get("events_scanned", 0) if _is_dict else getattr(advisory, "events_scanned", 0)  # type: ignore[union-attr]  # narrowed by _is_dict flag
    if not _warnings:
        if detail == "full":
            return (
                "[failure memory routing advisory]\n"
                f"events_scanned: {_scanned}\n"
                "warnings: 0"
            )
        return ""
    _wc = len(_warnings)
    _wword = "warning" if _wc == 1 else "warnings"
    if detail != "full":
        return f"failure memory advisory: {_wc} {_wword}"
    _summaries = (advisory.get("model_retry_summaries", []) if _is_dict else getattr(advisory, "model_retry_summaries", [])) or []  # type: ignore[union-attr]  # narrowed by _is_dict flag
    _hot_files = (advisory.get("hot_file_paths", []) if _is_dict else getattr(advisory, "hot_file_paths", [])) or []  # type: ignore[union-attr]  # narrowed by _is_dict flag
    lines = [
        "[failure memory routing advisory]",
        f"events_scanned: {_scanned}",
    ]
    if _summaries:
        lines.append("model retry failures:")
        for s in _summaries:
            if isinstance(s, dict):
                _m = s.get("model", "")
                _c = s.get("failure_count", 0)
                _ft = s.get("failure_types", []) or []
            else:
                _m = getattr(s, "model", "")
                _c = getattr(s, "failure_count", 0)
                _ft = getattr(s, "failure_types", []) or []
            _ft_str = f"  [{', '.join(_ft[:3])}]" if _ft else ""
            lines.append(f"  - {_m}: {_c} retries{_ft_str}")
    if _hot_files:
        lines.append("hot files:")
        for p in _hot_files:
            lines.append(f"  - {p}")
    lines.append(f"warnings: {_wc}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Native Plan Ledger v0
# ---------------------------------------------------------------------------

@dataclass
class NativePlanItem:
    index: int = 0
    title: str = ""
    status: str = "pending"  # pending | running | passed | failed | skipped
    evidence: str = ""
    raw_content_stored: bool = False

    def __post_init__(self) -> None:
        self.raw_content_stored = False


@dataclass
class NativePlanLedger:
    enabled: bool = True
    items: list[NativePlanItem] = field(default_factory=list)
    completed_count: int = 0
    failed_count: int = 0
    pending_count: int = 0
    raw_content_stored: bool = False

    def __post_init__(self) -> None:
        self.raw_content_stored = False


_VALID_PLAN_STATUSES: frozenset[str] = frozenset({"pending", "running", "passed", "failed", "skipped"})

_V0_PLAN_TITLES: list[str] = [
    "Understand task and repo context",
    "Generate patch",
    "Write patch to sandbox",
    "Run verification",
    "Record receipt",
]


def build_native_plan_ledger(
    task: str,
    planned_files: list[str] | None = None,
) -> NativePlanLedger:
    items = []
    for i, title in enumerate(_V0_PLAN_TITLES):
        evidence = ""
        if planned_files and title == "Write patch to sandbox":
            evidence = f"files={len(planned_files)}"
        items.append(NativePlanItem(index=i, title=title, status="pending", evidence=evidence))
    return NativePlanLedger(
        enabled=True,
        items=items,
        completed_count=0,
        failed_count=0,
        pending_count=len(items),
    )


def update_native_plan_ledger_status(
    ledger: NativePlanLedger,
    title_contains: str,
    status: str,
    evidence: str = "",
) -> NativePlanLedger:
    if status not in _VALID_PLAN_STATUSES:
        return ledger
    needle = title_contains.lower()
    for item in ledger.items:
        if needle in item.title.lower():
            item.status = status
            if evidence:
                item.evidence = evidence
            break
    ledger.completed_count = sum(1 for it in ledger.items if it.status == "passed")
    ledger.failed_count = sum(1 for it in ledger.items if it.status == "failed")
    ledger.pending_count = sum(1 for it in ledger.items if it.status == "pending")
    return ledger


_PLAN_STATUS_LABELS: dict[str, str] = {
    "passed": "passed",
    "failed": "failed",
    "running": "running",
    "skipped": "skipped",
    "pending": "pending",
}


def render_native_plan_ledger(
    ledger: NativePlanLedger | None,
    detail: str = "compact",
) -> str:
    if ledger is None or not ledger.enabled or not ledger.items:
        return ""
    total = len(ledger.items)
    header = f"plan ledger: {ledger.completed_count}/{total} passed, {ledger.failed_count} failed"
    if detail != "full":
        return header
    lines = [header]
    for item in ledger.items:
        label = _PLAN_STATUS_LABELS.get(item.status, item.status)
        ev = f"  ({item.evidence})" if item.evidence else ""
        lines.append(f"  {item.index + 1}. {label}  {item.title}{ev}")
    return "\n".join(lines)


def record_native_edit_loop_attempt(
    summary: NativeEditLoopSummary,
    *,
    attempt_index: int,
    purpose: str,
    files_written: list[str],
    verification_status: str,
    exit_code: int | None,
    output_chars: int,
) -> NativeEditLoopSummary:
    attempt = NativeEditLoopAttempt(
        attempt_index=attempt_index,
        purpose=purpose,
        files_written=list(files_written),
        verification_status=verification_status,
        exit_code=exit_code,
        output_chars=output_chars,
    )
    summary.attempts.append(attempt)
    if purpose == "repair":
        summary.repair_used = True
    summary.final_status = verification_status
    if (
        verification_status == "passed"
        or verification_status == "skipped"
        or attempt_index >= summary.max_attempts
    ):
        summary.completed = True
    return summary


def render_native_edit_loop_summary(
    summary: NativeEditLoopSummary | Any | None,
    detail: str = "compact",
) -> str:
    if summary is None:
        return ""
    if not getattr(summary, "enabled", True):
        return ""
    final_status = getattr(summary, "final_status", "") or ""
    attempts = getattr(summary, "attempts", []) or []
    max_attempts = getattr(summary, "max_attempts", 2)
    n = len(attempts)
    header = f"edit loop: {final_status} after {n}/{max_attempts} attempt(s)"
    if detail != "full" or not attempts:
        return header
    lines = [header]
    for att in attempts:
        idx = getattr(att, "attempt_index", 0)
        purpose = getattr(att, "purpose", "")
        vstatus = getattr(att, "verification_status", "")
        fcount = len(getattr(att, "files_written", []) or [])
        exit_code = getattr(att, "exit_code", None)
        chars = getattr(att, "output_chars", 0)
        exit_s = f" exit={exit_code}" if exit_code is not None else ""
        lines.append(
            f"  {idx}. {purpose}: {vstatus} files={fcount}{exit_s} chars={chars}"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Native Candidate Agents v0
# ---------------------------------------------------------------------------

@dataclass
class NativeCandidateAttempt:
    candidate_index: int = 0
    model: str = ""
    sandbox_path: str = ""
    files_written: list[str] = field(default_factory=list)
    verification_status: str = ""  # "passed" | "failed" | "skipped"
    exit_code: int | None = None
    output_chars: int = 0
    selected: bool = False
    selection_reason: str = ""
    score: float = 0.0
    score_reasons: list[str] = field(default_factory=list)
    raw_content_stored: bool = False

    def __post_init__(self) -> None:
        self.raw_content_stored = False


@dataclass
class NativeCandidateSummary:
    enabled: bool = False
    requested_count: int = 1
    completed_count: int = 0
    selected_index: int | None = None  # 1-based
    selection_reason: str = ""
    candidates: list[NativeCandidateAttempt] = field(default_factory=list)
    raw_content_stored: bool = False

    def __post_init__(self) -> None:
        self.raw_content_stored = False


def record_native_candidate_attempt(
    summary: NativeCandidateSummary,
    *,
    candidate_index: int,  # 1-based
    model: str,
    sandbox_path: str,
    files_written: list[str],
    verification_status: str,
    exit_code: int | None = None,
    output_chars: int = 0,
) -> NativeCandidateSummary:
    attempt = NativeCandidateAttempt(
        candidate_index=candidate_index,
        model=model,
        sandbox_path=sandbox_path,
        files_written=list(files_written),
        verification_status=verification_status,
        exit_code=exit_code,
        output_chars=output_chars,
    )
    summary.candidates.append(attempt)
    summary.completed_count = len(summary.candidates)
    return summary


def score_native_candidate_attempt(candidate: Any) -> tuple[float, list[str]]:
    """Compute a deterministic score for a single candidate attempt.

    Supports dataclass, dict, and SimpleNamespace candidates.
    Returns (score, reasons) without mutating the candidate.
    """
    def _g(key: str, default: Any = None) -> Any:
        if isinstance(candidate, dict):
            return candidate.get(key, default)
        return getattr(candidate, key, default)

    score = 0.0
    reasons: list[str] = []

    vstatus = _g("verification_status", "")
    if vstatus == "passed":
        score += 100
        reasons.append("verification passed")
    elif vstatus == "skipped":
        score += 20
        reasons.append("verification skipped")
    elif vstatus == "failed":
        score -= 50
        reasons.append("verification failed")

    exit_code = _g("exit_code", None)
    if exit_code == 0:
        score += 10
        reasons.append("exit code 0")
    elif exit_code is not None:
        score -= 10
        reasons.append(f"exit code {exit_code}")

    files_written = _g("files_written", []) or []
    fcount = len(files_written)
    if fcount == 0:
        score -= 20
        reasons.append("no files written")
    elif fcount <= 5:
        score += 10
        reasons.append("reasonable file count")
    else:
        score -= 5
        reasons.append("large file count")

    output_chars = _g("output_chars", 0) or 0
    if output_chars > 5000:
        score -= 5
        reasons.append("large verification output")

    score = max(-100.0, min(150.0, score))
    return score, reasons


def score_native_candidate_summary(
    summary: NativeCandidateSummary,
) -> NativeCandidateSummary:
    """Score every candidate in summary, mutating score and score_reasons in place."""
    for candidate in summary.candidates:
        s, r = score_native_candidate_attempt(candidate)
        candidate.score = s
        candidate.score_reasons = r
    return summary


def select_native_candidate(summary: NativeCandidateSummary) -> NativeCandidateSummary:
    """Score all candidates, select highest score (tie-break: lower candidate_index)."""
    if not summary.candidates:
        summary.selection_reason = "no candidates"
        return summary
    for candidate in summary.candidates:
        candidate.selected = False
        candidate.selection_reason = ""
    score_native_candidate_summary(summary)
    winner = max(
        summary.candidates,
        key=lambda c: (c.score, -c.candidate_index),
    )
    winner.selected = True
    first_reason = winner.score_reasons[0] if winner.score_reasons else ""
    reason = f"highest score: {winner.score} ({first_reason})"
    winner.selection_reason = reason
    summary.selected_index = winner.candidate_index  # 1-based
    summary.selection_reason = reason
    return summary


def render_native_candidate_summary(
    summary: NativeCandidateSummary | Any | None,
    detail: str = "compact",
) -> str:
    """Supports dataclass, SimpleNamespace, and dict objects."""

    def _v(obj: Any, key: str, default: Any = None) -> Any:
        if isinstance(obj, dict):
            return obj.get(key, default)
        return getattr(obj, key, default)

    if summary is None:
        return ""
    if not _v(summary, "enabled", False):
        return ""
    requested = _v(summary, "requested_count", 1)
    completed = _v(summary, "completed_count", 0)
    selected = _v(summary, "selected_index", None)  # 1-based or None
    candidates = _v(summary, "candidates", []) or []
    passed_count = sum(
        1 for c in candidates
        if _v(c, "verification_status", "") == "passed"
    )
    sel_str = str(selected) if selected is not None else "none"
    sel_candidate = next(
        (c for c in candidates if _v(c, "selected", False)), None
    )
    score_suffix = ""
    if sel_candidate is not None:
        sel_score = _v(sel_candidate, "score", 0.0)
        score_suffix = f", score={sel_score}"
    header = (
        f"candidates: {completed}/{requested} run,"
        f" selected={sel_str}, passed={passed_count}{score_suffix}"
    )
    if detail != "full" or not candidates:
        return header
    lines = [header]
    for c in candidates:
        idx = _v(c, "candidate_index", 0)  # 1-based
        vstatus = _v(c, "verification_status", "")
        fcount = len(_v(c, "files_written", []) or [])
        exit_code = _v(c, "exit_code", None)
        chars = _v(c, "output_chars", 0)
        is_sel = _v(c, "selected", False)
        cscore = _v(c, "score", 0.0)
        creasons = _v(c, "score_reasons", []) or []
        sel_marker = "selected " if is_sel else ""
        exit_s = f" exit={exit_code}" if exit_code is not None else ""
        lines.append(
            f"  {idx}. {sel_marker}{vstatus} score={cscore} files={fcount}{exit_s} chars={chars}"
        )
        if creasons:
            lines.append(f"     reason: {'; '.join(creasons)}")
    return "\n".join(lines)
