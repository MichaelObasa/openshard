from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from openshard.analysis.repo import RepoFacts
from openshard.execution.generator import ExecutionGenerator, ExecutionResult
from openshard.native.context import (
    CompactRunState,
    NativeCommandPolicyPreview,
    NativeContextBudget,
    NativeApprovalReceipt,
    NativeApprovalRequest,
    NativeChangeBudget,
    NativeChangeBudgetPreview,
    NativeChangeBudgetSoftGate,
    NativeContextPacket,
    NativeContextQualityAdvisory,
    NativeContextQualityScore,
    NativeDiffReview,
    NativeEvidence,
    NativeFileContext,
    NativeFileSnippet,
    NativeFinalReport,
    NativeObservation,
    NativePatchProposal,
    NativePlan,
    NativeVerificationCommandSummary,
    NativeVerificationLoop,
    NativeVerificationPlan,
    build_initial_context_budget,
    build_native_approval_receipt,
    build_native_budget_gate_approval_request,
    build_native_command_policy_preview,
    build_native_context_packet,
    build_native_change_budget,
    build_native_change_budget_preview,
    build_native_change_budget_soft_gate,
    build_native_context_quality_advisory,
    build_native_context_quality_score,
    build_native_diff_review,
    build_native_final_report,
    build_native_patch_proposal,
    build_native_verification_command_summary,
    build_native_verification_plan,
    render_native_change_budget,
    render_native_context_packet,
    render_native_context_quality_advisory,
    render_native_evidence,
    render_native_observation,
    render_native_plan,
)
from openshard.native.repo_context import (
    NativeRepoContextSummary,
    build_repo_context_summary,
    render_repo_context_summary,
)
from openshard.native.loop import NativeLoopTrace
from openshard.native.skills import match_builtin_skills, selected_skill_names
from openshard.native.tool_runner import NativeToolRunner
from openshard.native.tools import NativeToolCall, NativeToolResult


@dataclass
class NativeRunMeta:
    workflow: str = "native"
    executor: str = "native"
    execution_depth: str = "fast"
    selected_skills: list[str] = field(default_factory=list)
    context_budget: NativeContextBudget | None = None
    context_state: CompactRunState | None = None
    context_warnings: list[str] = field(default_factory=list)
    tool_trace: list[dict] = field(default_factory=list)
    repo_context_summary: NativeRepoContextSummary | None = None
    observation: NativeObservation | None = None
    evidence: NativeEvidence | None = None
    plan: NativePlan | None = None
    diff_review: NativeDiffReview | None = None
    write_path: str = "pipeline"
    verification_loop: NativeVerificationLoop | None = None
    verification_command_summary: NativeVerificationCommandSummary | None = None
    final_report: NativeFinalReport | None = None
    native_loop_steps: list[str] = field(default_factory=list)
    native_loop_trace: NativeLoopTrace = field(default_factory=NativeLoopTrace)
    native_backend: str = "builtin"
    native_backend_available: bool = True
    native_backend_notes: list[str] = field(default_factory=list)
    native_backend_proof: dict | None = None
    read_search_findings: list[str] = field(default_factory=list)
    patch_proposal: NativePatchProposal | None = None
    command_policy_preview: NativeCommandPolicyPreview | None = None
    file_context: NativeFileContext | None = None
    context_packet: NativeContextPacket | None = None
    context_quality_score: NativeContextQualityScore | None = None
    context_quality_advisory: NativeContextQualityAdvisory | None = None
    change_budget: NativeChangeBudget | None = None
    change_budget_preview: NativeChangeBudgetPreview | None = None
    change_budget_soft_gate: NativeChangeBudgetSoftGate | None = None
    approval_request: NativeApprovalRequest | None = None
    approval_receipt: NativeApprovalReceipt | None = None
    verification_plan: NativeVerificationPlan | None = None


_SEARCH_STOP_WORDS: frozenset[str] = frozenset({
    "the", "a", "an", "in", "on", "at", "to", "for", "of",
    "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did",
    "will", "would", "should", "could", "may", "might", "can",
})

_SEARCH_TRIGGER_WORDS: frozenset[str] = frozenset({"where", "find", "search", "locate"})

_MAX_READ_SEARCH_STEPS: int = 3
_MAX_LOOP_FINDINGS: int = 5
_MAX_FILE_CONTEXT_FILES: int = 3
_MAX_FILE_CONTEXT_CHARS_PER_FILE: int = 2000
_MAX_FILE_CONTEXT_TOTAL_CHARS: int = 5000


def _parse_search_result_line(line: str) -> tuple[str, int] | None:
    parts = line.split(":", 2)
    if len(parts) < 3:
        return None
    path, lineno_raw, _ = parts
    try:
        return path, int(lineno_raw)
    except ValueError:
        return None


def _extract_snippet_lines(content: str, lineno: int, *, radius: int = 2, max_lines: int = 8) -> list[str]:
    all_lines = content.splitlines()
    if not all_lines:
        return []
    idx = max(0, lineno - 1)
    start = max(0, idx - radius)
    end = min(len(all_lines), idx + radius + 1)
    selected = all_lines[start:end][:max_lines]
    return [f"{start + i + 1}: {line}" for i, line in enumerate(selected)]


def _build_file_snippets_from_search(
    runner,
    search_lines: list[str],
) -> tuple[list[NativeFileSnippet], list[dict]]:
    snippets: list[NativeFileSnippet] = []
    traces: list[dict] = []
    seen: list[str] = []

    for line in search_lines:
        parsed = _parse_search_result_line(line)
        if parsed is None:
            continue
        path, lineno = parsed
        if path in seen:
            continue
        seen.append(path)
        if len(seen) > 2:
            break

        call = NativeToolCall(tool_name="read_file", args={"path": path})
        result = runner.run(call)
        traces.append(runner.trace_entry(call, result))
        if not result.ok:
            continue
        snippet_lines = _extract_snippet_lines(result.output, lineno)
        snippets.append(NativeFileSnippet(path=path, lines=snippet_lines))

    return snippets, traces


def _extract_search_query(task: str) -> str | None:
    filtered = []
    for raw in task.lower().split():
        word = raw.strip(".,:;!?()[]{}\"'`")
        if word and word not in _SEARCH_STOP_WORDS and word not in _SEARCH_TRIGGER_WORDS and len(word) >= 3:
            filtered.append(word)
    if not filtered:
        return None
    return " ".join(filtered[:3])


def _build_native_plan(
    task: str,
    *,
    observation: NativeObservation | None = None,
    evidence: NativeEvidence | None = None,
    selected_skills: list[str] | None = None,
) -> NativePlan:
    selected_skills = selected_skills or []

    _strip = ".,:;!?()[]{}\"'`"
    words = {w.strip(_strip) for w in task.lower().split() if w.strip(_strip)}

    if words & {"search", "find", "where", "locate"}:
        intent = "search"
    elif words & {"test", "verify", "fail", "error", "debug", "fix"}:
        intent = "debug"
    elif "refactor" in words:
        intent = "refactor"
    elif words & {"add", "create", "implement", "build"}:
        intent = "implementation"
    else:
        intent = "standard"

    dirty = observation is not None and observation.dirty_diff_present
    security = "security-sensitive-change" in selected_skills
    risk = "medium" if (dirty or security) else "low"

    steps = ["inspect relevant files", "make the smallest safe change", "review the diff"]
    if evidence is not None:
        steps.insert(0, "use bounded search evidence to choose files")
    if intent == "debug":
        steps.append("run verification after changes")
    if dirty:
        steps.append("avoid overwriting existing user changes")

    warnings = []
    if dirty:
        warnings.append("dirty working tree detected")
    if security:
        warnings.append("security-sensitive task")

    return NativePlan(
        intent=intent,
        risk=risk,
        suggested_steps=steps,
        warnings=warnings,
    )


class NativeAgentExecutor:
    """Fast-path native executor. Delegates generation to ExecutionGenerator."""

    def __init__(
        self,
        provider=None,
        repo_root: Path | None = None,
        backend_name: str = "builtin",
        experimental_deepagents_run: bool = False,
        deepagents_model: str | None = None,
    ) -> None:
        from openshard.native.backends import get_backend

        self._gen = ExecutionGenerator(provider=provider)
        self.model = self._gen.model
        self.fixer_model = self._gen.fixer_model
        self.native_meta = NativeRunMeta()
        self._runner = NativeToolRunner(repo_root) if repo_root is not None else None
        self._backend = get_backend(backend_name)
        _available = self._backend.available()
        self.native_meta.native_backend = self._backend.name
        self.native_meta.native_backend_available = _available
        self._experimental_deepagents_run = experimental_deepagents_run
        self._deepagents_model = deepagents_model  # None in production; injectable for tests
        if backend_name == "deepagents" and not _available:
            self.native_meta.native_backend_notes = [
                "Install deepagents to enable this experimental backend."
            ]

    def record_loop_step(
        self,
        step: str,
        *,
        summary: str = "",
        metadata: dict | None = None,
    ) -> None:
        if step not in self.native_meta.native_loop_steps:
            self.native_meta.native_loop_steps.append(step)
        self.native_meta.native_loop_trace.record(step, summary=summary, metadata=metadata or {})

    def run_tool(self, call: NativeToolCall) -> NativeToolResult:
        if self._runner is None:
            result = NativeToolResult(
                tool_name=call.tool_name,
                ok=False,
                error="No repo_root configured for tool execution.",
            )
            self.native_meta.tool_trace.append(
                {
                    "tool": call.tool_name,
                    "ok": False,
                    "approved": call.approved,
                    "output_chars": 0,
                    "error": result.error,
                }
            )
            return result

        result = self._runner.run(call)
        self.native_meta.tool_trace.append(self._runner.trace_entry(call, result))
        return result

    def _run_preflight(self) -> None:
        if self._runner is None:
            return

        call = NativeToolCall(tool_name="list_files", args={"subdir": "."})
        result = self._runner.run(call)
        self.native_meta.tool_trace.append(self._runner.trace_entry(call, result))

        if result.ok:
            if self.native_meta.context_budget is None:
                self.native_meta.context_budget = build_initial_context_budget()
            self.native_meta.context_budget.repo_map_built = True
            self.native_meta.repo_context_summary = build_repo_context_summary(result.output)
            self.record_loop_step("repo_context")

    def _run_observe_phase(self, task: str, repo_facts=None) -> None:
        if self._runner is None:
            return

        observation = NativeObservation()

        diff_call = NativeToolCall(tool_name="get_git_diff", args={})
        diff_result = self._runner.run(diff_call)
        self.native_meta.tool_trace.append(self._runner.trace_entry(diff_call, diff_result))
        observation.observed_tools.append("get_git_diff")
        if diff_result.ok and diff_result.output.strip():
            observation.dirty_diff_present = True

        task_lower = task.lower()
        if any(trigger in task_lower.split() for trigger in _SEARCH_TRIGGER_WORDS):
            query = _extract_search_query(task)
            if query:
                search_call = NativeToolCall(tool_name="search_repo", args={"query": query})
                search_result = self._runner.run(search_call)
                self.native_meta.tool_trace.append(self._runner.trace_entry(search_call, search_result))
                observation.observed_tools.append("search_repo")
                if search_result.ok:
                    observation.search_matches_count = search_result.metadata.get("matches", 0)
                    raw_lines = [ln for ln in search_result.output.splitlines() if ln.strip()]
                    truncated = search_result.metadata.get("truncated", False) or len(raw_lines) > 3
                    file_snippets, read_traces = _build_file_snippets_from_search(
                        self._runner,
                        raw_lines[:3],
                    )
                    self.native_meta.tool_trace.extend(read_traces)
                    self.native_meta.evidence = NativeEvidence(
                        search_results=raw_lines[:3],
                        file_snippets=file_snippets,
                        truncated=truncated,
                    )
                    self.record_loop_step("evidence")

        self.native_meta.observation = observation
        self.record_loop_step("observation")

    def _run_read_search_loop(self, task: str) -> None:
        if self._runner is None:
            return

        task_lower = task.lower()
        if any(kw in task_lower for kw in ("test", "pytest", "failing", "bug", "flaky")):
            strategy = "test"
        elif any(kw in task_lower for kw in ("cli", "command", "argument")):
            strategy = "cli"
        else:
            strategy = "default"

        findings: list[str] = []
        files_checked: int = 0
        matched_terms: int = 0
        steps_taken: int = 0
        truncated: bool = False

        rcs = self.native_meta.repo_context_summary

        # Step 1: mine test_markers from already-collected repo context (no new tool call)
        if steps_taken < _MAX_READ_SEARCH_STEPS and rcs is not None:
            steps_taken += 1
            for marker in rcs.test_markers:
                if len(findings) >= _MAX_LOOP_FINDINGS:
                    truncated = True
                    break
                findings.append(f"test-marker:{marker}")
                matched_terms += 1

        # Step 2: mine package_files from already-collected repo context (no new tool call)
        if steps_taken < _MAX_READ_SEARCH_STEPS and rcs is not None:
            steps_taken += 1
            for pkg in rcs.package_files:
                if len(findings) >= _MAX_LOOP_FINDINGS:
                    truncated = True
                    break
                findings.append(f"package:{pkg}")
                matched_terms += 1

        # Step 3: one optional targeted search_repo call based on strategy
        if steps_taken < _MAX_READ_SEARCH_STEPS and len(findings) < _MAX_LOOP_FINDINGS:
            steps_taken += 1
            if strategy == "test":
                query = "test"
            elif strategy == "cli":
                query = "cli"
            else:
                query = _extract_search_query(task) or "main"

            search_call = NativeToolCall(tool_name="search_repo", args={"query": query})
            search_result = self._runner.run(search_call)
            self.native_meta.tool_trace.append(self._runner.trace_entry(search_call, search_result))

            if search_result.ok:
                raw_lines = [ln for ln in search_result.output.splitlines() if ln.strip()]
                for line in raw_lines:
                    if len(findings) >= _MAX_LOOP_FINDINGS:
                        truncated = True
                        break
                    parsed = _parse_search_result_line(line)
                    if parsed is not None:
                        path, _ = parsed
                        label = f"file:{path}"
                        if label not in findings:
                            findings.append(label)
                            files_checked += 1
                            matched_terms += 1

        self.native_meta.read_search_findings = findings[:_MAX_LOOP_FINDINGS]

        self.record_loop_step(
            "read_search",
            summary=f"strategy={strategy} steps={steps_taken} findings={len(findings)}",
            metadata={
                "steps": steps_taken,
                "findings": len(findings),
                "files_checked": files_checked,
                "matched_terms": matched_terms,
                "truncated": truncated,
                "strategy": strategy,
            },
        )

    def _run_file_context_phase(self) -> None:
        if self._runner is None:
            return
        findings = list(self.native_meta.read_search_findings)
        if not findings:
            return

        candidates: list[str] = []
        for f in findings:
            for prefix in ("file:", "test-marker:", "package:"):
                if f.startswith(prefix):
                    candidates.append(f[len(prefix):])
                    break

        safe: list[str] = []
        for p in candidates:
            if os.path.isabs(p):
                continue
            parts = p.replace("\\", "/").split("/")
            if ".." in parts or ".git" in parts:
                continue
            safe.append(p)

        if not safe:
            return

        files_read = 0
        total_chars = 0
        truncated = False
        paths_read: list[str] = []
        warnings: list[str] = []

        for path in safe[:_MAX_FILE_CONTEXT_FILES]:
            read_call = NativeToolCall(tool_name="read_file", args={"path": path})
            read_result = self._runner.run(read_call)
            self.native_meta.tool_trace.append(
                self._runner.trace_entry(read_call, read_result)
            )

            if not read_result.ok:
                warnings.append(f"could not read: {path}")
                continue

            content = read_result.output or ""
            chars = len(content)

            if total_chars + min(chars, _MAX_FILE_CONTEXT_CHARS_PER_FILE) > _MAX_FILE_CONTEXT_TOTAL_CHARS:
                truncated = True
                break
            if chars > _MAX_FILE_CONTEXT_CHARS_PER_FILE:
                chars = _MAX_FILE_CONTEXT_CHARS_PER_FILE
                truncated = True

            total_chars += chars
            paths_read.append(path)
            files_read += 1

        self.native_meta.file_context = NativeFileContext(
            files_read=files_read,
            paths=paths_read,
            total_chars=total_chars,
            truncated=truncated,
            warnings=warnings,
        )
        self.record_loop_step(
            "file_context",
            metadata={"files_read": files_read, "total_chars": total_chars, "truncated": truncated},
        )

    def _run_backend_proof_phase(self, task: str) -> None:
        if self._backend.name != "deepagents":
            return
        context: dict[str, Any] = {"experimental_deepagents_run": True}
        if self._deepagents_model:
            context["deepagents_model"] = self._deepagents_model
        rcs = self.native_meta.repo_context_summary
        if rcs is not None:
            context["repo_context_summary"] = " ".join(
                getattr(rcs, "likely_stack_markers", [])
            )
        result = self._backend.run(task=task, context=context)
        proof = result.metadata.get("proof")
        if proof:
            self.native_meta.native_backend_proof = proof
        if result.notes:
            self.native_meta.native_backend_notes.extend(result.notes)
        self.record_loop_step("backend_proof")

    def review_diff(self) -> NativeDiffReview | None:
        if self._runner is None:
            return None

        call = NativeToolCall(tool_name="get_git_diff", args={})
        result = self._runner.run(call)
        self.native_meta.tool_trace.append(self._runner.trace_entry(call, result))

        if not result.ok:
            return None

        review = build_native_diff_review(
            result.output,
            truncated=bool(result.metadata.get("truncated", False)),
        )
        self.native_meta.diff_review = review
        self.record_loop_step("diff_review")
        return review

    def build_command_policy_preview(
        self, verification_plan: Any | None
    ) -> NativeCommandPolicyPreview:
        preview = build_native_command_policy_preview(verification_plan)
        self.native_meta.command_policy_preview = preview
        self.record_loop_step(
            "command_policy",
            summary=(
                f"{preview.safe_count} safe / "
                f"{preview.needs_approval_count} approval / "
                f"{preview.blocked_count} blocked"
            ),
            metadata={
                "safe": preview.safe_count,
                "needs_approval": preview.needs_approval_count,
                "blocked": preview.blocked_count,
            },
        )
        return preview

    def build_verification_command_summary(
        self, verification_plan: Any | None
    ) -> NativeVerificationCommandSummary:
        summary = build_native_verification_command_summary(
            verification_loop=self.native_meta.verification_loop,
            verification_plan=verification_plan,
        )
        self.native_meta.verification_command_summary = summary
        self.record_loop_step(
            "verification_summary",
            summary=f"{summary.command_count} verification commands",
            metadata={
                "commands": summary.command_count,
                "safe": summary.safe_count,
                "needs_approval": summary.needs_approval_count,
                "blocked": summary.blocked_count,
                "passed": summary.passed,
                "retried": summary.retried,
            },
        )
        return summary

    def build_final_report(self) -> NativeFinalReport:
        report = build_native_final_report(
            selected_skills=self.native_meta.selected_skills,
            observation=self.native_meta.observation,
            evidence=self.native_meta.evidence,
            plan=self.native_meta.plan,
            verification_loop=self.native_meta.verification_loop,
            diff_review=self.native_meta.diff_review,
        )
        self.native_meta.final_report = report
        self.record_loop_step("final_report")
        return report

    def build_patch_proposal(self, files: list[Any]) -> NativePatchProposal:
        proposal = build_native_patch_proposal(files)
        self.native_meta.patch_proposal = proposal
        self.record_loop_step(
            "proposal",
            summary=f"{proposal.file_count} proposed file changes",
            metadata={
                "files": proposal.file_count,
                "change_types": len(set(proposal.change_types)),
            },
        )
        return proposal

    def build_change_budget_preview(self) -> NativeChangeBudgetPreview:
        preview = build_native_change_budget_preview(
            budget=self.native_meta.change_budget,
            proposal=self.native_meta.patch_proposal,
        )
        self.native_meta.change_budget_preview = preview
        self.record_loop_step(
            "change_budget_preview",
            summary=(
                f"{preview.proposed_files}/{preview.budget_max_files} files "
                f"action={preview.action}"
            ),
            metadata={
                "budget_max_files": preview.budget_max_files,
                "proposed_files": preview.proposed_files,
                "within_budget": preview.within_budget,
                "would_exceed_budget": preview.would_exceed_budget,
                "action": preview.action,
            },
        )
        return preview

    def build_change_budget_soft_gate(self) -> NativeChangeBudgetSoftGate:
        gate = build_native_change_budget_soft_gate(self.native_meta.change_budget_preview)
        self.native_meta.change_budget_soft_gate = gate
        self.record_loop_step(
            "change_budget_soft_gate",
            summary=f"{gate.action} approval={gate.requires_approval}",
            metadata={
                "requires_approval": gate.requires_approval,
                "action": gate.action,
            },
        )
        return gate

    def build_budget_gate_approval_request(self) -> NativeApprovalRequest:
        request = build_native_budget_gate_approval_request(
            gate=self.native_meta.change_budget_soft_gate,
            preview=self.native_meta.change_budget_preview,
        )
        self.native_meta.approval_request = request
        self.record_loop_step(
            "approval_request",
            summary=f"{request.source} approval={request.requires_approval}",
            metadata={
                "source": request.source,
                "requires_approval": request.requires_approval,
                "action": request.action,
                "proposed_files": request.proposed_files,
                "budget_max_files": request.budget_max_files,
            },
        )
        return request

    def build_approval_receipt(self, *, granted: bool) -> NativeApprovalReceipt:
        receipt = build_native_approval_receipt(
            self.native_meta.approval_request,
            granted=granted,
        )
        self.native_meta.approval_receipt = receipt
        self.record_loop_step(
            "approval_receipt",
            summary=f"{receipt.source} granted={receipt.granted}",
            metadata={
                "source": receipt.source,
                "requested": receipt.requested,
                "granted": receipt.granted,
                "action": receipt.action,
            },
        )
        return receipt

    def build_context_packet(self, task: str) -> NativeContextPacket:
        file_ctx = self.native_meta.file_context
        packet = build_native_context_packet(
            task=task,
            repo_context_summary=self.native_meta.repo_context_summary,
            read_search_findings=self.native_meta.read_search_findings,
            selected_skills=self.native_meta.selected_skills,
            native_backend=self.native_meta.native_backend,
            native_backend_available=self.native_meta.native_backend_available,
            native_backend_proof=self.native_meta.native_backend_proof,
            file_context_files=file_ctx.files_read if file_ctx is not None else 0,
        )
        self.native_meta.context_packet = packet
        self.record_loop_step(
            "context_packet",
            summary=f"{len(packet.sources)} sources, {len(packet.compact_paths)} paths",
            metadata={
                "sources": len(packet.sources),
                "paths": len(packet.compact_paths),
            },
        )
        return packet

    def build_context_quality_score(self) -> NativeContextQualityScore:
        packet = self.native_meta.context_packet
        if packet is None:
            packet = NativeContextPacket()
        score = build_native_context_quality_score(packet)
        self.native_meta.context_quality_score = score
        self.record_loop_step(
            "context_quality",
            summary=f"{score.level} {score.score}/{score.max_score}",
            metadata={
                "score": score.score,
                "max_score": score.max_score,
                "level": score.level,
                "reasons": score.reasons,
            },
        )
        return score

    def build_context_quality_advisory(self) -> NativeContextQualityAdvisory:
        advisory = build_native_context_quality_advisory(
            self.native_meta.context_quality_score
        )
        self.native_meta.context_quality_advisory = advisory
        self.record_loop_step(
            "context_quality_advisory",
            summary=advisory.recommendation,
            metadata={
                "level": advisory.level,
                "should_block": advisory.should_block,
                "warnings": len(advisory.warnings),
            },
        )
        return advisory

    def build_change_budget(self) -> NativeChangeBudget:
        budget = build_native_change_budget(self.native_meta.context_quality_advisory)
        self.native_meta.change_budget = budget
        self.record_loop_step(
            "change_budget",
            summary=f"{budget.level} max_files={budget.max_files} size={budget.max_change_size}",
            metadata={
                "level": budget.level,
                "max_files": budget.max_files,
                "max_change_size": budget.max_change_size,
            },
        )
        return budget

    def generate(
        self,
        task: str,
        model: str | None = None,
        repo_facts: RepoFacts | None = None,
        skills_context: str = "",
    ) -> ExecutionResult:
        self._run_preflight()
        if self._experimental_deepagents_run:
            self._run_backend_proof_phase(task)
        self._run_observe_phase(task, repo_facts=repo_facts)
        self._run_read_search_loop(task)
        self._run_file_context_phase()
        matches = match_builtin_skills(task, repo_facts=repo_facts)
        self.native_meta.selected_skills = selected_skill_names(matches)
        self.build_context_packet(task)
        self.build_context_quality_score()
        self.build_context_quality_advisory()
        self.build_change_budget()

        self.native_meta.plan = _build_native_plan(
            task,
            observation=self.native_meta.observation,
            evidence=self.native_meta.evidence,
            selected_skills=self.native_meta.selected_skills,
        )
        self.record_loop_step("plan")

        self.native_meta.verification_plan = build_native_verification_plan(
            task=task,
            plan=self.native_meta.plan,
            change_budget=self.native_meta.change_budget,
            read_search_findings=self.native_meta.read_search_findings,
            repo_facts=repo_facts,
        )
        self.record_loop_step("verification_plan")

        context_parts = []

        summary = self.native_meta.repo_context_summary
        if summary is not None:
            context_parts.append(render_repo_context_summary(summary))

        observation = self.native_meta.observation
        if observation is not None:
            context_parts.append(render_native_observation(observation))

        evidence = self.native_meta.evidence
        if evidence is not None:
            context_parts.append(render_native_evidence(evidence))

        context_parts.append(render_native_plan(self.native_meta.plan))

        packet_context = render_native_context_packet(self.native_meta.context_packet)
        if packet_context:
            context_parts.append(packet_context)

        advisory_context = render_native_context_quality_advisory(
            self.native_meta.context_quality_advisory
        )
        if advisory_context:
            context_parts.append(advisory_context)

        change_budget_context = render_native_change_budget(self.native_meta.change_budget)
        if change_budget_context:
            context_parts.append(change_budget_context)

        if skills_context:
            context_parts.append(skills_context)

        combined_context = "\n\n".join(context_parts) if context_parts else skills_context

        if context_parts and self.native_meta.context_budget is not None:
            self.native_meta.context_budget.estimated_tokens_used += (
                len(combined_context) // 4
            )

        result = self._gen.generate(
            task, model=model, repo_facts=repo_facts, skills_context=combined_context
        )
        self.record_loop_step("generation")
        return result