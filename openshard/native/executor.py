from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from openshard.analysis.repo import RepoFacts
from openshard.execution.generator import ExecutionGenerator, ExecutionResult
from openshard.native.context import (
    CompactRunState,
    NativeClarificationRequest,
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
    NativeContextProvenance,
    NativeContextUsageSummary,
    NativeDiffReview,
    NativeEvidence,
    NativeFailureMemory,
    NativeFileContext,
    NativeFileSnippet,
    NativeFinalReport,
    NativeObservation,
    NativePatchProposal,
    NativePlan,
    NativeVerificationCommandSummary,
    NativeVerificationLoop,
    NativeVerificationPlan,
    NativeRunTrustScore,
    NativeModelSelectionDecision,
    NativeModelCandidateScoring,
    NativeModelPolicy,
    NativeModelPolicyReceipt,
    NativeRoutingPreview,
    NativeRoutingReceipt,
    NativeTierDispatchReceipt,
    NativeValidationContract,
    OSNLoopMeta,
    OSNLoopStep,
    build_initial_context_budget,
    build_native_approval_receipt,
    build_native_budget_gate_approval_request,
    build_native_clarification_request,
    build_native_command_policy_preview,
    build_native_context_packet,
    build_native_change_budget,
    build_native_change_budget_preview,
    build_native_change_budget_soft_gate,
    build_native_context_quality_advisory,
    build_native_context_quality_score,
    build_native_context_provenance,
    build_native_context_usage_summary,
    build_native_diff_review,
    build_native_final_report,
    build_native_patch_proposal,
    build_native_verification_command_summary,
    build_native_verification_plan,
    build_native_validation_contract,
    build_native_model_policy,
    build_osn_loop_meta,
    render_native_change_budget,
    render_native_context_packet,
    render_native_context_quality_advisory,
    render_native_evidence,
    render_native_observation,
    render_native_plan,
    render_native_validation_contract,
    render_osn_loop_context,
)
from openshard.native.repo_context import (
    NativeRepoContextSummary,
    build_repo_context_summary,
    render_repo_context_summary,
)
from openshard.native.backends import DeepAgentsAdapterMeta, build_deepagents_adapter_meta
from openshard.native.loop import NativeLoopTrace
from openshard.native.skills import match_builtin_skills, selected_skill_names
from openshard.native.tool_runner import NativeToolRunner
from openshard.native.tools import NativeToolCall, NativeToolResult, NativeToolSearchEvent


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
    tool_search_events: list[NativeToolSearchEvent] = field(default_factory=list)
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
    clarification_request: NativeClarificationRequest | None = None
    validation_contract: NativeValidationContract | None = None
    context_usage_summary: NativeContextUsageSummary | None = None
    failure_memory: NativeFailureMemory | None = None
    osn_loop: OSNLoopMeta | None = None
    deepagents_adapter: DeepAgentsAdapterMeta | None = None
    context_provenance: NativeContextProvenance | None = None
    run_trust_score: NativeRunTrustScore | None = None
    model_selection_decision: NativeModelSelectionDecision | None = None
    model_candidate_scoring: NativeModelCandidateScoring | None = None
    model_policy: NativeModelPolicy | None = None
    model_policy_receipt: NativeModelPolicyReceipt | None = None
    routing_preview: NativeRoutingPreview | None = None
    routing_receipt: NativeRoutingReceipt | None = None
    tier_dispatch_receipt: NativeTierDispatchReceipt | None = None


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

_LOOP_ALLOWED_TOOLS: frozenset[str] = frozenset({
    "list_files",
    "read_file",
    "search_repo",
    "get_git_diff",
})
_MAX_OSN_LOOP_STEPS: int = 5
_MAX_OSN_QUEUE_CAP: int = 10
_MAX_OSN_CONSECUTIVE_EMPTY: int = 2


def _infer_result_count(tool_name: str, result: NativeToolResult) -> int:
    if not result.ok:
        return 0
    if tool_name == "search_repo":
        return result.metadata.get("matches", 0)
    if tool_name == "list_files":
        return len([ln for ln in result.output.splitlines() if ln.strip()])
    if tool_name in ("get_git_diff", "read_file"):
        return 1 if result.output.strip() else 0
    return 0


def _infer_result_quality(result_count: int, context_injected: bool) -> str:
    if result_count == 0:
        return "empty"
    if context_injected:
        return "useful"
    return "weak"


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


# ── Explicit-file outline helpers ─────────────────────────────────────────────

def _extract_module_docstring(lines: list[str]) -> str | None:
    """Return the first content line of the module docstring, or None."""
    i = 0
    while i < len(lines) and not lines[i].strip():
        i += 1
    if i >= len(lines):
        return None
    stripped = lines[i].strip()
    for delim in ('"""', "'''"):
        if stripped.startswith(delim):
            inner = stripped[len(delim):]
            if inner.endswith(delim) and len(inner) >= len(delim):
                return inner[: -len(delim)].strip() or None
            return inner.strip() or None
    return None


_IMPORT_RE = re.compile(r"^(?:from\s+([\w.]+)|import\s+([\w., ]+))")


def _collect_import_names(lines: list[str]) -> list[str]:
    """Return deduplicated root module names from top-of-file imports."""
    seen: list[str] = []
    in_docstring = False
    docstring_delim: str = ""
    for line in lines:
        stripped = line.strip()
        if in_docstring:
            if docstring_delim in stripped:
                in_docstring = False
            continue
        if not stripped or stripped.startswith("#"):
            continue
        for delim in ('"""', "'''"):
            if stripped.startswith(delim):
                rest = stripped[len(delim):]
                if delim not in rest:
                    in_docstring = True
                    docstring_delim = delim
                break
        if in_docstring:
            continue
        m = _IMPORT_RE.match(stripped)
        if m:
            raw = m.group(1) or m.group(2).split(",")[0].strip()
            root = raw.split(".")[0].split()[0]
            if root and root not in seen:
                seen.append(root)
        elif stripped.startswith("__") and "=" in stripped:
            continue
        elif stripped.startswith(("@", "def ", "class ", "async def ")):
            break
    return seen


_COMMAND_DECORATOR_RE = re.compile(r"^\s*@\w+\.(?:command|group)\b")


def _has_command_decorator(lines: list[str], def_idx: int) -> bool:
    """Return True if the def at def_idx is preceded by a @x.command/@x.group decorator."""
    i = def_idx - 1
    while i >= 0:
        stripped = lines[i].strip()
        if _COMMAND_DECORATOR_RE.match(lines[i]):
            return True
        if stripped.startswith("@") or not stripped or stripped.startswith("#"):
            i -= 1
            continue
        break
    return False


def _build_generic_outline(content: str, max_lines: int) -> list[str]:
    """Return first max_lines non-blank lines verbatim."""
    result: list[str] = []
    for line in content.splitlines():
        if line.strip():
            result.append(line)
            if len(result) >= max_lines:
                break
    return result


_TOP_LEVEL_DEF_RE = re.compile(r"^(?:async )?def (\w+)|^class (\w+)")


def _build_explicit_file_outline(content: str, path: str, *, max_lines: int = 25) -> list[str]:
    """Build a compact, bounded outline of *content* for explicit-file model evidence.

    For .py files: module docstring, import summary, top-level def/class names,
    and @xxx.command decorated functions labeled as [commands].
    For non-.py files: first max_lines non-blank lines verbatim.
    Falls back to generic outline when no Python symbols are found.
    """
    if not path.endswith(".py"):
        return _build_generic_outline(content, max_lines)

    lines = content.splitlines()
    outline: list[str] = []

    doc = _extract_module_docstring(lines)
    if doc:
        outline.append(f"[docstring] {doc}")

    import_names = _collect_import_names(lines)
    if import_names:
        shown = import_names[:6]
        suffix = ", ..." if len(import_names) > 6 else ""
        outline.append(f"[imports] {', '.join(shown)}{suffix}")

    defs: list[str] = []
    classes: list[str] = []
    commands: list[str] = []

    for idx, line in enumerate(lines):
        m = _TOP_LEVEL_DEF_RE.match(line)
        if m:
            name = m.group(1) or m.group(2)
            if m.group(2):
                classes.append(name)
            elif _has_command_decorator(lines, idx):
                commands.append(name)
            else:
                defs.append(name)

    if classes:
        outline.append(f"[classes] {', '.join(classes)}")
    if defs:
        outline.append(f"[defs] {', '.join(defs)}")
    if commands:
        outline.append(f"[commands] {', '.join(commands)}")

    if not outline:
        return _build_generic_outline(content, max_lines)

    return outline[:max_lines]


# ─────────────────────────────────────────────────────────────────────────────


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


_EXPLICIT_PATH_RE = re.compile(
    r"(?<![:/\w])"
    r"([\w][\w.-]*/[\w./][\w./-]*\.[\w]{1,10}"
    r"|[\w][\w.-]*\.(?:py|md|txt|toml|json|yaml|yml|cfg|ini|sh|js|ts|rs|go|rb|tsx|jsx|lock|env))"
    r"(?![\w/])"
)
_MAX_EXPLICIT_SNIPPET_FILES: int = 2


def _extract_explicit_file_paths(task: str) -> list[str]:
    """Return unique apparent repo-relative file paths found literally in task text."""
    seen: list[str] = []
    for m in _EXPLICIT_PATH_RE.finditer(task):
        path = m.group(1)
        if path not in seen:
            seen.append(path)
    return seen


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
        native_loop: str | None = None,
        model_policy: str | None = None,
    ) -> None:
        from openshard.native.backends import get_backend

        self._gen = ExecutionGenerator(provider=provider)
        self.model = self._gen.model
        self.fixer_model = self._gen.fixer_model
        self.native_meta = NativeRunMeta()
        self.native_meta.model_policy = build_native_model_policy(model_policy)
        self._runner = NativeToolRunner(repo_root) if repo_root is not None else None
        self._backend = get_backend(backend_name)
        _available = self._backend.available()
        self.native_meta.native_backend = self._backend.name
        self.native_meta.native_backend_available = _available
        self._experimental_deepagents_run = experimental_deepagents_run
        self._deepagents_model = deepagents_model  # None in production; injectable for tests
        self._native_loop = native_loop
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

    def _record_tool_search_event(
        self,
        tool_name: str,
        result: NativeToolResult,
        selected_reason: str = "",
        query: str = "",
        context_injected: bool = False,
        warnings: list[str] | None = None,
    ) -> NativeToolSearchEvent:
        count = _infer_result_count(tool_name, result)
        quality = _infer_result_quality(count, context_injected)
        event = NativeToolSearchEvent(
            tool_name=tool_name,
            selected_reason=selected_reason,
            query=query,
            result_count=count,
            result_quality=quality,
            context_injected=context_injected,
            warnings=list(warnings or []),
        )
        self.native_meta.tool_search_events.append(event)
        return event

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
        self._record_tool_search_event(
            "list_files", result,
            selected_reason="preflight scan",
            query=".",
            context_injected=result.ok,
        )

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
        self._record_tool_search_event(
            "get_git_diff", diff_result,
            selected_reason="observe dirty diff",
            context_injected=diff_result.ok and bool(diff_result.output.strip()),
        )
        observation.observed_tools.append("get_git_diff")
        if diff_result.ok and diff_result.output.strip():
            observation.dirty_diff_present = True

        evidence: NativeEvidence | None = None

        task_lower = task.lower()
        if any(trigger in task_lower.split() for trigger in _SEARCH_TRIGGER_WORDS):
            query = _extract_search_query(task)
            if query:
                search_call = NativeToolCall(tool_name="search_repo", args={"query": query})
                search_result = self._runner.run(search_call)
                self.native_meta.tool_trace.append(self._runner.trace_entry(search_call, search_result))
                self._record_tool_search_event(
                    "search_repo", search_result,
                    selected_reason="observe search trigger",
                    query=query,
                    context_injected=search_result.ok and search_result.metadata.get("matches", 0) > 0,
                )
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
                    evidence = NativeEvidence(
                        search_results=raw_lines[:3],
                        file_snippets=file_snippets,
                        truncated=truncated,
                    )

        # Gather content for any repo-relative file paths named explicitly in the task,
        # regardless of whether search trigger words are present.
        explicit_paths = _extract_explicit_file_paths(task)
        existing_snippet_paths = {s.path for s in evidence.file_snippets} if evidence else set()
        for path in explicit_paths:
            if len(existing_snippet_paths) >= _MAX_EXPLICIT_SNIPPET_FILES:
                break
            if path in existing_snippet_paths:
                continue
            if not self._is_safe_repo_path(path):
                continue
            read_call = NativeToolCall(tool_name="read_file", args={"path": path})
            result = self._runner.run(read_call)
            self.native_meta.tool_trace.append(self._runner.trace_entry(read_call, result))
            if not result.ok:
                continue
            snippet_lines = _build_explicit_file_outline(result.output, path)
            if not snippet_lines:
                continue
            snippet = NativeFileSnippet(path=path, lines=snippet_lines)
            if evidence is None:
                evidence = NativeEvidence(file_snippets=[snippet])
            else:
                evidence.file_snippets.append(snippet)
            existing_snippet_paths.add(path)

        if evidence is not None:
            self.native_meta.evidence = evidence
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
            self._record_tool_search_event(
                "search_repo", search_result,
                selected_reason=f"read-search strategy={strategy}",
                query=query,
                context_injected=search_result.ok and bool(search_result.output.strip()),
            )

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

    @staticmethod
    def _is_safe_repo_path(raw: str) -> bool:
        """Return True only for safe repo-relative paths.

        Rejects: absolute POSIX paths, Windows drive-letter paths, home-relative
        paths, POSIX traversal (../), backslash traversal (..), and any path
        that normalizes outside the repo root after separator normalization.
        """
        if not raw:
            return False
        normalized = raw.replace("\\", "/")
        if normalized.startswith("/"):
            return False
        if normalized.startswith("~"):
            return False
        if len(normalized) >= 2 and normalized[1] == ":":
            return False
        parts = normalized.split("/")
        if ".." in parts:
            return False
        return True

    @staticmethod
    def _sanitize_target_label(tool_name: str, raw: str) -> str:
        """Return a sanitized compact label for an OSN loop step.

        File paths: validated as safe repo-relative, backslashes normalized,
        truncated to 200 chars.
        Search steps: always returns the fixed label 'task_keyword_search'.
        """
        if tool_name == "search_repo":
            return "task_keyword_search"
        if tool_name in ("read_file", "list_files"):
            if not NativeAgentExecutor._is_safe_repo_path(raw):
                return ""
            normalized = raw.replace("\\", "/")
            return normalized[:200]
        return ""

    def _build_loop_step_queue(self, task: str) -> list[tuple[str, str, str]]:
        """Build a deterministic step queue from already-gathered findings.

        Returns list of (tool_name, raw_target, reason) tuples.
        All tool_names are in _LOOP_ALLOWED_TOOLS.
        No model or AI is consulted — queue is built from static prior data.
        """
        queue: list[tuple[str, str, str]] = []
        seen_paths: set[str] = set()

        for finding in self.native_meta.read_search_findings:
            if len(queue) >= _MAX_OSN_QUEUE_CAP:
                break
            if finding.startswith("file:"):
                path = finding[len("file:"):]
                if path and path not in seen_paths and self._is_safe_repo_path(path):
                    seen_paths.add(path)
                    queue.append(("read_file", path, "read_search_finding"))
            elif finding.startswith("test-marker:"):
                path = finding[len("test-marker:"):]
                if path and path not in seen_paths and self._is_safe_repo_path(path):
                    seen_paths.add(path)
                    queue.append(("read_file", path, "test_marker"))

        if len(queue) < _MAX_OSN_QUEUE_CAP:
            query = _extract_search_query(task)
            if query:
                queue.append(("search_repo", query, "task_keyword"))

        evidence = self.native_meta.evidence
        if evidence is not None:
            for snippet in evidence.file_snippets:
                if len(queue) >= _MAX_OSN_QUEUE_CAP:
                    break
                path = snippet.path
                if path and path not in seen_paths and self._is_safe_repo_path(path):
                    seen_paths.add(path)
                    queue.append(("read_file", path, "evidence_snippet"))

        return queue[:_MAX_OSN_QUEUE_CAP]

    @staticmethod
    def _osn_loop_args(tool_name: str, raw_target: str) -> dict:
        if tool_name == "read_file":
            return {"path": raw_target}
        if tool_name == "search_repo":
            return {"query": raw_target}
        if tool_name == "list_files":
            return {"subdir": raw_target or "."}
        if tool_name == "get_git_diff":
            return {}
        return {}

    def _run_experimental_loop(self, task: str) -> None:
        """Bounded deterministic OSN loop — safe read-only tools only.

        Runs only when --native-loop experimental is passed. write_file,
        run_command, and run_verification are structurally excluded — they
        are absent from _LOOP_ALLOWED_TOOLS and the queue builder never
        produces them.
        """
        if self._runner is None:
            return

        queue = self._build_loop_step_queue(task)
        steps_queued = len(queue)

        if steps_queued == 0:
            meta = OSNLoopMeta(
                enabled=True,
                steps_run=0,
                steps_queued=0,
                max_steps=_MAX_OSN_LOOP_STEPS,
                terminated_reason="no_steps",
            )
            self.native_meta.osn_loop = meta
            self.record_loop_step(
                "osn_loop",
                summary="no_steps queued",
                metadata={"steps_run": 0, "steps_queued": 0, "terminated_reason": "no_steps"},
            )
            return

        executed_steps: list[OSNLoopStep] = []
        consecutive_empty = 0
        warnings: list[str] = []
        terminated_reason = "complete"

        for idx, (tool_name, raw_target, reason) in enumerate(queue):
            if idx >= _MAX_OSN_LOOP_STEPS:
                terminated_reason = "max_steps"
                break

            if tool_name not in _LOOP_ALLOWED_TOOLS:
                warnings.append(f"skipped disallowed tool: {tool_name}")
                executed_steps.append(OSNLoopStep(
                    step_index=idx,
                    tool_name=tool_name,
                    target_label=self._sanitize_target_label(tool_name, raw_target),
                    reason=reason,
                    skipped=True,
                ))
                continue

            args = self._osn_loop_args(tool_name, raw_target)
            call = NativeToolCall(tool_name=tool_name, args=args)
            result = self._runner.run(call)
            self.native_meta.tool_trace.append(self._runner.trace_entry(call, result))

            output_chars = len(result.output) if result.ok else 0
            is_empty = result.ok and not result.output.strip()
            self._record_tool_search_event(
                tool_name, result,
                selected_reason=reason,
                query=str(raw_target),
                context_injected=result.ok and not is_empty,
            )

            if is_empty:
                consecutive_empty += 1
            else:
                consecutive_empty = 0

            executed_steps.append(OSNLoopStep(
                step_index=idx,
                tool_name=tool_name,
                target_label=self._sanitize_target_label(tool_name, raw_target),
                reason=reason,
                ok=result.ok,
                output_chars=output_chars,
                empty=is_empty,
            ))

            if consecutive_empty >= _MAX_OSN_CONSECUTIVE_EMPTY:
                terminated_reason = "consecutive_empty"
                break

        steps_run = sum(1 for s in executed_steps if not s.skipped)
        meta = build_osn_loop_meta(
            steps_run=steps_run,
            steps_queued=steps_queued,
            max_steps=_MAX_OSN_LOOP_STEPS,
            consecutive_empty=consecutive_empty,
            terminated_reason=terminated_reason,
            steps=executed_steps,
            warnings=warnings,
        )
        self.native_meta.osn_loop = meta
        self.record_loop_step(
            "osn_loop",
            summary=(
                f"steps={steps_run}/{_MAX_OSN_LOOP_STEPS}"
                f" paths={len(meta.paths_surfaced)}"
                f" reason={terminated_reason}"
            ),
            metadata={
                "steps_run": steps_run,
                "steps_queued": steps_queued,
                "paths_surfaced": len(meta.paths_surfaced),
                "terminated_reason": terminated_reason,
                "truncated": meta.truncated,
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

    def _run_deepagents_adapter_phase(self) -> None:
        meta = build_deepagents_adapter_meta()
        self.native_meta.deepagents_adapter = meta
        self.record_loop_step(
            "deepagents_adapter",
            summary=f"mode={meta.mode} available={meta.available}",
            metadata={"mode": meta.mode, "available": meta.available},
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
        if self._backend.name == "deepagents":
            self._run_deepagents_adapter_phase()
        if self._experimental_deepagents_run:
            self._run_backend_proof_phase(task)
        self._run_observe_phase(task, repo_facts=repo_facts)
        self._run_read_search_loop(task)
        if self._native_loop == "experimental":
            self._run_experimental_loop(task)
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

        self.native_meta.clarification_request = build_native_clarification_request(
            task=task,
            verification_plan=self.native_meta.verification_plan,
        )
        if self.native_meta.clarification_request.needed:
            self.record_loop_step("clarification_request")

        validation_contract = build_native_validation_contract(
            task=task,
            plan=self.native_meta.plan,
            verification_plan=self.native_meta.verification_plan,
            change_budget=self.native_meta.change_budget,
            change_budget_preview=None,
            change_budget_soft_gate=None,
            clarification_request=self.native_meta.clarification_request,
            context_quality_score=self.native_meta.context_quality_score,
        )
        self.native_meta.validation_contract = validation_contract
        self.record_loop_step("validation_contract")

        context_parts = []
        _injected_sources: set[str] = set()

        summary = self.native_meta.repo_context_summary
        if summary is not None:
            context_parts.append(render_repo_context_summary(summary))
            _injected_sources.add("repo_summary")

        observation = self.native_meta.observation
        if observation is not None:
            context_parts.append(render_native_observation(observation))
            _injected_sources.add("observation")

        evidence = self.native_meta.evidence
        if evidence is not None:
            context_parts.append(render_native_evidence(evidence, limit=3000, max_lines_per_snippet=25))
            _injected_sources.add("evidence")

        osn_block = render_osn_loop_context(self.native_meta.osn_loop)
        if osn_block:
            context_parts.append(osn_block)
            _injected_sources.add("osn_loop")

        rendered_plan = render_native_plan(self.native_meta.plan)
        if rendered_plan:
            context_parts.append(rendered_plan)
            _injected_sources.add("plan")

        rendered_contract = render_native_validation_contract(validation_contract)
        if rendered_contract:
            context_parts.append(rendered_contract)
            _injected_sources.add("validation_contract")

        packet_context = render_native_context_packet(self.native_meta.context_packet)
        if packet_context:
            context_parts.append(packet_context)
            _injected_sources.add("context_packet")

        advisory_context = render_native_context_quality_advisory(
            self.native_meta.context_quality_advisory
        )
        if advisory_context:
            context_parts.append(advisory_context)
            _injected_sources.add("advisory")

        change_budget_context = render_native_change_budget(self.native_meta.change_budget)
        if change_budget_context:
            context_parts.append(change_budget_context)
            _injected_sources.add("change_budget")

        if skills_context:
            context_parts.append(skills_context)
            _injected_sources.add("skills_context")

        combined_context = "\n\n".join(context_parts) if context_parts else skills_context

        if context_parts and self.native_meta.context_budget is not None:
            self.native_meta.context_budget.estimated_tokens_used += (
                len(combined_context) // 4
            )

        self.native_meta.context_usage_summary = build_native_context_usage_summary(
            repo_context_summary=self.native_meta.repo_context_summary,
            file_context=self.native_meta.file_context,
            context_packet=self.native_meta.context_packet,
            evidence=self.native_meta.evidence,
            observation=self.native_meta.observation,
            plan=self.native_meta.plan,
            context_quality_score=self.native_meta.context_quality_score,
            final_report=self.native_meta.final_report,
            diff_review=self.native_meta.diff_review,
            verification_loop=self.native_meta.verification_loop,
            total_chars=len(combined_context) if combined_context else 0,
        )

        self.native_meta.context_provenance = build_native_context_provenance(
            repo_context_summary=self.native_meta.repo_context_summary,
            observation=self.native_meta.observation,
            evidence=self.native_meta.evidence,
            read_search_findings=self.native_meta.read_search_findings,
            file_context=self.native_meta.file_context,
            context_packet=self.native_meta.context_packet,
            context_quality_score=self.native_meta.context_quality_score,
            context_quality_advisory=self.native_meta.context_quality_advisory,
            change_budget=self.native_meta.change_budget,
            plan=self.native_meta.plan,
            verification_plan=self.native_meta.verification_plan,
            clarification_request=self.native_meta.clarification_request,
            validation_contract=self.native_meta.validation_contract,
            context_usage_summary=self.native_meta.context_usage_summary,
            osn_loop=self.native_meta.osn_loop,
            skills_context=skills_context,
            injected_source_names=_injected_sources,
        )
        self.record_loop_step("context_provenance")

        result = self._gen.generate(
            task, model=model, repo_facts=repo_facts, skills_context=combined_context
        )
        self.record_loop_step("generation")
        return result