from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from openshard.analysis.repo import RepoFacts
from openshard.execution.generator import ExecutionGenerator, ExecutionResult
from openshard.native.context import (
    CompactRunState,
    NativeContextBudget,
    NativeDiffReview,
    NativeEvidence,
    NativeFileSnippet,
    NativeObservation,
    NativePlan,
    NativeVerificationLoop,
    build_initial_context_budget,
    build_native_diff_review,
    render_native_evidence,
    render_native_observation,
    render_native_plan,
)
from openshard.native.repo_context import (
    NativeRepoContextSummary,
    build_repo_context_summary,
    render_repo_context_summary,
)
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


_SEARCH_STOP_WORDS: frozenset[str] = frozenset({
    "the", "a", "an", "in", "on", "at", "to", "for", "of",
    "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did",
    "will", "would", "should", "could", "may", "might", "can",
})

_SEARCH_TRIGGER_WORDS: frozenset[str] = frozenset({"where", "find", "search", "locate"})


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

    def __init__(self, provider=None, repo_root: Path | None = None) -> None:
        self._gen = ExecutionGenerator(provider=provider)
        self.model = self._gen.model
        self.fixer_model = self._gen.fixer_model
        self.native_meta = NativeRunMeta()
        self._runner = NativeToolRunner(repo_root) if repo_root is not None else None

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

        self.native_meta.observation = observation

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
        return review

    def generate(
        self,
        task: str,
        model: str | None = None,
        repo_facts: RepoFacts | None = None,
        skills_context: str = "",
    ) -> ExecutionResult:
        self._run_preflight()
        self._run_observe_phase(task, repo_facts=repo_facts)
        matches = match_builtin_skills(task, repo_facts=repo_facts)
        self.native_meta.selected_skills = selected_skill_names(matches)

        self.native_meta.plan = _build_native_plan(
            task,
            observation=self.native_meta.observation,
            evidence=self.native_meta.evidence,
            selected_skills=self.native_meta.selected_skills,
        )

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

        if skills_context:
            context_parts.append(skills_context)

        combined_context = "\n\n".join(context_parts) if context_parts else skills_context

        if context_parts and self.native_meta.context_budget is not None:
            self.native_meta.context_budget.estimated_tokens_used += (
                len(combined_context) // 4
            )

        return self._gen.generate(
            task, model=model, repo_facts=repo_facts, skills_context=combined_context
        )