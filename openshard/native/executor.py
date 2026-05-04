from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from openshard.analysis.repo import RepoFacts
from openshard.execution.generator import ExecutionGenerator, ExecutionResult
from openshard.native.context import (
    CompactRunState,
    NativeContextBudget,
    NativeObservation,
    build_initial_context_budget,
    render_native_observation,
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


_SEARCH_STOP_WORDS: frozenset[str] = frozenset({
    "the", "a", "an", "in", "on", "at", "to", "for", "of",
    "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did",
    "will", "would", "should", "could", "may", "might", "can",
})

_SEARCH_TRIGGER_WORDS: frozenset[str] = frozenset({"where", "find", "search", "locate"})


def _extract_search_query(task: str) -> str | None:
    filtered = []
    for raw in task.lower().split():
        word = raw.strip(".,:;!?()[]{}\"'`")
        if word and word not in _SEARCH_STOP_WORDS and word not in _SEARCH_TRIGGER_WORDS and len(word) >= 3:
            filtered.append(word)
    if not filtered:
        return None
    return " ".join(filtered[:3])


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

        self.native_meta.observation = observation

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

        context_parts = []

        summary = self.native_meta.repo_context_summary
        if summary is not None:
            context_parts.append(render_repo_context_summary(summary))

        observation = self.native_meta.observation
        if observation is not None:
            context_parts.append(render_native_observation(observation))

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