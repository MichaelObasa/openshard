from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from openshard.analysis.repo import RepoFacts
from openshard.execution.generator import ExecutionGenerator, ExecutionResult
from openshard.native.context import (
    CompactRunState,
    NativeContextBudget,
    build_initial_context_budget,
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
            self.native_meta.tool_trace.append({
                "tool": call.tool_name,
                "ok": False,
                "approved": call.approved,
                "output_chars": 0,
                "error": result.error,
            })
            return result
        result = self._runner.run(call)
        self.native_meta.tool_trace.append(
            self._runner.trace_entry(call, result)
        )
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

    def generate(
        self,
        task: str,
        model: str | None = None,
        repo_facts: RepoFacts | None = None,
        skills_context: str = "",
    ) -> ExecutionResult:
        self._run_preflight()
        matches = match_builtin_skills(task, repo_facts=repo_facts)
        self.native_meta.selected_skills = selected_skill_names(matches)
        return self._gen.generate(
            task, model=model, repo_facts=repo_facts, skills_context=skills_context
        )
