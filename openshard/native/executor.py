from __future__ import annotations

from dataclasses import dataclass, field

from openshard.analysis.repo import RepoFacts
from openshard.execution.generator import ExecutionGenerator, ExecutionResult
from openshard.native.context import CompactRunState, NativeContextBudget
from openshard.native.skills import match_builtin_skills, selected_skill_names


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

    def __init__(self, provider=None) -> None:
        self._gen = ExecutionGenerator(provider=provider)
        self.model = self._gen.model
        self.fixer_model = self._gen.fixer_model
        self.native_meta = NativeRunMeta()

    def generate(
        self,
        task: str,
        model: str | None = None,
        repo_facts: RepoFacts | None = None,
        skills_context: str = "",
    ) -> ExecutionResult:
        matches = match_builtin_skills(task, repo_facts=repo_facts)
        self.native_meta.selected_skills = selected_skill_names(matches)
        return self._gen.generate(
            task, model=model, repo_facts=repo_facts, skills_context=skills_context
        )
