from __future__ import annotations

from dataclasses import dataclass, field

from openshard.execution.generator import ExecutionGenerator, ExecutionResult
from openshard.analysis.repo import RepoFacts


@dataclass
class NativeRunMeta:
    workflow: str = "native"
    executor: str = "native"
    execution_depth: str = "fast"
    selected_skills: list = field(default_factory=list)
    context_budget: int | None = None
    tool_trace: list = field(default_factory=list)


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
        return self._gen.generate(
            task, model=model, repo_facts=repo_facts, skills_context=skills_context
        )
