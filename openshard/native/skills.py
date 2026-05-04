from __future__ import annotations

from dataclasses import dataclass

from openshard.analysis.repo import RepoFacts


@dataclass
class NativeSkill:
    name: str
    description: str
    categories: list[str]
    triggers: list[str]
    priority: int = 0
    built_in: bool = True


@dataclass
class NativeSkillMatch:
    skill: NativeSkill
    reason: str
    score: float


_BUILTIN_SKILLS: list[NativeSkill] = [
    NativeSkill(
        name="core-engineering-discipline",
        description="General engineering principles for well-structured, maintainable changes.",
        categories=["engineering", "general"],
        triggers=["implement", "add", "build", "create", "refactor"],
        priority=0,
    ),
    NativeSkill(
        name="repo-map",
        description="Navigate and map the repository structure before making changes.",
        categories=["navigation", "exploration"],
        triggers=["find", "where", "map", "explore", "locate", "search"],
        priority=0,
    ),
    NativeSkill(
        name="context-discipline",
        description="Keep context small and focused; load only what is needed.",
        categories=["context", "efficiency"],
        triggers=["context", "compact", "memory"],
        priority=0,
    ),
    NativeSkill(
        name="token-discipline",
        description="Respect context window budget; avoid unnecessary token usage.",
        categories=["tokens", "efficiency"],
        triggers=["token", "budget"],
        priority=0,
    ),
    NativeSkill(
        name="safe-file-editing",
        description="Edit files carefully using minimal, targeted diffs.",
        categories=["editing", "safety"],
        triggers=["edit", "modify", "change", "update", "file"],
        priority=0,
    ),
    NativeSkill(
        name="test-discovery",
        description="Locate and understand the test suite before writing or running tests.",
        categories=["testing"],
        triggers=["test", "spec", "pytest", "unittest", "coverage", "assert"],
        priority=0,
    ),
    NativeSkill(
        name="verification-fix-loop",
        description="Run, observe failures, fix, and repeat until green.",
        categories=["testing", "debugging"],
        triggers=["fix", "debug", "verify", "fail", "broken", "error"],
        priority=0,
    ),
    NativeSkill(
        name="diff-review",
        description="Review diffs and pull requests with structured attention.",
        categories=["review", "git"],
        triggers=["review", "diff", "pr", "pull request"],
        priority=0,
    ),
    NativeSkill(
        name="security-sensitive-change",
        description="Apply extra care when touching auth, secrets, payments, or permissions.",
        categories=["security"],
        triggers=[
            "auth", "authentication", "authorization", "secret", "password",
            "payment", "iam", "credential", "permission", "encrypt",
        ],
        priority=10,
    ),
]


def list_builtin_skills() -> list[NativeSkill]:
    return list(_BUILTIN_SKILLS)


def match_builtin_skills(
    task: str,
    *,
    repo_facts: RepoFacts | None = None,
    max_skills: int = 3,
) -> list[NativeSkillMatch]:
    task_lower = task.lower()
    matches: list[NativeSkillMatch] = []

    for skill in _BUILTIN_SKILLS:
        hit = [t for t in skill.triggers if t in task_lower]
        if hit:
            score = len(hit) + skill.priority * 0.1
            reason = f"task matches: {', '.join(hit)}"
            matches.append(NativeSkillMatch(skill=skill, reason=reason, score=score))

    matches.sort(key=lambda m: (-m.score, -m.skill.priority, m.skill.name))
    return matches[:max_skills]


def selected_skill_names(matches: list[NativeSkillMatch]) -> list[str]:
    return [m.skill.name for m in matches]
