from dataclasses import dataclass

from openshard.analysis.repo import RepoFacts
from openshard.skills.discovery import SkillDef

_MAX_MATCHED = 5


@dataclass
class MatchedSkill:
    skill: SkillDef
    reasons: list[str]


def match_skills(
    skills: list[SkillDef],
    task: str,
    category: str,
    repo_facts: RepoFacts,
) -> list[MatchedSkill]:
    """Match skill defs against task, routing category, and repo facts.

    A skill qualifies if any of these hold:
    - skill.category == category
    - any skill keyword appears in task text
    - skill.framework matches repo_facts.framework
    - skill.languages overlap repo_facts.languages AND skill.category == category
    """
    task_lower = task.lower()
    matched: list[MatchedSkill] = []

    for skill in skills:
        reasons: list[str] = []

        if skill.category == category:
            reasons.append(f"category: {category}")

        for kw in skill.keywords:
            if kw.lower() in task_lower:
                reasons.append(f"keyword: {kw}")
                break

        if skill.framework and skill.framework == repo_facts.framework:
            reasons.append(f"framework: {skill.framework}")

        # Language match only qualifies when category also matches
        if skill.languages and skill.category == category:
            for lang in skill.languages:
                if lang in repo_facts.languages:
                    reasons.append(f"language: {lang}")
                    break

        if reasons:
            matched.append(MatchedSkill(skill=skill, reasons=reasons))

    matched.sort(key=lambda ms: len(ms.reasons), reverse=True)
    return matched[:_MAX_MATCHED]
