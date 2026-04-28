from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, cast, get_args

if TYPE_CHECKING:
    from openshard.analysis.repo import RepoFacts

ProfileName = Literal["native_light", "native_deep", "native_swarm"]

_VALID_PROFILES: frozenset[str] = frozenset(get_args(ProfileName))
_DEEP_CATEGORIES = {"security", "complex"}
_LONG_TASK_WORDS = 80


@dataclass
class ProfileDecision:
    profile: ProfileName
    reason: str


def select_profile(
    category: str,
    repo_facts: RepoFacts | None,
    task: str,
    override: str | None,
) -> ProfileDecision:
    """Select an execution profile for the given task signals.

    Priority:
    1. Explicit override always wins (raises ValueError if invalid).
    2. native_swarm is never auto-selected.
    3. security/complex category, risky paths, or long task → native_deep.
    4. Otherwise → native_light.
    """
    if override is not None:
        normalized = override.lower()
        if normalized not in _VALID_PROFILES:
            valid = ", ".join(sorted(_VALID_PROFILES))
            raise ValueError(f"Invalid profile {override!r}. Valid values: {valid}")
        return ProfileDecision(profile=cast(ProfileName, normalized), reason="explicit override")

    signals: list[str] = []

    if category in _DEEP_CATEGORIES:
        signals.append(f"{category} category")

    risky = bool(repo_facts and repo_facts.risky_paths)
    if risky:
        signals.append("risky paths detected")

    if len(task.split()) > _LONG_TASK_WORDS:
        signals.append("long task description")

    if signals:
        return ProfileDecision(profile="native_deep", reason="; ".join(signals))

    return ProfileDecision(profile="native_light", reason="simple/safe task")
