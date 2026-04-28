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


_ESCALATION_MIN_RUNS = 5
_ESCALATION_PASS_RATE_THRESHOLD = 0.70
_ESCALATION_RETRY_RATE_THRESHOLD = 0.30


def select_profile(
    category: str,
    repo_facts: RepoFacts | None,
    task: str,
    override: str | None,
    history_summary: dict[ProfileName, ProfileHistorySummary] | None = None,
) -> ProfileDecision:
    """Select an execution profile for the given task signals.

    Priority:
    1. Explicit override always wins (raises ValueError if invalid).
    2. native_swarm is never auto-selected.
    3. security/complex category, risky paths, or long task → native_deep.
    4. History-aware escalation: native_light → native_deep when past
       native_light performance is poor (low pass rate or high retry rate).
    5. Otherwise → native_light.
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

    if history_summary is not None:
        light = history_summary.get("native_light")
        if light is not None and light.runs >= _ESCALATION_MIN_RUNS:
            escalation_reasons: list[str] = []
            if light.verification_pass_rate is not None and light.verification_pass_rate < _ESCALATION_PASS_RATE_THRESHOLD:
                escalation_reasons.append("native_light history pass rate below threshold")
            if light.retry_rate is not None and light.retry_rate > _ESCALATION_RETRY_RATE_THRESHOLD:
                escalation_reasons.append("native_light history retry rate above threshold")
            if escalation_reasons:
                return ProfileDecision(profile="native_deep", reason="; ".join(escalation_reasons))

    return ProfileDecision(profile="native_light", reason="simple/safe task")


@dataclass
class ProfileHistorySummary:
    profile: ProfileName
    runs: int
    verification_pass_rate: float | None
    retry_rate: float | None
    avg_cost: float | None
    avg_duration: float | None


def build_profile_history_summary(
    runs: list[dict],
) -> dict[ProfileName, ProfileHistorySummary]:
    """Summarise past run history by execution profile.

    Always returns entries for all three profiles. Runs without an
    execution_profile field are ignored.
    """
    from openshard.history.metrics import compute_profile_stats

    stats = compute_profile_stats(runs)
    return {
        cast(ProfileName, profile): ProfileHistorySummary(
            profile=cast(ProfileName, profile),
            runs=data["runs_count"],
            verification_pass_rate=data["verification_pass_rate"],
            retry_rate=data["retry_rate"],
            avg_cost=data["avg_cost"],
            avg_duration=data["avg_duration"],
        )
        for profile, data in stats.items()
    }
