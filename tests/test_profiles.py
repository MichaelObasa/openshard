from __future__ import annotations

import pytest
from openshard.analysis.repo import RepoFacts
from openshard.routing.profiles import ProfileDecision, select_profile


def _facts(risky_paths: list[str] | None = None) -> RepoFacts:
    return RepoFacts(
        languages=["python"],
        package_files=[],
        framework=None,
        test_command=None,
        risky_paths=risky_paths or [],
        changed_files=[],
    )


def _short_task() -> str:
    return "fix typo in README"


def _long_task() -> str:
    return " ".join(["word"] * 81)


# ---------------------------------------------------------------------------
# native_light — simple/safe tasks
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("category", ["standard", "visual", "boilerplate"])
def test_light_for_simple_categories(category):
    result = select_profile(category, _facts(), _short_task(), None)
    assert result.profile == "native_light"
    assert result.reason == "simple/safe task"


def test_light_no_repo_facts():
    result = select_profile("standard", None, _short_task(), None)
    assert result.profile == "native_light"


# ---------------------------------------------------------------------------
# native_deep — serious categories
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("category", ["security", "complex"])
def test_deep_for_serious_categories(category):
    result = select_profile(category, _facts(), _short_task(), None)
    assert result.profile == "native_deep"
    assert category in result.reason


# ---------------------------------------------------------------------------
# native_deep — risky paths
# ---------------------------------------------------------------------------

def test_deep_when_risky_paths_present():
    result = select_profile("standard", _facts(["auth/tokens.py"]), _short_task(), None)
    assert result.profile == "native_deep"
    assert "risky paths" in result.reason


def test_light_when_no_risky_paths():
    result = select_profile("standard", _facts([]), _short_task(), None)
    assert result.profile == "native_light"


# ---------------------------------------------------------------------------
# native_deep — long task
# ---------------------------------------------------------------------------

def test_deep_when_task_is_long():
    result = select_profile("standard", _facts(), _long_task(), None)
    assert result.profile == "native_deep"
    assert "long task" in result.reason


def test_light_when_task_exactly_at_boundary():
    task = " ".join(["word"] * 80)  # exactly 80 words — not > 80
    result = select_profile("standard", _facts(), task, None)
    assert result.profile == "native_light"


# ---------------------------------------------------------------------------
# Multiple signals — reason concatenates
# ---------------------------------------------------------------------------

def test_reason_concatenates_multiple_signals():
    result = select_profile("security", _facts(["auth/login.py"]), _short_task(), None)
    assert result.profile == "native_deep"
    assert "security category" in result.reason
    assert "risky paths" in result.reason


def test_reason_all_three_signals():
    result = select_profile("security", _facts(["auth/login.py"]), _long_task(), None)
    assert "security category" in result.reason
    assert "risky paths" in result.reason
    assert "long task" in result.reason


# ---------------------------------------------------------------------------
# Explicit overrides
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("override", ["native_light", "native_deep", "native_swarm"])
def test_override_wins_for_all_profiles(override):
    result = select_profile("security", _facts(["auth/x.py"]), _long_task(), override)
    assert result.profile == override
    assert result.reason == "explicit override"


def test_override_wins_even_for_simple_task():
    result = select_profile("standard", _facts(), _short_task(), "native_swarm")
    assert result.profile == "native_swarm"
    assert result.reason == "explicit override"


def test_override_is_case_insensitive():
    result = select_profile("standard", _facts(), _short_task(), "NATIVE_DEEP")
    assert result.profile == "native_deep"


def test_invalid_override_raises():
    import pytest
    with pytest.raises(ValueError, match="Invalid profile"):
        select_profile("standard", _facts(), _short_task(), "turbo_agent")


# ---------------------------------------------------------------------------
# native_swarm never auto-selected
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("category", ["security", "complex", "standard", "visual", "boilerplate"])
def test_swarm_never_auto_selected(category):
    result = select_profile(category, _facts(["auth/x.py"]), _long_task(), None)
    assert result.profile != "native_swarm"


# ---------------------------------------------------------------------------
# Return type
# ---------------------------------------------------------------------------

def test_returns_profile_decision_dataclass():
    result = select_profile("standard", _facts(), _short_task(), None)
    assert isinstance(result, ProfileDecision)
    assert isinstance(result.profile, str)
    assert isinstance(result.reason, str)
