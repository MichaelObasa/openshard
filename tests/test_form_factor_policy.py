from __future__ import annotations

import pytest
from openshard.analysis.repo import RepoFacts
from openshard.routing.form_factor_policy import (
    ExecutionFormFactorDecision,
    _derive_risk_level,
    _native_loop_enabled,
    select_form_factor,
)


def _facts(risky_paths: list[str] | None = None) -> RepoFacts:
    return RepoFacts(
        languages=["python"],
        package_files=[],
        framework=None,
        test_command=None,
        risky_paths=risky_paths or [],
        changed_files=[],
    )


def _select(**kwargs) -> ExecutionFormFactorDecision:
    defaults = dict(
        category="standard",
        readonly=False,
        workflow="direct",
        profile_name="native_light",
        repo_facts=None,
        write_requested=True,
        verification_available=False,
        native_loop=None,
        experimental_deepagents_run=False,
        context_quality_level=None,
    )
    defaults.update(kwargs)
    return select_form_factor(**defaults)


# ---------------------------------------------------------------------------
# _native_loop_enabled helper
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("value", ["auto", "bounded", "true", "1", "yes", "on", "enabled"])
def test_native_loop_enabled_helper_positive_values(value):
    assert _native_loop_enabled(value) is True


@pytest.mark.parametrize("value", [None, "", "off", "none", "disabled", "false", "0", "no"])
def test_native_loop_enabled_helper_negative_values(value):
    assert _native_loop_enabled(value) is False


# ---------------------------------------------------------------------------
# _derive_risk_level helper
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("category", ["security", "auth", "payments", "infra", "migration", "infrastructure"])
def test_derive_risk_high_for_risk_categories(category):
    assert _derive_risk_level(category, None, write_requested=False) == "high"


def test_derive_risk_high_for_write_with_risky_paths():
    assert _derive_risk_level("standard", _facts(["payments/stripe.py"]), write_requested=True) == "high"


def test_derive_risk_not_high_for_readonly_with_risky_paths():
    assert _derive_risk_level("standard", _facts(["payments/stripe.py"]), write_requested=False) == "low"


def test_derive_risk_medium_for_complex():
    assert _derive_risk_level("complex", None, write_requested=False) == "medium"


def test_derive_risk_low_for_standard():
    assert _derive_risk_level("standard", None, write_requested=False) == "low"


# ---------------------------------------------------------------------------
# select_form_factor — core rules
# ---------------------------------------------------------------------------

def test_simple_readonly_selects_ask_direct():
    result = _select(readonly=True, write_requested=False)
    assert result.public_mode == "ask"
    assert result.internal_form_factor == "direct"
    assert result.confidence == "high"
    assert result.reason == "read-only task"


def test_simple_safe_write_selects_run_direct():
    result = _select(category="standard", workflow="direct")
    assert result.public_mode == "run"
    assert result.internal_form_factor == "direct"
    assert result.confidence == "high"
    assert result.reason == "simple safe task"


def test_staged_workflow_selects_run_staged():
    result = _select(category="standard", workflow="staged")
    assert result.public_mode == "run"
    assert result.internal_form_factor == "staged"
    assert result.confidence == "high"
    assert result.reason == "staged planning selected"


def test_security_category_selects_deep_run_native_loop():
    result = _select(category="security")
    assert result.public_mode == "deep-run"
    assert result.internal_form_factor == "native-loop-candidate"
    assert result.confidence == "medium"


def test_risky_paths_write_selects_deep_run_with_warning():
    result = _select(
        category="standard",
        repo_facts=_facts(["payments/billing.py", "auth/tokens.py"]),
        write_requested=True,
    )
    assert result.public_mode == "deep-run"
    assert result.internal_form_factor == "native-loop-candidate"
    assert result.risk_level == "high"
    assert any("risky path" in w for w in result.warnings)


def test_risky_paths_readonly_stays_ask():
    result = _select(
        category="standard",
        readonly=True,
        write_requested=False,
        repo_facts=_facts(["payments/billing.py"]),
    )
    assert result.public_mode == "ask"
    assert result.internal_form_factor == "direct"


def test_native_swarm_profile_selects_swarm_candidate():
    result = _select(profile_name="native_swarm")
    assert result.public_mode == "deep-run"
    assert result.internal_form_factor == "swarm-candidate"
    assert result.confidence == "medium"


def test_native_loop_enabled_selects_osn_run():
    result = _select(native_loop="auto")
    assert result.public_mode == "osn-run"
    assert result.internal_form_factor == "native-loop-candidate"
    assert result.confidence == "high"
    assert result.reason == "explicit native loop requested"


@pytest.mark.parametrize("value", ["off", "disabled", "none", "false", "0", None])
def test_native_loop_disabled_values_do_not_select_osn_run(value):
    result = _select(native_loop=value)
    assert result.public_mode != "osn-run"


def test_experimental_deepagents_selects_subagent_candidate():
    result = _select(experimental_deepagents_run=True)
    assert result.public_mode == "deep-run"
    assert result.internal_form_factor == "subagent-candidate"
    assert result.confidence == "medium"


def test_complex_staged_stays_run_not_deep_run():
    result = _select(category="complex", workflow="staged")
    assert result.public_mode == "run"
    assert result.internal_form_factor == "staged"
    assert result.risk_level == "medium"


def test_native_deep_profile_selects_deep_run():
    result = _select(profile_name="native_deep", category="standard")
    assert result.public_mode == "deep-run"
    assert result.internal_form_factor == "native-loop-candidate"


# ---------------------------------------------------------------------------
# Warning signals
# ---------------------------------------------------------------------------

def test_weak_context_adds_warning_not_changes_mode():
    result = _select(category="standard", workflow="direct", context_quality_level="weak")
    assert result.public_mode == "run"
    assert result.internal_form_factor == "direct"
    assert any("context quality" in w for w in result.warnings)


def test_unknown_context_adds_warning():
    result = _select(context_quality_level="unknown")
    assert any("context quality" in w for w in result.warnings)


def test_strong_context_adds_no_warning():
    result = _select(context_quality_level="strong")
    assert not any("context quality" in w for w in result.warnings)


def test_risky_paths_readonly_no_warning():
    result = _select(
        readonly=True,
        write_requested=False,
        repo_facts=_facts(["payments/stripe.py"]),
    )
    assert not any("risky path" in w for w in result.warnings)


# ---------------------------------------------------------------------------
# Priority ordering
# ---------------------------------------------------------------------------

def test_native_loop_takes_priority_over_readonly():
    result = _select(native_loop="auto", readonly=True, write_requested=False)
    assert result.public_mode == "osn-run"


def test_readonly_takes_priority_over_security_category():
    result = _select(category="security", readonly=True, write_requested=False)
    assert result.public_mode == "ask"


def test_deepagents_takes_priority_over_staged():
    result = _select(experimental_deepagents_run=True, workflow="staged")
    assert result.public_mode == "deep-run"
    assert result.internal_form_factor == "subagent-candidate"


# ---------------------------------------------------------------------------
# No side effects — policy is metadata only
# ---------------------------------------------------------------------------

def test_no_autonomous_execution_from_policy():
    # Calling select_form_factor with any combination of flags returns a pure
    # data object and has no observable side effects (no I/O, no state changes).
    for native_loop in ["auto", "bounded", None]:
        for deepagents in [True, False]:
            result = select_form_factor(
                category="standard",
                readonly=False,
                workflow="direct",
                profile_name="native_light",
                repo_facts=None,
                write_requested=True,
                verification_available=False,
                native_loop=native_loop,
                experimental_deepagents_run=deepagents,
            )
            assert isinstance(result, ExecutionFormFactorDecision)
            assert result.public_mode in {"ask", "run", "deep-run", "osn-run"}


# ---------------------------------------------------------------------------
# Output fields are complete
# ---------------------------------------------------------------------------

def test_decision_carries_all_required_fields():
    result = _select()
    assert result.public_mode is not None
    assert result.internal_form_factor is not None
    assert result.reason != ""
    assert result.confidence in {"low", "medium", "high"}
    assert result.risk_level in {"low", "medium", "high"}
    assert isinstance(result.read_only, bool)
    assert isinstance(result.write_requested, bool)
    assert isinstance(result.verification_available, bool)
    assert isinstance(result.warnings, list)
