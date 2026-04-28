from __future__ import annotations

import pytest
from openshard.analysis.repo import RepoFacts
from openshard.routing.workflow_selector import (
    MIN_CATEGORY_RUNS,
    WorkflowHistorySummary,
    build_workflow_history_summary,
    select_workflow,
    WorkflowDecision,
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


def _summary(retry_rate: float, pass_rate: float, n: int = MIN_CATEGORY_RUNS) -> WorkflowHistorySummary:
    return WorkflowHistorySummary(
        retry_rate=retry_rate,
        verification_pass_rate=pass_rate,
        sample_count=n,
    )


# ---------------------------------------------------------------------------
# Base rules
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("category", ["security", "complex"])
def test_base_staged_categories(category):
    result = select_workflow(category, _facts(), None, False)
    assert result.workflow == "staged"
    assert result.reason == "category defaults to staged"


@pytest.mark.parametrize("category", ["visual", "boilerplate", "standard"])
def test_base_direct_categories(category):
    result = select_workflow(category, _facts(), None, False)
    assert result.workflow == "direct"
    assert result.reason == "category defaults to direct"


# ---------------------------------------------------------------------------
# Escalation: risky paths
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("category", ["visual", "boilerplate", "standard"])
def test_risky_paths_escalates_direct_category(category):
    result = select_workflow(category, _facts(["auth/login.py"]), None, False)
    assert result.workflow == "staged"
    assert result.reason == "risky paths detected"


# ---------------------------------------------------------------------------
# Escalation: retry rate
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("category", ["visual", "boilerplate", "standard"])
def test_high_retry_rate_escalates(category):
    result = select_workflow(category, _facts(), _summary(0.4, 0.9), False)
    assert result.workflow == "staged"
    assert result.reason == "high retry rate in this category"


def test_retry_rate_below_threshold_does_not_escalate():
    assert select_workflow("standard", _facts(), _summary(0.39, 0.9), False).workflow == "direct"


# ---------------------------------------------------------------------------
# Escalation: verification pass rate
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("category", ["visual", "boilerplate", "standard"])
def test_low_pass_rate_escalates(category):
    result = select_workflow(category, _facts(), _summary(0.1, 0.3), False)
    assert result.workflow == "staged"
    assert result.reason == "low verification pass rate"


def test_pass_rate_above_threshold_does_not_escalate():
    assert select_workflow("standard", _facts(), _summary(0.1, 0.31), False).workflow == "direct"


# ---------------------------------------------------------------------------
# De-escalation
# ---------------------------------------------------------------------------

def test_deescalation_all_gates_pass():
    """security base → direct when history is excellent and no risky paths."""
    good = _summary(retry_rate=0.1, pass_rate=0.8)
    result = select_workflow("security", _facts(), good, False)
    assert result.workflow == "direct"
    assert result.reason == "history cleared gates for staged category"


def test_deescalation_complex_base():
    good = _summary(retry_rate=0.05, pass_rate=0.95)
    assert select_workflow("complex", _facts(), good, False).workflow == "direct"


def test_deescalation_blocked_by_risky_paths():
    good = _summary(retry_rate=0.1, pass_rate=0.9)
    assert select_workflow("security", _facts(["payment/stripe.py"]), good, False).workflow == "staged"


def test_deescalation_blocked_by_high_retry():
    borderline = _summary(retry_rate=0.2, pass_rate=0.9)
    assert select_workflow("security", _facts(), borderline, False).workflow == "staged"


def test_deescalation_blocked_by_low_pass_rate():
    borderline = _summary(retry_rate=0.05, pass_rate=0.79)
    assert select_workflow("security", _facts(), borderline, False).workflow == "staged"


# ---------------------------------------------------------------------------
# Insufficient history
# ---------------------------------------------------------------------------

def test_no_history_uses_base_rules():
    assert select_workflow("standard", _facts(), None, False).workflow == "direct"
    assert select_workflow("security", _facts(), None, False).workflow == "staged"


def test_insufficient_samples_skips_history_signals():
    low_sample = _summary(retry_rate=0.9, pass_rate=0.0, n=MIN_CATEGORY_RUNS - 1)
    # poor signals but too few samples → base rule wins (direct for standard)
    assert select_workflow("standard", _facts(), low_sample, False).workflow == "direct"


def test_insufficient_samples_skips_deescalation():
    low_sample = _summary(retry_rate=0.0, pass_rate=1.0, n=MIN_CATEGORY_RUNS - 1)
    # good signals but too few samples → base rule wins (staged for security)
    assert select_workflow("security", _facts(), low_sample, False).workflow == "staged"


# ---------------------------------------------------------------------------
# verify_enabled does not escalate alone
# ---------------------------------------------------------------------------

def test_verify_alone_does_not_escalate_standard():
    assert select_workflow("standard", _facts(), None, True).workflow == "direct"


def test_verify_alone_does_not_escalate_boilerplate():
    assert select_workflow("boilerplate", _facts(), None, True).workflow == "direct"


# ---------------------------------------------------------------------------
# build_workflow_history_summary
# ---------------------------------------------------------------------------

def test_build_summary_empty_runs():
    assert build_workflow_history_summary([], "standard") is None


def test_build_summary_no_matching_category():
    runs = [{"routing_category": "security", "retry_triggered": False}]
    assert build_workflow_history_summary(runs, "standard") is None


def test_build_summary_computes_rates():
    runs = [
        {"routing_category": "standard", "retry_triggered": True, "verification_passed": True},
        {"routing_category": "standard", "retry_triggered": False, "verification_passed": False},
        {"routing_category": "standard", "retry_triggered": False, "verification_passed": None},
        {"routing_category": "security", "retry_triggered": True, "verification_passed": True},
    ]
    s = build_workflow_history_summary(runs, "standard")
    assert s is not None
    assert s.sample_count == 3
    assert s.retry_rate == pytest.approx(1 / 3)
    # verification_passed is not None for 2 runs; 1 passed
    assert s.verification_pass_rate == pytest.approx(0.5)


def test_build_summary_no_verif_runs():
    runs = [
        {"routing_category": "standard", "retry_triggered": False},
        {"routing_category": "standard", "retry_triggered": False},
    ]
    s = build_workflow_history_summary(runs, "standard")
    assert s is not None
    assert s.verification_pass_rate == 0.0
