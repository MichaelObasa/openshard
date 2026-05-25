from __future__ import annotations

from openshard.tui.plan_mode import answer_plan_mode


def test_returns_string():
    assert isinstance(answer_plan_mode("refactor auth"), str)


def test_plan_header_present():
    assert "PLAN" in answer_plan_mode("refactor auth")


def test_plan_goal_section():
    result = answer_plan_mode("refactor auth")
    assert "Goal" in result
    assert "refactor auth" in result


def test_plan_suggested_approach():
    assert "Suggested approach" in answer_plan_mode("refactor auth")


def test_plan_risk_notes():
    assert "Risk notes" in answer_plan_mode("refactor auth")


def test_plan_next_step():
    assert "Next step" in answer_plan_mode("refactor auth")


def test_local_plan_only_disclaimer():
    result = answer_plan_mode("refactor auth")
    assert "No repo scan" in result
    assert "no provider call" in result
    assert "no files changed" in result


def test_empty_task_returns_usage_hint():
    result = answer_plan_mode("")
    assert "/plan" in result
    assert not result.startswith("PLAN\n")


def test_whitespace_only_task_returns_usage_hint():
    result = answer_plan_mode("   ")
    assert "/plan" in result
    assert not result.startswith("PLAN\n")


def test_task_text_preserved_in_output():
    assert "add rate limiting to the API" in answer_plan_mode("add rate limiting to the API")


def test_task_stripped():
    assert "refactor auth" in answer_plan_mode("  refactor auth  ")
