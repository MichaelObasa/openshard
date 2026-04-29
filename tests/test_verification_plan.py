from __future__ import annotations

import sys
from dataclasses import dataclass
import pytest

from openshard.verification.plan import (
    CommandSafety,
    VerificationKind,
    VerificationPlan,
    VerificationSource,
    build_verification_plan,
    classify_command_safety,
    parse_command_to_argv,
    render_verification_plan,
)


@dataclass
class _FakeRepoFacts:
    test_command: str | None = None


# ---------------------------------------------------------------------------
# classify_command_safety
# ---------------------------------------------------------------------------


def test_empty_argv_is_blocked():
    safety, reason = classify_command_safety([], VerificationSource.detected)
    assert safety == CommandSafety.blocked
    assert "empty" in reason.lower()


@pytest.mark.parametrize(
    "argv",
    [
        ["curl", "http://example.com"],
        ["wget", "http://example.com"],
        ["rm", "-rf", "/"],
        ["sudo", "apt-get", "install", "foo"],
        ["chmod", "777", "file.py"],
        ["chown", "root", "file.py"],
    ],
)
def test_blocked_commands(argv: list[str]):
    safety, _ = classify_command_safety(argv, VerificationSource.detected)
    assert safety == CommandSafety.blocked


@pytest.mark.parametrize(
    "argv",
    [
        ["python", "-m", "pytest"],
        ["python", "-m", "pytest", "tests/"],
        ["pytest", "--tb=short"],
        ["npm", "test"],
        ["npm", "test", "--", "--coverage"],
        ["cargo", "test"],
        ["go", "test", "./..."],
        ["bundle", "exec", "rspec"],
        ["mvn", "test"],
        [sys.executable, "-m", "pytest"],
    ],
)
def test_safe_commands(argv: list[str]):
    safety, _ = classify_command_safety(argv, VerificationSource.detected)
    assert safety == CommandSafety.safe


@pytest.mark.parametrize(
    "argv",
    [
        ["make", "test"],
        ["tox"],
        ["./run_tests.sh"],
        ["node", "jest"],
    ],
)
def test_unknown_commands_need_approval(argv: list[str]):
    safety, _ = classify_command_safety(argv, VerificationSource.detected)
    assert safety == CommandSafety.needs_approval


@pytest.mark.parametrize(
    "argv",
    [
        ["python", "-m", "pytest", "&&", "rm", "-rf", "."],
        ["pytest; rm -rf ."],
        ["sh", "-c", "pytest $(echo foo)"],
        ["bash", "-c", "pytest `whoami`"],
    ],
)
def test_shell_metacharacters_are_blocked(argv: list[str]):
    safety, reason = classify_command_safety(argv, VerificationSource.detected)
    assert safety == CommandSafety.blocked
    assert "metachar" in reason.lower() or "shell" in reason.lower()


def test_compound_pytest_rm_is_blocked():
    # "python -m pytest && rm -rf ." parsed as argv keeps && as a token
    argv = ["python", "-m", "pytest", "&&", "rm", "-rf", "."]
    safety, _ = classify_command_safety(argv, VerificationSource.detected)
    assert safety == CommandSafety.blocked


# ---------------------------------------------------------------------------
# parse_command_to_argv
# ---------------------------------------------------------------------------


def test_parse_simple_command():
    assert parse_command_to_argv("pytest tests/") == ["pytest", "tests/"]


def test_parse_quoted_args():
    result = parse_command_to_argv('python -m pytest "my tests/"')
    assert result[0] == "python"
    assert result[-1] in ('"my tests/"', "my tests/")  # posix/non-posix difference


def test_parse_fails_gracefully():
    # Unmatched quote — shlex raises ValueError; we return [command]
    result = parse_command_to_argv("pytest 'unterminated")
    # On posix shlex this may parse differently; just ensure no exception
    assert isinstance(result, list)
    assert len(result) >= 1


# ---------------------------------------------------------------------------
# build_verification_plan
# ---------------------------------------------------------------------------


def test_empty_plan_when_no_config_and_no_repo_facts():
    plan = build_verification_plan({}, _FakeRepoFacts(test_command=None))
    assert not plan.has_commands
    assert plan.commands == []


def test_config_list_command_becomes_safe_config_command():
    plan = build_verification_plan(
        {"verification_command": ["python", "-m", "pytest"]},
        _FakeRepoFacts(),
    )
    assert plan.has_commands
    cmd = plan.commands[0]
    assert cmd.source == VerificationSource.config
    assert cmd.safety == CommandSafety.safe
    assert cmd.argv == ["python", "-m", "pytest"]


def test_config_string_command_parses_correctly():
    plan = build_verification_plan(
        {"verification_command": "python -m pytest tests/"},
        _FakeRepoFacts(),
    )
    assert plan.has_commands
    cmd = plan.commands[0]
    assert cmd.source == VerificationSource.config
    assert cmd.argv[:3] == ["python", "-m", "pytest"]


def test_detected_pytest_command_is_safe_detected():
    plan = build_verification_plan({}, _FakeRepoFacts(test_command="pytest"))
    assert plan.has_commands
    cmd = plan.commands[0]
    assert cmd.source == VerificationSource.detected
    assert cmd.safety == CommandSafety.safe
    assert cmd.kind == VerificationKind.test


def test_detected_npm_test_is_safe_detected():
    plan = build_verification_plan({}, _FakeRepoFacts(test_command="npm test"))
    assert plan.has_commands
    cmd = plan.commands[0]
    assert cmd.source == VerificationSource.detected
    assert cmd.safety == CommandSafety.safe
    assert cmd.kind == VerificationKind.test


def test_unknown_detected_command_needs_approval():
    plan = build_verification_plan({}, _FakeRepoFacts(test_command="make test"))
    assert plan.has_commands
    cmd = plan.commands[0]
    assert cmd.source == VerificationSource.detected
    assert cmd.safety == CommandSafety.needs_approval


def test_config_takes_priority_over_detected():
    plan = build_verification_plan(
        {"verification_command": ["pytest"]},
        _FakeRepoFacts(test_command="npm test"),
    )
    assert plan.has_commands
    assert plan.commands[0].source == VerificationSource.config


def test_safe_test_command_named_tests():
    plan = build_verification_plan({"verification_command": ["pytest"]}, _FakeRepoFacts())
    assert plan.commands[0].name == "tests"


def test_unsafe_command_named_verification():
    plan = build_verification_plan({"verification_command": ["make", "test"]}, _FakeRepoFacts())
    assert plan.commands[0].name == "verification"


# ---------------------------------------------------------------------------
# has_commands
# ---------------------------------------------------------------------------


def test_has_commands_true():
    plan = build_verification_plan({"verification_command": ["pytest"]}, _FakeRepoFacts())
    assert plan.has_commands is True


def test_has_commands_false():
    plan = VerificationPlan()
    assert plan.has_commands is False


# ---------------------------------------------------------------------------
# render_verification_plan
# ---------------------------------------------------------------------------


def test_render_empty_plan():
    plan = VerificationPlan()
    rendered = render_verification_plan(plan)
    assert rendered.startswith("Verification")
    assert "no verification command detected" in rendered


def test_render_shows_source_safety_argv():
    plan = build_verification_plan({}, _FakeRepoFacts(test_command="pytest"))
    rendered = render_verification_plan(plan)
    assert "Verification" in rendered
    assert "safe" in rendered
    assert "detected" in rendered
    assert "pytest" in rendered


def test_render_shows_config_source():
    plan = build_verification_plan(
        {"verification_command": ["python", "-m", "pytest"]},
        _FakeRepoFacts(),
    )
    rendered = render_verification_plan(plan)
    assert "config" in rendered
    assert "safe" in rendered
    assert "python -m pytest" in rendered
