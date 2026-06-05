from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import click
import pytest
from click.testing import CliRunner

from openshard.history.native_steps import NativeStepEvent
from openshard.native.context import build_native_command_policy_preview
from openshard.verification.executor import run_verification_plan
from openshard.verification.plan import (
    CommandSafety,
    VerificationCommand,
    VerificationKind,
    VerificationPlan,
    VerificationSource,
    classify_command_safety,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_cmd(
    argv: list[str],
    safety: CommandSafety,
    reason: str = "",
) -> VerificationCommand:
    return VerificationCommand(
        name="tests" if safety == CommandSafety.safe else "verification",
        argv=argv,
        kind=VerificationKind.test,
        source=VerificationSource.detected,
        safety=safety,
        reason=reason or f"safety={safety.value}",
    )


def _safe_plan(argv: list[str] | None = None) -> VerificationPlan:
    argv = argv or ["python", "-m", "pytest"]
    return VerificationPlan(commands=[
        _make_cmd(argv, CommandSafety.safe, "matches safe prefix: python -m pytest"),
    ])


def _blocked_plan(argv: list[str] | None = None) -> VerificationPlan:
    argv = argv or ["rm", "-rf", "."]
    return VerificationPlan(commands=[
        _make_cmd(argv, CommandSafety.blocked, "blocked executable: 'rm'"),
    ])


def _proc(returncode: int = 0, stdout: str = "") -> MagicMock:
    p = MagicMock()
    p.returncode = returncode
    p.stdout = stdout
    return p


def _render(entry: dict, detail: str = "more") -> str:
    from openshard.cli.main import _render_log_entry

    @click.command()
    def cmd():
        _render_log_entry(entry, detail)

    return CliRunner().invoke(cmd).output


def _native_entry_with_policy(
    safe: int = 1,
    approval: int = 0,
    blocked: int = 0,
    vplan_cmds: list[dict] | None = None,
) -> dict:
    """Minimal run history entry that triggers native command policy rendering."""
    cpp = {
        "safe_count": safe,
        "needs_approval_count": approval,
        "blocked_count": blocked,
        "command_classes": ["safe"] if safe else ["blocked"] if blocked else [],
        "warnings": [],
        "command_records": [
            {
                "command": "python -m pytest tests/",
                "classification": "safe",
                "decision_reason": "matches safe prefix: python -m pytest",
                "raw_content_stored": False,
            }
        ] if safe else [],
    }
    entry: dict = {
        "task": "test task",
        "workflow": "native",
        "command_policy_preview": cpp,
    }
    if vplan_cmds is not None:
        entry["verification_plan"] = vplan_cmds
    return entry


# ---------------------------------------------------------------------------
# Safe commands allowed
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "argv",
    [
        ["npm", "run", "test"],
        ["npm", "run", "lint"],
        ["npm", "run", "typecheck"],
        ["git", "status"],
        ["git", "diff"],
        ["git", "diff", "--stat"],
        ["git", "rev-parse", "HEAD"],
        ["grep", "foo", "."],
        ["rg", "pattern", "src/"],
    ],
)
def test_safe_commands_allowed(argv: list[str]) -> None:
    safety, reason = classify_command_safety(argv, VerificationSource.detected)
    assert safety == CommandSafety.safe, (
        f"Expected safe for {argv!r}, got {safety!r}: {reason}"
    )


# ---------------------------------------------------------------------------
# Dangerous commands blocked
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "argv",
    [
        ["git", "push"],
        ["git", "push", "origin", "main"],
        ["git", "reset", "--hard"],
        ["git", "reset", "--hard", "HEAD~1"],
        ["git", "reset", "--mixed", "HEAD"],
        ["git", "clean", "-fd"],
        ["git", "clean", "-f"],
        ["npm", "publish"],
        ["kubectl", "apply", "-f", "deploy.yaml"],
        ["kubectl", "delete", "pod", "mypod"],
        ["terraform", "apply"],
        ["terraform", "destroy"],
        ["terraform", "import", "aws_instance.foo", "i-1234"],
        ["terraform", "state", "rm", "aws_instance.foo"],
        ["helm", "install", "myrelease", "mychart"],
        ["helm", "upgrade", "myrelease", "mychart"],
        ["helm", "uninstall", "myrelease"],
        ["ansible", "all", "-m", "ping"],
        ["ansible-playbook", "site.yml"],
        ["printenv"],
        ["docker", "push", "myimage"],
    ],
)
def test_dangerous_commands_blocked(argv: list[str]) -> None:
    safety, reason = classify_command_safety(argv, VerificationSource.detected)
    assert safety == CommandSafety.blocked, (
        f"Expected blocked for {argv!r}, got {safety!r}: {reason}"
    )


# ---------------------------------------------------------------------------
# Medium-risk commands require approval
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "argv",
    [
        ["npm", "install"],
        ["npm", "ci"],
        ["npm", "run", "build"],
        ["pip", "install", "requests"],
        ["pip3", "install", "-r", "requirements.txt"],
        ["yarn", "install"],
        ["yarn", "add", "lodash"],
        ["git", "checkout", "main"],
        ["git", "switch", "feat/new"],
        ["git", "branch", "-d", "old-branch"],
        ["git", "merge", "develop"],
        ["git", "rebase", "main"],
        ["make"],
        ["make", "build"],
        ["terraform", "plan"],
        ["terraform", "init"],
        ["cargo", "build"],
        ["go", "build", "./..."],
    ],
)
def test_medium_risk_requires_approval(argv: list[str]) -> None:
    safety, _ = classify_command_safety(argv, VerificationSource.detected)
    assert safety == CommandSafety.needs_approval, (
        f"Expected needs_approval for {argv!r}, got {safety!r}"
    )


# ---------------------------------------------------------------------------
# git reset flag edge cases
# ---------------------------------------------------------------------------


def test_git_reset_soft_not_blocked() -> None:
    """git reset without a destructive flag is approval-required, not blocked."""
    safety, _ = classify_command_safety(
        ["git", "reset", "HEAD~1"], VerificationSource.detected
    )
    assert safety == CommandSafety.needs_approval


def test_git_reset_keep_is_blocked() -> None:
    safety, reason = classify_command_safety(
        ["git", "reset", "--keep", "HEAD"], VerificationSource.detected
    )
    assert safety == CommandSafety.blocked
    assert "--keep" in reason


# ---------------------------------------------------------------------------
# grep/rg with shell metacharacters is blocked
# ---------------------------------------------------------------------------


def test_grep_with_pipe_is_blocked() -> None:
    """grep safe prefix only applies when no shell metacharacters are present."""
    safety, reason = classify_command_safety(
        ["grep", "foo", ".", "|", "head"], VerificationSource.detected
    )
    assert safety == CommandSafety.blocked
    assert "metacharacter" in reason.lower() or "shell" in reason.lower()


# ---------------------------------------------------------------------------
# Timeout enforcement
# ---------------------------------------------------------------------------


def test_timeout_enforced_non_capture() -> None:
    with tempfile.TemporaryDirectory() as td:
        plan = _safe_plan()
        with patch(
            "subprocess.run",
            side_effect=subprocess.TimeoutExpired(cmd=["python"], timeout=5.0),
        ):
            result = run_verification_plan(plan, Path(td), timeout=5.0)
    assert result == 1


def test_timeout_enforced_capture() -> None:
    with tempfile.TemporaryDirectory() as td:
        plan = _safe_plan()
        with patch(
            "subprocess.run",
            side_effect=subprocess.TimeoutExpired(cmd=["python"], timeout=5.0),
        ):
            exit_code, output = run_verification_plan(  # type: ignore[misc]
                plan, Path(td), capture=True, timeout=5.0
            )
    assert exit_code == 1
    assert "timed out" in output


# ---------------------------------------------------------------------------
# Raw stdout not stored
# ---------------------------------------------------------------------------


def test_native_step_event_raw_content_always_false() -> None:
    """NativeStepEvent.__post_init__ enforces raw_content_stored=False."""
    event = NativeStepEvent(raw_content_stored=True)  # type: ignore[call-arg]
    assert event.raw_content_stored is False


def test_exec_run_verification_metadata_no_raw_output() -> None:
    """_exec_run_verification metadata carries output_chars but not raw stdout text."""
    from openshard.native.tools import _exec_run_verification

    fake_facts = MagicMock()
    fake_facts.test_command = "python -m pytest"

    with tempfile.TemporaryDirectory() as td:
        with (
            patch("openshard.analysis.repo.analyze_repo", return_value=fake_facts),
            patch(
                "openshard.verification.executor.subprocess.run",
                return_value=_proc(0, "5 passed"),
            ),
        ):
            result = _exec_run_verification(Path(td))

    meta = result.metadata
    assert meta.get("raw_content_stored") is False
    assert isinstance(meta.get("output_chars"), int)
    assert "5 passed" not in str(meta)


# ---------------------------------------------------------------------------
# Verification still runs
# ---------------------------------------------------------------------------


def test_verification_still_runs() -> None:
    with tempfile.TemporaryDirectory() as td:
        plan = _safe_plan()
        with patch("subprocess.run", return_value=_proc(0, "3 passed")) as mock_run:
            exit_code, output = run_verification_plan(  # type: ignore[misc]
                plan, Path(td), capture=True
            )
    assert exit_code == 0
    mock_run.assert_called_once()


# ---------------------------------------------------------------------------
# Blocked commands recorded in metadata
# ---------------------------------------------------------------------------


def test_blocked_commands_metadata() -> None:
    """_exec_run_verification for a blocked command records classification=blocked."""
    from openshard.native.tools import _exec_run_verification

    fake_facts = MagicMock()
    fake_facts.test_command = "rm -rf ."

    with tempfile.TemporaryDirectory() as td:
        with patch("openshard.analysis.repo.analyze_repo", return_value=fake_facts):
            result = _exec_run_verification(Path(td))

    assert result.ok is False
    meta = result.metadata
    assert meta.get("classification") == "blocked"
    assert meta.get("raw_content_stored") is False
    assert meta.get("exit_code") == 1


# ---------------------------------------------------------------------------
# Rendering: old run records
# ---------------------------------------------------------------------------


def test_old_run_records_render_safely() -> None:
    """An entry with no native fields renders without crashing and no command policy block."""
    out = _render({"task": "old task"}, detail="full")
    assert "command policy" not in out


# ---------------------------------------------------------------------------
# Rendering: default output clean
# ---------------------------------------------------------------------------


def test_default_output_clean() -> None:
    """default detail does not show [native] block or command policy."""
    entry = _native_entry_with_policy(safe=1)
    out = _render(entry, detail="default")
    assert "command policy" not in out


# ---------------------------------------------------------------------------
# Rendering: last --more compact policy
# ---------------------------------------------------------------------------


def test_last_more_compact_policy() -> None:
    """--full renders compact command policy summary line."""
    entry = _native_entry_with_policy(safe=2, approval=1, blocked=0)
    out = _render(entry, detail="full")
    assert "command policy: 2 safe, 1 approval, 0 blocked" in out


# ---------------------------------------------------------------------------
# Rendering: last --full detailed policy
# ---------------------------------------------------------------------------


def test_last_full_detailed_policy() -> None:
    """--full renders per-command detail lines after the compact summary."""
    vplan = [
        {
            "name": "tests",
            "argv": ["python", "-m", "pytest", "tests/"],
            "kind": "test",
            "source": "detected",
            "safety": "safe",
            "reason": "matches safe prefix: python -m pytest",
        }
    ]
    entry = _native_entry_with_policy(safe=1, vplan_cmds=vplan)
    out = _render(entry, detail="full")
    assert "command policy: 1 safe" in out
    assert "[safe]" in out
    assert "matches safe prefix" in out


def test_last_full_blocked_detail() -> None:
    """--full shows [blocked] classification for blocked commands."""
    vplan = [
        {
            "name": "verification",
            "argv": ["git", "push"],
            "kind": "unknown",
            "source": "detected",
            "safety": "blocked",
            "reason": "blocked command: git push",
        }
    ]
    entry = _native_entry_with_policy(safe=0, blocked=1, vplan_cmds=vplan)
    out = _render(entry, detail="full")
    assert "[blocked]" in out
    assert "blocked command" in out


# ---------------------------------------------------------------------------
# NativeCommandPolicyPreview command_records
# ---------------------------------------------------------------------------


def test_command_policy_preview_records_populated() -> None:
    plan = _safe_plan(["python", "-m", "pytest", "tests/"])
    preview = build_native_command_policy_preview(plan)
    assert preview.safe_count == 1
    assert len(preview.command_records) == 1
    rec = preview.command_records[0]
    assert rec["classification"] == "safe"
    assert rec["command"] == "python -m pytest tests/"
    assert rec["raw_content_stored"] is False


def test_command_policy_preview_blocked_records() -> None:
    plan = _blocked_plan()
    preview = build_native_command_policy_preview(plan)
    assert preview.blocked_count == 1
    assert preview.command_records[0]["classification"] == "blocked"
