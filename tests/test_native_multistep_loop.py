"""Native multi-step edit/verify loop v0 — focused tests.

Covers:
  - NativeEditLoopAttempt / NativeEditLoopSummary dataclass invariants
  - record_native_edit_loop_attempt helper behaviour
  - render_native_edit_loop_summary output format (compact + full)
  - Pipeline integration: attempt recording, repair path, skipped path
  - Existing retry_metadata still present (non-regression)
"""
from __future__ import annotations

import tempfile
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import click
from click.testing import CliRunner

from openshard.analysis.repo import RepoFacts
from openshard.cli.main import cli
from openshard.native.context import (
    NativeApprovalRequest,
    NativeChangeBudgetSoftGate,
    NativeEditLoopAttempt,
    NativeEditLoopSummary,
    NativeSandboxMeta,
    record_native_edit_loop_attempt,
    render_native_edit_loop_summary,
)

# ---------------------------------------------------------------------------
# Shared helpers (mirror test_iterative_retry_loop.py patterns exactly)
# ---------------------------------------------------------------------------

_DEFAULT_CONFIG = {"approval_mode": "smart"}

_PYTHON_REPO_WITH_TEST_CMD = RepoFacts(
    languages=["python"],
    package_files=[],
    framework=None,
    test_command="python -m pytest",
    risky_paths=[],
    changed_files=[],
)

_PYTHON_REPO_NO_TEST_CMD = RepoFacts(
    languages=["python"],
    package_files=[],
    framework=None,
    test_command=None,
    risky_paths=[],
    changed_files=[],
)


def _render(entry: dict, detail: str = "more") -> str:
    from openshard.cli.main import _render_log_entry

    @click.command()
    def cmd():
        _render_log_entry(entry, detail)

    return CliRunner().invoke(cmd).output


def _edit_loop_entry(summary_dict: dict) -> dict:
    """Build a minimal native run history entry carrying an edit_loop_summary."""
    return {
        "task": "test task",
        "workflow": "native",
        "edit_loop_summary": summary_dict,
    }


def _make_native_mock(generate_side_effect=None):
    """Return a generator mock wired to look like NativeAgentExecutor."""
    from openshard.native.executor import NativeRunMeta

    g = MagicMock()
    if generate_side_effect is not None:
        g.generate.side_effect = generate_side_effect
    else:
        g.generate.return_value = MagicMock(
            usage=None,
            files=[],
            summary="done",
            notes=[],
        )
    g.model = "mock-model"
    g.fixer_model = "mock-fixer"
    meta = NativeRunMeta()
    g.native_meta = meta
    g.build_change_budget_soft_gate.return_value = NativeChangeBudgetSoftGate(
        requires_approval=False, reason="", action="allow"
    )
    g.build_budget_gate_approval_request.return_value = NativeApprovalRequest(
        source="change_budget_soft_gate",
        requires_approval=False,
        reason="",
        action="allow",
    )
    return g


def _make_manager_mock():
    m = MagicMock()
    inv = MagicMock()
    inv.models = []
    m.get_inventory.return_value = inv
    m.providers = {"openrouter": MagicMock()}
    return m


def _invoke_native_write(
    native_mock=None,
    *,
    verify_side_effects: list | None = None,
    repo: RepoFacts | None = None,
):
    """Invoke openshard run --workflow native --write with mocked infra."""
    if native_mock is None:
        native_mock = _make_native_mock()
    if repo is None:
        repo = _PYTHON_REPO_WITH_TEST_CMD

    _verif_calls = iter(verify_side_effects or [(0, "")])

    def _fake_verify(*args, **kwargs):
        capture = kwargs.get("capture", False)
        try:
            code, out = next(_verif_calls)
        except StopIteration:
            code, out = 0, ""
        return (code, out) if capture else code

    with tempfile.TemporaryDirectory() as _td:
        sandbox_meta = NativeSandboxMeta(sandbox_enabled=False, sandbox_type="none")
        with (
            patch("openshard.run.pipeline.NativeAgentExecutor", return_value=native_mock),
            patch("openshard.run.pipeline.ProviderManager", return_value=_make_manager_mock()),
            patch("openshard.cli.main.load_config", return_value=_DEFAULT_CONFIG),
            patch("openshard.run.pipeline.analyze_repo", return_value=repo),
            patch("openshard.run.pipeline._run_verification_plan", side_effect=_fake_verify),
            patch("openshard.run.pipeline._write_files"),
            patch(
                "openshard.native.sandbox.create_run_sandbox",
                return_value=(Path(_td), sandbox_meta),
            ),
            patch("openshard.run.pipeline._log_run"),
        ):
            runner = CliRunner()
            result = runner.invoke(
                cli,
                ["run", "--workflow", "native", "--write", "fix the bug"],
            )
    return result, native_mock


# ---------------------------------------------------------------------------
# NativeEditLoopAttempt — dataclass defaults and invariants
# ---------------------------------------------------------------------------


def test_attempt_default_attempt_index_zero() -> None:
    assert NativeEditLoopAttempt().attempt_index == 0


def test_attempt_default_purpose_empty() -> None:
    assert NativeEditLoopAttempt().purpose == ""


def test_attempt_default_verification_status_empty() -> None:
    assert NativeEditLoopAttempt().verification_status == ""


def test_attempt_default_files_written_empty_list() -> None:
    assert NativeEditLoopAttempt().files_written == []


def test_attempt_default_exit_code_none() -> None:
    assert NativeEditLoopAttempt().exit_code is None


def test_attempt_default_output_chars_zero() -> None:
    assert NativeEditLoopAttempt().output_chars == 0


def test_attempt_raw_content_stored_always_false() -> None:
    assert NativeEditLoopAttempt().raw_content_stored is False


def test_attempt_raw_content_stored_reset_by_post_init() -> None:
    a = NativeEditLoopAttempt(raw_content_stored=True)  # type: ignore[call-arg]
    assert a.raw_content_stored is False


def test_attempt_files_written_lists_are_independent() -> None:
    a1 = NativeEditLoopAttempt()
    a2 = NativeEditLoopAttempt()
    a1.files_written.append("foo.py")
    assert a2.files_written == []


# ---------------------------------------------------------------------------
# NativeEditLoopSummary — dataclass defaults and invariants
# ---------------------------------------------------------------------------


def test_summary_enabled_by_default() -> None:
    assert NativeEditLoopSummary().enabled is True


def test_summary_max_attempts_is_two() -> None:
    assert NativeEditLoopSummary().max_attempts == 2


def test_summary_attempts_empty_by_default() -> None:
    assert NativeEditLoopSummary().attempts == []


def test_summary_completed_false_by_default() -> None:
    assert NativeEditLoopSummary().completed is False


def test_summary_final_status_empty_by_default() -> None:
    assert NativeEditLoopSummary().final_status == ""


def test_summary_repair_used_false_by_default() -> None:
    assert NativeEditLoopSummary().repair_used is False


def test_summary_raw_content_stored_always_false() -> None:
    assert NativeEditLoopSummary().raw_content_stored is False


def test_summary_raw_content_stored_reset_by_post_init() -> None:
    s = NativeEditLoopSummary(raw_content_stored=True)  # type: ignore[call-arg]
    assert s.raw_content_stored is False


def test_summary_attempts_lists_are_independent() -> None:
    s1 = NativeEditLoopSummary()
    s2 = NativeEditLoopSummary()
    s1.attempts.append(NativeEditLoopAttempt())
    assert s2.attempts == []


def test_summary_serializes_cleanly() -> None:
    s = NativeEditLoopSummary()
    d = asdict(s)
    assert d["raw_content_stored"] is False
    assert d["attempts"] == []
    assert d["enabled"] is True


# ---------------------------------------------------------------------------
# record_native_edit_loop_attempt — helper behaviour
# ---------------------------------------------------------------------------


def test_record_appends_one_attempt() -> None:
    s = NativeEditLoopSummary()
    record_native_edit_loop_attempt(
        s,
        attempt_index=1,
        purpose="initial",
        files_written=[],
        verification_status="passed",
        exit_code=0,
        output_chars=0,
    )
    assert len(s.attempts) == 1


def test_record_two_calls_append_two_attempts() -> None:
    s = NativeEditLoopSummary()
    for idx, purpose in [(1, "initial"), (2, "repair")]:
        record_native_edit_loop_attempt(
            s,
            attempt_index=idx,
            purpose=purpose,
            files_written=[],
            verification_status="passed",
            exit_code=0,
            output_chars=0,
        )
    assert len(s.attempts) == 2


def test_record_repair_purpose_sets_repair_used() -> None:
    s = NativeEditLoopSummary()
    record_native_edit_loop_attempt(
        s,
        attempt_index=2,
        purpose="repair",
        files_written=[],
        verification_status="passed",
        exit_code=0,
        output_chars=0,
    )
    assert s.repair_used is True


def test_record_initial_purpose_leaves_repair_used_false() -> None:
    s = NativeEditLoopSummary()
    record_native_edit_loop_attempt(
        s,
        attempt_index=1,
        purpose="initial",
        files_written=[],
        verification_status="failed",
        exit_code=1,
        output_chars=0,
    )
    assert s.repair_used is False


def test_record_passed_at_index1_sets_completed_true() -> None:
    s = NativeEditLoopSummary()
    record_native_edit_loop_attempt(
        s,
        attempt_index=1,
        purpose="initial",
        files_written=[],
        verification_status="passed",
        exit_code=0,
        output_chars=0,
    )
    assert s.completed is True
    assert s.final_status == "passed"


def test_record_failed_at_index1_leaves_completed_false() -> None:
    s = NativeEditLoopSummary(max_attempts=2)
    record_native_edit_loop_attempt(
        s,
        attempt_index=1,
        purpose="initial",
        files_written=[],
        verification_status="failed",
        exit_code=1,
        output_chars=0,
    )
    assert s.completed is False
    assert s.final_status == "failed"


def test_record_failed_at_index2_sets_completed_true() -> None:
    s = NativeEditLoopSummary(max_attempts=2)
    record_native_edit_loop_attempt(
        s,
        attempt_index=2,
        purpose="repair",
        files_written=[],
        verification_status="failed",
        exit_code=1,
        output_chars=0,
    )
    assert s.completed is True


def test_record_skipped_sets_completed_true() -> None:
    s = NativeEditLoopSummary()
    record_native_edit_loop_attempt(
        s,
        attempt_index=1,
        purpose="initial",
        files_written=[],
        verification_status="skipped",
        exit_code=None,
        output_chars=0,
    )
    assert s.completed is True
    assert s.final_status == "skipped"


def test_record_attempt_raw_content_never_stored() -> None:
    s = NativeEditLoopSummary()
    record_native_edit_loop_attempt(
        s,
        attempt_index=1,
        purpose="initial",
        files_written=[],
        verification_status="passed",
        exit_code=0,
        output_chars=0,
    )
    assert s.attempts[0].raw_content_stored is False


def test_record_files_written_stored_as_copy() -> None:
    s = NativeEditLoopSummary()
    src = ["a.py", "b.py"]
    record_native_edit_loop_attempt(
        s,
        attempt_index=1,
        purpose="initial",
        files_written=src,
        verification_status="passed",
        exit_code=0,
        output_chars=0,
    )
    src.append("mutated.py")
    assert "mutated.py" not in s.attempts[0].files_written


# ---------------------------------------------------------------------------
# render_native_edit_loop_summary — output format
# ---------------------------------------------------------------------------


def test_render_none_returns_empty_string() -> None:
    assert render_native_edit_loop_summary(None) == ""


def test_render_disabled_summary_returns_empty_string() -> None:
    s = NativeEditLoopSummary(enabled=False)
    assert render_native_edit_loop_summary(s) == ""


def test_render_compact_is_single_line() -> None:
    s = NativeEditLoopSummary()
    record_native_edit_loop_attempt(
        s, attempt_index=1, purpose="initial", files_written=[],
        verification_status="passed", exit_code=0, output_chars=0,
    )
    out = render_native_edit_loop_summary(s, detail="compact")
    assert "\n" not in out


def test_render_compact_shows_final_status() -> None:
    s = NativeEditLoopSummary()
    record_native_edit_loop_attempt(
        s, attempt_index=1, purpose="initial", files_written=[],
        verification_status="passed", exit_code=0, output_chars=0,
    )
    out = render_native_edit_loop_summary(s, detail="compact")
    assert "passed" in out


def test_render_compact_shows_attempt_count() -> None:
    s = NativeEditLoopSummary()
    record_native_edit_loop_attempt(
        s, attempt_index=1, purpose="initial", files_written=[],
        verification_status="passed", exit_code=0, output_chars=0,
    )
    out = render_native_edit_loop_summary(s, detail="compact")
    assert "1/2" in out


def test_render_full_two_attempts_is_multiline() -> None:
    s = NativeEditLoopSummary()
    record_native_edit_loop_attempt(
        s, attempt_index=1, purpose="initial", files_written=["a.py"],
        verification_status="failed", exit_code=1, output_chars=842,
    )
    record_native_edit_loop_attempt(
        s, attempt_index=2, purpose="repair", files_written=["a.py"],
        verification_status="passed", exit_code=0, output_chars=120,
    )
    out = render_native_edit_loop_summary(s, detail="full")
    assert "\n" in out


def test_render_full_shows_purpose_initial() -> None:
    s = NativeEditLoopSummary()
    record_native_edit_loop_attempt(
        s, attempt_index=1, purpose="initial", files_written=[],
        verification_status="passed", exit_code=0, output_chars=0,
    )
    out = render_native_edit_loop_summary(s, detail="full")
    assert "initial" in out


def test_render_full_shows_exit_code() -> None:
    s = NativeEditLoopSummary()
    record_native_edit_loop_attempt(
        s, attempt_index=1, purpose="initial", files_written=[],
        verification_status="passed", exit_code=0, output_chars=0,
    )
    out = render_native_edit_loop_summary(s, detail="full")
    assert "exit=0" in out


def test_render_full_shows_chars() -> None:
    s = NativeEditLoopSummary()
    record_native_edit_loop_attempt(
        s, attempt_index=1, purpose="initial", files_written=[],
        verification_status="passed", exit_code=0, output_chars=120,
    )
    out = render_native_edit_loop_summary(s, detail="full")
    assert "chars=120" in out


def test_render_full_shows_files_count() -> None:
    s = NativeEditLoopSummary()
    record_native_edit_loop_attempt(
        s, attempt_index=1, purpose="initial", files_written=["a.py"],
        verification_status="passed", exit_code=0, output_chars=0,
    )
    out = render_native_edit_loop_summary(s, detail="full")
    assert "files=1" in out


def test_render_full_uses_attempt_index_directly() -> None:
    s = NativeEditLoopSummary()
    record_native_edit_loop_attempt(
        s, attempt_index=1, purpose="initial", files_written=["a.py"],
        verification_status="failed", exit_code=1, output_chars=842,
    )
    record_native_edit_loop_attempt(
        s, attempt_index=2, purpose="repair", files_written=["a.py"],
        verification_status="passed", exit_code=0, output_chars=120,
    )
    out = render_native_edit_loop_summary(s, detail="full")
    assert "  1. initial:" in out
    assert "  2. repair:" in out


def test_render_accepts_simple_namespace() -> None:
    s = NativeEditLoopSummary()
    record_native_edit_loop_attempt(
        s, attempt_index=1, purpose="initial", files_written=[],
        verification_status="passed", exit_code=0, output_chars=0,
    )
    ns = SimpleNamespace(**{
        "enabled": True,
        "max_attempts": 2,
        "final_status": "passed",
        "repair_used": False,
        "completed": True,
        "raw_content_stored": False,
        "attempts": [
            SimpleNamespace(**{
                "attempt_index": 1,
                "purpose": "initial",
                "files_written": [],
                "verification_status": "passed",
                "exit_code": 0,
                "output_chars": 0,
                "raw_content_stored": False,
            })
        ],
    })
    out = render_native_edit_loop_summary(ns, detail="compact")
    assert "passed" in out
    assert out != ""


def test_render_accepts_dict_via_simple_namespace() -> None:
    s = NativeEditLoopSummary()
    record_native_edit_loop_attempt(
        s, attempt_index=1, purpose="initial", files_written=["x.py"],
        verification_status="passed", exit_code=0, output_chars=50,
    )

    def _dict_to_ns(obj):
        if isinstance(obj, dict):
            return SimpleNamespace(**{k: _dict_to_ns(v) for k, v in obj.items()})
        if isinstance(obj, list):
            return [_dict_to_ns(i) for i in obj]
        return obj

    ns = _dict_to_ns(asdict(s))
    out = render_native_edit_loop_summary(ns, detail="full")
    assert "passed" in out
    assert "initial" in out


# ---------------------------------------------------------------------------
# Rendering integration: run_output.py shows edit loop at correct detail levels
# ---------------------------------------------------------------------------


def _make_edit_loop_summary_dict(
    *,
    final_status: str = "passed",
    attempts: list[dict] | None = None,
) -> dict:
    if attempts is None:
        attempts = [
            {
                "attempt_index": 1,
                "purpose": "initial",
                "files_written": ["src/foo.py"],
                "verification_status": final_status,
                "exit_code": 0 if final_status == "passed" else 1,
                "output_chars": 100,
                "raw_content_stored": False,
            }
        ]
    return {
        "enabled": True,
        "max_attempts": 2,
        "attempts": attempts,
        "completed": True,
        "final_status": final_status,
        "repair_used": False,
        "raw_content_stored": False,
    }


def test_default_output_does_not_show_edit_loop() -> None:
    entry = _edit_loop_entry(_make_edit_loop_summary_dict())
    out = _render(entry, detail="default")
    assert "edit loop" not in out


def test_more_shows_compact_edit_loop() -> None:
    entry = _edit_loop_entry(_make_edit_loop_summary_dict(final_status="passed"))
    out = _render(entry, detail="full")
    assert "edit loop" in out
    assert "passed" in out


def test_full_shows_attempt_detail() -> None:
    attempts = [
        {
            "attempt_index": 1,
            "purpose": "initial",
            "files_written": ["src/foo.py", "src/bar.py"],
            "verification_status": "failed",
            "exit_code": 1,
            "output_chars": 842,
            "raw_content_stored": False,
        },
        {
            "attempt_index": 2,
            "purpose": "repair",
            "files_written": ["src/foo.py"],
            "verification_status": "passed",
            "exit_code": 0,
            "output_chars": 120,
            "raw_content_stored": False,
        },
    ]
    entry = _edit_loop_entry(_make_edit_loop_summary_dict(
        final_status="passed", attempts=attempts
    ))
    out = _render(entry, detail="full")
    assert "edit loop" in out
    assert "initial" in out
    assert "repair" in out
    assert "chars=120" in out


# ---------------------------------------------------------------------------
# Pipeline integration — edit_loop_summary populated by pipeline
# ---------------------------------------------------------------------------


def test_pipeline_creates_edit_loop_summary() -> None:
    _, g = _invoke_native_write(verify_side_effects=[(0, "ok")])
    assert g.native_meta.edit_loop_summary is not None


def test_pipeline_edit_loop_enabled_true() -> None:
    _, g = _invoke_native_write(verify_side_effects=[(0, "ok")])
    assert g.native_meta.edit_loop_summary.enabled is True


def test_pipeline_edit_loop_max_attempts_two() -> None:
    _, g = _invoke_native_write(verify_side_effects=[(0, "ok")])
    assert g.native_meta.edit_loop_summary.max_attempts == 2


def test_pipeline_initial_pass_one_attempt() -> None:
    _, g = _invoke_native_write(verify_side_effects=[(0, "ok")])
    assert len(g.native_meta.edit_loop_summary.attempts) == 1


def test_pipeline_initial_pass_attempt_index_is_1() -> None:
    _, g = _invoke_native_write(verify_side_effects=[(0, "ok")])
    assert g.native_meta.edit_loop_summary.attempts[0].attempt_index == 1


def test_pipeline_initial_pass_purpose_is_initial() -> None:
    _, g = _invoke_native_write(verify_side_effects=[(0, "ok")])
    assert g.native_meta.edit_loop_summary.attempts[0].purpose == "initial"


def test_pipeline_initial_pass_verification_status_passed() -> None:
    _, g = _invoke_native_write(verify_side_effects=[(0, "ok")])
    assert g.native_meta.edit_loop_summary.attempts[0].verification_status == "passed"


def test_pipeline_initial_pass_final_status_passed() -> None:
    _, g = _invoke_native_write(verify_side_effects=[(0, "ok")])
    assert g.native_meta.edit_loop_summary.final_status == "passed"


def test_pipeline_initial_pass_repair_used_false() -> None:
    _, g = _invoke_native_write(verify_side_effects=[(0, "ok")])
    assert g.native_meta.edit_loop_summary.repair_used is False


def test_pipeline_initial_pass_completed_true() -> None:
    _, g = _invoke_native_write(verify_side_effects=[(0, "ok")])
    assert g.native_meta.edit_loop_summary.completed is True


def test_pipeline_initial_pass_generate_called_once() -> None:
    mock = _make_native_mock()
    _invoke_native_write(mock, verify_side_effects=[(0, "ok")])
    assert mock.generate.call_count == 1


def test_pipeline_fail_then_pass_two_attempts() -> None:
    _, g = _invoke_native_write(
        verify_side_effects=[(1, "FAILED: 1 error"), (0, "ok")]
    )
    assert len(g.native_meta.edit_loop_summary.attempts) == 2


def test_pipeline_fail_then_pass_attempt1_purpose_initial() -> None:
    _, g = _invoke_native_write(
        verify_side_effects=[(1, "FAILED: 1 error"), (0, "ok")]
    )
    assert g.native_meta.edit_loop_summary.attempts[0].purpose == "initial"


def test_pipeline_fail_then_pass_attempt1_index_is_1() -> None:
    _, g = _invoke_native_write(
        verify_side_effects=[(1, "FAILED: 1 error"), (0, "ok")]
    )
    assert g.native_meta.edit_loop_summary.attempts[0].attempt_index == 1


def test_pipeline_fail_then_pass_attempt1_status_failed() -> None:
    _, g = _invoke_native_write(
        verify_side_effects=[(1, "FAILED: 1 error"), (0, "ok")]
    )
    assert g.native_meta.edit_loop_summary.attempts[0].verification_status == "failed"


def test_pipeline_fail_then_pass_attempt2_purpose_repair() -> None:
    _, g = _invoke_native_write(
        verify_side_effects=[(1, "FAILED: 1 error"), (0, "ok")]
    )
    assert g.native_meta.edit_loop_summary.attempts[1].purpose == "repair"


def test_pipeline_fail_then_pass_attempt2_index_is_2() -> None:
    _, g = _invoke_native_write(
        verify_side_effects=[(1, "FAILED: 1 error"), (0, "ok")]
    )
    assert g.native_meta.edit_loop_summary.attempts[1].attempt_index == 2


def test_pipeline_fail_then_pass_attempt2_status_passed() -> None:
    _, g = _invoke_native_write(
        verify_side_effects=[(1, "FAILED: 1 error"), (0, "ok")]
    )
    assert g.native_meta.edit_loop_summary.attempts[1].verification_status == "passed"


def test_pipeline_fail_then_pass_repair_used_true() -> None:
    _, g = _invoke_native_write(
        verify_side_effects=[(1, "FAILED: 1 error"), (0, "ok")]
    )
    assert g.native_meta.edit_loop_summary.repair_used is True


def test_pipeline_fail_then_pass_final_status_passed() -> None:
    _, g = _invoke_native_write(
        verify_side_effects=[(1, "FAILED: 1 error"), (0, "ok")]
    )
    assert g.native_meta.edit_loop_summary.final_status == "passed"


def test_pipeline_fail_then_pass_completed_true() -> None:
    _, g = _invoke_native_write(
        verify_side_effects=[(1, "FAILED: 1 error"), (0, "ok")]
    )
    assert g.native_meta.edit_loop_summary.completed is True


def test_pipeline_fail_then_pass_generate_called_twice() -> None:
    mock = _make_native_mock()
    _invoke_native_write(mock, verify_side_effects=[(1, "FAILED"), (0, "ok")])
    assert mock.generate.call_count == 2


def test_pipeline_both_fail_two_attempts() -> None:
    _, g = _invoke_native_write(
        verify_side_effects=[(1, "fail1"), (1, "fail2")]
    )
    assert len(g.native_meta.edit_loop_summary.attempts) == 2


def test_pipeline_both_fail_final_status_failed() -> None:
    _, g = _invoke_native_write(
        verify_side_effects=[(1, "fail1"), (1, "fail2")]
    )
    assert g.native_meta.edit_loop_summary.final_status == "failed"


def test_pipeline_both_fail_completed_true() -> None:
    _, g = _invoke_native_write(
        verify_side_effects=[(1, "fail1"), (1, "fail2")]
    )
    assert g.native_meta.edit_loop_summary.completed is True


def test_pipeline_both_fail_repair_used_true() -> None:
    _, g = _invoke_native_write(
        verify_side_effects=[(1, "fail1"), (1, "fail2")]
    )
    assert g.native_meta.edit_loop_summary.repair_used is True


def test_pipeline_both_fail_generate_called_twice_only() -> None:
    mock = _make_native_mock()
    _invoke_native_write(mock, verify_side_effects=[(1, "fail1"), (1, "fail2")])
    assert mock.generate.call_count == 2


def test_pipeline_skipped_one_attempt() -> None:
    _, g = _invoke_native_write(repo=_PYTHON_REPO_NO_TEST_CMD)
    summary = g.native_meta.edit_loop_summary
    assert summary is not None
    assert len(summary.attempts) == 1


def test_pipeline_skipped_verification_status_skipped() -> None:
    _, g = _invoke_native_write(repo=_PYTHON_REPO_NO_TEST_CMD)
    assert g.native_meta.edit_loop_summary.attempts[0].verification_status == "skipped"


def test_pipeline_skipped_attempt_index_is_1() -> None:
    _, g = _invoke_native_write(repo=_PYTHON_REPO_NO_TEST_CMD)
    assert g.native_meta.edit_loop_summary.attempts[0].attempt_index == 1


def test_pipeline_skipped_completed_true() -> None:
    _, g = _invoke_native_write(repo=_PYTHON_REPO_NO_TEST_CMD)
    assert g.native_meta.edit_loop_summary.completed is True


def test_pipeline_raw_content_never_stored_initial_pass() -> None:
    _, g = _invoke_native_write(verify_side_effects=[(0, "ok")])
    for att in g.native_meta.edit_loop_summary.attempts:
        assert att.raw_content_stored is False


def test_pipeline_raw_content_never_stored_after_repair() -> None:
    _, g = _invoke_native_write(
        verify_side_effects=[(1, "FAILED"), (0, "ok")]
    )
    for att in g.native_meta.edit_loop_summary.attempts:
        assert att.raw_content_stored is False


def test_pipeline_retry_metadata_still_present_after_repair() -> None:
    """Non-regression: existing retry_metadata must survive the edit loop addition."""
    _, g = _invoke_native_write(
        verify_side_effects=[(1, "FAILED: assertion error"), (0, "ok")]
    )
    vloop = g.native_meta.verification_loop
    assert vloop is not None
    assert vloop.retry_metadata is not None
    assert vloop.retry_metadata.retry_attempted is True
    assert vloop.retry_metadata.retry_verification_status == "passed"
