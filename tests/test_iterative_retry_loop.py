"""Iterative edit/retry loop v0 — focused tests.

Covers:
  - RetryMetadata dataclass invariants
  - build_failure_summary classification (no raw content)
  - Pipeline: no retry on first-pass verification
  - Pipeline: retry triggered on first-fail
  - Pipeline: retry_once passed/failed recorded correctly
  - Pipeline: retry capped at one attempt
  - record_osn_loop_step metadata routing (NativeStepEvent vs OSN recorder)
  - Rendering: old records, default output, --more, --full
"""
from __future__ import annotations

import tempfile
from dataclasses import asdict
from pathlib import Path
from unittest.mock import MagicMock, patch

import click
from click.testing import CliRunner

from openshard.analysis.repo import RepoFacts
from openshard.cli.main import cli
from openshard.native.context import (
    NativeSandboxMeta,
    NativeVerificationLoop,
    RetryMetadata,
    build_failure_summary,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _render(entry: dict, detail: str = "more") -> str:
    from openshard.cli.main import _render_log_entry

    @click.command()
    def cmd():
        _render_log_entry(entry, detail)

    return CliRunner().invoke(cmd).output


def _vloop_entry(
    *,
    attempted: bool = True,
    passed: bool = True,
    retried: bool = False,
    retry_meta: dict | None = None,
) -> dict:
    vloop: dict = {
        "attempted": attempted,
        "passed": passed,
        "retried": retried,
        "exit_code": 0 if passed else 1,
        "output_chars": 0,
        "truncated": False,
    }
    if retry_meta is not None:
        vloop["retry_metadata"] = retry_meta
    return {"task": "test task", "workflow": "native", "verification_loop": vloop}


def _retry_meta_dict(
    *,
    status: str = "passed",
    failure_summary: str = "exit_code=1 failure_type=test_failure output_chars=50 raw_content_stored=false",
    patch_files: list[str] | None = None,
) -> dict:
    return {
        "retry_attempted": True,
        "retry_reason": "verification_failed",
        "failure_summary": failure_summary,
        "retry_patch_files": patch_files or ["src/foo.py"],
        "retry_verification_status": status,
        "raw_content_stored": False,
    }


# ---------------------------------------------------------------------------
# RetryMetadata — invariants
# ---------------------------------------------------------------------------


def test_retry_metadata_defaults() -> None:
    m = RetryMetadata()
    assert m.retry_attempted is False
    assert m.retry_reason == ""
    assert m.failure_summary == ""
    assert m.retry_patch_files == []
    assert m.retry_verification_status == ""
    assert m.raw_content_stored is False


def test_retry_metadata_raw_content_always_false() -> None:
    """__post_init__ enforces raw_content_stored=False regardless of init value."""
    m = RetryMetadata(raw_content_stored=True)  # type: ignore[call-arg]
    assert m.raw_content_stored is False


def test_retry_metadata_in_verification_loop() -> None:
    vloop = NativeVerificationLoop()
    assert vloop.retry_metadata is None
    vloop.retry_metadata = RetryMetadata(retry_attempted=True)
    assert vloop.retry_metadata.retry_attempted is True


def test_retry_metadata_serializes_cleanly() -> None:
    vloop = NativeVerificationLoop(
        attempted=True, retried=True,
        retry_metadata=RetryMetadata(
            retry_attempted=True,
            retry_reason="verification_failed",
            failure_summary="exit_code=1 failure_type=test_failure output_chars=30 raw_content_stored=false",
            retry_patch_files=["src/foo.py"],
            retry_verification_status="passed",
        ),
    )
    d = asdict(vloop)
    assert d["retry_metadata"]["retry_attempted"] is True
    assert d["retry_metadata"]["raw_content_stored"] is False
    assert "failure_type" in d["retry_metadata"]["failure_summary"]


# ---------------------------------------------------------------------------
# build_failure_summary — classification, no raw content
# ---------------------------------------------------------------------------


def test_build_failure_summary_syntax_error() -> None:
    out = build_failure_summary("SyntaxError: invalid syntax", exit_code=1)
    assert "syntax_error" in out
    assert "SyntaxError" not in out
    assert "invalid syntax" not in out


def test_build_failure_summary_import_error() -> None:
    out = build_failure_summary("ModuleNotFoundError: No module named 'foo'", exit_code=1)
    assert "import_error" in out
    assert "No module named" not in out


def test_build_failure_summary_assertion_error() -> None:
    out = build_failure_summary("AssertionError: expected True", exit_code=1)
    assert "assertion_error" in out
    assert "expected True" not in out


def test_build_failure_summary_type_error() -> None:
    out = build_failure_summary("TypeError: unsupported operand", exit_code=1)
    assert "type_error" in out


def test_build_failure_summary_generic_failure() -> None:
    out = build_failure_summary("FAILED test_foo.py::test_bar - assert 1 == 2", exit_code=1)
    assert "test_failure" in out
    assert "assert 1 == 2" not in out


def test_build_failure_summary_exit_code_present() -> None:
    out = build_failure_summary("some output", exit_code=2)
    assert "exit_code=2" in out


def test_build_failure_summary_output_chars_correct() -> None:
    raw = "x" * 100
    out = build_failure_summary(raw, exit_code=1)
    assert "output_chars=100" in out


def test_build_failure_summary_raw_content_stored_false() -> None:
    raw = "secret stdout content"
    out = build_failure_summary(raw, exit_code=1)
    assert "raw_content_stored=false" in out
    assert "secret stdout content" not in out


def test_retry_uses_summarized_failure_metadata_not_raw_output() -> None:
    """RetryMetadata.failure_summary contains no verbatim raw output."""
    raw = "FAILED: assert result == expected\nActualValue=42 ExpectedValue=99"
    summary = build_failure_summary(raw, exit_code=1)
    meta = RetryMetadata(
        retry_attempted=True,
        failure_summary=summary,
    )
    assert "ActualValue=42" not in meta.failure_summary
    assert "ExpectedValue=99" not in meta.failure_summary
    assert "raw_content_stored=false" in meta.failure_summary


# ---------------------------------------------------------------------------
# record_osn_loop_step — metadata routed only to NativeStepEvent
# ---------------------------------------------------------------------------


def test_record_osn_loop_step_metadata_not_passed_to_osn_recorder() -> None:
    """metadata kwarg must not reach OsnLoopRecorder.record_step (no such field)."""
    from openshard.native.executor import NativeAgentExecutor

    with patch("openshard.native.executor.ExecutionGenerator") as _gen_cls:
        _gen_cls.return_value = MagicMock(model="m", fixer_model="f")
        executor = NativeAgentExecutor(provider=MagicMock(), native_loop="experimental")

    executor._run_id = "test-run"
    with patch("openshard.history.native_steps.log_native_step_event"):
        # Should not raise TypeError even with metadata kwarg
        executor.record_osn_loop_step(
            "retry_diagnosis", "passed",
            result_summary="diagnosis",
            metadata={"failure_summary": "exit_code=1", "raw_content_stored": False},
        )

    step = next(
        s for s in executor._osn_recorder.summary.steps if s.step_name == "retry_diagnosis"
    )
    assert not hasattr(step, "metadata")


def test_record_osn_loop_step_metadata_reaches_step_event() -> None:
    """metadata kwarg is forwarded to NativeStepEvent."""
    from openshard.native.executor import NativeAgentExecutor
    from openshard.history.native_steps import NativeStepEvent

    captured: list[NativeStepEvent] = []

    with patch("openshard.native.executor.ExecutionGenerator") as _gen_cls:
        _gen_cls.return_value = MagicMock(model="m", fixer_model="f")
        executor = NativeAgentExecutor(provider=MagicMock(), native_loop="experimental")

    executor._run_id = "test-run"

    def _capture(event: NativeStepEvent) -> None:
        captured.append(event)

    with patch("openshard.history.native_steps.log_native_step_event", side_effect=_capture):
        executor.record_osn_loop_step(
            "retry_patch", "passed",
            result_summary="2 files patched",
            metadata={"files": ["a.py", "b.py"], "file_count": 2, "raw_content_stored": False},
        )

    assert captured
    assert captured[0].metadata["file_count"] == 2
    assert captured[0].metadata["raw_content_stored"] is False


# ---------------------------------------------------------------------------
# Pipeline: native verification loop — retry behaviour
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


def _make_native_mock(generate_side_effect=None):
    """Return a generator mock wired to look like NativeAgentExecutor."""
    from openshard.native.executor import NativeRunMeta
    from openshard.native.context import NativeChangeBudgetSoftGate, NativeApprovalRequest

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
):
    """Invoke openshard run --workflow native --write with mocked infra.

    verify_side_effects: list of (exit_code, output) tuples returned by each
    successive call to _run_verification_plan.
    """
    if native_mock is None:
        native_mock = _make_native_mock()

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
            patch("openshard.run.pipeline.analyze_repo", return_value=_PYTHON_REPO_WITH_TEST_CMD),
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


def test_no_retry_when_verification_passes() -> None:
    """When first verification passes, generate is called exactly once."""
    mock = _make_native_mock()
    result, g = _invoke_native_write(mock, verify_side_effects=[(0, "all passed")])
    assert result.exit_code == 0, result.output
    assert g.generate.call_count == 1
    vloop = g.native_meta.verification_loop
    assert vloop is None or not getattr(vloop, "retried", False)


def test_retry_triggered_on_verification_failure() -> None:
    """When first verification fails, generate is called a second time (retry)."""
    mock = _make_native_mock()
    result, g = _invoke_native_write(
        mock,
        verify_side_effects=[(1, "FAILED: 1 error"), (0, "all passed")],
    )
    assert result.exit_code == 0, result.output
    assert g.generate.call_count == 2


def test_retry_once_passed_recorded() -> None:
    """When second verification passes, retry_metadata.retry_verification_status is 'passed'."""
    mock = _make_native_mock()
    result, g = _invoke_native_write(
        mock,
        verify_side_effects=[(1, "FAILED test"), (0, "1 passed")],
    )
    assert result.exit_code == 0, result.output
    vloop = g.native_meta.verification_loop
    assert vloop is not None
    assert vloop.passed is True
    assert vloop.retried is True
    rmeta = vloop.retry_metadata
    assert rmeta is not None
    assert rmeta.retry_verification_status == "passed"
    assert rmeta.retry_attempted is True
    assert rmeta.retry_reason == "verification_failed"
    assert rmeta.raw_content_stored is False


def test_retry_once_failed_recorded() -> None:
    """When second verification also fails, retry_metadata.retry_verification_status is 'failed'."""
    mock = _make_native_mock()
    result, g = _invoke_native_write(
        mock,
        verify_side_effects=[(1, "FAILED test"), (1, "still failing")],
    )
    assert result.exit_code == 0, result.output
    vloop = g.native_meta.verification_loop
    assert vloop is not None
    assert vloop.passed is False
    rmeta = vloop.retry_metadata
    assert rmeta is not None
    assert rmeta.retry_verification_status == "failed"


def test_retry_capped_at_one_attempt() -> None:
    """Even if both verifications fail, generate is called exactly twice (one retry)."""
    mock = _make_native_mock()
    _invoke_native_write(
        mock,
        verify_side_effects=[(1, "fail1"), (1, "fail2")],
    )
    assert mock.generate.call_count == 2


def test_retry_patch_writes_through_existing_workspace() -> None:
    """The retry patch is written to the existing workspace, not a new path."""
    mock = _make_native_mock()
    with tempfile.TemporaryDirectory() as _td:
        sandbox_meta = NativeSandboxMeta(sandbox_enabled=False, sandbox_type="none")
        _workspace = Path(_td)

        write_calls: list = []

        def _capture_write(files, root):
            write_calls.append(root)

        _verif_iter = iter([(1, "FAILED test"), (0, "passed")])

        def _fake_verify(*args, **kwargs):
            capture = kwargs.get("capture", False)
            try:
                code, out = next(_verif_iter)
            except StopIteration:
                code, out = 0, ""
            return (code, out) if capture else code

        with (
            patch("openshard.run.pipeline.NativeAgentExecutor", return_value=mock),
            patch("openshard.run.pipeline.ProviderManager", return_value=_make_manager_mock()),
            patch("openshard.cli.main.load_config", return_value=_DEFAULT_CONFIG),
            patch("openshard.run.pipeline.analyze_repo", return_value=_PYTHON_REPO_WITH_TEST_CMD),
            patch("openshard.run.pipeline._run_verification_plan", side_effect=_fake_verify),
            patch("openshard.run.pipeline._write_files", side_effect=_capture_write),
            patch(
                "openshard.native.sandbox.create_run_sandbox",
                return_value=(_workspace, sandbox_meta),
            ),
            patch("openshard.run.pipeline._log_run"),
        ):
            runner = CliRunner()
            runner.invoke(cli, ["run", "--workflow", "native", "--write", "fix the bug"])

    assert len(write_calls) == 2, f"Expected 2 write calls, got {write_calls}"
    assert write_calls[0] == write_calls[1], "Retry must write to same workspace as initial write"


# ---------------------------------------------------------------------------
# Rendering: backward compatibility and detail levels
# ---------------------------------------------------------------------------


def test_old_run_records_render_safely() -> None:
    """An entry without verification_loop renders without crashing."""
    out = _render({"task": "old task"}, detail="full")
    assert "retry" not in out


def test_old_run_records_with_vloop_no_retry_metadata() -> None:
    """An entry with verification_loop but no retry_metadata renders safely."""
    entry = _vloop_entry(attempted=True, passed=True, retried=False)
    out = _render(entry, detail="full")
    assert "verification: passed" in out
    assert "retry:" not in out


def test_default_output_does_not_show_retry_detail() -> None:
    """default detail never shows retry metadata."""
    entry = _vloop_entry(
        attempted=True, passed=False, retried=True,
        retry_meta=_retry_meta_dict(status="passed"),
    )
    out = _render(entry, detail="default")
    assert "retry:" not in out


def test_last_more_shows_compact_retry_result_passed() -> None:
    """--full shows compact retry line when retry was attempted and passed."""
    entry = _vloop_entry(
        attempted=True, passed=True, retried=True,
        retry_meta=_retry_meta_dict(status="passed", patch_files=["src/a.py", "src/b.py"]),
    )
    out = _render(entry, detail="full")
    assert "retry: passed" in out
    assert "reason: verification_failed" in out
    assert "files patched: 2" in out


def test_last_more_shows_compact_retry_result_failed() -> None:
    """--full shows compact retry line when retry was attempted and failed."""
    entry = _vloop_entry(
        attempted=True, passed=False, retried=True,
        retry_meta=_retry_meta_dict(status="failed", patch_files=["src/foo.py"]),
    )
    out = _render(entry, detail="full")
    assert "retry: failed" in out
    assert "files patched: 1" in out


def test_last_more_does_not_show_failure_summary() -> None:
    """--more does not expose the failure_summary line (that's --full only)."""
    entry = _vloop_entry(
        attempted=True, passed=True, retried=True,
        retry_meta=_retry_meta_dict(status="passed"),
    )
    out = _render(entry, detail="more")
    assert "retry failure:" not in out


def test_last_full_shows_retry_diagnosis_and_result() -> None:
    """--full shows failure_summary and retry_patch_files."""
    entry = _vloop_entry(
        attempted=True, passed=False, retried=True,
        retry_meta=_retry_meta_dict(
            status="failed",
            failure_summary="exit_code=1 failure_type=test_failure output_chars=487 raw_content_stored=false",
            patch_files=["src/foo.py"],
        ),
    )
    out = _render(entry, detail="full")
    assert "retry: failed" in out
    assert "retry failure:" in out
    assert "failure_type=test_failure" in out
    assert "raw_content_stored=false" in out
    assert "retry patch: src/foo.py" in out


def test_last_full_no_retry_diagnosis_when_no_retry_attempted() -> None:
    """--full does not show retry failure/patch lines when retry was not attempted."""
    entry = _vloop_entry(attempted=True, passed=True, retried=False)
    out = _render(entry, detail="full")
    assert "retry failure:" not in out
    assert "retry patch:" not in out
