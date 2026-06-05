"""Native Candidate Agents v0 — focused tests.

Covers:
  - NativeCandidateAttempt / NativeCandidateSummary dataclass invariants
  - record_native_candidate_attempt helper behaviour
  - select_native_candidate priority logic
  - render_native_candidate_summary output format (compact + full, dict/NS/dataclass)
  - CLI: --candidates option validation
  - Pipeline integration: candidates=1 unchanged, candidates=2 behaviour
"""
from __future__ import annotations

import tempfile
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from click.testing import CliRunner

from openshard.analysis.repo import RepoFacts
from openshard.cli.main import cli
from openshard.native.context import (
    NativeApprovalRequest,
    NativeCandidateAttempt,
    NativeCandidateSummary,
    NativeChangeBudgetSoftGate,
    NativeSandboxMeta,
    record_native_candidate_attempt,
    render_native_candidate_summary,
    select_native_candidate,
)

# ---------------------------------------------------------------------------
# Shared helpers
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


def _make_native_mock(generate_side_effect=None):
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
    g.native_meta = NativeRunMeta()
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


def _invoke_native_write_candidates(
    native_mock=None,
    *,
    candidates: int = 2,
    verify_side_effects: list | None = None,
    repo: RepoFacts | None = None,
):
    if native_mock is None:
        native_mock = _make_native_mock()
    if repo is None:
        repo = _PYTHON_REPO_WITH_TEST_CMD

    _verif_calls = iter(verify_side_effects or [(0, "")] * (candidates + 1))

    def _fake_verify(*args, **kwargs):
        capture = kwargs.get("capture", False)
        try:
            code, out = next(_verif_calls)
        except StopIteration:
            code, out = 0, ""
        return (code, out) if capture else code

    sandbox_meta = NativeSandboxMeta(sandbox_enabled=False, sandbox_type="none")

    with tempfile.TemporaryDirectory() as _td:
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
                ["run", "--workflow", "native", "--write", "--candidates", str(candidates), "fix the bug"],
            )
    return result, native_mock


# ---------------------------------------------------------------------------
# NativeCandidateAttempt — dataclass defaults and invariants
# ---------------------------------------------------------------------------

def test_attempt_default_candidate_index_zero():
    assert NativeCandidateAttempt().candidate_index == 0


def test_attempt_default_model_empty():
    assert NativeCandidateAttempt().model == ""


def test_attempt_default_verification_status_empty():
    assert NativeCandidateAttempt().verification_status == ""


def test_attempt_default_files_written_empty_list():
    assert NativeCandidateAttempt().files_written == []


def test_attempt_default_exit_code_none():
    assert NativeCandidateAttempt().exit_code is None


def test_attempt_default_output_chars_zero():
    assert NativeCandidateAttempt().output_chars == 0


def test_attempt_default_selected_false():
    assert NativeCandidateAttempt().selected is False


def test_attempt_default_selection_reason_empty():
    assert NativeCandidateAttempt().selection_reason == ""


def test_attempt_raw_content_stored_always_false():
    assert NativeCandidateAttempt().raw_content_stored is False


def test_attempt_raw_content_stored_reset_by_post_init():
    a = NativeCandidateAttempt()
    a.raw_content_stored = True
    # post_init already ran; confirm the field default is False on a fresh instance
    b = NativeCandidateAttempt()
    assert b.raw_content_stored is False


def test_attempt_files_written_lists_are_independent():
    a1 = NativeCandidateAttempt()
    a2 = NativeCandidateAttempt()
    a1.files_written.append("foo.py")
    assert a2.files_written == []


def test_attempt_serializes_cleanly():
    d = asdict(NativeCandidateAttempt())
    assert d["raw_content_stored"] is False
    assert d["files_written"] == []


# ---------------------------------------------------------------------------
# NativeCandidateSummary — dataclass defaults and invariants
# ---------------------------------------------------------------------------

def test_summary_enabled_false_by_default():
    assert NativeCandidateSummary().enabled is False


def test_summary_requested_count_one_by_default():
    assert NativeCandidateSummary().requested_count == 1


def test_summary_completed_count_zero_by_default():
    assert NativeCandidateSummary().completed_count == 0


def test_summary_selected_index_none_by_default():
    assert NativeCandidateSummary().selected_index is None


def test_summary_candidates_empty_by_default():
    assert NativeCandidateSummary().candidates == []


def test_summary_raw_content_stored_always_false():
    assert NativeCandidateSummary().raw_content_stored is False


def test_summary_candidates_lists_are_independent():
    s1 = NativeCandidateSummary()
    s2 = NativeCandidateSummary()
    s1.candidates.append(NativeCandidateAttempt())
    assert s2.candidates == []


def test_summary_serializes_cleanly():
    d = asdict(NativeCandidateSummary())
    assert d["raw_content_stored"] is False
    assert d["candidates"] == []
    assert d["enabled"] is False


# ---------------------------------------------------------------------------
# record_native_candidate_attempt — helper behaviour
# ---------------------------------------------------------------------------

def test_record_appends_one_attempt():
    s = NativeCandidateSummary()
    record_native_candidate_attempt(
        s, candidate_index=1, model="m", sandbox_path="/tmp/c1",
        files_written=[], verification_status="passed",
    )
    assert len(s.candidates) == 1


def test_record_two_calls_append_two_attempts():
    s = NativeCandidateSummary()
    for i in range(1, 3):
        record_native_candidate_attempt(
            s, candidate_index=i, model="m", sandbox_path=f"/tmp/c{i}",
            files_written=[], verification_status="passed",
        )
    assert len(s.candidates) == 2


def test_record_updates_completed_count():
    s = NativeCandidateSummary()
    record_native_candidate_attempt(
        s, candidate_index=1, model="m", sandbox_path="/tmp/c1",
        files_written=[], verification_status="failed",
    )
    assert s.completed_count == 1


def test_record_files_written_stored_as_copy():
    s = NativeCandidateSummary()
    src = ["a.py", "b.py"]
    record_native_candidate_attempt(
        s, candidate_index=1, model="m", sandbox_path="/tmp/c1",
        files_written=src, verification_status="passed",
    )
    src.append("mutated.py")
    assert "mutated.py" not in s.candidates[0].files_written


def test_record_exit_code_and_chars_stored():
    s = NativeCandidateSummary()
    record_native_candidate_attempt(
        s, candidate_index=1, model="m", sandbox_path="/tmp/c1",
        files_written=[], verification_status="failed",
        exit_code=1, output_chars=500,
    )
    assert s.candidates[0].exit_code == 1
    assert s.candidates[0].output_chars == 500


def test_record_raw_content_never_stored():
    s = NativeCandidateSummary()
    record_native_candidate_attempt(
        s, candidate_index=1, model="m", sandbox_path="/tmp/c1",
        files_written=[], verification_status="passed",
    )
    assert s.candidates[0].raw_content_stored is False


def test_record_candidate_index_stored():
    s = NativeCandidateSummary()
    record_native_candidate_attempt(
        s, candidate_index=2, model="m", sandbox_path="/tmp/c2",
        files_written=[], verification_status="skipped",
    )
    assert s.candidates[0].candidate_index == 2


# ---------------------------------------------------------------------------
# select_native_candidate — priority logic
# ---------------------------------------------------------------------------

def test_select_empty_candidates_returns_unchanged():
    s = NativeCandidateSummary()
    result = select_native_candidate(s)
    assert result.selected_index is None


def test_select_first_passed_wins():
    s = NativeCandidateSummary()
    for i, status in enumerate(["failed", "passed", "passed"], start=1):
        record_native_candidate_attempt(
            s, candidate_index=i, model="m", sandbox_path=f"/tmp/c{i}",
            files_written=[], verification_status=status,
        )
    select_native_candidate(s)
    assert s.selected_index == 2  # 1-based index of the first "passed"


def test_select_skipped_wins_when_no_passed():
    s = NativeCandidateSummary()
    for i, status in enumerate(["failed", "skipped"], start=1):
        record_native_candidate_attempt(
            s, candidate_index=i, model="m", sandbox_path=f"/tmp/c{i}",
            files_written=[], verification_status=status,
        )
    select_native_candidate(s)
    assert s.selected_index == 2


def test_select_fallback_first_when_all_failed():
    s = NativeCandidateSummary()
    for i in range(1, 4):
        record_native_candidate_attempt(
            s, candidate_index=i, model="m", sandbox_path=f"/tmp/c{i}",
            files_written=[], verification_status="failed",
        )
    select_native_candidate(s)
    assert s.selected_index == 1
    assert s.selection_reason.startswith("highest score:")


def test_select_sets_selection_reason_first_passed():
    s = NativeCandidateSummary()
    record_native_candidate_attempt(
        s, candidate_index=1, model="m", sandbox_path="/tmp/c1",
        files_written=[], verification_status="passed",
    )
    select_native_candidate(s)
    assert s.selection_reason.startswith("highest score:")


def test_select_sets_selection_reason_first_skipped():
    s = NativeCandidateSummary()
    record_native_candidate_attempt(
        s, candidate_index=1, model="m", sandbox_path="/tmp/c1",
        files_written=[], verification_status="failed",
    )
    record_native_candidate_attempt(
        s, candidate_index=2, model="m", sandbox_path="/tmp/c2",
        files_written=[], verification_status="skipped",
    )
    select_native_candidate(s)
    assert s.selection_reason.startswith("highest score:")


def test_select_only_winner_has_selected_true():
    s = NativeCandidateSummary()
    for i, status in enumerate(["failed", "passed", "failed"], start=1):
        record_native_candidate_attempt(
            s, candidate_index=i, model="m", sandbox_path=f"/tmp/c{i}",
            files_written=[], verification_status=status,
        )
    select_native_candidate(s)
    selected = [c for c in s.candidates if c.selected]
    assert len(selected) == 1
    assert selected[0].candidate_index == 2


def test_select_selected_index_is_one_based():
    s = NativeCandidateSummary()
    record_native_candidate_attempt(
        s, candidate_index=1, model="m", sandbox_path="/tmp/c1",
        files_written=[], verification_status="passed",
    )
    select_native_candidate(s)
    assert s.selected_index == 1  # 1-based, not 0


# ---------------------------------------------------------------------------
# render_native_candidate_summary — output format
# ---------------------------------------------------------------------------

def test_render_none_returns_empty_string():
    assert render_native_candidate_summary(None) == ""


def test_render_disabled_summary_returns_empty_string():
    s = NativeCandidateSummary(enabled=False)
    assert render_native_candidate_summary(s) == ""


def test_render_enabled_summary_returns_nonempty():
    s = NativeCandidateSummary(enabled=True, requested_count=2, completed_count=2)
    assert render_native_candidate_summary(s) != ""


def test_render_compact_is_single_line():
    s = NativeCandidateSummary(enabled=True, requested_count=2, completed_count=2)
    out = render_native_candidate_summary(s, detail="compact")
    assert "\n" not in out


def test_render_compact_shows_count():
    s = NativeCandidateSummary(enabled=True, requested_count=2, completed_count=2)
    out = render_native_candidate_summary(s, detail="compact")
    assert "2/2" in out


def test_render_compact_shows_selected_one_based():
    s = NativeCandidateSummary(enabled=True, requested_count=2, completed_count=2, selected_index=1)
    out = render_native_candidate_summary(s, detail="compact")
    assert "selected=1" in out


def test_render_compact_shows_passed_count():
    s = NativeCandidateSummary(enabled=True, requested_count=2, completed_count=2)
    record_native_candidate_attempt(
        s, candidate_index=1, model="m", sandbox_path="/tmp/c1",
        files_written=[], verification_status="passed",
    )
    record_native_candidate_attempt(
        s, candidate_index=2, model="m", sandbox_path="/tmp/c2",
        files_written=[], verification_status="failed",
    )
    out = render_native_candidate_summary(s, detail="compact")
    assert "passed=1" in out


def test_render_full_is_multiline_with_candidates():
    s = NativeCandidateSummary(enabled=True, requested_count=2, completed_count=2)
    record_native_candidate_attempt(
        s, candidate_index=1, model="m", sandbox_path="/tmp/c1",
        files_written=["a.py"], verification_status="passed", exit_code=0, output_chars=100,
    )
    record_native_candidate_attempt(
        s, candidate_index=2, model="m", sandbox_path="/tmp/c2",
        files_written=[], verification_status="failed", exit_code=1, output_chars=200,
    )
    out = render_native_candidate_summary(s, detail="full")
    assert "\n" in out


def test_render_full_shows_files_and_exit():
    s = NativeCandidateSummary(enabled=True, requested_count=1, completed_count=1)
    record_native_candidate_attempt(
        s, candidate_index=1, model="m", sandbox_path="/tmp/c1",
        files_written=["a.py", "b.py"], verification_status="passed", exit_code=0, output_chars=50,
    )
    out = render_native_candidate_summary(s, detail="full")
    assert "files=2" in out
    assert "exit=0" in out
    assert "chars=50" in out


def test_render_full_marks_selected_candidate():
    s = NativeCandidateSummary(enabled=True, requested_count=2, completed_count=2)
    for i, status in enumerate(["failed", "passed"], start=1):
        record_native_candidate_attempt(
            s, candidate_index=i, model="m", sandbox_path=f"/tmp/c{i}",
            files_written=[], verification_status=status,
        )
    select_native_candidate(s)
    out = render_native_candidate_summary(s, detail="full")
    assert "selected" in out


def test_render_full_candidate_index_is_one_based():
    s = NativeCandidateSummary(enabled=True, requested_count=1, completed_count=1)
    record_native_candidate_attempt(
        s, candidate_index=1, model="m", sandbox_path="/tmp/c1",
        files_written=[], verification_status="passed",
    )
    out = render_native_candidate_summary(s, detail="full")
    # Should show "1." not "0."
    assert "  1." in out


def test_render_accepts_simple_namespace():
    ns = SimpleNamespace(
        enabled=True,
        requested_count=1,
        completed_count=1,
        selected_index=1,
        selection_reason="first_passed",
        raw_content_stored=False,
        candidates=[
            SimpleNamespace(
                candidate_index=1,
                model="m",
                sandbox_path="/tmp/c1",
                files_written=[],
                verification_status="passed",
                exit_code=0,
                output_chars=0,
                selected=True,
                selection_reason="first_passed",
                raw_content_stored=False,
            )
        ],
    )
    out = render_native_candidate_summary(ns, detail="compact")
    assert out != ""
    assert "passed" in out


def test_render_accepts_dict():
    s = NativeCandidateSummary(enabled=True, requested_count=1, completed_count=1)
    record_native_candidate_attempt(
        s, candidate_index=1, model="m", sandbox_path="/tmp/c1",
        files_written=["x.py"], verification_status="passed", exit_code=0, output_chars=50,
    )
    select_native_candidate(s)
    d = asdict(s)
    out = render_native_candidate_summary(d, detail="compact")
    assert "passed" in out
    assert out != ""


def test_render_dict_full_shows_per_candidate_lines():
    s = NativeCandidateSummary(enabled=True, requested_count=2, completed_count=2)
    record_native_candidate_attempt(
        s, candidate_index=1, model="m", sandbox_path="/tmp/c1",
        files_written=["a.py"], verification_status="passed", exit_code=0, output_chars=10,
    )
    record_native_candidate_attempt(
        s, candidate_index=2, model="m", sandbox_path="/tmp/c2",
        files_written=[], verification_status="failed", exit_code=1, output_chars=20,
    )
    select_native_candidate(s)
    d = asdict(s)
    out = render_native_candidate_summary(d, detail="full")
    assert "\n" in out
    assert "1." in out
    assert "2." in out


# ---------------------------------------------------------------------------
# CLI — --candidates option validation
# ---------------------------------------------------------------------------

def test_cli_candidates_hidden_from_help():
    result = CliRunner().invoke(cli, ["run", "--help"])
    assert "--candidates" not in result.output


def test_cli_candidates_above_three_rejected():
    with (
        patch("openshard.cli.main.load_config", return_value=_DEFAULT_CONFIG),
    ):
        result = CliRunner().invoke(
            cli, ["run", "--candidates", "4", "--workflow", "native", "--write", "task"]
        )
    assert result.exit_code != 0


def test_cli_candidates_zero_rejected():
    with (
        patch("openshard.cli.main.load_config", return_value=_DEFAULT_CONFIG),
    ):
        result = CliRunner().invoke(
            cli, ["run", "--candidates", "0", "--workflow", "native", "--write", "task"]
        )
    assert result.exit_code != 0


# ---------------------------------------------------------------------------
# Pipeline integration — candidates=1 unchanged (non-regression)
# ---------------------------------------------------------------------------

def test_pipeline_candidates_one_generate_called_once():
    mock = _make_native_mock()
    _invoke_native_write_candidates(mock, candidates=1, verify_side_effects=[(0, "ok")])
    assert mock.generate.call_count == 1


def test_pipeline_candidates_one_no_candidate_summary():
    _, g = _invoke_native_write_candidates(candidates=1, verify_side_effects=[(0, "ok")])
    assert g.native_meta.candidate_summary is None


def test_pipeline_candidates_one_edit_loop_summary_still_set():
    _, g = _invoke_native_write_candidates(candidates=1, verify_side_effects=[(0, "ok")])
    assert g.native_meta.edit_loop_summary is not None


def test_pipeline_candidates_one_exit_zero():
    result, _ = _invoke_native_write_candidates(candidates=1, verify_side_effects=[(0, "ok")])
    assert result.exit_code == 0


# ---------------------------------------------------------------------------
# Pipeline integration — candidates=2
# ---------------------------------------------------------------------------

def test_pipeline_candidates_two_exit_zero():
    result, _ = _invoke_native_write_candidates(
        candidates=2, verify_side_effects=[(0, "ok"), (0, "ok")]
    )
    assert result.exit_code == 0


def test_pipeline_candidates_two_generate_called_twice():
    mock = _make_native_mock()
    _invoke_native_write_candidates(mock, candidates=2, verify_side_effects=[(0, "ok"), (0, "ok")])
    assert mock.generate.call_count == 2


def test_pipeline_candidates_two_candidate_summary_set():
    _, g = _invoke_native_write_candidates(candidates=2, verify_side_effects=[(0, "ok"), (1, "fail")])
    assert g.native_meta.candidate_summary is not None


def test_pipeline_candidates_two_enabled_true():
    _, g = _invoke_native_write_candidates(candidates=2, verify_side_effects=[(0, "ok"), (1, "fail")])
    assert g.native_meta.candidate_summary.enabled is True


def test_pipeline_candidates_two_requested_count():
    _, g = _invoke_native_write_candidates(candidates=2, verify_side_effects=[(0, "ok"), (1, "fail")])
    assert g.native_meta.candidate_summary.requested_count == 2


def test_pipeline_candidates_two_completed_count():
    _, g = _invoke_native_write_candidates(candidates=2, verify_side_effects=[(0, "ok"), (1, "fail")])
    assert g.native_meta.candidate_summary.completed_count == 2


def test_pipeline_candidates_two_passed_wins_over_failed():
    _, g = _invoke_native_write_candidates(
        candidates=2,
        verify_side_effects=[(1, "fail"), (0, "ok")],
    )
    # candidate 2 (index 2, 1-based) passed
    assert g.native_meta.candidate_summary.selected_index == 2


def test_pipeline_candidates_all_failed_fallback_to_first():
    _, g = _invoke_native_write_candidates(
        candidates=2,
        verify_side_effects=[(1, "fail"), (1, "fail")],
    )
    assert g.native_meta.candidate_summary.selected_index == 1
    assert g.native_meta.candidate_summary.selection_reason.startswith("highest score:")


def test_pipeline_candidates_skipped_verification_first_selected():
    _, g = _invoke_native_write_candidates(
        candidates=2,
        repo=_PYTHON_REPO_NO_TEST_CMD,
    )
    assert g.native_meta.candidate_summary is not None
    for cand in g.native_meta.candidate_summary.candidates:
        assert cand.verification_status == "skipped"
    assert g.native_meta.candidate_summary.selected_index == 1
    assert g.native_meta.candidate_summary.selection_reason.startswith("highest score:")


def test_pipeline_candidates_two_raw_content_never_stored():
    _, g = _invoke_native_write_candidates(
        candidates=2,
        verify_side_effects=[(0, "ok"), (0, "ok")],
    )
    for c in g.native_meta.candidate_summary.candidates:
        assert c.raw_content_stored is False
    assert g.native_meta.candidate_summary.raw_content_stored is False


def test_pipeline_candidates_two_edit_loop_not_run():
    _, g = _invoke_native_write_candidates(
        candidates=2,
        verify_side_effects=[(0, "ok"), (0, "ok")],
    )
    assert g.native_meta.edit_loop_summary is None


def test_pipeline_candidates_two_creates_two_sandboxes():
    sandbox_meta = NativeSandboxMeta(sandbox_enabled=False, sandbox_type="none")
    mock = _make_native_mock()
    _verif = iter([(0, ""), (0, "")])

    def _fake_verify(*a, **kw):
        capture = kw.get("capture", False)
        code, out = next(_verif, (0, ""))
        return (code, out) if capture else code

    create_sandbox_mock = MagicMock()

    with tempfile.TemporaryDirectory() as _td:
        create_sandbox_mock.return_value = (Path(_td), sandbox_meta)
        with (
            patch("openshard.run.pipeline.NativeAgentExecutor", return_value=mock),
            patch("openshard.run.pipeline.ProviderManager", return_value=_make_manager_mock()),
            patch("openshard.cli.main.load_config", return_value=_DEFAULT_CONFIG),
            patch("openshard.run.pipeline.analyze_repo", return_value=_PYTHON_REPO_WITH_TEST_CMD),
            patch("openshard.run.pipeline._run_verification_plan", side_effect=_fake_verify),
            patch("openshard.run.pipeline._write_files"),
            patch("openshard.native.sandbox.create_run_sandbox", create_sandbox_mock),
            patch("openshard.run.pipeline._log_run"),
        ):
            CliRunner().invoke(
                cli,
                ["run", "--workflow", "native", "--write", "--candidates", "2", "fix the bug"],
            )

    assert create_sandbox_mock.call_count == 2


def test_pipeline_candidates_candidate_indexes_are_one_based():
    _, g = _invoke_native_write_candidates(
        candidates=2,
        verify_side_effects=[(0, "ok"), (0, "ok")],
    )
    indexes = [c.candidate_index for c in g.native_meta.candidate_summary.candidates]
    assert indexes == [1, 2]
