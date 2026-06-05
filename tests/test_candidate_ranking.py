"""Candidate Ranking / Scoring v0 — focused tests.

Covers:
  - score_native_candidate_attempt: scoring rules, clamping, multi-type support
  - score_native_candidate_summary: populates fields, preserves raw_content_stored
  - select_native_candidate: score-based selection, tie-breaking, edge cases
  - render_native_candidate_summary: score in compact + full output, backward compat
  - candidates-last CLI: Score column, real scores passed through, old-entry defaults
  - Regressions: single-candidate path, existing behaviour unchanged
"""
from __future__ import annotations

import json
import unittest
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

from click.testing import CliRunner

from openshard.cli.main import cli
from openshard.native.context import (
    NativeCandidateAttempt,
    NativeCandidateSummary,
    record_native_candidate_attempt,
    render_native_candidate_summary,
    score_native_candidate_attempt,
    score_native_candidate_summary,
    select_native_candidate,
)

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _attempt(
    *,
    candidate_index: int = 1,
    verification_status: str = "passed",
    exit_code: int | None = 0,
    files_written: list[str] | None = None,
    output_chars: int = 0,
) -> NativeCandidateAttempt:
    fw = ["a.py"] if files_written is None else files_written
    return NativeCandidateAttempt(
        candidate_index=candidate_index,
        verification_status=verification_status,
        exit_code=exit_code,
        files_written=fw,
        output_chars=output_chars,
    )


def _summary(*attempts: NativeCandidateAttempt) -> NativeCandidateSummary:
    s = NativeCandidateSummary(enabled=True, requested_count=len(attempts))
    s.candidates = list(attempts)
    s.completed_count = len(attempts)
    return s


def _make_candidate(
    index: int,
    sandbox_path: str = "",
    status: str = "passed",
    selected: bool = False,
    files: list[str] | None = None,
    exit_code: int | None = 0,
    score: float | None = None,
    score_reasons: list[str] | None = None,
) -> dict:
    rec: dict = {
        "candidate_index": index,
        "model": "mock-model",
        "sandbox_path": sandbox_path,
        "files_written": files or [],
        "verification_status": status,
        "exit_code": exit_code,
        "output_chars": 0,
        "selected": selected,
        "selection_reason": "first_passed" if selected else "",
        "raw_content_stored": False,
    }
    if score is not None:
        rec["score"] = score
    if score_reasons is not None:
        rec["score_reasons"] = score_reasons
    return rec


def _make_candidate_summary(candidates: list[dict]) -> dict:
    selected_index = next(
        (c["candidate_index"] for c in candidates if c.get("selected")), None
    )
    return {
        "enabled": True,
        "requested_count": len(candidates),
        "completed_count": len(candidates),
        "selected_index": selected_index,
        "selection_reason": "first_passed",
        "candidates": candidates,
        "raw_content_stored": False,
    }


def _write_runs(entries: list[dict]) -> Path:
    log_dir = Path(".openshard")
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "runs.jsonl"
    with log_path.open("w", encoding="utf-8") as fh:
        for e in entries:
            fh.write(json.dumps(e) + "\n")
    return log_path


def _make_entry(**kwargs) -> dict:
    return {
        "task": "test task",
        "timestamp": "2025-01-01T00:00:00Z",
        "execution_model": "anthropic/claude-sonnet-4.6",
        "executor": "native",
        "workflow": "native",
        "duration_seconds": 1.0,
        "files_created": 0,
        "files_updated": 0,
        "files_deleted": 0,
        "retry_triggered": False,
        "verification_attempted": False,
        "verification_passed": None,
        "workspace_path": None,
        "summary": "",
        "files_detail": [],
        "sandbox": None,
        **kwargs,
    }


# ---------------------------------------------------------------------------
# TestScoreNativeCandidateAttempt
# ---------------------------------------------------------------------------

class TestScoreNativeCandidateAttempt(unittest.TestCase):

    def test_passed_gets_strong_positive_score(self):
        s, r = score_native_candidate_attempt(_attempt(verification_status="passed"))
        self.assertGreaterEqual(s, 100)
        self.assertIn("verification passed", r)

    def test_failed_gets_negative_score(self):
        s, r = score_native_candidate_attempt(
            _attempt(verification_status="failed", exit_code=1)
        )
        self.assertLess(s, 0)
        self.assertIn("verification failed", r)

    def test_skipped_gets_moderate_positive_score(self):
        s, r = score_native_candidate_attempt(_attempt(verification_status="skipped"))
        self.assertGreater(s, 0)
        self.assertLess(s, 100)
        self.assertIn("verification skipped", r)

    def test_exit_code_0_adds_score(self):
        s_with, _ = score_native_candidate_attempt(
            _attempt(verification_status="skipped", exit_code=0)
        )
        s_without, _ = score_native_candidate_attempt(
            _attempt(verification_status="skipped", exit_code=None)
        )
        self.assertGreater(s_with, s_without)

    def test_nonzero_exit_code_penalises(self):
        s_bad, r = score_native_candidate_attempt(
            _attempt(verification_status="skipped", exit_code=2)
        )
        s_none, _ = score_native_candidate_attempt(
            _attempt(verification_status="skipped", exit_code=None)
        )
        self.assertLess(s_bad, s_none)
        self.assertTrue(any("exit code 2" in x for x in r))

    def test_no_files_penalised(self):
        s_none, r = score_native_candidate_attempt(
            _attempt(files_written=[])
        )
        s_some, _ = score_native_candidate_attempt(
            _attempt(files_written=["a.py"])
        )
        self.assertLess(s_none, s_some)
        self.assertIn("no files written", r)

    def test_reasonable_file_count_rewarded(self):
        s, r = score_native_candidate_attempt(
            _attempt(files_written=["a.py", "b.py", "c.py"])
        )
        self.assertIn("reasonable file count", r)

    def test_large_file_count_lightly_penalised(self):
        s_large, r = score_native_candidate_attempt(
            _attempt(files_written=[f"f{i}.py" for i in range(10)])
        )
        s_small, _ = score_native_candidate_attempt(
            _attempt(files_written=["a.py", "b.py"])
        )
        self.assertLess(s_large, s_small)
        self.assertIn("large file count", r)

    def test_large_output_chars_lightly_penalised(self):
        s_big, r = score_native_candidate_attempt(
            _attempt(output_chars=6000)
        )
        s_small, _ = score_native_candidate_attempt(
            _attempt(output_chars=100)
        )
        self.assertLess(s_big, s_small)
        self.assertIn("large verification output", r)

    def test_score_clamps_max(self):
        s, _ = score_native_candidate_attempt(
            _attempt(
                verification_status="passed",
                exit_code=0,
                files_written=["a.py"],
                output_chars=0,
            )
        )
        self.assertLessEqual(s, 150.0)

    def test_score_clamps_min(self):
        s, _ = score_native_candidate_attempt(
            _attempt(
                verification_status="failed",
                exit_code=99,
                files_written=[],
                output_chars=9999,
            )
        )
        self.assertGreaterEqual(s, -100.0)

    def test_supports_dict_candidate(self):
        d = {
            "verification_status": "passed",
            "exit_code": 0,
            "files_written": ["x.py"],
            "output_chars": 0,
        }
        s, r = score_native_candidate_attempt(d)
        self.assertIsInstance(s, float)
        self.assertIsInstance(r, list)
        self.assertIn("verification passed", r)

    def test_supports_simplenamespace_candidate(self):
        ns = SimpleNamespace(
            verification_status="failed",
            exit_code=1,
            files_written=[],
            output_chars=0,
        )
        s, r = score_native_candidate_attempt(ns)
        self.assertIsInstance(s, float)
        self.assertLess(s, 0)


# ---------------------------------------------------------------------------
# TestScoreNativeCandidateSummary
# ---------------------------------------------------------------------------

class TestScoreNativeCandidateSummary(unittest.TestCase):

    def test_populates_score_and_score_reasons(self):
        a = _attempt(verification_status="passed", exit_code=0)
        s = _summary(a)
        score_native_candidate_summary(s)
        self.assertNotEqual(s.candidates[0].score, 0.0)
        self.assertTrue(len(s.candidates[0].score_reasons) > 0)

    def test_preserves_raw_content_stored_false(self):
        a = _attempt()
        s = _summary(a)
        score_native_candidate_summary(s)
        self.assertFalse(s.raw_content_stored)
        self.assertFalse(s.candidates[0].raw_content_stored)

    def test_handles_empty_candidates_safely(self):
        s = NativeCandidateSummary(enabled=True)
        result = score_native_candidate_summary(s)
        self.assertEqual(result.candidates, [])


# ---------------------------------------------------------------------------
# TestSelectNativeCandidate
# ---------------------------------------------------------------------------

class TestSelectNativeCandidate(unittest.TestCase):

    def test_highest_score_selected(self):
        a1 = _attempt(candidate_index=1, verification_status="failed", exit_code=1, files_written=[])
        a2 = _attempt(candidate_index=2, verification_status="passed", exit_code=0)
        s = _summary(a1, a2)
        select_native_candidate(s)
        self.assertTrue(a2.selected)
        self.assertFalse(a1.selected)

    def test_passed_candidate_beats_failed_candidate(self):
        failed = _attempt(candidate_index=1, verification_status="failed", exit_code=1)
        passed = _attempt(candidate_index=2, verification_status="passed", exit_code=0)
        s = _summary(failed, passed)
        select_native_candidate(s)
        self.assertEqual(s.selected_index, 2)

    def test_tie_breaks_by_lower_candidate_index(self):
        # Give both identical inputs so scores must be equal
        a1 = _attempt(candidate_index=1, verification_status="passed", exit_code=0, files_written=["x.py"])
        a2 = _attempt(candidate_index=2, verification_status="passed", exit_code=0, files_written=["x.py"])
        s = _summary(a1, a2)
        select_native_candidate(s)
        self.assertEqual(s.selected_index, 1)

    def test_exactly_one_selected(self):
        a1 = _attempt(candidate_index=1)
        a2 = _attempt(candidate_index=2)
        a3 = _attempt(candidate_index=3)
        s = _summary(a1, a2, a3)
        select_native_candidate(s)
        selected = [c for c in s.candidates if c.selected]
        self.assertEqual(len(selected), 1)

    def test_selected_index_is_one_based(self):
        a1 = _attempt(candidate_index=1, verification_status="passed")
        a2 = _attempt(candidate_index=2, verification_status="failed")
        s = _summary(a1, a2)
        select_native_candidate(s)
        self.assertEqual(s.selected_index, 1)

    def test_selection_reason_includes_highest_score(self):
        a = _attempt(candidate_index=1, verification_status="passed", exit_code=0)
        s = _summary(a)
        select_native_candidate(s)
        self.assertIn("highest score:", s.selection_reason)

    def test_no_candidates_gives_selected_index_none(self):
        s = NativeCandidateSummary(enabled=True)
        select_native_candidate(s)
        self.assertIsNone(s.selected_index)
        self.assertEqual(s.selection_reason, "no candidates")

    def test_clears_previous_selected_state(self):
        a1 = _attempt(candidate_index=1, verification_status="failed")
        a2 = _attempt(candidate_index=2, verification_status="passed")
        a1.selected = True  # pre-set wrong winner
        s = _summary(a1, a2)
        select_native_candidate(s)
        self.assertFalse(a1.selected)
        self.assertTrue(a2.selected)


# ---------------------------------------------------------------------------
# TestRenderNativeCandidateSummary
# ---------------------------------------------------------------------------

class TestRenderNativeCandidateSummary(unittest.TestCase):

    def _scored_summary(self) -> NativeCandidateSummary:
        a = _attempt(candidate_index=1, verification_status="passed", exit_code=0)
        s = _summary(a)
        select_native_candidate(s)
        return s

    def test_compact_includes_score(self):
        s = self._scored_summary()
        out = render_native_candidate_summary(s, detail="compact")
        self.assertIn("score=", out)

    def test_full_includes_score_per_candidate(self):
        s = self._scored_summary()
        out = render_native_candidate_summary(s, detail="full")
        self.assertIn("score=", out)

    def test_full_includes_reason_line(self):
        s = self._scored_summary()
        out = render_native_candidate_summary(s, detail="full")
        self.assertIn("reason:", out)
        self.assertIn("verification passed", out)

    def test_old_candidate_records_without_score_render_safely(self):
        """A candidate dict missing score/score_reasons should not crash."""
        old_candidate = {
            "candidate_index": 1,
            "verification_status": "passed",
            "exit_code": 0,
            "files_written": ["a.py"],
            "output_chars": 0,
            "selected": True,
        }
        old_summary = {
            "enabled": True,
            "requested_count": 1,
            "completed_count": 1,
            "selected_index": 1,
            "selection_reason": "first_passed",
            "candidates": [old_candidate],
            "raw_content_stored": False,
        }
        out = render_native_candidate_summary(old_summary, detail="full")
        self.assertIn("score=0.0", out)

    def test_disabled_summary_returns_empty_string(self):
        s = NativeCandidateSummary(enabled=False)
        out = render_native_candidate_summary(s, detail="compact")
        self.assertEqual(out, "")


# ---------------------------------------------------------------------------
# TestCandidatesLastCLI
# ---------------------------------------------------------------------------

class TestCandidatesLastCLI(unittest.TestCase):

    def test_candidates_last_shows_score_column(self):
        runner = CliRunner()
        with runner.isolated_filesystem():
            cand = _make_candidate(1, selected=True, score=120.0)
            entry = _make_entry(candidate_summary=_make_candidate_summary([cand]))
            _write_runs([entry])
            result = runner.invoke(cli, ["candidates-last"])
        self.assertIn("Score", result.output)

    def test_candidates_last_shows_real_score_from_candidate_summary(self):
        """candidates-last must display the stored score, not fall back to 0.0."""
        runner = CliRunner()
        with runner.isolated_filesystem():
            cand = _make_candidate(1, selected=True, score=77.5, score_reasons=["verification passed"])
            entry = _make_entry(candidate_summary=_make_candidate_summary([cand]))
            _write_runs([entry])
            result = runner.invoke(cli, ["candidates-last"])
        self.assertIn("77.5", result.output)

    def test_old_candidate_records_show_zero_score(self):
        """Entries without score field must show 0.0, not crash."""
        runner = CliRunner()
        with runner.isolated_filesystem():
            cand = _make_candidate(1, selected=True)  # no score kwarg
            entry = _make_entry(candidate_summary=_make_candidate_summary([cand]))
            _write_runs([entry])
            result = runner.invoke(cli, ["candidates-last"])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("0.0", result.output)

    def test_no_raw_content_in_candidates_last(self):
        """candidates-last must never print file content."""
        runner = CliRunner()
        with runner.isolated_filesystem():
            cand = _make_candidate(1, selected=True, files=["secret.py"])
            entry = _make_entry(candidate_summary=_make_candidate_summary([cand]))
            _write_runs([entry])
            result = runner.invoke(cli, ["candidates-last"])
        self.assertNotIn("secret content", result.output)
        self.assertNotIn("def ", result.output)


# ---------------------------------------------------------------------------
# TestRegressionCandidates
# ---------------------------------------------------------------------------

class TestRegressionCandidates(unittest.TestCase):

    def test_single_candidate_deterministic_selection(self):
        """Single candidate: selected_index == 1, exactly one candidate selected."""
        a = _attempt(candidate_index=1, verification_status="passed", exit_code=0)
        s = _summary(a)
        result = select_native_candidate(s)
        self.assertEqual(result.selected_index, 1)
        self.assertTrue(result.candidates[0].selected)
        self.assertEqual(len([c for c in result.candidates if c.selected]), 1)

    def test_score_and_reasons_populated_after_select(self):
        """score and score_reasons are set as a side-effect of selection."""
        a = _attempt(candidate_index=1, verification_status="passed", exit_code=0)
        s = _summary(a)
        select_native_candidate(s)
        self.assertGreater(s.candidates[0].score, 0)
        self.assertTrue(len(s.candidates[0].score_reasons) > 0)

    def test_asdict_serializes_new_fields(self):
        """New score fields survive asdict() round-trip without error."""
        a = _attempt()
        d = asdict(a)
        self.assertIn("score", d)
        self.assertIn("score_reasons", d)

    def test_record_native_candidate_attempt_creates_default_score(self):
        """record_native_candidate_attempt leaves score=0.0 before scoring."""
        s = NativeCandidateSummary(enabled=True)
        record_native_candidate_attempt(
            s,
            candidate_index=1,
            model="m",
            sandbox_path="",
            files_written=[],
            verification_status="passed",
        )
        self.assertEqual(s.candidates[0].score, 0.0)
        self.assertEqual(s.candidates[0].score_reasons, [])


if __name__ == "__main__":
    unittest.main()
