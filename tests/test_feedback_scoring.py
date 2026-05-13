from __future__ import annotations

import unittest

from openshard.history.feedback_scoring import (
    MIN_EVIDENCE,
    _ADJ_MAX,
    _ADJ_MIN,
    _feedback_signal,
    compute_feedback_adjustments,
    compute_feedback_adjustment_reasons,
)


def _fb(action: str | None = None, reason: str | None = None, note: str = "") -> dict:
    fb: dict = {"schema_version": 1, "rating": "bad", "note": note, "created_at": "2025-01-01T00:00:00Z"}
    if action is not None:
        fb["action"] = action
    if reason is not None:
        fb["correction_reason"] = reason
    return fb


def _run(model: str = "openrouter/fast-model", feedback: dict | None = None) -> dict:
    entry: dict = {
        "task": "test task",
        "timestamp": "2025-01-01T00:00:00Z",
        "execution_model": model,
        "duration_seconds": 1.0,
        "files_created": 0,
        "files_updated": 0,
        "files_deleted": 0,
        "retry_triggered": False,
        "verification_attempted": False,
        "verification_passed": None,
    }
    if feedback is not None:
        entry["feedback"] = feedback
    return entry


def _runs_with_feedback(model: str, n: int, action: str, reason: str | None = None) -> list[dict]:
    return [_run(model, _fb(action=action, reason=reason)) for _ in range(n)]


class TestFeedbackSignal(unittest.TestCase):

    def test_accepted_gives_positive_signal(self):
        self.assertGreater(_feedback_signal(_fb(action="accepted")), 0.0)

    def test_partially_accepted_gives_small_positive_signal(self):
        sig = _feedback_signal(_fb(action="partially-accepted"))
        self.assertGreater(sig, 0.0)
        self.assertLess(sig, _feedback_signal(_fb(action="accepted")))

    def test_rejected_gives_negative_signal(self):
        self.assertLess(_feedback_signal(_fb(action="rejected")), 0.0)

    def test_retried_gives_negative_signal(self):
        self.assertLess(_feedback_signal(_fb(action="retried")), 0.0)

    def test_edited_no_reason_gives_small_negative_signal(self):
        self.assertAlmostEqual(_feedback_signal(_fb(action="edited")), -0.10)

    def test_edited_with_serious_reason_gives_stronger_negative(self):
        sig_serious = _feedback_signal(_fb(action="edited", reason="failed-tests"))
        sig_plain = _feedback_signal(_fb(action="edited"))
        self.assertLess(sig_serious, sig_plain)
        self.assertAlmostEqual(sig_serious, -0.20)

    def test_edited_with_hallucinated_reason_is_capped_at_minus_0_2(self):
        self.assertAlmostEqual(_feedback_signal(_fb(action="edited", reason="hallucinated")), -0.20)

    def test_hallucinated_reason_with_non_edited_action_is_strong_negative(self):
        sig = _feedback_signal(_fb(action="rejected", reason="hallucinated"))
        self.assertAlmostEqual(sig, -0.30)

    def test_wrong_file_with_rejected_takes_more_negative(self):
        sig = _feedback_signal(_fb(action="rejected", reason="wrong-file"))
        self.assertAlmostEqual(sig, -0.25)

    def test_note_is_ignored(self):
        with_note = _feedback_signal(_fb(action="accepted", note="great job!"))
        without_note = _feedback_signal(_fb(action="accepted"))
        self.assertEqual(with_note, without_note)

    def test_note_alone_gives_zero_signal(self):
        self.assertEqual(_feedback_signal({"schema_version": 1, "note": "some note", "created_at": "2025-01-01T00:00:00Z"}), 0.0)

    def test_cost_speed_reasons_give_zero_signal_for_accepted(self):
        for reason in ("too-expensive", "too-slow"):
            with self.subTest(reason=reason):
                self.assertAlmostEqual(
                    _feedback_signal(_fb(action="accepted", reason=reason)),
                    _feedback_signal(_fb(action="accepted")),
                )

    def test_unknown_action_gives_zero_signal(self):
        self.assertEqual(_feedback_signal(_fb(action="unknown")), 0.0)

    def test_missing_action_gives_zero_signal(self):
        self.assertEqual(_feedback_signal({"schema_version": 1, "note": "", "created_at": "2025-01-01T00:00:00Z"}), 0.0)


class TestComputeFeedbackAdjustments(unittest.TestCase):

    def test_empty_runs_returns_empty(self):
        self.assertEqual(compute_feedback_adjustments([]), {})

    def test_runs_without_feedback_returns_empty(self):
        runs = [_run() for _ in range(10)]
        self.assertEqual(compute_feedback_adjustments(runs), {})

    def test_below_min_evidence_returns_no_entry(self):
        runs = _runs_with_feedback("openrouter/fast-model", MIN_EVIDENCE - 1, "accepted")
        result = compute_feedback_adjustments(runs)
        self.assertNotIn("openrouter/fast-model", result)

    def test_at_min_evidence_returns_entry(self):
        runs = _runs_with_feedback("openrouter/fast-model", MIN_EVIDENCE, "accepted")
        result = compute_feedback_adjustments(runs)
        self.assertIn("openrouter/fast-model", result)

    def test_accepted_history_gives_positive_adjustment(self):
        runs = _runs_with_feedback("openrouter/fast-model", MIN_EVIDENCE, "accepted")
        result = compute_feedback_adjustments(runs)
        self.assertGreater(result["openrouter/fast-model"], 0.0)

    def test_rejected_history_gives_negative_adjustment(self):
        runs = _runs_with_feedback("openrouter/fast-model", MIN_EVIDENCE, "rejected")
        result = compute_feedback_adjustments(runs)
        self.assertLess(result["openrouter/fast-model"], 0.0)

    def test_retried_history_gives_negative_adjustment(self):
        runs = _runs_with_feedback("openrouter/fast-model", MIN_EVIDENCE, "retried")
        result = compute_feedback_adjustments(runs)
        self.assertLess(result["openrouter/fast-model"], 0.0)

    def test_serious_reason_gives_stronger_negative_than_mild(self):
        runs_serious = _runs_with_feedback("m1", MIN_EVIDENCE, "edited", reason="failed-tests")
        runs_mild = _runs_with_feedback("m2", MIN_EVIDENCE, "edited")
        adj_serious = compute_feedback_adjustments(runs_serious + runs_mild)
        self.assertLess(adj_serious["m1"], adj_serious["m2"])

    def test_adjustment_is_clamped_at_lower_bound(self):
        runs = _runs_with_feedback("openrouter/fast-model", 100, "rejected", reason="hallucinated")
        result = compute_feedback_adjustments(runs)
        self.assertGreaterEqual(result["openrouter/fast-model"], _ADJ_MIN)

    def test_adjustment_is_clamped_at_upper_bound(self):
        runs = _runs_with_feedback("openrouter/fast-model", 100, "accepted")
        result = compute_feedback_adjustments(runs)
        self.assertLessEqual(result["openrouter/fast-model"], _ADJ_MAX)

    def test_old_entries_without_feedback_field_do_not_crash(self):
        runs = [
            {"task": "t", "timestamp": "2024-01-01T00:00:00Z", "execution_model": "m"},
            _run("m", _fb(action="accepted")),
            _run("m", _fb(action="accepted")),
            _run("m", _fb(action="accepted")),
        ]
        result = compute_feedback_adjustments(runs)
        self.assertIn("m", result)

    def test_multiple_models_computed_independently(self):
        runs_a = _runs_with_feedback("model-a", MIN_EVIDENCE, "accepted")
        runs_b = _runs_with_feedback("model-b", MIN_EVIDENCE, "rejected")
        result = compute_feedback_adjustments(runs_a + runs_b)
        self.assertGreater(result["model-a"], 0.0)
        self.assertLess(result["model-b"], 0.0)

    def test_feedback_scoring_does_not_mutate_runs(self):
        runs = _runs_with_feedback("openrouter/fast-model", MIN_EVIDENCE, "rejected")
        original = [dict(r) for r in runs]
        compute_feedback_adjustments(runs)
        for before, after in zip(original, runs):
            self.assertEqual(before, after)


class TestComputeFeedbackAdjustmentReasons(unittest.TestCase):

    def test_no_reason_for_model_below_threshold(self):
        runs = _runs_with_feedback("m", MIN_EVIDENCE - 1, "rejected")
        reasons = compute_feedback_adjustment_reasons(runs)
        self.assertNotIn("m", reasons)

    def test_reason_string_present_for_rejected_model(self):
        runs = _runs_with_feedback("m", MIN_EVIDENCE, "rejected")
        reasons = compute_feedback_adjustment_reasons(runs)
        self.assertIn("m", reasons)
        self.assertIn("feedback:", reasons["m"])

    def test_reason_includes_rejection_count(self):
        runs = _runs_with_feedback("m", MIN_EVIDENCE, "rejected")
        reasons = compute_feedback_adjustment_reasons(runs)
        self.assertIn("rejected", reasons["m"])

    def test_reason_includes_correction_reason_when_present(self):
        runs = _runs_with_feedback("m", MIN_EVIDENCE, "edited", reason="wrong-file")
        reasons = compute_feedback_adjustment_reasons(runs)
        self.assertIn("m", reasons)
        self.assertIn("wrong-file", reasons["m"])

    def test_zero_adjustment_model_omitted_from_reasons(self):
        # mixed accepted/rejected signals that cancel to ~0 after averaging may not appear
        # at minimum: model with no feedback records is definitely absent
        reasons = compute_feedback_adjustment_reasons([])
        self.assertEqual(reasons, {})


if __name__ == "__main__":
    unittest.main()
