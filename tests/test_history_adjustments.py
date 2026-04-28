from __future__ import annotations

import unittest

from openshard.history.adjustments import (
    MIN_MODEL_RUNS,
    MIN_VERIFIED_RUNS,
    compute_history_adjustments,
)


def _run(
    model: str,
    verification_passed=None,
    retry_triggered: bool = False,
) -> dict:
    return {
        "execution_model": model,
        "verification_passed": verification_passed,
        "retry_triggered": retry_triggered,
        "duration_seconds": 5.0,
    }


def _runs(model: str, n: int, passed: bool | None = None, retry: bool = False) -> list[dict]:
    return [_run(model, verification_passed=passed, retry_triggered=retry) for _ in range(n)]


class TestComputeHistoryAdjustments(unittest.TestCase):

    def test_empty_runs_returns_empty_dict(self):
        self.assertEqual(compute_history_adjustments([]), {})

    def test_no_adjustment_below_min_model_runs(self):
        runs = _runs("m/a", MIN_MODEL_RUNS - 1, passed=True)
        adj = compute_history_adjustments(runs)
        self.assertEqual(adj.get("m/a", 0.0), 0.0)

    def test_adjustment_unlocked_at_min_model_runs(self):
        # Exactly MIN_MODEL_RUNS verified passing runs → should get bonus.
        runs = _runs("m/a", MIN_MODEL_RUNS, passed=True)
        adj = compute_history_adjustments(runs)
        # MIN_MODEL_RUNS >= MIN_VERIFIED_RUNS so pass-rate signal should fire
        if MIN_MODEL_RUNS >= MIN_VERIFIED_RUNS:
            self.assertGreater(adj.get("m/a", 0.0), 0.0)

    def test_no_pass_rate_signal_below_min_verified_runs(self):
        # 5 total runs but fewer than MIN_VERIFIED_RUNS have verification_passed set.
        # Pass rate signal should be suppressed; retry signal still available.
        runs = (
            _runs("m/b", MIN_MODEL_RUNS - MIN_VERIFIED_RUNS, passed=None)  # unverified
            + _runs("m/b", MIN_VERIFIED_RUNS - 1, passed=True)             # verified but under threshold
        )
        # Pad to meet MIN_MODEL_RUNS if needed
        while len(runs) < MIN_MODEL_RUNS:
            runs.append(_run("m/b", verification_passed=None))
        adj = compute_history_adjustments(runs)
        # No retry penalty and no pass-rate bonus/penalty → 0.0
        self.assertEqual(adj.get("m/b", 0.0), 0.0)

    def test_high_pass_rate_gives_bonus(self):
        # All runs verified passing → pass rate = 1.0 → +1.0 bonus
        runs = _runs("m/good", MIN_MODEL_RUNS, passed=True)
        adj = compute_history_adjustments(runs)
        self.assertAlmostEqual(adj["m/good"], 1.0)

    def test_low_pass_rate_gives_penalty(self):
        # All verified, all failing → pass rate = 0.0 → -1.0 penalty
        runs = _runs("m/bad", MIN_MODEL_RUNS, passed=False)
        adj = compute_history_adjustments(runs)
        self.assertAlmostEqual(adj["m/bad"], -1.0)

    def test_high_retry_rate_gives_penalty(self):
        # Half the runs triggered retry, none verified → retry penalty only
        n = MIN_MODEL_RUNS
        runs = (
            _runs("m/flaky", n // 2, retry=True)
            + _runs("m/flaky", n - n // 2, retry=False)
        )
        adj = compute_history_adjustments(runs)
        self.assertLess(adj["m/flaky"], 0.0)

    def test_combined_penalties_capped_at_minus_two(self):
        # Bad pass rate + high retry rate → capped at -2.0
        runs = _runs("m/terrible", MIN_MODEL_RUNS, passed=False, retry=True)
        adj = compute_history_adjustments(runs)
        self.assertAlmostEqual(adj["m/terrible"], -2.0)

    def test_bonus_capped_at_plus_one(self):
        # Even if both signals fire positively the cap is +1.0
        runs = _runs("m/great", MIN_MODEL_RUNS, passed=True, retry=False)
        adj = compute_history_adjustments(runs)
        self.assertLessEqual(adj["m/great"], 1.0)

    def test_model_not_in_history_returns_zero_via_dict_get(self):
        runs = _runs("m/known", MIN_MODEL_RUNS, passed=True)
        adj = compute_history_adjustments(runs)
        self.assertEqual(adj.get("m/unknown", 0.0), 0.0)

    def test_unverified_runs_not_counted_toward_verified_threshold(self):
        # MIN_MODEL_RUNS total runs, all unverified → no pass-rate signal, no retry → 0.0
        runs = _runs("m/noverif", MIN_MODEL_RUNS, passed=None, retry=False)
        adj = compute_history_adjustments(runs)
        self.assertEqual(adj.get("m/noverif", 0.0), 0.0)

    def test_multiple_models_computed_independently(self):
        runs = (
            _runs("m/good", MIN_MODEL_RUNS, passed=True)
            + _runs("m/bad", MIN_MODEL_RUNS, passed=False)
        )
        adj = compute_history_adjustments(runs)
        self.assertGreater(adj["m/good"], adj["m/bad"])


if __name__ == "__main__":
    unittest.main()
