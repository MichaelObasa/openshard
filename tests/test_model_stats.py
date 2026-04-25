from __future__ import annotations

import json
import unittest
from pathlib import Path

from click.testing import CliRunner

from openshard.history.metrics import compute_model_stats
from openshard.cli.main import cli


def _run(**kwargs) -> dict:
    defaults = {
        "timestamp": "2026-04-25T10:00:00Z",
        "task": "do something",
        "execution_model": "openrouter/fast-model",
        "duration_seconds": 5.0,
        "verification_passed": None,
        "estimated_cost": 0.01,
        "total_tokens": 1000,
        "retry_triggered": False,
    }
    defaults.update(kwargs)
    return defaults


class TestComputeModelStats(unittest.TestCase):

    def test_empty_returns_empty_dict(self):
        self.assertEqual(compute_model_stats([]), {})

    def test_single_run_basic_fields(self):
        stats = compute_model_stats([_run()])
        self.assertIn("openrouter/fast-model", stats)
        s = stats["openrouter/fast-model"]
        self.assertEqual(s["runs_count"], 1)
        self.assertAlmostEqual(s["avg_cost"], 0.01)
        self.assertAlmostEqual(s["avg_duration"], 5.0)
        self.assertEqual(s["avg_tokens"], 1000)
        self.assertEqual(s["retry_rate"], 0.0)
        self.assertIsNone(s["verification_pass_rate"])
        self.assertEqual(s["last_used_timestamp"], "2026-04-25T10:00:00Z")

    def test_runs_count_per_model(self):
        runs = [
            _run(execution_model="m/a"),
            _run(execution_model="m/a"),
            _run(execution_model="m/b"),
        ]
        stats = compute_model_stats(runs)
        self.assertEqual(stats["m/a"]["runs_count"], 2)
        self.assertEqual(stats["m/b"]["runs_count"], 1)

    def test_sorted_by_runs_count_descending(self):
        runs = [
            _run(execution_model="m/rare"),
            _run(execution_model="m/common"),
            _run(execution_model="m/common"),
        ]
        keys = list(compute_model_stats(runs).keys())
        self.assertEqual(keys[0], "m/common")
        self.assertEqual(keys[1], "m/rare")

    def test_avg_cost_averages_correctly(self):
        runs = [
            _run(execution_model="m/x", estimated_cost=0.02),
            _run(execution_model="m/x", estimated_cost=0.04),
        ]
        s = compute_model_stats(runs)["m/x"]
        self.assertAlmostEqual(s["avg_cost"], 0.03)

    def test_avg_cost_none_when_missing(self):
        entry = _run()
        del entry["estimated_cost"]
        s = compute_model_stats([entry])["openrouter/fast-model"]
        self.assertIsNone(s["avg_cost"])

    def test_avg_cost_partial_missing(self):
        e1 = _run(execution_model="m/x", estimated_cost=0.04)
        e2 = _run(execution_model="m/x")
        del e2["estimated_cost"]
        s = compute_model_stats([e1, e2])["m/x"]
        self.assertAlmostEqual(s["avg_cost"], 0.04)

    def test_avg_duration_averages_correctly(self):
        runs = [
            _run(execution_model="m/x", duration_seconds=4.0),
            _run(execution_model="m/x", duration_seconds=8.0),
        ]
        s = compute_model_stats(runs)["m/x"]
        self.assertAlmostEqual(s["avg_duration"], 6.0)

    def test_avg_duration_none_when_missing(self):
        entry = _run()
        del entry["duration_seconds"]
        s = compute_model_stats([entry])["openrouter/fast-model"]
        self.assertIsNone(s["avg_duration"])

    def test_avg_tokens_rounds(self):
        runs = [
            _run(execution_model="m/x", total_tokens=999),
            _run(execution_model="m/x", total_tokens=1001),
        ]
        s = compute_model_stats(runs)["m/x"]
        self.assertEqual(s["avg_tokens"], 1000)

    def test_avg_tokens_none_when_missing(self):
        entry = _run()
        del entry["total_tokens"]
        s = compute_model_stats([entry])["openrouter/fast-model"]
        self.assertIsNone(s["avg_tokens"])

    def test_verification_pass_rate_correct(self):
        runs = [
            _run(execution_model="m/x", verification_passed=True),
            _run(execution_model="m/x", verification_passed=True),
            _run(execution_model="m/x", verification_passed=False),
        ]
        s = compute_model_stats(runs)["m/x"]
        self.assertAlmostEqual(s["verification_pass_rate"], 2 / 3)

    def test_verification_pass_rate_none_when_all_null(self):
        runs = [_run(verification_passed=None), _run(verification_passed=None)]
        s = compute_model_stats(runs)["openrouter/fast-model"]
        self.assertIsNone(s["verification_pass_rate"])

    def test_verification_pass_rate_ignores_nulls(self):
        runs = [
            _run(execution_model="m/x", verification_passed=True),
            _run(execution_model="m/x", verification_passed=None),
        ]
        s = compute_model_stats(runs)["m/x"]
        self.assertAlmostEqual(s["verification_pass_rate"], 1.0)

    def test_retry_rate_correct(self):
        runs = [
            _run(execution_model="m/x", retry_triggered=True),
            _run(execution_model="m/x", retry_triggered=False),
            _run(execution_model="m/x", retry_triggered=False),
            _run(execution_model="m/x", retry_triggered=False),
        ]
        s = compute_model_stats(runs)["m/x"]
        self.assertAlmostEqual(s["retry_rate"], 0.25)

    def test_retry_rate_zero_when_field_missing(self):
        entry = _run()
        del entry["retry_triggered"]
        s = compute_model_stats([entry])["openrouter/fast-model"]
        self.assertAlmostEqual(s["retry_rate"], 0.0)

    def test_last_used_picks_latest_timestamp(self):
        runs = [
            _run(execution_model="m/x", timestamp="2026-04-23T08:00:00Z"),
            _run(execution_model="m/x", timestamp="2026-04-25T10:00:00Z"),
            _run(execution_model="m/x", timestamp="2026-04-24T09:00:00Z"),
        ]
        s = compute_model_stats(runs)["m/x"]
        self.assertEqual(s["last_used_timestamp"], "2026-04-25T10:00:00Z")

    def test_last_used_none_when_no_timestamps(self):
        entry = _run()
        del entry["timestamp"]
        s = compute_model_stats([entry])["openrouter/fast-model"]
        self.assertIsNone(s["last_used_timestamp"])

    def test_run_without_execution_model_skipped(self):
        entry = _run()
        del entry["execution_model"]
        stats = compute_model_stats([entry])
        self.assertEqual(stats, {})

    def test_malformed_run_missing_all_optional_fields(self):
        entry = {"execution_model": "m/minimal", "timestamp": "2026-01-01T00:00:00Z"}
        s = compute_model_stats([entry])["m/minimal"]
        self.assertEqual(s["runs_count"], 1)
        self.assertIsNone(s["avg_cost"])
        self.assertIsNone(s["avg_duration"])
        self.assertIsNone(s["avg_tokens"])
        self.assertIsNone(s["verification_pass_rate"])
        self.assertAlmostEqual(s["retry_rate"], 0.0)


class TestModelStatsCLI(unittest.TestCase):

    def _invoke(self, entries: list[dict] | None) -> str:
        runner = CliRunner()
        with runner.isolated_filesystem():
            if entries is not None:
                Path(".openshard").mkdir()
                with open(".openshard/runs.jsonl", "w") as f:
                    for e in entries:
                        f.write(json.dumps(e) + "\n")
            result = runner.invoke(cli, ["models", "stats"])
        return result.output

    def test_no_file_shows_message(self):
        out = self._invoke(None)
        self.assertIn("No run history", out)

    def test_empty_file_shows_message(self):
        out = self._invoke([])
        self.assertIn("No runs recorded", out)

    def test_output_has_header(self):
        out = self._invoke([_run()])
        self.assertIn("[model stats]", out)
        self.assertIn("model", out)
        self.assertIn("runs", out)

    def test_output_shows_model_id(self):
        out = self._invoke([_run(execution_model="openrouter/my-model")])
        self.assertIn("openrouter/my-model", out)

    def test_output_shows_run_count(self):
        out = self._invoke([_run(), _run()])
        self.assertIn("2", out)

    def test_output_shows_cost_with_dollar_sign(self):
        out = self._invoke([_run(estimated_cost=0.0123)])
        self.assertIn("$", out)

    def test_output_shows_dash_when_no_cost(self):
        entry = _run()
        del entry["estimated_cost"]
        out = self._invoke([entry])
        self.assertIn("-", out)

    def test_at_most_ten_models_shown(self):
        runs = [_run(execution_model=f"m/model-{i}") for i in range(15)]
        out = self._invoke(runs)
        model_lines = [ln for ln in out.splitlines() if "m/model-" in ln]
        self.assertLessEqual(len(model_lines), 10)

    def test_old_entries_without_optional_fields(self):
        old = {"task": "old", "timestamp": "2025-01-01T00:00:00Z", "execution_model": "m/old"}
        out = self._invoke([old])
        self.assertIn("[model stats]", out)
        self.assertIn("m/old", out)

    def test_malformed_jsonl_line_skipped(self):
        runner = CliRunner()
        with runner.isolated_filesystem():
            Path(".openshard").mkdir()
            with open(".openshard/runs.jsonl", "w") as f:
                f.write("not json\n")
                f.write(json.dumps(_run()) + "\n")
            result = runner.invoke(cli, ["models", "stats"])
        self.assertIn("[model stats]", result.output)


if __name__ == "__main__":
    unittest.main()
