from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from click.testing import CliRunner

from openshard.cli.main import _compute_metrics, cli


def _entry(**kwargs) -> dict:
    defaults = {
        "timestamp": "2026-04-25T10:00:00Z",
        "task": "do something",
        "execution_model": "openrouter/fast-model",
        "duration_seconds": 5.0,
        "verification_passed": None,
        "estimated_cost": 0.01,
    }
    defaults.update(kwargs)
    return defaults


class TestComputeMetrics(unittest.TestCase):

    def test_empty_returns_zeroed_structure(self):
        m = _compute_metrics([])
        self.assertEqual(m["total_runs"], 0)
        self.assertIsNone(m["total_cost"])
        self.assertIsNone(m["avg_cost"])
        self.assertEqual(m["avg_duration"], 0.0)
        self.assertIsNone(m["most_recent"])
        self.assertEqual(m["models"], {})
        self.assertEqual(m["categories"], {})
        self.assertEqual(m["verification"], {"passed": 0, "failed": 0, "unknown": 0})

    def test_total_runs(self):
        entries = [_entry(), _entry(), _entry()]
        m = _compute_metrics(entries)
        self.assertEqual(m["total_runs"], 3)

    def test_total_and_avg_cost(self):
        entries = [_entry(estimated_cost=0.01), _entry(estimated_cost=0.03)]
        m = _compute_metrics(entries)
        self.assertAlmostEqual(m["total_cost"], 0.04)
        self.assertAlmostEqual(m["avg_cost"], 0.02)

    def test_cost_none_when_all_missing(self):
        entries = [_entry(estimated_cost=None), _entry(estimated_cost=None)]
        # remove the key entirely to simulate old entries
        for e in entries:
            del e["estimated_cost"]
        m = _compute_metrics(entries)
        self.assertIsNone(m["total_cost"])
        self.assertIsNone(m["avg_cost"])

    def test_cost_partial_missing_counts_only_present(self):
        e1 = _entry(estimated_cost=0.04)
        e2 = _entry()
        del e2["estimated_cost"]
        m = _compute_metrics([e1, e2])
        self.assertAlmostEqual(m["total_cost"], 0.04)
        self.assertAlmostEqual(m["avg_cost"], 0.04)

    def test_avg_duration(self):
        entries = [_entry(duration_seconds=4.0), _entry(duration_seconds=8.0)]
        m = _compute_metrics(entries)
        self.assertAlmostEqual(m["avg_duration"], 6.0)

    def test_avg_duration_missing_field(self):
        e = _entry()
        del e["duration_seconds"]
        m = _compute_metrics([e])
        self.assertEqual(m["avg_duration"], 0.0)

    def test_most_recent_picks_latest(self):
        entries = [
            _entry(timestamp="2026-04-23T08:00:00Z"),
            _entry(timestamp="2026-04-25T10:00:00Z"),
            _entry(timestamp="2026-04-24T09:00:00Z"),
        ]
        m = _compute_metrics(entries)
        self.assertIn("2026-04-25", m["most_recent"])
        self.assertIn("UTC", m["most_recent"])

    def test_most_recent_none_when_no_timestamps(self):
        e = _entry()
        del e["timestamp"]
        m = _compute_metrics([e])
        self.assertIsNone(m["most_recent"])

    def test_models_counted(self):
        entries = [
            _entry(execution_model="openrouter/fast"),
            _entry(execution_model="openrouter/fast"),
            _entry(execution_model="openrouter/strong"),
        ]
        m = _compute_metrics(entries)
        self.assertEqual(m["models"]["openrouter/fast"], 2)
        self.assertEqual(m["models"]["openrouter/strong"], 1)

    def test_models_sorted_by_count(self):
        entries = [
            _entry(execution_model="openrouter/rare"),
            _entry(execution_model="openrouter/common"),
            _entry(execution_model="openrouter/common"),
        ]
        m = _compute_metrics(entries)
        keys = list(m["models"].keys())
        self.assertEqual(keys[0], "openrouter/common")
        self.assertEqual(keys[1], "openrouter/rare")

    def test_models_missing_field_skipped(self):
        e = _entry()
        del e["execution_model"]
        m = _compute_metrics([e])
        self.assertEqual(m["models"], {})

    def test_categories_counted(self):
        entries = [
            _entry(routing_category="security"),
            _entry(routing_category="standard"),
            _entry(routing_category="security"),
        ]
        m = _compute_metrics(entries)
        self.assertEqual(m["categories"]["security"], 2)
        self.assertEqual(m["categories"]["standard"], 1)

    def test_categories_absent_field_skipped(self):
        e = _entry()
        m = _compute_metrics([e])
        self.assertEqual(m["categories"], {})

    def test_verification_counts_passed_failed_unknown(self):
        entries = [
            _entry(verification_passed=True),
            _entry(verification_passed=True),
            _entry(verification_passed=False),
            _entry(verification_passed=None),
            _entry(),  # key present but None
        ]
        m = _compute_metrics(entries)
        self.assertEqual(m["verification"]["passed"], 2)
        self.assertEqual(m["verification"]["failed"], 1)
        self.assertEqual(m["verification"]["unknown"], 2)

    def test_verification_missing_key_counts_as_unknown(self):
        e = _entry()
        del e["verification_passed"]
        m = _compute_metrics([e])
        self.assertEqual(m["verification"]["unknown"], 1)

    def test_verification_counts_sum_to_total(self):
        entries = [_entry(verification_passed=True), _entry(verification_passed=False), _entry()]
        m = _compute_metrics(entries)
        v = m["verification"]
        self.assertEqual(v["passed"] + v["failed"] + v["unknown"], m["total_runs"])


class TestMetricsCLI(unittest.TestCase):

    def _run(self, entries: list[dict] | None) -> str:
        runner = CliRunner()
        with runner.isolated_filesystem():
            if entries is not None:
                Path(".openshard").mkdir()
                with open(".openshard/runs.jsonl", "w") as f:
                    for e in entries:
                        f.write(json.dumps(e) + "\n")
            result = runner.invoke(cli, ["metrics"])
        return result.output

    def test_no_file_message(self):
        out = self._run(None)
        self.assertIn("No run history", out)

    def test_empty_file_message(self):
        out = self._run([])
        self.assertIn("No runs recorded", out)

    def test_basic_output_sections(self):
        entries = [
            _entry(
                execution_model="anthropic/claude-sonnet-4.6",
                routing_category="standard",
                verification_passed=True,
                estimated_cost=0.05,
                duration_seconds=8.0,
            )
        ]
        out = self._run(entries)
        self.assertIn("[metrics]", out)
        self.assertIn("runs:", out)
        self.assertIn("total cost:", out)
        self.assertIn("avg cost/run:", out)
        self.assertIn("avg duration:", out)
        self.assertIn("most recent:", out)
        self.assertIn("models", out)
        self.assertIn("categories", out)
        self.assertIn("verification", out)

    def test_output_shows_run_count(self):
        entries = [_entry(), _entry()]
        out = self._run(entries)
        self.assertIn("2", out)

    def test_no_categories_section_when_absent(self):
        e = _entry()
        out = self._run([e])
        self.assertNotIn("categories", out)

    def test_cost_shown_as_dollars(self):
        out = self._run([_entry(estimated_cost=0.0123)])
        self.assertIn("$", out)

    def test_dash_when_no_cost(self):
        e = _entry()
        del e["estimated_cost"]
        out = self._run([e])
        self.assertIn("-", out)

    def test_old_entries_without_new_fields(self):
        old_entry = {"task": "old task", "timestamp": "2025-01-01T00:00:00Z"}
        out = self._run([old_entry])
        self.assertIn("[metrics]", out)
        self.assertIn("1", out)


if __name__ == "__main__":
    unittest.main()
