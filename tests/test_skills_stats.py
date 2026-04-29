from __future__ import annotations

import unittest
from pathlib import Path

from click.testing import CliRunner

from openshard.cli.main import cli
from openshard.history.metrics import compute_skill_stats


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run(**kwargs) -> dict:
    defaults: dict = {
        "task": "do something",
        "execution_model": "openrouter/fast-model",
    }
    defaults.update(kwargs)
    return defaults


# ---------------------------------------------------------------------------
# compute_skill_stats unit tests
# ---------------------------------------------------------------------------

class TestComputeSkillStats(unittest.TestCase):

    def test_empty_runs_returns_empty_dict(self):
        assert compute_skill_stats([]) == {}

    def test_runs_without_matched_skills_ignored(self):
        runs = [_run(), _run(matched_skills=[]), _run(matched_skills=None)]
        assert compute_skill_stats(runs) == {}

    def test_single_skill_counted(self):
        runs = [_run(matched_skills=["lint-check"])]
        stats = compute_skill_stats(runs)
        assert "lint-check" in stats
        assert stats["lint-check"]["runs_count"] == 1

    def test_multiple_runs_same_skill(self):
        runs = [_run(matched_skills=["lint-check"])] * 3
        stats = compute_skill_stats(runs)
        assert stats["lint-check"]["runs_count"] == 3

    def test_one_run_multiple_skills_contributes_to_each(self):
        runs = [_run(matched_skills=["skill-a", "skill-b"])]
        stats = compute_skill_stats(runs)
        assert stats["skill-a"]["runs_count"] == 1
        assert stats["skill-b"]["runs_count"] == 1

    def test_avg_cost_computed(self):
        runs = [
            _run(matched_skills=["s"], estimated_cost=0.10),
            _run(matched_skills=["s"], estimated_cost=0.20),
        ]
        stats = compute_skill_stats(runs)
        self.assertAlmostEqual(stats["s"]["avg_cost"], 0.15)

    def test_avg_cost_none_when_no_cost_data(self):
        runs = [_run(matched_skills=["s"])]
        stats = compute_skill_stats(runs)
        assert stats["s"]["avg_cost"] is None

    def test_avg_duration_computed(self):
        runs = [
            _run(matched_skills=["s"], duration_seconds=10.0),
            _run(matched_skills=["s"], duration_seconds=20.0),
        ]
        stats = compute_skill_stats(runs)
        self.assertAlmostEqual(stats["s"]["avg_duration"], 15.0)

    def test_avg_duration_none_when_no_duration_data(self):
        runs = [_run(matched_skills=["s"])]
        stats = compute_skill_stats(runs)
        assert stats["s"]["avg_duration"] is None

    def test_verification_pass_rate_all_pass(self):
        runs = [_run(matched_skills=["s"], verification_passed=True)] * 4
        stats = compute_skill_stats(runs)
        self.assertAlmostEqual(stats["s"]["verification_pass_rate"], 1.0)

    def test_verification_pass_rate_mixed(self):
        runs = [
            _run(matched_skills=["s"], verification_passed=True),
            _run(matched_skills=["s"], verification_passed=False),
        ]
        stats = compute_skill_stats(runs)
        self.assertAlmostEqual(stats["s"]["verification_pass_rate"], 0.5)

    def test_verification_pass_rate_none_when_no_verification_data(self):
        runs = [_run(matched_skills=["s"])]
        stats = compute_skill_stats(runs)
        assert stats["s"]["verification_pass_rate"] is None

    def test_retry_rate_computed(self):
        runs = [
            _run(matched_skills=["s"], retry_triggered=True),
            _run(matched_skills=["s"], retry_triggered=False),
            _run(matched_skills=["s"]),
        ]
        stats = compute_skill_stats(runs)
        self.assertAlmostEqual(stats["s"]["retry_rate"], 1 / 3)

    def test_retry_rate_zero_when_no_retries(self):
        runs = [_run(matched_skills=["s"])] * 2
        stats = compute_skill_stats(runs)
        self.assertAlmostEqual(stats["s"]["retry_rate"], 0.0)

    def test_sorted_by_runs_count_descending(self):
        runs = (
            [_run(matched_skills=["popular"])] * 5
            + [_run(matched_skills=["rare"])] * 1
        )
        stats = compute_skill_stats(runs)
        slugs = list(stats.keys())
        assert slugs[0] == "popular"
        assert slugs[1] == "rare"

    def test_unknown_verification_value_ignored(self):
        runs = [_run(matched_skills=["s"], verification_passed=None)]
        stats = compute_skill_stats(runs)
        assert stats["s"]["verification_pass_rate"] is None


# ---------------------------------------------------------------------------
# CLI integration tests
# ---------------------------------------------------------------------------

class TestSkillsStatsCLI(unittest.TestCase):

    def _invoke(self, entries: list[dict] | None) -> str:
        runner = CliRunner()
        with runner.isolated_filesystem():
            if entries is not None:
                Path(".openshard").mkdir()
                with open(".openshard/runs.jsonl", "w") as f:
                    for e in entries:
                        f.write(__import__("json").dumps(e) + "\n")
            result = runner.invoke(cli, ["skills", "stats"])
        return result.output

    def test_no_file_shows_message(self):
        out = self._invoke(None)
        self.assertIn("No run history", out)

    def test_empty_file_shows_message(self):
        out = self._invoke([])
        self.assertIn("No runs recorded", out)

    def test_no_skill_data_shows_message(self):
        out = self._invoke([_run(), _run(matched_skills=[])])
        self.assertIn("No skill data in run history.", out)

    def test_header_line_shown(self):
        out = self._invoke([_run(matched_skills=["my-skill"])])
        self.assertIn("[skill stats]", out)

    def test_skill_slug_shown_in_output(self):
        out = self._invoke([_run(matched_skills=["my-skill"])])
        self.assertIn("my-skill", out)

    def test_run_count_shown(self):
        out = self._invoke([_run(matched_skills=["s"])] * 3)
        self.assertIn("3", out)

    def test_multiple_skills_each_shown(self):
        out = self._invoke([
            _run(matched_skills=["alpha"]),
            _run(matched_skills=["beta"]),
        ])
        self.assertIn("alpha", out)
        self.assertIn("beta", out)

    def test_pass_rate_shown_when_verification_present(self):
        out = self._invoke([_run(matched_skills=["s"], verification_passed=True)] * 2)
        self.assertIn("100%", out)

    def test_dash_shown_when_no_verification_data(self):
        out = self._invoke([_run(matched_skills=["s"])])
        self.assertIn("-", out)

    def test_cost_shown_with_dollar_sign(self):
        out = self._invoke([_run(matched_skills=["s"], estimated_cost=0.0123)])
        self.assertIn("$", out)

    def test_malformed_jsonl_line_skipped(self):
        runner = CliRunner()
        with runner.isolated_filesystem():
            Path(".openshard").mkdir()
            with open(".openshard/runs.jsonl", "w") as f:
                f.write("not json\n")
                f.write(__import__("json").dumps(_run(matched_skills=["s"])) + "\n")
            result = runner.invoke(cli, ["skills", "stats"])
        self.assertIn("[skill stats]", result.output)

    def test_skills_group_help_accessible(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["skills", "--help"])
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertIn("stats", result.output, result.output)


if __name__ == "__main__":
    unittest.main()
