from __future__ import annotations

import json
import unittest
from pathlib import Path

import pytest
from click.testing import CliRunner

from openshard.analysis.repo import RepoFacts
from openshard.cli.main import cli
from openshard.history.metrics import ALL_PROFILES, compute_profile_stats
from openshard.routing.profiles import (
    ProfileDecision,
    ProfileHistorySummary,
    build_profile_history_summary,
    select_profile,
)


def _facts(risky_paths: list[str] | None = None) -> RepoFacts:
    return RepoFacts(
        languages=["python"],
        package_files=[],
        framework=None,
        test_command=None,
        risky_paths=risky_paths or [],
        changed_files=[],
    )


def _short_task() -> str:
    return "fix typo in README"


def _long_task() -> str:
    return " ".join(["word"] * 81)


# ---------------------------------------------------------------------------
# native_light — simple/safe tasks
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("category", ["standard", "visual", "boilerplate"])
def test_light_for_simple_categories(category):
    result = select_profile(category, _facts(), _short_task(), None)
    assert result.profile == "native_light"
    assert result.reason == "simple/safe task"


def test_light_no_repo_facts():
    result = select_profile("standard", None, _short_task(), None)
    assert result.profile == "native_light"


# ---------------------------------------------------------------------------
# native_deep — serious categories
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("category", ["security", "complex"])
def test_deep_for_serious_categories(category):
    result = select_profile(category, _facts(), _short_task(), None)
    assert result.profile == "native_deep"
    assert category in result.reason


# ---------------------------------------------------------------------------
# native_deep — risky paths
# ---------------------------------------------------------------------------

def test_deep_when_risky_paths_present():
    result = select_profile("standard", _facts(["auth/tokens.py"]), _short_task(), None)
    assert result.profile == "native_deep"
    assert "risky paths" in result.reason


def test_light_when_no_risky_paths():
    result = select_profile("standard", _facts([]), _short_task(), None)
    assert result.profile == "native_light"


# ---------------------------------------------------------------------------
# native_deep — long task
# ---------------------------------------------------------------------------

def test_deep_when_task_is_long():
    result = select_profile("standard", _facts(), _long_task(), None)
    assert result.profile == "native_deep"
    assert "long task" in result.reason


def test_light_when_task_exactly_at_boundary():
    task = " ".join(["word"] * 80)  # exactly 80 words — not > 80
    result = select_profile("standard", _facts(), task, None)
    assert result.profile == "native_light"


# ---------------------------------------------------------------------------
# Multiple signals — reason concatenates
# ---------------------------------------------------------------------------

def test_reason_concatenates_multiple_signals():
    result = select_profile("security", _facts(["auth/login.py"]), _short_task(), None)
    assert result.profile == "native_deep"
    assert "security category" in result.reason
    assert "risky paths" in result.reason


def test_reason_all_three_signals():
    result = select_profile("security", _facts(["auth/login.py"]), _long_task(), None)
    assert "security category" in result.reason
    assert "risky paths" in result.reason
    assert "long task" in result.reason


# ---------------------------------------------------------------------------
# Explicit overrides
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("override", ["native_light", "native_deep", "native_swarm"])
def test_override_wins_for_all_profiles(override):
    result = select_profile("security", _facts(["auth/x.py"]), _long_task(), override)
    assert result.profile == override
    assert result.reason == "explicit override"


def test_override_wins_even_for_simple_task():
    result = select_profile("standard", _facts(), _short_task(), "native_swarm")
    assert result.profile == "native_swarm"
    assert result.reason == "explicit override"


def test_override_is_case_insensitive():
    result = select_profile("standard", _facts(), _short_task(), "NATIVE_DEEP")
    assert result.profile == "native_deep"


def test_invalid_override_raises():
    import pytest
    with pytest.raises(ValueError, match="Invalid profile"):
        select_profile("standard", _facts(), _short_task(), "turbo_agent")


# ---------------------------------------------------------------------------
# native_swarm never auto-selected
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("category", ["security", "complex", "standard", "visual", "boilerplate"])
def test_swarm_never_auto_selected(category):
    result = select_profile(category, _facts(["auth/x.py"]), _long_task(), None)
    assert result.profile != "native_swarm"


# ---------------------------------------------------------------------------
# Return type
# ---------------------------------------------------------------------------

def test_returns_profile_decision_dataclass():
    result = select_profile("standard", _facts(), _short_task(), None)
    assert isinstance(result, ProfileDecision)
    assert isinstance(result.profile, str)
    assert isinstance(result.reason, str)


# ---------------------------------------------------------------------------
# compute_profile_stats — unit tests
# ---------------------------------------------------------------------------

def _prun(**kwargs) -> dict:
    defaults = {
        "timestamp": "2026-04-25T10:00:00Z",
        "task": "do something",
        "execution_profile": "native_light",
        "execution_model": "openrouter/fast-model",
        "duration_seconds": 5.0,
        "verification_passed": None,
        "estimated_cost": 0.01,
        "retry_triggered": False,
    }
    defaults.update(kwargs)
    return defaults


class TestComputeProfileStats(unittest.TestCase):

    def test_all_profiles_always_present(self):
        stats = compute_profile_stats([])
        for p in ALL_PROFILES:
            self.assertIn(p, stats)

    def test_empty_input_all_zero_counts(self):
        stats = compute_profile_stats([])
        for p in ALL_PROFILES:
            self.assertEqual(stats[p]["runs_count"], 0)

    def test_empty_input_sensible_none_values(self):
        stats = compute_profile_stats([])
        for p in ALL_PROFILES:
            self.assertIsNone(stats[p]["avg_cost"])
            self.assertIsNone(stats[p]["avg_duration"])
            self.assertIsNone(stats[p]["verification_pass_rate"])
            self.assertIsNone(stats[p]["retry_rate"])

    def test_run_without_profile_ignored(self):
        entry = _prun()
        del entry["execution_profile"]
        stats = compute_profile_stats([entry])
        for p in ALL_PROFILES:
            self.assertEqual(stats[p]["runs_count"], 0)

    def test_unknown_profile_ignored(self):
        entry = _prun(execution_profile="turbo_agent")
        stats = compute_profile_stats([entry])
        for p in ALL_PROFILES:
            self.assertEqual(stats[p]["runs_count"], 0)

    def test_runs_counted_per_profile(self):
        runs = [
            _prun(execution_profile="native_light"),
            _prun(execution_profile="native_light"),
            _prun(execution_profile="native_deep"),
        ]
        stats = compute_profile_stats(runs)
        self.assertEqual(stats["native_light"]["runs_count"], 2)
        self.assertEqual(stats["native_deep"]["runs_count"], 1)
        self.assertEqual(stats["native_swarm"]["runs_count"], 0)

    def test_avg_cost_correct(self):
        runs = [
            _prun(execution_profile="native_deep", estimated_cost=0.02),
            _prun(execution_profile="native_deep", estimated_cost=0.04),
        ]
        self.assertAlmostEqual(compute_profile_stats(runs)["native_deep"]["avg_cost"], 0.03)

    def test_avg_cost_none_when_missing(self):
        entry = _prun(execution_profile="native_light")
        del entry["estimated_cost"]
        self.assertIsNone(compute_profile_stats([entry])["native_light"]["avg_cost"])

    def test_avg_cost_partial_missing(self):
        e1 = _prun(execution_profile="native_light", estimated_cost=0.04)
        e2 = _prun(execution_profile="native_light")
        del e2["estimated_cost"]
        self.assertAlmostEqual(compute_profile_stats([e1, e2])["native_light"]["avg_cost"], 0.04)

    def test_avg_duration_correct(self):
        runs = [
            _prun(execution_profile="native_light", duration_seconds=4.0),
            _prun(execution_profile="native_light", duration_seconds=8.0),
        ]
        self.assertAlmostEqual(compute_profile_stats(runs)["native_light"]["avg_duration"], 6.0)

    def test_avg_duration_none_when_missing(self):
        entry = _prun(execution_profile="native_light")
        del entry["duration_seconds"]
        self.assertIsNone(compute_profile_stats([entry])["native_light"]["avg_duration"])

    def test_verification_pass_rate_correct(self):
        runs = [
            _prun(execution_profile="native_deep", verification_passed=True),
            _prun(execution_profile="native_deep", verification_passed=True),
            _prun(execution_profile="native_deep", verification_passed=False),
        ]
        self.assertAlmostEqual(
            compute_profile_stats(runs)["native_deep"]["verification_pass_rate"], 2 / 3
        )

    def test_verification_pass_rate_none_when_all_null(self):
        runs = [_prun(verification_passed=None), _prun(verification_passed=None)]
        self.assertIsNone(compute_profile_stats(runs)["native_light"]["verification_pass_rate"])

    def test_verification_pass_rate_ignores_nulls(self):
        runs = [
            _prun(execution_profile="native_light", verification_passed=True),
            _prun(execution_profile="native_light", verification_passed=None),
        ]
        self.assertAlmostEqual(
            compute_profile_stats(runs)["native_light"]["verification_pass_rate"], 1.0
        )

    def test_retry_rate_correct(self):
        runs = [
            _prun(execution_profile="native_swarm", retry_triggered=True),
            _prun(execution_profile="native_swarm", retry_triggered=False),
            _prun(execution_profile="native_swarm", retry_triggered=False),
            _prun(execution_profile="native_swarm", retry_triggered=False),
        ]
        self.assertAlmostEqual(compute_profile_stats(runs)["native_swarm"]["retry_rate"], 0.25)

    def test_retry_rate_none_when_no_runs(self):
        self.assertIsNone(compute_profile_stats([])["native_swarm"]["retry_rate"])

    def test_retry_rate_zero_when_none_triggered(self):
        runs = [_prun(execution_profile="native_light", retry_triggered=False)]
        self.assertAlmostEqual(compute_profile_stats(runs)["native_light"]["retry_rate"], 0.0)

    def test_profiles_isolated_from_each_other(self):
        runs = [
            _prun(execution_profile="native_light", estimated_cost=0.01),
            _prun(execution_profile="native_deep", estimated_cost=0.99),
        ]
        stats = compute_profile_stats(runs)
        self.assertAlmostEqual(stats["native_light"]["avg_cost"], 0.01)
        self.assertAlmostEqual(stats["native_deep"]["avg_cost"], 0.99)
        self.assertIsNone(stats["native_swarm"]["avg_cost"])


# ---------------------------------------------------------------------------
# profiles stats CLI — output tests
# ---------------------------------------------------------------------------

class TestProfilesStatsCLI(unittest.TestCase):

    def _invoke(self, entries: list[dict] | None) -> str:
        runner = CliRunner()
        with runner.isolated_filesystem():
            if entries is not None:
                Path(".openshard").mkdir()
                with open(".openshard/runs.jsonl", "w") as f:
                    for e in entries:
                        f.write(json.dumps(e) + "\n")
            result = runner.invoke(cli, ["profiles", "stats"])
        return result.output

    def test_no_file_shows_message(self):
        out = self._invoke(None)
        self.assertIn("No run history", out)

    def test_empty_file_shows_message(self):
        out = self._invoke([])
        self.assertIn("No runs recorded", out)

    def test_output_has_header(self):
        out = self._invoke([_prun()])
        self.assertIn("[profile stats]", out)
        self.assertIn("profile", out)
        self.assertIn("runs", out)

    def test_all_three_profiles_shown(self):
        out = self._invoke([_prun(execution_profile="native_light")])
        self.assertIn("native_light", out)
        self.assertIn("native_deep", out)
        self.assertIn("native_swarm", out)

    def test_run_count_shown(self):
        runs = [_prun(execution_profile="native_light")] * 3
        out = self._invoke(runs)
        self.assertIn("3", out)

    def test_cost_shown_with_dollar_sign(self):
        out = self._invoke([_prun(execution_profile="native_deep", estimated_cost=0.0456)])
        self.assertIn("$", out)

    def test_dash_shown_when_no_cost(self):
        entry = _prun(execution_profile="native_light")
        del entry["estimated_cost"]
        out = self._invoke([entry])
        self.assertIn("-", out)

    def test_zero_run_profiles_show_dashes(self):
        out = self._invoke([_prun(execution_profile="native_light")])
        lines = [ln for ln in out.splitlines() if "native_swarm" in ln]
        self.assertEqual(len(lines), 1)
        self.assertIn("-", lines[0])

    def test_runs_without_profile_excluded_from_count(self):
        entry_no_profile = _prun()
        del entry_no_profile["execution_profile"]
        out = self._invoke([entry_no_profile, _prun(execution_profile="native_deep")])
        self.assertIn("1 run with profile data", out)

    def test_malformed_jsonl_line_skipped(self):
        runner = CliRunner()
        with runner.isolated_filesystem():
            Path(".openshard").mkdir()
            with open(".openshard/runs.jsonl", "w") as f:
                f.write("not json\n")
                f.write(json.dumps(_prun()) + "\n")
            result = runner.invoke(cli, ["profiles", "stats"])
        self.assertIn("[profile stats]", result.output)


# ---------------------------------------------------------------------------
# build_profile_history_summary
# ---------------------------------------------------------------------------

class TestBuildProfileHistorySummary(unittest.TestCase):

    def test_all_profiles_returned(self):
        summary = build_profile_history_summary([])
        self.assertIn("native_light", summary)
        self.assertIn("native_deep", summary)
        self.assertIn("native_swarm", summary)

    def test_returns_profile_history_summary_instances(self):
        summary = build_profile_history_summary([])
        for s in summary.values():
            self.assertIsInstance(s, ProfileHistorySummary)

    def test_empty_history_zero_runs(self):
        summary = build_profile_history_summary([])
        for s in summary.values():
            self.assertEqual(s.runs, 0)

    def test_empty_history_none_rates(self):
        summary = build_profile_history_summary([])
        for s in summary.values():
            self.assertIsNone(s.verification_pass_rate)
            self.assertIsNone(s.retry_rate)
            self.assertIsNone(s.avg_cost)
            self.assertIsNone(s.avg_duration)

    def test_runs_counted_correctly(self):
        runs = [
            _prun(execution_profile="native_light"),
            _prun(execution_profile="native_light"),
            _prun(execution_profile="native_deep"),
        ]
        summary = build_profile_history_summary(runs)
        self.assertEqual(summary["native_light"].runs, 2)
        self.assertEqual(summary["native_deep"].runs, 1)
        self.assertEqual(summary["native_swarm"].runs, 0)

    def test_pass_rate_copied_correctly(self):
        runs = [
            _prun(execution_profile="native_deep", verification_passed=True),
            _prun(execution_profile="native_deep", verification_passed=True),
            _prun(execution_profile="native_deep", verification_passed=False),
        ]
        summary = build_profile_history_summary(runs)
        self.assertAlmostEqual(summary["native_deep"].verification_pass_rate, 2 / 3)

    def test_retry_rate_copied_correctly(self):
        runs = [
            _prun(execution_profile="native_light", retry_triggered=True),
            _prun(execution_profile="native_light", retry_triggered=False),
        ]
        summary = build_profile_history_summary(runs)
        self.assertAlmostEqual(summary["native_light"].retry_rate, 0.5)

    def test_avg_cost_copied_correctly(self):
        runs = [
            _prun(execution_profile="native_swarm", estimated_cost=0.10),
            _prun(execution_profile="native_swarm", estimated_cost=0.20),
        ]
        summary = build_profile_history_summary(runs)
        self.assertAlmostEqual(summary["native_swarm"].avg_cost, 0.15)

    def test_avg_duration_copied_correctly(self):
        runs = [
            _prun(execution_profile="native_light", duration_seconds=4.0),
            _prun(execution_profile="native_light", duration_seconds=8.0),
        ]
        summary = build_profile_history_summary(runs)
        self.assertAlmostEqual(summary["native_light"].avg_duration, 6.0)

    def test_runs_without_profile_ignored(self):
        entry = _prun()
        del entry["execution_profile"]
        summary = build_profile_history_summary([entry])
        for s in summary.values():
            self.assertEqual(s.runs, 0)

    def test_profile_field_matches_key(self):
        summary = build_profile_history_summary([_prun(execution_profile="native_deep")])
        for key, s in summary.items():
            self.assertEqual(s.profile, key)


if __name__ == "__main__":
    unittest.main()
