from __future__ import annotations

import json
import unittest
from pathlib import Path

from click.testing import CliRunner

from openshard.cli.main import cli
from openshard.history.completeness import (
    FIELD_DEFINITIONS,
    TOTAL_FIELDS,
    evaluate_completeness,
    score_receipt,
)
from openshard.history.shard_contract import build_shard_receipt


def _full_entry(**kwargs) -> dict:
    """A receipt entry that should populate (almost) every scored field."""
    entry = {
        "shard_id": "full01",
        "timestamp": "2026-04-25T10:00:00Z",
        "task": "implement a feature",
        "execution_model": "openrouter/some-model",
        "duration_seconds": 12.5,
        "estimated_cost": 0.0123,
        "files_created": 1,
        "files_updated": 1,
        "files_deleted": 0,
        "files_detail": [{"path": "a.py", "change_type": "create", "summary": ""}],
        "verification_attempted": True,
        "verification_passed": True,
        "file_context": {"files_read": 3, "paths": ["a.py", "b.py"]},
        "context_files_injected_count": 4,
        "context_utilisation_ratio": 0.5,
        "policy_decisions": [{"decision_id": "p1", "decision": "allow"}],
        "approval_receipt": {"granted": True, "reason": "ok"},
        "execution_spans": [{"span_id": "s1", "name": "plan", "duration_ms": 100}],
        "developer_feedback": {"rating": "good"},
    }
    entry.update(kwargs)
    return entry


def _sparse_entry(**kwargs) -> dict:
    """An old/minimal receipt entry missing all newer fields."""
    entry = {"task": "old task", "timestamp": "2025-01-01T00:00:00Z"}
    entry.update(kwargs)
    return entry


def _receipt(entry: dict):
    return build_shard_receipt(entry, index=0)


def _write_runs(entries: list[dict]) -> None:
    log_dir = Path(".openshard")
    log_dir.mkdir(parents=True, exist_ok=True)
    with (log_dir / "runs.jsonl").open("w", encoding="utf-8") as fh:
        for entry in entries:
            fh.write(json.dumps(entry) + "\n")


class TestScoreReceipt(unittest.TestCase):
    def test_full_receipt_scores_high(self):
        result = score_receipt(_receipt(_full_entry()))
        self.assertGreaterEqual(result.score_percent, 90)
        self.assertEqual(len(result.present_fields) + len(result.missing_fields), TOTAL_FIELDS)
        for core in ("task", "timestamp", "shard_id", "execution_model", "status"):
            self.assertIn(core, result.present_fields)

    def test_sparse_receipt_scores_low_with_missing(self):
        result = score_receipt(_receipt(_sparse_entry()))
        self.assertLess(result.score_percent, 50)
        # Newer fields should be reported missing.
        for missing in ("execution_model", "duration", "cost_estimate", "execution_spans",
                        "context_usage", "policy_decisions", "feedback"):
            self.assertIn(missing, result.missing_fields)
        # Core identity from the entry is still present.
        self.assertIn("task", result.present_fields)
        self.assertIn("timestamp", result.present_fields)

    def test_present_and_missing_are_disjoint_and_complete(self):
        result = score_receipt(_receipt(_full_entry()))
        self.assertEqual(
            set(result.present_fields) | set(result.missing_fields),
            {fd.name for fd in FIELD_DEFINITIONS},
        )
        self.assertEqual(
            set(result.present_fields) & set(result.missing_fields), set()
        )


class TestEvaluateCompleteness(unittest.TestCase):
    def test_empty_input_zeroed_report(self):
        report = evaluate_completeness([])
        self.assertEqual(report.runs_checked, 0)
        self.assertEqual(report.average_score_percent, 0)
        self.assertEqual(report.strong_fields, [])
        self.assertEqual(report.weak_fields, [])
        self.assertEqual(report.recommendations, [])
        self.assertEqual(report.receipts, [])

    def test_aggregation_math_and_partitioning(self):
        receipts = [_receipt(_full_entry()), _receipt(_sparse_entry())]
        report = evaluate_completeness(receipts)
        self.assertEqual(report.runs_checked, 2)

        # task present in both -> 100%; cost present in one -> 50%.
        self.assertEqual(report.field_presence["task"]["present"], 2)
        self.assertEqual(report.field_presence["task"]["presence_percent"], 100)
        self.assertEqual(report.field_presence["cost_estimate"]["present"], 1)
        self.assertEqual(report.field_presence["cost_estimate"]["presence_percent"], 50)

        # average is the mean of the two per-receipt percents.
        expected_avg = round(
            (score_receipt(receipts[0]).score_percent
             + score_receipt(receipts[1]).score_percent) / 2
        )
        self.assertEqual(report.average_score_percent, expected_avg)

        # task is strong; a field absent from both is weak.
        self.assertIn("task", report.strong_fields)
        self.assertIn("feedback", report.weak_fields)

    def test_weak_fields_sorted_ascending_then_name(self):
        report = evaluate_completeness([_receipt(_sparse_entry())])
        percents = [report.field_presence[n]["presence_percent"] for n in report.weak_fields]
        self.assertEqual(percents, sorted(percents))

    def test_recommendations_only_for_weak_fields(self):
        report = evaluate_completeness([_receipt(_full_entry())])
        # Full receipt has few/no weak fields; recommendations must not exceed them.
        self.assertLessEqual(len(report.recommendations), len(report.weak_fields))


class TestStatsCompletenessCLI(unittest.TestCase):
    def _invoke(self, entries: list[dict] | None, args: list[str]):
        runner = CliRunner()
        with runner.isolated_filesystem():
            if entries is not None:
                _write_runs(entries)
            return runner.invoke(cli, ["stats", "completeness"] + args)

    def test_no_history_human(self):
        result = self._invoke(None, [])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("No run history", result.output)

    def test_no_history_json_not_found(self):
        result = self._invoke(None, ["--json"])
        self.assertEqual(result.exit_code, 0)
        payload = json.loads(result.output)
        self.assertEqual(payload["status"], "not_found")
        self.assertEqual(payload["command"], "stats completeness")
        self.assertEqual(payload["runs_checked"], 0)

    def test_empty_file_json_not_found(self):
        result = self._invoke([], ["--json"])
        self.assertEqual(result.exit_code, 0)
        payload = json.loads(result.output)
        self.assertEqual(payload["status"], "not_found")

    def test_json_valid_and_complete(self):
        result = self._invoke([_full_entry(), _sparse_entry()], ["--json"])
        self.assertEqual(result.exit_code, 0)
        payload = json.loads(result.output)  # raises if not valid JSON-only
        self.assertEqual(payload["schema_version"], "1")
        self.assertEqual(payload["status"], "ok")
        self.assertEqual(payload["runs_checked"], 2)
        self.assertIn("field_presence", payload)
        self.assertEqual(len(payload["receipts"]), 2)
        for rc in payload["receipts"]:
            self.assertIn("shard_id", rc)
            self.assertIn("score_percent", rc)

    def test_corrupt_line_skipped(self):
        runner = CliRunner()
        with runner.isolated_filesystem():
            log_dir = Path(".openshard")
            log_dir.mkdir(parents=True, exist_ok=True)
            with (log_dir / "runs.jsonl").open("w", encoding="utf-8") as fh:
                fh.write(json.dumps(_full_entry()) + "\n")
                fh.write("{{not valid json}}\n")
                fh.write(json.dumps(_sparse_entry()) + "\n")
            result = runner.invoke(cli, ["stats", "completeness", "--json"])
        self.assertEqual(result.exit_code, 0)
        payload = json.loads(result.output)
        self.assertEqual(payload["runs_checked"], 2)

    def test_limit_restricts_to_most_recent(self):
        entries = [_sparse_entry() for _ in range(5)] + [_full_entry()]
        result = self._invoke(entries, ["--limit", "2", "--json"])
        payload = json.loads(result.output)
        self.assertEqual(payload["runs_checked"], 2)

    def test_human_output_short(self):
        result = self._invoke([_full_entry()], [])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("Shard receipt completeness", result.output)
        self.assertIn("average completeness", result.output)

    def test_no_secret_or_absolute_path_leak(self):
        secret = "sk-ABCD1234567890SECRETVALUE"
        abs_path = "/home/victim/secret/workspace/creds.txt"
        entry = _full_entry(
            workspace_path=abs_path,
            summary=f"used {secret} at {abs_path}",
            files_detail=[{"path": abs_path, "change_type": "update", "summary": secret}],
            secret_scan_result={"findings": [
                {"fingerprint": "fp1", "kind": "api_key", "path": "creds.txt", "line": 3}
            ]},
        )
        for args in ([], ["--json"]):
            result = self._invoke([entry], args)
            self.assertEqual(result.exit_code, 0)
            self.assertNotIn(secret, result.output)
            self.assertNotIn(abs_path, result.output)
            self.assertNotIn("/home/victim", result.output)


if __name__ == "__main__":
    unittest.main()
