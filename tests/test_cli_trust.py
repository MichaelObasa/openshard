"""Tests for the `openshard trust last` CLI command and `last --json` trust block."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from click.testing import CliRunner

from openshard.cli.main import cli

_GOOD_ENTRY = {
    "schema_version": "1.1",
    "task": "Add a helper function",
    "timestamp": "2026-06-03T00:00:00Z",
    "execution_model": "claude-opus-4.7",
    "duration_seconds": 2.5,
    "estimated_cost": 0.01,
    "files_updated": 1,
    "verification_attempted": True,
    "verification_passed": True,
    "file_context": {"paths": ["src/a.py"]},
    "context_files_injected_count": 3,
    "policy_decisions": [{"decision_id": "d1", "decision": "allow", "action": "write"}],
    "approval_receipt": {"granted": True},
    "execution_spans": [{"span_id": "s1", "name": "planning", "kind": "phase"}],
    "developer_feedback": {"outcome": "accepted"},
    "run_timeline": [
        {"event": "receipt_saved", "label": "Saved Shard receipt",
         "kind": "receipt", "status": "completed"}
    ],
    "summary": "done",
}

_FAILED_ENTRY = {**_GOOD_ENTRY, "verification_passed": False}


def _assert_no_unsafe(test: unittest.TestCase, text: str) -> None:
    for needle in ("C:\\", "C:/", "/Users/", "/home/", "sk-", "AKIA"):
        test.assertNotIn(needle, text, msg=f"unsafe substring {needle!r} leaked: {text}")


def _invoke(args: list[str], *, runs: list[dict] | None, interactions: list[dict] | None = None):
    runner = CliRunner()
    with tempfile.TemporaryDirectory() as td:
        log_dir = Path(td) / ".openshard"
        log_dir.mkdir()
        if runs is not None:
            (log_dir / "runs.jsonl").write_text(
                "".join(json.dumps(e) + "\n" for e in runs), encoding="utf-8"
            )
        if interactions is not None:
            (log_dir / "interactions.jsonl").write_text(
                "".join(json.dumps(e) + "\n" for e in interactions), encoding="utf-8"
            )
        with patch("openshard.cli.main.Path.cwd", return_value=Path(td)):
            result = runner.invoke(cli, args)
    return result


class TestTrustLast(unittest.TestCase):
    # --- no / empty history ------------------------------------------------

    def test_no_runs_file_human(self):
        result = _invoke(["trust", "last"], runs=None)
        self.assertEqual(result.exit_code, 0, msg=result.output)
        self.assertIn("No run history", result.output)

    def test_no_runs_file_json_is_not_found(self):
        result = _invoke(["trust", "last", "--json"], runs=None)
        self.assertEqual(result.exit_code, 0, msg=result.output)
        data = json.loads(result.output)
        self.assertEqual(data["command"], "trust last")
        self.assertEqual(data["status"], "not_found")
        self.assertIsNone(data["score"])
        self.assertEqual(data["penalties"], [])

    def test_empty_history_json_is_not_found(self):
        result = _invoke(["trust", "last", "--json"], runs=[])
        self.assertEqual(result.exit_code, 0, msg=result.output)
        data = json.loads(result.output)
        self.assertEqual(data["status"], "not_found")

    # --- valid runs --------------------------------------------------------

    def test_good_run_json_envelope(self):
        result = _invoke(["trust", "last", "--json"], runs=[_GOOD_ENTRY])
        self.assertEqual(result.exit_code, 0, msg=result.output)
        data = json.loads(result.output)  # raises if not valid JSON-only
        self.assertEqual(data["schema_version"], "1")
        self.assertEqual(data["command"], "trust last")
        self.assertEqual(data["status"], "ok")
        self.assertTrue(data["shard_id"])
        self.assertEqual(data["score"], 100)
        self.assertEqual(data["band"], "strong")
        self.assertIsInstance(data["signals"], dict)
        self.assertIsInstance(data["penalties"], list)
        self.assertIsInstance(data["warnings"], list)
        _assert_no_unsafe(self, result.output)

    def test_failed_run_lowers_score(self):
        result = _invoke(["trust", "last", "--json"], runs=[_FAILED_ENTRY])
        data = json.loads(result.output)
        self.assertLess(data["score"], 100)
        codes = {p["code"] for p in data["penalties"]}
        self.assertIn("verification_failed", codes)

    def test_skipped_run_with_changes_lowers_score_like_not_run(self):
        skipped = {
            **_GOOD_ENTRY,
            "verification_attempted": False,
            "verification_passed": None,
            "osn_verification_contract": {
                "enabled": True,
                "status": "skipped",
                "skipped_reason": "needs_approval: medium-risk command",
            },
        }
        result = _invoke(["trust", "last", "--json"], runs=[skipped])
        data = json.loads(result.output)
        self.assertEqual(data["signals"]["verification"], "skipped")
        codes = {p["code"] for p in data["penalties"]}
        self.assertIn("verification_not_run", codes)
        self.assertLess(data["score"], 100)

    def test_manual_review_run_is_weak_not_failure(self):
        mr = {
            **_GOOD_ENTRY,
            "verification_attempted": False,
            "verification_passed": None,
            "osn_verification_contract": {
                "enabled": True,
                "status": "manual_review",
                "manual_review_required": True,
            },
        }
        result = _invoke(["trust", "last", "--json"], runs=[mr])
        data = json.loads(result.output)
        self.assertEqual(data["signals"]["verification"], "manual_review")
        codes = {p["code"] for p in data["penalties"]}
        self.assertNotIn("verification_failed", codes)

    def test_human_output_is_not_json(self):
        result = _invoke(["trust", "last"], runs=[_GOOD_ENTRY])
        self.assertEqual(result.exit_code, 0, msg=result.output)
        self.assertIn("OpenShard Trust Score: 100 / 100 (strong)", result.output)
        self.assertIn("trust heuristic over recorded proof signals", result.output)
        with self.assertRaises(json.JSONDecodeError):
            json.loads(result.output)

    def test_latest_run_is_used(self):
        result = _invoke(["trust", "last", "--json"], runs=[_GOOD_ENTRY, _FAILED_ENTRY])
        data = json.loads(result.output)
        codes = {p["code"] for p in data["penalties"]}
        self.assertIn("verification_failed", codes)

    # --- robustness --------------------------------------------------------

    def test_corrupt_jsonl_line_is_skipped(self):
        runner = CliRunner()
        with tempfile.TemporaryDirectory() as td:
            log_dir = Path(td) / ".openshard"
            log_dir.mkdir()
            (log_dir / "runs.jsonl").write_text(
                "{ not json }\n" + json.dumps(_GOOD_ENTRY) + "\n", encoding="utf-8"
            )
            with patch("openshard.cli.main.Path.cwd", return_value=Path(td)):
                result = runner.invoke(cli, ["trust", "last", "--json"])
        self.assertEqual(result.exit_code, 0, msg=result.output)
        data = json.loads(result.output)
        self.assertEqual(data["status"], "ok")

    def test_missing_interactions_file_does_not_crash(self):
        result = _invoke(["trust", "last", "--json"], runs=[_GOOD_ENTRY], interactions=None)
        self.assertEqual(result.exit_code, 0, msg=result.output)
        data = json.loads(result.output)
        self.assertEqual(data["signals"]["interaction_events"], 0)

    def test_unsafe_interaction_event_is_counted(self):
        events = [{
            "schema_version": 1, "event_id": "e1", "run_id": _GOOD_ENTRY["timestamp"],
            "timestamp": "2026-06-03T00:00:01Z", "event_type": "unsafe_command",
            "summary": "ran a risky command", "severity": "high",
        }]
        result = _invoke(["trust", "last", "--json"], runs=[_GOOD_ENTRY], interactions=events)
        data = json.loads(result.output)
        codes = {p["code"] for p in data["penalties"]}
        self.assertIn("unsafe_interaction", codes)
        self.assertGreaterEqual(data["signals"]["interaction_events"], 1)

    def test_corrupt_interactions_file_does_not_crash(self):
        runner = CliRunner()
        with tempfile.TemporaryDirectory() as td:
            log_dir = Path(td) / ".openshard"
            log_dir.mkdir()
            (log_dir / "runs.jsonl").write_text(json.dumps(_GOOD_ENTRY) + "\n", encoding="utf-8")
            (log_dir / "interactions.jsonl").write_text("{ broken\n", encoding="utf-8")
            with patch("openshard.cli.main.Path.cwd", return_value=Path(td)):
                result = runner.invoke(cli, ["trust", "last", "--json"])
        self.assertEqual(result.exit_code, 0, msg=result.output)
        json.loads(result.output)


class TestLastJsonTrustBlock(unittest.TestCase):
    """The optional, additive trust block embedded in `last --json`."""

    def test_last_json_includes_trust_block(self):
        result = _invoke(["last", "--json"], runs=[_GOOD_ENTRY])
        self.assertEqual(result.exit_code, 0, msg=result.output)
        data = json.loads(result.output)
        # Existing 'run' block is untouched.
        self.assertIsInstance(data["run"], dict)
        # New additive trust block.
        self.assertIn("trust", data)
        self.assertEqual(data["trust"]["score"], 100)
        self.assertEqual(data["trust"]["band"], "strong")
        self.assertIsInstance(data["trust"]["penalties"], list)
        _assert_no_unsafe(self, result.output)


if __name__ == "__main__":
    unittest.main()
