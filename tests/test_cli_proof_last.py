"""Tests for `openshard proof last` and its `shard verify last` alias.

Covers the human view, the JSON envelope, no-runs behavior, unsafe records,
old / minimal records, alias parity, and that no raw unsafe content leaks.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from click.testing import CliRunner

from openshard.cli.main import cli

# A complete entry that should populate every required and recommended section.
_STRONG_ENTRY = {
    "schema_version": "1.2",
    "shard_id": "shard-20260604-0001",
    "timestamp": "2026-06-04T10:00:00Z",
    "task": "implement a feature",
    "workflow": "native",
    "execution_profile": "native_deep",
    "execution_model": "openrouter/some-model",
    "duration_seconds": 12.5,
    "estimated_cost": 0.0123,
    "files_created": 1,
    "files_updated": 1,
    "files_deleted": 0,
    "files_detail": [{"path": "a.py", "change_type": "create", "summary": ""}],
    "verification_attempted": True,
    "verification_passed": True,
    "review_checks": [
        {"name": "ruff", "status": "passed"},
        {"name": "pytest", "status": "passed"},
    ],
    "context_files_injected_count": 4,
    "context_utilisation_ratio": 0.5,
    "policy_decisions": [{"decision_id": "p1", "decision": "allow"}],
    "approval_receipt": {"granted": True, "reason": "ok"},
    "run_timeline": [
        {"event": "run_started", "label": "Run started", "kind": "run", "status": "completed"},
    ],
    "evidence_capsules": [
        {"capsule_id": "c1", "kind": "inspection", "summary": "read a.py"},
    ],
    "git_base_branch": "main",
    "git_head_commit_hash": "abc1234",
    "git_dirty": False,
    "summary": "Implemented the feature and verified it.",
    "developer_feedback": {"rating": "good", "action": "accepted"},
}

# A minimal / old-style record with only a couple of fields.
_MINIMAL_ENTRY = {"task": "do a small thing"}

# A record whose required sections are intact but whose timeline and provenance
# recommended sections are absent, so they surface as recommended gaps.
_NO_TIMELINE_PROVENANCE_ENTRY = {
    k: v
    for k, v in _STRONG_ENTRY.items()
    if k not in ("run_timeline", "evidence_capsules", "git_base_branch",
                 "git_head_commit_hash", "git_dirty")
}

_RAW_LEAK = "raw diff body that must never appear"


def _assert_no_unsafe(test: unittest.TestCase, text: str) -> None:
    for needle in ("C:\\", "C:/", "/Users/", "/home/", "sk-", "AKIA", _RAW_LEAK):
        test.assertNotIn(needle, text, msg=f"unsafe substring {needle!r} leaked: {text}")


def _invoke(args: list[str], *, runs: list[dict] | None):
    runner = CliRunner()
    with tempfile.TemporaryDirectory() as td:
        log_dir = Path(td) / ".openshard"
        log_dir.mkdir()
        if runs is not None:
            (log_dir / "runs.jsonl").write_text(
                "".join(json.dumps(e) + "\n" for e in runs), encoding="utf-8"
            )
        with patch("openshard.cli.main.Path.cwd", return_value=Path(td)):
            result = runner.invoke(cli, args)
    return result


class TestProofLastHuman(unittest.TestCase):
    def test_strong_entry_human(self):
        result = _invoke(["proof", "last"], runs=[_STRONG_ENTRY])
        self.assertEqual(result.exit_code, 0, msg=result.output)
        out = result.output
        # Leads with the product-facing heading, not the technical phrase.
        self.assertTrue(out.startswith("Proof for last run"), msg=out)
        self.assertIn("Status:", out)
        self.assertIn("Summary:", out)
        self.assertIn("Required proof:", out)
        self.assertIn("Missing recommended proof:", out)
        self.assertIn("Unsafe findings:", out)
        self.assertIn("Next action:", out)
        # The technical phrase appears only on the Contract detail line.
        self.assertIn("Contract: Shard Proof Contract v", out)
        self.assertNotIn("Shard Proof Contract v1\n", out[: len("Proof for last run") + 5])
        _assert_no_unsafe(self, out)


class TestProofLastHumanLabels(unittest.TestCase):
    def test_human_uses_plain_english_labels(self):
        result = _invoke(["proof", "last"], runs=[_NO_TIMELINE_PROVENANCE_ENTRY])
        out = result.output
        # Required sections read in plain language.
        self.assertIn("Verification:", out)
        # Recommended gaps use plain-English labels, not raw section names.
        self.assertIn("Step-by-step run events", out)
        self.assertIn("Repo state", out)
        # Raw technical section names must not appear in human output.
        self.assertNotIn("timeline", out)
        self.assertNotIn("provenance", out)
        self.assertNotIn("repo_state", out)
        _assert_no_unsafe(self, out)

    def test_json_keeps_raw_section_names(self):
        result = _invoke(["proof", "last", "--json"], runs=[_NO_TIMELINE_PROVENANCE_ENTRY])
        data = json.loads(result.output)
        names = [s["name"] for s in data["proof_contract"]["sections"]]
        # Technical names are preserved in JSON.
        self.assertIn("timeline", names)
        self.assertIn("provenance", names)


class TestProofLastJson(unittest.TestCase):
    def test_strong_entry_json(self):
        result = _invoke(["proof", "last", "--json"], runs=[_STRONG_ENTRY])
        self.assertEqual(result.exit_code, 0, msg=result.output)
        data = json.loads(result.output)  # raises if not valid JSON-only
        self.assertEqual(data["schema_version"], "1")
        self.assertEqual(data["command"], "proof last")
        self.assertEqual(data["status"], "ok")
        self.assertTrue(data["shard_id"])
        self.assertIsInstance(data["proof_contract"], dict)
        self.assertEqual(data["validation_errors"], [])
        self.assertTrue(data["recommendation"])
        # Full 17-section detail is carried in JSON, not the human view.
        self.assertEqual(len(data["proof_contract"]["sections"]), 17)
        _assert_no_unsafe(self, result.output)


class TestAliasParity(unittest.TestCase):
    def test_alias_human_matches(self):
        primary = _invoke(["proof", "last"], runs=[_STRONG_ENTRY])
        alias = _invoke(["shard", "verify", "last"], runs=[_STRONG_ENTRY])
        self.assertEqual(primary.exit_code, alias.exit_code)
        self.assertEqual(primary.output, alias.output)

    def test_alias_json_matches(self):
        primary = _invoke(["proof", "last", "--json"], runs=[_STRONG_ENTRY])
        alias = _invoke(["shard", "verify", "last", "--json"], runs=[_STRONG_ENTRY])
        self.assertEqual(primary.exit_code, alias.exit_code)
        self.assertEqual(primary.output, alias.output)
        # Both report the canonical machine command name.
        self.assertEqual(json.loads(alias.output)["command"], "proof last")


class TestNoRuns(unittest.TestCase):
    def test_no_runs_human(self):
        result = _invoke(["proof", "last"], runs=None)
        self.assertNotEqual(result.exit_code, 0)
        self.assertIn("No run history", result.output)

    def test_no_runs_json(self):
        result = _invoke(["proof", "last", "--json"], runs=None)
        self.assertNotEqual(result.exit_code, 0)
        data = json.loads(result.output)
        self.assertEqual(data["command"], "proof last")
        self.assertEqual(data["status"], "error")
        self.assertIsNone(data["proof_contract"])


class TestUnsafe(unittest.TestCase):
    def test_unsafe_contract_exits_nonzero(self):
        entry = {**_STRONG_ENTRY, "raw_diff": _RAW_LEAK}
        result = _invoke(["proof", "last", "--json"], runs=[entry])
        self.assertNotEqual(result.exit_code, 0)
        data = json.loads(result.output)
        self.assertEqual(data["status"], "unsafe")
        self.assertEqual(data["proof_contract"]["overall_status"], "unsafe")
        self.assertIn("blocked_field:raw_diff", data["proof_contract"]["unsafe_findings"])
        _assert_no_unsafe(self, result.output)

    def test_unsafe_contract_human(self):
        entry = {**_STRONG_ENTRY, "raw_diff": _RAW_LEAK}
        result = _invoke(["proof", "last"], runs=[entry])
        self.assertNotEqual(result.exit_code, 0)
        self.assertIn("blocked_field:raw_diff", result.output)
        _assert_no_unsafe(self, result.output)


class TestMinimalRecord(unittest.TestCase):
    def test_minimal_record_does_not_crash(self):
        result = _invoke(["proof", "last", "--json"], runs=[_MINIMAL_ENTRY])
        # Minimal records are not unsafe or invalid, so exit 0.
        self.assertEqual(result.exit_code, 0, msg=result.output)
        data = json.loads(result.output)
        self.assertEqual(data["status"], "ok")
        self.assertIsInstance(data["proof_contract"]["missing_required_sections"], list)
        self.assertIsInstance(data["validation_errors"], list)
        _assert_no_unsafe(self, result.output)


if __name__ == "__main__":
    unittest.main()
