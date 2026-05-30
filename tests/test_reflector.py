"""Tests for openshard.reflection.reflector."""
from __future__ import annotations

import json
import unittest
from pathlib import Path

from openshard.history.shard_contract import (
    EvidenceCapsule,
    ShardFinding,
    ShardReceipt,
    build_shard_receipt,
)
from openshard.reflection.reflector import (
    RunReflection,
    build_run_reflection,
    render_run_reflection,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _base_receipt(**kwargs) -> ShardReceipt:
    """Minimal valid ShardReceipt with sensible defaults."""
    defaults = dict(
        shard_id="shard-20260101-0001",
        created_at="2026-01-01T00:00:00Z",
        task_short="Add tests",
        task_full="Add unit tests for the auth module",
        agent="OpenShard Native",
        strategy="focused",
        model_display="claude-sonnet",
        risk="Low",
        sandbox="On",
        files_changed=0,
        checks_display="Not recorded",
        approval="Not recorded",
        cost_display="$0.0012",
        result="Tests added.",
        status="Not recorded",
        duration_seconds=None,
        error_class=None,
        schema_version="1.1",
    )
    defaults.update(kwargs)
    return ShardReceipt(**defaults)


def _strong_receipt() -> ShardReceipt:
    return _base_receipt(
        files_changed=3,
        checks_display="2 passed",
        status="Passed",
        check_results=["+ terraform fmt  passed", "+ terraform validate  passed"],
        run_timeline=[{"label": "Run started"}, {"label": "Checks ran"}],
        inspected_files=["auth/tokens.py", "auth/session.py"],
        policy_decisions=[{"decision_id": "pd-1", "action": "read", "decision": "allow"}],
        context_quality="Good",
        git_state="Clean",
        schema_version="1.1",
    )


def _failed_receipt() -> ShardReceipt:
    return _base_receipt(
        error_class="ToolCallError",
        status="Failed",
        checks_display="0/1 passed",
        check_results=["x verification  failed"],
        files_changed=0,
        schema_version="1.1",
    )


def _minimal_receipt() -> ShardReceipt:
    """Old/minimal receipt with few fields."""
    return _base_receipt(
        schema_version=None,
        run_timeline=[],
        inspected_files=[],
        check_results=[],
        policy_decisions=[],
        evidence_capsules=[],
        status="Not recorded",
        checks_display="Not recorded",
    )


# ---------------------------------------------------------------------------
# Scoring tests
# ---------------------------------------------------------------------------

class TestStrongRunScore(unittest.TestCase):

    def test_strong_run_scores_good_or_strong(self):
        r = build_run_reflection(_strong_receipt())
        self.assertGreaterEqual(r.score, 70)
        self.assertIn(r.level, ("strong", "good"))

    def test_strong_run_has_strengths(self):
        r = build_run_reflection(_strong_receipt())
        self.assertTrue(r.strengths)

    def test_strong_run_level_is_not_weak(self):
        r = build_run_reflection(_strong_receipt())
        self.assertNotEqual(r.level, "weak")


class TestFailedRunScore(unittest.TestCase):

    def test_failed_run_scores_low(self):
        r = build_run_reflection(_failed_receipt())
        self.assertLessEqual(r.score, 45)

    def test_failed_run_level_is_weak_or_fair(self):
        r = build_run_reflection(_failed_receipt())
        self.assertIn(r.level, ("weak", "fair"))

    def test_failed_run_has_warnings(self):
        r = build_run_reflection(_failed_receipt())
        self.assertTrue(r.warnings)


class TestWriteTaskWithoutChecks(unittest.TestCase):

    def test_gap_present_when_write_task_no_checks(self):
        receipt = _base_receipt(
            files_changed=5,
            status="Not recorded",
            check_results=[],
            checks_display="Not recorded",
        )
        r = build_run_reflection(receipt)
        gap_text = " ".join(r.gaps).lower()
        self.assertIn("verification", gap_text)

    def test_suggestion_includes_verification(self):
        receipt = _base_receipt(
            files_changed=5,
            status="Not recorded",
            check_results=[],
            checks_display="Not recorded",
        )
        r = build_run_reflection(receipt)
        suggestion_text = " ".join(r.suggestions).lower()
        self.assertIn("verification", suggestion_text)


class TestSecretScanWarning(unittest.TestCase):

    def _receipt_with_secret(self) -> ShardReceipt:
        cap = EvidenceCapsule(
            capsule_id="sec-001",
            kind="secret_scan",
            summary="Potential anthropic_key detected and redacted",
            source="secret_scanner",
        )
        return _base_receipt(evidence_capsules=[cap])

    def test_secret_scan_creates_warning(self):
        r = build_run_reflection(self._receipt_with_secret())
        warning_text = " ".join(r.warnings).lower()
        self.assertIn("secret", warning_text)

    def test_secret_scan_reduces_score(self):
        clean = build_run_reflection(_base_receipt())
        with_secret = build_run_reflection(self._receipt_with_secret())
        self.assertLess(with_secret.score, clean.score)

    def test_secret_scan_creates_suggestion(self):
        r = build_run_reflection(self._receipt_with_secret())
        suggestion_text = " ".join(r.suggestions).lower()
        self.assertIn("secret", suggestion_text)


class TestAdapterMetadata(unittest.TestCase):

    def test_available_adapter_adds_strength(self):
        receipt = _base_receipt(adapter="opencode", adapter_available=True)
        r = build_run_reflection(receipt)
        strength_text = " ".join(r.strengths).lower()
        self.assertIn("adapter", strength_text)

    def test_unavailable_adapter_creates_gap(self):
        receipt = _base_receipt(adapter="opencode", adapter_available=False)
        r = build_run_reflection(receipt)
        gap_text = " ".join(r.gaps).lower()
        self.assertIn("adapter", gap_text)

    def test_unavailable_adapter_reduces_score(self):
        available = build_run_reflection(_base_receipt(adapter="opencode", adapter_available=True))
        unavailable = build_run_reflection(_base_receipt(adapter="opencode", adapter_available=False))
        self.assertLess(unavailable.score, available.score)


class TestPolicyDecisions(unittest.TestCase):

    def test_policy_decisions_creates_strength(self):
        receipt = _base_receipt(
            policy_decisions=[{"decision_id": "pd-1", "action": "read", "decision": "allow"}]
        )
        r = build_run_reflection(receipt)
        strength_text = " ".join(r.strengths).lower()
        self.assertIn("policy", strength_text)

    def test_missing_policy_decisions_creates_gap(self):
        receipt = _base_receipt(policy_decisions=[])
        r = build_run_reflection(receipt)
        gap_text = " ".join(r.gaps).lower()
        self.assertIn("policy", gap_text)


class TestMissingEvidence(unittest.TestCase):

    def test_no_inspected_files_creates_gap(self):
        receipt = _base_receipt(inspected_files=[])
        r = build_run_reflection(receipt)
        gap_text = " ".join(r.gaps).lower()
        self.assertIn("file evidence", gap_text)


class TestApprovalDenied(unittest.TestCase):

    def test_approval_denied_creates_warning(self):
        receipt = _base_receipt(
            approval_required=True,
            approval_granted=False,
            approval="Required → Denied",
        )
        r = build_run_reflection(receipt)
        warning_text = " ".join(r.warnings).lower()
        self.assertIn("approval", warning_text)

    def test_approval_denied_low_score(self):
        receipt = _base_receipt(
            approval_required=True,
            approval_granted=False,
            approval="Required → Denied",
        )
        r = build_run_reflection(receipt)
        self.assertLessEqual(r.score, 40)


class TestMinimalOldReceipt(unittest.TestCase):

    def test_minimal_receipt_does_not_raise(self):
        r = build_run_reflection(_minimal_receipt())
        self.assertIsInstance(r, RunReflection)

    def test_minimal_receipt_score_in_range(self):
        r = build_run_reflection(_minimal_receipt())
        self.assertGreaterEqual(r.score, 0)
        self.assertLessEqual(r.score, 100)

    def test_minimal_receipt_renders_safely(self):
        r = build_run_reflection(_minimal_receipt())
        lines = render_run_reflection(r)
        self.assertTrue(len(lines) > 0)

    def test_minimal_receipt_has_old_format_gap(self):
        r = build_run_reflection(_minimal_receipt())
        gap_text = " ".join(r.gaps).lower()
        self.assertIn("old receipt", gap_text)


# ---------------------------------------------------------------------------
# Score invariant tests
# ---------------------------------------------------------------------------

class TestScoreInvariants(unittest.TestCase):

    def _variants(self):
        return [
            _base_receipt(),
            _strong_receipt(),
            _failed_receipt(),
            _minimal_receipt(),
            _base_receipt(adapter="opencode", adapter_available=False),
            _base_receipt(approval_required=True, approval_granted=False),
            _base_receipt(
                evidence_capsules=[
                    EvidenceCapsule("s1", "secret_scan", "Found key", "scanner")
                ]
            ),
            _base_receipt(
                findings=[ShardFinding(severity="Critical", message="SQL injection found")]
            ),
            _base_receipt(files_changed=10, check_results=[], status="Not recorded"),
            _base_receipt(schema_version="1.1", policy_decisions=[
                {"decision_id": "x", "action": "write", "decision": "deny"}
            ]),
        ]

    def test_score_always_in_range(self):
        for receipt in self._variants():
            r = build_run_reflection(receipt)
            self.assertGreaterEqual(r.score, 0, f"score < 0 for {receipt}")
            self.assertLessEqual(r.score, 100, f"score > 100 for {receipt}")

    def test_level_always_valid(self):
        valid_levels = {"strong", "good", "fair", "weak"}
        for receipt in self._variants():
            r = build_run_reflection(receipt)
            self.assertIn(r.level, valid_levels)

    def test_confidence_always_valid(self):
        valid_confs = {"high", "medium", "low"}
        for receipt in self._variants():
            r = build_run_reflection(receipt)
            self.assertIn(r.confidence, valid_confs)

    def test_level_mapping_deterministic(self):
        # Force specific score thresholds via known receipts
        strong_r = build_run_reflection(_strong_receipt())
        if strong_r.score >= 80:
            self.assertEqual(strong_r.level, "strong")
        elif strong_r.score >= 60:
            self.assertEqual(strong_r.level, "good")

        weak_r = build_run_reflection(_failed_receipt())
        if weak_r.score < 40:
            self.assertEqual(weak_r.level, "weak")
        elif weak_r.score < 60:
            self.assertEqual(weak_r.level, "fair")


# ---------------------------------------------------------------------------
# Render safety tests
# ---------------------------------------------------------------------------

class TestRenderSafety(unittest.TestCase):

    def test_rendered_output_does_not_contain_task_full(self):
        receipt = _base_receipt(task_full="Add unit tests for the auth module")
        r = build_run_reflection(receipt)
        output = "\n".join(render_run_reflection(r))
        self.assertNotIn("Add unit tests for the auth module", output)

    def test_rendered_output_does_not_contain_stdout_stderr_labels(self):
        receipt = _base_receipt(
            adapter_stdout_summary="stdout: some output",
            adapter_stderr_summary="stderr: some error",
        )
        r = build_run_reflection(receipt)
        output = "\n".join(render_run_reflection(r))
        self.assertNotIn("stdout: some output", output)
        self.assertNotIn("stderr: some error", output)

    def test_render_caps_lists_at_five(self):
        # Build a receipt that triggers many signals
        caps = [
            EvidenceCapsule(f"ec-{i}", "secret_scan", f"Secret {i}", "scanner")
            for i in range(10)
        ]
        receipt = _base_receipt(
            evidence_capsules=caps[:1],  # one secret is enough for the warning
            adapter="opencode",
            adapter_available=False,
            approval_required=True,
            approval_granted=False,
            findings=[ShardFinding("Critical", "Issue"), ShardFinding("High", "Issue 2")],
        )
        r = build_run_reflection(receipt)
        self.assertLessEqual(len(r.strengths), 5)
        self.assertLessEqual(len(r.gaps), 5)
        self.assertLessEqual(len(r.suggestions), 5)
        self.assertLessEqual(len(r.warnings), 5)

    def test_render_includes_required_sections(self):
        r = build_run_reflection(_strong_receipt())
        output = "\n".join(render_run_reflection(r))
        self.assertIn("RUN REFLECTION", output)
        self.assertIn("Score", output)
        self.assertIn("Summary", output)
        self.assertIn("Recommended next move", output)
        self.assertIn("Advisory only", output)

    def test_render_includes_confidence(self):
        r = build_run_reflection(_strong_receipt())
        output = "\n".join(render_run_reflection(r))
        self.assertIn("confidence:", output)


# ---------------------------------------------------------------------------
# UX / CLI integration tests
# ---------------------------------------------------------------------------

class TestCLIReflectLast(unittest.TestCase):

    def test_no_history_shows_helpful_message(self):
        from click.testing import CliRunner
        from openshard.cli.main import cli
        runner = CliRunner()
        with runner.isolated_filesystem():
            result = runner.invoke(cli, ["reflect", "last"])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("No run history", result.output)

    def test_with_history_renders_reflection(self):
        from click.testing import CliRunner
        from openshard.cli.main import cli

        entry = {
            "schema_version": "1.1",
            "timestamp": "2026-01-01T00:00:00Z",
            "task": "Add tests for auth module",
            "execution_model": "claude-sonnet",
            "duration_seconds": 12.5,
            "files_created": 1,
            "files_updated": 0,
            "files_deleted": 0,
            "verification_attempted": True,
            "verification_passed": True,
            "workspace_path": "/tmp/ws",
            "repo_name": "myrepo",
            "git_branch": "main",
            "git_dirty": False,
            "summary": "Added auth tests.",
        }
        runner = CliRunner()
        with runner.isolated_filesystem():
            Path(".openshard").mkdir()
            Path(".openshard/runs.jsonl").write_text(
                json.dumps(entry) + "\n", encoding="utf-8"
            )
            result = runner.invoke(cli, ["reflect", "last"])
        self.assertEqual(result.exit_code, 0, msg=result.output)
        self.assertIn("RUN REFLECTION", result.output)
        self.assertIn("Score", result.output)
        self.assertIn("confidence:", result.output)
        self.assertIn("Summary", result.output)
        self.assertIn("Recommended next move", result.output)
        self.assertIn("Advisory only", result.output)

    def test_reflect_output_does_not_expose_raw_task_text(self):
        from click.testing import CliRunner
        from openshard.cli.main import cli

        raw_task = "Add unit tests for the auth module with secret key sk-ant-abc123"
        entry = {
            "schema_version": "1.1",
            "timestamp": "2026-01-01T00:00:00Z",
            "task": raw_task,
            "execution_model": "claude-sonnet",
            "duration_seconds": 5.0,
            "files_created": 0,
            "files_updated": 0,
            "files_deleted": 0,
            "verification_attempted": None,
            "verification_passed": None,
            "workspace_path": "/tmp/ws",
            "repo_name": "myrepo",
            "summary": "Done.",
        }
        runner = CliRunner()
        with runner.isolated_filesystem():
            Path(".openshard").mkdir()
            Path(".openshard/runs.jsonl").write_text(
                json.dumps(entry) + "\n", encoding="utf-8"
            )
            result = runner.invoke(cli, ["reflect", "last"])
        self.assertNotIn(raw_task, result.output)


# ---------------------------------------------------------------------------
# build_shard_receipt integration
# ---------------------------------------------------------------------------

class TestBuildFromEntry(unittest.TestCase):

    def test_build_from_minimal_entry(self):
        entry = {
            "timestamp": "2026-01-01T00:00:00Z",
            "task": "Do something",
            "execution_model": "claude-sonnet",
            "duration_seconds": 3.0,
        }
        receipt = build_shard_receipt(entry, index=0)
        r = build_run_reflection(receipt)
        self.assertGreaterEqual(r.score, 0)
        self.assertLessEqual(r.score, 100)

    def test_build_from_entry_with_secret_scan(self):
        entry = {
            "schema_version": "1.1",
            "timestamp": "2026-01-01T00:00:00Z",
            "task": "Deploy app",
            "execution_model": "claude-sonnet",
            "duration_seconds": 8.0,
            "secret_scan_result": {
                "scanned_files_count": 5,
                "findings": [
                    {
                        "kind": "anthropic_key",
                        "path": "config.py",
                        "line": 12,
                        "redacted": "sk-a***123",
                        "severity": "Critical",
                        "fingerprint": "abc123def456",
                    }
                ],
                "blocked": True,
                "summary": "1 secret found.",
            },
        }
        receipt = build_shard_receipt(entry, index=0)
        r = build_run_reflection(receipt)
        warning_text = " ".join(r.warnings).lower()
        self.assertIn("secret", warning_text)
