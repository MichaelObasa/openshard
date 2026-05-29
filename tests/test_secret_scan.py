"""Tests for openshard/security/secret_scan.py and its receipt integration."""
from __future__ import annotations

import unittest
from pathlib import Path

from openshard.security.secret_scan import (
    SecretScanResult,
    _fingerprint,
    _is_noisy_path,
    _redact,
    scan_paths_for_secrets,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write(tmp_path: Path, name: str, content: str) -> Path:
    p = tmp_path / name
    p.write_text(content, encoding="utf-8")
    return p


def _scan_text(content: str, tmp_path: Path, name: str = "sample.txt") -> SecretScanResult:
    p = _write(tmp_path, name, content)
    return scan_paths_for_secrets([p])


# ---------------------------------------------------------------------------
# Pattern detection
# ---------------------------------------------------------------------------

class TestPatternDetection(unittest.TestCase):

    def test_detects_aws_key(self, tmp_path=None):
        if tmp_path is None:
            import tempfile
            with tempfile.TemporaryDirectory() as td:
                return self.test_detects_aws_key(Path(td))
        # Use a key that does not contain placeholder words like "example"
        result = _scan_text("AWS_ACCESS_KEY_ID=AKIA1234567890ABCDEF\n", tmp_path)
        self.assertEqual(len(result.findings), 1)
        self.assertEqual(result.findings[0].kind, "aws_access_key_id")

    def test_detects_github_classic_token(self, tmp_path=None):
        if tmp_path is None:
            import tempfile
            with tempfile.TemporaryDirectory() as td:
                return self.test_detects_github_classic_token(Path(td))
        result = _scan_text("token = ghp_ABCDEFGHIJKLMNOPQRSTUVWXYZabcd\n", tmp_path)
        self.assertEqual(len(result.findings), 1)
        self.assertEqual(result.findings[0].kind, "github_token")

    def test_detects_github_pat(self, tmp_path=None):
        if tmp_path is None:
            import tempfile
            with tempfile.TemporaryDirectory() as td:
                return self.test_detects_github_pat(Path(td))
        result = _scan_text("GH_TOKEN=github_pat_ABCDEFGHIJKLMNOPQRSTUVWXYZabc1234567\n", tmp_path)
        self.assertEqual(len(result.findings), 1)
        self.assertEqual(result.findings[0].kind, "github_pat")

    def test_detects_openai_key(self, tmp_path=None):
        if tmp_path is None:
            import tempfile
            with tempfile.TemporaryDirectory() as td:
                return self.test_detects_openai_key(Path(td))
        result = _scan_text("OPENAI_API_KEY=sk-ABCDEFGHIJKLMNOPQRSTUVWXYZabcdef123456\n", tmp_path)
        # Should detect openai_key (sk-) but NOT anthropic_key (sk-ant-)
        kinds = [f.kind for f in result.findings]
        self.assertIn("openai_key", kinds)
        self.assertNotIn("anthropic_key", kinds)

    def test_detects_anthropic_key(self, tmp_path=None):
        if tmp_path is None:
            import tempfile
            with tempfile.TemporaryDirectory() as td:
                return self.test_detects_anthropic_key(Path(td))
        result = _scan_text("ANTHROPIC_API_KEY=sk-ant-api03-ABCDEFGHIJKLMNOPQRSTUVWXYZabcdef\n", tmp_path)
        kinds = [f.kind for f in result.findings]
        self.assertIn("anthropic_key", kinds)
        self.assertNotIn("openai_key", kinds)

    def test_detects_generic_api_key_assignment(self, tmp_path=None):
        if tmp_path is None:
            import tempfile
            with tempfile.TemporaryDirectory() as td:
                return self.test_detects_generic_api_key_assignment(Path(td))
        result = _scan_text('api_key = "realvalue1234567890"\n', tmp_path)
        kinds = [f.kind for f in result.findings]
        self.assertIn("generic_secret_assignment", kinds)


# ---------------------------------------------------------------------------
# Placeholder suppression
# ---------------------------------------------------------------------------

class TestPlaceholderSuppression(unittest.TestCase):

    def test_ignores_short_value(self, tmp_path=None):
        if tmp_path is None:
            import tempfile
            with tempfile.TemporaryDirectory() as td:
                return self.test_ignores_short_value(Path(td))
        result = _scan_text("token=test\n", tmp_path)
        self.assertEqual(result.findings, [])

    def test_ignores_shell_variable_placeholder(self, tmp_path=None):
        if tmp_path is None:
            import tempfile
            with tempfile.TemporaryDirectory() as td:
                return self.test_ignores_shell_variable_placeholder(Path(td))
        result = _scan_text("api_key=${MY_SECRET_TOKEN}\n", tmp_path)
        self.assertEqual(result.findings, [])

    def test_ignores_terraform_variable_reference(self, tmp_path=None):
        if tmp_path is None:
            import tempfile
            with tempfile.TemporaryDirectory() as td:
                return self.test_ignores_terraform_variable_reference(Path(td))
        result = _scan_text('api_key = var.my_secret_token\n', tmp_path)
        self.assertEqual(result.findings, [])

    def test_ignores_changeme_placeholder(self, tmp_path=None):
        if tmp_path is None:
            import tempfile
            with tempfile.TemporaryDirectory() as td:
                return self.test_ignores_changeme_placeholder(Path(td))
        result = _scan_text('password = "changeme"\n', tmp_path)
        self.assertEqual(result.findings, [])

    def test_ignores_example_placeholder(self, tmp_path=None):
        if tmp_path is None:
            import tempfile
            with tempfile.TemporaryDirectory() as td:
                return self.test_ignores_example_placeholder(Path(td))
        result = _scan_text('password = "example-password-123456"\n', tmp_path)
        self.assertEqual(result.findings, [])


# ---------------------------------------------------------------------------
# Redaction and fingerprint
# ---------------------------------------------------------------------------

class TestRedactionAndFingerprint(unittest.TestCase):

    def test_redacted_never_contains_raw_secret(self, tmp_path=None):
        if tmp_path is None:
            import tempfile
            with tempfile.TemporaryDirectory() as td:
                return self.test_redacted_never_contains_raw_secret(Path(td))
        raw = "AKIA1234567890ABCDEF"
        result = _scan_text(f"key={raw}\n", tmp_path)
        self.assertTrue(result.findings, "expected at least one finding")
        for f in result.findings:
            self.assertNotEqual(f.redacted, raw)
            self.assertNotIn(raw, f.redacted)

    def test_redact_long_value_shows_prefix_suffix(self):
        val = "AKIAIOSFODNN7EXAMPLE"
        redacted = _redact(val)
        self.assertTrue(redacted.startswith(val[:4]))
        self.assertTrue(redacted.endswith(val[-4:]))
        self.assertIn("...", redacted)
        self.assertNotEqual(redacted, val)

    def test_redact_short_value_returns_stars(self):
        redacted = _redact("abc")
        self.assertEqual(redacted, "***")

    def test_fingerprint_stable_same_input(self):
        fp1 = _fingerprint("AKIAIOSFODNN7EXAMPLE")
        fp2 = _fingerprint("AKIAIOSFODNN7EXAMPLE")
        self.assertEqual(fp1, fp2)

    def test_fingerprint_differs_for_different_input(self):
        fp1 = _fingerprint("AKIAIOSFODNN7EXAMPLE")
        fp2 = _fingerprint("AKIAIOSFODNN7EXAMPLF")
        self.assertNotEqual(fp1, fp2)


# ---------------------------------------------------------------------------
# Scanner file behaviour
# ---------------------------------------------------------------------------

class TestScannerFileBehaviour(unittest.TestCase):

    def test_skips_git_dir(self, tmp_path=None):
        if tmp_path is None:
            import tempfile
            with tempfile.TemporaryDirectory() as td:
                return self.test_skips_git_dir(Path(td))
        git_dir = tmp_path / ".git"
        git_dir.mkdir()
        secret_file = git_dir / "config"
        secret_file.write_text("AKIAIOSFODNN7EXAMPLE\n", encoding="utf-8")
        result = scan_paths_for_secrets([secret_file])
        self.assertEqual(result.findings, [])
        self.assertEqual(result.scanned_files_count, 0)

    def test_skips_pycache_dir(self, tmp_path=None):
        if tmp_path is None:
            import tempfile
            with tempfile.TemporaryDirectory() as td:
                return self.test_skips_pycache_dir(Path(td))
        cache_dir = tmp_path / "__pycache__"
        cache_dir.mkdir()
        secret_file = cache_dir / "module.pyc"
        secret_file.write_text("AKIAIOSFODNN7EXAMPLE\n", encoding="utf-8")
        result = scan_paths_for_secrets([secret_file])
        self.assertEqual(result.findings, [])

    def test_skips_large_files(self, tmp_path=None):
        if tmp_path is None:
            import tempfile
            with tempfile.TemporaryDirectory() as td:
                return self.test_skips_large_files(Path(td))
        large_file = tmp_path / "large.txt"
        # Write just over 512 KB
        large_file.write_bytes(b"AKIAIOSFODNN7EXAMPLE\n" * 30000)
        result = scan_paths_for_secrets([large_file])
        self.assertEqual(result.scanned_files_count, 0)
        self.assertEqual(result.findings, [])

    def test_handles_binary_file_safely(self, tmp_path=None):
        if tmp_path is None:
            import tempfile
            with tempfile.TemporaryDirectory() as td:
                return self.test_handles_binary_file_safely(Path(td))
        binary_file = tmp_path / "image.bin"
        binary_file.write_bytes(bytes(range(256)) * 100)
        # Must not raise
        result = scan_paths_for_secrets([binary_file])
        self.assertIsInstance(result, SecretScanResult)
        self.assertEqual(result.findings, [])

    def test_handles_missing_file_safely(self, tmp_path=None):
        if tmp_path is None:
            import tempfile
            with tempfile.TemporaryDirectory() as td:
                return self.test_handles_missing_file_safely(Path(td))
        missing = tmp_path / "does_not_exist.txt"
        result = scan_paths_for_secrets([missing])
        self.assertIsInstance(result, SecretScanResult)

    def test_results_sorted_deterministic(self, tmp_path=None):
        if tmp_path is None:
            import tempfile
            with tempfile.TemporaryDirectory() as td:
                return self.test_results_sorted_deterministic(Path(td))
        f1 = _write(tmp_path, "a.txt", "AKIAIOSFODNN7EXAMPLE\n")
        f2 = _write(tmp_path, "b.txt", "ghp_ABCDEFGHIJKLMNOPQRSTUVWXYZabcd\n")
        result1 = scan_paths_for_secrets([f1, f2])
        result2 = scan_paths_for_secrets([f2, f1])
        self.assertEqual(
            [(f.path, f.kind) for f in result1.findings],
            [(f.path, f.kind) for f in result2.findings],
        )

    def test_deduplicates_same_secret(self, tmp_path=None):
        if tmp_path is None:
            import tempfile
            with tempfile.TemporaryDirectory() as td:
                return self.test_deduplicates_same_secret(Path(td))
        content = "AKIA1234567890ABCDEF\nAKIA1234567890ABCDEF\n"
        result = _scan_text(content, tmp_path)
        aws_findings = [f for f in result.findings if f.kind == "aws_access_key_id"]
        self.assertEqual(len(aws_findings), 1)


# ---------------------------------------------------------------------------
# Noisy path helper
# ---------------------------------------------------------------------------

class TestIsNoisyPath(unittest.TestCase):

    def test_git_dir_is_noisy(self):
        self.assertTrue(_is_noisy_path(".git/config"))

    def test_pycache_is_noisy(self):
        self.assertTrue(_is_noisy_path("src/__pycache__/mod.pyc"))

    def test_venv_is_noisy(self):
        self.assertTrue(_is_noisy_path(".venv/lib/site-packages/pkg.py"))

    def test_normal_path_is_not_noisy(self):
        self.assertFalse(_is_noisy_path("src/config.py"))

    def test_dist_at_root_is_noisy(self):
        self.assertTrue(_is_noisy_path("dist/main.js"))

    def test_dist_in_middle_is_not_noisy(self):
        self.assertFalse(_is_noisy_path("src/dist_utils.py"))


# ---------------------------------------------------------------------------
# Receipt integration
# ---------------------------------------------------------------------------

class TestReceiptIntegration(unittest.TestCase):

    def _entry_with_scan_result(self) -> dict:
        return {
            "task": "add feature",
            "timestamp": "2026-04-13T06:24:08.695472Z",
            "execution_model": "anthropic/claude-sonnet-4-6",
            "summary": "Feature implemented.",
            "secret_scan_result": {
                "scanned_files_count": 2,
                "blocked": False,
                "summary": "1 potential secret detected in 2 scanned files",
                "findings": [
                    {
                        "kind": "aws_access_key_id",
                        "path": "config/.env",
                        "line": 4,
                        "redacted": "AKIA...MPLE",
                        "severity": "Critical",
                        "fingerprint": "abc123def456",
                    }
                ],
            },
        }

    def test_evidence_capsule_summary_safe(self):
        from openshard.history.shard_contract import build_shard_receipt
        receipt = build_shard_receipt(self._entry_with_scan_result())
        secret_caps = [c for c in receipt.evidence_capsules if c.kind == "secret_scan"]
        self.assertEqual(len(secret_caps), 1)
        cap = secret_caps[0]
        # Summary must not contain a real key — only the redacted form or kind name
        self.assertNotIn("AKIAIOSFODNN7EXAMPLE", cap.summary)

    def test_evidence_capsule_fields_populated(self):
        from openshard.history.shard_contract import build_shard_receipt
        receipt = build_shard_receipt(self._entry_with_scan_result())
        secret_caps = [c for c in receipt.evidence_capsules if c.kind == "secret_scan"]
        self.assertEqual(len(secret_caps), 1)
        cap = secret_caps[0]
        self.assertEqual(cap.capsule_id, "abc123def456")
        self.assertEqual(cap.source, "secret_scanner")
        self.assertEqual(cap.path, "config/.env")
        self.assertEqual(cap.line, 4)
        self.assertEqual(cap.severity, "Critical")

    def test_full_receipt_no_raw_secret(self):
        from openshard.history.shard_contract import build_shard_receipt, render_full_shard_receipt
        raw_secret = "AKIAIOSFODNN7EXAMPLE"
        receipt = build_shard_receipt(self._entry_with_scan_result())
        rendered = render_full_shard_receipt(receipt)
        self.assertNotIn(raw_secret, rendered)

    def test_old_receipt_without_scan_renders_safely(self):
        from openshard.history.shard_contract import build_shard_receipt, render_full_shard_receipt, render_compact_shard_receipt
        entry = {
            "task": "old task",
            "timestamp": "2025-01-01T00:00:00Z",
            "execution_model": "anthropic/claude-sonnet-4-6",
            "summary": "Done.",
        }
        receipt = build_shard_receipt(entry)
        # Must not crash; must not show an empty EVIDENCE CAPSULES section
        full = render_full_shard_receipt(receipt)
        compact = render_compact_shard_receipt(receipt)
        self.assertIsInstance(full, str)
        self.assertIsInstance(compact, str)
        # No empty evidence capsules section should appear
        self.assertNotIn("EVIDENCE CAPSULES\n\n", full)

    def test_compact_receipt_shows_secrets_row_when_findings(self):
        from openshard.history.shard_contract import build_shard_receipt, render_compact_shard_receipt
        receipt = build_shard_receipt(self._entry_with_scan_result())
        compact = render_compact_shard_receipt(receipt)
        self.assertIn("Secrets", compact)
        self.assertIn("finding", compact)

    def test_compact_receipt_no_secrets_row_when_clean(self):
        from openshard.history.shard_contract import build_shard_receipt, render_compact_shard_receipt
        entry = {
            "task": "clean task",
            "timestamp": "2026-04-13T06:24:08.695472Z",
            "execution_model": "anthropic/claude-sonnet-4-6",
            "summary": "Done.",
        }
        receipt = build_shard_receipt(entry)
        compact = render_compact_shard_receipt(receipt)
        self.assertNotIn("Secrets", compact)

    def test_existing_evidence_capsules_preserved(self):
        from openshard.history.shard_contract import build_shard_receipt
        entry = self._entry_with_scan_result()
        entry["evidence_capsules"] = [
            {
                "capsule_id": "existing-cap-1",
                "kind": "iac_violation",
                "summary": "Missing resource tag",
                "source": "tflint",
                "path": "main.tf",
                "line": 10,
                "severity": "medium",
            }
        ]
        receipt = build_shard_receipt(entry)
        kinds = [c.kind for c in receipt.evidence_capsules]
        self.assertIn("iac_violation", kinds)
        self.assertIn("secret_scan", kinds)

    def test_compact_receipt_wording_honest(self):
        from openshard.history.shard_contract import build_shard_receipt, render_compact_shard_receipt
        receipt = build_shard_receipt(self._entry_with_scan_result())
        compact = render_compact_shard_receipt(receipt)
        # Must not imply blocking or model protection
        self.assertNotIn("blocked", compact.lower())
        self.assertNotIn("prevented", compact.lower())
        self.assertIn("see full receipt", compact.lower())
