"""Tests for evidence path filtering in openshard.history.shard_contract."""

from __future__ import annotations

import unittest

from openshard.history.shard_contract import (
    _build_file_evidence,
    _is_noisy_evidence_path,
)


class TestIsNoisyEvidencePath(unittest.TestCase):
    # ── noisy at any depth ──────────────────────────────────────────────────

    def test_pytest_cache_root(self):
        self.assertTrue(_is_noisy_evidence_path(".pytest_cache/v/cache/lastfailed"))

    def test_pytest_cache_nested(self):
        self.assertTrue(_is_noisy_evidence_path("packages/app/.pytest_cache/foo"))

    def test_git_root(self):
        self.assertTrue(_is_noisy_evidence_path(".git/config"))

    def test_pycache_root(self):
        self.assertTrue(_is_noisy_evidence_path("__pycache__/module.cpython-311.pyc"))

    def test_pycache_nested(self):
        self.assertTrue(_is_noisy_evidence_path("services/api/__pycache__/x.pyc"))

    def test_venv_root(self):
        self.assertTrue(_is_noisy_evidence_path(".venv/lib/python3.11/site-packages/foo.py"))

    def test_plain_venv_root(self):
        self.assertTrue(_is_noisy_evidence_path("venv/lib/foo.py"))

    def test_node_modules_root(self):
        self.assertTrue(_is_noisy_evidence_path("node_modules/lodash/index.js"))

    def test_node_modules_nested(self):
        self.assertTrue(_is_noisy_evidence_path("frontend/node_modules/react/index.js"))

    def test_mypy_cache_root(self):
        self.assertTrue(_is_noisy_evidence_path(".mypy_cache/3.11/openshard/__init__.data"))

    def test_ruff_cache_nested(self):
        self.assertTrue(_is_noisy_evidence_path("src/.ruff_cache/0.1.0/foo"))

    # ── noisy only at root ──────────────────────────────────────────────────

    def test_dist_root(self):
        self.assertTrue(_is_noisy_evidence_path("dist/main.js"))

    def test_build_root(self):
        self.assertTrue(_is_noisy_evidence_path("build/index.html"))

    def test_coverage_root(self):
        self.assertTrue(_is_noisy_evidence_path("coverage/lcov.info"))

    def test_tmp_root(self):
        self.assertTrue(_is_noisy_evidence_path("tmp/scratch.txt"))

    def test_temp_root(self):
        self.assertTrue(_is_noisy_evidence_path("temp/output.log"))

    def test_cache_root(self):
        self.assertTrue(_is_noisy_evidence_path("cache/data.bin"))

    # ── broad names nested should NOT be filtered ───────────────────────────

    def test_dist_nested_not_noisy(self):
        # "dist" nested inside a legitimate path is not filtered
        self.assertFalse(_is_noisy_evidence_path("packages/lib/dist/index.js"))

    def test_build_nested_not_noisy(self):
        self.assertFalse(_is_noisy_evidence_path("src/build_config/settings.py"))

    # ── normal source files must not be filtered ────────────────────────────

    def test_normal_python_file(self):
        self.assertFalse(_is_noisy_evidence_path("src/main.py"))

    def test_normal_tf_file(self):
        self.assertFalse(_is_noisy_evidence_path("infra/main.tf"))

    def test_normal_nested_file(self):
        self.assertFalse(_is_noisy_evidence_path("openshard/review/terraform_checker.py"))

    def test_empty_string(self):
        self.assertFalse(_is_noisy_evidence_path(""))

    def test_windows_style_path(self):
        # Backslash-separated paths are normalised before splitting
        self.assertTrue(_is_noisy_evidence_path(".pytest_cache\\v\\cache\\foo"))


class TestBuildFileEvidenceFiltering(unittest.TestCase):
    def test_noisy_inspected_path_excluded(self):
        result = _build_file_evidence(
            inspected=[".pytest_cache/lastfailed", "src/main.py"],
            referenced=[],
            touched=[],
        )
        paths = [fe.path for fe in result]
        self.assertIn("src/main.py", paths)
        self.assertNotIn(".pytest_cache/lastfailed", paths)

    def test_all_noisy_dirs_excluded_from_inspected(self):
        noisy = [
            ".git/config",
            "__pycache__/mod.pyc",
            ".venv/lib/foo.py",
            "venv/bin/python",
            "node_modules/pkg/index.js",
            "dist/bundle.js",
            "build/output.html",
            ".mypy_cache/3.11/x.data",
            ".pytest_cache/v/cache/lastfailed",
        ]
        result = _build_file_evidence(inspected=noisy, referenced=[], touched=[])
        self.assertEqual(result, [], f"Expected all filtered, got: {result}")

    def test_finding_source_at_noisy_path_is_kept(self):
        # A finding source in a noisy dir is still a real finding — do not suppress it.
        result = _build_file_evidence(
            inspected=[], referenced=["node_modules/evil-pkg/index.js"], touched=[]
        )
        self.assertEqual(len(result), 1)
        self.assertIn("finding_source", result[0].roles)

    def test_changed_noisy_path_is_kept(self):
        result = _build_file_evidence(
            inspected=[], referenced=[], touched=[".git/config"]
        )
        self.assertEqual(len(result), 1)
        self.assertIn("changed", result[0].roles)

    def test_normal_inspected_path_retained(self):
        result = _build_file_evidence(
            inspected=["openshard/review/terraform_checker.py"],
            referenced=[],
            touched=[],
        )
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].path, "openshard/review/terraform_checker.py")
        self.assertIn("inspected", result[0].roles)

    def test_nested_pycache_excluded(self):
        result = _build_file_evidence(
            inspected=["services/api/__pycache__/x.pyc", "services/api/app.py"],
            referenced=[],
            touched=[],
        )
        paths = [fe.path for fe in result]
        self.assertNotIn("services/api/__pycache__/x.pyc", paths)
        self.assertIn("services/api/app.py", paths)


if __name__ == "__main__":
    unittest.main()
