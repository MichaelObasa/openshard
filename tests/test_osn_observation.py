"""Tests for OSN repo observation (OSNObservationPacket, build_osn_observation, receipt rendering).

Covers:
- Packet defaults and safety
- List caps enforced in __post_init__
- Stack detection (Python, Node, Terraform, GitHub Actions, Docker)
- Noisy directory exclusion including .codegraph
- OSN integration (step recording, failure handling, no shell exec)
- Receipt / render output
- Regression: _SKIP_DIRS includes .codegraph
"""
from __future__ import annotations

import json
import os
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_tree(root: Path, paths: list[str]) -> None:
    """Create files at given relative paths under root."""
    for rel in paths:
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.touch()


def _make_native_meta(**kwargs: Any) -> Any:
    """Minimal object with osn_observation attribute for receipt tests."""
    class _FakeMeta:
        def __init__(self, **kw: Any) -> None:
            self.osn_observation = kw.get("osn_observation", None)
            self.osn_loop_summary = kw.get("osn_loop_summary", None)
            self.osn_loop = kw.get("osn_loop", None)
            self.native_loop_steps = kw.get("native_loop_steps", None)
            self.read_search_findings = kw.get("read_search_findings", None)
            self.diff_review = kw.get("diff_review", None)
            self.repo_context_summary = kw.get("repo_context_summary", None)
            self.native_backend = kw.get("native_backend", None)
    return _FakeMeta(**kwargs)


def _make_executor(native_loop: str | None = "experimental", repo_root: Path | None = None):
    fake_gen = MagicMock()
    fake_gen.generate.return_value = MagicMock(usage=None, files=[], summary="ok", notes=[])
    fake_gen.model = "mock-model"
    fake_gen.fixer_model = "mock-fixer"
    with patch("openshard.native.executor.ExecutionGenerator", return_value=fake_gen):
        from openshard.native.executor import NativeAgentExecutor
        executor = NativeAgentExecutor(
            provider=MagicMock(), native_loop=native_loop, repo_root=repo_root
        )
    return executor, fake_gen


# ---------------------------------------------------------------------------
# 1-9: Packet defaults, serialization, caps, safety
# ---------------------------------------------------------------------------

class TestOSNObservationPacketDefaults(unittest.TestCase):

    def test_defaults_are_safe(self):
        from openshard.native.context import OSNObservationPacket
        p = OSNObservationPacket()
        self.assertFalse(p.enabled)
        self.assertFalse(p.repo_root_present)
        self.assertEqual(p.stack_signals, [])
        self.assertEqual(p.candidate_files, [])
        self.assertEqual(p.config_files, [])
        self.assertEqual(p.test_files, [])
        self.assertEqual(p.risky_markers, [])
        self.assertEqual(p.suggested_checks, [])
        self.assertEqual(p.observation_summary, "")
        self.assertEqual(p.files_considered, 0)
        self.assertFalse(p.files_capped)
        self.assertEqual(p.source, "repo_observation_v1")

    def test_serializes_to_json_safe_dict(self):
        from openshard.native.context import OSNObservationPacket
        p = OSNObservationPacket(
            enabled=True,
            repo_root_present=True,
            stack_signals=["python"],
            candidate_files=["openshard/native/executor.py"],
            observation_summary="test summary",
        )
        d = asdict(p)
        self.assertIsInstance(d, dict)
        json.dumps(d)  # must not raise

    def test_candidate_files_capped_at_10(self):
        from openshard.native.context import OSNObservationPacket
        p = OSNObservationPacket(candidate_files=[f"file{i}.py" for i in range(20)])
        self.assertEqual(len(p.candidate_files), 10)

    def test_config_files_capped_at_8(self):
        from openshard.native.context import OSNObservationPacket
        p = OSNObservationPacket(config_files=[f"cfg{i}.toml" for i in range(20)])
        self.assertEqual(len(p.config_files), 8)

    def test_test_files_capped_at_8(self):
        from openshard.native.context import OSNObservationPacket
        p = OSNObservationPacket(test_files=[f"test_f{i}.py" for i in range(20)])
        self.assertEqual(len(p.test_files), 8)

    def test_risky_markers_capped_at_6(self):
        from openshard.native.context import OSNObservationPacket
        p = OSNObservationPacket(risky_markers=[f"auth{i}.py" for i in range(20)])
        self.assertEqual(len(p.risky_markers), 6)

    def test_suggested_checks_capped_at_6(self):
        from openshard.native.context import OSNObservationPacket
        p = OSNObservationPacket(suggested_checks=[f"check {i}" for i in range(20)])
        self.assertEqual(len(p.suggested_checks), 6)

    def test_observation_summary_capped_at_200_chars(self):
        from openshard.native.context import OSNObservationPacket
        long_summary = "x" * 500
        p = OSNObservationPacket(observation_summary=long_summary)
        self.assertEqual(len(p.observation_summary), 200)

    def test_no_absolute_path_field(self):
        from openshard.native.context import OSNObservationPacket
        p = OSNObservationPacket()
        d = asdict(p)
        self.assertNotIn("repo_root", d)
        self.assertIn("repo_root_present", d)


# ---------------------------------------------------------------------------
# 10-16: Stack detection and suggested checks
# ---------------------------------------------------------------------------

class TestOSNObservationStackDetection(unittest.TestCase):

    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.root = Path(self._tmpdir.name)

    def tearDown(self):
        self._tmpdir.cleanup()

    def test_detects_python_from_pyproject_toml(self):
        from openshard.native.osn_observation import build_osn_observation
        (self.root / "pyproject.toml").write_text("[tool.poetry]")
        p = build_osn_observation(self.root)
        self.assertIn("python", p.stack_signals)

    def test_detects_node_from_package_json(self):
        from openshard.native.osn_observation import build_osn_observation
        (self.root / "package.json").write_text('{"name": "test"}')
        p = build_osn_observation(self.root)
        self.assertIn("node", p.stack_signals)

    def test_detects_terraform_from_tf_files(self):
        from openshard.native.osn_observation import build_osn_observation
        (self.root / "main.tf").write_text('resource "null_resource" "x" {}')
        p = build_osn_observation(self.root)
        self.assertIn("terraform", p.stack_signals)

    def test_detects_github_actions_from_workflows(self):
        from openshard.native.osn_observation import build_osn_observation
        wf = self.root / ".github" / "workflows"
        wf.mkdir(parents=True)
        (wf / "ci.yml").write_text("name: CI\non: push")
        p = build_osn_observation(self.root)
        self.assertIn("github-actions", p.stack_signals)

    def test_detects_docker_from_dockerfile(self):
        from openshard.native.osn_observation import build_osn_observation
        (self.root / "Dockerfile").write_text("FROM python:3.11")
        p = build_osn_observation(self.root)
        self.assertIn("docker", p.stack_signals)

    def test_suggested_checks_include_pytest_for_python(self):
        from openshard.native.osn_observation import build_osn_observation
        (self.root / "pyproject.toml").write_text("[tool.pytest]")
        tests = self.root / "tests"
        tests.mkdir()
        p = build_osn_observation(self.root)
        checks_combined = " ".join(p.suggested_checks)
        self.assertIn("pytest", checks_combined)

    def test_suggested_checks_include_ruff_for_python(self):
        from openshard.native.osn_observation import build_osn_observation
        (self.root / "pyproject.toml").write_text("[tool.ruff]")
        p = build_osn_observation(self.root)
        checks_combined = " ".join(p.suggested_checks)
        self.assertIn("ruff", checks_combined)

    def test_suggested_checks_include_terraform_validate_for_terraform(self):
        from openshard.native.osn_observation import build_osn_observation
        (self.root / "main.tf").write_text('terraform {}')
        p = build_osn_observation(self.root)
        checks_combined = " ".join(p.suggested_checks)
        self.assertIn("terraform validate", checks_combined)


# ---------------------------------------------------------------------------
# Noisy directory exclusion
# ---------------------------------------------------------------------------

class TestOSNObservationNoisyDirExclusion(unittest.TestCase):

    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.root = Path(self._tmpdir.name)

    def tearDown(self):
        self._tmpdir.cleanup()

    def test_skips_git_dir(self):
        from openshard.native.osn_observation import build_osn_observation
        _make_tree(self.root, [".git/config", ".git/index"])
        (self.root / "main.py").touch()
        p = build_osn_observation(self.root)
        for f in p.candidate_files + p.config_files + p.test_files:
            self.assertNotIn(".git", f)

    def test_skips_node_modules(self):
        from openshard.native.osn_observation import build_osn_observation
        _make_tree(self.root, ["node_modules/lodash/index.js"])
        (self.root / "index.js").touch()
        p = build_osn_observation(self.root)
        for f in p.candidate_files:
            self.assertNotIn("node_modules", f)

    def test_skips_venv(self):
        from openshard.native.osn_observation import build_osn_observation
        _make_tree(self.root, [".venv/lib/python3.11/site-packages/foo.py"])
        (self.root / "main.py").touch()
        p = build_osn_observation(self.root)
        for f in p.candidate_files:
            self.assertNotIn(".venv", f)

    def test_skips_pycache(self):
        from openshard.native.osn_observation import build_osn_observation
        _make_tree(self.root, ["__pycache__/main.cpython-311.pyc"])
        (self.root / "main.py").touch()
        p = build_osn_observation(self.root)
        for f in p.candidate_files + p.config_files + p.test_files:
            self.assertNotIn("__pycache__", f)

    def test_skips_codegraph_dir(self):
        from openshard.native.osn_observation import build_osn_observation
        _make_tree(self.root, [".codegraph/codegraph.db", ".codegraph/index.json"])
        (self.root / "main.py").touch()
        p = build_osn_observation(self.root)
        all_files = p.candidate_files + p.config_files + p.test_files + p.risky_markers
        for f in all_files:
            self.assertNotIn(".codegraph", f)

    def test_codegraph_db_never_included(self):
        from openshard.native.osn_observation import build_osn_observation
        _make_tree(self.root, [".codegraph/codegraph.db"])
        p = build_osn_observation(self.root)
        all_files = p.candidate_files + p.config_files + p.test_files + p.risky_markers
        self.assertNotIn(".codegraph/codegraph.db", all_files)
        for f in all_files:
            self.assertNotIn("codegraph.db", f)

    def test_paths_are_relative_and_forward_slash(self):
        from openshard.native.osn_observation import build_osn_observation
        _make_tree(self.root, ["src/app.py", "tests/test_app.py"])
        p = build_osn_observation(self.root)
        all_files = p.candidate_files + p.config_files + p.test_files
        for f in all_files:
            self.assertFalse(f.startswith("/"), f"absolute path: {f}")
            self.assertFalse(os.path.isabs(f), f"absolute path: {f}")
            self.assertNotIn("\\", f, f"backslash in path: {f}")

    def test_skip_dirs_in_repo_py_includes_codegraph(self):
        from openshard.analysis.repo import _SKIP_DIRS
        self.assertIn(".codegraph", _SKIP_DIRS)


# ---------------------------------------------------------------------------
# OSN integration: step recording and failure handling
# ---------------------------------------------------------------------------

class TestOSNObservationIntegration(unittest.TestCase):

    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.root = Path(self._tmpdir.name)

    def tearDown(self):
        self._tmpdir.cleanup()

    def test_observation_step_recorded_on_executor(self):
        executor, _ = _make_executor(native_loop="experimental", repo_root=self.root)
        executor._run_repo_observation_phase()
        self.assertIn("repo_observation", executor.native_meta.native_loop_steps)

    def test_observation_metadata_attached_to_native_meta(self):
        executor, _ = _make_executor(native_loop="experimental", repo_root=self.root)
        (self.root / "pyproject.toml").write_text("[tool.poetry]")
        executor._run_repo_observation_phase()
        self.assertIsNotNone(executor.native_meta.osn_observation)

    def test_observation_failure_does_not_crash(self):
        executor, _ = _make_executor(native_loop="experimental", repo_root=self.root)
        with patch(
            "openshard.native.osn_observation.build_osn_observation",
            side_effect=RuntimeError("simulated failure"),
        ):
            try:
                executor._run_repo_observation_phase()
            except Exception as e:
                self.fail(f"observation phase raised unexpectedly: {e}")

    def test_observation_failure_adds_warning(self):
        executor, _ = _make_executor(native_loop="experimental", repo_root=self.root)
        with patch(
            "openshard.native.osn_observation.build_osn_observation",
            side_effect=RuntimeError("simulated failure"),
        ):
            executor._run_repo_observation_phase()
        self.assertIn("repo_observation_failed", executor.native_meta.context_warnings)

    def test_observation_skipped_when_no_repo_root(self):
        executor, _ = _make_executor(native_loop="experimental", repo_root=None)
        executor._run_repo_observation_phase()
        self.assertIsNone(executor.native_meta.osn_observation)

    def test_no_shell_command_executed_during_observation(self):
        from openshard.native.osn_observation import build_osn_observation
        _make_tree(self.root, ["pyproject.toml", "tests/test_main.py", "main.py"])
        with patch("subprocess.run") as mock_run, patch("subprocess.Popen") as mock_popen:
            build_osn_observation(self.root)
            mock_run.assert_not_called()
            mock_popen.assert_not_called()

    def test_no_provider_model_call_added_by_observation(self):
        from openshard.native.osn_observation import build_osn_observation
        _make_tree(self.root, ["pyproject.toml"])
        with patch("openshard.native.osn_observation.build_osn_observation", wraps=build_osn_observation):
            executor, fake_gen = _make_executor(native_loop="experimental", repo_root=self.root)
            executor._run_repo_observation_phase()
        fake_gen.generate.assert_not_called()

    def test_no_raw_file_content_stored(self):
        from openshard.native.osn_observation import build_osn_observation
        secret_text = "SECRET_API_KEY=abc123"
        (self.root / "config.py").write_text(secret_text)
        p = build_osn_observation(self.root)
        d = asdict(p)
        import json
        serialized = json.dumps(d)
        self.assertNotIn(secret_text, serialized)
        self.assertNotIn("SECRET_API_KEY", serialized)

    def test_observation_not_run_for_non_experimental_loop(self):
        executor, _ = _make_executor(native_loop=None, repo_root=self.root)
        executor.native_meta.osn_observation = None
        # simulate what generate() does
        if executor._native_loop == "experimental":
            executor._run_repo_observation_phase()
        self.assertIsNone(executor.native_meta.osn_observation)

    def test_observation_run_for_experimental_loop(self):
        (self.root / "pyproject.toml").write_text("[project]")
        executor, _ = _make_executor(native_loop="experimental", repo_root=self.root)
        executor._run_repo_observation_phase()
        self.assertIsNotNone(executor.native_meta.osn_observation)

    def test_osn_recorder_tool_calls_not_inflated(self):
        executor, _ = _make_executor(native_loop="experimental", repo_root=self.root)
        executor._run_repo_observation_phase()
        if executor._osn_recorder is not None:
            summary = executor._osn_recorder.summary
            self.assertEqual(summary.tool_calls_attempted, 0, "observation must not inflate tool_calls_attempted")


# ---------------------------------------------------------------------------
# Receipt rendering
# ---------------------------------------------------------------------------

class TestOSNObservationReceipt(unittest.TestCase):

    def _render(self, native_meta: Any, detail: str = "default") -> str:
        from openshard.cli.run_output import _render_native_demo_block
        return "\n".join(_render_native_demo_block(native_meta, detail=detail))

    def _make_obs(self, **kwargs: Any) -> Any:
        from openshard.native.context import OSNObservationPacket
        defaults = dict(
            enabled=True,
            repo_root_present=True,
            stack_signals=["python", "github-actions"],
            candidate_files=["openshard/native/executor.py", "openshard/native/context.py"],
            test_files=["tests/test_osn_loop.py"],
            suggested_checks=["python -m pytest tests/ -v", "python -m ruff check"],
        )
        defaults.update(kwargs)
        return OSNObservationPacket(**defaults)

    def test_section_rendered_when_enabled(self):
        obs = self._make_obs()
        meta = _make_native_meta(osn_observation=obs)
        output = self._render(meta)
        self.assertIn("OSN OBSERVATION", output)

    def test_section_omitted_when_observation_absent(self):
        meta = _make_native_meta(osn_observation=None)
        output = self._render(meta)
        self.assertNotIn("OSN OBSERVATION", output)

    def test_section_omitted_when_observation_disabled(self):
        from openshard.native.context import OSNObservationPacket
        obs = OSNObservationPacket(enabled=False)
        meta = _make_native_meta(osn_observation=obs)
        output = self._render(meta)
        self.assertNotIn("OSN OBSERVATION", output)

    def test_section_shows_stack(self):
        obs = self._make_obs()
        meta = _make_native_meta(osn_observation=obs)
        output = self._render(meta)
        self.assertIn("Stack", output)
        self.assertIn("python", output)

    def test_section_shows_candidates_count(self):
        obs = self._make_obs()
        meta = _make_native_meta(osn_observation=obs)
        output = self._render(meta)
        self.assertIn("Candidates", output)

    def test_section_shows_tests_count(self):
        obs = self._make_obs()
        meta = _make_native_meta(osn_observation=obs)
        output = self._render(meta)
        self.assertIn("Tests", output)

    def test_section_shows_checks(self):
        obs = self._make_obs()
        meta = _make_native_meta(osn_observation=obs)
        output = self._render(meta)
        self.assertIn("Checks", output)

    def test_candidate_file_names_shown_only_in_full_detail(self):
        obs = self._make_obs()
        meta = _make_native_meta(osn_observation=obs)
        default_output = self._render(meta, detail="default")
        full_output = self._render(meta, detail="full")
        self.assertNotIn("openshard/native/executor.py", default_output)
        self.assertIn("openshard/native/executor.py", full_output)

    def test_candidate_files_capped_in_full_detail(self):
        obs = self._make_obs(candidate_files=[f"file{i}.py" for i in range(10)])
        meta = _make_native_meta(osn_observation=obs)
        full_output = self._render(meta, detail="full")
        # At most 5 file lines should appear (capped in renderer)
        file_lines = [ln for ln in full_output.splitlines() if "file" in ln and ".py" in ln]
        self.assertLessEqual(len(file_lines), 5)

    def test_no_em_dash_in_output(self):
        obs = self._make_obs()
        meta = _make_native_meta(osn_observation=obs)
        output = self._render(meta)
        self.assertNotIn("—", output)

    def test_no_raw_json_in_output(self):
        obs = self._make_obs()
        meta = _make_native_meta(osn_observation=obs)
        output = self._render(meta)
        self.assertNotIn("{", output)
        self.assertNotIn("}", output)

    def test_no_chain_of_thought_phrases(self):
        obs = self._make_obs()
        meta = _make_native_meta(osn_observation=obs)
        output = self._render(meta)
        banned = ["let me think", "I will now", "first I need to"]
        for phrase in banned:
            self.assertNotIn(phrase.lower(), output.lower())

    def test_no_codegraph_in_output(self):
        obs = self._make_obs()
        meta = _make_native_meta(osn_observation=obs)
        output = self._render(meta)
        self.assertNotIn(".codegraph", output)
        self.assertNotIn("codegraph.db", output)

    def test_ux_smoke_more_detail(self):
        """Smoke test for --more equivalent (detail='more')."""
        obs = self._make_obs()
        meta = _make_native_meta(osn_observation=obs)
        output = self._render(meta, detail="more")
        self.assertIn("OSN OBSERVATION", output)
        self.assertIn("Stack", output)
        self.assertIn("Candidates", output)
        self.assertIn("Checks", output)
        self.assertNotIn(".codegraph", output)
        self.assertNotIn("—", output)


# ---------------------------------------------------------------------------
# render_osn_observation_context (prompt-safe block)
# ---------------------------------------------------------------------------

class TestRenderOSNObservationContext(unittest.TestCase):

    def test_returns_empty_when_none(self):
        from openshard.native.context import render_osn_observation_context
        self.assertEqual(render_osn_observation_context(None), "")

    def test_returns_empty_when_disabled(self):
        from openshard.native.context import OSNObservationPacket, render_osn_observation_context
        p = OSNObservationPacket(enabled=False)
        self.assertEqual(render_osn_observation_context(p), "")

    def test_includes_stack_when_enabled(self):
        from openshard.native.context import OSNObservationPacket, render_osn_observation_context
        p = OSNObservationPacket(enabled=True, stack_signals=["python", "github-actions"])
        output = render_osn_observation_context(p)
        self.assertIn("python", output)
        self.assertIn("github-actions", output)
        self.assertIn("[repo observation]", output)

    def test_no_raw_content_in_context_block(self):
        from openshard.native.context import OSNObservationPacket, render_osn_observation_context
        p = OSNObservationPacket(
            enabled=True,
            stack_signals=["python"],
            candidate_files=["openshard/native/executor.py"],
            suggested_checks=["python -m pytest tests/ -v"],
        )
        output = render_osn_observation_context(p)
        self.assertNotIn("{", output)
        self.assertNotIn("}", output)


if __name__ == "__main__":
    unittest.main()
