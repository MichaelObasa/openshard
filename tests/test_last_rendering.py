from __future__ import annotations

import unittest

import click
from click.testing import CliRunner

from openshard.cli.main import _model_label, _render_log_entry


def _render(entry: dict, detail: str = "more") -> str:
    @click.command()
    def cmd():
        _render_log_entry(entry, detail)

    return CliRunner().invoke(cmd).output


class TestLastRoutingDisplay(unittest.TestCase):

    def test_routing_section_shown_with_fields(self):
        entry = {
            "task": "do a thing",
            "routing_category": "standard",
            "routing_candidate_count": 5,
            "routing_selected_model": "openrouter/fast-model",
            "routing_selected_provider": "openrouter",
            "routing_used_fallback": False,
        }
        out = _render(entry)
        self.assertIn("[routing] category: standard", out)
        self.assertIn("candidates: 5", out)
        self.assertIn(_model_label("openrouter/fast-model"), out)
        self.assertIn("openrouter", out)

    def test_routing_section_fallback(self):
        entry = {
            "task": "do a thing",
            "routing_category": "visual",
            "routing_candidate_count": 0,
            "routing_selected_model": None,
            "routing_selected_provider": None,
            "routing_used_fallback": True,
        }
        out = _render(entry)
        self.assertIn("[routing] category: visual", out)
        self.assertIn("fallback (keyword routing)", out)

    def test_routing_section_absent_without_fields(self):
        entry = {"task": "do a thing"}
        out = _render(entry)
        self.assertNotIn("[routing]", out)


class TestLastProfileDisplay(unittest.TestCase):

    def test_more_shows_profile_when_present(self):
        entry = {
            "task": "do a thing",
            "execution_profile": "native_deep",
            "execution_profile_reason": "security category",
        }
        out = _render(entry, detail="more")
        self.assertIn("[profile] native_deep - security category", out)

    def test_full_shows_profile_when_present(self):
        entry = {
            "task": "do a thing",
            "execution_profile": "native_light",
            "execution_profile_reason": "simple/safe task",
        }
        out = _render(entry, detail="full")
        self.assertIn("[profile] native_light - simple/safe task", out)

    def test_profile_without_reason(self):
        entry = {
            "task": "do a thing",
            "execution_profile": "native_swarm",
        }
        out = _render(entry, detail="more")
        self.assertIn("[profile] native_swarm", out)

    def test_no_crash_on_entry_without_profile_fields(self):
        entry = {"task": "do a thing"}
        out = _render(entry, detail="more")
        self.assertNotIn("[profile]", out)

    def test_no_crash_on_entry_without_profile_fields_full(self):
        entry = {"task": "do a thing"}
        out = _render(entry, detail="full")
        self.assertNotIn("[profile]", out)

    def test_default_detail_does_not_show_profile(self):
        entry = {
            "task": "do a thing",
            "execution_profile": "native_deep",
            "execution_profile_reason": "security category",
        }
        out = _render(entry, detail="default")
        self.assertNotIn("[profile]", out)


class TestLastVerificationDisplay(unittest.TestCase):

    _PLAN_ENTRY = {
        "task": "do a thing",
        "verification_plan": [
            {
                "name": "tests",
                "argv": ["python", "-m", "pytest"],
                "kind": "test",
                "source": "detected",
                "safety": "safe",
                "reason": "matches safe prefix: python -m pytest",
            }
        ],
    }

    def test_more_shows_verification_plan(self):
        out = _render(self._PLAN_ENTRY, detail="more")
        self.assertIn("[verification]", out)
        self.assertIn("safe", out)
        self.assertIn("detected", out)

    def test_full_shows_verification_plan(self):
        out = _render(self._PLAN_ENTRY, detail="full")
        self.assertIn("[verification]", out)
        self.assertIn("safe", out)
        self.assertIn("detected", out)

    def test_argv_rendered_space_joined(self):
        out = _render(self._PLAN_ENTRY, detail="more")
        self.assertIn("python -m pytest", out)

    def test_default_detail_hides_verification_plan(self):
        out = _render(self._PLAN_ENTRY, detail="default")
        self.assertNotIn("[verification]", out)

    def test_old_entry_without_plan_no_crash(self):
        entry = {"task": "old run"}
        out = _render(entry, detail="more")
        self.assertNotIn("[verification]", out)

    def test_config_source_shown(self):
        entry = {
            "task": "do a thing",
            "verification_plan": [
                {
                    "name": "tests",
                    "argv": ["pytest"],
                    "kind": "test",
                    "source": "config",
                    "safety": "safe",
                    "reason": "matches safe prefix: pytest",
                }
            ],
        }
        out = _render(entry, detail="more")
        self.assertIn("config", out)
        self.assertIn("pytest", out)


def _native_entry() -> dict:
    return {
        "task": "add feature",
        "workflow": "native",
        "executor": "native",
        "repo_context_summary": {
            "likely_stack_markers": ["python"],
            "test_markers": ["pytest"],
            "total_files": 30,
            "top_level_dirs": ["openshard"],
            "package_files": ["pyproject.toml"],
            "truncated": False,
        },
        "observation": {
            "dirty_diff_present": False,
            "search_matches_count": 3,
            "observed_tools": [],
            "verification_available": True,
            "warnings": [],
        },
        "plan": {"intent": "implementation", "risk": "medium", "suggested_steps": [], "warnings": []},
        "write_path": "pipeline",
        "verification_loop": {
            "attempted": True,
            "passed": True,
            "retried": True,
            "exit_code": 0,
            "output_chars": 200,
            "truncated": False,
        },
        "diff_review": {
            "has_diff": True,
            "changed_files": ["a.py", "b.py"],
            "added_lines": 34,
            "removed_lines": 8,
            "output_chars": 500,
            "truncated": False,
        },
        "final_report": {
            "used_native_context": True,
            "observed_tools": [],
            "selected_skills": ["repo mapping", "safe file editing"],
            "plan_intent": "implementation",
            "plan_risk": "medium",
            "evidence_items": 3,
            "snippet_files": 2,
            "verification_attempted": True,
            "verification_passed": True,
            "verification_retried": True,
            "diff_files": ["a.py", "b.py"],
            "added_lines": 34,
            "removed_lines": 8,
            "warnings": [],
        },
    }


class TestLastNativeInspection(unittest.TestCase):

    def test_more_renders_native_block(self):
        out = _render(_native_entry(), detail="more")
        self.assertIn("[native]", out)
        self.assertIn("repo:", out)
        self.assertIn("python", out)

    def test_more_renders_native_summary(self):
        out = _render(_native_entry(), detail="more")
        self.assertIn("[native summary]", out)
        self.assertIn("context: yes", out)
        self.assertIn("repo mapping", out)

    def test_full_renders_native_blocks(self):
        out = _render(_native_entry(), detail="full")
        self.assertIn("[native]", out)
        self.assertIn("[native summary]", out)

    def test_default_hides_native_blocks(self):
        out = _render(_native_entry(), detail="default")
        self.assertNotIn("[native]", out)
        self.assertNotIn("[native summary]", out)

    def test_non_native_entry_no_native_blocks(self):
        entry = {"task": "standard run", "workflow": "standard"}
        out = _render(entry, detail="more")
        self.assertNotIn("[native]", out)
        self.assertNotIn("[native summary]", out)

    def test_partial_native_metadata_no_crash(self):
        entry = {"task": "native run", "workflow": "native"}
        out = _render(entry, detail="more")
        self.assertNotIn("[native]", out)  # no useful metadata → no block
        self.assertNotIn("[native summary]", out)

    def test_partial_real_metadata_shows_native_block(self):
        entry = {
            "task": "native run",
            "workflow": "native",
            "repo_context_summary": {
                "likely_stack_markers": ["python"],
                "test_markers": [],
                "total_files": 10,
                "top_level_dirs": ["src"],
                "package_files": [],
                "truncated": False,
            },
        }
        out = _render(entry, detail="more")
        self.assertIn("[native]", out)
        self.assertIn("repo:", out)
        self.assertNotIn("[native summary]", out)

    def test_raw_output_not_printed(self):
        entry = _native_entry()
        entry["diff_review"]["output_chars"] = 9999
        out = _render(entry, detail="full")
        self.assertNotIn("a.py", out)
        self.assertNotIn("b.py", out)
        self.assertIn("2 files", out)

    def test_loop_steps_rendered_in_native_block(self):
        entry = _native_entry()
        entry["native_loop_steps"] = ["repo_context", "observation", "plan", "generation"]
        out = _render(entry, detail="more")
        self.assertIn("loop:", out)
        self.assertIn("repo_context -> observation -> plan -> generation", out)

    def test_empty_loop_steps_no_loop_line(self):
        entry = _native_entry()
        entry["native_loop_steps"] = []
        out = _render(entry, detail="more")
        self.assertNotIn("loop:", out)

    def test_non_native_entry_no_loop_steps_rendered(self):
        entry = {"task": "standard run", "workflow": "standard", "native_loop_steps": ["plan"]}
        out = _render(entry, detail="more")
        self.assertNotIn("loop:", out)

    def test_render_native_demo_block_shows_backend_builtin(self):
        from types import SimpleNamespace
        from openshard.cli.run_output import _render_native_demo_block

        meta = SimpleNamespace(
            repo_context_summary=SimpleNamespace(likely_stack_markers=["python"], test_markers=[]),
            native_backend="builtin",
            native_backend_available=True,
            native_backend_notes=[],
            observation=None,
            plan=None,
            write_path="pipeline",
            verification_loop=None,
            diff_review=None,
            native_loop_steps=[],
        )
        lines = _render_native_demo_block(meta)
        joined = "\n".join(lines)
        self.assertIn("backend: builtin", joined)

    def test_render_native_demo_block_shows_backend_unavailable(self):
        from types import SimpleNamespace
        from openshard.cli.run_output import _render_native_demo_block

        meta = SimpleNamespace(
            repo_context_summary=SimpleNamespace(likely_stack_markers=["python"], test_markers=[]),
            native_backend="deepagents",
            native_backend_available=False,
            native_backend_notes=["Install deepagents to enable this experimental backend."],
            observation=None,
            plan=None,
            write_path="pipeline",
            verification_loop=None,
            diff_review=None,
            native_loop_steps=[],
        )
        lines = _render_native_demo_block(meta)
        joined = "\n".join(lines)
        self.assertIn("backend: deepagents unavailable", joined)

    def test_render_native_demo_block_no_backend_field_no_error(self):
        from types import SimpleNamespace
        from openshard.cli.run_output import _render_native_demo_block

        meta = SimpleNamespace(
            repo_context_summary=SimpleNamespace(likely_stack_markers=["python"], test_markers=[]),
            observation=None,
            plan=None,
            write_path="pipeline",
            verification_loop=None,
            diff_review=None,
            native_loop_steps=[],
        )
        lines = _render_native_demo_block(meta)
        self.assertNotIn("backend:", "\n".join(lines))

    def test_backend_line_appears_after_repo_and_before_observation(self):
        from types import SimpleNamespace
        from openshard.cli.run_output import _render_native_demo_block

        meta = SimpleNamespace(
            repo_context_summary=SimpleNamespace(likely_stack_markers=["python"], test_markers=[]),
            native_backend="builtin",
            native_backend_available=True,
            native_backend_notes=[],
            observation=SimpleNamespace(dirty_diff_present=False, search_matches_count=0),
            plan=None,
            write_path="pipeline",
            verification_loop=None,
            diff_review=None,
            native_loop_steps=[],
        )
        lines = _render_native_demo_block(meta)
        repo_idx = next((i for i, ln in enumerate(lines) if "repo:" in ln), -1)
        backend_idx = next((i for i, ln in enumerate(lines) if "backend:" in ln), -1)
        observation_idx = next((i for i, ln in enumerate(lines) if "observation:" in ln), -1)
        self.assertGreater(backend_idx, repo_idx)
        self.assertLess(backend_idx, observation_idx)

    def test_native_meta_from_entry_passes_backend_fields(self):
        entry = _native_entry()
        entry["native_backend"] = "deepagents"
        entry["native_backend_available"] = False
        entry["native_backend_notes"] = ["Install deepagents to enable this experimental backend."]
        out = _render(entry, detail="more")
        self.assertIn("backend: deepagents unavailable", out)

    def test_render_readonly_proof_line(self):
        from openshard.cli.run_output import _render_native_demo_block
        from openshard.native.executor import NativeRunMeta
        meta = NativeRunMeta()
        meta.native_backend = "deepagents"
        meta.native_backend_available = True
        meta.native_backend_proof = {"mode": "readonly_agent_proof"}
        lines = _render_native_demo_block(meta)
        joined = "\n".join(lines)
        self.assertIn("proof: readonly agent proof", joined)

    def test_render_unconfigured_proof_line(self):
        from openshard.cli.run_output import _render_native_demo_block
        from openshard.native.executor import NativeRunMeta
        meta = NativeRunMeta()
        meta.native_backend = "deepagents"
        meta.native_backend_available = True
        meta.native_backend_proof = {"mode": "readonly_agent_unconfigured"}
        lines = _render_native_demo_block(meta)
        joined = "\n".join(lines)
        self.assertIn("proof: readonly agent unconfigured", joined)

    def test_native_meta_from_entry_includes_proof(self):
        from openshard.cli.run_output import _native_meta_from_entry
        entry = {
            "workflow": "native",
            "native_backend": "deepagents",
            "native_backend_available": True,
            "native_backend_notes": [],
            "native_backend_proof": {"mode": "readonly_agent_proof", "summary": "ok"},
        }
        meta = _native_meta_from_entry(entry)
        self.assertIsNotNone(meta.native_backend_proof)
        self.assertEqual(meta.native_backend_proof.mode, "readonly_agent_proof")


if __name__ == "__main__":
    unittest.main()
