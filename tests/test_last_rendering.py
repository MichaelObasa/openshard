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

    def test_default_shows_receipt_line(self):
        out = _render(_native_entry(), detail="default")
        self.assertIn("Receipt saved", out)

    def test_default_receipt_files_count(self):
        out = _render(_native_entry(), detail="default")
        self.assertIn("2 files changed", out)

    def test_default_receipt_verification_passed(self):
        out = _render(_native_entry(), detail="default")
        self.assertIn("Verification passed", out)

    def test_default_receipt_no_risky_writes(self):
        out = _render(_native_entry(), detail="default")
        self.assertIn("No risky writes", out)

    def test_default_receipt_not_shown_for_non_native(self):
        entry = {"task": "standard run", "workflow": "standard"}
        out = _render(entry, detail="default")
        self.assertNotIn("Receipt saved", out)

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


class TestNativeLoopTraceRendering(unittest.TestCase):
    """Tests for native_loop_trace extraction and rendering in saved-run inspection."""

    def _entry_with_trace(self, trace_events=None):
        entry = _native_entry()
        entry["native_loop_steps"] = ["repo_context", "observation", "plan", "generation"]
        if trace_events is not None:
            entry["native_loop_trace"] = trace_events
        return entry

    # 1
    def test_saved_run_tolerates_missing_native_loop_trace(self):
        from openshard.cli.run_output import _native_meta_from_entry
        entry = _native_entry()
        # no native_loop_trace key
        meta = _native_meta_from_entry(entry)
        self.assertIsNotNone(meta)
        trace = getattr(meta, "native_loop_trace", None)
        self.assertIsNotNone(trace)
        # Should be an empty list (raw, before _dict_to_ns promotion)
        if isinstance(trace, list):
            self.assertEqual(trace, [])
        else:
            self.assertEqual(getattr(trace, "events", []), [])

    # 2
    def test_saved_run_with_empty_native_loop_trace_no_trace_section(self):
        entry = self._entry_with_trace(trace_events=[])
        out = _render(entry, detail="full")
        self.assertNotIn("loop trace:", out)

    # 3
    def test_full_detail_renders_loop_trace(self):
        events = [
            {"phase": "repo_context", "status": "completed", "summary": "", "metadata": {}},
            {"phase": "observation", "status": "completed", "summary": "", "metadata": {"files": 12}},
            {"phase": "write", "status": "completed", "summary": "", "metadata": {"files": 2}},
        ]
        entry = self._entry_with_trace(trace_events=events)
        out = _render(entry, detail="full")
        self.assertIn("loop trace:", out)
        self.assertIn("repo_context [completed]", out)
        self.assertIn("observation [completed]", out)
        self.assertIn("write [completed]", out)

    # 4
    def test_default_detail_hides_loop_trace(self):
        events = [
            {"phase": "repo_context", "status": "completed", "summary": "", "metadata": {}},
        ]
        entry = self._entry_with_trace(trace_events=events)
        out = _render(entry, detail="default")
        self.assertNotIn("loop trace:", out)

    # 5
    def test_more_detail_hides_loop_trace(self):
        events = [
            {"phase": "plan", "status": "completed", "summary": "", "metadata": {}},
        ]
        entry = self._entry_with_trace(trace_events=events)
        out = _render(entry, detail="more")
        self.assertNotIn("loop trace:", out)

    # 6
    def test_native_meta_from_entry_extracts_loop_trace(self):
        from openshard.cli.run_output import _native_meta_from_entry
        events = [{"phase": "plan", "status": "completed", "summary": "", "metadata": {}}]
        entry = self._entry_with_trace(trace_events=events)
        meta = _native_meta_from_entry(entry)
        raw_trace = getattr(meta, "native_loop_trace", None)
        self.assertIsNotNone(raw_trace)
        # After _dict_to_ns, items may be SimpleNamespace
        if isinstance(raw_trace, list):
            self.assertEqual(len(raw_trace), 1)
        else:
            self.assertEqual(len(getattr(raw_trace, "events", [])), 1)


class TestReadSearchFindingsRendering(unittest.TestCase):
    """Tests for read/search: N findings line in [native] block."""

    def _entry_with_findings(self, findings: list[str]) -> dict:
        entry = _native_entry()
        entry["read_search_findings"] = findings
        entry["native_loop_steps"] = [
            "repo_context", "observation", "read_search", "plan", "generation"
        ]
        return entry

    def test_read_search_line_shown_when_findings_present(self):
        from openshard.cli.run_output import _native_meta_from_entry, _render_native_demo_block
        entry = self._entry_with_findings(["test-marker:tests/", "file:src/auth.py"])
        meta = _native_meta_from_entry(entry)
        lines = _render_native_demo_block(meta)
        self.assertIn("  read/search: 2 findings", lines)

    def test_read_search_line_absent_when_findings_empty(self):
        from openshard.cli.run_output import _native_meta_from_entry, _render_native_demo_block
        entry = self._entry_with_findings([])
        meta = _native_meta_from_entry(entry)
        lines = _render_native_demo_block(meta)
        joined = "\n".join(lines)
        self.assertNotIn("read/search:", joined)

    def test_read_search_line_absent_when_field_missing(self):
        from openshard.cli.run_output import _native_meta_from_entry, _render_native_demo_block
        entry = _native_entry()
        meta = _native_meta_from_entry(entry)
        lines = _render_native_demo_block(meta)
        joined = "\n".join(lines)
        self.assertNotIn("read/search:", joined)

    def test_read_search_count_reflects_actual_list_length(self):
        from openshard.cli.run_output import _native_meta_from_entry, _render_native_demo_block
        findings = [
            "test-marker:tests/a.py",
            "test-marker:tests/b.py",
            "package:pyproject.toml",
            "file:src/main.py",
            "file:src/utils.py",
        ]
        entry = self._entry_with_findings(findings)
        meta = _native_meta_from_entry(entry)
        lines = _render_native_demo_block(meta)
        self.assertIn("  read/search: 5 findings", lines)

    def test_render_block_via_full_render_shows_read_search(self):
        entry = self._entry_with_findings(["file:src/main.py"])
        out = _render(entry, detail="more")
        self.assertIn("read/search: 1 findings", out)

    def test_raw_finding_labels_not_shown_in_default_block(self):
        entry = self._entry_with_findings(["test-marker:tests/secret_config.py"])
        out = _render(entry, detail="more")
        self.assertNotIn("test-marker:tests/secret_config.py", out)
        self.assertIn("read/search:", out)


class TestPatchProposalRendering(unittest.TestCase):
    """Tests for proposal: N files line in [native] block."""

    def _entry_with_proposal(self, file_count: int) -> dict:
        entry = _native_entry()
        entry["patch_proposal"] = {
            "file_count": file_count,
            "files": [f"file{i}.py" for i in range(file_count)],
            "change_types": ["update"] * file_count,
            "summaries": ["a change"] * file_count,
            "warnings": [],
        }
        return entry

    def test_proposal_line_shown_when_present(self):
        from openshard.cli.run_output import _native_meta_from_entry, _render_native_demo_block
        entry = self._entry_with_proposal(2)
        meta = _native_meta_from_entry(entry)
        lines = _render_native_demo_block(meta)
        self.assertIn("  proposal: 2 files", lines)

    def test_proposal_line_shows_count_only(self):
        from openshard.cli.run_output import _native_meta_from_entry, _render_native_demo_block
        entry = self._entry_with_proposal(3)
        meta = _native_meta_from_entry(entry)
        lines = _render_native_demo_block(meta)
        joined = "\n".join(lines)
        self.assertIn("proposal: 3 files", joined)
        self.assertNotIn("file0.py", joined)
        self.assertNotIn("update", joined.split("proposal:")[1] if "proposal:" in joined else "")

    def test_saved_run_inspection_renders_proposal_count(self):
        entry = self._entry_with_proposal(1)
        out = _render(entry, detail="more")
        self.assertIn("proposal: 1 files", out)

    def test_old_entry_without_patch_proposal_does_not_crash(self):
        from openshard.cli.run_output import _native_meta_from_entry, _render_native_demo_block
        entry = _native_entry()
        entry.pop("patch_proposal", None)
        meta = _native_meta_from_entry(entry)
        lines = _render_native_demo_block(meta)
        joined = "\n".join(lines)
        self.assertNotIn("proposal:", joined)

    def test_proposal_metadata_contains_no_raw_content(self):
        from dataclasses import asdict
        from openshard.native.context import NativePatchProposal
        proposal = NativePatchProposal(
            file_count=1,
            files=["a.py"],
            change_types=["create"],
            summaries=["created"],
        )
        d = asdict(proposal)
        self.assertNotIn("content", d)


class TestVerificationCommandSummaryRendering(unittest.TestCase):
    """Tests for verification commands: N safe, N approval, N blocked line in [native] block."""

    def _entry_with_summary(self, command_count: int, safe_count: int, needs_approval_count: int, blocked_count: int) -> dict:
        entry = _native_entry()
        entry["verification_command_summary"] = {
            "attempted": True,
            "command_count": command_count,
            "safe_count": safe_count,
            "needs_approval_count": needs_approval_count,
            "blocked_count": blocked_count,
            "passed": True,
            "retried": False,
            "warnings": [],
        }
        return entry

    def test_renders_verification_command_counts(self):
        from openshard.cli.run_output import _native_meta_from_entry, _render_native_demo_block
        entry = self._entry_with_summary(2, 2, 0, 0)
        meta = _native_meta_from_entry(entry)
        lines = _render_native_demo_block(meta)
        joined = "\n".join(lines)
        self.assertIn("verification commands: 2 safe, 0 approval, 0 blocked", joined)

    def test_no_verification_commands_line_when_count_zero(self):
        from openshard.cli.run_output import _native_meta_from_entry, _render_native_demo_block
        entry = self._entry_with_summary(0, 0, 0, 0)
        meta = _native_meta_from_entry(entry)
        lines = _render_native_demo_block(meta)
        joined = "\n".join(lines)
        self.assertNotIn("verification commands:", joined)

    def test_old_entry_without_verification_command_summary_does_not_crash(self):
        from openshard.cli.run_output import _native_meta_from_entry, _render_native_demo_block
        entry = _native_entry()
        entry.pop("verification_command_summary", None)
        meta = _native_meta_from_entry(entry)
        lines = _render_native_demo_block(meta)
        self.assertIsInstance(lines, list)

    def test_saved_run_inspection_renders_command_counts(self):
        entry = self._entry_with_summary(3, 2, 1, 0)
        out = _render(entry, detail="more")
        self.assertIn("verification commands:", out)
        self.assertIn("2 safe", out)

    def test_no_command_strings_in_output(self):
        from openshard.cli.run_output import _native_meta_from_entry, _render_native_demo_block
        entry = self._entry_with_summary(1, 1, 0, 0)
        meta = _native_meta_from_entry(entry)
        lines = _render_native_demo_block(meta)
        joined = "\n".join(lines)
        self.assertNotIn("python -m pytest", joined)
        self.assertNotIn("argv", joined)


class TestCommandPolicyPreviewRendering(unittest.TestCase):
    """Tests for command policy: N safe, N approval, N blocked line in [native] block."""

    def _entry_with_preview(self, safe_count: int, needs_approval_count: int, blocked_count: int) -> dict:
        entry = _native_entry()
        entry["command_policy_preview"] = {
            "safe_count": safe_count,
            "needs_approval_count": needs_approval_count,
            "blocked_count": blocked_count,
            "command_classes": ["safe"] if safe_count > 0 else [],
            "warnings": [],
        }
        return entry

    def test_renders_counts(self):
        from openshard.cli.run_output import _native_meta_from_entry, _render_native_demo_block
        entry = self._entry_with_preview(2, 0, 0)
        meta = _native_meta_from_entry(entry)
        lines = _render_native_demo_block(meta)
        joined = "\n".join(lines)
        self.assertIn("command policy: 2 safe, 0 approval, 0 blocked", joined)

    def test_no_line_when_total_zero(self):
        from openshard.cli.run_output import _native_meta_from_entry, _render_native_demo_block
        entry = self._entry_with_preview(0, 0, 0)
        meta = _native_meta_from_entry(entry)
        lines = _render_native_demo_block(meta)
        joined = "\n".join(lines)
        self.assertNotIn("command policy:", joined)

    def test_old_entry_no_crash(self):
        from openshard.cli.run_output import _native_meta_from_entry, _render_native_demo_block
        entry = _native_entry()
        entry.pop("command_policy_preview", None)
        meta = _native_meta_from_entry(entry)
        lines = _render_native_demo_block(meta)
        self.assertIsInstance(lines, list)

    def test_saved_run_shows_counts(self):
        entry = self._entry_with_preview(3, 1, 0)
        out = _render(entry, detail="more")
        self.assertIn("command policy:", out)
        self.assertIn("3 safe", out)

    def test_no_command_strings_in_output(self):
        from openshard.cli.run_output import _native_meta_from_entry, _render_native_demo_block
        entry = self._entry_with_preview(1, 0, 0)
        meta = _native_meta_from_entry(entry)
        lines = _render_native_demo_block(meta)
        joined = "\n".join(lines)
        self.assertNotIn("python -m pytest", joined)
        self.assertNotIn("argv", joined)
class TestNativeContextPacketRendering(unittest.TestCase):
    """Tests for context packet: N sources, N paths line in [native] block."""

    def _entry_with_packet(self, sources: list[str], compact_paths: list[str]) -> dict:
        entry = _native_entry()
        entry["context_packet"] = {
            "task_preview": "fix the bug",
            "sources": sources,
            "repo_stack": [],
            "test_marker_count": 0,
            "package_file_count": 0,
            "read_search_count": len(compact_paths),
            "selected_skills": [],
            "backend": "builtin",
            "backend_available": True,
            "backend_proof_mode": "",
            "compact_paths": compact_paths,
            "warnings": [],
        }
        return entry

    def test_context_packet_line_rendered(self):
        from openshard.cli.run_output import _native_meta_from_entry, _render_native_demo_block
        entry = self._entry_with_packet(
            sources=["backend", "repo_context"],
            compact_paths=["file:src/x.py"],
        )
        meta = _native_meta_from_entry(entry)
        lines = _render_native_demo_block(meta)
        self.assertIn("  context packet: 2 sources, 1 paths", lines)

    def test_context_packet_not_rendered_when_missing(self):
        from openshard.cli.run_output import _native_meta_from_entry, _render_native_demo_block
        entry = _native_entry()
        entry.pop("context_packet", None)
        meta = _native_meta_from_entry(entry)
        lines = _render_native_demo_block(meta)
        joined = "\n".join(lines)
        self.assertNotIn("context packet:", joined)

    def test_context_packet_not_rendered_when_no_sources(self):
        from openshard.cli.run_output import _native_meta_from_entry, _render_native_demo_block
        entry = self._entry_with_packet(sources=[], compact_paths=[])
        meta = _native_meta_from_entry(entry)
        lines = _render_native_demo_block(meta)
        joined = "\n".join(lines)
        self.assertNotIn("context packet:", joined)

    def test_saved_run_inspection_passes_context_packet(self):
        from openshard.cli.run_output import _native_meta_from_entry
        entry = self._entry_with_packet(
            sources=["backend", "read_search"],
            compact_paths=["file:src/main.py", "test-marker:tests/"],
        )
        meta = _native_meta_from_entry(entry)
        cp = getattr(meta, "context_packet", None)
        self.assertIsNotNone(cp)
        self.assertEqual(getattr(cp, "sources", None), ["backend", "read_search"])

    def test_raw_paths_not_rendered_in_default_output(self):
        from openshard.cli.run_output import _native_meta_from_entry, _render_native_demo_block
        entry = self._entry_with_packet(
            sources=["backend", "read_search"],
            compact_paths=["file:src/main.py", "test-marker:tests/"],
        )
        meta = _native_meta_from_entry(entry)
        lines = _render_native_demo_block(meta)
        joined = "\n".join(lines)
        self.assertNotIn("file:src/main.py", joined)
        self.assertNotIn("test-marker:tests/", joined)

class TestNativeFileContextRendering(unittest.TestCase):
    """Rendering tests for NativeFileContext compact line."""

    def _base_entry(self):
        return {
            "workflow": "native",
            "executor": "native",
            "native_loop_steps": [],
            "native_loop_trace": [],
            "read_search_findings": [],
            "write_path": "pipeline",
        }

    def test_file_context_compact_line_rendered(self):
        entry = self._base_entry()
        entry["file_context"] = {
            "files_read": 3,
            "paths": ["a.py", "b.py", "c.py"],
            "total_chars": 4210,
            "truncated": False,
            "warnings": [],
        }
        out = _render(entry, detail="more")
        self.assertIn("file context: 3 files, 4210 chars", out)

    def test_file_context_not_rendered_when_zero_files(self):
        entry = self._base_entry()
        entry["file_context"] = {
            "files_read": 0,
            "paths": [],
            "total_chars": 0,
            "truncated": False,
            "warnings": [],
        }
        out = _render(entry, detail="more")
        self.assertNotIn("file context:", out)

    def test_old_entry_without_file_context_does_not_crash(self):
        entry = self._base_entry()
        try:
            out = _render(entry, detail="more")
        except Exception as exc:
            self.fail(f"Rendering raised {exc}")
        self.assertIsInstance(out, str)


class TestContextQualityScoreRendering(unittest.TestCase):

    def _base_entry(self):
        return {
            "workflow": "native",
            "executor": "native",
            "native_loop_steps": [],
            "native_loop_trace": [],
        }

    def test_context_quality_line_rendered(self):
        entry = self._base_entry()
        entry["context_quality_score"] = {
            "score": 70,
            "max_score": 100,
            "level": "good",
            "reasons": ["repo_context", "read_search"],
            "warnings": [],
        }
        out = _render(entry, detail="more")
        self.assertIn("context quality: good 70/100", out)

    def test_context_quality_absent_does_not_crash(self):
        entry = self._base_entry()
        try:
            out = _render(entry, detail="more")
        except Exception as exc:
            self.fail(f"Rendering raised {exc}")
        self.assertNotIn("context quality:", out)

    def test_context_quality_none_does_not_crash(self):
        entry = self._base_entry()
        entry["context_quality_score"] = None
        try:
            out = _render(entry, detail="more")
        except Exception as exc:
            self.fail(f"Rendering raised {exc}")
        self.assertNotIn("context quality:", out)

    def test_native_meta_from_entry_populates_context_quality_score(self):
        from openshard.cli.run_output import _native_meta_from_entry
        entry = self._base_entry()
        entry["context_quality_score"] = {
            "score": 45,
            "max_score": 100,
            "level": "fair",
            "reasons": ["backend"],
            "warnings": [],
        }
        meta = _native_meta_from_entry(entry)
        self.assertIsNotNone(meta)
        cqs = getattr(meta, "context_quality_score", None)
        self.assertIsNotNone(cqs)
        self.assertEqual(getattr(cqs, "score", None), 45)
        self.assertEqual(getattr(cqs, "level", None), "fair")


class TestContextQualityAdvisoryRendering(unittest.TestCase):

    def _base_entry(self):
        return {
            "workflow": "native",
            "executor": "native",
            "native_loop_steps": [],
            "native_loop_trace": [],
        }

    def test_advisory_line_rendered_when_present(self):
        entry = self._base_entry()
        entry["context_quality_advisory"] = {
            "level": "good",
            "recommendation": "context is good enough for normal generation",
            "should_block": False,
            "warnings": [],
        }
        out = _render(entry, detail="more")
        self.assertIn("context advisory: context is good enough for normal generation", out)

    def test_advisory_line_absent_when_missing(self):
        entry = self._base_entry()
        out = _render(entry, detail="more")
        self.assertNotIn("context advisory:", out)

    def test_advisory_line_absent_when_none(self):
        entry = self._base_entry()
        entry["context_quality_advisory"] = None
        out = _render(entry, detail="more")
        self.assertNotIn("context advisory:", out)

    def test_saved_run_inspection_passes_advisory_through(self):
        from openshard.cli.run_output import _native_meta_from_entry
        entry = self._base_entry()
        entry["context_quality_advisory"] = {
            "level": "weak",
            "recommendation": "context is weak; prefer smaller changes or gather more context",
            "should_block": False,
            "warnings": ["context packet may be insufficient for confident generation"],
        }
        meta = _native_meta_from_entry(entry)
        self.assertIsNotNone(meta)
        cqa = getattr(meta, "context_quality_advisory", None)
        self.assertIsNotNone(cqa)
        self.assertEqual(getattr(cqa, "level", None), "weak")
        self.assertIn("weak", getattr(cqa, "recommendation", ""))

    def test_warnings_not_rendered_by_default(self):
        entry = self._base_entry()
        entry["context_quality_advisory"] = {
            "level": "fair",
            "recommendation": "context is usable but may need cautious generation",
            "should_block": False,
            "warnings": ["consider smaller changes if the task is risky"],
        }
        out = _render(entry, detail="more")
        self.assertNotIn("consider smaller changes", out)
        self.assertIn("context advisory:", out)


class TestNativeChangeBudgetRendering(unittest.TestCase):

    def _base_entry(self):
        return {
            "workflow": "native",
            "executor": "native",
            "native_loop_steps": [],
            "native_loop_trace": [],
        }

    def test_change_budget_line_rendered(self):
        entry = self._base_entry()
        entry["change_budget"] = {
            "level": "good",
            "max_files": 3,
            "max_change_size": "normal",
            "guidance": "normal generation is acceptable; avoid unnecessary broad refactors",
            "warnings": [],
        }
        out = _render(entry, detail="more")
        self.assertIn("change budget: 3 files, normal", out)

    def test_change_budget_absent_when_missing(self):
        entry = self._base_entry()
        out = _render(entry, detail="more")
        self.assertNotIn("change budget:", out)

    def test_change_budget_absent_when_none(self):
        entry = self._base_entry()
        entry["change_budget"] = None
        out = _render(entry, detail="more")
        self.assertNotIn("change budget:", out)

    def test_native_meta_from_entry_passes_change_budget(self):
        from openshard.cli.run_output import _native_meta_from_entry
        entry = self._base_entry()
        entry["change_budget"] = {
            "level": "fair",
            "max_files": 2,
            "max_change_size": "small",
            "guidance": "prefer a cautious, focused change",
            "warnings": ["context is usable but not strong"],
        }
        meta = _native_meta_from_entry(entry)
        self.assertIsNotNone(meta)
        cb = getattr(meta, "change_budget", None)
        self.assertIsNotNone(cb)
        self.assertEqual(getattr(cb, "max_files", None), 2)
        self.assertEqual(getattr(cb, "level", None), "fair")

    def test_guidance_not_rendered_in_default_output(self):
        entry = self._base_entry()
        entry["change_budget"] = {
            "level": "weak",
            "max_files": 1,
            "max_change_size": "small",
            "guidance": "UNIQUE_GUIDANCE_STRING_NOT_IN_OUTPUT",
            "warnings": [],
        }
        out = _render(entry, detail="more")
        self.assertNotIn("UNIQUE_GUIDANCE_STRING_NOT_IN_OUTPUT", out)
        self.assertIn("change budget:", out)


class TestNativeChangeBudgetPreviewRendering(unittest.TestCase):
    """Tests for budget preview: N/M files, action line in [native] block."""

    def _base_entry(self):
        return {
            "workflow": "native",
            "executor": "native",
            "native_loop_steps": [],
            "native_loop_trace": [],
        }

    def _entry_with_preview(self, proposed_files: int, budget_max_files: int, action: str) -> dict:
        entry = self._base_entry()
        entry["change_budget_preview"] = {
            "budget_max_files": budget_max_files,
            "proposed_files": proposed_files,
            "within_budget": proposed_files <= budget_max_files,
            "would_exceed_budget": proposed_files > budget_max_files,
            "action": action,
            "warnings": (
                [f"proposal has {proposed_files} files but budget allows {budget_max_files}"]
                if proposed_files > budget_max_files
                else []
            ),
        }
        return entry

    def test_budget_preview_line_rendered(self):
        entry = self._entry_with_preview(proposed_files=4, budget_max_files=2, action="warn")
        out = _render(entry, detail="more")
        self.assertIn("budget preview:", out)
        self.assertIn("4/2 files", out)
        self.assertIn("warn", out)

    def test_budget_preview_absent_when_missing(self):
        entry = self._base_entry()
        out = _render(entry, detail="more")
        self.assertNotIn("budget preview:", out)

    def test_budget_preview_absent_when_none(self):
        entry = self._base_entry()
        entry["change_budget_preview"] = None
        out = _render(entry, detail="more")
        self.assertNotIn("budget preview:", out)

    def test_native_meta_from_entry_passes_budget_preview(self):
        from openshard.cli.run_output import _native_meta_from_entry
        entry = self._entry_with_preview(proposed_files=1, budget_max_files=3, action="allow")
        meta = _native_meta_from_entry(entry)
        self.assertIsNotNone(meta)
        cbp = getattr(meta, "change_budget_preview", None)
        self.assertIsNotNone(cbp)
        self.assertEqual(getattr(cbp, "proposed_files", None), 1)
        self.assertEqual(getattr(cbp, "budget_max_files", None), 3)
        self.assertEqual(getattr(cbp, "action", None), "allow")

    def test_warnings_not_rendered_by_default(self):
        entry = self._entry_with_preview(proposed_files=4, budget_max_files=2, action="warn")
        out = _render(entry, detail="more")
        self.assertNotIn("proposal has 4 files but budget allows 2", out)


class TestNativeChangeBudgetSoftGateRendering(unittest.TestCase):
    """Tests for budget gate: action, approval=bool line in [native] block."""

    def _base_entry(self):
        return {
            "workflow": "native",
            "executor": "native",
            "native_loop_steps": [],
            "native_loop_trace": [],
        }

    def _entry_with_gate(self, action: str, requires_approval: bool, warnings: list | None = None) -> dict:
        entry = self._base_entry()
        entry["change_budget_soft_gate"] = {
            "requires_approval": requires_approval,
            "reason": "proposal exceeds advisory change budget" if requires_approval else "within budget",
            "action": action,
            "warnings": warnings if warnings is not None else [],
        }
        return entry

    def test_soft_gate_line_rendered(self):
        entry = self._entry_with_gate(action="require_approval", requires_approval=True)
        out = _render(entry, detail="more")
        self.assertIn("budget gate:", out)
        self.assertIn("require_approval", out)
        self.assertIn("approval=true", out)

    def test_soft_gate_absent_when_missing(self):
        entry = self._base_entry()
        out = _render(entry, detail="more")
        self.assertNotIn("budget gate:", out)

    def test_soft_gate_absent_when_none(self):
        entry = self._base_entry()
        entry["change_budget_soft_gate"] = None
        out = _render(entry, detail="more")
        self.assertNotIn("budget gate:", out)

    def test_native_meta_from_entry_passes_soft_gate(self):
        from openshard.cli.run_output import _native_meta_from_entry
        entry = self._entry_with_gate(action="allow", requires_approval=False)
        meta = _native_meta_from_entry(entry)
        self.assertIsNotNone(meta)
        gate = getattr(meta, "change_budget_soft_gate", None)
        self.assertIsNotNone(gate)
        self.assertEqual(getattr(gate, "action", None), "allow")
        self.assertEqual(getattr(gate, "requires_approval", None), False)

    def test_warnings_not_rendered_by_default(self):
        entry = self._entry_with_gate(
            action="require_approval",
            requires_approval=True,
            warnings=["proposal has 4 files but budget allows 2"],
        )
        out = _render(entry, detail="more")
        self.assertNotIn("proposal has 4 files but budget allows 2", out)


class TestNativeApprovalRequestRendering(unittest.TestCase):
    """Tests for approval request: source, required=bool line in [native] block."""

    def _base_entry(self):
        return {
            "workflow": "native",
            "executor": "native",
            "native_loop_steps": [],
            "native_loop_trace": [],
        }

    def _entry_with_approval(self, requires_approval: bool, source: str = "change_budget_soft_gate") -> dict:
        entry = self._base_entry()
        entry["approval_request"] = {
            "source": source,
            "requires_approval": requires_approval,
            "reason": "proposal exceeds advisory change budget" if requires_approval else "within budget",
            "action": "require_approval" if requires_approval else "allow",
            "proposed_files": 4 if requires_approval else 1,
            "budget_max_files": 2,
            "prompt": (
                "Proposal exceeds advisory change budget: 4 files proposed, budget allows 2. Proceed?"
                if requires_approval else ""
            ),
            "warnings": [],
        }
        return entry

    def test_approval_request_line_rendered(self):
        entry = self._entry_with_approval(requires_approval=True)
        out = _render(entry, detail="more")
        self.assertIn("approval request:", out)
        self.assertIn("change_budget_soft_gate", out)
        self.assertIn("required=true", out)

    def test_approval_request_absent_when_missing(self):
        entry = self._base_entry()
        out = _render(entry, detail="more")
        self.assertNotIn("approval request:", out)

    def test_approval_request_absent_when_none(self):
        entry = self._base_entry()
        entry["approval_request"] = None
        out = _render(entry, detail="more")
        self.assertNotIn("approval request:", out)

    def test_native_meta_from_entry_passes_approval_request(self):
        from openshard.cli.run_output import _native_meta_from_entry
        entry = self._entry_with_approval(requires_approval=False)
        meta = _native_meta_from_entry(entry)
        self.assertIsNotNone(meta)
        approval = getattr(meta, "approval_request", None)
        self.assertIsNotNone(approval)
        self.assertEqual(getattr(approval, "source", None), "change_budget_soft_gate")
        self.assertEqual(getattr(approval, "requires_approval", None), False)

    def test_prompt_not_rendered_by_default(self):
        entry = self._entry_with_approval(requires_approval=True)
        out = _render(entry, detail="more")
        self.assertNotIn("Proceed?", out)

    def test_warnings_not_rendered_by_default(self):
        entry = self._base_entry()
        entry["approval_request"] = {
            "source": "change_budget_soft_gate",
            "requires_approval": True,
            "reason": "proposal exceeds advisory change budget",
            "action": "require_approval",
            "proposed_files": 4,
            "budget_max_files": 2,
            "prompt": "Proposal exceeds advisory change budget: 4 files proposed, budget allows 2. Proceed?",
            "warnings": ["proposal has 4 files but budget allows 2"],
        }
        out = _render(entry, detail="more")
        self.assertNotIn("proposal has 4 files but budget allows 2", out)


class TestNativeApprovalReceiptRendering(unittest.TestCase):
    """Tests for approval receipt: source, granted=bool line in [native] block."""

    def _base_entry(self):
        return {
            "workflow": "native",
            "executor": "native",
            "native_loop_steps": [],
            "native_loop_trace": [],
        }

    def _entry_with_receipt(self, granted: bool, source: str = "change_budget_soft_gate") -> dict:
        entry = self._base_entry()
        entry["approval_receipt"] = {
            "source": source,
            "requested": True,
            "granted": granted,
            "action": "require_approval" if not granted else "allow",
            "reason": "proposal exceeds advisory change budget",
        }
        return entry

    def test_receipt_line_rendered(self):
        entry = self._entry_with_receipt(granted=True)
        out = _render(entry, detail="more")
        self.assertIn("approval receipt:", out)
        self.assertIn("change_budget_soft_gate", out)
        self.assertIn("granted=true", out)

    def test_absent_when_missing(self):
        entry = self._base_entry()
        out = _render(entry, detail="more")
        self.assertNotIn("approval receipt:", out)

    def test_absent_when_none(self):
        entry = self._base_entry()
        entry["approval_receipt"] = None
        out = _render(entry, detail="more")
        self.assertNotIn("approval receipt:", out)

    def test_native_meta_from_entry_passes_receipt_through(self):
        from openshard.cli.run_output import _native_meta_from_entry
        entry = self._entry_with_receipt(granted=True)
        meta = _native_meta_from_entry(entry)
        self.assertIsNotNone(meta)
        receipt = getattr(meta, "approval_receipt", None)
        self.assertIsNotNone(receipt)
        self.assertEqual(getattr(receipt, "source", None), "change_budget_soft_gate")
        self.assertTrue(getattr(receipt, "granted", None))

    def test_reason_not_rendered_by_default(self):
        entry = self._entry_with_receipt(granted=True)
        out = _render(entry, detail="more")
        self.assertNotIn("proposal exceeds advisory change budget", out)


class TestBaselineLineInLastDefault(unittest.TestCase):
    """Baseline estimate line: native + default detail only."""

    def _entry_with_tokens(self, pt: int, ct: int, cost: float | None = None) -> dict:
        entry = _native_entry()
        entry["prompt_tokens"] = pt
        entry["completion_tokens"] = ct
        entry["total_tokens"] = pt + ct
        if cost is not None:
            entry["estimated_cost"] = cost
        return entry

    def test_baseline_shown_in_native_default_with_tokens(self):
        out = _render(self._entry_with_tokens(1_000_000, 1_000_000), detail="default")
        self.assertIn("Baseline estimate:", out)

    def test_baseline_contains_sonnet46(self):
        out = _render(self._entry_with_tokens(1_000_000, 1_000_000), detail="default")
        self.assertIn("Sonnet 4.6", out)

    def test_baseline_contains_gpt55(self):
        out = _render(self._entry_with_tokens(1_000_000, 1_000_000), detail="default")
        self.assertIn("GPT-5.5", out)

    def test_baseline_hidden_in_native_more(self):
        out = _render(self._entry_with_tokens(1_000_000, 1_000_000), detail="more")
        self.assertNotIn("Baseline estimate:", out)

    def test_baseline_hidden_in_native_full(self):
        out = _render(self._entry_with_tokens(1_000_000, 1_000_000), detail="full")
        self.assertNotIn("Baseline estimate:", out)

    def test_baseline_omitted_when_tokens_zero(self):
        out = _render(self._entry_with_tokens(0, 0), detail="default")
        self.assertNotIn("Baseline estimate:", out)

    def test_baseline_omitted_when_tokens_missing_from_entry(self):
        entry = {"task": "old run", "workflow": "native", "executor": "native"}
        out = _render(entry, detail="default")
        self.assertNotIn("Baseline estimate:", out)

    def test_baseline_omitted_for_non_native_entry(self):
        entry = {
            "task": "non-native run",
            "prompt_tokens": 1_000_000,
            "completion_tokens": 1_000_000,
        }
        out = _render(entry, detail="default")
        self.assertNotIn("Baseline estimate:", out)

    def test_baseline_appears_after_time_line(self):
        out = _render(self._entry_with_tokens(1_000_000, 1_000_000), detail="default")
        time_idx = out.index("Time:")
        baseline_idx = out.index("Baseline estimate:")
        self.assertGreater(baseline_idx, time_idx)

    def test_baseline_appears_before_receipt_line(self):
        out = _render(self._entry_with_tokens(1_000_000, 1_000_000), detail="default")
        baseline_idx = out.index("Baseline estimate:")
        receipt_idx = out.index("Receipt saved")
        self.assertLess(baseline_idx, receipt_idx)

    def test_multiplier_shown_when_cost_present(self):
        out = _render(self._entry_with_tokens(1_000_000, 1_000_000, cost=0.01), detail="default")
        self.assertIn("x higher", out)

    def test_multiplier_absent_when_no_cost(self):
        entry = self._entry_with_tokens(1_000_000, 1_000_000)
        entry.pop("estimated_cost", None)
        out = _render(entry, detail="default")
        self.assertNotIn("x higher", out)


class TestContextUsageRendering(unittest.TestCase):
    """Rendering tests for context usage, file context truncated, and context warnings."""

    def _base_entry(self):
        return {
            "workflow": "native",
            "executor": "native",
            "native_loop_steps": [],
            "native_loop_trace": [],
            "read_search_findings": [],
            "write_path": "pipeline",
        }

    def _entry_with_usage(
        self,
        used=True,
        evidence_items=0,
        snippet_files=0,
        fc_truncated=False,
        fc_files=0,
        cqs_warnings=None,
        cp_warnings=None,
        fc_warnings=None,
    ):
        entry = self._base_entry()
        entry["final_report"] = {
            "used_native_context": used,
            "observed_tools": [],
            "selected_skills": [],
            "plan_intent": None,
            "plan_risk": None,
            "evidence_items": evidence_items,
            "snippet_files": snippet_files,
            "verification_attempted": False,
            "verification_passed": False,
            "verification_retried": False,
            "diff_files": [],
            "added_lines": 0,
            "removed_lines": 0,
            "warnings": [],
        }
        if fc_files > 0 or fc_truncated or fc_warnings:
            entry["file_context"] = {
                "files_read": fc_files,
                "paths": [],
                "total_chars": 1000 * fc_files,
                "truncated": fc_truncated,
                "warnings": fc_warnings or [],
            }
        if cqs_warnings is not None:
            entry["context_quality_score"] = {
                "score": 50,
                "max_score": 100,
                "level": "fair",
                "reasons": [],
                "warnings": cqs_warnings,
            }
        if cp_warnings is not None:
            entry["context_packet"] = {
                "task_preview": "",
                "sources": ["repo_context"],
                "repo_stack": [],
                "test_marker_count": 0,
                "package_file_count": 0,
                "read_search_count": 0,
                "selected_skills": [],
                "backend": "builtin",
                "backend_available": True,
                "backend_proof_mode": "",
                "compact_paths": [],
                "file_context_files": 0,
                "warnings": cp_warnings,
            }
        return entry

    # --- context usage line ---

    def test_context_usage_line_rendered_when_used(self):
        entry = self._entry_with_usage(used=True)
        out = _render(entry, detail="more")
        self.assertIn("context usage: used=yes", out)

    def test_context_usage_line_rendered_when_not_used(self):
        entry = self._entry_with_usage(used=False)
        out = _render(entry, detail="more")
        self.assertIn("context usage: used=no", out)

    def test_context_usage_includes_evidence_count(self):
        entry = self._entry_with_usage(used=True, evidence_items=5)
        out = _render(entry, detail="more")
        self.assertIn("evidence=5 items", out)

    def test_context_usage_omits_zero_evidence(self):
        entry = self._entry_with_usage(used=True, evidence_items=0)
        out = _render(entry, detail="more")
        self.assertNotIn("evidence=0", out)

    def test_context_usage_includes_snippet_count(self):
        entry = self._entry_with_usage(used=True, snippet_files=2)
        out = _render(entry, detail="more")
        self.assertIn("2 snippets", out)

    def test_context_usage_omits_zero_snippets(self):
        entry = self._entry_with_usage(used=True, snippet_files=0)
        out = _render(entry, detail="more")
        self.assertNotIn("0 snippets", out)

    def test_context_usage_absent_when_no_final_report(self):
        entry = self._base_entry()
        out = _render(entry, detail="more")
        self.assertNotIn("context usage:", out)

    def test_context_usage_not_in_default_output(self):
        entry = self._entry_with_usage(used=True, evidence_items=3)
        out = _render(entry, detail="default")
        self.assertNotIn("context usage:", out)

    def test_saved_run_context_usage_reconstructed(self):
        from openshard.cli.run_output import _native_meta_from_entry, _render_native_demo_block
        entry = self._entry_with_usage(used=True, evidence_items=4, snippet_files=1)
        meta = _native_meta_from_entry(entry)
        lines = _render_native_demo_block(meta)
        joined = "\n".join(lines)
        self.assertIn("context usage: used=yes, evidence=4 items, 1 snippets", joined)

    # --- file context truncated ---

    def test_file_context_shows_truncated_flag(self):
        entry = self._entry_with_usage(fc_files=2, fc_truncated=True)
        out = _render(entry, detail="more")
        self.assertIn("file context: 2 files, 2000 chars, truncated", out)

    def test_file_context_no_truncated_flag_when_false(self):
        entry = self._entry_with_usage(fc_files=2, fc_truncated=False)
        out = _render(entry, detail="more")
        self.assertIn("file context: 2 files, 2000 chars", out)
        self.assertNotIn("truncated", out)

    # --- context warnings ---

    def test_context_warnings_count_shown_for_one(self):
        entry = self._entry_with_usage(cqs_warnings=["context packet may be insufficient"])
        out = _render(entry, detail="more")
        self.assertIn("context warnings: 1 warning", out)

    def test_context_warnings_count_shown_for_multiple(self):
        entry = self._entry_with_usage(
            cqs_warnings=["weak quality"],
            cp_warnings=["no repo context", "no skills"],
        )
        out = _render(entry, detail="more")
        self.assertIn("context warnings: 3 warnings", out)

    def test_context_warnings_raw_text_not_rendered(self):
        warning_text = "context packet may be insufficient for generation"
        entry = self._entry_with_usage(cqs_warnings=[warning_text])
        out = _render(entry, detail="more")
        self.assertNotIn(warning_text, out)

    def test_context_warnings_from_packet(self):
        entry = self._entry_with_usage(cp_warnings=["no repo context found"])
        out = _render(entry, detail="more")
        self.assertIn("context warnings: 1 warning", out)

    def test_context_warnings_from_file_context(self):
        entry = self._entry_with_usage(fc_files=1, fc_warnings=["read failed for one file"])
        out = _render(entry, detail="more")
        self.assertIn("context warnings: 1 warning", out)

    def test_context_warnings_absent_when_none(self):
        entry = self._entry_with_usage(
            cqs_warnings=[],
            cp_warnings=[],
            fc_warnings=[],
        )
        out = _render(entry, detail="more")
        self.assertNotIn("context warnings:", out)


class TestNativeFailureMemoryRendering(unittest.TestCase):
    def _base_entry(self) -> dict:
        return {
            "workflow": "native",
            "executor": "native",
        }

    def _entry_with_failure_memory(self, lessons: list[dict]) -> dict:
        entry = self._base_entry()
        entry["failure_memory"] = {
            "has_lessons": bool(lessons),
            "lessons": lessons,
        }
        return entry

    def test_no_block_when_failure_memory_absent(self):
        out = _render(self._base_entry(), detail="more")
        self.assertNotIn("failure memory:", out)

    def test_no_block_when_has_lessons_false(self):
        entry = self._base_entry()
        entry["failure_memory"] = {"has_lessons": False, "lessons": []}
        out = _render(entry, detail="more")
        self.assertNotIn("failure memory:", out)

    def test_block_shown_with_two_lessons(self):
        entry = self._entry_with_failure_memory([
            {"lesson_type": "weak_context", "reason": "score 20/100"},
            {"lesson_type": "missing_verification", "reason": "no commands"},
        ])
        out = _render(entry, detail="more")
        self.assertIn("failure memory:", out)
        self.assertIn("weak_context", out)
        self.assertIn("missing_verification", out)

    def test_block_shown_with_singular_lesson(self):
        entry = self._entry_with_failure_memory([
            {"lesson_type": "failed_verification", "reason": "exit code 1"},
        ])
        out = _render(entry, detail="more")
        self.assertIn("failure memory: failed_verification", out)

    def test_compact_rendering_shows_labels_not_reasons(self):
        entry = self._entry_with_failure_memory([
            {"lesson_type": "weak_context", "reason": "score 20/100"},
            {"lesson_type": "missing_verification", "reason": "no commands available"},
        ])
        out = _render(entry, detail="more")
        self.assertIn("weak_context", out)
        self.assertIn("missing_verification", out)
        self.assertNotIn("score 20/100", out)
        self.assertNotIn("no commands available", out)

    def test_full_rendering_shows_reasons(self):
        entry = self._entry_with_failure_memory([
            {"lesson_type": "weak_context", "reason": "score 20/100"},
        ])
        out = _render(entry, detail="full")
        self.assertIn("failure memory: 1 lesson", out)
        self.assertIn("weak_context: score 20/100", out)

    def test_all_known_lesson_types_render_as_labels(self):
        all_types = [
            "weak_context",
            "unknown_task_type",
            "failed_verification",
            "unsafe_command",
            "approval_required",
            "approval_rejected",
            "patch_too_broad",
            "missing_verification",
            "context_truncated",
            "warnings_present",
        ]
        lessons = [{"lesson_type": t, "reason": f"reason for {t}"} for t in all_types]
        entry = self._entry_with_failure_memory(lessons)
        out = _render(entry, detail="more")
        self.assertIn("failure memory:", out)
        for t in all_types:
            self.assertIn(t, out)

    def test_old_entry_without_failure_memory_is_safe(self):
        entry = {
            "workflow": "native",
            "executor": "native",
            "diff_review": {"changed_files": 1, "added_lines": 5, "removed_lines": 2, "truncated": False, "warnings": []},
        }
        out = _render(entry, detail="more")
        self.assertNotIn("failure memory:", out)


if __name__ == "__main__":
    unittest.main()
