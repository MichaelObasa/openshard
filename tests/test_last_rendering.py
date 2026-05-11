from __future__ import annotations

import unittest
from dataclasses import asdict

import click
from click.testing import CliRunner

from openshard.cli.main import _model_label, _render_log_entry
from openshard.native.context import (
    NativeContextQualityScore,
    NativeModelCandidateScore,
    NativeModelCandidateScoring,
    NativeModelRoleDecision,
    NativeModelSelectionDecision,
    NativeObservation,
    NativePlan,
    NativeRunTrustFactor,
    NativeRunTrustScore,
    NativeValidationContract,
    build_native_context_provenance,
    build_native_model_candidate_scoring,
)
from openshard.cli.run_output import _native_meta_from_entry
from openshard.native.context import build_native_model_policy, NativeModelPolicyReceipt


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
        self.assertIn("Category: standard", out)
        self.assertIn("Initial candidate:", out)
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
        self.assertIn("Category: visual", out)
        self.assertIn("Initial candidate: fallback (keyword routing)", out)

    def test_routing_section_absent_without_fields(self):
        entry = {"task": "do a thing"}
        out = _render(entry)
        self.assertNotIn("[routing]", out)

    def test_routing_no_tier_dispatch_note_without_receipt(self):
        entry = {
            "task": "do a thing",
            "routing_category": "standard",
            "routing_selected_model": "openrouter/fast-model",
            "routing_selected_provider": "openrouter",
            "routing_used_fallback": False,
        }
        out = _render(entry)
        self.assertNotIn("tier dispatch changed", out)

    def test_routing_tier_dispatch_applied_shows_note(self):
        entry = {
            "task": "do a thing",
            "routing_category": "standard",
            "routing_selected_model": "openrouter/deepseek-v4",
            "routing_selected_provider": "openrouter",
            "routing_used_fallback": False,
            "tier_dispatch_receipt": {"enabled": True, "applied": True},
        }
        out = _render(entry)
        self.assertIn("Note: tier dispatch changed the work model shown below.", out)


class TestLastProfileDisplay(unittest.TestCase):

    def test_more_shows_profile_when_present(self):
        entry = {
            "task": "do a thing",
            "execution_profile": "native_deep",
            "execution_profile_reason": "security category",
        }
        out = _render(entry, detail="more")
        self.assertIn("Execution", out)
        self.assertIn("Mode: Deep Run", out)
        self.assertIn("Reason: security category", out)

    def test_full_shows_profile_when_present(self):
        entry = {
            "task": "do a thing",
            "execution_profile": "native_light",
            "execution_profile_reason": "simple/safe task",
        }
        out = _render(entry, detail="full")
        self.assertIn("Execution", out)
        self.assertIn("Mode: Run", out)
        self.assertIn("Reason: simple/safe task", out)

    def test_profile_without_reason(self):
        entry = {
            "task": "do a thing",
            "execution_profile": "native_swarm",
        }
        out = _render(entry, detail="more")
        self.assertIn("Execution", out)
        self.assertIn("Mode: Team Run", out)

    def test_no_crash_on_entry_without_profile_fields(self):
        entry = {"task": "do a thing"}
        out = _render(entry, detail="more")
        self.assertNotIn("Execution", out)

    def test_no_crash_on_entry_without_profile_fields_full(self):
        entry = {"task": "do a thing"}
        out = _render(entry, detail="full")
        self.assertNotIn("Execution", out)

    def test_default_detail_does_not_show_profile(self):
        entry = {
            "task": "do a thing",
            "execution_profile": "native_deep",
            "execution_profile_reason": "security category",
        }
        out = _render(entry, detail="default")
        self.assertNotIn("Execution", out)


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
        self.assertIn("Verification", out)
        self.assertIn("safe", out)
        self.assertIn("detected", out)

    def test_full_shows_verification_plan(self):
        out = _render(self._PLAN_ENTRY, detail="full")
        self.assertIn("Verification", out)
        self.assertIn("safe", out)
        self.assertIn("detected", out)

    def test_argv_rendered_space_joined(self):
        out = _render(self._PLAN_ENTRY, detail="more")
        self.assertIn("python -m pytest", out)

    def test_default_detail_hides_verification_plan(self):
        out = _render(self._PLAN_ENTRY, detail="default")
        self.assertNotIn("Verification", out)

    def test_old_entry_without_plan_no_crash(self):
        entry = {"task": "old run"}
        out = _render(entry, detail="more")
        self.assertNotIn("Verification", out)

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


class TestLastReadonlyTaskTypeBlock(unittest.TestCase):

    _RO_ENTRY = {
        "task": "explain the routing engine",
        "routing_rationale": "read-only analysis",
    }

    def test_more_shows_task_type_for_readonly(self):
        out = _render(self._RO_ENTRY, detail="more")
        self.assertIn("Task type", out)
        self.assertIn("Read-only analysis", out)
        self.assertIn("Reason:", out)

    def test_full_shows_task_type_for_readonly(self):
        out = _render(self._RO_ENTRY, detail="full")
        self.assertIn("Task type", out)
        self.assertIn("Read-only analysis", out)

    def test_default_does_not_show_task_type(self):
        out = _render(self._RO_ENTRY, detail="default")
        self.assertNotIn("Task type", out)

    def test_write_task_does_not_show_task_type(self):
        entry = {"task": "add a helper function", "routing_rationale": "standard feature implementation"}
        out = _render(entry, detail="more")
        self.assertNotIn("Task type", out)


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


class TestNativeValidationContractRendering(unittest.TestCase):

    def _base_entry(self):
        return {
            "workflow": "native",
            "executor": "native",
            "native_loop_steps": [],
            "native_loop_trace": [],
        }

    def _entry_with_contract(self, strength="strong", checks=3, cmds=1):
        entry = self._base_entry()
        entry["validation_contract"] = {
            "intent": "add feature",
            "risk_level": "low",
            "expected_change_scope": "3 files expected",
            "acceptance_checks": [f"check {i}" for i in range(checks)],
            "verification_commands": [f"cmd {i}" for i in range(cmds)],
            "approval_expected": False,
            "strength": strength,
            "warnings": [],
        }
        return entry

    def test_absent_on_old_runs_does_not_crash(self):
        entry = self._base_entry()
        out = _render(entry, detail="more")
        self.assertNotIn("validation contract:", out)

    def test_compact_line_appears_at_more_detail(self):
        entry = self._entry_with_contract(strength="strong", checks=3, cmds=1)
        out = _render(entry, detail="more")
        self.assertIn("validation contract:", out)
        self.assertIn("strong", out)
        self.assertIn("3 checks", out)
        self.assertIn("1 verification command", out)

    def test_compact_line_singular_check(self):
        entry = self._entry_with_contract(strength="fair", checks=1, cmds=0)
        out = _render(entry, detail="more")
        self.assertIn("1 check", out)
        self.assertNotIn("1 checks", out)

    def test_compact_line_singular_command(self):
        entry = self._entry_with_contract(strength="strong", checks=2, cmds=1)
        out = _render(entry, detail="more")
        self.assertIn("1 verification command", out)
        self.assertNotIn("1 verification commands", out)

    def test_compact_line_plural_checks(self):
        entry = self._entry_with_contract(strength="strong", checks=3, cmds=1)
        out = _render(entry, detail="more")
        self.assertIn("3 checks", out)

    def test_compact_line_plural_commands(self):
        entry = self._entry_with_contract(strength="strong", checks=2, cmds=3)
        out = _render(entry, detail="more")
        self.assertIn("3 verification commands", out)

    def test_full_detail_includes_multiline_render(self):
        entry = self._entry_with_contract(strength="strong", checks=2, cmds=1)
        out = _render(entry, detail="full")
        self.assertIn("[validation contract]", out)
        self.assertIn("intent:", out)
        self.assertIn("risk:", out)
        self.assertIn("strength:", out)

    def test_absent_when_validation_contract_is_none(self):
        entry = self._base_entry()
        entry["validation_contract"] = None
        out = _render(entry, detail="more")
        self.assertNotIn("validation contract:", out)

    def test_warnings_not_rendered_in_output(self):
        entry = self._base_entry()
        entry["validation_contract"] = {
            "intent": "task",
            "risk_level": "low",
            "expected_change_scope": "unknown",
            "acceptance_checks": [],
            "verification_commands": [],
            "approval_expected": False,
            "strength": "weak",
            "warnings": ["UNIQUE_WARNING_NOT_IN_OUTPUT"],
        }
        out = _render(entry, detail="more")
        self.assertNotIn("UNIQUE_WARNING_NOT_IN_OUTPUT", out)

    def test_native_meta_from_entry_round_trips_validation_contract(self):
        from openshard.cli.run_output import _native_meta_from_entry
        entry = self._entry_with_contract(strength="fair", checks=2, cmds=0)
        meta = _native_meta_from_entry(entry)
        self.assertIsNotNone(meta)
        vc = getattr(meta, "validation_contract", None)
        self.assertIsNotNone(vc)
        self.assertEqual(getattr(vc, "strength", None), "fair")
        self.assertEqual(len(getattr(vc, "acceptance_checks", [])), 2)
        self.assertEqual(len(getattr(vc, "verification_commands", [])), 0)

    def test_pipeline_dict_round_trips_via_native_meta_from_entry(self):
        """Prove the full save→load path: asdict output reconstructed by _native_meta_from_entry."""
        from dataclasses import asdict
        from openshard.cli.run_output import _native_meta_from_entry
        from openshard.native.context import NativeValidationContract
        original = NativeValidationContract(
            intent="add auth",
            risk_level="medium",
            expected_change_scope="2 files expected",
            acceptance_checks=["tests pass", "no regressions"],
            verification_commands=["pytest"],
            approval_expected=False,
            strength="strong",
            warnings=[],
        )
        entry = self._base_entry()
        entry["validation_contract"] = asdict(original)
        meta = _native_meta_from_entry(entry)
        self.assertIsNotNone(meta)
        vc = getattr(meta, "validation_contract", None)
        self.assertIsNotNone(vc)
        self.assertEqual(getattr(vc, "intent", None), "add auth")
        self.assertEqual(getattr(vc, "risk_level", None), "medium")
        self.assertEqual(getattr(vc, "strength", None), "strong")
        self.assertEqual(getattr(vc, "acceptance_checks", None), ["tests pass", "no regressions"])
        self.assertEqual(getattr(vc, "verification_commands", None), ["pytest"])


class TestNativeContextProvenanceRendering(unittest.TestCase):

    def _base_entry(self) -> dict:
        return {"workflow": "native", "executor": "native"}

    def _entry_with_provenance(self, **provenance_kwargs) -> dict:
        p = build_native_context_provenance(**provenance_kwargs)
        entry = self._base_entry()
        entry["context_provenance"] = asdict(p)
        return entry

    def test_old_run_without_provenance_does_not_crash(self):
        entry = self._base_entry()
        out = _render(entry, detail="more")
        self.assertNotIn("context provenance:", out)

    def test_compact_provenance_line_renders_in_more(self):
        obs = NativeObservation(observed_tools=["read", "grep"])
        entry = self._entry_with_provenance(
            observation=obs,
            injected_source_names={"observation"},
        )
        out = _render(entry, detail="more")
        self.assertIn("context provenance:", out)
        self.assertIn("sources", out)
        self.assertIn("injected", out)
        self.assertIn("items", out)

    def test_gaps_line_renders_when_has_gaps(self):
        cqs = NativeContextQualityScore(level="weak")
        entry = self._entry_with_provenance(context_quality_score=cqs)
        out = _render(entry, detail="more")
        self.assertIn("context provenance gaps:", out)

    def test_gaps_line_absent_when_no_gaps(self):
        cqs = NativeContextQualityScore(level="good")
        entry = self._entry_with_provenance(context_quality_score=cqs)
        out = _render(entry, detail="more")
        self.assertNotIn("context provenance gaps:", out)

    def test_full_detail_renders_source_lines(self):
        obs = NativeObservation(observed_tools=["read"])
        plan = NativePlan(suggested_steps=["step1"])
        entry = self._entry_with_provenance(
            observation=obs,
            plan=plan,
            injected_source_names={"observation", "plan"},
        )
        out = _render(entry, detail="full")
        self.assertIn("provenance source:", out)
        self.assertIn("observation", out)
        self.assertIn("plan", out)

    def test_provenance_absent_from_default_output(self):
        obs = NativeObservation(observed_tools=["read"])
        entry = self._entry_with_provenance(observation=obs)
        out = _render(entry, detail="default")
        self.assertNotIn("context provenance:", out)
        self.assertNotIn("provenance source:", out)

    def test_warning_text_not_rendered_in_block(self):
        cqs = NativeContextQualityScore(level="weak")
        vc = NativeValidationContract(strength="weak")
        entry = self._entry_with_provenance(
            context_quality_score=cqs,
            validation_contract=vc,
        )
        out = _render(entry, detail="more")
        self.assertNotIn("context quality weak", out)
        self.assertNotIn("validation contract weak", out)
        self.assertIn("context provenance gaps:", out)

    def test_warning_count_rendered_not_raw_text(self):
        cqs = NativeContextQualityScore(level="weak")
        vc = NativeValidationContract(strength="weak")
        entry = self._entry_with_provenance(
            context_quality_score=cqs,
            validation_contract=vc,
        )
        out = _render(entry, detail="more")
        self.assertIn("2 warnings", out)

    def test_native_meta_from_entry_roundtrips_provenance(self):
        obs = NativeObservation(observed_tools=["read", "grep"])
        plan = NativePlan(suggested_steps=["s1"])
        p = build_native_context_provenance(
            observation=obs,
            plan=plan,
            injected_source_names={"observation", "plan"},
        )
        entry = {"workflow": "native", "context_provenance": asdict(p)}
        meta = _native_meta_from_entry(entry)
        prov = getattr(meta, "context_provenance", None)
        self.assertIsNotNone(prov)
        self.assertEqual(getattr(prov, "used_sources", None), p.used_sources)
        self.assertEqual(getattr(prov, "injected_sources", None), p.injected_sources)
        self.assertEqual(getattr(prov, "total_items", None), p.total_items)
        self.assertEqual(getattr(prov, "has_gaps", None), p.has_gaps)

    def test_dict_based_source_items_render_without_crash(self):
        prov_dict = {
            "sources": [
                {"name": "repo_summary", "used": True, "injected": True, "item_count": 1, "summary": "1 summary"},
                {"name": "plan", "used": True, "injected": True, "item_count": 2, "summary": "2 steps"},
            ],
            "injected_sources": 2,
            "used_sources": 2,
            "total_items": 3,
            "has_gaps": False,
            "warnings": [],
        }
        entry = {"workflow": "native", "context_provenance": prov_dict}
        out = _render(entry, detail="full")
        self.assertIn("provenance source: repo_summary", out)
        self.assertIn("provenance source: plan", out)


class TestNativeRunTrustScoreRendering(unittest.TestCase):

    def _base_entry(self) -> dict:
        return {"workflow": "native", "executor": "native"}

    def _entry_with_trust(self, score: int = 72, level: str = "good", warnings=None, blockers=None, factors=None) -> dict:
        entry = self._base_entry()
        entry["run_trust_score"] = {
            "score": score,
            "level": level,
            "factors": factors or [],
            "warnings": warnings or [],
            "blockers": blockers or [],
        }
        return entry

    def test_old_run_without_run_trust_score_does_not_crash(self):
        entry = self._base_entry()
        out = _render(entry, detail="more")
        self.assertNotIn("run trust:", out)

    def test_compact_line_renders_when_present_in_more(self):
        entry = self._entry_with_trust(score=72, level="good")
        out = _render(entry, detail="more")
        self.assertIn("run trust: 72/100 good", out)

    def test_warning_count_line_renders_when_warnings_exist(self):
        entry = self._entry_with_trust(warnings=["verification failed", "context truncated"])
        out = _render(entry, detail="more")
        self.assertIn("run trust warnings: 2 warnings", out)

    def test_blocker_count_line_renders_when_blockers_exist(self):
        entry = self._entry_with_trust(blockers=["verification failed"])
        out = _render(entry, detail="more")
        self.assertIn("run trust blockers: 1 blocker", out)

    def test_trust_absent_from_default_output(self):
        entry = self._entry_with_trust()
        out = _render(entry, detail="default")
        self.assertNotIn("run trust:", out)

    def test_full_detail_renders_run_trust_block(self):
        entry = self._entry_with_trust(
            score=82,
            level="good",
            factors=[{"name": "verification_passed", "impact": 20, "reason": "verification passed"}],
        )
        out = _render(entry, detail="full")
        self.assertIn("[run trust]", out)
        self.assertIn("score: 82/100", out)
        self.assertIn("level: good", out)

    def test_full_detail_renders_factor_lines(self):
        entry = self._entry_with_trust(
            score=82,
            level="good",
            factors=[{"name": "verification_passed", "impact": 20, "reason": "verification passed"}],
        )
        out = _render(entry, detail="full")
        self.assertIn("verification_passed", out)
        self.assertIn("+20", out)

    def test_native_meta_from_entry_roundtrips_run_trust_score(self):
        trust = NativeRunTrustScore(
            score=72,
            level="good",
            factors=[NativeRunTrustFactor(name="verification_passed", impact=20, reason="verification passed")],
            warnings=["context truncated"],
            blockers=[],
        )
        entry = {"workflow": "native", "run_trust_score": asdict(trust)}
        meta = _native_meta_from_entry(entry)
        rts = getattr(meta, "run_trust_score", None)
        self.assertIsNotNone(rts)
        self.assertEqual(getattr(rts, "score", None), 72)
        self.assertEqual(getattr(rts, "level", None), "good")

    def test_dict_based_factor_rendering_in_full(self):
        entry = self._entry_with_trust(
            score=50,
            level="fair",
            factors=[
                {"name": "verification_not_attempted", "impact": -10, "reason": "verification not attempted"},
                {"name": "validation_contract_strong", "impact": 15, "reason": "strong validation contract"},
            ],
        )
        out = _render(entry, detail="full")
        self.assertIn("verification_not_attempted", out)
        self.assertIn("validation_contract_strong", out)
        self.assertIn("+15", out)

    def test_compact_does_not_expose_raw_warning_or_blocker_text(self):
        entry = self._entry_with_trust(
            warnings=["verification failed", "blocked commands detected"],
            blockers=["verification failed"],
        )
        out = _render(entry, detail="more")
        self.assertNotIn("verification failed", out)
        self.assertNotIn("blocked commands detected", out)
        self.assertIn("2 warnings", out)
        self.assertIn("1 blocker", out)


class TestNativeModelSelectionDecisionRendering(unittest.TestCase):
    def _base_entry(self) -> dict:
        return {"workflow": "native", "executor": "native"}

    def _msd_dict(
        self,
        strategy: str = "cost-balanced",
        task_type: str = "feature",
        risk_level: str = "medium",
        confidence: str = "high",
        fallback_reason: str = "",
        warnings: list | None = None,
    ) -> dict:
        return asdict(NativeModelSelectionDecision(
            strategy=strategy,
            task_type=task_type,
            risk_level=risk_level,
            roles=[
                NativeModelRoleDecision(role="planner", model_tier="frontier-reasoning-model", cost_tier="high", reason="planning"),
                NativeModelRoleDecision(role="executor", model_tier="balanced-coding-model", cost_tier="medium", reason="balanced"),
                NativeModelRoleDecision(role="validator", model_tier="independent-validator-model", cost_tier="medium", reason="independent"),
            ],
            warnings=warnings or [],
            fallback_reason=fallback_reason,
            confidence=confidence,
        ))

    def test_old_entry_no_msd_does_not_crash_more(self):
        entry = self._base_entry()
        out = _render(entry, detail="more")
        self.assertNotIn("model selection:", out)

    def test_old_entry_no_msd_does_not_crash_full(self):
        entry = self._base_entry()
        out = _render(entry, detail="full")
        self.assertNotIn("model selection:", out)

    def test_compact_line_renders_at_more(self):
        entry = self._base_entry()
        entry["model_selection_decision"] = self._msd_dict(strategy="cost-balanced")
        out = _render(entry, detail="more")
        self.assertIn("model selection:", out)
        self.assertIn("cost-balanced", out)

    def test_full_block_renders_at_full(self):
        entry = self._base_entry()
        entry["model_selection_decision"] = self._msd_dict(strategy="frontier-heavy", risk_level="high")
        out = _render(entry, detail="full")
        self.assertIn("[model selection]", out)
        self.assertIn("strategy:", out)
        self.assertIn("roles:", out)

    def test_model_selection_not_shown_at_default(self):
        entry = self._base_entry()
        entry["model_selection_decision"] = self._msd_dict()
        out = _render(entry, detail="default")
        self.assertNotIn("model selection:", out)

    def test_native_meta_from_entry_roundtrips_msd(self):
        entry = self._base_entry()
        entry["model_selection_decision"] = self._msd_dict(
            strategy="context-cautious",
            confidence="medium",
        )
        meta = _native_meta_from_entry(entry)
        msd = getattr(meta, "model_selection_decision", None)
        self.assertIsNotNone(msd)
        strategy = getattr(msd, "strategy", None)
        confidence = getattr(msd, "confidence", None)
        self.assertEqual(strategy, "context-cautious")
        self.assertEqual(confidence, "medium")

    def test_full_renders_role_tiers(self):
        entry = self._base_entry()
        entry["model_selection_decision"] = self._msd_dict()
        out = _render(entry, detail="full")
        self.assertIn("frontier-reasoning-model", out)
        self.assertIn("independent-validator-model", out)

    def test_synced_role_appears_in_full_output(self):
        entry = self._base_entry()
        entry["model_selection_decision"] = self._msd_dict(
            strategy="cost-balanced",
            warnings=["model selection adjusted by policy enforcement"],
        )
        out = _render(entry, detail="full")
        self.assertIn("balanced-coding-model", out)

    def test_warning_count_renders_without_raw_text_in_full(self):
        entry = self._base_entry()
        entry["model_selection_decision"] = self._msd_dict(
            warnings=["model selection adjusted by policy enforcement"],
        )
        out = _render(entry, detail="full")
        self.assertIn("warnings: 1", out)
        self.assertNotIn("model selection adjusted by policy enforcement", out)

    def test_warnings_count_in_compact_output(self):
        entry = self._base_entry()
        entry["model_selection_decision"] = self._msd_dict(
            warnings=["model selection adjusted by policy enforcement"],
        )
        out = _render(entry, detail="more")
        self.assertIn("warnings=1", out)

    def test_old_entries_without_warnings_render_safely(self):
        entry = self._base_entry()
        msd = self._msd_dict()
        msd.pop("warnings", None)
        entry["model_selection_decision"] = msd
        out = _render(entry, detail="more")
        self.assertIn("model selection:", out)
        self.assertNotIn("warnings=", out)


class TestNativeModelCandidateScoringRendering(unittest.TestCase):
    def _base_entry(self) -> dict:
        return {"workflow": "native", "executor": "native"}

    def _mcs_dict(
        self,
        strategy: str = "cost-balanced",
        confidence: str = "medium",
        include_candidates: bool = False,
    ) -> dict:
        candidates = []
        if include_candidates:
            candidates = [
                asdict(NativeModelCandidateScore(
                    role="planner",
                    candidate="frontier-reasoning-model",
                    score=87,
                    capability_score=25,
                    reason="capability+25",
                )),
                asdict(NativeModelCandidateScore(
                    role="executor",
                    candidate="balanced-coding-model",
                    score=76,
                    capability_score=15,
                    reason="capability+15",
                )),
                asdict(NativeModelCandidateScore(
                    role="executor",
                    candidate="low-cost-coding-model",
                    score=62,
                    cost_score=20,
                    reason="cost+20",
                )),
            ]
        return asdict(NativeModelCandidateScoring(
            strategy=strategy,
            confidence=confidence,
            selected_by_role={
                "planner": "frontier-reasoning-model",
                "executor": "balanced-coding-model",
                "validator": "independent-validator-model",
            },
            candidates=candidates,
            warnings=[],
        ))

    def test_absent_model_candidate_scoring_does_not_crash_more(self):
        entry = self._base_entry()
        try:
            out = _render(entry, detail="more")
        except Exception as exc:
            self.fail(f"render raised {exc}")
        self.assertNotIn("model candidates:", out)

    def test_absent_model_candidate_scoring_does_not_crash_full(self):
        entry = self._base_entry()
        try:
            out = _render(entry, detail="full")
        except Exception as exc:
            self.fail(f"render raised {exc}")
        self.assertNotIn("model candidates:", out)

    def test_present_more_renders_compact_line(self):
        entry = self._base_entry()
        entry["model_candidate_scoring"] = self._mcs_dict(strategy="cost-balanced", confidence="high")
        out = _render(entry, detail="more")
        self.assertIn("model candidates:", out)
        self.assertIn("cost-balanced", out)

    def test_present_compact_does_not_render(self):
        entry = self._base_entry()
        entry["model_candidate_scoring"] = self._mcs_dict()
        out = _render(entry, detail="default")
        self.assertNotIn("model candidates:", out)

    def test_full_renders_block_header(self):
        entry = self._base_entry()
        entry["model_candidate_scoring"] = self._mcs_dict(include_candidates=True)
        out = _render(entry, detail="full")
        self.assertIn("[model candidates]", out)

    def test_full_renders_strategy_and_confidence(self):
        entry = self._base_entry()
        entry["model_candidate_scoring"] = self._mcs_dict(
            strategy="frontier-heavy",
            confidence="low",
            include_candidates=True,
        )
        out = _render(entry, detail="full")
        self.assertIn("strategy: frontier-heavy", out)
        self.assertIn("confidence: low", out)

    def test_full_renders_selected_roles(self):
        entry = self._base_entry()
        entry["model_candidate_scoring"] = self._mcs_dict(include_candidates=True)
        out = _render(entry, detail="full")
        self.assertIn("planner:", out)
        self.assertIn("executor:", out)
        self.assertIn("validator:", out)

    def test_full_renders_score_lines(self):
        entry = self._base_entry()
        entry["model_candidate_scoring"] = self._mcs_dict(include_candidates=True)
        out = _render(entry, detail="full")
        self.assertIn("scores:", out)
        self.assertIn("planner/frontier-reasoning-model: 87", out)

    def test_full_renders_warnings_count(self):
        entry = self._base_entry()
        mcs = self._mcs_dict(include_candidates=True)
        mcs["warnings"] = ["low trust run may reduce model selection confidence"]
        entry["model_candidate_scoring"] = mcs
        out = _render(entry, detail="full")
        self.assertIn("warnings: 1", out)

    def test_native_meta_from_entry_roundtrips_model_candidate_scoring(self):
        entry = self._base_entry()
        entry["model_candidate_scoring"] = self._mcs_dict(strategy="context-cautious", confidence="low")
        meta = _native_meta_from_entry(entry)
        mcs = getattr(meta, "model_candidate_scoring", None)
        self.assertIsNotNone(mcs)
        strategy = getattr(mcs, "strategy", None)
        confidence = getattr(mcs, "confidence", None)
        self.assertEqual(strategy, "context-cautious")
        self.assertEqual(confidence, "low")

    def test_builder_result_asdict_roundtrips_through_entry(self):
        from dataclasses import asdict as _asdict
        sc = build_native_model_candidate_scoring()
        entry = self._base_entry()
        entry["model_candidate_scoring"] = _asdict(sc)
        meta = _native_meta_from_entry(entry)
        mcs = getattr(meta, "model_candidate_scoring", None)
        self.assertIsNotNone(mcs)
        self.assertEqual(getattr(mcs, "strategy", None), "cost-balanced")

    def test_full_with_none_scoring_after_present_does_not_crash(self):
        entry = self._base_entry()
        entry["model_candidate_scoring"] = None
        try:
            out = _render(entry, detail="full")
        except Exception as exc:
            self.fail(f"render raised {exc}")
        self.assertNotIn("model candidates:", out)

    def test_old_entry_without_blocked_candidates_renders_more(self):
        entry = self._base_entry()
        mcs = self._mcs_dict(strategy="cost-balanced", confidence="medium")
        # Simulate old entry: no blocked_candidates key
        mcs.pop("blocked_candidates", None)
        entry["model_candidate_scoring"] = mcs
        try:
            out = _render(entry, detail="more")
        except Exception as exc:
            self.fail(f"render raised {exc}")
        self.assertIn("model candidates:", out)
        self.assertNotIn("blocked=", out)

    def test_old_entry_without_blocked_candidates_renders_full(self):
        entry = self._base_entry()
        mcs = self._mcs_dict(include_candidates=True)
        mcs.pop("blocked_candidates", None)
        entry["model_candidate_scoring"] = mcs
        try:
            out = _render(entry, detail="full")
        except Exception as exc:
            self.fail(f"render raised {exc}")
        self.assertIn("[model candidates]", out)
        self.assertNotIn("blocked:", out)

    def test_blocked_count_appears_in_more(self):
        entry = self._base_entry()
        mcs = self._mcs_dict(strategy="cost-balanced", confidence="high")
        mcs["blocked_candidates"] = ["planner/frontier-reasoning-model", "executor/frontier-reasoning-model"]
        entry["model_candidate_scoring"] = mcs
        out = _render(entry, detail="more")
        self.assertIn("blocked=2", out)

    def test_zero_blocked_not_shown_in_compact(self):
        entry = self._base_entry()
        mcs = self._mcs_dict()
        mcs["blocked_candidates"] = []
        entry["model_candidate_scoring"] = mcs
        out = _render(entry, detail="more")
        self.assertNotIn("blocked=", out)

    def test_blocked_section_appears_in_full(self):
        entry = self._base_entry()
        mcs = self._mcs_dict(include_candidates=True)
        mcs["blocked_candidates"] = ["planner/frontier-reasoning-model"]
        entry["model_candidate_scoring"] = mcs
        out = _render(entry, detail="full")
        self.assertIn("blocked:", out)
        self.assertIn("planner/frontier-reasoning-model", out)

    def test_no_blocked_section_in_full_when_empty(self):
        entry = self._base_entry()
        mcs = self._mcs_dict(include_candidates=True)
        mcs["blocked_candidates"] = []
        entry["model_candidate_scoring"] = mcs
        out = _render(entry, detail="full")
        self.assertNotIn("blocked:", out)


class TestNativeModelPolicyRendering(unittest.TestCase):
    def _base_entry(self) -> dict:
        return {"workflow": "native", "executor": "native"}

    def _entry_with_policy(self, mode: str) -> dict:
        entry = self._base_entry()
        entry["model_policy"] = asdict(build_native_model_policy(mode))
        return entry

    def test_model_policy_hidden_at_default_detail(self):
        entry = self._entry_with_policy("auto")
        out = _render(entry, detail="default")
        self.assertNotIn("model policy", out)

    def test_model_policy_compact_line_at_more(self):
        entry = self._entry_with_policy("auto")
        out = _render(entry, detail="more")
        self.assertIn("model policy:", out)
        self.assertIn("auto", out)

    def test_model_policy_frontier_allowed_flag_at_more(self):
        entry = self._entry_with_policy("auto")
        out = _render(entry, detail="more")
        self.assertIn("frontier allowed", out)

    def test_model_policy_frontier_blocked_flag_at_more(self):
        entry = self._entry_with_policy("open-source-only")
        out = _render(entry, detail="more")
        self.assertIn("frontier blocked", out)

    def test_model_policy_cheapest_safe_at_more(self):
        entry = self._entry_with_policy("cheapest-safe")
        out = _render(entry, detail="more")
        self.assertIn("model policy:", out)
        self.assertIn("cheapest-safe", out)
        self.assertIn("prefer low cost", out)

    def test_model_policy_local_only_at_more(self):
        entry = self._entry_with_policy("local-only")
        out = _render(entry, detail="more")
        self.assertIn("local only", out)
        self.assertIn("frontier blocked", out)

    def test_model_policy_full_section_header(self):
        entry = self._entry_with_policy("auto")
        out = _render(entry, detail="full")
        self.assertIn("[model policy]", out)

    def test_model_policy_full_shows_mode(self):
        entry = self._entry_with_policy("cheapest-safe")
        out = _render(entry, detail="full")
        self.assertIn("mode: cheapest-safe", out)

    def test_model_policy_full_shows_all_fields(self):
        entry = self._entry_with_policy("auto")
        out = _render(entry, detail="full")
        self.assertIn("prefer_low_cost:", out)
        self.assertIn("require_open_source:", out)
        self.assertIn("require_local:", out)
        self.assertIn("allow_frontier:", out)
        self.assertIn("warnings:", out)

    def test_model_policy_full_boolean_values(self):
        entry = self._entry_with_policy("open-source-only")
        out = _render(entry, detail="full")
        self.assertIn("require_open_source: yes", out)
        self.assertIn("allow_frontier: no", out)
        self.assertIn("require_local: no", out)

    def test_model_policy_warnings_shown_at_more(self):
        entry = self._base_entry()
        p = build_native_model_policy("custom")
        entry["model_policy"] = asdict(p)
        out = _render(entry, detail="more")
        self.assertIn("warning(s)", out)

    def test_model_policy_none_no_crash_at_more(self):
        entry = self._base_entry()
        try:
            out = _render(entry, detail="more")
        except Exception as exc:
            self.fail(f"render raised {exc}")
        self.assertNotIn("model policy", out)

    def test_model_policy_none_no_crash_at_full(self):
        entry = self._base_entry()
        try:
            out = _render(entry, detail="full")
        except Exception as exc:
            self.fail(f"render raised {exc}")
        self.assertNotIn("model policy", out)

    def test_default_output_unchanged_when_policy_absent(self):
        entry_without = self._base_entry()
        entry_with_none = self._base_entry()
        entry_with_none["model_policy"] = None
        out_without = _render(entry_without, detail="more")
        out_with_none = _render(entry_with_none, detail="more")
        self.assertNotIn("model policy", out_without)
        self.assertNotIn("model policy", out_with_none)

    def test_native_meta_from_entry_extracts_model_policy(self):
        entry = self._entry_with_policy("frontier-heavy")
        nm = _native_meta_from_entry(entry)
        mp = getattr(nm, "model_policy", None)
        self.assertIsNotNone(mp)
        mode = mp.get("mode") if isinstance(mp, dict) else getattr(mp, "mode", None)
        self.assertEqual(mode, "frontier-heavy")

    def test_native_meta_from_entry_missing_policy_returns_none(self):
        entry = self._base_entry()
        nm = _native_meta_from_entry(entry)
        mp = getattr(nm, "model_policy", "MISSING")
        self.assertIsNone(mp)


class TestNativeModelPolicyReceiptRendering(unittest.TestCase):
    def _base_entry(self) -> dict:
        return {"workflow": "native", "executor": "native"}

    def _entry_with_receipt(self, **kwargs) -> dict:
        entry = self._base_entry()
        entry["model_policy_receipt"] = asdict(NativeModelPolicyReceipt(**kwargs))
        return entry

    def test_receipt_hidden_at_default_detail(self):
        entry = self._entry_with_receipt(active=True, blocked_count=2)
        out = _render(entry, detail="default")
        self.assertNotIn("model policy receipt", out)

    def test_compact_line_shown_at_more(self):
        entry = self._entry_with_receipt(active=True, blocked_count=2)
        out = _render(entry, detail="more")
        self.assertIn("model policy receipt:", out)

    def test_compact_line_shows_active_and_blocked(self):
        entry = self._entry_with_receipt(active=True, blocked_count=3)
        out = _render(entry, detail="more")
        self.assertIn("active=true", out)
        self.assertIn("blocked=3", out)

    def test_full_block_header_shown_at_full(self):
        entry = self._entry_with_receipt(active=True)
        out = _render(entry, detail="full")
        self.assertIn("[model policy receipt]", out)

    def test_full_block_shows_all_fields(self):
        entry = self._entry_with_receipt(
            active=True,
            mode="open-source-only",
            affected_selection=True,
            blocked_count=3,
            changed_roles=["executor"],
            warnings_count=2,
            summary="policy active: blocked 3 candidates and changed 1 role",
        )
        out = _render(entry, detail="full")
        self.assertIn("mode: open-source-only", out)
        self.assertIn("affected_selection: yes", out)
        self.assertIn("blocked_count: 3", out)
        self.assertIn("warnings_count: 2", out)
        self.assertIn("summary: policy active: blocked 3 candidates and changed 1 role", out)

    def test_changed_roles_list_rendered(self):
        entry = self._entry_with_receipt(
            active=True,
            affected_selection=True,
            changed_roles=["executor"],
        )
        out = _render(entry, detail="full")
        self.assertIn("- executor", out)

    def test_empty_changed_roles_rendered(self):
        entry = self._entry_with_receipt(active=True, changed_roles=[])
        out = _render(entry, detail="full")
        self.assertIn("changed_roles: []", out)

    def test_old_entry_without_receipt_no_crash(self):
        entry = self._base_entry()
        try:
            out = _render(entry, detail="more")
        except Exception as exc:
            self.fail(f"render raised {exc}")
        self.assertNotIn("receipt", out)

    def test_entry_with_none_receipt_no_crash(self):
        entry = self._base_entry()
        entry["model_policy_receipt"] = None
        try:
            out = _render(entry, detail="more")
        except Exception as exc:
            self.fail(f"render raised {exc}")
        self.assertNotIn("model policy receipt", out)

    def test_native_meta_from_entry_extracts_receipt(self):
        entry = self._base_entry()
        entry["model_policy_receipt"] = {"active": True, "mode": "open-source-only",
                                          "affected_selection": False, "blocked_count": 0,
                                          "changed_roles": [], "warnings_count": 0, "summary": ""}
        nm = _native_meta_from_entry(entry)
        mpr = getattr(nm, "model_policy_receipt", None)
        self.assertIsNotNone(mpr)
        active = mpr.get("active") if isinstance(mpr, dict) else getattr(mpr, "active", None)
        self.assertTrue(active)

    def test_receipt_not_shown_at_default_when_absent(self):
        entry = self._base_entry()
        out = _render(entry, detail="default")
        self.assertNotIn("receipt", out)


class TestRoutingPreviewRendering(unittest.TestCase):
    """Tests for routing_preview compact and full rendering."""

    def _preview_ns(self, **overrides):
        from types import SimpleNamespace
        defaults = dict(
            strategy="cost-balanced",
            policy_mode="cheapest-safe",
            planner_tier="frontier-reasoning-model",
            executor_tier="balanced-coding-model",
            validator_tier="independent-validator-model",
            risk_level="medium",
            confidence="high",
            trust_level="good",
            blocked_count=3,
            policy_affected=True,
            warnings=["w1"],
            summary="cost-balanced | planner=frontier-reasoning-model ...",
        )
        defaults.update(overrides)
        return SimpleNamespace(**defaults)

    def _meta_with_preview(self, preview):
        from types import SimpleNamespace
        ns = SimpleNamespace
        return ns(
            routing_preview=preview,
            repo_context_summary=None, observation=None, plan=None,
            write_path="pipeline", verification_loop=None,
            verification_command_summary=None, diff_review=None,
            final_report=None, native_loop_steps=[], native_loop_trace=ns(events=[]),
            native_backend=None, native_backend_available=True,
            native_backend_notes=[], native_backend_proof=None,
            read_search_findings=[], patch_proposal=None,
            command_policy_preview=None, context_packet=None,
            file_context=None, context_quality_score=None,
            context_quality_advisory=None, change_budget=None,
            change_budget_preview=None, change_budget_soft_gate=None,
            approval_request=None, approval_receipt=None,
            verification_plan=None, clarification_request=None,
            context_usage_summary=None, failure_memory=None,
            osn_loop=None, deepagents_adapter=None,
            validation_contract=None, context_provenance=None,
            run_trust_score=None, model_selection_decision=None,
            model_candidate_scoring=None, model_policy=None,
            model_policy_receipt=None,
        )

    def _entry_with_preview(self, **rp_overrides):
        rp = dict(
            strategy="cost-balanced",
            policy_mode="cheapest-safe",
            planner_tier="frontier-reasoning-model",
            executor_tier="balanced-coding-model",
            validator_tier="independent-validator-model",
            risk_level="medium",
            confidence="high",
            trust_level="good",
            blocked_count=3,
            policy_affected=True,
            warnings=["w1"],
            summary="",
        )
        rp.update(rp_overrides)
        return {
            "task": "add feature",
            "workflow": "native",
            "executor": "native",
            "routing_preview": rp,
        }

    # --- compact line (--more) ---

    def test_compact_line_contains_strategy(self):
        from openshard.cli.run_output import _render_native_demo_block
        meta = self._meta_with_preview(self._preview_ns())
        combined = "\n".join(_render_native_demo_block(meta, detail="more"))
        self.assertIn("cost-balanced", combined)

    def test_compact_line_contains_changed_yes(self):
        from openshard.cli.run_output import _render_native_demo_block
        meta = self._meta_with_preview(self._preview_ns(policy_affected=True))
        combined = "\n".join(_render_native_demo_block(meta, detail="more"))
        self.assertIn("changed=yes", combined)

    def test_compact_line_contains_changed_no(self):
        from openshard.cli.run_output import _render_native_demo_block
        meta = self._meta_with_preview(self._preview_ns(policy_affected=False))
        combined = "\n".join(_render_native_demo_block(meta, detail="more"))
        self.assertIn("changed=no", combined)

    def test_compact_line_contains_blocked_count(self):
        from openshard.cli.run_output import _render_native_demo_block
        meta = self._meta_with_preview(self._preview_ns(blocked_count=3))
        combined = "\n".join(_render_native_demo_block(meta, detail="more"))
        self.assertIn("blocked=3", combined)

    def test_compact_line_contains_trust_level(self):
        from openshard.cli.run_output import _render_native_demo_block
        meta = self._meta_with_preview(self._preview_ns(trust_level="good"))
        combined = "\n".join(_render_native_demo_block(meta, detail="more"))
        self.assertIn("trust=good", combined)

    def test_compact_line_contains_confidence(self):
        from openshard.cli.run_output import _render_native_demo_block
        meta = self._meta_with_preview(self._preview_ns(confidence="high"))
        combined = "\n".join(_render_native_demo_block(meta, detail="more"))
        self.assertIn("confidence=high", combined)

    def test_compact_line_does_not_expose_raw_warning_text(self):
        from openshard.cli.run_output import _render_native_demo_block
        meta = self._meta_with_preview(self._preview_ns(warnings=["secret-warning-text"]))
        combined = "\n".join(_render_native_demo_block(meta, detail="more"))
        self.assertNotIn("secret-warning-text", combined)

    def test_compact_line_no_warnings_token_when_empty(self):
        from openshard.cli.run_output import _render_native_demo_block
        meta = self._meta_with_preview(self._preview_ns(warnings=[]))
        combined = "\n".join(_render_native_demo_block(meta, detail="more"))
        self.assertNotIn("warnings=", combined)

    # --- full block (--full) uses new field labels ---

    def test_full_block_uses_planner_label(self):
        from openshard.cli.run_output import _render_native_demo_block
        meta = self._meta_with_preview(self._preview_ns())
        combined = "\n".join(_render_native_demo_block(meta, detail="full"))
        self.assertIn("planner:", combined)
        self.assertNotIn("planner_tier:", combined)

    def test_full_block_uses_executor_label(self):
        from openshard.cli.run_output import _render_native_demo_block
        meta = self._meta_with_preview(self._preview_ns())
        combined = "\n".join(_render_native_demo_block(meta, detail="full"))
        self.assertIn("executor:", combined)
        self.assertNotIn("executor_tier:", combined)

    def test_full_block_uses_validator_label(self):
        from openshard.cli.run_output import _render_native_demo_block
        meta = self._meta_with_preview(self._preview_ns())
        combined = "\n".join(_render_native_demo_block(meta, detail="full"))
        self.assertIn("validator:", combined)
        self.assertNotIn("validator_tier:", combined)

    def test_full_block_uses_risk_label(self):
        from openshard.cli.run_output import _render_native_demo_block
        meta = self._meta_with_preview(self._preview_ns())
        combined = "\n".join(_render_native_demo_block(meta, detail="full"))
        self.assertIn("risk:", combined)
        self.assertNotIn("risk_level:", combined)

    def test_full_block_uses_trust_label(self):
        from openshard.cli.run_output import _render_native_demo_block
        meta = self._meta_with_preview(self._preview_ns())
        combined = "\n".join(_render_native_demo_block(meta, detail="full"))
        self.assertIn("trust:", combined)
        self.assertNotIn("trust_level:", combined)

    def test_full_block_uses_policy_affected_label(self):
        from openshard.cli.run_output import _render_native_demo_block
        meta = self._meta_with_preview(self._preview_ns())
        combined = "\n".join(_render_native_demo_block(meta, detail="full"))
        self.assertIn("policy_affected:", combined)

    def test_full_block_shows_warnings_count_not_text(self):
        from openshard.cli.run_output import _render_native_demo_block
        meta = self._meta_with_preview(self._preview_ns(warnings=["secret-text"]))
        combined = "\n".join(_render_native_demo_block(meta, detail="full"))
        self.assertIn("warnings:", combined)
        self.assertNotIn("secret-text", combined)

    # --- missing routing_preview does not crash ---

    def test_missing_routing_preview_no_crash_more(self):
        from openshard.cli.run_output import _render_native_demo_block
        meta = self._meta_with_preview(None)
        try:
            _render_native_demo_block(meta, detail="more")
        except Exception as exc:
            self.fail(f"crashed with missing routing_preview: {exc}")

    def test_missing_routing_preview_no_crash_full(self):
        from openshard.cli.run_output import _render_native_demo_block
        meta = self._meta_with_preview(None)
        try:
            _render_native_demo_block(meta, detail="full")
        except Exception as exc:
            self.fail(f"crashed with missing routing_preview: {exc}")

    def test_missing_routing_preview_shows_no_routing_section(self):
        from openshard.cli.run_output import _render_native_demo_block
        meta = self._meta_with_preview(None)
        combined = "\n".join(_render_native_demo_block(meta, detail="more"))
        self.assertNotIn("routing preview", combined)

    # --- default detail hides routing preview ---

    def test_default_detail_hides_routing_preview(self):
        out = _render(self._entry_with_preview(), detail="default")
        self.assertNotIn("routing preview", out)

    # --- integration: dict entry round-trip ---

    def test_compact_via_entry_contains_all_key_fields(self):
        out = _render(self._entry_with_preview(), detail="more")
        self.assertIn("routing preview:", out)
        self.assertIn("changed=yes", out)
        self.assertIn("blocked=3", out)
        self.assertIn("trust=good", out)
        self.assertIn("confidence=high", out)

    def test_full_via_entry_uses_new_labels(self):
        out = _render(self._entry_with_preview(), detail="full")
        self.assertIn("[routing preview]", out)
        self.assertIn("planner:", out)
        self.assertNotIn("planner_tier:", out)


class TestLastRoutingReceiptDisplay(unittest.TestCase):
    """Tests for routing_receipt rendering via _render_log_entry (entry dict round-trip)."""

    def _receipt_dict(self, **kwargs) -> dict:
        defaults = {
            "strategy": "frontier-heavy",
            "planner_tier": "frontier",
            "executor_tier": "fast",
            "validator_tier": "low-cost",
            "policy_mode": "strict",
            "policy_affected": True,
            "blocked_count": 2,
            "trust_level": "good",
            "confidence": "high",
            "warnings_count": 1,
            "summary": "frontier-heavy | policy=strict",
        }
        defaults.update(kwargs)
        return defaults

    def _entry_with_receipt(self, **kwargs) -> dict:
        return {
            "task": "native run",
            "workflow": "native",
            "executor": "native",
            "routing_receipt": self._receipt_dict(**kwargs),
        }

    def test_more_hides_receipt(self):
        out = _render(self._entry_with_receipt(), detail="more")
        self.assertNotIn("routing receipt", out)

    def test_full_renders_receipt_section(self):
        out = _render(self._entry_with_receipt(), detail="full")
        self.assertIn("[routing receipt]", out)
        self.assertIn("strategy:", out)
        self.assertIn("planner:", out)
        self.assertIn("executor:", out)
        self.assertIn("validator:", out)
        self.assertIn("policy_mode:", out)
        self.assertIn("policy_affected:", out)
        self.assertIn("blocked_count:", out)
        self.assertIn("trust:", out)
        self.assertIn("confidence:", out)
        self.assertIn("warnings_count:", out)
        self.assertIn("summary:", out)

    def test_default_hides_receipt(self):
        out = _render(self._entry_with_receipt(), detail="default")
        self.assertNotIn("routing receipt", out)

    def test_old_entry_without_receipt_does_not_crash(self):
        entry = {
            "task": "native run",
            "workflow": "native",
            "executor": "native",
            # no routing_receipt key
        }
        out = _render(entry, detail="more")
        self.assertNotIn("routing receipt", out)

    def test_policy_affected_shown_as_yes(self):
        out = _render(self._entry_with_receipt(policy_affected=True), detail="full")
        self.assertIn("yes", out)

    def test_policy_affected_shown_as_no(self):
        out = _render(self._entry_with_receipt(policy_affected=False), detail="full")
        self.assertIn("no", out)


class TestLastReadonlyLabels(unittest.TestCase):
    """Stage and model-line labels use analysis wording for saved read-only runs."""

    def _ro_entry(self) -> dict:
        return {
            "task": "what does openshard/cli/main.py do?",
            "routing_rationale": "read-only analysis",
            "stage_runs": [
                {"model": "z-ai/glm-5.1", "stage_type": "planning",        "duration": 0.5, "cost": 0.0001},
                {"model": "z-ai/glm-5.1", "stage_type": "implementation",  "duration": 1.2, "cost": 0.0003},
            ],
        }

    def _write_entry(self) -> dict:
        return {
            "task": "fix the bug in utils.py",
            "routing_rationale": "standard feature implementation",
            "stage_runs": [
                {"model": "z-ai/glm-5.1", "stage_type": "planning",        "duration": 0.5, "cost": 0.0001},
                {"model": "z-ai/glm-5.1", "stage_type": "implementation",  "duration": 1.2, "cost": 0.0003},
            ],
        }

    def test_readonly_stage_label_says_analysis(self):
        out = _render(self._ro_entry(), detail="more")
        self.assertIn("Analysis", out)

    def test_readonly_stage_label_not_implementation(self):
        out = _render(self._ro_entry(), detail="more")
        self.assertNotIn("Implementation", out)

    def test_readonly_model_line_says_analysis(self):
        out = _render(self._ro_entry(), detail="more")
        self.assertIn("analysis", out)

    def test_readonly_model_line_not_implementation(self):
        out = _render(self._ro_entry(), detail="more")
        self.assertNotIn("implementation", out)

    def test_write_stage_label_says_implementation(self):
        out = _render(self._write_entry(), detail="more")
        self.assertIn("Implementation", out)

    def test_write_model_line_says_implementation(self):
        out = _render(self._write_entry(), detail="more")
        self.assertIn("implementation", out)

    def test_readonly_default_detail_no_stage_section(self):
        out = _render(self._ro_entry(), detail="default")
        self.assertNotIn("Stages", out)

    def test_write_default_detail_no_stage_section(self):
        out = _render(self._write_entry(), detail="default")
        self.assertNotIn("Stages", out)


class TestFeedbackRendering(unittest.TestCase):
    """Tests for the Developer feedback section in _render_log_entry."""

    def _entry(self, rating: str = "good", note: str = "") -> dict:
        return {
            "task": "do a thing",
            "feedback": {
                "schema_version": 1,
                "rating": rating,
                "note": note,
                "created_at": "2025-01-01T00:00:00Z",
            },
        }

    def test_more_shows_developer_feedback_header(self):
        out = _render(self._entry(), detail="more")
        self.assertIn("Developer feedback", out)

    def test_full_shows_developer_feedback_header(self):
        out = _render(self._entry(), detail="full")
        self.assertIn("Developer feedback", out)

    def test_default_hides_feedback(self):
        out = _render(self._entry(), detail="default")
        self.assertNotIn("Developer feedback", out)
        self.assertNotIn("Rating:", out)

    def test_rating_shown(self):
        out = _render(self._entry(rating="good"), detail="more")
        self.assertIn("Rating: good", out)

    def test_note_shown_when_present(self):
        out = _render(self._entry(note="GLM was good enough for this analysis"), detail="more")
        self.assertIn("Note: GLM was good enough for this analysis", out)

    def test_note_line_absent_when_empty(self):
        out = _render(self._entry(note=""), detail="more")
        self.assertNotIn("Note:", out)

    def test_bad_rating_rendered(self):
        out = _render(self._entry(rating="bad"), detail="more")
        self.assertIn("Rating: bad", out)

    def test_mixed_rating_rendered(self):
        out = _render(self._entry(rating="mixed"), detail="full")
        self.assertIn("Rating: mixed", out)

    def test_old_entry_no_feedback_no_crash_more(self):
        entry = {"task": "old run"}
        out = _render(entry, detail="more")
        self.assertNotIn("Developer feedback", out)

    def test_old_entry_no_feedback_no_crash_full(self):
        entry = {"task": "old run"}
        out = _render(entry, detail="full")
        self.assertNotIn("Developer feedback", out)

    def test_old_entry_no_feedback_no_crash_default(self):
        entry = {"task": "old run"}
        out = _render(entry, detail="default")
        self.assertNotIn("Developer feedback", out)

    def test_feedback_rendered_after_notes(self):
        entry = {
            "task": "do a thing",
            "notes": ["a pipeline note"],
            "feedback": {"schema_version": 1, "rating": "good", "note": ""},
        }
        out = _render(entry, detail="more")
        self.assertGreater(out.index("Developer feedback"), out.index("Notes"))


if __name__ == "__main__":
    unittest.main()
