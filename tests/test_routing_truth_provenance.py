"""Tests for routing truth provenance fields (PR 6).

Validates:
- New RoutingTruth fields have safe defaults for empty/legacy entries
- Fields are read from Shard entry correctly
- routing_truth_to_dict includes all new fields
- render_routing_truth_lines emits provenance at more/full detail
"""
from __future__ import annotations

from openshard.history.routing_truth import (
    RoutingTruth,
    build_routing_truth,
    render_routing_truth_lines,
    routing_truth_to_dict,
)

# ---------------------------------------------------------------------------
# Safe defaults on empty / legacy entries
# ---------------------------------------------------------------------------

class TestRoutingTruthDefaults:
    def test_empty_entry_all_new_fields_at_defaults(self):
        rt = build_routing_truth({})
        assert rt.model_resolution == "unknown"
        assert rt.feedback_routing_applied is False
        assert rt.mode_policy_applied is False
        assert rt.executor_source == "unknown"

    def test_none_entry_treated_as_empty(self):
        rt = build_routing_truth(None)
        assert rt.model_resolution == "unknown"

    def test_legacy_entry_without_new_fields_safe(self):
        # Old Shard with no provenance fields — must not raise
        entry = {
            "execution_model": "anthropic/claude-sonnet-4.6",
            "routing_model": "anthropic/claude-sonnet-4.6",
            "routing_rationale": "security task",
        }
        rt = build_routing_truth(entry)
        assert rt.model_resolution == "unknown"
        assert rt.feedback_routing_applied is False
        assert rt.executor_source == "unknown"


# ---------------------------------------------------------------------------
# Fields read from entry
# ---------------------------------------------------------------------------

class TestRoutingTruthFieldReading:
    def test_model_resolution_registry_read(self):
        rt = build_routing_truth({"model_resolution": "registry"})
        assert rt.model_resolution == "registry"

    def test_model_resolution_hardcoded_read(self):
        rt = build_routing_truth({"model_resolution": "hardcoded"})
        assert rt.model_resolution == "hardcoded"

    def test_feedback_routing_applied_true(self):
        rt = build_routing_truth({"feedback_routing_applied": True})
        assert rt.feedback_routing_applied is True

    def test_feedback_routing_applied_false_by_default(self):
        rt = build_routing_truth({"feedback_routing_applied": False})
        assert rt.feedback_routing_applied is False

    def test_mode_policy_applied_true(self):
        rt = build_routing_truth({"mode_policy_applied": True})
        assert rt.mode_policy_applied is True

    def test_executor_source_advisory(self):
        rt = build_routing_truth({"executor_source": "advisory"})
        assert rt.executor_source == "advisory"

    def test_executor_source_override(self):
        rt = build_routing_truth({"executor_source": "override"})
        assert rt.executor_source == "override"

    def test_executor_source_heuristic(self):
        rt = build_routing_truth({"executor_source": "heuristic"})
        assert rt.executor_source == "heuristic"


# ---------------------------------------------------------------------------
# routing_truth_to_dict includes new fields
# ---------------------------------------------------------------------------

class TestRoutingTruthToDict:
    def test_all_new_fields_present_in_dict(self):
        rt = build_routing_truth({
            "model_resolution": "registry",
            "feedback_routing_applied": True,
            "mode_policy_applied": True,
            "executor_source": "advisory",
        })
        d = routing_truth_to_dict(rt)
        assert "model_resolution" in d
        assert "feedback_routing_applied" in d
        assert "mode_policy_applied" in d
        assert "executor_source" in d

    def test_dict_values_match_entry(self):
        rt = build_routing_truth({
            "model_resolution": "registry",
            "feedback_routing_applied": True,
            "executor_source": "advisory",
        })
        d = routing_truth_to_dict(rt)
        assert d["model_resolution"] == "registry"
        assert d["feedback_routing_applied"] is True
        assert d["executor_source"] == "advisory"

    def test_legacy_entry_dict_has_safe_defaults(self):
        rt = build_routing_truth({"execution_model": "some/model"})
        d = routing_truth_to_dict(rt)
        assert d["model_resolution"] == "unknown"
        assert d["feedback_routing_applied"] is False
        assert d["executor_source"] == "unknown"


# ---------------------------------------------------------------------------
# render_routing_truth_lines provenance output
# ---------------------------------------------------------------------------

class TestRenderRoutingTruthLines:
    def _make_rt(self, **kwargs) -> RoutingTruth:
        defaults = dict(
            runtime_model="test/model",
            routing_mode="keyword",
            selection_source="deterministic",
            role_dispatch_status="not_dispatched",
            role_selection_mode="unavailable",
            planner_model=None,
            executor_model=None,
            validator_model=None,
            planner_dispatched=False,
            executor_dispatched=False,
            validator_dispatched=False,
            routing_truth_summary="test",
            model_resolution="unknown",
            feedback_routing_applied=False,
            mode_policy_applied=False,
            executor_source="unknown",
        )
        defaults.update(kwargs)
        return RoutingTruth(**defaults)

    def test_default_detail_no_provenance_line(self):
        rt = self._make_rt(model_resolution="registry", executor_source="advisory")
        lines = render_routing_truth_lines(rt, detail="default")
        assert not any("Routing:" in ln for ln in lines)

    def test_more_detail_emits_registry_line(self):
        rt = self._make_rt(model_resolution="registry")
        lines = render_routing_truth_lines(rt, detail="more")
        assert any("model=registry" in ln for ln in lines)

    def test_more_detail_emits_executor_advisory_line(self):
        rt = self._make_rt(executor_source="advisory")
        lines = render_routing_truth_lines(rt, detail="more")
        assert any("executor=advisory" in ln for ln in lines)

    def test_more_detail_emits_feedback_applied_line(self):
        rt = self._make_rt(feedback_routing_applied=True)
        lines = render_routing_truth_lines(rt, detail="more")
        assert any("feedback=applied" in ln for ln in lines)

    def test_more_detail_emits_mode_policy_line(self):
        rt = self._make_rt(mode_policy_applied=True)
        lines = render_routing_truth_lines(rt, detail="more")
        assert any("mode_policy=applied" in ln for ln in lines)

    def test_more_detail_no_provenance_line_when_all_defaults(self):
        rt = self._make_rt()  # all defaults: unknown, False, unknown
        lines = render_routing_truth_lines(rt, detail="more")
        assert not any("Routing:" in ln for ln in lines)

    def test_more_detail_hardcoded_resolution_not_emitted(self):
        # "hardcoded" is a default-equivalent value — should not emit the line
        rt = self._make_rt(model_resolution="hardcoded")
        lines = render_routing_truth_lines(rt, detail="more")
        assert not any("model=hardcoded" in ln for ln in lines)

    def test_heuristic_executor_source_not_emitted(self):
        rt = self._make_rt(executor_source="heuristic")
        lines = render_routing_truth_lines(rt, detail="more")
        assert not any("executor=heuristic" in ln for ln in lines)

    def test_full_detail_same_as_more_for_new_fields(self):
        rt = self._make_rt(model_resolution="registry", executor_source="advisory")
        more_lines = render_routing_truth_lines(rt, detail="more")
        full_lines = render_routing_truth_lines(rt, detail="full")
        # Both should emit the provenance line
        assert any("Routing:" in ln for ln in more_lines)
        assert any("Routing:" in ln for ln in full_lines)
