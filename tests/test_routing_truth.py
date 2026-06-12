"""Tests for honest routing truth (openshard/history/routing_truth.py) and the
proof/receipt surfaces that consume it.

Covers the contract from fix/routing-truth-in-proof:
* A single-model run reports the real runtime routing mode (never an advisory
  value) while marking per-role selection advisory_only / not_dispatched.
* Experimental tier dispatch that actually ran is recorded as dispatched.
* Receipt/proof output never implies planner/executor/validator ran when they
  did not.
* Old/minimal records render safely.
* JSON output stays valid and backward compatible.
"""

from __future__ import annotations

import json
import unittest

from click.testing import CliRunner

from openshard.cli.main import cli
from openshard.history.proof_contract import build_shard_proof_contract
from openshard.history.routing_truth import (
    build_routing_truth,
    render_routing_truth_lines,
    routing_truth_to_dict,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _keyword_entry(**kwargs) -> dict:
    """Single-model keyword run with advisory per-role metadata."""
    entry = {
        "schema_version": "1.2",
        "task": "add a helper",
        "timestamp": "2025-01-01T00:00:00Z",
        "execution_model": "z-ai/glm-5.1",
        "routing_model": "z-ai/glm-5.1",
        "routing_rationale": "standard task",
        "model_selection_decision": {
            "strategy": "cost-balanced",
            "confidence": "medium",
            "roles": [
                {"role": "planner", "model_tier": "frontier-reasoning-model"},
                {"role": "executor", "model_tier": "balanced-coding-model"},
                {"role": "validator", "model_tier": "independent-validator-model"},
            ],
        },
    }
    entry.update(kwargs)
    return entry


def _scored_entry(**kwargs) -> dict:
    entry = {
        "schema_version": "1.2",
        "task": "secure the token store",
        "timestamp": "2025-01-01T00:00:00Z",
        "execution_model": "anthropic/claude-sonnet-4.6",
        "routing_category": "security",
        "routing_selected_model": "anthropic/claude-sonnet-4.6",
        "routing_candidates": ["anthropic/claude-sonnet-4.6", "z-ai/glm-5.1"],
        "routing_scores": {"anthropic/claude-sonnet-4.6": 14.0, "z-ai/glm-5.1": 11.0},
    }
    entry.update(kwargs)
    return entry


def _tier_dispatch_entry(validator_applied: bool = True, **kwargs) -> dict:
    tdr = {
        "enabled": True,
        "applied": True,
        "tier_source": "candidate_scoring",
        "planner_model": "anthropic/claude-sonnet-4.6",
        "executor_model": "z-ai/glm-5.1",
        "validator_model": "anthropic/claude-sonnet-4.6",
        "planner_model_actual": "anthropic/claude-sonnet-4.6",
        "executor_model_actual": "z-ai/glm-5.1",
        "validator_model_actual": (
            "anthropic/claude-sonnet-4.6" if validator_applied else None
        ),
        "validator_dispatch_status": "applied" if validator_applied else "reserved",
    }
    entry = {
        "schema_version": "1.2",
        "task": "tier dispatch run",
        "timestamp": "2025-01-01T00:00:00Z",
        "execution_model": "z-ai/glm-5.1",
        "routing_category": "complex",
        "tier_dispatch_receipt": tdr,
    }
    entry.update(kwargs)
    return entry


# ---------------------------------------------------------------------------
# 1. Single-model default run
# ---------------------------------------------------------------------------


class TestSingleModelRun(unittest.TestCase):
    def test_keyword_run_runtime_and_role_truth(self) -> None:
        rt = build_routing_truth(_keyword_entry())
        self.assertEqual(rt.runtime_model, "z-ai/glm-5.1")
        self.assertEqual(rt.routing_mode, "keyword")
        self.assertEqual(rt.selection_source, "deterministic")
        self.assertEqual(rt.role_selection_mode, "advisory_only")
        self.assertEqual(rt.role_dispatch_status, "not_dispatched")
        self.assertFalse(rt.planner_dispatched)
        self.assertFalse(rt.executor_dispatched)
        self.assertFalse(rt.validator_dispatched)

    def test_routing_mode_never_advisory_flavoured(self) -> None:
        # Even with rich per-role advisory metadata, the runtime routing mode
        # must describe the real single-model choice.
        rt = build_routing_truth(_keyword_entry())
        self.assertIn(
            rt.routing_mode,
            {"keyword", "scored", "user_selected", "tier_dispatch", "unknown"},
        )
        self.assertNotIn("advisory", rt.routing_mode)

    def test_scored_run_runtime_mode(self) -> None:
        rt = build_routing_truth(_scored_entry())
        self.assertEqual(rt.routing_mode, "scored")
        self.assertEqual(rt.selection_source, "deterministic")
        self.assertEqual(rt.runtime_model, "anthropic/claude-sonnet-4.6")
        # No per-role metadata at all -> nothing to overclaim.
        self.assertEqual(rt.role_selection_mode, "unavailable")
        self.assertEqual(rt.role_dispatch_status, "not_dispatched")

    def test_advisory_recommended_models_recorded_not_dispatched(self) -> None:
        rt = build_routing_truth(_keyword_entry())
        # Recommended tiers are surfaced, but never marked dispatched.
        self.assertEqual(rt.planner_model, "frontier-reasoning-model")
        self.assertEqual(rt.executor_model, "balanced-coding-model")
        self.assertEqual(rt.validator_model, "independent-validator-model")
        self.assertFalse(
            rt.planner_dispatched or rt.executor_dispatched or rt.validator_dispatched
        )


# ---------------------------------------------------------------------------
# 2. Experimental tier dispatch
# ---------------------------------------------------------------------------


class TestTierDispatch(unittest.TestCase):
    def test_full_dispatch(self) -> None:
        rt = build_routing_truth(_tier_dispatch_entry(validator_applied=True))
        self.assertEqual(rt.routing_mode, "tier_dispatch")
        self.assertEqual(rt.selection_source, "experimental")
        self.assertEqual(rt.role_selection_mode, "dispatched")
        self.assertEqual(rt.role_dispatch_status, "dispatched")
        self.assertTrue(rt.planner_dispatched)
        self.assertTrue(rt.executor_dispatched)
        self.assertTrue(rt.validator_dispatched)

    def test_partial_dispatch(self) -> None:
        rt = build_routing_truth(_tier_dispatch_entry(validator_applied=False))
        self.assertEqual(rt.routing_mode, "tier_dispatch")
        self.assertEqual(rt.role_dispatch_status, "partially_dispatched")
        self.assertTrue(rt.planner_dispatched)
        self.assertTrue(rt.executor_dispatched)
        self.assertFalse(rt.validator_dispatched)

    def test_enabled_but_not_applied_is_advisory(self) -> None:
        entry = _tier_dispatch_entry()
        entry["tier_dispatch_receipt"]["applied"] = False
        rt = build_routing_truth(entry)
        # Reserved/not-applied dispatch must not claim a dispatched runtime mode.
        self.assertNotEqual(rt.routing_mode, "tier_dispatch")
        self.assertEqual(rt.role_dispatch_status, "not_dispatched")


# ---------------------------------------------------------------------------
# 3. Receipt/proof output does not imply per-role dispatch
# ---------------------------------------------------------------------------


class TestNoOverclaim(unittest.TestCase):
    def test_render_default_states_advisory(self) -> None:
        rt = build_routing_truth(_keyword_entry())
        lines = render_routing_truth_lines(rt, "default")
        text = "\n".join(lines)
        self.assertIn("advisory only", text)
        self.assertIn("not dispatched", text)

    def test_render_dispatched_names_roles(self) -> None:
        rt = build_routing_truth(_tier_dispatch_entry())
        lines = render_routing_truth_lines(rt, "more")
        text = "\n".join(lines)
        self.assertIn("dispatched", text)
        self.assertIn("planner", text)

    def test_proof_contract_model_detail_carries_role_truth(self) -> None:
        contract = build_shard_proof_contract(_keyword_entry())
        model_section = next(s for s in contract["sections"] if s["name"] == "model")
        self.assertEqual(model_section["status"], "present")
        self.assertIn("advisory_only", model_section["detail"])

    def test_proof_contract_keeps_17_sections(self) -> None:
        contract = build_shard_proof_contract(_keyword_entry())
        self.assertEqual(len(contract["sections"]), 17)


# ---------------------------------------------------------------------------
# 4. Old/minimal records render safely
# ---------------------------------------------------------------------------


class TestLegacyRecords(unittest.TestCase):
    def test_minimal_entry(self) -> None:
        rt = build_routing_truth({"task": "x", "timestamp": "2024-01-01T00:00:00Z"})
        self.assertEqual(rt.routing_mode, "unknown")
        self.assertEqual(rt.selection_source, "unknown")
        self.assertEqual(rt.role_selection_mode, "unavailable")
        self.assertEqual(rt.role_dispatch_status, "not_dispatched")

    def test_non_dict_entry(self) -> None:
        rt = build_routing_truth(None)
        self.assertEqual(rt.role_selection_mode, "unavailable")
        self.assertIsNone(rt.runtime_model)

    def test_to_dict_is_json_serialisable(self) -> None:
        d = routing_truth_to_dict(build_routing_truth(_keyword_entry()))
        json.loads(json.dumps(d))
        self.assertIn("routing_mode", d)
        self.assertIn("role_selection_mode", d)


# ---------------------------------------------------------------------------
# 5. CLI surfaces: human + JSON
# ---------------------------------------------------------------------------


def _write_runs(entries: list[dict]) -> None:
    from pathlib import Path
    log_dir = Path(".openshard")
    log_dir.mkdir(parents=True, exist_ok=True)
    with (log_dir / "runs.jsonl").open("w", encoding="utf-8") as fh:
        for entry in entries:
            fh.write(json.dumps(entry) + "\n")


class TestCliSurfaces(unittest.TestCase):
    def test_last_default_states_advisory(self) -> None:
        runner = CliRunner()
        with runner.isolated_filesystem():
            _write_runs([_keyword_entry()])
            result = runner.invoke(cli, ["last"])
            self.assertEqual(result.exit_code, 0)
            self.assertIn("advisory only", result.output)
            self.assertIn("not dispatched", result.output)

    def test_last_json_valid_and_has_routing_truth(self) -> None:
        runner = CliRunner()
        with runner.isolated_filesystem():
            _write_runs([_keyword_entry()])
            result = runner.invoke(cli, ["last", "--json"])
            self.assertEqual(result.exit_code, 0)
            payload = json.loads(result.output)
            run = payload["run"]
            rt = run["routing_truth"]
            self.assertEqual(rt["routing_mode"], "keyword")
            self.assertEqual(rt["role_selection_mode"], "advisory_only")
            # Backward compatible: existing keys still present.
            self.assertIn("execution_model", run)
            self.assertIn("routing_selected_model", run)

    def test_old_record_renders_without_error(self) -> None:
        runner = CliRunner()
        with runner.isolated_filesystem():
            _write_runs([{"task": "legacy", "timestamp": "2024-01-01T00:00:00Z"}])
            result = runner.invoke(cli, ["last"])
            self.assertEqual(result.exit_code, 0)
            result_json = runner.invoke(cli, ["last", "--json"])
            self.assertEqual(result_json.exit_code, 0)
            json.loads(result_json.output)


# ---------------------------------------------------------------------------
# 6. Per-role dispatch lists (dispatched_roles / advisory_only_roles) — v1
# ---------------------------------------------------------------------------


def _receipt_entry(
    executor_model_actual: str | None = None,
    validator_model_actual: str | None = None,
    planner_model_actual: str | None = None,
    validator_dispatch_status: str = "",
) -> dict:
    """Tier dispatch entry with configurable actual-model fields."""
    tdr: dict = {
        "enabled": True,
        "applied": True,
        "tier_source": "category_fallback",
        "planner_model": "anthropic/claude-sonnet-4.6",
        "executor_model": "z-ai/glm-5.1",
        "validator_model": "anthropic/claude-sonnet-4.6",
        "planner_model_actual": planner_model_actual,
        "executor_model_actual": executor_model_actual,
        "validator_model_actual": validator_model_actual,
        "validator_dispatch_status": validator_dispatch_status,
    }
    return {
        "schema_version": "1.2",
        "task": "dispatch test",
        "timestamp": "2025-01-01T00:00:00Z",
        "execution_model": "z-ai/glm-5.1",
        "tier_dispatch_receipt": tdr,
    }


class TestDispatchedRoleLists(unittest.TestCase):
    def test_executor_dispatched_when_actual_set(self) -> None:
        entry = _receipt_entry(executor_model_actual="anthropic/claude-sonnet-4.6")
        rt = build_routing_truth(entry)
        self.assertTrue(rt.executor_dispatched)
        self.assertIn("executor", rt.dispatched_roles)
        self.assertNotIn("executor", rt.advisory_only_roles)

    def test_validator_dispatched_when_actual_set(self) -> None:
        entry = _receipt_entry(
            validator_model_actual="z-ai/glm-5.1",
            validator_dispatch_status="applied",
        )
        rt = build_routing_truth(entry)
        self.assertTrue(rt.validator_dispatched)
        self.assertIn("validator", rt.dispatched_roles)
        self.assertNotIn("validator", rt.advisory_only_roles)

    def test_validator_dispatched_via_status_alone(self) -> None:
        # validator_dispatch_status="applied" is sufficient even without model_actual.
        entry = _receipt_entry(validator_dispatch_status="applied")
        rt = build_routing_truth(entry)
        self.assertTrue(rt.validator_dispatched)
        self.assertIn("validator", rt.dispatched_roles)

    def test_planner_always_advisory_no_actual_field(self) -> None:
        # planner_model_actual is never set by real code paths.
        entry = _receipt_entry(
            executor_model_actual="z-ai/glm-5.1",
            validator_model_actual="anthropic/claude-sonnet-4.6",
        )
        rt = build_routing_truth(entry)
        self.assertFalse(rt.planner_dispatched)
        self.assertIn("planner", rt.advisory_only_roles)
        self.assertNotIn("planner", rt.dispatched_roles)

    def test_advisory_only_roles_no_tier_dispatch(self) -> None:
        rt = build_routing_truth(_keyword_entry())
        self.assertEqual(rt.dispatched_roles, [])
        self.assertIn("planner", rt.advisory_only_roles)
        self.assertIn("executor", rt.advisory_only_roles)
        self.assertIn("validator", rt.advisory_only_roles)

    def test_advisory_only_roles_minimal_entry(self) -> None:
        rt = build_routing_truth({"task": "x", "timestamp": "2024-01-01T00:00:00Z"})
        self.assertEqual(rt.dispatched_roles, [])
        self.assertEqual(set(rt.advisory_only_roles), {"planner", "executor", "validator"})

    def test_dispatched_roles_partial_executor_only(self) -> None:
        entry = _receipt_entry(executor_model_actual="z-ai/glm-5.1")
        rt = build_routing_truth(entry)
        self.assertEqual(rt.dispatched_roles, ["executor"])
        self.assertIn("planner", rt.advisory_only_roles)
        self.assertIn("validator", rt.advisory_only_roles)
        self.assertNotIn("executor", rt.advisory_only_roles)

    def test_dispatched_roles_full_dispatch(self) -> None:
        rt = build_routing_truth(_tier_dispatch_entry(validator_applied=True))
        self.assertEqual(set(rt.dispatched_roles), {"planner", "executor", "validator"})
        self.assertEqual(rt.advisory_only_roles, [])

    def test_dispatched_and_advisory_lists_are_exhaustive(self) -> None:
        # Every role must appear in exactly one list.
        for entry in [
            _keyword_entry(),
            _tier_dispatch_entry(validator_applied=True),
            _tier_dispatch_entry(validator_applied=False),
            _receipt_entry(executor_model_actual="z-ai/glm-5.1"),
        ]:
            rt = build_routing_truth(entry)
            all_roles = set(rt.dispatched_roles) | set(rt.advisory_only_roles)
            self.assertEqual(all_roles, {"planner", "executor", "validator"}, msg=str(entry))
            overlap = set(rt.dispatched_roles) & set(rt.advisory_only_roles)
            self.assertEqual(overlap, set(), msg=str(entry))

    def test_new_fields_json_serialisable(self) -> None:
        for entry in [_keyword_entry(), _tier_dispatch_entry()]:
            d = routing_truth_to_dict(build_routing_truth(entry))
            roundtripped = json.loads(json.dumps(d))
            self.assertIn("dispatched_roles", roundtripped)
            self.assertIn("advisory_only_roles", roundtripped)
            self.assertIsInstance(roundtripped["dispatched_roles"], list)
            self.assertIsInstance(roundtripped["advisory_only_roles"], list)


# ---------------------------------------------------------------------------
# 5. render_routing_truth_lines — role truth block at more/full detail
# ---------------------------------------------------------------------------


class TestRenderRoleTruthBlock(unittest.TestCase):
    """The Dispatched roles / Advisory only lines at more/full detail level."""

    def _lines(self, entry: dict, detail: str = "more") -> list[str]:
        return render_routing_truth_lines(build_routing_truth(entry), detail)

    def test_more_shows_dispatched_roles_executor_validator(self) -> None:
        entry = _receipt_entry(
            executor_model_actual="z-ai/glm-5.1",
            validator_dispatch_status="applied",
        )
        lines = self._lines(entry)
        self.assertTrue(any("Dispatched roles: executor, validator" in ln for ln in lines))

    def test_more_shows_advisory_only_planner(self) -> None:
        entry = _receipt_entry(
            executor_model_actual="z-ai/glm-5.1",
            validator_dispatch_status="applied",
        )
        lines = self._lines(entry)
        self.assertTrue(any("Advisory only: planner" in ln for ln in lines))

    def test_more_no_dispatch_shows_none(self) -> None:
        # advisory-only entry: executor and validator not dispatched
        lines = self._lines(_keyword_entry())
        self.assertTrue(any("Dispatched roles: none" in ln for ln in lines))

    def test_more_planner_always_advisory(self) -> None:
        # Even with executor+validator dispatched, planner is advisory-only.
        entry = _receipt_entry(
            executor_model_actual="z-ai/glm-5.1",
            validator_dispatch_status="applied",
        )
        lines = self._lines(entry)
        combined = "\n".join(lines)
        self.assertIn("Advisory only: planner", combined)
        self.assertNotIn("Dispatched roles: planner", combined)

    def test_default_detail_no_role_truth_block(self) -> None:
        # The new block must NOT appear at default detail.
        lines = self._lines(_keyword_entry(), detail="default")
        combined = "\n".join(lines)
        self.assertNotIn("Dispatched roles:", combined)
        self.assertNotIn("Advisory only:", combined)

    def test_legacy_shard_role_unavailable_renders_safely(self) -> None:
        # Minimal entry has no tier dispatch and no advisory metadata →
        # ROLE_UNAVAILABLE → role-truth block must not appear.
        entry = {"task": "x", "timestamp": "2024-01-01T00:00:00Z"}
        lines = self._lines(entry)
        combined = "\n".join(lines)
        self.assertNotIn("Dispatched roles:", combined)
        self.assertNotIn("Advisory only:", combined)

    def test_full_detail_same_as_more(self) -> None:
        entry = _receipt_entry(executor_model_actual="z-ai/glm-5.1")
        more_lines = self._lines(entry, detail="more")
        full_lines = self._lines(entry, detail="full")
        self.assertEqual(more_lines, full_lines)


if __name__ == "__main__":
    unittest.main()
