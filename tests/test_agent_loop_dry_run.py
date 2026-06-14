from __future__ import annotations

import unittest

from openshard.native.agent_loop_dry_run import DryRunStep, run_dry_loop
from openshard.native.agent_loop_types import (
    AgentAction,
    AgentDecision,
    AgentObservation,
    IterationCost,
    ToolResult,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _action(kind: str = "read_file", *, needs_approval: bool = False, idx: int = 0) -> AgentAction:
    return AgentAction(
        action_id=f"a{idx}",
        iteration_index=idx,
        kind=kind,
        args={"path": "foo.py"},
        rationale="dry-run test",
        needs_approval=needs_approval,
    )


def _result(ok: bool = True, action_id: str = "a0") -> ToolResult:
    return ToolResult(action_id=action_id, ok=ok, output="ok" if ok else None,
                      error=None if ok else "boom")


def _obs(status: str = "success", action_id: str = "a0") -> AgentObservation:
    return AgentObservation(
        action_id=action_id,
        iteration_index=0,
        status=status,
        summary="test observation",
    )


def _decision(kind: str = "continue", stop_reason=None, idx: int = 0) -> AgentDecision:
    return AgentDecision(
        iteration_index=idx,
        decision=kind,
        reason="dry-run",
        stop_reason=stop_reason,
    )


def _step(
    action_kind: str = "read_file",
    decision_kind: str = "continue",
    *,
    needs_approval: bool = False,
    approval_granted: bool = True,
    ok: bool = True,
    idx: int = 0,
    stop_reason=None,
) -> DryRunStep:
    a = _action(action_kind, needs_approval=needs_approval, idx=idx)
    return DryRunStep(
        action=a,
        tool_result=_result(ok=ok, action_id=a.action_id),
        observation=_obs(action_id=a.action_id),
        decision=_decision(decision_kind, stop_reason=stop_reason, idx=idx),
        approval_granted=approval_granted,
    )


# ---------------------------------------------------------------------------
# Empty / trivial
# ---------------------------------------------------------------------------

class TestEmptySteps(unittest.TestCase):
    def test_empty_steps_completes(self):
        result = run_dry_loop("empty task", [])
        self.assertTrue(result.state.completed)
        self.assertEqual(result.state.current_state, "completed")
        self.assertEqual(result.state.iteration_count, 0)
        self.assertEqual(result.state.stop_reason, "completed")

    def test_empty_steps_has_start_event(self):
        result = run_dry_loop("t", [])
        self.assertTrue(any(e.event_type == "loop_started" for e in result.events))
        self.assertTrue(any(e.event_type == "loop_halted" for e in result.events))


# ---------------------------------------------------------------------------
# Single step
# ---------------------------------------------------------------------------

class TestSingleStep(unittest.TestCase):
    def _run(self, decision_kind="completed", stop_reason=None):
        return run_dry_loop("task", [_step(decision_kind=decision_kind,
                                           stop_reason=stop_reason)])

    def test_single_continue_step_completes_naturally(self):
        result = self._run("continue")
        self.assertEqual(result.state.current_state, "completed")
        self.assertTrue(result.state.completed)

    def test_single_completed_step(self):
        result = self._run("completed", stop_reason="completed")
        self.assertEqual(result.state.current_state, "completed")
        self.assertTrue(result.state.completed)

    def test_iteration_count_is_one(self):
        result = self._run()
        self.assertEqual(result.state.iteration_count, 1)

    def test_receipt_stored(self):
        result = self._run()
        self.assertEqual(len(result.state.iteration_history), 1)

    def test_receipt_fields_populated(self):
        result = self._run()
        r = result.state.iteration_history[0]
        self.assertIsNotNone(r.action)
        self.assertIsNotNone(r.result)
        self.assertIsNotNone(r.observation)
        self.assertIsNotNone(r.decision)
        self.assertIsNotNone(r.cost)
        self.assertEqual(r.iteration_index, 1)

    def test_raw_content_stored_false_on_result(self):
        result = self._run()
        self.assertFalse(result.raw_content_stored)
        self.assertFalse(result.state.raw_content_stored)
        r = result.state.iteration_history[0]
        self.assertFalse(r.raw_content_stored)
        self.assertFalse(r.result.raw_content_stored)


# ---------------------------------------------------------------------------
# Multi-step
# ---------------------------------------------------------------------------

class TestMultiStep(unittest.TestCase):
    def _three_continues(self):
        return run_dry_loop("task", [
            _step(idx=0),
            _step(idx=1),
            _step(idx=2),
        ])

    def test_three_continues_completes(self):
        result = self._three_continues()
        self.assertEqual(result.state.current_state, "completed")
        self.assertEqual(result.state.iteration_count, 3)
        self.assertEqual(len(result.state.iteration_history), 3)

    def test_receipt_indices_sequential(self):
        result = self._three_continues()
        for i, r in enumerate(result.state.iteration_history, start=1):
            self.assertEqual(r.iteration_index, i)

    def test_final_completed_step_halts(self):
        result = run_dry_loop("task", [
            _step(idx=0),
            _step(decision_kind="completed", stop_reason="completed", idx=1),
            _step(idx=2),  # should never run
        ])
        self.assertEqual(result.state.iteration_count, 2)
        self.assertEqual(len(result.state.iteration_history), 2)
        self.assertTrue(result.state.completed)


# ---------------------------------------------------------------------------
# Max iterations
# ---------------------------------------------------------------------------

class TestMaxIterations(unittest.TestCase):
    def test_halts_at_max(self):
        steps = [_step(idx=i) for i in range(10)]
        result = run_dry_loop("task", steps, max_iterations=3)
        self.assertEqual(result.state.current_state, "max_iterations_reached")
        self.assertEqual(result.state.stop_reason, "max_iterations_reached")
        self.assertEqual(result.state.iteration_count, 3)

    def test_halted_event_recorded(self):
        steps = [_step(idx=i) for i in range(5)]
        result = run_dry_loop("task", steps, max_iterations=2)
        types = [e.event_type for e in result.events]
        self.assertIn("loop_halted", types)


# ---------------------------------------------------------------------------
# Approval gate
# ---------------------------------------------------------------------------

class TestApprovalGate(unittest.TestCase):
    def test_approved_action_continues(self):
        result = run_dry_loop("task", [
            _step("write_file", needs_approval=True, approval_granted=True),
        ])
        self.assertEqual(result.state.current_state, "completed")
        self.assertEqual(result.state.iteration_count, 1)

    def test_approval_request_stored_in_receipt(self):
        result = run_dry_loop("task", [
            _step("write_file", needs_approval=True, approval_granted=True),
        ])
        r = result.state.iteration_history[0]
        self.assertIsNotNone(r.approval_request)
        self.assertIsNotNone(r.approval_receipt)
        self.assertTrue(r.approval_receipt.granted)

    def test_denied_approval_blocks_loop(self):
        result = run_dry_loop("task", [
            _step("write_file", needs_approval=True, approval_granted=False),
        ])
        self.assertEqual(result.state.current_state, "blocked")
        self.assertEqual(result.state.stop_reason, "approval_denied")
        self.assertFalse(result.state.completed)

    def test_denied_stops_before_subsequent_steps(self):
        result = run_dry_loop("task", [
            _step("write_file", needs_approval=True, approval_granted=False, idx=0),
            _step(idx=1),  # should never run
        ])
        self.assertEqual(result.state.iteration_count, 1)
        self.assertEqual(len(result.state.iteration_history), 1)

    def test_denied_receipt_has_approval_denied_decision(self):
        result = run_dry_loop("task", [
            _step("write_file", needs_approval=True, approval_granted=False),
        ])
        r = result.state.iteration_history[0]
        self.assertEqual(r.decision.stop_reason, "approval_denied")

    def test_approval_events_recorded(self):
        result = run_dry_loop("task", [
            _step("write_file", needs_approval=True, approval_granted=True),
        ])
        types = [e.event_type for e in result.events]
        self.assertIn("approval_requested", types)
        self.assertIn("approval_received", types)


# ---------------------------------------------------------------------------
# Terminal decision states
# ---------------------------------------------------------------------------

class TestTerminalDecisions(unittest.TestCase):
    def _run_terminal(self, decision_kind: str, stop_reason: str):
        return run_dry_loop("task", [
            _step(decision_kind=decision_kind, stop_reason=stop_reason),
            _step(idx=1),  # must not run
        ])

    def test_blocked_terminal(self):
        r = self._run_terminal("blocked", "repeated_failure")
        self.assertEqual(r.state.current_state, "blocked")
        self.assertFalse(r.state.completed)
        self.assertEqual(r.state.iteration_count, 1)

    def test_unsafe_terminal(self):
        r = self._run_terminal("unsafe", "unsafe_action_detected")
        self.assertEqual(r.state.current_state, "unsafe")
        self.assertFalse(r.state.completed)

    def test_hand_off_terminal(self):
        r = self._run_terminal("hand_off", "handed_to_human")
        self.assertEqual(r.state.current_state, "handed_to_human")
        self.assertFalse(r.state.completed)

    def test_stop_reason_propagated(self):
        r = self._run_terminal("blocked", "policy_violation")
        self.assertEqual(r.state.stop_reason, "policy_violation")


# ---------------------------------------------------------------------------
# Event log
# ---------------------------------------------------------------------------

class TestEventLog(unittest.TestCase):
    def test_event_order_for_single_step(self):
        result = run_dry_loop("task", [_step(decision_kind="completed",
                                             stop_reason="completed")])
        types = [e.event_type for e in result.events]
        loop_started_i = types.index("loop_started")
        iteration_started_i = types.index("iteration_started")
        action_selected_i = types.index("action_selected")
        action_executed_i = types.index("action_executed")
        decision_made_i = types.index("decision_made")
        loop_halted_i = types.index("loop_halted")
        self.assertLess(loop_started_i, iteration_started_i)
        self.assertLess(iteration_started_i, action_selected_i)
        self.assertLess(action_selected_i, action_executed_i)
        self.assertLess(action_executed_i, decision_made_i)
        self.assertLess(decision_made_i, loop_halted_i)

    def test_all_events_have_run_id(self):
        result = run_dry_loop("task", [_step()], run_id="myrun")
        for e in result.events:
            self.assertEqual(e.run_id, "myrun")

    def test_all_events_have_timestamp(self):
        result = run_dry_loop("task", [_step()])
        for e in result.events:
            self.assertNotEqual(e.timestamp, "")

    def test_all_events_raw_content_false(self):
        result = run_dry_loop("task", [_step()])
        for e in result.events:
            self.assertFalse(e.raw_content_stored)

    def test_run_id_propagated_to_state(self):
        result = run_dry_loop("task", [], run_id="r123")
        self.assertEqual(result.state.run_id, "r123")


# ---------------------------------------------------------------------------
# Cost tracking
# ---------------------------------------------------------------------------

class TestCostTracking(unittest.TestCase):
    def test_cost_stored_in_receipt(self):
        cost = IterationCost(input_tokens=100, output_tokens=50,
                             estimated_usd=0.01, cumulative_usd=0.01)
        step = DryRunStep(
            action=_action(),
            tool_result=_result(),
            observation=_obs(),
            decision=_decision(),
            cost=cost,
        )
        result = run_dry_loop("task", [step])
        r = result.state.iteration_history[0]
        self.assertEqual(r.cost.input_tokens, 100)
        self.assertEqual(r.cost.estimated_usd, 0.01)

    def test_default_cost_is_zero(self):
        result = run_dry_loop("task", [_step()])
        r = result.state.iteration_history[0]
        self.assertEqual(r.cost.estimated_usd, 0.0)


if __name__ == "__main__":
    unittest.main()
