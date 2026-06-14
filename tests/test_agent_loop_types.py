from __future__ import annotations

import unittest

from openshard.native.agent_loop_types import (
    AgentAction,
    AgentDecision,
    AgentLoopEvent,
    AgentLoopState,
    AgentObservation,
    ApprovalReceipt,
    ApprovalRequest,
    IterationCost,
    ReceiptIteration,
    ToolResult,
)


class TestAgentAction(unittest.TestCase):
    def _make(self, **kw):
        defaults = dict(
            action_id="a1",
            iteration_index=0,
            kind="read_file",
            args={"path": "foo.py"},
            rationale="need to read foo",
            needs_approval=False,
        )
        return AgentAction(**{**defaults, **kw})

    def test_defaults(self):
        a = self._make()
        self.assertEqual(a.estimated_cost_usd, 0.0)
        self.assertFalse(a.needs_approval)

    def test_needs_approval_set(self):
        a = self._make(kind="write_file", needs_approval=True)
        self.assertTrue(a.needs_approval)

    def test_args_preserved(self):
        a = self._make(args={"path": "bar.py", "content": "hello"})
        self.assertEqual(a.args["content"], "hello")


class TestToolResult(unittest.TestCase):
    def test_ok_result(self):
        r = ToolResult(action_id="a1", ok=True, output="some text")
        self.assertTrue(r.ok)
        self.assertIsNone(r.error)
        self.assertFalse(r.raw_content_stored)

    def test_error_result(self):
        r = ToolResult(action_id="a1", ok=False, error="boom")
        self.assertFalse(r.ok)
        self.assertIsNone(r.output)

    def test_raw_content_stored_always_false(self):
        r = ToolResult(action_id="a1", ok=True)
        self.assertFalse(r.raw_content_stored)

    def test_duration_default(self):
        r = ToolResult(action_id="a1", ok=True)
        self.assertEqual(r.duration_ms, 0)


class TestAgentObservation(unittest.TestCase):
    def test_defaults(self):
        obs = AgentObservation(
            action_id="a1",
            iteration_index=0,
            status="success",
            summary="file read ok",
        )
        self.assertEqual(obs.key_findings, [])
        self.assertEqual(obs.new_risks_detected, [])
        self.assertEqual(obs.verification_status, "not_run")

    def test_key_findings_stored(self):
        obs = AgentObservation(
            action_id="a1",
            iteration_index=1,
            status="partial",
            summary="partial read",
            key_findings=["found X"],
        )
        self.assertIn("found X", obs.key_findings)


class TestApprovalRequest(unittest.TestCase):
    def _action(self):
        return AgentAction(
            action_id="a1",
            iteration_index=0,
            kind="write_file",
            args={"path": "out.py"},
            rationale="write result",
            needs_approval=True,
        )

    def test_fields(self):
        req = ApprovalRequest(
            request_id="r1",
            iteration_index=0,
            action=self._action(),
            risk_reason="mutates file",
            policy_gate="write_gate",
        )
        self.assertEqual(req.request_id, "r1")
        self.assertEqual(req.action.kind, "write_file")


class TestApprovalReceipt(unittest.TestCase):
    def test_granted(self):
        rec = ApprovalReceipt(request_id="r1", granted=True)
        self.assertTrue(rec.granted)
        self.assertIsNone(rec.modified_action)
        self.assertEqual(rec.timestamp, "")

    def test_denied_with_reason(self):
        rec = ApprovalReceipt(request_id="r1", granted=False, reason="too risky")
        self.assertFalse(rec.granted)
        self.assertEqual(rec.reason, "too risky")


class TestAgentDecision(unittest.TestCase):
    def test_continue(self):
        d = AgentDecision(iteration_index=0, decision="continue", reason="more to do")
        self.assertIsNone(d.stop_reason)
        self.assertIsNone(d.next_action_hint)

    def test_completed_with_stop_reason(self):
        d = AgentDecision(
            iteration_index=3,
            decision="completed",
            reason="done",
            stop_reason="completed",
        )
        self.assertEqual(d.stop_reason, "completed")


class TestIterationCost(unittest.TestCase):
    def test_defaults_all_zero(self):
        c = IterationCost()
        self.assertEqual(c.input_tokens, 0)
        self.assertEqual(c.output_tokens, 0)
        self.assertEqual(c.tool_calls, 0)
        self.assertEqual(c.estimated_usd, 0.0)
        self.assertEqual(c.cumulative_usd, 0.0)

    def test_cumulative_tracks_separately(self):
        c = IterationCost(estimated_usd=0.05, cumulative_usd=0.15)
        self.assertEqual(c.cumulative_usd, 0.15)


class TestReceiptIteration(unittest.TestCase):
    def test_defaults(self):
        r = ReceiptIteration()
        self.assertEqual(r.schema_version, 1)
        self.assertIsNone(r.action)
        self.assertIsNone(r.approval_request)
        self.assertIsNone(r.result)
        self.assertIsNone(r.observation)
        self.assertIsNone(r.decision)
        self.assertIsNone(r.cost)
        self.assertFalse(r.raw_content_stored)

    def test_schema_version(self):
        self.assertEqual(ReceiptIteration().schema_version, 1)


class TestAgentLoopState(unittest.TestCase):
    def test_defaults(self):
        s = AgentLoopState()
        self.assertEqual(s.schema_version, 1)
        self.assertEqual(s.max_iterations, 20)
        self.assertEqual(s.iteration_count, 0)
        self.assertEqual(s.current_state, "received_task")
        self.assertFalse(s.completed)
        self.assertFalse(s.raw_content_stored)
        self.assertIsNone(s.stop_reason)

    def test_is_terminal_false_initially(self):
        s = AgentLoopState()
        self.assertFalse(s.is_terminal())

    def test_is_terminal_completed(self):
        s = AgentLoopState(current_state="completed")
        self.assertTrue(s.is_terminal())

    def test_is_terminal_all_terminal_states(self):
        terminal_states = (
            "completed", "blocked", "unsafe", "failed",
            "max_iterations_reached", "handed_to_human",
        )
        for state in terminal_states:
            with self.subTest(state=state):
                self.assertTrue(AgentLoopState(current_state=state).is_terminal())

    def test_is_terminal_false_for_non_terminal(self):
        non_terminal = (
            "received_task", "understood_task", "planned", "awaiting_approval",
            "acting", "observed", "inspecting", "deciding_next_step",
        )
        for state in non_terminal:
            with self.subTest(state=state):
                self.assertFalse(AgentLoopState(current_state=state).is_terminal())

    def test_iterations_remaining(self):
        s = AgentLoopState(max_iterations=20, iteration_count=7)
        self.assertEqual(s.iterations_remaining(), 13)

    def test_iterations_remaining_never_negative(self):
        s = AgentLoopState(max_iterations=5, iteration_count=99)
        self.assertEqual(s.iterations_remaining(), 0)

    def test_iteration_history_starts_empty(self):
        s = AgentLoopState()
        self.assertEqual(s.iteration_history, [])

    def test_budget_not_stored_in_state(self):
        # Budget caps are injected at runtime, not stored on the state object.
        s = AgentLoopState()
        self.assertFalse(hasattr(s, "max_budget_usd"))


class TestAgentLoopEvent(unittest.TestCase):
    def test_defaults(self):
        e = AgentLoopEvent()
        self.assertEqual(e.schema_version, 1)
        self.assertEqual(e.event_type, "loop_started")
        self.assertEqual(e.state, "received_task")
        self.assertEqual(e.summary, "")
        self.assertIsNone(e.stop_reason)
        self.assertFalse(e.raw_content_stored)

    def test_event_type_set(self):
        e = AgentLoopEvent(event_type="action_executed", state="acting")
        self.assertEqual(e.event_type, "action_executed")
        self.assertEqual(e.state, "acting")


if __name__ == "__main__":
    unittest.main()
