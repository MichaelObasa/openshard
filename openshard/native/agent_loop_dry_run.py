from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime

from openshard.native.agent_loop_types import (
    AgentAction,
    AgentDecision,
    AgentDecisionKind,
    AgentLoopEvent,
    AgentLoopEventType,
    AgentLoopState,
    AgentObservation,
    ApprovalReceipt,
    ApprovalRequest,
    IterationCost,
    LoopState,
    ReceiptIteration,
    ToolResult,
)


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _uid() -> str:
    return uuid.uuid4().hex[:12]


@dataclass
class DryRunStep:
    """Pre-scripted inputs for one loop iteration.

    The tracer consumes these instead of calling an LLM or real tools.
    Set approval_granted=False to simulate a denial — the loop halts
    with stop_reason='approval_denied'.
    """
    action: AgentAction
    tool_result: ToolResult
    observation: AgentObservation
    decision: AgentDecision
    approval_granted: bool = True
    cost: IterationCost = field(default_factory=IterationCost)


@dataclass
class DryRunLoopResult:
    """Outcome of a dry-run loop trace.

    state.iteration_history holds one ReceiptIteration per iteration.
    events is the ordered audit log of AgentLoopEvent entries.
    raw_content_stored is always False — no tool output is persisted.
    """
    state: AgentLoopState
    events: list[AgentLoopEvent] = field(default_factory=list)
    raw_content_stored: bool = False


def _decision_to_state(decision: AgentDecision) -> LoopState:
    mapping: dict[AgentDecisionKind, LoopState] = {
        "continue": "acting",
        "completed": "completed",
        "blocked": "blocked",
        "unsafe": "unsafe",
        "hand_off": "handed_to_human",
    }
    return mapping.get(decision.decision, "failed")


def run_dry_loop(
    task: str,
    steps: list[DryRunStep],
    *,
    max_iterations: int = 20,
    run_id: str | None = None,
) -> DryRunLoopResult:
    """Walk the agent loop state machine with pre-scripted steps.

    No LLM calls, no tool side-effects. Verifies state transitions,
    receipt assembly, approval gates, and halt conditions.
    """
    rid = run_id or _uid()
    loop_state = AgentLoopState(
        run_id=rid,
        task=task,
        max_iterations=max_iterations,
        current_state="received_task",
    )
    events: list[AgentLoopEvent] = []

    def record(event_type: AgentLoopEventType, summary: str = "") -> None:
        events.append(AgentLoopEvent(
            event_id=_uid(),
            run_id=rid,
            timestamp=_now(),
            event_type=event_type,
            iteration_index=loop_state.iteration_count,
            state=loop_state.current_state,
            summary=summary[:120],
            stop_reason=loop_state.stop_reason,
        ))

    loop_state.current_state = "understood_task"
    loop_state.current_state = "planned"
    record("loop_started", f"dry-run: task={task[:60]!r}, steps={len(steps)}")

    finished_naturally = True

    for step in steps:
        if loop_state.is_terminal():
            finished_naturally = False
            break

        if loop_state.iteration_count >= loop_state.max_iterations:
            loop_state.current_state = "max_iterations_reached"
            loop_state.stop_reason = "max_iterations_reached"
            record("loop_halted", "max_iterations reached before step executed")
            finished_naturally = False
            break

        loop_state.iteration_count += 1
        loop_state.current_state = "acting"
        receipt = ReceiptIteration(
            iteration_index=loop_state.iteration_count,
            timestamp=_now(),
            action=step.action,
        )

        record("iteration_started", f"iteration {loop_state.iteration_count}")
        record("action_selected", f"kind={step.action.kind} id={step.action.action_id}")

        # --- approval gate ---
        if step.action.needs_approval:
            req = ApprovalRequest(
                request_id=_uid(),
                iteration_index=loop_state.iteration_count,
                action=step.action,
                risk_reason=f"kind '{step.action.kind}' requires approval",
                policy_gate="needs_approval",
            )
            receipt.approval_request = req
            loop_state.current_state = "awaiting_approval"
            record("approval_requested", f"gate=needs_approval kind={step.action.kind}")

            rec = ApprovalReceipt(
                request_id=req.request_id,
                granted=step.approval_granted,
                timestamp=_now(),
            )
            receipt.approval_receipt = rec
            record("approval_received", f"granted={step.approval_granted}")

            if not step.approval_granted:
                loop_state.current_state = "blocked"
                loop_state.stop_reason = "approval_denied"
                receipt.decision = AgentDecision(
                    iteration_index=loop_state.iteration_count,
                    decision="blocked",
                    reason="approval denied",
                    stop_reason="approval_denied",
                )
                loop_state.iteration_history.append(receipt)
                record("loop_halted", "approval denied — loop blocked")
                finished_naturally = False
                break

            loop_state.current_state = "acting"

        # --- tool result + observation ---
        receipt.result = step.tool_result
        loop_state.current_state = "observed"
        receipt.observation = step.observation
        record("action_executed", f"ok={step.tool_result.ok} id={step.action.action_id}")
        record("observation_recorded", step.observation.summary[:120])

        # --- decision ---
        loop_state.current_state = "deciding_next_step"
        receipt.decision = step.decision
        receipt.cost = step.cost
        record("decision_made", f"decision={step.decision.decision}")

        next_state = _decision_to_state(step.decision)
        loop_state.current_state = next_state

        if step.decision.stop_reason:
            loop_state.stop_reason = step.decision.stop_reason

        loop_state.iteration_history.append(receipt)

        if loop_state.is_terminal():
            loop_state.completed = next_state == "completed"
            record("loop_halted", f"terminal state={next_state}")
            finished_naturally = False
            break

    if finished_naturally and not loop_state.is_terminal():
        loop_state.current_state = "completed"
        loop_state.completed = True
        loop_state.stop_reason = "completed"
        record("loop_halted", "all scripted steps consumed — marked completed")

    return DryRunLoopResult(state=loop_state, events=events)
