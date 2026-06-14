from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

# ---------------------------------------------------------------------------
# Literal types — used as discriminated unions throughout the agent loop.
# Keeping these as module-level Literals (not enums) matches the existing
# pattern in loop.py (NativeLoopPhase) and is cheaper to extend later.
# ---------------------------------------------------------------------------

StopReason = Literal[
    "completed",
    "max_iterations_reached",
    "budget_exceeded",
    "unsafe_action_detected",
    "repeated_failure",
    "approval_denied",
    "handed_to_human",
    "blocked_tool",
    "unknown_tool",
    "hallucinated_path",
    "policy_violation",
    "exception",
]

# ActionKind mirrors the safe/needs_approval/blocked split in tools.py.
# "ask_human" and "finish" are loop-level signals, not tool calls.
ActionKind = Literal[
    "list_files",
    "read_file",
    "search_repo",
    "get_git_diff",
    "write_file",       # needs_approval
    "run_command",      # needs_approval
    "run_verification", # safe
    "ask_human",        # always asks
    "finish",           # terminal — signals task complete
]

LoopState = Literal[
    "received_task",
    "understood_task",
    "planned",
    "awaiting_approval",
    "acting",
    "observed",
    "inspecting",
    "deciding_next_step",
    "completed",
    "blocked",
    "unsafe",
    "failed",
    "max_iterations_reached",
    "handed_to_human",
]

AgentDecisionKind = Literal[
    "continue",
    "completed",
    "blocked",
    "unsafe",
    "hand_off",
]

ObservationStatus = Literal[
    "success",
    "failure",
    "partial",
    "unexpected",
]

VerificationStatus = Literal[
    "passed",
    "failed",
    "skipped",
    "not_run",
]

AgentLoopEventType = Literal[
    "loop_started",
    "iteration_started",
    "action_selected",
    "approval_requested",
    "approval_received",
    "action_executed",
    "observation_recorded",
    "decision_made",
    "loop_halted",
]

# ---------------------------------------------------------------------------
# Core action / result types
# ---------------------------------------------------------------------------

@dataclass
class AgentAction:
    """A single tool call chosen by the model for one loop iteration."""
    action_id: str
    iteration_index: int
    kind: ActionKind
    # args is Any so each tool can define its own shape without coupling here.
    args: dict[str, Any]
    # rationale is capped at 200 chars before storage.
    rationale: str
    needs_approval: bool
    estimated_cost_usd: float = 0.0


@dataclass
class ToolResult:
    """Sanitised outcome of executing an AgentAction.

    Raw stdout/stderr must never be stored in logs. Callers must sanitise
    output before setting it here. raw_content_stored is always False.
    """
    action_id: str
    ok: bool
    output: str | None = None
    error: str | None = None
    exit_code: int | None = None
    duration_ms: int = 0
    raw_content_stored: bool = False


@dataclass
class AgentObservation:
    """What the loop learned from a ToolResult."""
    action_id: str
    iteration_index: int
    status: ObservationStatus
    # summary is capped at 120 chars before storage.
    summary: str
    key_findings: list[str] = field(default_factory=list)
    new_risks_detected: list[str] = field(default_factory=list)
    verification_status: VerificationStatus = "not_run"

# ---------------------------------------------------------------------------
# Approval types
# ---------------------------------------------------------------------------

@dataclass
class ApprovalRequest:
    """Emitted before any needs_approval action is executed."""
    request_id: str
    iteration_index: int
    action: AgentAction
    risk_reason: str
    policy_gate: str


@dataclass
class ApprovalReceipt:
    """Human decision recorded after an ApprovalRequest."""
    request_id: str
    granted: bool
    # modified_action allows the human to change args before approval.
    modified_action: AgentAction | None = None
    reason: str | None = None
    timestamp: str = ""

# ---------------------------------------------------------------------------
# Decision
# ---------------------------------------------------------------------------

@dataclass
class AgentDecision:
    """The loop's choice at the end of each iteration."""
    iteration_index: int
    decision: AgentDecisionKind
    reason: str
    stop_reason: StopReason | None = None
    next_action_hint: str | None = None

# ---------------------------------------------------------------------------
# Cost tracking
# ---------------------------------------------------------------------------

@dataclass
class IterationCost:
    """Token and dollar cost for one loop iteration.

    cumulative_usd is the running total across all iterations so far.
    Budget enforcement (if any) is done by the executor, not here.
    """
    input_tokens: int = 0
    output_tokens: int = 0
    tool_calls: int = 0
    estimated_usd: float = 0.0
    cumulative_usd: float = 0.0

# ---------------------------------------------------------------------------
# Per-iteration receipt
# ---------------------------------------------------------------------------

@dataclass
class ReceiptIteration:
    """Complete record of one agent loop iteration.

    All optional fields are None until that phase runs. raw_content_stored
    is always False — raw tool output is never persisted.
    """
    schema_version: int = 1
    iteration_index: int = 0
    timestamp: str = ""
    action: AgentAction | None = None
    approval_request: ApprovalRequest | None = None
    approval_receipt: ApprovalReceipt | None = None
    result: ToolResult | None = None
    observation: AgentObservation | None = None
    decision: AgentDecision | None = None
    cost: IterationCost | None = None
    raw_content_stored: bool = False

# ---------------------------------------------------------------------------
# Loop state — the live object carried across all iterations
# ---------------------------------------------------------------------------

@dataclass
class AgentLoopState:
    """Mutable state object for one agent loop run.

    max_iterations defaults to 20. Budget caps are not stored here —
    they are injected at runtime by the executor so the user can configure
    them per run without changing this schema.
    """
    schema_version: int = 1
    run_id: str = ""
    task: str = ""
    max_iterations: int = 20
    iteration_count: int = 0
    current_state: LoopState = "received_task"
    iteration_history: list[ReceiptIteration] = field(default_factory=list)
    stop_reason: StopReason | None = None
    completed: bool = False
    raw_content_stored: bool = False

    def is_terminal(self) -> bool:
        """True when the loop must not start another iteration."""
        return self.current_state in (
            "completed",
            "blocked",
            "unsafe",
            "failed",
            "max_iterations_reached",
            "handed_to_human",
        )

    def iterations_remaining(self) -> int:
        return max(0, self.max_iterations - self.iteration_count)

# ---------------------------------------------------------------------------
# Audit event — one entry per significant loop transition
# ---------------------------------------------------------------------------

@dataclass
class AgentLoopEvent:
    """Lightweight audit record written to native_steps.jsonl per transition.

    summary is capped at 120 chars. raw_content_stored is always False.
    """
    schema_version: int = 1
    event_id: str = ""
    run_id: str = ""
    timestamp: str = ""
    event_type: AgentLoopEventType = "loop_started"
    iteration_index: int = 0
    state: LoopState = "received_task"
    summary: str = ""
    stop_reason: StopReason | None = None
    raw_content_stored: bool = False
