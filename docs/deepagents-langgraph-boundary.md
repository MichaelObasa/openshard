# DeepAgents / LangGraph Backend Boundary

**OpenShard decides. Backend executes.**

This document defines where a DeepAgents or LangGraph execution backend fits in
OpenShard's architecture, what it is allowed to do, what it is forbidden from
doing, and what it must return. It is the authoritative reference for any future
backend adapter implementation.

---

## Why this document exists

OpenShard has an experimental OSN loop with pipeline-level step receipts and a
`backend_proof` phase. The next step is not to plug in DeepAgents immediately.
The next step is to define the boundary — so that when integration happens,
the control plane stays intact.

---

## What OpenShard owns (control plane)

OpenShard is the product brain. These responsibilities never move to a backend:

| Responsibility | Why it stays in OpenShard |
|---|---|
| Task classification | Determines form factor and routing |
| Workflow / form-factor decision | `ask`, `run`, `deep-run`, `osn-run` |
| Model routing | Which provider, which model, which tier |
| Model policy | Cost caps, risk tiers, approval modes |
| Risk scoring | Path safety, stack mismatch, change budget |
| Approval gates | `auto`, `smart`, `ask` — enforced by `GateEvaluator` |
| Verification plan | What to verify and how |
| Cost tracking | Token counts, duration, provider cost |
| Run receipt | Append-only `.openshard/runs.jsonl` |
| Audit trail | Every phase logged via `NativeLoopTrace` |
| Feedback / outcome learning | `feedback_scoring.py`, history adjustments |

---

## What a backend may do (execution plane)

A backend is an execution environment that receives a bounded task packet and
returns structured evidence. It may:

- Receive a bounded task packet from OpenShard (task text + compact context)
- Use only tools that OpenShard explicitly approves for this invocation
- Return proposed file changes or execution result metadata
- Return compact trace metadata (steps, subagents, tool calls)
- Return subagent summaries
- Return cost / token / duration info if available
- Return verification recommendations (not decisions)
- Return failure / retry metadata
- Return warnings about blocked or skipped actions

---

## Why DeepAgents must be constrained

DeepAgents is a capable agent framework. That capability is exactly why
OpenShard must own the safety layer. Specifically:

- **DeepAgents can use filesystem tools.** It may read or write files outside
  OpenShard's safe write path unless explicitly restricted.
- **DeepAgents can spawn subagents.** Subagents can each independently take
  actions — scope can multiply silently.
- **DeepAgents supports human-in-the-loop.** This is powerful but must be
  initiated by OpenShard's approval gate, not by the backend autonomously.
- **DeepAgents can use shell execution when sandbox backends are enabled.**
  Shell access bypasses path safety and command policy unless OpenShard
  intercepts it.

Therefore OpenShard must remain the policy, approval, safety, verification,
and receipt layer. The backend is an execution engine, not a decision engine.

---

## What a backend is forbidden from doing

| Forbidden action | Reason |
|---|---|
| Choosing final model policy | OpenShard routes. Backend does not re-route. |
| Bypassing OpenShard approval gates | All writes and high-cost operations must pass `GateEvaluator` |
| Bypassing path safety | `security/paths.py` is enforced by OpenShard, not the backend |
| Writing files directly outside OpenShard's safe write path | Backends propose. OpenShard writes. |
| Running arbitrary shell commands | Command policy is OpenShard's responsibility |
| Mutating run history directly | Run receipts are written only by `pipeline.py` |
| Deciding final verification policy | Backends suggest. OpenShard verifies. |
| Hiding tool calls from OpenShard | All tool invocations must appear in the returned trace |
| Storing raw transcripts by default | `raw_content_stored` must remain `False` |
| Uploading private repo context | Compact summaries only — no raw file contents, diffs, or paths |
| Becoming the public product mode | DeepAgents/LangGraph is not the product brain |

**DeepAgents/LangGraph is not the product brain.**
**Backend traces are evidence, not authority.**

---

## Required backend return contract

Every backend must return a `BackendExecutionResult`
(defined in `openshard/native/backends.py`):

```python
@dataclass
class BackendExecutionResult:
    backend_name: str = ""
    backend_version: str | None = None
    mode: str = ""
    steps: list[str] = field(default_factory=list)
    subagents_used: int = 0
    tools_used: list[str] = field(default_factory=list)
    proposed_files: list[str] = field(default_factory=list)
    verification_suggestion: str = ""
    cost: float | None = None
    duration: float | None = None
    tokens: int | None = None
    warnings: list[str] = field(default_factory=list)
    blocked_actions: list[str] = field(default_factory=list)
    raw_content_stored: bool = False  # must remain False
```

### Field notes

- `proposed_files` — paths only; no file contents in the result object
- `verification_suggestion` — free text; OpenShard decides whether to act on it
- `raw_content_stored` — adapters must never set this to `True`; raw transcripts
  and raw file contents are not stored by default
- `blocked_actions` — list of actions the backend attempted but did not execute
  (OpenShard may log these for audit)

---

## How a backend maps to form factors

| OpenShard form factor | Backend role |
|---|---|
| `ask` | No backend involvement (fast path) |
| `run` | No backend involvement (builtin native backend) |
| `deep-run` | DeepAgents backend could later handle the execution subgraph |
| `osn-run` | LangGraph backend could later provide durable execution / checkpointing |

Neither mapping is wired today. This document defines the contract so the
wiring can happen safely when the OSN loop receipt infrastructure is ready.

---

## Why OSN loop receipts come first

Backend integration requires that OpenShard can inspect, log, and audit what a
backend did. The OSN loop `backend_proof` phase and step receipts give
OpenShard that visibility. Without it, a backend becomes a black box —
OpenShard cannot reconstruct what happened for audit, cost accounting, or
feedback learning.

**Backend integration follows OSN loop receipts. It does not precede them.**

---

## Why this stays experimental and flag-gated

- DeepAgents integration is gated behind `--experimental-deepagents-run` and
  `experimental_deepagents_run` context flag.
- `DeepAgentsNativeBackend.run()` returns a stub result unless the flag is set.
- No execution side effects occur without explicit opt-in.
- A backend that is not flag-gated can silently change behaviour across all
  runs — that is not acceptable for an audit-trail system.

---

## When subagents add value vs. overkill

Subagents are useful when:
- The task has genuinely independent subtasks that can proceed in parallel
- Each subtask requires separate context isolation
- The task is too long for a single context window
- Human-in-the-loop checkpoints are needed mid-execution

Subagents are overkill when:
- The task is a simple read, refactor, or single-file change
- The task can be completed sequentially with a single model call
- Spawning a subagent adds latency without reducing complexity
- The overhead of merging subagent results exceeds the benefit

OpenShard's form-factor policy (`routing/form_factor_policy.py`) makes this
decision. A backend does not decide whether to use subagents — it reports
how many it used and what they did.

---

## How run history captures backend traces

Backend traces are written into `.openshard/runs.jsonl` as structured metadata
alongside the existing run receipt fields. The rules:

- Compact summaries only — no raw file contents, diffs, or transcripts
- `BackendExecutionResult` is serialized via `dataclasses.asdict()` and
  embedded in the run entry's metadata
- `raw_content_stored: false` must be present and enforced
- Tool calls must appear in `tools_used` — hidden tool calls are a contract
  violation
- Subagent count must appear in `subagents_used` — hidden subagents are a
  contract violation
- Blocked actions must appear in `blocked_actions` for audit completeness

**Raw transcripts and raw file contents are not stored by default.**

---

## Relationship to existing code

| Symbol | Location | Role |
|---|---|---|
| `NativeAgentBackend` | `openshard/native/backends.py` | Protocol all backends implement |
| `NativeBackendResult` | `openshard/native/backends.py` | Existing lightweight result (summary, notes, metadata) |
| `BackendExecutionResult` | `openshard/native/backends.py` | Full typed contract defined here |
| `DeepAgentsNativeBackend` | `openshard/native/backends.py` | Stub adapter (flag-gated) |
| `DeepAgentsAdapterMeta` | `openshard/native/backends.py` | Import-only probe result |
| `backend_proof` | `openshard/native/loop.py` | `NativeLoopPhase` where backend evidence is recorded |
| `GateEvaluator` | `openshard/execution/gates.py` | Approval gate OpenShard owns |
| `ExecutionFormFactorDecision` | `openshard/routing/form_factor_policy.py` | Form-factor OpenShard decides |
