"""Pure, deterministic Shard-receipt completeness scoring.

This module contains no I/O. It takes already-built ``ShardReceipt`` objects and
reduces them to a completeness heuristic: per receipt, how many of an explicit
set of valuable fields were captured, and across recent receipts, which fields
are consistently present (strong) or missing (weak).

This is a receipt-quality heuristic, not analytics. Scoring is a uniform,
unweighted count of present fields — deliberately simple and easy to review.

The output never contains raw secret values or absolute paths: it emits only
static field-name strings, integer counts/percents, and the receipt ``shard_id``
(already a safe short identifier).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable

from openshard.ci.policy_check import _verification_status

if TYPE_CHECKING:
    from openshard.history.shard_contract import ShardReceipt


@dataclass(frozen=True)
class FieldDef:
    """A single scored completeness field: a name and a presence predicate."""

    name: str
    present: Callable[["ShardReceipt"], bool]


# Explicit, ordered field set. Each field is chosen so that *present* means a
# richer receipt and *absent* means genuinely missing information. Fields whose
# absence is good (errors only appear on failure; a clean run has no secret
# findings) are deliberately excluded from scoring.
FIELD_DEFINITIONS: list[FieldDef] = [
    FieldDef("task", lambda r: bool(r.task_full.strip())),
    FieldDef("timestamp", lambda r: bool(r.created_at.strip())),
    FieldDef("shard_id", lambda r: bool(r.shard_id.strip())),
    FieldDef(
        "execution_model",
        lambda r: bool(r.model_display.strip()) and r.model_display.strip() != "Not recorded",
    ),
    FieldDef(
        "status",
        lambda r: bool(r.status.strip()) and r.status.strip() != "Not recorded",
    ),
    FieldDef("duration", lambda r: r.duration_seconds is not None),
    FieldDef("cost_estimate", lambda r: r.cost_raw is not None),
    FieldDef("files_changed", lambda r: bool(r.files_detail) or r.files_changed > 0),
    FieldDef("inspected_files", lambda r: bool(r.inspected_files) or bool(r.file_evidence)),
    FieldDef(
        "verification",
        lambda r: _verification_status(r) in {"passed", "failed", "not_run"},
    ),
    FieldDef(
        "context_usage",
        lambda r: r.context_utilisation_ratio is not None
        or r.context_files_injected_count is not None,
    ),
    FieldDef("policy_decisions", lambda r: bool(r.policy_decisions)),
    FieldDef(
        "approval_state",
        lambda r: r.approval_required
        or r.approval_granted is not None
        or bool(r.approval_reason.strip()),
    ),
    FieldDef("execution_spans", lambda r: bool(r.execution_spans)),
    FieldDef("feedback", lambda r: r.developer_feedback is not None),
]

TOTAL_FIELDS: int = len(FIELD_DEFINITIONS)

# Thresholds (presence_percent) for partitioning fields in the aggregate report.
STRONG_THRESHOLD = 80
WEAK_THRESHOLD = 50

# Cap on how many recommendations are emitted, to keep output short.
MAX_RECOMMENDATIONS = 5

# Deterministic, plain-English guidance per field. Only emitted for weak fields.
RECOMMENDATIONS: dict[str, str] = {
    "task": "task text missing; ensure runs record the task.",
    "timestamp": "timestamp missing; ensure runs record a timestamp.",
    "shard_id": "shard_id missing; ensure receipts carry a shard id.",
    "execution_model": "execution model rarely recorded; capture the model used per run.",
    "status": "verification status rarely recorded; record check outcomes.",
    "duration": "duration rarely recorded; capture run duration.",
    "cost_estimate": "cost estimate rarely recorded; capture estimated cost per run.",
    "files_changed": "changed-files detail rarely recorded; capture file changes.",
    "inspected_files": "inspected-file evidence rarely recorded; capture context files read.",
    "verification": "verification rarely recorded; run and record verification.",
    "context_usage": "context usage missing; populate context utilisation metadata.",
    "policy_decisions": "policy decisions rarely recorded; capture policy gate outcomes.",
    "approval_state": "approval state rarely recorded; capture approval decisions.",
    "execution_spans": "execution spans rarely captured; record span metadata on native runs.",
    "feedback": "developer feedback rarely recorded; capture feedback outcomes.",
}


@dataclass
class ReceiptCompleteness:
    """Per-receipt completeness result."""

    shard_id: str
    score: float
    score_percent: int
    present_fields: list[str]
    missing_fields: list[str]


@dataclass
class CompletenessReport:
    """Aggregate completeness across a set of receipts."""

    runs_checked: int
    average_score_percent: int
    field_presence: dict[str, dict]
    strong_fields: list[str]
    weak_fields: list[str]
    recommendations: list[str]
    receipts: list[ReceiptCompleteness] = field(default_factory=list)


def score_receipt(receipt: "ShardReceipt") -> ReceiptCompleteness:
    """Score a single receipt against ``FIELD_DEFINITIONS``. Never raises."""
    present: list[str] = []
    missing: list[str] = []
    for fd in FIELD_DEFINITIONS:
        try:
            is_present = bool(fd.present(receipt))
        except Exception:
            is_present = False
        (present if is_present else missing).append(fd.name)

    score = len(present) / TOTAL_FIELDS if TOTAL_FIELDS else 0.0
    return ReceiptCompleteness(
        shard_id=receipt.shard_id or "",
        score=score,
        score_percent=round(score * 100),
        present_fields=present,
        missing_fields=missing,
    )


def evaluate_completeness(receipts: list["ShardReceipt"]) -> CompletenessReport:
    """Aggregate completeness over receipts. Never raises.

    Empty input yields a zeroed report with empty collections.
    """
    scored = [score_receipt(r) for r in receipts]
    runs_checked = len(scored)

    if runs_checked == 0:
        return CompletenessReport(
            runs_checked=0,
            average_score_percent=0,
            field_presence={fd.name: {"present": 0, "missing": 0, "presence_percent": 0}
                            for fd in FIELD_DEFINITIONS},
            strong_fields=[],
            weak_fields=[],
            recommendations=[],
            receipts=[],
        )

    average_score_percent = round(sum(s.score_percent for s in scored) / runs_checked)

    field_presence: dict[str, dict] = {}
    for fd in FIELD_DEFINITIONS:
        present = sum(1 for s in scored if fd.name in s.present_fields)
        missing = runs_checked - present
        field_presence[fd.name] = {
            "present": present,
            "missing": missing,
            "presence_percent": round(present / runs_checked * 100),
        }

    # Strong: high presence, sorted by presence descending then name.
    strong_fields = [
        name for name, _ in sorted(
            ((n, p) for n, p in field_presence.items()
             if p["presence_percent"] >= STRONG_THRESHOLD),
            key=lambda x: (-x[1]["presence_percent"], x[0]),
        )
    ]
    # Weak: low presence, sorted by presence ascending then name.
    weak = sorted(
        ((n, p) for n, p in field_presence.items()
         if p["presence_percent"] <= WEAK_THRESHOLD),
        key=lambda x: (x[1]["presence_percent"], x[0]),
    )
    weak_fields = [name for name, _ in weak]

    recommendations = [
        RECOMMENDATIONS[name]
        for name, _ in weak[:MAX_RECOMMENDATIONS]
        if name in RECOMMENDATIONS
    ]

    return CompletenessReport(
        runs_checked=runs_checked,
        average_score_percent=average_score_percent,
        field_presence=field_presence,
        strong_fields=strong_fields,
        weak_fields=weak_fields,
        recommendations=recommendations,
        receipts=scored,
    )
