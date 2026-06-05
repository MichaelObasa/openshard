from __future__ import annotations

import datetime
import json
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from openshard.history.jsonl_store import append_jsonl

_NATIVE_STEPS_PATH = Path(".openshard") / "native_steps.jsonl"


@dataclass
class NativeStepEvent:
    schema_version: int = 1
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    run_id: str = ""
    timestamp: str = field(
        default_factory=lambda: datetime.datetime.now(
            datetime.UTC
        ).strftime("%Y-%m-%dT%H:%M:%SZ")
    )
    step_index: int = 0
    step_name: str = ""
    stage: str = ""
    status: str = ""
    summary: str = ""
    tool_name: str = ""
    policy_decision: str = ""
    approval_required: bool = False
    approval_granted: bool | None = None
    verification_status: str = ""
    retry_count: int = 0
    duration_ms: int | None = None
    metadata: dict = field(default_factory=dict)
    raw_content_stored: bool = False

    def __post_init__(self) -> None:
        self.raw_content_stored = False


def _event_to_dict(event: NativeStepEvent) -> dict:
    return {
        "schema_version": event.schema_version,
        "event_id": event.event_id,
        "run_id": event.run_id,
        "timestamp": event.timestamp,
        "step_index": event.step_index,
        "step_name": event.step_name,
        "stage": event.stage,
        "status": event.status,
        "summary": event.summary,
        "tool_name": event.tool_name,
        "policy_decision": event.policy_decision,
        "approval_required": event.approval_required,
        "approval_granted": event.approval_granted,
        "verification_status": event.verification_status,
        "retry_count": event.retry_count,
        "duration_ms": event.duration_ms,
        "metadata": event.metadata,
        "raw_content_stored": False,
    }


def _dict_to_event(d: dict) -> NativeStepEvent:
    return NativeStepEvent(
        schema_version=d.get("schema_version", 1),
        event_id=d.get("event_id", ""),
        run_id=d.get("run_id", ""),
        timestamp=d.get("timestamp", ""),
        step_index=d.get("step_index", 0),
        step_name=d.get("step_name", ""),
        stage=d.get("stage", ""),
        status=d.get("status", ""),
        summary=d.get("summary", ""),
        tool_name=d.get("tool_name", ""),
        policy_decision=d.get("policy_decision", ""),
        approval_required=bool(d.get("approval_required", False)),
        approval_granted=d.get("approval_granted"),
        verification_status=d.get("verification_status", ""),
        retry_count=d.get("retry_count", 0),
        duration_ms=d.get("duration_ms"),
        metadata=d.get("metadata") or {},
        raw_content_stored=False,
    )


def log_native_step_event(event: NativeStepEvent) -> None:
    steps_path = Path.cwd() / _NATIVE_STEPS_PATH
    d = _event_to_dict(event)
    d["raw_content_stored"] = False
    append_jsonl(steps_path, d)


def load_native_step_events() -> list[NativeStepEvent]:
    steps_path = Path.cwd() / _NATIVE_STEPS_PATH
    if not steps_path.exists():
        return []
    events: list[NativeStepEvent] = []
    for line in steps_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            events.append(_dict_to_event(json.loads(line)))
        except (json.JSONDecodeError, KeyError, TypeError):
            continue
    return events


def native_step_events_for_run(run_id: str) -> list[NativeStepEvent]:
    return [e for e in load_native_step_events() if e.run_id == run_id]
