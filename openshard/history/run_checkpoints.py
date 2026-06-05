from __future__ import annotations

import datetime
import json
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from openshard.history.jsonl_store import append_jsonl

_CHECKPOINT_PATH = Path(".openshard") / "run_checkpoints.jsonl"


@dataclass
class NativeRunCheckpointEvent:
    schema_version: int = 1
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    run_id: str = ""
    timestamp: str = field(
        default_factory=lambda: datetime.datetime.now(datetime.UTC).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        )
    )
    workflow: str = ""
    executor: str = ""
    stage: str = ""              # plan | generate | sandbox_write | verify | retry | receipt | final
    status: str = ""             # started | passed | failed | skipped
    workspace_path: str = ""
    sandbox_path: str = ""
    files: list[str] = field(default_factory=list)
    verification_status: str = ""
    retry_used: bool = False
    reason: str = ""
    raw_content_stored: bool = False

    def __post_init__(self) -> None:
        self.raw_content_stored = False


def _event_to_dict(event: NativeRunCheckpointEvent) -> dict:
    return {
        "schema_version": event.schema_version,
        "event_id": event.event_id,
        "run_id": event.run_id,
        "timestamp": event.timestamp,
        "workflow": event.workflow,
        "executor": event.executor,
        "stage": event.stage,
        "status": event.status,
        "workspace_path": event.workspace_path,
        "sandbox_path": event.sandbox_path,
        "files": list(event.files),
        "verification_status": event.verification_status,
        "retry_used": event.retry_used,
        "reason": event.reason,
        "raw_content_stored": False,
    }


def _dict_to_event(d: dict) -> NativeRunCheckpointEvent:
    return NativeRunCheckpointEvent(
        schema_version=int(d.get("schema_version", 1)),
        event_id=str(d.get("event_id", "")),
        run_id=str(d.get("run_id", "")),
        timestamp=str(d.get("timestamp", "")),
        workflow=str(d.get("workflow", "")),
        executor=str(d.get("executor", "")),
        stage=str(d.get("stage", "")),
        status=str(d.get("status", "")),
        workspace_path=str(d.get("workspace_path", "")),
        sandbox_path=str(d.get("sandbox_path", "")),
        files=list(d.get("files", [])),
        verification_status=str(d.get("verification_status", "")),
        retry_used=bool(d.get("retry_used", False)),
        reason=str(d.get("reason", "")),
        raw_content_stored=False,
    )


def log_run_checkpoint_event(event: NativeRunCheckpointEvent) -> None:
    path = Path.cwd() / _CHECKPOINT_PATH
    append_jsonl(path, _event_to_dict(event))


def load_run_checkpoint_events() -> list[NativeRunCheckpointEvent]:
    path = Path.cwd() / _CHECKPOINT_PATH
    if not path.exists():
        return []
    events: list[NativeRunCheckpointEvent] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            events.append(_dict_to_event(json.loads(line)))
        except (json.JSONDecodeError, KeyError, TypeError):
            continue
    return events


def run_checkpoints_for_run(run_id: str) -> list[NativeRunCheckpointEvent]:
    return [e for e in load_run_checkpoint_events() if e.run_id == run_id]


def recent_run_checkpoints(limit: int = 10) -> list[NativeRunCheckpointEvent]:
    events = load_run_checkpoint_events()
    return events[-limit:] if len(events) > limit else events
