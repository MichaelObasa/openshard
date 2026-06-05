from __future__ import annotations

import datetime
import json
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from openshard.history.jsonl_store import append_jsonl

_FAILURE_MEMORY_PATH = Path(".openshard") / "failure_memory.jsonl"


@dataclass
class NativeFailureMemoryEvent:
    schema_version: int = 1
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    run_id: str = ""
    timestamp: str = field(
        default_factory=lambda: datetime.datetime.now(
            datetime.timezone.utc
        ).strftime("%Y-%m-%dT%H:%M:%SZ")
    )
    task_summary: str = ""
    failure_type: str = ""
    exit_code: int = 0
    output_chars: int = 0
    verification_status: str = "failed"
    retry_attempted: bool = False
    retry_succeeded: bool = False
    retry_patch_files: list[str] = field(default_factory=list)
    related_file_paths: list[str] = field(default_factory=list)
    model: str = ""
    workflow: str = ""
    raw_content_stored: bool = False

    def __post_init__(self) -> None:
        self.raw_content_stored = False


def parse_failure_summary(s: str) -> dict[str, str]:
    """Parse 'exit_code=1 failure_type=test_failure output_chars=842 ...' into a dict."""
    result: dict[str, str] = {}
    for token in s.split():
        if "=" in token:
            k, _, v = token.partition("=")
            result[k] = v
    return result


def _event_to_dict(event: NativeFailureMemoryEvent) -> dict:
    return {
        "schema_version": event.schema_version,
        "event_id": event.event_id,
        "run_id": event.run_id,
        "timestamp": event.timestamp,
        "task_summary": event.task_summary,
        "failure_type": event.failure_type,
        "exit_code": event.exit_code,
        "output_chars": event.output_chars,
        "verification_status": event.verification_status,
        "retry_attempted": event.retry_attempted,
        "retry_succeeded": event.retry_succeeded,
        "retry_patch_files": list(event.retry_patch_files),
        "related_file_paths": list(event.related_file_paths),
        "model": event.model,
        "workflow": event.workflow,
        "raw_content_stored": False,
    }


def _dict_to_event(d: dict) -> NativeFailureMemoryEvent:
    return NativeFailureMemoryEvent(
        schema_version=d.get("schema_version", 1),
        event_id=d.get("event_id", ""),
        run_id=d.get("run_id", ""),
        timestamp=d.get("timestamp", ""),
        task_summary=d.get("task_summary", ""),
        failure_type=d.get("failure_type", ""),
        exit_code=d.get("exit_code", 0),
        output_chars=d.get("output_chars", 0),
        verification_status=d.get("verification_status", "failed"),
        retry_attempted=bool(d.get("retry_attempted", False)),
        retry_succeeded=bool(d.get("retry_succeeded", False)),
        retry_patch_files=list(d.get("retry_patch_files") or []),
        related_file_paths=list(d.get("related_file_paths") or []),
        model=d.get("model", ""),
        workflow=d.get("workflow", ""),
        raw_content_stored=False,
    )


def log_failure_memory_event(event: NativeFailureMemoryEvent) -> None:
    path = Path.cwd() / _FAILURE_MEMORY_PATH
    d = _event_to_dict(event)
    d["raw_content_stored"] = False
    append_jsonl(path, d)


def load_failure_memory_events() -> list[NativeFailureMemoryEvent]:
    path = Path.cwd() / _FAILURE_MEMORY_PATH
    if not path.exists():
        return []
    events: list[NativeFailureMemoryEvent] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            events.append(_dict_to_event(json.loads(line)))
        except (json.JSONDecodeError, KeyError, TypeError):
            continue
    return events


def failure_memory_events_for_run(run_id: str) -> list[NativeFailureMemoryEvent]:
    return [e for e in load_failure_memory_events() if e.run_id == run_id]


def recent_failure_memory(limit: int = 10) -> list[NativeFailureMemoryEvent]:
    events = load_failure_memory_events()
    return events[-limit:] if len(events) > limit else events
