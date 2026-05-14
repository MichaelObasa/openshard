from __future__ import annotations

import datetime
import json
import uuid
from dataclasses import dataclass, field
from pathlib import Path

_INTERACTIONS_PATH = Path(".openshard") / "interactions.jsonl"


@dataclass
class DeveloperInteractionEvent:
    schema_version: int = 1
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    run_id: str = ""
    timestamp: str = field(
        default_factory=lambda: datetime.datetime.now(
            datetime.timezone.utc
        ).strftime("%Y-%m-%dT%H:%M:%SZ")
    )
    actor: str = "developer"
    event_type: str = ""
    summary: str = ""
    related_stage: str = ""
    related_file_paths: list[str] = field(default_factory=list)
    correction_reason: str | None = None
    severity: str = "info"
    accepted: bool | None = None
    metadata: dict = field(default_factory=dict)
    raw_content_stored: bool = False

    def __post_init__(self) -> None:
        self.raw_content_stored = False


def _event_to_dict(event: DeveloperInteractionEvent) -> dict:
    return {
        "schema_version": event.schema_version,
        "event_id": event.event_id,
        "run_id": event.run_id,
        "timestamp": event.timestamp,
        "actor": event.actor,
        "event_type": event.event_type,
        "summary": event.summary,
        "related_stage": event.related_stage,
        "related_file_paths": event.related_file_paths,
        "correction_reason": event.correction_reason,
        "severity": event.severity,
        "accepted": event.accepted,
        "metadata": event.metadata,
        "raw_content_stored": False,
    }


def _dict_to_event(d: dict) -> DeveloperInteractionEvent:
    return DeveloperInteractionEvent(
        schema_version=d.get("schema_version", 1),
        event_id=d.get("event_id", ""),
        run_id=d.get("run_id", ""),
        timestamp=d.get("timestamp", ""),
        actor=d.get("actor", "developer"),
        event_type=d.get("event_type", ""),
        summary=d.get("summary", ""),
        related_stage=d.get("related_stage", ""),
        related_file_paths=d.get("related_file_paths") or [],
        correction_reason=d.get("correction_reason"),
        severity=d.get("severity", "info"),
        accepted=d.get("accepted"),
        metadata=d.get("metadata") or {},
        raw_content_stored=False,
    )


def log_interaction_event(event: DeveloperInteractionEvent) -> None:
    interactions_path = Path.cwd() / _INTERACTIONS_PATH
    interactions_path.parent.mkdir(parents=True, exist_ok=True)
    d = _event_to_dict(event)
    d["raw_content_stored"] = False
    with interactions_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(d) + "\n")


def load_interaction_events() -> list[DeveloperInteractionEvent]:
    interactions_path = Path.cwd() / _INTERACTIONS_PATH
    if not interactions_path.exists():
        return []
    events: list[DeveloperInteractionEvent] = []
    for line in interactions_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            events.append(_dict_to_event(json.loads(line)))
        except (json.JSONDecodeError, KeyError, TypeError):
            continue
    return events


def interaction_events_for_run(run_id: str) -> list[DeveloperInteractionEvent]:
    return [e for e in load_interaction_events() if e.run_id == run_id]
