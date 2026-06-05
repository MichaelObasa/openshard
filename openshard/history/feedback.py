from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from openshard.history.jsonl_store import append_jsonl

ALLOWED_OUTCOMES = ["accepted", "rejected", "partial", "useful", "wrong", "needs-retry"]
_FEEDBACK_PATH = ".openshard/feedback.jsonl"
SCHEMA_VERSION = 1


@dataclass
class FeedbackRecord:
    feedback_id: str
    shard_id: str
    run_timestamp: str
    task_short: str
    outcome: str
    note: str
    created_at: str
    source: str = "cli"
    schema_version: int = SCHEMA_VERSION


def build_feedback_record(entry: dict, run_index: int, outcome: str, note: str) -> FeedbackRecord:
    from openshard.history.shard_contract import _make_shard_id

    timestamp = entry.get("timestamp", "")
    task = entry.get("task", "") or ""
    task_short = task[:70]
    now = datetime.now(UTC)
    feedback_id = now.strftime("fb-%Y%m%d-%H%M%S-") + f"{now.microsecond:06d}"
    shard_id = _make_shard_id(timestamp, run_index)
    created_at = now.strftime("%Y-%m-%dT%H:%M:%SZ")
    return FeedbackRecord(
        feedback_id=feedback_id,
        shard_id=shard_id,
        run_timestamp=timestamp,
        task_short=task_short,
        outcome=outcome,
        note=note,
        created_at=created_at,
    )


def _record_to_dict(r: FeedbackRecord) -> dict:
    return {
        "feedback_id": r.feedback_id,
        "shard_id": r.shard_id,
        "run_timestamp": r.run_timestamp,
        "task_short": r.task_short,
        "outcome": r.outcome,
        "note": r.note,
        "created_at": r.created_at,
        "source": r.source,
        "schema_version": r.schema_version,
    }


def _dict_to_record(d: dict) -> FeedbackRecord:
    return FeedbackRecord(
        feedback_id=d["feedback_id"],
        shard_id=d["shard_id"],
        run_timestamp=d.get("run_timestamp", ""),
        task_short=d.get("task_short", ""),
        outcome=d["outcome"],
        note=d.get("note", ""),
        created_at=d.get("created_at", ""),
        source=d.get("source", "cli"),
        schema_version=d.get("schema_version", SCHEMA_VERSION),
    )


def log_feedback_record(record: FeedbackRecord, cwd: Path | None = None) -> None:
    base = cwd if cwd is not None else Path.cwd()
    path = base / _FEEDBACK_PATH
    append_jsonl(path, _record_to_dict(record))


def load_feedback_records(cwd: Path | None = None) -> list[FeedbackRecord]:
    base = cwd if cwd is not None else Path.cwd()
    path = base / _FEEDBACK_PATH
    if not path.exists():
        return []
    records: list[FeedbackRecord] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            records.append(_dict_to_record(json.loads(line)))
        except (json.JSONDecodeError, KeyError, TypeError):
            continue
    return records


def load_feedback_for_shard(shard_id: str, cwd: Path | None = None) -> list[FeedbackRecord]:
    return [r for r in load_feedback_records(cwd) if r.shard_id == shard_id]
