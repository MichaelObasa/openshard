from __future__ import annotations

import json
import random
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from openshard.history.jsonl_store import append_jsonl

_MEMORY_PATH = Path(".openshard") / "memory.jsonl"


@dataclass
class MemoryEntry:
    entry_id: str
    run_id: str
    task_short: str
    outcome: str
    reason: str | None
    recorded_at: str
    schema_version: int = 1


def build_memory_entry(run_entry: dict, outcome: str, reason: str | None) -> MemoryEntry:
    now = datetime.now(UTC)
    entry_id = f"mem-{now.strftime('%Y%m%d-%H%M%S')}-{random.randint(0, 999999):06d}"
    task_short = run_entry.get("task", "")[:120].strip()
    run_id = run_entry.get("timestamp", "")
    recorded_at = now.isoformat()
    return MemoryEntry(
        entry_id=entry_id,
        run_id=run_id,
        task_short=task_short,
        outcome=outcome,
        reason=reason,
        recorded_at=recorded_at,
    )


def log_memory_entry(entry: MemoryEntry, cwd: Path | None = None) -> None:
    base = cwd if cwd is not None else Path.cwd()
    memory_path = base / _MEMORY_PATH
    append_jsonl(memory_path, _entry_to_dict(entry))


def load_memory_entries(cwd: Path | None = None) -> list[MemoryEntry]:
    base = cwd if cwd is not None else Path.cwd()
    memory_path = base / _MEMORY_PATH
    if not memory_path.exists():
        return []
    entries: list[MemoryEntry] = []
    for line in memory_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            entries.append(_dict_to_entry(json.loads(line)))
        except (json.JSONDecodeError, KeyError, TypeError):
            continue
    return entries


def _entry_to_dict(entry: MemoryEntry) -> dict:
    return {
        "schema_version": entry.schema_version,
        "entry_id": entry.entry_id,
        "run_id": entry.run_id,
        "task_short": entry.task_short,
        "outcome": entry.outcome,
        "reason": entry.reason,
        "recorded_at": entry.recorded_at,
    }


def _dict_to_entry(d: dict) -> MemoryEntry:
    return MemoryEntry(
        schema_version=d.get("schema_version", 1),
        entry_id=d.get("entry_id", ""),
        run_id=d.get("run_id", ""),
        task_short=d.get("task_short", ""),
        outcome=d.get("outcome", ""),
        reason=d.get("reason"),
        recorded_at=d.get("recorded_at", ""),
    )
