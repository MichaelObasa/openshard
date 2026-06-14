from __future__ import annotations

import dataclasses
import json
from pathlib import Path

from openshard.native.agent_loop_types import AgentLoopEvent, ReceiptIteration


def _serialise(obj: object) -> object:
    """Recursively convert dataclasses and known primitives to JSON-safe types.

    Handles nested dataclasses, lists, dicts, and None.
    raw_content_stored is forced to False at the field level — not left to
    the caller — so the invariant is enforced at write time, not just declared.
    """
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        result = {}
        for f in dataclasses.fields(obj):
            value = getattr(obj, f.name)
            if f.name == "raw_content_stored":
                result[f.name] = False
            else:
                result[f.name] = _serialise(value)
        return result
    if isinstance(obj, list):
        return [_serialise(item) for item in obj]
    if isinstance(obj, dict):
        return {k: _serialise(v) for k, v in obj.items()}
    return obj


class ReceiptEmitter:
    """Appends serialised ReceiptIteration records to a JSONL file.

    Each emit() call is one atomic append: open → write → close.
    This avoids holding file descriptors across long-running loops and
    ensures each record lands on disk even if the process is interrupted.

    raw_content_stored is forced to False by _serialise at write time,
    not just by convention. The file is created if it does not exist.
    """

    def __init__(self, path: Path) -> None:
        self._path = Path(path)

    def emit(self, receipt: ReceiptIteration) -> None:
        record = _serialise(receipt)
        with self._path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, default=str) + "\n")

    def emit_event(self, event: AgentLoopEvent) -> None:
        record = _serialise(event)
        with self._path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, default=str) + "\n")

    def read_all(self) -> list[dict]:
        """Return all records written so far, in order."""
        if not self._path.exists():
            return []
        records = []
        with self._path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return records

    def record_count(self) -> int:
        return len(self.read_all())
