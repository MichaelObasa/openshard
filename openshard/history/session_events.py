from __future__ import annotations

import json
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

_EVENTS_FILE = Path(".openshard") / "session_events.jsonl"


def _truncate(text: str, limit: int = 300) -> str:
    if len(text) <= limit:
        return text
    return text[:limit]


class SessionEventWriter:
    """Append-only writer for local session events.

    Writes to .openshard/session_events.jsonl.  All errors are swallowed so
    the TUI never crashes due to session tracing.
    """

    def __init__(self, base_path: Path | None = None) -> None:
        self._events_path = (base_path or Path.cwd()) / _EVENTS_FILE

    def write(
        self,
        event_type: str,
        session_id: str,
        *,
        run_id: str | None = None,
        shard_id: str | None = None,
        command: str | None = None,
        summary: str = "",
        metadata: dict | None = None,
    ) -> None:
        event = {
            "schema_version": 1,
            "event_id": str(uuid.uuid4()),
            "session_id": session_id,
            "run_id": run_id,
            "shard_id": shard_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "source": "tui",
            "event_type": event_type,
            "command": command,
            "summary": summary,
            "raw_text_stored": False,
            "metadata": metadata or {},
        }
        try:
            self._events_path.parent.mkdir(parents=True, exist_ok=True)
            with self._events_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(event) + "\n")
        except OSError as exc:
            print(f"[session_events] write failed: {exc}", file=sys.stderr)
