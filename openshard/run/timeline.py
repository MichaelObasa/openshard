from __future__ import annotations

import sys
from dataclasses import dataclass, field


def _stdout_supports_unicode() -> bool:
    try:
        enc = getattr(sys.stdout, "encoding", None) or "utf-8"
        "━—✖⚠✓".encode(enc)
        return True
    except (UnicodeEncodeError, LookupError):
        return False


def timeline_symbol(status: str) -> str:
    """Return the display symbol for a run event status. Checks stdout encoding each call."""
    _ok = _stdout_supports_unicode()
    return {
        "completed": "✓" if _ok else "+",
        "failed":    "✖" if _ok else "x",
        "skipped":   "-",
        "started":   "→" if _ok else ">",
    }.get(status, "✓" if _ok else "+")


def normalize_timeline(events: list) -> list[dict]:
    """Normalise raw timeline event list for rendering.

    - Drops non-dict events and events with empty labels.
    - Defaults kind to "run" and status to "completed".
    - Deduplicates receipt_saved: keeps the first occurrence only.
    """
    seen_receipt_saved = False
    result: list[dict] = []
    for ev in events:
        if not isinstance(ev, dict):
            continue
        label = (ev.get("label") or "").strip()
        if not label:
            continue
        event_key = ev.get("event", "")
        if event_key == "receipt_saved":
            if seen_receipt_saved:
                continue
            seen_receipt_saved = True
        normalized: dict = {
            "event": event_key,
            "label": label,
            "kind": ev.get("kind") or "run",
            "status": ev.get("status") or "completed",
        }
        for optional in ("detail", "target", "count", "metadata"):
            val = ev.get(optional)
            if val is not None:
                normalized[optional] = val
        result.append(normalized)
    return result


@dataclass
class RunTimelineEvent:
    """A single product-level event in the run activity feed."""

    event: str    # stable machine key, e.g. "repo_scanned"
    label: str    # user-facing display text
    kind: str     # "run"|"workflow"|"scan"|"route"|"model"|"review"|"receipt"|"check"|"tool"|"plan"|"summary"
    status: str = "completed"    # "started"|"completed"|"skipped"|"failed"
    detail: str | None = None    # muted supplementary text
    target: str | None = None    # file / command / model / workflow name
    count: int | None = None     # numeric metadata
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        d: dict = {
            "event": self.event,
            "label": self.label,
            "kind": self.kind,
            "status": self.status,
        }
        if self.detail is not None:
            d["detail"] = self.detail
        if self.target is not None:
            d["target"] = self.target
        if self.count is not None:
            d["count"] = self.count
        if self.metadata:
            d["metadata"] = self.metadata
        return d

    @classmethod
    def from_dict(cls, d: dict) -> RunTimelineEvent:
        return cls(
            event=d.get("event", ""),
            label=d.get("label", ""),
            kind=d.get("kind", "run"),
            status=d.get("status", "completed"),
            detail=d.get("detail"),
            target=d.get("target"),
            count=d.get("count"),
            metadata=d.get("metadata", {}),
        )
