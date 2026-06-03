from __future__ import annotations

import re
import sys
from dataclasses import dataclass, field

# Versioned timeline schema (additive; old receipts without it still render).
TIMELINE_SCHEMA_VERSION = "1"

# Caps for sanitised timeline strings / sizes.
_MAX_TIMELINE_EVENTS = 40
_MAX_LABEL_CHARS = 80
_MAX_DETAIL_CHARS = 120
_MAX_TARGET_CHARS = 80
_MAX_METADATA_VALUE_CHARS = 80
_MAX_METADATA_KEYS = 10

# Allowed enums. Anything else falls back to a safe default.
_ALLOWED_STATUSES = frozenset({"started", "completed", "skipped", "failed", "warning"})
_ALLOWED_KINDS = frozenset(
    {"run", "workflow", "scan", "route", "model", "review", "receipt",
     "check", "tool", "plan", "summary"}
)

# Secret-like token patterns. A match means the whole string is dropped, never
# emitted, so partial secrets cannot leak.
_SECRET_PATTERNS = (
    re.compile(r"sk-[A-Za-z0-9_\-]{8,}"),                    # OpenAI-style keys
    re.compile(r"AKIA[0-9A-Z]{8,}"),                         # AWS access key id
    re.compile(r"(?i)\b(?:api[_-]?key|token|secret|password)\s*[=:]\s*\S+"),
    re.compile(r"(?i)\bbearer\s+\S+"),                       # bearer tokens
    re.compile(r"[A-Za-z0-9_\-+/]{32,}"),                    # long opaque key-like run
)


def _is_absolute_path(p: str) -> bool:
    """Return True if p looks like an absolute local path. Never emit absolute paths."""
    if not p:
        return False
    if p.startswith("/") or p.startswith("\\"):
        return True
    # Windows drive letter: C:\ or C:/
    if len(p) >= 3 and p[1] == ":" and p[2] in ("/", "\\"):
        return True
    return False


def _looks_like_secret(s: str) -> bool:
    return any(pat.search(s) for pat in _SECRET_PATTERNS)


def _sanitize_text(s, limit: int) -> "str | None":
    """Sanitise an untrusted timeline string for safe export.

    - Coerces to str; returns None for non-str/empty input.
    - Strips CR/LF and other control characters.
    - Drops the value entirely (returns None) if it looks like an absolute local
      path or contains a secret-like token, so partial values never leak.
    - Caps length to ``limit``.
    """
    if not isinstance(s, str):
        return None
    # Remove control chars (incl. CR/LF/tab) before any pattern checks.
    cleaned = "".join(ch for ch in s if ch == " " or ch.isprintable()).strip()
    if not cleaned:
        return None
    if _is_absolute_path(cleaned) or ".codegraph" in cleaned:
        return None
    if _looks_like_secret(cleaned):
        return None
    return cleaned[:limit]


def _sanitize_metadata(metadata) -> dict:
    """Keep only small, safe scalar metadata values.

    Drops nested dicts/lists/blobs. String values pass through ``_sanitize_text``
    (path + secret redaction, capped); values that drop to None are omitted.
    """
    if not isinstance(metadata, dict):
        return {}
    safe: dict = {}
    for key, val in metadata.items():
        if len(safe) >= _MAX_METADATA_KEYS:
            break
        if not isinstance(key, str):
            continue
        if isinstance(val, bool) or isinstance(val, (int, float)):
            safe[key] = val
        elif isinstance(val, str):
            clean = _sanitize_text(val, _MAX_METADATA_VALUE_CHARS)
            if clean is not None:
                safe[key] = clean
        # else: drop non-scalar (dict/list/None/blob)
    return safe


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


def make_timeline_event(
    event: str,
    label: str,
    *,
    kind: str = "run",
    status: str = "completed",
    detail: "str | None" = None,
    target: "str | None" = None,
    count: "int | None" = None,
    metadata: "dict | None" = None,
) -> RunTimelineEvent:
    """Construct a sanitised RunTimelineEvent.

    Single chokepoint for timeline construction: validates enums, redacts paths
    and secret-like tokens from all strings, caps lengths, and keeps metadata to
    small safe scalars only. Labels for normal safe events pass through unchanged.
    """
    _event = event if isinstance(event, str) else ""
    _kind = kind if kind in _ALLOWED_KINDS else "run"
    _status = status if status in _ALLOWED_STATUSES else "completed"

    _label = _sanitize_text(label, _MAX_LABEL_CHARS)
    if not _label:
        # label is required; fall back to a safe generic derived from the event
        # key (never the raw, possibly-unsafe value).
        _label = (_event.replace("_", " ").strip() or "event")[:_MAX_LABEL_CHARS]

    _detail = _sanitize_text(detail, _MAX_DETAIL_CHARS) if detail is not None else None
    _target = _sanitize_text(target, _MAX_TARGET_CHARS) if target is not None else None

    _count: "int | None" = None
    if isinstance(count, bool):
        _count = None
    elif isinstance(count, int) and count >= 0:
        _count = count

    return RunTimelineEvent(
        event=_event,
        label=_label,
        kind=_kind,
        status=_status,
        detail=_detail,
        target=_target,
        count=_count,
        metadata=_sanitize_metadata(metadata),
    )


def project_timeline_for_export(events: list) -> list[dict]:
    """Project stored timeline events to a safe, stable shape for machine output.

    Normalises, caps the event count, re-runs every value through the sanitiser
    (defence in depth at the export boundary), and emits a stable dict per event:
    ``{event, label, kind, status}`` plus ``detail``/``target``/``count`` only
    when present and safe. Pure, no I/O.
    """
    normalized = normalize_timeline(list(events) if events else [])
    result: list[dict] = []
    for ev in normalized[:_MAX_TIMELINE_EVENTS]:
        safe = make_timeline_event(
            ev.get("event", ""),
            ev.get("label", ""),
            kind=ev.get("kind", "run"),
            status=ev.get("status", "completed"),
            detail=ev.get("detail"),
            target=ev.get("target"),
            count=ev.get("count"),
            metadata=ev.get("metadata"),
        )
        row: dict = {
            "event": safe.event,
            "label": safe.label,
            "kind": safe.kind,
            "status": safe.status,
        }
        if safe.detail is not None:
            row["detail"] = safe.detail
        if safe.target is not None:
            row["target"] = safe.target
        if safe.count is not None:
            row["count"] = safe.count
        result.append(row)
    return result
