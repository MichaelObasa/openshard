"""Shared sanitization primitives for safe shard and timeline data export.

These helpers are intentionally in a neutral layer so both ``openshard.run``
and ``openshard.history`` modules can import without creating cross-layer
dependencies.

All public functions are pure (no I/O, no side effects) and never raise.
"""

from __future__ import annotations

import re

# Caps for sanitised metadata values / key counts.
_MAX_METADATA_VALUE_CHARS = 80
_MAX_METADATA_KEYS = 10

# Secret-like token patterns.  A match means the whole string is dropped —
# never emitted — so partial secrets cannot leak.
SECRET_PATTERNS = (
    re.compile(r"sk-[A-Za-z0-9_\-]{8,}"),                    # OpenAI-style keys
    re.compile(r"AKIA[0-9A-Z]{8,}"),                          # AWS access key id
    re.compile(r"(?i)\b(?:api[_-]?key|token|secret|password)\s*[=:]\s*\S+"),
    re.compile(r"(?i)\bbearer\s+\S+"),                        # bearer tokens
    re.compile(r"[A-Za-z0-9_\-+/]{32,}"),                    # long opaque key-like run
)


def is_absolute_path(p: str) -> bool:
    """Return True if *p* looks like an absolute local path."""
    if not p:
        return False
    if p.startswith("/") or p.startswith("\\"):
        return True
    if len(p) >= 3 and p[1] == ":" and p[2] in ("/", "\\"):
        return True
    return False


def looks_like_secret(s: str) -> bool:
    """Return True if *s* matches any known secret-like pattern."""
    return any(pat.search(s) for pat in SECRET_PATTERNS)


def sanitize_text(s: object, limit: int) -> str | None:
    """Sanitise an untrusted string for safe export.

    - Coerces to str; returns None for non-str/empty input.
    - Strips CR/LF and other control characters.
    - Drops the value entirely (returns None) if it looks like an absolute
      local path or contains a secret-like token.
    - Caps length to *limit*.
    """
    if not isinstance(s, str):
        return None
    cleaned = "".join(ch for ch in s if ch == " " or ch.isprintable()).strip()
    if not cleaned:
        return None
    if is_absolute_path(cleaned) or ".codegraph" in cleaned:
        return None
    if looks_like_secret(cleaned):
        return None
    return cleaned[:limit]


def sanitize_metadata(metadata: object) -> dict:
    """Keep only small, safe scalar metadata values.

    Drops nested dicts/lists/blobs.  String values pass through
    ``sanitize_text`` (path + secret redaction, capped); values that drop to
    None are omitted.
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
            clean = sanitize_text(val, _MAX_METADATA_VALUE_CHARS)
            if clean is not None:
                safe[key] = clean
        # else: drop non-scalar (dict/list/None/blob)
    return safe
