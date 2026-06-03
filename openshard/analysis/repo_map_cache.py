"""Local repo-map cache IO (v1).

Side-effecting load/save only - construction stays pure in ``analysis/repo_map.py``.
Cache files live under ``.openshard/cache/repo-<fingerprint>.json`` (the ``.openshard``
dir is already in every scanner's skip set, so the cache never self-scans).

Mirrors the conventions of ``providers/cache.py``: ``mkdir(parents=True)`` then
``json.dumps(indent=2)``; tolerant load that returns None on missing/corrupt files.
"""
from __future__ import annotations

import json
from pathlib import Path

_CACHE_SUBDIR = Path(".openshard") / "cache"


def cache_path_for(fingerprint: str, *, base: Path | None = None) -> tuple[Path, str]:
    """Return (absolute_path, relative_display) for a fingerprint's cache file.

    The display string is always a relative forward-slash path - never absolute.
    """
    root = base if base is not None else Path.cwd()
    rel = _CACHE_SUBDIR / f"repo-{fingerprint}.json"
    abs_path = root / rel
    display = rel.as_posix()
    return abs_path, display


def load_repo_map_cache(path: Path) -> dict | None:
    """Return the parsed cache dict, or None if missing or unparseable."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None
    return data if isinstance(data, dict) else None


def save_repo_map_cache(path: Path, data: dict) -> None:
    """Write *data* as pretty JSON, creating the cache directory if needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
