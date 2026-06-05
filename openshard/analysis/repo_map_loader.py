"""Repo-map load-or-build orchestration (v1).

Keeps the cache *decision* in one place so every consumer (`repo map`,
`repo plan`) shares the same behaviour: a clean git tree reuses the cache, a
dirty tree always rebuilds, ``--refresh`` forces a rebuild. Construction stays
pure in ``repo_map.py`` and IO stays in ``repo_map_cache.py`` - this module only
wires them together, so neither of those gains a side-effect or an import cycle.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from openshard.analysis.repo_map import (
    _DIRTY_WARNING,
    MAX_WARNINGS,
    RepoMap,
    build_repo_map,
    compute_repo_fingerprint,
)
from openshard.analysis.repo_map_cache import (
    cache_path_for,
    load_repo_map_cache,
    save_repo_map_cache,
)


@dataclass
class LoadedRepoMap:
    repo_map: dict  # RepoMap.to_dict(), with the final warnings list applied
    cache_hit: bool
    cache_path_display: str  # relative forward-slash path - never absolute
    warnings: list[str]


def load_or_build_repo_map(root: Path, *, refresh: bool = False) -> LoadedRepoMap:
    """Load the repo map from cache, or build and cache it. Never raises.

    The only side effect is writing ``.openshard/cache/repo-<fingerprint>.json``
    on a cache miss / dirty tree / ``--refresh`` - identical to what
    ``openshard repo map`` already does.
    """
    fingerprint, git, _ = compute_repo_fingerprint(root)
    cache_abs, cache_display = cache_path_for(fingerprint, base=root)
    dirty = git.dirty

    cached = None
    if not refresh and not dirty:
        cached = load_repo_map_cache(cache_abs)

    if cached is not None:
        repo_map_obj = RepoMap.from_dict(cached)
        cache_hit = True
    else:
        repo_map_obj = build_repo_map(root)
        save_repo_map_cache(cache_abs, repo_map_obj.to_dict())
        cache_hit = False

    repo_map_dict = repo_map_obj.to_dict()

    # The dirty-rebuild reason is a cache-decision warning, surfaced alongside
    # the construction warnings the map already carries.
    warnings = list(repo_map_dict.get("warnings", []))
    if dirty and _DIRTY_WARNING not in warnings:
        warnings.append(_DIRTY_WARNING)
    warnings = warnings[:MAX_WARNINGS]
    repo_map_dict["warnings"] = warnings

    return LoadedRepoMap(
        repo_map=repo_map_dict,
        cache_hit=cache_hit,
        cache_path_display=cache_display,
        warnings=warnings,
    )
