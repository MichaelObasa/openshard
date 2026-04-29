from __future__ import annotations

import unicodedata
from pathlib import Path


class UnsafePathError(Exception):
    """Raised when a model-provided path fails safety validation."""


def resolve_safe_repo_path(repo_root: Path, user_path: str) -> Path:
    """
    Resolve *user_path* relative to *repo_root* and validate it is safe to write.

    Raises UnsafePathError for any rejected input.
    Returns the resolved absolute Path on success.
    """
    # Reject empty
    if not user_path or not user_path.strip():
        raise UnsafePathError("Path must not be empty.")

    # Reject control characters (ASCII < 32, DEL 0x7f, Unicode category Cc)
    for ch in user_path:
        if unicodedata.category(ch) == "Cc" or ord(ch) < 32 or ord(ch) == 127:
            raise UnsafePathError(f"Path contains control character: {ch!r}")

    # Reject home-expansion patterns
    if user_path.lstrip().startswith("~"):
        raise UnsafePathError("Path must not start with '~' (home expansion rejected).")

    p = Path(user_path)

    # Reject absolute paths
    if p.is_absolute():
        raise UnsafePathError(f"Path must be relative, got absolute: {user_path!r}")

    # Reject '..' components anywhere in the path
    if ".." in p.parts:
        raise UnsafePathError(f"Path must not contain '..': {user_path!r}")

    # Reject symlinks before resolution (is_symlink() does not follow links)
    candidate = repo_root / p
    if candidate.is_symlink():
        raise UnsafePathError(f"Path is a symlink: {candidate}")

    # Resolve against repo root and confirm containment
    resolved = candidate.resolve()
    repo_resolved = repo_root.resolve()
    try:
        rel = resolved.relative_to(repo_resolved)
    except ValueError:
        raise UnsafePathError(
            f"Resolved path escapes repo root: {resolved} not under {repo_resolved}"
        )

    # Reject paths under .git/
    if rel.parts and rel.parts[0] == ".git":
        raise UnsafePathError(f"Path must not target .git/: {user_path!r}")

    # Reject .openshard/runs.jsonl specifically
    if rel == Path(".openshard") / "runs.jsonl":
        raise UnsafePathError("Path must not target .openshard/runs.jsonl.")

    return resolved
