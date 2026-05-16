"""Promote files from a native run sandbox/worktree into the real repo."""
from __future__ import annotations

import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from openshard.security.paths import UnsafePathError, resolve_safe_repo_path

_WALK_EXCLUDES = {".git", ".openshard", "__pycache__", ".pytest_cache"}


@dataclass
class SandboxApplyResult:
    applied: bool = False
    sandbox_path: str = ""
    files_applied: list[str] = field(default_factory=list)
    files_skipped: list[str] = field(default_factory=list)
    reason: str = ""
    raw_content_stored: bool = False

    def __post_init__(self) -> None:
        self.raw_content_stored = False


def extract_sandbox_path_from_entry(entry: dict) -> str:
    """Return the worktree path from a run entry, or '' if not applicable."""
    sandbox = entry.get("sandbox")
    if not sandbox or not isinstance(sandbox, dict):
        return ""
    if sandbox.get("sandbox_type") != "worktree":
        return ""
    worktree_path = sandbox.get("worktree_path")
    if not worktree_path:
        return ""
    if not Path(worktree_path).exists():
        return ""
    return worktree_path


def list_sandbox_changed_files(repo_root: Path, sandbox_path: Path) -> list[str]:
    """Return relative paths of files changed in the sandbox vs HEAD.

    Runs two git commands (diff + ls-files) and returns their deduped union.
    Falls back to a filesystem walk if git fails.
    """
    try:
        diff = subprocess.run(
            ["git", "diff", "--name-only"],
            cwd=str(sandbox_path),
            capture_output=True,
            text=True,
            timeout=10,
        )
        ls = subprocess.run(
            ["git", "ls-files", "--others", "--exclude-standard"],
            cwd=str(sandbox_path),
            capture_output=True,
            text=True,
            timeout=10,
        )
        if diff.returncode == 0 and ls.returncode == 0:
            seen: set[str] = set()
            result: list[str] = []
            for line in diff.stdout.splitlines() + ls.stdout.splitlines():
                f = line.strip()
                if f and f not in seen:
                    seen.add(f)
                    result.append(f)
            return result
    except Exception:
        pass

    # Fallback: walk the sandbox, skip excluded dirs
    files: list[str] = []
    for p in sandbox_path.rglob("*"):
        if not p.is_file():
            continue
        rel = p.relative_to(sandbox_path)
        if rel.parts and rel.parts[0] in _WALK_EXCLUDES:
            continue
        files.append(str(rel).replace("\\", "/"))
    return files


def apply_sandbox_changes(repo_root: Path, sandbox_path: Path) -> SandboxApplyResult:
    """Copy changed sandbox files into repo_root. No deletions in v0."""
    files = list_sandbox_changed_files(repo_root, sandbox_path)
    if not files:
        return SandboxApplyResult(
            sandbox_path=str(sandbox_path),
            reason="No sandbox changes to apply.",
        )

    result = SandboxApplyResult(sandbox_path=str(sandbox_path))
    for rel in files:
        try:
            dest = resolve_safe_repo_path(repo_root, rel)
        except UnsafePathError as exc:
            result.files_skipped.append(f"{rel} (unsafe: {exc})")
            continue

        src = sandbox_path / rel
        if not src.exists() or src.is_dir():
            result.files_skipped.append(f"{rel} (not found in sandbox)")
            continue

        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(src.read_bytes())
        result.files_applied.append(rel)

    result.applied = bool(result.files_applied)
    return result
