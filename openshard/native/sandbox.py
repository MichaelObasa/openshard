"""Git worktree sandbox for isolated native write runs."""
from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

from openshard.native.context import NativeSandboxMeta


def _detect_git_root(cwd: Path) -> Path | None:
    """Return git repo root via `git rev-parse --show-toplevel`, or None if not in a repo."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=str(cwd),
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0 and result.stdout.strip():
            return Path(result.stdout.strip())
    except Exception:
        pass
    return None


def _detect_git_state(cwd: Path) -> tuple[str | None, str | None]:
    """
    Return (branch, full_commit_hash) from cwd.

    Returns (None, None) on any failure.
    Returns (None, hash) when HEAD is detached (git outputs "HEAD" for --abbrev-ref).
    Never raises.
    """
    branch: str | None = None
    commit: str | None = None
    try:
        b = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=str(cwd),
            capture_output=True,
            text=True,
            timeout=10,
        )
        if b.returncode == 0 and b.stdout.strip() and b.stdout.strip() != "HEAD":
            branch = b.stdout.strip()
    except Exception:
        pass
    try:
        c = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(cwd),
            capture_output=True,
            text=True,
            timeout=10,
        )
        if c.returncode == 0 and c.stdout.strip():
            commit = c.stdout.strip()
    except Exception:
        pass
    return branch, commit


def _safe_branch_name(run_id: str) -> str:
    safe = run_id.replace(":", "-").replace(".", "-").replace(" ", "-")
    return f"osn/run-{safe[:40]}"


def create_run_sandbox(repo_root: Path, run_id: str) -> tuple[Path, NativeSandboxMeta]:
    """
    Return (workspace_path, meta).

    Tries to create a git worktree when inside a git repo.
    Falls back to a plain temp dir if not a git repo or if worktree creation fails.
    Never raises — all exceptions are captured into fallback_reason.
    """
    git_root = _detect_git_root(repo_root)
    if git_root is None:
        _base_branch, _base_commit = _detect_git_state(repo_root)
        tmp = Path(tempfile.mkdtemp())
        return tmp, NativeSandboxMeta(
            sandbox_enabled=True,
            sandbox_type="temp",
            fallback_reason="not a git repo",
            git_base_branch=_base_branch,
            git_base_commit_hash=_base_commit,
            safe_workspace_display_name="osn-temp",
        )

    _base_branch, _base_commit = _detect_git_state(git_root)
    branch = _safe_branch_name(run_id)
    tmp = Path(tempfile.mkdtemp())
    worktree_path = tmp / "wt"
    try:
        result = subprocess.run(
            ["git", "worktree", "add", "-b", branch, str(worktree_path)],
            cwd=str(git_root),
            capture_output=True,
            text=True,
            timeout=15,
        )
        if result.returncode != 0:
            raise RuntimeError(result.stderr.strip() or "git worktree add failed")
        return worktree_path, NativeSandboxMeta(
            sandbox_enabled=True,
            sandbox_type="worktree",
            worktree_path=str(worktree_path),
            worktree_branch=branch,
            git_base_branch=_base_branch,
            git_base_commit_hash=_base_commit,
            safe_workspace_display_name=branch,
        )
    except Exception as exc:
        fallback_tmp = Path(tempfile.mkdtemp())
        return fallback_tmp, NativeSandboxMeta(
            sandbox_enabled=True,
            sandbox_type="temp",
            fallback_reason=str(exc)[:200],
            git_base_branch=_base_branch,
            git_base_commit_hash=_base_commit,
            safe_workspace_display_name="osn-temp",
        )
