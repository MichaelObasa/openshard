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
        tmp = Path(tempfile.mkdtemp())
        return tmp, NativeSandboxMeta(
            sandbox_enabled=True,
            sandbox_type="temp",
            fallback_reason="not a git repo",
        )

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
        )
    except Exception as exc:
        fallback_tmp = Path(tempfile.mkdtemp())
        return fallback_tmp, NativeSandboxMeta(
            sandbox_enabled=True,
            sandbox_type="temp",
            fallback_reason=str(exc)[:200],
        )
