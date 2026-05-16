"""Extract and redact diffs from a native run sandbox/worktree."""
from __future__ import annotations

import subprocess
from dataclasses import dataclass, field
from pathlib import Path

_SECRET_PATTERNS = [
    "api_key",
    "apikey",
    "secret",
    "token",
    "password",
    "private_key",
    "client_secret",
]


@dataclass
class SandboxDiffResult:
    available: bool = False
    sandbox_path: str = ""
    files_changed: list[str] = field(default_factory=list)
    diff_text: str = ""
    stat_text: str = ""
    reason: str = ""
    raw_content_stored: bool = False

    def __post_init__(self) -> None:
        self.raw_content_stored = False


def redact_diff_text(text: str) -> str:
    out = []
    for line in text.splitlines():
        if any(pat in line.lower() for pat in _SECRET_PATTERNS):
            out.append("[redacted possible secret line]")
        else:
            out.append(line)
    return "\n".join(out)


def get_sandbox_diff(sandbox_path: Path, *, full: bool = False) -> SandboxDiffResult:
    try:
        stat_proc = subprocess.run(
            ["git", "diff", "--stat"],
            cwd=str(sandbox_path),
            capture_output=True,
            text=True,
            timeout=10,
        )
        if stat_proc.returncode != 0:
            return SandboxDiffResult(
                sandbox_path=str(sandbox_path),
                reason=stat_proc.stderr.strip() or "git diff --stat failed",
            )

        name_proc = subprocess.run(
            ["git", "diff", "--name-only"],
            cwd=str(sandbox_path),
            capture_output=True,
            text=True,
            timeout=10,
        )
        ls_proc = subprocess.run(
            ["git", "ls-files", "--others", "--exclude-standard"],
            cwd=str(sandbox_path),
            capture_output=True,
            text=True,
            timeout=10,
        )

        seen: set[str] = set()
        files_changed: list[str] = []
        for line in (name_proc.stdout or "").splitlines() + (ls_proc.stdout or "").splitlines():
            f = line.strip()
            if f and f not in seen:
                seen.add(f)
                files_changed.append(f)

        diff_text = ""
        if full:
            diff_proc = subprocess.run(
                ["git", "diff"],
                cwd=str(sandbox_path),
                capture_output=True,
                text=True,
                timeout=10,
            )
            if diff_proc.returncode == 0:
                diff_text = redact_diff_text(diff_proc.stdout)

        return SandboxDiffResult(
            available=True,
            sandbox_path=str(sandbox_path),
            files_changed=files_changed,
            diff_text=diff_text,
            stat_text=stat_proc.stdout.strip(),
        )
    except Exception as exc:
        return SandboxDiffResult(
            sandbox_path=str(sandbox_path),
            reason=str(exc),
        )
