from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path


@dataclass
class OpenCodeAvailability:
    available: bool
    path: str | None
    reason: str | None
    install_guidance: list[str]


@dataclass
class OpenCodeRunResult:
    exit_code: int
    stdout_summary: str
    stderr_summary: str
    duration_ms: int
    command: list[str]
    executor: str = "OpenCode"


_INSTALL_GUIDANCE_WIN32: list[str] = [
    "npm i -g opencode-ai@latest",
    "scoop install opencode",
    "choco install opencode",
]

_INSTALL_GUIDANCE_DARWIN: list[str] = [
    "brew install anomalyco/tap/opencode",
    "brew install opencode",
    "npm i -g opencode-ai@latest",
    "curl -fsSL https://opencode.ai/install | bash",
    "mise use -g opencode",
]

_INSTALL_GUIDANCE_LINUX: list[str] = [
    "curl -fsSL https://opencode.ai/install | bash",
    "npm i -g opencode-ai@latest",
    "mise use -g opencode",
]

_INSTALL_GUIDANCE_ALL: list[str] = [
    "npm i -g opencode-ai@latest",
    "scoop install opencode",
    "choco install opencode",
    "brew install anomalyco/tap/opencode",
    "curl -fsSL https://opencode.ai/install | bash",
    "mise use -g opencode",
]


def detect_opencode() -> OpenCodeAvailability:
    """Detect whether opencode is available. Never raises."""
    found = shutil.which("opencode")
    if found:
        return OpenCodeAvailability(available=True, path=found, reason=None, install_guidance=[])

    if sys.platform == "win32":
        appdata = os.environ.get("APPDATA", "")
        if appdata:
            for name in ("opencode.cmd", "opencode.ps1", "opencode"):
                candidate = Path(appdata) / "npm" / name
                if candidate.is_file():
                    return OpenCodeAvailability(
                        available=True, path=str(candidate), reason=None, install_guidance=[]
                    )

    guidance = get_opencode_install_guidance()
    return OpenCodeAvailability(
        available=False,
        path=None,
        reason="opencode not found on PATH",
        install_guidance=guidance,
    )


def get_opencode_install_guidance(platform: str | None = None) -> list[str]:
    """Return platform-appropriate install options for OpenCode."""
    plat = platform if platform is not None else sys.platform
    if plat == "win32":
        return list(_INSTALL_GUIDANCE_WIN32)
    if plat == "darwin":
        return list(_INSTALL_GUIDANCE_DARWIN)
    if plat.startswith("linux"):
        return list(_INSTALL_GUIDANCE_LINUX)
    return list(_INSTALL_GUIDANCE_ALL)


def build_opencode_command(task: str, repo_path: Path) -> list[str]:  # noqa: ARG001
    """Return the opencode argv list for *task*. Always a list, never a shell string."""
    return ["opencode", "run", task]


def run_opencode_task(
    task: str,
    repo_path: Path,
    timeout_s: int = 300,
) -> OpenCodeRunResult:
    """Run opencode for *task* in *repo_path*. Never raises.

    Returns exit_code=127 when opencode is not available.
    Returns exit_code=-1 on timeout.
    stdout/stderr summaries are truncated to 1000 characters each.
    """
    avail = detect_opencode()
    if not avail.available:
        return OpenCodeRunResult(
            exit_code=127,
            stdout_summary="",
            stderr_summary=avail.reason or "opencode not found",
            duration_ms=0,
            command=["opencode"],
        )

    cmd = build_opencode_command(task, repo_path)
    start = time.monotonic()
    try:
        proc = subprocess.run(
            cmd,
            cwd=repo_path,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout_s,
        )
        duration_ms = int((time.monotonic() - start) * 1000)
        return OpenCodeRunResult(
            exit_code=proc.returncode,
            stdout_summary=(proc.stdout or "")[:1000],
            stderr_summary=(proc.stderr or "")[:1000],
            duration_ms=duration_ms,
            command=cmd,
        )
    except subprocess.TimeoutExpired:
        duration_ms = int((time.monotonic() - start) * 1000)
        return OpenCodeRunResult(
            exit_code=-1,
            stdout_summary="",
            stderr_summary=f"Timed out after {timeout_s}s",
            duration_ms=duration_ms,
            command=cmd,
        )
