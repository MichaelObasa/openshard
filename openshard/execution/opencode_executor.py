from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import sys
from pathlib import Path

from openshard.config.settings import load_config
from openshard.execution.generator import ChangedFile, ExecutionResult

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_OPENCODE_MODEL_PREFIX = "openrouter/"


def _resolve_opencode_binary() -> str:
    """Return the path to the opencode executable, or raise RuntimeError."""
    found = shutil.which("opencode")
    if found:
        return found

    if sys.platform == "win32":
        appdata = os.environ.get("APPDATA", "")
        if appdata:
            for name in ("opencode.cmd", "opencode.ps1", "opencode"):
                candidate = Path(appdata) / "npm" / name
                if candidate.is_file():
                    return str(candidate)

    raise RuntimeError(
        "opencode binary not found. "
        "Install OpenCode and ensure it is on your PATH."
    )


# ---------------------------------------------------------------------------
# Executor
# ---------------------------------------------------------------------------

class OpenCodeExecutor:
    """Execute tasks by calling the ``opencode`` CLI as a subprocess.

    Conforms to the same interface as :class:`ExecutionGenerator` so the
    ``run`` command can swap backends without any other changes.

    The *workspace* parameter of :meth:`generate` must point to a directory
    that already contains the target codebase (populated by the CLI before
    this method is called).  OpenCode is invoked directly inside that
    directory, reads the existing files, and writes its changes in place.
    A before/after snapshot is used to classify each changed file as
    create / update / delete.
    """

    def __init__(self) -> None:
        config = load_config()
        self.model: str = self._resolve_model(config)
        self.fixer_model: str = self._resolve_fixer_model(config)

    def generate(
        self, task: str, model: str | None = None, workspace: Path | None = None
    ) -> ExecutionResult:
        """Run ``opencode run`` for *task* and return a parsed result.

        *workspace* — directory to run in.  Must already contain the repo
        files.  If ``None``, falls back to the current working directory
        (useful for ad-hoc / dry-run calls, but not recommended for write
        mode).

        *model* overrides the configured execution model when provided.
        """
        effective_model = model or self.model
        oc_model = f"{_OPENCODE_MODEL_PREFIX}{effective_model}"
        run_in = workspace if workspace is not None else Path.cwd()

        binary = _resolve_opencode_binary()
        before = _snapshot(run_in)

        proc = subprocess.run(
            [binary, "run", "--model", oc_model, task],
            cwd=run_in,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )

        raw_output = proc.stdout or ""

        if proc.returncode != 0:
            tail = raw_output[-2000:] if len(raw_output) > 2000 else raw_output
            raise RuntimeError(
                f"opencode exited with code {proc.returncode}.\n"
                f"Output:\n{tail}"
            )

        after = _snapshot(run_in)
        files = _classify_changes(run_in, before, after)
        summary = _extract_summary(raw_output)

        return ExecutionResult(
            summary=summary,
            files=files,
            notes=[],
            usage=None,
        )

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_model(config: dict) -> str:
        tiers: list[dict] = config.get("model_tiers", [])
        if not tiers:
            raise RuntimeError("No model_tiers defined in config.yml")
        if config.get("execution_model"):
            return config["execution_model"]
        balanced = next((t for t in tiers if t.get("name") == "balanced"), None)
        return (balanced or tiers[0])["model"]

    @staticmethod
    def _resolve_fixer_model(config: dict) -> str:
        tiers: list[dict] = config.get("model_tiers", [])
        if not tiers:
            raise RuntimeError("No model_tiers defined in config.yml")
        if config.get("fixer_model"):
            return config["fixer_model"]
        powerful = next((t for t in tiers if t.get("name") == "powerful"), None)
        balanced = next((t for t in tiers if t.get("name") == "balanced"), None)
        return (powerful or balanced or tiers[0])["model"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _snapshot(root: Path) -> dict[str, str]:
    """Return ``{relative_posix_path: md5_hex}`` for every file under *root*."""
    snap: dict[str, str] = {}
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        rel = path.relative_to(root).as_posix()
        try:
            snap[rel] = hashlib.md5(path.read_bytes()).hexdigest()
        except OSError:
            pass
    return snap


def _classify_changes(
    root: Path, before: dict[str, str], after: dict[str, str]
) -> list[ChangedFile]:
    """Return one :class:`ChangedFile` per file that was created, updated, or deleted."""
    files: list[ChangedFile] = []
    for rel in sorted(set(before) | set(after)):
        in_before = rel in before
        in_after  = rel in after
        if in_before and not in_after:
            files.append(ChangedFile(
                path=rel, change_type="delete", content="", summary=""
            ))
        elif not in_before and in_after:
            files.append(ChangedFile(
                path=rel, change_type="create",
                content=_read_file(root / rel), summary=""
            ))
        elif before[rel] != after[rel]:
            files.append(ChangedFile(
                path=rel, change_type="update",
                content=_read_file(root / rel), summary=""
            ))
        # unchanged — skip
    return files


def _read_file(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def _extract_summary(output: str) -> str:
    """Return the last non-empty line from OpenCode stdout, or a fallback."""
    for line in reversed(output.splitlines()):
        line = line.strip()
        if line:
            return line[:200]
    return "OpenCode execution completed."
