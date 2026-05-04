from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from openshard.security.paths import UnsafePathError, resolve_safe_repo_path


@dataclass
class NativeTool:
    name: str
    description: str
    risk: str  # "safe" | "needs_approval" | "blocked"
    categories: list[str]


@dataclass
class NativeToolCall:
    tool_name: str
    args: dict
    approved: bool = False


@dataclass
class NativeToolResult:
    tool_name: str
    ok: bool
    output: str = ""
    error: str | None = None
    metadata: dict = field(default_factory=dict)


_BUILTIN_TOOLS: list[NativeTool] = [
    NativeTool(
        name="list_files",
        description="List files in the repository or a subdirectory.",
        risk="safe",
        categories=["repo", "navigation"],
    ),
    NativeTool(
        name="read_file",
        description="Read the contents of a file within the repository.",
        risk="safe",
        categories=["repo", "navigation"],
    ),
    NativeTool(
        name="search_repo",
        description="Search for patterns or symbols across repository files.",
        risk="safe",
        categories=["repo", "navigation"],
    ),
    NativeTool(
        name="get_git_diff",
        description="Retrieve the current git diff for inspection.",
        risk="safe",
        categories=["repo", "inspection"],
    ),
    NativeTool(
        name="write_file",
        description="Write or overwrite a file within the repository.",
        risk="needs_approval",
        categories=["repo", "mutation"],
    ),
    NativeTool(
        name="run_verification",
        description="Run the project verification plan (tests, lint, typecheck).",
        risk="safe",
        categories=["verification"],
    ),
    NativeTool(
        name="run_command",
        description="Execute an arbitrary shell command.",
        risk="blocked",
        categories=["shell"],
    ),
]


def list_native_tools() -> list[NativeTool]:
    return list(_BUILTIN_TOOLS)


def get_native_tool(name: str) -> NativeTool | None:
    for tool in _BUILTIN_TOOLS:
        if tool.name == name:
            return tool
    return None


def classify_native_tool(name: str) -> str:
    tool = get_native_tool(name)
    return tool.risk if tool is not None else "blocked"


def compact_tool_result(output: str, limit: int = 4000) -> str:
    if len(output) <= limit:
        return output
    return output[:limit] + f"\n[truncated: output exceeded {limit} chars]"


_IGNORE_DIRS: frozenset[str] = frozenset({".git", "__pycache__", ".openshard"})
_IGNORE_SUFFIXES: frozenset[str] = frozenset({".pyc"})


def _exec_list_files(repo_root: Path, subdir: str = ".") -> NativeToolResult:
    try:
        if subdir == ".":
            base = repo_root.resolve()
        else:
            base = resolve_safe_repo_path(repo_root, subdir)
    except UnsafePathError as exc:
        return NativeToolResult(tool_name="list_files", ok=False, error=str(exc))

    repo_resolved = repo_root.resolve()
    paths: list[str] = []

    for dirpath, dirnames, filenames in os.walk(base):
        dirnames[:] = [d for d in dirnames if d not in _IGNORE_DIRS]
        for fname in filenames:
            if Path(fname).suffix in _IGNORE_SUFFIXES:
                continue
            full = Path(dirpath) / fname
            try:
                rel = full.relative_to(repo_resolved)
                paths.append(str(rel))
            except ValueError:
                continue

    output = "\n".join(sorted(paths))
    return NativeToolResult(tool_name="list_files", ok=True, output=output)


def _exec_read_file(repo_root: Path, path: str) -> NativeToolResult:
    try:
        safe = resolve_safe_repo_path(repo_root, path)
        text = safe.read_text(encoding="utf-8", errors="replace")
        return NativeToolResult(
            tool_name="read_file",
            ok=True,
            output=compact_tool_result(text),
        )
    except UnsafePathError as exc:
        return NativeToolResult(tool_name="read_file", ok=False, error=str(exc))
    except OSError as exc:
        return NativeToolResult(tool_name="read_file", ok=False, error=str(exc))
