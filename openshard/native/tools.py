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
_BINARY_SUFFIXES: frozenset[str] = frozenset({
    ".png", ".jpg", ".jpeg", ".gif", ".ico", ".svg",
    ".pdf", ".zip", ".tar", ".gz", ".exe", ".bin",
    ".woff", ".woff2", ".ttf", ".eot",
})


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


def _exec_search_repo(
    repo_root: Path,
    query: str,
    *,
    max_matches: int = 50,
) -> NativeToolResult:
    if not query or not query.strip():
        return NativeToolResult(
            tool_name="search_repo",
            ok=False,
            error="search_repo requires a non-empty query.",
        )

    try:
        max_matches_int = int(max_matches)
    except (TypeError, ValueError):
        max_matches_int = 50
    max_matches_int = max(1, max_matches_int)

    query_lower = query.strip().lower()
    repo_resolved = repo_root.resolve()
    matches: list[str] = []
    truncated = False

    for dirpath, dirnames, filenames in os.walk(repo_resolved):
        dirnames[:] = [d for d in dirnames if d not in _IGNORE_DIRS]
        for fname in filenames:
            p = Path(fname)
            if p.suffix in _IGNORE_SUFFIXES or p.suffix in _BINARY_SUFFIXES:
                continue
            full = Path(dirpath) / fname
            try:
                rel = str(full.relative_to(repo_resolved))
            except ValueError:
                continue
            try:
                text = full.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            for lineno, line in enumerate(text.splitlines(), start=1):
                if query_lower in line.lower():
                    matches.append(f"{rel}:{lineno}:{line.rstrip()}")
                    if len(matches) >= max_matches_int:
                        truncated = True
                        break
            if truncated:
                break

    output = compact_tool_result("\n".join(matches))
    return NativeToolResult(
        tool_name="search_repo",
        ok=True,
        output=output,
        metadata={"matches": len(matches), "truncated": truncated},
    )


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
