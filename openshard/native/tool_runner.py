from __future__ import annotations

from pathlib import Path

from openshard.native.tools import (
    NativeToolCall,
    NativeToolResult,
    _exec_list_files,
    _exec_read_file,
    _exec_search_repo,
    classify_native_tool,
)


class NativeToolRunner:
    """Executes allowed deterministic native tools against a fixed repo root."""

    def __init__(self, repo_root: Path) -> None:
        self._repo_root = repo_root

    def run(self, call: NativeToolCall) -> NativeToolResult:
        risk = classify_native_tool(call.tool_name)

        if risk == "blocked":
            return NativeToolResult(
                tool_name=call.tool_name,
                ok=False,
                error=f"Tool '{call.tool_name}' is blocked.",
            )

        if risk == "needs_approval" and not call.approved:
            return NativeToolResult(
                tool_name=call.tool_name,
                ok=False,
                error=f"Tool '{call.tool_name}' requires approval.",
            )

        args = call.args if isinstance(call.args, dict) else {}

        if call.tool_name == "write_file":
            return NativeToolResult(
                tool_name=call.tool_name,
                ok=False,
                error="Tool 'write_file' is not implemented yet.",
            )

        if call.tool_name == "list_files":
            return _exec_list_files(self._repo_root, args.get("subdir", "."))

        if call.tool_name == "read_file":
            return _exec_read_file(self._repo_root, args.get("path", ""))

        if call.tool_name == "search_repo":
            return _exec_search_repo(
                self._repo_root,
                args.get("query", ""),
                max_matches=args.get("max_matches", 50),
            )

        return NativeToolResult(
            tool_name=call.tool_name,
            ok=False,
            error=f"Tool '{call.tool_name}' is not yet implemented.",
        )

    def trace_entry(self, call: NativeToolCall, result: NativeToolResult) -> dict:
        return {
            "tool": call.tool_name,
            "ok": result.ok,
            "approved": call.approved,
            "output_chars": len(result.output),
            "error": result.error,
        }
