from __future__ import annotations

from pathlib import Path

from openshard.native.agent_loop_types import AgentAction, ToolResult
from openshard.native.tools import (
    NativeToolResult,
    _exec_get_git_diff,
    _exec_list_files,
    _exec_read_file,
    _exec_run_verification,
    _exec_search_repo,
    classify_native_tool,
)

# ActionKinds that are loop-level signals — never dispatched to a tool executor.
_LOOP_SIGNALS: frozenset[str] = frozenset({"ask_human", "finish"})


def _wrap(nr: NativeToolResult, action_id: str) -> ToolResult:
    """Convert a NativeToolResult into a ToolResult.

    output='' becomes None so callers can do a simple truthiness check.
    raw_content_stored is always False — no raw output is persisted.
    """
    return ToolResult(
        action_id=action_id,
        ok=nr.ok,
        output=nr.output or None,
        error=nr.error,
        exit_code=nr.metadata.get("exit_code") if nr.metadata else None,
        duration_ms=int(nr.metadata.get("duration_ms", 0)) if nr.metadata else 0,
    )


class ReadOnlyToolRunner:
    """Dispatches safe, read-only AgentActions to their executor functions.

    - 'safe' tools: dispatched immediately against repo_root.
    - 'needs_approval' / 'blocked' tools: rejected — a ToolResult with ok=False
      is returned so the loop can handle it without crashing.
    - Loop signals ('ask_human', 'finish'): rejected — the loop must handle
      these before calling run().
    - Unknown kinds: rejected with an explicit error.

    No writes are ever performed here. Phase 4 introduces write dispatch
    behind the approval gate.
    """

    def __init__(self, repo_root: Path) -> None:
        self._root = Path(repo_root)

    def run(self, action: AgentAction) -> ToolResult:
        kind = action.kind

        if kind in _LOOP_SIGNALS:
            return ToolResult(
                action_id=action.action_id,
                ok=False,
                error=f"'{kind}' is a loop signal — handle it at the loop level before dispatch",
            )

        risk = classify_native_tool(kind)

        if risk == "blocked":
            return ToolResult(
                action_id=action.action_id,
                ok=False,
                error=f"tool '{kind}' is blocked and cannot be executed",
            )

        if risk == "needs_approval":
            return ToolResult(
                action_id=action.action_id,
                ok=False,
                error=f"tool '{kind}' requires approval — pass through the approval gate first",
            )

        return self._dispatch(action)

    def _dispatch(self, action: AgentAction) -> ToolResult:
        kind = action.kind
        args = action.args

        if kind == "list_files":
            nr = _exec_list_files(self._root, args.get("subdir", "."))

        elif kind == "read_file":
            path = args.get("path", "")
            if not path:
                return ToolResult(
                    action_id=action.action_id,
                    ok=False,
                    error="read_file requires a non-empty 'path' argument",
                )
            nr = _exec_read_file(self._root, path)

        elif kind == "search_repo":
            query = args.get("query", "")
            max_matches = args.get("max_matches", 50)
            nr = _exec_search_repo(self._root, query, max_matches=max_matches)

        elif kind == "get_git_diff":
            nr = _exec_get_git_diff(self._root)

        elif kind == "run_verification":
            nr = _exec_run_verification(self._root)

        else:
            return ToolResult(
                action_id=action.action_id,
                ok=False,
                error=f"unknown tool kind: '{kind}'",
            )

        return _wrap(nr, action.action_id)
