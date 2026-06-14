from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

from openshard.native.agent_loop_types import AgentAction, ApprovalReceipt, ToolResult
from openshard.native.agent_loop_tool_runner import ReadOnlyToolRunner
from openshard.security.paths import UnsafePathError, resolve_safe_repo_path


@dataclass
class WriteGrant:
    """Carries the ApprovalReceipt that authorises a specific write_file action.

    Both fields must match the action being executed:
    - receipt.granted must be True
    - action_id must equal AgentAction.action_id

    A grant is single-use by convention — the caller should not reuse
    the same grant for a different action.
    """
    receipt: ApprovalReceipt
    action_id: str


def _exec_write_file(
    repo_root: Path,
    path: str,
    content: str,
    action_id: str,
) -> ToolResult:
    """Write content to a repo-relative path.

    Uses resolve_safe_repo_path to prevent path traversal.
    Creates parent directories as needed.
    raw_content_stored is always False — content length is recorded, not content.
    """
    if not path:
        return ToolResult(
            action_id=action_id,
            ok=False,
            error="write_file requires a non-empty 'path' argument",
        )

    t0 = time.monotonic()
    try:
        safe = resolve_safe_repo_path(repo_root, path)
        safe.parent.mkdir(parents=True, exist_ok=True)
        safe.write_text(content, encoding="utf-8")
        duration_ms = int((time.monotonic() - t0) * 1000)
        return ToolResult(
            action_id=action_id,
            ok=True,
            output=f"wrote {len(content)} chars to {path}",
            duration_ms=duration_ms,
        )
    except UnsafePathError as exc:
        return ToolResult(action_id=action_id, ok=False, error=str(exc))
    except OSError as exc:
        return ToolResult(action_id=action_id, ok=False, error=str(exc))


class GatedToolRunner:
    """Extends read-only dispatch with approval-gated write_file support.

    Safe (read-only) actions are delegated to ReadOnlyToolRunner unchanged.
    write_file requires a WriteGrant with a matching action_id and a receipt
    where granted=True. Without a valid grant the action is rejected inline —
    no filesystem access occurs.

    run_command remains blocked and loop signals remain rejected regardless
    of any grant provided.
    """

    def __init__(self, repo_root: Path) -> None:
        self._root = Path(repo_root)
        self._read_runner = ReadOnlyToolRunner(repo_root)

    def run(self, action: AgentAction, *, grant: WriteGrant | None = None) -> ToolResult:
        if action.kind == "write_file":
            return self._run_write(action, grant)
        return self._read_runner.run(action)

    def _run_write(self, action: AgentAction, grant: WriteGrant | None) -> ToolResult:
        if grant is None:
            return ToolResult(
                action_id=action.action_id,
                ok=False,
                error="write_file requires a WriteGrant — none provided",
            )

        if not grant.receipt.granted:
            return ToolResult(
                action_id=action.action_id,
                ok=False,
                error="write_file approval was denied",
            )

        if grant.action_id != action.action_id:
            return ToolResult(
                action_id=action.action_id,
                ok=False,
                error=(
                    f"grant action_id mismatch: "
                    f"expected '{action.action_id}', got '{grant.action_id}'"
                ),
            )

        return _exec_write_file(
            self._root,
            action.args.get("path", ""),
            action.args.get("content", ""),
            action.action_id,
        )
