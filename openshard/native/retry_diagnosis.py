"""OSN Retry Diagnosis v1.

Deterministic, safe diagnosis object built from recorded OSN loop and
verification signals. No shell execution, no provider calls, no raw output.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

_MAX_STR: int = 120
_MAX_LIST: int = 5

_VALID_STATUSES: frozenset[str] = frozenset({
    "not_needed",
    "allowed",
    "used",
    "exhausted",
    "blocked",
    "manual_review",
    "unknown",
})

_ALLOWED_CHANGES_DEFAULT: list[str] = [
    "adjust only files touched in this run",
    "fix failing verification path",
]

_BLOCKED_CHANGES_DEFAULT: list[str] = [
    "do not expand scope",
    "do not touch unrelated files",
    "do not bypass approval",
    "do not ignore failed checks",
]

_BLOCKED_CHANGES_APPROVAL: list[str] = [
    "do not expand scope",
    "do not bypass approval",
]

_NEXT_ACTIONS: dict[str, str] = {
    "allowed": "fix failing verification path",
    "exhausted": "manual review required before retry",
    "blocked": "address approval blockers",
    "manual_review": "run focused verification before proceeding",
    "used": "verify result before merging",
    "unknown": "review run state",
}

_FAILURE_KINDS: dict[str, str] = {
    "blocked": "approval_denied",
    "exhausted": "verification_failed",
    "allowed": "verification_failed",
    "manual_review": "missing_verification",
    "unknown": "unknown",
}

_FAILURE_SUMMARIES: dict[str, str] = {
    "blocked": "approval denied",
    "exhausted": "verification failed - retry limit reached",
    "allowed": "verification failed",
    "manual_review": "verification not attempted on write task",
    "unknown": "verification state could not be determined",
}

_RETRY_REASONS: dict[str, str] = {
    "allowed": "verification failed",
    "exhausted": "verification failed",
    "used": "verification failed on first attempt",
}


@dataclass
class OSNRetryDiagnosis:
    """Safe, bounded retry diagnosis for OSN runs.

    Never stores raw command output, raw file contents, raw prompts, or
    absolute paths. All list fields capped. No em dash characters.
    """

    enabled: bool = False
    retry_allowed: bool = False
    retry_used: bool = False
    retry_count: int = 0
    retry_limit: int = 1
    status: str = "not_needed"
    source: str = "osn_retry_diagnosis_v1"
    failure_kind: str = ""
    failure_summary: str = ""
    retry_reason: str = ""
    allowed_changes: list[str] = field(default_factory=list)
    blocked_changes: list[str] = field(default_factory=list)
    next_action: str = ""
    manual_review_required: bool = False
    raw_content_stored: bool = False

    def __post_init__(self) -> None:
        self.raw_content_stored = False
        if self.status not in _VALID_STATUSES:
            self.status = "unknown"
        self.failure_kind = self.failure_kind[:_MAX_STR]
        self.failure_summary = self.failure_summary[:_MAX_STR]
        self.retry_reason = self.retry_reason[:_MAX_STR]
        self.next_action = self.next_action[:_MAX_STR]
        self.allowed_changes = [s[:_MAX_STR] for s in self.allowed_changes[:_MAX_LIST]]
        self.blocked_changes = [s[:_MAX_STR] for s in self.blocked_changes[:_MAX_LIST]]


def build_osn_retry_diagnosis(
    *,
    osn_loop_summary: object | None = None,
    osn_verification_contract: object | None = None,
    approval_required: bool = False,
    approval_granted: bool = False,
    retry_limit: int = 1,
) -> OSNRetryDiagnosis:
    """Build a deterministic retry diagnosis from recorded OSN signals.

    Does not execute shell commands. Does not call providers or models. Reads
    only from osn_loop_summary and osn_verification_contract.
    """
    if osn_loop_summary is None or not getattr(osn_loop_summary, "enabled", False):
        return OSNRetryDiagnosis()

    retry_used: bool = bool(getattr(osn_loop_summary, "retry_used", False))
    retry_count: int = int(getattr(osn_loop_summary, "retry_count", 0) or 0)
    stopped_reason: str = getattr(osn_loop_summary, "stopped_reason", "") or ""
    v_attempted: bool = bool(getattr(osn_loop_summary, "verification_attempted", False))
    v_passed: bool | None = getattr(osn_loop_summary, "verification_passed", None)

    vc_status: str | None = (
        getattr(osn_verification_contract, "status", None)
        if osn_verification_contract is not None
        else None
    )
    vc_manual: bool = bool(
        getattr(osn_verification_contract, "manual_review_required", False)
        if osn_verification_contract is not None
        else False
    )

    # Determine status in priority order
    if approval_required and not approval_granted:
        status = "blocked"
    elif v_attempted and v_passed is True and retry_used:
        status = "used"
    elif v_attempted and v_passed is True and not retry_used:
        status = "not_needed"
    elif stopped_reason == "retry_limit":
        status = "exhausted"
    elif v_attempted and v_passed is False and retry_count >= retry_limit:
        status = "exhausted"
    elif v_attempted and v_passed is False and retry_count < retry_limit:
        status = "allowed"
    elif not v_attempted and vc_status == "skipped":
        status = "manual_review"
    elif not v_attempted:
        status = "not_needed"
    else:
        status = "unknown"

    failure_kind = _FAILURE_KINDS.get(status, "")
    failure_summary = _FAILURE_SUMMARIES.get(status, "")
    retry_reason = _RETRY_REASONS.get(status, "")
    retry_allowed = status == "allowed"
    manual_review_required = status in ("exhausted", "blocked", "manual_review") or vc_manual
    next_action = _NEXT_ACTIONS.get(status, "")

    if status in ("allowed", "used"):
        allowed_changes = list(_ALLOWED_CHANGES_DEFAULT)
        blocked_changes = list(_BLOCKED_CHANGES_DEFAULT)
    elif status in ("exhausted", "manual_review"):
        allowed_changes = []
        blocked_changes = list(_BLOCKED_CHANGES_DEFAULT)
    elif status == "blocked":
        allowed_changes = []
        blocked_changes = list(_BLOCKED_CHANGES_APPROVAL)
    else:
        allowed_changes = []
        blocked_changes = []

    # not_needed with no retry metadata - no useful diagnosis to show
    enabled = status != "not_needed"

    return OSNRetryDiagnosis(
        enabled=enabled,
        retry_allowed=retry_allowed,
        retry_used=retry_used,
        retry_count=retry_count,
        retry_limit=retry_limit,
        status=status,
        failure_kind=failure_kind,
        failure_summary=failure_summary,
        retry_reason=retry_reason,
        allowed_changes=allowed_changes,
        blocked_changes=blocked_changes,
        next_action=next_action,
        manual_review_required=manual_review_required,
    )


def render_osn_retry_receipt(diag: Any, *, detail: str = "default") -> list[str]:
    """Return receipt lines for OSN RETRY block. No raw output, no em dashes."""
    if diag is None or not getattr(diag, "enabled", False):
        return []
    status = getattr(diag, "status", "unknown") or "unknown"
    retry_used = getattr(diag, "retry_used", False)
    retry_count = getattr(diag, "retry_count", 0)
    retry_limit = getattr(diag, "retry_limit", 1)
    failure_kind = getattr(diag, "failure_kind", "") or ""
    failure_summary = getattr(diag, "failure_summary", "") or ""
    next_action = getattr(diag, "next_action", "") or ""
    manual_review_required = getattr(diag, "manual_review_required", False)
    lines: list[str] = []
    lines.append("  OSN RETRY")
    lines.append(f"    Status       {status}")
    lines.append(f"    Used         {'yes' if retry_used else 'no'}")
    lines.append(f"    Count        {retry_count}/{retry_limit}")
    if failure_kind:
        lines.append(f"    Failure      {failure_kind}")
    if failure_summary:
        lines.append(f"    Reason       {failure_summary}")
    if next_action:
        lines.append(f"    Next         {next_action}")
    if manual_review_required:
        lines.append("    Review       yes")
    return lines
