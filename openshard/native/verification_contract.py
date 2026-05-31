from __future__ import annotations

from dataclasses import dataclass, field

_ALLOWED_STATUSES: frozenset[str] = frozenset(
    {"not_run", "passed", "failed", "skipped", "impossible", "manual_review", "unknown"}
)

_MAX_CHECKS: int = 8
_MAX_REASON_CHARS: int = 200
_MAX_SUMMARY_CHARS: int = 200


@dataclass
class OSNVerificationContract:
    """Safe, bounded verification contract for OSN runs.

    Never stores raw command output, raw file contents, raw prompts, or absolute paths.
    All list fields are capped in __post_init__. No em dash characters.
    """

    enabled: bool = False
    status: str = "not_run"
    required: bool = False
    source: str = "osn_verification_contract_v1"
    expected_checks: list[str] = field(default_factory=list)
    attempted_checks: list[str] = field(default_factory=list)
    passed_checks: list[str] = field(default_factory=list)
    failed_checks: list[str] = field(default_factory=list)
    skipped_checks: list[str] = field(default_factory=list)
    missing_checks: list[str] = field(default_factory=list)
    skipped_reason: str = ""
    manual_review_required: bool = False
    summary: str = ""

    def __post_init__(self) -> None:
        self.expected_checks = self.expected_checks[:_MAX_CHECKS]
        self.attempted_checks = self.attempted_checks[:_MAX_CHECKS]
        self.passed_checks = self.passed_checks[:_MAX_CHECKS]
        self.failed_checks = self.failed_checks[:_MAX_CHECKS]
        self.skipped_checks = self.skipped_checks[:_MAX_CHECKS]
        self.missing_checks = self.missing_checks[:_MAX_CHECKS]
        if len(self.skipped_reason) > _MAX_REASON_CHARS:
            self.skipped_reason = self.skipped_reason[:_MAX_REASON_CHARS]
        if len(self.summary) > _MAX_SUMMARY_CHARS:
            self.summary = self.summary[:_MAX_SUMMARY_CHARS]
        if self.status not in _ALLOWED_STATUSES:
            self.status = "unknown"


def build_osn_verification_contract(
    *,
    osn_observation: object | None,
    osn_loop_summary: object | None,
    is_write_task: bool = False,
    verification_loop: object | None = None,
) -> OSNVerificationContract:
    """Build a deterministic verification contract from recorded signals.

    Does not execute shell commands. Does not call providers. Does not invent
    per-check proof. Only populates check lists when data proves it.
    """
    contract = OSNVerificationContract(enabled=True)

    # Pull expected checks from observation
    if osn_observation is not None:
        raw_checks = getattr(osn_observation, "suggested_checks", []) or []
        contract.expected_checks = list(raw_checks)[:_MAX_CHECKS]

    # Populate per-check proof from real execution results
    _loop_skipped_reasons: list[str] = []
    if verification_loop is not None:
        _ca = getattr(verification_loop, "check_attempted", []) or []
        _cp = getattr(verification_loop, "check_passed", []) or []
        _cf = getattr(verification_loop, "check_failed", []) or []
        _cs = getattr(verification_loop, "check_skipped", []) or []
        _loop_skipped_reasons = list(getattr(verification_loop, "check_skipped_reasons", []) or [])
        contract.attempted_checks = list(_ca)[:_MAX_CHECKS]
        contract.passed_checks = list(_cp)[:_MAX_CHECKS]
        contract.failed_checks = list(_cf)[:_MAX_CHECKS]
        contract.skipped_checks = list(_cs)[:_MAX_CHECKS]
        # Seed expected_checks from execution plan when observation has none
        if not contract.expected_checks:
            contract.expected_checks = (_ca + _cs)[:_MAX_CHECKS]

    if osn_loop_summary is None:
        contract.status = "not_run"
        contract.summary = "no loop summary available"
        return contract

    v_attempted: bool = bool(getattr(osn_loop_summary, "verification_attempted", False))
    v_passed: bool | None = getattr(osn_loop_summary, "verification_passed", None)
    v_status: str = getattr(osn_loop_summary, "verification_status", "") or ""
    stopped: str = getattr(osn_loop_summary, "stopped_reason", "") or ""

    if not v_attempted:
        contract.status = "skipped"
        contract.skipped_reason = "verification not attempted"
        if is_write_task:
            contract.manual_review_required = True
        _summary_parts = ["skipped"]
        if is_write_task:
            _summary_parts.append("write task requires manual review")
        contract.summary = ", ".join(_summary_parts)
    elif v_passed is True:
        contract.status = "passed"
        # Do not populate attempted_checks or passed_checks from expected_checks -
        # no per-check proof exists. Status alone records the outcome.
        contract.summary = "verification passed"
    elif v_passed is False:
        contract.status = "failed"
        contract.manual_review_required = True
        contract.summary = "verification failed"
    elif v_status == "skipped":
        contract.status = "skipped"
        contract.skipped_reason = "verification was skipped"
        if is_write_task:
            contract.manual_review_required = True
        contract.summary = "verification skipped"
    elif v_attempted:
        contract.status = "unknown"
        contract.manual_review_required = True
        contract.summary = "verification attempted but outcome unknown"
    else:
        contract.status = "unknown"
        contract.summary = "verification state unknown"

    # stopped_reason=verification_failed overrides status if not already failed
    if stopped == "verification_failed" and contract.status != "failed":
        contract.status = "failed"
        contract.manual_review_required = True
        contract.summary = "stopped: verification failed"

    # Prefer specific command reason over the generic fallback when we have one
    if _loop_skipped_reasons and contract.skipped_checks:
        contract.skipped_reason = _loop_skipped_reasons[0][:_MAX_REASON_CHARS]

    # missing_checks = expected checks not found in attempted
    contract.missing_checks = [
        c for c in contract.expected_checks if c not in contract.attempted_checks
    ][:_MAX_CHECKS]

    # Enforce caps and char limits via __post_init__ logic
    if len(contract.summary) > _MAX_SUMMARY_CHARS:
        contract.summary = contract.summary[:_MAX_SUMMARY_CHARS]
    if len(contract.skipped_reason) > _MAX_REASON_CHARS:
        contract.skipped_reason = contract.skipped_reason[:_MAX_REASON_CHARS]

    return contract


def render_osn_verification_context(contract: OSNVerificationContract | None) -> str:
    """Prompt-safe one-liner for context injection. No raw content, no absolute paths."""
    if contract is None or not contract.enabled:
        return ""
    parts = [f"[osn verification] status={contract.status}"]
    if contract.expected_checks:
        parts.append(f"expected={', '.join(contract.expected_checks[:3])}")
    if contract.manual_review_required:
        parts.append("manual-review=yes")
    return " ".join(parts)


def render_osn_verification_receipt(
    contract: OSNVerificationContract | None,
    *,
    detail: str = "default",
) -> list[str]:
    """Return receipt lines for OSN VERIFICATION block. No raw output, no em dashes."""
    if contract is None or not contract.enabled:
        return []
    lines: list[str] = []
    lines.append("  OSN VERIFICATION")
    lines.append(f"    Status       {contract.status}")
    if contract.expected_checks:
        lines.append(f"    Expected     {', '.join(contract.expected_checks[:4])}")
    if contract.attempted_checks:
        lines.append(f"    Attempted    {', '.join(contract.attempted_checks[:4])}")
    if contract.passed_checks:
        lines.append(f"    Passed       {', '.join(contract.passed_checks[:4])}")
    if contract.failed_checks:
        lines.append(f"    Failed       {', '.join(contract.failed_checks[:4])}")
    if contract.skipped_checks:
        lines.append(f"    Skipped      {', '.join(contract.skipped_checks[:4])}")
    if contract.missing_checks:
        lines.append(f"    Missing      {', '.join(contract.missing_checks[:4])}")
    lines.append(f"    Review       {'yes' if contract.manual_review_required else 'no'}")
    if contract.skipped_reason:
        lines.append(f"    Reason       {contract.skipped_reason}")
    if detail == "full" and contract.summary:
        lines.append(f"    Summary      {contract.summary}")
    return lines
