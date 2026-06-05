"""Pure, deterministic CI policy evaluation over a Shard receipt.

This module contains no I/O: it takes an already-loaded run entry and its
built ``ShardReceipt`` and reduces them to a single CI verdict
(``pass`` | ``warn`` | ``fail`` | ``skip``) with the matching exit code.

Decision rules (priority order):

1. ``fail`` (exit 1) when verification failed, or manual review is required by
   an independent gate (denied approval, progress blockers, retry diagnosis,
   policy deny).
2. ``warn`` (exit 0) when verification was not run / skipped / needs manual
   review / is unknown, or redacted secret-scan findings are present.
3. ``pass`` (exit 0) otherwise.

``--strict`` promotes any ``warn`` to ``fail``. The ``skip`` state (no run to
evaluate) is produced by the caller, not here.

Reasons and warnings are short, plain-English strings. They never contain raw
secret values or absolute paths.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from openshard.history.proof_signals import (
    secret_scan_finding_count,
    verification_status_from_receipt,
)

if TYPE_CHECKING:
    from openshard.history.shard_contract import ShardReceipt


@dataclass
class CICheckResult:
    """Outcome of a CI policy check over a single receipt."""

    status: str  # "pass" | "warn" | "fail" | "skip"
    exit_code: int
    shard_id: str | None
    reasons: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    checks: dict = field(default_factory=dict)


# Thin backward-compatible aliases over the shared, public proof signals in
# ``openshard.history.proof_signals`` (the single source of truth). Kept so
# existing imports of these private names continue to work.
_verification_status = verification_status_from_receipt


def _independent_manual_review(entry: dict, receipt: "ShardReceipt") -> bool:
    """Manual review required by a gate independent of verification.

    These always block CI: denied approval, progress blockers, retry diagnosis,
    and policy deny. Verification-driven review (skipped or manual_review) is
    excluded here so it warns rather than fails.
    """
    if receipt.approval_required and not receipt.approval_granted:
        return True

    prog = entry.get("osn_progress_memory")
    if isinstance(prog, dict) and prog.get("blockers"):
        return True

    retry = entry.get("osn_retry_diagnosis")
    if isinstance(retry, dict) and retry.get("manual_review_required"):
        return True

    if any(
        isinstance(pd, dict) and pd.get("decision") == "deny"
        for pd in receipt.policy_decisions
    ):
        return True

    return False


def _manual_review_required(entry: dict, receipt: "ShardReceipt") -> bool:
    """Full manual-review detection, including the verification contract signal.

    Used by failure classification and trust scoring as a weak signal. CI uses
    the narrower ``_independent_manual_review`` so a skipped or manual_review
    verification warns rather than fails.
    """
    if _independent_manual_review(entry, receipt):
        return True

    verif = entry.get("osn_verification_contract")
    if isinstance(verif, dict) and verif.get("manual_review_required"):
        return True

    return False


_secret_scan_findings = secret_scan_finding_count


def evaluate_ci_check(
    entry: dict, receipt: "ShardReceipt", *, strict: bool = False
) -> CICheckResult:
    """Evaluate a single run entry/receipt into a CI verdict. Never raises."""
    verification = _verification_status(receipt)
    manual_review = _independent_manual_review(entry, receipt)
    secret_findings = _secret_scan_findings(receipt)

    reasons: list[str] = []
    warnings: list[str] = []
    status = "pass"

    # Blocking conditions.
    if verification == "failed":
        reasons.append("Verification failed.")
        status = "fail"
    if manual_review:
        reasons.append("Manual review required.")
        status = "fail"

    # Non-blocking concerns (only when not already failing). A skipped or
    # manual_review verification is a weak signal, so it warns rather than fails.
    if status != "fail":
        if verification == "not_run":
            warnings.append("Verification was not run.")
            status = "warn"
        elif verification == "skipped":
            reason = (getattr(receipt, "verification_reason", "") or "").strip()
            warnings.append(
                f"Verification was skipped: {reason}" if reason else "Verification was skipped."
            )
            status = "warn"
        elif verification == "manual_review":
            warnings.append("Verification needs manual review.")
            status = "warn"
        elif verification == "unknown":
            warnings.append("Verification status could not be determined from the receipt.")
            status = "warn"
        if secret_findings > 0:
            warnings.append(
                f"{secret_findings} secret-scan finding(s) detected (redacted)."
            )
            status = "warn"

    # Strict mode: promote warnings to blocking failures.
    if strict and status == "warn":
        status = "fail"
        reasons.extend(warnings)
        warnings = []

    exit_code = 1 if status == "fail" else 0

    return CICheckResult(
        status=status,
        exit_code=exit_code,
        shard_id=receipt.shard_id or None,
        reasons=reasons,
        warnings=warnings,
        checks={
            "verification": verification,
            "manual_review_required": manual_review,
            "secret_scan_findings": secret_findings,
        },
    )
