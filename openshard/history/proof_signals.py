"""Neutral, public proof-signal helpers derived from a Shard receipt.

These are small, pure reductions of a ``ShardReceipt`` into proof signals that
several consumers need (the proof contract, the CI policy check, completeness).
They live in the ``history`` layer - the home of core Shard logic - so that
``history`` modules never have to reach into private ``ci`` helpers.

All functions are pure (no I/O), never raise, and emit only safe tokens / ints.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from openshard.history.shard_contract import ShardReceipt


def verification_status_from_receipt(receipt: "ShardReceipt") -> str:
    """Map a receipt's status into a verification enum.

    Returns one of: ``passed`` | ``failed`` | ``not_run`` | ``unknown``.
    """
    try:
        status = (receipt.status or "").strip()
        if status.startswith("Checks:"):
            # Review-style checks: failed if any sub-check failed, else passed.
            display = (receipt.checks_display or "").lower()
            return "failed" if "failed" in display else "passed"
        return {
            "Passed": "passed",
            "Failed": "failed",
            "No checks run": "not_run",
            "Not recorded": "unknown",
        }.get(status, "unknown")
    except Exception:
        return "unknown"


def secret_scan_finding_count(receipt: "ShardReceipt") -> int:
    """Count redacted secret-scan evidence capsules on the receipt."""
    try:
        return sum(
            1
            for ec in (receipt.evidence_capsules or [])
            if getattr(ec, "kind", None) == "secret_scan"
        )
    except Exception:
        return 0
