"""Reflector v0 - deterministic post-run reflection from a ShardReceipt.

Advisory only.  Does not change execution behaviour, mutate history,
or make any provider / model / network calls.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from openshard.history.shard_contract import ShardReceipt

_MAX_LIST = 5


@dataclass
class ReflectionSignal:
    kind: str
    severity: str  # "positive" | "negative" | "neutral"
    summary: str


@dataclass
class RunReflection:
    score: int               # 0–100
    level: str               # "strong" | "good" | "fair" | "weak"
    confidence: str          # "high" | "medium" | "low"
    summary: str
    strengths: list[str] = field(default_factory=list)
    gaps: list[str] = field(default_factory=list)
    suggestions: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    recommended_next_move: str | None = None
    source: str = "shard_receipt"
    version: str = "reflector_v0"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _level(score: int) -> str:
    if score >= 80:
        return "strong"
    if score >= 60:
        return "good"
    if score >= 40:
        return "fair"
    return "weak"


def _compute_confidence(receipt: ShardReceipt) -> str:
    """How much evidence does OpenShard have to judge this run?"""
    count = 0
    if receipt.schema_version:
        count += 1
    if receipt.check_results or receipt.status in ("Passed", "Failed"):
        count += 1
    if receipt.policy_decisions:
        count += 1
    if receipt.run_timeline:
        count += 1
    if receipt.evidence_capsules or receipt.inspected_files:
        count += 1
    if receipt.git_state is not None:
        count += 1
    if count >= 4:
        return "high"
    if count >= 2:
        return "medium"
    return "low"


def _has_secret_scan_findings(receipt: ShardReceipt) -> bool:
    return any(ec.kind == "secret_scan" for ec in receipt.evidence_capsules)


def _has_critical_or_high_findings(receipt: ShardReceipt) -> bool:
    return any(f.severity in ("Critical", "High") for f in receipt.findings)


def _checks_passed(receipt: ShardReceipt) -> bool:
    """True when checks were recorded and all passed (no failures)."""
    if receipt.status == "Passed":
        return True
    if receipt.check_results and "failed" not in receipt.checks_display.lower():
        return True
    return False


def _checks_failed(receipt: ShardReceipt) -> bool:
    """True when checks were recorded and at least one failed."""
    if receipt.status == "Failed":
        return True
    if receipt.check_results and "failed" in receipt.checks_display.lower():
        return True
    return False


def _any_checks_recorded(receipt: ShardReceipt) -> bool:
    return bool(receipt.check_results) or receipt.status in ("Passed", "Failed")


def _compute_summary(receipt: ShardReceipt, level: str) -> str:
    if receipt.approval_granted is False:
        return "Approval was denied for this run."
    if _has_secret_scan_findings(receipt):
        return "Secret scan findings were detected in this run."
    if receipt.error_class is not None:
        return "This run encountered an error."
    if _checks_failed(receipt):
        return "Verification checks did not pass for this run."
    if level == "strong":
        return "This run was well-evidenced with strong control signals."
    if level == "good":
        return "This run had useful evidence, but some control signals were missing."
    if level == "fair":
        return "This run had limited evidence or missing control signals."
    return "This run had significant gaps in evidence or did not complete."


def _pick_recommended_next_move(
    receipt: ShardReceipt,
    level: str,
    has_secret: bool,
    has_high_findings: bool,
    no_checks_write: bool,
) -> str | None:
    if receipt.approval_granted is False:
        return "Address approval blockers before re-running this task."
    if has_secret:
        return "Review secret scan warnings before sharing this context."
    if receipt.error_class is not None:
        return "Investigate the error before retrying."
    if _checks_failed(receipt):
        return "Review check failures before merging."
    if no_checks_write:
        return "Run focused verification before merging."
    if has_high_findings:
        return "Review high-severity findings before trusting this result."
    if receipt.adapter is not None and receipt.adapter_available is False:
        return "Check adapter availability before running adapter tasks."
    if level == "strong":
        return "This run looks good enough to review manually."
    if level == "good":
        return "Review any gaps noted above before merging."
    return "Use a staged workflow for the next risky write task."


# ---------------------------------------------------------------------------
# Main builder
# ---------------------------------------------------------------------------

def build_run_reflection(receipt: ShardReceipt) -> RunReflection:
    """Build a RunReflection from a ShardReceipt.  Pure and deterministic."""
    score = 50
    strengths: list[str] = []
    gaps: list[str] = []
    suggestions: list[str] = []
    warnings_: list[str] = []

    # --- error class --------------------------------------------------------
    if receipt.error_class is None:
        score += 10
        strengths.append("No errors recorded")
    else:
        score -= 20
        warnings_.append(f"Error recorded ({receipt.error_class})")

    # --- checks/verification ------------------------------------------------
    if _checks_passed(receipt):
        score += 15
        strengths.append("Checks recorded and passed")
    elif _any_checks_recorded(receipt):
        score += 8
        strengths.append("Checks recorded")

    if _checks_failed(receipt):
        score -= 15
        gaps.append("Checks did not pass")
        suggestions.append("Review check failures before merging.")

    # --- no checks for a write task -----------------------------------------
    _is_write_task = receipt.files_changed > 0
    _no_checks = not _any_checks_recorded(receipt)
    no_checks_write = _is_write_task and _no_checks
    if no_checks_write:
        gaps.append("No verification checks recorded for a write task")
        suggestions.append("Run focused verification before merging.")

    # --- run timeline -------------------------------------------------------
    if receipt.run_timeline:
        score += 5
        strengths.append("Run timeline present")
    else:
        gaps.append("Run timeline not recorded")

    # --- file evidence ------------------------------------------------------
    if receipt.inspected_files:
        score += 8
        strengths.append("File evidence present")
    else:
        gaps.append("No file evidence recorded")
        if "Add more file evidence" not in " ".join(suggestions):
            suggestions.append("Add more file evidence to improve context quality.")

    # --- policy decisions ---------------------------------------------------
    if receipt.policy_decisions:
        score += 5
        strengths.append("Policy decisions recorded")
    else:
        gaps.append("Policy decisions not recorded")

    # --- context quality ----------------------------------------------------
    if receipt.context_quality in ("Good", "Partial"):
        score += 5
        strengths.append(f"Context quality recorded ({receipt.context_quality.lower()})")
    elif receipt.context_quality == "Weak":
        gaps.append("Context quality recorded as weak")
        suggestions.append("Add more file evidence to improve context quality.")
    else:
        gaps.append("Context quality not recorded")

    # --- git state ----------------------------------------------------------
    if receipt.git_state is not None:
        score += 3
        strengths.append("Git state recorded")

    # --- approval -----------------------------------------------------------
    if receipt.approval_required:
        if receipt.approval_granted is True:
            score += 5
            strengths.append("Approval granted")
        elif receipt.approval_granted is False:
            score -= 25
            warnings_.append("Approval was denied")
            suggestions.append("Address approval blockers before re-running this task.")

    # --- adapter ------------------------------------------------------------
    is_adapter_run = receipt.adapter is not None
    if is_adapter_run:
        if receipt.adapter_available is True:
            score += 3
            strengths.append("Adapter metadata captured")
        elif receipt.adapter_available is False:
            score -= 10
            gaps.append("Adapter was unavailable")
            suggestions.append("Check adapter availability before running adapter tasks.")

    # --- schema version (structured receipt) --------------------------------
    if receipt.schema_version:
        score += 3
        strengths.append("Structured receipt recorded")
    else:
        gaps.append("Old receipt format - some fields may be missing")

    # --- evidence capsules --------------------------------------------------
    has_secret = _has_secret_scan_findings(receipt)
    if receipt.evidence_capsules:
        if not has_secret:
            score += 3
            strengths.append("Evidence capsules present")
        # secret findings are handled separately below

    # --- secret scan --------------------------------------------------------
    if has_secret:
        score -= 10
        warnings_.append("Secret scan findings detected")
        suggestions.append("Review secret scan warnings before sharing this context.")

    # --- high/critical findings ---------------------------------------------
    has_high_findings = _has_critical_or_high_findings(receipt)
    if has_high_findings:
        score -= 5
        warnings_.append("High-severity findings present")
        suggestions.append("Review high-severity findings before trusting this result.")

    # --- clamp and level ----------------------------------------------------
    score = max(0, min(100, score))
    level = _level(score)
    confidence = _compute_confidence(receipt)
    summary = _compute_summary(receipt, level)

    recommended = _pick_recommended_next_move(
        receipt, level, has_secret, has_high_findings, no_checks_write
    )

    return RunReflection(
        score=score,
        level=level,
        confidence=confidence,
        summary=summary,
        strengths=strengths[:_MAX_LIST],
        gaps=gaps[:_MAX_LIST],
        suggestions=suggestions[:_MAX_LIST],
        warnings=warnings_[:_MAX_LIST],
        recommended_next_move=recommended,
    )


# ---------------------------------------------------------------------------
# Renderer
# ---------------------------------------------------------------------------

def render_run_reflection(reflection: RunReflection) -> list[str]:
    """Return a list of plain-text lines for CLI display."""
    lines: list[str] = []

    lines.append("RUN REFLECTION")
    lines.append("")
    lines.append("Score")
    lines.append(f"  {reflection.score}/100 - {reflection.level}  (confidence: {reflection.confidence})")
    lines.append("")
    lines.append("Summary")
    lines.append(f"  {reflection.summary}")

    if reflection.strengths:
        lines.append("")
        lines.append("What went well")
        for item in reflection.strengths[:_MAX_LIST]:
            lines.append(f"  - {item}")

    if reflection.gaps:
        lines.append("")
        lines.append("What was missing")
        for item in reflection.gaps[:_MAX_LIST]:
            lines.append(f"  - {item}")

    if reflection.warnings:
        lines.append("")
        lines.append("Warnings")
        for item in reflection.warnings[:_MAX_LIST]:
            lines.append(f"  - {item}")

    if reflection.suggestions:
        lines.append("")
        lines.append("Suggestions")
        for item in reflection.suggestions[:_MAX_LIST]:
            lines.append(f"  - {item}")

    if reflection.recommended_next_move:
        lines.append("")
        lines.append("Recommended next move")
        lines.append(f"  {reflection.recommended_next_move}")

    lines.append("")
    lines.append("Advisory only")
    lines.append("  This reflection does not change execution behaviour.")

    return lines
