from __future__ import annotations

from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

_EXECUTOR_LABELS: dict[str, str] = {
    "native": "OpenShard Native",
    "opencode": "OpenCode",
    "direct": "Local Plan",
    "staged": "Staged Plan",
}

_EVIDENCE_FIT: dict[str, str] = {
    "native": "strong",
    "opencode": "adequate",
    "direct": "none",
    "staged": "none",
}

_COST_CLASS: dict[str, str] = {
    "native": "local",
    "opencode": "provider",
    "direct": "free",
    "staged": "free",
}

_HIGH_RISK_CATEGORIES: frozenset[str] = frozenset(
    {"security", "auth", "payments", "infra", "migration", "infrastructure"}
)


@dataclass
class AdvisoryCandidate:
    executor: str
    label: str
    available: bool
    score: float          # 0.0–100.0
    reasons: list[str]
    warnings: list[str]
    cost_class: str | None
    risk_fit: str | None  # "strong" | "adequate" | "weak"
    evidence_fit: str | None  # "strong" | "adequate" | "none"


@dataclass
class ExecutorAdvisoryResult:
    recommended: AdvisoryCandidate
    alternatives: list[AdvisoryCandidate]  # sorted descending by score
    warnings: list[str]
    version: str = "executor_advisory_v2"
    advisory_only: bool = True


# ---------------------------------------------------------------------------
# Scoring — pure, deterministic, no I/O
# ---------------------------------------------------------------------------

_BASE_SCORES: dict[str, float] = {
    "native": 70.0,
    "opencode": 55.0,
    "direct": 50.0,
    "staged": 45.0,
}


def _score_native(
    category: str,
    read_only: bool,
    risky_paths: list[str],
    write_task: bool,
) -> tuple[float, list[str], list[str]]:
    score = _BASE_SCORES["native"]
    reasons: list[str] = ["default executor", "supports receipts, checks, policy"]
    warnings: list[str] = []

    if write_task:
        score += 10.0
        reasons.append("best fit for write task")
    if category in _HIGH_RISK_CATEGORIES:
        score += 10.0
        reasons.append(f"enforces policy checks for {category} task")
    if risky_paths:
        score += 8.0
        warnings.append(f"write touches {len(risky_paths)} risky path(s); review recommended")
    if category == "complex":
        score += 8.0
        reasons.append("handles multi-file tasks well")
    if read_only:
        # Native is fine read-only but less differentiated
        score -= 5.0

    return min(score, 100.0), reasons, warnings


def _score_opencode(
    category: str,
    read_only: bool,
    available: bool,
    opencode_preference: bool,
) -> tuple[float, list[str], list[str]]:
    score = _BASE_SCORES["opencode"]
    reasons: list[str] = ["external agent executor"]
    warnings: list[str] = []

    if not available:
        score -= 20.0
        warnings.append("opencode not found on PATH; install to use this path")
    else:
        reasons.append("receipt metadata supported")

    if opencode_preference and available:
        score += 30.0
        reasons.append("explicit user preference")

    if read_only:
        score += 5.0
        reasons.append("suitable for read-only tasks")

    # OpenCode cannot outrank Native by default — enforce a ceiling unless
    # the user has an explicit preference and the task is not high-risk.
    if not opencode_preference:
        score = min(score, _BASE_SCORES["native"] - 1.0)

    if category in _HIGH_RISK_CATEGORIES:
        score -= 5.0
        warnings.append(f"{category} task: prefer paths with stronger receipt evidence")

    return max(0.0, min(score, 100.0)), reasons, warnings


def _score_direct(
    category: str,
    read_only: bool,
) -> tuple[float, list[str], list[str]]:
    score = _BASE_SCORES["direct"]
    reasons: list[str] = ["safe read-only path", "no provider calls", "free to run"]
    warnings: list[str] = []

    if read_only:
        score += 15.0
        reasons.append("ideal for read-only explanation or review")
    if category == "boilerplate":
        score += 10.0
        reasons.append("low complexity task")
    if category in _HIGH_RISK_CATEGORIES:
        score -= 10.0
        warnings.append("direct plan lacks receipt evidence for high-risk tasks")

    return max(0.0, min(score, 100.0)), reasons, warnings


def _score_staged(
    category: str,
    read_only: bool,
    write_task: bool,
) -> tuple[float, list[str], list[str]]:
    score = _BASE_SCORES["staged"]
    reasons: list[str] = ["two-phase plan then execute"]
    warnings: list[str] = []

    if write_task:
        score += 5.0
        reasons.append("suitable for write tasks")
    if category == "complex":
        score += 10.0
        reasons.append("plan-first approach for multi-file tasks")
    if category in _HIGH_RISK_CATEGORIES:
        score += 10.0
        reasons.append("structured approach for risk-relevant tasks")
    if read_only:
        score -= 5.0

    return max(0.0, min(score, 100.0)), reasons, warnings


def _risk_fit(executor: str, category: str, read_only: bool) -> str:
    if executor == "native":
        if category in _HIGH_RISK_CATEGORIES:
            return "strong"
        return "adequate"
    if executor == "staged":
        if category in _HIGH_RISK_CATEGORIES:
            return "adequate"
        return "adequate"
    if executor == "opencode":
        if category in _HIGH_RISK_CATEGORIES:
            return "weak"
        return "adequate"
    # direct
    if read_only:
        return "adequate"
    return "weak"


# ---------------------------------------------------------------------------
# Public ranking function — pure, no I/O
# ---------------------------------------------------------------------------

def rank_executors(
    task: str,  # noqa: ARG001  — reserved for future text signals
    *,
    category: str = "standard",
    risk_level: str = "low",  # noqa: ARG001  — reserved; signals already fold into category
    read_only: bool = False,
    opencode_available: bool = False,
    opencode_preference: bool = False,
    risky_paths: list[str] | None = None,
    _for_dispatch: bool = False,
) -> ExecutorAdvisoryResult:
    """Return a deterministic executor ranking. Pure — no I/O, no external calls.

    Args:
        _for_dispatch: when ``True``, the caller intends to apply the
            recommended executor to actual dispatch. The returned
            ``advisory_only`` flag is set to ``False`` in that case,
            accurately reflecting that this ranking drove a real decision.
            Callers that only want the ranking for display should leave
            this ``False`` (default).
    """
    paths = risky_paths or []
    write_task = not read_only
    global_warnings: list[str] = []

    # Score each executor
    nat_score, nat_reasons, nat_warn = _score_native(category, read_only, paths, write_task)
    oc_score, oc_reasons, oc_warn = _score_opencode(category, read_only, opencode_available, opencode_preference)
    dir_score, dir_reasons, dir_warn = _score_direct(category, read_only)
    stg_score, stg_reasons, stg_warn = _score_staged(category, read_only, write_task)

    def _candidate(executor: str, score: float, reasons: list[str], warnings: list[str]) -> AdvisoryCandidate:
        available = True if executor != "opencode" else opencode_available
        return AdvisoryCandidate(
            executor=executor,
            label=_EXECUTOR_LABELS[executor],
            available=available,
            score=round(score, 1),
            reasons=reasons,
            warnings=warnings,
            cost_class=_COST_CLASS[executor],
            risk_fit=_risk_fit(executor, category, read_only),
            evidence_fit=_EVIDENCE_FIT[executor],
        )

    candidates = [
        _candidate("native", nat_score, nat_reasons, nat_warn),
        _candidate("opencode", oc_score, oc_reasons, oc_warn),
        _candidate("direct", dir_score, dir_reasons, dir_warn),
        _candidate("staged", stg_score, stg_reasons, stg_warn),
    ]

    # Collect any global warnings from all candidates
    for c in candidates:
        global_warnings.extend(c.warnings)
    # Deduplicate preserving order
    seen: set[str] = set()
    deduped: list[str] = []
    for w in global_warnings:
        if w not in seen:
            seen.add(w)
            deduped.append(w)

    # Sort descending by score; tie-break by executor name for determinism
    candidates.sort(key=lambda c: (-c.score, c.executor))

    recommended = candidates[0]
    alternatives = candidates[1:]

    return ExecutorAdvisoryResult(
        recommended=recommended,
        alternatives=alternatives,
        warnings=deduped,
        advisory_only=not _for_dispatch,
    )


# ---------------------------------------------------------------------------
# Compact renderer — safe strings only
# ---------------------------------------------------------------------------

def render_executor_advisory(result: ExecutorAdvisoryResult) -> list[str]:
    """Return compact human-readable lines. No raw prompts or internal data."""
    lines: list[str] = []
    lines.append("ADVISORY RANKING")
    lines.append("")

    rec = result.recommended
    lines.append("  Recommended")
    lines.append(f"    {rec.label} — {rec.score:.0f}/100")
    if rec.reasons:
        lines.append(f"    {'; '.join(rec.reasons)}")

    if result.alternatives:
        lines.append("")
        lines.append("  Alternatives")
        for alt in result.alternatives:
            avail = "yes" if alt.available else "no"
            lines.append(f"    {alt.label} — {alt.score:.0f}/100   available: {avail}")
            if alt.reasons:
                lines.append(f"      {'; '.join(alt.reasons[:2])}")

    global_warns = [w for w in result.warnings if w]
    if global_warns:
        lines.append("")
        lines.append("  Warnings")
        for w in global_warns:
            lines.append(f"    {w}")

    lines.append("")
    lines.append("  [advisory only — does not change execution defaults]")
    return lines
