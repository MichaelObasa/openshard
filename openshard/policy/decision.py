from __future__ import annotations

import datetime
import uuid
from dataclasses import dataclass

_SEVERITY_RANK: dict[str | None, int] = {
    None: 0,
    "info": 1,
    "low": 2,
    "medium": 3,
    "high": 4,
    "critical": 5,
}

_DECISION_RANK: dict[str, int] = {
    "not_applicable": -1,
    "allow": 0,
    "ask": 1,
    "deny": 2,
}

_MAX_FIELD_LEN = 200


def _safe_text(value: str | None) -> str | None:
    if value is None:
        return None
    cleaned = value.replace("\n", " ").replace("\r", " ").strip()
    return cleaned[:_MAX_FIELD_LEN]


def _now_iso() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


@dataclass
class PolicyDecision:
    decision_id: str
    action: str
    resource: str | None
    decision: str  # "allow" | "ask" | "deny" | "not_applicable"
    reason: str | None = None
    source: str | None = None  # "path_policy"|"secret_scan"|"approval_gate"|"validator"|"unknown"
    severity: str | None = None  # "critical"|"high"|"medium"|"low"|"info"|None
    approval_required: bool = False
    approval_granted: bool | None = None
    scope: str | None = None
    created_at: str | None = None


def make_allow(
    action: str,
    resource: str | None = None,
    reason: str | None = None,
    source: str | None = None,
    scope: str | None = None,
) -> PolicyDecision:
    return PolicyDecision(
        decision_id=str(uuid.uuid4()),
        action=action,
        resource=_safe_text(resource),
        decision="allow",
        reason=_safe_text(reason),
        source=source,
        severity=None,
        approval_required=False,
        approval_granted=None,
        scope=scope,
        created_at=_now_iso(),
    )


def make_ask(
    action: str,
    resource: str | None = None,
    reason: str | None = None,
    source: str | None = None,
    scope: str | None = None,
) -> PolicyDecision:
    return PolicyDecision(
        decision_id=str(uuid.uuid4()),
        action=action,
        resource=_safe_text(resource),
        decision="ask",
        reason=_safe_text(reason),
        source=source,
        severity=None,
        approval_required=True,
        approval_granted=None,
        scope=scope,
        created_at=_now_iso(),
    )


def make_deny(
    action: str,
    resource: str | None = None,
    reason: str | None = None,
    source: str | None = None,
    severity: str | None = None,
    scope: str | None = None,
) -> PolicyDecision:
    return PolicyDecision(
        decision_id=str(uuid.uuid4()),
        action=action,
        resource=_safe_text(resource),
        decision="deny",
        reason=_safe_text(reason),
        source=source,
        severity=severity,
        approval_required=False,
        approval_granted=None,
        scope=scope,
        created_at=_now_iso(),
    )


def resolve_policy_decisions(decisions: list[PolicyDecision]) -> PolicyDecision:
    """
    Deterministic resolution: deny > ask > allow > not_applicable.

    not_applicable entries are excluded before ranking. Among denies,
    highest severity wins. Tie-break: lexicographic ascending decision_id.
    """
    active = [d for d in decisions if d.decision != "not_applicable"]

    if not active:
        return PolicyDecision(
            decision_id="resolve-not-applicable",
            action="none",
            resource=None,
            decision="not_applicable",
            reason="no applicable policy decisions",
        )

    def _sort_key(d: PolicyDecision) -> tuple[int, int, str]:
        return (
            -_DECISION_RANK.get(d.decision, 0),
            -_SEVERITY_RANK.get(d.severity, 0),
            d.decision_id,
        )

    return sorted(active, key=_sort_key)[0]
