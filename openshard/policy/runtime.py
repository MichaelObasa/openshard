from __future__ import annotations

import dataclasses
import uuid

from openshard.policy.decision import PolicyDecision, _now_iso, _safe_text


def _dedup_key(d: dict) -> tuple:
    return (d.get("source"), d.get("action"), d.get("decision"), d.get("reason"))


def _dedup_decisions(existing: list[dict], new: list[dict]) -> list[dict]:
    seen = {_dedup_key(d) for d in existing}
    result = []
    for d in new:
        k = _dedup_key(d)
        if k not in seen:
            seen.add(k)
            result.append(d)
    return result


def _pd_asdict(pd: PolicyDecision) -> dict:
    return dataclasses.asdict(pd)


def build_runtime_policy_decisions(
    approval_request: dict | None = None,
    approval_receipt: dict | None = None,
    secret_scan_result: dict | None = None,
    validator_policy: dict | None = None,
    readonly: bool | None = None,
) -> list[dict]:
    """
    Build policy decision records from existing runtime metadata.
    Returns JSON-safe dicts. Never raises — bad inputs are silently skipped.
    Does not change runtime behaviour; only records what already happened.
    """
    decisions: list[dict] = []

    try:
        decisions.extend(_approval_decisions(approval_request, approval_receipt))
    except Exception:
        pass

    try:
        ro_decision = _readonly_decision(readonly, approval_request)
        if ro_decision is not None:
            decisions.append(ro_decision)
    except Exception:
        pass

    try:
        ss_decision = _secret_scan_decision(secret_scan_result)
        if ss_decision is not None:
            decisions.append(ss_decision)
    except Exception:
        pass

    try:
        vp_decision = _validator_policy_decision(validator_policy)
        if vp_decision is not None:
            decisions.append(vp_decision)
    except Exception:
        pass

    return decisions


def _approval_decisions(
    request: dict | None,
    receipt: dict | None,
) -> list[dict]:
    if not isinstance(request, dict):
        return []
    if not request.get("requires_approval"):
        return []

    action = _safe_text(str(request.get("action") or "write")) or "write"
    raw_reason = _safe_text(str(receipt.get("reason") or "")) if isinstance(receipt, dict) else None

    if receipt is None or not isinstance(receipt, dict):
        pd = PolicyDecision(
            decision_id=str(uuid.uuid4()),
            action=action,
            resource=None,
            decision="ask",
            reason="approval required",
            source="approval_gate",
            severity=None,
            approval_required=True,
            approval_granted=None,
            created_at=_now_iso(),
        )
        return [_pd_asdict(pd)]

    granted = bool(receipt.get("granted"))
    receipt_action = _safe_text(str(receipt.get("action") or action)) or action

    if granted:
        pd = PolicyDecision(
            decision_id=str(uuid.uuid4()),
            action=receipt_action,
            resource=None,
            decision="allow",
            reason=raw_reason or "approval granted",
            source="approval_gate",
            severity=None,
            approval_required=True,
            approval_granted=True,
            created_at=_now_iso(),
        )
    else:
        pd = PolicyDecision(
            decision_id=str(uuid.uuid4()),
            action=receipt_action,
            resource=None,
            decision="deny",
            reason=raw_reason or "approval denied",
            source="approval_gate",
            severity="high",
            approval_required=True,
            approval_granted=False,
            created_at=_now_iso(),
        )

    return [_pd_asdict(pd)]


def _readonly_decision(readonly: bool | None, approval_request: dict | None) -> dict | None:
    if not readonly:
        return None
    if isinstance(approval_request, dict) and approval_request.get("requires_approval"):
        return None
    pd = PolicyDecision(
        decision_id=str(uuid.uuid4()),
        action="read_only_review",
        resource=None,
        decision="allow",
        reason="read-only task; writes not requested",
        source="path_policy",
        severity=None,
        approval_required=False,
        approval_granted=None,
        created_at=_now_iso(),
    )
    return _pd_asdict(pd)


def _secret_scan_decision(secret_scan_result: dict | None) -> dict | None:
    if not isinstance(secret_scan_result, dict):
        return None
    findings = secret_scan_result.get("findings") or []
    if not isinstance(findings, list) or not findings:
        return None
    pd = PolicyDecision(
        decision_id=str(uuid.uuid4()),
        action="secret_scan_review",
        resource=None,
        decision="not_applicable",
        reason="secret-like values detected; recorded warning only",
        source="secret_scan",
        severity="high",
        approval_required=False,
        approval_granted=None,
        created_at=_now_iso(),
    )
    return _pd_asdict(pd)


def _validator_policy_decision(validator_policy: dict | None) -> dict | None:
    if not isinstance(validator_policy, dict):
        return None
    ran = validator_policy.get("run")
    if ran is None:
        return None
    raw_reason = _safe_text(str(validator_policy.get("reason") or "")) or ""
    if ran:
        pd = PolicyDecision(
            decision_id=str(uuid.uuid4()),
            action="validator_review",
            resource=None,
            decision="allow",
            reason="validator run approved",
            source="validator",
            severity=None,
            approval_required=False,
            approval_granted=None,
            created_at=_now_iso(),
        )
    else:
        skip_reason = f"validator skipped: {raw_reason}" if raw_reason else "validator skipped"
        pd = PolicyDecision(
            decision_id=str(uuid.uuid4()),
            action="validator_review",
            resource=None,
            decision="not_applicable",
            reason=_safe_text(skip_reason),
            source="validator",
            severity=None,
            approval_required=False,
            approval_granted=None,
            created_at=_now_iso(),
        )
    return _pd_asdict(pd)
