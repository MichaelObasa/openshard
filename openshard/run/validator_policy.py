from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ValidatorPolicyDecision:
    run: bool
    reason: str


def should_run_validator(
    *,
    has_validator_model: bool,
    dry_run: bool,
    can_dispatch: bool,
    tier_dispatch_applied: bool,
    readonly_task: bool,
    routing_category: str,
    execution_profile: str,
    workflow: str,
    risky_paths_count: int,
    verification_attempted: bool,
) -> ValidatorPolicyDecision:
    """Decide whether to run the validator stage for this task.

    readonly_task is an absolute skip — no run condition overrides it.
    """
    if not has_validator_model:
        return ValidatorPolicyDecision(run=False, reason="no validator model")
    if dry_run:
        return ValidatorPolicyDecision(run=False, reason="dry run")
    if not can_dispatch:
        return ValidatorPolicyDecision(run=False, reason="tier dispatch not enabled")
    if not tier_dispatch_applied:
        return ValidatorPolicyDecision(run=False, reason="tier dispatch not applied")
    if readonly_task:
        return ValidatorPolicyDecision(run=False, reason="read-only task")

    if routing_category in ("security", "complex"):
        return ValidatorPolicyDecision(run=True, reason="security/complex category")
    if execution_profile in ("native_deep", "native_swarm"):
        return ValidatorPolicyDecision(run=True, reason="deep/swarm profile")
    if risky_paths_count > 0:
        return ValidatorPolicyDecision(run=True, reason="risky paths detected")
    if verification_attempted:
        return ValidatorPolicyDecision(run=True, reason="verification attempted")
    if workflow == "staged":
        return ValidatorPolicyDecision(run=True, reason="staged write task")

    return ValidatorPolicyDecision(run=False, reason="simple/safe task")
