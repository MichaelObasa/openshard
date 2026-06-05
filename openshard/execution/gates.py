from __future__ import annotations

from dataclasses import dataclass

from openshard.policy.decision import PolicyDecision, resolve_policy_decisions

SAFE_COMMANDS = [
    "python -m pytest",
    "npm test",
    "cargo test",
    "go test",
    "bundle exec rspec",
    "mvn test",
]

VALID_APPROVAL_MODES = {"auto", "smart", "ask"}


@dataclass
class GateDecision:
    required: bool
    reason: str


def _normalize(path: str) -> str:
    return path.replace("\\", "/")


def _risky_match(file_path: str, risky_paths: set[str]) -> bool:
    """Return True if file_path matches any risky path.

    Matches if: equal, risky contained in file, or file contained in risky.
    """
    fp = _normalize(file_path)
    for rp in risky_paths:
        rp_norm = _normalize(rp)
        if fp == rp_norm or rp_norm in fp or fp in rp_norm:
            return True
    return False


class GateEvaluator:
    def __init__(self, approval_mode: str, risky_paths: list[str] | None, cost_threshold: float):
        self.approval_mode = approval_mode
        self.risky_paths = set(risky_paths or [])
        self.cost_threshold = cost_threshold

    def check_file_write(self, files: list[str]) -> GateDecision:
        if self.approval_mode == "ask":
            return GateDecision(True, f"File write requested: {', '.join(files[:3])}")
        if self.approval_mode == "smart" and self.risky_paths:
            hits = [f for f in files if _risky_match(f, self.risky_paths)]
            if hits:
                return GateDecision(True, f"Writing to risky paths: {', '.join(hits[:3])}")
        return GateDecision(False, "")

    def check_shell_command(self, cmd: str) -> GateDecision:
        if self.approval_mode == "ask":
            return GateDecision(True, f"Shell command: {cmd}")
        if self.approval_mode == "smart":
            if any(cmd.startswith(safe) for safe in SAFE_COMMANDS):
                return GateDecision(False, "")
            return GateDecision(True, f"Shell command not in whitelist: {cmd}")
        return GateDecision(False, "")

    def check_high_cost(self, estimate: float) -> GateDecision:
        if self.approval_mode == "ask":
            return GateDecision(True, f"Estimated cost ${estimate:.4f}")
        if self.approval_mode == "smart" and estimate > self.cost_threshold:
            return GateDecision(True, f"Estimated cost ${estimate:.4f} exceeds threshold ${self.cost_threshold:.2f}")
        return GateDecision(False, "")

    def check_risky_paths(self, files: list[str]) -> GateDecision:
        if self.approval_mode in ("ask", "smart") and self.risky_paths:
            hits = [f for f in files if _risky_match(f, self.risky_paths)]
            if hits:
                return GateDecision(True, f"Risky paths detected: {', '.join(hits[:3])}")
        return GateDecision(False, "")

    def check_stack_mismatch(self, mismatches: list[str]) -> GateDecision:
        if mismatches:
            return GateDecision(True, f"Stack mismatch: {', '.join(mismatches[:3])}")
        return GateDecision(False, "")


def resolve_gate_decisions(decisions: list[GateDecision]) -> GateDecision:
    """Combine multiple gate decisions through the canonical policy resolver's
    deny > ask > allow ordering instead of ad-hoc if/elif logic.

    ``required=True`` maps to ``"ask"``, ``required=False`` to ``"allow"``.
    Decisions are passed in priority order; index-based ``decision_id`` values
    (``gate-0000``, ``gate-0001``, ...) make the resolver's lexicographic
    tie-break preserve that input order, so equal-rank ties are deterministic.

    Returns a single GateDecision the caller can act on unchanged. The
    ``"deny"`` branch is future-safe only — ``GateDecision`` cannot express
    deny today, so gates never produce it.
    """
    if not decisions:
        return GateDecision(False, "")
    pds = [
        PolicyDecision(
            decision_id=f"gate-{i:04d}",
            action="gate",
            resource=None,
            decision="ask" if d.required else "allow",
            reason=d.reason or "",
        )
        for i, d in enumerate(decisions)
    ]
    resolved = resolve_policy_decisions(pds)
    required = resolved.decision in ("ask", "deny")
    return GateDecision(required, resolved.reason or "")
