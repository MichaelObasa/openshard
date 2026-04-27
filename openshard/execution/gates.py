from __future__ import annotations

from dataclasses import dataclass
from typing import List

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
    def __init__(self, approval_mode: str, risky_paths: List[str] | None, cost_threshold: float):
        self.approval_mode = approval_mode
        self.risky_paths = set(risky_paths or [])
        self.cost_threshold = cost_threshold

    def check_file_write(self, files: List[str]) -> GateDecision:
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

    def check_risky_paths(self, files: List[str]) -> GateDecision:
        if self.approval_mode in ("ask", "smart") and self.risky_paths:
            hits = [f for f in files if _risky_match(f, self.risky_paths)]
            if hits:
                return GateDecision(True, f"Risky paths detected: {', '.join(hits[:3])}")
        return GateDecision(False, "")

    def check_stack_mismatch(self, mismatches: List[str]) -> GateDecision:
        if mismatches:
            return GateDecision(True, f"Stack mismatch: {', '.join(mismatches[:3])}")
        return GateDecision(False, "")
