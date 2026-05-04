from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass
class NativeBackendResult:
    summary: str = ""
    notes: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


class NativeAgentBackend(Protocol):
    name: str

    def available(self) -> bool: ...

    def run(self, *, task: str, context: dict[str, Any]) -> NativeBackendResult: ...


class BuiltinNativeBackend:
    name = "builtin"

    def available(self) -> bool:
        return True

    def run(self, *, task: str, context: dict[str, Any]) -> NativeBackendResult:
        return NativeBackendResult(
            summary="builtin native backend selected",
            metadata={"backend": self.name},
        )


class DeepAgentsNativeBackend:
    name = "deepagents"

    def available(self) -> bool:
        try:
            import deepagents  # noqa: F401

            return True
        except Exception:
            return False

    def run(self, *, task: str, context: dict[str, Any]) -> NativeBackendResult:
        if not self.available():
            return NativeBackendResult(
                summary="deepagents backend unavailable",
                notes=["Install deepagents to enable this experimental backend."],
                metadata={"backend": self.name, "available": False},
            )
        if context.get("experimental_run"):
            _payload = {
                "task": task,
                "experimental_run": True,
                "repo_context_summary": context.get("repo_context_summary"),
                "native_backend": self.name,
            }
            _t0 = time.monotonic()
            proof = _deepagents_proof_fn(task, _payload)
            _duration_ms = int((time.monotonic() - _t0) * 1000)
            return NativeBackendResult(
                summary=proof.get("summary", "deepagents sandbox proof completed"),
                notes=list(proof.get("notes", [])),
                metadata={
                    "backend": self.name,
                    "available": True,
                    "mode": proof.get("mode", "sandbox_proof"),
                    "summary": proof.get("summary", ""),
                    "notes": list(proof.get("notes", [])),
                    "duration_ms": _duration_ms,
                },
            )
        return NativeBackendResult(
            summary="deepagents backend available",
            notes=[
                "DeepAgents backend is detected but not yet allowed to mutate the repo.",
                "OpenShard still owns write, verification, diff review, and final reporting.",
            ],
            metadata={"backend": self.name, "available": True, "mode": "stub"},
        )


def _default_deepagents_proof(task: str, context: dict[str, Any]) -> dict[str, Any]:
    return {
        "mode": "sandbox_proof",
        "summary": "deepagents sandbox proof completed",
        "notes": [],
    }


_deepagents_proof_fn = _default_deepagents_proof


def get_backend(name: str) -> NativeAgentBackend:
    if name == "deepagents":
        return DeepAgentsNativeBackend()
    return BuiltinNativeBackend()
