from __future__ import annotations

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
        return NativeBackendResult(
            summary="deepagents backend available",
            notes=[
                "DeepAgents backend is detected but not yet allowed to mutate the repo.",
                "OpenShard still owns write, verification, diff review, and final reporting.",
            ],
            metadata={"backend": self.name, "available": True, "mode": "stub"},
        )


def get_backend(name: str) -> NativeAgentBackend:
    if name == "deepagents":
        return DeepAgentsNativeBackend()
    return BuiltinNativeBackend()
