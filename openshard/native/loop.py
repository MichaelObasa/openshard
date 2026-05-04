from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

NativeLoopPhase = Literal[
    "repo_context",
    "observation",
    "evidence",
    "plan",
    "backend_proof",
    "generation",
    "proposal",
    "write",
    "verification",
    "diff_review",
    "final_report",
]


@dataclass
class NativeLoopEvent:
    phase: str
    status: str = "completed"
    summary: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class NativeLoopTrace:
    events: list[NativeLoopEvent] = field(default_factory=list)

    def record(
        self,
        phase: str,
        *,
        status: str = "completed",
        summary: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.events.append(
            NativeLoopEvent(
                phase=phase,
                status=status,
                summary=summary,
                metadata=metadata or {},
            )
        )

    def phases(self) -> list[str]:
        return [event.phase for event in self.events]
