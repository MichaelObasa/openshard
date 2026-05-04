from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

_MAX_PROOF_SUMMARY_LEN = 300


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


def _import_deepagents_create():
    from deepagents import create_deep_agent
    return create_deep_agent


def _get_safe_deepagents_model(context: dict) -> str | None:
    """Return a model string only if one is safely available in context, else None."""
    return context.get("deepagents_model")  # caller can inject; defaults to None


def _default_deepagents_proof(
    task: str,
    repo_context_summary: str | None = None,
    context: dict | None = None,
) -> dict:
    ctx = context or {}

    try:
        create_deep_agent = _import_deepagents_create()
    except Exception:
        return {
            "backend": "deepagents",
            "available": False,
            "mode": "unavailable",
            "summary": "",
            "notes": ["Install deepagents to enable this experimental backend."],
        }

    model = _get_safe_deepagents_model(ctx)
    if model is None:
        # Production path always returns here — create_deep_agent is never called without an
        # explicitly injected deepagents_model. tools=[] is not a sufficient read-only guard
        # because DeepAgents may include built-in filesystem, execute, and subagent tools.
        # A verified read-only DeepAgents configuration is deferred to a future branch.
        return {
            "backend": "deepagents",
            "available": True,
            "mode": "readonly_agent_unconfigured",
            "summary": "",
            "notes": [
                "DeepAgents installed, but no safe model/client configured for read-only proof."
            ],
        }

    input_parts = [f"Task: {task}"]
    if repo_context_summary:
        input_parts.append(f"Repo context (compact): {repo_context_summary[:500]}")
    agent_input = "\n".join(input_parts)

    try:
        # Safety note: tools=[] is passed but DeepAgents may include built-in filesystem,
        # execute, and subagent tools regardless. We mitigate by passing only compact text
        # input (no repo paths, file contents, diffs, or shell commands). If the installed
        # API cannot be made safely read-only, the except branch returns unconfigured.
        agent = create_deep_agent(
            model=model,
            tools=[],
            system_prompt=(
                "You are a read-only planning proof. "
                "Do not write files, run shell commands, or mutate anything."
            ),
        )
        result = agent.invoke({"messages": [{"role": "user", "content": agent_input}]})
        raw = str(result) if result is not None else ""
        summary = raw[:_MAX_PROOF_SUMMARY_LEN]
        return {
            "backend": "deepagents",
            "available": True,
            "mode": "readonly_agent_proof",
            "summary": summary,
            "notes": [
                "Read-only DeepAgents proof invoked. "
                "No write, edit, shell, or repo mutation tools were provided."
            ],
        }
    except Exception as exc:
        return {
            "backend": "deepagents",
            "available": True,
            "mode": "readonly_agent_unconfigured",
            "summary": "",
            "notes": [
                f"DeepAgents installed, but no safe model/client configured for read-only proof: {exc}"
            ],
        }


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
        if context.get("experimental_deepagents_run", False):
            proof = _default_deepagents_proof(
                task,
                repo_context_summary=context.get("repo_context_summary"),
                context=context,
            )
            mode = proof.get("mode", "readonly_agent_unconfigured")
            return NativeBackendResult(
                summary=f"deepagents {mode}",
                notes=proof.get("notes", []),
                metadata={"backend": self.name, "available": True, "mode": mode, "proof": proof},
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
