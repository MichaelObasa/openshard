from __future__ import annotations

from openshard.native.context import NativeOSNLoopStep, NativeOSNLoopSummary

_DISABLED_VALUES: frozenset[str] = frozenset({"off", "none", "disabled", "false", "0", "no", ""})


def should_enable_osn_recorder(native_loop: str | None) -> bool:
    """Return True only for the explicit experimental value."""
    return native_loop == "experimental"


class OsnLoopRecorder:
    """Records pipeline-level OSN loop steps into NativeOSNLoopSummary.

    Created only when native_loop == "experimental". All recording calls on
    NativeAgentExecutor are no-ops when the recorder is None.
    """

    def __init__(self) -> None:
        self._summary = NativeOSNLoopSummary(enabled=True, mode="experimental", max_steps=11)

    def record_step(
        self,
        step_name: str,
        status: str,
        *,
        tool_name: str = "",
        reason: str = "",
        result_summary: str = "",
        context_injected: bool = False,
        approval_required: bool = False,
        verification_status: str = "",
        warnings: list[str] | None = None,
    ) -> NativeOSNLoopStep:
        """Append a pipeline step. Enforces max_steps — len(steps) never exceeds it."""
        # Already at capacity: add warning once, return last step
        if len(self._summary.steps) >= self._summary.max_steps:
            if "max steps reached" not in self._summary.warnings:
                self._summary.warnings.append("max steps reached")
            return self._summary.steps[-1]

        # Last slot available but step is not final_receipt: occupy with blocked marker
        if (
            len(self._summary.steps) == self._summary.max_steps - 1
            and step_name != "final_receipt"
        ):
            if "max steps reached" not in self._summary.warnings:
                self._summary.warnings.append("max steps reached")
            blocked = NativeOSNLoopStep(
                step_index=len(self._summary.steps),
                step_name="max_steps_exceeded",
                status="blocked",
                warnings=["max steps reached"],
            )
            self._summary.steps.append(blocked)
            self._summary.steps_taken = len(self._summary.steps)
            return blocked

        step = NativeOSNLoopStep(
            step_index=len(self._summary.steps),
            step_name=step_name,
            status=status,
            tool_name=tool_name,
            reason=reason,
            result_summary=result_summary[:120],  # hard cap — no raw content
            context_injected=context_injected,
            approval_required=approval_required,
            verification_status=verification_status,
            warnings=warnings or [],
        )
        self._summary.steps.append(step)
        self._summary.steps_taken = len(self._summary.steps)
        if approval_required:
            self._summary.approval_required = True
        return step

    def complete(
        self,
        *,
        stopped_reason: str = "completed",
        verification_status: str = "",
        retry_used: bool = False,
        approval_granted: bool = False,
    ) -> None:
        """Finalise the summary. Overrides stopped_reason if max steps were hit without final_receipt."""
        _has_final_receipt = any(s.step_name == "final_receipt" for s in self._summary.steps)
        if "max steps reached" in self._summary.warnings and not _has_final_receipt:
            self._summary.completed = False
            self._summary.stopped_reason = "max steps reached"
        else:
            self._summary.completed = True
            self._summary.stopped_reason = stopped_reason
        self._summary.verification_status = verification_status
        self._summary.retry_used = retry_used
        if approval_granted:
            self._summary.approval_granted = True

    @property
    def summary(self) -> NativeOSNLoopSummary:
        return self._summary
