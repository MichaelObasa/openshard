from __future__ import annotations

import uuid

from openshard.native.context import (
    _MAX_REPEATED_BLOCKED_TOOL,
    _MAX_RETRY_COUNT,
    _MAX_STEP_EVENTS_RECORDED,
    NativeOSNLoopStep,
    NativeOSNLoopSummary,
    normalize_osn_stop_reason,
)

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
        self._summary = NativeOSNLoopSummary(
            enabled=True,
            mode="experimental",
            max_steps=11,
            loop_id=str(uuid.uuid4()),
        )

    def record_step(
        self,
        step_name: str,
        status: str,
        *,
        tool_name: str = "",
        target_label: str = "",
        reason: str = "",
        result_summary: str = "",
        blocked_reason: str = "",
        context_injected: bool = False,
        approval_required: bool = False,
        verification_status: str = "",
        warnings: list[str] | None = None,
    ) -> NativeOSNLoopStep:
        """Append a pipeline step. Enforces max_steps and _MAX_STEP_EVENTS_RECORDED."""
        # Hard safety cap across all recorded events
        if len(self._summary.steps) >= _MAX_STEP_EVENTS_RECORDED:
            if "step_events_cap_reached" not in self._summary.warnings:
                self._summary.warnings.append("step_events_cap_reached")
            return self._summary.steps[-1]

        # Already at max_steps capacity: add warning once, return last step
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
            self._summary.blocked_steps += 1
            return blocked

        step = NativeOSNLoopStep(
            step_index=len(self._summary.steps),
            step_name=step_name,
            status=status,
            tool_name=tool_name,
            target_label=target_label,
            reason=reason,
            result_summary=result_summary[:120],  # hard cap — no raw content
            blocked_reason=blocked_reason,
            context_injected=context_injected,
            approval_required=approval_required,
            verification_status=verification_status,
            warnings=warnings or [],
        )
        self._summary.steps.append(step)
        self._summary.steps_taken = len(self._summary.steps)

        # Update counters
        if step_name != "max_steps_exceeded":
            self._summary.attempted_steps += 1
        if status == "passed":
            self._summary.completed_steps += 1
        elif status == "failed":
            self._summary.failed_steps += 1
        elif status == "blocked":
            self._summary.blocked_steps += 1

        if tool_name:
            self._summary.tool_calls_attempted += 1
            if status == "passed":
                self._summary.tool_calls_completed += 1
            elif status == "blocked":
                self._summary.tool_calls_blocked += 1

        if approval_required:
            self._summary.approval_required = True

        # Warn when repeated blocked tool attempts hit the limit
        if self._summary.blocked_steps >= _MAX_REPEATED_BLOCKED_TOOL:
            warn = "repeated_blocked_tool_limit"
            if warn not in self._summary.warnings:
                self._summary.warnings.append(warn)

        return step

    def complete(
        self,
        *,
        stopped_reason: str = "completed",
        verification_status: str = "",
        retry_used: bool = False,
        retry_count: int = 0,
        approval_granted: bool = False,
    ) -> None:
        """Finalise the summary. Normalizes stopped_reason; sets all final fields."""
        _has_final_receipt = any(s.step_name == "final_receipt" for s in self._summary.steps)

        if "max steps reached" in self._summary.warnings and not _has_final_receipt:
            raw_reason = "max_steps"
        elif retry_count > _MAX_RETRY_COUNT:
            raw_reason = "retry_limit"
        else:
            raw_reason = stopped_reason

        self._summary.stopped_reason = normalize_osn_stop_reason(raw_reason)
        self._summary.completed = self._summary.stopped_reason == "completed"
        self._summary.verification_status = verification_status
        self._summary.retry_used = retry_used
        self._summary.retry_count = retry_count

        if approval_granted:
            self._summary.approval_granted = True

        # Verification fields
        if verification_status:
            self._summary.verification_attempted = True
            if verification_status == "passed":
                self._summary.verification_passed = True
            elif verification_status == "failed":
                self._summary.verification_passed = False
            # else None (skipped or unknown)

        # Final status - prefer explicit stopped_reason when not completed
        self._summary.final_status = (
            "completed" if self._summary.completed else self._summary.stopped_reason
        )

    @property
    def summary(self) -> NativeOSNLoopSummary:
        return self._summary
