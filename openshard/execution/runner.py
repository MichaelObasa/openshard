from __future__ import annotations

from openshard.routing.engine import RoutingEngine


class TaskRunner:
    """Execute a task against the model selected by :class:`RoutingEngine`."""

    def __init__(self) -> None:
        self.engine = RoutingEngine()

    def run(self, task: str) -> str:
        """Route and execute *task*, returning the model's response."""
        model = self.engine.select_model(task)
        # TODO: wire up real model calls via httpx / SDK.
        return f"[stub] Would send {task!r} to model '{model}'"
