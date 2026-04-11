from __future__ import annotations

from openshard.config.settings import get_api_key
from openshard.providers.openrouter import ChatResponse, OpenRouterClient
from openshard.routing.engine import RoutingEngine


class TaskRunner:
    """Execute a task against the model selected by :class:`RoutingEngine`."""

    def __init__(self) -> None:
        self.engine = RoutingEngine()
        self.client = OpenRouterClient(get_api_key())

    def run(self, task: str) -> ChatResponse:
        """Route *task* to the appropriate model and return the response."""
        model = self.engine.select_model(task)
        return self.client.send_request(model, task)
