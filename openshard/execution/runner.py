from __future__ import annotations

from openshard.providers.openrouter import ChatResponse, OpenRouterClient
from openshard.routing.engine import RoutingEngine


class TaskRunner:
    """Execute a task against the model selected by :class:`RoutingEngine`."""

    def __init__(self) -> None:
        self.engine = RoutingEngine()
        api_key: str = self.engine.config.get("openrouter_api_key", "")
        if not api_key:
            raise ValueError(
                "openrouter_api_key is not set in config.yml. "
                "Add your key or set OPENSHARD_CONFIG to a config file that contains it."
            )
        self.client = OpenRouterClient(api_key)

    def run(self, task: str) -> ChatResponse:
        """Route *task* to the appropriate model and return the response."""
        model = self.engine.select_model(task)
        return self.client.send_request(model, task)
