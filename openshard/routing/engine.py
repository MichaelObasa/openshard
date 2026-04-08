from __future__ import annotations

from openshard.config.settings import load_config


class RoutingEngine:
    """Select the appropriate model tier for a given task."""

    def __init__(self) -> None:
        self.config = load_config()

    def select_model(self, task: str) -> str:
        """Return the model identifier best suited for *task*.

        Routing logic is intentionally minimal for now; replace with a
        real heuristic or classifier as the project matures.
        """
        tiers: list[dict] = self.config.get("model_tiers", [])
        if not tiers:
            raise RuntimeError("No model tiers defined in config.yml")
        # Default: return the first (highest-priority) tier.
        return tiers[0]["model"]
