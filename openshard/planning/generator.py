from __future__ import annotations

import json
import re
from dataclasses import dataclass

from openshard.config.settings import get_api_key, load_config
from openshard.providers.openrouter import OpenRouterClient, OpenRouterError, UsageStats

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are a task planning assistant. Respond with valid JSON only — no markdown, \
no prose, no code fences.

Return exactly this structure:
{
  "summary": "<one sentence description of the task>",
  "stages": [
    {
      "name": "<stage name>",
      "tier": "<strong|medium|cheap>",
      "reasoning": "<one sentence>"
    }
  ]
}

Tier definitions:
- strong: complex reasoning, architecture decisions, open-ended generation
- medium: standard implementation tasks, moderate complexity
- cheap: simple extraction, formatting, lookup, or boilerplate

Use as many stages as the task genuinely requires. Do not pad.\
"""

# ---------------------------------------------------------------------------
# Return types
# ---------------------------------------------------------------------------

_VALID_TIERS = {"strong", "medium", "cheap"}


@dataclass
class PlanStage:
    name: str
    tier: str       # "strong" | "medium" | "cheap"
    reasoning: str


@dataclass
class ExecutionPlan:
    summary: str
    stages: list[PlanStage]
    usage: UsageStats | None = None


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------

class PlanGenerator:
    """Generate a structured execution plan for a task via OpenRouter."""

    def __init__(self) -> None:
        config = load_config()
        tiers: list[dict] = config.get("model_tiers", [])
        if not tiers:
            raise RuntimeError("No model_tiers defined in config.yml")
        self.model: str = config.get("planning_model") or tiers[0]["model"]
        self.client = OpenRouterClient(get_api_key())

    def generate(self, task: str) -> ExecutionPlan:
        """Call the model and return a parsed :class:`ExecutionPlan`."""
        response = self.client.send_request(
            model=self.model,
            prompt=task,
            system=_SYSTEM_PROMPT,
        )
        plan = self._parse(response.content)
        plan.usage = response.usage
        return plan

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _parse(self, raw: str) -> ExecutionPlan:
        # Strip markdown code fences defensively — some models add them
        # even when told not to.
        text = re.sub(r"^```(?:json)?\s*", "", raw.strip())
        text = re.sub(r"\s*```$", "", text)

        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            raise OpenRouterError(
                f"Model returned invalid JSON ({exc}).\nRaw response:\n{raw}"
            ) from exc

        stages = [
            PlanStage(
                name=s.get("name", "Unnamed stage"),
                tier=s.get("tier", "medium") if s.get("tier") in _VALID_TIERS else "medium",
                reasoning=s.get("reasoning", ""),
            )
            for s in data.get("stages", [])
        ]

        return ExecutionPlan(
            summary=data.get("summary", ""),
            stages=stages,
        )
