from __future__ import annotations

import json
import re
from dataclasses import dataclass

from openshard.config.settings import get_api_key, load_config
from openshard.providers.base import BaseProvider, ProviderError, UsageStats
from openshard.providers.openrouter import OpenRouterClient

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
    """Generate a structured execution plan for a task via a provider."""

    def __init__(self, provider: BaseProvider | None = None) -> None:
        config = load_config()
        tiers: list[dict] = config.get("model_tiers", [])

        # Priority: config.planning_model > mode_policy > model_tiers[0]
        self.model: str
        self._model_source: str

        if config.get("planning_model"):
            self.model = config["planning_model"]
            self._model_source = "config"
        else:
            _policy_model: str | None = None
            try:
                from openshard.models.mode_policy import model_policy_for_mode as _mpm
                _policy = _mpm("plan")
                if _policy is not None:
                    _policy_model = _policy.default_model_id
            except Exception:
                pass

            if _policy_model:
                self.model = _policy_model
                self._model_source = "mode_policy"
            elif tiers:
                self.model = tiers[0]["model"]
                self._model_source = "model_tiers"
            else:
                raise RuntimeError(
                    "No model_tiers defined in config.yml and no mode policy available"
                )

        self.client: BaseProvider = (
            provider if provider is not None else OpenRouterClient(get_api_key())
        )

    def generate(self, task: str) -> ExecutionPlan:
        """Call the model and return a parsed :class:`ExecutionPlan`."""
        response = self.client.execute(
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
            raise ProviderError(
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
