from __future__ import annotations

import json
import re
from dataclasses import dataclass

from openshard.config.settings import get_api_key, load_config
from openshard.providers.openrouter import OpenRouterClient, OpenRouterError

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are a software execution assistant. Respond with valid JSON only — no markdown, \
no prose, no code fences.

Return exactly this structure:
{
  "summary": "<short description of what was done>",
  "files": [
    {
      "path": "<relative/file/path>",
      "change_type": "<create|update|delete>",
      "content": "<full file content for create/update, empty string for delete>",
      "summary": "<short explanation of the change>"
    }
  ],
  "notes": [
    "<short note 1>",
    "<short note 2>"
  ]
}

Rules:
- summary: one sentence, past tense, describes what was accomplished
- files: list every file that would be created, updated, or deleted
- content: complete file contents for create/update; empty string for delete
- notes: important follow-up actions or caveats; omit if none
- change_type must be exactly one of: create, update, delete\
"""

# ---------------------------------------------------------------------------
# Return types
# ---------------------------------------------------------------------------

_VALID_CHANGE_TYPES = {"create", "update", "delete"}


@dataclass
class ChangedFile:
    path: str
    change_type: str   # "create" | "update" | "delete"
    content: str
    summary: str


@dataclass
class ExecutionResult:
    summary: str
    files: list[ChangedFile]
    notes: list[str]


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------

class ExecutionGenerator:
    """Generate a structured execution result for a task via OpenRouter."""

    def __init__(self) -> None:
        config = load_config()
        self.model: str = self._resolve_model(config)
        self.client = OpenRouterClient(get_api_key())

    def generate(self, task: str) -> ExecutionResult:
        """Call the model and return a parsed :class:`ExecutionResult`."""
        response = self.client.send_request(
            model=self.model,
            prompt=task,
            system=_SYSTEM_PROMPT,
        )
        return self._parse(response.content)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_model(config: dict) -> str:
        tiers: list[dict] = config.get("model_tiers", [])
        if not tiers:
            raise RuntimeError("No model_tiers defined in config.yml")
        if config.get("execution_model"):
            return config["execution_model"]
        balanced = next((t for t in tiers if t.get("name") == "balanced"), None)
        return (balanced or tiers[0])["model"]

    def _parse(self, raw: str) -> ExecutionResult:
        text = re.sub(r"^```(?:json)?\s*", "", raw.strip())
        text = re.sub(r"\s*```$", "", text)

        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            raise OpenRouterError(
                f"Model returned invalid JSON ({exc}).\nRaw response:\n{raw}"
            ) from exc

        files = [
            ChangedFile(
                path=f.get("path", ""),
                change_type=f.get("change_type") if f.get("change_type") in _VALID_CHANGE_TYPES else "update",
                content=f.get("content", ""),
                summary=f.get("summary", ""),
            )
            for f in data.get("files", [])
        ]

        return ExecutionResult(
            summary=data.get("summary", ""),
            files=files,
            notes=[n for n in data.get("notes", []) if n],
        )
