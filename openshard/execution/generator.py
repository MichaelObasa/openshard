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
    usage: UsageStats | None = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_json_object(text: str) -> str | None:
    """Return the first top-level {...} substring from *text*, or None.

    Walks a brace counter from the first '{' so nested objects and braces
    inside string values are handled correctly.
    """
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    in_string = False
    escape_next = False
    for i, ch in enumerate(text[start:], start):
        if escape_next:
            escape_next = False
            continue
        if ch == "\\" and in_string:
            escape_next = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------

class ExecutionGenerator:
    """Generate a structured execution result for a task via OpenRouter."""

    def __init__(self) -> None:
        config = load_config()
        self.model: str = self._resolve_model(config)
        self.fixer_model: str = self._resolve_fixer_model(config)
        self.client = OpenRouterClient(get_api_key())

    def generate(self, task: str, model: str | None = None) -> ExecutionResult:
        """Call the model and return a parsed :class:`ExecutionResult`.

        *model* overrides the default execution model when provided.
        """
        response = self.client.send_request(
            model=model or self.model,
            prompt=task,
            system=_SYSTEM_PROMPT,
        )
        result = self._parse(response.content)
        result.usage = response.usage
        return result

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

    @staticmethod
    def _resolve_fixer_model(config: dict) -> str:
        tiers: list[dict] = config.get("model_tiers", [])
        if not tiers:
            raise RuntimeError("No model_tiers defined in config.yml")
        if config.get("fixer_model"):
            return config["fixer_model"]
        powerful = next((t for t in tiers if t.get("name") == "powerful"), None)
        balanced = next((t for t in tiers if t.get("name") == "balanced"), None)
        return (powerful or balanced or tiers[0])["model"]

    def _parse(self, raw: str) -> ExecutionResult:
        text = re.sub(r"^```(?:json)?\s*", "", raw.strip())
        text = re.sub(r"\s*```$", "", text)

        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            extracted = _extract_json_object(raw)
            if extracted is None:
                raise OpenRouterError(
                    f"Model returned no extractable JSON object.\nRaw response:\n{raw}"
                )
            try:
                data = json.loads(extracted)
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
