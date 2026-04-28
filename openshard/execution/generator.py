from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from openshard.config.settings import get_api_key, load_config
from openshard.providers.base import BaseProvider, ProviderError, UsageStats
from openshard.providers.openrouter import OpenRouterClient

if TYPE_CHECKING:
    from openshard.analysis.repo import RepoFacts

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
# Stack guard
# ---------------------------------------------------------------------------

# Extensions that are always allowed regardless of repo language.
_DOC_EXTS: frozenset[str] = frozenset({
    ".md", ".rst", ".txt", ".json", ".yml", ".yaml", ".toml", ".cfg",
    ".ini", ".lock", ".gitignore", ".env", ".sh", ".bat", ".html",
    ".css", ".scss", ".svg", ".xml", ".dockerfile",
})

# Extensions whose presence implies a specific language.
_EXT_TO_LANG: dict[str, str] = {
    ".py": "python",
    ".js": "javascript",
    ".jsx": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".rs": "rust",
    ".go": "go",
    ".java": "java",
    ".rb": "ruby",
    ".php": "php",
    ".cs": "csharp",
    ".cpp": "cpp",
    ".c": "c",
    ".swift": "swift",
    ".kt": "kotlin",
}


def check_stack_mismatch(files: list[ChangedFile], repo_facts: RepoFacts) -> list[str]:
    """Return paths of generated files whose language is absent from the repo stack."""
    if not repo_facts.languages:
        return []
    repo_langs = set(repo_facts.languages)
    mismatches: list[str] = []
    for f in files:
        ext = Path(f.path).suffix.lower()
        if not ext or ext in _DOC_EXTS:
            continue
        lang = _EXT_TO_LANG.get(ext)
        if lang and lang not in repo_langs:
            mismatches.append(f.path)
    return mismatches


def _build_repo_context(repo_facts: RepoFacts) -> str:
    """Format repo facts into a short context block for the task prompt."""
    parts: list[str] = []
    if repo_facts.languages:
        parts.append(f"Languages: {', '.join(repo_facts.languages)}")
    if repo_facts.test_command:
        parts.append(f"Test command: {repo_facts.test_command}")
    if repo_facts.framework:
        parts.append(f"Framework: {repo_facts.framework}")
    if repo_facts.package_files:
        parts.append(f"Package files: {', '.join(repo_facts.package_files)}")
    if not parts:
        return ""
    lines = ["Repo context:"] + [f"  {p}" for p in parts]
    lines.append("Generate files that match this repo's stack and language(s).")
    return "\n".join(lines)


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
    """Generate a structured execution result for a task via a provider."""

    def __init__(self, provider: BaseProvider | None = None) -> None:
        config = load_config()
        self.model: str = self._resolve_model(config)
        self.fixer_model: str = self._resolve_fixer_model(config)
        self.client: BaseProvider = (
            provider if provider is not None else OpenRouterClient(get_api_key())
        )

    def generate(
        self,
        task: str,
        model: str | None = None,
        repo_facts: RepoFacts | None = None,
        skills_context: str = "",
    ) -> ExecutionResult:
        """Call the model and return a parsed :class:`ExecutionResult`.

        *model* overrides the default execution model when provided.
        *repo_facts* appends repo stack context to the prompt so the model
        generates files that match the detected language and test tooling.
        *skills_context* prepends matched skill hints before repo context.
        """
        prompt = task
        if repo_facts is not None:
            ctx = _build_repo_context(repo_facts)
            if ctx:
                prompt = f"{ctx}\n\n{task}"
        if skills_context:
            prompt = f"{skills_context}\n\n{prompt}"
        response = self.client.execute(
            model=model or self.model,
            prompt=prompt,
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
                raise ProviderError(
                    f"Model returned no extractable JSON object.\nRaw response:\n{raw}"
                )
            try:
                data = json.loads(extracted)
            except json.JSONDecodeError as exc:
                raise ProviderError(
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
