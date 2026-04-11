from __future__ import annotations

from dataclasses import dataclass

import httpx

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_BASE_URL = "https://openrouter.ai/api/v1"
_TIMEOUT = 60.0  # seconds


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class OpenRouterError(Exception):
    """Base error for all OpenRouter failures."""


class AuthError(OpenRouterError):
    """Raised when the API key is invalid or missing (HTTP 401/403)."""


class RateLimitError(OpenRouterError):
    """Raised when the API rate limit is exceeded (HTTP 429)."""


# ---------------------------------------------------------------------------
# Return types
# ---------------------------------------------------------------------------

@dataclass
class ModelInfo:
    id: str
    name: str
    pricing: dict  # {"prompt": str, "completion": str, ...} — varies by model


@dataclass
class UsageStats:
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


@dataclass
class ChatResponse:
    content: str
    model: str
    usage: UsageStats


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------

class OpenRouterClient:
    """Thin HTTP client for the OpenRouter API."""

    def __init__(self, api_key: str) -> None:
        if not api_key:
            raise ValueError("api_key must not be empty")
        self._client = httpx.Client(
            base_url=_BASE_URL,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            timeout=_TIMEOUT,
        )

    # ------------------------------------------------------------------
    # Public methods
    # ------------------------------------------------------------------

    def list_models(self) -> list[ModelInfo]:
        """Return all models available on OpenRouter."""
        data = self._get("/models")
        return [
            ModelInfo(
                id=m.get("id", ""),
                name=m.get("name", m.get("id", "")),
                pricing=m.get("pricing", {}),
            )
            for m in data.get("data", [])
        ]

    def send_request(
        self, model: str, prompt: str, system: str | None = None
    ) -> ChatResponse:
        """Send *prompt* to *model* and return a structured response.

        *system* is an optional system-role message prepended to the conversation.
        """
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        payload = {
            "model": model,
            "messages": messages,
        }
        data = self._post("/chat/completions", payload)

        choices = data.get("choices", [])
        if not choices:
            raise OpenRouterError("API returned no choices in response")

        content = choices[0].get("message", {}).get("content", "")
        usage_raw = data.get("usage", {})
        usage = UsageStats(
            prompt_tokens=usage_raw.get("prompt_tokens", 0),
            completion_tokens=usage_raw.get("completion_tokens", 0),
            total_tokens=usage_raw.get("total_tokens", 0),
        )
        return ChatResponse(content=content, model=data.get("model", model), usage=usage)

    def close(self) -> None:
        self._client.close()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get(self, path: str) -> dict:
        try:
            response = self._client.get(path)
        except httpx.RequestError as exc:
            raise OpenRouterError(f"Network error: {exc}") from exc
        return self._parse(response)

    def _post(self, path: str, payload: dict) -> dict:
        try:
            response = self._client.post(path, json=payload)
        except httpx.RequestError as exc:
            raise OpenRouterError(f"Network error: {exc}") from exc
        return self._parse(response)

    def _parse(self, response: httpx.Response) -> dict:
        status = response.status_code
        if status in (401, 403):
            raise AuthError(f"Authentication failed (HTTP {status})")
        if status == 429:
            raise RateLimitError("Rate limit exceeded — try again later")
        if status >= 400:
            try:
                detail = response.json().get("error", {}).get("message", response.text)
            except Exception:
                detail = response.text
            raise OpenRouterError(f"API error (HTTP {status}): {detail}")
        return response.json()
