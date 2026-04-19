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


# ---------------------------------------------------------------------------
# Pricing snapshot
# ---------------------------------------------------------------------------

# Dollars per million tokens — (prompt, completion).  Updated 2026-04.
# Prices marked ~est are approximate; verify current rates at openrouter.ai/models.
MODEL_PRICING: dict[str, tuple[float, float]] = {
    # Anthropic
    "anthropic/claude-haiku-4.5":           (0.80,   4.00),
    "anthropic/claude-haiku-4.5-20251001":  (0.80,   4.00),
    "anthropic/claude-sonnet-4.6":          (3.00,  15.00),
    "anthropic/claude-opus-4.6":            (15.00, 75.00),
    "anthropic/claude-opus-4.7":            (15.00, 75.00),   # ~est
    # Main worker
    "z-ai/glm-5.1":                         (0.10,   0.10),   # ~est
    # Cheap coding
    "deepseek/deepseek-v3.2":              (0.14,   0.28),   # ~est
    # Visual / multimodal
    "moonshotai/kimi-k2.5":                 (0.45,   2.20),
    # Long-horizon
    "minimax/m2.7":                         (0.20,   1.10),   # ~est
    # Tiny helpers
    "openai/gpt-5.4-nano":                  (0.10,   0.40),   # ~est
}


def compute_cost(
    model: str, prompt_tokens: int, completion_tokens: int
) -> float | None:
    """Return estimated cost in USD from token counts, or None if model unknown."""
    pricing = MODEL_PRICING.get(model)
    if pricing is None:
        return None
    p_per_m, c_per_m = pricing
    return (prompt_tokens * p_per_m + completion_tokens * c_per_m) / 1_000_000


@dataclass
class UsageStats:
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    estimated_cost: float | None = None


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
        raw_cost = usage_raw.get("cost")
        resolved_model = data.get("model", model)
        estimated_cost = float(raw_cost) if raw_cost is not None else None
        usage = UsageStats(
            prompt_tokens=usage_raw.get("prompt_tokens", 0),
            completion_tokens=usage_raw.get("completion_tokens", 0),
            total_tokens=usage_raw.get("total_tokens", 0),
            estimated_cost=estimated_cost,
        )
        # Fallback: compute cost from token counts when provider omits it
        if usage.estimated_cost is None and usage.total_tokens > 0:
            usage.estimated_cost = compute_cost(
                resolved_model, usage.prompt_tokens, usage.completion_tokens
            )
        return ChatResponse(content=content, model=resolved_model, usage=usage)

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
