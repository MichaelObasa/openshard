from __future__ import annotations

import httpx

from openshard.providers.base import (
    BaseProvider,
    ProviderAuthError,
    ProviderError,
    ProviderRateLimitError,
)

# Re-export shared data types so existing imports from this module keep working.
from openshard.providers.base import ChatResponse, ModelInfo, UsageStats
from openshard.providers.cache import load_cache

__all__ = [
    "OpenRouterClient",
    "OpenRouterError",
    "AuthError",
    "RateLimitError",
    "ModelInfo",
    "UsageStats",
    "ChatResponse",
    "MODEL_PRICING",
    "compute_cost",
]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_BASE_URL = "https://openrouter.ai/api/v1"
_TIMEOUT = 60.0  # seconds


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class OpenRouterError(ProviderError):
    """Base error for all OpenRouter failures."""


class AuthError(ProviderAuthError, OpenRouterError):
    """Raised when the API key is invalid or missing (HTTP 401/403)."""


class RateLimitError(ProviderRateLimitError, OpenRouterError):
    """Raised when the API rate limit is exceeded (HTTP 429)."""


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
    "deepseek/deepseek-v4-flash":          (0.10,   0.28),   # ~est
    "deepseek/deepseek-v4-pro":            (0.27,   1.10),   # ~est
    # Visual / multimodal
    "moonshotai/kimi-k2.5":                 (0.45,   2.20),
    # Long-horizon
    "minimax/m2.7":                         (0.20,   1.10),   # ~est
    # OpenAI
    "openai/gpt-4o":                        (2.50,  10.00),
    "openai/gpt-4o-mini":                   (0.15,   0.60),
    # GPT-5 family
    "openai/gpt-5.5":                       (5.00,  30.00),   # ~est
    # Tiny helpers
    "openai/gpt-5.4-nano":                  (0.10,   0.40),   # ~est
}


def _cost_from_cache(
    model: str, prompt_tokens: int, completion_tokens: int
) -> float | None:
    """Return cost using per-token pricing from the local OpenRouter model cache."""
    cache = load_cache()
    if not cache:
        return None
    for entry in cache.get("models", {}).get("openrouter", []):
        if entry.get("id") == model:
            pricing = entry.get("pricing") or {}
            try:
                p = float(pricing["prompt"])
                c = float(pricing["completion"])
            except (KeyError, TypeError, ValueError):
                return None
            return prompt_tokens * p + completion_tokens * c
    return None


def compute_cost(
    model: str, prompt_tokens: int, completion_tokens: int
) -> float | None:
    """Return estimated cost in USD from token counts, or None if model unknown."""
    pricing = MODEL_PRICING.get(model)
    if pricing is not None:
        p_per_m, c_per_m = pricing
        return (prompt_tokens * p_per_m + completion_tokens * c_per_m) / 1_000_000
    return _cost_from_cache(model, prompt_tokens, completion_tokens)


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------

class OpenRouterClient(BaseProvider):
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
    # BaseProvider interface
    # ------------------------------------------------------------------

    def list_models(self) -> list[ModelInfo]:
        """Return all models available on OpenRouter."""
        data = self._get("/models")
        return [
            ModelInfo(
                id=m.get("id", ""),
                name=m.get("name", m.get("id", "")),
                pricing=m.get("pricing", {}),
                context_window=m.get("context_length"),
                max_output_tokens=(m.get("top_provider") or {}).get("max_completion_tokens"),
                supports_vision="image" in (
                    (m.get("architecture") or {}).get("modality") or ""
                ),
                supports_tools=bool(
                    m.get("supported_parameters")
                    and "tools" in m["supported_parameters"]
                ),
            )
            for m in data.get("data", [])
        ]

    def execute(
        self, model: str, prompt: str, system: str | None = None
    ) -> ChatResponse:
        """Send *prompt* to *model* and return a structured response."""
        return self.send_request(model, prompt, system)

    def get_model_info(self, model_id: str) -> ModelInfo | None:
        """Return info for *model_id*, or None if not listed."""
        for m in self.list_models():
            if m.id == model_id:
                return m
        return None

    # ------------------------------------------------------------------
    # Legacy method — kept for internal callers; prefer execute()
    # ------------------------------------------------------------------

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
