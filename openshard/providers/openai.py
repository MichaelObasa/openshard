from __future__ import annotations

from openshard.providers.base import (
    BaseProvider,
    ChatResponse,
    ModelInfo,
    ProviderAuthError,
    ProviderError,
    ProviderRateLimitError,
    UsageStats,
)
from openshard.providers.openrouter import compute_cost

try:
    import openai as _sdk
except ImportError:
    _sdk = None  # type: ignore[assignment]

# Default model used when routing selects a non-OpenAI model.
# Expressed in OpenRouter-style so MODEL_PRICING lookup and _model_label work.
DEFAULT_MODEL = "openai/gpt-4o-mini"


def _normalize_model_id(model: str) -> str:
    """Strip the 'openai/' prefix so the bare ID is sent to the API.

    openai/gpt-4o-mini  ->  gpt-4o-mini
    gpt-4o-mini         ->  gpt-4o-mini  (already bare, pass through)
    """
    if model.startswith("openai/"):
        return model[len("openai/"):]
    return model


class OpenAIProvider(BaseProvider):
    """Direct OpenAI API provider using the official SDK."""

    def __init__(self, api_key: str) -> None:
        if _sdk is None:
            raise ProviderError(
                "The 'openai' package is required to use the OpenAI provider.\n"
                "Install it with:  pip install openai"
            )
        if not api_key:
            raise ProviderAuthError(
                "OPENAI_API_KEY is not set.\n"
                "Export it before running:\n\n"
                "  export OPENAI_API_KEY=your_key_here\n\n"
                "Obtain a key from https://platform.openai.com/api-keys"
            )
        self._client = _sdk.OpenAI(api_key=api_key)

    # ------------------------------------------------------------------
    # BaseProvider interface
    # ------------------------------------------------------------------

    def list_models(self) -> list[ModelInfo]:
        """Return all models available from the OpenAI API."""
        try:
            page = self._client.models.list()
            return [
                ModelInfo(
                    id=m.id,
                    name=m.id,
                    pricing={},
                )
                for m in page.data
            ]
        except _sdk.AuthenticationError as exc:
            raise ProviderAuthError(str(exc)) from exc
        except _sdk.RateLimitError as exc:
            raise ProviderRateLimitError(str(exc)) from exc
        except _sdk.APIError as exc:
            raise ProviderError(str(exc)) from exc

    def execute(
        self, model: str, prompt: str, system: str | None = None
    ) -> ChatResponse:
        """Send *prompt* to *model* via the OpenAI chat completions API."""
        native_model = _normalize_model_id(model)
        messages: list[dict] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        try:
            response = self._client.chat.completions.create(
                model=native_model,
                messages=messages,
            )
        except _sdk.AuthenticationError as exc:
            raise ProviderAuthError(str(exc)) from exc
        except _sdk.RateLimitError as exc:
            raise ProviderRateLimitError(str(exc)) from exc
        except _sdk.APIError as exc:
            raise ProviderError(str(exc)) from exc

        content = response.choices[0].message.content or "" if response.choices else ""
        resolved_model = response.model or native_model
        raw_usage = response.usage
        usage = UsageStats(
            prompt_tokens=raw_usage.prompt_tokens if raw_usage else 0,
            completion_tokens=raw_usage.completion_tokens if raw_usage else 0,
            total_tokens=raw_usage.total_tokens if raw_usage else 0,
        )
        or_model = f"openai/{native_model}"
        usage.estimated_cost = compute_cost(
            or_model, usage.prompt_tokens, usage.completion_tokens
        )
        return ChatResponse(content=content, model=resolved_model, usage=usage)

    def get_model_info(self, model_id: str) -> ModelInfo | None:
        """Return info for *model_id*, or None if not found."""
        native = _normalize_model_id(model_id)
        for m in self.list_models():
            if m.id == native or m.id == model_id:
                return m
        return None
