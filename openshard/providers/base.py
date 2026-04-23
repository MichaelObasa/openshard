from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


# ---------------------------------------------------------------------------
# Provider-neutral exceptions
# ---------------------------------------------------------------------------

class ProviderError(Exception):
    """Base error for all provider failures."""


class ProviderAuthError(ProviderError):
    """Raised when the API key is invalid or missing."""


class ProviderRateLimitError(ProviderError):
    """Raised when the API rate limit is exceeded."""


# ---------------------------------------------------------------------------
# Shared data types
# ---------------------------------------------------------------------------

@dataclass
class ModelInfo:
    id: str
    name: str
    pricing: dict  # {"prompt": str, "completion": str, ...} — varies by provider/model
    context_window: int | None = None
    max_output_tokens: int | None = None
    supports_vision: bool = False
    supports_tools: bool = False


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
# Abstract base provider
# ---------------------------------------------------------------------------

class BaseProvider(ABC):
    """Minimal interface that every provider must implement."""

    @abstractmethod
    def list_models(self) -> list[ModelInfo]:
        """Return all models available from this provider."""

    @abstractmethod
    def execute(
        self, model: str, prompt: str, system: str | None = None
    ) -> ChatResponse:
        """Send *prompt* to *model* and return a structured response."""

    @abstractmethod
    def get_model_info(self, model_id: str) -> ModelInfo | None:
        """Return info for a single model, or None if not found."""
