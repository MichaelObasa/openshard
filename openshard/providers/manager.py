from __future__ import annotations

import datetime
import os
import sys
import time
from dataclasses import dataclass

from openshard.providers.base import BaseProvider, ModelInfo, ProviderAuthError, ProviderError
from openshard.providers.cache import build_cache_entry, is_stale, load_cache, save_cache


@dataclass
class InventoryEntry:
    provider: str
    model: ModelInfo


@dataclass
class UnifiedInventory:
    models: list[InventoryEntry]
    generated_at: str
    provider_count: int


_PROVIDER_ENV: dict[str, str] = {
    "openrouter": "OPENROUTER_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
}


def _make_provider(name: str, key: str) -> BaseProvider:
    if name == "openrouter":
        from openshard.providers.openrouter import OpenRouterClient
        return OpenRouterClient(key)
    if name == "anthropic":
        from openshard.providers.anthropic import AnthropicProvider
        return AnthropicProvider(key)
    from openshard.providers.openai import OpenAIProvider
    return OpenAIProvider(key)


def _rehydrate(rows: list[dict]) -> list[ModelInfo]:
    return [
        ModelInfo(
            id=row["id"],
            name=row["name"],
            pricing=row.get("pricing", {}),
            context_window=row.get("context_window"),
            max_output_tokens=row.get("max_output_tokens"),
            supports_vision=row.get("supports_vision", False),
            supports_tools=row.get("supports_tools", False),
        )
        for row in rows
    ]


class ProviderManager:
    def __init__(self) -> None:
        self.providers: dict[str, BaseProvider] = {}
        for name, env_var in _PROVIDER_ENV.items():
            key = os.environ.get(env_var, "")
            if not key:
                continue
            try:
                self.providers[name] = _make_provider(name, key)
            except ProviderAuthError as exc:
                print(f"warning: {name} skipped ({exc})", file=sys.stderr)

    def get_inventory(self, refresh: bool = False) -> UnifiedInventory:
        cache = load_cache() or {"cached_at": 0.0, "models": {}}
        cache_fresh = not is_stale(cache.get("cached_at", 0.0))
        cached_models: dict = cache.get("models", {})

        entries: list[InventoryEntry] = []
        cache_updated = False

        for name, provider in self.providers.items():
            use_cache = not refresh and cache_fresh and name in cached_models
            if use_cache:
                models = _rehydrate(cached_models[name])
            else:
                try:
                    models = provider.list_models()
                    cached_models[name] = build_cache_entry(name, models)[name]
                    cache_updated = True
                except ProviderError as exc:
                    print(f"warning: {name} list_models failed ({exc})", file=sys.stderr)
                    continue

            for m in models:
                entries.append(InventoryEntry(provider=name, model=m))

        if cache_updated:
            cache["models"] = cached_models
            cache["cached_at"] = time.time()
            save_cache(cache)

        generated_at = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        return UnifiedInventory(
            models=entries,
            generated_at=generated_at,
            provider_count=len({e.provider for e in entries}),
        )
