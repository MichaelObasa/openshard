from __future__ import annotations

import pytest

from openshard.providers.openrouter import compute_cost

_FAKE_CACHE = {
    "cached_at": 9999999999.0,
    "models": {
        "openrouter": [
            {
                "id": "openai/gpt-5.4-mini",
                "pricing": {"prompt": "0.00000075", "completion": "0.0000045"},
            },
            {
                "id": "bad/malformed",
                "pricing": {"prompt": "not-a-number", "completion": "0.001"},
            },
            {
                "id": "bad/empty-pricing",
                "pricing": {},
            },
        ]
    },
}


def test_known_model_uses_model_pricing():
    # anthropic/claude-sonnet-4.6 is in MODEL_PRICING — no cache access needed
    cost = compute_cost("anthropic/claude-sonnet-4.6", 1_000_000, 1_000_000)
    assert cost == pytest.approx(3.00 + 15.00)


def test_model_pricing_takes_priority_over_cache(monkeypatch):
    # Even with a cache patched to return different prices, MODEL_PRICING wins
    fake = {
        "cached_at": 0,
        "models": {
            "openrouter": [
                {"id": "anthropic/claude-sonnet-4.6", "pricing": {"prompt": "99", "completion": "99"}},
            ]
        },
    }
    monkeypatch.setattr("openshard.providers.openrouter.load_cache", lambda: fake)
    cost = compute_cost("anthropic/claude-sonnet-4.6", 1_000_000, 1_000_000)
    assert cost == pytest.approx(3.00 + 15.00)


def test_cache_fallback_computes_cost(monkeypatch):
    # openai/gpt-5.4-mini is absent from MODEL_PRICING but present in cache
    monkeypatch.setattr("openshard.providers.openrouter.load_cache", lambda: _FAKE_CACHE)
    cost = compute_cost("openai/gpt-5.4-mini", 1_000_000, 500_000)
    expected = 1_000_000 * 0.00000075 + 500_000 * 0.0000045
    assert cost == pytest.approx(expected)


def test_missing_model_returns_none(monkeypatch):
    monkeypatch.setattr("openshard.providers.openrouter.load_cache", lambda: _FAKE_CACHE)
    assert compute_cost("totally/unknown", 100, 100) is None


def test_no_cache_returns_none(monkeypatch):
    monkeypatch.setattr("openshard.providers.openrouter.load_cache", lambda: None)
    assert compute_cost("openai/gpt-5.4-mini", 100, 100) is None


def test_malformed_pricing_string_returns_none(monkeypatch):
    monkeypatch.setattr("openshard.providers.openrouter.load_cache", lambda: _FAKE_CACHE)
    assert compute_cost("bad/malformed", 100, 100) is None


def test_missing_pricing_keys_returns_none(monkeypatch):
    monkeypatch.setattr("openshard.providers.openrouter.load_cache", lambda: _FAKE_CACHE)
    assert compute_cost("bad/empty-pricing", 100, 100) is None
