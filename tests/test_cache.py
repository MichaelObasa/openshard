from __future__ import annotations

import time

import openshard.providers.cache as cache_mod
from openshard.providers.base import ModelInfo
from openshard.providers.cache import (
    CACHE_TTL_HOURS,
    build_cache_entry,
    is_stale,
    load_cache,
    save_cache,
)


def test_is_stale_old_timestamp():
    old = time.time() - (CACHE_TTL_HOURS + 1) * 3600
    assert is_stale(old) is True


def test_is_stale_recent_timestamp():
    recent = time.time() - 60
    assert is_stale(recent) is False


def test_build_cache_entry_defaults():
    m = ModelInfo(id="x/y", name="Y", pricing={"prompt": "1"})
    result = build_cache_entry("testprovider", [m])
    assert "testprovider" in result
    rows = result["testprovider"]
    assert len(rows) == 1
    row = rows[0]
    assert row["id"] == "x/y"
    assert row["name"] == "Y"
    assert row["pricing"] == {"prompt": "1"}
    assert row["context_window"] is None
    assert row["max_output_tokens"] is None
    assert row["supports_vision"] is False
    assert row["supports_tools"] is False


def test_build_cache_entry_with_capabilities():
    m = ModelInfo(
        id="a/b", name="B", pricing={},
        context_window=128000, max_output_tokens=4096,
        supports_vision=True, supports_tools=True,
    )
    result = build_cache_entry("prov", [m])
    row = result["prov"][0]
    assert row["context_window"] == 128000
    assert row["max_output_tokens"] == 4096
    assert row["supports_vision"] is True
    assert row["supports_tools"] is True


def test_load_cache_missing_file(tmp_path, monkeypatch):
    monkeypatch.setattr(cache_mod, "CACHE_PATH", tmp_path / "nonexistent.json")
    assert load_cache() is None


def test_save_and_load_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr(cache_mod, "CACHE_PATH", tmp_path / "model_cache.json")
    data = {
        "cached_at": 1234567890.0,
        "models": {
            "openrouter": [{"id": "x/y", "name": "Y", "pricing": {}}],
        },
    }
    save_cache(data)
    result = load_cache()
    assert result == data
