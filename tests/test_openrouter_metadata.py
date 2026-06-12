from __future__ import annotations

import json
import sys
import unittest
import urllib.error
from pathlib import Path
from unittest.mock import MagicMock, patch

from click.testing import CliRunner

from openshard.cli.main import cli
from openshard.models.openrouter_fetcher import (
    OpenRouterCacheError,
    OpenRouterFetchError,
    fetch_openrouter_models,
    load_openrouter_cache,
    normalize_model,
    save_openrouter_cache,
)

_FULL_RAW = {
    "id": "anthropic/claude-opus-4.8",
    "canonical_slug": "claude-opus-4.8",
    "name": "Anthropic: Claude Opus 4.8",
    "created": 1749600000,
    "description": "A capable model.",
    "context_length": 200000,
    "architecture": {
        "modality": "text+image->text",
        "input_modalities": ["text", "image"],
        "output_modalities": ["text"],
        "tokenizer": "Claude",
        "instruct_type": None,
    },
    "pricing": {
        "prompt": "0.000015",
        "completion": "0.000075",
        "request": "0",
        "image": "0.0048",
        "audio": None,
        "input_cache_read": "0.0000015",
        "input_cache_write": "0.00001875",
        "web_search": None,
        "internal_reasoning": None,
    },
    "top_provider": {
        "context_length": 200000,
        "max_completion_tokens": 32000,
        "is_moderated": False,
    },
    "supported_parameters": ["temperature", "tools", "reasoning"],
    "per_request_limits": None,
    "knowledge_cutoff": "2025-01-01",
    "expiration_date": None,
    # fields that should be excluded from normalized output
    "benchmarks": {"design_arena": []},
    "links": {"details": "https://openrouter.ai/anthropic/claude-opus-4.8"},
    "hugging_face_id": None,
    "supported_voices": None,
    "default_parameters": {},
}


def _make_http_response(payload: dict) -> MagicMock:
    body = json.dumps(payload).encode()
    mock_resp = MagicMock()
    mock_resp.read.return_value = body
    mock_resp.__enter__ = lambda s: s
    mock_resp.__exit__ = MagicMock(return_value=False)
    return mock_resp


class TestFetchOpenRouterModels(unittest.TestCase):
    def test_fetch_valid_response(self):
        resp = _make_http_response({"data": [_FULL_RAW]})
        with patch("urllib.request.urlopen", return_value=resp):
            models = fetch_openrouter_models()
        self.assertEqual(len(models), 1)
        self.assertEqual(models[0]["id"], "anthropic/claude-opus-4.8")

    def test_fetch_invalid_json(self):
        mock_resp = MagicMock()
        mock_resp.read.return_value = b"not json{"
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)
        with patch("urllib.request.urlopen", return_value=mock_resp):
            with self.assertRaises(OpenRouterFetchError) as ctx:
                fetch_openrouter_models()
        self.assertIn("Invalid JSON", str(ctx.exception))

    def test_fetch_network_failure(self):
        with patch(
            "urllib.request.urlopen",
            side_effect=urllib.error.URLError("connection refused"),
        ):
            with self.assertRaises(OpenRouterFetchError) as ctx:
                fetch_openrouter_models()
        self.assertIn("Network error", str(ctx.exception))

    def test_fetch_missing_data_key(self):
        resp = _make_http_response({"models": []})
        with patch("urllib.request.urlopen", return_value=resp):
            with self.assertRaises(OpenRouterFetchError) as ctx:
                fetch_openrouter_models()
        self.assertIn("data", str(ctx.exception))


class TestNormalizeModel(unittest.TestCase):
    def test_normalize_full(self):
        result = normalize_model(_FULL_RAW)
        self.assertEqual(result["id"], "anthropic/claude-opus-4.8")
        self.assertEqual(result["context_length"], 200000)
        self.assertEqual(result["pricing"]["prompt"], "0.000015")
        self.assertEqual(result["architecture"]["tokenizer"], "Claude")
        self.assertEqual(result["top_provider"]["is_moderated"], False)
        self.assertEqual(result["supported_parameters"], ["temperature", "tools", "reasoning"])

    def test_normalize_excludes_benchmarks_links(self):
        result = normalize_model(_FULL_RAW)
        self.assertNotIn("benchmarks", result)
        self.assertNotIn("links", result)
        self.assertNotIn("hugging_face_id", result)
        self.assertNotIn("supported_voices", result)
        self.assertNotIn("default_parameters", result)

    def test_normalize_missing_optional_fields(self):
        result = normalize_model({"id": "some/model"})
        self.assertEqual(result["id"], "some/model")
        self.assertIsNone(result["context_length"])
        self.assertIsNone(result["architecture"])
        self.assertIsNone(result["pricing"])
        self.assertIsNone(result["top_provider"])

    def test_normalize_partial_nested(self):
        raw = {"id": "x/y", "architecture": {"modality": "text->text"}}
        result = normalize_model(raw)
        self.assertEqual(result["architecture"]["modality"], "text->text")
        self.assertIsNone(result["architecture"]["tokenizer"])


class TestCacheIO(unittest.TestCase):
    def test_save_and_load_roundtrip(self, tmp_path=None):
        import tempfile

        with tempfile.TemporaryDirectory() as td:
            cache_path = Path(td) / "openrouter-models.json"
            models = [normalize_model(_FULL_RAW)]
            save_openrouter_cache(models, path=cache_path)
            loaded = load_openrouter_cache(path=cache_path)
        self.assertIsNotNone(loaded)
        self.assertEqual(loaded["schema_version"], "1")
        self.assertEqual(loaded["model_count"], 1)
        self.assertEqual(loaded["models"][0]["id"], "anthropic/claude-opus-4.8")

    def test_load_cache_missing(self, tmp_path=None):
        import tempfile

        with tempfile.TemporaryDirectory() as td:
            missing = Path(td) / "does-not-exist.json"
            result = load_openrouter_cache(path=missing)
        self.assertIsNone(result)

    def test_load_cache_corrupt(self):
        import tempfile

        with tempfile.TemporaryDirectory() as td:
            corrupt = Path(td) / "openrouter-models.json"
            corrupt.write_text("{ bad json }", encoding="utf-8")
            with self.assertRaises(OpenRouterCacheError) as ctx:
                load_openrouter_cache(path=corrupt)
        self.assertIn("corrupt", str(ctx.exception).lower())

    def test_save_no_api_key_in_cache(self):
        import tempfile

        with tempfile.TemporaryDirectory() as td:
            cache_path = Path(td) / "openrouter-models.json"
            save_openrouter_cache([normalize_model(_FULL_RAW)], path=cache_path)
            content = cache_path.read_text(encoding="utf-8")
        # Ensure no env-style key or common key patterns appear
        self.assertNotIn("OPENROUTER_API_KEY", content)
        self.assertNotIn("api_key", content.lower())


class TestSyncCommand(unittest.TestCase):
    def test_sync_command_writes_cache(self):
        import tempfile

        runner = CliRunner()
        with tempfile.TemporaryDirectory() as td:
            cache_path = Path(td) / "openrouter-models.json"
            resp = _make_http_response({"data": [_FULL_RAW]})
            with patch("urllib.request.urlopen", return_value=resp), patch(
                "openshard.models.openrouter_fetcher._DEFAULT_CACHE_PATH", cache_path
            ):
                result = runner.invoke(cli, ["models", "sync-openrouter"])
            self.assertEqual(result.exit_code, 0, result.output)
            self.assertTrue(cache_path.exists())
            loaded = json.loads(cache_path.read_text())
            self.assertEqual(loaded["model_count"], 1)

    def test_sync_command_no_api_key_written(self):
        import tempfile

        runner = CliRunner()
        with tempfile.TemporaryDirectory() as td:
            cache_path = Path(td) / "openrouter-models.json"
            resp = _make_http_response({"data": [_FULL_RAW]})
            with patch("urllib.request.urlopen", return_value=resp), patch(
                "openshard.models.openrouter_fetcher._DEFAULT_CACHE_PATH", cache_path
            ):
                runner.invoke(cli, ["models", "sync-openrouter"])
            if cache_path.exists():
                content = cache_path.read_text()
                self.assertNotIn("OPENROUTER_API_KEY", content)

    def test_sync_command_network_error(self):
        runner = CliRunner()
        with patch(
            "urllib.request.urlopen",
            side_effect=urllib.error.URLError("timeout"),
        ):
            result = runner.invoke(cli, ["models", "sync-openrouter"])
        self.assertNotEqual(result.exit_code, 0)
        self.assertIn("Network error", result.output)


class TestCacheCommand(unittest.TestCase):
    def test_cache_command_present(self):
        import tempfile

        runner = CliRunner()
        with tempfile.TemporaryDirectory() as td:
            cache_path = Path(td) / "openrouter-models.json"
            save_openrouter_cache([normalize_model(_FULL_RAW)], path=cache_path)
            with patch(
                "openshard.models.openrouter_fetcher._DEFAULT_CACHE_PATH", cache_path
            ):
                result = runner.invoke(cli, ["models", "openrouter-cache"])
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertIn("present", result.output)
        self.assertIn("1", result.output)
        self.assertIn("anthropic/claude-opus-4.8", result.output)

    def test_cache_command_missing(self):
        import tempfile

        runner = CliRunner()
        with tempfile.TemporaryDirectory() as td:
            missing = Path(td) / "openrouter-models.json"
            with patch(
                "openshard.models.openrouter_fetcher._DEFAULT_CACHE_PATH", missing
            ):
                result = runner.invoke(cli, ["models", "openrouter-cache"])
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertIn("No OpenRouter cache found", result.output)
        self.assertIn("sync-openrouter", result.output)

    def test_cache_command_corrupt(self):
        import tempfile

        runner = CliRunner()
        with tempfile.TemporaryDirectory() as td:
            corrupt = Path(td) / "openrouter-models.json"
            corrupt.write_text("{ bad json }", encoding="utf-8")
            with patch(
                "openshard.models.openrouter_fetcher._DEFAULT_CACHE_PATH", corrupt
            ):
                result = runner.invoke(cli, ["models", "openrouter-cache"])
        self.assertNotEqual(result.exit_code, 0)
        self.assertIn("corrupt", result.output.lower())

    def test_cache_command_offline(self):
        """Cache inspection works without any network access."""
        import tempfile

        runner = CliRunner()
        with tempfile.TemporaryDirectory() as td:
            cache_path = Path(td) / "openrouter-models.json"
            save_openrouter_cache([normalize_model(_FULL_RAW)], path=cache_path)
            # No patch on urlopen — if it were called the test would still pass
            # but real network access during tests is undesirable.
            with patch(
                "openshard.models.openrouter_fetcher._DEFAULT_CACHE_PATH", cache_path
            ), patch("urllib.request.urlopen", side_effect=AssertionError("network called")):
                result = runner.invoke(cli, ["models", "openrouter-cache"])
        self.assertEqual(result.exit_code, 0, result.output)


class TestNoRoutingDependency(unittest.TestCase):
    def test_routing_engine_does_not_import_openrouter_fetcher(self):
        import importlib

        engine = importlib.import_module("openshard.routing.engine")
        # Confirm the fetcher module is not in sys.modules as a dependency of engine
        fetcher_key = "openshard.models.openrouter_fetcher"
        # Remove fetcher from sys.modules so we can test clean import
        sys.modules.pop(fetcher_key, None)
        # Re-import engine — if it pulls in the fetcher, the key will reappear
        importlib.reload(engine)
        self.assertNotIn(
            fetcher_key,
            sys.modules,
            "openshard.routing.engine must not import openrouter_fetcher",
        )


class TestExistingCommandsUnaffected(unittest.TestCase):
    def test_models_list_still_works(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["models", "list"])
        self.assertEqual(result.exit_code, 0, result.output)

    def test_models_show_still_works(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["models", "show", "anthropic/claude-sonnet-4.6"])
        self.assertEqual(result.exit_code, 0, result.output)


if __name__ == "__main__":
    unittest.main()
