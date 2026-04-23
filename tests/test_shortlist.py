from __future__ import annotations

import unittest

from openshard.providers.base import ModelInfo
from openshard.providers.manager import InventoryEntry
from openshard.scoring.shortlist import (
    build_shortlist,
    extract_version,
    is_alias,
    is_trusted_model,
)


def _entry(model_id: str, provider: str = "openrouter") -> InventoryEntry:
    return InventoryEntry(
        provider=provider,
        model=ModelInfo(id=model_id, name=model_id, pricing={}),
    )


class TestIsTrustedModel(unittest.TestCase):

    def test_trusted_plain(self):
        self.assertTrue(is_trusted_model("claude-sonnet-4.6"))

    def test_trusted_with_provider_prefix(self):
        self.assertTrue(is_trusted_model("anthropic/claude-opus-4.7"))

    def test_trusted_alias(self):
        self.assertTrue(is_trusted_model("~anthropic/claude-haiku-latest"))

    def test_untrusted(self):
        self.assertFalse(is_trusted_model("some-random-model-v2"))

    def test_untrusted_embedding(self):
        self.assertFalse(is_trusted_model("text-embedding-3-large"))


class TestIsAlias(unittest.TestCase):

    def test_alias(self):
        self.assertTrue(is_alias("~anthropic/claude-opus-latest"))

    def test_not_alias(self):
        self.assertFalse(is_alias("anthropic/claude-opus-4.7"))


class TestExtractVersion(unittest.TestCase):

    def test_simple_version(self):
        self.assertEqual(extract_version("claude-opus-4.7"), (4, 7))

    def test_minor_greater_than_nine(self):
        self.assertEqual(extract_version("claude-sonnet-4.20"), (4, 20))

    def test_no_version(self):
        self.assertIsNone(extract_version("claude-opus-latest"))

    def test_version_with_prefix(self):
        self.assertEqual(extract_version("anthropic/claude-haiku-4.5"), (4, 5))


class TestBuildShortlist(unittest.TestCase):

    def test_untrusted_excluded(self):
        entries = [_entry("random-model-v1"), _entry("claude-sonnet-4.6")]
        result = build_shortlist(entries)
        ids = [e.model.id for e in result]
        self.assertNotIn("random-model-v1", ids)
        self.assertIn("claude-sonnet-4.6", ids)

    def test_alias_dropped_when_versioned_exists(self):
        entries = [
            _entry("~anthropic/claude-opus-latest"),
            _entry("anthropic/claude-opus-4.7"),
        ]
        result = build_shortlist(entries)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].model.id, "anthropic/claude-opus-4.7")

    def test_alias_kept_when_no_versioned(self):
        entries = [_entry("~anthropic/claude-opus-latest")]
        result = build_shortlist(entries)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].model.id, "~anthropic/claude-opus-latest")

    def test_newest_version_selected(self):
        entries = [
            _entry("claude-opus-4.5"),
            _entry("claude-opus-4.7"),
            _entry("claude-opus-4.6"),
        ]
        result = build_shortlist(entries)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].model.id, "claude-opus-4.7")

    def test_multiple_families_preserved(self):
        entries = [
            _entry("claude-sonnet-4.6"),
            _entry("claude-haiku-4.5"),
        ]
        result = build_shortlist(entries)
        ids = {e.model.id for e in result}
        self.assertEqual(ids, {"claude-sonnet-4.6", "claude-haiku-4.5"})

    def test_empty_input_safe(self):
        self.assertEqual(build_shortlist([]), [])

    def test_versioned_beats_alias_same_family(self):
        entries = [
            _entry("~anthropic/claude-sonnet-latest"),
            _entry("anthropic/claude-sonnet-4.6"),
        ]
        result = build_shortlist(entries)
        self.assertEqual(result[0].model.id, "anthropic/claude-sonnet-4.6")
