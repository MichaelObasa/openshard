from __future__ import annotations

import unittest

from openshard.cli.main import _build_model_line
from openshard.providers.base import ModelInfo
from openshard.providers.manager import InventoryEntry
from openshard.routing.engine import RoutingDecision
from openshard.scoring.filter import prefilter_coding
from openshard.scoring.requirements import TaskRequirements
from openshard.scoring.scorer import select_with_info


def _make_entry(model_id: str, provider: str = "openrouter", **kwargs) -> InventoryEntry:
    return InventoryEntry(
        provider=provider,
        model=ModelInfo(
            id=model_id,
            name=model_id,
            pricing=kwargs.get("pricing", {}),
            context_window=kwargs.get("context_window"),
            max_output_tokens=None,
            supports_vision=kwargs.get("supports_vision", False),
            supports_tools=kwargs.get("supports_tools", False),
        ),
    )


class TestSelectWithInfo(unittest.TestCase):

    def test_select_with_info_returns_winner(self):
        entry = _make_entry("openrouter/fast-model")
        reqs = TaskRequirements()
        result = select_with_info([entry], reqs, "standard")
        self.assertFalse(result.used_fallback)
        self.assertEqual(result.selected_model, "openrouter/fast-model")
        self.assertEqual(result.candidate_count, 1)

    def test_select_with_info_fallback_when_empty(self):
        entry = _make_entry("openrouter/no-vision", supports_vision=False)
        reqs = TaskRequirements(needs_vision=True)
        result = select_with_info([entry], reqs, "visual")
        self.assertTrue(result.used_fallback)
        self.assertIsNone(result.selected_model)
        self.assertIsNone(result.selected_provider)
        self.assertEqual(result.candidate_count, 0)

    def test_select_with_info_category_preserved(self):
        entry = _make_entry("openrouter/fast-model")
        reqs = TaskRequirements()
        result = select_with_info([entry], reqs, "security")
        self.assertEqual(result.category, "security")

    def test_select_with_info_provider_preserved(self):
        entry = _make_entry("claude-sonnet", provider="anthropic")
        reqs = TaskRequirements()
        result = select_with_info([entry], reqs, "standard")
        self.assertEqual(result.selected_provider, "anthropic")
        self.assertEqual(result.selected_model, "claude-sonnet")


class TestPrefilterCoding(unittest.TestCase):

    def test_embedding_model_excluded(self):
        entry = _make_entry("openai/text-embedding-3-small")
        self.assertEqual(prefilter_coding([entry]), [])

    def test_whisper_model_excluded(self):
        entry = _make_entry("openai/whisper-1")
        self.assertEqual(prefilter_coding([entry]), [])

    def test_dalle_model_excluded(self):
        entry = _make_entry("openai/dall-e-3")
        self.assertEqual(prefilter_coding([entry]), [])

    def test_tts_model_excluded(self):
        entry = _make_entry("openai/tts-1")
        self.assertEqual(prefilter_coding([entry]), [])

    def test_image_path_excluded(self):
        entry = _make_entry("somevendor/model/image")
        self.assertEqual(prefilter_coding([entry]), [])

    def test_image_variant_excluded(self):
        entry = _make_entry("gpt-5.4-image")
        self.assertEqual(prefilter_coding([entry]), [])

    def test_vision_model_excluded(self):
        entry = _make_entry("gemini-vision")
        self.assertEqual(prefilter_coding([entry]), [])

    def test_claude_passes_through(self):
        entry = _make_entry("claude-sonnet-4.6")
        self.assertEqual(prefilter_coding([entry]), [entry])

    def test_coding_model_passes_through(self):
        entry = _make_entry("openrouter/fast-model")
        self.assertEqual(prefilter_coding([entry]), [entry])


class TestModelLineAlignment(unittest.TestCase):

    def _decision(self, model: str) -> RoutingDecision:
        return RoutingDecision(model=model, category="standard", rationale="standard feature implementation")

    def test_model_line_uses_routed_model_when_provided(self):
        decision = self._decision("z-ai/glm-5.1")
        line = _build_model_line(decision, [], model="anthropic/claude-opus-4.7")
        self.assertIn("Opus", line)
        self.assertNotIn("GLM", line)

    def test_model_line_falls_back_to_routing_decision_when_no_model(self):
        decision = self._decision("z-ai/glm-5.1")
        line = _build_model_line(decision, [], model=None)
        self.assertIn("GLM", line)

    def test_model_line_keyword_and_scored_cannot_diverge(self):
        decision = self._decision("z-ai/glm-5.1")
        scored_winner = "anthropic/claude-sonnet-4.6"
        line = _build_model_line(decision, [], model=scored_winner)
        self.assertNotEqual(line, _build_model_line(decision, [], model=None))


class TestWinnerCost(unittest.TestCase):

    def test_select_with_info_returns_cost_when_present(self):
        # $0.10/M = 0.0000001 per token; expect per-million value back
        entry = _make_entry("openrouter/fast-model", pricing={"prompt": "0.0000001"})
        reqs = TaskRequirements()
        result = select_with_info([entry], reqs, "standard")
        self.assertAlmostEqual(result.selected_cost_per_m, 0.10)

    def test_select_with_info_cost_none_when_pricing_missing(self):
        entry = _make_entry("openrouter/no-price")
        reqs = TaskRequirements()
        result = select_with_info([entry], reqs, "standard")
        self.assertIsNone(result.selected_cost_per_m)

    def test_select_with_info_cost_none_when_pricing_unparseable(self):
        entry = _make_entry("openrouter/weird", pricing={"prompt": "n/a"})
        reqs = TaskRequirements()
        result = select_with_info([entry], reqs, "standard")
        self.assertIsNone(result.selected_cost_per_m)


if __name__ == "__main__":
    unittest.main()
