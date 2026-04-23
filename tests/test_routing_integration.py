from __future__ import annotations

import unittest
from unittest.mock import MagicMock, call, patch

from click.testing import CliRunner

from openshard.cli.main import cli
from openshard.providers.base import ModelInfo
from openshard.providers.manager import InventoryEntry


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


def _fake_result():
    r = MagicMock()
    r.usage = None
    r.files = []
    r.summary = "done"
    r.notes = []
    return r


def _make_generator_mock():
    g = MagicMock()
    g.generate.return_value = _fake_result()
    g.model = "mock-default-model"
    g.fixer_model = "mock-fixer-model"
    return g


def _make_manager_mock(entries: list[InventoryEntry], provider_names: list[str]):
    m = MagicMock()
    inv = MagicMock()
    inv.models = entries
    m.get_inventory.return_value = inv
    m.providers = {p: MagicMock() for p in provider_names}
    return m


class TestScoredRoutingIntegration(unittest.TestCase):

    def _run(self, args: list[str], manager_mock, generator_mock):
        with patch("openshard.cli.main.ProviderManager", return_value=manager_mock), \
             patch("openshard.cli.main.ExecutionGenerator", return_value=generator_mock), \
             patch("openshard.cli.main.get_api_key", return_value="test-key"), \
             patch("openshard.cli.main._log_run"):
            runner = CliRunner()
            result = runner.invoke(cli, ["run"] + args)
        return result

    def test_scored_selection_used(self):
        """When inventory has a matching entry, its model ID reaches generate()."""
        task = "implement a feature"
        entry = _make_entry("openrouter/fast-model", pricing={"prompt": "0.0000005"})
        manager = _make_manager_mock([entry], ["openrouter"])
        generator = _make_generator_mock()

        self._run([task], manager, generator)

        generator.generate.assert_called_once_with(task, model="openrouter/fast-model")

    def test_fallback_when_no_candidate(self):
        """When the only inventory entry fails hard filter, routing decision model is used."""
        task = "add a ui component"
        # visual category → needs_vision=True; this entry has supports_vision=False
        entry = _make_entry("openrouter/no-vision", supports_vision=False)
        manager = _make_manager_mock([entry], ["openrouter"])
        generator = _make_generator_mock()

        self._run([task], manager, generator)

        # routing_decision.model for "visual" category is moonshotai/kimi-k2.5
        generator.generate.assert_called_once_with(task, model="moonshotai/kimi-k2.5")

    def test_provider_manager_failure_uses_fallback(self):
        """When ProviderManager.get_inventory raises, routing decision model is used."""
        task = "implement a feature"
        manager = MagicMock()
        manager.get_inventory.side_effect = RuntimeError("network error")
        generator = _make_generator_mock()

        self._run([task], manager, generator)

        # standard task → MODEL_MAIN
        generator.generate.assert_called_once_with(task, model="z-ai/glm-5.1")

    def test_provider_flag_restricts_candidates(self):
        """With --provider openrouter, only openrouter entries are considered.

        The anthropic entry has a large context window (score bonus) that would
        win without filtering, proving the filter is applied when it matters.
        """
        task = "implement a feature"
        openrouter_entry = _make_entry("openrouter/basic", pricing={"prompt": "0.0000005"})
        # anthropic entry scores higher (200K context → +2 bonus) but must be excluded
        anthropic_entry = _make_entry(
            "anthropic/claude-large", provider="anthropic", context_window=200_000,
            pricing={"prompt": "0.0000005"},
        )
        manager = _make_manager_mock([openrouter_entry, anthropic_entry], ["openrouter", "anthropic"])
        generator = _make_generator_mock()

        self._run([task, "--provider", "openrouter"], manager, generator)

        generator.generate.assert_called_once_with(task, model="openrouter/basic")

    def test_scored_routing_logged(self):
        """_log_run is called with a ScoredRoutingResult that reflects the winning candidate."""
        task = "implement a feature"
        entry = _make_entry("openrouter/fast-model", pricing={"prompt": "0.0000005"})
        manager = _make_manager_mock([entry], ["openrouter"])
        generator = _make_generator_mock()

        with patch("openshard.cli.main.ProviderManager", return_value=manager), \
             patch("openshard.cli.main.ExecutionGenerator", return_value=generator), \
             patch("openshard.cli.main.get_api_key", return_value="test-key"), \
             patch("openshard.cli.main._log_run") as mock_log:
            runner = CliRunner()
            runner.invoke(cli, ["run", task])

        mock_log.assert_called_once()
        _scored = mock_log.call_args.kwargs.get("_scored")
        self.assertIsNotNone(_scored)
        self.assertEqual(_scored.category, "standard")
        self.assertEqual(_scored.selected_model, "openrouter/fast-model")
        self.assertEqual(_scored.selected_provider, "openrouter")
        self.assertFalse(_scored.used_fallback)
