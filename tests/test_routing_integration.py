from __future__ import annotations

import unittest
from unittest.mock import ANY, MagicMock, call, patch

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
             patch("openshard.cli.main.analyze_repo"), \
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

        generator.generate.assert_called_once_with(task, model="openrouter/fast-model", repo_facts=ANY)

    def test_fallback_when_no_candidate(self):
        """When the only inventory entry fails hard filter, routing decision model is used."""
        task = "add a ui component"
        # visual category → needs_vision=True; this entry has supports_vision=False
        entry = _make_entry("openrouter/no-vision", supports_vision=False)
        manager = _make_manager_mock([entry], ["openrouter"])
        generator = _make_generator_mock()

        self._run([task], manager, generator)

        # routing_decision.model for "visual" category is moonshotai/kimi-k2.5
        generator.generate.assert_called_once_with(task, model="moonshotai/kimi-k2.5", repo_facts=ANY)

    def test_provider_manager_failure_uses_fallback(self):
        """When ProviderManager.get_inventory raises, routing decision model is used."""
        task = "implement a feature"
        manager = MagicMock()
        manager.get_inventory.side_effect = RuntimeError("network error")
        generator = _make_generator_mock()

        self._run([task], manager, generator)

        # standard task → MODEL_MAIN
        generator.generate.assert_called_once_with(task, model="z-ai/glm-5.1", repo_facts=ANY)

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

        generator.generate.assert_called_once_with(task, model="openrouter/basic", repo_facts=ANY)

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


class TestRoutingDisplayConsistency(unittest.TestCase):
    """Verify that the early [routing] line in --more output matches the final selected model."""

    def _run(self, args: list[str], manager_mock, generator_mock):
        with patch("openshard.cli.main.ProviderManager", return_value=manager_mock), \
             patch("openshard.cli.main.ExecutionGenerator", return_value=generator_mock), \
             patch("openshard.cli.main.get_api_key", return_value="test-key"), \
             patch("openshard.cli.main.analyze_repo"), \
             patch("openshard.cli.main._log_run"):
            runner = CliRunner()
            result = runner.invoke(cli, ["run"] + args)
        return result

    def test_routing_line_shows_scored_model_not_keyword_model(self):
        """With --more, the [routing] line shows the scored model, not the keyword-routed one.

        Keyword routing for 'implement a feature' picks GLM-5.1, but the inventory
        has fast-model which scoring selects instead.  The display line must agree.
        """
        task = "implement a feature"
        entry = _make_entry("openrouter/fast-model", pricing={"prompt": "0.0000005"})
        manager = _make_manager_mock([entry], ["openrouter"])
        generator = _make_generator_mock()

        result = self._run([task, "--more"], manager, generator)

        # Find the single early routing summary line (contains " - " rationale separator)
        routing_lines = [
            ln for ln in result.output.splitlines()
            if "[routing]" in ln and " - " in ln
        ]
        self.assertEqual(len(routing_lines), 1, result.output)
        # _model_label("openrouter/fast-model") → "Fast Model"
        self.assertIn("Fast Model", routing_lines[0])
        # Keyword-routed model (GLM-5.1) must NOT appear in this line
        self.assertNotIn("GLM", routing_lines[0])

    def test_routing_line_uses_fallback_model_when_scoring_finds_no_candidate(self):
        """When no inventory entry passes the hard filter, the fallback (keyword) model is shown."""
        task = "add a ui component"  # routes to visual category → needs_vision=True
        # Entry lacks vision support → hard-filtered out → fallback
        entry = _make_entry("openrouter/no-vision", supports_vision=False)
        manager = _make_manager_mock([entry], ["openrouter"])
        generator = _make_generator_mock()

        result = self._run([task, "--more"], manager, generator)

        routing_lines = [
            ln for ln in result.output.splitlines()
            if "[routing]" in ln and " - " in ln
        ]
        self.assertEqual(len(routing_lines), 1, result.output)
        # Fallback for visual category is kimi-k2.5; the label should NOT say fast-model
        self.assertNotIn("fast-model", routing_lines[0])
        self.assertNotIn("no-vision", routing_lines[0])
