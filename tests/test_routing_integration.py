from __future__ import annotations

import unittest
from unittest.mock import ANY, MagicMock, call, patch

from click.testing import CliRunner

from openshard.analysis.repo import RepoFacts
from openshard.cli.main import cli
from openshard.providers.base import ModelInfo
from openshard.providers.manager import InventoryEntry

_AUTO_CONFIG = {"approval_mode": "auto"}

_PYTHON_REPO = RepoFacts(
    languages=["python"], package_files=[], framework=None,
    test_command=None, risky_paths=[], changed_files=[],
)


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
             patch("openshard.cli.main.load_config", return_value=_AUTO_CONFIG), \
             patch("openshard.cli.main.analyze_repo", return_value=_PYTHON_REPO), \
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

        generator.generate.assert_called_once_with(task, model="openrouter/fast-model", repo_facts=ANY, skills_context="")

    def test_fallback_when_no_candidate(self):
        """When the only inventory entry fails hard filter, routing decision model is used."""
        task = "add a ui component"
        # visual category → needs_vision=True; this entry has supports_vision=False
        entry = _make_entry("openrouter/no-vision", supports_vision=False)
        manager = _make_manager_mock([entry], ["openrouter"])
        generator = _make_generator_mock()

        self._run([task], manager, generator)

        # routing_decision.model for "visual" category is moonshotai/kimi-k2.5
        generator.generate.assert_called_once_with(task, model="moonshotai/kimi-k2.5", repo_facts=ANY, skills_context="")

    def test_provider_manager_failure_uses_fallback(self):
        """When ProviderManager.get_inventory raises, routing decision model is used."""
        task = "implement a feature"
        manager = MagicMock()
        manager.get_inventory.side_effect = RuntimeError("network error")
        generator = _make_generator_mock()

        self._run([task], manager, generator)

        # standard task → MODEL_MAIN
        generator.generate.assert_called_once_with(task, model="z-ai/glm-5.1", repo_facts=ANY, skills_context="")

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

        generator.generate.assert_called_once_with(task, model="openrouter/basic", repo_facts=ANY, skills_context="")

    def test_scored_routing_logged(self):
        """_log_run is called with a ScoredRoutingResult that reflects the winning candidate."""
        task = "implement a feature"
        entry = _make_entry("openrouter/fast-model", pricing={"prompt": "0.0000005"})
        manager = _make_manager_mock([entry], ["openrouter"])
        generator = _make_generator_mock()

        with patch("openshard.cli.main.ProviderManager", return_value=manager), \
             patch("openshard.cli.main.ExecutionGenerator", return_value=generator), \
             patch("openshard.cli.main.get_api_key", return_value="test-key"), \
             patch("openshard.cli.main.load_config", return_value=_AUTO_CONFIG), \
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
             patch("openshard.cli.main.load_config", return_value=_AUTO_CONFIG), \
             patch("openshard.cli.main.analyze_repo", return_value=_PYTHON_REPO), \
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

    def test_default_routing_line_shows_scored_model(self):
        """Default (no --more) routing line shows the scored model, not the keyword-routed one."""
        task = "implement a feature"
        entry = _make_entry("openrouter/fast-model", pricing={"prompt": "0.0000005"})
        manager = _make_manager_mock([entry], ["openrouter"])
        generator = _make_generator_mock()

        result = self._run([task], manager, generator)

        routing_lines = [
            ln for ln in result.output.splitlines()
            if ln.strip().startswith("Routing -")
        ]
        self.assertEqual(len(routing_lines), 1, result.output)
        self.assertIn("Fast Model", routing_lines[0])
        self.assertNotIn("GLM", routing_lines[0])


class TestApprovalFlag(unittest.TestCase):
    """Verify --approval flag overrides config and triggers gates correctly."""

    def _run_with_write(self, args: list[str], generator_mock, approval_mode="auto"):
        config = {"approval_mode": approval_mode}
        with patch("openshard.cli.main.ProviderManager"), \
             patch("openshard.cli.main.ExecutionGenerator", return_value=generator_mock), \
             patch("openshard.cli.main.get_api_key", return_value="test-key"), \
             patch("openshard.cli.main.load_config", return_value=config), \
             patch("openshard.cli.main.analyze_repo", return_value=_PYTHON_REPO), \
             patch("openshard.cli.main._log_run"), \
             patch("openshard.cli.main._write_files"):
            runner = CliRunner()
            result = runner.invoke(cli, ["run"] + args, input="n\n")
        return result

    def test_approval_ask_flag_triggers_gate(self):
        """--approval ask shows gate prompt before file write without editing config."""
        generator = _make_generator_mock()
        generator.generate.return_value.files = [MagicMock(path="helper.py")]
        result = self._run_with_write(
            ["add a helper", "--write", "--approval", "ask"],
            generator,
            approval_mode="auto",
        )
        assert "[gate]" in result.output, result.output

    def test_config_approval_mode_honored_without_flag(self):
        """When --approval is not passed, config approval_mode is used."""
        generator = _make_generator_mock()
        generator.generate.return_value.files = [MagicMock(path="helper.py")]
        result = self._run_with_write(
            ["add a helper", "--write"],
            generator,
            approval_mode="ask",
        )
        assert "[gate]" in result.output, result.output

    def test_auto_config_no_gate_without_flag(self):
        """With config auto and no --approval flag, no gate prompt appears."""
        generator = _make_generator_mock()
        generator.generate.return_value.files = [MagicMock(path="helper.py")]
        result = self._run_with_write(
            ["add a helper", "--write"],
            generator,
            approval_mode="auto",
        )
        assert "[gate]" not in result.output, result.output

    def test_invalid_approval_flag_fails_cleanly(self):
        """--approval with an invalid value exits with a usage error."""
        runner = CliRunner()
        result = runner.invoke(cli, ["run", "some task", "--approval", "banana"])
        assert result.exit_code != 0
        assert "Invalid value" in result.output or "Error" in result.output


class TestHistoryScoringDisplay(unittest.TestCase):
    """Verify history-scoring lines appear in --more output when the flag is set."""

    def _run(self, args: list[str], manager_mock, generator_mock,
             adjustments=None, reasons=None):
        adjustments = adjustments or {}
        reasons = reasons or {}
        with patch("openshard.cli.main.ProviderManager", return_value=manager_mock), \
             patch("openshard.cli.main.ExecutionGenerator", return_value=generator_mock), \
             patch("openshard.cli.main.get_api_key", return_value="test-key"), \
             patch("openshard.cli.main.load_config", return_value=_AUTO_CONFIG), \
             patch("openshard.cli.main.analyze_repo", return_value=_PYTHON_REPO), \
             patch("openshard.cli.main._log_run"), \
             patch("openshard.cli.main.load_runs", return_value=[]), \
             patch("openshard.cli.main.compute_history_adjustments", return_value=adjustments), \
             patch("openshard.cli.main.compute_history_adjustment_reasons", return_value=reasons):
            runner = CliRunner()
            result = runner.invoke(cli, ["run"] + args)
        return result

    def test_history_scoring_enabled_line_shown(self):
        """[routing] history scoring: enabled appears in --more output when flag is set."""
        entry = _make_entry("openrouter/fast-model", pricing={"prompt": "0.0000005"})
        manager = _make_manager_mock([entry], ["openrouter"])
        generator = _make_generator_mock()

        result = self._run(["implement a feature", "--more", "--history-scoring"], manager, generator)

        self.assertIn("[routing] history scoring: enabled", result.output, result.output)

    def test_history_nonzero_adjustment_shown(self):
        """Non-zero adjustment for selected model shows value and reason in --more output."""
        entry = _make_entry("openrouter/fast-model", pricing={"prompt": "0.0000005"})
        manager = _make_manager_mock([entry], ["openrouter"])
        generator = _make_generator_mock()
        adjustments = {"openrouter/fast-model": 1.0}
        reasons = {"openrouter/fast-model": "high pass rate"}

        result = self._run(
            ["implement a feature", "--more", "--history-scoring"],
            manager, generator,
            adjustments=adjustments, reasons=reasons,
        )

        self.assertIn("+1.0", result.output, result.output)
        self.assertIn("high pass rate", result.output, result.output)
        self.assertIn("← selected", result.output, result.output)

    def test_history_scoring_hidden_without_flag(self):
        """history scoring lines must NOT appear when --history-scoring is absent."""
        entry = _make_entry("openrouter/fast-model", pricing={"prompt": "0.0000005"})
        manager = _make_manager_mock([entry], ["openrouter"])
        generator = _make_generator_mock()

        result = self._run(["implement a feature", "--more"], manager, generator)

        self.assertNotIn("history scoring", result.output, result.output)


class TestDeepSeekBoilerplateModel(unittest.TestCase):
    """DeepSeek V4 Flash is the boilerplate model; V3.2 is no longer the default."""

    def test_boilerplate_keyword_routes_to_v4_flash(self):
        from openshard.routing.engine import route, MODEL_CHEAP
        decision = route("add a simple validation helper")
        self.assertEqual(decision.category, "boilerplate")
        self.assertEqual(decision.model, MODEL_CHEAP)
        self.assertIn("v4-flash", MODEL_CHEAP)

    def test_model_cheap_is_not_v3_2(self):
        from openshard.routing.engine import MODEL_CHEAP
        self.assertNotIn("v3.2", MODEL_CHEAP)
