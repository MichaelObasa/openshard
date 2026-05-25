from __future__ import annotations

import unittest

from click.testing import CliRunner

from openshard.cli.main import cli
from openshard.models.advisory import ModelAdvisory, recommend_models


class TestRecommendModels(unittest.TestCase):
    def test_cheap_control_excludes_experimental_by_default(self):
        results = recommend_models(role="cheap_control")
        self.assertGreater(len(results), 0)
        for advisory in results:
            self.assertFalse(
                advisory.model.experimental,
                f"Model {advisory.model.id} is experimental but should be excluded",
            )

    def test_cheap_control_with_include_experimental(self):
        results = recommend_models(role="cheap_control", include_experimental=True, limit=10)
        experimental_ids = {a.model.id for a in results if a.model.experimental}
        # qwen3.6-flash, gemma-4-26b, gemma-4-31b, ibm-granite are experimental cheap_control
        self.assertGreater(
            len(experimental_ids), 0, "Expected at least one experimental cheap_control model"
        )

    def test_required_capability_reasoning_includes_key_models(self):
        results = recommend_models(
            required_capabilities=("reasoning",),
            include_experimental=True,
            limit=10,
        )
        ids = {a.model.id for a in results}
        self.assertIn("x-ai/grok-4.3", ids)
        self.assertIn("anthropic/claude-sonnet-4.6", ids)

    def test_max_cost_cheap_excludes_mid_and_expensive(self):
        results = recommend_models(max_cost_class="cheap", include_experimental=True, limit=20)
        valid = {"free", "tiny", "cheap"}
        for advisory in results:
            self.assertIn(
                advisory.model.cost_class,
                valid,
                f"Model {advisory.model.id} has cost_class={advisory.model.cost_class} which exceeds cheap",
            )

    def test_risk_high_top_results_are_strong_frontier_or_reasoning(self):
        results = recommend_models(risk="high", include_experimental=True, limit=10)
        self.assertGreater(len(results), 0)
        for advisory in results[:3]:
            m = advisory.model
            self.assertTrue(
                m.tier in {"strong", "frontier"} or m.supports_reasoning,
                f"{m.id} (tier={m.tier}, reasoning={m.supports_reasoning}) should be strong/frontier or reasoning for risk=high",
            )

    def test_returns_list_of_model_advisory_instances(self):
        results = recommend_models()
        self.assertIsInstance(results, list)
        for r in results:
            self.assertIsInstance(r, ModelAdvisory)

    def test_unknown_capability_returns_empty(self):
        results = recommend_models(required_capabilities=("telekinesis",))
        self.assertEqual(results, [])

    def test_unknown_role_returns_empty(self):
        results = recommend_models(role="nonexistent_role_xyz")
        self.assertEqual(results, [])

    def test_deterministic_ordering(self):
        kwargs = dict(role="cheap_control", risk="low", include_experimental=True)
        call1 = recommend_models(**kwargs)
        call2 = recommend_models(**kwargs)
        self.assertEqual(
            [a.model.id for a in call1],
            [a.model.id for a in call2],
        )


class TestModelsRecommendCommand(unittest.TestCase):
    def setUp(self):
        self.runner = CliRunner()

    def test_recommend_role_cheap_control_exits_zero(self):
        result = self.runner.invoke(cli, ["models", "recommend", "--role", "cheap_control"])
        self.assertEqual(result.exit_code, 0, result.output)

    def test_recommend_role_cheap_control_output_contains_gemini(self):
        result = self.runner.invoke(cli, ["models", "recommend", "--role", "cheap_control"])
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertIn("gemini", result.output.lower())

    def test_include_experimental_shows_experimental_model(self):
        result = self.runner.invoke(
            cli,
            ["models", "recommend", "--include-experimental", "--limit", "10"],
        )
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertTrue(
            "grok-build" in result.output or "poolside" in result.output,
            f"Expected experimental model in output:\n{result.output}",
        )

    def test_unknown_capability_shows_informative_message(self):
        result = self.runner.invoke(
            cli,
            ["models", "recommend", "--capability", "unknown_xyz"],
        )
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertTrue(
            "unknown" in result.output.lower() or "no results" in result.output.lower(),
            f"Expected informative message:\n{result.output}",
        )
