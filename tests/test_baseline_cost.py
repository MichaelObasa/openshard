from __future__ import annotations

import unittest
from unittest.mock import patch

from openshard.cost.baseline import BASELINE_MODELS, format_baseline_line


class TestFormatBaselineLine(unittest.TestCase):

    # --- Guard conditions ---

    def test_returns_none_when_both_tokens_zero(self):
        self.assertIsNone(format_baseline_line(0, 0))

    def test_returns_none_when_all_models_unresolvable(self):
        with patch("openshard.cost.baseline.compute_cost", return_value=None):
            self.assertIsNone(format_baseline_line(1_000, 500))

    # --- Label format ---

    def test_starts_with_baseline_estimate_label(self):
        result = format_baseline_line(1_000_000, 1_000_000)
        self.assertIsNotNone(result)
        self.assertTrue(result.startswith("Baseline estimate: "))

    def test_contains_gpt55_label(self):
        result = format_baseline_line(1_000_000, 1_000_000)
        self.assertIn("GPT-5.5", result)

    def test_contains_sonnet46_label(self):
        result = format_baseline_line(1_000_000, 1_000_000)
        self.assertIn("Sonnet 4.6", result)

    def test_models_separated_by_comma_space(self):
        result = format_baseline_line(1_000_000, 1_000_000)
        self.assertIn(", ", result)

    # --- Numeric correctness ---

    def test_sonnet46_cost_computed_correctly(self):
        # anthropic/claude-sonnet-4.6: (3.00, 15.00) per million
        # 500k prompt + 500k completion = 1.50 + 7.50 = $9.000
        result = format_baseline_line(500_000, 500_000)
        self.assertIn("Sonnet 4.6 $9.000", result)

    def test_gpt55_cost_computed_correctly(self):
        # openai/gpt-5.5: (5.00, 30.00) per million
        # 500k prompt + 500k completion = 2.50 + 15.00 = $17.500
        result = format_baseline_line(500_000, 500_000)
        self.assertIn("GPT-5.5 $17.500", result)

    def test_three_decimal_places(self):
        import re
        result = format_baseline_line(1_000, 1_000)
        self.assertIsNotNone(result)
        prices = re.findall(r"\$(\d+\.\d+)", result)
        self.assertGreater(len(prices), 0)
        for p in prices:
            self.assertEqual(len(p.split(".")[1]), 3)

    # --- Multiplier: absent without actual_cost ---

    def test_no_multiplier_when_actual_cost_none(self):
        result = format_baseline_line(1_000_000, 1_000_000, actual_cost=None)
        self.assertNotIn("x higher", result)

    def test_no_multiplier_when_actual_cost_zero(self):
        result = format_baseline_line(1_000_000, 1_000_000, actual_cost=0.0)
        self.assertNotIn("x higher", result)

    # --- Multiplier: present with actual_cost ---

    def test_multiplier_present_when_actual_cost_positive(self):
        result = format_baseline_line(1_000_000, 1_000_000, actual_cost=0.001)
        self.assertIn("x higher", result)

    def test_multiplier_whole_number_at_or_above_10x(self):
        # Sonnet 4.6 at 1M/1M tokens = $18.00; actual = $1.00 → ratio 18x (>=10)
        result = format_baseline_line(1_000_000, 1_000_000, actual_cost=1.0)
        # Should use round(), not .1f
        self.assertIn("~18x higher", result)

    def test_multiplier_one_decimal_below_10x(self):
        # Sonnet 4.6 at 1M/1M = $18.00; actual = $5.00 → ratio 3.6x (<10)
        result = format_baseline_line(1_000_000, 1_000_000, actual_cost=5.0)
        self.assertIn("~3.6x higher", result)

    def test_multiplier_in_parens_after_price(self):
        result = format_baseline_line(1_000_000, 1_000_000, actual_cost=0.01)
        # Each entry should look like "Label $X.XXX (~Nx higher)"
        self.assertRegex(result, r"\$\d+\.\d{3} \(~[\d.]+x higher\)")

    # --- Partial resolution ---

    def test_partial_model_resolution_still_renders(self):
        def fake_compute(model, pt, ct):
            if "gpt" in model:
                return None
            from openshard.providers.openrouter import compute_cost as real
            return real(model, pt, ct)

        with patch("openshard.cost.baseline.compute_cost", side_effect=fake_compute):
            result = format_baseline_line(1_000_000, 1_000_000)
        self.assertIsNotNone(result)
        self.assertIn("Sonnet 4.6", result)
        self.assertNotIn("GPT-5.5", result)

    # --- Custom models override ---

    def test_custom_models_parameter_respected(self):
        custom = [("Custom", "anthropic/claude-sonnet-4.6")]
        result = format_baseline_line(1_000_000, 1_000_000, models=custom)
        self.assertIn("Custom", result)
        self.assertNotIn("GPT-5.5", result)

    # --- No prohibited language ---

    def test_no_savings_claim(self):
        result = format_baseline_line(1_000_000, 1_000_000, actual_cost=0.01)
        self.assertNotIn("save", result.lower())
        self.assertNotIn("%", result)
        self.assertNotIn("cheaper", result.lower())


class TestBaselineModelsConstant(unittest.TestCase):

    def test_has_two_entries(self):
        self.assertEqual(len(BASELINE_MODELS), 2)

    def test_gpt55_is_first(self):
        self.assertEqual(BASELINE_MODELS[0][1], "openai/gpt-5.5")

    def test_sonnet46_is_second(self):
        self.assertEqual(BASELINE_MODELS[1][1], "anthropic/claude-sonnet-4.6")
