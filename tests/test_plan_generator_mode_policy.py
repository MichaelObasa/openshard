"""Tests for mode policy wiring into PlanGenerator (PR 4).

Validates:
- PlanGenerator uses mode_policy when no config.planning_model is set
- Config planning_model always wins over mode policy
- Tier fallback still works when no policy is available
- ModeModelPolicy.advisory_only correctly reflects wired vs. pending state
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# ModeModelPolicy advisory_only state
# ---------------------------------------------------------------------------

class TestModeModelPolicyAdvisoryOnly:
    def test_plan_policy_is_wired(self):
        from openshard.models.mode_policy import model_policy_for_mode
        policy = model_policy_for_mode("plan")
        assert policy is not None
        assert policy.advisory_only is False, (
            "_PLAN_POLICY.advisory_only must be False since PlanGenerator is wired"
        )

    def test_ask_policy_is_still_advisory(self):
        from openshard.models.mode_policy import model_policy_for_mode
        policy = model_policy_for_mode("ask")
        assert policy is not None
        assert policy.advisory_only is True, (
            "_ASK_POLICY.advisory_only must remain True until TUI ask makes provider calls"
        )

    def test_plan_policy_has_default_model_id(self):
        from openshard.models.mode_policy import model_policy_for_mode
        policy = model_policy_for_mode("plan")
        assert isinstance(policy.default_model_id, str) and policy.default_model_id

    def test_plan_policy_mode_is_plan(self):
        from openshard.models.mode_policy import model_policy_for_mode
        policy = model_policy_for_mode("plan")
        assert policy.mode == "plan"


# ---------------------------------------------------------------------------
# PlanGenerator model source priority
# ---------------------------------------------------------------------------

def _mock_provider():
    p = MagicMock()
    p.execute.return_value = MagicMock(
        content='{"summary": "test", "stages": []}',
        usage=MagicMock(prompt_tokens=10, completion_tokens=5, total_tokens=15, estimated_cost=None),
    )
    return p


class TestPlanGeneratorModelSource:
    def _config_with_tiers(self, planning_model=None):
        cfg = {
            "model_tiers": [
                {"name": "fast", "model": "tier/model-fast"},
                {"name": "balanced", "model": "tier/model-balanced"},
            ]
        }
        if planning_model:
            cfg["planning_model"] = planning_model
        return cfg

    def test_config_planning_model_wins_over_policy(self):
        with patch("openshard.planning.generator.load_config",
                   return_value=self._config_with_tiers(planning_model="config/explicit-model")):
            with patch("openshard.planning.generator.get_api_key", return_value="key"):
                gen = __import__(
                    "openshard.planning.generator", fromlist=["PlanGenerator"]
                ).PlanGenerator(provider=_mock_provider())
        assert gen.model == "config/explicit-model"
        assert gen._model_source == "config"

    def test_mode_policy_used_when_no_config_planning_model(self):
        from openshard.models.mode_policy import model_policy_for_mode
        expected = model_policy_for_mode("plan").default_model_id

        with patch("openshard.planning.generator.load_config",
                   return_value=self._config_with_tiers()):
            with patch("openshard.planning.generator.get_api_key", return_value="key"):
                from openshard.planning.generator import PlanGenerator
                gen = PlanGenerator(provider=_mock_provider())

        assert gen.model == expected
        assert gen._model_source == "mode_policy"

    def test_model_tiers_fallback_when_policy_unavailable(self):
        with patch("openshard.planning.generator.load_config",
                   return_value=self._config_with_tiers()):
            with patch("openshard.planning.generator.get_api_key", return_value="key"):
                with patch("openshard.models.mode_policy.model_policy_for_mode", return_value=None):
                    from openshard.planning.generator import PlanGenerator
                    gen = PlanGenerator(provider=_mock_provider())

        assert gen.model == "tier/model-fast"
        assert gen._model_source == "model_tiers"

    def test_model_tiers_fallback_when_policy_raises(self):
        with patch("openshard.planning.generator.load_config",
                   return_value=self._config_with_tiers()):
            with patch("openshard.planning.generator.get_api_key", return_value="key"):
                with patch(
                    "openshard.models.mode_policy.model_policy_for_mode",
                    side_effect=RuntimeError("policy unavailable"),
                ):
                    from openshard.planning.generator import PlanGenerator
                    gen = PlanGenerator(provider=_mock_provider())

        assert gen.model == "tier/model-fast"
        assert gen._model_source == "model_tiers"

    def test_raises_when_no_tiers_and_no_policy(self):
        with patch("openshard.planning.generator.load_config", return_value={}):
            with patch("openshard.planning.generator.get_api_key", return_value="key"):
                with patch("openshard.models.mode_policy.model_policy_for_mode", return_value=None):
                    from openshard.planning.generator import PlanGenerator
                    with pytest.raises(RuntimeError, match="No model_tiers"):
                        PlanGenerator(provider=_mock_provider())

    def test_no_planning_model_in_config_uses_policy_not_tier(self):
        """Regression: before this fix, empty planning_model fell through to tiers."""
        from openshard.models.mode_policy import model_policy_for_mode
        policy_model = model_policy_for_mode("plan").default_model_id
        tier_model = "tier/model-fast"

        with patch("openshard.planning.generator.load_config",
                   return_value=self._config_with_tiers()):
            with patch("openshard.planning.generator.get_api_key", return_value="key"):
                from openshard.planning.generator import PlanGenerator
                gen = PlanGenerator(provider=_mock_provider())

        assert gen.model == policy_model
        assert gen.model != tier_model
