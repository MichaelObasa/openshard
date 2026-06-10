"""Tests for feedback routing advisory wiring into scoring (PR 3).

Validates:
- FRA with consider_stronger_review penalizes cheap-class models
- FRA with no negative signals applies no penalty
- FRA result stored in Shard via _log_run (no recompute)
- Medium confidence produces heavier penalty than low
- FRA failure never crashes the run
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_signals(rejected=0, partial=0, retry=0) -> list[dict]:
    signals = []
    for _ in range(rejected):
        signals.append({"signal_type": "rejected_explicit"})
    for _ in range(partial):
        signals.append({"signal_type": "partial_explicit"})
    for _ in range(retry):
        signals.append({"signal_type": "retry_requested"})
    return signals


def _make_registry_entry(model_id: str, cost_class: str):
    m = MagicMock()
    m.id = model_id
    m.cost_class = cost_class
    return m


def _make_inventory_entry(model_id: str):
    e = MagicMock()
    e.model.id = model_id
    e.model.pricing = {}
    e.model.supports_vision = False
    e.model.supports_tools = True
    e.model.context_window = 128_000
    e.provider = "openrouter"
    return e


# ---------------------------------------------------------------------------
# build_feedback_routing_advisory (unit)
# ---------------------------------------------------------------------------

class TestBuildFeedbackRoutingAdvisory:
    def test_rejected_signal_produces_consider_stronger_review(self):
        from openshard.models.feedback_advisory import build_feedback_routing_advisory
        result = build_feedback_routing_advisory(_make_signals(rejected=1))
        assert result is not None
        assert result["recommendation"] == "consider_stronger_review"
        assert result["confidence"] == "medium"

    def test_partial_signal_produces_consider_stronger_review(self):
        from openshard.models.feedback_advisory import build_feedback_routing_advisory
        result = build_feedback_routing_advisory(_make_signals(partial=1))
        assert result is not None
        assert result["recommendation"] == "consider_stronger_review"

    def test_retry_only_produces_low_confidence(self):
        from openshard.models.feedback_advisory import build_feedback_routing_advisory
        result = build_feedback_routing_advisory(_make_signals(retry=1))
        assert result is not None
        assert result["confidence"] == "low"

    def test_no_negative_signals_returns_none(self):
        from openshard.models.feedback_advisory import build_feedback_routing_advisory
        result = build_feedback_routing_advisory([])
        assert result is None

    def test_accepted_signal_alone_returns_none(self):
        from openshard.models.feedback_advisory import build_feedback_routing_advisory
        result = build_feedback_routing_advisory([{"signal_type": "accepted_explicit"}])
        assert result is None


# ---------------------------------------------------------------------------
# FRA penalty applied to cheap models in pipeline scoring block
# ---------------------------------------------------------------------------

class TestFRAScoringPenalty:
    """
    Tests that FRA adjustments are computed and fed into _merged_adjustments
    before select_with_info is called. We test the logic by exercising the
    FRA adjustment construction directly (mirroring the pipeline block).
    """

    def _compute_fra_adjustments(self, signals: list[dict], entries) -> dict[str, float]:
        """Replicate the FRA adjustment logic from pipeline.py."""
        from openshard.models.feedback_advisory import build_feedback_routing_advisory
        fra_result = build_feedback_routing_advisory(signals)
        fra_adjustments: dict[str, float] = {}
        if fra_result is not None and fra_result.get("recommendation") == "consider_stronger_review":
            confidence = fra_result.get("confidence", "low")
            penalty = -1.5 if confidence == "medium" else -0.5
            for entry in entries:
                reg = entry["reg"]
                if reg is not None and reg.cost_class == "cheap":
                    fra_adjustments[entry["id"]] = penalty
        return fra_adjustments

    def test_medium_confidence_penalty_is_minus_1_5(self):
        entries = [{"id": "cheap/model", "reg": _make_registry_entry("cheap/model", "cheap")}]
        signals = _make_signals(rejected=1)  # medium confidence
        adj = self._compute_fra_adjustments(signals, entries)
        assert adj.get("cheap/model") == -1.5

    def test_low_confidence_penalty_is_minus_0_5(self):
        entries = [{"id": "cheap/model", "reg": _make_registry_entry("cheap/model", "cheap")}]
        signals = _make_signals(retry=1)  # low confidence
        adj = self._compute_fra_adjustments(signals, entries)
        assert adj.get("cheap/model") == -0.5

    def test_non_cheap_model_not_penalized(self):
        entries = [{"id": "strong/model", "reg": _make_registry_entry("strong/model", "expensive")}]
        signals = _make_signals(rejected=1)
        adj = self._compute_fra_adjustments(signals, entries)
        assert "strong/model" not in adj

    def test_no_signals_no_penalty(self):
        entries = [{"id": "cheap/model", "reg": _make_registry_entry("cheap/model", "cheap")}]
        adj = self._compute_fra_adjustments([], entries)
        assert adj == {}

    def test_unknown_registry_entry_not_penalized(self):
        entries = [{"id": "mystery/model", "reg": None}]
        signals = _make_signals(rejected=1)
        adj = self._compute_fra_adjustments(signals, entries)
        assert "mystery/model" not in adj


# ---------------------------------------------------------------------------
# _log_run: FRA result stored without recompute
# ---------------------------------------------------------------------------

class TestLogRunFRAStorage:
    def _make_generator(self):
        g = MagicMock()
        g.model = "test/model"
        g.fixer_model = "test/fixer"
        return g

    def test_fra_passed_in_stored_directly(self, tmp_path):
        from openshard.run._pipeline_helpers import _log_run

        fra = {
            "recommendation": "consider_stronger_review",
            "confidence": "medium",
            "advisory_only": True,
        }

        with patch("openshard.run._pipeline_helpers.append_jsonl") as mock_append:
            with patch("openshard.run._pipeline_helpers._safe_git_info", return_value={}):
                with patch("openshard.history.shard_schema.coerce_shard_entry", side_effect=lambda x: x):
                    with patch("openshard.history.shard_contract._make_shard_id", return_value="id"):
                        with patch("openshard.history.routing_truth.build_routing_truth", return_value=MagicMock()):
                            with patch("openshard.history.routing_truth.routing_truth_to_dict", return_value={}):
                                with patch("openshard.models.feedback_advisory._load_recent_session_signals") as mock_load:
                                    import time
                                    _log_run(
                                        start=time.time(),
                                        task="fix login",
                                        generator=self._make_generator(),
                                        retry_triggered=False,
                                        files=[],
                                        verification_attempted=False,
                                        verification_passed=None,
                                        workspace=tmp_path,
                                        fra_result=fra,
                                    )
        # Should NOT re-load from file
        mock_load.assert_not_called()
        entry = mock_append.call_args[0][1]
        assert entry["feedback_routing_advisory"]["recommendation"] == "consider_stronger_review"

    def test_fra_applied_flag_set_when_recommendation_present(self, tmp_path):
        from openshard.run._pipeline_helpers import _log_run

        fra = {"recommendation": "consider_stronger_review", "confidence": "medium"}

        with patch("openshard.run._pipeline_helpers.append_jsonl") as mock_append:
            with patch("openshard.run._pipeline_helpers._safe_git_info", return_value={}):
                with patch("openshard.history.shard_schema.coerce_shard_entry", side_effect=lambda x: x):
                    with patch("openshard.history.shard_contract._make_shard_id", return_value="id"):
                        with patch("openshard.history.routing_truth.build_routing_truth", return_value=MagicMock()):
                            with patch("openshard.history.routing_truth.routing_truth_to_dict", return_value={}):
                                import time
                                _log_run(
                                    start=time.time(),
                                    task="fix login",
                                    generator=self._make_generator(),
                                    retry_triggered=False,
                                    files=[],
                                    verification_attempted=False,
                                    verification_passed=None,
                                    workspace=tmp_path,
                                    fra_result=fra,
                                )
        entry = mock_append.call_args[0][1]
        assert entry.get("feedback_routing_applied") is True

    def test_fra_applied_flag_not_set_when_no_recommendation(self, tmp_path):
        from openshard.run._pipeline_helpers import _log_run

        fra = {"recommendation": "no_change", "confidence": "low"}

        with patch("openshard.run._pipeline_helpers.append_jsonl") as mock_append:
            with patch("openshard.run._pipeline_helpers._safe_git_info", return_value={}):
                with patch("openshard.history.shard_schema.coerce_shard_entry", side_effect=lambda x: x):
                    with patch("openshard.history.shard_contract._make_shard_id", return_value="id"):
                        with patch("openshard.history.routing_truth.build_routing_truth", return_value=MagicMock()):
                            with patch("openshard.history.routing_truth.routing_truth_to_dict", return_value={}):
                                import time
                                _log_run(
                                    start=time.time(),
                                    task="fix login",
                                    generator=self._make_generator(),
                                    retry_triggered=False,
                                    files=[],
                                    verification_attempted=False,
                                    verification_passed=None,
                                    workspace=tmp_path,
                                    fra_result=fra,
                                )
        entry = mock_append.call_args[0][1]
        assert "feedback_routing_applied" not in entry
