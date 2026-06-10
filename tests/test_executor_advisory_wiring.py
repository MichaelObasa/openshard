"""Tests for executor advisory wiring (PR 5).

Covers:
- rank_executors _for_dispatch flag correctly sets advisory_only
- _suggest_executor with _for_dispatch=True uses rank_executors
- _suggest_executor backward compatibility (no _for_dispatch) uses heuristic
- _log_run stores executor_source and executor_advisory fields
- Fallback to heuristic when rank_executors raises
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

# ---------------------------------------------------------------------------
# rank_executors._for_dispatch flag
# ---------------------------------------------------------------------------

class TestRankExecutorsForDispatch:
    def test_advisory_only_false_when_for_dispatch_true(self):
        from openshard.routing.executor_advisory import rank_executors
        result = rank_executors("fix the auth bug", _for_dispatch=True)
        assert result.advisory_only is False

    def test_advisory_only_true_when_for_dispatch_false(self):
        from openshard.routing.executor_advisory import rank_executors
        result = rank_executors("fix the auth bug", _for_dispatch=False)
        assert result.advisory_only is True

    def test_advisory_only_true_by_default(self):
        from openshard.routing.executor_advisory import rank_executors
        result = rank_executors("fix the auth bug")
        assert result.advisory_only is True

    def test_recommended_executor_is_string(self):
        from openshard.routing.executor_advisory import rank_executors
        result = rank_executors("add validation helper", _for_dispatch=True)
        assert isinstance(result.recommended.executor, str)
        assert result.recommended.executor in ("native", "opencode", "direct", "staged")

    def test_dispatch_result_has_score_in_range(self):
        from openshard.routing.executor_advisory import rank_executors
        result = rank_executors("add validation helper", _for_dispatch=True)
        assert 0.0 <= result.recommended.score <= 100.0


# ---------------------------------------------------------------------------
# _suggest_executor
# ---------------------------------------------------------------------------

class TestSuggestExecutor:
    def test_returns_three_tuple(self):
        from openshard.run._pipeline_helpers import _suggest_executor
        executor, reason, advisory = _suggest_executor("fix the login flow")
        assert isinstance(executor, str)
        assert isinstance(reason, str)
        assert advisory is None  # heuristic path when _for_dispatch=False

    def test_default_path_short_task_returns_direct(self):
        from openshard.run._pipeline_helpers import _suggest_executor
        executor, _, advisory = _suggest_executor("add a simple helper")
        assert executor == "direct"
        assert advisory is None

    def test_default_path_large_task_returns_native(self):
        from openshard.run._pipeline_helpers import _suggest_executor
        task = " ".join(["word"] * 65)  # > 60 words
        executor, _, _ = _suggest_executor(task)
        assert executor == "native"

    def test_for_dispatch_calls_rank_executors(self):
        from openshard.routing.executor_advisory import AdvisoryCandidate, ExecutorAdvisoryResult
        mock_result = MagicMock(spec=ExecutorAdvisoryResult)
        mock_result.recommended = MagicMock(spec=AdvisoryCandidate)
        mock_result.recommended.executor = "native"
        mock_result.recommended.reasons = ["default executor"]
        mock_result.advisory_only = False

        with patch("openshard.routing.executor_advisory.rank_executors", return_value=mock_result) as mock_rank:
            from openshard.run._pipeline_helpers import _suggest_executor
            executor, reason, advisory = _suggest_executor(
                "fix the auth bug",
                category="security",
                _for_dispatch=True,
            )

        mock_rank.assert_called_once()
        assert executor == "native"
        assert advisory is mock_result
        assert advisory.advisory_only is False

    def test_for_dispatch_falls_back_to_heuristic_on_error(self):
        with patch(
            "openshard.routing.executor_advisory.rank_executors",
            side_effect=RuntimeError("advisor down"),
        ):
            from openshard.run._pipeline_helpers import _suggest_executor
            executor, reason, advisory = _suggest_executor(
                "add a simple helper",
                _for_dispatch=True,
            )
        # Heuristic fallback: short focused task → direct
        assert executor == "direct"
        assert advisory is None

    def test_opencode_not_returned_without_preference(self):
        from openshard.run._pipeline_helpers import _suggest_executor
        executor, _, _ = _suggest_executor("fix everything", _for_dispatch=False)
        assert executor != "opencode"


# ---------------------------------------------------------------------------
# _log_run stores executor fields
# ---------------------------------------------------------------------------

class TestLogRunExecutorFields:
    def _make_minimal_generator(self):
        g = MagicMock()
        g.model = "test/model"
        g.fixer_model = "test/fixer"
        return g

    def _make_advisory_result(self, executor="native", score=80.0, advisory_only=False):
        from openshard.routing.executor_advisory import AdvisoryCandidate, ExecutorAdvisoryResult
        rec = MagicMock(spec=AdvisoryCandidate)
        rec.executor = executor
        rec.score = score
        rec.reasons = ["default executor", "supports receipts"]
        rec.warnings = []

        result = MagicMock(spec=ExecutorAdvisoryResult)
        result.recommended = rec
        result.alternatives = []
        result.advisory_only = advisory_only
        return result

    def test_executor_source_stored_in_entry(self, tmp_path):

        from openshard.run._pipeline_helpers import _log_run

        log_file = tmp_path / ".openshard" / "runs.jsonl"
        log_file.parent.mkdir(parents=True)

        with patch("openshard.run._pipeline_helpers._LOG_PATH",
                   new=log_file.relative_to(tmp_path)):
            with patch("openshard.run._pipeline_helpers.append_jsonl") as mock_append:
                with patch("openshard.run._pipeline_helpers._safe_git_info", return_value={}):
                    with patch("openshard.history.shard_schema.coerce_shard_entry", side_effect=lambda x: x):
                        with patch("openshard.history.shard_contract._make_shard_id", return_value="test-id"):
                            with patch("openshard.history.routing_truth.build_routing_truth", return_value=MagicMock()):
                                with patch("openshard.history.routing_truth.routing_truth_to_dict", return_value={}):
                                    import time
                                    _log_run(
                                        start=time.time(),
                                        task="fix the login flow",
                                        generator=self._make_minimal_generator(),
                                        retry_triggered=False,
                                        files=[],
                                        verification_attempted=False,
                                        verification_passed=None,
                                        workspace=tmp_path,
                                        executor_source="advisory",
                                        executor_advisory_result=self._make_advisory_result(),
                                    )

        mock_append.assert_called_once()
        entry = mock_append.call_args[0][1]
        assert entry.get("executor_source") == "advisory"
        assert "executor_advisory" in entry
        assert entry["executor_advisory"]["recommended"] == "native"
        assert entry["executor_advisory"]["advisory_only"] is False

    def test_unknown_executor_source_not_stored(self, tmp_path):
        from openshard.run._pipeline_helpers import _log_run

        with patch("openshard.run._pipeline_helpers.append_jsonl") as mock_append:
            with patch("openshard.run._pipeline_helpers._safe_git_info", return_value={}):
                with patch("openshard.history.shard_schema.coerce_shard_entry", side_effect=lambda x: x):
                    with patch("openshard.history.shard_contract._make_shard_id", return_value="test-id"):
                        with patch("openshard.history.routing_truth.build_routing_truth", return_value=MagicMock()):
                            with patch("openshard.history.routing_truth.routing_truth_to_dict", return_value={}):
                                import time
                                _log_run(
                                    start=time.time(),
                                    task="fix the login flow",
                                    generator=self._make_minimal_generator(),
                                    retry_triggered=False,
                                    files=[],
                                    verification_attempted=False,
                                    verification_passed=None,
                                    workspace=tmp_path,
                                    executor_source="unknown",  # should NOT be stored
                                )

        entry = mock_append.call_args[0][1]
        assert "executor_source" not in entry

    def test_fra_result_passed_in_stored_without_recompute(self, tmp_path):
        from openshard.run._pipeline_helpers import _log_run

        fra = {
            "recommendation": "consider_stronger_review",
            "confidence": "medium",
            "advisory_only": True,
        }

        with patch("openshard.run._pipeline_helpers.append_jsonl") as mock_append:
            with patch("openshard.run._pipeline_helpers._safe_git_info", return_value={}):
                with patch("openshard.history.shard_schema.coerce_shard_entry", side_effect=lambda x: x):
                    with patch("openshard.history.shard_contract._make_shard_id", return_value="test-id"):
                        with patch("openshard.history.routing_truth.build_routing_truth", return_value=MagicMock()):
                            with patch("openshard.history.routing_truth.routing_truth_to_dict", return_value={}):
                                with patch("openshard.models.feedback_advisory._load_recent_session_signals") as mock_load:
                                    import time
                                    _log_run(
                                        start=time.time(),
                                        task="fix the login flow",
                                        generator=self._make_minimal_generator(),
                                        retry_triggered=False,
                                        files=[],
                                        verification_attempted=False,
                                        verification_passed=None,
                                        workspace=tmp_path,
                                        fra_result=fra,
                                    )

        # FRA was passed in — should NOT recompute from file
        mock_load.assert_not_called()
        entry = mock_append.call_args[0][1]
        assert entry.get("feedback_routing_advisory") == fra
