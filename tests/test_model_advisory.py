from __future__ import annotations

import unittest

from click.testing import CliRunner

from openshard.cli.main import cli
from openshard.models.advisory import ModelAdvisory, build_advisory_for_storage, recommend_models


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
            limit=25,
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


class TestBuildAdvisoryForStorage(unittest.TestCase):
    def test_high_risk_returns_candidates(self):
        candidates, _ = build_advisory_for_storage(risk="high")
        self.assertGreater(len(candidates), 0)

    def test_high_risk_max_3_candidates(self):
        candidates, _ = build_advisory_for_storage(risk="high")
        self.assertLessEqual(len(candidates), 3)

    def test_low_risk_returns_candidates(self):
        candidates, _ = build_advisory_for_storage(risk="low")
        self.assertGreater(len(candidates), 0)

    def test_medium_risk_returns_candidates(self):
        candidates, _ = build_advisory_for_storage(risk="medium")
        self.assertGreater(len(candidates), 0)

    def test_unknown_risk_returns_empty_candidates(self):
        candidates, meta = build_advisory_for_storage(risk="unknown")
        self.assertEqual(candidates, [])
        self.assertIsInstance(meta, dict)

    def test_none_risk_returns_empty_candidates(self):
        candidates, meta = build_advisory_for_storage(risk=None)
        self.assertEqual(candidates, [])
        self.assertIsInstance(meta, dict)

    def test_meta_has_version_rules_v1(self):
        _, meta = build_advisory_for_storage(risk="high")
        self.assertEqual(meta["version"], "rules_v1")

    def test_meta_advisory_only_is_true(self):
        _, meta = build_advisory_for_storage(risk="high")
        self.assertIs(meta["advisory_only"], True)

    def test_meta_records_risk_input(self):
        _, meta = build_advisory_for_storage(risk="high")
        self.assertEqual(meta["risk"], "high")

    def test_meta_none_risk_records_none(self):
        _, meta = build_advisory_for_storage(risk=None)
        self.assertIsNone(meta["risk"])

    def test_meta_has_role_none(self):
        _, meta = build_advisory_for_storage(risk="high")
        self.assertIsNone(meta["role"])

    def test_meta_has_required_capabilities_empty(self):
        _, meta = build_advisory_for_storage(risk="high")
        self.assertEqual(meta["required_capabilities"], [])

    def test_result_structure_has_required_fields(self):
        candidates, _ = build_advisory_for_storage(risk="high")
        required = {"model_id", "display_name", "tier", "cost_class", "experimental", "reasons"}
        for c in candidates:
            self.assertEqual(required, required & c.keys(), f"Missing fields in {c}")

    def test_reasons_is_list_of_strings(self):
        candidates, _ = build_advisory_for_storage(risk="high")
        for c in candidates:
            self.assertIsInstance(c["reasons"], list)
            for r in c["reasons"]:
                self.assertIsInstance(r, str)

    def test_experimental_false_by_default(self):
        candidates, _ = build_advisory_for_storage(risk="high")
        for c in candidates:
            self.assertFalse(
                c["experimental"],
                f"Model {c['model_id']} is experimental but should be excluded by default",
            )

    def test_result_is_json_serializable(self):
        import json
        candidates, meta = build_advisory_for_storage(risk="high")
        json.dumps({"candidates": candidates, "meta": meta})


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


def _sig(signal_type: str) -> dict:
    return {"signal_type": signal_type, "schema_version": 1, "source": "rules_v1"}


class TestFeedbackRoutingAdvisory(unittest.TestCase):

    def setUp(self):
        from openshard.models.feedback_advisory import build_feedback_routing_advisory
        self.build = build_feedback_routing_advisory

    # -- pure helper: negative signals trigger advisory --

    def test_retry_requested_gives_low_confidence(self):
        result = self.build([_sig("retry_requested")])
        self.assertIsNotNone(result)
        self.assertEqual(result["confidence"], "low")
        self.assertEqual(result["recommendation"], "consider_stronger_review")

    def test_partial_explicit_gives_medium_confidence(self):
        result = self.build([_sig("partial_explicit")])
        self.assertIsNotNone(result)
        self.assertEqual(result["confidence"], "medium")

    def test_rejected_explicit_gives_medium_confidence(self):
        result = self.build([_sig("rejected_explicit")])
        self.assertIsNotNone(result)
        self.assertEqual(result["confidence"], "medium")

    def test_partial_and_rejected_both_present_reason_includes_both(self):
        signals = [_sig("partial_explicit"), _sig("rejected_explicit")]
        result = self.build(signals)
        self.assertIsNotNone(result)
        self.assertIn("partial feedback", result["reason"])
        self.assertIn("rejected feedback", result["reason"])
        self.assertEqual(result["confidence"], "medium")

    # -- pure helper: non-negative signals produce no advisory --

    def test_accepted_explicit_alone_returns_none(self):
        self.assertIsNone(self.build([_sig("accepted_explicit")]))

    def test_inspected_result_alone_returns_none(self):
        self.assertIsNone(self.build([_sig("inspected_result")]))

    def test_continued_session_alone_returns_none(self):
        self.assertIsNone(self.build([_sig("continued_session")]))

    def test_empty_list_returns_none(self):
        self.assertIsNone(self.build([]))

    # -- counts and metadata --

    def test_multiple_negative_signals_counted_correctly(self):
        signals = [
            _sig("retry_requested"),
            _sig("retry_requested"),
            _sig("partial_explicit"),
        ]
        result = self.build(signals)
        self.assertIsNotNone(result)
        self.assertEqual(result["signals_considered"]["retry_requested"], 2)
        self.assertEqual(result["signals_considered"]["partial_explicit"], 1)
        self.assertEqual(result["signals_considered"]["rejected_explicit"], 0)

    def test_advisory_only_always_true(self):
        result = self.build([_sig("retry_requested")])
        self.assertIs(result["advisory_only"], True)

    def test_version_always_rules_v1(self):
        result = self.build([_sig("partial_explicit")])
        self.assertEqual(result["version"], "rules_v1")

    def test_reason_uses_recent_local_wording(self):
        result = self.build([_sig("retry_requested")])
        self.assertIn("Recent local session signals", result["reason"])

    def test_signals_window_keys_present(self):
        result = self.build([_sig("partial_explicit")])
        w = result["signals_window"]
        self.assertIn("source", w)
        self.assertIn("max_recent_signals", w)
        self.assertIn("signals_read", w)
        self.assertIn("signals_used", w)
        self.assertEqual(w["source"], "session_signals.jsonl")

    def test_signals_window_counts_match(self):
        signals = [_sig("partial_explicit"), _sig("accepted_explicit")]
        result = self.build(signals)
        self.assertIsNotNone(result)
        self.assertEqual(result["signals_window"]["signals_read"], 2)
        self.assertEqual(result["signals_window"]["signals_used"], 1)

    # -- _load_recent_session_signals: file I/O helpers --

    def test_load_missing_file_returns_empty(self):
        import tempfile
        from pathlib import Path

        from openshard.models.feedback_advisory import _load_recent_session_signals
        with tempfile.TemporaryDirectory() as d:
            result = _load_recent_session_signals(Path(d) / "nope.jsonl")
        self.assertEqual(result, [])

    def test_load_invalid_json_lines_skipped(self):
        import json
        import tempfile
        from pathlib import Path

        from openshard.models.feedback_advisory import _load_recent_session_signals
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "signals.jsonl"
            p.write_text(
                'not json\n'
                + json.dumps(_sig("retry_requested")) + '\n'
                + '{broken\n',
                encoding="utf-8",
            )
            result = _load_recent_session_signals(p)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["signal_type"], "retry_requested")

    def test_load_returns_last_25_of_30(self):
        import json
        import tempfile
        from pathlib import Path

        from openshard.models.feedback_advisory import _load_recent_session_signals
        signals = []
        for i in range(30):
            s = _sig("accepted_explicit")
            s["_idx"] = i
            signals.append(s)
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "signals.jsonl"
            p.write_text("\n".join(json.dumps(s) for s in signals) + "\n", encoding="utf-8")
            result = _load_recent_session_signals(p, limit=25)
        self.assertEqual(len(result), 25)
        self.assertEqual(result[0]["_idx"], 5)
        self.assertEqual(result[-1]["_idx"], 29)

    def test_load_drops_unknown_signal_types(self):
        import json
        import tempfile
        from pathlib import Path

        from openshard.models.feedback_advisory import _load_recent_session_signals
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "signals.jsonl"
            p.write_text(
                json.dumps({"signal_type": "mystery_type"}) + "\n"
                + json.dumps(_sig("retry_requested")) + "\n",
                encoding="utf-8",
            )
            result = _load_recent_session_signals(p)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["signal_type"], "retry_requested")

    # -- bounded window: old signals outside window are ignored --

    def test_old_signals_outside_window_ignored(self):
        # 25 accepted signals fill the window; the older rejected is at position 26 (not passed in)
        accepted_window = [_sig("accepted_explicit") for _ in range(25)]
        # Simulate what _load_recent_session_signals returns: only the last 25 (all accepted)
        result = self.build(accepted_window)
        self.assertIsNone(result)

    def test_recent_signals_within_window_trigger_advisory(self):
        signals = [_sig("accepted_explicit")] * 20 + [_sig("rejected_explicit")] * 2
        result = self.build(signals)
        self.assertIsNotNone(result)
        self.assertEqual(result["confidence"], "medium")
