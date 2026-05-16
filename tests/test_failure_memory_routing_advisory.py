"""Tests for NativeFailureMemoryRoutingAdvisory.

Covers:
  - Dataclass defaults and advisory_only invariant
  - No raw_content_stored field
  - Builder: no events -> no warnings
  - Builder: model retry failures at/above threshold -> warnings
  - Builder: file path frequency at/above threshold -> warnings
  - Builder: below-threshold cases -> no warnings
  - advisory_only=True always
  - asdict() serialization round-trip (JSON-serializable)
  - Old records without this field render safely
  - Default openshard last output stays clean
  - --more shows compact advisory when warnings present; silent otherwise
  - --full shows model/file detail; calls renderer as single source of truth
  - render_native_failure_memory_routing_advisory function directly
"""
from __future__ import annotations

import json
import unittest
from dataclasses import asdict, fields
from types import SimpleNamespace

import click
from click.testing import CliRunner

from openshard.history.failure_memory import (
    NativeFailureMemoryEvent,
    _event_to_dict,
)
from openshard.native.context import (
    NativeFailureMemoryRoutingAdvisory,
    NativeModelRetryFailureSummary,
    build_native_failure_memory_routing_advisory,
    render_native_failure_memory_routing_advisory,
)
from openshard.cli.run_output import (
    _native_meta_from_entry,
    _render_native_demo_block,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_event(**kwargs) -> NativeFailureMemoryEvent:
    defaults: dict = dict(
        run_id="2026-01-01T00:00:00Z",
        task_summary="fix the bug",
        failure_type="test_failure",
        exit_code=1,
        output_chars=100,
        retry_attempted=True,
        retry_succeeded=False,
        model="claude-sonnet-4-6",
        related_file_paths=[],
    )
    defaults.update(kwargs)
    return NativeFailureMemoryEvent(**defaults)


def _write_events(events: list[NativeFailureMemoryEvent], path) -> None:
    from pathlib import Path
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as fh:
        for evt in events:
            fh.write(json.dumps(_event_to_dict(evt)) + "\n")


def _meta_with_advisory(**kwargs) -> SimpleNamespace:
    """Return a SimpleNamespace with failure_memory_routing_advisory set."""
    adv_fields = dict(
        events_scanned=10,
        model_retry_summaries=[
            SimpleNamespace(
                model="claude-sonnet-4-6",
                failure_count=2,
                failure_types=["test_failure", "test_failure"],
            )
        ],
        hot_file_paths=["openshard/run/pipeline.py"],
        warnings=[
            "model 'claude-sonnet-4-6' has 2 repeated retry failures",
            "file 'openshard/run/pipeline.py' appears in 2 recent failure events",
        ],
        advisory_only=True,
    )
    adv_fields.update(kwargs)
    return SimpleNamespace(
        failure_memory_routing_advisory=SimpleNamespace(**adv_fields)
    )


def _render_entry(entry: dict, detail: str) -> str:
    from openshard.cli.main import _render_log_entry

    @click.command()
    def cmd():
        _render_log_entry(entry, detail)

    return CliRunner().invoke(cmd).output


def _make_run_entry(**kwargs) -> dict:
    base = {
        "task": "test task",
        "timestamp": "2026-01-01T00:00:00Z",
        "execution_model": "claude-sonnet-4-6",
        "duration_seconds": 1.0,
        "files_created": 0,
        "files_updated": 0,
        "files_deleted": 0,
        "retry_triggered": False,
        "verification_attempted": False,
        "verification_passed": None,
        "workspace_path": None,
        "summary": "",
        "workflow": "native",
        "executor": "native",
    }
    base.update(kwargs)
    return base


# ---------------------------------------------------------------------------
# Dataclass defaults
# ---------------------------------------------------------------------------

class TestNativeModelRetryFailureSummaryDefaults(unittest.TestCase):

    def test_model_default(self):
        s = NativeModelRetryFailureSummary()
        self.assertEqual(s.model, "")

    def test_failure_count_default(self):
        s = NativeModelRetryFailureSummary()
        self.assertEqual(s.failure_count, 0)

    def test_failure_types_default_empty(self):
        s = NativeModelRetryFailureSummary()
        self.assertEqual(s.failure_types, [])

    def test_failure_types_independent_instances(self):
        a = NativeModelRetryFailureSummary()
        b = NativeModelRetryFailureSummary()
        a.failure_types.append("x")
        self.assertEqual(b.failure_types, [])


class TestNativeFailureMemoryRoutingAdvisoryDefaults(unittest.TestCase):

    def test_events_scanned_default(self):
        a = NativeFailureMemoryRoutingAdvisory()
        self.assertEqual(a.events_scanned, 0)

    def test_model_retry_summaries_default_empty(self):
        a = NativeFailureMemoryRoutingAdvisory()
        self.assertEqual(a.model_retry_summaries, [])

    def test_hot_file_paths_default_empty(self):
        a = NativeFailureMemoryRoutingAdvisory()
        self.assertEqual(a.hot_file_paths, [])

    def test_warnings_default_empty(self):
        a = NativeFailureMemoryRoutingAdvisory()
        self.assertEqual(a.warnings, [])

    def test_advisory_only_default_true(self):
        a = NativeFailureMemoryRoutingAdvisory()
        self.assertTrue(a.advisory_only)

    def test_raw_content_stored_field_absent(self):
        field_names = {f.name for f in fields(NativeFailureMemoryRoutingAdvisory)}
        self.assertNotIn("raw_content_stored", field_names)

    def test_sub_dataclass_raw_content_stored_absent(self):
        field_names = {f.name for f in fields(NativeModelRetryFailureSummary)}
        self.assertNotIn("raw_content_stored", field_names)

    def test_lists_independent_across_instances(self):
        a = NativeFailureMemoryRoutingAdvisory()
        b = NativeFailureMemoryRoutingAdvisory()
        a.warnings.append("x")
        self.assertEqual(b.warnings, [])


# ---------------------------------------------------------------------------
# Builder: no events
# ---------------------------------------------------------------------------

class TestBuildAdvisoryNoEvents(unittest.TestCase):

    def setUp(self):
        self._runner = CliRunner()
        self._td = self._runner.isolated_filesystem()
        self._td.__enter__()

    def tearDown(self):
        self._td.__exit__(None, None, None)

    def test_returns_advisory_instance(self):
        adv = build_native_failure_memory_routing_advisory()
        self.assertIsInstance(adv, NativeFailureMemoryRoutingAdvisory)

    def test_events_scanned_zero(self):
        adv = build_native_failure_memory_routing_advisory()
        self.assertEqual(adv.events_scanned, 0)

    def test_no_warnings(self):
        adv = build_native_failure_memory_routing_advisory()
        self.assertEqual(adv.warnings, [])

    def test_no_model_retry_summaries(self):
        adv = build_native_failure_memory_routing_advisory()
        self.assertEqual(adv.model_retry_summaries, [])

    def test_no_hot_file_paths(self):
        adv = build_native_failure_memory_routing_advisory()
        self.assertEqual(adv.hot_file_paths, [])

    def test_advisory_only_true(self):
        adv = build_native_failure_memory_routing_advisory()
        self.assertTrue(adv.advisory_only)


# ---------------------------------------------------------------------------
# Builder: model retry failures
# ---------------------------------------------------------------------------

class TestBuildAdvisoryModelFailures(unittest.TestCase):

    def setUp(self):
        self._runner = CliRunner()
        self._td = self._runner.isolated_filesystem()
        self._td.__enter__()
        from pathlib import Path
        self._path = Path(".openshard") / "failure_memory.jsonl"

    def tearDown(self):
        self._td.__exit__(None, None, None)

    def test_single_retry_failure_no_warning(self):
        _write_events([_make_event(model="model-a", retry_attempted=True)], self._path)
        adv = build_native_failure_memory_routing_advisory()
        self.assertEqual(adv.warnings, [])

    def test_two_retry_failures_same_model_produces_warning(self):
        events = [
            _make_event(model="model-a", retry_attempted=True),
            _make_event(model="model-a", retry_attempted=True),
        ]
        _write_events(events, self._path)
        adv = build_native_failure_memory_routing_advisory()
        self.assertEqual(len([w for w in adv.warnings if "model-a" in w]), 1)

    def test_warning_text_contains_model_name(self):
        events = [_make_event(model="my-model", retry_attempted=True)] * 2
        _write_events(events, self._path)
        adv = build_native_failure_memory_routing_advisory()
        self.assertTrue(any("my-model" in w for w in adv.warnings))

    def test_warning_text_contains_failure_count(self):
        events = [_make_event(model="model-x", retry_attempted=True)] * 3
        _write_events(events, self._path)
        adv = build_native_failure_memory_routing_advisory()
        model_warnings = [w for w in adv.warnings if "model-x" in w]
        self.assertTrue(any("3" in w for w in model_warnings))

    def test_three_retry_failures_still_one_warning_per_model(self):
        events = [_make_event(model="model-a", retry_attempted=True)] * 3
        _write_events(events, self._path)
        adv = build_native_failure_memory_routing_advisory()
        model_warnings = [w for w in adv.warnings if "model-a" in w]
        self.assertEqual(len(model_warnings), 1)

    def test_two_models_both_at_threshold_two_model_warnings(self):
        events = (
            [_make_event(model="model-a", retry_attempted=True)] * 2
            + [_make_event(model="model-b", retry_attempted=True)] * 2
        )
        _write_events(events, self._path)
        adv = build_native_failure_memory_routing_advisory()
        model_warnings = [w for w in adv.warnings if "model" in w and "repeated" in w]
        self.assertEqual(len(model_warnings), 2)

    def test_retry_not_attempted_events_not_counted(self):
        events = [
            _make_event(model="model-a", retry_attempted=False),
            _make_event(model="model-a", retry_attempted=False),
        ]
        _write_events(events, self._path)
        adv = build_native_failure_memory_routing_advisory()
        self.assertEqual(adv.warnings, [])

    def test_empty_model_field_ignored(self):
        events = [
            _make_event(model="", retry_attempted=True),
            _make_event(model="", retry_attempted=True),
        ]
        _write_events(events, self._path)
        adv = build_native_failure_memory_routing_advisory()
        self.assertEqual(adv.model_retry_summaries, [])
        self.assertEqual(adv.warnings, [])

    def test_model_retry_summaries_populated_at_threshold(self):
        events = [_make_event(model="model-z", retry_attempted=True)] * 2
        _write_events(events, self._path)
        adv = build_native_failure_memory_routing_advisory()
        models = [s.model for s in adv.model_retry_summaries]
        self.assertIn("model-z", models)

    def test_failure_types_captured_in_summary(self):
        events = [
            _make_event(model="model-a", retry_attempted=True, failure_type="test_failure"),
            _make_event(model="model-a", retry_attempted=True, failure_type="syntax_error"),
        ]
        _write_events(events, self._path)
        adv = build_native_failure_memory_routing_advisory()
        summaries = {s.model: s for s in adv.model_retry_summaries}
        self.assertIn("model-a", summaries)
        ftypes = summaries["model-a"].failure_types
        self.assertIn("test_failure", ftypes)
        self.assertIn("syntax_error", ftypes)

    def test_mixed_retry_only_retried_counted(self):
        events = [
            _make_event(model="model-a", retry_attempted=True),
            _make_event(model="model-a", retry_attempted=False),
        ]
        _write_events(events, self._path)
        adv = build_native_failure_memory_routing_advisory()
        summaries = {s.model: s for s in adv.model_retry_summaries}
        if "model-a" in summaries:
            self.assertEqual(summaries["model-a"].failure_count, 1)
        self.assertEqual(adv.warnings, [])


# ---------------------------------------------------------------------------
# Builder: file path frequency
# ---------------------------------------------------------------------------

class TestBuildAdvisoryFileFrequency(unittest.TestCase):

    def setUp(self):
        self._runner = CliRunner()
        self._td = self._runner.isolated_filesystem()
        self._td.__enter__()
        from pathlib import Path
        self._path = Path(".openshard") / "failure_memory.jsonl"

    def tearDown(self):
        self._td.__exit__(None, None, None)

    def test_single_file_appearance_no_warning(self):
        _write_events(
            [_make_event(related_file_paths=["foo.py"])], self._path
        )
        adv = build_native_failure_memory_routing_advisory()
        file_warnings = [w for w in adv.warnings if "foo.py" in w]
        self.assertEqual(file_warnings, [])

    def test_two_appearances_same_file_produces_warning(self):
        events = [
            _make_event(related_file_paths=["bar.py"]),
            _make_event(related_file_paths=["bar.py"]),
        ]
        _write_events(events, self._path)
        adv = build_native_failure_memory_routing_advisory()
        self.assertTrue(any("bar.py" in w for w in adv.warnings))

    def test_warning_text_contains_file_path(self):
        events = [_make_event(related_file_paths=["src/main.py"])] * 2
        _write_events(events, self._path)
        adv = build_native_failure_memory_routing_advisory()
        self.assertTrue(any("src/main.py" in w for w in adv.warnings))

    def test_warning_text_contains_appearance_count(self):
        events = [_make_event(related_file_paths=["x.py"])] * 3
        _write_events(events, self._path)
        adv = build_native_failure_memory_routing_advisory()
        file_warnings = [w for w in adv.warnings if "x.py" in w]
        self.assertTrue(any("3" in w for w in file_warnings))

    def test_file_in_hot_file_paths(self):
        events = [_make_event(related_file_paths=["hot.py"])] * 2
        _write_events(events, self._path)
        adv = build_native_failure_memory_routing_advisory()
        self.assertIn("hot.py", adv.hot_file_paths)

    def test_file_below_threshold_not_in_hot_files(self):
        _write_events([_make_event(related_file_paths=["cold.py"])], self._path)
        adv = build_native_failure_memory_routing_advisory()
        self.assertNotIn("cold.py", adv.hot_file_paths)

    def test_empty_related_file_paths_ignored(self):
        events = [_make_event(related_file_paths=[])] * 3
        _write_events(events, self._path)
        adv = build_native_failure_memory_routing_advisory()
        self.assertEqual(adv.hot_file_paths, [])

    def test_multiple_files_per_event_all_counted(self):
        _write_events(
            [_make_event(related_file_paths=["a.py", "b.py", "c.py"])], self._path
        )
        adv = build_native_failure_memory_routing_advisory()
        self.assertEqual(adv.hot_file_paths, [])

    def test_retry_not_attempted_events_still_count_for_files(self):
        events = [
            _make_event(retry_attempted=False, related_file_paths=["x.py"]),
            _make_event(retry_attempted=False, related_file_paths=["x.py"]),
        ]
        _write_events(events, self._path)
        adv = build_native_failure_memory_routing_advisory()
        self.assertIn("x.py", adv.hot_file_paths)

    def test_advisory_only_always_true_with_file_warnings(self):
        events = [_make_event(related_file_paths=["z.py"])] * 2
        _write_events(events, self._path)
        adv = build_native_failure_memory_routing_advisory()
        self.assertTrue(adv.advisory_only)


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------

class TestNativeFailureMemoryRoutingAdvisorySerialization(unittest.TestCase):

    def setUp(self):
        self._runner = CliRunner()
        self._td = self._runner.isolated_filesystem()
        self._td.__enter__()

    def tearDown(self):
        self._td.__exit__(None, None, None)

    def test_asdict_json_serializable_no_warnings(self):
        adv = build_native_failure_memory_routing_advisory()
        d = asdict(adv)
        json.dumps(d)

    def test_asdict_json_serializable_with_summaries(self):
        adv = NativeFailureMemoryRoutingAdvisory(
            events_scanned=5,
            model_retry_summaries=[
                NativeModelRetryFailureSummary(
                    model="m1", failure_count=2, failure_types=["test_failure"]
                )
            ],
            hot_file_paths=["a.py"],
            warnings=["model 'm1' has 2 repeated retry failures"],
            advisory_only=True,
        )
        d = asdict(adv)
        json.dumps(d)

    def test_asdict_expected_keys_present(self):
        d = asdict(NativeFailureMemoryRoutingAdvisory())
        for key in ("events_scanned", "model_retry_summaries", "hot_file_paths", "warnings", "advisory_only"):
            self.assertIn(key, d)

    def test_asdict_raw_content_stored_absent(self):
        d = asdict(NativeFailureMemoryRoutingAdvisory())
        self.assertNotIn("raw_content_stored", d)

    def test_advisory_only_always_true_in_dict(self):
        adv = build_native_failure_memory_routing_advisory()
        self.assertTrue(asdict(adv)["advisory_only"])

    def test_nested_model_retry_summaries_serialized(self):
        adv = NativeFailureMemoryRoutingAdvisory(
            model_retry_summaries=[
                NativeModelRetryFailureSummary(model="m", failure_count=3, failure_types=["e"])
            ]
        )
        d = asdict(adv)
        self.assertEqual(len(d["model_retry_summaries"]), 1)
        self.assertEqual(d["model_retry_summaries"][0]["model"], "m")
        self.assertEqual(d["model_retry_summaries"][0]["failure_count"], 3)

    def test_json_dumps_nested_does_not_raise(self):
        adv = NativeFailureMemoryRoutingAdvisory(
            model_retry_summaries=[
                NativeModelRetryFailureSummary(model="x", failure_count=2, failure_types=["t"])
            ],
            hot_file_paths=["f.py"],
            warnings=["w1"],
        )
        json.dumps(asdict(adv))


# ---------------------------------------------------------------------------
# Backward compatibility: old records
# ---------------------------------------------------------------------------

class TestNativeFailureMemoryRoutingAdvisoryBackwardCompat(unittest.TestCase):

    def _old_native_entry(self, **extra) -> dict:
        base = _make_run_entry()
        base.pop("failure_memory_routing_advisory", None)
        base.update(extra)
        return base

    def test_old_entry_without_advisory_key_no_crash(self):
        entry = self._old_native_entry()
        self.assertNotIn("failure_memory_routing_advisory", entry)
        meta = _native_meta_from_entry(entry)
        advisory = getattr(meta, "failure_memory_routing_advisory", None)
        self.assertIsNone(advisory)

    def test_render_demo_block_none_advisory_no_crash(self):
        meta = SimpleNamespace(failure_memory_routing_advisory=None)
        lines = _render_native_demo_block(meta, detail="more")
        combined = "\n".join(lines)
        self.assertNotIn("failure memory advisory", combined)

    def test_render_demo_block_missing_attribute_no_crash(self):
        meta = SimpleNamespace()
        lines = _render_native_demo_block(meta, detail="full")
        self.assertIsInstance(lines, list)

    def test_render_demo_block_old_entry_no_advisory_text(self):
        meta = _native_meta_from_entry(self._old_native_entry())
        lines = _render_native_demo_block(meta, detail="full")
        combined = "\n".join(lines)
        self.assertNotIn("failure memory advisory", combined)
        self.assertNotIn("failure memory routing advisory", combined)


# ---------------------------------------------------------------------------
# Default output stays clean
# ---------------------------------------------------------------------------

class TestNativeFailureMemoryRoutingAdvisoryDefaultOutput(unittest.TestCase):

    def test_default_detail_no_advisory(self):
        entry = _make_run_entry(
            failure_memory_routing_advisory={
                "events_scanned": 5,
                "model_retry_summaries": [],
                "hot_file_paths": [],
                "warnings": ["model 'x' has 2 repeated retry failures"],
                "advisory_only": True,
            }
        )
        out = _render_entry(entry, detail="default")
        self.assertNotIn("failure memory advisory", out)
        self.assertNotIn("advisory", out.lower())

    def test_no_advisory_key_default_detail_no_crash(self):
        entry = _make_run_entry()
        out = _render_entry(entry, detail="default")
        self.assertNotIn("failure memory advisory", out)


# ---------------------------------------------------------------------------
# --more rendering
# ---------------------------------------------------------------------------

class TestNativeFailureMemoryRoutingAdvisoryMoreRendering(unittest.TestCase):

    def test_more_shows_compact_summary_when_warnings_present(self):
        meta = _meta_with_advisory()
        lines = _render_native_demo_block(meta, detail="more")
        combined = "\n".join(lines)
        self.assertIn("failure memory advisory", combined)
        self.assertIn("2 warnings", combined)

    def test_more_does_not_show_model_detail(self):
        meta = _meta_with_advisory()
        lines = _render_native_demo_block(meta, detail="more")
        combined = "\n".join(lines)
        self.assertNotIn("claude-sonnet-4-6", combined)
        self.assertNotIn("pipeline.py", combined)

    def test_more_silent_when_no_warnings(self):
        meta = SimpleNamespace(
            failure_memory_routing_advisory=SimpleNamespace(
                events_scanned=3,
                model_retry_summaries=[],
                hot_file_paths=[],
                warnings=[],
                advisory_only=True,
            )
        )
        lines = _render_native_demo_block(meta, detail="more")
        combined = "\n".join(lines)
        self.assertNotIn("failure memory advisory", combined)

    def test_more_does_not_expose_raw_warning_text(self):
        meta = _meta_with_advisory()
        lines = _render_native_demo_block(meta, detail="more")
        combined = "\n".join(lines)
        self.assertNotIn("repeated retry failures", combined)
        self.assertNotIn("appears in", combined)

    def test_more_singular_warning_word(self):
        meta = SimpleNamespace(
            failure_memory_routing_advisory=SimpleNamespace(
                events_scanned=5,
                model_retry_summaries=[],
                hot_file_paths=[],
                warnings=["single warning"],
                advisory_only=True,
            )
        )
        lines = _render_native_demo_block(meta, detail="more")
        combined = "\n".join(lines)
        self.assertIn("1 warning", combined)
        self.assertNotIn("1 warnings", combined)


# ---------------------------------------------------------------------------
# --full rendering
# ---------------------------------------------------------------------------

class TestNativeFailureMemoryRoutingAdvisoryFullRendering(unittest.TestCase):

    def test_full_shows_section_header(self):
        meta = _meta_with_advisory()
        lines = _render_native_demo_block(meta, detail="full")
        combined = "\n".join(lines)
        self.assertIn("[failure memory routing advisory]", combined)

    def test_full_shows_events_scanned(self):
        meta = _meta_with_advisory(events_scanned=7)
        lines = _render_native_demo_block(meta, detail="full")
        combined = "\n".join(lines)
        self.assertIn("events_scanned: 7", combined)

    def test_full_shows_model_retry_section(self):
        meta = _meta_with_advisory()
        lines = _render_native_demo_block(meta, detail="full")
        combined = "\n".join(lines)
        self.assertIn("model retry failures:", combined)

    def test_full_shows_model_name(self):
        meta = _meta_with_advisory()
        lines = _render_native_demo_block(meta, detail="full")
        combined = "\n".join(lines)
        self.assertIn("claude-sonnet-4-6", combined)

    def test_full_shows_hot_files_section(self):
        meta = _meta_with_advisory()
        lines = _render_native_demo_block(meta, detail="full")
        combined = "\n".join(lines)
        self.assertIn("hot files:", combined)

    def test_full_shows_hot_file_path(self):
        meta = _meta_with_advisory()
        lines = _render_native_demo_block(meta, detail="full")
        combined = "\n".join(lines)
        self.assertIn("pipeline.py", combined)

    def test_full_shows_warning_count(self):
        meta = _meta_with_advisory()
        lines = _render_native_demo_block(meta, detail="full")
        combined = "\n".join(lines)
        self.assertIn("warnings: 2", combined)

    def test_full_dict_advisory_renders_without_crash(self):
        advisory_dict = {
            "events_scanned": 8,
            "model_retry_summaries": [
                {"model": "claude-opus-4-5", "failure_count": 3, "failure_types": ["test_failure"]}
            ],
            "hot_file_paths": ["src/main.py"],
            "warnings": ["model 'claude-opus-4-5' has 3 repeated retry failures"],
            "advisory_only": True,
        }
        meta = SimpleNamespace(failure_memory_routing_advisory=advisory_dict)
        lines = _render_native_demo_block(meta, detail="full")
        combined = "\n".join(lines)
        self.assertIn("failure memory advisory", combined)

    def test_full_no_warnings_silent_in_demo_block(self):
        meta = SimpleNamespace(
            failure_memory_routing_advisory=SimpleNamespace(
                events_scanned=3,
                model_retry_summaries=[],
                hot_file_paths=[],
                warnings=[],
                advisory_only=True,
            )
        )
        lines = _render_native_demo_block(meta, detail="full")
        combined = "\n".join(lines)
        self.assertNotIn("failure memory advisory", combined)
        self.assertNotIn("[failure memory routing advisory]", combined)


# ---------------------------------------------------------------------------
# render_native_failure_memory_routing_advisory function
# ---------------------------------------------------------------------------

class TestRenderNativeFailureMemoryRoutingAdvisoryFunction(unittest.TestCase):

    def test_none_returns_empty_string(self):
        self.assertEqual(render_native_failure_memory_routing_advisory(None), "")

    def test_none_compact_returns_empty_string(self):
        self.assertEqual(render_native_failure_memory_routing_advisory(None, detail="compact"), "")

    def test_none_full_returns_empty_string(self):
        self.assertEqual(render_native_failure_memory_routing_advisory(None, detail="full"), "")

    def test_no_warnings_compact_returns_empty(self):
        adv = NativeFailureMemoryRoutingAdvisory(warnings=[])
        self.assertEqual(render_native_failure_memory_routing_advisory(adv, detail="compact"), "")

    def test_no_warnings_full_returns_block_with_zero_warnings(self):
        adv = NativeFailureMemoryRoutingAdvisory(events_scanned=5, warnings=[])
        out = render_native_failure_memory_routing_advisory(adv, detail="full")
        self.assertIn("[failure memory routing advisory]", out)
        self.assertIn("warnings: 0", out)
        self.assertIn("events_scanned: 5", out)

    def test_compact_one_warning_singular(self):
        adv = NativeFailureMemoryRoutingAdvisory(warnings=["w1"])
        out = render_native_failure_memory_routing_advisory(adv, detail="compact")
        self.assertIn("1 warning", out)
        self.assertNotIn("1 warnings", out)

    def test_compact_two_warnings_plural(self):
        adv = NativeFailureMemoryRoutingAdvisory(warnings=["w1", "w2"])
        out = render_native_failure_memory_routing_advisory(adv, detail="compact")
        self.assertIn("2 warnings", out)

    def test_compact_does_not_expose_raw_warning_text(self):
        adv = NativeFailureMemoryRoutingAdvisory(warnings=["secret warning content"])
        out = render_native_failure_memory_routing_advisory(adv, detail="compact")
        self.assertNotIn("secret warning content", out)

    def test_full_includes_model_detail(self):
        adv = NativeFailureMemoryRoutingAdvisory(
            events_scanned=4,
            model_retry_summaries=[
                NativeModelRetryFailureSummary(
                    model="model-x", failure_count=2, failure_types=["test_failure"]
                )
            ],
            warnings=["model 'model-x' has 2 repeated retry failures"],
        )
        out = render_native_failure_memory_routing_advisory(adv, detail="full")
        self.assertIn("model-x", out)
        self.assertIn("2 retries", out)

    def test_full_includes_file_detail(self):
        adv = NativeFailureMemoryRoutingAdvisory(
            events_scanned=2,
            hot_file_paths=["special/file.py"],
            warnings=["file 'special/file.py' appears in 2 recent failure events"],
        )
        out = render_native_failure_memory_routing_advisory(adv, detail="full")
        self.assertIn("hot files:", out)
        self.assertIn("special/file.py", out)

    def test_full_warning_count_matches_list_length(self):
        warnings = ["w1", "w2", "w3"]
        adv = NativeFailureMemoryRoutingAdvisory(warnings=warnings)
        out = render_native_failure_memory_routing_advisory(adv, detail="full")
        self.assertIn("warnings: 3", out)

    def test_full_dict_style_advisory_no_crash(self):
        advisory_dict = {
            "events_scanned": 6,
            "model_retry_summaries": [
                {"model": "m", "failure_count": 2, "failure_types": ["x"]}
            ],
            "hot_file_paths": [],
            "warnings": ["w"],
            "advisory_only": True,
        }
        out = render_native_failure_memory_routing_advisory(advisory_dict, detail="full")
        self.assertIn("[failure memory routing advisory]", out)

    def test_failure_types_truncated_to_three_in_full(self):
        adv = NativeFailureMemoryRoutingAdvisory(
            events_scanned=6,
            model_retry_summaries=[
                NativeModelRetryFailureSummary(
                    model="m",
                    failure_count=6,
                    failure_types=["a", "b", "c", "d", "e", "f"],
                )
            ],
            warnings=["model 'm' has 6 repeated retry failures"],
        )
        out = render_native_failure_memory_routing_advisory(adv, detail="full")
        self.assertIn("a, b, c", out)
        self.assertNotIn("d, e", out)


if __name__ == "__main__":
    unittest.main()
