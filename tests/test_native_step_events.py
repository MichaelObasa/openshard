from __future__ import annotations

import json
import unittest
from pathlib import Path

import click
from click.testing import CliRunner

from openshard.cli.main import _render_log_entry
from openshard.history.native_steps import (
    NativeStepEvent,
    _dict_to_event,
    _event_to_dict,
    load_native_step_events,
    log_native_step_event,
    native_step_events_for_run,
)
from openshard.native.executor import _step_to_stage
from openshard.security.paths import UnsafePathError, resolve_safe_repo_path


def _make_event(**kwargs) -> NativeStepEvent:
    defaults: dict = dict(
        run_id="2025-01-01T00:00:00Z",
        step_name="preflight",
        status="passed",
    )
    defaults.update(kwargs)
    return NativeStepEvent(**defaults)


def _write_steps(events: list[NativeStepEvent]) -> Path:
    path = Path(".openshard") / "native_steps.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for evt in events:
            fh.write(json.dumps(_event_to_dict(evt)) + "\n")
    return path


def _render(entry: dict, detail: str = "more") -> str:
    @click.command()
    def cmd():
        _render_log_entry(entry, detail)

    return CliRunner().invoke(cmd).output


def _make_run_entry(**kwargs) -> dict:
    base = {
        "task": "test task",
        "timestamp": "2025-01-01T00:00:00Z",
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
        "native": {},
    }
    base.update(kwargs)
    return base


class TestNativeStepEventDataclass(unittest.TestCase):

    def test_defaults_are_correct(self):
        evt = NativeStepEvent()
        self.assertEqual(evt.schema_version, 1)
        self.assertEqual(evt.run_id, "")
        self.assertEqual(evt.step_index, 0)
        self.assertEqual(evt.step_name, "")
        self.assertEqual(evt.stage, "")
        self.assertEqual(evt.status, "")
        self.assertEqual(evt.summary, "")
        self.assertEqual(evt.tool_name, "")
        self.assertEqual(evt.policy_decision, "")
        self.assertIs(evt.approval_required, False)
        self.assertIsNone(evt.approval_granted)
        self.assertEqual(evt.verification_status, "")
        self.assertEqual(evt.retry_count, 0)
        self.assertIsNone(evt.duration_ms)
        self.assertEqual(evt.metadata, {})
        self.assertIs(evt.raw_content_stored, False)

    def test_post_init_enforces_raw_content_stored_false(self):
        evt = NativeStepEvent(raw_content_stored=True)
        self.assertIs(evt.raw_content_stored, False)

    def test_event_id_is_unique(self):
        a = NativeStepEvent()
        b = NativeStepEvent()
        self.assertNotEqual(a.event_id, b.event_id)
        self.assertTrue(len(a.event_id) > 0)

    def test_timestamp_format(self):
        evt = NativeStepEvent()
        self.assertRegex(evt.timestamp, r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")


class TestNativeStepEventSerialization(unittest.TestCase):

    def test_round_trip_all_fields(self):
        evt = NativeStepEvent(
            run_id="run-xyz",
            step_index=3,
            step_name="verify",
            stage="verification",
            status="passed",
            summary="tests passed",
            tool_name="run_verification",
            policy_decision="allowed",
            approval_required=True,
            approval_granted=True,
            verification_status="passed",
            retry_count=1,
            duration_ms=420,
            metadata={"exit_code": 0},
        )
        d = _event_to_dict(evt)
        roundtrip = _dict_to_event(d)
        self.assertEqual(roundtrip.run_id, "run-xyz")
        self.assertEqual(roundtrip.step_index, 3)
        self.assertEqual(roundtrip.step_name, "verify")
        self.assertEqual(roundtrip.stage, "verification")
        self.assertEqual(roundtrip.status, "passed")
        self.assertEqual(roundtrip.summary, "tests passed")
        self.assertEqual(roundtrip.tool_name, "run_verification")
        self.assertEqual(roundtrip.policy_decision, "allowed")
        self.assertIs(roundtrip.approval_required, True)
        self.assertIs(roundtrip.approval_granted, True)
        self.assertEqual(roundtrip.verification_status, "passed")
        self.assertEqual(roundtrip.retry_count, 1)
        self.assertEqual(roundtrip.duration_ms, 420)
        self.assertEqual(roundtrip.metadata, {"exit_code": 0})
        self.assertIs(roundtrip.raw_content_stored, False)

    def test_all_required_keys_present(self):
        d = _event_to_dict(_make_event())
        for key in (
            "schema_version", "event_id", "run_id", "timestamp",
            "step_index", "step_name", "stage", "status", "summary",
            "tool_name", "policy_decision", "approval_required", "approval_granted",
            "verification_status", "retry_count", "duration_ms", "metadata",
            "raw_content_stored",
        ):
            self.assertIn(key, d, f"missing key: {key}")

    def test_raw_content_stored_always_false_in_dict(self):
        evt = NativeStepEvent(raw_content_stored=True)
        d = _event_to_dict(evt)
        self.assertIs(d["raw_content_stored"], False)

    def test_dict_to_event_ignores_raw_content_stored_true(self):
        d = _event_to_dict(_make_event())
        d["raw_content_stored"] = True
        evt = _dict_to_event(d)
        self.assertIs(evt.raw_content_stored, False)


class TestNativeStepEventLogging(unittest.TestCase):

    def test_append_only_logging(self):
        runner = CliRunner()
        with runner.isolated_filesystem():
            log_native_step_event(_make_event(step_name="preflight"))
            log_native_step_event(_make_event(step_name="observe"))
            log_native_step_event(_make_event(step_name="gather_context"))
            path = Path(".openshard") / "native_steps.jsonl"
            lines = [ln for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
            self.assertEqual(len(lines), 3)
            names = [json.loads(ln)["step_name"] for ln in lines]
            self.assertEqual(names, ["preflight", "observe", "gather_context"])

    def test_append_does_not_overwrite(self):
        runner = CliRunner()
        with runner.isolated_filesystem():
            log_native_step_event(_make_event(step_name="first"))
            log_native_step_event(_make_event(step_name="second"))
            events = load_native_step_events()
            self.assertEqual(events[0].step_name, "first")
            self.assertEqual(events[1].step_name, "second")

    def test_directory_created_automatically(self):
        runner = CliRunner()
        with runner.isolated_filesystem():
            self.assertFalse(Path(".openshard").exists())
            log_native_step_event(_make_event())
            self.assertTrue((Path(".openshard") / "native_steps.jsonl").exists())

    def test_stored_json_has_raw_content_stored_false(self):
        runner = CliRunner()
        with runner.isolated_filesystem():
            evt = NativeStepEvent(raw_content_stored=True)
            log_native_step_event(evt)
            path = Path(".openshard") / "native_steps.jsonl"
            stored = json.loads(path.read_text(encoding="utf-8").strip())
            self.assertIs(stored["raw_content_stored"], False)


class TestLoadNativeStepEvents(unittest.TestCase):

    def test_load_returns_native_step_event_instances(self):
        runner = CliRunner()
        with runner.isolated_filesystem():
            _write_steps([
                _make_event(run_id="run-A", step_name="preflight", status="passed"),
                _make_event(run_id="run-A", step_name="observe", status="passed"),
            ])
            events = load_native_step_events()
            self.assertEqual(len(events), 2)
            self.assertIsInstance(events[0], NativeStepEvent)
            self.assertEqual(events[0].step_name, "preflight")
            self.assertEqual(events[1].step_name, "observe")

    def test_load_skips_malformed_lines(self):
        runner = CliRunner()
        with runner.isolated_filesystem():
            path = Path(".openshard") / "native_steps.jsonl"
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("w", encoding="utf-8") as fh:
                fh.write(json.dumps(_event_to_dict(_make_event(step_name="good"))) + "\n")
                fh.write("not valid json\n")
                fh.write("\n")
                fh.write(json.dumps(_event_to_dict(_make_event(step_name="also-good"))) + "\n")
            events = load_native_step_events()
            self.assertEqual(len(events), 2)
            self.assertEqual(events[0].step_name, "good")
            self.assertEqual(events[1].step_name, "also-good")

    def test_missing_file_returns_empty_list(self):
        runner = CliRunner()
        with runner.isolated_filesystem():
            result = load_native_step_events()
            self.assertEqual(result, [])


class TestFilterByRunId(unittest.TestCase):

    def test_filter_by_run_id(self):
        runner = CliRunner()
        with runner.isolated_filesystem():
            _write_steps([
                _make_event(run_id="run-A", step_name="preflight"),
                _make_event(run_id="run-A", step_name="observe"),
                _make_event(run_id="run-B", step_name="preflight"),
            ])
            a_events = native_step_events_for_run("run-A")
            self.assertEqual(len(a_events), 2)
            b_events = native_step_events_for_run("run-B")
            self.assertEqual(len(b_events), 1)
            c_events = native_step_events_for_run("run-C")
            self.assertEqual(c_events, [])

    def test_missing_file_for_run_returns_empty_list(self):
        runner = CliRunner()
        with runner.isolated_filesystem():
            result = native_step_events_for_run("run-X")
            self.assertEqual(result, [])


class TestRawContentStoredEnforcement(unittest.TestCase):

    def test_post_init_always_false(self):
        evt = NativeStepEvent(raw_content_stored=True)
        self.assertIs(evt.raw_content_stored, False)

    def test_log_stores_false(self):
        runner = CliRunner()
        with runner.isolated_filesystem():
            evt = NativeStepEvent()
            log_native_step_event(evt)
            path = Path(".openshard") / "native_steps.jsonl"
            raw = json.loads(path.read_text(encoding="utf-8").strip())
            self.assertIs(raw["raw_content_stored"], False)

    def test_loaded_event_always_false(self):
        runner = CliRunner()
        with runner.isolated_filesystem():
            path = Path(".openshard") / "native_steps.jsonl"
            path.parent.mkdir(parents=True, exist_ok=True)
            record = _event_to_dict(_make_event())
            record["raw_content_stored"] = True
            path.write_text(json.dumps(record) + "\n", encoding="utf-8")
            events = load_native_step_events()
            self.assertEqual(len(events), 1)
            self.assertIs(events[0].raw_content_stored, False)


class TestPathProtection(unittest.TestCase):

    def test_native_steps_path_is_rejected(self):
        repo_root = Path.cwd()
        with self.assertRaises(UnsafePathError) as ctx:
            resolve_safe_repo_path(repo_root, ".openshard/native_steps.jsonl")
        self.assertIn("native_steps.jsonl", str(ctx.exception))

    def test_runs_still_rejected(self):
        repo_root = Path.cwd()
        with self.assertRaises(UnsafePathError):
            resolve_safe_repo_path(repo_root, ".openshard/runs.jsonl")

    def test_interactions_still_rejected(self):
        repo_root = Path.cwd()
        with self.assertRaises(UnsafePathError):
            resolve_safe_repo_path(repo_root, ".openshard/interactions.jsonl")


class TestStepToStageMapping(unittest.TestCase):

    def test_known_steps_map_correctly(self):
        cases = {
            "preflight": "setup",
            "observe": "setup",
            "gather_context": "setup",
            "plan_update": "planning",
            "budget_check": "planning",
            "generate_patch": "generation",
            "approval": "approval",
            "safe_write": "write",
            "verify": "verification",
            "retry_once": "retry",
            "final_receipt": "done",
            "max_steps_exceeded": "done",
        }
        for step_name, expected_stage in cases.items():
            with self.subTest(step_name=step_name):
                self.assertEqual(_step_to_stage(step_name), expected_stage)

    def test_unknown_step_falls_back_to_pipeline(self):
        self.assertEqual(_step_to_stage("some_unknown_step"), "pipeline")


class TestStepRendering(unittest.TestCase):

    def test_default_detail_shows_no_step_events(self):
        runner = CliRunner()
        with runner.isolated_filesystem():
            _write_steps([_make_event(run_id="2025-01-01T00:00:00Z", step_name="preflight")])
            entry = _make_run_entry(timestamp="2025-01-01T00:00:00Z")
            out = _render(entry, detail="default")
            self.assertNotIn("[native steps]", out)

    def test_more_detail_shows_compact_count(self):
        runner = CliRunner()
        with runner.isolated_filesystem():
            _write_steps([
                _make_event(run_id="2025-01-01T00:00:00Z", step_name="preflight"),
                _make_event(run_id="2025-01-01T00:00:00Z", step_name="observe"),
                _make_event(run_id="2025-01-01T00:00:00Z", step_name="gather_context"),
            ])
            entry = _make_run_entry(timestamp="2025-01-01T00:00:00Z")
            out = _render(entry, detail="full")
            self.assertIn("[native steps]", out)
            self.assertIn("preflight", out)

    def test_full_detail_shows_step_timeline(self):
        runner = CliRunner()
        with runner.isolated_filesystem():
            _write_steps([
                _make_event(run_id="2025-01-01T00:00:00Z", step_name="preflight", stage="setup", step_index=0, summary="repo_map_ok=True"),
                _make_event(run_id="2025-01-01T00:00:00Z", step_name="observe", stage="setup", step_index=1, summary="dirty=False"),
            ])
            entry = _make_run_entry(timestamp="2025-01-01T00:00:00Z")
            out = _render(entry, detail="full")
            self.assertIn("[native steps]", out)
            self.assertIn("preflight", out)
            self.assertIn("observe", out)
            self.assertIn("setup", out)
            self.assertIn("repo_map_ok=True", out)

    def test_no_step_events_shows_nothing(self):
        runner = CliRunner()
        with runner.isolated_filesystem():
            entry = _make_run_entry(timestamp="2025-01-01T00:00:00Z")
            out = _render(entry, detail="full")
            self.assertNotIn("[native steps]", out)

    def test_non_native_run_shows_no_step_events(self):
        runner = CliRunner()
        with runner.isolated_filesystem():
            _write_steps([_make_event(run_id="2025-01-01T00:00:00Z", step_name="preflight")])
            entry = {
                "task": "test task",
                "timestamp": "2025-01-01T00:00:00Z",
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
                "workflow": "direct",
            }
            out = _render(entry, detail="full")
            self.assertNotIn("[native steps]", out)


class TestExecutorLogStepEvent(unittest.TestCase):

    def _make_executor(self):
        from openshard.native.executor import NativeAgentExecutor
        executor = object.__new__(NativeAgentExecutor)
        executor._osn_recorder = None
        executor._run_id = ""
        executor._step_counter = 0
        return executor

    def test_log_step_event_noop_without_run_id(self):
        runner = CliRunner()
        with runner.isolated_filesystem():
            executor = self._make_executor()
            executor._run_id = ""
            executor.log_step_event("preflight", "passed")
            path = Path(".openshard") / "native_steps.jsonl"
            self.assertFalse(path.exists())

    def test_log_step_event_writes_event_when_run_id_set(self):
        runner = CliRunner()
        with runner.isolated_filesystem():
            executor = self._make_executor()
            executor._run_id = "2025-01-01T00:00:00Z"
            executor.log_step_event("preflight", "passed")
            events = load_native_step_events()
            self.assertEqual(len(events), 1)
            self.assertEqual(events[0].run_id, "2025-01-01T00:00:00Z")
            self.assertEqual(events[0].step_name, "preflight")
            self.assertEqual(events[0].status, "passed")

    def test_step_counter_increments(self):
        runner = CliRunner()
        with runner.isolated_filesystem():
            executor = self._make_executor()
            executor._run_id = "2025-01-01T00:00:00Z"
            executor.log_step_event("preflight", "passed")
            executor.log_step_event("observe", "passed")
            executor.log_step_event("gather_context", "passed")
            events = load_native_step_events()
            self.assertEqual([e.step_index for e in events], [0, 1, 2])

    def test_record_osn_loop_step_also_calls_log_step_event(self):
        runner = CliRunner()
        with runner.isolated_filesystem():
            executor = self._make_executor()
            executor._run_id = "2025-01-01T00:00:00Z"
            executor.record_osn_loop_step("preflight", "passed", result_summary="repo_map_ok=True")
            events = load_native_step_events()
            self.assertEqual(len(events), 1)
            self.assertEqual(events[0].step_name, "preflight")
            self.assertEqual(events[0].summary, "repo_map_ok=True")

    def test_summary_capped_at_120_chars(self):
        runner = CliRunner()
        with runner.isolated_filesystem():
            executor = self._make_executor()
            executor._run_id = "2025-01-01T00:00:00Z"
            executor.log_step_event("preflight", "passed", summary="x" * 200)
            events = load_native_step_events()
            self.assertLessEqual(len(events[0].summary), 120)

    def test_raw_content_stored_remains_false(self):
        runner = CliRunner()
        with runner.isolated_filesystem():
            executor = self._make_executor()
            executor._run_id = "2025-01-01T00:00:00Z"
            executor.log_step_event("preflight", "passed", metadata={"raw": "some content"})
            events = load_native_step_events()
            self.assertIs(events[0].raw_content_stored, False)
            path = Path(".openshard") / "native_steps.jsonl"
            raw = json.loads(path.read_text(encoding="utf-8").strip())
            self.assertIs(raw["raw_content_stored"], False)


if __name__ == "__main__":
    unittest.main()
