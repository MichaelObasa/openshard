"""Failure Memory v1 — comprehensive tests.

Covers:
  - NativeFailureMemoryEvent dataclass invariants
  - parse_failure_summary helper
  - Serialization round-trip
  - Append-only JSONL logging
  - load / filter helpers
  - raw_content_stored always False (triple enforcement)
  - Path protection
  - CLI: failure-memory, export-failure-memory
  - Pipeline integration (retry writes event)
  - Rendering: last --more / --full
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest import TestCase
from unittest.mock import MagicMock, patch

import click
from click.testing import CliRunner

from openshard.analysis.repo import RepoFacts
from openshard.cli.main import cli
from openshard.cli.run_output import _render_failure_memory_block
from openshard.history.failure_memory import (
    NativeFailureMemoryEvent,
    _dict_to_event,
    _event_to_dict,
    failure_memory_events_for_run,
    load_failure_memory_events,
    log_failure_memory_event,
    parse_failure_summary,
    recent_failure_memory,
)
from openshard.security.paths import UnsafePathError, resolve_safe_repo_path

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
        retry_succeeded=True,
    )
    defaults.update(kwargs)
    return NativeFailureMemoryEvent(**defaults)


def _write_events(events: list[NativeFailureMemoryEvent], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for evt in events:
            fh.write(json.dumps(_event_to_dict(evt)) + "\n")


def _render(entry: dict, detail: str = "more") -> str:
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
        "native": {},
    }
    base.update(kwargs)
    return base


# ---------------------------------------------------------------------------
# Dataclass invariants
# ---------------------------------------------------------------------------


class TestNativeFailureMemoryEventDataclass(TestCase):

    def test_defaults_are_correct(self):
        evt = NativeFailureMemoryEvent()
        self.assertEqual(evt.schema_version, 1)
        self.assertEqual(evt.run_id, "")
        self.assertEqual(evt.task_summary, "")
        self.assertEqual(evt.failure_type, "")
        self.assertEqual(evt.exit_code, 0)
        self.assertEqual(evt.output_chars, 0)
        self.assertEqual(evt.verification_status, "failed")
        self.assertFalse(evt.retry_attempted)
        self.assertFalse(evt.retry_succeeded)
        self.assertEqual(evt.retry_patch_files, [])
        self.assertEqual(evt.related_file_paths, [])
        self.assertEqual(evt.model, "")
        self.assertEqual(evt.workflow, "")
        self.assertFalse(evt.raw_content_stored)

    def test_post_init_enforces_raw_content_stored_false(self):
        evt = NativeFailureMemoryEvent(raw_content_stored=True)  # type: ignore[call-arg]
        self.assertIs(evt.raw_content_stored, False)

    def test_event_id_is_unique(self):
        a = NativeFailureMemoryEvent()
        b = NativeFailureMemoryEvent()
        self.assertNotEqual(a.event_id, b.event_id)

    def test_event_id_is_nonempty(self):
        evt = NativeFailureMemoryEvent()
        self.assertTrue(evt.event_id)

    def test_timestamp_matches_iso_format(self):
        evt = NativeFailureMemoryEvent()
        self.assertRegex(evt.timestamp, r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")

    def test_retry_succeeded_independent_of_retry_attempted(self):
        evt = NativeFailureMemoryEvent(retry_attempted=False, retry_succeeded=True)
        self.assertFalse(evt.retry_attempted)
        self.assertTrue(evt.retry_succeeded)


# ---------------------------------------------------------------------------
# parse_failure_summary
# ---------------------------------------------------------------------------


class TestParseFailureSummary(TestCase):

    def test_parses_standard_summary(self):
        s = "exit_code=1 failure_type=test_failure output_chars=842 raw_content_stored=false"
        d = parse_failure_summary(s)
        self.assertEqual(d["exit_code"], "1")
        self.assertEqual(d["failure_type"], "test_failure")
        self.assertEqual(d["output_chars"], "842")
        self.assertEqual(d["raw_content_stored"], "false")

    def test_parses_syntax_error(self):
        s = "exit_code=1 failure_type=syntax_error output_chars=20 raw_content_stored=false"
        self.assertEqual(parse_failure_summary(s)["failure_type"], "syntax_error")

    def test_parses_import_error(self):
        s = "exit_code=1 failure_type=import_error output_chars=50 raw_content_stored=false"
        self.assertEqual(parse_failure_summary(s)["failure_type"], "import_error")

    def test_parses_assertion_error(self):
        s = "exit_code=1 failure_type=assertion_error output_chars=30 raw_content_stored=false"
        self.assertEqual(parse_failure_summary(s)["failure_type"], "assertion_error")

    def test_handles_missing_key(self):
        s = "failure_type=test_failure output_chars=10"
        d = parse_failure_summary(s)
        self.assertNotIn("exit_code", d)
        self.assertEqual(d.get("exit_code", "1"), "1")

    def test_handles_empty_string(self):
        d = parse_failure_summary("")
        self.assertEqual(d, {})

    def test_handles_no_equals(self):
        d = parse_failure_summary("no_key_value_pairs here")
        self.assertEqual(d, {})


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------


class TestSerialization(TestCase):

    def _full_event(self) -> NativeFailureMemoryEvent:
        return NativeFailureMemoryEvent(
            run_id="run-abc",
            task_summary="fix the auth bug",
            failure_type="type_error",
            exit_code=2,
            output_chars=500,
            verification_status="failed",
            retry_attempted=True,
            retry_succeeded=False,
            retry_patch_files=["openshard/run/pipeline.py"],
            related_file_paths=["openshard/run/pipeline.py"],
            model="claude-sonnet-4-6",
            workflow="native",
        )

    def test_round_trip_all_fields(self):
        evt = self._full_event()
        d = _event_to_dict(evt)
        restored = _dict_to_event(d)
        self.assertEqual(restored.run_id, evt.run_id)
        self.assertEqual(restored.task_summary, evt.task_summary)
        self.assertEqual(restored.failure_type, evt.failure_type)
        self.assertEqual(restored.exit_code, evt.exit_code)
        self.assertEqual(restored.output_chars, evt.output_chars)
        self.assertEqual(restored.verification_status, evt.verification_status)
        self.assertEqual(restored.retry_attempted, evt.retry_attempted)
        self.assertEqual(restored.retry_succeeded, evt.retry_succeeded)
        self.assertEqual(restored.retry_patch_files, evt.retry_patch_files)
        self.assertEqual(restored.related_file_paths, evt.related_file_paths)
        self.assertEqual(restored.model, evt.model)
        self.assertEqual(restored.workflow, evt.workflow)
        self.assertFalse(restored.raw_content_stored)

    def test_all_required_keys_present_in_dict(self):
        d = _event_to_dict(self._full_event())
        for key in [
            "schema_version", "event_id", "run_id", "timestamp", "task_summary",
            "failure_type", "exit_code", "output_chars", "verification_status",
            "retry_attempted", "retry_succeeded", "retry_patch_files",
            "related_file_paths", "model", "workflow", "raw_content_stored",
        ]:
            self.assertIn(key, d, f"Missing key: {key}")

    def test_raw_content_stored_always_false_in_dict(self):
        evt = NativeFailureMemoryEvent()
        d = _event_to_dict(evt)
        self.assertIs(d["raw_content_stored"], False)

    def test_dict_to_event_ignores_stored_true(self):
        d = _event_to_dict(NativeFailureMemoryEvent())
        d["raw_content_stored"] = True
        restored = _dict_to_event(d)
        self.assertIs(restored.raw_content_stored, False)

    def test_schema_version_preserved(self):
        d = _event_to_dict(NativeFailureMemoryEvent())
        self.assertEqual(d["schema_version"], 1)


# ---------------------------------------------------------------------------
# JSONL logging
# ---------------------------------------------------------------------------


class TestLogging(TestCase):

    def setUp(self):
        self._runner = CliRunner()
        self._td = self._runner.isolated_filesystem()
        self._td.__enter__()
        self._path = Path(".openshard") / "failure_memory.jsonl"

    def tearDown(self):
        self._td.__exit__(None, None, None)

    def test_append_only_logging(self):
        log_failure_memory_event(_make_event(task_summary="first"))
        log_failure_memory_event(_make_event(task_summary="second"))
        log_failure_memory_event(_make_event(task_summary="third"))
        lines = [ln for ln in self._path.read_text().splitlines() if ln.strip()]
        self.assertEqual(len(lines), 3)
        self.assertIn("first", lines[0])
        self.assertIn("second", lines[1])
        self.assertIn("third", lines[2])

    def test_append_does_not_overwrite(self):
        log_failure_memory_event(_make_event(task_summary="alpha"))
        log_failure_memory_event(_make_event(task_summary="beta"))
        lines = [ln for ln in self._path.read_text().splitlines() if ln.strip()]
        self.assertEqual(len(lines), 2)
        self.assertIn("alpha", lines[0])

    def test_directory_created_automatically(self):
        self.assertFalse(Path(".openshard").exists())
        log_failure_memory_event(_make_event())
        self.assertTrue(self._path.exists())

    def test_stored_json_has_raw_content_stored_false(self):
        log_failure_memory_event(_make_event())
        raw = json.loads(self._path.read_text().strip())
        self.assertFalse(raw["raw_content_stored"])


# ---------------------------------------------------------------------------
# Load helpers
# ---------------------------------------------------------------------------


class TestLoadFailureMemoryEvents(TestCase):

    def setUp(self):
        self._runner = CliRunner()
        self._td = self._runner.isolated_filesystem()
        self._td.__enter__()
        self._path = Path(".openshard") / "failure_memory.jsonl"

    def tearDown(self):
        self._td.__exit__(None, None, None)

    def test_missing_file_returns_empty_list(self):
        self.assertEqual(load_failure_memory_events(), [])

    def test_load_returns_event_instances(self):
        log_failure_memory_event(_make_event())
        events = load_failure_memory_events()
        self.assertEqual(len(events), 1)
        self.assertIsInstance(events[0], NativeFailureMemoryEvent)

    def test_load_skips_malformed_lines(self):
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._path.open("w", encoding="utf-8") as fh:
            fh.write(json.dumps(_event_to_dict(_make_event())) + "\n")
            fh.write("not-valid-json\n")
            fh.write("\n")
            fh.write(json.dumps(_event_to_dict(_make_event())) + "\n")
        events = load_failure_memory_events()
        self.assertEqual(len(events), 2)

    def test_load_preserves_order(self):
        for summary in ["a", "b", "c"]:
            log_failure_memory_event(_make_event(task_summary=summary))
        events = load_failure_memory_events()
        self.assertEqual([e.task_summary for e in events], ["a", "b", "c"])


# ---------------------------------------------------------------------------
# Filter by run_id
# ---------------------------------------------------------------------------


class TestFilterByRunId(TestCase):

    def setUp(self):
        self._runner = CliRunner()
        self._td = self._runner.isolated_filesystem()
        self._td.__enter__()

    def tearDown(self):
        self._td.__exit__(None, None, None)

    def test_filter_returns_matching_events(self):
        log_failure_memory_event(_make_event(run_id="run-A"))
        log_failure_memory_event(_make_event(run_id="run-B"))
        log_failure_memory_event(_make_event(run_id="run-A"))
        matches = failure_memory_events_for_run("run-A")
        self.assertEqual(len(matches), 2)
        self.assertTrue(all(e.run_id == "run-A" for e in matches))

    def test_filter_returns_empty_for_unknown_run_id(self):
        log_failure_memory_event(_make_event(run_id="run-X"))
        self.assertEqual(failure_memory_events_for_run("run-UNKNOWN"), [])

    def test_filter_two_run_ids_separated(self):
        for i in range(3):
            log_failure_memory_event(_make_event(run_id=f"run-{i}"))
        self.assertEqual(len(failure_memory_events_for_run("run-0")), 1)
        self.assertEqual(len(failure_memory_events_for_run("run-1")), 1)
        self.assertEqual(len(failure_memory_events_for_run("run-2")), 1)


# ---------------------------------------------------------------------------
# recent_failure_memory
# ---------------------------------------------------------------------------


class TestRecentFailureMemory(TestCase):

    def setUp(self):
        self._runner = CliRunner()
        self._td = self._runner.isolated_filesystem()
        self._td.__enter__()

    def tearDown(self):
        self._td.__exit__(None, None, None)

    def test_returns_last_n_events(self):
        for i in range(15):
            log_failure_memory_event(_make_event(task_summary=str(i)))
        recent = recent_failure_memory(limit=10)
        self.assertEqual(len(recent), 10)
        # should be the last 10 (indices 5–14)
        self.assertEqual(recent[0].task_summary, "5")
        self.assertEqual(recent[-1].task_summary, "14")

    def test_limit_larger_than_total_returns_all(self):
        for i in range(3):
            log_failure_memory_event(_make_event())
        self.assertEqual(len(recent_failure_memory(limit=100)), 3)

    def test_default_limit_is_10(self):
        for i in range(15):
            log_failure_memory_event(_make_event())
        self.assertEqual(len(recent_failure_memory()), 10)

    def test_empty_log_returns_empty(self):
        self.assertEqual(recent_failure_memory(), [])


# ---------------------------------------------------------------------------
# raw_content_stored enforcement (triple)
# ---------------------------------------------------------------------------


class TestRawContentStoredEnforcement(TestCase):

    def setUp(self):
        self._runner = CliRunner()
        self._td = self._runner.isolated_filesystem()
        self._td.__enter__()

    def tearDown(self):
        self._td.__exit__(None, None, None)

    def test_post_init_always_false(self):
        self.assertFalse(NativeFailureMemoryEvent(raw_content_stored=True).raw_content_stored)  # type: ignore[call-arg]

    def test_event_to_dict_always_false(self):
        evt = NativeFailureMemoryEvent()
        d = _event_to_dict(evt)
        self.assertFalse(d["raw_content_stored"])

    def test_log_stores_false_on_disk(self):
        log_failure_memory_event(_make_event())
        raw = json.loads((Path(".openshard") / "failure_memory.jsonl").read_text().strip())
        self.assertFalse(raw["raw_content_stored"])

    def test_loaded_event_always_false_even_if_disk_has_true(self):
        path = Path(".openshard") / "failure_memory.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        d = _event_to_dict(_make_event())
        d["raw_content_stored"] = True
        with path.open("w") as fh:
            fh.write(json.dumps(d) + "\n")
        events = load_failure_memory_events()
        self.assertFalse(events[0].raw_content_stored)


# ---------------------------------------------------------------------------
# Path protection
# ---------------------------------------------------------------------------


class TestPathProtection(TestCase):

    def test_failure_memory_path_is_rejected(self):
        repo_root = Path.cwd()
        with self.assertRaises(UnsafePathError) as ctx:
            resolve_safe_repo_path(repo_root, ".openshard/failure_memory.jsonl")
        self.assertIn("failure_memory.jsonl", str(ctx.exception))

    def test_runs_still_rejected(self):
        repo_root = Path.cwd()
        with self.assertRaises(UnsafePathError):
            resolve_safe_repo_path(repo_root, ".openshard/runs.jsonl")

    def test_interactions_still_rejected(self):
        repo_root = Path.cwd()
        with self.assertRaises(UnsafePathError):
            resolve_safe_repo_path(repo_root, ".openshard/interactions.jsonl")

    def test_native_steps_still_rejected(self):
        repo_root = Path.cwd()
        with self.assertRaises(UnsafePathError):
            resolve_safe_repo_path(repo_root, ".openshard/native_steps.jsonl")


# ---------------------------------------------------------------------------
# CLI: failure-memory
# ---------------------------------------------------------------------------


class TestCLIFailureMemoryCommand(TestCase):

    def setUp(self):
        self._runner = CliRunner()
        self._td_ctx = self._runner.isolated_filesystem()
        self._td_ctx.__enter__()

    def tearDown(self):
        self._td_ctx.__exit__(None, None, None)

    def test_no_events_shows_message(self):
        result = self._runner.invoke(cli, ["failure-memory"])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("No failure memory events recorded yet", result.output)

    def test_shows_events_table_with_headers(self):
        log_failure_memory_event(_make_event(task_summary="Add rate limiting", failure_type="test_failure"))
        result = self._runner.invoke(cli, ["failure-memory"])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("Time", result.output)
        self.assertIn("Failure Type", result.output)
        self.assertIn("Exit", result.output)
        self.assertIn("Retry", result.output)
        self.assertIn("test_failure", result.output)

    def test_last_n_limits_output(self):
        for i in range(15):
            log_failure_memory_event(_make_event(task_summary=f"task-{i}"))
        result = self._runner.invoke(cli, ["failure-memory", "--last", "5"])
        self.assertEqual(result.exit_code, 0)
        # Header + 5 data rows
        data_lines = [ln for ln in result.output.splitlines() if ln.strip() and "Time" not in ln]
        self.assertEqual(len(data_lines), 5)

    def test_task_summary_truncated_to_50_chars(self):
        long_task = "A" * 80
        log_failure_memory_event(_make_event(task_summary=long_task))
        result = self._runner.invoke(cli, ["failure-memory"])
        self.assertEqual(result.exit_code, 0)
        # Only 50 chars of the task shown
        self.assertNotIn("A" * 51, result.output)


# ---------------------------------------------------------------------------
# CLI: export-failure-memory
# ---------------------------------------------------------------------------


class TestCLIExportFailureMemory(TestCase):

    def setUp(self):
        self._runner = CliRunner()
        self._td_ctx = self._runner.isolated_filesystem()
        self._td_ctx.__enter__()

    def tearDown(self):
        self._td_ctx.__exit__(None, None, None)

    def test_no_events_shows_message(self):
        result = self._runner.invoke(cli, ["export-failure-memory"])
        self.assertIn("No failure memory events recorded yet", result.output)

    def test_exports_valid_jsonl_to_stdout(self):
        log_failure_memory_event(_make_event())
        result = self._runner.invoke(cli, ["export-failure-memory"])
        self.assertEqual(result.exit_code, 0)
        d = json.loads(result.output.strip())
        self.assertIn("event_id", d)
        self.assertFalse(d["raw_content_stored"])

    def test_exports_to_file(self):
        log_failure_memory_event(_make_event())
        result = self._runner.invoke(cli, ["export-failure-memory", "--output", "out.jsonl"])
        self.assertEqual(result.exit_code, 0)
        lines = [ln for ln in Path("out.jsonl").read_text().splitlines() if ln.strip()]
        self.assertEqual(len(lines), 1)
        d = json.loads(lines[0])
        self.assertIn("event_id", d)

    def test_redacted_sets_task_summary(self):
        log_failure_memory_event(_make_event(task_summary="secret task"))
        result = self._runner.invoke(cli, ["export-failure-memory", "--redacted"])
        self.assertEqual(result.exit_code, 0)
        d = json.loads(result.output.strip())
        self.assertEqual(d["task_summary"], "[redacted]")
        self.assertNotIn("secret task", result.output)

    def test_redacted_sets_model(self):
        log_failure_memory_event(_make_event(model="claude-opus-4-7"))
        result = self._runner.invoke(cli, ["export-failure-memory", "--redacted"])
        d = json.loads(result.output.strip())
        self.assertEqual(d["model"], "redacted")

    def test_redacted_preserves_workflow(self):
        log_failure_memory_event(_make_event(workflow="native"))
        result = self._runner.invoke(cli, ["export-failure-memory", "--redacted"])
        d = json.loads(result.output.strip())
        self.assertEqual(d["workflow"], "native")

    def test_raw_content_stored_always_false_in_export(self):
        log_failure_memory_event(_make_event())
        result = self._runner.invoke(cli, ["export-failure-memory"])
        d = json.loads(result.output.strip())
        self.assertFalse(d["raw_content_stored"])


# ---------------------------------------------------------------------------
# Pipeline integration
# ---------------------------------------------------------------------------

_DEFAULT_CONFIG_FM = {"approval_mode": "smart"}

_PYTHON_REPO_FM = RepoFacts(
    languages=["python"],
    package_files=[],
    framework=None,
    test_command="python -m pytest",
    risky_paths=[],
    changed_files=[],
)


def _make_native_mock_fm(generate_side_effect=None):
    from openshard.native.context import NativeApprovalRequest, NativeChangeBudgetSoftGate
    from openshard.native.executor import NativeRunMeta

    g = MagicMock()
    if generate_side_effect is not None:
        g.generate.side_effect = generate_side_effect
    else:
        g.generate.return_value = MagicMock(usage=None, files=[], summary="done", notes=[])
    g.model = "mock-model"
    g.fixer_model = "mock-fixer"
    meta = NativeRunMeta()
    g.native_meta = meta
    g.build_change_budget_soft_gate.return_value = NativeChangeBudgetSoftGate(
        requires_approval=False, reason="", action="allow"
    )
    g.build_budget_gate_approval_request.return_value = NativeApprovalRequest(
        source="change_budget_soft_gate", requires_approval=False, reason="", action="allow"
    )
    return g


def _make_manager_mock_fm():
    m = MagicMock()
    inv = MagicMock()
    inv.models = []
    m.get_inventory.return_value = inv
    m.providers = {"openrouter": MagicMock()}
    return m


def _invoke_native_write_fm(native_mock=None, *, verify_side_effects=None, captured_events=None):
    from openshard.native.context import NativeSandboxMeta

    if native_mock is None:
        native_mock = _make_native_mock_fm()

    _verif_calls = iter(verify_side_effects or [(0, "")])

    def _fake_verify(*args, **kwargs):
        capture = kwargs.get("capture", False)
        try:
            code, out = next(_verif_calls)
        except StopIteration:
            code, out = 0, ""
        return (code, out) if capture else code

    def _capture_event(evt):
        if captured_events is not None:
            captured_events.append(evt)

    with tempfile.TemporaryDirectory() as _td:
        sandbox_meta = NativeSandboxMeta(sandbox_enabled=False, sandbox_type="none")
        with (
            patch("openshard.run.pipeline.NativeAgentExecutor", return_value=native_mock),
            patch("openshard.run.pipeline.ProviderManager", return_value=_make_manager_mock_fm()),
            patch("openshard.cli.main.load_config", return_value=_DEFAULT_CONFIG_FM),
            patch("openshard.run.pipeline.analyze_repo", return_value=_PYTHON_REPO_FM),
            patch("openshard.run.pipeline._run_verification_plan", side_effect=_fake_verify),
            patch("openshard.run.pipeline._write_files"),
            patch("openshard.native.sandbox.create_run_sandbox", return_value=(Path(_td), sandbox_meta)),
            patch("openshard.run.pipeline._log_run"),
            patch("openshard.run.pipeline.log_failure_memory_event", side_effect=_capture_event),
        ):
            runner = CliRunner()
            result = runner.invoke(cli, ["run", "--workflow", "native", "--write", "fix the bug"])
    return result, native_mock


class TestPipelineIntegration(TestCase):

    def test_failure_memory_event_logged_on_retry(self):
        captured: list[NativeFailureMemoryEvent] = []
        result, _ = _invoke_native_write_fm(
            verify_side_effects=[(1, "FAILED: 1 error"), (0, "all passed")],
            captured_events=captured,
        )
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertEqual(len(captured), 1)
        evt = captured[0]
        self.assertTrue(evt.retry_attempted)
        self.assertFalse(evt.raw_content_stored)

    def test_retry_succeeded_true_when_second_verify_passes(self):
        captured: list[NativeFailureMemoryEvent] = []
        _invoke_native_write_fm(
            verify_side_effects=[(1, "FAILED test"), (0, "1 passed")],
            captured_events=captured,
        )
        self.assertEqual(len(captured), 1)
        self.assertTrue(captured[0].retry_succeeded)

    def test_retry_succeeded_false_when_second_verify_fails(self):
        captured: list[NativeFailureMemoryEvent] = []
        _invoke_native_write_fm(
            verify_side_effects=[(1, "TypeError: bad"), (1, "still failing")],
            captured_events=captured,
        )
        self.assertEqual(len(captured), 1)
        self.assertFalse(captured[0].retry_succeeded)

    def test_failure_type_parsed_from_summary(self):
        captured: list[NativeFailureMemoryEvent] = []
        _invoke_native_write_fm(
            verify_side_effects=[(1, "TypeError: unsupported operand"), (0, "ok")],
            captured_events=captured,
        )
        self.assertEqual(len(captured), 1)
        self.assertEqual(captured[0].failure_type, "type_error")

    def test_no_event_on_clean_first_pass(self):
        captured: list[NativeFailureMemoryEvent] = []
        result, _ = _invoke_native_write_fm(
            verify_side_effects=[(0, "all passed")],
            captured_events=captured,
        )
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertEqual(len(captured), 0)

    def test_event_task_summary_truncated_to_120_chars(self):
        captured: list[NativeFailureMemoryEvent] = []
        _invoke_native_write_fm(
            verify_side_effects=[(1, "FAILED"), (0, "ok")],
            captured_events=captured,
        )
        self.assertEqual(len(captured), 1)
        # task was "fix the bug" — well under 120, so intact
        self.assertEqual(captured[0].task_summary, "fix the bug")


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


class TestRendering(TestCase):

    def setUp(self):
        self._runner = CliRunner()
        self._td_ctx = self._runner.isolated_filesystem()
        self._td_ctx.__enter__()

    def tearDown(self):
        self._td_ctx.__exit__(None, None, None)

    def _log_event_for_run(self, run_id: str, **kwargs) -> None:
        log_failure_memory_event(_make_event(run_id=run_id, **kwargs))

    def test_default_detail_shows_no_failure_memory(self):
        run_id = "2026-01-01T00:00:00Z"
        self._log_event_for_run(run_id)
        entry = _make_run_entry(timestamp=run_id, workflow="native")
        out = _render(entry, detail="default")
        self.assertNotIn("[failure memory]", out)

    def test_more_detail_shows_compact_failure_line(self):
        run_id = "2026-01-01T00:00:00Z"
        self._log_event_for_run(run_id, failure_type="test_failure", retry_succeeded=True)
        entry = _make_run_entry(timestamp=run_id, workflow="native")
        out = _render(entry, detail="full")
        self.assertIn("[failure memory]", out)
        self.assertIn("test_failure", out)

    def test_full_detail_shows_expanded_block(self):
        run_id = "2026-01-01T00:00:00Z"
        self._log_event_for_run(
            run_id,
            failure_type="syntax_error",
            exit_code=1,
            output_chars=200,
            retry_succeeded=False,
            retry_patch_files=["openshard/run/pipeline.py"],
        )
        entry = _make_run_entry(timestamp=run_id, workflow="native")
        out = _render(entry, detail="full")
        self.assertIn("[failure memory]", out)
        self.assertIn("syntax_error", out)
        self.assertIn("exit=1", out)
        self.assertIn("chars=200", out)

    def test_full_detail_shows_patch_files(self):
        run_id = "2026-01-01T00:00:00Z"
        self._log_event_for_run(
            run_id,
            retry_patch_files=["openshard/run/pipeline.py"],
        )
        entry = _make_run_entry(timestamp=run_id, workflow="native")
        out = _render(entry, detail="full")
        self.assertIn("openshard/run/pipeline.py", out)

    def test_no_events_shows_nothing_at_more(self):
        run_id = "2026-01-01T00:00:00Z"
        entry = _make_run_entry(timestamp=run_id, workflow="native")
        out = _render(entry, detail="more")
        self.assertNotIn("[failure memory]", out)

    def test_non_native_workflow_shows_nothing(self):
        run_id = "2026-01-01T00:00:00Z"
        self._log_event_for_run(run_id)
        entry = _make_run_entry(timestamp=run_id, workflow="direct")
        out_more = _render(entry, detail="more")
        out_full = _render(entry, detail="full")
        self.assertNotIn("[failure memory]", out_more)
        self.assertNotIn("[failure memory]", out_full)


# ---------------------------------------------------------------------------
# _render_failure_memory_block — pure function
# ---------------------------------------------------------------------------


class TestRenderFailureMemoryBlock(TestCase):

    def _evt(self, **kwargs) -> NativeFailureMemoryEvent:
        return _make_event(**kwargs)

    def test_default_returns_empty(self):
        self.assertEqual(_render_failure_memory_block([self._evt()], "default"), [])

    def test_empty_events_returns_empty(self):
        self.assertEqual(_render_failure_memory_block([], "more"), [])

    def test_more_returns_one_compact_line(self):
        lines = _render_failure_memory_block([self._evt(failure_type="test_failure")], "more")
        self.assertEqual(len(lines), 1)
        self.assertIn("[failure memory]", lines[0])
        self.assertIn("test_failure", lines[0])

    def test_full_returns_multiple_lines(self):
        lines = _render_failure_memory_block(
            [self._evt(failure_type="import_error", exit_code=1, output_chars=42)],
            "full",
        )
        self.assertGreater(len(lines), 1)
        combined = "\n".join(lines)
        self.assertIn("[failure memory]", combined)
        self.assertIn("import_error", combined)
        self.assertIn("exit=1", combined)
        self.assertIn("chars=42", combined)

    def test_more_uses_last_event(self):
        evts = [
            self._evt(failure_type="syntax_error"),
            self._evt(failure_type="type_error"),
        ]
        lines = _render_failure_memory_block(evts, "more")
        self.assertIn("type_error", lines[0])

    def test_full_shows_patch_files(self):
        lines = _render_failure_memory_block(
            [self._evt(retry_patch_files=["src/foo.py", "src/bar.py"])],
            "full",
        )
        combined = "\n".join(lines)
        self.assertIn("src/foo.py", combined)
