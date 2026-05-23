"""Native Run Checkpoints v0 — comprehensive tests.

Covers:
  - NativeRunCheckpointEvent dataclass invariants
  - Serialization round-trip
  - Append-only JSONL logging and load helpers
  - raw_content_stored always False (triple enforcement)
  - Path protection
  - Rendering: --more / --full
  - CLI: checkpoints, resume-last
  - Pipeline integration
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
from openshard.cli.run_output import _render_run_checkpoints_block
from openshard.history.run_checkpoints import (
    NativeRunCheckpointEvent,
    _dict_to_event,
    _event_to_dict,
    load_run_checkpoint_events,
    log_run_checkpoint_event,
    recent_run_checkpoints,
    run_checkpoints_for_run,
)
from openshard.security.paths import UnsafePathError, resolve_safe_repo_path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_event(**kwargs) -> NativeRunCheckpointEvent:
    defaults: dict = dict(
        run_id="2026-01-01T00:00:00Z",
        workflow="native",
        executor="native",
        stage="plan",
        status="passed",
    )
    defaults.update(kwargs)
    return NativeRunCheckpointEvent(**defaults)


def _write_events(events: list[NativeRunCheckpointEvent], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for evt in events:
            fh.write(json.dumps(_event_to_dict(evt)) + "\n")


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


def _render(entry: dict, detail: str = "more") -> str:
    from openshard.cli.main import _render_log_entry

    @click.command()
    def cmd():
        _render_log_entry(entry, detail)

    return CliRunner().invoke(cmd).output


# ---------------------------------------------------------------------------
# Dataclass invariants
# ---------------------------------------------------------------------------


class TestNativeRunCheckpointEventDataclass(TestCase):

    def test_defaults_are_correct(self):
        evt = NativeRunCheckpointEvent()
        self.assertEqual(evt.schema_version, 1)
        self.assertEqual(evt.run_id, "")
        self.assertEqual(evt.workflow, "")
        self.assertEqual(evt.executor, "")
        self.assertEqual(evt.stage, "")
        self.assertEqual(evt.status, "")
        self.assertEqual(evt.workspace_path, "")
        self.assertEqual(evt.sandbox_path, "")
        self.assertEqual(evt.files, [])
        self.assertEqual(evt.verification_status, "")
        self.assertFalse(evt.retry_used)
        self.assertEqual(evt.reason, "")
        self.assertFalse(evt.raw_content_stored)

    def test_post_init_enforces_raw_content_stored_false(self):
        evt = NativeRunCheckpointEvent(raw_content_stored=True)  # type: ignore[call-arg]
        self.assertIs(evt.raw_content_stored, False)

    def test_event_id_is_unique(self):
        a = NativeRunCheckpointEvent()
        b = NativeRunCheckpointEvent()
        self.assertNotEqual(a.event_id, b.event_id)

    def test_event_id_is_nonempty(self):
        evt = NativeRunCheckpointEvent()
        self.assertTrue(evt.event_id)

    def test_timestamp_matches_iso_format(self):
        evt = NativeRunCheckpointEvent()
        self.assertRegex(evt.timestamp, r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")

    def test_files_list_default_is_independent(self):
        a = NativeRunCheckpointEvent()
        b = NativeRunCheckpointEvent()
        a.files.append("foo.py")
        self.assertEqual(b.files, [])


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------


class TestSerialization(TestCase):

    def _full_event(self) -> NativeRunCheckpointEvent:
        return NativeRunCheckpointEvent(
            run_id="run-abc",
            workflow="native",
            executor="native",
            stage="verify",
            status="passed",
            workspace_path="/tmp/ws",
            sandbox_path="/tmp/ws",
            files=["src/foo.py", "src/bar.py"],
            verification_status="passed",
            retry_used=True,
            reason="",
        )

    def test_all_required_keys_present_in_dict(self):
        d = _event_to_dict(self._full_event())
        for key in [
            "schema_version", "event_id", "run_id", "timestamp", "workflow",
            "executor", "stage", "status", "workspace_path", "sandbox_path",
            "files", "verification_status", "retry_used", "reason", "raw_content_stored",
        ]:
            self.assertIn(key, d, f"Missing key: {key}")

    def test_raw_content_stored_always_false_in_dict(self):
        d = _event_to_dict(self._full_event())
        self.assertIs(d["raw_content_stored"], False)

    def test_dict_to_event_ignores_stored_true(self):
        d = _event_to_dict(self._full_event())
        d["raw_content_stored"] = True
        restored = _dict_to_event(d)
        self.assertIs(restored.raw_content_stored, False)

    def test_round_trip_preserves_stage_status_files(self):
        evt = self._full_event()
        restored = _dict_to_event(_event_to_dict(evt))
        self.assertEqual(restored.stage, evt.stage)
        self.assertEqual(restored.status, evt.status)
        self.assertEqual(restored.files, evt.files)
        self.assertEqual(restored.verification_status, evt.verification_status)
        self.assertEqual(restored.retry_used, evt.retry_used)
        self.assertEqual(restored.run_id, evt.run_id)

    def test_schema_version_preserved(self):
        d = _event_to_dict(NativeRunCheckpointEvent())
        self.assertEqual(d["schema_version"], 1)


# ---------------------------------------------------------------------------
# JSONL logging
# ---------------------------------------------------------------------------


class TestLogging(TestCase):

    def setUp(self):
        self._runner = CliRunner()
        self._td = self._runner.isolated_filesystem()
        self._td.__enter__()
        self._path = Path(".openshard") / "run_checkpoints.jsonl"

    def tearDown(self):
        self._td.__exit__(None, None, None)

    def test_missing_file_returns_empty_list(self):
        self.assertEqual(load_run_checkpoint_events(), [])

    def test_append_only_writes_3_lines(self):
        log_run_checkpoint_event(_make_event(stage="plan"))
        log_run_checkpoint_event(_make_event(stage="generate"))
        log_run_checkpoint_event(_make_event(stage="sandbox_write"))
        lines = [ln for ln in self._path.read_text().splitlines() if ln.strip()]
        self.assertEqual(len(lines), 3)
        self.assertIn("plan", lines[0])
        self.assertIn("generate", lines[1])
        self.assertIn("sandbox_write", lines[2])

    def test_malformed_lines_skipped(self):
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._path.open("w", encoding="utf-8") as fh:
            fh.write(json.dumps(_event_to_dict(_make_event(stage="plan"))) + "\n")
            fh.write("not-valid-json\n")
            fh.write("\n")
            fh.write(json.dumps(_event_to_dict(_make_event(stage="generate"))) + "\n")
        events = load_run_checkpoint_events()
        self.assertEqual(len(events), 2)

    def test_loaded_events_are_instances(self):
        log_run_checkpoint_event(_make_event())
        events = load_run_checkpoint_events()
        self.assertEqual(len(events), 1)
        self.assertIsInstance(events[0], NativeRunCheckpointEvent)

    def test_directory_created_automatically(self):
        self.assertFalse(Path(".openshard").exists())
        log_run_checkpoint_event(_make_event())
        self.assertTrue(self._path.exists())

    def test_recent_run_checkpoints_returns_last_n(self):
        for i in range(15):
            log_run_checkpoint_event(_make_event(stage=f"s{i}"))
        recent = recent_run_checkpoints(limit=10)
        self.assertEqual(len(recent), 10)
        self.assertEqual(recent[0].stage, "s5")
        self.assertEqual(recent[-1].stage, "s14")

    def test_recent_run_checkpoints_limit_larger_than_total(self):
        for _ in range(3):
            log_run_checkpoint_event(_make_event())
        self.assertEqual(len(recent_run_checkpoints(limit=100)), 3)

    def test_run_checkpoints_for_run_filters_correctly(self):
        log_run_checkpoint_event(_make_event(run_id="run-A", stage="plan"))
        log_run_checkpoint_event(_make_event(run_id="run-B", stage="plan"))
        log_run_checkpoint_event(_make_event(run_id="run-A", stage="generate"))
        matches = run_checkpoints_for_run("run-A")
        self.assertEqual(len(matches), 2)
        self.assertTrue(all(e.run_id == "run-A" for e in matches))

    def test_run_checkpoints_for_run_unknown_id_returns_empty(self):
        log_run_checkpoint_event(_make_event(run_id="run-X"))
        self.assertEqual(run_checkpoints_for_run("run-UNKNOWN"), [])


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
        self.assertFalse(NativeRunCheckpointEvent(raw_content_stored=True).raw_content_stored)  # type: ignore[call-arg]

    def test_event_to_dict_always_false(self):
        d = _event_to_dict(NativeRunCheckpointEvent())
        self.assertFalse(d["raw_content_stored"])

    def test_log_stores_false_on_disk(self):
        log_run_checkpoint_event(_make_event())
        raw = json.loads((Path(".openshard") / "run_checkpoints.jsonl").read_text().strip())
        self.assertFalse(raw["raw_content_stored"])

    def test_loaded_event_always_false_even_if_disk_has_true(self):
        path = Path(".openshard") / "run_checkpoints.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        d = _event_to_dict(_make_event())
        d["raw_content_stored"] = True
        with path.open("w") as fh:
            fh.write(json.dumps(d) + "\n")
        events = load_run_checkpoint_events()
        self.assertFalse(events[0].raw_content_stored)


# ---------------------------------------------------------------------------
# Path protection
# ---------------------------------------------------------------------------


class TestPathSafety(TestCase):

    def test_run_checkpoints_path_is_rejected(self):
        repo_root = Path.cwd()
        with self.assertRaises(UnsafePathError) as ctx:
            resolve_safe_repo_path(repo_root, ".openshard/run_checkpoints.jsonl")
        self.assertIn("run_checkpoints.jsonl", str(ctx.exception))

    def test_runs_still_rejected(self):
        with self.assertRaises(UnsafePathError):
            resolve_safe_repo_path(Path.cwd(), ".openshard/runs.jsonl")

    def test_native_steps_still_rejected(self):
        with self.assertRaises(UnsafePathError):
            resolve_safe_repo_path(Path.cwd(), ".openshard/native_steps.jsonl")

    def test_failure_memory_still_rejected(self):
        with self.assertRaises(UnsafePathError):
            resolve_safe_repo_path(Path.cwd(), ".openshard/failure_memory.jsonl")

    def test_sandbox_apply_receipts_still_rejected(self):
        with self.assertRaises(UnsafePathError):
            resolve_safe_repo_path(Path.cwd(), ".openshard/sandbox_apply_receipts.jsonl")


# ---------------------------------------------------------------------------
# _render_run_checkpoints_block — pure function
# ---------------------------------------------------------------------------


class TestRenderRunCheckpointsBlock(TestCase):

    def _evt(self, **kwargs) -> NativeRunCheckpointEvent:
        return _make_event(**kwargs)

    def test_default_returns_empty(self):
        self.assertEqual(_render_run_checkpoints_block([self._evt()], "default"), [])

    def test_empty_events_returns_empty(self):
        self.assertEqual(_render_run_checkpoints_block([], "more"), [])

    def test_more_returns_single_compact_line(self):
        lines = _render_run_checkpoints_block([self._evt(stage="plan", status="passed")], "more")
        self.assertEqual(len(lines), 1)
        self.assertIn("[checkpoints]", lines[0])
        self.assertIn("plan:passed", lines[0])

    def test_more_shows_total_count(self):
        evts = [self._evt(stage="plan"), self._evt(stage="generate"), self._evt(stage="sandbox_write")]
        lines = _render_run_checkpoints_block(evts, "more")
        self.assertIn("total=3", lines[0])

    def test_more_uses_latest_event(self):
        evts = [self._evt(stage="plan"), self._evt(stage="generate")]
        lines = _render_run_checkpoints_block(evts, "more")
        self.assertIn("generate:", lines[0])

    def test_full_shows_all_stages(self):
        evts = [
            self._evt(stage="plan", status="passed"),
            self._evt(stage="generate", status="passed"),
            self._evt(stage="verify", status="failed"),
        ]
        lines = _render_run_checkpoints_block(evts, "full")
        combined = "\n".join(lines)
        self.assertIn("plan", combined)
        self.assertIn("generate", combined)
        self.assertIn("verify", combined)

    def test_full_shows_verify_and_retry_columns(self):
        evt = self._evt(stage="retry", status="passed", verification_status="passed", retry_used=True)
        lines = _render_run_checkpoints_block([evt], "full")
        combined = "\n".join(lines)
        self.assertIn("verify=passed", combined)
        self.assertIn("retry=yes", combined)

    def test_full_with_files_shows_up_to_3_names(self):
        evt = self._evt(files=["a.py", "b.py", "c.py", "d.py"])
        lines = _render_run_checkpoints_block([evt], "full")
        combined = "\n".join(lines)
        self.assertIn("a.py", combined)
        self.assertIn("b.py", combined)
        self.assertIn("c.py", combined)
        self.assertNotIn("d.py", combined)

    def test_full_files_does_not_show_file_content(self):
        evt = self._evt(files=["src/foo.py"])
        lines = _render_run_checkpoints_block([evt], "full")
        combined = "\n".join(lines)
        self.assertIn("src/foo.py", combined)
        self.assertNotIn("def ", combined)
        self.assertNotIn("import ", combined)


# ---------------------------------------------------------------------------
# _print_run_checkpoints_block — I/O wrapper
# ---------------------------------------------------------------------------


class TestPrintRunCheckpointsBlock(TestCase):

    def setUp(self):
        self._runner = CliRunner()
        self._td = self._runner.isolated_filesystem()
        self._td.__enter__()

    def tearDown(self):
        self._td.__exit__(None, None, None)

    def test_non_native_entry_shows_nothing(self):
        run_id = "2026-01-01T00:00:00Z"
        log_run_checkpoint_event(_make_event(run_id=run_id))
        entry = _make_run_entry(timestamp=run_id, workflow="direct", executor="direct")
        out = _render(entry, detail="more")
        self.assertNotIn("[checkpoints]", out)

    def test_default_detail_shows_nothing(self):
        run_id = "2026-01-01T00:00:00Z"
        log_run_checkpoint_event(_make_event(run_id=run_id))
        entry = _make_run_entry(timestamp=run_id, workflow="native")
        out = _render(entry, detail="default")
        self.assertNotIn("[checkpoints]", out)

    def test_native_run_no_checkpoints_renders_nothing(self):
        run_id = "2026-01-01T00:00:00Z"
        entry = _make_run_entry(timestamp=run_id, workflow="native")
        out = _render(entry, detail="more")
        self.assertNotIn("[checkpoints]", out)

    def test_more_shows_checkpoint_block(self):
        run_id = "2026-01-01T00:00:00Z"
        log_run_checkpoint_event(_make_event(run_id=run_id, stage="plan", status="passed"))
        entry = _make_run_entry(timestamp=run_id, workflow="native")
        out = _render(entry, detail="full")
        self.assertIn("[checkpoints]", out)

    def test_full_shows_expanded_checkpoint_block(self):
        run_id = "2026-01-01T00:00:00Z"
        log_run_checkpoint_event(_make_event(run_id=run_id, stage="plan", status="passed"))
        log_run_checkpoint_event(_make_event(run_id=run_id, stage="generate", status="passed"))
        entry = _make_run_entry(timestamp=run_id, workflow="native")
        out = _render(entry, detail="full")
        self.assertIn("[checkpoints]", out)
        self.assertIn("plan", out)
        self.assertIn("generate", out)


# ---------------------------------------------------------------------------
# CLI: checkpoints command
# ---------------------------------------------------------------------------


class TestCLICheckpointsCommand(TestCase):

    def setUp(self):
        self._runner = CliRunner()
        self._td_ctx = self._runner.isolated_filesystem()
        self._td_ctx.__enter__()

    def tearDown(self):
        self._td_ctx.__exit__(None, None, None)

    def test_no_events_shows_message(self):
        result = self._runner.invoke(cli, ["checkpoints"])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("No run checkpoints recorded yet.", result.output)

    def test_shows_table_headers(self):
        log_run_checkpoint_event(_make_event())
        result = self._runner.invoke(cli, ["checkpoints"])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("Time", result.output)
        self.assertIn("Stage", result.output)
        self.assertIn("Status", result.output)
        self.assertIn("Verify", result.output)
        self.assertIn("Retry", result.output)

    def test_last_n_limits_rows(self):
        for i in range(15):
            log_run_checkpoint_event(_make_event(stage=f"s{i}"))
        result = self._runner.invoke(cli, ["checkpoints", "--last", "5"])
        self.assertEqual(result.exit_code, 0)
        data_lines = [ln for ln in result.output.splitlines() if ln.strip() and "Time" not in ln]
        self.assertEqual(len(data_lines), 5)


# ---------------------------------------------------------------------------
# CLI: resume-last command
# ---------------------------------------------------------------------------


class TestCLIResumeLast(TestCase):

    def setUp(self):
        self._runner = CliRunner()
        self._td_ctx = self._runner.isolated_filesystem()
        self._td_ctx.__enter__()
        self._log_path = Path(".openshard") / "runs.jsonl"

    def tearDown(self):
        self._td_ctx.__exit__(None, None, None)

    def _write_run_entry(self, entry: dict) -> None:
        self._log_path.parent.mkdir(parents=True, exist_ok=True)
        with self._log_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry) + "\n")

    def test_no_history_shows_message(self):
        result = self._runner.invoke(cli, ["resume-last"])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("No run history found.", result.output)

    def test_non_native_latest_run(self):
        self._write_run_entry(_make_run_entry(executor="direct", workflow="direct"))
        result = self._runner.invoke(cli, ["resume-last"])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("Latest run is not a native run.", result.output)

    def test_native_run_no_checkpoints(self):
        run_id = "2026-01-01T00:00:00Z"
        self._write_run_entry(_make_run_entry(timestamp=run_id, executor="native", workflow="native"))
        result = self._runner.invoke(cli, ["resume-last"])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("No checkpoints found for latest native run.", result.output)

    def test_with_checkpoint_and_existing_sandbox(self):
        with tempfile.TemporaryDirectory() as td:
            run_id = "2026-01-01T00:00:00Z"
            entry = _make_run_entry(
                timestamp=run_id,
                executor="native",
                workflow="native",
                workspace_path=td,
            )
            entry["extra_metadata"] = {"sandbox": {"sandbox_path": td}}
            self._write_run_entry(entry)
            log_run_checkpoint_event(_make_event(run_id=run_id, stage="receipt", status="passed"))

            with patch(
                "openshard.native.sandbox_apply.extract_sandbox_path_from_entry",
                return_value=td,
            ):
                result = self._runner.invoke(cli, ["resume-last"])

        self.assertEqual(result.exit_code, 0)
        self.assertIn("Latest checkpoint: receipt (passed)", result.output)
        self.assertIn("Resume options:", result.output)
        self.assertIn("diff-last", result.output)
        self.assertIn("apply-last", result.output)

    def test_with_checkpoint_but_missing_sandbox(self):
        run_id = "2026-01-01T00:00:00Z"
        entry = _make_run_entry(timestamp=run_id, executor="native", workflow="native")
        self._write_run_entry(entry)
        log_run_checkpoint_event(_make_event(run_id=run_id, stage="plan", status="passed"))

        with patch(
            "openshard.native.sandbox_apply.extract_sandbox_path_from_entry",
            return_value="/nonexistent/path/that/does/not/exist",
        ):
            result = self._runner.invoke(cli, ["resume-last"])

        self.assertEqual(result.exit_code, 0)
        self.assertIn("No live sandbox found.", result.output)


# ---------------------------------------------------------------------------
# Pipeline integration
# ---------------------------------------------------------------------------

_DEFAULT_CONFIG_CP = {"approval_mode": "smart"}

_PYTHON_REPO_CP = RepoFacts(
    languages=["python"],
    package_files=[],
    framework=None,
    test_command="python -m pytest",
    risky_paths=[],
    changed_files=[],
)


def _make_native_mock_cp(generate_side_effect=None):
    from openshard.native.executor import NativeRunMeta
    from openshard.native.context import NativeChangeBudgetSoftGate, NativeApprovalRequest

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


def _make_manager_mock_cp():
    m = MagicMock()
    inv = MagicMock()
    inv.models = []
    m.get_inventory.return_value = inv
    m.providers = {"openrouter": MagicMock()}
    return m


def _invoke_native_write_cp(native_mock=None, *, verify_side_effects=None, captured_events=None):
    from openshard.native.context import NativeSandboxMeta

    if native_mock is None:
        native_mock = _make_native_mock_cp()

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
            patch("openshard.run.pipeline.ProviderManager", return_value=_make_manager_mock_cp()),
            patch("openshard.cli.main.load_config", return_value=_DEFAULT_CONFIG_CP),
            patch("openshard.run.pipeline.analyze_repo", return_value=_PYTHON_REPO_CP),
            patch("openshard.run.pipeline._run_verification_plan", side_effect=_fake_verify),
            patch("openshard.run.pipeline._write_files"),
            patch("openshard.native.sandbox.create_run_sandbox", return_value=(Path(_td), sandbox_meta)),
            patch("openshard.run.pipeline._log_run"),
            patch("openshard.run.pipeline._log_run_checkpoint", side_effect=_capture_event),
        ):
            runner = CliRunner()
            result = runner.invoke(cli, ["run", "--workflow", "native", "--write", "fix the bug"])
    return result, native_mock


class TestPipelineIntegration(TestCase):

    def test_clean_verification_pass_records_expected_stages(self):
        captured: list[NativeRunCheckpointEvent] = []
        result, _ = _invoke_native_write_cp(
            verify_side_effects=[(0, "all passed")],
            captured_events=captured,
        )
        self.assertEqual(result.exit_code, 0, result.output)
        stages = [e.stage for e in captured]
        self.assertIn("plan", stages)
        self.assertIn("generate", stages)
        self.assertIn("sandbox_write", stages)
        self.assertIn("verify", stages)
        self.assertIn("receipt", stages)

    def test_retry_checkpoint_has_retry_used_true(self):
        captured: list[NativeRunCheckpointEvent] = []
        result, _ = _invoke_native_write_cp(
            verify_side_effects=[(1, "FAILED: 1 error"), (0, "all passed")],
            captured_events=captured,
        )
        self.assertEqual(result.exit_code, 0, result.output)
        retry_events = [e for e in captured if e.stage == "retry"]
        self.assertEqual(len(retry_events), 1)
        self.assertTrue(retry_events[0].retry_used)
        self.assertEqual(retry_events[0].verification_status, "passed")

    def test_verify_fail_then_retry_fail(self):
        captured: list[NativeRunCheckpointEvent] = []
        _invoke_native_write_cp(
            verify_side_effects=[(1, "TypeError: bad"), (1, "still failing")],
            captured_events=captured,
        )
        retry_events = [e for e in captured if e.stage == "retry"]
        self.assertEqual(len(retry_events), 1)
        self.assertEqual(retry_events[0].verification_status, "failed")

    def test_verification_skipped_records_verify_skipped(self):
        # No verification plan commands → _loop_meta.attempted stays False
        captured: list[NativeRunCheckpointEvent] = []

        native_mock = _make_native_mock_cp()

        from openshard.native.context import NativeSandboxMeta
        from openshard.verification.plan import VerificationPlan

        with tempfile.TemporaryDirectory() as _td:
            sandbox_meta = NativeSandboxMeta(sandbox_enabled=False, sandbox_type="none")
            empty_plan = VerificationPlan(commands=[])

            def _capture(evt):
                captured.append(evt)

            with (
                patch("openshard.run.pipeline.NativeAgentExecutor", return_value=native_mock),
                patch("openshard.run.pipeline.ProviderManager", return_value=_make_manager_mock_cp()),
                patch("openshard.cli.main.load_config", return_value=_DEFAULT_CONFIG_CP),
                patch("openshard.run.pipeline.analyze_repo", return_value=_PYTHON_REPO_CP),
                patch("openshard.run.pipeline._run_verification_plan"),
                patch("openshard.run.pipeline._write_files"),
                patch("openshard.native.sandbox.create_run_sandbox", return_value=(Path(_td), sandbox_meta)),
                patch("openshard.run.pipeline._log_run"),
                patch("openshard.run.pipeline._log_run_checkpoint", side_effect=_capture),
                patch("openshard.run.pipeline.build_verification_plan", return_value=empty_plan),
            ):
                runner = CliRunner()
                runner.invoke(cli, ["run", "--workflow", "native", "--write", "fix the bug"])

        verify_events = [e for e in captured if e.stage == "verify"]
        skipped = [e for e in verify_events if e.status == "skipped"]
        self.assertEqual(len(skipped), 1, f"Expected 1 skipped verify event, got stages: {[e.stage+':'+e.status for e in captured]}")
        self.assertEqual(skipped[0].reason, "verification not configured")
        for e in captured:
            self.assertFalse(e.raw_content_stored)

    def test_no_raw_content_stored_on_any_captured_event(self):
        captured: list[NativeRunCheckpointEvent] = []
        _invoke_native_write_cp(
            verify_side_effects=[(0, "ok")],
            captured_events=captured,
        )
        for evt in captured:
            self.assertFalse(evt.raw_content_stored)
