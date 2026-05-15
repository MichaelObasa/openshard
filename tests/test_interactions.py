from __future__ import annotations

import json
import unittest
from pathlib import Path

from click.testing import CliRunner

from openshard.cli.main import cli
from openshard.history.interactions import (
    DeveloperInteractionEvent,
    _dict_to_event,
    _event_to_dict,
    interaction_events_for_run,
    load_interaction_events,
    log_interaction_event,
)


def _make_event(**kwargs) -> DeveloperInteractionEvent:
    defaults: dict = dict(
        run_id="2025-01-01T00:00:00Z",
        event_type="feedback_accepted",
        summary="test event",
    )
    defaults.update(kwargs)
    return DeveloperInteractionEvent(**defaults)


def _write_interactions(events: list[DeveloperInteractionEvent]) -> Path:
    path = Path(".openshard") / "interactions.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for evt in events:
            fh.write(json.dumps(_event_to_dict(evt)) + "\n")
    return path


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
    }
    base.update(kwargs)
    return base


def _write_runs(entries: list[dict]) -> Path:
    path = Path(".openshard") / "runs.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for e in entries:
            fh.write(json.dumps(e) + "\n")
    return path


class TestEventSerialization(unittest.TestCase):

    def test_event_serialization(self):
        evt = DeveloperInteractionEvent(
            run_id="run-123",
            event_type="feedback_rejected",
            summary="rejected bad output",
            related_stage="execution",
            related_file_paths=["src/foo.py"],
            correction_reason="wrong-file",
            severity="warning",
            accepted=False,
            metadata={"rating": "bad"},
        )
        d = _event_to_dict(evt)
        self.assertEqual(d["run_id"], "run-123")
        self.assertEqual(d["event_type"], "feedback_rejected")
        self.assertEqual(d["summary"], "rejected bad output")
        self.assertEqual(d["related_stage"], "execution")
        self.assertEqual(d["related_file_paths"], ["src/foo.py"])
        self.assertEqual(d["correction_reason"], "wrong-file")
        self.assertEqual(d["severity"], "warning")
        self.assertIs(d["accepted"], False)
        self.assertEqual(d["metadata"], {"rating": "bad"})
        self.assertIs(d["raw_content_stored"], False)

        roundtrip = _dict_to_event(d)
        self.assertEqual(roundtrip.run_id, "run-123")
        self.assertEqual(roundtrip.event_type, "feedback_rejected")
        self.assertEqual(roundtrip.correction_reason, "wrong-file")
        self.assertIs(roundtrip.accepted, False)
        self.assertIs(roundtrip.raw_content_stored, False)

    def test_all_required_keys_present(self):
        d = _event_to_dict(_make_event())
        for key in (
            "schema_version", "event_id", "run_id", "timestamp", "actor",
            "event_type", "summary", "related_stage", "related_file_paths",
            "correction_reason", "severity", "accepted", "metadata",
            "raw_content_stored",
        ):
            self.assertIn(key, d, f"missing key: {key}")


class TestAppendOnlyLogging(unittest.TestCase):

    def test_append_only_logging(self):
        runner = CliRunner()
        with runner.isolated_filesystem():
            log_interaction_event(_make_event(event_type="feedback_accepted"))
            log_interaction_event(_make_event(event_type="feedback_rejected"))
            log_interaction_event(_make_event(event_type="feedback_edited"))

            path = Path(".openshard") / "interactions.jsonl"
            lines = [ln for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
            self.assertEqual(len(lines), 3)
            types = [json.loads(ln)["event_type"] for ln in lines]
            self.assertEqual(types, ["feedback_accepted", "feedback_rejected", "feedback_edited"])

    def test_append_does_not_overwrite(self):
        runner = CliRunner()
        with runner.isolated_filesystem():
            log_interaction_event(_make_event(event_type="first"))
            log_interaction_event(_make_event(event_type="second"))
            events = load_interaction_events()
            self.assertEqual(events[0].event_type, "first")
            self.assertEqual(events[1].event_type, "second")


class TestLoadInteractionEvents(unittest.TestCase):

    def test_load_interaction_events(self):
        runner = CliRunner()
        with runner.isolated_filesystem():
            evt_a = _make_event(run_id="run-A", event_type="feedback_accepted", summary="A")
            evt_b = _make_event(run_id="run-B", event_type="feedback_rejected", summary="B")
            _write_interactions([evt_a, evt_b])

            loaded = load_interaction_events()
            self.assertEqual(len(loaded), 2)
            self.assertIsInstance(loaded[0], DeveloperInteractionEvent)
            self.assertEqual(loaded[0].run_id, "run-A")
            self.assertEqual(loaded[0].event_type, "feedback_accepted")
            self.assertEqual(loaded[1].run_id, "run-B")
            self.assertEqual(loaded[1].event_type, "feedback_rejected")

    def test_load_skips_malformed_lines(self):
        runner = CliRunner()
        with runner.isolated_filesystem():
            path = Path(".openshard") / "interactions.jsonl"
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("w", encoding="utf-8") as fh:
                fh.write(json.dumps(_event_to_dict(_make_event(event_type="good"))) + "\n")
                fh.write("not valid json\n")
                fh.write("\n")
                fh.write(json.dumps(_event_to_dict(_make_event(event_type="also-good"))) + "\n")

            loaded = load_interaction_events()
            self.assertEqual(len(loaded), 2)
            self.assertEqual(loaded[0].event_type, "good")
            self.assertEqual(loaded[1].event_type, "also-good")


class TestFilterByRunId(unittest.TestCase):

    def test_filter_by_run_id(self):
        runner = CliRunner()
        with runner.isolated_filesystem():
            _write_interactions([
                _make_event(run_id="run-A", event_type="feedback_accepted"),
                _make_event(run_id="run-A", event_type="feedback_edited"),
                _make_event(run_id="run-B", event_type="feedback_rejected"),
            ])
            a_events = interaction_events_for_run("run-A")
            self.assertEqual(len(a_events), 2)
            b_events = interaction_events_for_run("run-B")
            self.assertEqual(len(b_events), 1)
            c_events = interaction_events_for_run("run-C")
            self.assertEqual(c_events, [])


class TestMissingFileReturnsEmptyList(unittest.TestCase):

    def test_missing_file_returns_empty_list(self):
        runner = CliRunner()
        with runner.isolated_filesystem():
            result = load_interaction_events()
            self.assertEqual(result, [])

    def test_missing_file_for_run_returns_empty_list(self):
        runner = CliRunner()
        with runner.isolated_filesystem():
            result = interaction_events_for_run("run-X")
            self.assertEqual(result, [])


class TestRedactedExport(unittest.TestCase):

    def test_redacted_export(self):
        runner = CliRunner()
        with runner.isolated_filesystem():
            _write_interactions([
                _make_event(
                    run_id="run-1",
                    event_type="feedback_rejected",
                    summary="sensitive details here",
                    metadata={"rating": "bad", "note": "private note"},
                )
            ])
            result = runner.invoke(cli, ["export-interactions", "--redacted"])
            self.assertEqual(result.exit_code, 0)
            row = json.loads(result.output.strip())
            self.assertEqual(row["summary"], "[redacted]")
            self.assertEqual(row["metadata"], {})
            self.assertEqual(row["event_type"], "feedback_rejected")
            self.assertEqual(row["run_id"], "run-1")
            self.assertIn("timestamp", row)
            self.assertIs(row["raw_content_stored"], False)

    def test_unredacted_export_preserves_fields(self):
        runner = CliRunner()
        with runner.isolated_filesystem():
            _write_interactions([
                _make_event(
                    summary="original summary",
                    metadata={"rating": "good"},
                )
            ])
            result = runner.invoke(cli, ["export-interactions"])
            self.assertEqual(result.exit_code, 0)
            row = json.loads(result.output.strip())
            self.assertEqual(row["summary"], "original summary")
            self.assertEqual(row["metadata"], {"rating": "good"})

    def test_redacted_export_to_file(self):
        runner = CliRunner()
        with runner.isolated_filesystem():
            _write_interactions([_make_event(summary="private")])
            result = runner.invoke(cli, ["export-interactions", "--redacted", "--output", "out.jsonl"])
            self.assertEqual(result.exit_code, 0)
            row = json.loads(Path("out.jsonl").read_text(encoding="utf-8").strip())
            self.assertEqual(row["summary"], "[redacted]")


class TestFeedbackCorrectionIntegration(unittest.TestCase):

    def test_feedback_rejected_logs_interaction_event(self):
        runner = CliRunner()
        with runner.isolated_filesystem():
            _write_runs([_make_run_entry(timestamp="2025-06-01T12:00:00Z")])
            result = runner.invoke(
                cli,
                ["feedback", "--action", "rejected", "--reason", "wrong-file", "--rating", "bad"],
            )
            self.assertEqual(result.exit_code, 0)
            events = load_interaction_events()
            self.assertEqual(len(events), 1)
            evt = events[0]
            self.assertEqual(evt.event_type, "feedback_rejected")
            self.assertIs(evt.accepted, False)
            self.assertEqual(evt.run_id, "2025-06-01T12:00:00Z")
            self.assertEqual(evt.correction_reason, "wrong-file")
            self.assertEqual(evt.metadata.get("rating"), "bad")
            self.assertIs(evt.raw_content_stored, False)

    def test_feedback_accepted_logs_accepted_true(self):
        runner = CliRunner()
        with runner.isolated_filesystem():
            _write_runs([_make_run_entry(timestamp="2025-06-01T12:00:00Z")])
            result = runner.invoke(cli, ["feedback", "--action", "accepted", "--rating", "good"])
            self.assertEqual(result.exit_code, 0)
            events = load_interaction_events()
            self.assertEqual(len(events), 1)
            self.assertIs(events[0].accepted, True)
            self.assertEqual(events[0].event_type, "feedback_accepted")

    def test_feedback_note_only_logs_event(self):
        runner = CliRunner()
        with runner.isolated_filesystem():
            _write_runs([_make_run_entry()])
            result = runner.invoke(cli, ["feedback", "--note", "looks fine"])
            self.assertEqual(result.exit_code, 0)
            events = load_interaction_events()
            self.assertEqual(len(events), 1)
            self.assertEqual(events[0].event_type, "feedback_noted")

    def test_feedback_partially_accepted_logs_accepted_true(self):
        runner = CliRunner()
        with runner.isolated_filesystem():
            _write_runs([_make_run_entry()])
            result = runner.invoke(cli, ["feedback", "--action", "partially-accepted"])
            self.assertEqual(result.exit_code, 0)
            events = load_interaction_events()
            self.assertIs(events[0].accepted, True)
            self.assertEqual(events[0].event_type, "feedback_partially_accepted")


class TestRawContentStoredRemainsFalse(unittest.TestCase):

    def test_post_init_enforces_false(self):
        evt = DeveloperInteractionEvent(raw_content_stored=True)
        self.assertIs(evt.raw_content_stored, False)

    def test_stored_json_has_false(self):
        runner = CliRunner()
        with runner.isolated_filesystem():
            evt = DeveloperInteractionEvent(raw_content_stored=True)
            log_interaction_event(evt)
            path = Path(".openshard") / "interactions.jsonl"
            stored = json.loads(path.read_text(encoding="utf-8").strip())
            self.assertIs(stored["raw_content_stored"], False)

    def test_loaded_event_has_false(self):
        runner = CliRunner()
        with runner.isolated_filesystem():
            path = Path(".openshard") / "interactions.jsonl"
            path.parent.mkdir(parents=True, exist_ok=True)
            record = _event_to_dict(_make_event())
            record["raw_content_stored"] = True
            path.write_text(json.dumps(record) + "\n", encoding="utf-8")
            loaded = load_interaction_events()
            self.assertEqual(len(loaded), 1)
            self.assertIs(loaded[0].raw_content_stored, False)

    def test_dict_to_event_always_false(self):
        d = _event_to_dict(_make_event())
        d["raw_content_stored"] = True
        evt = _dict_to_event(d)
        self.assertIs(evt.raw_content_stored, False)


class TestInteractionsCLI(unittest.TestCase):

    def test_interactions_no_events(self):
        runner = CliRunner()
        with runner.isolated_filesystem():
            result = runner.invoke(cli, ["interactions"])
            self.assertEqual(result.exit_code, 0)
            self.assertIn("No interaction events", result.output)

    def test_interactions_shows_recent(self):
        runner = CliRunner()
        with runner.isolated_filesystem():
            _write_interactions([
                _make_event(event_type="feedback_accepted", summary="first"),
                _make_event(event_type="feedback_rejected", summary="second"),
            ])
            result = runner.invoke(cli, ["interactions", "--last", "1"])
            self.assertEqual(result.exit_code, 0)
            self.assertIn("feedback_rejected", result.output)
            self.assertNotIn("feedback_accepted", result.output)

    def test_export_interactions_no_events(self):
        runner = CliRunner()
        with runner.isolated_filesystem():
            result = runner.invoke(cli, ["export-interactions"])
            self.assertEqual(result.exit_code, 0)
            self.assertIn("No interaction events", result.output)

    def test_export_interactions_jsonl_output(self):
        runner = CliRunner()
        with runner.isolated_filesystem():
            _write_interactions([
                _make_event(event_type="feedback_accepted"),
                _make_event(event_type="feedback_rejected"),
            ])
            result = runner.invoke(cli, ["export-interactions"])
            self.assertEqual(result.exit_code, 0)
            lines = [ln for ln in result.output.strip().splitlines() if ln.strip()]
            self.assertEqual(len(lines), 2)
            for ln in lines:
                row = json.loads(ln)
                self.assertIn("event_type", row)
                self.assertIs(row["raw_content_stored"], False)


if __name__ == "__main__":
    unittest.main()
