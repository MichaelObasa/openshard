from __future__ import annotations

import unittest

from openshard.run.timeline import (
    _MAX_TIMELINE_EVENTS,
    make_timeline_event,
    project_timeline_for_export,
)


class TestMakeTimelineEvent(unittest.TestCase):
    def test_safe_event_passes_through_unchanged(self):
        ev = make_timeline_event("repo_scanned", "Scanned repo", kind="scan")
        self.assertEqual(ev.event, "repo_scanned")
        self.assertEqual(ev.label, "Scanned repo")
        self.assertEqual(ev.kind, "scan")
        self.assertEqual(ev.status, "completed")

    def test_strips_control_chars_and_newlines(self):
        ev = make_timeline_event("x", "line1\r\nline2\tend", kind="run")
        self.assertNotIn("\n", ev.label)
        self.assertNotIn("\r", ev.label)
        self.assertNotIn("\t", ev.label)

    def test_caps_long_label(self):
        # Use a spaced phrase so it is not mistaken for an opaque secret token.
        ev = make_timeline_event("x", "word " * 100, kind="run")
        self.assertLessEqual(len(ev.label), 80)

    def test_caps_long_detail_and_target(self):
        ev = make_timeline_event(
            "x", "ok", kind="run", detail="word " * 100, target="part " * 100
        )
        self.assertLessEqual(len(ev.detail), 120)
        self.assertLessEqual(len(ev.target), 80)

    def test_drops_absolute_path_target(self):
        for bad in (r"C:\secret\key.txt", "/Users/alice/.env", "/home/bob/creds"):
            ev = make_timeline_event("x", "ok", kind="run", target=bad)
            self.assertIsNone(ev.target, msg=bad)

    def test_invalid_status_falls_back(self):
        ev = make_timeline_event("x", "ok", kind="run", status="bogus")
        self.assertEqual(ev.status, "completed")

    def test_invalid_kind_falls_back(self):
        ev = make_timeline_event("x", "ok", kind="bogus")
        self.assertEqual(ev.kind, "run")

    def test_count_coercion(self):
        self.assertEqual(make_timeline_event("x", "ok", count=3).count, 3)
        self.assertIsNone(make_timeline_event("x", "ok", count=-1).count)
        self.assertIsNone(make_timeline_event("x", "ok", count=True).count)
        self.assertIsNone(make_timeline_event("x", "ok", count="5").count)

    def test_label_fallback_when_dropped(self):
        # An all-secret label is dropped, then derived from the event key.
        ev = make_timeline_event("repo_scanned", "sk-abcdefghijklmnop", kind="scan")
        self.assertTrue(ev.label)
        self.assertNotIn("sk-abcdefghijklmnop", ev.label)
        self.assertEqual(ev.label, "repo scanned")

    def test_metadata_scalars_only(self):
        ev = make_timeline_event(
            "x", "ok",
            metadata={
                "n": 5, "f": 1.5, "b": True, "s": "fine",
                "nested": {"a": 1}, "lst": [1, 2], "none": None,
            },
        )
        self.assertEqual(ev.metadata["n"], 5)
        self.assertEqual(ev.metadata["f"], 1.5)
        self.assertEqual(ev.metadata["b"], True)
        self.assertEqual(ev.metadata["s"], "fine")
        self.assertNotIn("nested", ev.metadata)
        self.assertNotIn("lst", ev.metadata)
        self.assertNotIn("none", ev.metadata)


class TestSecretRedaction(unittest.TestCase):
    SECRETS = (
        "sk-ABCDEFGHIJKLMNOP1234",
        "AKIA1234567890ABCDEF",
        "token=supersecretvalue",
        "api_key=abc123def456",
        "apikey: abc123def456",
        "secret=hunter2hunter2",
        "password=hunter2hunter2",
        "Bearer ya29.abcdefg",
        "A" * 40,  # long opaque key-like run
    )

    def test_secret_dropped_from_fields(self):
        for secret in self.SECRETS:
            ev = make_timeline_event(
                "x", f"prefix {secret}", kind="run",
                detail=secret, target=secret,
                metadata={"v": secret},
            )
            self.assertNotIn(secret, ev.label, msg=secret)
            self.assertIsNone(ev.detail, msg=secret)
            self.assertIsNone(ev.target, msg=secret)
            self.assertNotIn("v", ev.metadata, msg=secret)

    def test_no_secret_in_projected_output(self):
        events = [
            {"event": "x", "label": "got sk-ABCDEFGHIJKLMNOP1234",
             "kind": "run", "status": "completed",
             "detail": "token=leakme1234567890", "target": "AKIA1234567890ABCDEF",
             "metadata": {"k": "secret=leakvalue123"}},
        ]
        out = project_timeline_for_export(events)
        blob = repr(out)
        for needle in ("sk-ABCDEFGHIJKLMNOP", "token=leakme", "AKIA1234567890", "secret=leak"):
            self.assertNotIn(needle, blob, msg=needle)


class TestProjectTimelineForExport(unittest.TestCase):
    def test_empty_returns_empty_list(self):
        self.assertEqual(project_timeline_for_export([]), [])
        self.assertEqual(project_timeline_for_export(None), [])

    def test_stable_shape(self):
        out = project_timeline_for_export(
            [{"event": "repo_scanned", "label": "Scanned repo", "kind": "scan",
              "status": "completed", "count": 2, "target": "ok", "detail": "fine"}]
        )
        self.assertEqual(len(out), 1)
        row = out[0]
        self.assertEqual(row["event"], "repo_scanned")
        self.assertEqual(row["label"], "Scanned repo")
        self.assertEqual(row["kind"], "scan")
        self.assertEqual(row["status"], "completed")
        self.assertEqual(row["count"], 2)
        self.assertEqual(row["target"], "ok")
        self.assertEqual(row["detail"], "fine")

    def test_omits_empty_optionals(self):
        out = project_timeline_for_export(
            [{"event": "x", "label": "Did something", "kind": "run", "status": "completed"}]
        )
        self.assertNotIn("detail", out[0])
        self.assertNotIn("target", out[0])
        self.assertNotIn("count", out[0])

    def test_drops_events_without_label(self):
        out = project_timeline_for_export(
            [{"event": "x", "label": "", "kind": "run"},
             {"event": "y", "label": "Real", "kind": "run"}]
        )
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["label"], "Real")

    def test_dedupes_receipt_saved(self):
        out = project_timeline_for_export(
            [{"event": "receipt_saved", "label": "Saved Shard receipt", "kind": "receipt"},
             {"event": "receipt_saved", "label": "Saved Shard receipt", "kind": "receipt"}]
        )
        self.assertEqual(sum(1 for e in out if e["event"] == "receipt_saved"), 1)

    def test_caps_event_count(self):
        many = [{"event": f"e{i}", "label": f"Event {i}", "kind": "run"} for i in range(100)]
        out = project_timeline_for_export(many)
        self.assertLessEqual(len(out), _MAX_TIMELINE_EVENTS)

    def test_failed_status_preserved(self):
        out = project_timeline_for_export(
            [{"event": "model_response_received", "label": "Model request failed",
              "kind": "model", "status": "failed"}]
        )
        self.assertEqual(out[0]["status"], "failed")


if __name__ == "__main__":
    unittest.main()
