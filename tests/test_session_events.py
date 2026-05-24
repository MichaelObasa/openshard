from __future__ import annotations

import json
from unittest.mock import patch

import pytest

from openshard.history.session_events import SessionEventWriter, _truncate


# ── _truncate ───────────────────────────────────────────────────────────────


def test_truncate_short_string_unchanged():
    assert _truncate("hello", 300) == "hello"


def test_truncate_long_string_clipped():
    text = "x" * 500
    result = _truncate(text, 300)
    assert len(result) == 300
    assert result == "x" * 300


def test_truncate_exact_limit_unchanged():
    text = "y" * 300
    assert _truncate(text, 300) == text


# ── SessionEventWriter ──────────────────────────────────────────────────────


def test_writer_appends_valid_jsonl(tmp_path):
    writer = SessionEventWriter(base_path=tmp_path)
    writer.write("session_started", "sid-1", summary="TUI session started")

    events_file = tmp_path / ".openshard" / "session_events.jsonl"
    assert events_file.exists()
    lines = events_file.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    event = json.loads(lines[0])
    assert event["event_type"] == "session_started"
    assert event["session_id"] == "sid-1"
    assert event["schema_version"] == 1
    assert event["source"] == "tui"
    assert event["raw_text_stored"] is False
    assert "event_id" in event
    assert "timestamp" in event


def test_session_id_stable_across_writes(tmp_path):
    writer = SessionEventWriter(base_path=tmp_path)
    sid = "stable-session-id"
    writer.write("session_started", sid)
    writer.write("command_invoked", sid, command="/help")

    lines = (tmp_path / ".openshard" / "session_events.jsonl").read_text().splitlines()
    assert len(lines) == 2
    for line in lines:
        assert json.loads(line)["session_id"] == sid


def test_command_invoked_event_shape(tmp_path):
    writer = SessionEventWriter(base_path=tmp_path)
    writer.write("command_invoked", "s1", command="/last full", summary="last_full")

    event = json.loads((tmp_path / ".openshard" / "session_events.jsonl").read_text())
    assert event["event_type"] == "command_invoked"
    assert event["command"] == "/last full"
    assert event["summary"] == "last_full"
    assert event["raw_text_stored"] is False
    assert event["schema_version"] == 1


def test_user_message_truncated_at_300(tmp_path):
    writer = SessionEventWriter(base_path=tmp_path)
    long_task = "a" * 500
    writer.write("user_message", "s1", summary=_truncate(long_task, 300))

    event = json.loads((tmp_path / ".openshard" / "session_events.jsonl").read_text())
    assert len(event["summary"]) == 300


def test_run_completed_links_run_id(tmp_path):
    writer = SessionEventWriter(base_path=tmp_path)
    writer.write(
        "openshard_response",
        "s1",
        run_id="2026-05-24T10:00:00Z",
        shard_id="shard-20260524-0001",
        summary="OpenShard response completed",
        metadata={"exit_code": 0, "is_run": True},
    )

    event = json.loads((tmp_path / ".openshard" / "session_events.jsonl").read_text())
    assert event["run_id"] == "2026-05-24T10:00:00Z"
    assert event["shard_id"] == "shard-20260524-0001"
    assert event["metadata"]["exit_code"] == 0
    assert event["metadata"]["is_run"] is True


def test_missing_openshard_dir_auto_created(tmp_path):
    base = tmp_path / "nested" / "workspace"
    base.mkdir(parents=True)
    writer = SessionEventWriter(base_path=base)
    writer.write("session_started", "s1")
    assert (base / ".openshard" / "session_events.jsonl").exists()


def test_write_error_does_not_raise(tmp_path):
    writer = SessionEventWriter(base_path=tmp_path)
    with patch("pathlib.Path.mkdir", side_effect=OSError("no space")):
        writer.write("session_started", "s1")  # must not raise


def test_multiple_events_appended_in_order(tmp_path):
    writer = SessionEventWriter(base_path=tmp_path)
    for i, et in enumerate(["session_started", "user_message", "openshard_response"]):
        writer.write(et, "s1", summary=f"event-{i}")

    lines = (tmp_path / ".openshard" / "session_events.jsonl").read_text().splitlines()
    assert len(lines) == 3
    types = [json.loads(line)["event_type"] for line in lines]
    assert types == ["session_started", "user_message", "openshard_response"]


# ── TUI integration ─────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_session_started_event_on_mount(tmp_path):
    from openshard.tui.app import OpenShardTui
    from openshard.tui.state import get_guardrails

    app = OpenShardTui(path=tmp_path)
    app._git_info = {"project_name": "test", "branch": "main", "state": "clean"}
    app._recent_runs = []
    app._guardrails = get_guardrails()

    async with app.run_test(size=(120, 55)):
        pass

    events_file = tmp_path / ".openshard" / "session_events.jsonl"
    assert events_file.exists()
    lines = events_file.read_text().splitlines()
    types = [json.loads(line)["event_type"] for line in lines]
    assert "session_started" in types


@pytest.mark.asyncio
async def test_clear_does_not_delete_session_events(tmp_path):
    from openshard.tui.app import OpenShardTui, TaskInput
    from openshard.tui.state import get_guardrails

    app = OpenShardTui(path=tmp_path)
    app._git_info = {"project_name": "test", "branch": "main", "state": "clean"}
    app._recent_runs = []
    app._guardrails = get_guardrails()

    events_file = tmp_path / ".openshard" / "session_events.jsonl"

    async with app.run_test(size=(120, 55)) as pilot:
        ta = app.query_one("#task-input", TaskInput)
        ta.focus()
        ta.load_text("/clear")
        await pilot.press("enter")
        await pilot.pause(delay=0.1)

    assert events_file.exists()
    lines = events_file.read_text().splitlines()
    assert len(lines) >= 1  # at least session_started remains
