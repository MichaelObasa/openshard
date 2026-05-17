from __future__ import annotations

import pytest
from textual.widgets import Input, Label, Static

from openshard.tui.app import OpenShardTui
from openshard.tui.state import get_guardrails

_SIZE = (120, 55)


def _make_app(tmp_path, recent_runs=None, git_info=None):
    app = OpenShardTui(path=tmp_path)
    app._git_info = git_info or {"project_name": "test-project", "branch": "main", "state": "clean"}
    app._recent_runs = [] if recent_runs is None else recent_runs
    app._guardrails = get_guardrails()
    return app


def _text(widget) -> str:
    return str(widget.content)


# ── Hero box: left column ──────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_hero_brand_contains_openshard(tmp_path):
    app = _make_app(tmp_path)
    async with app.run_test(size=_SIZE) as _:
        assert "OpenShard" in _text(app.query_one("#hero-brand", Static))


@pytest.mark.asyncio
async def test_hero_tagline_contains_expected_text(tmp_path):
    app = _make_app(tmp_path)
    async with app.run_test(size=_SIZE) as _:
        assert "control layer for AI" in _text(app.query_one("#hero-tagline", Static))


@pytest.mark.asyncio
async def test_hero_project_shows_project_name(tmp_path):
    app = _make_app(tmp_path)
    async with app.run_test(size=_SIZE) as _:
        assert "test-project" in _text(app.query_one("#hero-project", Static))


@pytest.mark.asyncio
async def test_hero_project_shows_branch(tmp_path):
    app = _make_app(tmp_path)
    async with app.run_test(size=_SIZE) as _:
        assert "main" in _text(app.query_one("#hero-project", Static))


@pytest.mark.asyncio
async def test_hero_project_shows_clean_state(tmp_path):
    app = _make_app(tmp_path)
    async with app.run_test(size=_SIZE) as _:
        assert "Clean" in _text(app.query_one("#hero-project", Static))


@pytest.mark.asyncio
async def test_dirty_state_renders_as_changes_pending(tmp_path):
    app = _make_app(tmp_path, git_info={"project_name": "p", "branch": "main", "state": "dirty"})
    async with app.run_test(size=_SIZE) as _:
        text = _text(app.query_one("#hero-project", Static))
        assert "Changes pending" in text
        assert "dirty" not in text


# ── Hero box: right column ─────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_hero_controls_heading_says_run_controls(tmp_path):
    app = _make_app(tmp_path)
    async with app.run_test(size=_SIZE) as _:
        assert "Run Controls" in _text(app.query_one("#hero-controls-heading", Label))


@pytest.mark.asyncio
async def test_hero_controls_shows_all_guardrail_keys(tmp_path):
    app = _make_app(tmp_path)
    async with app.run_test(size=_SIZE) as _:
        text = _text(app.query_one("#hero-controls", Static))
        for key in ("Agent", "Model", "Sandbox", "Approval", "Checks", "Receipts"):
            assert key in text


@pytest.mark.asyncio
async def test_hero_controls_shows_guardrail_values(tmp_path):
    app = _make_app(tmp_path)
    async with app.run_test(size=_SIZE) as _:
        text = _text(app.query_one("#hero-controls", Static))
        assert "OpenShard Native" in text
        assert "Smart" in text


@pytest.mark.asyncio
async def test_recent_activity_shows_only_latest_run(tmp_path):
    runs = [
        {"task": "new task", "timestamp": "2026-01-02T00:00", "status": "passed", "duration": "1.0s"},
        {"task": "old task", "timestamp": "2026-01-01T00:00", "status": "failed", "duration": "3.0s"},
    ]
    app = _make_app(tmp_path, recent_runs=runs)
    async with app.run_test(size=_SIZE) as _:
        text = _text(app.query_one("#hero-activity", Static))
        assert "new task" in text
        assert "old task" not in text


@pytest.mark.asyncio
async def test_recent_activity_no_runs_fallback(tmp_path):
    app = _make_app(tmp_path, recent_runs=[])
    async with app.run_test(size=_SIZE) as _:
        assert "No recent activity" in _text(app.query_one("#hero-activity", Static))


@pytest.mark.asyncio
async def test_recent_activity_shows_title_case_status(tmp_path):
    runs = [{"task": "t", "timestamp": "2026-01-01T00:00", "status": "read-only", "duration": "0.1s"}]
    app = _make_app(tmp_path, recent_runs=runs)
    async with app.run_test(size=_SIZE) as _:
        text = _text(app.query_one("#hero-activity", Static))
        assert "Read-only" in text


# ── Input behaviour ────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_input_has_placeholder(tmp_path):
    app = _make_app(tmp_path)
    async with app.run_test(size=_SIZE) as _:
        assert app.query_one("#task-input", Input).placeholder == "Type a task or /help"


@pytest.mark.asyncio
async def test_empty_input_does_not_set_status(tmp_path):
    app = _make_app(tmp_path)
    async with app.run_test(size=_SIZE) as pilot:
        app.query_one("#task-input", Input).focus()
        await pilot.press("enter")
        assert _text(app.query_one("#status-msg", Label)).strip() == ""


@pytest.mark.asyncio
async def test_plain_task_shows_run_command_hint(tmp_path):
    app = _make_app(tmp_path)
    async with app.run_test(size=_SIZE) as pilot:
        inp = app.query_one("#task-input", Input)
        inp.focus()
        inp.value = "fix the bug"
        await pilot.press("enter")
        text = _text(app.query_one("#status-msg", Label))
        assert "Running tasks from the TUI is coming soon" in text
        assert 'openshard run "fix the bug"' in text


@pytest.mark.asyncio
async def test_openshard_command_shows_shell_hint(tmp_path):
    app = _make_app(tmp_path)
    async with app.run_test(size=_SIZE) as pilot:
        inp = app.query_one("#task-input", Input)
        inp.focus()
        inp.value = 'openshard run "test"'
        await pilot.press("enter")
        text = _text(app.query_one("#status-msg", Label))
        assert "Running tasks from the TUI is coming soon" in text
        assert "directly in the shell" in text


@pytest.mark.asyncio
async def test_enter_clears_input(tmp_path):
    app = _make_app(tmp_path)
    async with app.run_test(size=_SIZE) as pilot:
        inp = app.query_one("#task-input", Input)
        inp.focus()
        inp.value = "some task"
        await pilot.press("enter")
        assert inp.value == ""


# ── Forbidden terms ────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_no_forbidden_terms_in_hero(tmp_path):
    app = _make_app(tmp_path)
    async with app.run_test(size=_SIZE) as _:
        owned = {
            "#hero-controls": Static,
            "#hero-activity": Static,
            "#hero-quickstart": Static,
            "#hero-controls-heading": Label,
            "#hero-activity-heading": Label,
        }
        for widget_id, widget_type in owned.items():
            text = _text(app.query_one(widget_id, widget_type)).lower()
            for forbidden in ("routing advisory", "verification loop", "plan ledger"):
                assert forbidden not in text, f"Found '{forbidden}' in {widget_id}"
            assert "candidate" not in text, f"Found 'candidate' in {widget_id}"
