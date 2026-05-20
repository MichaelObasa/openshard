from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from textual.containers import ScrollableContainer
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


# ── Output panel ───────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_output_panel_is_present(tmp_path):
    app = _make_app(tmp_path)
    async with app.run_test(size=_SIZE) as _:
        panel = app.query_one("#output-panel", ScrollableContainer)
        assert panel is not None


@pytest.mark.asyncio
async def test_help_command_shows_help_in_panel(tmp_path):
    app = _make_app(tmp_path)
    async with app.run_test(size=_SIZE) as pilot:
        inp = app.query_one("#task-input", Input)
        inp.focus()
        inp.value = "/help"
        await pilot.press("enter")
        text = _text(app.query_one("#output-content", Static))
        assert "Supported commands" in text
        assert "/last" in text
        assert "/quit" in text


@pytest.mark.asyncio
async def test_unknown_slash_command_shows_error(tmp_path):
    app = _make_app(tmp_path)
    async with app.run_test(size=_SIZE) as pilot:
        inp = app.query_one("#task-input", Input)
        inp.focus()
        inp.value = "/badcommand"
        await pilot.press("enter")
        text = _text(app.query_one("#output-content", Static))
        assert "Unknown command" in text


@pytest.mark.asyncio
async def test_clear_command_empties_panel(tmp_path):
    app = _make_app(tmp_path)
    async with app.run_test(size=_SIZE) as pilot:
        inp = app.query_one("#task-input", Input)
        inp.focus()
        # First populate the panel via /help
        inp.value = "/help"
        await pilot.press("enter")
        # Then clear it
        inp.value = "/clear"
        await pilot.press("enter")
        text = _text(app.query_one("#output-content", Static))
        assert text.strip() == ""


@pytest.mark.asyncio
async def test_quit_command_exits_app(tmp_path):
    app = _make_app(tmp_path)
    with patch.object(app, "exit") as mock_exit:
        async with app.run_test(size=_SIZE) as pilot:
            inp = app.query_one("#task-input", Input)
            inp.focus()
            inp.value = "/quit"
            await pilot.press("enter")
        mock_exit.assert_called_once()


@pytest.mark.asyncio
async def test_plain_task_invokes_cli_runner(tmp_path):
    app = _make_app(tmp_path)
    mock_result = MagicMock()
    mock_result.output = "some output"
    mock_result.exit_code = 0
    mock_result.exception = None

    with patch("openshard.tui.app.CliRunner") as mock_runner_cls:
        mock_runner_cls.return_value.invoke.return_value = mock_result
        async with app.run_test(size=_SIZE) as pilot:
            inp = app.query_one("#task-input", Input)
            inp.focus()
            inp.value = "explain this repo"
            await pilot.press("enter")
            await pilot.pause(delay=0.3)

        calls = mock_runner_cls.return_value.invoke.call_args_list
        assert len(calls) == 1
        # CliRunner.invoke(cli, ["run", task], input="", catch_exceptions=True)
        assert "run" in str(calls[0])
        assert "explain this repo" in str(calls[0])


@pytest.mark.asyncio
async def test_run_output_appears_in_panel(tmp_path):
    app = _make_app(tmp_path)
    mock_result = MagicMock()
    mock_result.output = "Result: all good\n"
    mock_result.exit_code = 0
    mock_result.exception = None

    with patch("openshard.tui.app.CliRunner") as mock_runner_cls:
        mock_runner_cls.return_value.invoke.return_value = mock_result
        async with app.run_test(size=_SIZE) as pilot:
            inp = app.query_one("#task-input", Input)
            inp.focus()
            inp.value = "explain this repo"
            await pilot.press("enter")
            await pilot.pause(delay=0.3)
            text = _text(app.query_one("#output-content", Static))
        assert "Result: all good" in text
        assert "Done." in text


@pytest.mark.asyncio
async def test_failed_run_shows_failure_status(tmp_path):
    app = _make_app(tmp_path)
    mock_result = MagicMock()
    mock_result.output = "Error occurred\n"
    mock_result.exit_code = 1
    mock_result.exception = None

    with patch("openshard.tui.app.CliRunner") as mock_runner_cls:
        mock_runner_cls.return_value.invoke.return_value = mock_result
        async with app.run_test(size=_SIZE) as pilot:
            inp = app.query_one("#task-input", Input)
            inp.focus()
            inp.value = "break something"
            await pilot.press("enter")
            await pilot.pause(delay=0.3)
            text = _text(app.query_one("#output-content", Static))
        assert "Failed" in text


@pytest.mark.asyncio
async def test_recent_activity_refreshes_after_run(tmp_path):
    app = _make_app(tmp_path)
    mock_result = MagicMock()
    mock_result.output = "done\n"
    mock_result.exit_code = 0
    mock_result.exception = None

    new_runs = [{"task": "refreshed task", "timestamp": "2026-05-18T00:00", "status": "passed", "duration": "2.0s"}]

    with patch("openshard.tui.app.CliRunner") as mock_runner_cls:
        mock_runner_cls.return_value.invoke.return_value = mock_result
        with patch("openshard.tui.app.OpenShardTui._on_cli_result", wraps=app._on_cli_result):
            with patch("openshard.tui.state.load_recent_runs", return_value=new_runs) as mock_load:
                async with app.run_test(size=_SIZE) as pilot:
                    inp = app.query_one("#task-input", Input)
                    inp.focus()
                    inp.value = "explain this repo"
                    await pilot.press("enter")
                    await pilot.pause(delay=0.3)
                mock_load.assert_called()


# ── Workflow packs commands ────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_packs_command_shows_workflow_packs_heading(tmp_path):
    app = _make_app(tmp_path)
    async with app.run_test(size=_SIZE) as pilot:
        inp = app.query_one("#task-input", Input)
        inp.focus()
        inp.value = "/packs"
        await pilot.press("enter")
        text = _text(app.query_one("#output-content", Static))
        assert "Workflow packs" in text


@pytest.mark.asyncio
async def test_packs_command_shows_known_pack_ids(tmp_path):
    app = _make_app(tmp_path)
    async with app.run_test(size=_SIZE) as pilot:
        inp = app.query_one("#task-input", Input)
        inp.focus()
        inp.value = "/packs"
        await pilot.press("enter")
        text = _text(app.query_one("#output-content", Static))
        assert "production-iac-hardening" in text
        assert "repo-explanation" in text


@pytest.mark.asyncio
async def test_packs_command_shows_usage_hint(tmp_path):
    app = _make_app(tmp_path)
    async with app.run_test(size=_SIZE) as pilot:
        inp = app.query_one("#task-input", Input)
        inp.focus()
        inp.value = "/packs"
        await pilot.press("enter")
        text = _text(app.query_one("#output-content", Static))
        assert "/pack <pack-id>" in text


@pytest.mark.asyncio
async def test_pack_show_renders_title(tmp_path):
    app = _make_app(tmp_path)
    async with app.run_test(size=_SIZE) as pilot:
        inp = app.query_one("#task-input", Input)
        inp.focus()
        inp.value = "/pack production-iac-hardening"
        await pilot.press("enter")
        text = _text(app.query_one("#output-content", Static))
        assert "Production IaC hardening review" in text


@pytest.mark.asyncio
async def test_pack_show_renders_category(tmp_path):
    app = _make_app(tmp_path)
    async with app.run_test(size=_SIZE) as pilot:
        inp = app.query_one("#task-input", Input)
        inp.focus()
        inp.value = "/pack production-iac-hardening"
        await pilot.press("enter")
        text = _text(app.query_one("#output-content", Static))
        assert "infrastructure" in text


@pytest.mark.asyncio
async def test_pack_show_renders_prompt(tmp_path):
    app = _make_app(tmp_path)
    async with app.run_test(size=_SIZE) as pilot:
        inp = app.query_one("#task-input", Input)
        inp.focus()
        inp.value = "/pack production-iac-hardening"
        await pilot.press("enter")
        text = _text(app.query_one("#output-content", Static))
        assert "Terraform" in text


@pytest.mark.asyncio
async def test_pack_show_unknown_id_shows_error(tmp_path):
    app = _make_app(tmp_path)
    async with app.run_test(size=_SIZE) as pilot:
        inp = app.query_one("#task-input", Input)
        inp.focus()
        inp.value = "/pack unknown-xyz"
        await pilot.press("enter")
        text = _text(app.query_one("#output-content", Static))
        assert "Unknown pack" in text
        assert "repo-explanation" in text


@pytest.mark.asyncio
async def test_pack_no_id_shows_usage(tmp_path):
    app = _make_app(tmp_path)
    async with app.run_test(size=_SIZE) as pilot:
        inp = app.query_one("#task-input", Input)
        inp.focus()
        inp.value = "/pack"
        await pilot.press("enter")
        text = _text(app.query_one("#output-content", Static))
        assert "Usage: /pack" in text


@pytest.mark.asyncio
async def test_packs_command_does_not_invoke_cli_runner(tmp_path):
    app = _make_app(tmp_path)
    with patch("openshard.tui.app.CliRunner") as mock_runner_cls:
        async with app.run_test(size=_SIZE) as pilot:
            inp = app.query_one("#task-input", Input)
            inp.focus()
            inp.value = "/packs"
            await pilot.press("enter")
        mock_runner_cls.return_value.invoke.assert_not_called()


@pytest.mark.asyncio
async def test_pack_show_does_not_invoke_cli_runner(tmp_path):
    app = _make_app(tmp_path)
    with patch("openshard.tui.app.CliRunner") as mock_runner_cls:
        async with app.run_test(size=_SIZE) as pilot:
            inp = app.query_one("#task-input", Input)
            inp.focus()
            inp.value = "/pack production-iac-hardening"
            await pilot.press("enter")
        mock_runner_cls.return_value.invoke.assert_not_called()


@pytest.mark.asyncio
async def test_packs_output_no_forbidden_strings(tmp_path):
    app = _make_app(tmp_path)
    async with app.run_test(size=_SIZE) as pilot:
        inp = app.query_one("#task-input", Input)
        inp.focus()
        inp.value = "/packs"
        await pilot.press("enter")
        text = _text(app.query_one("#output-content", Static))
        for forbidden in ("Tunic Pay", "Mercury", "Volant", "AKIA"):
            assert forbidden not in text
