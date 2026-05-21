from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from textual.containers import ScrollableContainer
from textual.widgets import Label, Static

from openshard.tui.app import OpenShardTui, TaskInput, _extract_receipt_block
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


# ── Compact header ─────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_header_brand_contains_openshard(tmp_path):
    app = _make_app(tmp_path)
    async with app.run_test(size=_SIZE) as _:
        assert "OpenShard" in _text(app.query_one("#header-brand", Static))


@pytest.mark.asyncio
async def test_header_tagline_contains_expected_text(tmp_path):
    app = _make_app(tmp_path)
    async with app.run_test(size=_SIZE) as _:
        assert "layer for AI coding agents" in _text(app.query_one("#header-tagline", Static))


@pytest.mark.asyncio
async def test_header_project_shows_project_name(tmp_path):
    app = _make_app(tmp_path)
    async with app.run_test(size=_SIZE) as _:
        assert "test-project" in _text(app.query_one("#header-project", Static))


@pytest.mark.asyncio
async def test_header_project_shows_branch(tmp_path):
    app = _make_app(tmp_path)
    async with app.run_test(size=_SIZE) as _:
        assert "main" in _text(app.query_one("#header-project", Static))


@pytest.mark.asyncio
async def test_header_project_shows_clean_state(tmp_path):
    app = _make_app(tmp_path)
    async with app.run_test(size=_SIZE) as _:
        assert "Clean" in _text(app.query_one("#header-project", Static))


@pytest.mark.asyncio
async def test_dirty_state_renders_as_changes_pending(tmp_path):
    app = _make_app(tmp_path, git_info={"project_name": "p", "branch": "main", "state": "dirty"})
    async with app.run_test(size=_SIZE) as _:
        text = _text(app.query_one("#header-project", Static))
        assert "Changes pending" in text
        assert "dirty" not in text


# ── Status strip ───────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_status_strip_shows_expected_content(tmp_path):
    app = _make_app(tmp_path)
    async with app.run_test(size=_SIZE) as _:
        text = _text(app.query_one("#status-strip", Static))
        for term in ("Sandbox On", "Approval Smart", "Checks Auto", "Receipts On"):
            assert term in text


@pytest.mark.asyncio
async def test_no_forbidden_terms_in_status_strip(tmp_path):
    app = _make_app(tmp_path)
    async with app.run_test(size=_SIZE) as _:
        text = _text(app.query_one("#status-strip", Static)).lower()
        for forbidden in ("routing advisory", "verification loop", "plan ledger"):
            assert forbidden not in text, f"Found '{forbidden}' in #status-strip"
        assert "candidate" not in text, "Found 'candidate' in #status-strip"


# ── Mode strip ─────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_mode_strip_shows_mode_and_executor(tmp_path):
    app = _make_app(tmp_path)
    async with app.run_test(size=_SIZE) as _:
        left = _text(app.query_one("#mode-strip-left", Static))
        right = _text(app.query_one("#mode-strip-right", Static))
        assert "Auto mode" in left
        assert "OpenShard Native" in right


# ── Slash command menu ─────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_slash_menu_hidden_when_composer_empty(tmp_path):
    app = _make_app(tmp_path)
    async with app.run_test(size=_SIZE) as _:
        slash_menu = app.query_one("#slash-menu", Static)
        assert slash_menu.display is False


@pytest.mark.asyncio
async def test_slash_menu_shows_when_text_starts_with_slash(tmp_path):
    app = _make_app(tmp_path)
    async with app.run_test(size=_SIZE) as pilot:
        ta = app.query_one("#task-input", TaskInput)
        ta.load_text("/")
        await pilot.pause()
        slash_menu = app.query_one("#slash-menu", Static)
        assert slash_menu.display is True


@pytest.mark.asyncio
async def test_slash_menu_hides_after_submit(tmp_path):
    app = _make_app(tmp_path)
    async with app.run_test(size=_SIZE) as pilot:
        ta = app.query_one("#task-input", TaskInput)
        ta.focus()
        ta.load_text("/help")
        await pilot.press("enter")
        await pilot.pause()
        slash_menu = app.query_one("#slash-menu", Static)
        assert slash_menu.display is False


# ── Task composer ──────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_task_input_is_multiline_composer(tmp_path):
    app = _make_app(tmp_path)
    async with app.run_test(size=_SIZE) as _:
        ta = app.query_one("#task-input", TaskInput)
        assert ta is not None
        assert ta.BORDER_TITLE == "Type a task or / for commands"


@pytest.mark.asyncio
async def test_empty_input_does_not_set_status(tmp_path):
    app = _make_app(tmp_path)
    async with app.run_test(size=_SIZE) as pilot:
        app.query_one("#task-input", TaskInput).focus()
        await pilot.press("enter")
        assert _text(app.query_one("#status-msg", Label)).strip() == ""


@pytest.mark.asyncio
async def test_enter_clears_input(tmp_path):
    app = _make_app(tmp_path)
    async with app.run_test(size=_SIZE) as pilot:
        ta = app.query_one("#task-input", TaskInput)
        ta.focus()
        ta.load_text("some task")
        await pilot.press("enter")
        assert ta.text == ""


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
        ta = app.query_one("#task-input", TaskInput)
        ta.focus()
        ta.load_text("/help")
        await pilot.press("enter")
        text = _text(app.query_one("#output-content", Static))
        assert "Supported commands" in text
        assert "/last" in text
        assert "/quit" in text


@pytest.mark.asyncio
async def test_unknown_slash_command_shows_error(tmp_path):
    app = _make_app(tmp_path)
    async with app.run_test(size=_SIZE) as pilot:
        ta = app.query_one("#task-input", TaskInput)
        ta.focus()
        ta.load_text("/badcommand")
        await pilot.press("enter")
        text = _text(app.query_one("#output-content", Static))
        assert "Unknown command" in text


@pytest.mark.asyncio
async def test_clear_command_empties_panel(tmp_path):
    app = _make_app(tmp_path)
    async with app.run_test(size=_SIZE) as pilot:
        ta = app.query_one("#task-input", TaskInput)
        ta.focus()
        # First populate the panel via /help
        ta.load_text("/help")
        await pilot.press("enter")
        # Then clear it
        ta.load_text("/clear")
        await pilot.press("enter")
        text = _text(app.query_one("#output-content", Static))
        assert text.strip() == ""


@pytest.mark.asyncio
async def test_quit_command_exits_app(tmp_path):
    app = _make_app(tmp_path)
    with patch.object(app, "exit") as mock_exit:
        async with app.run_test(size=_SIZE) as pilot:
            ta = app.query_one("#task-input", TaskInput)
            ta.focus()
            ta.load_text("/quit")
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
            ta = app.query_one("#task-input", TaskInput)
            ta.focus()
            ta.load_text("explain this repo")
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
            ta = app.query_one("#task-input", TaskInput)
            ta.focus()
            ta.load_text("explain this repo")
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
            ta = app.query_one("#task-input", TaskInput)
            ta.focus()
            ta.load_text("break something")
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
                    ta = app.query_one("#task-input", TaskInput)
                    ta.focus()
                    ta.load_text("explain this repo")
                    await pilot.press("enter")
                    await pilot.pause(delay=0.3)
                mock_load.assert_called()


# ── Workflow packs commands ────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_packs_command_shows_workflow_packs_heading(tmp_path):
    app = _make_app(tmp_path)
    async with app.run_test(size=_SIZE) as pilot:
        ta = app.query_one("#task-input", TaskInput)
        ta.focus()
        ta.load_text("/packs")
        await pilot.press("enter")
        text = _text(app.query_one("#output-content", Static))
        assert "Workflow packs" in text


@pytest.mark.asyncio
async def test_packs_command_shows_known_pack_ids(tmp_path):
    app = _make_app(tmp_path)
    async with app.run_test(size=_SIZE) as pilot:
        ta = app.query_one("#task-input", TaskInput)
        ta.focus()
        ta.load_text("/packs")
        await pilot.press("enter")
        text = _text(app.query_one("#output-content", Static))
        assert "production-iac-hardening" in text
        assert "repo-explanation" in text


@pytest.mark.asyncio
async def test_packs_command_shows_usage_hint(tmp_path):
    app = _make_app(tmp_path)
    async with app.run_test(size=_SIZE) as pilot:
        ta = app.query_one("#task-input", TaskInput)
        ta.focus()
        ta.load_text("/packs")
        await pilot.press("enter")
        text = _text(app.query_one("#output-content", Static))
        assert "/pack <pack-id>" in text


@pytest.mark.asyncio
async def test_pack_show_renders_title(tmp_path):
    app = _make_app(tmp_path)
    async with app.run_test(size=_SIZE) as pilot:
        ta = app.query_one("#task-input", TaskInput)
        ta.focus()
        ta.load_text("/pack production-iac-hardening")
        await pilot.press("enter")
        text = _text(app.query_one("#output-content", Static))
        assert "Production IaC hardening review" in text


@pytest.mark.asyncio
async def test_pack_show_renders_summary(tmp_path):
    app = _make_app(tmp_path)
    async with app.run_test(size=_SIZE) as pilot:
        ta = app.query_one("#task-input", TaskInput)
        ta.focus()
        ta.load_text("/pack production-iac-hardening")
        await pilot.press("enter")
        text = _text(app.query_one("#output-content", Static))
        assert "Hardens" in text


@pytest.mark.asyncio
async def test_pack_show_renders_compact_cta(tmp_path):
    app = _make_app(tmp_path)
    async with app.run_test(size=_SIZE) as pilot:
        ta = app.query_one("#task-input", TaskInput)
        ta.focus()
        ta.load_text("/pack production-iac-hardening")
        await pilot.press("enter")
        text = _text(app.query_one("#output-content", Static))
        assert "Loaded into composer" in text


@pytest.mark.asyncio
async def test_pack_show_renders_prompt_text_in_panel(tmp_path):
    app = _make_app(tmp_path)
    async with app.run_test(size=_SIZE) as pilot:
        ta = app.query_one("#task-input", TaskInput)
        ta.focus()
        ta.load_text("/pack production-iac-hardening")
        await pilot.press("enter")
        text = _text(app.query_one("#output-content", Static))
        assert "Terraform" in text


@pytest.mark.asyncio
async def test_pack_show_unknown_id_shows_error(tmp_path):
    app = _make_app(tmp_path)
    async with app.run_test(size=_SIZE) as pilot:
        ta = app.query_one("#task-input", TaskInput)
        ta.focus()
        ta.load_text("/pack unknown-xyz")
        await pilot.press("enter")
        text = _text(app.query_one("#output-content", Static))
        assert "Unknown pack" in text
        assert "repo-explanation" in text


@pytest.mark.asyncio
async def test_pack_no_id_shows_usage(tmp_path):
    app = _make_app(tmp_path)
    async with app.run_test(size=_SIZE) as pilot:
        ta = app.query_one("#task-input", TaskInput)
        ta.focus()
        ta.load_text("/pack")
        await pilot.press("enter")
        text = _text(app.query_one("#output-content", Static))
        assert "Usage: /pack" in text


@pytest.mark.asyncio
async def test_packs_command_does_not_invoke_cli_runner(tmp_path):
    app = _make_app(tmp_path)
    with patch("openshard.tui.app.CliRunner") as mock_runner_cls:
        async with app.run_test(size=_SIZE) as pilot:
            ta = app.query_one("#task-input", TaskInput)
            ta.focus()
            ta.load_text("/packs")
            await pilot.press("enter")
        mock_runner_cls.return_value.invoke.assert_not_called()


@pytest.mark.asyncio
async def test_pack_show_does_not_invoke_cli_runner(tmp_path):
    app = _make_app(tmp_path)
    with patch("openshard.tui.app.CliRunner") as mock_runner_cls:
        async with app.run_test(size=_SIZE) as pilot:
            ta = app.query_one("#task-input", TaskInput)
            ta.focus()
            ta.load_text("/pack production-iac-hardening")
            await pilot.press("enter")
        mock_runner_cls.return_value.invoke.assert_not_called()


@pytest.mark.asyncio
async def test_packs_output_no_forbidden_strings(tmp_path):
    app = _make_app(tmp_path)
    async with app.run_test(size=_SIZE) as pilot:
        ta = app.query_one("#task-input", TaskInput)
        ta.focus()
        ta.load_text("/packs")
        await pilot.press("enter")
        text = _text(app.query_one("#output-content", Static))
        for forbidden in ("Tunic Pay", "Mercury", "Volant", "AKIA"):
            assert forbidden not in text


@pytest.mark.asyncio
async def test_pack_show_preloads_prompt_into_composer(tmp_path):
    app = _make_app(tmp_path)
    async with app.run_test(size=_SIZE) as pilot:
        ta = app.query_one("#task-input", TaskInput)
        ta.focus()
        ta.load_text("/pack production-iac-hardening")
        await pilot.press("enter")
        # Pack prompt should be preloaded into the composer
        assert "Terraform" in ta.text


# ── _extract_receipt_block() unit tests ───────────────────────────────────


def test_extract_receipt_separator_format():
    output = (
        "Planning - mapping out implementation approach...\n"
        "Executing - Sonnet 4.6 handling logic...\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        "RECEIPT — shard-20260520-0001\n"
        "1 file changed. Verification passed. Receipt saved.\n"
    )
    result = _extract_receipt_block(output)
    assert result is not None
    assert "RECEIPT" in result
    assert "1 file changed" in result
    # separator line is included
    assert "━" in result
    # noisy lines are excluded
    assert "Planning" not in result


def test_extract_receipt_rich_box_format():
    output = (
        "Planning - mapping out...\n"
        "Executing - step 1...\n"
        "╭─ OpenShard Receipt ────────────────────────────╮\n"
        "│ 2 files changed. Verification passed.          │\n"
        "╰────────────────────────────────────────────────╯\n"
    )
    result = _extract_receipt_block(output)
    assert result is not None
    assert "╭─ OpenShard Receipt" in result
    assert "2 files changed" in result
    assert "Planning" not in result


def test_extract_receipt_neither_returns_none():
    output = (
        "Planning - mapping out implementation approach...\n"
        "Executing - Sonnet 4.6 handling logic...\n"
        "Some other output line.\n"
    )
    assert _extract_receipt_block(output) is None


def test_extract_receipt_separator_preferred_over_rich_box():
    output = (
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        "RECEIPT — shard-20260520-0002\n"
        "Receipt saved.\n"
        "╭─ OpenShard Receipt ──╮\n"
        "│ also here            │\n"
        "╰──────────────────────╯\n"
    )
    result = _extract_receipt_block(output)
    assert result is not None
    assert result.startswith("━")
