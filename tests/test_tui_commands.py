from __future__ import annotations

from openshard.tui.commands import TuiCommand, parse_tui_input


# ── Plain natural language ─────────────────────────────────────────────────


def test_plain_text_parses_as_run_task():
    result = parse_tui_input("explain this repo")
    assert result.cmd == TuiCommand.RUN_TASK
    assert result.task == "explain this repo"


def test_plain_text_with_leading_whitespace_parses_as_run_task():
    result = parse_tui_input("  fix the bug  ")
    assert result.cmd == TuiCommand.RUN_TASK
    assert result.task == "fix the bug"


# ── openshard run variations ───────────────────────────────────────────────


def test_openshard_run_quoted_parses_as_run_task():
    result = parse_tui_input('openshard run "explain this repo"')
    assert result.cmd == TuiCommand.RUN_TASK
    assert result.task == "explain this repo"


def test_openshard_run_unquoted_multi_word_parses_as_run_task():
    result = parse_tui_input("openshard run explain this repo")
    assert result.cmd == TuiCommand.RUN_TASK
    assert result.task == "explain this repo"


def test_openshard_run_single_word_task():
    result = parse_tui_input("openshard run refactor")
    assert result.cmd == TuiCommand.RUN_TASK
    assert result.task == "refactor"


def test_openshard_run_no_task_is_unknown():
    result = parse_tui_input("openshard run")
    assert result.cmd == TuiCommand.UNKNOWN


# ── Unsupported openshard subcommands ─────────────────────────────────────


def test_openshard_diff_last_is_unknown():
    result = parse_tui_input("openshard diff-last")
    assert result.cmd == TuiCommand.UNKNOWN


def test_openshard_apply_last_is_unknown():
    result = parse_tui_input("openshard apply-last")
    assert result.cmd == TuiCommand.UNKNOWN


def test_openshard_last_is_unknown():
    result = parse_tui_input("openshard last")
    assert result.cmd == TuiCommand.UNKNOWN


def test_openshard_arbitrary_command_is_unknown():
    result = parse_tui_input("openshard candidates-last")
    assert result.cmd == TuiCommand.UNKNOWN


# ── Slash commands ─────────────────────────────────────────────────────────


def test_slash_help_parses_as_help():
    assert parse_tui_input("/help").cmd == TuiCommand.HELP


def test_slash_last_parses_as_last():
    result = parse_tui_input("/last")
    assert result.cmd == TuiCommand.LAST


def test_slash_last_more_parses_as_last_more():
    result = parse_tui_input("/last more")
    assert result.cmd == TuiCommand.LAST_MORE


def test_slash_clear_parses_as_clear():
    assert parse_tui_input("/clear").cmd == TuiCommand.CLEAR


def test_slash_quit_parses_as_quit():
    assert parse_tui_input("/quit").cmd == TuiCommand.QUIT


def test_unknown_slash_command_is_unknown():
    assert parse_tui_input("/foo").cmd == TuiCommand.UNKNOWN


def test_unknown_slash_command_does_not_return_run_task():
    result = parse_tui_input("/shell rm -rf /")
    assert result.cmd == TuiCommand.UNKNOWN
    assert result.task is None
