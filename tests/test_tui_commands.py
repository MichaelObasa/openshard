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


def test_slash_last_full_parses_as_last_full():
    result = parse_tui_input("/last full")
    assert result.cmd == TuiCommand.LAST_FULL


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


# ── Workflow pack commands ─────────────────────────────────────────────────


def test_slash_packs_parses_as_packs():
    result = parse_tui_input("/packs")
    assert result.cmd == TuiCommand.PACKS
    assert result.pack_id is None


def test_slash_packs_uppercase_parses_as_packs():
    result = parse_tui_input("/PACKS")
    assert result.cmd == TuiCommand.PACKS


def test_slash_packs_with_whitespace_parses_as_packs():
    result = parse_tui_input("  /packs  ")
    assert result.cmd == TuiCommand.PACKS


def test_slash_pack_with_id_parses_as_pack_show():
    result = parse_tui_input("/pack production-iac-hardening")
    assert result.cmd == TuiCommand.PACK_SHOW
    assert result.pack_id == "production-iac-hardening"


def test_slash_pack_uppercase_id_is_lowercased():
    result = parse_tui_input("/PACK Production-IaC-Hardening")
    assert result.cmd == TuiCommand.PACK_SHOW
    assert result.pack_id == "production-iac-hardening"


def test_slash_pack_no_id_parses_as_pack_show_without_id():
    result = parse_tui_input("/pack")
    assert result.cmd == TuiCommand.PACK_SHOW
    assert result.pack_id is None


def test_slash_pack_trailing_spaces_only_has_no_id():
    result = parse_tui_input("/pack   ")
    assert result.cmd == TuiCommand.PACK_SHOW
    assert result.pack_id is None


def test_slash_packs_with_id_parses_as_pack_show():
    result = parse_tui_input("/packs production-iac-hardening")
    assert result.cmd == TuiCommand.PACK_SHOW
    assert result.pack_id == "production-iac-hardening"


def test_slash_packs_with_id_uppercase_is_lowercased():
    result = parse_tui_input("/PACKS Production-IaC-Hardening")
    assert result.cmd == TuiCommand.PACK_SHOW
    assert result.pack_id == "production-iac-hardening"


def test_slash_packs_alone_still_lists_packs():
    result = parse_tui_input("/packs")
    assert result.cmd == TuiCommand.PACKS
    assert result.pack_id is None


# ── Feedback commands ──────────────────────────────────────────────────────


def test_slash_feedback_accepted_parses():
    result = parse_tui_input("/feedback accepted")
    assert result.cmd == TuiCommand.FEEDBACK
    assert result.feedback_outcome == "accepted"
    assert result.feedback_reason is None


def test_slash_feedback_rejected_with_reason():
    result = parse_tui_input("/feedback rejected wording was wrong")
    assert result.cmd == TuiCommand.FEEDBACK
    assert result.feedback_outcome == "rejected"
    assert result.feedback_reason == "wording was wrong"


def test_slash_feedback_partial_no_reason():
    result = parse_tui_input("/feedback partial")
    assert result.cmd == TuiCommand.FEEDBACK
    assert result.feedback_outcome == "partial"
    assert result.feedback_reason is None


def test_slash_feedback_no_outcome_is_unknown():
    result = parse_tui_input("/feedback")
    assert result.cmd == TuiCommand.UNKNOWN


def test_slash_feedback_invalid_outcome_is_unknown():
    result = parse_tui_input("/feedback badvalue")
    assert result.cmd == TuiCommand.UNKNOWN


def test_slash_feedback_abandoned_parses():
    result = parse_tui_input("/feedback abandoned")
    assert result.cmd == TuiCommand.FEEDBACK
    assert result.feedback_outcome == "abandoned"


def test_slash_feedback_retried_parses():
    result = parse_tui_input("/feedback retried")
    assert result.cmd == TuiCommand.FEEDBACK
    assert result.feedback_outcome == "retried"


# ── Ask mode commands ──────────────────────────────────────────────────────────


def test_slash_ask_with_question_parses_as_ask():
    result = parse_tui_input("/ask what models do you support?")
    assert result.cmd == TuiCommand.ASK
    assert result.question == "what models do you support?"


def test_slash_ask_bare_parses_as_ask_with_empty_question():
    result = parse_tui_input("/ask")
    assert result.cmd == TuiCommand.ASK
    assert result.question == ""


def test_slash_ask_case_insensitive():
    result = parse_tui_input("/ASK what models")
    assert result.cmd == TuiCommand.ASK


def test_slash_ask_question_not_in_task():
    result = parse_tui_input("/ask reasoning models")
    assert result.question == "reasoning models"
    assert result.task is None


def test_fast_path_what_models_parses_as_ask():
    result = parse_tui_input("what models do you have")
    assert result.cmd == TuiCommand.ASK


def test_fast_path_which_models_parses_as_ask():
    result = parse_tui_input("which models are available")
    assert result.cmd == TuiCommand.ASK


def test_fast_path_what_commands_parses_as_ask():
    result = parse_tui_input("what commands can I use")
    assert result.cmd == TuiCommand.ASK


def test_fast_path_what_is_openshard_parses_as_ask():
    result = parse_tui_input("what is openshard")
    assert result.cmd == TuiCommand.ASK


def test_fast_path_what_does_openshard_parses_as_ask():
    result = parse_tui_input("what does openshard do")
    assert result.cmd == TuiCommand.ASK


def test_regression_plain_task_still_run_task():
    result = parse_tui_input("fix the authentication bug")
    assert result.cmd == TuiCommand.RUN_TASK
    assert result.task == "fix the authentication bug"


# ── Plan mode commands ────────────────────────────────────────────────────────


def test_slash_plan_with_task_parses_as_plan():
    result = parse_tui_input("/plan refactor auth")
    assert result.cmd == TuiCommand.PLAN
    assert result.task == "refactor auth"
    assert result.question is None


def test_slash_plan_bare_parses_as_plan_with_no_task():
    result = parse_tui_input("/plan")
    assert result.cmd == TuiCommand.PLAN
    assert result.task is None


def test_slash_plan_uppercase_parses_as_plan():
    result = parse_tui_input("/PLAN refactor auth")
    assert result.cmd == TuiCommand.PLAN
    assert result.task == "refactor auth"


def test_slash_plan_trailing_spaces_has_no_task():
    result = parse_tui_input("/plan   ")
    assert result.cmd == TuiCommand.PLAN
    assert result.task is None


def test_slash_plan_multiword_task():
    result = parse_tui_input("/plan add tests for the payment module")
    assert result.cmd == TuiCommand.PLAN
    assert result.task == "add tests for the payment module"


def test_fast_path_plan_prefix_parses_as_plan():
    result = parse_tui_input("plan refactor auth safely")
    assert result.cmd == TuiCommand.PLAN


def test_fast_path_how_should_i_parses_as_plan():
    result = parse_tui_input("how should i refactor the auth module")
    assert result.cmd == TuiCommand.PLAN


def test_fast_path_safest_way_parses_as_plan():
    result = parse_tui_input("what is the safest way to deploy this")
    assert result.cmd == TuiCommand.PLAN


def test_fast_path_how_do_i_approach_parses_as_plan():
    result = parse_tui_input("how do i approach this refactor")
    assert result.cmd == TuiCommand.PLAN


def test_regression_fix_the_auth_bug_is_run_task():
    result = parse_tui_input("fix the auth bug")
    assert result.cmd == TuiCommand.RUN_TASK
    assert result.task == "fix the auth bug"


def test_regression_ask_fast_path_still_wins_over_plan():
    result = parse_tui_input("what models do you have")
    assert result.cmd == TuiCommand.ASK


# ── TUI help and CLI discoverability ──────────────────────────────────────────


def test_help_text_mentions_reflect_last():
    from openshard.tui.app import _HELP_TEXT
    assert "openshard reflect last" in _HELP_TEXT


def test_help_text_mentions_pr_comment():
    from openshard.tui.app import _HELP_TEXT
    assert "openshard pr comment" in _HELP_TEXT


def test_slash_menu_mentions_reflect_last():
    from openshard.tui.app import _SLASH_MENU_TEXT
    assert "openshard reflect last" in _SLASH_MENU_TEXT


def test_slash_menu_mentions_pr_comment():
    from openshard.tui.app import _SLASH_MENU_TEXT
    assert "openshard pr comment" in _SLASH_MENU_TEXT


def test_help_text_no_em_dash():
    from openshard.tui.app import _HELP_TEXT
    assert "—" not in _HELP_TEXT


def test_slash_menu_no_em_dash():
    from openshard.tui.app import _SLASH_MENU_TEXT
    assert "—" not in _SLASH_MENU_TEXT


def test_ask_commands_text_mentions_reflect_last():
    from openshard.tui.ask_mode import _COMMANDS_TEXT
    assert "openshard reflect last" in _COMMANDS_TEXT


def test_ask_commands_text_mentions_pr_comment():
    from openshard.tui.ask_mode import _COMMANDS_TEXT
    assert "openshard pr comment" in _COMMANDS_TEXT


def test_last_text_mentions_reflect_last():
    from openshard.tui.ask_mode import _LAST_TEXT
    assert "openshard reflect last" in _LAST_TEXT


def test_last_text_mentions_pr_comment():
    from openshard.tui.ask_mode import _LAST_TEXT
    assert "openshard pr comment" in _LAST_TEXT


def test_last_text_no_em_dash():
    from openshard.tui.ask_mode import _LAST_TEXT
    assert "—" not in _LAST_TEXT
