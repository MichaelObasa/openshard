from __future__ import annotations

from openshard.tui.action_blocks import (
    render_action_block,
    render_actions_section,
    render_check_actions_section,
)


# ---------------------------------------------------------------------------
# render_action_block
# ---------------------------------------------------------------------------


def test_render_action_block_title_only():
    result = render_action_block("Scanned repo")
    assert result == "  Scanned repo"
    assert "↳" not in result


def test_render_action_block_with_detail():
    result = render_action_block("Loaded workflow pack", "ready")
    assert "  Loaded workflow pack" in result
    assert "↳ [dim]ready[/dim]" in result


def test_render_action_block_empty_detail_omitted():
    result = render_action_block("Something", None)
    assert "↳" not in result


# ---------------------------------------------------------------------------
# render_actions_section
# ---------------------------------------------------------------------------


def test_render_actions_section_empty_list():
    assert render_actions_section([]) == ""


def test_render_actions_section_all_empty_labels():
    events = [{"label": "", "status": "completed"}, {"label": None, "status": "completed"}]
    assert render_actions_section(events) == ""


def test_render_actions_section_completed_events():
    events = [
        {"label": "Loaded workflow pack", "status": "completed"},
        {"label": "Scanned repo", "status": "completed"},
    ]
    result = render_actions_section(events)
    assert "[bold]ACTIONS[/bold]" in result
    assert "  Loaded workflow pack" in result
    assert "  Scanned repo" in result


def test_render_actions_section_skipped_event():
    events = [{"label": "Some step", "status": "skipped"}]
    result = render_actions_section(events)
    assert "↳ [dim]skipped[/dim]" in result


def test_render_actions_section_failed_event():
    events = [{"label": "Failed step", "status": "failed"}]
    result = render_actions_section(events)
    assert "↳ [dim]failed[/dim]" in result


def test_render_actions_section_count_event():
    events = [{"label": "Found raw findings", "status": "completed", "count": 27}]
    result = render_actions_section(events)
    assert "↳ [dim]27 total[/dim]" in result


def test_render_actions_section_explicit_detail():
    events = [{"label": "Custom step", "status": "completed", "detail": "my custom detail"}]
    result = render_actions_section(events)
    assert "↳ [dim]my custom detail[/dim]" in result


def test_render_actions_section_explicit_detail_takes_priority_over_count():
    events = [{"label": "Step", "status": "completed", "detail": "explicit", "count": 5}]
    result = render_actions_section(events)
    assert "explicit" in result
    assert "5 total" not in result


def test_render_actions_section_ends_with_newline():
    events = [{"label": "Scanned repo", "status": "completed"}]
    result = render_actions_section(events)
    assert result.endswith("\n")


# ---------------------------------------------------------------------------
# render_check_actions_section
# ---------------------------------------------------------------------------


def test_render_check_actions_section_empty_list():
    assert render_check_actions_section([]) == ""


def test_render_check_actions_section_all_empty_names():
    checks = [{"name": "", "status": "passed"}, {"name": None, "status": "passed"}]
    assert render_check_actions_section(checks) == ""


def test_render_check_actions_section_passed():
    checks = [{"name": "terraform fmt", "status": "passed"}]
    result = render_check_actions_section(checks)
    assert "[bold]CHECK ACTIONS[/bold]" in result
    assert "[green]✓[/green]" in result
    assert "terraform fmt" in result
    assert "passed" in result


def test_render_check_actions_section_failed():
    checks = [{"name": "terraform validate", "status": "failed"}]
    result = render_check_actions_section(checks)
    assert "[red]✗[/red]" in result
    assert "failed" in result


def test_render_check_actions_section_skipped_with_reason():
    checks = [{"name": "tflint", "status": "skipped", "reason": "tflint not installed"}]
    result = render_check_actions_section(checks)
    assert "[dim]-[/dim]" in result
    assert "skipped — tflint not installed" in result


def test_render_check_actions_section_skipped_no_reason():
    checks = [{"name": "tflint", "status": "skipped"}]
    result = render_check_actions_section(checks)
    assert "skipped" in result
    assert " — " not in result


def test_render_check_actions_section_with_summary():
    checks = [{"name": "unit tests", "status": "passed", "summary": "42 tests passed"}]
    result = render_check_actions_section(checks)
    assert "42 tests passed" in result
    assert "passed" in result


def test_render_check_actions_section_failed_with_summary():
    checks = [{"name": "unit tests", "status": "failed", "summary": "3 tests failed"}]
    result = render_check_actions_section(checks)
    assert "3 tests failed" in result


def test_render_check_actions_section_ends_with_newline():
    checks = [{"name": "terraform fmt", "status": "passed"}]
    result = render_check_actions_section(checks)
    assert result.endswith("\n")


def test_render_check_actions_section_multiple_checks():
    checks = [
        {"name": "terraform fmt", "status": "passed"},
        {"name": "tflint", "status": "skipped", "reason": "tflint not installed"},
        {"name": "unit tests", "status": "failed"},
    ]
    result = render_check_actions_section(checks)
    assert "terraform fmt" in result
    assert "tflint" in result
    assert "unit tests" in result
    assert "[green]✓[/green]" in result
    assert "[dim]-[/dim]" in result
    assert "[red]✗[/red]" in result
