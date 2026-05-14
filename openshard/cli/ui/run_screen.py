from __future__ import annotations

from dataclasses import dataclass

import click
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from .theme import ACCENT_STYLE, BORDER_STYLE, ERROR_STYLE, GOOD_STYLE, MUTED_STYLE, VALUE_STYLE


@dataclass
class StageDisplay:
    name: str    # "Route", "Plan", "Work", "Ask", "Verify", "Receipt"
    status: str  # "routed" | "passed" | "failed" | "skipped" | "saved" | "ask"
    detail: str  # pre-formatted, ASCII-safe


_STATUS_STYLE: dict[str, str] = {
    "passed":  GOOD_STYLE,
    "failed":  ERROR_STYLE,
    "skipped": MUTED_STYLE,
    "saved":   ACCENT_STYLE,
    "ask":     ACCENT_STYLE,
    "routed":  VALUE_STYLE,
}


def render_run_stages(stages: list[StageDisplay], console: Console) -> None:
    """Render stage table using Rich grid layout."""
    grid = Table.grid(padding=(0, 2))
    grid.add_column(width=9)
    grid.add_column(width=10)
    grid.add_column()
    for s in stages:
        grid.add_row(
            Text(s.name, style=MUTED_STYLE),
            Text(s.status, style=_STATUS_STYLE.get(s.status, VALUE_STYLE)),
            Text(s.detail, style=MUTED_STYLE),
        )
    console.print()
    console.print(grid)


def render_run_stages_plain(stages: list[StageDisplay]) -> None:
    """Plain-text fallback for NO_COLOR or dumb terminals."""
    click.echo("")
    for s in stages:
        click.echo(f"  {s.name:<9}  {s.status:<10}  {s.detail}")


def _receipt_rows(
    stages: list[StageDisplay],
    mode_label: str | None,
    cost_str: str | None,
) -> list[tuple[str, str, str]]:
    """Return (label, value, style_key) tuples for the receipt panel.

    style_key is a key into _STATUS_STYLE or one of the sentinel values
    "value" / "muted" / "accent" for non-status rows.
    """
    stage_map = {s.name: s for s in stages}
    is_readonly = any(s.name == "Ask" for s in stages)
    receipt_stage = stage_map.get("Receipt")

    rows: list[tuple[str, str, str]] = []

    if mode_label:
        rows.append(("Mode", mode_label, "value"))

    route = stage_map.get("Route")
    if route:
        rows.append(("Route", route.detail, "value"))

    # Plan: always omit from panel

    if not is_readonly:
        work = stage_map.get("Work")
        if work:
            files = receipt_stage.detail if receipt_stage and receipt_stage.detail else ""
            value = f"{work.status} -- {files}" if files else work.status
            rows.append(("Work", value, work.status))

    verify = stage_map.get("Verify")
    if verify:
        value = f"{verify.status} -- {verify.detail}" if verify.detail else verify.status
        rows.append(("Verify", value, verify.status))

    if cost_str:
        rows.append(("Cost", cost_str, "value"))

    if is_readonly and receipt_stage and receipt_stage.detail:
        rows.append(("Result", receipt_stage.detail, "muted"))

    rows.append(("Receipt", "saved", "accent"))

    return rows


def render_receipt_panel(
    stages: list[StageDisplay],
    mode_label: str | None,
    cost_str: str | None,
    console: Console,
) -> None:
    """Render the post-run receipt as a Rich Panel with purple border."""
    _STYLE_MAP: dict[str, str] = {
        **_STATUS_STYLE,
        "value":  VALUE_STYLE,
        "muted":  MUTED_STYLE,
        "accent": ACCENT_STYLE,
    }

    grid = Table.grid(padding=(0, 2))
    grid.add_column(width=9)
    grid.add_column()

    for label, value, style_key in _receipt_rows(stages, mode_label, cost_str):
        grid.add_row(
            Text(label, style=MUTED_STYLE),
            Text(value, style=_STYLE_MAP.get(style_key, VALUE_STYLE)),
        )

    panel = Panel(
        grid,
        title="OpenShard Receipt",
        title_align="left",
        border_style=BORDER_STYLE,
        box=box.ROUNDED,
        padding=(0, 1),
    )
    console.print()
    console.print(panel)


def render_receipt_panel_plain(
    stages: list[StageDisplay],
    mode_label: str | None,
    cost_str: str | None,
) -> None:
    """Plain-text fallback for NO_COLOR or dumb terminals."""
    click.echo("")
    click.echo("  OpenShard Receipt")
    click.echo("")
    for label, value, _ in _receipt_rows(stages, mode_label, cost_str):
        click.echo(f"  {label:<9}  {value}")
