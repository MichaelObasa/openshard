from __future__ import annotations

from dataclasses import dataclass

import click
from rich.console import Console
from rich.table import Table
from rich.text import Text

from .theme import ACCENT_STYLE, ERROR_STYLE, GOOD_STYLE, MUTED_STYLE, VALUE_STYLE


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
