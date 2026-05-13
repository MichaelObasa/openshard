from __future__ import annotations

import os
import sys

from rich.console import Console


def make_console() -> Console:
    """Return a console that stays readable in tests and redirected output."""
    no_color = bool(os.environ.get("NO_COLOR")) or os.environ.get("TERM") == "dumb"
    return Console(
        file=sys.stdout,
        force_terminal=False,
        color_system=None if no_color else "auto",
        highlight=False,
        soft_wrap=True,
    )

