from __future__ import annotations

import time
from importlib.metadata import version as _pkg_version
from pathlib import Path

from click.testing import CliRunner
from textual import on, work
from textual.app import App, ComposeResult
from textual.containers import Horizontal, ScrollableContainer, Vertical
from textual.events import Key
from textual.message import Message
from textual.widgets import Label, Static, TextArea

from openshard.tui.commands import TuiCommand, parse_tui_input

_BRAND_ANSI_SHADOW = (
    "██████╗ ██████╗ ███████╗███╗   ██╗███████╗██╗  ██╗ █████╗ ██████╗ ██████╗\n"
    "██╔═══██╗██╔══██╗██╔════╝████╗  ██║██╔════╝██║  ██║██╔══██╗██╔══██╗██╔══██╗\n"
    "██║   ██║██████╔╝█████╗  ██╔██╗ ██║███████╗███████║███████║██████╔╝██║  ██║\n"
    "██║   ██║██╔═══╝ ██╔══╝  ██║╚██╗██║╚════██║██╔══██║██╔══██║██╔══██╗██║  ██║\n"
    "╚██████╔╝██║     ███████╗██║ ╚████║███████║██║  ██║██║  ██║██║  ██║██████╔╝\n"
    " ╚═════╝ ╚═╝     ╚══════╝╚═╝  ╚═══╝╚══════╝╚═╝  ╚═╝╚═╝  ╚═╝╚═╝  ╚═╝╚═════╝"
)
_BRAND_FALLBACK = "✦ OpenShard"
_BRAND_MIN_WIDTH = 100  # conservative threshold; avoids Windows Terminal wrapping/clipping

try:
    _VERSION = f"v{_pkg_version('openshard')}"
except Exception:
    _VERSION = "v0.1.0"

TAGLINE = "The control layer for AI coding agents"

_MODE_STRIP_DEFAULT = "Auto mode"
_STATUS_STRIP = "↳ Sandbox [ON]     ↳ Receipts [ON]     ↳ Checks [AUTO]     ↳ Approval [SMART]"
_SLASH_MENU_TEXT = (
    "  /help          Show help\n"
    "  /packs         List workflow packs\n"
    "  /pack <id>     Load a workflow pack\n"
    "  /last          Show most recent run\n"
    "  /last more     Show more detail\n"
    "  /clear         Clear output\n"
    "  /quit          Exit"
)

_GIT_STATE_LABELS = {
    "clean": "Clean",
    "dirty": "Changes pending",
    "unknown": "Unknown",
}

_STATUS_LABELS = {
    "passed": "Passed",
    "failed": "Failed",
    "read-only": "Read-only",
    "unknown": "Unknown",
}

_STATUS_COLORS = {
    "passed": "green",
    "failed": "red",
    "read-only": "dim",
    "unknown": "dim",
}

_HELP_TEXT = (
    "Supported commands:\n"
    "  /help           Show this help\n"
    "  /last           Show most recent run\n"
    "  /last more      Show more details of most recent run\n"
    "  /clear          Clear output area\n"
    "  /quit           Exit the TUI\n"
    "  /packs          List available workflow packs\n"
    "  /pack <id>      Show a workflow pack prompt\n"
    "  (plain text)    Run as openshard task\n"
    "\n"
    "Composer keys:\n"
    "  Enter           Submit task\n"
    "  Ctrl+J          Insert newline\n"
    "  Shift+Enter     Insert newline (if terminal supports it)"
)


def _render_packs_list() -> str:
    from openshard.workflow_packs.packs import load_packs

    packs = load_packs()
    col = 32
    lines = ["Workflow packs", ""]
    for p in packs:
        lines.append(f"{p.id:<{col}}{p.title}")
    lines += ["", "Use:", "  /pack <pack-id>"]
    return "\n".join(lines)


def _render_pack_detail(pack_id: str | None) -> str:
    from openshard.workflow_packs.packs import get_pack, load_packs

    if pack_id is None:
        ids = "\n".join(f"  {p.id}" for p in load_packs())
        return f"Usage: /pack <pack-id>\n\nAvailable packs:\n{ids}"
    try:
        p = get_pack(pack_id)
    except KeyError:
        ids = "\n".join(f"  {p.id}" for p in load_packs())
        return f'Unknown pack: "{pack_id}"\n\nAvailable packs:\n{ids}'
    return f"Workflow pack selected: {p.title}"


def _extract_receipt_block(output: str) -> str | None:
    """Return the receipt block from CLI output, or None if not found.

    Tries two formats in order:
    1. Separator receipt: a line where the stripped text starts with "RECEIPT"
       and contains "—" or "-" (e.g. "RECEIPT — shard-20260520-0001").
       Walks back one line to include the preceding separator rule.
    2. Rich box receipt: a line containing "╭─ OpenShard Receipt".
    """
    lines = output.splitlines()

    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("RECEIPT") and ("—" in stripped or "-" in stripped):
            return "\n".join(lines[max(0, i - 1):])

    for i, line in enumerate(lines):
        if "╭─ OpenShard Receipt" in line:
            return "\n".join(lines[i:])

    return None


class TaskInput(TextArea):
    """Multi-line task composer. Enter submits; Ctrl+J / Shift+Enter insert newline."""

    BORDER_TITLE = "Type a task or / for commands"

    class Submit(Message):
        def __init__(self, text: str) -> None:
            self.text = text
            super().__init__()

    def on_key(self, event: Key) -> None:
        if event.key == "enter":
            event.prevent_default()
            self.post_message(self.Submit(self.text))
        elif event.key in ("ctrl+j", "shift+enter"):
            event.prevent_default()
            self.insert("\n")


class OpenShardTui(App):
    CSS_PATH = "styles.tcss"

    def __init__(self, path: Path | None = None) -> None:
        super().__init__()
        self._path = path or Path.cwd()
        self._git_info: dict = {}
        self._recent_runs: list[dict] | None = None
        self._guardrails: dict = {}
        self._output_lines: list[str] = []
        self._run_in_progress: bool = False
        self._run_start: float = 0.0

    def compose(self) -> ComposeResult:
        with Horizontal(id="header-card"):
            with Vertical(id="header-left"):
                yield Static("", id="header-brand")
                yield Static(TAGLINE, id="header-tagline")
                yield Static("", id="header-project")
            yield Static(_VERSION, id="header-version")
        yield ScrollableContainer(Static("", id="output-content"), id="output-panel")
        with Horizontal(id="mode-strip"):
            yield Static(_MODE_STRIP_DEFAULT, id="mode-strip-left")
            yield Static("Agent: OpenShard Native (OSN)", id="mode-strip-right")
        yield Static(_SLASH_MENU_TEXT, id="slash-menu")
        yield TaskInput(id="task-input")
        yield Static(_STATUS_STRIP, id="status-strip", markup=False)
        yield Label("", id="status-msg")

    def on_mount(self) -> None:
        from openshard.tui.state import get_guardrails, load_git_info, load_recent_runs

        if not self._git_info:
            self._git_info = load_git_info(self._path)
        if not self._guardrails:
            self._guardrails = get_guardrails()
        if self._recent_runs is None:
            self._recent_runs = load_recent_runs(self._path)

        self._refresh_widgets()
        self._update_brand()

    def on_resize(self, event) -> None:
        self._update_brand()

    def _update_brand(self) -> None:
        brand = _BRAND_ANSI_SHADOW if self.size.width >= _BRAND_MIN_WIDTH else _BRAND_FALLBACK
        self.query_one("#header-brand", Static).update(brand)

    def _refresh_widgets(self) -> None:
        gi = self._git_info
        state_key = gi.get("state", "unknown")
        state_label = _GIT_STATE_LABELS.get(state_key, "Unknown")
        col = 26
        _repo_color = {"clean": "green", "dirty": "yellow", "unknown": "dim"}
        color = _repo_color.get(state_key, "dim")
        pname = gi.get("project_name", "unknown")
        branch = gi.get("branch", "unknown")
        labels = f"[dim]{'PROJECT':<{col}}{'BRANCH':<{col}}REPO[/dim]"
        values = (
            f"[white]{pname:<{col}}[/white]"
            f"[white]{branch:<{col}}[/white]"
            f"[{color}]{state_label}[/{color}]"
        )
        self.query_one("#header-project", Static).update(f"{labels}\n{values}")

    @on(TextArea.Changed, "#task-input")
    def _on_task_input_changed(self, event: TextArea.Changed) -> None:
        text = event.text_area.text
        self.query_one("#slash-menu").display = text.startswith("/") and " " not in text

    def on_task_input_submit(self, event: TaskInput.Submit) -> None:
        raw = event.text.strip()
        if not raw:
            return
        self.query_one("#task-input", TaskInput).load_text("")
        parsed = parse_tui_input(raw)

        if parsed.cmd == TuiCommand.HELP:
            self._append_output(_HELP_TEXT)
        elif parsed.cmd == TuiCommand.CLEAR:
            self._clear_output()
        elif parsed.cmd == TuiCommand.QUIT:
            self.exit()
        elif parsed.cmd == TuiCommand.LAST:
            self._run_cli_async(["last"])
        elif parsed.cmd == TuiCommand.LAST_MORE:
            self._run_cli_async(["last", "--more"])
        elif parsed.cmd == TuiCommand.RUN_TASK:
            self._append_output(f"> {raw}")
            self._start_run_status(raw)
            self._run_cli_async(["run", parsed.task], refresh_after=True, is_run=True)
        elif parsed.cmd == TuiCommand.PACKS:
            self._append_output(_render_packs_list())
        elif parsed.cmd == TuiCommand.PACK_SHOW:
            self._append_output(_render_pack_detail(parsed.pack_id))
            if parsed.pack_id:
                try:
                    from openshard.workflow_packs.packs import get_pack
                    p = get_pack(parsed.pack_id)
                    self.query_one("#task-input", TaskInput).load_text(p.prompt)
                except KeyError:
                    pass
        else:
            self._append_output(f"Unknown command: {raw}\nType /help for supported commands.")

    def _start_run_status(self, task_text: str) -> None:
        self._run_in_progress = True
        self._run_start = time.monotonic()
        short = task_text[:60].rstrip()
        if len(task_text) > 60:
            short += "…"
        self.query_one("#mode-strip-left", Static).update(
            f"[dim]  ● Executing: {short}  (0s)[/dim]"
        )
        self.set_timer(1.0, self._tick_run_status)

    def _tick_run_status(self) -> None:
        if not self._run_in_progress:
            return
        elapsed = int(time.monotonic() - self._run_start)
        try:
            self.query_one("#mode-strip-left", Static).update(
                f"[dim]  ● Executing  {elapsed}s  — /last more for full output[/dim]"
            )
        except Exception:
            return
        self.set_timer(1.0, self._tick_run_status)

    @work(thread=True)
    def _run_cli_async(
        self, args: list[str], refresh_after: bool = False, is_run: bool = False
    ) -> None:
        # Temporary v0 bridge: routes TUI input through existing Click commands via
        # CliRunner so we avoid shell=True and stay within supported code paths.
        from openshard.cli.main import cli as openshard_cli

        runner = CliRunner()
        result = runner.invoke(openshard_cli, args, input="", catch_exceptions=True)
        output = result.output or (str(result.exception) if result.exception else "")
        status = "Done." if result.exit_code == 0 else f"Failed (exit {result.exit_code})."
        self.call_from_thread(self._on_cli_result, output, status, refresh_after, is_run)

    def _on_cli_result(
        self, output: str, status: str, refresh_after: bool, is_run: bool = False
    ) -> None:
        self._run_in_progress = False
        self.query_one("#mode-strip-left", Static).update(_MODE_STRIP_DEFAULT)

        if is_run:
            receipt_block = _extract_receipt_block(output)
            if receipt_block is not None:
                display = receipt_block.rstrip("\n") + "\n" + status
            else:
                tail = "\n".join(output.splitlines()[-30:])
                display = tail.rstrip("\n") + "\n" + status
        else:
            display = output.rstrip("\n") + "\n" + status

        self._append_output(display)

        if refresh_after:
            from openshard.tui.state import load_recent_runs

            self._recent_runs = load_recent_runs(self._path)
            self._refresh_widgets()

    def _append_output(self, text: str) -> None:
        self._output_lines.append(text)
        self.query_one("#output-content", Static).update("\n".join(self._output_lines))
        self.query_one("#output-panel", ScrollableContainer).scroll_end(animate=False)

    def _clear_output(self) -> None:
        self._output_lines.clear()
        self.query_one("#output-content", Static).update("")
