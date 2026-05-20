from __future__ import annotations

import time
from pathlib import Path

from click.testing import CliRunner
from textual import work
from textual.app import App, ComposeResult
from textual.containers import Horizontal, ScrollableContainer, Vertical
from textual.events import Key
from textual.message import Message
from textual.widgets import Label, Static, TextArea

from openshard.tui.commands import TuiCommand, parse_tui_input

BRAND = "✦ OpenShard"
TAGLINE = "The control layer for AI coding agents."

_WORDMARK_WIDE_MIN_WIDTH = 140

# 6-line ANSI Shadow wordmark for wide terminals (≥140 cols)
_WORDMARK_WIDE = (
    "██████╗ ██████╗ ███████╗███╗   ██╗███████╗██╗  ██╗ █████╗ ██████╗ ██████╗\n"
    "██╔═══██╗██╔══██╗██╔════╝████╗  ██║██╔════╝██║  ██║██╔══██╗██╔══██╗██╔══██╗\n"
    "██║   ██║██████╔╝█████╗  ██╔██╗ ██║███████╗███████║███████║██████╔╝██║  ██║\n"
    "██║   ██║██╔═══╝ ██╔══╝  ██║╚██╗██║╚════██║██╔══██║██╔══██║██╔══██╗██║  ██║\n"
    "╚██████╔╝██║     ███████╗██║ ╚████║███████║██║  ██║██║  ██║██║  ██║██████╔╝\n"
    " ╚═════╝ ╚═╝     ╚══════╝╚═╝  ╚═══╝╚══════╝╚═╝  ╚═╝╚═╝  ╚═╝╚═╝  ╚═╝╚═════╝"
)

# 2-line compact wordmark for medium terminals (90–139 cols)
_WORDMARK_NARROW = (
    "█▀█ █▀█ █▀▀ █▄ █ █▀ █ █ ▄▀█ █▀█ █▀▄\n"
    "█▄█ █▀▀ ██▄ █ ▀█ ▄█ █▀█ █▀█ █▀▄ █▄▀"
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

# Two quick-start commands shown inside the hero box
_HERO_QUICKSTART = [
    "explain this repo",
    "/last more",
]

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
    return "\n".join([
        f"Workflow pack: {p.title}",
        "",
        "Category:",
        p.category,
        "",
        "Summary:",
        p.summary,
        "",
        "Prompt:",
        p.prompt,
        "",
        "Next step:",
        "  The prompt above has been loaded into the composer.",
        "  Edit it if needed, then press Enter to run.",
        "  Or run from the shell: openshard run \"<paste prompt>\"",
    ])


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

    BORDER_TITLE = "Type a task or /help"

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
        with Horizontal(id="hero"):
            with Vertical(id="hero-left"):
                yield Static("", id="wordmark")
                yield Static(BRAND, id="hero-brand")
                yield Static(TAGLINE, id="hero-tagline")
                yield Static("", id="hero-project")
            with Vertical(id="hero-right"):
                yield Label("Run Controls", id="hero-controls-heading")
                yield Static("", id="hero-controls")
                yield Label("Quick start", id="hero-quickstart-heading")
                yield Static("", id="hero-quickstart")
                yield Label("Recent activity", id="hero-activity-heading")
                yield Static("", id="hero-activity")
        yield Static("", id="prompt-line")
        yield Static("", id="below-commands")
        yield ScrollableContainer(Static("", id="output-content"), id="output-panel")
        yield Static("", id="run-status")
        yield TaskInput(id="task-input")
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

    def on_resize(self, event) -> None:
        self._update_for_width(event.size.width)

    def _update_for_width(self, width: int) -> None:
        """Swap wordmark, brand visibility, column widths, and hero height by terminal width."""
        is_wide = width >= _WORDMARK_WIDE_MIN_WIDTH
        try:
            self.query_one("#wordmark", Static).update(
                _WORDMARK_WIDE if is_wide else (_WORDMARK_NARROW if width >= 90 else "")
            )
            self.query_one("#hero-brand", Static).update("" if is_wide else BRAND)
            hero = self.query_one("#hero", Horizontal)
            hero.styles.height = 18 if is_wide else 15
            hero_left = self.query_one("#hero-left", Vertical)
            hero_right = self.query_one("#hero-right", Vertical)
            if is_wide:
                hero_left.styles.width = "3fr"
                hero_right.styles.width = "2fr"
            else:
                hero_left.styles.width = "2fr"
                hero_right.styles.width = "3fr"
        except Exception:
            pass

    def _refresh_widgets(self) -> None:
        self._update_for_width(self.size.width)

        gi = self._git_info
        state_label = _GIT_STATE_LABELS.get(gi.get("state", "unknown"), "Unknown")
        project_text = (
            f"[dim]Project[/dim]\n"
            f"  {gi.get('project_name', 'unknown')}\n"
            f"[dim]Branch[/dim]\n"
            f"  {gi.get('branch', 'unknown')}\n"
            f"[dim]Repo[/dim]\n"
            f"  {state_label}"
        )
        self.query_one("#hero-project", Static).update(project_text)

        items = list(self._guardrails.items())
        pairs = [(items[i], items[i + 3]) for i in range(3)]
        ctrl_lines = [
            f"[dim]{k1:<8}[/dim] {v1:<16}  [dim]{k2:<9}[/dim] {v2}"
            for (k1, v1), (k2, v2) in pairs
        ]
        self.query_one("#hero-controls", Static).update("\n".join(ctrl_lines))

        qs_text = "\n".join(f"[dim]> {cmd}[/dim]" for cmd in _HERO_QUICKSTART)
        self.query_one("#hero-quickstart", Static).update(qs_text)

        if self._recent_runs:
            run = self._recent_runs[0]
            label = _STATUS_LABELS.get(run["status"], run["status"])
            color = _STATUS_COLORS.get(run["status"], "dim")
            activity = (
                f"[dim]{run['task']}[/dim]"
                f" · [{color}]{label}[/{color}]"
                f" · [dim]{run['duration']}[/dim]"
            )
        else:
            activity = "[dim]No recent activity[/dim]"
        self.query_one("#hero-activity", Static).update(activity)

        self.query_one("#prompt-line", Static).update("")

        self.query_one("#below-commands", Static).update(
            "[dim]OpenShard ready · /help · /packs · /last more[/dim]"
        )

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
        self.query_one("#run-status", Static).update(
            f"[dim]  ● Executing: {short}  (0s)[/dim]"
        )
        self.set_timer(1.0, self._tick_run_status)

    def _tick_run_status(self) -> None:
        if not self._run_in_progress:
            return
        elapsed = int(time.monotonic() - self._run_start)
        self.query_one("#run-status", Static).update(
            f"[dim]  ● Executing  {elapsed}s  — /last more for full output[/dim]"
        )
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
        self.query_one("#run-status", Static).update("")

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
