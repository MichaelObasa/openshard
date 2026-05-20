from __future__ import annotations

from pathlib import Path

from click.testing import CliRunner
from textual import work
from textual.app import App, ComposeResult
from textual.containers import Horizontal, ScrollableContainer, Vertical
from textual.widgets import Input, Label, Static

from openshard.tui.commands import TuiCommand, parse_tui_input

BRAND = "✦ OpenShard"
TAGLINE = "The control layer for AI coding agents."
PLACEHOLDER = "Type a task or /help"

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

# Full try list shown below the hero box
_BELOW_COMMANDS = [
    "explain this repo",
    "/last",
    "/last more",
    "/help",
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
    "  (plain text)    Run as openshard task"
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
        "How to run:",
        "Copy the prompt above into the input, or run:",
        '  openshard run "<paste prompt here>"',
    ])


class OpenShardTui(App):
    CSS_PATH = "styles.tcss"

    def __init__(self, path: Path | None = None) -> None:
        super().__init__()
        self._path = path or Path.cwd()
        self._git_info: dict = {}
        self._recent_runs: list[dict] | None = None
        self._guardrails: dict = {}
        self._output_lines: list[str] = []

    def compose(self) -> ComposeResult:
        with Horizontal(id="hero"):
            with Vertical(id="hero-left"):
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
        yield Input(placeholder=PLACEHOLDER, id="task-input")
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

    def _refresh_widgets(self) -> None:
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

        self.query_one("#prompt-line", Static).update(
            "[dim]> OpenShard ready. What would you like to build?[/dim]"
        )

        below_text = "[dim]Try these inside the TUI:[/dim]\n" + "\n".join(
            f"[dim]> {cmd}[/dim]" for cmd in _BELOW_COMMANDS
        )
        self.query_one("#below-commands", Static).update(below_text)

    def on_input_submitted(self, event: Input.Submitted) -> None:
        raw = event.value.strip()
        if not raw:
            return
        self.query_one("#task-input", Input).clear()
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
            self._append_output(f"> {raw}\nRunning...")
            self._run_cli_async(["run", parsed.task], refresh_after=True)
        elif parsed.cmd == TuiCommand.PACKS:
            self._append_output(_render_packs_list())
        elif parsed.cmd == TuiCommand.PACK_SHOW:
            self._append_output(_render_pack_detail(parsed.pack_id))
        else:
            self._append_output(f"Unknown command: {raw}\nType /help for supported commands.")

    @work(thread=True)
    def _run_cli_async(self, args: list[str], refresh_after: bool = False) -> None:
        # Temporary v0 bridge: routes TUI input through existing Click commands via
        # CliRunner so we avoid shell=True and stay within supported code paths.
        from openshard.cli.main import cli as openshard_cli

        runner = CliRunner()
        result = runner.invoke(openshard_cli, args, input="", catch_exceptions=True)
        output = result.output or (str(result.exception) if result.exception else "")
        status = "Done." if result.exit_code == 0 else f"Failed (exit {result.exit_code})."
        self.call_from_thread(self._on_cli_result, output, status, refresh_after)

    def _on_cli_result(self, output: str, status: str, refresh_after: bool) -> None:
        self._append_output(output.rstrip("\n") + "\n" + status)
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
