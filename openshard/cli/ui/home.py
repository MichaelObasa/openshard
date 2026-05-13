from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path
from typing import Any

from rich import box
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from openshard.analysis.repo import RepoFacts, analyze_repo
from openshard.config.settings import load_config
from openshard.run.pipeline import _LOG_PATH

from .console import make_console
from .theme import MUTED_STYLE, SECTION_STYLE, TITLE_STYLE

TAGLINE = "The control layer for AI coding agents."
SUPPORT_LINE = "Route tasks. Gate risky actions. Verify results. Keep an execution record."
TRUST_LINE = "Local-first by default. Your runs, receipts, and policies stay under your control."


@dataclass
class HomeState:
    title: str
    repo_path: str
    git_state: str
    stack_state: str
    route_state: list[tuple[str, str]]
    receipts: list[dict[str, Any]]


def render_home() -> None:
    """Render the OpenShard home screen without provider/model side effects."""
    console = make_console()
    console.print(_build_home_panel(_collect_home_state()))


def _collect_home_state() -> HomeState:
    cwd = Path.cwd()
    config = _safe_load_config()
    facts = _safe_analyze_repo(cwd)
    return HomeState(
        title=_title(),
        repo_path=str(cwd),
        git_state=_git_state(cwd, facts),
        stack_state=_stack_state(facts),
        route_state=_route_state(config),
        receipts=_recent_receipts(cwd / _LOG_PATH),
    )


def _title() -> str:
    try:
        version = metadata.version("openshard")
    except metadata.PackageNotFoundError:
        return "OpenShard dev"
    return f"OpenShard v{version}"


def _safe_load_config() -> dict[str, Any]:
    try:
        return load_config()
    except Exception:
        return {}


def _safe_analyze_repo(path: Path) -> RepoFacts | None:
    try:
        return analyze_repo(path)
    except Exception:
        return None


def _git_state(path: Path, facts: RepoFacts | None) -> str:
    try:
        result = subprocess.run(
            ["git", "status", "--short", "--branch"],
            cwd=str(path),
            capture_output=True,
            text=True,
            timeout=3,
        )
    except Exception:
        result = None

    if result is None or result.returncode != 0:
        changed = len(facts.changed_files) if facts else 0
        if changed:
            return f"{changed} changed file(s)"
        return "git state unavailable"

    lines = [line for line in result.stdout.splitlines() if line.strip()]
    changed = [line for line in lines if not line.startswith("##")]
    branch = lines[0].removeprefix("## ").strip() if lines else "unknown branch"
    if changed:
        return f"{branch} - {len(changed)} changed file(s)"
    return f"{branch} - git clean"


def _stack_state(facts: RepoFacts | None) -> str:
    if facts is None:
        return "repo analysis unavailable"

    parts: list[str] = []
    if facts.languages:
        parts.append(", ".join(lang.title() for lang in facts.languages))
    if facts.framework:
        parts.append(facts.framework)
    if facts.test_command:
        parts.append("tests detected")
    elif facts.package_files:
        parts.append(", ".join(facts.package_files[:3]))
    return " - ".join(parts) if parts else "stack not detected"


def _route_state(config: dict[str, Any]) -> list[tuple[str, str]]:
    if not config:
        return [
            ("Route", "Config unavailable"),
            ("Risk", "Repo signals only"),
            ("Verify", "Not configured"),
            ("Cost", "No run cost yet"),
            ("Receipt", "Local history"),
        ]

    workflow = config.get("workflow") or config.get("executor") or "auto"
    execution_model = config.get("execution_model") or _tier_model(config, "balanced")
    planning_model = config.get("planning_model") or _tier_model(config, "powerful")
    verify = config.get("verification_command") or "auto-detect when requested"
    cost_gate = config.get("cost_gate_threshold")
    free_mode = config.get("free_mode")
    free_label = str(free_mode) if free_mode is not None else "not configured"

    return [
        ("Route", f"{workflow} workflow"),
        ("Risk", "Policy gates from repo/task signals"),
        ("Verify", _shorten(str(verify), 44)),
        ("Cost", f"gate ${cost_gate:.2f}" if isinstance(cost_gate, (int, float)) else "not configured"),
        ("Receipt", "saved to .openshard/runs.jsonl"),
        ("Free mode", free_label),
        ("Run model", _shorten(str(execution_model or "not configured"), 44)),
        ("Plan model", _shorten(str(planning_model or "not configured"), 44)),
    ]


def _tier_model(config: dict[str, Any], tier_name: str) -> str | None:
    for tier in config.get("model_tiers", []) or []:
        if isinstance(tier, dict) and tier.get("name") == tier_name:
            model = tier.get("model")
            return str(model) if model else None
    return None


def _recent_receipts(log_path: Path) -> list[dict[str, Any]]:
    if not log_path.exists():
        return []

    entries: list[dict[str, Any]] = []
    try:
        lines = log_path.read_text(encoding="utf-8").splitlines()
    except Exception:
        return []

    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            parsed = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            entries.append(parsed)
    return entries[-3:]


def _build_home_panel(state: HomeState) -> Panel:
    grid = Table.grid(padding=(0, 1))
    grid.expand = True

    header = Text()
    header.append(state.title, style=TITLE_STYLE)
    header.append("\n")
    header.append(TAGLINE)
    header.append("\n")
    header.append(SUPPORT_LINE, style=MUTED_STYLE)
    header.append("\n")
    header.append(TRUST_LINE, style=MUTED_STYLE)
    grid.add_row(header)
    grid.add_row("")

    grid.add_row(_repo_table(state))
    grid.add_row("")
    grid.add_row(_route_table(state.route_state))
    grid.add_row("")
    grid.add_row(_receipts_table(state.receipts))
    grid.add_row("")
    grid.add_row(_commands_table())

    return Panel(grid, box=box.ROUNDED, border_style="cyan", padding=(1, 2))


def _repo_table(state: HomeState) -> Table:
    table = Table.grid(padding=(0, 2))
    table.add_column(style=SECTION_STYLE, no_wrap=True)
    table.add_column()
    table.add_row("Repo", state.repo_path)
    table.add_row("Git", state.git_state)
    table.add_row("Stack", state.stack_state)
    return table


def _route_table(rows: list[tuple[str, str]]) -> Table:
    table = Table.grid(padding=(0, 2))
    table.add_column(style=SECTION_STYLE, no_wrap=True)
    table.add_column()
    for label, value in rows:
        table.add_row(label, value)
    return table


def _receipts_table(receipts: list[dict[str, Any]]) -> Table:
    table = Table.grid(padding=(0, 2))
    table.add_column(style=SECTION_STYLE, no_wrap=True)
    table.add_column()
    if not receipts:
        table.add_row("Recent receipts", "No recent receipts yet.")
        return table

    table.add_row("Recent receipts", "")
    for entry in reversed(receipts):
        table.add_row("", _receipt_line(entry))
    return table


def _commands_table() -> Table:
    table = Table.grid(padding=(0, 2))
    table.add_column(style=SECTION_STYLE, no_wrap=True)
    table.add_column()
    table.add_row("Try", 'openshard run "explain this repo"')
    table.add_row("", "openshard last --full")
    table.add_row("", "openshard demo-run")
    table.add_row("", "openshard export-runs --preview")
    return table


def _receipt_line(entry: dict[str, Any]) -> str:
    mode = (
        entry.get("execution_mode_label")
        or entry.get("execution_form_factor")
        or entry.get("workflow")
        or "Run"
    )
    task = _shorten(str(entry.get("task") or entry.get("summary") or "untitled run"), 34)
    verify = _verify_label(entry)
    cost = entry.get("estimated_cost")
    cost_label = f"${cost:.4f}" if isinstance(cost, (int, float)) else "-"
    return f"{mode} - {task} - {verify} - {cost_label} - receipt"


def _verify_label(entry: dict[str, Any]) -> str:
    if entry.get("verification_passed") is True:
        return "verified"
    if entry.get("verification_passed") is False:
        return "verify failed"
    if entry.get("verification_attempted") is False:
        return "verify skipped"
    return "verify unknown"


def _shorten(value: str, limit: int) -> str:
    if len(value) <= limit:
        return value
    return value[: limit - 3].rstrip() + "..."
