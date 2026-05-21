from __future__ import annotations

import shlex
from dataclasses import dataclass
from enum import Enum


class TuiCommand(Enum):
    RUN_TASK = "run_task"
    LAST = "last"
    LAST_MORE = "last_more"
    HELP = "help"
    CLEAR = "clear"
    QUIT = "quit"
    PACKS = "packs"
    PACK_SHOW = "pack_show"
    UNKNOWN = "unknown"


@dataclass
class ParsedCommand:
    cmd: TuiCommand
    task: str | None = None
    pack_id: str | None = None


def parse_tui_input(text: str) -> ParsedCommand:
    text = text.strip()
    if not text:
        return ParsedCommand(TuiCommand.UNKNOWN)

    if text.startswith("/"):
        lower = text.lower()
        if lower == "/packs":
            return ParsedCommand(TuiCommand.PACKS)
        if lower.startswith("/packs "):
            pack_id = text[7:].strip().lower()
            return ParsedCommand(TuiCommand.PACK_SHOW, pack_id=pack_id or None)
        if lower == "/pack":
            return ParsedCommand(TuiCommand.PACK_SHOW, pack_id=None)
        if lower.startswith("/pack "):
            pack_id = text[6:].strip().lower()
            return ParsedCommand(TuiCommand.PACK_SHOW, pack_id=pack_id or None)
        match lower:
            case "/help":
                return ParsedCommand(TuiCommand.HELP)
            case "/last more":
                return ParsedCommand(TuiCommand.LAST_MORE)
            case "/last":
                return ParsedCommand(TuiCommand.LAST)
            case "/clear":
                return ParsedCommand(TuiCommand.CLEAR)
            case "/quit":
                return ParsedCommand(TuiCommand.QUIT)
            case _:
                return ParsedCommand(TuiCommand.UNKNOWN)

    if text.startswith("openshard "):
        try:
            parts = shlex.split(text)
        except ValueError:
            return ParsedCommand(TuiCommand.UNKNOWN)
        # Only `openshard run <task>` is supported; all other subcommands are UNKNOWN
        if len(parts) >= 3 and parts[1] == "run":
            task = " ".join(parts[2:])
            return ParsedCommand(TuiCommand.RUN_TASK, task=task) if task else ParsedCommand(TuiCommand.UNKNOWN)
        return ParsedCommand(TuiCommand.UNKNOWN)

    return ParsedCommand(TuiCommand.RUN_TASK, task=text)
