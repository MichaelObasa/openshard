from __future__ import annotations

import shlex
from dataclasses import dataclass
from enum import Enum

from openshard.tui.plan_mode import PLAN_FAST_PATHS


class TuiCommand(Enum):
    RUN_TASK = "run_task"
    LAST = "last"
    LAST_MORE = "last_more"
    LAST_FULL = "last_full"
    FEEDBACK = "feedback"
    HELP = "help"
    CLEAR = "clear"
    QUIT = "quit"
    PACKS = "packs"
    PACK_SHOW = "pack_show"
    ASK = "ask"
    PLAN = "plan"
    UNKNOWN = "unknown"


_VALID_FEEDBACK_OUTCOMES = {"accepted", "rejected", "partial", "abandoned", "retried"}


@dataclass
class ParsedCommand:
    cmd: TuiCommand
    task: str | None = None
    pack_id: str | None = None
    feedback_outcome: str | None = None
    feedback_reason: str | None = None
    question: str | None = None


_ASK_FAST_PATH: tuple[str, ...] = (
    "what models",
    "which models",
    "what commands",
    "what is openshard",
    "what does openshard",
)


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
        if lower.startswith("/feedback"):
            rest = text[9:].strip()
            if not rest:
                return ParsedCommand(TuiCommand.UNKNOWN)
            parts = rest.split(None, 1)
            outcome = parts[0].lower()
            if outcome not in _VALID_FEEDBACK_OUTCOMES:
                return ParsedCommand(TuiCommand.UNKNOWN)
            reason = parts[1] if len(parts) > 1 else None
            return ParsedCommand(TuiCommand.FEEDBACK, feedback_outcome=outcome, feedback_reason=reason)
        if lower == "/ask" or lower.startswith("/ask "):
            question = text[4:].strip() if lower.startswith("/ask ") else ""
            return ParsedCommand(TuiCommand.ASK, question=question)
        if lower == "/plan" or lower.startswith("/plan "):
            task = text[6:].strip() if lower.startswith("/plan ") else None
            return ParsedCommand(TuiCommand.PLAN, task=task or None)
        match lower:
            case "/help":
                return ParsedCommand(TuiCommand.HELP)
            case "/last full":
                return ParsedCommand(TuiCommand.LAST_FULL)
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

    _lower = text.lower()
    if any(_lower.startswith(p) for p in _ASK_FAST_PATH):
        return ParsedCommand(TuiCommand.ASK, question=text)
    if any(_lower.startswith(p) for p in PLAN_FAST_PATHS):
        return ParsedCommand(TuiCommand.PLAN, task=text)
    return ParsedCommand(TuiCommand.RUN_TASK, task=text)
