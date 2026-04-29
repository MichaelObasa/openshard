from __future__ import annotations

import re
import shlex
import sys
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass

_SHELL_METACHAR = re.compile(r"&&|\|(?!\|)|;|\$\(|`")

_SAFE_PREFIXES: list[tuple[str, ...]] = [
    ("python", "-m", "pytest"),
    ("python3", "-m", "pytest"),
    (sys.executable, "-m", "pytest"),
    ("pytest",),
    ("npm", "test"),
    ("cargo", "test"),
    ("go", "test"),
    ("bundle", "exec", "rspec"),
    ("mvn", "test"),
]

_BLOCKED_COMMANDS = {"curl", "wget", "rm", "sudo", "chmod", "chown"}

_TEST_KINDS: list[tuple[str, ...]] = [
    ("pytest",),
    ("python", "-m", "pytest"),
    ("python3", "-m", "pytest"),
    (sys.executable, "-m", "pytest"),
    ("npm", "test"),
    ("cargo", "test"),
    ("go", "test"),
    ("bundle", "exec", "rspec"),
    ("mvn", "test"),
]


class VerificationSource(str, Enum):
    config = "config"
    detected = "detected"
    user = "user"
    eval = "eval"


class VerificationKind(str, Enum):
    test = "test"
    lint = "lint"
    typecheck = "typecheck"
    build = "build"
    format_check = "format_check"
    unknown = "unknown"


class CommandSafety(str, Enum):
    safe = "safe"
    needs_approval = "needs_approval"
    blocked = "blocked"


@dataclass
class VerificationCommand:
    name: str
    argv: list[str]
    kind: VerificationKind
    source: VerificationSource
    safety: CommandSafety
    reason: str


@dataclass
class VerificationPlan:
    commands: list[VerificationCommand] = field(default_factory=list)

    @property
    def has_commands(self) -> bool:
        return bool(self.commands)


def classify_command_safety(
    argv: list[str], source: VerificationSource  # noqa: ARG001
) -> tuple[CommandSafety, str]:
    if not argv:
        return CommandSafety.blocked, "empty argv"

    for token in argv:
        if _SHELL_METACHAR.search(token):
            return CommandSafety.blocked, f"shell metacharacter in token: {token!r}"

    joined = " ".join(argv)
    if _SHELL_METACHAR.search(joined):
        return CommandSafety.blocked, "shell metacharacters detected"

    executable = argv[0].lower()
    base = executable.split("/")[-1].split("\\")[-1]
    if base in _BLOCKED_COMMANDS or executable in _BLOCKED_COMMANDS:
        return CommandSafety.blocked, f"blocked executable: {argv[0]!r}"

    for prefix in _SAFE_PREFIXES:
        if len(argv) >= len(prefix) and tuple(argv[: len(prefix)]) == prefix:
            return CommandSafety.safe, f"matches safe prefix: {' '.join(prefix)}"

    return CommandSafety.needs_approval, "unrecognised command requires approval"


def parse_command_to_argv(command: str) -> list[str]:
    try:
        posix = sys.platform != "win32"
        return shlex.split(command, posix=posix)
    except ValueError:
        return [command]


def _infer_kind(argv: list[str]) -> VerificationKind:
    for prefix in _TEST_KINDS:
        if len(argv) >= len(prefix) and tuple(argv[: len(prefix)]) == prefix:
            return VerificationKind.test
    return VerificationKind.unknown


def build_verification_plan(config: dict, repo_facts) -> VerificationPlan:
    argv: list[str] | None = None
    source: VerificationSource | None = None

    raw = config.get("verification_command")
    if isinstance(raw, list) and raw:
        argv = [str(t) for t in raw]
        source = VerificationSource.config
    elif isinstance(raw, str) and raw.strip():
        argv = parse_command_to_argv(raw.strip())
        source = VerificationSource.config
    elif repo_facts is not None and getattr(repo_facts, "test_command", None):
        argv = parse_command_to_argv(repo_facts.test_command)
        source = VerificationSource.detected

    if argv is None or source is None:
        return VerificationPlan()

    safety, reason = classify_command_safety(argv, source)
    kind = _infer_kind(argv)
    name = "tests" if (safety == CommandSafety.safe and kind == VerificationKind.test) else "verification"

    cmd = VerificationCommand(
        name=name,
        argv=argv,
        kind=kind,
        source=source,
        safety=safety,
        reason=reason,
    )
    return VerificationPlan(commands=[cmd])


def render_verification_plan(plan: VerificationPlan) -> str:
    lines = ["Verification"]
    if not plan.has_commands:
        lines.append("  no verification command detected")
    else:
        for cmd in plan.commands:
            argv_str = " ".join(cmd.argv)
            lines.append(f"  {cmd.name}  {cmd.safety.value}  {cmd.source.value}  {argv_str}")
    return "\n".join(lines)
