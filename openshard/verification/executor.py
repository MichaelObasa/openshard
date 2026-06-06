from __future__ import annotations

import subprocess
from pathlib import Path

import click

from openshard.verification.plan import CommandSafety, VerificationPlan

_DEFAULT_TIMEOUT: float = 120.0


def confirm_or_abort(reason: str) -> None:
    click.echo(f"\n[gate] {reason}")
    if not click.confirm("Proceed?", default=False):
        click.echo("Aborted!")
        raise SystemExit(0)


def run_verification_plan(
    plan: VerificationPlan,
    cwd: Path,
    gate=None,
    label: str = "[verify]",
    capture: bool = False,
    detail: str = "default",
    timeout: float = _DEFAULT_TIMEOUT,
) -> int | tuple[int, str]:
    """Execute the first VerificationCommand from *plan*.

    - No commands  → returns 0; announces nothing-to-run (matches old behaviour).
    - blocked      → skips subprocess, returns 1 with a clear message.
    - needs_approval with gate → calls confirm_or_abort when approval required.
    - safe         → executes argv directly (never shell=True).

    capture=False: streams output live, returns int exit code.
    capture=True: captures silently, returns (exit_code, output).

    timeout: seconds before the subprocess is killed (default 120). Callers that
    depend on captured output still receive a string on timeout; raw stdout is
    never stored in run history or NativeStepEvent metadata.
    """
    if not plan.has_commands:
        if not capture:
            click.echo(f"  {label} no test command detected")
        return (0, "") if capture else 0

    cmd = plan.commands[0]

    if cmd.safety == CommandSafety.blocked:
        msg = f"  {label} blocked: {cmd.reason}"
        if not capture:
            click.echo(msg)
        return (1, msg) if capture else 1

    if cmd.safety == CommandSafety.needs_approval and gate is not None:
        _sc_dec = gate.check_shell_command(" ".join(cmd.argv))
        if _sc_dec.required:
            confirm_or_abort(_sc_dec.reason)

    if not capture:
        click.echo(f"  {label} running: {' '.join(cmd.argv)}")

    try:
        proc = subprocess.run(  # type: ignore[call-overload]  # dynamic text/encoding kwargs via ** unpacking
            cmd.argv,
            cwd=cwd,
            stdout=subprocess.PIPE if capture else None,
            stderr=subprocess.STDOUT if capture else None,
            text=capture,
            timeout=timeout,
            **({"encoding": "utf-8", "errors": "replace"} if capture else {}),
        )
    except subprocess.TimeoutExpired:
        msg = f"  {label} timed out after {timeout}s"
        if not capture:
            click.echo(msg)
        return (1, msg) if capture else 1
    except FileNotFoundError:
        msg = f"  {label} not found"
        if not capture:
            click.echo(msg)
        return (1, msg) if capture else 1

    if capture:
        return proc.returncode, proc.stdout or ""

    if proc.returncode == 0:
        click.echo(f"  {label} passed")
    else:
        click.echo(f"  {label} failed (exit code {proc.returncode})")
    return proc.returncode
