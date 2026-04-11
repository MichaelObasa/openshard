import subprocess
import sys
from pathlib import Path

import click

from openshard.execution.generator import ChangedFile, ExecutionGenerator
from openshard.planning.generator import PlanGenerator
from openshard.providers.openrouter import AuthError, OpenRouterError, RateLimitError


@click.group()
@click.version_option()
def cli():
    """OpenShard - intelligent task routing and execution."""


@cli.command()
@click.argument("task")
def plan(task: str):
    """Analyse TASK and produce a structured execution plan."""
    try:
        generator = PlanGenerator()
    except ValueError as exc:
        raise click.ClickException(str(exc))

    try:
        result = generator.generate(task)
    except AuthError:
        raise click.ClickException(
            "Authentication failed. Check that OPENROUTER_API_KEY is valid."
        )
    except RateLimitError:
        raise click.ClickException("Rate limit exceeded. Wait a moment then try again.")
    except OpenRouterError as exc:
        raise click.ClickException(f"API error: {exc}")

    click.echo(f"\nTask: {task}\n")
    click.echo(f"Summary: {result.summary}\n")
    for i, stage in enumerate(result.stages, 1):
        click.echo(f"Stage {i}: {stage.name} [{stage.tier}]")
        click.echo(f"  {stage.reasoning}")


@cli.command()
@click.argument("task")
@click.option("--write", is_flag=True, default=False, help="Write generated files to disk.")
@click.option("--verify", is_flag=True, default=False, help="Run verification after writing (requires --write).")
def run(task: str, write: bool, verify: bool):
    """Execute TASK and return a structured result."""
    try:
        generator = ExecutionGenerator()
    except ValueError as exc:
        raise click.ClickException(str(exc))

    try:
        result = generator.generate(task)
    except AuthError:
        raise click.ClickException(
            "Authentication failed. Check that OPENROUTER_API_KEY is valid."
        )
    except RateLimitError:
        raise click.ClickException("Rate limit exceeded. Wait a moment then try again.")
    except OpenRouterError as exc:
        raise click.ClickException(f"API error: {exc}")

    click.echo(f"\nTask: {task}\n")
    click.echo(f"Summary: {result.summary}\n")
    if result.files:
        click.echo("Files changed")
        for f in result.files:
            click.echo(f"  - {f.path} [{f.change_type}] - {f.summary}")
    if result.notes:
        click.echo("\nNotes")
        for note in result.notes:
            click.echo(f"  - {note}")

    if verify and not write:
        raise click.ClickException("--verify requires --write.")

    if write:
        click.echo("")
        _write_files(result.files)

    if write and verify:
        click.echo("")
        _run_verification(Path.cwd())


def _write_files(files: list[ChangedFile]) -> None:
    cwd = Path.cwd().resolve()
    for f in files:
        if not f.path:
            click.echo(f"  [skip] empty path")
            continue

        target = (cwd / f.path).resolve()
        if not str(target).startswith(str(cwd)):
            click.echo(f"  [skip] unsafe path rejected: {f.path}")
            continue

        if f.change_type == "delete":
            if target.exists():
                target.unlink()
                click.echo(f"  [deleted] {f.path}")
            else:
                click.echo(f"  [skip] not found: {f.path}")
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(f.content, encoding="utf-8")
            click.echo(f"  [written] {f.path}")


def _detect_command(cwd: Path) -> list[str] | None:
    if (cwd / "package.json").exists():
        return ["npm", "test"]
    if (cwd / "pyproject.toml").exists() or (cwd / "tests").is_dir():
        return ["pytest"]
    return None


def _run_verification(cwd: Path) -> None:
    cmd = _detect_command(cwd)
    if cmd is None:
        click.echo("  [verify] no test command detected")
        return

    click.echo(f"  [verify] running: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=cwd)
    if result.returncode == 0:
        click.echo("  [verify] passed")
    else:
        click.echo(f"  [verify] failed (exit code {result.returncode})")
        sys.exit(result.returncode)


@cli.command()
def report():
    """Display a summary report of recent executions."""
    click.echo("[report] Fetching execution report...")

if __name__ == "__main__":
    cli()