import subprocess
import sys
import tempfile
import time
from pathlib import Path

import click

from openshard.config.settings import load_config
from openshard.execution.generator import ChangedFile, ExecutionGenerator, ExecutionResult
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
@click.option("--dry-run", is_flag=True, default=False, help="Preview files without writing.")
def run(task: str, write: bool, verify: bool, dry_run: bool):
    """Execute TASK and return a structured result."""
    start = time.time()
    retry_triggered = False

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

    final_files = result.files

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

    if dry_run:
        _print_dry_run(result.files)
        _print_summary(start, generator, retry_triggered, final_files)
        return

    if verify and not write:
        raise click.ClickException("--verify requires --write.")

    if write:
        workspace = Path(tempfile.mkdtemp())
        click.echo(f"\n  [workspace] {workspace}")
        _write_files(result.files, workspace)

    if write and verify:
        click.echo("")
        code = _run_verification(workspace)
        if code != 0:
            retry_triggered = True
            click.echo("  [retry] attempting fix")
            _, verify_output = _run_verification(workspace, capture=True)
            retry_prompt = _build_retry_prompt(task, result, verify_output)
            try:
                retry_result = generator.generate(retry_prompt, model=generator.fixer_model)
            except AuthError:
                raise click.ClickException(
                    "Authentication failed. Check that OPENROUTER_API_KEY is valid."
                )
            except RateLimitError:
                raise click.ClickException("Rate limit exceeded. Wait a moment then try again.")
            except OpenRouterError as exc:
                raise click.ClickException(f"API error: {exc}")
            final_files = retry_result.files
            _write_files(retry_result.files, workspace)
            code = _run_verification(workspace, label="[retry]")
        if code != 0:
            _print_summary(start, generator, retry_triggered, final_files)
            sys.exit(code)

    _print_summary(start, generator, retry_triggered, final_files)


def _print_summary(
    start: float,
    generator: ExecutionGenerator,
    retry_triggered: bool,
    files: list[ChangedFile],
) -> None:
    elapsed = time.time() - start
    created  = sum(1 for f in files if f.change_type == "create")
    updated  = sum(1 for f in files if f.change_type == "update")
    deleted  = sum(1 for f in files if f.change_type == "delete")
    click.echo("\n[summary]")
    click.echo(f"  time:            {elapsed:.1f}s")
    click.echo(f"  execution_model: {generator.model}")
    if retry_triggered:
        click.echo(f"  fixer_model:     {generator.fixer_model}")
    click.echo(f"  retry:           {'yes' if retry_triggered else 'no'}")
    click.echo(f"  files:           {created} created / {updated} updated / {deleted} deleted")


def _print_dry_run(files: list[ChangedFile]) -> None:
    if not files:
        click.echo("\n(no files to preview)")
        return
    click.echo("")
    for f in files:
        click.echo(f"--- {f.path} [{f.change_type}] ---")
        if f.change_type == "delete" or not f.content:
            click.echo("(no content — file will be deleted)")
        else:
            click.echo(f.content)
        click.echo("")


def _write_files(files: list[ChangedFile], root: Path) -> None:
    cwd = root.resolve()
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


def _detect_command(cwd: Path, config: dict | None = None) -> list[str] | None:
    if config is None:
        try:
            config = load_config()
        except Exception:
            config = {}
    cmd = config.get("verification_command")
    if cmd:
        return list(cmd)
    if (cwd / "package.json").exists():
        return ["npm", "test"]
    if (cwd / "pyproject.toml").exists() or (cwd / "tests").is_dir():
        return [sys.executable, "-m", "pytest"]
    return None


def _run_verification(
    cwd: Path, label: str = "[verify]", capture: bool = False
) -> int | tuple[int, str]:
    """Run the detected test command.

    capture=False (default): streams output live, returns exit code as int.
    capture=True: captures stdout+stderr silently, returns (exit_code, output).
    """
    cmd = _detect_command(cwd)
    if cmd is None:
        if not capture:
            click.echo(f"  {label} no test command detected")
        return (0, "") if capture else 0

    if not capture:
        click.echo(f"  {label} running: {' '.join(cmd)}")
    proc = subprocess.run(
        cmd,
        cwd=cwd,
        stdout=subprocess.PIPE if capture else None,
        stderr=subprocess.STDOUT if capture else None,
        text=capture,
    )
    if capture:
        return proc.returncode, proc.stdout or ""

    if proc.returncode == 0:
        click.echo(f"  {label} passed")
    else:
        click.echo(f"  {label} failed (exit code {proc.returncode})")
    return proc.returncode


def _build_retry_prompt(task: str, result: ExecutionResult, verify_output: str) -> str:
    lines = [
        "The previous implementation failed verification.",
        "Fix the issues and return updated files only.",
        "",
        f"Original task: {task}",
        f"Previous summary: {result.summary}",
        "",
        "Files that were written:",
    ]
    for f in result.files:
        lines += [
            f"  path: {f.path}",
            f"  change_type: {f.change_type}",
            f"  summary: {f.summary}",
            "  content:",
            f.content,
            "",
        ]
    lines += [
        "Verification output:",
        verify_output.strip() if verify_output.strip() else "(no output)",
        "",
        "Instructions:",
        "- Goal: make the existing tests pass by fixing the implementation, not by changing the tests.",
        "- Prefer fixing implementation files. Only modify a test file if it contains a clear bug.",
        "- Do NOT mark any test as xfail or skip.",
        "- Do NOT delete any failing test.",
        "- Do NOT weaken any assertion.",
        "- Do NOT change expected values to match wrong output.",
        "",
        "Respond with valid JSON only — no markdown, no prose, no code fences.",
        'Use exactly this schema: {"summary": "...", "files": [{"path": "...", '
        '"change_type": "create|update|delete", "content": "...", "summary": "..."}], "notes": [...]}',
    ]
    return "\n".join(lines)


@cli.command()
def report():
    """Display a summary report of recent executions."""
    click.echo("[report] Fetching execution report...")

if __name__ == "__main__":
    cli()