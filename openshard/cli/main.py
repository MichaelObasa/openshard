import datetime
import json
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
@click.option("--more", is_flag=True, default=False, help="Show file list, retry info, model names, and token breakdown.")
@click.option("--full", is_flag=True, default=False, help="Show all details: workspace, verification command, retry prompt, raw output.")
@click.option("--no-shrink", is_flag=True, default=False, help="Disable output shrinking for long results.")
@click.option(
    "--mode",
    type=click.Choice(["auto", "ask", "smart"], case_sensitive=False),
    default=None,
    help="Execution mode: auto (always proceed), ask (always prompt), smart (prompt on risky tasks).",
)
def run(task: str, write: bool, verify: bool, dry_run: bool, more: bool, full: bool, no_shrink: bool, mode: str | None):
    """Execute TASK and return a structured result."""
    detail = "full" if full else ("more" if more else "default")
    start = time.time()
    retry_triggered = False
    workspace: Path | None = None
    verification_passed: bool | None = None
    usage = None
    retry_usage = None

    try:
        generator = ExecutionGenerator()
    except ValueError as exc:
        raise click.ClickException(str(exc))

    click.echo("\nRunning task...")
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

    usage = result.usage
    final_files = result.files

    click.echo("\n✔ Done")
    click.echo(result.summary)
    if result.files:
        click.echo("\nFiles")
        shrinking = _should_shrink(result.files, no_shrink)
        files_to_show = result.files[:5] if shrinking else result.files
        for f in files_to_show:
            if detail == "default":
                click.echo(f"  {f.path} ({_CHANGE_LABEL.get(f.change_type, f.change_type)})")
            else:
                click.echo(f"  {f.path} ({_CHANGE_LABEL.get(f.change_type, f.change_type)})  — {f.summary}")
        if shrinking and len(result.files) > 5:
            click.echo(f"  ... and {len(result.files) - 5} more (use --no-shrink to see all)")
    if result.notes:
        click.echo("\nNotes")
        for note in result.notes:
            click.echo(f"  {note}")

    if dry_run:
        if _should_shrink(result.files, no_shrink):
            _print_shrunk(result.files, result.summary)
        else:
            _print_dry_run(result.files)
        _print_summary(start, generator, retry_triggered, final_files, usage=usage, detail=detail)
        try:
            _log_run(start, task, generator, retry_triggered, final_files,
                     verification_attempted=False, verification_passed=None,
                     workspace=None, usage=usage)
        except Exception as exc:
            click.echo(f"  [log] warning: {exc}")
        return

    if verify and not write:
        raise click.ClickException("--verify requires --write.")

    if write:
        effective_mode = _resolve_mode(load_config(), mode)
        if effective_mode == "ask":
            risks = _detect_risks(task)
            if not _confirm_proceed(risks if risks else ["write requested"]):
                click.echo("  Aborted.")
                return
        elif effective_mode == "smart":
            risks = _detect_risks(task)
            if risks and not _confirm_proceed(risks):
                click.echo("  Aborted.")
                return

        workspace = Path(tempfile.mkdtemp())
        if detail == "full":
            click.echo(f"\n  [workspace] {workspace}")
        _write_files(result.files, workspace)

    if write and verify:
        click.echo("")
        code = _run_verification(workspace, detail=detail)
        if code != 0:
            retry_triggered = True
            if detail != "default":
                click.echo("  Retrying with fixer model...")
            _, verify_output = _run_verification(workspace, capture=True)
            retry_prompt = _build_retry_prompt(task, result, verify_output)
            if detail == "full":
                snippet = retry_prompt[:300] + ("..." if len(retry_prompt) > 300 else "")
                click.echo(f"\n  [retry prompt] {snippet}")
                if verify_output.strip():
                    click.echo(f"\n  [verify output] {verify_output.strip()[:300]}")
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
            retry_usage = retry_result.usage
            final_files = retry_result.files
            _write_files(retry_result.files, workspace)
            code = _run_verification(workspace, label="[retry]", detail=detail)
        verification_passed = code == 0
        if code != 0:
            _print_summary(start, generator, retry_triggered, final_files,
                           usage=usage, retry_usage=retry_usage, detail=detail)
            try:
                _log_run(start, task, generator, retry_triggered, final_files,
                         verification_attempted=True, verification_passed=False,
                         workspace=workspace, usage=usage, retry_usage=retry_usage)
            except Exception as exc:
                click.echo(f"  [log] warning: {exc}")
            sys.exit(code)

    _print_summary(start, generator, retry_triggered, final_files,
                   usage=usage, retry_usage=retry_usage, detail=detail)
    try:
        _log_run(start, task, generator, retry_triggered, final_files,
                 verification_attempted=(write and verify),
                 verification_passed=verification_passed,
                 workspace=workspace, usage=usage, retry_usage=retry_usage)
    except Exception as exc:
        click.echo(f"  [log] warning: {exc}")


_LOG_PATH = Path(".openshard") / "runs.jsonl"


def _log_run(
    start: float,
    task: str,
    generator: ExecutionGenerator,
    retry_triggered: bool,
    files: list[ChangedFile],
    verification_attempted: bool,
    verification_passed: bool | None,
    workspace: Path | None,
    usage=None,
    retry_usage=None,
) -> None:
    entry: dict = {
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat().replace("+00:00", "Z"),
        "task": task,
        "execution_model": generator.model,
        "retry_triggered": retry_triggered,
        "duration_seconds": round(time.time() - start, 2),
        "files_created": sum(1 for f in files if f.change_type == "create"),
        "files_updated": sum(1 for f in files if f.change_type == "update"),
        "files_deleted": sum(1 for f in files if f.change_type == "delete"),
        "verification_attempted": verification_attempted,
        "verification_passed": verification_passed,
        "workspace_path": str(workspace) if workspace else None,
    }
    if retry_triggered:
        entry["fixer_model"] = generator.fixer_model
    if usage is not None:
        entry["prompt_tokens"] = usage.prompt_tokens
        entry["completion_tokens"] = usage.completion_tokens
        entry["total_tokens"] = usage.total_tokens
        entry["estimated_cost"] = usage.estimated_cost
    if retry_usage is not None:
        entry["retry_prompt_tokens"] = retry_usage.prompt_tokens
        entry["retry_completion_tokens"] = retry_usage.completion_tokens
        entry["retry_total_tokens"] = retry_usage.total_tokens
        entry["retry_estimated_cost"] = retry_usage.estimated_cost

    log_path = Path.cwd() / _LOG_PATH
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry) + "\n")


def _print_summary(
    start: float,
    generator: ExecutionGenerator,
    retry_triggered: bool,
    files: list[ChangedFile],
    usage=None,
    retry_usage=None,
    detail: str = "default",
) -> None:
    elapsed = time.time() - start
    cost_str = (
        f"${usage.estimated_cost:.4f}"
        if usage is not None and usage.estimated_cost is not None
        else "-"
    )

    if detail == "default":
        click.echo(f"\nTime: {elapsed:.1f}s   Cost: {cost_str}")
        return

    # more and full
    click.echo(f"\nModel: {generator.model}")
    if retry_triggered:
        click.echo(f"Fixer model: {generator.fixer_model}")
        click.echo("Retried: yes")
    if usage is not None:
        click.echo(
            f"Tokens: {usage.prompt_tokens} prompt / "
            f"{usage.completion_tokens} completion / "
            f"{usage.total_tokens} total"
        )
    if detail == "full":
        created = sum(1 for f in files if f.change_type == "create")
        updated = sum(1 for f in files if f.change_type == "update")
        deleted = sum(1 for f in files if f.change_type == "delete")
        click.echo(f"Files: {created} created / {updated} updated / {deleted} deleted")
        if retry_triggered and retry_usage is not None:
            retry_cost_str = (
                f"${retry_usage.estimated_cost:.4f}"
                if retry_usage.estimated_cost is not None else "-"
            )
            click.echo(
                f"Retry tokens: {retry_usage.prompt_tokens} prompt / "
                f"{retry_usage.completion_tokens} completion / "
                f"{retry_usage.total_tokens} total"
            )
            click.echo(f"Retry cost: {retry_cost_str}")
    click.echo(f"Time: {elapsed:.1f}s   Cost: {cost_str}")


_VALID_MODES = {"auto", "ask", "smart"}

_HIGH_RISK_KEYWORDS: dict[str, list[str]] = {
    "auth":        ["auth", "login", "logout", "jwt", "oauth", "session", "token", "password"],
    "payments":    ["payment", "stripe", "billing", "invoice", "charge"],
    "secrets":     ["secret", "api key", "credential", "private key"],
    "env/config":  [".env", "dotenv", "environment variable"],
    "infra":       ["deploy", "kubernetes", "docker", "terraform", "ci/cd", "pipeline"],
    "migrations":  ["migration", "migrate", "schema", "alter table", "drop table"],
    "deletes":     ["delete", "remove", "drop", "purge", "wipe", "truncate"],
    "permissions": ["permission", "rbac", "acl", "role", "privilege"],
    "security":    ["security", "firewall", "ssl", "tls", "certificate", "encryption"],
}


def _resolve_mode(config: dict, cli_mode: str | None) -> str:
    if cli_mode:
        return cli_mode.lower()
    value = config.get("mode", "smart")
    return value if value in _VALID_MODES else "smart"


def _detect_risks(task: str) -> list[str]:
    lower = task.lower()
    return [
        label
        for label, keywords in _HIGH_RISK_KEYWORDS.items()
        if any(kw in lower for kw in keywords)
    ]


def _confirm_proceed(risks: list[str]) -> bool:
    labels = ", ".join(risks)
    click.echo(f"\n  This looks high-risk ({labels}).")
    answer = click.prompt("  Proceed? [y/N]", default="N", show_default=False)
    return answer.strip().lower() == "y"


_CHANGE_LABEL = {"create": "created", "update": "updated", "delete": "deleted"}

_SHRINK_CHAR_THRESHOLD = 6_000
_SHRINK_LINE_THRESHOLD = 1_500
_SHRINK_ERROR_PATTERNS = ("error", "exception", "failed", "traceback")


def _should_shrink(files: list[ChangedFile], no_shrink: bool) -> bool:
    if no_shrink:
        return False
    total_chars = sum(len(f.content) for f in files)
    total_lines = sum(f.content.count("\n") for f in files)
    return total_chars > _SHRINK_CHAR_THRESHOLD or total_lines > _SHRINK_LINE_THRESHOLD


def _print_shrunk(files: list[ChangedFile], result_summary: str) -> None:
    total_chars = sum(len(f.content) for f in files)
    click.echo(
        f"\n  Output condensed — {len(files)} file(s), ~{total_chars} chars."
        " Use --no-shrink to see full content."
    )
    click.echo(f"\n{result_summary}\n")
    click.echo("Files")
    for f in files[:5]:
        click.echo(f"  {f.path} ({f.change_type})  — {f.summary}")
    if len(files) > 5:
        click.echo(f"  ... and {len(files) - 5} more")

    error_lines: list[str] = []
    for f in files:
        for line in f.content.splitlines():
            if any(pat in line.lower() for pat in _SHRINK_ERROR_PATTERNS):
                error_lines.append(line.strip())
                if len(error_lines) >= 5:
                    break
        if len(error_lines) >= 5:
            break
    if error_lines:
        click.echo("\nErrors detected:")
        for line in error_lines:
            click.echo(f"  {line}")


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
    cwd: Path, label: str = "[verify]", capture: bool = False, detail: str = "default"
) -> int | tuple[int, str]:
    """Run the detected test command.

    capture=False (default): streams output live, returns exit code as int.
    capture=True: captures stdout+stderr silently, returns (exit_code, output).
    """
    cmd = _detect_command(cwd)
    if cmd is None:
        if not capture and detail == "full":
            click.echo(f"  {label} no test command detected")
        return (0, "") if capture else 0

    if not capture and detail == "full":
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

    if detail == "full":
        if proc.returncode == 0:
            click.echo(f"  {label} passed")
        else:
            click.echo(f"  {label} failed (exit code {proc.returncode})")
    else:
        if proc.returncode == 0:
            click.echo("  ✔ Verified")
        else:
            click.echo(f"  ✗ Verification failed (exit {proc.returncode})")
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
@click.argument("task")
def explain(task: str):
    """Explain how OpenShard would approach TASK and which model strengths apply."""
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

    strong = [s for s in result.stages if s.tier == "strong"]
    medium = [s for s in result.stages if s.tier == "medium"]
    cheap  = [s for s in result.stages if s.tier == "cheap"]

    click.echo(f"\nTask: {task}\n")
    click.echo(f"Summary: {result.summary}\n")

    if strong:
        click.echo("Hard parts (strong model):")
        for s in strong:
            click.echo(f"  - {s.name}: {s.reasoning}")
        click.echo("")

    if medium:
        click.echo("Standard parts:")
        for s in medium:
            click.echo(f"  - {s.name}: {s.reasoning}")
        click.echo("")

    if cheap:
        click.echo("Low-risk parts (cheap model):")
        for s in cheap:
            click.echo(f"  - {s.name}: {s.reasoning}")
        click.echo("")

    click.echo("Retry / fix:")
    if strong:
        click.echo("  Complex task — fixer would benefit from a stronger model.")
    else:
        click.echo("  Straightforward task — default fixer model should be sufficient.")


@cli.command()
def report():
    """Display a summary report of recent executions."""
    log_path = Path.cwd() / _LOG_PATH

    if not log_path.exists():
        click.echo("No run history found. Run 'openshard run' to get started.")
        return

    entries = []
    for line in log_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            entries.append(json.loads(line))
        except json.JSONDecodeError:
            continue

    if not entries:
        click.echo("No runs recorded yet.")
        return

    total         = len(entries)
    verify_passed = sum(1 for e in entries if e.get("verification_passed") is True)
    verify_failed = sum(1 for e in entries if e.get("verification_passed") is False)
    retry_count   = sum(1 for e in entries if e.get("retry_triggered") is True)
    avg_duration  = sum(e.get("duration_seconds", 0) for e in entries) / total
    total_tokens  = sum(e.get("total_tokens", 0) for e in entries)
    costs = [e["estimated_cost"] for e in entries if e.get("estimated_cost") is not None]
    total_cost    = sum(costs) if costs else None
    avg_cost      = total_cost / len(costs) if costs else None

    click.echo("\n[report]")
    click.echo(f"  total runs:             {total}")
    click.echo(f"  successful verifications: {verify_passed}")
    click.echo(f"  failed verifications:   {verify_failed}")
    click.echo(f"  retries triggered:      {retry_count}")
    click.echo(f"  average duration:       {avg_duration:.1f}s")
    click.echo(f"  total tokens:           {total_tokens}")
    click.echo(f"  total cost:             {'$' + f'{total_cost:.4f}' if total_cost is not None else '-'}")
    click.echo(f"  average cost per run:   {'$' + f'{avg_cost:.4f}' if avg_cost is not None else '-'}")

    click.echo("\n  recent runs:")
    for entry in entries[-5:][::-1]:
        ts = entry.get("timestamp", "")
        ts = ts.rstrip("Z").replace("T", " ").split(".")[0]
        task  = entry.get("task", "")[:50]
        model = entry.get("execution_model", "")
        retry = "yes" if entry.get("retry_triggered") else "no"
        vp    = entry.get("verification_passed")
        vstr  = "passed" if vp is True else ("failed" if vp is False else "-")
        click.echo(f"  {ts}  {task}")
        click.echo(f"    model: {model}  retry: {retry}  verify: {vstr}")

if __name__ == "__main__":
    cli()