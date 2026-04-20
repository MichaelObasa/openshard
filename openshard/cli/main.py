import datetime
import json
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

import click

from openshard.config.settings import load_config
from openshard.execution.generator import ChangedFile, ExecutionGenerator, ExecutionResult
from openshard.execution.opencode_executor import OpenCodeExecutor
from openshard.planning.generator import PlanGenerator
from openshard.providers.openrouter import (
    AuthError, OpenRouterError, RateLimitError,
    MODEL_PRICING, compute_cost,
)
from openshard.routing.engine import ESCALATION_CHAIN, RoutingDecision, route
from openshard.execution.stages import (
    Stage, StageRun, split_task, route_stage, should_use_stages, run_planning_stage,
)


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
    "--executor",
    type=click.Choice(["direct", "opencode"], case_sensitive=False),
    default=None,
    help="Execution backend: direct (default, calls OpenRouter API) or opencode (calls opencode CLI).",
)
def run(task: str, write: bool, verify: bool, dry_run: bool, more: bool, full: bool, no_shrink: bool, executor: str | None):
    """Execute TASK and return a structured result."""
    detail = "full" if full else ("more" if more else "default")
    start = time.time()
    retry_triggered = False
    workspace: Path | None = None
    verification_passed: bool | None = None
    usage = None
    retry_usage = None

    try:
        _cfg = load_config()
        _cfg_executor = _cfg.get("executor", "direct").lower()
        _policy_executor, _policy_reason = _suggest_executor(task)
        if executor:
            # CLI flag is absolute; policy note is suppressed
            effective_executor = executor.lower()
            _policy_reason = ""
        elif _cfg_executor != "direct":
            # Config explicitly overrides default
            effective_executor = _cfg_executor
            _policy_reason = ""
        else:
            # Auto-route by policy; direct remains the default
            effective_executor = _policy_executor
        if effective_executor == "opencode":
            generator: ExecutionGenerator | OpenCodeExecutor = OpenCodeExecutor()
        else:
            generator = ExecutionGenerator()
    except (ValueError, RuntimeError) as exc:
        raise click.ClickException(str(exc))

    opencode_mode = (effective_executor == "opencode")
    routing_decision: RoutingDecision | None = route(task) if not opencode_mode else None
    _use_stages = (
        not opencode_mode
        and routing_decision is not None
        and should_use_stages(routing_decision.category)
    )
    stage_runs: list[StageRun] = []

    if opencode_mode:
        workspace = Path(tempfile.mkdtemp())
        _copy_cwd_to_workspace(workspace)
        if detail == "full":
            click.echo(f"\n  [workspace] {workspace}")

    spinner = _Spinner()
    click.echo("")
    if detail != "default" and _policy_reason:
        click.echo(f"  [executor] {effective_executor} - {_policy_reason}")
    if routing_decision is not None:
        if detail != "default":
            click.echo(f"  [routing] {routing_decision.model} - {routing_decision.rationale}")
        hint = _pre_run_cost_hint(routing_decision.model, task)
        if hint:
            click.echo(f"  Cost estimate: {hint}")
    _routed_model = routing_decision.model if routing_decision else None

    # --- Stage-based execution (direct mode, security/complex tasks) ----------
    _impl_task = task          # may be augmented with a plan
    result = None

    if _use_stages:
        stages = split_task(task)
        for _stage in stages:
            _stage_t0 = time.time()

            if _stage.stage_type == "planning":
                click.echo("  Planning...")
                spinner.start("Planning...")
                try:
                    _plan_text, _plan_usage = run_planning_stage(generator.client, task)
                    _impl_task = (
                        f"Task: {task}\n\nImplementation plan:\n{_plan_text}"
                        "\n\nExecute the task following the plan above."
                    )
                    stage_runs.append(StageRun(
                        stage=_stage,
                        model=route_stage(_stage),
                        duration=time.time() - _stage_t0,
                        cost=_plan_usage.estimated_cost if _plan_usage else None,
                        summary="Implementation plan produced",
                    ))
                except (AuthError, RateLimitError, OpenRouterError):
                    _impl_task = task   # planning failed — fall back to plain task
                    if detail == "full":
                        click.echo("  [stages] planning call failed, continuing without plan")
                finally:
                    spinner.stop()

            elif _stage.stage_type == "implementation":
                _stage_model = _routed_model or route_stage(_stage)
                _impl_reason = _RATIONALE_SHORT.get(
                    routing_decision.rationale if routing_decision else "",
                    routing_decision.category if routing_decision else "implementation",
                )
                click.echo(f"  Using {_model_label(_stage_model)} ({_impl_reason})...")
                spinner.start("Executing...")
                try:
                    result = generator.generate(_impl_task, model=_stage_model)
                except RuntimeError as exc:
                    raise click.ClickException(str(exc))
                except AuthError:
                    raise click.ClickException(
                        "Authentication failed. Check that OPENROUTER_API_KEY is valid."
                    )
                except RateLimitError:
                    raise click.ClickException("Rate limit exceeded. Wait a moment then try again.")
                except OpenRouterError as exc:
                    raise click.ClickException(f"API error: {exc}")
                finally:
                    spinner.stop()
                stage_runs.append(StageRun(
                    stage=_stage,
                    model=_stage_model,
                    duration=time.time() - _stage_t0,
                    cost=result.usage.estimated_cost if result.usage else None,
                    summary=result.summary,
                ))

    # --- Single-stage execution (simple tasks, opencode, stages not triggered) -
    if result is None:
        if routing_decision is not None:
            _single_reason = _RATIONALE_SHORT.get(routing_decision.rationale, routing_decision.category)
            click.echo(f"  Using {_model_label(_routed_model)} ({_single_reason})...")
        elif opencode_mode:
            click.echo("  Running with OpenCode...")
        else:
            click.echo("  Executing...")
        spinner.start("Executing...")
        try:
            if opencode_mode:
                result = generator.generate(task, workspace=workspace)
            else:
                result = generator.generate(task, model=_routed_model)
        except RuntimeError as exc:
            raise click.ClickException(str(exc))
        except AuthError:
            raise click.ClickException(
                "Authentication failed. Check that OPENROUTER_API_KEY is valid."
            )
        except RateLimitError:
            raise click.ClickException("Rate limit exceeded. Wait a moment then try again.")
        except OpenRouterError as exc:
            raise click.ClickException(f"API error: {exc}")
        finally:
            spinner.stop()

    usage = result.usage
    # When stages ran, fold the planning cost into the reported total
    if stage_runs and usage is not None:
        total_stage_cost = sum(sr.cost for sr in stage_runs if sr.cost is not None)
        if usage.estimated_cost is not None:
            usage.estimated_cost = total_stage_cost
        else:
            usage.estimated_cost = total_stage_cost or None
    final_files = result.files

    click.echo("\nDone")
    click.echo(result.summary)

    _model_line = _build_model_line(routing_decision, stage_runs)
    if _model_line:
        click.echo(f"\n{_model_line}")

    # Stages — shown before file count so the reader sees what ran (--more only)
    if detail != "default" and stage_runs:
        click.echo("\nStages")
        for sr in stage_runs:
            _sr_cost = f"${sr.cost:.4f}" if sr.cost is not None else "-"
            click.echo(f"  {sr.stage.stage_type.capitalize()} ({sr.model}): {sr.duration:.1f}s, {_sr_cost}")

    if result.files:
        _fc = sum(1 for f in result.files if f.change_type == "create")
        _fu = sum(1 for f in result.files if f.change_type == "update")
        _fd = sum(1 for f in result.files if f.change_type == "delete")
        _counts = ", ".join(p for p in [
            f"{_fc} created" if _fc else "",
            f"{_fu} updated" if _fu else "",
            f"{_fd} deleted" if _fd else "",
        ] if p)
        click.echo(f"\nFiles: {_counts}")
        if detail != "default":
            for f in result.files:
                _desc = f" - {f.summary}" if f.summary else ""
                click.echo(f"  {f.path}{_desc}")

    if detail != "default" and result.notes:
        _notes = [_truncate_note(n) for n in result.notes if n][:3]
        if _notes:
            click.echo("\nNotes")
            for note in _notes:
                click.echo(f"  {note}")

    if dry_run:
        if _should_shrink(result.files, no_shrink):
            _print_shrunk(result.files, result.summary)
        else:
            _print_dry_run(result.files)
        _print_summary(start, generator, retry_triggered, final_files, usage=usage, detail=detail, model=_routed_model, stage_runs=stage_runs)
        try:
            _log_run(start, task, generator, retry_triggered, final_files,
                     verification_attempted=False, verification_passed=None,
                     workspace=None, usage=usage, model=_routed_model,
                     summary=result.summary, notes=result.notes,
                     stage_runs=stage_runs, routing_decision=routing_decision)
        except Exception as exc:
            click.echo(f"  [log] warning: {exc}")
        return

    if verify and not write:
        raise click.ClickException("--verify requires --write.")

    if write:
        if not opencode_mode:
            workspace = Path(tempfile.mkdtemp())
            if detail == "full":
                click.echo(f"\n  [workspace] {workspace}")
            _write_files(result.files, workspace)
        # OpenCode: workspace already created and populated before generate().

    if write and verify:
        click.echo("")
        code = _run_verification(workspace, detail=detail)
        # Escalation loop: try each model in chain until verification passes.
        # Direct mode uses ESCALATION_CHAIN (sonnet → opus).
        # OpenCode mode uses a single fixer-model retry (no chain).
        _escalation = ESCALATION_CHAIN if not opencode_mode else [generator.fixer_model]
        _last_attempt = result
        for _esc_model in _escalation:
            if code == 0:
                break
            retry_triggered = True
            _, verify_output = _run_verification(workspace, capture=True)
            retry_prompt = _build_retry_prompt(task, _last_attempt, verify_output)
            if detail == "full":
                snippet = retry_prompt[:300] + ("..." if len(retry_prompt) > 300 else "")
                click.echo(f"\n  [retry prompt] {snippet}")
                if verify_output.strip():
                    click.echo(f"\n  [verify output] {verify_output.strip()[:300]}")
            _esc_label = _esc_model.split("/")[-1]
            click.echo(f"  Retrying with {_esc_label}...")
            spinner.start(f"Retrying with {_esc_label}...")
            try:
                if opencode_mode:
                    _last_attempt = generator.generate(
                        retry_prompt, model=_esc_model, workspace=workspace
                    )
                else:
                    _last_attempt = generator.generate(retry_prompt, model=_esc_model)
            except RuntimeError as exc:
                raise click.ClickException(str(exc))
            except AuthError:
                raise click.ClickException(
                    "Authentication failed. Check that OPENROUTER_API_KEY is valid."
                )
            except RateLimitError:
                raise click.ClickException("Rate limit exceeded. Wait a moment then try again.")
            except OpenRouterError as exc:
                raise click.ClickException(f"API error: {exc}")
            finally:
                spinner.stop()
            retry_usage = _last_attempt.usage
            final_files = _last_attempt.files
            if not opencode_mode:
                _write_files(_last_attempt.files, workspace)
            code = _run_verification(workspace, label=f"[retry/{_esc_label}]", detail=detail)
        verification_passed = code == 0
        if code != 0:
            _print_summary(start, generator, retry_triggered, final_files,
                           usage=usage, retry_usage=retry_usage, detail=detail, model=_routed_model, stage_runs=stage_runs)
            try:
                _log_run(start, task, generator, retry_triggered, final_files,
                         verification_attempted=True, verification_passed=False,
                         workspace=workspace, usage=usage, retry_usage=retry_usage, model=_routed_model,
                         summary=result.summary, notes=result.notes,
                         stage_runs=stage_runs, routing_decision=routing_decision)
            except Exception as exc:
                click.echo(f"  [log] warning: {exc}")
            sys.exit(code)

    _print_summary(start, generator, retry_triggered, final_files,
                   usage=usage, retry_usage=retry_usage, detail=detail, model=_routed_model, stage_runs=stage_runs)
    try:
        _log_run(start, task, generator, retry_triggered, final_files,
                 verification_attempted=(write and verify),
                 verification_passed=verification_passed,
                 workspace=workspace, usage=usage, retry_usage=retry_usage, model=_routed_model,
                 summary=result.summary, notes=result.notes,
                 stage_runs=stage_runs, routing_decision=routing_decision)
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
    model: str | None = None,
    summary: str = "",
    notes: list | None = None,
    stage_runs=None,
    routing_decision=None,
) -> None:
    entry: dict = {
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat().replace("+00:00", "Z"),
        "task": task,
        "execution_model": model or generator.model,
        "retry_triggered": retry_triggered,
        "duration_seconds": round(time.time() - start, 2),
        "files_created": sum(1 for f in files if f.change_type == "create"),
        "files_updated": sum(1 for f in files if f.change_type == "update"),
        "files_deleted": sum(1 for f in files if f.change_type == "delete"),
        "verification_attempted": verification_attempted,
        "verification_passed": verification_passed,
        "workspace_path": str(workspace) if workspace else None,
        "summary": summary,
        "files_detail": [
            {"path": f.path, "change_type": f.change_type, "summary": f.summary or ""}
            for f in files
        ],
    }
    if notes:
        entry["notes"] = [n.split("\n")[0][:300] for n in notes if n][:5]
    if stage_runs:
        entry["stage_runs"] = [
            {
                "stage_type": sr.stage.stage_type,
                "model": sr.model,
                "duration": round(sr.duration, 2),
                "cost": sr.cost,
            }
            for sr in stage_runs
        ]
    if routing_decision is not None:
        entry["routing_model"] = routing_decision.model
        entry["routing_rationale"] = routing_decision.rationale
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
    model: str | None = None,
    stage_runs: list[StageRun] | None = None,
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
    if not stage_runs:
        click.echo(f"\nModel: {model or generator.model}")
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
    click.echo(f"\nTime: {elapsed:.1f}s   Cost: {cost_str}")


# ---------------------------------------------------------------------------
# Execution policy
# ---------------------------------------------------------------------------

_OPENCODE_SIGNALS: list[tuple[list[str], str]] = [
    (["all files", "every file", "multiple files", "many files"], "multi-file scope"),
    (["throughout the", "across the codebase", "across all files"], "multi-file scope"),
    # refactor / architecture / codebase are now handled in direct mode via minimax
]


def _suggest_executor(task: str) -> tuple[str, str]:
    """Return ``(executor, reason)`` recommendation for *task*.

    - Defaults to ``"direct"`` for short, focused tasks.
    - Returns ``"opencode"`` for large, multi-file, or ambiguous tasks.
    """
    lower = task.lower()
    if len(task.split()) > 60:
        return "opencode", "large or complex task"
    for keywords, label in _OPENCODE_SIGNALS:
        if any(kw in lower for kw in keywords):
            return "opencode", label
    return "direct", "focused task"


# ---------------------------------------------------------------------------
# Pre-run cost hint
# ---------------------------------------------------------------------------

_SYSTEM_OVERHEAD_TOKENS = 200   # rough execution system-prompt size


def _pre_run_cost_hint(model: str, task: str) -> str | None:
    """Return a rough pre-run cost range string, or None if pricing is unknown.

    Uses the prompt token estimate + a 0.5×–3× completion range heuristic.
    Clearly labelled as an estimate — not shown as exact.
    """
    if model not in MODEL_PRICING:
        return None
    prompt_tokens = _SYSTEM_OVERHEAD_TOKENS + len(task) // 4
    cost_low  = compute_cost(model, prompt_tokens, max(100, prompt_tokens // 2))
    cost_high = compute_cost(model, prompt_tokens, prompt_tokens * 3)
    if cost_low is None or cost_high is None:
        return None
    return f"~${cost_low:.4f}-${cost_high:.4f}"



_MODEL_SHORT: dict[str, str] = {
    "deepseek/deepseek-v3.2":          "DeepSeek V3.2",
    "z-ai/glm-5.1":                    "GLM-5.1",
    "anthropic/claude-sonnet-4.6":     "Sonnet 4.6",
    "anthropic/claude-opus-4.7":       "Opus 4.7",
    "moonshotai/kimi-k2.5":            "Kimi K2.5",
    "minimax/m2.7":                    "MiniMax M2.7",
}

_RATIONALE_SHORT: dict[str, str] = {
    "security-sensitive code requires careful reasoning": "security-sensitive",
    "UI or visual task routed to multimodal specialist":  "UI / visual",
    "multi-file or long-horizon task":                    "complex task",
    "low-risk boilerplate task":                          "boilerplate",
    "standard feature implementation":                    "standard coding",
}


def _model_label(model: str) -> str:
    return _MODEL_SHORT.get(model, model.split("/")[-1])


def _build_model_line(
    routing_decision: "RoutingDecision | None",
    stage_runs: "list[StageRun]",
) -> str | None:
    """Return a single 'Model: ...' or 'Models: ...' line for default output."""
    if stage_runs:
        seen: dict[str, list[str]] = {}
        for sr in stage_runs:
            label = _model_label(sr.model)
            seen.setdefault(label, []).append(sr.stage.stage_type)
        parts = []
        for label, types in seen.items():
            reason = " + ".join(types)
            parts.append(f"{label} ({reason})")
        prefix = "Model" if len(seen) == 1 else "Models"
        return f"{prefix}: {', '.join(parts)}"

    if routing_decision is not None:
        label = _model_label(routing_decision.model)
        reason = _RATIONALE_SHORT.get(routing_decision.rationale, "")
        suffix = f" ({reason})" if reason else ""
        return f"Model: {label}{suffix}"

    return None


def _truncate_note(text: str, limit: int = 200) -> str:
    line = text.split("\n")[0]
    if len(line) <= limit:
        return line
    cut = line.rfind(" ", 0, limit)
    return line[:cut] + "..." if cut > 0 else line[:limit] + "..."


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
        f"\n  Output condensed: {len(files)} file(s), ~{total_chars} chars."
        " Use --no-shrink to see full content."
    )
    click.echo(f"\n{result_summary}\n")
    click.echo("Files")
    for f in files[:5]:
        click.echo(f"  {f.path} ({f.change_type}) - {f.summary}")
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
            click.echo("(no content, file will be deleted)")
        else:
            click.echo(f.content)
        click.echo("")


class _Spinner:
    """Animated terminal spinner with phase label and elapsed time."""

    _FRAMES_UNICODE = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
    _FRAMES_ASCII   = ["-", "\\", "|", "/"]

    def __init__(self) -> None:
        self.phase: str = ""
        self._t0: float = 0.0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self, phase: str) -> None:
        self.phase = phase
        self._t0 = time.time()
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        if self._thread is None:
            return
        self._stop.set()
        self._thread.join()
        self._thread = None
        sys.stdout.write("\r" + " " * 72 + "\r")
        sys.stdout.flush()

    def _run(self) -> None:
        encoding = getattr(sys.stdout, "encoding", "utf-8") or "utf-8"
        frames = self._FRAMES_UNICODE if encoding.lower().replace("-", "") in ("utf8", "utf16", "utf32") else self._FRAMES_ASCII
        i = 0
        while not self._stop.wait(0.1):
            elapsed = time.time() - self._t0
            frame = frames[i % len(frames)]
            line = f"  {frame} {self.phase}  {elapsed:.1f}s"
            sys.stdout.write(f"\r{line:<70}")
            sys.stdout.flush()
            i += 1


def _copy_cwd_to_workspace(workspace: Path) -> None:
    """Copy the current working directory into *workspace*, excluding noise."""
    shutil.copytree(
        Path.cwd(),
        workspace,
        dirs_exist_ok=True,
        ignore=shutil.ignore_patterns(
            # version control
            ".git",
            # dependency trees
            "node_modules", ".venv", "venv", "env", ".tox",
            # build / cache artefacts
            "__pycache__", "*.pyc", "*.egg-info",
            ".pytest_cache", ".mypy_cache",
            "dist", "build", "coverage", ".next",
            # openshard-internal files OpenCode doesn't need
            ".openshard", ".claude",
            # binary assets — waste context tokens for no gain
            "*.png", "*.jpg", "*.jpeg", "*.gif", "*.svg", "*.ico",
            "*.pdf", "*.zip", "*.tar", "*.gz",
        ),
    )


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

    if not capture:
        if detail == "full":
            click.echo(f"  {label} running: {' '.join(cmd)}")
        else:
            click.echo("  Verifying...")
    proc = subprocess.run(
        cmd,
        cwd=cwd,
        stdout=subprocess.PIPE if capture else None,
        stderr=subprocess.STDOUT if capture else None,
        text=capture,
        **({"encoding": "utf-8", "errors": "replace"} if capture else {}),
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
            click.echo("  Verified")
        else:
            click.echo(f"  Verification failed (exit {proc.returncode})")
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
        click.echo("  Complex task - fixer would benefit from a stronger model.")
    else:
        click.echo("  Straightforward task - default fixer model should be sufficient.")


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

def _render_log_entry(entry: dict, detail: str) -> None:
    """Render a stored run log entry at the requested detail level."""
    ts = entry.get("timestamp", "").rstrip("Z").replace("T", " ").split(".")[0]
    task = entry.get("task", "")
    summary = entry.get("summary", "")
    stage_runs_data: list[dict] = entry.get("stage_runs", [])
    routing_model: str = entry.get("routing_model", "")
    routing_rationale: str = entry.get("routing_rationale", "")
    files_detail: list[dict] = entry.get("files_detail", [])
    notes: list[str] = entry.get("notes", [])

    click.echo(f"\nTask: {task}")
    if ts:
        click.echo(f"At: {ts} UTC")
    click.echo("\nDone")
    if summary:
        click.echo(summary)

    # Model line — always shown
    if stage_runs_data:
        seen: dict[str, list[str]] = {}
        for sr in stage_runs_data:
            lbl = _model_label(sr["model"])
            seen.setdefault(lbl, []).append(sr["stage_type"])
        parts = [f"{lbl} ({' + '.join(types)})" for lbl, types in seen.items()]
        prefix = "Model" if len(seen) == 1 else "Models"
        click.echo(f"\n{prefix}: {', '.join(parts)}")
    elif routing_model:
        lbl = _model_label(routing_model)
        reason = _RATIONALE_SHORT.get(routing_rationale, "")
        suffix = f" ({reason})" if reason else ""
        click.echo(f"\nModel: {lbl}{suffix}")

    # Stages (--more / --full)
    if detail != "default" and stage_runs_data:
        click.echo("\nStages")
        for sr in stage_runs_data:
            cost_s = f"${sr['cost']:.4f}" if sr.get("cost") is not None else "-"
            click.echo(f"  {sr['stage_type'].capitalize()} ({sr['model']}): {sr['duration']:.1f}s, {cost_s}")

    # Files
    fc = entry.get("files_created", 0)
    fu = entry.get("files_updated", 0)
    fd = entry.get("files_deleted", 0)
    if fc or fu or fd:
        counts = ", ".join(p for p in [
            f"{fc} created" if fc else "",
            f"{fu} updated" if fu else "",
            f"{fd} deleted" if fd else "",
        ] if p)
        click.echo(f"\nFiles: {counts}")
        if detail != "default":
            for f in files_detail:
                desc = f" - {f['summary']}" if f.get("summary") else ""
                click.echo(f"  {f['path']}{desc}")

    # Notes (--more / --full)
    if detail != "default" and notes:
        _notes = [_truncate_note(n) for n in notes if n][:3]
        if _notes:
            click.echo("\nNotes")
            for note in _notes:
                click.echo(f"  {note}")

    # Token / model detail (--more / --full)
    if detail != "default":
        full_model = entry.get("execution_model", "")
        if full_model and not stage_runs_data:
            click.echo(f"\nModel: {full_model}")
        pt = entry.get("prompt_tokens", 0)
        ct = entry.get("completion_tokens", 0)
        tt = entry.get("total_tokens", 0)
        if tt:
            click.echo(f"Tokens: {pt} prompt / {ct} completion / {tt} total")
        if entry.get("retry_triggered"):
            click.echo("Retried: yes")
        if detail == "full":
            vp = entry.get("verification_passed")
            if vp is not None:
                click.echo(f"Verification: {'passed' if vp else 'failed'}")
            ws = entry.get("workspace_path")
            if ws:
                click.echo(f"Workspace: {ws}")

    duration = entry.get("duration_seconds", 0)
    cost = entry.get("estimated_cost")
    cost_str = f"${cost:.4f}" if cost is not None else "-"
    click.echo(f"\nTime: {duration:.1f}s   Cost: {cost_str}")


@cli.command()
@click.option("--more", is_flag=True, default=False, help="Show file list, model names, and token breakdown.")
@click.option("--full", is_flag=True, default=False, help="Show all stored details including verification and workspace.")
def last(more: bool, full: bool):
    """Show details of the most recent run without rerunning it."""
    detail = "full" if full else ("more" if more else "default")
    log_path = Path.cwd() / _LOG_PATH
    if not log_path.exists():
        click.echo("No run history found. Run a task first with 'openshard run'.")
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
    _render_log_entry(entries[-1], detail)


if __name__ == "__main__":
    cli()