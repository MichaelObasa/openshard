import json
import sys
import tempfile
import time
from pathlib import Path

import click

from openshard.config.settings import load_config, get_api_key, get_anthropic_api_key, get_openai_api_key
from openshard.planning.generator import PlanGenerator
from openshard.providers.base import ProviderAuthError, ProviderError, ProviderRateLimitError
from openshard.run.pipeline import (
    RunPipeline,
    _LOG_PATH,
    _suggest_executor as _suggest_executor,
    _pre_run_cost_hint as _pre_run_cost_hint,
    _parse_cost_hint as _parse_cost_hint,
    _log_run as _log_run,
    _copy_cwd_to_workspace,
    _build_retry_prompt as _build_retry_prompt,
    _write_files as _write_files,
    _detect_command as _detect_command,
    _run_verification as _run_verification,
    _run_verification_plan as _run_verification_plan,
    confirm_or_abort as confirm_or_abort,
)
from openshard.cli.run_output import (
    _Spinner as _Spinner,
    _print_summary as _print_summary,
    _render_repo_summary as _render_repo_summary,
    _format_model_slug as _format_model_slug,
    _model_label,
    _build_routing_line as _build_routing_line,
    _exec_message as _exec_message,
    _build_model_line as _build_model_line,
    _truncate_note,
    _should_shrink as _should_shrink,
    _print_shrunk as _print_shrunk,
    _print_dry_run as _print_dry_run,
    _RATIONALE_SHORT,
)
from openshard.evals.registry import load_eval_tasks
from openshard.evals.runner import append_eval_result, run_eval_task


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
    except ProviderAuthError:
        raise click.ClickException(
            "Authentication failed. Check that your provider API key is valid."
        )
    except ProviderRateLimitError:
        raise click.ClickException("Rate limit exceeded. Wait a moment then try again.")
    except ProviderError as exc:
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
    "--workflow",
    type=click.Choice(
        ["auto", "direct", "staged", "native", "opencode", "claude-code", "codex"],
        case_sensitive=False,
    ),
    default=None,
    help=(
        "Execution workflow: auto (default, policy-driven), direct (single-pass API call), "
        "staged (planning then implementation), native (native agent - not yet available), "
        "opencode (OpenCode CLI), claude-code (not yet available), codex (not yet available)."
    ),
)
@click.option(
    "--profile",
    type=click.Choice(["native_light", "native_deep", "native_swarm"], case_sensitive=False),
    default=None,
    help="Execution profile: native_light (fast/simple), native_deep (thorough/complex), native_swarm (experimental, never auto-selected).",
)
@click.option(
    "--executor",
    type=click.Choice(["direct", "opencode"], case_sensitive=False),
    default=None,
    help="[DEPRECATED] Use --workflow instead. Execution backend: direct or opencode.",
)
@click.option("--plan", "plan_flag", is_flag=True, default=False, help="Show execution plan and prompt for approval before running.")
@click.option(
    "--approval",
    type=click.Choice(["auto", "smart", "ask"], case_sensitive=False),
    default=None,
    help="Override config approval_mode for this run: auto (silent), smart (prompt on risk), ask (always prompt).",
)
@click.option(
    "--provider",
    type=click.Choice(["openrouter", "anthropic", "openai"], case_sensitive=False),
    default=None,
    help="API provider: openrouter (default), anthropic (requires ANTHROPIC_API_KEY), or openai (requires OPENAI_API_KEY).",
)
@click.option("--history-scoring", "history_scoring", is_flag=True, default=False, help="Apply run-history bonuses/penalties to model scoring (opt-in).")
@click.option("--eval-scoring", "eval_scoring", is_flag=True, default=False, help="Apply eval-run bonuses/penalties to model scoring (opt-in).")
def run(task: str, write: bool, verify: bool, dry_run: bool, more: bool, full: bool, no_shrink: bool, workflow: str | None, profile: str | None, executor: str | None, plan_flag: bool, approval: str | None, provider: str | None, history_scoring: bool, eval_scoring: bool):
    """Execute TASK and return a structured result."""
    try:
        config = load_config()
    except (ValueError, RuntimeError) as exc:
        raise click.ClickException(str(exc))
    detail = "full" if full else ("more" if more else "default")
    pipeline = RunPipeline(
        config,
        write=write,
        verify=verify,
        dry_run=dry_run,
        no_shrink=no_shrink,
        workflow=workflow,
        profile=profile,
        executor=executor,
        plan_flag=plan_flag,
        approval=approval,
        provider=provider,
        history_scoring=history_scoring,
        eval_scoring=eval_scoring,
        detail=detail,
    )
    result = pipeline.run(task)
    if result.exit_code != 0:
        sys.exit(result.exit_code)


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
    except ProviderAuthError:
        raise click.ClickException(
            "Authentication failed. Check that your provider API key is valid."
        )
    except ProviderRateLimitError:
        raise click.ClickException("Rate limit exceeded. Wait a moment then try again.")
    except ProviderError as exc:
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


@cli.group(invoke_without_command=True)
@click.pass_context
@click.option(
    "--provider",
    type=click.Choice(["openrouter", "anthropic", "openai"], case_sensitive=False),
    default=None,
    help="Filter by provider. Omit to show all.",
)
@click.option("--refresh", is_flag=True, default=False, help="Fetch models from provider API and update cache.")
def models(ctx: click.Context, provider: str | None, refresh: bool):
    """List cached models and capabilities, or run a subcommand."""
    if ctx.invoked_subcommand is not None:
        return
    if provider is None:
        from openshard.providers.manager import ProviderManager
        manager = ProviderManager()
        if not manager.providers:
            click.echo("No providers configured. Set at least one API key.")
            return
        inventory = manager.get_inventory(refresh=refresh)
        if refresh:
            from collections import Counter
            counts = Counter(e.provider for e in inventory.models)
            for pname, count in sorted(counts.items()):
                click.echo(f"  {pname}: {count} models cached")
            return
        header = f"{'Provider':<12}  {'Model ID':<50}  {'Context':>9}  {'MaxOut':>7}  {'Vision':<6}  {'Tools':<5}"
        click.echo(header)
        click.echo("-" * len(header))
        for entry in inventory.models:
            m = entry.model
            ctx2 = str(m.context_window or "-")
            out = str(m.max_output_tokens or "-")
            vis = "yes" if m.supports_vision else "no"
            tls = "yes" if m.supports_tools else "no"
            click.echo(f"{entry.provider:<12}  {m.id:<50}  {ctx2:>9}  {out:>7}  {vis:<6}  {tls:<5}")
        return

    from openshard.providers.cache import (
        CACHE_TTL_HOURS,
        build_cache_entry,
        is_stale,
        load_cache,
        save_cache,
    )

    _all_providers = ["openrouter", "anthropic", "openai"]
    targets = [provider.lower()] if provider else _all_providers

    if refresh:
        cache = load_cache() or {"cached_at": 0.0, "models": {}}
        for pname in targets:
            try:
                if pname == "openrouter":
                    from openshard.providers.openrouter import OpenRouterClient
                    client = OpenRouterClient(get_api_key())
                elif pname == "anthropic":
                    from openshard.providers.anthropic import AnthropicProvider
                    client = AnthropicProvider(get_anthropic_api_key())
                else:
                    from openshard.providers.openai import OpenAIProvider
                    client = OpenAIProvider(get_openai_api_key())
                model_list = client.list_models()
                cache["models"].update(build_cache_entry(pname, model_list))
                click.echo(f"  {pname}: {len(model_list)} models cached")
            except (ValueError, ProviderAuthError, ProviderError) as exc:
                click.echo(f"  {pname}: skipped ({exc})")
        cache["cached_at"] = time.time()
        save_cache(cache)
        return

    cache = load_cache()
    if cache is None:
        click.echo("No model cache found. Run 'openshard models --refresh' to populate it.")
        return
    if is_stale(cache.get("cached_at", 0.0)):
        click.echo(f"Cache is older than {CACHE_TTL_HOURS}h. Run 'openshard models --refresh' to update.")
        return

    header = f"{'Provider':<12}  {'Model ID':<50}  {'Context':>9}  {'MaxOut':>7}  {'Vision':<6}  {'Tools':<5}"
    click.echo(header)
    click.echo("-" * len(header))
    for pname in targets:
        for m in cache.get("models", {}).get(pname, []):
            ctx2 = str(m.get("context_window") or "-")
            out = str(m.get("max_output_tokens") or "-")
            vis = "yes" if m.get("supports_vision") else "no"
            tls = "yes" if m.get("supports_tools") else "no"
            click.echo(f"{pname:<12}  {m['id']:<50}  {ctx2:>9}  {out:>7}  {vis:<6}  {tls:<5}")


@models.command("stats")
def models_stats():
    """Show per-model performance stats from run history."""
    from openshard.history.metrics import compute_model_stats, load_runs

    runs = load_runs()
    if not runs:
        log_path = Path.cwd() / _LOG_PATH
        if not log_path.exists():
            click.echo("No run history found. Run 'openshard run' to get started.")
        else:
            click.echo("No runs recorded yet.")
        return

    stats = compute_model_stats(runs)
    if not stats:
        click.echo("No model data in run history.")
        return

    total_runs = len(runs)
    model_count = len(stats)
    click.echo(f"[model stats]  {model_count} model{'s' if model_count != 1 else ''}  (from {total_runs} run{'s' if total_runs != 1 else ''})\n")

    col_model = 48
    header = (
        f"  {'model':<{col_model}}  {'runs':>5}  {'avg cost':>9}  {'avg dur':>8}  {'pass rate':>9}  {'retry':>6}"
    )
    click.echo(header)
    click.echo("  " + "-" * (len(header) - 2))

    for model_id, s in list(stats.items())[:10]:
        runs_n = s["runs_count"]
        avg_cost = f"${s['avg_cost']:.4f}" if s["avg_cost"] is not None else "-"
        avg_dur = f"{s['avg_duration']:.1f}s" if s["avg_duration"] is not None else "-"
        pass_rate = f"{s['verification_pass_rate']:.0%}" if s["verification_pass_rate"] is not None else "-"
        retry = f"{s['retry_rate']:.0%}"
        mid = model_id if len(model_id) <= col_model else model_id[: col_model - 1] + "..."
        click.echo(f"  {mid:<{col_model}}  {runs_n:>5}  {avg_cost:>9}  {avg_dur:>8}  {pass_rate:>9}  {retry:>6}")


@cli.group(invoke_without_command=True)
@click.pass_context
def profiles(ctx: click.Context):
    """Profile management commands."""
    if ctx.invoked_subcommand is None:
        click.echo(ctx.get_help())


@profiles.command("stats")
def profiles_stats():
    """Show per-profile performance stats from run history."""
    from openshard.history.metrics import compute_profile_stats, load_runs

    runs = load_runs()
    if not runs:
        log_path = Path.cwd() / _LOG_PATH
        if not log_path.exists():
            click.echo("No run history found. Run 'openshard run' to get started.")
        else:
            click.echo("No runs recorded yet.")
        return

    stats = compute_profile_stats(runs)
    profiled_runs = sum(s["runs_count"] for s in stats.values())

    click.echo(f"[profile stats]  {profiled_runs} run{'s' if profiled_runs != 1 else ''} with profile data\n")

    col = 16
    header = f"  {'profile':<{col}}  {'runs':>5}  {'avg cost':>9}  {'avg dur':>8}  {'pass rate':>9}  {'retry':>6}"
    click.echo(header)
    click.echo("  " + "-" * (len(header) - 2))

    for profile, s in stats.items():
        n = s["runs_count"]
        avg_cost = f"${s['avg_cost']:.4f}" if s["avg_cost"] is not None else "-"
        avg_dur = f"{s['avg_duration']:.1f}s" if s["avg_duration"] is not None else "-"
        pass_rate = f"{s['verification_pass_rate']:.0%}" if s["verification_pass_rate"] is not None else "-"
        retry = f"{s['retry_rate']:.0%}" if s["retry_rate"] is not None else "-"
        click.echo(f"  {profile:<{col}}  {n:>5}  {avg_cost:>9}  {avg_dur:>8}  {pass_rate:>9}  {retry:>6}")


@cli.group(invoke_without_command=True)
@click.pass_context
def skills(ctx: click.Context):
    """Skills management commands."""
    if ctx.invoked_subcommand is None:
        click.echo(ctx.get_help())


@skills.command("stats")
def skills_stats():
    """Show per-skill performance stats from run history."""
    from openshard.history.metrics import compute_skill_stats, load_runs

    runs = load_runs()
    if not runs:
        log_path = Path.cwd() / _LOG_PATH
        if not log_path.exists():
            click.echo("No run history found. Run 'openshard run' to get started.")
        else:
            click.echo("No runs recorded yet.")
        return

    stats = compute_skill_stats(runs)
    if not stats:
        click.echo("No skill data in run history.")
        return

    skill_runs = sum(s["runs_count"] for s in stats.values())
    click.echo(f"[skill stats]  {len(stats)} skill{'s' if len(stats) != 1 else ''}  (from {skill_runs} skill-matched run{'s' if skill_runs != 1 else ''})\n")

    col = 28
    header = f"  {'skill':<{col}}  {'runs':>5}  {'avg cost':>9}  {'avg dur':>8}  {'pass rate':>9}  {'retry':>6}"
    click.echo(header)
    click.echo("  " + "-" * (len(header) - 2))

    for slug, s in stats.items():
        n = s["runs_count"]
        avg_cost = f"${s['avg_cost']:.4f}" if s["avg_cost"] is not None else "-"
        avg_dur = f"{s['avg_duration']:.1f}s" if s["avg_duration"] is not None else "-"
        pass_rate = f"{s['verification_pass_rate']:.0%}" if s["verification_pass_rate"] is not None else "-"
        retry = f"{s['retry_rate']:.0%}" if s["retry_rate"] is not None else "-"
        label = slug if len(slug) <= col else slug[: col - 1] + "..."
        click.echo(f"  {label:<{col}}  {n:>5}  {avg_cost:>9}  {avg_dur:>8}  {pass_rate:>9}  {retry:>6}")


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

    from openshard.history.metrics import compute_profile_stats, compute_skill_stats
    _profile_stats = compute_profile_stats(entries)
    _profiled = sum(s["runs_count"] for s in _profile_stats.values())
    if _profiled > 0:
        click.echo("\n  profiles:")
        for _pname, _ps in _profile_stats.items():
            _n = _ps["runs_count"]
            _pc = f"${_ps['avg_cost']:.4f}" if _ps["avg_cost"] is not None else "-"
            _pd = f"{_ps['avg_duration']:.1f}s" if _ps["avg_duration"] is not None else "-"
            _pp = f"{_ps['verification_pass_rate']:.0%}" if _ps["verification_pass_rate"] is not None else "-"
            _pr = f"{_ps['retry_rate']:.0%}" if _ps["retry_rate"] is not None else "-"
            _run_label = "run" if _n == 1 else "runs"
            click.echo(f"    {_pname:<14}  {_n} {_run_label}  pass: {_pp}  retry: {_pr}  {_pc}  {_pd}")

    _skill_stats = compute_skill_stats(entries)
    if _skill_stats:
        click.echo("\n  skills (top 5):")
        for _slug, _ss in list(_skill_stats.items())[:5]:
            _sn = _ss["runs_count"]
            _sc = f"${_ss['avg_cost']:.4f}" if _ss["avg_cost"] is not None else "-"
            _sd = f"{_ss['avg_duration']:.1f}s" if _ss["avg_duration"] is not None else "-"
            _sp = f"{_ss['verification_pass_rate']:.0%}" if _ss["verification_pass_rate"] is not None else "-"
            _sr = f"{_ss['retry_rate']:.0%}" if _ss["retry_rate"] is not None else "-"
            _run_label = "run" if _sn == 1 else "runs"
            click.echo(f"    {_slug:<24}  {_sn} {_run_label}  pass: {_sp}  retry: {_sr}  {_sc}  {_sd}")

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

def _compute_metrics(entries: list[dict]) -> dict:
    from collections import Counter

    costs = [e["estimated_cost"] for e in entries if e.get("estimated_cost") is not None]
    total_cost = sum(costs) if costs else None
    avg_cost = total_cost / len(costs) if costs else None

    durations = [e["duration_seconds"] for e in entries if "duration_seconds" in e]
    avg_duration = sum(durations) / len(durations) if durations else 0.0

    models = Counter(e["execution_model"] for e in entries if e.get("execution_model"))
    categories = Counter(e["routing_category"] for e in entries if e.get("routing_category"))

    v_passed = sum(1 for e in entries if e.get("verification_passed") is True)
    v_failed = sum(1 for e in entries if e.get("verification_passed") is False)
    v_unknown = len(entries) - v_passed - v_failed

    timestamps = [e["timestamp"] for e in entries if e.get("timestamp")]
    most_recent = max(timestamps) if timestamps else None
    if most_recent:
        most_recent = most_recent.rstrip("Z").replace("T", " ").split(".")[0] + " UTC"

    return {
        "total_runs": len(entries),
        "total_cost": total_cost,
        "avg_cost": avg_cost,
        "avg_duration": avg_duration,
        "most_recent": most_recent,
        "models": dict(models.most_common()),
        "categories": dict(categories.most_common()),
        "verification": {"passed": v_passed, "failed": v_failed, "unknown": v_unknown},
    }


@cli.command()
def metrics():
    """Show aggregated metrics from run history."""
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

    m = _compute_metrics(entries)

    def _cost(v: float | None) -> str:
        return f"${v:.4f}" if v is not None else "-"

    click.echo("\n[metrics]")
    click.echo(f"  runs:             {m['total_runs']}")
    click.echo(f"  total cost:       {_cost(m['total_cost'])}")
    click.echo(f"  avg cost/run:     {_cost(m['avg_cost'])}")
    click.echo(f"  avg duration:     {m['avg_duration']:.1f}s")
    click.echo(f"  most recent:      {m['most_recent'] or '-'}")

    if m["models"]:
        click.echo("\n  models")
        for model_id, count in m["models"].items():
            label = _model_label(model_id)
            click.echo(f"    {label:<26} {count}")

    if m["categories"]:
        click.echo("\n  categories")
        for cat, count in m["categories"].items():
            click.echo(f"    {cat:<26} {count}")

    v = m["verification"]
    click.echo("\n  verification")
    click.echo(f"    passed           {v['passed']}")
    click.echo(f"    failed           {v['failed']}")
    click.echo(f"    not attempted    {v['unknown']}")


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
            click.echo(f"  {sr['stage_type'].capitalize()} ({_model_label(sr['model'])}): {sr['duration']:.1f}s, {cost_s}")

    # Routing (--more / --full)
    if detail != "default" and "routing_category" in entry:
        click.echo(f"\n  [routing] category: {entry['routing_category']}")
        if entry.get("routing_used_fallback"):
            click.echo(f"  [routing] candidates: {entry.get('routing_candidate_count')} -> fallback (keyword routing)")
        elif entry.get("routing_selected_model"):
            _prov = entry.get("routing_selected_provider")
            _prov_suffix = f" ({_prov})" if _prov else ""
            click.echo(f"  [routing] candidates: {entry.get('routing_candidate_count')} -> {_model_label(entry['routing_selected_model'])}{_prov_suffix}")

    # Execution profile (--more / --full)
    if detail != "default" and entry.get("execution_profile"):
        _profile = entry["execution_profile"]
        _reason = entry.get("execution_profile_reason", "")
        _profile_line = f"[profile] {_profile}"
        if _reason:
            _profile_line += f" - {_reason}"
        click.echo(f"  {_profile_line}")

    # Verification plan (--more / --full)
    if detail != "default" and "verification_plan" in entry:
        for _vc in entry["verification_plan"]:
            _argv_str = " ".join(_vc["argv"])
            click.echo(f"  [verification] {_vc['name']} {_vc['safety']} {_vc['source']} {_argv_str}")

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


@cli.group(invoke_without_command=True)
@click.pass_context
def eval(ctx: click.Context):
    """Eval harness commands."""
    if ctx.invoked_subcommand is None:
        click.echo(ctx.get_help())


@eval.command("list")
@click.option("--suite", default="basic", show_default=True, help="Eval suite to list.")
def eval_list(suite: str):
    """List available eval tasks."""
    from openshard.evals.registry import load_eval_tasks

    try:
        tasks = load_eval_tasks(suite)
    except FileNotFoundError as exc:
        raise click.ClickException(str(exc))

    col_id = 24
    col_title = 40
    header = f"  {'id':<{col_id}}  {'title':<{col_title}}  category"
    click.echo(header)
    click.echo("  " + "-" * (len(header) - 2))
    for task in tasks:
        tid = task.id if len(task.id) <= col_id else task.id[: col_id - 1] + "..."
        ttitle = task.title if len(task.title) <= col_title else task.title[: col_title - 1] + "..."
        click.echo(f"  {tid:<{col_id}}  {ttitle:<{col_title}}  {task.category}")


@eval.command("validate")
@click.option("--suite", default="basic", show_default=True, help="Eval suite to validate.")
def eval_validate(suite: str):
    """Validate that all eval tasks in a suite load correctly."""
    from openshard.evals.registry import load_eval_tasks

    try:
        tasks = load_eval_tasks(suite)
    except FileNotFoundError as exc:
        raise click.ClickException(str(exc))

    errors: list[str] = []
    for task in tasks:
        if not task.prompt.strip():
            errors.append(f"{task.id}: prompt.txt is empty")

    if errors:
        for err in errors:
            click.echo(f"FAIL  {err}")
        raise click.ClickException(f"{len(errors)} task(s) failed validation.")

    click.echo(f"OK  {len(tasks)} task(s) passed validation for suite '{suite}'.")


@eval.command("run")
@click.option("--suite", default="basic", show_default=True, help="Eval suite to run.")
@click.option(
    "--model",
    default="anthropic/claude-haiku-4-5-20251001",
    show_default=True,
    help="Model for execution.",
)
def eval_run(suite: str, model: str):
    """Run all eval tasks in a suite and report pass/fail."""
    import tempfile

    from openshard.evals.registry import load_eval_tasks
    from openshard.evals.runner import append_eval_result, run_eval_task
    from openshard.run.pipeline import _copy_cwd_to_workspace

    try:
        tasks = load_eval_tasks(suite)
    except FileNotFoundError as exc:
        raise click.ClickException(str(exc))

    log_path = Path(".openshard") / "eval-runs.jsonl"
    passed_count = failed_count = 0

    for task in tasks:
        with tempfile.TemporaryDirectory(prefix=f"openshard-eval-{task.id}-") as tmp:
            workspace = Path(tmp)
            _copy_cwd_to_workspace(workspace)
            result = run_eval_task(task, model=model, suite=suite, workspace_root=workspace)
        append_eval_result(result, log_path)
        status = "PASS" if result.passed else "FAIL"
        extra = f"  ({result.error})" if result.error else ""
        click.echo(f"{status}  {task.id:<28}  {result.duration_seconds:.1f}s{extra}")
        if result.passed:
            passed_count += 1
        else:
            failed_count += 1

    click.echo(f"\n{passed_count} passed, {failed_count} failed  — results in {log_path}")
    if failed_count:
        raise SystemExit(1)


@eval.command("compare")
@click.option("--suite", default="basic", show_default=True, help="Eval suite to run.")
@click.option("--models", required=True, help="Comma-separated list of model slugs.")
def eval_compare(suite: str, models: str):
    """Run an eval suite across multiple models and print a comparison summary."""
    model_list = [m.strip() for m in models.split(",") if m.strip()]
    if not model_list:
        raise click.ClickException("--models must contain at least one model slug.")

    try:
        tasks = load_eval_tasks(suite)
    except FileNotFoundError as exc:
        raise click.ClickException(str(exc))

    log_path = Path(".openshard") / "eval-runs.jsonl"
    click.echo(f"\nRunning suite '{suite}' ({len(tasks)} tasks) across {len(model_list)} model(s)...\n")

    results_by_model: dict[str, list] = {}

    for model in model_list:
        click.echo(f"[{model}]")
        model_results = []
        for task in tasks:
            with tempfile.TemporaryDirectory(prefix=f"openshard-eval-{task.id}-") as tmp:
                workspace = Path(tmp)
                _copy_cwd_to_workspace(workspace)
                result = run_eval_task(task, model=model, suite=suite, workspace_root=workspace)
            append_eval_result(result, log_path)
            status = "PASS" if result.passed else "FAIL"
            extra = f"  ({result.error})" if result.error else ""
            click.echo(f"  {status}  {task.id:<28}  {result.duration_seconds:.1f}s{extra}")
            model_results.append(result)
        results_by_model[model] = model_results
        click.echo()

    all_results = [r for rs in results_by_model.values() for r in rs]
    show_tokens = any(r.total_tokens > 0 for r in all_results)

    header = f"{'Model':<44} {'Runs':>4}  {'Pass':>4}  {'Fail':>4}  {'Rate':>7}  {'Avg Dur':>8}"
    if show_tokens:
        header += f"  {'Avg Tok':>8}"
    header += f"  {'Unsafe':>6}"
    click.echo(header)

    any_failures = False
    for model, model_results in results_by_model.items():
        runs = len(model_results)
        passed_count = sum(1 for r in model_results if r.passed)
        failed_count = runs - passed_count
        if failed_count:
            any_failures = True
        rate = passed_count / runs * 100 if runs else 0.0
        avg_dur = sum(r.duration_seconds for r in model_results) / runs if runs else 0.0
        unsafe = sum(len(r.unsafe_files) for r in model_results)
        row = f"{model:<44} {runs:>4}  {passed_count:>4}  {failed_count:>4}  {rate:>6.1f}%  {avg_dur:>7.1f}s"
        if show_tokens:
            avg_tok = sum(r.total_tokens for r in model_results) / runs if runs else 0.0
            row += f"  {avg_tok:>8.0f}"
        row += f"  {unsafe:>6}"
        click.echo(row)

    click.echo(f"\nResults appended to {log_path}")

    if len(model_list) > 1:
        from openshard.evals.stats import rank_models
        ranking = rank_models(results_by_model)
        click.echo("\n[ranking]")
        if all(entry.pass_count == 0 for entry in ranking):
            click.echo("  no passing runs; cost-per-pass ranking unavailable")
        else:
            for entry in ranking:
                pass_pct = f"{entry.pass_rate:.0%}"
                cost_str = f"${entry.cost_per_pass:.4f}" if entry.cost_per_pass is not None else "-"
                tok_str = f"{entry.avg_tokens:,.0f}" if entry.avg_tokens is not None else "-"
                click.echo(
                    f"  {entry.rank}. {entry.model:<44}"
                    f"  pass: {pass_pct:>4}"
                    f"  cost/pass: {cost_str:<10}"
                    f"  avg tokens: {tok_str:>7}"
                    f"  avg dur: {entry.avg_duration:.1f}s"
                    f"  unsafe: {entry.unsafe_count}"
                )

    if any_failures:
        raise SystemExit(1)


@eval.command("report")
@click.option("--suite", default=None, help="Filter by eval suite name.")
@click.option("--model", default=None, help="Filter by model slug.")
def eval_report(suite: str | None, model: str | None):
    """Summarize eval run results from .openshard/eval-runs.jsonl."""
    import collections

    log_path = Path(".openshard") / "eval-runs.jsonl"

    if not log_path.exists():
        click.echo("No eval runs found. Run `openshard eval run` first.")
        return

    records: list[dict] = []
    for raw in log_path.read_text(encoding="utf-8").splitlines():
        raw = raw.strip()
        if not raw:
            continue
        try:
            records.append(json.loads(raw))
        except json.JSONDecodeError:
            continue

    if suite:
        records = [r for r in records if r.get("suite") == suite]
    if model:
        records = [r for r in records if r.get("model") == model]

    if not records:
        click.echo("No records match the given filters.")
        return

    total = len(records)
    passed = sum(1 for r in records if r.get("passed"))
    failed = total - passed
    pass_rate = passed / total * 100
    avg_dur = sum(r.get("duration_seconds", 0) for r in records) / total

    tokens = [r.get("total_tokens", 0) for r in records]
    show_tokens = any(t > 0 for t in tokens)
    avg_tok = sum(tokens) / total if show_tokens else 0

    unsafe_count = sum(len(r.get("unsafe_files", [])) for r in records)

    parts = []
    if suite:
        parts.append(f"suite={suite}")
    if model:
        parts.append(f"model={model}")
    header_suffix = "  " + "  ".join(parts) if parts else ""

    click.echo(f"Eval Report{header_suffix}")
    click.echo("-" * 42)
    click.echo(f"  Total runs:    {total}")
    click.echo(f"  Passed:        {passed}  ({pass_rate:.1f}%)")
    click.echo(f"  Failed:        {failed}")
    click.echo(f"  Avg duration:  {avg_dur:.1f}s")
    if show_tokens:
        click.echo(f"  Avg tokens:    {avg_tok:.0f}")
    click.echo(f"  Unsafe files:  {unsafe_count}")

    by_task: dict[str, list[dict]] = collections.defaultdict(list)
    for r in records:
        by_task[r.get("task_id", "unknown")].append(r)

    click.echo("\nBy task:")
    for tid in sorted(by_task):
        grp = by_task[tid]
        p = sum(1 for r in grp if r.get("passed"))
        n = len(grp)
        pct = p / n * 100
        click.echo(f"  {tid:<28}  {p}/{n}  ({pct:.1f}%)")

    by_model: dict[str, list[dict]] = collections.defaultdict(list)
    for r in records:
        by_model[r.get("model", "unknown")].append(r)

    click.echo("\nBy model:")
    for mdl in sorted(by_model):
        grp = by_model[mdl]
        p = sum(1 for r in grp if r.get("passed"))
        n = len(grp)
        pct = p / n * 100
        click.echo(f"  {mdl:<44}  {p}/{n}  ({pct:.1f}%)")


@eval.command("stats")
@click.option("--suite", default=None, help="Filter by suite name.")
@click.option("--model", default=None, help="Filter by model slug.")
@click.option("--task", default=None, help="Filter by task_id.")
@click.option("--by-category", is_flag=True, default=False, help="Group results by task category.")
def eval_stats(suite: str | None, model: str | None, task: str | None, by_category: bool):
    """Show grouped pass/fail stats from .openshard/eval-runs.jsonl."""
    from openshard.evals.stats import EVAL_RUNS_PATH, compute_category_stats, compute_eval_stats, load_eval_runs

    records = load_eval_runs(Path.cwd() / EVAL_RUNS_PATH)

    if by_category:
        from openshard.evals.registry import build_category_map

        suites = {r["suite"] for r in records if r.get("suite")}
        if suite:
            suites = {suite} & suites
        category_maps: dict[str, dict[str, str]] = {}
        for s in suites:
            try:
                category_maps[s] = build_category_map(s)
            except FileNotFoundError:
                category_maps[s] = {}

        rows = compute_category_stats(records, category_maps, suite=suite, model=model, task=task)

        if not rows:
            click.echo("No eval results found.")
            return

        click.echo("\n[eval stats --by-category]")
        parts = []
        if suite:
            parts.append(f"suite: {suite}")
        if model:
            parts.append(f"model: {model}")
        if task:
            parts.append(f"task: {task}")
        if parts:
            click.echo("  " + "  ".join(parts))

        header = (
            f"  {'suite':<10}  {'category':<12}  {'model':<44}"
            f"  {'runs':>5}  {'pass':>5}  {'fail':>5}  {'pass%':>6}"
            f"  {'avg_dur':>8}  {'avg_tokens':>11}  {'cost/pass':>10}  {'unsafe':>7}"
        )
        click.echo(f"\n{header}")
        for s in rows:
            tok = f"{s.avg_total_tokens:,.0f}" if s.avg_total_tokens is not None else "-"
            cpp = f"${s.cost_per_pass:.4f}" if s.cost_per_pass is not None else "-"
            click.echo(
                f"  {s.suite:<10}  {s.category:<12}  {s.model:<44}"
                f"  {s.run_count:>5}  {s.pass_count:>5}  {s.fail_count:>5}  {s.pass_rate:>5.0%}"
                f"  {s.avg_duration:>7.1f}s  {tok:>11}  {cpp:>10}  {s.unsafe_file_count:>7}"
            )

        total_runs = sum(s.run_count for s in rows)
        total_pass = sum(s.pass_count for s in rows)
        total_fail = sum(s.fail_count for s in rows)
        overall_rate = total_pass / total_runs if total_runs else 0.0
        click.echo(f"\n  total: {total_runs} runs  pass: {total_pass}  fail: {total_fail}  pass rate: {overall_rate:.0%}")
        return

    rows = compute_eval_stats(records, suite=suite, model=model, task=task)

    if not rows:
        click.echo("No eval results found.")
        return

    click.echo("\n[eval stats]")
    parts = []
    if suite:
        parts.append(f"suite: {suite}")
    if model:
        parts.append(f"model: {model}")
    if task:
        parts.append(f"task: {task}")
    if parts:
        click.echo("  " + "  ".join(parts))

    header = (
        f"  {'suite':<10}  {'model':<44}  {'task_id':<24}"
        f"  {'runs':>5}  {'pass':>5}  {'fail':>5}  {'pass%':>6}"
        f"  {'avg_dur':>8}  {'avg_tokens':>11}  {'unsafe':>7}"
    )
    click.echo(f"\n{header}")
    for s in rows:
        tok = f"{s.avg_total_tokens:,.0f}" if s.avg_total_tokens is not None else "-"
        click.echo(
            f"  {s.suite:<10}  {s.model:<44}  {s.task_id:<24}"
            f"  {s.run_count:>5}  {s.pass_count:>5}  {s.fail_count:>5}  {s.pass_rate:>5.0%}"
            f"  {s.avg_duration:>7.1f}s  {tok:>11}  {s.unsafe_file_count:>7}"
        )

    total_runs = sum(s.run_count for s in rows)
    total_pass = sum(s.pass_count for s in rows)
    total_fail = sum(s.fail_count for s in rows)
    overall_rate = total_pass / total_runs if total_runs else 0.0
    click.echo(f"\n  total: {total_runs} runs  pass: {total_pass}  fail: {total_fail}  pass rate: {overall_rate:.0%}")


if __name__ == "__main__":
    cli()