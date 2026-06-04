import datetime
import json
import os
import sys
import tempfile
import time
from pathlib import Path

import click

from openshard import __version__
from openshard.config.settings import (
    load_config,
    load_config_safe,
    save_config,
    find_config_path,
    get_onboarding,
    get_api_key,
    get_anthropic_api_key,
    get_openai_api_key,
)
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
    _render_repo_map as _render_repo_map,
    _render_repo_plan as _render_repo_plan,
    _format_model_slug as _format_model_slug,
    _model_label,
    _profile_display_label,
    _build_routing_line as _build_routing_line,
    _exec_message as _exec_message,
    _build_model_line as _build_model_line,
    _truncate_note,
    _should_shrink as _should_shrink,
    _print_shrunk as _print_shrunk,
    _print_dry_run as _print_dry_run,
    _render_native_inspection,
    _native_meta_from_entry,
    _RATIONALE_SHORT,
    _PUBLIC_MODE_LABEL,
)
from openshard.evals.registry import load_eval_tasks
from openshard.evals.runner import append_eval_result, run_eval_task
from openshard.history.sandbox_apply_receipts import (
    SandboxApplyReceipt,
    log_sandbox_apply_receipt,
    recent_sandbox_apply_receipts,
)


@click.group(invoke_without_command=True)
@click.version_option(version=__version__, prog_name="openshard")
@click.pass_context
def cli(ctx: click.Context):
    """OpenShard - intelligent task routing and execution."""
    if ctx.invoked_subcommand is None:
        from openshard.cli.ui.home import render_home

        render_home()


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
@click.option(
    "--native-backend",
    "native_backend",
    type=click.Choice(["builtin", "deepagents"], case_sensitive=False),
    default=None,
    help="Native workflow backend (default: builtin). deepagents is experimental/stub only. Ignored for non-native workflows.",
)
@click.option(
    "--experimental-deepagents-run",
    "experimental_deepagents_run",
    is_flag=True,
    default=False,
    help="Invoke a minimal read-only DeepAgents agent as a proof step. Requires --native-backend deepagents. No write or shell tools are provided.",
)
@click.option(
    "--experimental-tier-dispatch",
    "experimental_tier_dispatch",
    is_flag=True,
    default=False,
    help="[Experimental] Resolve routing tier names to model IDs and use them during execution. Recorded in run log; shown at --more/--full.",
)
@click.option(
    "--native-loop",
    "native_loop",
    type=click.Choice(["experimental"], case_sensitive=False),
    default=None,
    help="Enable experimental bounded native loop. Requires --workflow native. Runs additional deterministic read-only tool steps before generation.",
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
@click.option("--feedback-scoring", "feedback_scoring", is_flag=True, default=False, help="Apply developer-feedback bonuses/penalties to model scoring (opt-in).")
@click.option(
    "--model-policy",
    "model_policy",
    type=click.Choice(["auto", "cheapest-safe", "frontier-heavy", "open-source-only", "local-only", "custom"], case_sensitive=False),
    default=None,
    help="Model selection policy mode (metadata-only v1): auto, cheapest-safe, frontier-heavy, open-source-only, local-only, custom.",
)
@click.option(
    "--candidates",
    default=1,
    type=click.IntRange(1, 3),
    help="Run multiple native candidate agents and select the best verified result (1–3, native --write only).",
)
def run(task: str, write: bool, verify: bool, dry_run: bool, more: bool, full: bool, no_shrink: bool, workflow: str | None, profile: str | None, executor: str | None, native_backend: str | None, experimental_deepagents_run: bool, experimental_tier_dispatch: bool, native_loop: str | None, plan_flag: bool, approval: str | None, provider: str | None, history_scoring: bool, eval_scoring: bool, feedback_scoring: bool, model_policy: str | None, candidates: int):
    """Execute TASK and return a structured result."""
    if native_loop is not None and workflow != "native":
        raise click.UsageError("--native-loop experimental requires --workflow native")
    try:
        config = load_config()
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
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
        feedback_scoring=feedback_scoring,
        detail=detail,
        native_backend=native_backend,
        experimental_deepagents_run=experimental_deepagents_run,
        experimental_tier_dispatch=experimental_tier_dispatch,
        native_loop=native_loop,
        model_policy=model_policy,
        candidates=candidates,
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


# ---------------------------------------------------------------------------
# Registry inspection helpers and commands.
# ---------------------------------------------------------------------------

_COL_ID = 45
_COL_PROV = 12
_COL_TIER = 17
_COL_COST = 10
_COL_EXP_R = 13


def _print_registry_table(entries: list) -> None:
    header = (
        f"  {'ID':<{_COL_ID}}  {'Provider':<{_COL_PROV}}  {'Tier':<{_COL_TIER}}  Experimental"
    )
    click.echo(header)
    click.echo("  " + "-" * (len(header) - 2))
    for entry in entries:
        exp = "yes" if entry.experimental else "no"
        mid = entry.id if len(entry.id) <= _COL_ID else entry.id[: _COL_ID - 1] + "..."
        click.echo(f"  {mid:<{_COL_ID}}  {entry.provider:<{_COL_PROV}}  {entry.tier:<{_COL_TIER}}  {exp}")


@models.command("list")
def models_list():
    """List all registered models."""
    from openshard.models.registry import all_models

    _print_registry_table(all_models())


@models.command("show")
@click.argument("model_id")
def models_show(model_id: str):
    """Show full details for a model."""
    from openshard.models.registry import get_model

    entry = get_model(model_id)
    if entry is None:
        raise click.ClickException(f"Model not found: {model_id}")
    w = 12
    roles_str = ", ".join(entry.roles) if entry.roles else "-"
    ctx_str = str(entry.context_length) if entry.context_length is not None else "-"
    click.echo(f"  {'Model':<{w}}  {entry.display_name}")
    click.echo(f"  {'ID':<{w}}  {entry.id}")
    click.echo(f"  {'Provider':<{w}}  {entry.provider}")
    click.echo(f"  {'Tier':<{w}}  {entry.tier}")
    click.echo(f"  {'Roles':<{w}}  {roles_str}")
    click.echo(f"  {'Experimental':<{w}}  {'yes' if entry.experimental else 'no'}")
    click.echo(f"  {'Context':<{w}}  {ctx_str}")
    click.echo(f"  {'Tools':<{w}}  {'yes' if entry.supports_tools else 'no'}")
    click.echo(f"  {'Structured':<{w}}  {'yes' if entry.supports_structured_outputs else 'no'}")
    click.echo(f"  {'Reasoning':<{w}}  {'yes' if entry.supports_reasoning else 'no'}")
    click.echo(f"  {'Multimodal':<{w}}  {'yes' if entry.supports_multimodal else 'no'}")
    click.echo(f"  {'Latency':<{w}}  {entry.latency_class}")
    click.echo(f"  {'Cost class':<{w}}  {entry.cost_class}")


@models.command("role")
@click.argument("role")
def models_role(role: str):
    """List models that have the given role."""
    from openshard.models.registry import models_by_role

    entries = models_by_role(role)
    if not entries:
        click.echo(f"No models found for role: {role}")
        return
    _print_registry_table(entries)


_VALID_CAPABILITIES = ("tools", "structured_outputs", "reasoning", "multimodal")


@models.command("capabilities")
@click.argument("capability")
def models_capabilities(capability: str):
    """List models that support a capability (tools, structured_outputs, reasoning, multimodal)."""
    from openshard.models.registry import models_by_capability

    if capability not in _VALID_CAPABILITIES:
        click.echo(f"Unknown capability: {capability}")
        click.echo(f"Accepted: {', '.join(_VALID_CAPABILITIES)}")
        return
    entries = models_by_capability(capability)
    if not entries:
        click.echo(f"No models found for capability: {capability}")
        return
    _print_registry_table(entries)


@models.command("experimental")
def models_experimental():
    """List all experimental models."""
    from openshard.models.registry import all_models

    entries = [e for e in all_models() if e.experimental]
    if not entries:
        click.echo("No experimental models registered.")
        return
    _print_registry_table(entries)


_VALID_COST_CLASSES = ("free", "tiny", "cheap", "mid", "expensive")


@models.command("recommend")
@click.option("--role", default=None, help="Filter/score by role name.")
@click.option(
    "--risk",
    type=click.Choice(["low", "medium", "high"], case_sensitive=False),
    default=None,
    help="Risk level hint.",
)
@click.option(
    "--capability",
    "capabilities",
    multiple=True,
    help="Required capability (tools, structured_outputs, reasoning, multimodal). Repeatable.",
)
@click.option(
    "--max-cost",
    "max_cost_class",
    type=click.Choice(_VALID_COST_CLASSES, case_sensitive=False),
    default=None,
    help="Maximum cost class.",
)
@click.option("--include-experimental", "include_experimental", is_flag=True, default=False)
@click.option("--limit", default=5, show_default=True, type=click.IntRange(min=1))
def models_recommend(
    role: str | None,
    risk: str | None,
    capabilities: tuple[str, ...],
    max_cost_class: str | None,
    include_experimental: bool,
    limit: int,
) -> None:
    """Recommend advisory models for a use case (does not change routing)."""
    from openshard.models.advisory import recommend_models
    from openshard.models.registry import CAPABILITY_NAMES

    unknown = [c for c in capabilities if c not in CAPABILITY_NAMES]
    if unknown:
        click.echo(
            f"No results: unknown capability '{unknown[0]}'. "
            f"Accepted: {', '.join(sorted(CAPABILITY_NAMES))}"
        )
        return

    results = recommend_models(
        role=role,
        risk=risk,
        required_capabilities=tuple(capabilities),
        max_cost_class=max_cost_class,
        include_experimental=include_experimental,
        limit=limit,
    )

    if not results:
        click.echo("No models matched the given criteria.")
        return

    header = (
        f"  {'ID':<{_COL_ID}}  {'Tier':<{_COL_TIER}}  "
        f"{'Cost':<{_COL_COST}}  {'Experimental':<{_COL_EXP_R}}  Reasons"
    )
    click.echo(header)
    click.echo("  " + "-" * (len(header) - 2))
    for advisory in results:
        m = advisory.model
        mid = m.id if len(m.id) <= _COL_ID else m.id[: _COL_ID - 1] + "..."
        exp = "yes" if m.experimental else "no"
        reasons_str = "; ".join(advisory.reasons) if advisory.reasons else "-"
        click.echo(
            f"  {mid:<{_COL_ID}}  {m.tier:<{_COL_TIER}}  "
            f"{m.cost_class:<{_COL_COST}}  {exp:<{_COL_EXP_R}}  {reasons_str}"
        )


@models.command("mode")
@click.argument("mode")
def models_mode(mode: str) -> None:
    """Show the advisory model policy for a mode (ask, plan, run)."""
    from openshard.models.mode_policy import model_policy_for_mode
    from openshard.models.registry import display_name_for

    _LABEL = 12
    mode = mode.strip().lower()
    if mode not in ("ask", "plan", "run"):
        raise click.ClickException(
            f"Unknown mode '{mode}'. Supported modes: ask, plan, run."
        )
    policy = model_policy_for_mode(mode)
    if policy is None:
        click.echo(f"{'Mode':<{_LABEL}}run")
        click.echo(
            f"{'Status':<{_LABEL}}Run routing remains controlled by existing routing policy"
        )
        return
    default_display = display_name_for(policy.default_model_id, policy.default_model_id)
    fallback_display = ", ".join(
        display_name_for(fid, fid) for fid in policy.fallback_model_ids
    )
    if mode == "ask":
        status = "Advisory only - Ask Mode is still local deterministic"
    else:
        status = "Advisory only - Plan Mode is still local deterministic"
    click.echo(f"{'Mode':<{_LABEL}}{mode}")
    click.echo(f"{'Default':<{_LABEL}}{default_display}")
    click.echo(f"{'Fallbacks':<{_LABEL}}{fallback_display}")
    click.echo(f"{'Status':<{_LABEL}}{status}")


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


@cli.group(invoke_without_command=True)
@click.pass_context
def advisory(ctx: click.Context):
    """Executor advisory ranking commands (does not change routing defaults)."""
    if ctx.invoked_subcommand is None:
        click.echo(ctx.get_help())


@advisory.command("rank")
@click.option("--task", required=True, help="Task description to rank executors for.")
@click.option(
    "--risk",
    type=click.Choice(["low", "medium", "high"], case_sensitive=False),
    default="low",
    show_default=True,
    help="Risk level hint.",
)
@click.option(
    "--category",
    default="standard",
    show_default=True,
    help="Task category (security, complex, boilerplate, visual, standard).",
)
@click.option(
    "--opencode-preference",
    "opencode_preference",
    is_flag=True,
    default=False,
    help="Signal that OpenCode is preferred when available.",
)
def advisory_rank(
    task: str,
    risk: str,
    category: str,
    opencode_preference: bool,
) -> None:
    """Rank executor paths for a task. Advisory only — does not change execution defaults."""
    from openshard.execution.opencode_adapter import detect_opencode
    from openshard.routing.engine import is_readonly_task
    from openshard.routing.executor_advisory import rank_executors, render_executor_advisory

    availability = detect_opencode()
    read_only = is_readonly_task(task)

    result = rank_executors(
        task,
        category=category,
        risk_level=risk,
        read_only=read_only,
        opencode_available=availability.available,
        opencode_preference=opencode_preference,
        risky_paths=[],
    )

    for line in render_executor_advisory(result):
        click.echo(line)


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


def _render_log_entry(entry: dict, detail: str, index: int | None = None) -> None:
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
    _stored_findings = entry.get("findings") or []
    if _stored_findings:
        from openshard.cli.run_output import render_review_tldr_memo
        from openshard.history.shard_contract import ShardFinding
        _sf_list = [
            ShardFinding(severity=f.get("severity", "Note"), message=f.get("message", ""))
            for f in _stored_findings
            if isinstance(f, dict)
        ]
        _review_files = [fd.get("path", "") for fd in files_detail if fd.get("path")]
        click.echo("\nReview complete")
        click.echo(render_review_tldr_memo(_sf_list, _review_files))
    elif entry.get("is_review_task") or (
        entry.get("review_domain") and entry.get("review_domain") not in ("generic_review", "terraform_iac")
    ):
        from openshard.cli.run_output import render_review_fallback_memo
        from openshard.review.domain_files import no_files_message
        _review_files = [fd.get("path", "") for fd in files_detail if fd.get("path")]
        _domain_files_log = entry.get("domain_files") or []
        # Prefer domain-discovered evidence files over the changed-files list.
        # Changed files are always empty for read-only reviews; domain_files
        # contains the files that were actually inspected for the review domain.
        _evidence_files = _domain_files_log or _review_files
        click.echo("\nReview complete")
        if not _evidence_files:
            _no_msg = no_files_message(entry.get("review_domain", ""))
            if _no_msg:
                click.echo(_no_msg)
            else:
                click.echo(render_review_fallback_memo(
                    [],
                    include_diagnostic=(detail in ("more", "full")),
                ))
        else:
            click.echo(render_review_fallback_memo(
                _evidence_files,
                include_diagnostic=(detail in ("more", "full")),
                is_evidence=True,
            ))
    else:
        click.echo("\nDone")
        if summary:
            click.echo(summary)

    # Model line — only in default mode; receipt shows model in --more / --full.
    # Prefer routing_selected_model (scored routing) over routing_model (keyword
    # routing) so the displayed model matches what the receipt shows.
    _is_ro = routing_rationale == "read-only analysis"
    _routing_selected: str = entry.get("routing_selected_model", "")
    if detail == "default":
        if stage_runs_data:
            seen: dict[str, list[str]] = {}
            for sr in stage_runs_data:
                lbl = _model_label(sr["model"])
                stype = "analysis" if (_is_ro and sr["stage_type"] == "implementation") else sr["stage_type"]
                seen.setdefault(lbl, []).append(stype)
            parts = [f"{lbl} ({' + '.join(types)})" for lbl, types in seen.items()]
            prefix = "Model" if len(seen) == 1 else "Models"
            click.echo(f"\n{prefix}: {', '.join(parts)}")
        elif _routing_selected or routing_model:
            _display_model = _routing_selected or routing_model
            from openshard.history.shard_contract import display_model_name as _sc_display_model_name
            lbl = _sc_display_model_name(_display_model)
            if _routing_selected and _routing_selected != routing_model:
                # Scored routing overrode the keyword pick — mirror receipt format.
                click.echo(f"\nModel: Auto → {lbl}")
            else:
                reason = _RATIONALE_SHORT.get(routing_rationale, "")
                suffix = f" ({reason})" if reason else ""
                click.echo(f"\nModel: {lbl}{suffix}")
        _df_compact = entry.get("developer_feedback")
        if _df_compact:
            click.echo(f"Feedback: {_df_compact.get('outcome', '')}")

    # Shard receipt (--more / --full) — shown near top before diagnostic blocks
    if detail != "default":
        from openshard.history.shard_contract import (
            build_shard_receipt,
            render_compact_shard_receipt,
            render_full_shard_receipt,
        )
        _shard = build_shard_receipt(entry, index)
        click.echo("")
        click.echo(render_compact_shard_receipt(_shard))
        click.echo("")
        click.echo(render_full_shard_receipt(_shard, detail=detail))

    # Proof summary (--more only) - compact OSN proof presence check
    if detail == "more":
        from openshard.cli.run_output import _print_proof_summary
        _proof_nm = _native_meta_from_entry(entry)
        if _proof_nm is not None:
            _print_proof_summary(_proof_nm)

    # Stages (--full only)
    if detail == "full" and stage_runs_data:
        click.echo("\nStages")
        for sr in stage_runs_data:
            cost_s = f"${sr['cost']:.4f}" if sr.get("cost") is not None else "-"
            _stage_label = "Analysis" if (_is_ro and sr['stage_type'] == "implementation") else sr['stage_type'].capitalize()
            click.echo(f"  {_stage_label} ({_model_label(sr['model'])}): {sr['duration']:.1f}s, {cost_s}")

    # Task type (--full only, read-only only)
    if detail == "full" and _is_ro:
        click.echo("\nTask type")
        click.echo("  Read-only analysis")
        click.echo("  Reason: The prompt asks a question, so file changes are blocked.")

    # Routing (--full only)
    if detail == "full" and "routing_category" in entry:
        click.echo("\n  Routing")
        click.echo(f"    Category: {entry['routing_category']}")
        if entry.get("routing_used_fallback"):
            click.echo("    Initial candidate: fallback (keyword routing)")
        elif entry.get("routing_selected_model"):
            _prov = entry.get("routing_selected_provider")
            _prov_suffix = f" ({_prov})" if _prov else ""
            click.echo(f"    Initial candidate: {_model_label(entry['routing_selected_model'])}{_prov_suffix}")
        _tdr_check = entry.get("tier_dispatch_receipt")
        if _tdr_check and _tdr_check.get("enabled") and _tdr_check.get("applied"):
            click.echo("    Note: tier dispatch changed the work model shown below.")
        if entry.get("routing_feedback_scoring_used"):
            _fb_adjs = entry.get("routing_feedback_adjustments") or {}
            _fb_rsns = entry.get("routing_feedback_reasons") or {}
            if _fb_adjs:
                click.echo("    Feedback scoring:")
                for _fm, _fa in _fb_adjs.items():
                    _rsn = _fb_rsns.get(_fm, "")
                    _rsn_str = f" ({_rsn})" if _rsn else ""
                    click.echo(f"      {_model_label(_fm)}: {_fa:+.2f}{_rsn_str}")
            else:
                click.echo("    Feedback scoring: enabled (no adjustment)")

    # Execution profile (--full only)
    if detail == "full" and entry.get("execution_profile"):
        _profile = entry["execution_profile"]
        _reason = entry.get("execution_profile_reason", "")
        click.echo("  Execution")
        _is_ro = entry.get("routing_rationale") == "read-only analysis"
        click.echo(f"    Mode: {_profile_display_label(_profile, is_readonly=_is_ro)}")
        if _reason:
            click.echo(f"    Reason: {_reason}")

    # Form factor (--full only)
    if detail == "full" and "form_factor" in entry:
        _ff = entry["form_factor"]
        _ff_pub = _PUBLIC_MODE_LABEL.get(_ff["public_mode"], _ff["public_mode"].title())
        click.echo("\n  Form factor")
        click.echo(f"    Public mode:  {_ff_pub}")
        click.echo(f"    Internal:     {_ff['internal_form_factor']}")
        click.echo(f"    Reason:       {_ff['reason']}")
        click.echo(f"    Confidence:   {_ff['confidence']}")
        click.echo(f"    Risk:         {_ff['risk_level']}")
        if _ff.get("context_quality"):
            click.echo(f"    Context:      {_ff['context_quality']}")
        for _w in _ff.get("warnings", []):
            click.echo(f"    Warning:      {_w}")

    # Verification plan (--full only)
    if detail == "full" and "verification_plan" in entry:
        _vp_raw = entry["verification_plan"]
        # Native runs store verification_plan as {"commands": [...]} (asdict of VerificationPlan).
        # Non-native runs store it as a plain list of command dicts.
        if isinstance(_vp_raw, dict):
            _vp_cmds = _vp_raw.get("commands") or []
        elif isinstance(_vp_raw, list):
            _vp_cmds = _vp_raw
        else:
            _vp_cmds = []
        for _vc in _vp_cmds:
            if not isinstance(_vc, dict):
                continue
            _argv_str = " ".join(_vc.get("argv") or [])
            click.echo("  Verification")
            click.echo(f"    Name:    {_vc.get('name', '')}")
            click.echo(f"    Safety:  {_vc.get('safety', '')}")
            click.echo(f"    Source:  {_vc.get('source', '')}")
            click.echo(f"    Command: {_argv_str}")

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
        if detail == "full":
            for f in files_detail:
                desc = f" - {f['summary']}" if f.get("summary") else ""
                click.echo(f"  {f['path']}{desc}")

    # Notes (--full only)
    if detail == "full" and notes:
        _notes = [_truncate_note(n) for n in notes if n][:3]
        if _notes:
            click.echo("\nNotes")
            for note in _notes:
                click.echo(f"  {note}")

    # Developer feedback (--more / --full)
    _feedback = entry.get("feedback")
    if detail in ("more", "full") and _feedback:
        click.echo("\nDeveloper feedback")
        _action = _feedback.get("action")
        _reason = _feedback.get("correction_reason")
        _rating = _feedback.get("rating") or ""
        _fb_note = _feedback.get("note", "")
        if _action:
            click.echo(f"  Action: {_action}")
        if _reason:
            click.echo(f"  Reason: {_reason}")
        if _rating:
            click.echo(f"  Rating: {_rating}")
        if _fb_note:
            click.echo(f"  Note: {_fb_note}")

    # Feedback Signals v0 (--more / --full)
    if detail in ("more", "full") and index is not None:
        from openshard.history.feedback import load_feedback_for_shard
        from openshard.history.shard_contract import _make_shard_id
        _shard_id = _make_shard_id(entry.get("timestamp", ""), index)
        _fb_signals = load_feedback_for_shard(_shard_id)
        if _fb_signals:
            _fb = _fb_signals[-1]
            click.echo("\nFEEDBACK")
            click.echo(f"{'Outcome':<12}{_fb.outcome}")
            if _fb.note:
                click.echo(f"{'Note':<12}{_fb.note}")

    # Developer feedback v1 (in-entry, --more / --full)
    _dev_feedback = entry.get("developer_feedback")
    if detail in ("more", "full") and _dev_feedback:
        click.echo("\n  FEEDBACK")
        click.echo(f"  {'Outcome':<12}{_dev_feedback.get('outcome', '')}")
        if _dev_feedback.get("edited"):
            click.echo(f"  {'Edited':<12}yes")
        if _dev_feedback.get("manual_fix_required"):
            click.echo(f"  {'Manual fix':<12}yes")
        if _dev_feedback.get("ci_passed"):
            click.echo(f"  {'CI':<12}passed")
        elif _dev_feedback.get("ci_failed"):
            click.echo(f"  {'CI':<12}failed")
        if _dev_feedback.get("pr_created"):
            click.echo(f"  {'PR':<12}created")
        if _dev_feedback.get("pr_merged"):
            click.echo(f"  {'PR':<12}merged")
        if _dev_feedback.get("reason"):
            click.echo(f"  {'Reason':<12}{_dev_feedback['reason']}")

    # Token / model detail (--full only)
    if detail == "full":
        full_model = entry.get("execution_model", "")
        if full_model and not stage_runs_data and not routing_model:
            click.echo(f"\nModel: {full_model}")
        pt = entry.get("prompt_tokens", 0)
        ct = entry.get("completion_tokens", 0)
        tt = entry.get("total_tokens", 0)
        if tt:
            click.echo(f"Tokens: {pt} prompt / {ct} completion / {tt} total")
        if entry.get("retry_triggered"):
            click.echo("Retried: yes")
        vp = entry.get("verification_passed")
        if vp is not None:
            click.echo(f"Verification: {'passed' if vp else 'failed'}")
        ws = entry.get("workspace_path")
        if ws:
            click.echo(f"Workspace: {ws}")

    # Native inspection (--full only)
    if detail == "full":
        _render_native_inspection(entry, detail)

    # Tier dispatch for non-native runs (native gets it inside _render_native_inspection)
    if detail == "full" and entry.get("workflow") != "native":
        _tdr = entry.get("tier_dispatch_receipt")
        _vpol = entry.get("validator_policy")
        if _tdr and _tdr.get("enabled"):
            from openshard.cli.run_output import _render_tier_dispatch_block
            _init_model = entry.get("routing_selected_model")
            _vr = entry.get("validator_result")
            _is_direct_ask = _is_ro and not any(
                sr.get("stage_type") == "planning"
                for sr in (entry.get("stage_runs") or [])
            )
            for line in _render_tier_dispatch_block(_tdr, detail, initial_model=_init_model, validator_result=_vr, validator_policy=_vpol, is_ask=_is_direct_ask):
                click.echo(line)
        elif _vpol and not _vpol.get("run"):
            click.echo(f"\nValidator: skipped — {_vpol.get('reason', '')}")

    duration = entry.get("duration_seconds", 0)
    cost = entry.get("estimated_cost")
    cost_str = f"${cost:.4f}" if cost is not None else "-"

    # Compact RECEIPT — default view only, appears before Time/Cost footer
    if detail == "default":
        from openshard.history.shard_contract import build_shard_receipt, render_compact_shard_receipt
        _shard = build_shard_receipt(entry, index)
        click.echo("")
        click.echo(render_compact_shard_receipt(_shard))

    click.echo(f"\nTime: {duration:.1f}s   Cost: {cost_str}")

    # Baseline comparison and cost section (after Time/Cost footer)
    if detail == "default":
        _nm = _native_meta_from_entry(entry)
        if _nm is not None:
            from openshard.cost.baseline import format_baseline_line
            _pt = entry.get("prompt_tokens") or 0
            _ct = entry.get("completion_tokens") or 0
            _bl = format_baseline_line(_pt, _ct, actual_cost=cost)
            if _bl is not None:
                click.echo(_bl)
    elif detail == "more":
        from openshard.cost.baseline import (
            compute_baseline_comparison,
            format_concise_comparison_lines,
        )
        _pt = entry.get("prompt_tokens") or 0
        _ct = entry.get("completion_tokens") or 0
        _cmp = compute_baseline_comparison(_pt, _ct, actual_cost=cost)
        if _cmp is not None:
            click.echo("\nCOST COMPARISON")
            click.echo(f"  OpenShard selected: {_shard.model_display}")
            click.echo(f"  Run cost: ${_cmp['actual_cost_usd']:.4f}")
            _rows = format_concise_comparison_lines(_pt, _ct, _cmp["actual_cost_usd"])
            if _rows:
                click.echo("")
                click.echo("  Estimated same-token baseline:")
                for _row in _rows:
                    click.echo(_row)
            click.echo("")
            click.echo("  Method: same-token API price estimate. Real single-model cost may differ.")
    elif detail == "full":
        from openshard.cost.baseline import (
            compute_baseline_comparison,
            format_full_comparison_lines,
        )
        _pt = entry.get("prompt_tokens") or 0
        _ct = entry.get("completion_tokens") or 0
        _cmp = compute_baseline_comparison(_pt, _ct, actual_cost=cost)
        if _cmp is not None:
            click.echo("\nCost comparison")
            click.echo(f"  OpenShard: ${_cmp['actual_cost_usd']:.4f}")
            _rows = format_full_comparison_lines(_pt, _ct, _cmp["actual_cost_usd"])
            if _rows:
                click.echo("")
                click.echo("  Compared with")
                for _row in _rows:
                    click.echo(_row)
            click.echo("")
            click.echo("  Method: same-token API price estimate. Real single-model cost may differ.")


@cli.command()
@click.option("--more", is_flag=True, default=False, help="Show file list, model names, and token breakdown.")
@click.option("--full", is_flag=True, default=False, help="Show all stored details including verification and workspace.")
@click.option("--json", "as_json", is_flag=True, default=False, help="Machine-readable output.")
def last(more: bool, full: bool, as_json: bool):
    """Show details of the most recent run without rerunning it."""
    log_path = Path.cwd() / _LOG_PATH

    if as_json:
        from openshard.history.shard_contract import build_shard_receipt

        entries = _load_run_entries(log_path)
        if not entries:
            click.echo(json.dumps(_machine_envelope("last", "not_found", run=None), indent=2))
            return
        entry = entries[-1]
        receipt = build_shard_receipt(entry, index=len(entries) - 1)
        from openshard.history.proof_contract import build_shard_proof_contract
        from openshard.history.trust_score import evaluate_trust_score

        _ts = evaluate_trust_score(
            entry, receipt,
            interaction_event_types=_interaction_event_types(entry.get("timestamp", "")),
        )
        payload = _machine_envelope(
            "last", "ok", shard_id=receipt.shard_id,
            run=_export_run_entry(entry, include_timeline=True, receipt=receipt),
            trust={
                "score": _ts.score,
                "band": _ts.band,
                "penalties": [
                    {"code": p.code, "points": p.points, "reason": p.reason}
                    for p in _ts.penalties
                ],
            },
            proof_contract=build_shard_proof_contract(entry),
        )
        click.echo(json.dumps(payload, indent=2))
        return

    detail = "full" if full else ("more" if more else "default")
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
    _render_log_entry(entries[-1], detail, index=len(entries) - 1)


@cli.group("trust")
def trust_group() -> None:
    """Run Trust Score — a heuristic over recorded proof signals."""


@trust_group.command("last")
@click.option("--json", "as_json", is_flag=True, default=False, help="Machine-readable output.")
def trust_last(as_json: bool) -> None:
    """Show the trust score for the most recent run.

    This is a deterministic trust heuristic over recorded proof signals, not a
    safety guarantee or certification.
    """
    from openshard.history.shard_contract import build_shard_receipt
    from openshard.history.trust_score import evaluate_trust_score, format_human, to_payload

    log_path = Path.cwd() / _LOG_PATH
    entries = _load_run_entries(log_path)
    if not entries:
        if as_json:
            click.echo(json.dumps(
                _machine_envelope("trust last", "not_found", score=None, signals={}, penalties=[]),
                indent=2,
            ))
        else:
            click.echo("No run history found. Run a task first with 'openshard run'.")
        return

    entry = entries[-1]
    receipt = build_shard_receipt(entry, index=len(entries) - 1)
    ts = evaluate_trust_score(
        entry, receipt,
        interaction_event_types=_interaction_event_types(entry.get("timestamp", "")),
    )
    if as_json:
        payload = _machine_envelope(
            "trust last", ts.status, shard_id=ts.shard_id, warnings=list(ts.warnings),
            **to_payload(ts),
        )
        click.echo(json.dumps(payload, indent=2))
        return
    for line in format_human(ts):
        click.echo(line)


@cli.group("reflect")
def reflect_group() -> None:
    """Post-run reflection and advisory review."""


@reflect_group.command("last")
@click.option("--json", "as_json", is_flag=True, default=False, help="Machine-readable output.")
def reflect_last(as_json: bool) -> None:
    """Show a reflection on the most recent run."""
    from openshard.reflection.reflector import build_run_reflection, render_run_reflection
    from openshard.history.shard_contract import build_shard_receipt

    log_path = Path.cwd() / _LOG_PATH
    entries = _load_run_entries(log_path)
    if not entries:
        if as_json:
            click.echo(json.dumps(_machine_envelope("reflect last", "not_found", reflection=None), indent=2))
        else:
            click.echo("No run history found. Run a task first with 'openshard run'.")
        return
    receipt = build_shard_receipt(entries[-1], index=len(entries) - 1)
    reflection = build_run_reflection(receipt)
    if as_json:
        from dataclasses import asdict

        payload = _machine_envelope(
            "reflect last", "ok", shard_id=receipt.shard_id,
            warnings=list(reflection.warnings), reflection=asdict(reflection),
        )
        click.echo(json.dumps(payload, indent=2))
        return
    for line in render_run_reflection(reflection):
        click.echo(line)


@cli.group("pr", invoke_without_command=True)
@click.pass_context
def pr_group(ctx: click.Context) -> None:
    """PR comment and local export utilities."""
    if ctx.invoked_subcommand is None:
        click.echo(ctx.get_help())


@pr_group.command("comment")
@click.option("--output", default=None, help="Write output to this path instead of stdout.")
@click.option("--json", "as_json", is_flag=True, default=False, help="Machine-readable output.")
@click.option("--github-step-summary", "github_step_summary", is_flag=True, default=False,
              help="Append the markdown receipt to $GITHUB_STEP_SUMMARY (GitHub Actions).")
@click.option("--github-output", "github_output", is_flag=True, default=False,
              help="Write safe key=value outputs to $GITHUB_OUTPUT (GitHub Actions).")
def pr_comment(output: str | None, as_json: bool, github_step_summary: bool, github_output: bool) -> None:
    """Generate a GitHub-ready PR comment from the latest OpenShard run.

    The --github-step-summary and --github-output flags write to the files
    referenced by $GITHUB_STEP_SUMMARY and $GITHUB_OUTPUT. This is a local,
    file-based Actions layer only: no GitHub API, no gh, no network, no auth.
    """
    from openshard.history.shard_contract import build_shard_receipt
    from openshard.github.pr_comment import build_pr_comment_summary, render_pr_comment

    log_path = Path.cwd() / _LOG_PATH
    entries = _load_run_entries(log_path)

    if not entries:
        ss_available = ss_written = go_available = go_written = False
        if github_output:
            go_available, go_written = _write_github_output(
                {"openshard_available": "false", "openshard_status": "not_found"}
            )
        if github_step_summary:
            ss_available, ss_written = _write_github_step_summary(
                "## OpenShard\n\nNo OpenShard run found. Run an OpenShard task first."
            )
        if as_json:
            body: dict = {"summary": None}
            if github_step_summary:
                body["github_step_summary_available"] = ss_available
                body["github_step_summary_written"] = ss_written
            if github_output:
                body["github_output_available"] = go_available
                body["github_output_written"] = go_written
            click.echo(json.dumps(_machine_envelope("pr comment", "not_found", **body), indent=2))
        else:
            click.echo("No run history found. Run an OpenShard task first.")
            _warn_missing_github_env(github_step_summary, ss_available, github_output, go_available)
        return

    entry = entries[-1]
    receipt = build_shard_receipt(entry, index=len(entries) - 1)
    summary = build_pr_comment_summary(entry, receipt)
    markdown = render_pr_comment(summary)

    if as_json:
        from dataclasses import asdict

        body = {"summary": asdict(summary)}
        if output:
            output_path = Path(output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(json.dumps(
                _machine_envelope("pr comment", "ok", shard_id=receipt.shard_id,
                                  warnings=list(summary.warnings), summary=asdict(summary)),
                indent=2) + "\n", encoding="utf-8")
            body["written"] = True
            body["output_path_display"] = _safe_output_display(output)
        if github_output:
            go_pairs = _github_output_pairs(
                "ok", receipt.shard_id, summary.manual_review_required,
                output_path_key="openshard_output_path" if output else None,
                output_display=_safe_output_display(output) if output else None,
            )
            go_available, go_written = _write_github_output(go_pairs)
            body["github_output_available"] = go_available
            body["github_output_written"] = go_written
        if github_step_summary:
            ss_available, ss_written = _write_github_step_summary(markdown)
            body["github_step_summary_available"] = ss_available
            body["github_step_summary_written"] = ss_written
        payload = _machine_envelope(
            "pr comment", "ok", shard_id=receipt.shard_id,
            warnings=list(summary.warnings), **body,
        )
        click.echo(json.dumps(payload, indent=2))
        return

    comment_path_display: str | None = None
    if output:
        output_path = Path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(markdown, encoding="utf-8")
        if output_path.exists():
            comment_path_display = _safe_output_display(output)
        click.echo(f"PR comment written to {output}")
    else:
        click.echo(markdown)

    ss_available = go_available = True
    if github_output:
        go_pairs = _github_output_pairs(
            "ok", receipt.shard_id, summary.manual_review_required,
            output_path_key="openshard_comment_path" if comment_path_display else None,
            output_display=comment_path_display,
        )
        go_available, _ = _write_github_output(go_pairs)
    if github_step_summary:
        ss_available, _ = _write_github_step_summary(markdown)
    _warn_missing_github_env(github_step_summary, ss_available, github_output, go_available)


@cli.group("ci", invoke_without_command=True)
@click.pass_context
def ci_group(ctx: click.Context) -> None:
    """CI policy checks over the latest OpenShard run receipt."""
    if ctx.invoked_subcommand is None:
        click.echo(ctx.get_help())


@ci_group.command("check")
@click.option("--json", "as_json", is_flag=True, default=False,
              help="Machine-readable output (valid JSON only).")
@click.option("--strict", is_flag=True, default=False,
              help="Treat warnings as failures (exit 1).")
@click.option("--github-output", "github_output", is_flag=True, default=False,
              help="Write safe key=value outputs to $GITHUB_OUTPUT (GitHub Actions).")
def ci_check(as_json: bool, strict: bool, github_output: bool) -> None:
    """Evaluate the latest Shard receipt and return a CI verdict.

    Local, deterministic, receipt-based: reduces the most recent OpenShard run
    to pass / warn / fail / skip. No GitHub API, no gh, no network, no auth.
    Exit code is 1 only for fail (and for warnings under --strict); otherwise 0.
    """
    from openshard.history.shard_contract import build_shard_receipt
    from openshard.ci.policy_check import evaluate_ci_check

    log_path = Path.cwd() / _LOG_PATH
    entries = _load_run_entries(log_path)

    if not entries:
        reasons = ["No OpenShard run found to evaluate."]
        if as_json:
            body: dict = {
                "exit_code": 0,
                "reasons": reasons,
                "checks": {
                    "verification": "unknown",
                    "manual_review_required": False,
                    "secret_scan_findings": 0,
                },
            }
            if github_output:
                go_available, go_written = _ci_write_github_output("skip", 0, len(reasons))
                body["github_output_available"] = go_available
                body["github_output_written"] = go_written
            click.echo(json.dumps(_machine_envelope("ci check", "skip", **body), indent=2))
        else:
            click.echo("OpenShard CI Check: skip")
            for reason in reasons:
                click.echo(f"  - {reason}")
            if github_output:
                go_available, _ = _ci_write_github_output("skip", 0, len(reasons))
                if not go_available:
                    click.echo(
                        "warning: GITHUB_OUTPUT is not set; outputs not written.", err=True
                    )
        sys.exit(0)

    entry = entries[-1]
    receipt = build_shard_receipt(entry, index=len(entries) - 1)
    result = evaluate_ci_check(entry, receipt, strict=strict)

    if as_json:
        body = {
            "exit_code": result.exit_code,
            "reasons": result.reasons,
            "checks": result.checks,
        }
        if github_output:
            go_available, go_written = _ci_write_github_output(
                result.status, result.exit_code, len(result.reasons)
            )
            body["github_output_available"] = go_available
            body["github_output_written"] = go_written
        payload = _machine_envelope(
            "ci check", result.status, shard_id=result.shard_id,
            warnings=result.warnings, **body,
        )
        click.echo(json.dumps(payload, indent=2))
        sys.exit(result.exit_code)

    click.echo(f"OpenShard CI Check: {result.status}")
    for reason in result.reasons:
        click.echo(f"  - {reason}")
    for warning in result.warnings:
        click.echo(f"  - {warning}")
    if github_output:
        go_available, _ = _ci_write_github_output(
            result.status, result.exit_code, len(result.reasons)
        )
        if not go_available:
            click.echo("warning: GITHUB_OUTPUT is not set; outputs not written.", err=True)
    sys.exit(result.exit_code)


@cli.group("repo", invoke_without_command=True)
@click.pass_context
def repo_group(ctx: click.Context) -> None:
    """Local repo-map cache: build, reuse, and inspect repo metadata."""
    if ctx.invoked_subcommand is None:
        click.echo(ctx.get_help())


@repo_group.command("map")
@click.option("--json", "as_json", is_flag=True, default=False,
              help="Machine-readable output (valid JSON only).")
@click.option("--refresh", is_flag=True, default=False,
              help="Rebuild the repo map instead of reusing the cache.")
@click.option("--output", default=None,
              help="Also write the repo-map JSON to this path.")
def repo_map_cmd(as_json: bool, refresh: bool, output: str | None) -> None:
    """Build or load the local repo-map cache for the current directory.

    Local, deterministic, metadata-only: no model calls, no network. Caches under
    .openshard/cache/repo-<fingerprint>.json. A clean git tree reuses the cache; a
    dirty tree always rebuilds; --refresh forces a rebuild. Output and cache contain
    no file contents, raw secrets, or absolute paths.
    """
    from openshard.analysis.repo_map_loader import load_or_build_repo_map

    root = Path.cwd()
    loaded = load_or_build_repo_map(root, refresh=refresh)
    repo_map_dict = loaded.repo_map
    cache_hit = loaded.cache_hit
    cache_display = loaded.cache_path_display
    warnings = loaded.warnings

    output_display: str | None = None
    if output:
        output_path = Path(output)
        if output_path.parent != Path("."):
            output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(repo_map_dict, indent=2) + "\n", encoding="utf-8")
        output_display = _safe_output_display(output)

    if as_json:
        body: dict = {
            "cache_hit": cache_hit,
            "cache_path_display": cache_display,
            "repo_map": repo_map_dict,
        }
        if output_display is not None:
            body["output_path_display"] = output_display
        payload = _machine_envelope("repo map", "ok", warnings=warnings, **body)
        click.echo(json.dumps(payload, indent=2))
        return

    _render_repo_map(repo_map_dict, cache_hit=cache_hit, cache_path_display=cache_display)
    if output_display is not None:
        click.echo(f"  Wrote: {output_display}")


@repo_group.command("plan")
@click.argument("task")
@click.option("--json", "as_json", is_flag=True, default=False,
              help="Machine-readable output (valid JSON only).")
@click.option("--refresh", is_flag=True, default=False,
              help="Rebuild the repo map instead of reusing the cache.")
def repo_plan_cmd(task: str, as_json: bool, refresh: bool) -> None:
    """Produce a read-only, repo-aware plan for TASK from repo-map metadata.

    Local, deterministic, metadata-only: no model calls, no network, no file
    contents read, no source files written, and the task is never executed. Uses
    the repo-map cache (a clean git tree reuses it; a dirty tree or --refresh
    rebuilds). Output contains no file contents, raw secrets, or absolute paths;
    the task argument is sanitised before display.
    """
    from openshard.analysis.repo_map_loader import load_or_build_repo_map
    from openshard.planning.repo_plan import build_repo_aware_plan

    root = Path.cwd()
    loaded = load_or_build_repo_map(root, refresh=refresh)
    plan = build_repo_aware_plan(task, loaded.repo_map, cache_hit=loaded.cache_hit)

    if as_json:
        payload = _machine_envelope(
            "repo plan", "ok", warnings=loaded.warnings, **plan.to_dict()
        )
        click.echo(json.dumps(payload, indent=2))
        return

    render_dict = plan.to_dict()
    render_dict["warnings"] = loaded.warnings
    _render_repo_plan(
        render_dict,
        cache_hit=loaded.cache_hit,
        cache_path_display=loaded.cache_path_display,
    )


@cli.group("stats", invoke_without_command=True)
@click.pass_context
def stats_group(ctx: click.Context) -> None:
    """Receipt-quality stats over recent OpenShard run receipts."""
    if ctx.invoked_subcommand is None:
        click.echo(ctx.get_help())


@stats_group.command("completeness")
@click.option("--json", "as_json", is_flag=True, default=False,
              help="Machine-readable output (valid JSON only).")
@click.option("--limit", default=20, type=click.IntRange(min=1),
              help="Evaluate only the most recent N receipts (default 20).")
def stats_completeness(as_json: bool, limit: int) -> None:
    """Score recent Shard receipts for completeness.

    Local, deterministic, receipt-based: reports a completeness heuristic plus
    the fields that are consistently present (strong) or missing (weak) across
    recent runs. No network, no model calls; no secrets or absolute paths leak.
    """
    from openshard.history.shard_contract import build_shard_receipt
    from openshard.history.completeness import evaluate_completeness

    log_path = Path.cwd() / _LOG_PATH
    entries = _load_run_entries(log_path)

    if not entries:
        if as_json:
            payload = _machine_envelope(
                "stats completeness", "not_found",
                runs_checked=0,
                average_score_percent=0,
                field_presence={},
                strong_fields=[],
                weak_fields=[],
                recommendations=[],
                receipts=[],
            )
            click.echo(json.dumps(payload, indent=2))
        else:
            click.echo("No run history found. Run 'openshard run' to get started.")
        return

    recent = entries[-limit:]
    receipts = [
        build_shard_receipt(entry, index=len(entries) - len(recent) + offset)
        for offset, entry in enumerate(recent)
    ]
    report = evaluate_completeness(receipts)

    if as_json:
        payload = _machine_envelope(
            "stats completeness", "ok",
            runs_checked=report.runs_checked,
            average_score_percent=report.average_score_percent,
            field_presence=report.field_presence,
            strong_fields=report.strong_fields,
            weak_fields=report.weak_fields,
            recommendations=report.recommendations,
            receipts=[
                {
                    "shard_id": rc.shard_id,
                    "score_percent": rc.score_percent,
                    "present_fields": rc.present_fields,
                    "missing_fields": rc.missing_fields,
                }
                for rc in report.receipts
            ],
        )
        click.echo(json.dumps(payload, indent=2))
        return

    click.echo("\nShard receipt completeness")
    click.echo(f"  runs checked:         {report.runs_checked}")
    click.echo(f"  average completeness: {report.average_score_percent}%")

    if report.strong_fields:
        click.echo(f"\n  strong fields:  {', '.join(report.strong_fields)}")
    if report.weak_fields:
        weak_display = ", ".join(
            f"{name} ({report.field_presence[name]['presence_percent']}%)"
            for name in report.weak_fields
        )
        click.echo(f"  weak fields:    {weak_display}")

    if report.recommendations:
        click.echo("\n  recommendations:")
        for rec in report.recommendations:
            click.echo(f"    - {rec}")


@stats_group.command("failures")
@click.option("--json", "as_json", is_flag=True, default=False,
              help="Machine-readable output (valid JSON only).")
@click.option("--limit", default=20, type=click.IntRange(min=1),
              help="Classify only the most recent N receipts (default 20).")
def stats_failures(as_json: bool, limit: int) -> None:
    """Classify recent runs into stable failure categories.

    Local, deterministic, receipt-based: reads existing Shard/run metadata and
    classifies each recent run into one failure category (or no failure). No
    network, no model calls; no secrets, raw error messages, or absolute paths
    leak.
    """
    from openshard.history.shard_contract import build_shard_receipt
    from openshard.history.failures import evaluate_failures

    log_path = Path.cwd() / _LOG_PATH
    entries = _load_run_entries(log_path)

    if not entries:
        if as_json:
            payload = _machine_envelope(
                "stats failures", "not_found",
                runs_checked=0,
                category_counts={},
                top_categories=[],
                recommendations=[],
                failures=[],
            )
            click.echo(json.dumps(payload, indent=2))
        else:
            click.echo("No run history found. Run 'openshard run' to get started.")
        return

    recent = entries[-limit:]
    pairs = [
        (entry, build_shard_receipt(entry, index=len(entries) - len(recent) + offset))
        for offset, entry in enumerate(recent)
    ]
    report = evaluate_failures(pairs)

    if as_json:
        payload = _machine_envelope(
            "stats failures", "ok",
            runs_checked=report.runs_checked,
            category_counts=report.category_counts,
            top_categories=report.top_categories,
            recommendations=report.recommendations,
            failures=[
                {
                    "shard_id": fc.shard_id,
                    "category": fc.category,
                    "confidence": fc.confidence,
                    "reasons": fc.reasons,
                    "signals": fc.signals,
                }
                for fc in report.failures
            ],
        )
        click.echo(json.dumps(payload, indent=2))
        return

    failure_total = sum(
        count for name, count in report.category_counts.items()
        if name != "no_failure_detected"
    )

    click.echo("\nFailure taxonomy")
    click.echo(f"  runs checked: {report.runs_checked}")
    click.echo(f"  failures:     {failure_total}")

    if report.top_categories:
        click.echo("\n  top failure categories:")
        for tc in report.top_categories:
            click.echo(f"    {tc['category']:<22}{tc['count']}")

    if report.recommendations:
        click.echo("\n  recommendations:")
        for rec in report.recommendations:
            click.echo(f"    - {rec}")

    if report.failures:
        click.echo("\n  recent failures:")
        for fc in report.failures:
            click.echo(f"    {fc.shard_id:<12}{fc.category} ({fc.confidence})")


@cli.command("apply-last")
@click.option("--dry-run", is_flag=True, default=False, help="Show what would be applied without copying files.")
@click.option("--file", "include_files", multiple=True, help="Only apply this relative file path. Can be used multiple times.")
@click.option("--exclude", "exclude_files", multiple=True, help="Exclude this relative file path. Can be used multiple times.")
@click.option("--candidate", "candidate_index", default=None, type=click.IntRange(min=1),
              help="Apply files from a specific candidate (1-based index).")
def apply_last(dry_run: bool, include_files: tuple[str, ...], exclude_files: tuple[str, ...], candidate_index: int | None) -> None:
    """Promote files from the most recent sandbox run into the real repo."""
    from openshard.native.sandbox_apply import (
        apply_sandbox_changes,
        extract_sandbox_path_from_entry,
        extract_candidate_sandbox_path_from_entry,
        filter_sandbox_changed_files,
        list_sandbox_changed_files,
    )

    log_path = Path.cwd() / _LOG_PATH
    entries = _load_run_entries(log_path)
    if not entries:
        click.echo("No run history found. Run a task first with 'openshard run'.")
        return

    entry = entries[-1]

    if entry.get("executor") != "native":
        click.echo("Latest run is not a native run.")
        return

    if candidate_index is not None:
        sandbox_path_str = extract_candidate_sandbox_path_from_entry(entry, candidate_index)
        if not sandbox_path_str:
            click.echo(f"Candidate {candidate_index} has no sandbox path to apply.")
            return
    else:
        sandbox_path_str = extract_sandbox_path_from_entry(entry)
        if not sandbox_path_str:
            click.echo("Latest native run has no sandbox path to apply.")
            return

    include = list(include_files) or None
    exclude = list(exclude_files) or None

    sandbox_path = Path(sandbox_path_str)
    all_files = list_sandbox_changed_files(Path.cwd(), sandbox_path)
    files = filter_sandbox_changed_files(all_files, include=include, exclude=exclude)

    if not files:
        if include or exclude:
            click.echo("No sandbox changes matched the apply selection.")
        else:
            click.echo("No sandbox changes to apply.")
        log_sandbox_apply_receipt(SandboxApplyReceipt(
            source_run_id=entry.get("timestamp", ""),
            sandbox_path=str(sandbox_path),
            applied=False,
            files_applied=[],
            files_skipped=[],
            dry_run=False,
            reason="No sandbox changes to apply.",
        ))
        return

    click.echo(f"Sandbox: {sandbox_path_str}")
    click.echo("")
    if dry_run:
        click.echo(f"Would apply {len(files)} file(s):")
        for f in files:
            click.echo(f"  - {f}")
        log_sandbox_apply_receipt(SandboxApplyReceipt(
            source_run_id=entry.get("timestamp", ""),
            sandbox_path=str(sandbox_path),
            applied=False,
            files_applied=[],
            files_skipped=[],
            dry_run=True,
            reason="dry run",
        ))
        return

    click.echo(f"Files to apply ({len(files)}):")
    for f in files:
        click.echo(f"  - {f}")
    click.echo("")

    result = apply_sandbox_changes(Path.cwd(), sandbox_path, include=include, exclude=exclude)

    log_sandbox_apply_receipt(SandboxApplyReceipt(
        source_run_id=entry.get("timestamp", ""),
        sandbox_path=str(sandbox_path),
        applied=result.applied,
        files_applied=list(result.files_applied),
        files_skipped=list(result.files_skipped),
        dry_run=False,
        reason=result.reason,
    ))

    if result.reason and not result.files_applied:
        raise click.ClickException(result.reason)

    click.echo(f"Applied {len(result.files_applied)} file(s) from sandbox.")
    for f in result.files_applied:
        click.echo(f"  - {f}")
    if result.files_skipped:
        click.echo(f"Skipped {len(result.files_skipped)} file(s).")
        for f in result.files_skipped:
            click.echo(f"  [skipped] {f}")


@cli.command("candidates-last")
def candidates_last() -> None:
    """Show the candidate table for the most recent native run."""
    from openshard.native.sandbox_apply import get_candidate_records_from_entry

    log_path = Path.cwd() / _LOG_PATH
    entries = _load_run_entries(log_path)
    if not entries:
        click.echo("No run history found.")
        return

    entry = entries[-1]
    if entry.get("executor") != "native":
        click.echo("Latest run is not a native run.")
        return

    records = get_candidate_records_from_entry(entry)
    if not records:
        click.echo("No candidate summary available.")
        return

    click.echo(f"{'Candidate':<10} {'Status':<10} {'Selected':<10} {'Score':<8} {'Files':<7} Exit")
    for r in records:
        idx = r.get("candidate_index", "?")
        status = r.get("verification_status", "")
        selected = "yes" if r.get("selected") else "no"
        score = r.get("score", 0.0)
        score_s = f"{score:.1f}"
        files = len(r.get("files_written") or [])
        ec = r.get("exit_code")
        exit_s = str(ec) if ec is not None else "-"
        click.echo(f"{idx:<10} {status:<10} {selected:<10} {score_s:<8} {files:<7} {exit_s}")


@cli.command("diff-last")
@click.option("--full", "show_full", is_flag=True, default=False,
              help="Show full sandbox diff, not just stat summary.")
@click.option("--candidate", "candidate_index", default=None, type=click.IntRange(min=1),
              help="Show diff for a specific candidate (1-based index).")
def diff_last(show_full: bool, candidate_index: int | None) -> None:
    """Preview the diff from the most recent native sandbox run."""
    from openshard.native.sandbox_diff import get_sandbox_diff
    from openshard.native.sandbox_apply import (
        extract_sandbox_path_from_entry,
        extract_candidate_sandbox_path_from_entry,
    )

    log_path = Path.cwd() / _LOG_PATH
    if not log_path.exists():
        click.echo("No run history found.")
        return

    entries = _load_run_entries(log_path)
    if not entries:
        click.echo("No runs recorded yet.")
        return

    entry = entries[-1]

    if entry.get("executor") != "native":
        click.echo("Latest run is not a native run.")
        return

    if candidate_index is not None:
        sandbox_path_str = extract_candidate_sandbox_path_from_entry(entry, candidate_index)
        if not sandbox_path_str:
            click.echo(f"Candidate {candidate_index} has no sandbox path to diff.")
            return
    else:
        sandbox_path_str = extract_sandbox_path_from_entry(entry)
        if not sandbox_path_str:
            click.echo("Latest native run has no sandbox path to diff.")
            return

    result = get_sandbox_diff(Path(sandbox_path_str), full=show_full)

    if not result.available:
        click.echo(f"No sandbox diff available: {result.reason}")
        return

    click.echo(f"Sandbox: {sandbox_path_str}")
    click.echo(f"Changed files ({len(result.files_changed)}):")
    for f in result.files_changed:
        click.echo(f"  - {f}")

    if result.stat_text:
        click.echo("")
        click.echo("Diff stat:")
        click.echo(result.stat_text)

    if show_full and result.diff_text:
        click.echo("")
        click.echo("Diff:")
        click.echo(result.diff_text)


@cli.command("apply-receipts")
@click.option("--last", "last_n", default=10, type=click.IntRange(min=1), show_default=True)
def apply_receipts_cmd(last_n: int) -> None:
    """Show recent sandbox apply receipts."""
    receipts = recent_sandbox_apply_receipts(limit=last_n)
    if not receipts:
        click.echo("No sandbox apply receipts recorded yet.")
        return

    header = f"{'Time':<18} {'Applied':<8} {'Skipped':<8} {'Dry Run':<8} Sandbox"
    click.echo(header)
    for r in receipts:
        ts = r.timestamp[:16].replace("T", " ")
        dry = "yes" if r.dry_run else "no"
        click.echo(f"{ts:<18} {r.applied_count:<8} {r.skipped_count:<8} {dry:<8} {r.sandbox_path}")


@cli.command("checkpoints")
@click.option("--last", "last_n", default=10, type=click.IntRange(min=1), show_default=True)
def checkpoints_cmd(last_n: int) -> None:
    """Show recent native run checkpoints."""
    from openshard.history.run_checkpoints import recent_run_checkpoints
    events = recent_run_checkpoints(limit=last_n)
    if not events:
        click.echo("No run checkpoints recorded yet.")
        return
    header = f"{'Time':<18} {'Run':<22} {'Stage':<14} {'Status':<10} {'Verify':<10} Retry"
    click.echo(header)
    for evt in events:
        ts = evt.timestamp[:16].replace("T", " ")
        run_short = evt.run_id[:20] if evt.run_id else "-"
        retry_s = "yes" if evt.retry_used else "no"
        verify_s = evt.verification_status or "-"
        click.echo(f"{ts:<18} {run_short:<22} {evt.stage:<14} {evt.status:<10} {verify_s:<10} {retry_s}")


@cli.command("resume-last")
def resume_last() -> None:
    """Show safe resume guidance for the most recent native run (v0)."""
    from openshard.native.sandbox_apply import extract_sandbox_path_from_entry
    from openshard.history.run_checkpoints import run_checkpoints_for_run

    log_path = Path.cwd() / _LOG_PATH
    entries = _load_run_entries(log_path)
    if not entries:
        click.echo("No run history found.")
        return

    entry = entries[-1]
    if entry.get("executor") != "native":
        click.echo("Latest run is not a native run.")
        return

    run_id = entry.get("timestamp", "")
    checkpoints = run_checkpoints_for_run(run_id)
    if not checkpoints:
        click.echo("No checkpoints found for latest native run.")
        return

    latest = checkpoints[-1]
    click.echo(f"Latest checkpoint: {latest.stage} ({latest.status})")

    sandbox_path_str = extract_sandbox_path_from_entry(entry)
    if sandbox_path_str and Path(sandbox_path_str).exists():
        click.echo("\nResume options:")
        click.echo("  openshard diff-last --full")
        click.echo("  openshard apply-last --dry-run")
        click.echo("  openshard apply-last")
    else:
        click.echo("\nNo live sandbox found. This run can be inspected, but not resumed from workspace.")


def _load_run_entries(log_path: Path) -> list[dict]:
    if not log_path.exists():
        return []
    entries: list[dict] = []
    for line in log_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            entries.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return entries


def _interaction_event_types(run_id: str) -> list[str]:
    """Load sanitised developer-interaction event *types* for a run. Never raises.

    Returns types only (e.g. ``"unsafe_command"``) — never raw summaries — so the
    trust evaluator stays leak-free. Missing/corrupt interaction files yield ``[]``.
    """
    if not run_id:
        return []
    try:
        from openshard.history.interactions import interaction_events_for_run

        return [e.event_type for e in interaction_events_for_run(run_id)]
    except Exception:
        return []


# Schema version for the machine-readable (--json) output contract. Bump only on
# breaking envelope changes; the underlying structures keep their own source/version.
_MACHINE_OUTPUT_SCHEMA_VERSION = "1"


def _machine_envelope(
    command: str,
    status: str,
    shard_id: str | None = None,
    warnings: list[str] | None = None,
    **body: object,
) -> dict:
    """Build the stable machine-readable envelope shared by all --json commands."""
    return {
        "schema_version": _MACHINE_OUTPUT_SCHEMA_VERSION,
        "command": command,
        "status": status,
        "shard_id": shard_id,
        "warnings": warnings or [],
        **body,
    }


def _safe_output_display(output: str) -> str:
    """Return a path safe to echo in JSON: relative as-is, absolute -> bare filename."""
    p = output.replace("\\", "/")
    if p.startswith("/") or (len(p) > 1 and p[1] == ":"):
        return Path(output).name
    return output


def _write_github_step_summary(markdown: str) -> tuple[bool, bool]:
    """Append markdown to $GITHUB_STEP_SUMMARY. Returns (available, written).

    available is True when the env var is set; written is True when the append
    succeeded. No network, no secrets - markdown is already sanitized upstream.
    """
    path = os.environ.get("GITHUB_STEP_SUMMARY")
    if not path:
        return (False, False)
    try:
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(markdown.rstrip("\n") + "\n")
        return (True, True)
    except OSError:
        return (True, False)


def _write_github_output(pairs: dict[str, str]) -> tuple[bool, bool]:
    """Append safe key=value lines to $GITHUB_OUTPUT. Returns (available, written).

    Values are coerced to str and stripped of CR/LF so each output stays on a
    single line (no heredoc / invalid-character cases). Keys are fixed literals.
    """
    path = os.environ.get("GITHUB_OUTPUT")
    if not path:
        return (False, False)
    try:
        with open(path, "a", encoding="utf-8") as fh:
            for key, value in pairs.items():
                safe_value = str(value).replace("\r", " ").replace("\n", " ")
                fh.write(f"{key}={safe_value}\n")
        return (True, True)
    except OSError:
        return (True, False)


def _github_output_pairs(
    status: str,
    shard_id: str | None,
    manual_review_required: bool | None,
    output_path_key: str | None = None,
    output_display: str | None = None,
) -> dict[str, str]:
    """Build the safe scalar key=value map for $GITHUB_OUTPUT."""
    pairs: dict[str, str] = {
        "openshard_available": "true" if status == "ok" else "false",
        "openshard_status": status,
    }
    if status == "ok":
        if shard_id:
            pairs["openshard_shard_id"] = shard_id
        if manual_review_required is not None:
            pairs["openshard_manual_review_required"] = "true" if manual_review_required else "false"
        if output_path_key and output_display:
            pairs[output_path_key] = output_display
    return pairs


def _ci_write_github_output(
    status: str, exit_code: int, reasons_count: int
) -> tuple[bool, bool]:
    """Write the CI-check scalar outputs to $GITHUB_OUTPUT. Returns (available, written)."""
    return _write_github_output(
        {
            "openshard_ci_status": status,
            "openshard_ci_exit_code": str(exit_code),
            "openshard_ci_reasons_count": str(reasons_count),
        }
    )


def _warn_missing_github_env(
    github_step_summary: bool,
    ss_available: bool,
    github_output: bool,
    go_available: bool,
) -> None:
    """Warn on stderr (human mode only) when a requested GitHub env var is unset."""
    if github_step_summary and not ss_available:
        click.echo("warning: GITHUB_STEP_SUMMARY is not set; step summary not written.", err=True)
    if github_output and not go_available:
        click.echo("warning: GITHUB_OUTPUT is not set; outputs not written.", err=True)


_ALLOWED_OUTCOMES_V1 = ["accepted", "rejected", "partial", "abandoned", "retried"]


@cli.command()
@click.option(
    "--outcome",
    type=click.Choice(_ALLOWED_OUTCOMES_V1, case_sensitive=False),
    required=True,
    help="Outcome: accepted, rejected, partial, abandoned, or retried.",
)
@click.option("--reason", default=None, help="Optional free-text reason.")
@click.option("--edited", is_flag=True, default=False, help="You edited the output manually.")
@click.option("--manual-fix-required", is_flag=True, default=False, help="A manual fix was required.")
@click.option("--ci-passed", is_flag=True, default=False, help="CI passed after this run.")
@click.option("--ci-failed", is_flag=True, default=False, help="CI failed after this run.")
@click.option("--pr-created", is_flag=True, default=False, help="A PR was created from this run.")
@click.option("--pr-merged", is_flag=True, default=False, help="The PR was merged.")
def feedback(
    outcome: str,
    reason: str | None,
    edited: bool,
    manual_fix_required: bool,
    ci_passed: bool,
    ci_failed: bool,
    pr_created: bool,
    pr_merged: bool,
) -> None:
    """Record developer feedback for the most recent run."""
    log_path = Path.cwd() / _LOG_PATH
    if not log_path.exists():
        raise click.ClickException("No run history found. Run a task first with 'openshard run'.")
    entries = _load_run_entries(log_path)
    if not entries:
        raise click.ClickException("No run history found. Run a task first with 'openshard run'.")
    df: dict = {
        "schema_version": 1,
        "outcome": outcome.lower(),
        "reason": reason or None,
        "edited": edited,
        "manual_fix_required": manual_fix_required,
        "ci_passed": ci_passed,
        "ci_failed": ci_failed,
        "pr_created": pr_created,
        "pr_merged": pr_merged,
        "recorded_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "source": "cli",
    }
    entries[-1]["developer_feedback"] = df
    with log_path.open("w", encoding="utf-8") as fh:
        for entry in entries:
            fh.write(json.dumps(entry) + "\n")
    click.echo(f"Feedback recorded: {outcome.lower()}")
    try:
        from openshard.history.interactions import DeveloperInteractionEvent, log_interaction_event
        _event_type_map = {
            "accepted": "feedback_accepted",
            "rejected": "feedback_rejected",
            "partial": "feedback_partial",
            "abandoned": "feedback_abandoned",
            "retried": "feedback_retried",
        }
        _accepted_map = {
            "accepted": True,
            "partial": True,
            "rejected": False,
            "retried": False,
        }
        _run_id = entries[-1].get("timestamp") or ""
        _evt = DeveloperInteractionEvent(
            run_id=_run_id,
            event_type=_event_type_map.get(outcome.lower(), "feedback_noted"),
            summary=f"feedback outcome={outcome.lower()}",
            correction_reason=reason,
            accepted=_accepted_map.get(outcome.lower()),
            metadata={"edited": edited, "ci_passed": ci_passed, "ci_failed": ci_failed},
        )
        log_interaction_event(_evt)
    except Exception:
        pass


@cli.command("feedback-stats")
def feedback_stats() -> None:
    """Show local developer feedback stats."""
    log_path = Path.cwd() / _LOG_PATH
    entries = _load_run_entries(log_path)

    if not log_path.exists():
        click.echo("No run history found. Run a task first with 'openshard run'.")
        return

    if not entries:
        click.echo("No runs recorded yet.")
        return

    total = len(entries)
    feedback_entries = [e for e in entries if isinstance(e.get("feedback"), dict)]
    feedback_count = len(feedback_entries)

    if feedback_count == 0:
        click.echo("Developer feedback\n")
        click.echo(f"Runs: {total}")
        click.echo("Feedback: 0 recorded")
        click.echo("\nTip: add feedback with 'openshard feedback --rating good'")
        return

    counts = {"good": 0, "mixed": 0, "bad": 0}
    by_model: dict[str, dict[str, int]] = {}
    action_counts: dict[str, int] = {
        "accepted": 0, "rejected": 0, "edited": 0,
        "retried": 0, "partially-accepted": 0, "unknown": 0,
    }
    reason_counts: dict[str, int] = {}
    by_category: dict[str, dict[str, int]] = {}
    for entry in feedback_entries:
        fb = entry["feedback"]
        rating = fb.get("rating", "")
        if rating in counts:
            counts[rating] += 1
        raw_model = entry.get("execution_model") or entry.get("model") or "unknown"
        label = _model_label(raw_model)
        bucket = by_model.setdefault(label, {"good": 0, "mixed": 0, "bad": 0})
        if rating in bucket:
            bucket[rating] += 1
        action = fb.get("action")
        if action in action_counts:
            action_counts[action] += 1
        reason = fb.get("correction_reason")
        if reason:
            reason_counts[reason] = reason_counts.get(reason, 0) + 1
        category = entry.get("routing_category")
        if category:
            cat_bucket = by_category.setdefault(category, {"good": 0, "mixed": 0, "bad": 0})
            if rating in cat_bucket:
                cat_bucket[rating] += 1

    percent = round(feedback_count / total * 100)

    click.echo("Developer feedback\n")
    click.echo(f"Runs: {total}")
    click.echo(f"Feedback: {feedback_count} recorded ({percent}%)")
    click.echo(f"Good:  {counts['good']}")
    click.echo(f"Mixed: {counts['mixed']}")
    click.echo(f"Bad:   {counts['bad']}")

    click.echo("\nBy model")
    for lbl, bucket in by_model.items():
        click.echo(f"  {lbl}: good={bucket['good']} mixed={bucket['mixed']} bad={bucket['bad']}")

    shown_actions = [(a, c) for a, c in action_counts.items() if c > 0]
    if shown_actions:
        click.echo("\nBy action")
        for action, count in shown_actions:
            click.echo(f"  {action}: {count}")

    if reason_counts:
        click.echo("\nCorrection reasons")
        for reason, count in sorted(reason_counts.items(), key=lambda x: -x[1]):
            click.echo(f"  {reason}: {count}")

    if by_category:
        click.echo("\nBy category")
        for cat, cat_bucket in by_category.items():
            click.echo(f"  {cat}: good={cat_bucket['good']} mixed={cat_bucket['mixed']} bad={cat_bucket['bad']}")

    notes = [
        (e["feedback"].get("rating", ""), e["feedback"].get("note", ""))
        for e in reversed(feedback_entries)
        if e["feedback"].get("note", "").strip()
    ][:5]

    if notes:
        click.echo("\nRecent notes")
        for rating, note in notes:
            click.echo(f"  {rating} — {note}")


def _baseline_export_fields(
    prompt_tokens: int,
    completion_tokens: int,
    actual_cost: float | None,
) -> dict:
    from openshard.cost.baseline import compute_baseline_comparison
    cmp = compute_baseline_comparison(prompt_tokens, completion_tokens, actual_cost)
    if cmp is None:
        return {
            "frontier_baseline_cost_usd": None,
            "estimated_saving_usd": None,
            "estimated_saving_percent": None,
        }
    return {
        "frontier_baseline_cost_usd": cmp["frontier_baseline_cost_usd"],
        "estimated_saving_usd": cmp["estimated_saving_usd"],
        "estimated_saving_percent": cmp["estimated_saving_percent"],
    }


def _export_run_entry(entry: dict, include_notes: bool = False, include_timeline: bool = False, receipt=None) -> dict:  # receipt: ShardReceipt | None
    stage_runs = entry.get("stage_runs") or []
    is_ro = entry.get("routing_rationale") == "read-only analysis"

    def _stage_key(sr: dict) -> str | None:
        return sr.get("stage_type") or sr.get("stage")

    _ANALYSIS_STAGE_TYPES = {"analysis", "implementation", "execution", "work"}

    planning_model = next(
        (sr.get("model") for sr in stage_runs if _stage_key(sr) == "planning"), None
    )
    analysis_model = next(
        (sr.get("model") for sr in stage_runs if _stage_key(sr) in _ANALYSIS_STAGE_TYPES), None
    )

    feedback = entry.get("feedback") or {}
    tdr = entry.get("tier_dispatch_receipt") or {}

    row: dict = {
        "task":                      entry.get("task"),
        "timestamp":                 entry.get("timestamp"),
        "workflow":                  entry.get("workflow"),
        "execution_model":           entry.get("execution_model"),
        "planning_model":            planning_model,
        "analysis_model":            analysis_model,
        "routing_category":          entry.get("routing_category"),
        "routing_rationale":         entry.get("routing_rationale"),
        "routing_selected_model":    entry.get("routing_selected_model"),
        "routing_selected_provider": entry.get("routing_selected_provider"),
        "execution_profile":         entry.get("execution_profile"),
        "execution_mode_label":      _profile_display_label(entry.get("execution_profile"), is_readonly=is_ro),
        "verification_attempted":    entry.get("verification_attempted"),
        "verification_passed":       entry.get("verification_passed"),
        "duration_seconds":          entry.get("duration_seconds"),
        "total_cost_usd":            entry.get("estimated_cost"),
        "prompt_tokens":             entry.get("prompt_tokens"),
        "completion_tokens":         entry.get("completion_tokens"),
        "total_tokens":              entry.get("total_tokens"),
        "files_created":             entry.get("files_created"),
        "files_updated":             entry.get("files_updated"),
        "files_deleted":             entry.get("files_deleted"),
        "feedback_rating":           feedback.get("rating"),
        "feedback_note":             feedback.get("note"),
        "feedback_action":           feedback.get("action"),
        "correction_reason":         feedback.get("correction_reason"),
        "tier_dispatch_enabled":     tdr.get("enabled"),
        "tier_dispatch_applied":     tdr.get("applied"),
        "tier_dispatch_work_model":  tdr.get("executor_model"),
        "summary":                   entry.get("summary"),
        **_baseline_export_fields(
            entry.get("prompt_tokens") or 0,
            entry.get("completion_tokens") or 0,
            entry.get("estimated_cost"),
        ),
    }
    if include_notes:
        row["notes"] = entry.get("notes") or []
    if include_timeline:
        from openshard.run.timeline import project_timeline_for_export
        row["timeline"] = project_timeline_for_export(entry.get("run_timeline") or [])
    if receipt is not None:
        from dataclasses import asdict as _asdict
        row["provenance"] = [_asdict(p) for p in receipt.provenance]
    else:
        row["provenance"] = []
    return row


def _render_export_preview(rows: list[dict]) -> None:
    _TW, _MW, _MDW, _CW, _SW = 21, 10, 11, 10, 10
    click.echo("Export preview")
    click.echo(f"\nRuns: {len(rows)}\n")
    click.echo(
        "Time".ljust(_TW) + "Mode".ljust(_MW) + "Model".ljust(_MDW)
        + "Cost".ljust(_CW) + "Saving".ljust(_SW) + "Feedback"
    )
    for row in rows:
        ts = (row.get("timestamp") or "").rstrip("Z").replace("T", " ")[:16]
        mode = row.get("execution_mode_label") or "-"
        model_raw = row.get("execution_model") or ""
        model = _model_label(model_raw) if model_raw else "-"
        cost = row.get("total_cost_usd")
        cost_s = f"${cost:.4f}" if cost is not None else "-"
        pct = row.get("estimated_saving_percent")
        saving_s = f"{pct}%" if pct is not None else "-"
        feedback = row.get("feedback_rating") or "-"
        click.echo(
            ts.ljust(_TW) + mode.ljust(_MW) + model.ljust(_MDW)
            + cost_s.ljust(_CW) + saving_s.ljust(_SW) + feedback
        )


@cli.command("export-runs")
@click.option("--output", default=None, help="Write JSONL to this path instead of stdout.")
@click.option("--limit", default=None, type=click.IntRange(min=1), help="Export most recent N entries.")
@click.option("--with-notes", is_flag=True, default=False, help="Include run notes in export.")
@click.option("--preview", is_flag=True, default=False, help="Print a human-readable table instead of JSONL.")
def export_runs(output: str | None, limit: int | None, with_notes: bool, preview: bool) -> None:
    """Export run history as clean JSONL for eval analysis and review."""
    if preview and output:
        raise click.UsageError("--preview and --output cannot be used together; preview is terminal-only.")
    log_path = Path.cwd() / _LOG_PATH
    if not log_path.exists():
        click.echo("No run history found. Run a task first with 'openshard run'.")
        return
    entries = _load_run_entries(log_path)
    if not entries:
        click.echo("No runs recorded yet.")
        return
    if limit is not None:
        entries = entries[-limit:]
    rows = [_export_run_entry(e, include_notes=with_notes) for e in entries]
    if preview:
        _render_export_preview(rows)
        return
    lines = "\n".join(json.dumps(r) for r in rows)
    if output:
        output_path = Path(output)
        if output_path.parent != Path("."):
            output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(lines + "\n", encoding="utf-8")
    else:
        click.echo(lines)


@cli.command("interactions")
@click.option(
    "--last",
    "last_n",
    default=10,
    type=click.IntRange(min=1),
    show_default=True,
    help="Show the most recent N interaction events.",
)
def interactions_cmd(last_n: int) -> None:
    """Show recent developer interaction events."""
    from openshard.history.interactions import load_interaction_events
    events = load_interaction_events()
    if not events:
        click.echo("No interaction events recorded yet.")
        return
    recent = events[-last_n:]
    _TW, _ETW, _SW = 20, 30, 8
    click.echo("Time".ljust(_TW) + "Event Type".ljust(_ETW) + "Accept".ljust(_SW) + "Summary")
    for evt in recent:
        ts = (evt.timestamp or "").rstrip("Z").replace("T", " ")[:16]
        etype = (evt.event_type or "-")[:_ETW - 1]
        accepted = (
            "yes" if evt.accepted is True
            else "no" if evt.accepted is False
            else "-"
        )
        summary = (evt.summary or "")[:60]
        click.echo(ts.ljust(_TW) + etype.ljust(_ETW) + accepted.ljust(_SW) + summary)


@cli.command("export-interactions")
@click.option("--output", default=None, help="Write JSONL to this path instead of stdout.")
@click.option(
    "--redacted",
    is_flag=True,
    default=False,
    help="Replace summary with '[redacted]' and metadata with {}.",
)
def export_interactions(output: str | None, redacted: bool) -> None:
    """Export developer interaction events as JSONL."""
    from openshard.history.interactions import load_interaction_events, _event_to_dict
    events = load_interaction_events()
    if not events:
        click.echo("No interaction events recorded yet.")
        return
    rows: list[dict] = []
    for evt in events:
        d = _event_to_dict(evt)
        if redacted:
            d["summary"] = "[redacted]"
            d["correction_reason"] = None
            d["related_file_paths"] = []
            d["metadata"] = {}
        rows.append(d)
    lines = "\n".join(json.dumps(r) for r in rows)
    if output:
        output_path = Path(output)
        if output_path.parent != Path("."):
            output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(lines + "\n", encoding="utf-8")
    else:
        click.echo(lines)


@cli.command("failure-memory")
@click.option(
    "--last",
    "last_n",
    default=10,
    type=click.IntRange(min=1),
    show_default=True,
    help="Show the most recent N failure memory events.",
)
def failure_memory_cmd(last_n: int) -> None:
    """Show recent native verification failure events."""
    from openshard.history.failure_memory import recent_failure_memory
    events = recent_failure_memory(limit=last_n)
    if not events:
        click.echo("No failure memory events recorded yet.")
        return
    _TW, _FTW, _MW = 20, 18, 8
    click.echo(
        "Time".ljust(_TW) + "Failure Type".ljust(_FTW)
        + "Exit".ljust(_MW) + "Retry".ljust(_MW) + "Task"
    )
    for evt in events:
        ts = (evt.timestamp or "").rstrip("Z").replace("T", " ")[:16]
        ftype = (evt.failure_type or "-")[:_FTW - 1]
        retry_s = "yes" if evt.retry_attempted else "no"
        task_s = (evt.task_summary or "")[:50]
        click.echo(
            ts.ljust(_TW) + ftype.ljust(_FTW)
            + str(evt.exit_code).ljust(_MW) + retry_s.ljust(_MW) + task_s
        )


@cli.command("export-failure-memory")
@click.option("--output", default=None, help="Write JSONL to this path instead of stdout.")
@click.option(
    "--redacted",
    is_flag=True,
    default=False,
    help="Replace task_summary with '[redacted]' and model with 'redacted'.",
)
def export_failure_memory(output: str | None, redacted: bool) -> None:
    """Export native failure memory events as JSONL."""
    from openshard.history.failure_memory import load_failure_memory_events, _event_to_dict
    events = load_failure_memory_events()
    if not events:
        click.echo("No failure memory events recorded yet.")
        return
    rows: list[dict] = []
    for evt in events:
        d = _event_to_dict(evt)
        if redacted:
            d["task_summary"] = "[redacted]"
            d["model"] = "redacted"
        rows.append(d)
    lines = "\n".join(json.dumps(r) for r in rows)
    if output:
        output_path = Path(output)
        if output_path.parent != Path("."):
            output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(lines + "\n", encoding="utf-8")
    else:
        click.echo(lines)


def _demo_default() -> None:
    click.echo("OpenShard demo")
    click.echo("")
    click.echo("This walkthrough shows the main steps OpenShard can follow during a task.")
    click.echo("")
    click.echo("1. Understand the task")
    click.echo("   OpenShard reads your task prompt and decides whether it involves")
    click.echo("   reading (questions, explanations) or writing (code changes, new files).")
    click.echo("")
    click.echo("2. Choose a workflow")
    click.echo("   A workflow is selected — direct (single pass), staged (plan then code),")
    click.echo("   or native (agent loop). You can override this with --workflow.")
    click.echo("")
    click.echo("3. Choose models")
    click.echo("   One or more models are chosen based on task complexity and your config.")
    click.echo("   The planning step typically uses a strong reasoning model.")
    click.echo("")
    click.echo("4. Protect files")
    click.echo("   Read-only tasks (questions, explain, summarise) never write files.")
    click.echo("   Write-guarded paths block accidental changes outside the project root.")
    click.echo("")
    click.echo("5. Record the run")
    click.echo("   Each run is appended to .openshard/runs.jsonl with timing, model,")
    click.echo("   cost, and file counts so you can review history later.")
    click.echo("")
    click.echo("6. Capture feedback")
    click.echo("   After a run you can rate it with 'openshard feedback --rating good'.")
    click.echo("   Ratings are stored alongside the run entry.")


def _demo_readonly() -> None:
    click.echo("Scenario: readonly")
    click.echo("")
    click.echo("Read-only tasks are prompts that ask a question or request an explanation")
    click.echo("without asking OpenShard to change anything — for example:")
    click.echo("")
    click.echo('  openshard run "what does openshard/cli/main.py do?"')
    click.echo('  openshard run "explain the pipeline execution flow"')
    click.echo("")
    click.echo("OpenShard detects these tasks automatically and enforces two protections:")
    click.echo("")
    click.echo("  - File writes are blocked. Even if a model returns file changes, they")
    click.echo("    are discarded and a notice is shown.")
    click.echo("  - The model receives an explicit instruction not to return file changes.")
    click.echo("")
    click.echo("File protection behaviour")
    click.echo("  The write guard is enforced regardless of the --write flag.")
    click.echo("  This means a read-only task cannot accidentally write files even if")
    click.echo("  you pass --write on the command line.")


def _demo_tier_dispatch() -> None:
    click.echo("Scenario: tier-dispatch")
    click.echo("")
    click.echo("Tier dispatch is a model selection strategy where the routing stage")
    click.echo("assigns each part of a task to a tier (fast, standard, capable),")
    click.echo("then resolves those tier names to actual model IDs before execution.")
    click.echo("")
    click.echo("Model plan / dispatch")
    click.echo("  Planning usually uses the strongest reasoning model because it decides")
    click.echo("  how the task should be handled.")
    click.echo("  A work model is then dispatched for the main task. For standard tasks")
    click.echo("  this can be a balanced model like GLM-5.1; for harder tasks it can be")
    click.echo("  a stronger model.")
    click.echo("")
    click.echo("  To enable tier dispatch on a run:")
    click.echo('    openshard run "your task" --experimental-tier-dispatch')
    click.echo("")
    click.echo("  The dispatch decision is recorded in the run log and visible at:")
    click.echo("    openshard last --more")


def _demo_feedback() -> None:
    click.echo("Scenario: feedback")
    click.echo("")
    click.echo("After each run you can attach a developer rating to the run entry.")
    click.echo("This lets you track which tasks and models produced good results.")
    click.echo("")
    click.echo("Developer feedback capture")
    click.echo("  Rate the most recent run with one of three values:")
    click.echo("")
    click.echo("    openshard feedback --rating good")
    click.echo("    openshard feedback --rating mixed")
    click.echo("    openshard feedback --rating bad")
    click.echo("")
    click.echo("  Add an optional note:")
    click.echo('    openshard feedback --rating mixed --note "output was close but missed edge case"')
    click.echo("")
    click.echo("  Feedback is stored in .openshard/runs.jsonl alongside the run entry.")
    click.echo("  You can view it with 'openshard last --more'.")


@cli.command()
@click.option(
    "--scenario",
    type=click.Choice(["readonly", "tier-dispatch", "feedback"], case_sensitive=False),
    default=None,
    help="Show a focused walkthrough for a specific scenario.",
)
def demo(scenario: str | None) -> None:
    """Show a walkthrough of OpenShard concepts and common scenarios."""
    if scenario is None:
        _demo_default()
    elif scenario.lower() == "readonly":
        _demo_readonly()
    elif scenario.lower() == "tier-dispatch":
        _demo_tier_dispatch()
    else:
        _demo_feedback()


def _demo_run() -> None:
    click.echo("Task: Add rate limiting to the API gateway")
    click.echo("")
    click.echo("Execution")
    click.echo("  Mode: Run")
    click.echo("")
    click.echo("Routing")
    click.echo("  Category: standard")
    click.echo("  Initial candidate: Sonnet 4.6")
    click.echo("  Candidates: 8")
    click.echo("  Workflow: staged")
    click.echo("  Reason: standard coding task")
    click.echo("")
    click.echo("Model plan")
    click.echo("  Planning: Sonnet 4.6")
    click.echo("  Work: GLM-5.1")
    click.echo("  Validator: Sonnet 4.6 (reserved)")
    click.echo("")
    click.echo("Dispatch")
    click.echo("  Applied: yes")
    click.echo("  Source: demo")
    click.echo("  Work model: GLM-5.1")
    click.echo("  Initial candidate: Sonnet 4.6")
    click.echo("")
    click.echo("Verification")
    click.echo("  Name: tests")
    click.echo("  Safety: safe")
    click.echo("  Source: demo")
    click.echo("  Command: python -m pytest")
    click.echo("")
    click.echo("Time: 9.5s   Cost: $0.0133")
    click.echo("Feedback: openshard feedback --rating good")


@cli.command("demo-run")
def demo_run() -> None:
    """Show a realistic sample run without making provider calls or writing files."""
    _demo_run()


@cli.command()
def tui() -> None:
    """Launch the interactive OpenShard home screen."""
    try:
        from openshard.tui.app import OpenShardTui
    except ImportError:
        raise click.ClickException(
            "The TUI requires the 'textual' package. "
            "Please reinstall OpenShard or install textual."
        )
    OpenShardTui().run()


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


# Default location for locally-generated eval cases (sibling of eval-runs.jsonl).
_GENERATED_EVALS_DIR = Path(".openshard") / "evals" / "generated"


def _resolve_eval_output_path(
    eval_id: str, output: str | None
) -> tuple[Path, list[str]]:
    """Resolve the write target for an eval case, falling back when unsafe.

    Returns (path, warnings). A custom ``--output`` is honoured only when it is a
    safe relative path (no absolute/drive prefix, no ``..`` traversal); otherwise
    we fall back to the default generated location and record a warning. The
    eval id is already sanitised by the caller, so the default path never embeds
    unsanitised shard/task data.
    """
    default_path = Path.cwd() / _GENERATED_EVALS_DIR / f"{eval_id}.json"
    if not output:
        return default_path, []

    candidate = Path(output)
    normalized = output.replace("\\", "/")
    is_absolute = (
        candidate.is_absolute()
        or normalized.startswith("/")  # POSIX-absolute, even when run on Windows
        or (len(normalized) > 1 and normalized[1] == ":")  # drive-letter prefix
    )
    has_traversal = ".." in candidate.parts
    if is_absolute or has_traversal:
        return (
            default_path,
            ["Ignored unsafe --output path; wrote to default location instead."],
        )
    return Path.cwd() / candidate, []


def _write_eval_case(path: Path, case: dict, force: bool) -> None:
    """Write the eval case JSON to ``path``. Raises on collision unless ``force``."""
    if path.exists() and not force:
        raise FileExistsError(str(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(case, indent=2) + "\n", encoding="utf-8")


@eval.command("create-from-last")
@click.option("--json", "as_json", is_flag=True, default=False,
              help="Machine-readable output (valid JSON only).")
@click.option("--output", "output", default=None,
              help="Write the eval case to this relative path instead of the default.")
@click.option("--force", is_flag=True, default=False,
              help="Overwrite the output file if it already exists.")
def eval_create_from_last(as_json: bool, output: str | None, force: bool) -> None:
    """Convert the latest failed/rejected/partial Shard into a safe eval case.

    Local, deterministic, receipt-based. Reads the most recent run from history,
    classifies it with the failure taxonomy, and — only when it carries a
    failure/correction signal — writes a redacted, versioned eval-case JSON file
    under .openshard/evals/generated/. No network, no model calls; no secrets,
    raw file contents, diffs, transcripts, error messages, or absolute paths leak.
    """
    from openshard.history.shard_contract import build_shard_receipt
    from openshard.history.failures import classify_failure
    from openshard.evals.case_builder import build_eval_case, is_eligible

    command = "eval create-from-last"
    log_path = Path.cwd() / _LOG_PATH
    entries = _load_run_entries(log_path)

    if not entries:
        if as_json:
            payload = _machine_envelope(
                command, "not_eligible",
                eval_id=None, source_shard_id=None, failure_category=None,
                output_path_display=None,
                warnings=["No run history found."],
            )
            click.echo(json.dumps(payload, indent=2))
        else:
            click.echo("No eval case created because there is no run history.")
        return

    entry = entries[-1]
    receipt = build_shard_receipt(entry, index=len(entries) - 1)
    classification = classify_failure(entry, receipt)

    if not is_eligible(classification):
        message = (
            "No eval case created because the latest Shard has no "
            "failure/correction signal."
        )
        if as_json:
            payload = _machine_envelope(
                command, "not_eligible",
                shard_id=receipt.shard_id,
                eval_id=None,
                source_shard_id=receipt.shard_id,
                failure_category=classification.category,
                output_path_display=None,
            )
            click.echo(json.dumps(payload, indent=2))
        else:
            click.echo(message)
        return

    created_at = datetime.datetime.now(datetime.timezone.utc).isoformat()
    case = build_eval_case(receipt, classification, created_at)
    eval_id = case["eval_id"]
    target, warnings = _resolve_eval_output_path(eval_id, output)

    try:
        _write_eval_case(target, case, force)
    except FileExistsError:
        display = _safe_output_display(str(target))
        if as_json:
            payload = _machine_envelope(
                command, "error",
                shard_id=receipt.shard_id,
                eval_id=eval_id,
                source_shard_id=receipt.shard_id,
                failure_category=classification.category,
                output_path_display=display,
                warnings=warnings + [f"File already exists: {display}. Use --force to overwrite."],
            )
            click.echo(json.dumps(payload, indent=2))
        else:
            click.echo(f"Eval case not written: {display} already exists. Use --force to overwrite.")
        raise SystemExit(1)
    except OSError as exc:
        display = _safe_output_display(str(target))
        if as_json:
            payload = _machine_envelope(
                command, "error",
                shard_id=receipt.shard_id,
                eval_id=eval_id,
                source_shard_id=receipt.shard_id,
                failure_category=classification.category,
                output_path_display=display,
                warnings=warnings + [f"Could not write eval case to {display}: {type(exc).__name__}."],
            )
            click.echo(json.dumps(payload, indent=2))
        else:
            click.echo(f"Eval case not written: could not write to {display} ({type(exc).__name__}).")
        raise SystemExit(1)

    display = _safe_output_display(str(target))
    if as_json:
        payload = _machine_envelope(
            command, "created",
            shard_id=receipt.shard_id,
            eval_id=eval_id,
            source_shard_id=receipt.shard_id,
            failure_category=classification.category,
            output_path_display=display,
            warnings=warnings,
        )
        click.echo(json.dumps(payload, indent=2))
        return

    click.echo("Created eval case from the latest failed Shard.")
    click.echo(f"  eval id:          {eval_id}")
    click.echo(f"  source shard id:  {receipt.shard_id}")
    click.echo(f"  failure category: {classification.category}")
    click.echo(f"  output:           {display}")
    for warning in warnings:
        click.echo(f"  warning:          {warning}")


@cli.group("packs", invoke_without_command=True)
@click.pass_context
def packs(ctx: click.Context):
    """Workflow pack commands."""
    if ctx.invoked_subcommand is None:
        click.echo(ctx.get_help())


@packs.command("list")
def packs_list():
    """List all available workflow packs."""
    from openshard.workflow_packs.packs import load_packs

    packs_ = load_packs()
    col_id = 32
    col_title = 40
    for p in packs_:
        pid = p.id if len(p.id) <= col_id else p.id[: col_id - 1] + "..."
        ptitle = p.title if len(p.title) <= col_title else p.title[: col_title - 1] + "..."
        click.echo(f"  {pid:<{col_id}}  {ptitle:<{col_title}}  {p.category}")


@packs.command("show")
@click.argument("pack_id")
def packs_show(pack_id: str):
    """Show full metadata for a workflow pack."""
    from openshard.workflow_packs.packs import get_pack, load_packs

    try:
        p = get_pack(pack_id)
    except KeyError:
        available = ", ".join(p.id for p in load_packs())
        raise click.ClickException(f"Unknown pack {pack_id!r}. Available: {available}")

    click.echo(f"ID:                    {p.id}")
    click.echo(f"Title:                 {p.title}")
    click.echo(f"Category:              {p.category}")
    click.echo(f"Summary:               {p.summary}")
    click.echo(f"Recommended context:   {p.recommended_context}")
    click.echo(f"Expected receipt value: {p.expected_receipt_value}")
    click.echo(f"Safety notes:          {p.safety_notes}")
    click.echo(f"Tags:                  {', '.join(p.tags)}")


@packs.command("prompt")
@click.argument("pack_id")
def packs_prompt(pack_id: str):
    """Print only the prompt text for a workflow pack (ready to copy or run)."""
    from openshard.workflow_packs.packs import get_pack, load_packs

    try:
        p = get_pack(pack_id)
    except KeyError:
        available = ", ".join(p.id for p in load_packs())
        raise click.ClickException(f"Unknown pack {pack_id!r}. Available: {available}")

    click.echo(p.prompt)
    click.echo("")
    click.echo(
        "This command only prints the prompt. "
        "To run it, cd into the target repo and use `openshard tui` or `openshard run`."
    )


@cli.group()
def adapters() -> None:
    """External adapter utilities."""


@adapters.command("doctor")
def adapters_doctor() -> None:
    """Check external adapter availability and show setup guidance."""
    from openshard.execution.opencode_adapter import detect_opencode

    click.echo("\nOpenShard Adapter Doctor\n")
    click.echo("OpenCode")
    avail = detect_opencode()
    if avail.available:
        click.echo("  Status:  detected")
        click.echo(f"  Path:    {avail.path}")
    else:
        click.echo("  Status:  not installed")
        click.echo(f"  Reason:  {avail.reason}")
        click.echo("  Install options:")
        for opt in avail.install_guidance:
            click.echo(f"    {opt}")
        click.echo("  After installing, verify with:")
        click.echo("    opencode --version")
        click.echo("    openshard adapters doctor")
    click.echo("")


@cli.group()
def session() -> None:
    """Local session utilities."""


@session.command("infer")
@click.option(
    "--path",
    default=None,
    help="Override base directory (default: current working directory).",
)
def session_infer(path: str | None) -> None:
    """Infer behavioural signals from session events and write to session_signals.jsonl."""
    from openshard.history.session_signals import run_inference

    base_path = Path(path) if path else None
    signals = run_inference(base_path)
    click.echo(f"Inferred {len(signals)} signal(s).")


def _current_state() -> dict:
    """Build the shared onboarding state from the current on-disk config."""
    from openshard.config import onboarding as ob

    config, valid, path = load_config_safe()
    return ob.build_state(
        version=__version__,
        config_found=path is not None,
        config_path=path,
        config_valid=valid,
        onboarding=get_onboarding(config),
    )


def _echo_warnings_next_steps(state: dict) -> None:
    warnings = state.get("warnings") or []
    next_steps = state.get("next_steps") or []
    if warnings:
        click.echo("\nWarnings:")
        for w in warnings:
            click.echo(f"  ! {w}")
    if next_steps:
        click.echo("\nNext steps:")
        for s in next_steps:
            click.echo(f"  - {s}")


@cli.command()
@click.option("--json", "as_json", is_flag=True, default=False, help="Machine-readable output.")
@click.option("--yes", "-y", "assume_yes", is_flag=True, default=False,
              help="Non-interactive: apply defaults/flags without prompting.")
@click.option("--mode", "mode", default=None, help="Usage mode (see options).")
@click.option("--provider", "provider", default=None, help="Provider preference (see options).")
@click.option("--model-mode", "model_mode", default=None, help="Model mode (see options).")
@click.option("--output-mode", "output_mode", default=None, help="Output mode (see options).")
@click.option("--force", is_flag=True, default=False, help="Overwrite existing onboarding without prompting.")
def init(as_json: bool, assume_yes: bool, mode: str | None, provider: str | None,
         model_mode: str | None, output_mode: str | None, force: bool) -> None:
    """Set up OpenShard for first use (interactive, or --yes / --json)."""
    from openshard.config import onboarding as ob

    mode_keys = [k for k, _, _ in ob.MODES]
    provider_keys = [k for k, _, _ in ob.PROVIDERS]
    model_mode_keys = [k for k, _, _ in ob.MODEL_MODES]
    output_mode_keys = [k for k, _, _ in ob.OUTPUT_MODES]

    def _validate(name: str, value: str | None, allowed: list[str]) -> None:
        if value is not None and value not in allowed:
            raise click.BadParameter(
                f"'{value}' is not a valid {name}. Choose from: {', '.join(allowed)}.",
            )

    _validate("mode", mode, mode_keys)
    _validate("provider", provider, provider_keys)
    _validate("model-mode", model_mode, model_mode_keys)
    _validate("output-mode", output_mode, output_mode_keys)

    # --json without --yes: read-only discovery. Never writes.
    if as_json and not assume_yes:
        payload = {"options": ob.options_catalog(), "state": _current_state()}
        click.echo(json.dumps(payload, indent=2))
        return

    def _prompt(label: str, items: list[tuple[str, str, str]], default: str) -> str:
        click.echo(f"\n{label}:")
        for key, opt_label, desc in items:
            click.echo(f"  {key:<16} {opt_label} — {desc}")
        keys = [k for k, _, _ in items]
        return click.prompt("  Choice", type=click.Choice(keys), default=default,
                            show_choices=False)

    if assume_yes:
        sel_mode = mode or ("native" if ob.any_api_key_present() else "local_only")
        sel_provider = provider or ob.default_provider()
        sel_model_mode = model_mode or "balanced"
        sel_output_mode = output_mode or "human"
    else:
        sel_mode = mode or _prompt("Usage mode", ob.MODES,
                                   "native" if ob.any_api_key_present() else "local_only")
        sel_provider = provider or _prompt("Provider", ob.PROVIDERS, ob.default_provider())
        sel_model_mode = model_mode or _prompt("Model mode", ob.MODEL_MODES, "balanced")
        sel_output_mode = output_mode or _prompt("Output mode", ob.OUTPUT_MODES, "human")

        click.echo("\nSafety:")
        for note in ob.SAFETY_NOTES:
            click.echo(f"  - {note}")

    existing = find_config_path()
    overwrite_warning = None
    if existing is not None and not force:
        if assume_yes:
            overwrite_warning = (
                "Existing config found; onboarding settings were replaced "
                "(model settings preserved)."
            )
        else:
            if not click.confirm(
                "\nConfig already exists. Overwrite onboarding settings? "
                "(existing model settings are preserved)",
                default=True,
            ):
                click.echo("Aborted; no changes made.")
                return

    onboarding_block = {
        "schema_version": ob.SCHEMA_VERSION,
        "mode": sel_mode,
        "provider": sel_provider,
        "model_mode": sel_model_mode,
        "output_mode": sel_output_mode,
        "completed_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }

    # Merge over the full loaded base so model_tiers and friends are preserved.
    base = load_config_safe()[0]
    base["onboarding"] = onboarding_block
    written_path = save_config(base)

    config, valid, _ = load_config_safe()
    state = ob.build_state(
        version=__version__,
        config_found=True,
        config_path=written_path,
        config_valid=valid,
        onboarding=get_onboarding(config),
    )
    if overwrite_warning:
        state["warnings"].insert(0, overwrite_warning)

    if as_json:
        click.echo(json.dumps(state, indent=2))
        return

    click.echo(f"\nWrote {state['config_path_display']}")
    click.echo(f"  mode:        {sel_mode}")
    click.echo(f"  provider:    {sel_provider}")
    click.echo(f"  model_mode:  {sel_model_mode}")
    click.echo(f"  output_mode: {sel_output_mode}")
    _echo_warnings_next_steps(state)


@cli.command()
@click.option("--json", "as_json", is_flag=True, default=False, help="Machine-readable output.")
def doctor(as_json: bool) -> None:
    """Diagnose OpenShard configuration and setup state."""
    from openshard.config import onboarding as ob

    state = _current_state()
    state["git_repo"] = ob.detect_git_repo()

    if as_json:
        click.echo(json.dumps(state, indent=2))
        return

    click.echo("\nOpenShard Doctor\n")
    click.echo(f"  version:      {state['openshard_version']}")
    click.echo(f"  config found: {'yes' if state['config_found'] else 'no'}")
    click.echo(f"  config path:  {state['config_path_display'] or '-'}")
    click.echo(f"  config valid: {'yes' if state['config_valid'] else 'no'}")
    click.echo(f"  git repo:     {'yes' if state['git_repo'] else 'no'}")
    click.echo("\n  Onboarding:")
    click.echo(f"    mode:        {state['mode'] or '-'}")
    click.echo(f"    provider:    {state['provider'] or '-'}")
    click.echo(f"    model_mode:  {state['model_mode'] or '-'}")
    click.echo(f"    output_mode: {state['output_mode'] or '-'}")
    click.echo("\n  API keys (environment):")
    for prov, present in ob.api_key_present().items():
        click.echo(f"    {prov:<12} {'yes' if present else 'no'}")
    _echo_warnings_next_steps(state)
    click.echo("")


@cli.group("config")
def config_cmd() -> None:
    """Configuration utilities."""


@config_cmd.command("show")
@click.option("--json", "as_json", is_flag=True, default=False, help="Machine-readable output.")
def config_show(as_json: bool) -> None:
    """Show the active configuration with secrets redacted."""
    import yaml

    from openshard.config import onboarding as ob

    config, valid, _ = load_config_safe()
    safe = ob.redact(config)

    if as_json:
        click.echo(json.dumps(safe, indent=2))
        return

    if not valid:
        click.echo("Warning: config could not be parsed; showing safe defaults.\n")
    click.echo(yaml.safe_dump(safe, sort_keys=False, default_flow_style=False).rstrip())


if __name__ == "__main__":
    cli()
