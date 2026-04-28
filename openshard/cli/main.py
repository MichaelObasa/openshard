import datetime
import json
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

import click

from openshard.config.settings import get_api_key, get_anthropic_api_key, get_openai_api_key, load_config
from openshard.execution.gates import GateEvaluator, VALID_APPROVAL_MODES
from openshard.execution.generator import ChangedFile, ExecutionGenerator, ExecutionResult, check_stack_mismatch
from openshard.execution.opencode_executor import OpenCodeExecutor
from openshard.planning.generator import PlanGenerator
from openshard.providers.base import ProviderAuthError, ProviderError, ProviderRateLimitError
from openshard.providers.openrouter import MODEL_PRICING, compute_cost
from openshard.providers.manager import ProviderManager
from openshard.routing.engine import ESCALATION_CHAIN, MODEL_STRONG, RoutingDecision, route
from openshard.scoring.requirements import requirements_from_category
from openshard.scoring.scorer import ScoredRoutingResult, select_with_info
from openshard.execution.stages import (
    Stage, StageRun, split_task, route_stage, should_use_stages, run_planning_stage,
)
from openshard.analysis.repo import analyze_repo, RepoFacts
from openshard.history.metrics import load_runs
from openshard.history.adjustments import compute_history_adjustments, compute_history_adjustment_reasons
from openshard.routing.workflow_selector import WorkflowHistorySummary, build_workflow_history_summary, select_workflow


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
        "staged (planning then implementation), native (native agent — not yet available), "
        "opencode (OpenCode CLI), claude-code (not yet available), codex (not yet available)."
    ),
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
def run(task: str, write: bool, verify: bool, dry_run: bool, more: bool, full: bool, no_shrink: bool, workflow: str | None, executor: str | None, plan_flag: bool, approval: str | None, provider: str | None, history_scoring: bool):
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
        _cfg_workflow = _cfg.get("workflow", "").strip().lower()
        _cfg_executor_legacy = _cfg.get("executor", "direct").strip().lower()
        _use_history_scoring = history_scoring or bool(_cfg.get("history_scoring", False))
        _policy_executor, _policy_reason = _suggest_executor(task)

        # --workflow > --executor (deprecated) > config.workflow > config.executor > auto
        _show_executor_deprecation = False
        if workflow is not None:
            _wf = workflow.lower()
            if _wf in ("native", "claude-code", "codex"):
                raise click.ClickException(
                    f"--workflow {_wf!r} is not yet available. "
                    "Use: auto, direct, staged, or opencode."
                )
            effective_workflow = _wf
            _policy_reason = ""
        elif executor is not None:
            _show_executor_deprecation = True
            effective_workflow = executor.lower()
            _policy_reason = ""
        elif _cfg_workflow:
            if _cfg_workflow in ("native", "claude-code", "codex"):
                raise click.ClickException(
                    f"config.workflow: {_cfg_workflow!r} is not yet available. "
                    "Use: auto, direct, staged, or opencode."
                )
            effective_workflow = _cfg_workflow
            _policy_reason = ""
        elif _cfg_executor_legacy != "direct":
            effective_workflow = _cfg_executor_legacy
            _policy_reason = ""
        else:
            effective_workflow = "auto"

        if _show_executor_deprecation:
            click.echo(
                "  [deprecation] --executor is deprecated; use --workflow instead.",
                err=True,
            )

        # Map workflow to effective_executor and stage-forcing behaviour.
        # _force_stages=True → always stage; False → never stage; None → auto (routing-driven).
        _force_stages: bool | None = None
        if effective_workflow == "auto":
            effective_executor = _policy_executor
        elif effective_workflow == "direct":
            effective_executor = "direct"
            _force_stages = False
        elif effective_workflow == "staged":
            effective_executor = "direct"
            _force_stages = True
            _policy_reason = ""
        elif effective_workflow == "opencode":
            effective_executor = "opencode"
            _force_stages = False
            _policy_reason = ""
        else:
            # Fallback (shouldn't reach here after earlier validation)
            effective_executor = "direct"

        # Resolve provider (only applies to direct executor)
        _provider_instance = None
        _provider_name = (provider or "openrouter").lower()
        if effective_executor != "opencode":
            if _provider_name == "anthropic":
                from openshard.providers.anthropic import AnthropicProvider
                _provider_instance = AnthropicProvider(get_anthropic_api_key())
            elif _provider_name == "openai":
                from openshard.providers.openai import OpenAIProvider
                _provider_instance = OpenAIProvider(get_openai_api_key())

        if effective_executor == "opencode":
            generator: ExecutionGenerator | OpenCodeExecutor = OpenCodeExecutor()
        else:
            generator = ExecutionGenerator(provider=_provider_instance)
    except (ValueError, RuntimeError) as exc:
        raise click.ClickException(str(exc))

    opencode_mode = (effective_executor == "opencode")
    routing_decision: RoutingDecision | None = route(task) if not opencode_mode else None
    # When using a third-party provider, non-native routed models are not available.
    # Fall back to a sensible default for each provider.
    if (
        _provider_name == "anthropic"
        and routing_decision is not None
        and not routing_decision.model.startswith("anthropic/")
    ):
        routing_decision = RoutingDecision(
            model=generator.model,
            category=routing_decision.category,
            rationale=routing_decision.rationale,
        )
    elif (
        _provider_name == "openai"
        and routing_decision is not None
        and not routing_decision.model.startswith("openai/")
    ):
        from openshard.providers.openai import DEFAULT_MODEL as _OPENAI_DEFAULT
        routing_decision = RoutingDecision(
            model=_OPENAI_DEFAULT,
            category=routing_decision.category,
            rationale=routing_decision.rationale,
        )
    if _force_stages is True:
        _use_stages = not opencode_mode
    elif _force_stages is False:
        _use_stages = False
    else:
        _use_stages = None  # resolved below after history + repo_facts
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
    _routed_model = routing_decision.model if routing_decision else None
    _scored: ScoredRoutingResult | None = None
    _runs: list[dict] = []

    # Attempt scored model selection; fall back silently to keyword routing.
    if not opencode_mode and routing_decision is not None:
        try:
            _mgr = ProviderManager()
            _inv = _mgr.get_inventory()
            _reqs = requirements_from_category(routing_decision.category)
            _entries = [e for e in _inv.models if e.provider == _provider_name]
            _hist_adjustments: dict[str, float] | None = None
            _hist_reasons: dict[str, str] = {}
            if _use_history_scoring:
                try:
                    _runs = load_runs()
                    _hist_adjustments = compute_history_adjustments(_runs)
                    _hist_reasons = compute_history_adjustment_reasons(_runs)
                except Exception:
                    _runs = []
            _scored = select_with_info(_entries, _reqs, routing_decision.category, history_adjustments=_hist_adjustments)
            if (
                _scored.selected_model is not None
                and _mgr.providers.get(_scored.selected_provider) is not None
            ):
                _routed_model = _scored.selected_model
            else:
                _scored = ScoredRoutingResult(
                    category=_scored.category,
                    requirements=_scored.requirements,
                    candidate_count=_scored.candidate_count,
                    selected_model=None,
                    selected_provider=None,
                    used_fallback=True,
                )
        except Exception:
            pass

    _repo_facts: RepoFacts | None = None
    try:
        _repo_facts = analyze_repo(Path.cwd())
    except Exception:
        pass

    if _use_stages is None:
        _category = routing_decision.category if routing_decision else "standard"
        _workflow_history_summary: WorkflowHistorySummary | None = None
        if _use_history_scoring and _runs:
            _workflow_history_summary = build_workflow_history_summary(_runs, _category)
        _wf_decision = select_workflow(
            category=_category,
            repo_facts=_repo_facts,
            history_summary=_workflow_history_summary,
            verify_enabled=verify,
        )
        _wf_choice = _wf_decision.workflow
        _wf_reason = _wf_decision.reason
        _use_stages = not opencode_mode and (_wf_choice == "staged")

    _cfg_approval = _cfg.get("approval_mode", "auto").strip().lower()
    if _cfg_approval not in VALID_APPROVAL_MODES:
        raise click.ClickException(
            f"Invalid approval_mode {_cfg_approval!r} in config. "
            f"Valid values: {', '.join(sorted(VALID_APPROVAL_MODES))}"
        )
    _approval_mode = approval.lower() if approval else _cfg_approval
    _cost_threshold = float(_cfg.get("cost_gate_threshold", 0.10))
    gate = GateEvaluator(
        approval_mode=_approval_mode,
        risky_paths=_repo_facts.risky_paths if _repo_facts is not None else [],
        cost_threshold=_cost_threshold,
    )

    # Routing line is printed after scoring so the model label reflects the actual selection.
    if routing_decision is not None:
        if detail != "default":
            click.echo(f"  [routing] {_model_label(_routed_model)} - {routing_decision.rationale}")
        hint = _pre_run_cost_hint(routing_decision.model, task)
        if hint:
            click.echo(f"  Cost estimate: {hint}")
            _cost_val = _parse_cost_hint(hint)
            if _cost_val is not None:
                _cost_dec = gate.check_high_cost(_cost_val)
                if _cost_dec.required:
                    confirm_or_abort(_cost_dec.reason)

    if detail != "default" and _scored is not None:
        _req = _scored.requirements
        _req_parts: list[str] = []
        if _req.security_sensitive:
            _req_parts.append("security_sensitive")
        if _req.needs_vision:
            _req_parts.append("needs_vision")
        if _req.needs_tools:
            _req_parts.append("needs_tools")
        if _req.complexity != "standard":
            _req_parts.append(_req.complexity)
        if _req.min_context_window:
            _req_parts.append(f"ctx>={_req.min_context_window // 1_000}k")
        _req_str = ", ".join(_req_parts) if _req_parts else "none"
        click.echo(f"  [routing] category: {_scored.category}, requirements: {_req_str}")
        if _scored.used_fallback:
            click.echo(f"  [routing] candidates: {_scored.candidate_count} → fallback (keyword routing)")
        else:
            _cost_str = f"cost: ${_scored.selected_cost_per_m:.2f}/M" if _scored.selected_cost_per_m is not None else "cost: unknown"
            click.echo(f"  [routing] candidates: {_scored.candidate_count} → {_model_label(_scored.selected_model)} ({_cost_str})")
        if _use_history_scoring:
            click.echo("  [routing] history scoring: enabled")
            _nonzero = [
                (m, adj)
                for m, adj in _scored.history_adjustments.items()
                if m in set(_scored.candidates) and adj != 0.0
            ]
            for _hm, _hadj in _nonzero:
                _rsn = _hist_reasons.get(_hm, "")
                _rsn_str = f" ({_rsn})" if _rsn else ""
                _marker = " ← selected" if _hm == _scored.selected_model else ""
                click.echo(f"  [routing] history: {_model_label(_hm)}: {_hadj:+.1f}{_rsn_str}{_marker}")

    if detail != "default":
        if _force_stages is None:
            _wf_display = _wf_choice
            _wf_display_reason = _wf_reason
        elif _force_stages:
            _wf_display = "staged"
            _wf_display_reason = "forced by workflow setting"
        else:
            _wf_display = "direct"
            _wf_display_reason = "forced by workflow setting"
        click.echo(f"  [routing] workflow: {_wf_display}")
        click.echo(f"  [routing] reason: {_wf_display_reason}")

    if detail == "default":
        _routing_msg = _build_routing_line(routing_decision, _use_stages, actual_model=_routed_model)
        if _routing_msg:
            click.echo(_routing_msg)

    # --- --plan gate: show plan and prompt for approval before executing --------
    _impl_task = task          # may be augmented with a plan
    _plan_already_done = False

    if plan_flag:
        if effective_workflow == "opencode":
            _shape_desc = "opencode (delegated to OpenCode CLI)"
        elif _use_stages:
            _shape_desc = "staged (planning → implementation)"
        else:
            _shape_desc = "direct single-pass"
        click.echo(f"\n  Workflow:  {_shape_desc}")
        click.echo(f"  Model:     {_model_label(_routed_model or 'unknown')}")
        _hint = _pre_run_cost_hint(routing_decision.model if routing_decision else "", task)
        if _hint:
            click.echo(f"  Est. cost: {_hint}")
        if not opencode_mode:
            spinner.start("Planning - generating implementation plan")
            try:
                _plan_text, _ = run_planning_stage(generator.client, task)
                _impl_task = (
                    f"Task: {task}\n\nImplementation plan:\n{_plan_text}"
                    "\n\nExecute the task following the plan above."
                )
                _plan_already_done = True
                spinner.stop()
                click.echo(f"\n  Plan:\n{_plan_text}")
            except Exception:
                spinner.stop()
                click.echo("  [plan] planning call failed; will proceed without pre-run plan")
        click.echo("")
        if not click.confirm("Proceed?", default=False):
            raise click.Abort()

    # --- Stage-based execution (direct mode, security/complex tasks) ----------
    result = None

    if _use_stages:
        stages = split_task(task)
        for _stage in stages:
            _stage_t0 = time.time()

            if _stage.stage_type == "planning":
                if _plan_already_done:
                    # --plan gate already ran the planning call; skip to avoid double billing.
                    continue
                spinner.start("Planning - mapping out implementation approach")
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
                except ProviderError:
                    _impl_task = task   # planning failed — fall back to plain task
                    if detail == "full":
                        click.echo("  [stages] planning call failed, continuing without plan")
                finally:
                    spinner.stop()

            elif _stage.stage_type == "implementation":
                _stage_model = _routed_model or route_stage(_stage)
                spinner.start(_exec_message(
                    _stage_model,
                    routing_decision.rationale if routing_decision else "",
                ))
                try:
                    result = generator.generate(_impl_task, model=_stage_model, repo_facts=_repo_facts)
                except RuntimeError as exc:
                    raise click.ClickException(str(exc))
                except ProviderAuthError:
                    raise click.ClickException(
                        "Authentication failed. Check that your provider API key is valid."
                    )
                except ProviderRateLimitError:
                    raise click.ClickException("Rate limit exceeded. Wait a moment then try again.")
                except ProviderError as exc:
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
        _single_msg = (
            _exec_message(_routed_model, routing_decision.rationale)
            if routing_decision is not None
            else ("Executing - running with OpenCode" if opencode_mode else "Executing - running task")
        )
        spinner.start(_single_msg)
        try:
            if opencode_mode:
                result = generator.generate(task, workspace=workspace)
            else:
                result = generator.generate(task, model=_routed_model, repo_facts=_repo_facts)
        except RuntimeError as exc:
            raise click.ClickException(str(exc))
        except ProviderAuthError:
            raise click.ClickException(
                "Authentication failed. Check that your provider API key is valid."
            )
        except ProviderRateLimitError:
            raise click.ClickException("Rate limit exceeded. Wait a moment then try again.")
        except ProviderError as exc:
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

    if not opencode_mode and _repo_facts is not None:
        _mismatches = check_stack_mismatch(final_files, _repo_facts)
        _sm_dec = gate.check_stack_mismatch(_mismatches)
        if _sm_dec.required:
            _lang_str = ", ".join(_repo_facts.languages) if _repo_facts.languages else "unknown"
            confirm_or_abort(f"{_sm_dec.reason} (repo stack: {_lang_str})")

    click.echo("\nDone")
    click.echo(result.summary)

    _model_line = _build_model_line(routing_decision, stage_runs, model=_routed_model)
    if _model_line:
        click.echo(f"\n{_model_line}")

    # Stages — shown before file count so the reader sees what ran (--more only)
    if detail != "default" and stage_runs:
        click.echo("\nStages")
        for sr in stage_runs:
            _sr_cost = f"${sr.cost:.4f}" if sr.cost is not None else "-"
            click.echo(f"  {sr.stage.stage_type.capitalize()} ({_model_label(sr.model)}): {sr.duration:.1f}s, {_sr_cost}")

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

    if detail != "default":
        try:
            _render_repo_summary(analyze_repo(Path.cwd()))
        except Exception:
            click.echo("\n  [repo] Repo summary unavailable")

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
                     stage_runs=stage_runs, routing_decision=routing_decision,
                     _scored=_scored, repo_facts=_repo_facts)
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
            _file_paths = [f.path for f in result.files if f.path]
            _fw_dec = gate.check_file_write(_file_paths)
            _rp_dec = gate.check_risky_paths(_file_paths)
            if _fw_dec.required:
                confirm_or_abort(_fw_dec.reason)
            elif _rp_dec.required:
                confirm_or_abort(_rp_dec.reason)
            _write_files(result.files, workspace)
        # OpenCode: workspace already created and populated before generate().

    if write and verify:
        click.echo("")
        _vcmd = _detect_command(workspace)
        if _vcmd:
            _sc_dec = gate.check_shell_command(" ".join(_vcmd))
            if _sc_dec.required:
                confirm_or_abort(_sc_dec.reason)
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
            spinner.start(f"Retrying - escalating to {_model_label(_esc_model)}")
            try:
                if opencode_mode:
                    _last_attempt = generator.generate(
                        retry_prompt, model=_esc_model, workspace=workspace
                    )
                else:
                    _last_attempt = generator.generate(retry_prompt, model=_esc_model)
            except RuntimeError as exc:
                raise click.ClickException(str(exc))
            except ProviderAuthError:
                raise click.ClickException(
                    "Authentication failed. Check that your provider API key is valid."
                )
            except ProviderRateLimitError:
                raise click.ClickException("Rate limit exceeded. Wait a moment then try again.")
            except ProviderError as exc:
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
                         stage_runs=stage_runs, routing_decision=routing_decision,
                         _scored=_scored, repo_facts=_repo_facts)
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
                 stage_runs=stage_runs, routing_decision=routing_decision,
                 _scored=_scored, repo_facts=_repo_facts)
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
    _scored: ScoredRoutingResult | None = None,
    repo_facts: RepoFacts | None = None,
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
    if _scored is not None:
        entry["routing_category"] = _scored.category
        entry["routing_candidate_count"] = _scored.candidate_count
        entry["routing_selected_model"] = _scored.selected_model
        entry["routing_selected_provider"] = _scored.selected_provider
        entry["routing_used_fallback"] = _scored.used_fallback
        if _scored.candidates:
            entry["routing_candidates"] = _scored.candidates
        if _scored.scores:
            entry["routing_scores"] = _scored.scores
        if _scored.scores_raw:
            entry["routing_scores_raw"] = _scored.scores_raw
        if _scored.history_adjustments:
            entry["routing_adjustments"] = _scored.history_adjustments
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
    if repo_facts is not None:
        entry["repo_facts"] = {
            "languages": repo_facts.languages,
            "package_files": repo_facts.package_files,
            "framework": repo_facts.framework,
            "test_command": repo_facts.test_command,
            "risky_paths_count": len(repo_facts.risky_paths),
            "risky_paths_sample": repo_facts.risky_paths[:3],
            "changed_files_count": len(repo_facts.changed_files),
            "changed_files_sample": repo_facts.changed_files[:3],
        }

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
    if retry_triggered:
        click.echo(f"Fixer model: {_model_label(generator.fixer_model)}")
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


def _render_repo_summary(facts: RepoFacts) -> None:
    click.echo("\nRepo")
    if facts.languages:
        click.echo(f"  Languages: {', '.join(facts.languages)}")
    if facts.package_files:
        click.echo(f"  Packages: {', '.join(facts.package_files)}")
    if facts.framework:
        click.echo(f"  Framework: {facts.framework}")
    if facts.test_command:
        click.echo(f"  Tests: {facts.test_command}")
    if facts.risky_paths:
        n = len(facts.risky_paths)
        sample = ", ".join(facts.risky_paths[:3])
        suffix = f" + {n - 3} more" if n > 3 else ""
        click.echo(f"  Risky: {n} paths  ({sample}{suffix})")
    if facts.changed_files:
        n = len(facts.changed_files)
        sample = ", ".join(facts.changed_files[:3])
        suffix = f" + {n - 3} more" if n > 3 else ""
        click.echo(f"  Changed: {n} files  ({sample}{suffix})")


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

_SYSTEM_OVERHEAD_TOKENS = 500   # execution system-prompt is substantial


def _pre_run_cost_hint(model: str, task: str) -> str | None:
    """Return a conservative pre-run cost range, or None if pricing is unknown.

    Assumes 300–1500 completion tokens — small code output to a full module.
    """
    if model not in MODEL_PRICING:
        return None
    prompt_tokens = _SYSTEM_OVERHEAD_TOKENS + len(task) // 4
    cost_low  = compute_cost(model, prompt_tokens, 300)
    cost_high = compute_cost(model, prompt_tokens, 1500)
    if cost_low is None or cost_high is None:
        return None
    return f"~${cost_low:.4f}-${cost_high:.4f}"


def _parse_cost_hint(hint: str | None) -> float | None:
    # Temporary: parses dollar amount from formatted hint string.
    # TODO: replace with structured cost metadata from the routing layer.
    if not hint:
        return None
    m = re.search(r'\$([\d.]+)', hint)
    return float(m.group(1)) if m else None


_MODEL_SHORT: dict[str, str] = {
    "deepseek/deepseek-v4-flash":      "DeepSeek V4 Flash",
    "deepseek/deepseek-v4-pro":        "DeepSeek V4 Pro",
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


_ABBREV_WORDS = {"gpt", "llm", "ai", "api", "url", "id", "ui", "ml"}


def _format_model_slug(raw: str) -> str:
    """Format an unknown model ID into a readable label.

    gpt-5.4-nano  -> GPT-5.4 Nano
    gemini-2.0-flash -> Gemini 2.0 Flash
    llama-3.3-70b -> Llama 3.3 70B
    """
    parts = [p for p in raw.split("/")[-1].split("-") if p]
    tagged: list[tuple[str, str]] = []
    for part in parts:
        lower = part.lower()
        if lower in _ABBREV_WORDS:
            tagged.append(("abbrev", part.upper()))
        elif re.match(r"^v\d", lower):
            tagged.append(("version", part[0].upper() + part[1:]))
        elif re.match(r"^\d+[a-z]+$", lower):
            tagged.append(("version", re.sub(r"[a-z]+$", lambda m: m.group().upper(), part)))
        elif part[0].isdigit():
            tagged.append(("version", part))
        else:
            tagged.append(("word", part.capitalize()))
    out = ""
    for i, (kind, text) in enumerate(tagged):
        if i == 0:
            out = text
        elif kind == "version" and tagged[i - 1][0] == "abbrev":
            out += "-" + text
        else:
            out += " " + text
    return out


def _model_label(model: str) -> str:
    return _MODEL_SHORT.get(model, _format_model_slug(model))


def _build_routing_line(
    routing_decision: "RoutingDecision | None",
    use_stages: bool,
    actual_model: str | None = None,
) -> str | None:
    """One-line routing summary shown in default output before execution starts."""
    if routing_decision is None:
        return None
    impl_label = _model_label(actual_model or routing_decision.model)
    reason = _RATIONALE_SHORT.get(routing_decision.rationale, routing_decision.category)
    if use_stages:
        plan_label = _model_label(MODEL_STRONG)
        if plan_label == impl_label:
            return f"  Routing - {impl_label} for planning and {reason}"
        return f"  Routing - {plan_label} for planning -> {impl_label} for {reason}"
    return f"  Routing - {impl_label} for {reason}"


def _exec_message(model: str, rationale: str) -> str:
    """Human-readable spinner message for the execution phase."""
    label = _model_label(model)
    desc = {
        "security-sensitive code requires careful reasoning": f"{label} handling security-sensitive logic",
        "UI or visual task routed to multimodal specialist":  f"{label} handling UI work",
        "multi-file or long-horizon task":                    f"{label} working through multi-file changes",
        "low-risk boilerplate task":                          f"{label} generating boilerplate",
        "standard feature implementation":                    f"{label} writing implementation",
    }
    return "Executing - " + desc.get(rationale, f"{label} running task")


def _build_model_line(
    routing_decision: "RoutingDecision | None",
    stage_runs: "list[StageRun]",
    model: str | None = None,
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
        label = _model_label(model or routing_decision.model)
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
    """Animated progress line: looping dots + elapsed time, updates in place."""

    _DOTS = [".", "..", "..."]

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
        sys.stdout.write("\r" + " " * 82 + "\r")
        sys.stdout.flush()

    def _run(self) -> None:
        i = 0
        while not self._stop.wait(0.4):
            elapsed = time.time() - self._t0
            dots = self._DOTS[i % len(self._DOTS)]
            line = f"  {self.phase}{dots}   {elapsed:.1f}s"
            sys.stdout.write(f"\r{line:<82}")
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
        **({"encoding": "utf-8", "errors": "replace"} if capture else {}),
    )
    if capture:
        return proc.returncode, proc.stdout or ""

    if proc.returncode == 0:
        click.echo(f"  {label} passed")
    else:
        click.echo(f"  {label} failed (exit code {proc.returncode})")
    return proc.returncode


def confirm_or_abort(reason: str) -> None:
    click.echo(f"\n[gate] {reason}")
    if not click.confirm("Proceed?", default=False):
        click.echo("Aborted!")
        raise SystemExit(0)


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
        mid = model_id if len(model_id) <= col_model else model_id[: col_model - 1] + "…"
        click.echo(f"  {mid:<{col_model}}  {runs_n:>5}  {avg_cost:>9}  {avg_dur:>8}  {pass_rate:>9}  {retry:>6}")


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
            click.echo(f"  [routing] candidates: {entry.get('routing_candidate_count')} → fallback (keyword routing)")
        elif entry.get("routing_selected_model"):
            _prov = entry.get("routing_selected_provider")
            _prov_suffix = f" ({_prov})" if _prov else ""
            click.echo(f"  [routing] candidates: {entry.get('routing_candidate_count')} → {_model_label(entry['routing_selected_model'])}{_prov_suffix}")

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