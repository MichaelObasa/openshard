import datetime
import json
import re
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import click

from openshard.analysis.repo import RepoFacts, analyze_repo
from openshard.cli.run_output import (
    _Spinner,
    _build_model_line,
    _build_routing_line,
    _exec_message,
    _model_label,
    _print_dry_run,
    _print_shrunk,
    _print_summary,
    _render_repo_summary,
    _should_shrink,
    _truncate_note,
)
from openshard.config.settings import get_anthropic_api_key, get_openai_api_key
from openshard.execution.gates import GateEvaluator, VALID_APPROVAL_MODES
from openshard.execution.generator import (
    ChangedFile,
    ExecutionGenerator,
    ExecutionResult,
    check_stack_mismatch,
)
from openshard.execution.opencode_executor import OpenCodeExecutor
from openshard.native.context import NativeVerificationLoop, render_verification_failure_context
from openshard.native.executor import NativeAgentExecutor
from openshard.execution.stages import StageRun, split_task, route_stage, run_planning_stage
from openshard.history.adjustments import (
    compute_history_adjustments,
    compute_history_adjustment_reasons,
)
from openshard.history.metrics import load_runs
from openshard.providers.base import ProviderAuthError, ProviderError, ProviderRateLimitError
from openshard.providers.manager import ProviderManager
from openshard.providers.openrouter import MODEL_PRICING, compute_cost
from openshard.routing.engine import ESCALATION_CHAIN, RoutingDecision, route
from openshard.routing.profiles import (
    ProfileDecision,
    ProfileHistorySummary,
    build_profile_history_summary,
    select_profile,
)
from openshard.routing.workflow_selector import (
    WorkflowHistorySummary,
    build_workflow_history_summary,
    select_workflow,
)
from openshard.scoring.requirements import requirements_from_category
from openshard.scoring.scorer import ScoredRoutingResult, select_with_info
from openshard.security.paths import resolve_safe_repo_path, UnsafePathError
from openshard.skills.context import build_skills_context
from openshard.skills.discovery import discover_skills
from openshard.skills.matcher import MatchedSkill, match_skills
from openshard.verification.executor import run_verification_plan as _run_verification_plan, confirm_or_abort  # noqa: F401
from openshard.verification.plan import CommandSafety, VerificationPlan, build_verification_plan

@dataclass
class RunResult:
    """Structured return value from RunPipeline.run()."""
    exit_code: int = 0
    start: float = 0.0
    generator: Any = None
    retry_triggered: bool = False
    final_files: list = field(default_factory=list)
    usage: Any = None
    retry_usage: Any = None
    routed_model: str | None = None
    stage_runs: list = field(default_factory=list)
    routing_decision: Any = None
    scored: Any = None
    repo_facts: Any = None
    matched_skills: list = field(default_factory=list)
    profile_decision: Any = None
    verification_plan: Any = None
    verification_attempted: bool = False
    verification_passed: bool | None = None
    workspace: Any = None
    result_summary: str = ""
    result_notes: list = field(default_factory=list)
    native_meta: Any = None


class RunPipeline:
    """Owns the run lifecycle: routing, execution, write/verify/retry, and logging."""

    def __init__(
        self,
        config: dict,
        *,
        write: bool,
        verify: bool,
        dry_run: bool,
        no_shrink: bool,
        workflow: str | None,
        profile: str | None,
        executor: str | None,
        plan_flag: bool,
        approval: str | None,
        provider: str | None,
        history_scoring: bool,
        eval_scoring: bool,
        detail: str,
        native_backend: str | None = None,
        experimental_deepagents_run: bool = False,
    ) -> None:
        self._config = config
        self._write = write
        self._verify = verify
        self._dry_run = dry_run
        self._no_shrink = no_shrink
        self._workflow = workflow
        self._profile = profile
        self._executor = executor
        self._plan_flag = plan_flag
        self._approval = approval
        self._provider = provider
        self._history_scoring = history_scoring
        self._eval_scoring = eval_scoring
        self._detail = detail
        self._native_backend = native_backend or "builtin"
        self._experimental_deepagents_run = experimental_deepagents_run

    def run(self, task: str) -> RunResult:  # noqa: C901
        result_obj = RunResult()
        detail = self._detail
        write = self._write
        verify = self._verify
        dry_run = self._dry_run
        no_shrink = self._no_shrink
        workflow = self._workflow
        profile = self._profile
        executor = self._executor
        plan_flag = self._plan_flag
        approval = self._approval
        provider = self._provider
        history_scoring = self._history_scoring
        eval_scoring = self._eval_scoring

        start = time.time()
        result_obj.start = start
        retry_triggered = False
        workspace: Path | None = None
        verification_passed: bool | None = None
        usage = None
        retry_usage = None

        try:
            _cfg = self._config
            _cfg_workflow = _cfg.get("workflow", "").strip().lower()
            _cfg_executor_legacy = _cfg.get("executor", "direct").strip().lower()
            _use_history_scoring = history_scoring or bool(_cfg.get("history_scoring", False))
            _use_eval_scoring = eval_scoring or bool(_cfg.get("eval_scoring", False))
            _policy_executor, _policy_reason = _suggest_executor(task)

            # --workflow > --executor (deprecated) > config.workflow > config.executor > auto
            _show_executor_deprecation = False
            if workflow is not None:
                _wf = workflow.lower()
                if _wf in ("claude-code", "codex"):
                    raise click.ClickException(
                        f"--workflow {_wf!r} is not yet available. "
                        "Use: auto, direct, staged, opencode, or native."
                    )
                effective_workflow = _wf
                _policy_reason = ""
            elif executor is not None:
                _show_executor_deprecation = True
                effective_workflow = executor.lower()
                _policy_reason = ""
            elif _cfg_workflow:
                if _cfg_workflow in ("claude-code", "codex"):
                    raise click.ClickException(
                        f"config.workflow: {_cfg_workflow!r} is not yet available. "
                        "Use: auto, direct, staged, opencode, or native."
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
            elif effective_workflow == "native":
                effective_executor = "native"
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
                generator: ExecutionGenerator | OpenCodeExecutor | NativeAgentExecutor = OpenCodeExecutor()
            elif effective_executor == "native":
                generator = NativeAgentExecutor(
                    provider=_provider_instance,
                    repo_root=Path.cwd(),
                    backend_name=self._native_backend,
                    experimental_deepagents_run=self._experimental_deepagents_run,
                )
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
                _eval_adjustments: dict[str, float] = {}
                _eval_reasons: dict[str, str] = {}
                _cat_adjustments: dict[str, float] = {}
                _cat_reasons: dict[str, str] = {}
                _eval_records: list[dict] = []
                if _use_eval_scoring:
                    try:
                        from openshard.evals.stats import EVAL_RUNS_PATH, compute_eval_stats, load_eval_runs
                        from openshard.evals.adjustments import compute_eval_adjustments, compute_eval_adjustment_reasons
                        _eval_records = load_eval_runs(Path.cwd() / EVAL_RUNS_PATH)
                        _eval_stats_data = compute_eval_stats(_eval_records)
                        _eval_adjustments = compute_eval_adjustments(_eval_stats_data)
                        _eval_reasons = compute_eval_adjustment_reasons(_eval_stats_data)
                    except Exception:
                        pass
                    if routing_decision and routing_decision.category and _eval_records:
                        try:
                            from openshard.evals.registry import build_category_map
                            from openshard.evals.stats import compute_category_stats
                            from openshard.evals.adjustments import (
                                compute_category_eval_adjustments,
                                compute_category_eval_adjustment_reasons,
                            )
                            _suites = {r.get("suite") for r in _eval_records if r.get("suite")}
                            _cat_maps: dict[str, dict[str, str]] = {}
                            for _s in _suites:
                                try:
                                    _cat_maps[str(_s)] = build_category_map(str(_s))
                                except Exception:
                                    pass
                            if _cat_maps:
                                _cat_stats = compute_category_stats(_eval_records, _cat_maps)
                                _cat_adjustments = compute_category_eval_adjustments(
                                    _cat_stats, routing_decision.category
                                )
                                _cat_reasons = compute_category_eval_adjustment_reasons(
                                    _cat_stats, routing_decision.category
                                )
                        except Exception:
                            pass
                _merged_adjustments: dict[str, float] | None = None
                if _hist_adjustments is not None or _eval_adjustments or _cat_adjustments:
                    _merged_adjustments = dict(_hist_adjustments or {})
                    _all_eval_models = set(_eval_adjustments) | set(_cat_adjustments)
                    for _em in _all_eval_models:
                        _combined = _eval_adjustments.get(_em, 0.0) + _cat_adjustments.get(_em, 0.0)
                        from openshard.evals.adjustments import _ADJ_MIN, _ADJ_MAX
                        _combined = max(_ADJ_MIN, min(_ADJ_MAX, _combined))
                        if _combined != 0.0:
                            _merged_adjustments[_em] = _merged_adjustments.get(_em, 0.0) + _combined
                _scored = select_with_info(_entries, _reqs, routing_decision.category, history_adjustments=_merged_adjustments)
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

        _verification_plan = build_verification_plan(_cfg, _repo_facts)

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

        _profile_history_summary: dict[str, ProfileHistorySummary] | None = None
        if _use_history_scoring and _runs:
            _profile_history_summary = build_profile_history_summary(_runs)
        _profile_decision: ProfileDecision = select_profile(
            category=routing_decision.category if routing_decision else "standard",
            repo_facts=_repo_facts,
            task=task,
            override=profile,
            history_summary=_profile_history_summary,
        )

        _cfg_approval = _cfg.get("approval_mode", "smart").strip().lower()
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
                click.echo(f"  [routing] candidates: {_scored.candidate_count} -> fallback (keyword routing)")
            else:
                _cost_str = f"cost: ${_scored.selected_cost_per_m:.2f}/M" if _scored.selected_cost_per_m is not None else "cost: unknown"
                click.echo(f"  [routing] candidates: {_scored.candidate_count} -> {_model_label(_scored.selected_model)} ({_cost_str})")
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
                    _marker = " <- selected" if _hm == _scored.selected_model else ""
                    click.echo(f"  [routing] history: {_model_label(_hm)}: {_hadj:+.1f}{_rsn_str}{_marker}")
            if _use_eval_scoring:
                click.echo("  [routing] eval scoring: enabled")
                _eval_nonzero = [
                    (m, adj)
                    for m, adj in _eval_adjustments.items()
                    if m in set(_scored.candidates) and adj != 0.0
                ]
                for _em, _ea in _eval_nonzero:
                    _rsn = _eval_reasons.get(_em, "")
                    _rsn_str = f" ({_rsn})" if _rsn else ""
                    _marker = " <- selected" if _em == _scored.selected_model else ""
                    click.echo(f"  [routing] eval: {_model_label(_em)}: {_ea:+.1f}{_rsn_str}{_marker}")
                if not _eval_nonzero:
                    click.echo("  [routing] eval: no relevant stats (no adjustment)")
                _cat_label = routing_decision.category if routing_decision else "?"
                _cat_nonzero = [
                    (m, adj)
                    for m, adj in _cat_adjustments.items()
                    if m in set(_scored.candidates) and adj != 0.0
                ]
                if _cat_nonzero:
                    for _cm, _ca in _cat_nonzero:
                        _rsn = _cat_reasons.get(_cm, "")
                        _rsn_str = f" ({_rsn})" if _rsn else ""
                        _marker = " <- selected" if _cm == _scored.selected_model else ""
                        click.echo(f"  [routing] eval [{_cat_label}]: {_model_label(_cm)}: {_ca:+.1f}{_rsn_str}{_marker}")
                else:
                    click.echo(f"  [routing] eval [{_cat_label}]: no category evidence (global only)")

        _matched_skills: list[MatchedSkill] = []
        if _repo_facts is not None and routing_decision is not None:
            try:
                _local_skills = discover_skills(Path.cwd())
                if _local_skills:
                    _matched_skills = match_skills(
                        _local_skills, task, routing_decision.category, _repo_facts
                    )
            except Exception:
                pass
        _skills_ctx = build_skills_context(_matched_skills)

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
            click.echo(f"  [profile] {_profile_decision.profile} - {_profile_decision.reason}")
            if _verification_plan.has_commands:
                for _vcmd in _verification_plan.commands:
                    _argv_str = " ".join(_vcmd.argv)
                    click.echo(f"  [verification] {_vcmd.name} {_vcmd.safety.value} {_vcmd.source.value} {_argv_str}")
            else:
                click.echo("  [verification] no verification command detected")

        if detail != "default" and _matched_skills:
            click.echo("\nSkills")
            for _ms in _matched_skills:
                _rsn = ", ".join(_ms.reasons)
                click.echo(f"  {_ms.skill.slug}: {_ms.skill.name} ({_rsn})")

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
                _shape_desc = "staged (planning -> implementation)"
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
                    _plan_text, _ = run_planning_stage(generator.client, task, skills_context=_skills_ctx)
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
        exec_result = None

        if dry_run:
            exec_result = ExecutionResult(summary="(dry run — no provider call)", files=[], notes=[], usage=None)

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
                        _plan_text, _plan_usage = run_planning_stage(generator.client, task, skills_context=_skills_ctx)
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
                        exec_result = generator.generate(_impl_task, model=_stage_model, repo_facts=_repo_facts, skills_context=_skills_ctx)
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
                        cost=exec_result.usage.estimated_cost if exec_result.usage else None,
                        summary=exec_result.summary,
                    ))

        # --- Single-stage execution (simple tasks, opencode, stages not triggered) -
        if exec_result is None:
            _single_msg = (
                _exec_message(_routed_model, routing_decision.rationale)
                if routing_decision is not None
                else ("Executing - running with OpenCode" if opencode_mode else "Executing - running task")
            )
            spinner.start(_single_msg)
            try:
                if opencode_mode:
                    exec_result = generator.generate(task, workspace=workspace)
                else:
                    exec_result = generator.generate(task, model=_routed_model, repo_facts=_repo_facts, skills_context=_skills_ctx)
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

        usage = exec_result.usage
        # When stages ran, fold the planning cost into the reported total
        if stage_runs and usage is not None:
            total_stage_cost = sum(sr.cost for sr in stage_runs if sr.cost is not None)
            if usage.estimated_cost is not None:
                usage.estimated_cost = total_stage_cost
            else:
                usage.estimated_cost = total_stage_cost or None
        final_files = exec_result.files

        if not opencode_mode and _repo_facts is not None:
            _mismatches = check_stack_mismatch(final_files, _repo_facts)
            _sm_dec = gate.check_stack_mismatch(_mismatches)
            if _sm_dec.required:
                _lang_str = ", ".join(_repo_facts.languages) if _repo_facts.languages else "unknown"
                confirm_or_abort(f"{_sm_dec.reason} (repo stack: {_lang_str})")

        click.echo("\nDone")
        click.echo(exec_result.summary)

        _model_line = _build_model_line(routing_decision, stage_runs, model=_routed_model)
        if _model_line:
            click.echo(f"\n{_model_line}")

        # Stages — shown before file count so the reader sees what ran (--more only)
        if detail != "default" and stage_runs:
            click.echo("\nStages")
            for sr in stage_runs:
                _sr_cost = f"${sr.cost:.4f}" if sr.cost is not None else "-"
                click.echo(f"  {sr.stage.stage_type.capitalize()} ({_model_label(sr.model)}): {sr.duration:.1f}s, {_sr_cost}")

        if exec_result.files:
            _fc = sum(1 for f in exec_result.files if f.change_type == "create")
            _fu = sum(1 for f in exec_result.files if f.change_type == "update")
            _fd = sum(1 for f in exec_result.files if f.change_type == "delete")
            _counts = ", ".join(p for p in [
                f"{_fc} created" if _fc else "",
                f"{_fu} updated" if _fu else "",
                f"{_fd} deleted" if _fd else "",
            ] if p)
            click.echo(f"\nFiles: {_counts}")
            if detail != "default":
                for f in exec_result.files:
                    _desc = f" - {f.summary}" if f.summary else ""
                    click.echo(f"  {f.path}{_desc}")

        if detail != "default" and exec_result.notes:
            _notes = [_truncate_note(n) for n in exec_result.notes if n][:3]
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
            if _should_shrink(exec_result.files, no_shrink):
                _print_shrunk(exec_result.files, exec_result.summary)
            else:
                _print_dry_run(exec_result.files)
            _print_summary(start, generator, retry_triggered, final_files, usage=usage, detail=detail, model=_routed_model, stage_runs=stage_runs)
            try:
                _log_run(start, task, generator, retry_triggered, final_files,
                         verification_attempted=False, verification_passed=None,
                         workspace=None, usage=usage, model=_routed_model,
                         summary=exec_result.summary, notes=exec_result.notes,
                         stage_runs=stage_runs, routing_decision=routing_decision,
                         _scored=_scored, repo_facts=_repo_facts,
                         matched_skills=_matched_skills,
                         profile_decision=_profile_decision,
                         verification_plan=_verification_plan)
            except Exception as exc:
                click.echo(f"  [log] warning: {exc}")
            result_obj.exit_code = 0
            result_obj.generator = generator
            result_obj.final_files = final_files
            result_obj.usage = usage
            result_obj.routed_model = _routed_model
            result_obj.stage_runs = stage_runs
            result_obj.routing_decision = routing_decision
            result_obj.scored = _scored
            result_obj.repo_facts = _repo_facts
            result_obj.matched_skills = _matched_skills
            result_obj.profile_decision = _profile_decision
            result_obj.verification_plan = _verification_plan
            result_obj.result_summary = exec_result.summary
            result_obj.result_notes = exec_result.notes or []
            return result_obj

        if verify and not write:
            raise click.ClickException("--verify requires --write.")

        if write:
            if not opencode_mode:
                workspace = Path(tempfile.mkdtemp())
                if detail == "full":
                    click.echo(f"\n  [workspace] {workspace}")
                _file_paths = [f.path for f in exec_result.files if f.path]
                _fw_dec = gate.check_file_write(_file_paths)
                _rp_dec = gate.check_risky_paths(_file_paths)
                if _fw_dec.required:
                    confirm_or_abort(_fw_dec.reason)
                elif _rp_dec.required:
                    confirm_or_abort(_rp_dec.reason)
                _write_files(exec_result.files, workspace)
                if effective_executor == "native":
                    if hasattr(generator, "record_loop_step"):
                        generator.record_loop_step("write")
                    generator.review_diff()
            # OpenCode: workspace already created and populated before generate().

        # Native controlled verification loop — one retry, safe commands only, no --verify flag required
        if effective_executor == "native" and write and not dry_run and not verify:
            _loop_meta = NativeVerificationLoop()
            generator.native_meta.verification_loop = _loop_meta
            if (
                _verification_plan.has_commands
                and all(cmd.safety == CommandSafety.safe for cmd in _verification_plan.commands)
            ):
                _loop_meta.attempted = True
                if hasattr(generator, "record_loop_step"):
                    generator.record_loop_step("verification")
                _loop_code, _loop_output = _run_verification_plan(
                    _verification_plan, workspace, capture=True
                )
                _loop_meta.exit_code = _loop_code
                _loop_meta.output_chars = len(_loop_output)
                _loop_meta.truncated = len(_loop_output) > 1200
                _loop_meta.passed = _loop_code == 0
                if _loop_code != 0:
                    _failure_ctx = render_verification_failure_context(
                        _loop_output, exit_code=_loop_code
                    )
                    _retry_skills_ctx = "\n\n".join(
                        p for p in [_skills_ctx, _failure_ctx] if p
                    )
                    spinner.start("Retrying - native verification failed, regenerating")
                    try:
                        _retry_result = generator.generate(
                            task,
                            model=_routed_model,
                            repo_facts=_repo_facts,
                            skills_context=_retry_skills_ctx,
                        )
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
                    _loop_meta.retried = True
                    exec_result = _retry_result
                    final_files = _retry_result.files
                    _write_files(_retry_result.files, workspace)
                    # Run verification once more after retry — update metadata with final result
                    _loop_code2, _loop_output2 = _run_verification_plan(
                        _verification_plan, workspace, capture=True
                    )
                    _loop_meta.exit_code = _loop_code2
                    _loop_meta.output_chars = len(_loop_output2)
                    _loop_meta.truncated = len(_loop_output2) > 1200
                    _loop_meta.passed = _loop_code2 == 0

        if write and verify:
            click.echo("")
            code = _run_verification_plan(_verification_plan, workspace, gate=gate, detail=detail)
            # Escalation loop: try each model in chain until verification passes.
            # Direct mode uses ESCALATION_CHAIN (sonnet → opus).
            # OpenCode mode uses a single fixer-model retry (no chain).
            _escalation = ESCALATION_CHAIN if not opencode_mode else [generator.fixer_model]
            _last_attempt = exec_result
            _can_escalate = (
                _verification_plan.has_commands
                and _verification_plan.commands[0].safety != CommandSafety.blocked
            )
            if _can_escalate:
                for _esc_model in _escalation:
                    if code == 0:
                        break
                    retry_triggered = True
                    _, verify_output = _run_verification_plan(
                        _verification_plan, workspace, gate=None, capture=True
                    )
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
                        if effective_executor == "native":
                            generator.review_diff()
                    code = _run_verification_plan(
                        _verification_plan, workspace, gate=None,
                        label=f"[retry/{_esc_label}]", detail=detail,
                    )
            verification_passed = code == 0
            if code != 0:
                _print_summary(start, generator, retry_triggered, final_files,
                               usage=usage, retry_usage=retry_usage, detail=detail, model=_routed_model, stage_runs=stage_runs)
                try:
                    _log_run(start, task, generator, retry_triggered, final_files,
                             verification_attempted=True, verification_passed=False,
                             workspace=workspace, usage=usage, retry_usage=retry_usage, model=_routed_model,
                             summary=exec_result.summary, notes=exec_result.notes,
                             stage_runs=stage_runs, routing_decision=routing_decision,
                             _scored=_scored, repo_facts=_repo_facts,
                             matched_skills=_matched_skills,
                             profile_decision=_profile_decision,
                             verification_plan=_verification_plan)
                except Exception as exc:
                    click.echo(f"  [log] warning: {exc}")
                result_obj.exit_code = code
                result_obj.generator = generator
                result_obj.retry_triggered = retry_triggered
                result_obj.final_files = final_files
                result_obj.usage = usage
                result_obj.retry_usage = retry_usage
                result_obj.routed_model = _routed_model
                result_obj.stage_runs = stage_runs
                result_obj.routing_decision = routing_decision
                result_obj.scored = _scored
                result_obj.repo_facts = _repo_facts
                result_obj.matched_skills = _matched_skills
                result_obj.profile_decision = _profile_decision
                result_obj.verification_plan = _verification_plan
                result_obj.verification_attempted = True
                result_obj.verification_passed = False
                result_obj.workspace = workspace
                result_obj.result_summary = exec_result.summary
                result_obj.result_notes = exec_result.notes or []
                return result_obj

        _print_summary(start, generator, retry_triggered, final_files,
                       usage=usage, retry_usage=retry_usage, detail=detail, model=_routed_model, stage_runs=stage_runs)
        if effective_executor == "native" and hasattr(generator, "build_final_report"):
            generator.build_final_report()
        _native_meta = generator.native_meta if effective_executor == "native" else None
        if _native_meta is not None:
            from openshard.cli.run_output import _print_native_demo_block, _print_native_summary
            _print_native_demo_block(_native_meta)
            _print_native_summary(_native_meta)
        _extra_metadata: dict | None = None
        if _native_meta is not None:
            _extra_metadata = {
                "workflow": _native_meta.workflow,
                "executor": _native_meta.executor,
                "execution_depth": _native_meta.execution_depth,
                "selected_skills": _native_meta.selected_skills,
                "context_budget": asdict(_native_meta.context_budget) if _native_meta.context_budget is not None else None,
                "context_state": asdict(_native_meta.context_state) if _native_meta.context_state is not None else None,
                "context_warnings": _native_meta.context_warnings,
                "tool_trace": _native_meta.tool_trace,
                "repo_context_summary": asdict(_native_meta.repo_context_summary) if _native_meta.repo_context_summary is not None else None,
                "observation": asdict(_native_meta.observation) if _native_meta.observation is not None else None,
                "evidence": asdict(_native_meta.evidence) if _native_meta.evidence is not None else None,
                "plan": asdict(_native_meta.plan) if _native_meta.plan is not None else None,
                "diff_review": asdict(_native_meta.diff_review) if _native_meta.diff_review is not None else None,
                "write_path": _native_meta.write_path,
                "verification_loop": (
                    asdict(_native_meta.verification_loop)
                    if _native_meta.verification_loop is not None
                    else None
                ),
                "final_report": (
                    asdict(_native_meta.final_report)
                    if _native_meta.final_report is not None
                    else None
                ),
                "native_loop_steps": list(_native_meta.native_loop_steps),
                "native_loop_trace": (
                    [
                        {
                            "phase": event.phase,
                            "status": event.status,
                            "summary": event.summary,
                            "metadata": event.metadata,
                        }
                        for event in getattr(_native_meta.native_loop_trace, "events", [])
                    ]
                    if _native_meta is not None
                    else []
                ),
                "native_backend": getattr(_native_meta, "native_backend", "builtin"),
                "native_backend_available": getattr(_native_meta, "native_backend_available", True),
                "native_backend_notes": list(getattr(_native_meta, "native_backend_notes", [])),
                "native_backend_proof": getattr(_native_meta, "native_backend_proof", None),
            }
        try:
            _log_run(start, task, generator, retry_triggered, final_files,
                     verification_attempted=(write and verify),
                     verification_passed=verification_passed,
                     workspace=workspace, usage=usage, retry_usage=retry_usage, model=_routed_model,
                     summary=exec_result.summary, notes=exec_result.notes,
                     stage_runs=stage_runs, routing_decision=routing_decision,
                     _scored=_scored, repo_facts=_repo_facts,
                     matched_skills=_matched_skills,
                     profile_decision=_profile_decision,
                     verification_plan=_verification_plan,
                     extra_metadata=_extra_metadata)
        except Exception as exc:
            click.echo(f"  [log] warning: {exc}")

        result_obj.exit_code = 0
        result_obj.generator = generator
        result_obj.retry_triggered = retry_triggered
        result_obj.final_files = final_files
        result_obj.usage = usage
        result_obj.retry_usage = retry_usage
        result_obj.routed_model = _routed_model
        result_obj.stage_runs = stage_runs
        result_obj.routing_decision = routing_decision
        result_obj.scored = _scored
        result_obj.repo_facts = _repo_facts
        result_obj.matched_skills = _matched_skills
        result_obj.profile_decision = _profile_decision
        result_obj.verification_plan = _verification_plan
        result_obj.verification_attempted = (write and verify)
        result_obj.verification_passed = verification_passed
        result_obj.workspace = workspace
        result_obj.result_summary = exec_result.summary
        result_obj.result_notes = exec_result.notes or []
        result_obj.native_meta = _native_meta
        return result_obj


_LOG_PATH = Path(".openshard") / "runs.jsonl"

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


def _serialize_verification_plan(plan: VerificationPlan) -> list[dict]:
    return [
        {
            "name": cmd.name,
            "argv": cmd.argv,
            "kind": cmd.kind.value,
            "source": cmd.source.value,
            "safety": cmd.safety.value,
            "reason": cmd.reason,
        }
        for cmd in plan.commands
    ]


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
    matched_skills: list[MatchedSkill] | None = None,
    profile_decision: ProfileDecision | None = None,
    verification_plan: VerificationPlan | None = None,
    extra_metadata: dict | None = None,
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

    if matched_skills:
        entry["matched_skills"] = [m.skill.slug for m in matched_skills]
        entry["matched_skill_reasons"] = {m.skill.slug: m.reasons for m in matched_skills}

    if profile_decision is not None:
        entry["execution_profile"] = profile_decision.profile
        entry["execution_profile_reason"] = profile_decision.reason

    if verification_plan is not None and verification_plan.has_commands:
        entry["verification_plan"] = _serialize_verification_plan(verification_plan)

    if extra_metadata:
        entry.update(extra_metadata)

    log_path = Path.cwd() / _LOG_PATH
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry) + "\n")


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


def _write_files(files: list[ChangedFile], root: Path) -> None:
    cwd = root.resolve()
    for f in files:
        try:
            target = resolve_safe_repo_path(cwd, f.path)
        except UnsafePathError:
            click.echo(f"  [skip] unsafe path rejected: {f.path!r}")
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
            from openshard.config.settings import load_config
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
