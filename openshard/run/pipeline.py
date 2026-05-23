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
    _build_routing_line,
    _exec_message,
    _extract_structured_findings,
    _extract_findings_from_model_answer,
    _extract_findings_from_review_files,
    _model_label,
    _profile_display_label,
    _print_dry_run,
    _print_shrunk,
    _print_summary,
    _should_shrink,
    render_post_run,
)
from openshard.run.timeline import RunTimelineEvent
from openshard.config.settings import get_anthropic_api_key, get_openai_api_key
from openshard.execution.gates import GateEvaluator, VALID_APPROVAL_MODES
from openshard.execution.generator import (
    ChangedFile,
    ExecutionGenerator,
    ExecutionResult,
    check_stack_mismatch,
)
from openshard.execution.opencode_executor import OpenCodeExecutor
from openshard.native.context import (
    NativeCandidateSummary,
    NativeEditLoopSummary,
    NativeVerificationLoop,
    RetryMetadata,
    build_failure_summary,
    record_native_candidate_attempt,
    record_native_edit_loop_attempt,
    render_verification_failure_context,
    select_native_candidate,
)
from openshard.native.executor import NativeAgentExecutor
from openshard.execution.stages import Stage, StageRun, split_task, route_stage, run_planning_stage, run_validator_stage
from openshard.history.adjustments import (
    compute_history_adjustments,
    compute_history_adjustment_reasons,
)
from openshard.history.feedback_scoring import (
    compute_feedback_adjustments,
    compute_feedback_adjustment_reasons,
)
from openshard.history.failure_memory import (
    NativeFailureMemoryEvent,
    parse_failure_summary,
    log_failure_memory_event,
)
from openshard.history.run_checkpoints import (
    NativeRunCheckpointEvent,
    log_run_checkpoint_event as _log_run_checkpoint,
)
from openshard.history.metrics import load_runs
from openshard.providers.base import ProviderAuthError, ProviderError, ProviderRateLimitError
from openshard.providers.manager import ProviderManager
from openshard.providers.openrouter import MODEL_PRICING, compute_cost
from openshard.routing.engine import ESCALATION_CHAIN, MODEL_STRONG, RoutingDecision, route, is_readonly_task
from openshard.routing.form_factor_policy import ExecutionFormFactorDecision, select_form_factor
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
from openshard.run.validator_policy import ValidatorPolicyDecision, should_run_validator
from openshard.verification.executor import run_verification_plan as _run_verification_plan, confirm_or_abort  # noqa: F401
from openshard.verification.plan import CommandSafety, VerificationPlan, build_verification_plan


def _build_explicit_file_context(task: str, *, root: Path | None = None) -> str:
    """Return a rendered evidence block for repo-relative file paths named in *task*.

    Used by the direct/staged execution path where NativeAgentExecutor is not active.
    Returns "" when no safe, readable, named files are found.
    """
    from openshard.native.executor import (
        _extract_explicit_file_paths,
        _build_explicit_file_outline,
        _MAX_EXPLICIT_SNIPPET_FILES,
    )
    from openshard.native.context import NativeEvidence, NativeFileSnippet, render_native_evidence

    paths = _extract_explicit_file_paths(task)
    if not paths:
        return ""

    repo_root = root or Path.cwd()
    snippets: list[NativeFileSnippet] = []
    for path_str in paths[:_MAX_EXPLICIT_SNIPPET_FILES]:
        try:
            resolved = resolve_safe_repo_path(repo_root, path_str)
        except UnsafePathError:
            continue
        if not resolved.is_file():
            continue
        try:
            content = resolved.read_text(encoding="utf-8", errors="replace")
            outline = _build_explicit_file_outline(content, path_str)
            if not outline:
                continue
            snippets.append(NativeFileSnippet(path=path_str, lines=outline))
        except Exception:
            continue

    if not snippets:
        return ""
    return render_native_evidence(
        NativeEvidence(file_snippets=snippets),
        limit=3000,
        max_lines_per_snippet=25,
    )


def _emit_review_debug(
    *,
    effective_executor: str,
    selected_workflow: str,
    opencode_explicit: bool,
    task: str,
    exec_result_summary: str,
    exec_result_files_count: int,
    exec_result_notes: list,
    usage_info: Any,
    extraction_source: "str | None",
    findings_count: int,
) -> None:
    import os as _os
    if _os.environ.get("OPENSHARD_DEBUG_REVIEW", "").strip() != "1":
        return
    _public_label = "OpenCode" if effective_executor == "opencode" else "OpenShard Native"
    lines = [
        "=== OPENSHARD_DEBUG_REVIEW ===",
        f"executor_path:     {effective_executor}",
        f"public_label:      {_public_label}",
        f"selected_workflow: {selected_workflow}",
        f"opencode_explicit: {opencode_explicit}",
        f"is_review_task:    {'STRUCTURED_FINDINGS:' in task}",
        f"task_has_suffix:   {'After completing your analysis' in task}",
        f"task_preview:      {task[:500]!r}",
        "---",
        f"summary_len:       {len(exec_result_summary)}",
        f"summary_preview:   {exec_result_summary[:3000]!r}",
        "---",
        f"files_count:       {exec_result_files_count}",
        f"notes:             {exec_result_notes!r}",
        f"usage:             {usage_info!r}",
        "---",
        f"extraction_source: {extraction_source or 'none (fallback)'}",
        f"findings_count:    {findings_count}",
        "=== END DEBUG ===",
    ]
    block = "\n".join(lines) + "\n"
    sys.stderr.write(block)
    sys.stderr.flush()
    _dot_openshard = Path.cwd() / ".openshard"
    if _dot_openshard.is_dir():
        _log_dir = _dot_openshard / "debug"
        _log_dir.mkdir(parents=True, exist_ok=True)
        _log_file = _log_dir / "review-debug.log"
    else:
        import tempfile as _tmp
        _log_file = Path(_tmp.gettempdir()) / "openshard-review-debug.log"
    try:
        with _log_file.open("a", encoding="utf-8") as _f:
            _f.write(block)
    except OSError:
        pass


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
    form_factor_decision: Any = None


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
        feedback_scoring: bool,
        detail: str,
        native_backend: str | None = None,
        experimental_deepagents_run: bool = False,
        experimental_tier_dispatch: bool = False,
        native_loop: str | None = None,
        model_policy: str | None = None,
        candidates: int = 1,
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
        self._feedback_scoring = feedback_scoring
        self._detail = detail
        self._native_backend = native_backend or "builtin"
        self._experimental_deepagents_run = experimental_deepagents_run
        self._experimental_tier_dispatch = experimental_tier_dispatch
        self._native_loop = native_loop
        self._model_policy = model_policy
        self._candidates = candidates

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
        feedback_scoring = self._feedback_scoring

        start = time.time()
        result_obj.start = start
        _run_id = datetime.datetime.fromtimestamp(
            start, tz=datetime.timezone.utc
        ).isoformat().replace("+00:00", "Z")
        retry_triggered = False
        workspace: Path | None = None
        verification_passed: bool | None = None
        usage = None
        retry_usage = None
        _timeline: list[RunTimelineEvent] = []

        try:
            _timeline.append(RunTimelineEvent("run_started", "Started run", kind="run"))
            _cfg = self._config
            _cfg_workflow = _cfg.get("workflow", "").strip().lower()
            _cfg_executor_legacy = _cfg.get("executor", "direct").strip().lower()
            _use_history_scoring = history_scoring or bool(_cfg.get("history_scoring", False))
            _use_eval_scoring = eval_scoring or bool(_cfg.get("eval_scoring", False))
            _use_feedback_scoring = feedback_scoring or bool(_cfg.get("feedback_scoring", False))
            _policy_executor, _policy_reason = _suggest_executor(task)

            # --workflow > --executor (deprecated) > config.workflow > config.executor > auto
            _show_executor_deprecation = False
            _pack_label: str = ""
            if workflow is not None:
                _wf = workflow.lower()
                if _wf in ("claude-code", "codex"):
                    raise click.ClickException(
                        f"--workflow {_wf!r} is not yet available. "
                        "Use: auto, direct, staged, opencode, or native."
                    )
                effective_workflow = _wf
                _policy_reason = ""
                _timeline.append(RunTimelineEvent("workflow_pack_loaded", "Loaded workflow pack", kind="workflow", target=_wf))
                try:
                    from openshard.workflow_packs.packs import load_packs as _load_packs_fn
                    _pack_label = next(
                        (p.title for p in _load_packs_fn() if p.prompt and task.startswith(p.prompt)),
                        "",
                    )
                except Exception:
                    _pack_label = ""
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
                    native_loop=self._native_loop,
                    model_policy=self._model_policy,
                )
                if hasattr(generator, "_run_id"):
                    generator._run_id = _run_id
            else:
                generator = ExecutionGenerator(provider=_provider_instance)
        except (ValueError, RuntimeError) as exc:
            raise click.ClickException(str(exc))

        opencode_mode = (effective_executor == "opencode")
        routing_decision: RoutingDecision | None = route(task) if not opencode_mode else None
        if routing_decision is not None and is_readonly_task(task):
            routing_decision.rationale = "read-only analysis"
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

        def _safe_checkpoint(stage: str, status: str, **kwargs) -> None:
            try:
                _wp = str(workspace) if workspace is not None else ""
                _log_run_checkpoint(NativeRunCheckpointEvent(
                    run_id=_run_id,
                    workflow=effective_workflow,
                    executor=effective_executor,
                    stage=stage,
                    status=status,
                    workspace_path=_wp,
                    sandbox_path=_wp,
                    files=list(kwargs.get("files", [])),
                    verification_status=kwargs.get("verification_status", ""),
                    retry_used=bool(kwargs.get("retry_used", False)),
                    reason=kwargs.get("reason", ""),
                ))
            except Exception:
                pass

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
        _feedback_receipt: dict | None = None

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
                _feedback_adjustments: dict[str, float] = {}
                _feedback_reasons: dict[str, str] = {}
                if _use_feedback_scoring:
                    try:
                        from openshard.history.interactions import load_interaction_events
                        _fb_runs = load_runs()
                        _fb_events = load_interaction_events()
                        _feedback_adjustments = compute_feedback_adjustments(
                            _fb_runs, interaction_events=_fb_events
                        )
                        _feedback_reasons = compute_feedback_adjustment_reasons(
                            _fb_runs, interaction_events=_fb_events
                        )
                        _feedback_receipt = {
                            "routing_feedback_scoring_used": True,
                            "routing_feedback_adjustments": {
                                m: round(v, 3) for m, v in _feedback_adjustments.items()
                            },
                            "routing_feedback_reasons": dict(_feedback_reasons),
                        }
                    except Exception:
                        pass
                _merged_adjustments: dict[str, float] | None = None
                if _hist_adjustments is not None or _eval_adjustments or _cat_adjustments or _feedback_adjustments:
                    _merged_adjustments = dict(_hist_adjustments or {})
                    _all_eval_models = set(_eval_adjustments) | set(_cat_adjustments)
                    for _em in _all_eval_models:
                        _combined = _eval_adjustments.get(_em, 0.0) + _cat_adjustments.get(_em, 0.0)
                        from openshard.evals.adjustments import _ADJ_MIN, _ADJ_MAX
                        _combined = max(_ADJ_MIN, min(_ADJ_MAX, _combined))
                        if _combined != 0.0:
                            _merged_adjustments[_em] = _merged_adjustments.get(_em, 0.0) + _combined
                    _merged_clamp_min, _merged_clamp_max = -2.0, 1.0
                    for _fm, _fval in _feedback_adjustments.items():
                        _merged_adjustments[_fm] = max(
                            _merged_clamp_min, min(_merged_clamp_max, _merged_adjustments.get(_fm, 0.0) + _fval)
                        )
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
            _timeline.append(RunTimelineEvent("repo_scanned", "Scanned repo", kind="scan"))
        except Exception:
            pass

        if _routed_model:
            _timeline.append(RunTimelineEvent(
                "model_selected",
                f"Routed to {_model_label(_routed_model)}",
                kind="route",
                target=_routed_model,
            ))

        _readonly_task = is_readonly_task(task)
        _is_sf_task = "STRUCTURED_FINDINGS:" in task
        _generation_max_tokens: int = 8192 if _is_sf_task else 16384

        _verification_plan = build_verification_plan(_cfg, _repo_facts)

        if (
            effective_executor == "native"
            and _verification_plan.has_commands
            and hasattr(generator, "build_command_policy_preview")
        ):
            generator.build_command_policy_preview(_verification_plan)

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
                readonly=_readonly_task,
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

        _form_factor_decision: ExecutionFormFactorDecision = select_form_factor(
            category=routing_decision.category if routing_decision else "standard",
            readonly=_readonly_task,
            workflow="staged" if _use_stages else "direct",
            profile_name=_profile_decision.profile,
            repo_facts=_repo_facts,
            write_requested=write,
            verification_available=bool(
                _verification_plan is not None and _verification_plan.has_commands
            ),
            native_loop=self._native_loop,
            experimental_deepagents_run=self._experimental_deepagents_run,
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

        # Routing section lines are collected here and printed together after workflow is resolved.
        _routing_lines: list[str] = []
        if routing_decision is not None:
            if detail != "default":
                _routing_lines.append(f"  Task type: {routing_decision.rationale}")
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
            _routing_lines.append(f"  Category: {_scored.category}")
            if _req_str != "none":
                _routing_lines.append(f"  Requirements: {_req_str}")
            if _scored.used_fallback:
                _routing_lines.append(f"  Candidates: {_scored.candidate_count} (fallback keyword routing)")
            else:
                _cost_str = f"cost: ${_scored.selected_cost_per_m:.2f}/M" if _scored.selected_cost_per_m is not None else "cost: unknown"
                _routing_lines.append(f"  Initial candidate: {_model_label(_scored.selected_model)} ({_cost_str})")
                _routing_lines.append(f"  Candidates: {_scored.candidate_count}")
            if _use_history_scoring:
                _routing_lines.append("  History scoring: enabled")
                _nonzero = [
                    (m, adj)
                    for m, adj in _scored.history_adjustments.items()
                    if m in set(_scored.candidates) and adj != 0.0
                ]
                for _hm, _hadj in _nonzero:
                    _rsn = _hist_reasons.get(_hm, "")
                    _rsn_str = f" ({_rsn})" if _rsn else ""
                    _marker = " <- selected" if _hm == _scored.selected_model else ""
                    _routing_lines.append(f"    {_model_label(_hm)}: {_hadj:+.1f}{_rsn_str}{_marker}")
            if _use_eval_scoring:
                _routing_lines.append("  eval scoring: enabled")
                _eval_nonzero = [
                    (m, adj)
                    for m, adj in _eval_adjustments.items()
                    if m in set(_scored.candidates) and adj != 0.0
                ]
                for _em, _ea in _eval_nonzero:
                    _rsn = _eval_reasons.get(_em, "")
                    _rsn_str = f" ({_rsn})" if _rsn else ""
                    _marker = " <- selected" if _em == _scored.selected_model else ""
                    _routing_lines.append(f"    {_model_label(_em)}: {_ea:+.1f}{_rsn_str}{_marker}")
                if not _eval_nonzero:
                    _routing_lines.append("    No relevant stats (no adjustment)")
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
                        _routing_lines.append(f"    [{_cat_label}] {_model_label(_cm)}: {_ca:+.1f}{_rsn_str}{_marker}")
                else:
                    _routing_lines.append(f"    [{_cat_label}] No category evidence (global only)")
            if _use_feedback_scoring:
                _routing_lines.append("  Feedback scoring: enabled")
                _fb_nonzero = [
                    (m, adj)
                    for m, adj in _feedback_adjustments.items()
                    if m in set(_scored.candidates) and adj != 0.0
                ]
                for _fm, _fa in _fb_nonzero:
                    _rsn = _feedback_reasons.get(_fm, "")
                    _rsn_str = f" ({_rsn})" if _rsn else ""
                    _marker = " <- selected" if _fm == _scored.selected_model else ""
                    _routing_lines.append(f"    {_model_label(_fm)}: {_fa:+.1f}{_rsn_str}{_marker}")
                if not _fb_nonzero:
                    _routing_lines.append("    No relevant feedback (no adjustment)")

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

        if effective_executor != "native":
            _explicit_ctx = _build_explicit_file_context(task)
            if _explicit_ctx:
                _skills_ctx = f"{_skills_ctx}\n\n{_explicit_ctx}" if _skills_ctx else _explicit_ctx

        if (_readonly_task or _is_sf_task) and not dry_run:
            if "STRUCTURED_FINDINGS:" in task:
                # Review/security pack: model must put COMPLETE multi-line analysis in summary field.
                # The generic "one sentence, third-person" instruction conflicts with this requirement.
                _rv_instruction = (
                    "\n\n[IMPORTANT] This is a read-only security/configuration review task. "
                    "Do not propose file changes. The `files` array must be empty ([]). "
                    "Your `summary` field must contain your COMPLETE multi-line analysis — "
                    "severity sections and the STRUCTURED_FINDINGS line must be inside the summary. "
                    "Do NOT truncate the summary to one sentence for this task."
                )
                _skills_ctx = f"{_skills_ctx}{_rv_instruction}" if _skills_ctx else _rv_instruction
            else:
                _ro_instruction = (
                    "\n\n[IMPORTANT] This is a read-only analysis/explanation task. "
                    "Do not propose file changes. Do not create, update, delete, or rewrite files. "
                    "The `files` field in your response must be empty ([]). "
                    "Write your response as a third-person explanation of what the code does "
                    "(e.g. 'This file handles…', 'The function does…'). "
                    "Do NOT use phrases like 'Implemented', 'Updated', 'Created', or 'Added' "
                    "in your summary — those imply you made changes."
                )
                _skills_ctx = f"{_skills_ctx}{_ro_instruction}" if _skills_ctx else _ro_instruction

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
            _routing_lines.append(f"  Workflow: {_wf_display}")
            _routing_lines.append(f"  Reason: {_wf_display_reason}")
            if _routing_lines:
                click.echo("Routing")
                for _rl in _routing_lines:
                    click.echo(_rl)
            click.echo("")
            click.echo("Execution")
            click.echo(f"  Mode: {_profile_display_label(_profile_decision.profile, is_readonly=_readonly_task)}")
            _exec_reason = "read-only task — direct analysis" if _readonly_task else _profile_decision.reason
            click.echo(f"  Reason: {_exec_reason}")
            click.echo("")
            click.echo("Verification")
            if _verification_plan.has_commands:
                _first_vcmd = True
                for _vcmd in _verification_plan.commands:
                    if not _first_vcmd:
                        click.echo("")
                    _first_vcmd = False
                    _argv_str = " ".join(_vcmd.argv)
                    click.echo(f"  Name: {_vcmd.name}")
                    click.echo(f"  Safety: {_vcmd.safety.value}")
                    click.echo(f"  Source: {_vcmd.source.value}")
                    click.echo(f"  Command: {_argv_str}")
            else:
                click.echo("  No verification command detected")

        if detail != "default" and _matched_skills:
            click.echo("\nSkills")
            for _ms in _matched_skills:
                _rsn = ", ".join(_ms.reasons)
                click.echo(f"  {_ms.skill.slug}: {_ms.skill.name} ({_rsn})")

        if detail == "default":
            _routing_msg = _build_routing_line(routing_decision, _use_stages, actual_model=_routed_model)
            if _routing_msg:
                click.echo(_routing_msg)

        # --- Experimental tier dispatch (non-native, non-opencode only) ----------
        _tier_dispatch_receipt = None
        _dispatch_planner_model: str | None = None
        _dispatch_executor_model: str | None = None
        _effective_model: str | None = None
        _validator_result: dict | None = None
        _validator_policy: ValidatorPolicyDecision | None = None

        _can_dispatch = (
            self._experimental_tier_dispatch
            and not opencode_mode
            and effective_executor != "native"
        )
        if _can_dispatch:
            from openshard.native.dispatch import resolve_tier, resolve_tier_for_category
            from openshard.native.context import NativeTierDispatchReceipt

            _cat = routing_decision.category if routing_decision else None
            _p_tier = "frontier-reasoning-model"
            _p_model, _p_fb, _p_reason = resolve_tier(_p_tier)
            _e_model, _e_tier, _e_fb, _e_reason = resolve_tier_for_category(_cat)
            _v_tier = "independent-validator-model"
            _v_model, _v_fb, _v_reason = resolve_tier(_v_tier)

            _any_fb = _p_fb or _e_fb or _v_fb
            _warns = [r for r in [
                (f"planner fallback: {_p_reason}" if _p_fb and _p_reason else ""),
                (f"executor fallback: {_e_reason}" if _e_fb and _e_reason else ""),
                (f"validator fallback: {_v_reason}" if _v_fb and _v_reason else ""),
            ] if r]
            _first_reason = next((r for r in (_p_reason, _e_reason, _v_reason) if r), "")

            _applied = not dry_run
            _not_applied_reason = "dry-run" if dry_run else ""

            _tier_dispatch_receipt = NativeTierDispatchReceipt(
                enabled=True,
                applied=_applied,
                tier_source="category_fallback",
                planner_tier=_p_tier,
                planner_model=_p_model,
                executor_tier=_e_tier or "",
                executor_model=_e_model,
                validator_tier=_v_tier,
                validator_model=_v_model,
                fallback_used=_any_fb,
                fallback_reason=_not_applied_reason or _first_reason,
                warnings=_warns,
            )
            if _applied:
                _dispatch_planner_model = _p_model
                _dispatch_executor_model = _e_model

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
                    _plan_text, _ = run_planning_stage(
                        generator.client, task,
                        skills_context=_skills_ctx,
                        model=_dispatch_planner_model or MODEL_STRONG,
                    )
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
                    if _plan_already_done or dry_run:
                        continue
                    spinner.start("Planning - mapping out implementation approach")
                    try:
                        _plan_model = _dispatch_planner_model or MODEL_STRONG
                        _plan_text, _plan_usage = run_planning_stage(
                            generator.client, task,
                            skills_context=_skills_ctx,
                            model=_plan_model,
                        )
                        _impl_task = (
                            f"Task: {task}\n\nImplementation plan:\n{_plan_text}"
                            "\n\nExecute the task following the plan above."
                        )
                        stage_runs.append(StageRun(
                            stage=_stage,
                            model=_plan_model,
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
                    if dry_run:
                        continue
                    _stage_model = _dispatch_executor_model or _routed_model or route_stage(_stage)
                    spinner.start(_exec_message(
                        _stage_model,
                        routing_decision.rationale if routing_decision else "",
                    ))
                    _stage_ok = False
                    try:
                        exec_result = generator.generate(_impl_task, model=_stage_model, repo_facts=_repo_facts, skills_context=_skills_ctx, max_tokens=_generation_max_tokens, is_review_task=_is_sf_task)
                        _stage_ok = True
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
                        if not _stage_ok:
                            _timeline.append(RunTimelineEvent("model_response_received", "Model request failed", kind="model", status="failed"))
                    _timeline.append(RunTimelineEvent("model_response_received", "Model responded", kind="model", target=_stage_model))
                    stage_runs.append(StageRun(
                        stage=_stage,
                        model=_stage_model,
                        duration=time.time() - _stage_t0,
                        cost=exec_result.usage.estimated_cost if exec_result.usage else None,
                        summary=exec_result.summary,
                    ))

        # --- Single-stage execution (simple tasks, opencode, stages not triggered) -
        if exec_result is None:
            _effective_model = _dispatch_executor_model or _routed_model
            _single_msg = (
                _exec_message(_effective_model, routing_decision.rationale)
                if routing_decision is not None
                else ("Executing - running with OpenCode" if opencode_mode else "Executing - running task")
            )
            if effective_executor == "native":
                from openshard.native.context import build_native_plan_ledger, update_native_plan_ledger_status
                generator.native_meta.plan_ledger = build_native_plan_ledger(task)
                generator.native_meta.plan_ledger = update_native_plan_ledger_status(
                    generator.native_meta.plan_ledger, "Understand task", "passed", evidence="context prepared"
                )
                _safe_checkpoint("plan", "passed")
            spinner.start(_single_msg)
            _single_ok = False
            try:
                if opencode_mode:
                    exec_result = generator.generate(task, workspace=workspace)
                else:
                    exec_result = generator.generate(task, model=_effective_model, repo_facts=_repo_facts, skills_context=_skills_ctx, max_tokens=_generation_max_tokens, is_review_task=_is_sf_task)
                _single_ok = True
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
                if not _single_ok:
                    _timeline.append(RunTimelineEvent("model_response_received", "Model request failed", kind="model", status="failed"))
            _timeline.append(RunTimelineEvent("model_response_received", "Model responded", kind="model", target=_effective_model))
            if effective_executor == "native" and generator.native_meta.plan_ledger is not None:
                from openshard.native.context import update_native_plan_ledger_status
                generator.native_meta.plan_ledger = update_native_plan_ledger_status(
                    generator.native_meta.plan_ledger, "Generate patch", "passed",
                    evidence=f"files={len(exec_result.files)}",
                )
                _safe_checkpoint("generate", "passed", files=[f.path for f in exec_result.files])

        # --- Multi-candidate path (candidates > 1, native + write only) ---
        _multi_candidate_done = False
        if self._candidates > 1 and effective_executor == "native" and write and not dry_run:
            from openshard.native.sandbox import create_run_sandbox as _create_mc_sandbox
            _candidate_summary = NativeCandidateSummary(
                enabled=True,
                requested_count=self._candidates,
            )
            _candidate_workspaces: list[Path] = []
            _candidate_results: list = []
            _candidate_sb_metas: list = []
            _all_safe = (
                _verification_plan.has_commands
                and all(cmd.safety == CommandSafety.safe for cmd in _verification_plan.commands)
            )
            for _ci in range(self._candidates):
                if _ci == 0:
                    _cand_result = exec_result
                else:
                    spinner.start(f"Candidate {_ci + 1}/{self._candidates} - generating")
                    try:
                        _cand_result = generator.generate(
                            task,
                            model=_effective_model,
                            repo_facts=_repo_facts,
                            skills_context=_skills_ctx,
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
                _cand_workspace, _cand_sb_meta = _create_mc_sandbox(
                    Path.cwd(), f"{_run_id}-candidate-{_ci + 1}"
                )
                _write_files(_cand_result.files, _cand_workspace)
                if _all_safe:
                    _cand_exit, _cand_out = _run_verification_plan(
                        _verification_plan, _cand_workspace, capture=True
                    )
                    _cand_vstatus = "passed" if _cand_exit == 0 else "failed"
                    _cand_exit_code: int | None = _cand_exit
                    _cand_output_chars = len(_cand_out or "")
                else:
                    _cand_vstatus = "skipped"
                    _cand_exit_code = None
                    _cand_output_chars = 0
                record_native_candidate_attempt(
                    _candidate_summary,
                    candidate_index=_ci + 1,
                    model=_effective_model or generator.model or "",
                    sandbox_path=str(_cand_workspace),
                    files_written=[f.path for f in _cand_result.files],
                    verification_status=_cand_vstatus,
                    exit_code=_cand_exit_code,
                    output_chars=_cand_output_chars,
                )
                _candidate_workspaces.append(_cand_workspace)
                _candidate_results.append(_cand_result)
                _candidate_sb_metas.append(_cand_sb_meta)
            select_native_candidate(_candidate_summary)
            if _candidate_summary.selected_index is None:
                raise click.ClickException("No candidate result was selected.")
            generator.native_meta.candidate_summary = _candidate_summary
            _selected_list_i = _candidate_summary.selected_index - 1
            exec_result = _candidate_results[_selected_list_i]
            workspace = _candidate_workspaces[_selected_list_i]
            generator.native_meta.sandbox = _candidate_sb_metas[_selected_list_i]
            _multi_candidate_done = True

        usage = exec_result.usage
        # When stages ran, fold the planning cost into the reported total
        if stage_runs and usage is not None:
            total_stage_cost = sum(sr.cost for sr in stage_runs if sr.cost is not None)
            if usage.estimated_cost is not None:
                usage.estimated_cost = total_stage_cost
            else:
                usage.estimated_cost = total_stage_cost or None

        # --- Validator policy + stage (experimental tier dispatch only) --------------
        if _can_dispatch and _tier_dispatch_receipt is not None:
            _validator_policy = should_run_validator(
                has_validator_model=_tier_dispatch_receipt.validator_model is not None,
                dry_run=dry_run,
                can_dispatch=_can_dispatch,
                tier_dispatch_applied=_tier_dispatch_receipt.applied,
                readonly_task=_readonly_task,
                routing_category=routing_decision.category if routing_decision else "standard",
                execution_profile=_profile_decision.profile if _profile_decision else "native_light",
                workflow="staged" if _use_stages else "direct",
                risky_paths_count=len(_repo_facts.risky_paths) if _repo_facts else 0,
                verification_attempted=bool(write and verify),
            )

        if (
            _validator_policy is not None
            and _validator_policy.run
            and exec_result is not None
        ):
            _val_t0 = time.time()
            spinner.start("Validating - reviewing implementation result")
            try:
                _val_dict, _val_usage = run_validator_stage(
                    generator.client,
                    task=task,
                    result_summary=exec_result.summary,
                    notes=exec_result.notes or [],
                    model=_tier_dispatch_receipt.validator_model,
                )
                _validator_result = _val_dict
                stage_runs.append(StageRun(
                    stage=Stage(stage_type="validation", description="Review implementation result"),
                    model=_tier_dispatch_receipt.validator_model,
                    duration=time.time() - _val_t0,
                    cost=_val_usage.estimated_cost if _val_usage else None,
                    summary=f"verdict: {_val_dict.get('verdict', '?')}",
                ))
            except Exception as _val_exc:
                _validator_result = {
                    "verdict": "error",
                    "summary": str(_val_exc)[:200],
                    "model": _tier_dispatch_receipt.validator_model,
                }
                if detail == "full":
                    click.echo(f"  [validator] warning: {_val_exc}")
            finally:
                spinner.stop()

        # Record validator dispatch outcome on the non-native tier dispatch receipt.
        # (_can_dispatch already excludes native, so _tier_dispatch_receipt here is
        # always the non-native receipt built at lines 742-755.)
        if _tier_dispatch_receipt is not None:
            if _validator_result is not None:
                _tier_dispatch_receipt.validator_dispatch_status = "applied"
            elif _validator_policy is not None and not _validator_policy.run:
                _tier_dispatch_receipt.validator_dispatch_status = "skipped"

        # Safety net: discard generated file changes for read-only tasks even if the
        # model ignored the read-only instruction injected into the context.
        # For dry-run, exec_result.files is already [] so this block never fires.
        if (_readonly_task or _is_sf_task) and exec_result.files:
            click.echo(f"\nRead-only task — {len(exec_result.files)} generated change(s) discarded.")
            exec_result = ExecutionResult(
                summary=exec_result.summary,
                files=[],
                notes=exec_result.notes,
                usage=exec_result.usage,
            )

        final_files = exec_result.files

        if not opencode_mode and _repo_facts is not None:
            _mismatches = check_stack_mismatch(final_files, _repo_facts)
            _sm_dec = gate.check_stack_mismatch(_mismatches)
            if _sm_dec.required:
                _lang_str = ", ".join(_repo_facts.languages) if _repo_facts.languages else "unknown"
                confirm_or_abort(f"{_sm_dec.reason} (repo stack: {_lang_str})")

        _is_review_task = "STRUCTURED_FINDINGS:" in task
        _TASK_SUFFIX_STRIP = "\n\nAfter completing your analysis"
        _task_display = task[:task.index(_TASK_SUFFIX_STRIP)].rstrip() if _TASK_SUFFIX_STRIP in task else task
        _clean_summary, _extracted_findings = _extract_structured_findings(exec_result.summary)
        _extraction_source: "str | None" = None
        if _extracted_findings:
            _extraction_source = "STRUCTURED_FINDINGS"
        elif _is_review_task:
            _extracted_findings = _extract_findings_from_model_answer(_clean_summary)
            if _extracted_findings:
                _extraction_source = "plain-text"
            elif final_files:
                _extracted_findings = _extract_findings_from_review_files(final_files)
                if _extracted_findings:
                    _extraction_source = "markdown-files"

        # Deterministic static analysis for Terraform review packs.
        # Runs unconditionally when this is a review task so the demo is reliable
        # even when the model does not produce valid STRUCTURED_FINDINGS JSON.
        # Static findings are merged AFTER model findings so model context is preserved.
        if _is_review_task:
            try:
                from openshard.review.terraform_checker import scan_terraform
                _static_findings = scan_terraform(Path.cwd())
                if _static_findings:
                    # Merge: keep model findings first, then append static ones that
                    # are not exact message duplicates.
                    _existing_msgs = {f.message for f in _extracted_findings}
                    _new_static = [
                        f for f in _static_findings if f.message not in _existing_msgs
                    ]
                    _extracted_findings = list(_extracted_findings) + _new_static
                    if _new_static:
                        _extraction_source = _extraction_source or "static-review"
            except Exception:
                pass
            _timeline.append(RunTimelineEvent(
                "static_findings_detected",
                f"Found {len(_extracted_findings)} raw findings",
                kind="scan",
                count=len(_extracted_findings),
            ))

        if _is_review_task:
            _emit_review_debug(
                effective_executor=effective_executor,
                selected_workflow=effective_workflow,
                opencode_explicit=(effective_executor == "opencode"),
                task=task,
                exec_result_summary=exec_result.summary,
                exec_result_files_count=len(exec_result.files),
                exec_result_notes=exec_result.notes or [],
                usage_info=exec_result.usage,
                extraction_source=_extraction_source,
                findings_count=len(_extracted_findings),
            )
        _findings_extra: dict = {}
        if _is_review_task:
            _findings_extra["is_review_task"] = True
        if _extracted_findings:
            _findings_extra["findings"] = [
                {
                    "severity": f.severity,
                    "message": f.message,
                    **({"path": f.path} if f.path else {}),
                    **({"line": f.line} if f.line is not None else {}),
                    "source": "static-review" if f.path else "model",
                }
                for f in _extracted_findings
            ]
            click.echo("\nReview complete")
        elif _is_review_task:
            click.echo("\nReview complete")
        else:
            click.echo("\nDone")
            click.echo(_clean_summary)
        _tier_validated = _validator_result is not None
        _tier_passed = (
            _validator_result.get("verdict", "fail") in ("pass", "warn")
            if _tier_validated else None
        )
        _mode_label = _profile_display_label(
            _profile_decision.profile if _profile_decision is not None else None,
            is_readonly=_readonly_task,
        )
        _receipt_risk = (
            _form_factor_decision.risk_level.capitalize()
            if _form_factor_decision and _form_factor_decision.risk_level
            else "Not recorded"
        )
        # Review tasks are always at least High risk regardless of routing category
        if _is_review_task and _receipt_risk in ("Not recorded", "Low"):
            _receipt_risk = "High"
        _receipt_sandbox = "Off" if (_readonly_task or _is_review_task) else "Not recorded"
        _receipt_approval = "Not required" if _readonly_task else "Not recorded"
        _receipt_index: "int | None" = None
        try:
            _log_p = Path.cwd() / _LOG_PATH
            if _log_p.exists():
                with _log_p.open("r", encoding="utf-8") as _lf:
                    _receipt_index = sum(1 for _ in _lf)
        except Exception:
            pass
        if _is_review_task:
            _timeline.append(RunTimelineEvent(
                "review_memo_rendered", "Generated review memo",
                kind="review", count=len(_extracted_findings),
            ))
        _timeline.append(RunTimelineEvent("run_completed", "Run completed", kind="run"))
        render_post_run(
            stage_runs=stage_runs,
            routing_decision=routing_decision,
            verification_attempted=_tier_validated,
            verification_passed=_tier_passed,
            readonly_task=_readonly_task,
            validator_policy=_validator_policy,
            validator_result=_validator_result,
            final_files=exec_result.files,
            detail=detail,
            notes=exec_result.notes or [],
            repo_facts=_repo_facts,
            mode_label=_mode_label,
            task=_task_display,
            usage=usage,
            run_id=_run_id,
            run_index=_receipt_index,
            risk=_receipt_risk,
            sandbox=_receipt_sandbox,
            approval=_receipt_approval,
            is_native=(effective_executor == "native"),
            exec_result_summary=_clean_summary,
            findings=_extracted_findings if _extracted_findings else None,
            is_review_task=_is_review_task,
            generator_model=getattr(generator, "model", None),
            run_timeline=[e.to_dict() for e in _timeline],
            run_label=_pack_label,
        )

        if dry_run:
            if _should_shrink(exec_result.files, no_shrink):
                _print_shrunk(exec_result.files, exec_result.summary)
            else:
                _print_dry_run(exec_result.files)
            _print_summary(start, generator, retry_triggered, final_files, usage=usage, detail=detail, model=_routed_model, stage_runs=stage_runs)
            _dr_extra: dict | None = None
            if _tier_dispatch_receipt is not None:
                _dr_extra = {"tier_dispatch_receipt": asdict(_tier_dispatch_receipt)}
            if _validator_result is not None:
                if _dr_extra is None:
                    _dr_extra = {}
                _dr_extra["validator_result"] = _validator_result
            if _validator_policy is not None:
                if _dr_extra is None:
                    _dr_extra = {}
                _dr_extra["validator_policy"] = {"run": _validator_policy.run, "reason": _validator_policy.reason}
            if _feedback_receipt is not None:
                if _dr_extra is None:
                    _dr_extra = {}
                _dr_extra.update(_feedback_receipt)
            if _findings_extra:
                if _dr_extra is None:
                    _dr_extra = {}
                _dr_extra.update(_findings_extra)
            try:
                _log_run(start, _task_display, generator, retry_triggered, final_files,
                         verification_attempted=False, verification_passed=None,
                         workspace=None, usage=usage, model=_routed_model,
                         summary=_clean_summary, notes=exec_result.notes,
                         stage_runs=stage_runs, routing_decision=routing_decision,
                         _scored=_scored, repo_facts=_repo_facts,
                         matched_skills=_matched_skills,
                         profile_decision=_profile_decision,
                         verification_plan=_verification_plan,
                         form_factor_decision=_form_factor_decision,
                         extra_metadata=_dr_extra,
                         run_timeline=[e.to_dict() for e in _timeline])
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
            result_obj.form_factor_decision = _form_factor_decision
            result_obj.result_summary = exec_result.summary
            result_obj.result_notes = exec_result.notes or []
            return result_obj

        if verify and not write:
            raise click.ClickException("--verify requires --write.")

        if write:
            if not opencode_mode:
                if effective_executor == "native" and hasattr(generator, "native_meta") and not _multi_candidate_done:
                    from openshard.native.sandbox import create_run_sandbox as _create_sandbox
                    workspace, _sb_meta = _create_sandbox(Path.cwd(), _run_id)
                    generator.native_meta.sandbox = _sb_meta
                else:
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
                if effective_executor == "native" and hasattr(generator, "build_patch_proposal"):
                    generator.build_patch_proposal(exec_result.files)
                if effective_executor == "native" and hasattr(generator, "build_change_budget_preview"):
                    generator.build_change_budget_preview()
                if effective_executor == "native" and hasattr(generator, "build_change_budget_soft_gate"):
                    _soft_gate = generator.build_change_budget_soft_gate()
                    _approval_request = None
                    if effective_executor == "native" and hasattr(generator, "build_budget_gate_approval_request"):
                        _approval_request = generator.build_budget_gate_approval_request()
                    if _soft_gate.requires_approval:
                        approval_prompt = (
                            _approval_request.prompt
                            if _approval_request is not None and _approval_request.prompt
                            else _soft_gate.reason
                        )
                        confirm_or_abort(approval_prompt)
                        if effective_executor == "native" and hasattr(generator, "build_approval_receipt"):
                            generator.build_approval_receipt(granted=True)
                if not _multi_candidate_done:
                    _write_files(exec_result.files, workspace)
                if effective_executor == "native":
                    if hasattr(generator, "record_loop_step"):
                        generator.record_loop_step("write")
                    if hasattr(generator, "record_osn_loop_step"):
                        generator.record_osn_loop_step("safe_write", "passed")
                    generator.review_diff()
                    if generator.native_meta.plan_ledger is not None:
                        from openshard.native.context import update_native_plan_ledger_status
                        generator.native_meta.plan_ledger = update_native_plan_ledger_status(
                            generator.native_meta.plan_ledger, "Write patch", "passed",
                            evidence="sandbox write complete",
                        )
                    _safe_checkpoint("sandbox_write", "passed", files=[f.path for f in exec_result.files])
            # OpenCode: workspace already created and populated before generate().

        # Native controlled verification loop — one retry, safe commands only, no --verify flag required
        if effective_executor == "native" and write and not dry_run and not verify and not _multi_candidate_done:
            generator.native_meta.edit_loop_summary = NativeEditLoopSummary(max_attempts=2)
            _edit_loop = generator.native_meta.edit_loop_summary
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
                _vst_first = "passed" if _loop_meta.passed else "failed"
                if hasattr(generator, "record_osn_loop_step"):
                    generator.record_osn_loop_step("verify", _vst_first, verification_status=_vst_first)
                _safe_checkpoint("verify", _vst_first, verification_status=_vst_first)
                record_native_edit_loop_attempt(
                    _edit_loop,
                    attempt_index=1,
                    purpose="initial",
                    files_written=[f.path for f in exec_result.files],
                    verification_status=_vst_first,
                    exit_code=_loop_code,
                    output_chars=_loop_meta.output_chars,
                )
                if _loop_code != 0:
                    # Build structured failure metadata — no raw content stored
                    _fsummary = build_failure_summary(_loop_output, exit_code=_loop_code)
                    _retry_meta = RetryMetadata(
                        retry_attempted=True,
                        retry_reason="verification_failed",
                        failure_summary=_fsummary,
                    )
                    _loop_meta.retry_metadata = _retry_meta
                    if hasattr(generator, "record_osn_loop_step"):
                        generator.record_osn_loop_step(
                            "retry_diagnosis", "passed",
                            result_summary="verification failed — diagnosis recorded",
                            metadata={"failure_summary": _fsummary, "raw_content_stored": False},
                        )
                    # Raw failure context injected into retry prompt only — never stored
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
                    _write_files(_retry_result.files, workspace)  # workspace is sandbox/worktree-backed
                    _retry_patch_files = [f.path for f in _retry_result.files]
                    _retry_meta.retry_patch_files = _retry_patch_files
                    if hasattr(generator, "record_osn_loop_step"):
                        generator.record_osn_loop_step(
                            "retry_patch", "passed",
                            result_summary=f"retry patch written: {len(_retry_patch_files)} file(s)",
                            metadata={
                                "files": _retry_patch_files,
                                "file_count": len(_retry_patch_files),
                                "raw_content_stored": False,
                            },
                        )
                    # Second verification run
                    _loop_code2, _loop_output2 = _run_verification_plan(
                        _verification_plan, workspace, capture=True
                    )
                    _loop_meta.exit_code = _loop_code2
                    _loop_meta.output_chars = len(_loop_output2)
                    _loop_meta.truncated = len(_loop_output2) > 1200
                    _loop_meta.passed = _loop_code2 == 0
                    _retry_vst = "passed" if _loop_meta.passed else "failed"
                    _retry_meta.retry_verification_status = _retry_vst
                    record_native_edit_loop_attempt(
                        _edit_loop,
                        attempt_index=2,
                        purpose="repair",
                        files_written=[f.path for f in _retry_result.files],
                        verification_status=_retry_vst,
                        exit_code=_loop_code2,
                        output_chars=len(_loop_output2),
                    )
                    if hasattr(generator, "record_osn_loop_step"):
                        generator.record_osn_loop_step(
                            "retry_once", _retry_vst, verification_status=_retry_vst,
                        )
                    _safe_checkpoint("retry", _retry_vst, retry_used=True, verification_status=_retry_vst)
                    # Failure Memory v1: one structured event per retry, best-effort
                    try:
                        _fm_parsed = parse_failure_summary(_fsummary)
                        log_failure_memory_event(NativeFailureMemoryEvent(
                            run_id=_run_id,
                            task_summary=task[:120],
                            failure_type=_fm_parsed.get("failure_type", "test_failure"),
                            exit_code=int(_fm_parsed.get("exit_code", "1")),
                            output_chars=int(_fm_parsed.get("output_chars", "0")),
                            verification_status="failed",
                            retry_attempted=True,
                            retry_succeeded=(_retry_vst == "passed"),
                            retry_patch_files=list(_retry_patch_files),
                            related_file_paths=list(_retry_patch_files),
                            model=_routed_model or generator.model,
                            workflow=effective_workflow,
                        ))
                    except Exception:
                        pass
            if generator.native_meta.plan_ledger is not None:
                from openshard.native.context import update_native_plan_ledger_status
                _vplan_status = (
                    ("passed" if _loop_meta.passed else "failed")
                    if _loop_meta.attempted
                    else "skipped"
                )
                generator.native_meta.plan_ledger = update_native_plan_ledger_status(
                    generator.native_meta.plan_ledger, "Run verification", _vplan_status
                )
            if not _loop_meta.attempted:
                record_native_edit_loop_attempt(
                    _edit_loop,
                    attempt_index=1,
                    purpose="initial",
                    files_written=[f.path for f in exec_result.files],
                    verification_status="skipped",
                    exit_code=None,
                    output_chars=0,
                )
                _safe_checkpoint("verify", "skipped", reason="verification not configured")

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
                    _vf_extra: dict | None = None
                    if _tier_dispatch_receipt is not None:
                        from dataclasses import asdict as _asdict
                        _vf_extra = {"tier_dispatch_receipt": _asdict(_tier_dispatch_receipt)}
                    if _validator_result is not None:
                        if _vf_extra is None:
                            _vf_extra = {}
                        _vf_extra["validator_result"] = _validator_result
                    if _validator_policy is not None:
                        if _vf_extra is None:
                            _vf_extra = {}
                        _vf_extra["validator_policy"] = {"run": _validator_policy.run, "reason": _validator_policy.reason}
                    if _feedback_receipt is not None:
                        if _vf_extra is None:
                            _vf_extra = {}
                        _vf_extra.update(_feedback_receipt)
                    if _findings_extra:
                        if _vf_extra is None:
                            _vf_extra = {}
                        _vf_extra.update(_findings_extra)
                    _log_run(start, _task_display, generator, retry_triggered, final_files,
                             verification_attempted=True, verification_passed=False,
                             workspace=workspace, usage=usage, retry_usage=retry_usage, model=_routed_model,
                             summary=_clean_summary, notes=exec_result.notes,
                             stage_runs=stage_runs, routing_decision=routing_decision,
                             _scored=_scored, repo_facts=_repo_facts,
                             matched_skills=_matched_skills,
                             profile_decision=_profile_decision,
                             verification_plan=_verification_plan,
                             form_factor_decision=_form_factor_decision,
                             extra_metadata=_vf_extra,
                             run_timeline=[e.to_dict() for e in _timeline])
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
                result_obj.form_factor_decision = _form_factor_decision
                result_obj.verification_attempted = True
                result_obj.verification_passed = False
                result_obj.workspace = workspace
                result_obj.result_summary = exec_result.summary
                result_obj.result_notes = exec_result.notes or []
                return result_obj

        _print_summary(start, generator, retry_triggered, final_files,
                       usage=usage, retry_usage=retry_usage, detail=detail, model=_routed_model, stage_runs=stage_runs)
        if (
            effective_executor == "native"
            and hasattr(generator, "build_verification_command_summary")
            and generator.native_meta.verification_loop is not None
            and generator.native_meta.verification_loop.attempted
        ):
            generator.build_verification_command_summary(_verification_plan)
        if effective_executor == "native" and hasattr(generator, "build_final_report"):
            generator.build_final_report()
        _native_meta = generator.native_meta if effective_executor == "native" else None
        if _native_meta is not None:
            from openshard.native.context import build_native_failure_memory
            _native_meta.failure_memory = build_native_failure_memory(
                context_quality_score=_native_meta.context_quality_score,
                clarification_request=_native_meta.clarification_request,
                verification_loop=_native_meta.verification_loop,
                command_policy_preview=_native_meta.command_policy_preview,
                approval_request=_native_meta.approval_request,
                approval_receipt=_native_meta.approval_receipt,
                change_budget_preview=_native_meta.change_budget_preview,
                verification_plan=_native_meta.verification_plan,
                context_usage_summary=_native_meta.context_usage_summary,
            )
            from openshard.native.context import build_native_run_trust_score
            _native_meta.run_trust_score = build_native_run_trust_score(
                context_quality_score=_native_meta.context_quality_score,
                validation_contract=_native_meta.validation_contract,
                context_provenance=_native_meta.context_provenance,
                verification_loop=_native_meta.verification_loop,
                command_policy_preview=_native_meta.command_policy_preview,
                change_budget_preview=_native_meta.change_budget_preview,
                change_budget_soft_gate=_native_meta.change_budget_soft_gate,
                approval_request=_native_meta.approval_request,
                approval_receipt=_native_meta.approval_receipt,
                failure_memory=_native_meta.failure_memory,
                context_usage_summary=_native_meta.context_usage_summary,
                final_report=_native_meta.final_report,
            )
            from openshard.native.context import build_native_verification_contract_result
            _native_meta.verification_contract_result = build_native_verification_contract_result(
                validation_contract=_native_meta.validation_contract,
                verification_loop=_native_meta.verification_loop,
            )
            from openshard.native.context import build_native_model_selection_decision
            _native_meta.model_selection_decision = build_native_model_selection_decision(
                verification_plan=_native_meta.verification_plan,
                validation_contract=_native_meta.validation_contract,
                context_quality_score=_native_meta.context_quality_score,
                context_provenance=_native_meta.context_provenance,
                run_trust_score=_native_meta.run_trust_score,
                change_budget=_native_meta.change_budget,
                failure_memory=_native_meta.failure_memory,
            )
            from openshard.native.context import build_native_model_candidate_scoring
            _native_meta.model_candidate_scoring = build_native_model_candidate_scoring(
                model_selection_decision=_native_meta.model_selection_decision,
                verification_plan=_native_meta.verification_plan,
                validation_contract=_native_meta.validation_contract,
                context_quality_score=_native_meta.context_quality_score,
                context_provenance=_native_meta.context_provenance,
                run_trust_score=_native_meta.run_trust_score,
                context_usage_summary=_native_meta.context_usage_summary,
                failure_memory=_native_meta.failure_memory,
                model_policy=_native_meta.model_policy,
            )
            import copy
            _before_sync = copy.deepcopy(_native_meta.model_selection_decision)
            from openshard.native.context import sync_native_model_selection_decision_with_candidate_scoring
            _native_meta.model_selection_decision = sync_native_model_selection_decision_with_candidate_scoring(
                model_selection_decision=_native_meta.model_selection_decision,
                model_candidate_scoring=_native_meta.model_candidate_scoring,
                model_policy=_native_meta.model_policy,
            )
            from openshard.native.context import build_native_model_policy_receipt
            _native_meta.model_policy_receipt = build_native_model_policy_receipt(
                model_policy=_native_meta.model_policy,
                model_selection_decision_before=_before_sync,
                model_selection_decision_after=_native_meta.model_selection_decision,
                model_candidate_scoring=_native_meta.model_candidate_scoring,
            )
            from openshard.native.context import build_native_routing_preview
            _native_meta.routing_preview = build_native_routing_preview(
                model_candidate_scoring=_native_meta.model_candidate_scoring,
                model_selection_decision=_native_meta.model_selection_decision,
                model_policy_receipt=_native_meta.model_policy_receipt,
                run_trust_score=_native_meta.run_trust_score,
            )
            from openshard.native.context import build_native_routing_receipt
            _native_meta.routing_receipt = build_native_routing_receipt(
                routing_preview=_native_meta.routing_preview,
                model_policy_receipt=_native_meta.model_policy_receipt,
                run_trust_score=_native_meta.run_trust_score,
            )
            if self._experimental_tier_dispatch:
                from openshard.native.context import build_native_tier_dispatch_receipt
                _native_meta.tier_dispatch_receipt = build_native_tier_dispatch_receipt(
                    routing_receipt=_native_meta.routing_receipt,
                    model_candidate_scoring=_native_meta.model_candidate_scoring,
                    routing_category=routing_decision.category if routing_decision else None,
                    experimental_tier_dispatch=True,
                    applied=False,
                    not_applied_reason="routing decisions resolved post-execution",
                    executor_model_actual=_effective_model,
                    validator_dispatch_status="reserved",
                )
                _tier_dispatch_receipt = _native_meta.tier_dispatch_receipt
        if _native_meta is not None:
            from openshard.native.context import build_native_failure_memory_routing_advisory
            _native_meta.failure_memory_routing_advisory = (
                build_native_failure_memory_routing_advisory()
            )
        if _native_meta is not None and detail != "default":
            from openshard.cli.run_output import _print_native_demo_block, _print_native_summary
            _print_native_demo_block(_native_meta, detail=detail)
            _print_native_summary(_native_meta, detail=detail)
        if _native_meta is None and _tier_dispatch_receipt is not None and detail != "default":
            from openshard.cli.run_output import _print_tier_dispatch_block
            _is_direct_ask = _readonly_task and not _use_stages
            _print_tier_dispatch_block(_tier_dispatch_receipt, detail, validator_result=_validator_result, validator_policy=_validator_policy, is_ask=_is_direct_ask)
        if _native_meta is not None and _native_meta.plan_ledger is not None:
            from openshard.native.context import update_native_plan_ledger_status
            _native_meta.plan_ledger = update_native_plan_ledger_status(
                _native_meta.plan_ledger, "Record receipt", "passed"
            )
        if effective_executor == "native":
            _safe_checkpoint("receipt", "passed")
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
                "tool_search_events": [
                    asdict(e) if not isinstance(e, dict) else e
                    for e in _native_meta.tool_search_events
                ],
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
                "verification_command_summary": (
                    asdict(_native_meta.verification_command_summary)
                    if _native_meta is not None
                    and _native_meta.verification_command_summary is not None
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
                "read_search_findings": list(getattr(_native_meta, "read_search_findings", [])),
                "context_packet": (
                    asdict(_native_meta.context_packet)
                    if _native_meta is not None and _native_meta.context_packet is not None
                    else None
                ),
                "patch_proposal": (
                    asdict(_native_meta.patch_proposal)
                    if _native_meta.patch_proposal is not None
                    else None
                ),
                "command_policy_preview": (
                    asdict(_native_meta.command_policy_preview)
                    if _native_meta is not None
                    and _native_meta.command_policy_preview is not None
                    else None
                ),
                "file_context": (
                    asdict(_native_meta.file_context)
                    if _native_meta is not None and _native_meta.file_context is not None
                    else None
                ),
                "context_quality_score": (
                    asdict(_native_meta.context_quality_score)
                    if _native_meta is not None and _native_meta.context_quality_score is not None
                    else None
                ),
                "context_quality_advisory": (
                    asdict(_native_meta.context_quality_advisory)
                    if _native_meta is not None
                    and _native_meta.context_quality_advisory is not None
                    else None
                ),
                "change_budget": (
                    asdict(_native_meta.change_budget)
                    if _native_meta is not None and _native_meta.change_budget is not None
                    else None
                ),
                "change_budget_preview": (
                    asdict(_native_meta.change_budget_preview)
                    if _native_meta is not None and _native_meta.change_budget_preview is not None
                    else None
                ),
                "change_budget_soft_gate": (
                    asdict(_native_meta.change_budget_soft_gate)
                    if _native_meta is not None and _native_meta.change_budget_soft_gate is not None
                    else None
                ),
                "approval_request": (
                    asdict(_native_meta.approval_request)
                    if _native_meta is not None and _native_meta.approval_request is not None
                    else None
                ),
                "approval_receipt": (
                    asdict(_native_meta.approval_receipt)
                    if _native_meta is not None and _native_meta.approval_receipt is not None
                    else None
                ),
                "verification_plan": (
                    asdict(_native_meta.verification_plan)
                    if _native_meta is not None and _native_meta.verification_plan is not None
                    else None
                ),
                "clarification_request": (
                    asdict(_native_meta.clarification_request)
                    if _native_meta is not None and _native_meta.clarification_request is not None
                    else None
                ),
                "context_usage_summary": (
                    asdict(_native_meta.context_usage_summary)
                    if _native_meta is not None and _native_meta.context_usage_summary is not None
                    else None
                ),
                "failure_memory": (
                    asdict(_native_meta.failure_memory)
                    if _native_meta is not None and _native_meta.failure_memory is not None
                    else None
                ),
                "osn_loop": (
                    asdict(_native_meta.osn_loop)
                    if _native_meta is not None and _native_meta.osn_loop is not None
                    else None
                ),
                "osn_loop_summary": (
                    asdict(_native_meta.osn_loop_summary)
                    if _native_meta is not None and _native_meta.osn_loop_summary is not None
                    else None
                ),
                "deepagents_adapter": (
                    asdict(_native_meta.deepagents_adapter)
                    if _native_meta is not None and _native_meta.deepagents_adapter is not None
                    else None
                ),
                "validation_contract": (
                    asdict(_native_meta.validation_contract)
                    if _native_meta is not None and _native_meta.validation_contract is not None
                    else None
                ),
                "verification_contract_result": (
                    asdict(_native_meta.verification_contract_result)
                    if _native_meta is not None and _native_meta.verification_contract_result is not None
                    else None
                ),
                "context_provenance": (
                    asdict(_native_meta.context_provenance)
                    if _native_meta is not None and _native_meta.context_provenance is not None
                    else None
                ),
                "run_trust_score": (
                    asdict(_native_meta.run_trust_score)
                    if _native_meta is not None and _native_meta.run_trust_score is not None
                    else None
                ),
                "model_selection_decision": (
                    asdict(_native_meta.model_selection_decision)
                    if _native_meta is not None and _native_meta.model_selection_decision is not None
                    else None
                ),
                "model_candidate_scoring": (
                    asdict(_native_meta.model_candidate_scoring)
                    if _native_meta is not None and _native_meta.model_candidate_scoring is not None
                    else None
                ),
                "model_policy": (
                    asdict(_native_meta.model_policy)
                    if _native_meta is not None and _native_meta.model_policy is not None
                    else None
                ),
                "model_policy_receipt": (
                    asdict(_native_meta.model_policy_receipt)
                    if _native_meta is not None and _native_meta.model_policy_receipt is not None
                    else None
                ),
                "routing_preview": (
                    asdict(_native_meta.routing_preview)
                    if _native_meta is not None and _native_meta.routing_preview is not None
                    else None
                ),
                "routing_receipt": (
                    asdict(_native_meta.routing_receipt)
                    if _native_meta is not None and _native_meta.routing_receipt is not None
                    else None
                ),
                "tier_dispatch_receipt": (
                    asdict(_native_meta.tier_dispatch_receipt)
                    if _native_meta is not None and _native_meta.tier_dispatch_receipt is not None
                    else None
                ),
                "sandbox": (
                    asdict(_native_meta.sandbox)
                    if _native_meta is not None and _native_meta.sandbox is not None
                    else None
                ),
                "failure_memory_routing_advisory": (
                    asdict(_native_meta.failure_memory_routing_advisory)
                    if _native_meta is not None
                    and _native_meta.failure_memory_routing_advisory is not None
                    else None
                ),
                "plan_ledger": (
                    asdict(_native_meta.plan_ledger)
                    if _native_meta.plan_ledger is not None
                    else None
                ),
                "edit_loop_summary": (
                    asdict(_native_meta.edit_loop_summary)
                    if _native_meta.edit_loop_summary is not None
                    else None
                ),
                "candidate_summary": (
                    asdict(_native_meta.candidate_summary)
                    if _native_meta.candidate_summary is not None
                    else None
                ),
            }
        # For non-native staged/direct runs, save tier_dispatch_receipt at top level
        if _native_meta is None and _tier_dispatch_receipt is not None:
            from dataclasses import asdict as _asdict
            _extra_metadata = {"tier_dispatch_receipt": _asdict(_tier_dispatch_receipt)}
        if _native_meta is None and _validator_result is not None:
            if _extra_metadata is None:
                _extra_metadata = {}
            _extra_metadata["validator_result"] = _validator_result
        if _native_meta is None and _validator_policy is not None:
            if _extra_metadata is None:
                _extra_metadata = {}
            _extra_metadata["validator_policy"] = {"run": _validator_policy.run, "reason": _validator_policy.reason}
        if _feedback_receipt is not None:
            if _extra_metadata is None:
                _extra_metadata = {}
            _extra_metadata.update(_feedback_receipt)
        # Finalise OSN loop summary before serialising to run history
        if effective_executor == "native" and hasattr(generator, "complete_osn_loop"):
            _vloop = getattr(generator.native_meta, "verification_loop", None)
            _approval_rcpt = getattr(generator.native_meta, "approval_receipt", None)
            generator.record_osn_loop_step("final_receipt", "passed")
            generator.complete_osn_loop(
                stopped_reason="completed",
                verification_status=(
                    "passed" if (_vloop and _vloop.passed)
                    else ("failed" if (_vloop and _vloop.attempted) else "")
                ),
                retry_used=bool(_vloop and _vloop.retried),
                approval_granted=bool(_approval_rcpt and _approval_rcpt.granted),
            )
            # Refresh snapshot: _extra_metadata was built before complete_osn_loop ran
            if _extra_metadata is not None and _native_meta is not None:
                _extra_metadata["osn_loop_summary"] = (
                    asdict(_native_meta.osn_loop_summary)
                    if _native_meta.osn_loop_summary is not None
                    else None
                )
        if _findings_extra:
            if _extra_metadata is None:
                _extra_metadata = {}
            _extra_metadata.update(_findings_extra)
        try:
            _log_run(start, _task_display, generator, retry_triggered, final_files,
                     verification_attempted=(write and verify),
                     verification_passed=verification_passed,
                     workspace=workspace, usage=usage, retry_usage=retry_usage, model=_routed_model,
                     summary=_clean_summary, notes=exec_result.notes,
                     stage_runs=stage_runs, routing_decision=routing_decision,
                     _scored=_scored, repo_facts=_repo_facts,
                     matched_skills=_matched_skills,
                     profile_decision=_profile_decision,
                     verification_plan=_verification_plan,
                     form_factor_decision=_form_factor_decision,
                     extra_metadata=_extra_metadata,
                     run_id=_run_id,
                     run_timeline=[e.to_dict() for e in _timeline])
        except Exception as exc:
            click.echo(f"  [log] warning: {exc}")

        if _native_meta is not None and detail == "default":
            from openshard.cost.baseline import format_baseline_line
            _pt = getattr(usage, "prompt_tokens", 0) or 0
            _ct = getattr(usage, "completion_tokens", 0) or 0
            _actual = getattr(usage, "estimated_cost", None)
            _bl = format_baseline_line(_pt, _ct, actual_cost=_actual)
            if _bl is not None:
                click.echo(_bl)
            from openshard.cli.run_output import _print_native_receipt
            _print_native_receipt(_native_meta)

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
        result_obj.form_factor_decision = _form_factor_decision
        result_obj.verification_attempted = (write and verify)
        result_obj.verification_passed = verification_passed
        result_obj.workspace = workspace
        result_obj.result_summary = exec_result.summary
        result_obj.result_notes = exec_result.notes or []
        result_obj.native_meta = _native_meta
        return result_obj


_LOG_PATH = Path(".openshard") / "runs.jsonl"

_NATIVE_SIGNALS: list[tuple[list[str], str]] = [
    (["all files", "every file", "multiple files", "many files"], "multi-file scope"),
    (["throughout the", "across the codebase", "across all files"], "multi-file scope"),
    # refactor / architecture / codebase are now handled in direct mode via minimax
]


def _suggest_executor(task: str) -> tuple[str, str]:
    """Return ``(executor, reason)`` recommendation for *task*.

    - Defaults to ``"direct"`` for short, focused tasks.
    - Returns ``"native"`` for large, multi-file, or complex tasks.
    - ``"opencode"`` is never returned here; it requires explicit ``--workflow opencode``.
    """
    lower = task.lower()
    if len(task.split()) > 60:
        return "native", "large or complex task"
    for keywords, label in _NATIVE_SIGNALS:
        if any(kw in lower for kw in keywords):
            return "native", label
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
    form_factor_decision: ExecutionFormFactorDecision | None = None,
    extra_metadata: dict | None = None,
    run_id: str | None = None,
    run_timeline: list | None = None,
) -> None:
    entry: dict = {
        "timestamp": run_id if run_id else datetime.datetime.now(datetime.timezone.utc).isoformat().replace("+00:00", "Z"),
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

    if form_factor_decision is not None:
        entry["form_factor"] = {
            "public_mode":            form_factor_decision.public_mode,
            "internal_form_factor":   form_factor_decision.internal_form_factor,
            "reason":                 form_factor_decision.reason,
            "confidence":             form_factor_decision.confidence,
            "risk_level":             form_factor_decision.risk_level,
            "read_only":              form_factor_decision.read_only,
            "write_requested":        form_factor_decision.write_requested,
            "verification_available": form_factor_decision.verification_available,
            "context_quality":        form_factor_decision.context_quality,
            "warnings":               form_factor_decision.warnings,
        }

    if verification_plan is not None and verification_plan.has_commands:
        entry["verification_plan"] = _serialize_verification_plan(verification_plan)

    if extra_metadata:
        entry.update(extra_metadata)

    if run_timeline is not None:
        _tl = list(run_timeline)
        _tl.append(RunTimelineEvent("receipt_saved", "Saved Shard receipt", kind="receipt").to_dict())
        entry["run_timeline"] = _tl

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
