import datetime
import json
import re
import shutil
import time
from pathlib import Path

from openshard.analysis.repo import RepoFacts
from openshard.execution.generator import ChangedFile, ExecutionGenerator, ExecutionResult
from openshard.providers.openrouter import MODEL_PRICING, compute_cost
from openshard.routing.profiles import ProfileDecision
from openshard.scoring.scorer import ScoredRoutingResult
from openshard.skills.matcher import MatchedSkill
from openshard.verification.plan import VerificationPlan

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
