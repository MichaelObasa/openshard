import re
import sys
import threading
import time
from types import SimpleNamespace
from typing import Any

import click

from openshard.analysis.repo import RepoFacts
from openshard.execution.generator import ChangedFile, ExecutionGenerator
from openshard.execution.stages import StageRun
from openshard.routing.engine import MODEL_STRONG, RoutingDecision


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
    "read-only analysis":                                 "read-only analysis",
}

_ABBREV_WORDS = {"gpt", "llm", "ai", "api", "url", "id", "ui", "ml"}

_PROFILE_LABEL: dict[str, str] = {
    "native_light": "Run",
    "native_deep": "Deep Run",
    "native_swarm": "Deep Run",
    "native": "OSN Run",
}


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


def _profile_display_label(profile: str | None, is_readonly: bool = False) -> str:
    if is_readonly:
        return "Ask"
    if not profile:
        return "-"
    return _PROFILE_LABEL.get(profile, profile.replace("_", " ").title())


def _build_routing_line(
    routing_decision: RoutingDecision | None,
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
        "read-only analysis":                                 f"{label} analysing",
    }
    return "Executing - " + desc.get(rationale, f"{label} running task")


def _build_model_line(
    routing_decision: RoutingDecision | None,
    stage_runs: list[StageRun],
    model: str | None = None,
) -> str | None:
    """Return a single 'Model: ...' or 'Models: ...' line for default output."""
    _is_ro = routing_decision is not None and routing_decision.rationale == "read-only analysis"
    if stage_runs:
        seen: dict[str, list[str]] = {}
        for sr in stage_runs:
            label = _model_label(sr.model)
            stype = "analysis" if (_is_ro and sr.stage.stage_type == "implementation") else sr.stage.stage_type
            seen.setdefault(label, []).append(stype)
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


def _print_native_summary(native_meta: Any, detail: str = "default") -> None:
    """Render compact native execution summary."""
    report = getattr(native_meta, "final_report", None)
    if report is None:
        return

    click.echo("\n[native summary]")

    ctx = "yes" if report.used_native_context else "no"
    click.echo(f"  context: {ctx}")

    if report.selected_skills:
        click.echo(f"  skills: {', '.join(report.selected_skills)}")

    if report.plan_intent or report.plan_risk:
        parts = [p for p in [report.plan_intent, report.plan_risk] if p]
        click.echo(f"  plan: {' / '.join(parts)}")

    evidence_parts = []
    if report.evidence_items:
        evidence_parts.append(f"{report.evidence_items} items")
    if report.snippet_files:
        evidence_parts.append(f"{report.snippet_files} snippets")
    if evidence_parts:
        click.echo(f"  evidence: {', '.join(evidence_parts)}")

    if report.verification_attempted:
        v_parts = []
        if report.verification_retried:
            v_parts.append("retried")
        v_parts.append("passed" if report.verification_passed else "failed")
        click.echo(f"  verification: {', '.join(v_parts)}")

    if report.diff_files:
        n = len(report.diff_files)
        file_word = "file" if n == 1 else "files"
        click.echo(
            f"  diff: {n} {file_word}, +{report.added_lines} / -{report.removed_lines}"
        )

    for warning in report.warnings:
        click.echo(f"  warning: {warning}")


def _render_native_receipt(native_meta: Any) -> str:
    """Return a clean one-line execution receipt for a native run. Pure, no I/O."""
    parts: list[str] = []

    report = getattr(native_meta, "final_report", None)
    diff_review = getattr(native_meta, "diff_review", None)

    n_files = 0
    if report is not None and getattr(report, "diff_files", None):
        n_files = len(report.diff_files)
    elif diff_review is not None:
        n_files = len(getattr(diff_review, "changed_files", []) or [])

    if n_files == 1:
        parts.append("1 file changed")
    elif n_files > 0:
        parts.append(f"{n_files} files changed")

    if report is not None and getattr(report, "verification_attempted", False):
        parts.append(
            "Verification passed"
            if getattr(report, "verification_passed", False)
            else "Verification failed"
        )

    approval_receipt = getattr(native_meta, "approval_receipt", None)
    if approval_receipt is not None and getattr(approval_receipt, "granted", False):
        parts.append("Write approved")
    else:
        parts.append("No risky writes")

    parts.append("Receipt saved")
    return ". ".join(parts) + "."


def _print_native_receipt(native_meta: Any) -> None:
    receipt = _render_native_receipt(native_meta)
    if receipt:
        click.echo(receipt)


def _loop_event_value(event: Any, key: str, default: Any = "") -> Any:
    if isinstance(event, dict):
        return event.get(key, default)
    return getattr(event, key, default)


def _render_native_demo_block(native_meta: Any, detail: str = "default") -> list[str]:
    """Return ordered indented lines for the [native] block. Pure, no I/O."""
    if native_meta is None:
        return []

    lines: list[str] = []
    has_content = False

    repo_summary = getattr(native_meta, "repo_context_summary", None)
    if repo_summary is not None:
        has_content = True
        stack = getattr(repo_summary, "likely_stack_markers", [])
        tests = getattr(repo_summary, "test_markers", [])
        parts: list[str] = list(stack) if stack else ["unknown"]
        if tests:
            parts.append("tests detected")
        lines.append(f"  repo: {', '.join(parts)}")

    native_backend = getattr(native_meta, "native_backend", None)
    if native_backend:
        _backend_available = getattr(native_meta, "native_backend_available", True)
        _backend_suffix = "" if _backend_available else " unavailable"
        lines.append(f"  backend: {native_backend}{_backend_suffix}")
        has_content = True

    backend_proof = getattr(native_meta, "native_backend_proof", None)
    if backend_proof:
        proof_mode = backend_proof.get("mode", "unknown")
        readable = proof_mode.replace("_", " ")
        lines.append(f"  proof: {readable}")
        has_content = True

    deepagents_adapter = getattr(native_meta, "deepagents_adapter", None)
    if deepagents_adapter is not None:
        _da_mode = getattr(deepagents_adapter, "mode", "unknown")
        _da_readable = _da_mode.replace("_", " ")
        _da_version = getattr(deepagents_adapter, "version", None)
        _da_suffix = f" [v{_da_version}]" if _da_version else ""
        lines.append(f"  deepagents adapter: {_da_readable}{_da_suffix}")
        _da_notes = getattr(deepagents_adapter, "notes", []) or []
        for note in _da_notes:
            lines.append(f"  deepagents adapter note: {note}")
        has_content = True

    observation = getattr(native_meta, "observation", None)
    if observation is not None:
        has_content = True
        dirty = "yes" if getattr(observation, "dirty_diff_present", False) else "no"
        search_ev = "yes" if getattr(observation, "search_matches_count", 0) > 0 else "no"
        lines.append(f"  observation: dirty tree {dirty}, search evidence {search_ev}")

    plan = getattr(native_meta, "plan", None)
    if plan is not None:
        has_content = True
        intent = getattr(plan, "intent", "")
        risk = getattr(plan, "risk", "")
        lines.append(f"  plan: {intent} / {risk}")

    write_path = getattr(native_meta, "write_path", "pipeline")
    lines.append(f"  write path: {write_path}")

    vloop = getattr(native_meta, "verification_loop", None)
    if vloop is not None and getattr(vloop, "attempted", False):
        has_content = True
        v_parts: list[str] = []
        if getattr(vloop, "retried", False):
            v_parts.append("retried")
        v_parts.append("passed" if getattr(vloop, "passed", False) else "failed")
        lines.append(f"  verification: {', '.join(v_parts)}")

    vcs = getattr(native_meta, "verification_command_summary", None)
    if vcs is not None and getattr(vcs, "command_count", 0) > 0:
        lines.append(
            f"  verification commands: {getattr(vcs, 'safe_count', 0)} safe, "
            f"{getattr(vcs, 'needs_approval_count', 0)} approval, "
            f"{getattr(vcs, 'blocked_count', 0)} blocked"
        )
        has_content = True

    cpp = getattr(native_meta, "command_policy_preview", None)
    if cpp is not None:
        _cpp_total = (
            getattr(cpp, "safe_count", 0)
            + getattr(cpp, "needs_approval_count", 0)
            + getattr(cpp, "blocked_count", 0)
        )
        if _cpp_total > 0:
            lines.append(
                f"  command policy: {getattr(cpp, 'safe_count', 0)} safe, "
                f"{getattr(cpp, 'needs_approval_count', 0)} approval, "
                f"{getattr(cpp, 'blocked_count', 0)} blocked"
            )
            has_content = True

    diff_review = getattr(native_meta, "diff_review", None)
    if diff_review is not None and getattr(diff_review, "has_diff", False):
        has_content = True
        changed = getattr(diff_review, "changed_files", [])
        n = len(changed)
        file_word = "file" if n == 1 else "files"
        added = getattr(diff_review, "added_lines", 0)
        removed = getattr(diff_review, "removed_lines", 0)
        lines.append(f"  diff: {n} {file_word}, +{added} / -{removed}")

    loop_steps = getattr(native_meta, "native_loop_steps", None)
    if loop_steps:
        lines.append(f"  loop: {' -> '.join(loop_steps)}")
        has_content = True

    read_search_findings = getattr(native_meta, "read_search_findings", None)
    if read_search_findings:
        lines.append(f"  read/search: {len(read_search_findings)} findings")
        has_content = True

    osn_loop = getattr(native_meta, "osn_loop", None)
    if osn_loop is not None and getattr(osn_loop, "enabled", False):
        steps_run = getattr(osn_loop, "steps_run", 0)
        max_steps = getattr(osn_loop, "max_steps", 0)
        paths = getattr(osn_loop, "paths_surfaced", []) or []
        reason = getattr(osn_loop, "terminated_reason", "")
        trunc = ", truncated" if getattr(osn_loop, "truncated", False) else ""
        lines.append(
            f"  osn loop: {steps_run}/{max_steps} steps,"
            f" {len(paths)} paths, reason={reason}{trunc}"
        )
        has_content = True

    cp = getattr(native_meta, "context_packet", None)
    if cp is not None:
        _cp_sources = len(getattr(cp, "sources", []) or [])
        _cp_paths = len(getattr(cp, "compact_paths", []) or [])
        if _cp_sources > 0:
            lines.append(f"  context packet: {_cp_sources} sources, {_cp_paths} paths")
            has_content = True

    cqs = getattr(native_meta, "context_quality_score", None)
    if cqs is not None:
        _cqs_level = getattr(cqs, "level", "unknown")
        _cqs_score = getattr(cqs, "score", 0)
        _cqs_max = getattr(cqs, "max_score", 100)
        lines.append(f"  context quality: {_cqs_level} {_cqs_score}/{_cqs_max}")
        has_content = True

    cqa = getattr(native_meta, "context_quality_advisory", None)
    if cqa is not None:
        recommendation = getattr(cqa, "recommendation", "")
        if recommendation:
            lines.append(f"  context advisory: {recommendation}")
            has_content = True

    _report_for_usage = getattr(native_meta, "final_report", None)
    if _report_for_usage is not None:
        _used = "yes" if getattr(_report_for_usage, "used_native_context", False) else "no"
        _usage_parts = [f"used={_used}"]
        _ev = getattr(_report_for_usage, "evidence_items", 0)
        _sn = getattr(_report_for_usage, "snippet_files", 0)
        if _ev:
            _usage_parts.append(f"evidence={_ev} items")
        if _sn:
            _usage_parts.append(f"{_sn} snippets")
        lines.append(f"  context usage: {', '.join(_usage_parts)}")
        has_content = True

    budget = getattr(native_meta, "change_budget", None)
    if budget is not None:
        max_files = getattr(budget, "max_files", 0)
        size = getattr(budget, "max_change_size", "")
        if max_files:
            lines.append(f"  change budget: {max_files} files, {size}")
            has_content = True

    vplan = getattr(native_meta, "verification_plan", None)
    if vplan is not None:
        _vplan_type = getattr(vplan, "task_type", "unknown")
        _vplan_risk = getattr(vplan, "risk_level", "unknown")
        lines.append(f"  verification plan: {_vplan_type}, risk={_vplan_risk}")
        has_content = True
        _vplan_scope = getattr(vplan, "likely_files_or_folders", []) or []
        if _vplan_scope:
            _scope_str = ", ".join(_vplan_scope[:3])
            if len(_vplan_scope) > 3:
                _scope_str += f" (+{len(_vplan_scope) - 3})"
            lines.append(f"  vplan scope: {_scope_str}")
        _vplan_cmds = getattr(vplan, "suggested_verification_commands", []) or []
        if _vplan_cmds:
            lines.append(f"  vplan verify: {_vplan_cmds[0]}")
        _vplan_blocked = getattr(vplan, "blocked_commands", []) or []
        if _vplan_blocked:
            lines.append("  vplan policy: blocked destructive/network commands")

    clarification_request = getattr(native_meta, "clarification_request", None)
    if clarification_request is not None and getattr(clarification_request, "needed", False):
        _cr_question = getattr(clarification_request, "question", None) or ""
        _cr_opts = getattr(clarification_request, "options", []) or []
        _cr_custom = getattr(clarification_request, "allows_custom", False)
        _cr_label = f'"{_cr_question}"' if _cr_question else "clarification needed"
        lines.append(f"  clarification: needed — {_cr_label}")
        _opt_str = f"{len(_cr_opts)} option{'s' if len(_cr_opts) != 1 else ''}"
        if _cr_custom:
            _opt_str += " + custom answer allowed"
        lines.append(f"  clarification options: {_opt_str}")
        has_content = True

    validation_contract = getattr(native_meta, "validation_contract", None)
    if validation_contract is not None:
        _vc_strength = getattr(validation_contract, "strength", "unknown")
        _vc_checks = len(getattr(validation_contract, "acceptance_checks", None) or [])
        _vc_cmds = len(getattr(validation_contract, "verification_commands", None) or [])
        _checks_word = "check" if _vc_checks == 1 else "checks"
        _cmds_word = "verification command" if _vc_cmds == 1 else "verification commands"
        lines.append(
            f"  validation contract: {_vc_strength}, {_vc_checks} {_checks_word}, {_vc_cmds} {_cmds_word}"
        )
        has_content = True
        if detail == "full":
            from openshard.native.context import NativeValidationContract as _NVC
            from openshard.native.context import render_native_validation_contract as _render_vc
            _vc_obj = _NVC(
                intent=getattr(validation_contract, "intent", ""),
                risk_level=getattr(validation_contract, "risk_level", "unknown"),
                expected_change_scope=getattr(validation_contract, "expected_change_scope", "unknown"),
                acceptance_checks=list(getattr(validation_contract, "acceptance_checks", None) or []),
                verification_commands=list(getattr(validation_contract, "verification_commands", None) or []),
                approval_expected=getattr(validation_contract, "approval_expected", False),
                strength=getattr(validation_contract, "strength", "weak"),
                warnings=list(getattr(validation_contract, "warnings", None) or []),
            )
            _full_contract = _render_vc(_vc_obj)
            if _full_contract:
                for _vc_line in _full_contract.splitlines():
                    lines.append(f"  {_vc_line}")

    budget_preview = getattr(native_meta, "change_budget_preview", None)
    if budget_preview is not None:
        _proposed = getattr(budget_preview, "proposed_files", 0)
        _max_files = getattr(budget_preview, "budget_max_files", 0)
        _action = getattr(budget_preview, "action", "")
        if _max_files:
            lines.append(f"  budget preview: {_proposed}/{_max_files} files, {_action}")
            has_content = True

    gate = getattr(native_meta, "change_budget_soft_gate", None)
    if gate is not None:
        action = getattr(gate, "action", "")
        requires = getattr(gate, "requires_approval", False)
        if action:
            lines.append(f"  budget gate: {action}, approval={str(requires).lower()}")
            has_content = True

    approval = getattr(native_meta, "approval_request", None)
    if approval is not None:
        requires = getattr(approval, "requires_approval", False)
        source = getattr(approval, "source", "")
        if source:
            lines.append(f"  approval request: {source}, required={str(requires).lower()}")
            has_content = True

    receipt = getattr(native_meta, "approval_receipt", None)
    if receipt is not None:
        source = getattr(receipt, "source", "")
        granted = getattr(receipt, "granted", False)
        if source:
            lines.append(f"  approval receipt: {source}, granted={str(granted).lower()}")
            has_content = True

    patch_proposal = getattr(native_meta, "patch_proposal", None)
    if patch_proposal is not None:
        _count = getattr(patch_proposal, "file_count", 0)
        lines.append(f"  proposal: {_count} files")
        has_content = True

    fc = getattr(native_meta, "file_context", None)
    if fc is not None:
        _fc_files = getattr(fc, "files_read", 0)
        if _fc_files > 0:
            _fc_chars = getattr(fc, "total_chars", 0)
            _fc_trunc = ", truncated" if getattr(fc, "truncated", False) else ""
            lines.append(f"  file context: {_fc_files} files, {_fc_chars} chars{_fc_trunc}")
            has_content = True

    tool_search_events = getattr(native_meta, "tool_search_events", None) or []
    if tool_search_events:
        count = len(tool_search_events)
        lines.append(f"  tool/search events: {count}")
        if detail == "full":
            for ev in tool_search_events:
                _tn = _loop_event_value(ev, "tool_name", "?")
                _rq = _loop_event_value(ev, "result_quality", "unknown")
                _rc = _loop_event_value(ev, "result_count", 0)
                _sr = _loop_event_value(ev, "selected_reason", "") or ""
                _q = _loop_event_value(ev, "query", "") or ""
                _ci = _loop_event_value(ev, "context_injected", False)
                _q_part = f'  query="{_q}"' if _q else ""
                _ci_part = ", injected" if _ci else ""
                lines.append(f"    {_tn}  {_rq}  {_rc} results  {_sr}{_q_part}{_ci_part}")
        has_content = True

    _cqs_w = getattr(getattr(native_meta, "context_quality_score", None), "warnings", []) or []
    _cp_w = getattr(getattr(native_meta, "context_packet", None), "warnings", []) or []
    _fc_w = getattr(getattr(native_meta, "file_context", None), "warnings", []) or []
    _ctx_warning_count = len(_cqs_w) + len(_cp_w) + len(_fc_w)
    if _ctx_warning_count > 0:
        _w_word = "warning" if _ctx_warning_count == 1 else "warnings"
        lines.append(f"  context warnings: {_ctx_warning_count} {_w_word}")
        has_content = True

    cus = getattr(native_meta, "context_usage_summary", None)
    if cus is not None:
        _cus_parts = [
            f"{getattr(cus, 'total_chars', 0)} chars",
            f"{getattr(cus, 'selected_files_count', 0)} files",
            f"{getattr(cus, 'compact_paths_count', 0)} paths",
            f"{getattr(cus, 'evidence_items_count', 0)} evidence",
            f"{getattr(cus, 'snippet_count', 0)} snippets",
        ]
        if getattr(cus, "any_truncated", False):
            _cus_parts.append("truncated")
        _cus_warn = getattr(cus, "failure_warning_count", 0)
        if _cus_warn:
            _cus_parts.append(f"{_cus_warn} warnings")
        lines.append(f"  context summary: {', '.join(_cus_parts)}")
        has_content = True

    fm = getattr(native_meta, "failure_memory", None)
    if fm is not None and getattr(fm, "has_lessons", False):
        _lessons = getattr(fm, "lessons", []) or []
        _l_count = len(_lessons)
        _l_word = "lesson" if _l_count == 1 else "lessons"
        if detail == "full":
            lines.append(f"  failure memory: {_l_count} {_l_word}")
            for _lesson in _lessons:
                _lt = getattr(_lesson, "lesson_type", None) or (_lesson.get("lesson_type", "") if isinstance(_lesson, dict) else "")
                _lr = getattr(_lesson, "reason", None) or (_lesson.get("reason", "") if isinstance(_lesson, dict) else "")
                lines.append(f"    {_lt}: {_lr}")
        else:
            _labels = [
                getattr(_l, "lesson_type", None) or (_l.get("lesson_type", "") if isinstance(_l, dict) else "")
                for _l in _lessons
            ]
            lines.append(f"  failure memory: {', '.join(_labels)}")
        has_content = True

    if detail == "full":
        osn_loop_full = getattr(native_meta, "osn_loop", None)
        if osn_loop_full is not None and getattr(osn_loop_full, "enabled", False):
            osn_steps = getattr(osn_loop_full, "steps", []) or []
            _MAX_OSN_RENDER_STEPS = 8
            if osn_steps:
                lines.append("  osn loop steps:")
                for s in osn_steps[:_MAX_OSN_RENDER_STEPS]:
                    s_idx = _loop_event_value(s, "step_index", "?")
                    s_tool = _loop_event_value(s, "tool_name", "?")
                    s_label = _loop_event_value(s, "target_label", "")
                    s_ok = _loop_event_value(s, "ok", False)
                    s_chars = _loop_event_value(s, "output_chars", 0)
                    s_skip = _loop_event_value(s, "skipped", False)
                    s_status = "skip" if s_skip else ("ok" if s_ok else "fail")
                    lines.append(f"    [{s_idx}] {s_tool}({s_label!r}) {s_status} chars={s_chars}")
                has_content = True

    if detail == "full":
        loop_trace = getattr(native_meta, "native_loop_trace", None)
        if loop_trace is None:
            loop_trace = []
        if isinstance(loop_trace, list):
            trace_events = loop_trace
        else:
            trace_events = getattr(loop_trace, "events", [])
        if trace_events:
            lines.append("  loop trace:")
            for event in trace_events:
                phase = _loop_event_value(event, "phase", "")
                status = _loop_event_value(event, "status", "completed")
                meta = _loop_event_value(event, "metadata", {})
                meta_str = ""
                if isinstance(meta, dict) and meta:
                    meta_str = " " + " ".join(f"{k}: {v}" for k, v in meta.items())
                elif hasattr(meta, "__dict__"):
                    d = {k: v for k, v in vars(meta).items() if not k.startswith("_")}
                    if d:
                        meta_str = " " + " ".join(f"{k}: {v}" for k, v in d.items())
                lines.append(f"    {phase} [{status}]{meta_str}")
            has_content = True

    if detail in ("more", "full"):
        prov = getattr(native_meta, "context_provenance", None)
        if prov is not None:
            _p_used = getattr(prov, "used_sources", 0)
            _p_inj = getattr(prov, "injected_sources", 0)
            _p_items = getattr(prov, "total_items", 0)
            lines.append(f"  context provenance: {_p_used} sources, {_p_inj} injected, {_p_items} items")
            has_content = True
            _p_gaps = getattr(prov, "has_gaps", False)
            _p_warns = getattr(prov, "warnings", []) or []
            if _p_gaps:
                lines.append(f"  context provenance gaps: {len(_p_warns)} warnings")
            if detail == "full":
                _p_sources = getattr(prov, "sources", []) or []
                for _src in _p_sources:
                    if isinstance(_src, dict):
                        _sn = _src.get("name", "")
                        _su = "yes" if _src.get("used", False) else "no"
                        _si = "yes" if _src.get("injected", False) else "no"
                        _sc = _src.get("item_count", 0)
                    else:
                        _sn = getattr(_src, "name", "")
                        _su = "yes" if getattr(_src, "used", False) else "no"
                        _si = "yes" if getattr(_src, "injected", False) else "no"
                        _sc = getattr(_src, "item_count", 0)
                    lines.append(f"    provenance source: {_sn} used={_su} injected={_si} items={_sc}")

    if detail in ("more", "full"):
        rts = getattr(native_meta, "run_trust_score", None)
        if rts is not None:
            _rts_score = getattr(rts, "score", 0)
            _rts_level = getattr(rts, "level", "unknown")
            _rts_warnings = getattr(rts, "warnings", []) or []
            _rts_blockers = getattr(rts, "blockers", []) or []
            lines.append(f"  run trust: {_rts_score}/100 {_rts_level}")
            if _rts_warnings:
                _wc = len(_rts_warnings)
                lines.append(f"  run trust warnings: {_wc} {'warning' if _wc == 1 else 'warnings'}")
            if _rts_blockers:
                _bc = len(_rts_blockers)
                lines.append(f"  run trust blockers: {_bc} {'blocker' if _bc == 1 else 'blockers'}")
            if detail == "full":
                _rts_factors = getattr(rts, "factors", []) or []
                lines.append("  [run trust]")
                lines.append(f"  score: {_rts_score}/100")
                lines.append(f"  level: {_rts_level}")
                if _rts_factors:
                    lines.append("  factors:")
                    for _f in _rts_factors:
                        if isinstance(_f, dict):
                            _fn = _f.get("name", "")
                            _fi = _f.get("impact", 0)
                            _fr = _f.get("reason", "")
                        else:
                            _fn = getattr(_f, "name", "")
                            _fi = getattr(_f, "impact", 0)
                            _fr = getattr(_f, "reason", "")
                        _sign = "+" if _fi >= 0 else ""
                        lines.append(f"    - {_fn} ({_sign}{_fi}): {_fr}")
                lines.append(f"  warnings: {len(_rts_warnings)}")
                lines.append(f"  blockers: {len(_rts_blockers)}")
            has_content = True

    if detail in ("more", "full"):
        msd = getattr(native_meta, "model_selection_decision", None)
        if msd is not None:
            _msd_strategy = getattr(msd, "strategy", "unknown")
            _msd_confidence = getattr(msd, "confidence", "unknown")
            _msd_roles = getattr(msd, "roles", []) or []
            _planner_tier = ""
            _executor_tier = ""
            _validator_tier = ""
            for _r in _msd_roles:
                _rname = _r.get("role", "") if isinstance(_r, dict) else getattr(_r, "role", "")
                _rtier = _r.get("model_tier", "") if isinstance(_r, dict) else getattr(_r, "model_tier", "")
                if _rname == "planner":
                    _planner_tier = _rtier
                if _rname == "executor":
                    _executor_tier = _rtier
                if _rname == "validator":
                    _validator_tier = _rtier
            _msd_warnings_compact = getattr(msd, "warnings", []) or []
            _msd_compact = (
                f"  model selection: {_msd_strategy},"
                f" confidence={_msd_confidence},"
                f" planner={_planner_tier},"
                f" executor={_executor_tier},"
                f" validator={_validator_tier}"
            )
            if _msd_warnings_compact:
                _msd_compact += f", warnings={len(_msd_warnings_compact)}"
            lines.append(_msd_compact)
            has_content = True
            if detail == "full":
                _msd_task = getattr(msd, "task_type", "unknown")
                _msd_risk = getattr(msd, "risk_level", "unknown")
                _msd_fallback = getattr(msd, "fallback_reason", "")
                _msd_warnings = getattr(msd, "warnings", []) or []
                lines.append("  [model selection]")
                lines.append(f"  strategy: {_msd_strategy}")
                lines.append(f"  task_type: {_msd_task}")
                lines.append(f"  risk_level: {_msd_risk}")
                lines.append(f"  confidence: {_msd_confidence}")
                if _msd_fallback:
                    lines.append(f"  fallback: {_msd_fallback}")
                if _msd_roles:
                    lines.append("  roles:")
                    for _r in _msd_roles:
                        if isinstance(_r, dict):
                            _rn = _r.get("role", "")
                            _rt = _r.get("model_tier", "")
                            _rc = _r.get("cost_tier", "")
                            _rr = _r.get("reason", "")
                        else:
                            _rn = getattr(_r, "role", "")
                            _rt = getattr(_r, "model_tier", "")
                            _rc = getattr(_r, "cost_tier", "")
                            _rr = getattr(_r, "reason", "")
                        lines.append(f"    - {_rn}: {_rt} ({_rc}) — {_rr}")
                lines.append(f"  warnings: {len(_msd_warnings)}")

    mcs = getattr(native_meta, "model_candidate_scoring", None)
    if mcs is not None and detail in ("more", "full"):
        from openshard.native.context import render_native_model_candidate_scoring
        _mcs_line = render_native_model_candidate_scoring(mcs, detail="compact")
        if _mcs_line:
            lines.append(f"  {_mcs_line}")
            has_content = True
        if detail == "full":
            _mcs_selected_raw = getattr(mcs, "selected_by_role", {}) or {}
            _mcs_selected = _mcs_selected_raw if isinstance(_mcs_selected_raw, dict) else (vars(_mcs_selected_raw) if hasattr(_mcs_selected_raw, "__dict__") else {})
            _mcs_candidates = getattr(mcs, "candidates", []) or []
            _mcs_warnings = getattr(mcs, "warnings", []) or []
            lines.append("  [model candidates]")
            lines.append(f"  strategy: {getattr(mcs, 'strategy', 'cost-balanced')}")
            lines.append(f"  confidence: {getattr(mcs, 'confidence', 'medium')}")
            if _mcs_selected:
                lines.append("  selected:")
                for _role, _tier in _mcs_selected.items():
                    lines.append(f"    - {_role}: {_tier}")
            if _mcs_candidates:
                lines.append("  scores:")
                for _c in _mcs_candidates:
                    if isinstance(_c, dict):
                        _cr = _c.get("role", "")
                        _cc = _c.get("candidate", "")
                        _cs = _c.get("score", 0)
                    else:
                        _cr = getattr(_c, "role", "")
                        _cc = getattr(_c, "candidate", "")
                        _cs = getattr(_c, "score", 0)
                    lines.append(f"    - {_cr}/{_cc}: {_cs}")
            _mcs_blocked = getattr(mcs, "blocked_candidates", []) or []
            if _mcs_blocked:
                lines.append("  blocked:")
                for _b in _mcs_blocked:
                    lines.append(f"    - {_b}")
            lines.append(f"  warnings: {len(_mcs_warnings)}")

    if detail in ("more", "full"):
        mp = getattr(native_meta, "model_policy", None)
        if mp is not None:
            _mp_mode = mp.get("mode", "auto") if isinstance(mp, dict) else getattr(mp, "mode", "auto")
            _mp_allow_frontier = mp.get("allow_frontier", True) if isinstance(mp, dict) else getattr(mp, "allow_frontier", True)
            _mp_prefer_low_cost = mp.get("prefer_low_cost", False) if isinstance(mp, dict) else getattr(mp, "prefer_low_cost", False)
            _mp_require_os = mp.get("require_open_source", False) if isinstance(mp, dict) else getattr(mp, "require_open_source", False)
            _mp_require_local = mp.get("require_local", False) if isinstance(mp, dict) else getattr(mp, "require_local", False)
            _mp_warnings = (mp.get("warnings", []) if isinstance(mp, dict) else getattr(mp, "warnings", [])) or []
            _mp_flags: list[str] = []
            if _mp_require_local:
                _mp_flags.append("local only")
            elif _mp_require_os:
                _mp_flags.append("open source only")
            if _mp_prefer_low_cost:
                _mp_flags.append("prefer low cost")
            _mp_flags.append("frontier allowed" if _mp_allow_frontier else "frontier blocked")
            _mp_summary = ", ".join(_mp_flags)
            _mp_warn_suffix = f", {len(_mp_warnings)} warning(s)" if _mp_warnings else ""
            lines.append(f"  model policy: {_mp_mode}, {_mp_summary}{_mp_warn_suffix}")
            has_content = True
            if detail == "full":
                lines.append("  [model policy]")
                lines.append(f"  mode: {_mp_mode}")
                lines.append(f"  prefer_low_cost: {'yes' if _mp_prefer_low_cost else 'no'}")
                lines.append(f"  require_open_source: {'yes' if _mp_require_os else 'no'}")
                lines.append(f"  require_local: {'yes' if _mp_require_local else 'no'}")
                lines.append(f"  allow_frontier: {'yes' if _mp_allow_frontier else 'no'}")
                lines.append(f"  warnings: {len(_mp_warnings)}")

    if detail in ("more", "full"):
        mpr = getattr(native_meta, "model_policy_receipt", None)
        if mpr is not None:
            _mpr_active = mpr.get("active", False) if isinstance(mpr, dict) else getattr(mpr, "active", False)
            _mpr_blocked = mpr.get("blocked_count", 0) if isinstance(mpr, dict) else getattr(mpr, "blocked_count", 0)
            _mpr_changed = (mpr.get("changed_roles", []) if isinstance(mpr, dict) else getattr(mpr, "changed_roles", [])) or []
            lines.append(
                f"  model policy receipt: active={'true' if _mpr_active else 'false'},"
                f" blocked={_mpr_blocked}, changed_roles={len(_mpr_changed)}"
            )
            has_content = True
            if detail == "full":
                _mpr_mode = mpr.get("mode", "auto") if isinstance(mpr, dict) else getattr(mpr, "mode", "auto")
                _mpr_affected = mpr.get("affected_selection", False) if isinstance(mpr, dict) else getattr(mpr, "affected_selection", False)
                _mpr_warnings = mpr.get("warnings_count", 0) if isinstance(mpr, dict) else getattr(mpr, "warnings_count", 0)
                _mpr_summary = mpr.get("summary", "") if isinstance(mpr, dict) else getattr(mpr, "summary", "")
                lines.append("  [model policy receipt]")
                lines.append(f"  mode: {_mpr_mode}")
                lines.append(f"  affected_selection: {'yes' if _mpr_affected else 'no'}")
                lines.append(f"  blocked_count: {_mpr_blocked}")
                if _mpr_changed:
                    lines.append("  changed_roles:")
                    for _cr in _mpr_changed:
                        lines.append(f"    - {_cr}")
                else:
                    lines.append("  changed_roles: []")
                lines.append(f"  warnings_count: {_mpr_warnings}")
                lines.append(f"  summary: {_mpr_summary}")

    if detail in ("more", "full"):
        rp = getattr(native_meta, "routing_preview", None)
        if rp is not None:
            def _rp(key, default=""):
                return rp.get(key, default) if isinstance(rp, dict) else getattr(rp, key, default)

            _rp_strategy   = _rp("strategy", "")
            _rp_policy     = _rp("policy_mode", "")
            _rp_planner    = _rp("planner_tier", "")
            _rp_executor   = _rp("executor_tier", "")
            _rp_validator  = _rp("validator_tier", "")
            _rp_risk       = _rp("risk_level", "")
            _rp_confidence = _rp("confidence", "")
            _rp_trust      = _rp("trust_level", "")
            _rp_blocked    = _rp("blocked_count", 0)
            _rp_changed    = _rp("policy_affected", False)
            _rp_warnings   = _rp("warnings", [])
            _changed_str   = "yes" if _rp_changed else "no"
            lines.append(
                f"  routing preview: {_rp_strategy}"
                f" | planner={_rp_planner} executor={_rp_executor} validator={_rp_validator}"
                f" | policy={_rp_policy} changed={_changed_str}"
                f" blocked={_rp_blocked} trust={_rp_trust} confidence={_rp_confidence}"
            )
            has_content = True
            if detail == "full":
                lines.append("  [routing preview]")
                lines.append(f"  strategy:                {_rp_strategy}")
                lines.append(f"  policy_mode:             {_rp_policy}")
                lines.append(f"  planner:                 {_rp_planner}")
                lines.append(f"  executor:                {_rp_executor}")
                lines.append(f"  validator:               {_rp_validator}")
                lines.append(f"  risk:                    {_rp_risk}")
                lines.append(f"  confidence:              {_rp_confidence}")
                lines.append(f"  trust:                   {_rp_trust}")
                lines.append(f"  blocked_count:           {_rp_blocked}")
                lines.append(f"  policy_affected:         {'yes' if _rp_changed else 'no'}")
                lines.append(f"  warnings:                {len(_rp_warnings or [])}")

    if detail == "full":
        rr = getattr(native_meta, "routing_receipt", None)
        if rr is not None:
            def _rr(key, default=""):
                return rr.get(key, default) if isinstance(rr, dict) else getattr(rr, key, default)

            _rr_strategy   = _rr("strategy", "")
            _rr_policy     = _rr("policy_mode", "")
            _rr_trust      = _rr("trust_level", "")
            _rr_blocked    = _rr("blocked_count", 0)
            _rr_confidence = _rr("confidence", "")
            _rr_planner    = _rr("planner_tier", "")
            _rr_executor   = _rr("executor_tier", "")
            _rr_validator  = _rr("validator_tier", "")
            _rr_affected   = _rr("policy_affected", False)
            _rr_warnings   = _rr("warnings_count", 0)
            _rr_summary    = _rr("summary", "")

            has_content = True
            lines.append("  [routing receipt]")
            lines.append(f"  strategy:        {_rr_strategy}")
            lines.append(f"  planner:         {_rr_planner}")
            lines.append(f"  executor:        {_rr_executor}")
            lines.append(f"  validator:       {_rr_validator}")
            lines.append(f"  policy_mode:     {_rr_policy}")
            lines.append(f"  policy_affected: {'yes' if _rr_affected else 'no'}")
            lines.append(f"  blocked_count:   {_rr_blocked}")
            lines.append(f"  trust:           {_rr_trust}")
            lines.append(f"  confidence:      {_rr_confidence}")
            lines.append(f"  warnings_count:  {_rr_warnings}")
            lines.append(f"  summary:         {_rr_summary}")

    if detail in ("more", "full"):
        _tdr = getattr(native_meta, "tier_dispatch_receipt", None)
        if _tdr is not None:
            _tdr_lines = _render_tier_dispatch_block(_tdr, detail)
            lines.extend(_tdr_lines)
            if _tdr_lines:
                has_content = True

    return lines if has_content else []


def _print_native_demo_block(native_meta: Any, detail: str = "default") -> None:
    """Print the [native] demo block via click.echo."""
    if native_meta is None:
        return
    body = _render_native_demo_block(native_meta, detail=detail)
    if not body:
        return
    click.echo("\n[native]")
    for line in body:
        click.echo(line)


def _dict_to_ns(obj: Any) -> Any:
    """Recursively convert nested dicts to SimpleNamespace for attribute access."""
    if isinstance(obj, dict):
        return SimpleNamespace(**{k: _dict_to_ns(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [_dict_to_ns(item) for item in obj]
    return obj


def _native_meta_from_entry(entry: dict) -> Any | None:
    """Extract native metadata from a history entry dict, or None if not a native run."""
    is_native = (
        entry.get("workflow") == "native"
        or entry.get("executor") == "native"
    )
    if not is_native:
        return None
    return _dict_to_ns({
        "repo_context_summary": entry.get("repo_context_summary"),
        "observation": entry.get("observation"),
        "plan": entry.get("plan"),
        "write_path": entry.get("write_path", "pipeline"),
        "verification_loop": entry.get("verification_loop"),
        "verification_command_summary": entry.get("verification_command_summary"),
        "diff_review": entry.get("diff_review"),
        "final_report": entry.get("final_report"),
        "native_loop_steps": entry.get("native_loop_steps", []),
        "native_loop_trace": entry.get("native_loop_trace", []),
        "native_backend": entry.get("native_backend", None),
        "native_backend_available": entry.get("native_backend_available", True),
        "native_backend_notes": entry.get("native_backend_notes", []),
        "native_backend_proof": entry.get("native_backend_proof"),
        "read_search_findings": entry.get("read_search_findings", []),
        "patch_proposal": entry.get("patch_proposal"),
        "command_policy_preview": entry.get("command_policy_preview"),
        "context_packet": entry.get("context_packet"),
        "file_context": entry.get("file_context"),
        "context_quality_score": entry.get("context_quality_score"),
        "context_quality_advisory": entry.get("context_quality_advisory"),
        "change_budget": entry.get("change_budget"),
        "change_budget_preview": entry.get("change_budget_preview"),
        "change_budget_soft_gate": entry.get("change_budget_soft_gate"),
        "approval_request": entry.get("approval_request"),
        "approval_receipt": entry.get("approval_receipt"),
        "verification_plan": entry.get("verification_plan"),
        "clarification_request": entry.get("clarification_request"),
        "context_usage_summary": entry.get("context_usage_summary"),
        "failure_memory": entry.get("failure_memory"),
        "osn_loop": entry.get("osn_loop"),
        "deepagents_adapter": entry.get("deepagents_adapter"),
        "validation_contract": entry.get("validation_contract"),
        "context_provenance": entry.get("context_provenance"),
        "run_trust_score": entry.get("run_trust_score"),
        "model_selection_decision": entry.get("model_selection_decision"),
        "model_candidate_scoring": entry.get("model_candidate_scoring"),
        "model_policy": entry.get("model_policy"),
        "model_policy_receipt": entry.get("model_policy_receipt"),
        "routing_preview": entry.get("routing_preview"),
        "routing_receipt": entry.get("routing_receipt"),
        "tier_dispatch_receipt": entry.get("tier_dispatch_receipt"),
        "tool_search_events": entry.get("tool_search_events", []),
    })


def _render_native_inspection(entry: dict, detail: str) -> None:
    """Render stored native [native] and [native summary] blocks for openshard last."""
    native_meta = _native_meta_from_entry(entry)
    if native_meta is None:
        return
    _print_native_demo_block(native_meta, detail=detail)
    _print_native_summary(native_meta, detail)


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


def _validator_policy_field(policy: Any, field: str, default: Any = None) -> Any:
    if policy is None:
        return default
    if isinstance(policy, dict):
        return policy.get(field, default)
    return getattr(policy, field, default)


def _render_tier_dispatch_block(tdr: Any, detail: str, initial_model: str | None = None, validator_result: dict | None = None, validator_policy: Any = None, is_ask: bool = False) -> list[str]:
    """Render tier dispatch receipt as lines. tdr can be dict or SimpleNamespace."""
    if detail not in ("more", "full"):
        return []
    def _g(key: str, default: Any = "") -> Any:
        return tdr.get(key, default) if isinstance(tdr, dict) else getattr(tdr, key, default)
    if not _g("enabled", False):
        return []
    p_tier  = _g("planner_tier", "")
    p_model = _g("planner_model") or "-"
    e_tier  = _g("executor_tier", "")
    e_model = _g("executor_model") or "-"
    v_tier  = _g("validator_tier", "")
    v_model = _g("validator_model") or "-"
    applied = _g("applied", False)
    fb      = _g("fallback_used", False)
    source  = _g("tier_source", "")
    warns   = _g("warnings", []) or []

    def _lbl(m: str) -> str:
        return _model_label(m) if m and m != "-" else "-"

    _policy_skipped = validator_policy and not _validator_policy_field(validator_policy, "run")
    _skip_reason = _validator_policy_field(validator_policy, "reason", "skipped") if _policy_skipped else ""

    source_label = source.replace("_", " ")
    lines: list[str] = []

    if is_ask:
        # Direct Ask mode: single-pass, no planning stage ran.
        lines.append("  Model plan")
        if detail == "more":
            lines.append(f"    Ask: {_lbl(e_model)}")
            if validator_result:
                _vv = validator_result.get("verdict", "?")
                lines.append(f"    Validator: {_lbl(v_model)} ({_vv})")
            elif _policy_skipped:
                lines.append(f"    Validator: skipped — {_skip_reason}")
            else:
                lines.append(f"    Validator: {_lbl(v_model)} (reserved)")
        else:  # full
            lines.append(f"    Ask: {_lbl(e_model)}")
            lines.append(f"      Model: {e_model}")
            lines.append("")
            lines.append("    Validator:")
            if validator_result:
                _vv = validator_result.get("verdict", "?")
                _vsum = validator_result.get("summary", "")
                lines.append(f"      Verdict: {_vv}")
                if _vsum:
                    lines.append(f"      Summary: {_vsum}")
            elif _policy_skipped:
                lines.append(f"      Skipped: {_skip_reason}")
            else:
                lines.append("      Used: no, reserved for validation")
        return lines

    if detail == "more":
        lines.append("  Model plan")
        lines.append(f"    Planning:  {_lbl(p_model)}")
        lines.append(f"    Work:      {_lbl(e_model)}")
        if validator_result:
            _vv = validator_result.get("verdict", "?")
            lines.append(f"    Validator: {_lbl(v_model)} ({_vv})")
        elif _policy_skipped:
            lines.append(f"    Validator: {_lbl(v_model)} (skipped — {_skip_reason})")
        else:
            lines.append(f"    Validator: {_lbl(v_model)} (reserved)")
        lines.append("")
        lines.append("  Dispatch")
        lines.append(f"    Applied: {'yes' if applied else 'no'}")
        lines.append(f"    Source:  {source_label}")
        lines.append(f"    Work model: {_lbl(e_model)}")
        if initial_model:
            lines.append(f"    Initial candidate: {_model_label(initial_model)}")
        if fb:
            lines.append("    Fallback: yes")
        if warns:
            lines.append(f"    Warnings: {len(warns)}")
    else:  # full
        reason = _g("fallback_reason", "")
        lines.append("  Model plan")
        lines.append(f"    Planning: {_lbl(p_model)}")
        lines.append(f"      Tier:  {p_tier}")
        lines.append(f"      Model: {p_model}")
        lines.append("")
        lines.append(f"    Work: {_lbl(e_model)}")
        lines.append(f"      Tier:  {e_tier}")
        lines.append(f"      Model: {e_model}")
        lines.append("")
        lines.append(f"    Validator: {_lbl(v_model)}")
        lines.append(f"      Tier:  {v_tier}")
        lines.append(f"      Model: {v_model}")
        if validator_result:
            _vv = validator_result.get("verdict", "?")
            _vsum = validator_result.get("summary", "")
            lines.append(f"      Verdict: {_vv}")
            if _vsum:
                lines.append(f"      Summary: {_vsum}")
        elif _policy_skipped:
            lines.append(f"      Skipped: {_skip_reason}")
        else:
            lines.append("      Used:  no, reserved for validation")
        lines.append("")
        lines.append("  Dispatch")
        lines.append(f"    Applied: {'yes' if applied else 'no'}")
        lines.append(f"    Source:  {source_label}")
        lines.append(f"    Work model: {_lbl(e_model)}")
        if initial_model:
            lines.append(f"    Initial candidate: {_model_label(initial_model)}")
        lines.append(f"    Fallback: {'yes' if fb else 'no'}")
        lines.append(f"    Warnings: {len(warns)}")
        if reason:
            lines.append(f"    Reason:  {reason}")
        for w in warns:
            lines.append(f"      - {w}")
    return lines


def _print_tier_dispatch_block(tdr: Any, detail: str, validator_result: dict | None = None, validator_policy: Any = None, is_ask: bool = False) -> None:
    for line in _render_tier_dispatch_block(tdr, detail, validator_result=validator_result, validator_policy=validator_policy, is_ask=is_ask):
        click.echo(line)
