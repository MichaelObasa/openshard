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
    }
    return "Executing - " + desc.get(rationale, f"{label} running task")


def _build_model_line(
    routing_decision: RoutingDecision | None,
    stage_runs: list[StageRun],
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

    cp = getattr(native_meta, "context_packet", None)
    if cp is not None:
        _cp_sources = len(getattr(cp, "sources", []) or [])
        _cp_paths = len(getattr(cp, "compact_paths", []) or [])
        if _cp_sources > 0:
            lines.append(f"  context packet: {_cp_sources} sources, {_cp_paths} paths")
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
            lines.append(f"  file context: {_fc_files} files, {_fc_chars} chars")
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
 osn-context-packet
        "context_packet": entry.get("context_packet"),

        "file_context": entry.get("file_context"),
main
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
