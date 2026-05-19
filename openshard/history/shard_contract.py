from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

_PROFILE_TO_STRATEGY: dict[str, str] = {
    "native_light": "Single",
    "native_deep": "Plan + Execute",
    "native_swarm": "Multi-stage",
}

_RISK_LABELS: dict[str, str] = {
    "critical": "Critical",
    "high": "High",
    "medium": "Medium",
    "low": "Low",
}

_STAGE_DISPLAY_LABELS: dict[str, str] = {
    "planning": "Planning",
    "implementation": "Execution",
    "analysis": "Analysis",
    "ask": "Ask",
    "verify": "Verify",
    "retry": "Retry",
}

# Full provider/slug → friendly display name.
# Keys are lowercase. Values use full model family names (not abbreviations).
_MODEL_FRIENDLY_NAMES: dict[str, str] = {
    "deepseek/deepseek-v4-pro": "DeepSeek V4 Pro",
    "deepseek/deepseek-v4-flash": "DeepSeek V4 Flash",
    "anthropic/claude-sonnet-4.6": "Claude Sonnet 4.6",
    "anthropic/claude-sonnet-4-6": "Claude Sonnet 4.6",
    "anthropic/claude-opus-4.7": "Claude Opus 4.7",
    "anthropic/claude-opus-4-7": "Claude Opus 4.7",
    "anthropic/claude-haiku-4.5": "Claude Haiku 4.5",
    "anthropic/claude-haiku-4-5": "Claude Haiku 4.5",
    "openai/gpt-5.5": "GPT-5.5",
    "openai/gpt-5.5-thinking": "GPT-5.5 Thinking",
    "z-ai/glm-5.1": "GLM-5.1",
    "qwen/qwen-3.6-plus": "Qwen 3.6 Plus",
    "google/gemini-3.1-pro": "Gemini 3.1 Pro",
}

# Words rendered in ALL CAPS in model names (abbreviations and well-known initialisms).
_ABBREV_WORDS: frozenset[str] = frozenset({"gpt", "llm", "ai", "api", "url", "id", "ui", "ml", "glm"})

_SEP: str = "━" * 46
_INDENT: str = "  "
_COL: int = 12


def _display_model_name(slug: str) -> str:
    """Convert a provider/model slug to a user-friendly display name.

    Checks an explicit lookup table first; falls back to a best-effort formatter.
    Keeps raw slugs in stored history — only used in rendered receipt output.
    """
    if not slug:
        return slug
    key = slug.lower().strip()
    if key in _MODEL_FRIENDLY_NAMES:
        return _MODEL_FRIENDLY_NAMES[key]
    # Try without provider prefix
    name_key = key.split("/", 1)[-1]
    if name_key in _MODEL_FRIENDLY_NAMES:
        return _MODEL_FRIENDLY_NAMES[name_key]
    return _format_model_slug_shard(slug.split("/", 1)[-1])


def _format_model_slug_shard(name: str) -> str:
    """Best-effort formatter for unknown model slugs (e.g. 'gemini-2.0-flash' → 'Gemini 2.0 Flash')."""
    parts = [p for p in name.split("-") if p]
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


@dataclass
class ShardReceipt:
    shard_id: str
    created_at: str
    task_short: str
    task_full: str
    agent: str
    strategy: str
    model_display: str
    risk: str
    sandbox: str
    files_changed: int
    checks_display: str
    approval: str
    cost_display: str
    result: str
    status: str
    duration_seconds: Optional[float]
    repo: Optional[str] = None
    branch: Optional[str] = None
    git_state: Optional[str] = None
    files_touched: list[str] = field(default_factory=list)
    files_detail: list[dict] = field(default_factory=list)
    allowed_paths: list[str] = field(default_factory=list)
    blocked_paths: list[str] = field(default_factory=list)
    blocked_commands: list[str] = field(default_factory=list)
    check_results: list[str] = field(default_factory=list)
    diff_added: Optional[int] = None
    diff_removed: Optional[int] = None
    cost_raw: Optional[float] = None
    # Each tuple is (friendly_stage_label, friendly_model_name).
    model_stages: list[tuple[str, str]] = field(default_factory=list)


def _make_shard_id(timestamp: str, index: Optional[int]) -> str:
    try:
        dt = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
        date_str = dt.strftime("%Y%m%d")
    except (ValueError, AttributeError, TypeError):
        date_str = datetime.now(timezone.utc).strftime("%Y%m%d")
    n = (index + 1) if index is not None else 1
    return f"shard-{date_str}-{n:04d}"


def _trunc(s: str, n: int) -> str:
    if not s or len(s) <= n:
        return s or ""
    return s[: n - 1] + "…"


def build_shard_receipt(entry: dict, index: Optional[int] = None) -> ShardReceipt:
    """Convert a raw run-history entry dict into a ShardReceipt. Never raises."""
    timestamp = entry.get("timestamp") or ""
    task = entry.get("task") or ""

    is_native = entry.get("workflow") == "native" or entry.get("executor") == "native"
    agent = "OpenShard Native" if is_native else "OpenShard"

    profile = entry.get("execution_profile")
    strategy = _PROFILE_TO_STRATEGY.get(profile) if profile else None
    strategy = strategy or "Not recorded"

    routing_model = entry.get("routing_selected_model")
    exec_model = entry.get("execution_model")
    if routing_model:
        model_display = f"Auto → {_display_model_name(routing_model)}"
    elif exec_model:
        model_display = _display_model_name(exec_model)
    else:
        model_display = "Not recorded"

    form_factor = entry.get("form_factor") or {}
    plan = entry.get("plan") or {}
    risk_raw = form_factor.get("risk_level") or plan.get("risk")
    risk = _RISK_LABELS.get(str(risk_raw).lower(), str(risk_raw).capitalize()) if risk_raw else "Not recorded"

    write_path = entry.get("write_path")
    ff_read_only = form_factor.get("read_only")
    if write_path == "sandbox":
        sandbox = "On"
    elif write_path == "pipeline":
        sandbox = "Off"
    elif ff_read_only:
        sandbox = "Not required"
    else:
        sandbox = "Not recorded"

    fc = entry.get("files_created") or 0
    fu = entry.get("files_updated") or 0
    fd = entry.get("files_deleted") or 0
    files_changed = fc + fu + fd
    if files_changed == 0:
        fr = entry.get("final_report") or {}
        diff = entry.get("diff_review") or {}
        diff_files = fr.get("diff_files") or diff.get("changed_files") or []
        if diff_files:
            files_changed = len(diff_files)

    v_attempted = entry.get("verification_attempted")
    v_passed = entry.get("verification_passed")
    if v_attempted is None:
        fr = entry.get("final_report") or {}
        v_attempted = fr.get("verification_attempted")
        if v_passed is None:
            v_passed = fr.get("verification_passed")

    if v_attempted is None:
        checks_display = "Not recorded"
        status = "Not recorded"
    elif not v_attempted:
        checks_display = "Not run"
        status = "No checks run"
    elif v_passed is True:
        checks_display = "1/1 passed"
        status = "Passed"
    elif v_passed is False:
        checks_display = "0/1 passed"
        status = "Failed"
    else:
        checks_display = "Not run"
        status = "No checks run"

    approval_receipt_raw = entry.get("approval_receipt") or {}
    if approval_receipt_raw:
        if approval_receipt_raw.get("granted"):
            approval = "Required → Granted"
        else:
            approval = "Required → Pending"
    elif ff_read_only or write_path == "pipeline":
        approval = "Not required"
    else:
        approval = "Not recorded"

    cost_raw = entry.get("estimated_cost")
    cost_display = f"${cost_raw:.4f}" if cost_raw is not None else "Not recorded"

    summary = entry.get("summary") or ""
    result = _trunc(summary.split("\n")[0] if summary else "", 120) or "Not recorded"

    files_detail_raw = entry.get("files_detail") or []
    files_touched = [f["path"] for f in files_detail_raw if isinstance(f, dict) and "path" in f]

    diff_review = entry.get("diff_review") or {}
    diff_added = diff_review.get("added_lines")
    diff_removed = diff_review.get("removed_lines")
    if diff_added is None:
        fr = entry.get("final_report") or {}
        diff_added = fr.get("added_lines")
    if diff_removed is None:
        fr = entry.get("final_report") or {}
        diff_removed = fr.get("removed_lines")

    stage_runs = entry.get("stage_runs") or []
    _is_ro = entry.get("routing_rationale") == "read-only analysis" or bool(form_factor.get("read_only"))
    model_stages: list[tuple[str, str]] = [
        (
            _STAGE_DISPLAY_LABELS.get(
                "analysis" if (_is_ro and s["stage_type"] == "implementation") else s["stage_type"],
                s["stage_type"].capitalize(),
            ),
            _display_model_name(s["model"]),
        )
        for s in stage_runs
        if isinstance(s, dict) and "stage_type" in s and "model" in s
    ]

    command_policy = entry.get("command_policy") or {}
    allowed_paths = list(command_policy.get("allowed_paths") or [])
    blocked_paths = list(command_policy.get("blocked_paths") or [])
    blocked_commands = list(command_policy.get("blocked_commands") or [])

    return ShardReceipt(
        shard_id=_make_shard_id(timestamp, index),
        created_at=timestamp,
        task_short=_trunc(task, 70),
        task_full=task,
        agent=agent,
        strategy=strategy,
        model_display=model_display,
        risk=risk,
        sandbox=sandbox,
        files_changed=files_changed,
        checks_display=checks_display,
        approval=approval,
        cost_display=cost_display,
        result=result,
        status=status,
        duration_seconds=entry.get("duration_seconds"),
        files_detail=files_detail_raw,
        files_touched=files_touched,
        diff_added=diff_added,
        diff_removed=diff_removed,
        cost_raw=cost_raw,
        model_stages=model_stages,
        allowed_paths=allowed_paths,
        blocked_paths=blocked_paths,
        blocked_commands=blocked_commands,
    )


def _row(label: str, value: str, width: int = _COL) -> str:
    return f"{_INDENT}{label:<{width}}{value}"


def _models_label_and_value(receipt: ShardReceipt) -> tuple[str, str]:
    """Return (label, value) for the Model/Models row in the compact receipt."""
    if receipt.model_stages:
        unique = list(dict.fromkeys(m for _, m in receipt.model_stages))
        if len(unique) == 1:
            return "Model", unique[0]
        if len(unique) == 2:
            return "Models", f"{unique[0]} → {unique[1]}"
        return "Models", ", ".join(unique)
    return "Model", receipt.model_display


def render_compact_shard_receipt(receipt: ShardReceipt) -> str:
    """Render a bordered, column-aligned RECEIPT block. Pure, no I/O."""
    file_str = f"{receipt.files_changed} file{'s' if receipt.files_changed != 1 else ''}"
    model_label, model_value = _models_label_and_value(receipt)

    lines = [
        _SEP,
        f"{_INDENT}RECEIPT — {receipt.shard_id}",
        _SEP,
        _row("Task", receipt.task_short),
        _row("Agent", receipt.agent),
        _row("Strategy", receipt.strategy),
        _row(model_label, model_value),
        _row("Risk", receipt.risk),
        _row("Sandbox", receipt.sandbox),
        _row("Changed", file_str),
        _row("Checks", receipt.checks_display),
        _row("Approval", receipt.approval),
        _row("Cost", receipt.cost_display),
        _row("Result", receipt.result),
        _SEP,
    ]
    return "\n".join(lines)


def _fmt_timestamp(ts: str) -> str:
    if not ts:
        return "-"
    try:
        dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        return dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    except (ValueError, AttributeError):
        return ts


def render_full_shard_receipt(receipt: ShardReceipt) -> str:
    """Render a full structured SHARD block with consistent separator style. Pure, no I/O."""
    lines: list[str] = []

    lines += [_SEP, f"{_INDENT}SHARD — {receipt.shard_id}", _SEP, ""]

    lines += [f"{_INDENT}TASK", f"{_INDENT}{receipt.task_full}", ""]

    lines.append(f"{_INDENT}EXECUTION")
    lines.append(_row("Agent", receipt.agent))
    lines.append(_row("Strategy", receipt.strategy))

    if receipt.model_stages:
        unique = list(dict.fromkeys(m for _, m in receipt.model_stages))
        if len(unique) == 1:
            lines.append(_row("Model", unique[0]))
        else:
            lines.append(f"{_INDENT}Models")
            _stage_col = max(len(s) for s, _ in receipt.model_stages) + 2
            for stage_lbl, model_name in receipt.model_stages:
                lines.append(f"{_INDENT}  {stage_lbl:<{_stage_col}}{model_name}")
    else:
        lines.append(_row("Model", receipt.model_display))

    dur = f"{receipt.duration_seconds:.1f}s" if receipt.duration_seconds is not None else "-"
    lines.append(_row("Duration", dur))
    lines.append(_row("Status", receipt.status))
    lines.append("")

    lines.append(f"{_INDENT}CONTEXT")
    lines.append(_row("Repo", receipt.repo or "-"))
    lines.append(_row("Branch", receipt.branch or "-"))
    lines.append(_row("Git state", receipt.git_state or "-"))
    if receipt.files_touched:
        touched_str = ", ".join(receipt.files_touched[:5])
        if len(receipt.files_touched) > 5:
            touched_str += f" (+{len(receipt.files_touched) - 5} more)"
        lines.append(_row("Touched", touched_str))
    else:
        lines.append(_row("Touched", "-"))
    lines.append("")

    lines.append(f"{_INDENT}POLICY")
    lines.append(_row("Risk", receipt.risk))
    lines.append(_row("Sandbox", receipt.sandbox))
    if receipt.allowed_paths:
        lines.append(_row("Allowed", ", ".join(receipt.allowed_paths[:3])))
    if receipt.blocked_paths:
        lines.append(_row("Blocked", ", ".join(receipt.blocked_paths[:3])))
    if receipt.blocked_commands:
        lines.append(_row("Commands", f"{len(receipt.blocked_commands)} blocked"))
    lines.append(_row("Approval", receipt.approval))
    lines.append("")

    lines.append(f"{_INDENT}CHECKS")
    if receipt.check_results:
        for cr in receipt.check_results:
            lines.append(f"{_INDENT}  {cr}")
    else:
        lines.append(f"{_INDENT}{receipt.checks_display}")
    lines.append("")

    lines.append(f"{_INDENT}CHANGES")
    file_str = f"{receipt.files_changed} file{'s' if receipt.files_changed != 1 else ''}"
    if receipt.diff_added is not None and receipt.diff_removed is not None:
        file_str += f" changed (+{receipt.diff_added} / -{receipt.diff_removed})"
    elif receipt.files_changed > 0:
        file_str += " changed"
    lines.append(f"{_INDENT}{file_str}")
    for f in receipt.files_detail[:10]:
        if isinstance(f, dict) and "path" in f:
            lines.append(f"{_INDENT}  {f['path']}")
    if len(receipt.files_detail) > 10:
        lines.append(f"{_INDENT}  (+{len(receipt.files_detail) - 10} more)")
    lines.append("")

    lines.append(f"{_INDENT}COST")
    lines.append(f"{_INDENT}{receipt.cost_display}")
    lines.append("")

    lines += [_SEP, f"{_INDENT}RECEIPT"]
    lines.append(_row("Shard ID", receipt.shard_id))
    lines.append(_row("Created", _fmt_timestamp(receipt.created_at)))
    lines.append(_row("Result", receipt.result))
    lines.append(_SEP)

    return "\n".join(lines)
