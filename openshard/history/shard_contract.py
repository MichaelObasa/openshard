from __future__ import annotations

import re
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path, PureWindowsPath
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
    "claude-sonnet-4-6": "Claude Sonnet 4.6",
    "claude-sonnet-4.6": "Claude Sonnet 4.6",
    "claude-opus-4-7": "Claude Opus 4.7",
    "claude-opus-4.7": "Claude Opus 4.7",
    "claude-haiku-4-5": "Claude Haiku 4.5",
    "claude-haiku-4.5": "Claude Haiku 4.5",
    "openai/gpt-5.5": "GPT-5.5",
    "openai/gpt-5.5-thinking": "GPT-5.5 Thinking",
    "z-ai/glm-5.1": "GLM-5.1",
    "qwen/qwen-3.6-plus": "Qwen 3.6 Plus",
    "google/gemini-3.1-pro": "Gemini 3.1 Pro",
}

# Words rendered in ALL CAPS in model names (abbreviations and well-known initialisms).
_ABBREV_WORDS: frozenset[str] = frozenset({"gpt", "llm", "ai", "api", "url", "id", "ui", "ml", "glm"})

def _stdout_supports_unicode() -> bool:
    try:
        enc = getattr(sys.stdout, "encoding", None) or "utf-8"
        "━—✖⚠✓".encode(enc)
        return True
    except (UnicodeEncodeError, LookupError):
        return False


_UNICODE_OK: bool = _stdout_supports_unicode()
_SEP: str = ("━" if _UNICODE_OK else "-") * 40
_EM: str = "—" if _UNICODE_OK else "-"
_INDENT: str = "  "
_COL: int = 12

_SEVERITY_ORDER: list[str] = ["Critical", "High", "Medium", "Low", "Note"]

_FINDING_ICONS: dict[str, str] = {
    "Critical": "✖" if _UNICODE_OK else "X",
    "High": "⚠" if _UNICODE_OK else "!",
    "Medium": "~",
    "Low": "✓" if _UNICODE_OK else "+",
    "Note": "-",
}


@dataclass
class ShardFinding:
    severity: str
    message: str
    path: str | None = None
    line: int | None = None


@dataclass
class FileEvidence:
    path: str
    roles: list[str]  # ordered subset of: inspected, finding_source, changed


_ROLE_ORDER = ["inspected", "finding_source", "changed"]

_ROLE_LABELS: dict[str, str] = {
    "inspected": "inspected/read context",
    "finding_source": "finding source",
    "changed": "changed",
}


def _build_file_evidence(
    inspected: list[str],
    referenced: list[str],
    touched: list[str],
) -> list[FileEvidence]:
    acc: dict[str, set[str]] = {}
    for p in inspected:
        acc.setdefault(p, set()).add("inspected")
    for p in referenced:
        acc.setdefault(p, set()).add("finding_source")
    for p in touched:
        acc.setdefault(p, set()).add("changed")
    result = [
        FileEvidence(path=p, roles=[r for r in _ROLE_ORDER if r in roles])
        for p, roles in acc.items()
    ]
    return sorted(result, key=lambda e: e.path)


def _safe_str_list(val: object) -> list[str]:
    if not isinstance(val, list):
        return []
    return [item for item in val if isinstance(item, str)]


def _coerce_finding_list(val: object, default_severity: str = "Note") -> list[ShardFinding]:
    if not isinstance(val, list):
        return []
    out: list[ShardFinding] = []
    for item in val:
        if isinstance(item, dict) and "message" in item:
            sev = item.get("severity") or default_severity
            if sev not in _SEVERITY_ORDER:
                sev = "Note"
            line_val = item.get("line")
            out.append(ShardFinding(
                severity=sev,
                message=str(item["message"]),
                path=item.get("path") or None,
                line=int(line_val) if line_val is not None else None,
            ))
        elif isinstance(item, str):
            out.append(ShardFinding(severity=default_severity, message=item))
    return out


_METADATA_NOISE_PHRASES: tuple[str, ...] = (
    "has no tags block",
    "has no labels block",
    "missing required tags",
    "missing required labels",
    "add owner and environment",
)

_DEFAULT_FINDING_CAPS: dict[str, int] = {"Critical": 3, "High": 3, "Medium": 2, "Low": 1}


def _is_metadata_noise(f: ShardFinding) -> bool:
    msg = f.message.lower()
    return any(p in msg for p in _METADATA_NOISE_PHRASES)


def group_review_findings(
    findings: list[ShardFinding],
    *,
    caps: "dict[str, int] | None" = None,
) -> "tuple[list[ShardFinding], ShardFinding | None, int, int]":
    """Group and rank findings for compact display.

    Returns (visible_substantive, grouped_metadata_or_None, hidden_substantive_count, raw_total).

    visible_substantive: deduplicated non-metadata findings sorted Critical→Low,
        each severity capped by *caps* (defaults to Critical=3, High=3, Medium=2, Low=1).
    grouped_metadata_or_None: a single synthetic ShardFinding (severity=Medium) that
        summarises all metadata-noise findings, or None if none found.
    hidden_substantive_count: substantive findings cut by the cap.
    raw_total: len(findings) before any grouping.
    """
    effective_caps = dict(_DEFAULT_FINDING_CAPS)
    if caps:
        effective_caps.update(caps)

    raw_total = len(findings)

    # Deduplicate by (severity, message)
    seen: set[tuple[str, str]] = set()
    deduped: list[ShardFinding] = []
    for f in findings:
        key = (f.severity, f.message)
        if key not in seen:
            seen.add(key)
            deduped.append(f)

    substantive = [f for f in deduped if not _is_metadata_noise(f)]
    metadata    = [f for f in deduped if _is_metadata_noise(f)]

    # Sort substantive by severity order
    substantive.sort(
        key=lambda f: _SEVERITY_ORDER.index(f.severity) if f.severity in _SEVERITY_ORDER else len(_SEVERITY_ORDER),
    )

    # Apply per-severity caps
    visible: list[ShardFinding] = []
    hidden_count = 0
    by_sev: dict[str, list[ShardFinding]] = {}
    for f in substantive:
        by_sev.setdefault(f.severity, []).append(f)
    for sev in _SEVERITY_ORDER:
        group = by_sev.get(sev, [])
        cap = effective_caps.get(sev, 1)
        visible.extend(group[:cap])
        hidden_count += max(0, len(group) - cap)

    # Build grouped metadata finding
    meta_group: "ShardFinding | None" = None
    if metadata:
        # Detect term (labels vs tags) from messages
        uses_labels = any("label" in f.message.lower() for f in metadata)
        has_gcp     = any("google_" in f.message for f in metadata)
        term   = "labels" if uses_labels else "tags"
        prefix = "GCP " if has_gcp else ""

        # Extract up to 3 resource names from messages (pattern: "Resource TYPE.NAME ...")
        examples: list[str] = []
        seen_ex: set[str] = set()
        for f in metadata:
            parts = f.message.split()
            if len(parts) >= 2 and parts[0].lower() == "resource":
                name = parts[1]
                if name not in seen_ex:
                    seen_ex.add(name)
                    examples.append(name)
                    if len(examples) >= 3:
                        break
        ex_str = ", ".join(examples)
        n = len(metadata)
        msg = (
            f"{n} {prefix}resources are missing ownership/environment {term}, "
            "making cost tracking and incident response harder."
        )
        if ex_str:
            msg += f"\n  Examples: {ex_str}"
        meta_group = ShardFinding(severity="Medium", message=msg)

    return visible, meta_group, hidden_count, raw_total


def _extract_findings(entry: dict) -> list[ShardFinding]:
    findings: list[ShardFinding] = []

    findings.extend(_coerce_finding_list(entry.get("findings")))

    for note in _safe_str_list(entry.get("agent_notes")):
        findings.append(ShardFinding(severity="Note", message=note))

    fr = entry.get("final_report") or {}
    findings.extend(_coerce_finding_list(fr.get("findings")))

    for w in _safe_str_list(fr.get("warnings")):
        findings.append(ShardFinding(severity="Note", message=w))

    dr = entry.get("diff_review") or {}
    for w in _safe_str_list(dr.get("warnings")):
        findings.append(ShardFinding(severity="Note", message=w))

    pl = entry.get("plan") or {}
    for w in _safe_str_list(pl.get("warnings")):
        findings.append(ShardFinding(severity="Note", message=w))

    obs = entry.get("observation") or {}
    for w in _safe_str_list(obs.get("warnings")):
        findings.append(ShardFinding(severity="Note", message=w))

    findings.sort(key=lambda f: _SEVERITY_ORDER.index(f.severity) if f.severity in _SEVERITY_ORDER else 99)
    return findings


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
    # Fall back to centralized registry for models not in the local table.
    from openshard.models.registry import display_name_for as _reg_display
    reg_name = _reg_display(slug)
    if reg_name != slug:
        return reg_name
    return _format_model_slug_shard(slug.split("/", 1)[-1])


def display_model_name(slug: str) -> str:
    """Public wrapper — convert a provider/model slug to a user-friendly display name."""
    return _display_model_name(slug)


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
    context_quality: Optional[str] = None
    files_read_count: Optional[int] = None
    inspected_files: list[str] = field(default_factory=list)
    files_referenced: list[str] = field(default_factory=list)
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
    findings: list[ShardFinding] = field(default_factory=list)
    agent_notes: list[str] = field(default_factory=list)
    run_timeline: list = field(default_factory=list)
    developer_feedback: Optional[dict] = None
    approval_required: bool = False
    approval_granted: Optional[bool] = None
    approval_reason: str = ""
    file_evidence: list[FileEvidence] = field(default_factory=list)
    model_advisory: list[dict] = field(default_factory=list)
    feedback_routing_advisory: dict | None = None


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
    return s[: n - 1] + ("…" if _UNICODE_OK else ".")


_MAX_RESULT: int = 60

# Words that signal a dangling clause when a sentence is clipped at them.
_RE_TRAILING_CONNECTIVE = re.compile(
    r"\s+(and|or|with|for|but|as|of|to|in|a|an|the|covering|including|"
    r"such|which|that|when|where|while|by|on|at|from|its|their|this|these|"
    r"its|both|also|well|via|across|using|within)\s*$",
    re.IGNORECASE,
)


def _result_display(summary: str) -> str:
    """Return a short, complete result line from a full run summary.

    Prefers the first complete sentence when one is found within _MAX_RESULT.
    Falls back to a clean word-boundary clip. Never appends an ellipsis or
    leaves a dangling clause.
    """
    if not summary:
        return "Not recorded"
    line = summary.split("\n")[0].strip()
    if not line:
        return "Not recorded"
    # Always try first complete sentence before considering full-line length.
    for sep in (". ", "; ", "! ", "? "):
        idx = line.find(sep, 0, _MAX_RESULT)
        if idx != -1:
            candidate = line[:idx + 1].strip()
            if len(candidate) >= 4:
                return candidate
    # No internal sentence boundary — use full line when it fits.
    if len(line) <= _MAX_RESULT:
        return line
    # Long line: clip at a word boundary, strip trailing connectives, add ellipsis.
    clipped = line[:_MAX_RESULT]
    sp = clipped.rfind(" ")
    if sp > _MAX_RESULT // 3:
        clipped = clipped[:sp]
    clipped = clipped.rstrip(" ,;:")
    clipped = _RE_TRAILING_CONNECTIVE.sub("", clipped).rstrip(" ,;:")
    return clipped or line[:_MAX_RESULT]


def _workspace_folder_name(raw: object) -> str | None:
    if not raw:
        return None
    value = str(raw).rstrip("\\/")
    if not value:
        return None
    if "\\" in value:
        return PureWindowsPath(value).name or None
    return Path(value).name or None


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
    _sr_quick = entry.get("stage_runs") or []
    _first_stage_model = next(
        (_display_model_name(s["model"]) for s in _sr_quick if isinstance(s, dict) and s.get("model")),
        None,
    )
    if routing_model:
        model_display = f"Auto → {_display_model_name(routing_model)}"
    elif exec_model:
        model_display = _display_model_name(exec_model)
    elif _first_stage_model:
        model_display = _first_stage_model
    else:
        model_display = "Not recorded"

    form_factor = entry.get("form_factor") or {}
    plan = entry.get("plan") or {}
    risk_raw = form_factor.get("risk_level") or plan.get("risk")
    risk = _RISK_LABELS.get(str(risk_raw).lower(), str(risk_raw).capitalize()) if risk_raw else "Not recorded"
    # Mirror the live-receipt review task risk floor: review runs are always at least High
    if entry.get("is_review_task") and risk in ("Not recorded", "Low"):
        risk = "High"

    write_path = entry.get("write_path")
    ff_read_only = form_factor.get("read_only")
    if write_path == "sandbox":
        sandbox = "On"
    elif write_path == "pipeline":
        sandbox = "Off"
    elif ff_read_only or entry.get("is_review_task"):
        sandbox = "Off"
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

    check_results: list[str] = []
    _review_checks_raw = entry.get("review_checks")
    if _review_checks_raw and isinstance(_review_checks_raw, list):
        checks_display, check_results = _format_review_checks(_review_checks_raw)
        status = f"Checks: {checks_display}"

    approval_receipt_raw = entry.get("approval_receipt") or {}
    if approval_receipt_raw:
        if approval_receipt_raw.get("granted"):
            approval = "Required → Granted"
        else:
            approval = "Required → Denied"
    elif ff_read_only or write_path == "pipeline":
        approval = "Not required"
    else:
        approval = "Not recorded"
    _approval_required = bool(approval_receipt_raw)
    _approval_granted: Optional[bool] = (
        approval_receipt_raw.get("granted") if approval_receipt_raw else None
    )
    _approval_reason: str = approval_receipt_raw.get("reason", "") if approval_receipt_raw else ""

    cost_raw = entry.get("estimated_cost")
    if cost_raw is None:
        _sr_costs = [
            s["cost"] for s in (entry.get("stage_runs") or [])
            if isinstance(s, dict) and s.get("cost") is not None
        ]
        if _sr_costs:
            cost_raw = sum(_sr_costs)
    cost_display = f"${cost_raw:.4f}" if cost_raw is not None else "Not recorded"

    summary = entry.get("summary") or ""
    _review_findings_raw = entry.get("findings") or []
    if _review_findings_raw:
        _fp_for_result = [
            f["path"] for f in (entry.get("files_detail") or [])
            if isinstance(f, dict) and f.get("path")
        ]
        _findings_objs = [
            ShardFinding(severity=f.get("severity", "Note"), message=f.get("message", ""))
            for f in _review_findings_raw if isinstance(f, dict)
        ]
        _vis_sub, _meta_g, _, _raw_total = group_review_findings(_findings_objs)
        _vis_areas = len(_vis_sub) + (1 if _meta_g else 0)
        _base = (
            f"{_raw_total} {'issue' if _raw_total == 1 else 'issues'} found"
            if _raw_total == _vis_areas
            else f"{_vis_areas} issue areas found. {_raw_total} raw findings recorded"
        )
        result = _base + ("; review files created." if _fp_for_result else ".")
    elif entry.get("is_review_task"):
        _dom = entry.get("review_domain") or ""
        _dfiles = entry.get("domain_files") or []
        if not _dfiles and _dom:
            from openshard.review.domain_files import no_files_message
            _msg = no_files_message(_dom)
            result = _msg if _msg else "Review completed."
        elif _dfiles and _dom == "docs_onboarding":
            _readme = next((f for f in _dfiles if "readme" in f.lower()), _dfiles[0])
            result = f"{_readme} inspected."
        else:
            result = "Review completed."
    else:
        result = _result_display(summary) or "Not recorded"

    files_detail_raw = entry.get("files_detail") or []
    files_touched = [f["path"] for f in files_detail_raw if isinstance(f, dict) and "path" in f]

    diff_review = entry.get("diff_review") or {}
    if not files_touched:
        _dr_changed = diff_review.get("changed_files")
        if isinstance(_dr_changed, list):
            files_touched = [f for f in _dr_changed if isinstance(f, str)]

    # For runs that explicitly recorded zero file changes, clear files_touched so
    # that any stale files_detail entries (e.g. a model-generated report that was
    # discarded by the read-only safety net) do not appear as changed evidence.
    # Only fires when the entry carries explicit counters — older/minimal entries
    # that lack these keys but have diff_review.changed_files are left untouched.
    _change_counter_keys = ("files_created", "files_updated", "files_deleted")
    _has_explicit_counters = any(k in entry for k in _change_counter_keys)
    if _has_explicit_counters:
        _changed_count = sum(int(entry.get(k) or 0) for k in _change_counter_keys)
        if _changed_count == 0:
            files_touched = []

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

    findings = _extract_findings(entry)
    agent_notes = _safe_str_list(entry.get("agent_notes"))

    repo: Optional[str] = entry.get("repo_name") or None
    if repo is None:
        repo = _workspace_folder_name(entry.get("workspace_path"))

    obs = entry.get("observation") or {}
    _dirty = obs.get("dirty_diff_present")
    if _dirty is True:
        git_state = "Changes pending"
    elif _dirty is False:
        git_state = "Clean"
    else:
        git_state = None
    if git_state is None:
        _gd = entry.get("git_dirty")
        if _gd is True:
            git_state = "Changes pending"
        elif _gd is False:
            git_state = "Clean"

    cqs = entry.get("context_quality_score") or {}
    _cqs_level = cqs.get("level") if isinstance(cqs, dict) else None
    if _cqs_level in ("good", "strong"):
        context_quality: Optional[str] = "Good"
    elif _cqs_level == "fair":
        context_quality = "Partial"
    elif _cqs_level == "weak":
        context_quality = "Weak"
    else:
        context_quality = None

    _fc = entry.get("file_context") or {}
    _fc_read = _fc.get("files_read")
    _fc_paths = _fc.get("paths")
    if type(_fc_read) is int:
        files_read_count: Optional[int] = _fc_read
    else:
        _fr2 = entry.get("final_report") or {}
        _snip = _fr2.get("snippet_files")
        files_read_count = _snip if type(_snip) is int else None
    inspected_files = [p for p in _fc_paths if isinstance(p, str)] if isinstance(_fc_paths, list) else []
    files_referenced: list[str] = sorted({f.path for f in findings if f.path})
    # Merge domain-specific evidence files (CI/CD, auth, docs, tests) as inspected.
    _domain_files_raw = entry.get("domain_files") or []
    _domain_inspected = [p for p in _domain_files_raw if isinstance(p, str)]
    file_evidence = _build_file_evidence(
        inspected_files + _domain_inspected, files_referenced, files_touched
    )

    _adv_raw = entry.get("model_advisory")
    _model_advisory: list[dict] = []
    if isinstance(_adv_raw, list):
        for _a in _adv_raw:
            if isinstance(_a, dict) and "model_id" in _a:
                _model_advisory.append(_a)

    _fra_raw = entry.get("feedback_routing_advisory")
    _feedback_routing_advisory: dict | None = None
    if isinstance(_fra_raw, dict) and _fra_raw.get("advisory_only") is True:
        _feedback_routing_advisory = _fra_raw

    return ShardReceipt(
        shard_id=entry.get("shard_id") or _make_shard_id(timestamp, index),
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
        approval_required=_approval_required,
        approval_granted=_approval_granted,
        approval_reason=_approval_reason,
        cost_display=cost_display,
        result=result,
        status=status,
        duration_seconds=entry.get("duration_seconds"),
        repo=repo,
        branch=entry.get("git_branch") or None,
        git_state=git_state,
        context_quality=context_quality,
        files_read_count=files_read_count,
        inspected_files=inspected_files,
        files_referenced=files_referenced,
        files_detail=files_detail_raw,
        files_touched=files_touched,
        diff_added=diff_added,
        diff_removed=diff_removed,
        cost_raw=cost_raw,
        model_stages=model_stages,
        allowed_paths=allowed_paths,
        blocked_paths=blocked_paths,
        blocked_commands=blocked_commands,
        findings=findings,
        agent_notes=agent_notes,
        check_results=check_results,
        run_timeline=[e for e in (entry.get("run_timeline") or []) if isinstance(e, dict) and e.get("label")],
        developer_feedback=entry.get("developer_feedback") or None,
        file_evidence=file_evidence,
        model_advisory=_model_advisory,
        feedback_routing_advisory=_feedback_routing_advisory,
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


def _truncate_compact(text: str, max_chars: int) -> str:
    """Return first line of text, word-safely truncated to max_chars with … if cut."""
    line = text.split("\n")[0]
    if len(line) <= max_chars:
        return line
    cut = line.rfind(" ", 0, max_chars)
    if cut > 0:
        return line[:cut] + "…"
    return line[:max_chars] + "…"


def render_compact_shard_receipt(receipt: ShardReceipt) -> str:
    """Render a bordered, column-aligned RECEIPT block. Pure, no I/O."""
    file_str = f"{receipt.files_changed} file{'s' if receipt.files_changed != 1 else ''}"
    model_label, model_value = _models_label_and_value(receipt)

    lines = [
        _SEP,
        f"{_INDENT}RECEIPT {_EM} {receipt.shard_id}",
        _SEP,
        _row("Task", receipt.task_short),
        _row("Executor", receipt.agent),
        _row(model_label, model_value),
        _row("Risk", receipt.risk),
        _row("Sandbox", receipt.sandbox),
        _row("Changed", file_str),
        _row("Checks", receipt.checks_display),
        _row("Approval", receipt.approval),
        _row("Cost", receipt.cost_display),
        _row("Result", receipt.result),
    ]
    if receipt.developer_feedback:
        lines.append(_row("Feedback", receipt.developer_feedback.get("outcome", "")))

    _TOP_SEVERITIES = {"Critical", "High", "Medium"}
    top_raw = [f for f in receipt.findings if f.severity in _TOP_SEVERITIES]
    if top_raw:
        # Use generous per-severity caps — the hard limit is the total of 5 visible slots.
        # group_review_findings handles dedup and metadata grouping.
        visible_sub, meta_group, _, _ = group_review_findings(
            top_raw,
            caps={"Critical": 5, "High": 5, "Medium": 5, "Low": 0},
        )
        compact_list: list[ShardFinding] = list(visible_sub)
        if meta_group and len(compact_list) < 5:
            compact_list.append(meta_group)
        compact_list = compact_list[:5]
        hidden_receipt = max(0, len(top_raw) - len(compact_list))
        lines.append(_SEP)
        lines.append(f"{_INDENT}FINDINGS")
        for f in compact_list:
            icon = _FINDING_ICONS.get(f.severity, "⚠" if _UNICODE_OK else "!")
            lines.append(f"{_INDENT}{icon}  {_truncate_compact(f.message, 79)}")
        if hidden_receipt > 0:
            lines.append(f"{_INDENT}+{hidden_receipt} more findings recorded.")

    # Warning line: only shown when a note contains an explicit blocker keyword
    _WARNING_KEYWORDS = ("DO NOT RUN", "WARNING:", "BLOCKER:", "DANGER:", "DO NOT APPLY")
    warning_note = next(
        (n for n in receipt.agent_notes if any(kw in n.upper() for kw in _WARNING_KEYWORDS)),
        None,
    )
    if warning_note:
        lines += [_SEP, f"{_INDENT}{warning_note}"]

    lines.append(_SEP)
    return "\n".join(lines)


def _format_review_checks(checks: list[dict]) -> tuple[str, list[str]]:
    """Return (checks_display, per-check lines) for a list of review check dicts."""
    passed = [c for c in checks if c.get("status") == "passed"]
    failed = [c for c in checks if c.get("status") == "failed"]
    skipped = [c for c in checks if c.get("status") == "skipped"]

    parts: list[str] = []
    if failed:
        parts.append(f"{len(failed)} failed")
    if passed:
        parts.append(f"{len(passed)} passed")
    if skipped:
        parts.append(f"{len(skipped)} skipped")
    checks_display = ", ".join(parts) if parts else "Not run"

    pass_icon = "✓" if _UNICODE_OK else "+"
    fail_icon = "✖" if _UNICODE_OK else "x"
    skip_icon = "-"
    lines: list[str] = []
    for c in checks:
        status = c.get("status", "skipped")
        name = c.get("name", "check")
        reason = c.get("reason") or ""
        summary = c.get("summary") or ""
        if status == "passed":
            lines.append(f"{pass_icon} {name:<22}{summary if summary else 'passed'}")
        elif status == "failed":
            lines.append(f"{fail_icon} {name:<22}{summary if summary else 'failed'}")
        else:
            suffix = f"skipped — {reason}" if reason else "skipped"
            lines.append(f"{skip_icon} {name:<22}{suffix}")

    return checks_display, lines


def build_live_run_receipt(
    *,
    task: str,
    run_id: str,
    run_index: "Optional[int]",
    agent: str,
    stage_runs: list,
    routing_model: "Optional[str]",
    risk: str,
    sandbox: str,
    files_changed: int,
    verification_attempted: "Optional[bool]",
    verification_passed: "Optional[bool]",
    approval: str,
    estimated_cost: "Optional[float]",
    result_summary: str,
    result: "Optional[str]" = None,
    agent_notes: "Optional[list[str]]" = None,
    findings: "Optional[list[ShardFinding]]" = None,
    run_timeline: "Optional[list[dict]]" = None,
    review_checks: "Optional[list[dict]]" = None,
    routing_selected_model: "Optional[str]" = None,
) -> ShardReceipt:
    """Build a ShardReceipt from live run metadata (before log write). Pure, no I/O.

    result: pre-formatted result string that bypasses _result_display() processing.
            Use when the caller has already computed a clean result (e.g. two-count
            review format "N issue areas found. M raw findings recorded.").
    routing_selected_model: the final scored model (if scoring ran); adds the
            "Auto → {model}" prefix to match build_shard_receipt output.
    """
    _model_stages: list[tuple[str, str]] = [
        (
            _STAGE_DISPLAY_LABELS.get(
                getattr(getattr(sr, "stage", None), "stage_type", "") or "",
                (getattr(getattr(sr, "stage", None), "stage_type", None) or "").capitalize(),
            ),
            _display_model_name(getattr(sr, "model", "") or ""),
        )
        for sr in stage_runs
        if getattr(getattr(sr, "stage", None), "stage_type", None) and getattr(sr, "model", None)
    ]
    if routing_selected_model:
        _model_display = f"Auto → {_display_model_name(routing_selected_model)}"
    elif routing_model:
        _model_display = _display_model_name(routing_model)
    elif _model_stages:
        _model_display = _model_stages[0][1]
    else:
        _model_display = "Not recorded"

    if verification_attempted is None or not verification_attempted:
        _checks = "Not run"
        _status = "No checks run"
    elif verification_passed is True:
        _checks = "1/1 passed"
        _status = "Passed"
    elif verification_passed is False:
        _checks = "0/1 passed"
        _status = "Failed"
    else:
        _checks = "Not run"
        _status = "No checks run"

    _check_results: list[str] = []
    if review_checks:
        _checks, _check_results = _format_review_checks(review_checks)
        _status = f"Checks: {_checks}"

    _cost_display = f"${estimated_cost:.4f}" if estimated_cost is not None else "Not recorded"
    _result = result if result is not None else (_result_display(result_summary or "") or "Not recorded")

    return ShardReceipt(
        shard_id=_make_shard_id(run_id, run_index),
        created_at=run_id,
        task_short=_trunc(task, 70),
        task_full=task,
        agent=agent,
        strategy="Not recorded",
        model_display=_model_display,
        risk=risk,
        sandbox=sandbox,
        files_changed=files_changed,
        checks_display=_checks,
        approval=approval,
        cost_display=_cost_display,
        result=_result,
        status=_status,
        duration_seconds=None,
        model_stages=_model_stages,
        agent_notes=[n.split("\n")[0][:300] for n in (agent_notes or []) if n][:5],
        findings=list(findings) if findings else [],
        run_timeline=list(run_timeline) if run_timeline else [],
        check_results=_check_results,
    )


def _fmt_timestamp(ts: str) -> str:
    if not ts:
        return "-"
    try:
        dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        return dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    except (ValueError, AttributeError):
        return ts


def render_full_shard_receipt(receipt: ShardReceipt, detail: str = "full") -> str:
    """Render a full structured SHARD block with consistent separator style. Pure, no I/O."""
    lines: list[str] = []

    lines += [_SEP, f"{_INDENT}SHARD {_EM} {receipt.shard_id}", _SEP, ""]

    lines += [f"{_INDENT}TASK", f"{_INDENT}{receipt.task_full}", ""]

    lines.append(f"{_INDENT}EXECUTION")
    lines.append(_row("Executor", receipt.agent))
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

    if receipt.run_timeline:
        _chk_ok = _UNICODE_OK
        _chk = "✓" if _chk_ok else "+"
        _fail = "✖" if _chk_ok else "x"
        lines.append(f"{_INDENT}TIMELINE")
        for _ev in receipt.run_timeline:
            _st = _ev.get("status", "completed") if isinstance(_ev, dict) else getattr(_ev, "status", "completed")
            _lbl = _ev.get("label", "") if isinstance(_ev, dict) else getattr(_ev, "label", "")
            _sym = _chk if _st != "failed" else _fail
            lines.append(f"{_INDENT}  {_sym} {_lbl}")
        lines.append("")

    lines.append(f"{_INDENT}CONTEXT")
    lines.append(_row("Repo", receipt.repo or "Not recorded"))
    lines.append(_row("Branch", receipt.branch or "Not recorded"))
    lines.append(_row("Git state", receipt.git_state or "Not recorded"))
    lines.append(_row("Quality", receipt.context_quality or "Not recorded"))
    if receipt.files_read_count is not None:
        lines.append(_row("Read", f"{receipt.files_read_count} file{'s' if receipt.files_read_count != 1 else ''}"))
    else:
        lines.append(_row("Read", "Not recorded"))
    touched_count = len(receipt.files_touched)
    if touched_count > 0:
        lines.append(_row("Touched", f"{touched_count} file{'s' if touched_count != 1 else ''}"))
    else:
        lines.append(_row("Touched", "-"))
    lines.append("")

    lines.append(f"{_INDENT}FILE EVIDENCE")
    if receipt.file_evidence:
        _fe_cap = 12
        for _fe in receipt.file_evidence[:_fe_cap]:
            lines.append(f"{_INDENT}  {_fe.path}")
            for _role in _fe.roles:
                lines.append(f"{_INDENT}    {_ROLE_LABELS[_role]}")
            lines.append("")
        if len(receipt.file_evidence) > _fe_cap:
            lines.append(f"{_INDENT}  (+{len(receipt.file_evidence) - _fe_cap} more)")
            lines.append("")
    else:
        lines.append(f"{_INDENT}  Not recorded")
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

    if receipt.approval_required:
        lines.append(f"{_INDENT}APPROVAL")
        lines.append(_row("Required", "yes"))
        lines.append(_row("Status", "granted" if receipt.approval_granted else "denied"))
        if receipt.approval_reason:
            lines.append(_row("Reason", receipt.approval_reason))
        if not receipt.approval_granted:
            lines.append(_row("Result", "Writes blocked"))
        lines.append("")

    lines.append(f"{_INDENT}CHECKS")
    if receipt.check_results:
        for cr in receipt.check_results:
            lines.append(f"{_INDENT}  {cr}")
    else:
        lines.append(f"{_INDENT}{receipt.checks_display}")
    lines.append("")

    lines.append(f"{_INDENT}FINDINGS")
    if receipt.findings:
        # Show substantive findings (no cap in full receipt) grouped by severity.
        visible_sub, meta_group, _, _ = group_review_findings(
            receipt.findings,
            caps={"Critical": 999, "High": 999, "Medium": 999, "Low": 999},
        )
        current_severity: str | None = None
        for finding in visible_sub:
            if finding.severity != current_severity:
                if current_severity is not None:
                    lines.append("")
                lines.append(f"{_INDENT}{finding.severity.upper()}")
                current_severity = finding.severity
            icon = _FINDING_ICONS.get(finding.severity, "-")
            msg = finding.message
            if finding.path:
                loc = f"{finding.path}:{finding.line}" if finding.line is not None else finding.path
                msg = f"{msg}  [{loc}]"
            lines.append(f"{_INDENT}  {icon} {msg}")
        # Metadata section: show all examples (up to 10)
        if meta_group:
            if current_severity is not None:
                lines.append("")
            lines.append(f"{_INDENT}METADATA")
            # Collect all metadata findings for the expanded view
            meta_findings = [f for f in receipt.findings if _is_metadata_noise(f)]
            n = len(meta_findings)
            uses_labels = any("label" in f.message.lower() for f in meta_findings)
            has_gcp     = any("google_" in f.message for f in meta_findings)
            term   = "labels" if uses_labels else "tags"
            prefix = "GCP " if has_gcp else ""
            lines.append(f"{_INDENT}  ~ {n} {prefix}resources missing ownership/environment {term}")
            examples: list[str] = []
            seen_ex: set[str] = set()
            for f in meta_findings:
                parts = f.message.split()
                if len(parts) >= 2 and parts[0].lower() == "resource":
                    name = parts[1]
                    if name not in seen_ex:
                        seen_ex.add(name)
                        examples.append(name)
            for ex in examples[:10]:
                lines.append(f"{_INDENT}    {ex}")
            if len(examples) > 10:
                lines.append(f"{_INDENT}    ...and {len(examples) - 10} more")
    else:
        lines.append(f"{_INDENT}  No structured findings recorded.")
    lines.append("")

    if receipt.model_advisory:
        lines.append(f"{_INDENT}MODEL ADVISORY")
        lines.append(f"{_INDENT}  Advisory only — routing unchanged")
        lines.append(f"{_INDENT}  Generated from risk signal only")
        for _adv in receipt.model_advisory:
            _name = _adv.get("display_name") or _adv.get("model_id", "?")
            lines.append(f"{_INDENT}  {_name}")
            for _r in _adv.get("reasons", [])[:3]:
                lines.append(f"{_INDENT}    ↳ {_r}")
        lines.append("")

    if detail == "full" and receipt.feedback_routing_advisory:
        _fra = receipt.feedback_routing_advisory
        lines.append(f"{_INDENT}FEEDBACK ROUTING ADVISORY")
        lines.append(f"{_INDENT}  Advisory only — routing unchanged")
        _rec = _fra.get("recommendation", "").replace("_", " ")
        lines.append(f"{_INDENT}  Recommendation  {_rec}")
        lines.append(f"{_INDENT}  Confidence      {_fra.get('confidence', '')}")
        _reason = _fra.get("reason", "")
        if _reason:
            lines.append(f"{_INDENT}  Reason          {_reason}")
        _sigs = _fra.get("signals_considered") or {}
        _sig_parts = [f"{k}={v}" for k, v in _sigs.items() if isinstance(v, int) and v > 0]
        if _sig_parts:
            lines.append(f"{_INDENT}  Signals         {', '.join(_sig_parts)}")
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
