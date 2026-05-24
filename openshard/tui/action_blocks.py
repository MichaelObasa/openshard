from __future__ import annotations

_MAX_INSPECTED = 5
_MAX_FINDINGS = 8


def render_action_block(title: str, detail: str | None = None) -> str:
    lines = [f"  {title}"]
    if detail:
        lines.append(f"    ↳ [dim]{detail}[/dim]")
    return "\n".join(lines)


def _event_detail(ev: dict) -> str | None:
    if ev.get("detail"):
        return str(ev["detail"])
    count = ev.get("count")
    if count is not None:
        return f"{count} total"
    status = ev.get("status", "completed")
    if status == "skipped":
        return "skipped"
    if status == "failed":
        return "failed"
    return None


def render_actions_section(events: list[dict]) -> str:
    if not events:
        return ""
    lines = ["[bold]ACTIONS[/bold]"]
    for ev in events:
        label = str(ev.get("label") or "").strip()
        if not label:
            continue
        lines.append(render_action_block(label, _event_detail(ev)))
    if len(lines) == 1:
        return ""
    return "\n".join(lines) + "\n"


def render_check_actions_section(checks: list[dict]) -> str:
    if not checks:
        return ""
    lines = ["[bold]CHECK ACTIONS[/bold]"]
    for check in checks:
        name = str(check.get("name") or "").strip()
        if not name:
            continue
        status = check.get("status", "")
        if status == "passed":
            icon = "[green]✓[/green]"
            detail: str | None = check.get("summary") or "passed"
        elif status == "failed":
            icon = "[red]✗[/red]"
            detail = check.get("summary") or "failed"
        elif status == "skipped":
            icon = "[dim]-[/dim]"
            reason = check.get("reason") or ""
            detail = f"skipped — {reason}" if reason else "skipped"
        else:
            icon = " "
            detail = status or None
        lines.append(render_action_block(f"{icon} {name}", detail))
    if len(lines) == 1:
        return ""
    return "\n".join(lines) + "\n"


def render_evidence_section(
    inspected_files: list[str], files_with_findings: list[str]
) -> str:
    if not inspected_files and not files_with_findings:
        return ""
    lines = ["[bold]EVIDENCE[/bold]"]

    shown_i = inspected_files[:_MAX_INSPECTED]
    for p in shown_i:
        lines.append(render_action_block(f"Read {p}", "inspected file"))
        lines.append("")
    if len(inspected_files) > _MAX_INSPECTED:
        lines.append(f"  +{len(inspected_files) - _MAX_INSPECTED} more inspected files")
        lines.append("")

    shown_f = files_with_findings[:_MAX_FINDINGS]
    for p in shown_f:
        lines.append(render_action_block(f"Finding source {p}", "file with findings"))
        lines.append("")
    if len(files_with_findings) > _MAX_FINDINGS:
        lines.append(f"  +{len(files_with_findings) - _MAX_FINDINGS} more files with findings")
        lines.append("")

    while lines and lines[-1] == "":
        lines.pop()
    return "\n".join(lines) + "\n"
