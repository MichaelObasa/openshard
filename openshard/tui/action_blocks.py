from __future__ import annotations


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
