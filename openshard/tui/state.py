from __future__ import annotations

import json
import subprocess
from pathlib import Path


def load_git_info(path: Path) -> dict:
    name = path.name or "unknown"
    try:
        result = subprocess.run(
            ["git", "status", "--short", "--branch"],
            cwd=str(path),
            capture_output=True,
            text=True,
            timeout=3,
        )
    except Exception:
        return {"project_name": name, "branch": "unknown", "state": "unknown"}

    if result.returncode != 0:
        return {"project_name": name, "branch": "unknown", "state": "unknown"}

    lines = [line for line in result.stdout.splitlines() if line.strip()]
    if not lines:
        return {"project_name": name, "branch": "unknown", "state": "unknown"}

    branch_raw = lines[0].removeprefix("## ").strip()
    branch = branch_raw.split("...")[0].split(" ")[0] or "unknown"
    changed = [line for line in lines if not line.startswith("##")]
    state = "dirty" if changed else "clean"
    return {"project_name": name, "branch": branch, "state": state}


def load_recent_runs(path: Path, limit: int = 5) -> list[dict]:
    jsonl_path = path / ".openshard" / "runs.jsonl"
    if not jsonl_path.exists():
        return []

    raw: list[dict] = []
    for line in jsonl_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            raw.append(json.loads(line))
        except json.JSONDecodeError:
            continue

    recent = raw[-limit:] if len(raw) > limit else raw
    recent = list(reversed(recent))

    result = []
    for entry in recent:
        task = str(entry.get("task") or "untitled")[:48]
        ts = str(entry.get("timestamp") or "")[:16]

        dur_val = entry.get("duration_seconds")
        duration = f"{dur_val:.1f}s" if isinstance(dur_val, (int, float)) else "-"

        vp = entry.get("verification_passed")
        if vp is True:
            status = "passed"
        elif vp is False:
            status = "failed"
        elif (entry.get("form_factor") or {}).get("read_only") is True:
            status = "read-only"
        elif (
            vp is None
            and entry.get("files_created", -1) == 0
            and entry.get("files_updated", -1) == 0
            and entry.get("files_deleted", -1) == 0
        ):
            status = "read-only"
        else:
            status = "unknown"

        result.append({"task": task, "timestamp": ts, "duration": duration, "status": status})

    return result


def get_guardrails() -> dict:
    return {
        "Agent": "OpenShard Native",
        "Model": "Auto",
        "Sandbox": "On",
        "Approval": "Smart",
        "Checks": "Auto",
        "Receipts": "On",
    }
