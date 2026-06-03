"""Repo-aware Plan Mode v1 - pure, deterministic planner.

Turns a repo-map dict (the safe, bounded metadata produced by
``analysis/repo_map.py``) plus a task string into a structured plan. This module
is side-effect-free: no model calls, no network, no file reads, no writes. It
consumes only metadata - it never reads file contents - so plan steps use honest
wording ("Suggested files to inspect", never "Files inspected").

The task argument is untrusted user text and is sanitised by ``_safe_task_text``
before it appears in any output (human or JSON): absolute local paths and
secret-like tokens are stripped, length is capped.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

_TASK_CAP = 200
_TASK_PLACEHOLDER = "OpenShard repo-aware plan"
_REDACTED = "<redacted>"
_PATH_TOKEN = "<path>"

# Absolute local paths. The Windows drive letter must not be part of a longer
# word (the lookbehind keeps "http://" from matching on its trailing "p").
_WIN_ABS_RE = re.compile(r"(?<![A-Za-z])[A-Za-z]:[\\/][^\s]*")
_POSIX_ABS_RE = re.compile(r"/(?:Users|home)/[^\s]*", re.IGNORECASE)
# Secret-like tokens.
_KV_SECRET_RE = re.compile(
    r"(?i)\b(?:token|secret|api[_-]?key|password|passwd|pwd)\s*[=:]\s*\S+"
)
_SK_RE = re.compile(r"\bsk-[A-Za-z0-9_-]{6,}")
_AKIA_RE = re.compile(r"\bAKIA[0-9A-Z]{6,}")
# Any remaining long opaque token (keys, hashes) - redact conservatively.
_LONG_OPAQUE_RE = re.compile(r"\b[A-Za-z0-9_\-]{32,}\b")


def _safe_task_text(value: object) -> str:
    """Sanitise untrusted task text for display in human and JSON output.

    Accepts only strings; strips CR/LF and whitespace; removes absolute local
    paths and secret-like tokens; caps length. Returns a neutral placeholder if
    nothing readable remains. Never carries file contents, diffs, env values, or
    secret-looking values through.
    """
    if not isinstance(value, str):
        return _TASK_PLACEHOLDER
    s = value.replace("\r", " ").replace("\n", " ")
    # Secrets first (a key=value pair may also look path-like).
    s = _KV_SECRET_RE.sub(_REDACTED, s)
    s = _SK_RE.sub(_REDACTED, s)
    s = _AKIA_RE.sub(_REDACTED, s)
    # Absolute local paths -> neutral token.
    s = _WIN_ABS_RE.sub(_PATH_TOKEN, s)
    s = _POSIX_ABS_RE.sub(_PATH_TOKEN, s)
    # Anything still opaque and long (keys, hashes) -> redact.
    s = _LONG_OPAQUE_RE.sub(_REDACTED, s)
    s = " ".join(s.split())
    if len(s) > _TASK_CAP:
        s = s[:_TASK_CAP].rstrip()
    return s or _TASK_PLACEHOLDER


@dataclass
class RepoAwarePlan:
    """A deterministic, read-only plan derived from repo-map metadata."""

    task: str  # already sanitised via _safe_task_text
    repo_context: dict
    plan_steps: list[str]
    safety_notes: list[str]
    warnings: list[str]

    def to_dict(self) -> dict:
        """The ``--json`` body (the machine envelope adds the outer fields)."""
        return {
            "task": self.task,
            "repo_context": dict(self.repo_context),
            "plan_steps": list(self.plan_steps),
            "safety_notes": list(self.safety_notes),
        }


def _build_repo_context(repo_map: dict, *, cache_hit: bool) -> dict:
    summary = repo_map.get("summary", {}) or {}
    git = repo_map.get("git", {}) or {}
    return {
        "languages": list(summary.get("languages", []) or []),
        "frameworks": list(summary.get("frameworks", []) or []),
        "package_managers": list(summary.get("package_managers", []) or []),
        "test_commands": list(summary.get("test_commands", []) or []),
        "important_files": list(repo_map.get("important_files", []) or []),
        "risky_areas": list(repo_map.get("risky_areas", []) or []),
        "git_dirty": bool(git.get("dirty", False)),
        "cache_hit": bool(cache_hit),
    }


def _build_plan_steps(ctx: dict) -> list[str]:
    languages = ctx["languages"]
    test_commands = ctx["test_commands"]
    steps: list[str] = []
    if ctx["important_files"]:
        steps.append(
            "Inspect the suggested files to understand current structure and conventions."
        )
    else:
        steps.append(
            "Inspect the repository structure and conventions relevant to the task."
        )
    if languages:
        steps.append(
            f"Identify the current {languages[0]} patterns and test style near the change area."
        )
    else:
        steps.append("Identify the current patterns and test style near the change area.")
    steps.append("Make the smallest focused change scoped to the task.")
    if test_commands:
        steps.append(f"Run {test_commands[0]} to verify the change.")
    else:
        steps.append(
            "No test command detected; run targeted checks appropriate to the change."
        )
    steps.append("Review the Shard receipt before applying changes.")
    return steps


def _build_safety_notes(repo_map: dict, ctx: dict) -> list[str]:
    git = repo_map.get("git", {}) or {}
    is_git = bool(git.get("branch")) or bool(git.get("head_commit"))
    notes: list[str] = []
    if ctx["git_dirty"]:
        notes.append("Repo is dirty; avoid mixing unrelated changes.")
    if not is_git:
        notes.append("Not a git repository; plan is based on file layout only.")
    if ctx["risky_areas"]:
        notes.append(
            "Risky/secret-like files are present; do not read or expose their contents."
        )
    if not ctx["languages"] and not ctx["frameworks"]:
        notes.append("No stack detected; plan is generic.")
    if not ctx["test_commands"]:
        notes.append("No test command detected; verification steps are generic.")
    # Honesty note - v1 reads no file contents.
    notes.append("Plan is metadata-only; suggested files have not been read.")
    return notes


def build_repo_aware_plan(task: object, repo_map: dict, *, cache_hit: bool) -> RepoAwarePlan:
    """Build a :class:`RepoAwarePlan` from a repo-map dict and a task.

    Pure and deterministic. ``task`` is sanitised; ``repo_map`` is the dict form
    of a :class:`~openshard.analysis.repo_map.RepoMap`.
    """
    ctx = _build_repo_context(repo_map, cache_hit=cache_hit)
    return RepoAwarePlan(
        task=_safe_task_text(task),
        repo_context=ctx,
        plan_steps=_build_plan_steps(ctx),
        safety_notes=_build_safety_notes(repo_map, ctx),
        warnings=list(repo_map.get("warnings", []) or []),
    )
