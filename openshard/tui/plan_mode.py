from __future__ import annotations

from pathlib import Path

from openshard.analysis.repo import RepoFacts, analyze_repo

PLAN_FAST_PATHS: tuple[str, ...] = (
    "plan ",
    "how should i ",
    "what is the safest way to ",
    "how do i approach ",
)

_USAGE = (
    "Plan Mode needs a task. Example:\n"
    "  /plan refactor the auth module safely\n"
    "  /plan add tests for the payment module\n"
    "  /plan review this Terraform repo before changing anything"
)

_APPROACH = (
    "  Suggested approach\n"
    "    1. Inspect relevant files and repo shape\n"
    "    2. Identify risk areas\n"
    "    3. Propose a small change scope\n"
    "    4. Run targeted checks\n"
    "    5. Review receipt before applying anything risky"
)

_RISK_NOTES = (
    "  Risk notes\n"
    "    - Auth/payment/infra/security tasks should require stronger review\n"
    "    - Destructive writes should require approval"
)

_NEXT_STEP = (
    "  Next step\n"
    "    Run the task normally when ready, or use a workflow pack if available."
)

_MAX_RELEVANT_FILES = 10

# Mirrors analysis/repo._SKIP_DIRS — kept local to avoid private import dependency.
_PLAN_SKIP_DIRS: frozenset[str] = frozenset({
    ".git", "node_modules", "__pycache__", ".venv", "venv",
    ".mypy_cache", ".pytest_cache", ".tox", "dist", "build",
    ".next", ".nuxt", ".openshard",
})


def _is_noisy(p: Path, root: Path) -> bool:
    try:
        return any(part in _PLAN_SKIP_DIRS for part in p.relative_to(root).parts)
    except ValueError:
        return True


def _detect_extra_stack(path: Path) -> dict[str, list[str]]:
    extras: dict[str, list[str]] = {}
    tf_files = sorted(
        str(p.relative_to(path)).replace("\\", "/")
        for p in path.rglob("*.tf")
        if not _is_noisy(p, path)
    )
    if tf_files:
        extras["terraform"] = tf_files[:3]
    workflows_dir = path / ".github" / "workflows"
    if workflows_dir.is_dir():
        wf_files = sorted(
            str(p.relative_to(path)).replace("\\", "/")
            for p in workflows_dir.iterdir()
            if p.suffix in {".yml", ".yaml"}
        )
        if wf_files:
            extras["github_actions"] = wf_files[:2]
    docker_files = [
        fname for fname in ("Dockerfile", "docker-compose.yml", "docker-compose.yaml")
        if (path / fname).is_file()
    ]
    if docker_files:
        extras["docker"] = docker_files
    cicd_files = [
        fname for fname in ("azure-pipelines.yml", ".gitlab-ci.yml", "Jenkinsfile")
        if (path / fname).is_file()
    ]
    if cicd_files:
        extras["cicd"] = cicd_files
    return extras


def _gather_relevant_files(
    path: Path, facts: RepoFacts, extra_stack: dict[str, list[str]]
) -> list[str]:
    seen: set[str] = set()
    files: list[str] = []

    def _add(f: str) -> None:
        if len(files) < _MAX_RELEVANT_FILES and f not in seen:
            seen.add(f)
            files.append(f)

    for f in facts.package_files:
        _add(f)
    for f in extra_stack.get("terraform", []):
        _add(f)
    for f in extra_stack.get("github_actions", []):
        _add(f)
    for f in extra_stack.get("docker", []):
        _add(f)
    for f in extra_stack.get("cicd", []):
        _add(f)
    for f in facts.risky_paths[:3]:
        _add(f)
    return files


def _risk_level(facts: RepoFacts, extra_stack: dict[str, list[str]]) -> str:
    if facts.risky_paths and ("terraform" in extra_stack or "cicd" in extra_stack):
        return "High"
    if facts.risky_paths:
        return "Medium"
    return "Low"


def _build_stack_line(facts: RepoFacts, extra_stack: dict[str, list[str]]) -> str:
    _DISPLAY = {
        "terraform": "Terraform",
        "github_actions": "GitHub Actions",
        "docker": "Docker",
        "cicd": "CI/CD",
    }
    parts = list(facts.languages)
    for key, label in _DISPLAY.items():
        if key in extra_stack:
            parts.append(label)
    return ", ".join(parts) if parts else "unknown"


def _answer_local_only(task: str) -> str:
    return (
        "PLAN\n"
        "\n"
        "  Local plan only\n"
        "    No repo scan, no provider call, no files changed.\n"
        "\n"
        f"  Goal\n"
        f"    {task}\n"
        "\n"
        f"{_APPROACH}\n"
        "\n"
        f"{_RISK_NOTES}\n"
        "\n"
        f"{_NEXT_STEP}"
    )


def _answer_repo_aware(task: str, path: Path) -> str:
    facts = analyze_repo(path)
    extra_stack = _detect_extra_stack(path)
    relevant_files = _gather_relevant_files(path, facts, extra_stack)
    risk = _risk_level(facts, extra_stack)
    stack_line = _build_stack_line(facts, extra_stack)

    files_section = ""
    if relevant_files:
        file_lines = "\n".join(f"    {f}" for f in relevant_files)
        files_section = f"\n  Relevant files found\n{file_lines}\n"

    checks: list[str] = []
    if facts.test_command:
        checks.append(facts.test_command)
    if "python" in facts.languages and "ruff" not in (facts.test_command or ""):
        checks.append("ruff check")
    if "terraform" in extra_stack:
        checks.append("terraform validate")
    checks_section = ""
    if checks:
        check_lines = "\n".join(f"    · {c}" for c in checks)
        checks_section = f"\n  Checks to run\n{check_lines}\n"

    return (
        "PLAN\n"
        "\n"
        "  Repo-aware read-only plan\n"
        "    No files changed  ·  No provider call\n"
        "\n"
        f"  Goal\n"
        f"    {task}\n"
        "\n"
        "  Repo signals\n"
        f"    Stack: {stack_line}\n"
        f"    Relevant files found: {len(relevant_files)}\n"
        f"    Risk level: {risk}\n"
        f"{files_section}"
        f"{checks_section}"
        "\n"
        f"{_RISK_NOTES}\n"
        "\n"
        "  Policy note\n"
        "    Read-only planning only  ·  Writes require Run Mode\n"
        "\n"
        f"{_NEXT_STEP}"
    )


def answer_plan_mode(task: str, path: Path | str | None = None) -> str:
    if not task.strip():
        return _USAGE
    if path is not None:
        try:
            return _answer_repo_aware(task.strip(), Path(path))
        except Exception:
            pass
    return _answer_local_only(task.strip())
