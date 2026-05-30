"""Bounded repo observation for OSN execution.

Runs before any planning or tool steps. Never executes shell commands.
Never stores raw file contents, absolute paths, raw prompts, or command output.
"""
from __future__ import annotations

import json
from pathlib import Path

from openshard.native.context import (
    OSNObservationPacket,
    _OBS_MAX_CANDIDATE_FILES,
    _OBS_MAX_CONFIG_FILES,
    _OBS_MAX_TEST_FILES,
    _OBS_MAX_RISKY_MARKERS,
    _OBS_MAX_SUGGESTED_CHECKS,
)

# Noisy dirs to skip - mirrors analysis/repo.py _SKIP_DIRS with .codegraph added.
# Defined locally to avoid depending on private internals of analysis/repo.py.
_OBS_SKIP_DIRS: frozenset[str] = frozenset({
    ".git", "node_modules", "__pycache__", ".venv", "venv",
    ".mypy_cache", ".pytest_cache", ".tox", "dist", "build",
    ".next", ".nuxt", ".openshard", ".codegraph",
})

# Source file extensions treated as candidate files
_CANDIDATE_EXTS: frozenset[str] = frozenset({
    ".py", ".js", ".jsx", ".ts", ".tsx", ".go", ".rs", ".rb",
    ".java", ".cs", ".cpp", ".c", ".swift", ".kt", ".php",
})

# Config/CI file names to collect
_CONFIG_NAMES: frozenset[str] = frozenset({
    "pyproject.toml", "package.json", "Cargo.toml", "go.mod",
    "requirements.txt", "setup.py", "setup.cfg",
    "docker-compose.yml", "docker-compose.yaml",
    "azure-pipelines.yml", ".gitlab-ci.yml",
    "Makefile",
})

# Risky path keyword fragments
_RISKY_KEYWORDS: tuple[str, ...] = (
    "auth", "payment", "secret", "credential", "migration",
    "password", "token", "security",
)

# Max files to walk before setting files_capped
_WALK_FILE_CAP: int = 2000

# Max bytes to read from package.json when checking for test script
_PKG_JSON_SIZE_CAP: int = 8192


def _rel(path: Path, root: Path) -> str:
    """Return a relative, forward-slash path string. Never an absolute path."""
    return str(path.relative_to(root)).replace("\\", "/")


def _walk_bounded(root: Path):
    """Yield (Path, rel_str) for non-noisy files under root, capped at _WALK_FILE_CAP."""
    count = 0
    for p in root.rglob("*"):
        if count >= _WALK_FILE_CAP:
            return
        parts = p.relative_to(root).parts
        if any(part in _OBS_SKIP_DIRS for part in parts):
            continue
        if p.is_file():
            yield p, _rel(p, root)
            count += 1


def _has_tf_files(root: Path) -> bool:
    """Return True if any .tf file exists outside noisy dirs."""
    for p in root.rglob("*.tf"):
        try:
            parts = p.relative_to(root).parts
        except ValueError:
            continue
        if not any(part in _OBS_SKIP_DIRS for part in parts):
            return True
    return False


def _detect_stack_signals(root: Path) -> list[str]:
    """Detect stack from file presence only. No subprocess or file content reads (except package.json size)."""
    signals: list[str] = []

    if any((root / f).is_file() for f in ("pyproject.toml", "requirements.txt", "setup.py", "setup.cfg")):
        signals.append("python")

    if (root / "package.json").is_file():
        signals.append("node")

    dockerfile = (root / "Dockerfile").is_file()
    compose = (root / "docker-compose.yml").is_file() or (root / "docker-compose.yaml").is_file()
    if dockerfile or compose:
        signals.append("docker")

    gha_dir = root / ".github" / "workflows"
    if gha_dir.is_dir() and any(gha_dir.glob("*.yml")):
        signals.append("github-actions")

    if _has_tf_files(root):
        signals.append("terraform")

    if (root / "pyproject.toml").is_file() and (root / "openshard").is_dir():
        signals.append("openshard")

    ci_files = ("azure-pipelines.yml", ".gitlab-ci.yml")
    if any((root / f).is_file() for f in ci_files):
        if "github-actions" not in signals:
            signals.append("ci-cd")

    return signals


def _suggest_checks(stack_signals: list[str], root: Path) -> list[str]:
    """Return suggested check command strings based on detected stack. Never executes them."""
    checks: list[str] = []

    if "python" in stack_signals:
        tests_dir = root / "tests"
        if tests_dir.is_dir():
            checks.append("python -m pytest tests/ -v")
        else:
            checks.append("python -m pytest -v")
        checks.append("python -m ruff check")

    if "terraform" in stack_signals:
        checks.append("terraform fmt -check")
        checks.append("terraform validate")

    if "node" in stack_signals:
        test_cmd = _node_test_command(root)
        checks.append(test_cmd)

    return checks[:_OBS_MAX_SUGGESTED_CHECKS]


def _node_test_command(root: Path) -> str:
    """Infer npm/pnpm/yarn test command from package.json. Returns generic fallback on any error."""
    pkg_path = root / "package.json"
    try:
        size = pkg_path.stat().st_size if pkg_path.is_file() else 0
        if 0 < size <= _PKG_JSON_SIZE_CAP:
            pkg = json.loads(pkg_path.read_text(encoding="utf-8"))
            if isinstance(pkg, dict) and "test" in pkg.get("scripts", {}):
                if (root / "pnpm-lock.yaml").is_file():
                    return "pnpm test"
                if (root / "yarn.lock").is_file():
                    return "yarn test"
                return "npm test"
    except Exception:
        pass
    return "npm test"


def build_osn_observation(repo_root: str | Path) -> OSNObservationPacket:
    """Build a bounded, safe repo observation packet.

    Never executes shell commands. Never stores file contents, absolute paths,
    raw prompts, or command output. Safe to call before any OSN planning.
    """
    root = Path(repo_root)
    if not root.is_dir():
        return OSNObservationPacket(enabled=False, repo_root_present=False)

    candidate_files: list[str] = []
    config_files: list[str] = []
    test_files: list[str] = []
    risky_markers: list[str] = []
    files_considered = 0
    files_capped = False

    for p, rel in _walk_bounded(root):
        files_considered += 1
        if files_considered >= _WALK_FILE_CAP:
            files_capped = True

        name = p.name
        name_lower = name.lower()

        if p.suffix.lower() in _CANDIDATE_EXTS:
            if len(candidate_files) < _OBS_MAX_CANDIDATE_FILES:
                candidate_files.append(rel)

        if name in _CONFIG_NAMES:
            if len(config_files) < _OBS_MAX_CONFIG_FILES:
                config_files.append(rel)

        # GitHub Actions YAML
        if ".github/workflows" in rel and name.endswith((".yml", ".yaml")):
            if len(config_files) < _OBS_MAX_CONFIG_FILES and rel not in config_files:
                config_files.append(rel)

        if (
            name_lower.startswith("test_")
            or name_lower.endswith("_test.py")
            or name_lower.endswith(".test.ts")
            or name_lower.endswith(".test.js")
            or name_lower.endswith(".spec.ts")
            or name_lower.endswith(".spec.js")
        ):
            if len(test_files) < _OBS_MAX_TEST_FILES:
                test_files.append(rel)

        if any(kw in name_lower for kw in _RISKY_KEYWORDS):
            if len(risky_markers) < _OBS_MAX_RISKY_MARKERS:
                risky_markers.append(rel)

    stack_signals = _detect_stack_signals(root)
    suggested_checks = _suggest_checks(stack_signals, root)

    stack_str = ", ".join(stack_signals) if stack_signals else "unknown"
    summary = (
        f"Repo observation: stack={stack_str}, "
        f"candidates={len(candidate_files)}, "
        f"tests={len(test_files)}, "
        f"checks={len(suggested_checks)}"
    )

    return OSNObservationPacket(
        enabled=True,
        repo_root_present=True,
        stack_signals=stack_signals,
        candidate_files=candidate_files,
        config_files=config_files,
        test_files=test_files,
        risky_markers=risky_markers,
        suggested_checks=suggested_checks,
        observation_summary=summary,
        files_considered=files_considered,
        files_capped=files_capped,
        source="repo_observation_v1",
    )
