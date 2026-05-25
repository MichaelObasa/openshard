from __future__ import annotations

from pathlib import Path

# Directories that should never contribute evidence files.
_EXCLUDED_DIRS: frozenset[str] = frozenset({
    ".openshard", ".git", "node_modules", ".venv", "venv", "env",
    "__pycache__", "dist", "build", ".tox", ".mypy_cache",
    ".pytest_cache", ".next", ".cache",
})


def _is_excluded(path: Path, repo_root: Path) -> bool:
    """Return True if *path* is under any excluded directory."""
    try:
        parts = path.relative_to(repo_root).parts
    except ValueError:
        return False
    return bool(parts and parts[0] in _EXCLUDED_DIRS)


_DOMAIN_GLOB_PATTERNS: dict[str, list[str]] = {
    "cicd": [
        ".github/workflows/*",
        ".gitlab-ci.yml",
        "azure-pipelines.yml",
        "Jenkinsfile",
        "buildspec.yml",
        "Dockerfile",
        "docker-compose.yml",
    ],
    "tests": [
        "tests/*",
        "test_*.py",
        "*_test.py",
        "*.spec.js",
        "*.spec.ts",
        "*.test.js",
        "*.test.ts",
    ],
    "docs_onboarding": [
        "README*",
        "docs/*",
        "CONTRIBUTING*",
        "DEVELOPMENT*",
        "SETUP*",
        "ONBOARDING*",
    ],
}

_AUTH_NAME_TERMS: tuple[str, ...] = (
    "auth", "login", "session", "token", "jwt", "oauth",
    "middleware", "security",
)

_NO_FILES_MESSAGES: dict[str, str] = {
    "cicd":            "No CI/CD configuration files were found in the scanned repo.",
    "auth_security":   "No obvious auth-related code paths were found in this repo.",
    "tests":           "No test files were found.",
    "docs_onboarding": "No README or onboarding files were found.",
    "generic_review":  "",
    "terraform_iac":   "",
}

_MAX_FILES = 20


def find_review_domain_files(repo_root: Path, domain: str) -> list[str]:
    """Return up to _MAX_FILES paths relevant to *domain* under *repo_root*.

    Uses glob patterns or name-substring matching. Read-only, no side effects.
    Paths are returned relative to repo_root using forward slashes.
    Files under excluded directories (.openshard, .git, node_modules, etc.) are
    always omitted.
    """
    found: list[str] = []

    if domain == "auth_security":
        for p in repo_root.rglob("*"):
            if not p.is_file() or _is_excluded(p, repo_root):
                continue
            name = p.name.lower()
            if any(term in name for term in _AUTH_NAME_TERMS):
                found.append(p.relative_to(repo_root).as_posix())
            if len(found) >= _MAX_FILES:
                break
        return found

    patterns = _DOMAIN_GLOB_PATTERNS.get(domain, [])
    for pat in patterns:
        for match in repo_root.glob(pat):
            if match.is_file() and not _is_excluded(match, repo_root):
                found.append(match.relative_to(repo_root).as_posix())
        if len(found) >= _MAX_FILES:
            break
    return found[:_MAX_FILES]


def no_files_message(domain: str) -> str:
    """Return the honest fallback message when no files were found for *domain*."""
    return _NO_FILES_MESSAGES.get(domain, "")
