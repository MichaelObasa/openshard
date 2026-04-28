from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass
class RepoFacts:
    languages: list[str]
    package_files: list[str]
    framework: str | None
    test_command: str | None
    risky_paths: list[str]
    changed_files: list[str]


_EXT_TO_LANG: dict[str, str] = {
    ".py": "python",
    ".js": "javascript",
    ".jsx": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".rs": "rust",
    ".go": "go",
    ".java": "java",
    ".rb": "ruby",
    ".php": "php",
    ".cs": "csharp",
    ".cpp": "cpp",
    ".c": "c",
    ".swift": "swift",
    ".kt": "kotlin",
}

_KNOWN_PACKAGE_FILES: frozenset[str] = frozenset({
    "package.json",
    "requirements.txt",
    "pyproject.toml",
    "Cargo.toml",
    "go.mod",
    "pom.xml",
    "build.gradle",
    "Gemfile",
    "composer.json",
    "setup.py",
    "setup.cfg",
})

_RISKY_KEYWORDS: tuple[str, ...] = (
    "auth", "payment", "config", "env", "security", "migration",
)

_SKIP_DIRS: frozenset[str] = frozenset({
    ".git", "node_modules", "__pycache__", ".venv", "venv",
    ".mypy_cache", ".pytest_cache", ".tox", "dist", "build",
    ".next", ".nuxt", ".openshard",
})


def _walk_repo(path: Path):
    for p in path.rglob("*"):
        rel_parts = p.relative_to(path).parts
        if any(part in _SKIP_DIRS for part in rel_parts):
            continue
        yield p


def _detect_languages(path: Path) -> list[str]:
    seen: set[str] = set()
    for p in _walk_repo(path):
        if p.is_file():
            lang = _EXT_TO_LANG.get(p.suffix.lower())
            if lang:
                seen.add(lang)
    return sorted(seen)


def _detect_package_files(path: Path) -> list[str]:
    return sorted(name for name in _KNOWN_PACKAGE_FILES if (path / name).is_file())


def _detect_framework(path: Path, package_files: list[str]) -> str | None:
    if "package.json" in package_files:
        try:
            pkg = json.loads((path / "package.json").read_text(encoding="utf-8"))
            deps = {**pkg.get("dependencies", {}), **pkg.get("devDependencies", {})}
            for fw in ("next", "react", "vue", "angular", "express", "fastify", "svelte"):
                if fw in deps:
                    return fw
        except Exception:
            pass

    for fname in ("requirements.txt", "pyproject.toml", "setup.cfg"):
        if fname not in package_files:
            continue
        try:
            content = (path / fname).read_text(encoding="utf-8").lower()
            for fw in ("django", "fastapi", "flask", "starlette", "tornado"):
                if fw in content:
                    return fw
        except Exception:
            pass

    return None


def _detect_test_command(path: Path, package_files: list[str]) -> str | None:
    if "package.json" in package_files:
        try:
            pkg = json.loads((path / "package.json").read_text(encoding="utf-8"))
            if "test" in pkg.get("scripts", {}):
                return "npm test"
        except Exception:
            pass

    if any(f in package_files for f in ("pyproject.toml", "setup.cfg", "requirements.txt")):
        for fname in ("pyproject.toml", "setup.cfg", "requirements.txt"):
            if fname not in package_files:
                continue
            try:
                if "pytest" in (path / fname).read_text(encoding="utf-8").lower():
                    return "python -m pytest"
            except Exception:
                pass
        if any(path.rglob("test_*.py")):
            return "python -m pytest"

    if "Cargo.toml" in package_files:
        return "cargo test"

    if "go.mod" in package_files:
        return "go test ./..."

    if "Gemfile" in package_files:
        return "bundle exec rspec"

    if "pom.xml" in package_files:
        return "mvn test"

    return None


def _detect_risky_paths(path: Path) -> list[str]:
    risky: list[str] = []
    for p in _walk_repo(path):
        name = p.name.lower()
        if any(kw in name for kw in _RISKY_KEYWORDS):
            risky.append(str(p.relative_to(path)).replace("\\", "/"))
    return sorted(risky)


def _detect_changed_files(path: Path) -> list[str]:
    try:
        result = subprocess.run(
            ["git", "diff", "--name-only", "HEAD"],
            cwd=str(path),
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return [f for f in result.stdout.splitlines() if f and not f.startswith(".openshard/")]
    except Exception:
        pass
    return []


def analyze_repo(path: str | Path) -> RepoFacts:
    p = Path(path)
    package_files = _detect_package_files(p)
    return RepoFacts(
        languages=_detect_languages(p),
        package_files=package_files,
        framework=_detect_framework(p, package_files),
        test_command=_detect_test_command(p, package_files),
        risky_paths=_detect_risky_paths(p),
        changed_files=_detect_changed_files(p),
    )
