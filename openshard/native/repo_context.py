from __future__ import annotations

from dataclasses import dataclass, field

_PACKAGE_FILES = frozenset({
    "pyproject.toml",
    "package.json",
    "pnpm-lock.yaml",
    "yarn.lock",
    "package-lock.json",
    "requirements.txt",
    "go.mod",
    "Cargo.toml",
    "composer.json",
    "pom.xml",
    "build.gradle",
})

_TEST_PATTERNS = ("tests/", "test_", "_test.", ".spec.", ".test.")

_EXT_TO_STACK: dict[str, str] = {
    ".py": "python",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".js": "javascript",
    ".jsx": "javascript",
    ".go": "go",
    ".rs": "rust",
    ".java": "java",
}

_PKG_TO_STACK: dict[str, str] = {
    "package.json": "node",
    "pyproject.toml": "python",
    "requirements.txt": "python",
    "Cargo.toml": "rust",
    "go.mod": "go",
}


@dataclass
class NativeRepoContextSummary:
    total_files: int = 0
    top_level_dirs: list[str] = field(default_factory=list)
    package_files: list[str] = field(default_factory=list)
    test_markers: list[str] = field(default_factory=list)
    likely_stack_markers: list[str] = field(default_factory=list)
    truncated: bool = False


def build_repo_context_summary(
    file_list_output: str, *, max_items: int = 20
) -> NativeRepoContextSummary:
    paths = [p for p in file_list_output.splitlines() if p.strip()]

    top_dirs: set[str] = set()
    pkg_files: set[str] = set()
    test_markers: set[str] = set()
    stack_markers: set[str] = set()

    for path in paths:
        parts = path.replace("\\", "/").split("/")
        basename = parts[-1]

        if len(parts) > 1:
            top_dirs.add(parts[0])

        if basename in _PACKAGE_FILES:
            pkg_files.add(basename)

        for pattern in _TEST_PATTERNS:
            if pattern in path.replace("\\", "/"):
                test_markers.add(path)
                break

        ext = "." + basename.rsplit(".", 1)[-1] if "." in basename else ""
        if ext in _EXT_TO_STACK:
            stack_markers.add(_EXT_TO_STACK[ext])
        if basename in _PKG_TO_STACK:
            stack_markers.add(_PKG_TO_STACK[basename])

    truncated = False

    def _compact(s: set[str]) -> list[str]:
        nonlocal truncated
        lst = sorted(s)
        if len(lst) > max_items:
            truncated = True
            return lst[:max_items]
        return lst

    return NativeRepoContextSummary(
        total_files=len(paths),
        top_level_dirs=_compact(top_dirs),
        package_files=_compact(pkg_files),
        test_markers=_compact(test_markers),
        likely_stack_markers=_compact(stack_markers),
        truncated=truncated,
    )
