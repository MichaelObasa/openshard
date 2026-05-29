"""Secret scan recording for OpenShard — v1.

Scans files already read as context, after the run completes and before the
Shard entry is logged. Records findings as safe, redacted EvidenceCapsules.
Does NOT prevent secrets from reaching the model in this version.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path

_MAX_FILE_BYTES = 512 * 1024  # 512 KB

# Directories skipped regardless of depth (mirrors shard_contract._NOISY_EVIDENCE_ANY_SEGMENT)
_SKIP_ANY_SEGMENT: frozenset[str] = frozenset({
    "__pycache__", ".pytest_cache", ".venv", "venv", "node_modules",
    ".mypy_cache", ".ruff_cache", ".tox", ".next", ".git",
})

# Directories skipped only when they are the root path component
_SKIP_ROOT_SEGMENT: frozenset[str] = frozenset({
    "dist", "build", "coverage", "cache", ".cache", "tmp", "temp",
})

# Placeholder values — suppress match if the captured secret contains any of these
_PLACEHOLDER_SUBSTRINGS = (
    "your-api-key", "changeme", "example", "dummy", "placeholder",
    "<token>", "xxxxx",
)
_PLACEHOLDER_RE = re.compile(
    r"(\$\{[^}]+\}|var\.[a-z_][a-z0-9_]*)",
    re.IGNORECASE,
)

# Secret patterns — each entry: (kind, compiled_pattern, value_group_index)
# value_group_index is the group that contains the raw secret value for
# fingerprinting and redaction; 0 means the whole match.
_PATTERNS: list[tuple[str, re.Pattern[str], int]] = [
    (
        "anthropic_key",
        re.compile(r"sk-ant-[A-Za-z0-9_\-]{20,}"),
        0,
    ),
    (
        "openai_key",
        re.compile(r"sk-(?!ant-)[A-Za-z0-9_\-]{20,}"),
        0,
    ),
    (
        "aws_access_key_id",
        re.compile(r"AKIA[0-9A-Z]{16}"),
        0,
    ),
    (
        "github_pat",
        re.compile(r"github_pat_[A-Za-z0-9_]{22,}"),
        0,
    ),
    (
        "github_token",
        re.compile(r"ghp_[A-Za-z0-9_]{20,}"),
        0,
    ),
    (
        "generic_secret_assignment",
        re.compile(
            r"(?i)(api_key|apikey|secret|token|password|private_key)"
            r"\s*[:=]\s*[\"']?([^\"'\s]{12,})"
        ),
        2,  # group 2 is the value
    ),
]


@dataclass
class SecretFinding:
    kind: str
    path: str | None
    line: int | None
    redacted: str
    severity: str
    fingerprint: str  # sha256[:12] of raw match — stable dedup key


@dataclass
class SecretScanResult:
    scanned_files_count: int
    findings: list[SecretFinding] = field(default_factory=list)
    blocked: bool = False
    summary: str = ""


def _is_noisy_path(path_str: str) -> bool:
    parts = path_str.replace("\\", "/").split("/")
    if not parts:
        return False
    if any(part in _SKIP_ANY_SEGMENT for part in parts):
        return True
    return parts[0] in _SKIP_ROOT_SEGMENT


def _fingerprint(raw: str) -> str:
    return hashlib.sha256(raw.encode("utf-8", errors="replace")).hexdigest()[:12]


def _redact(raw: str) -> str:
    if len(raw) >= 10:
        return f"{raw[:4]}...{raw[-4:]}"
    return "***"


def _is_placeholder(value: str) -> bool:
    lower = value.lower()
    if any(sub in lower for sub in _PLACEHOLDER_SUBSTRINGS):
        return True
    if _PLACEHOLDER_RE.search(value):
        return True
    # short values are likely templates, not real secrets
    if len(value) < 12:
        return True
    return False


def _scan_line(
    line: str,
    lineno: int,
    path_str: str | None,
    seen_fingerprints: set[str],
    findings: list[SecretFinding],
) -> None:
    for kind, pattern, value_group in _PATTERNS:
        for m in pattern.finditer(line):
            raw_value = m.group(value_group) if value_group else m.group(0)
            if _is_placeholder(raw_value):
                continue
            fp = _fingerprint(raw_value)
            if fp in seen_fingerprints:
                continue
            seen_fingerprints.add(fp)
            severity = "Critical" if kind in ("aws_access_key_id",) else "High"
            findings.append(SecretFinding(
                kind=kind,
                path=path_str,
                line=lineno,
                redacted=_redact(raw_value),
                severity=severity,
                fingerprint=fp,
            ))


def scan_paths_for_secrets(
    paths: list[Path],
    root: Path | None = None,
) -> SecretScanResult:
    """Scan a bounded list of paths for obvious secret-like values.

    Skips noisy dirs, binary files, and files above the size limit.
    Never raises — returns partial results on error.
    """
    seen_fingerprints: set[str] = set()
    findings: list[SecretFinding] = []
    scanned = 0

    for path in paths:
        path_str = str(path)
        # Normalise to repo-relative for display if root is provided
        display_path: str | None
        try:
            display_path = str(path.relative_to(root)) if root else path_str
        except ValueError:
            display_path = path_str

        # Skip noisy paths
        check_str = display_path or path_str
        if _is_noisy_path(check_str):
            continue

        # Skip missing
        try:
            if not path.is_file():
                continue
        except OSError:
            continue

        # Skip oversized files
        try:
            if path.stat().st_size > _MAX_FILE_BYTES:
                continue
        except OSError:
            continue

        # Read as text; skip binary/unreadable
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue

        scanned += 1
        for lineno, line in enumerate(text.splitlines(), start=1):
            _scan_line(line, lineno, display_path, seen_fingerprints, findings)

    findings.sort(key=lambda f: (f.path or "", f.line or 0, f.kind))

    count = len(findings)
    summary = (
        f"{count} potential secret{'s' if count != 1 else ''} detected in {scanned} scanned file{'s' if scanned != 1 else ''}"
        if count
        else f"No secrets detected in {scanned} scanned file{'s' if scanned != 1 else ''}"
    )

    return SecretScanResult(
        scanned_files_count=scanned,
        findings=findings,
        blocked=False,
        summary=summary,
    )
