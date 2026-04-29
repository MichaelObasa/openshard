from __future__ import annotations

from pathlib import Path

import pytest

from openshard.security.paths import UnsafePathError, resolve_safe_repo_path

def _symlink_ok() -> bool:
    import tempfile

    try:
        with tempfile.TemporaryDirectory() as d:
            src = Path(d) / "src.txt"
            src.touch()
            (Path(d) / "lnk").symlink_to(src)
        return True
    except OSError:
        return False


_can_symlink = pytest.mark.skipif(not _symlink_ok(), reason="symlink creation not available")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def ok(tmp_path: Path, user_path: str) -> Path:
    return resolve_safe_repo_path(tmp_path, user_path)


def bad(tmp_path: Path, user_path: str) -> None:
    with pytest.raises(UnsafePathError):
        resolve_safe_repo_path(tmp_path, user_path)


# ---------------------------------------------------------------------------
# Allowed paths
# ---------------------------------------------------------------------------

def test_allow_src_file(tmp_path):
    result = ok(tmp_path, "src/app.py")
    assert result == (tmp_path / "src" / "app.py").resolve()


def test_allow_tests_file(tmp_path):
    result = ok(tmp_path, "tests/test_app.py")
    assert result == (tmp_path / "tests" / "test_app.py").resolve()


def test_allow_docs_file(tmp_path):
    result = ok(tmp_path, "docs/readme.md")
    assert result == (tmp_path / "docs" / "readme.md").resolve()


def test_allow_nested_path(tmp_path):
    result = ok(tmp_path, "a/b/c/d.py")
    assert result == (tmp_path / "a" / "b" / "c" / "d.py").resolve()


def test_allow_top_level_file(tmp_path):
    result = ok(tmp_path, "README.md")
    assert result == (tmp_path / "README.md").resolve()


def test_allow_openshard_non_protected(tmp_path):
    # .openshard/ is not blanket-blocked; only runs.jsonl is protected
    result = ok(tmp_path, ".openshard/skills/foo/SKILL.md")
    assert result == (tmp_path / ".openshard" / "skills" / "foo" / "SKILL.md").resolve()


# ---------------------------------------------------------------------------
# Empty paths
# ---------------------------------------------------------------------------

def test_reject_empty_string(tmp_path):
    bad(tmp_path, "")


def test_reject_whitespace_only(tmp_path):
    bad(tmp_path, "   ")


# ---------------------------------------------------------------------------
# Absolute paths
# ---------------------------------------------------------------------------

def test_reject_absolute_unix(tmp_path):
    bad(tmp_path, "/etc/passwd")


def test_reject_absolute_tmp(tmp_path):
    bad(tmp_path, "/tmp/evil.py")


# ---------------------------------------------------------------------------
# Dot-dot traversal
# ---------------------------------------------------------------------------

def test_reject_dotdot_simple(tmp_path):
    bad(tmp_path, "../sibling")


def test_reject_dotdot_etc(tmp_path):
    bad(tmp_path, "../../etc/passwd")


def test_reject_dotdot_in_middle(tmp_path):
    bad(tmp_path, "src/../../../etc/passwd")


def test_reject_dotdot_only(tmp_path):
    bad(tmp_path, "..")


# ---------------------------------------------------------------------------
# Home expansion
# ---------------------------------------------------------------------------

def test_reject_tilde_ssh(tmp_path):
    bad(tmp_path, "~/.ssh/id_rsa")


def test_reject_tilde_home(tmp_path):
    bad(tmp_path, "~/foo")


def test_reject_tilde_bare(tmp_path):
    bad(tmp_path, "~")


# ---------------------------------------------------------------------------
# Control characters
# ---------------------------------------------------------------------------

def test_reject_null_byte(tmp_path):
    bad(tmp_path, "src/app\x00.py")


def test_reject_unit_separator(tmp_path):
    bad(tmp_path, "src/app\x1f.py")


def test_reject_del_character(tmp_path):
    bad(tmp_path, "src/app\x7f.py")


def test_reject_newline(tmp_path):
    bad(tmp_path, "src/app\n.py")


# ---------------------------------------------------------------------------
# Paths that escape repo root after resolve()
# ---------------------------------------------------------------------------

@_can_symlink
def test_reject_escape_via_resolve(tmp_path):
    # Construct a path that after normalization leaves the root.
    # On Windows, a path like "C:\Windows\system32" is absolute and caught
    # earlier; here we verify the resolve() containment check as a backstop
    # using a relative path that happens to point outside when joined.
    # We create an actual symlink inside tmp_path pointing outside.
    outside = tmp_path.parent / "outside_file.txt"
    outside.touch()
    link = tmp_path / "escape_link"
    link.symlink_to(outside)
    with pytest.raises(UnsafePathError):
        resolve_safe_repo_path(tmp_path, "escape_link")


# ---------------------------------------------------------------------------
# .git/ paths
# ---------------------------------------------------------------------------

def test_reject_git_config(tmp_path):
    bad(tmp_path, ".git/config")


def test_reject_git_hooks(tmp_path):
    bad(tmp_path, ".git/hooks/pre-commit")


def test_reject_git_root(tmp_path):
    bad(tmp_path, ".git")


# ---------------------------------------------------------------------------
# Protected .openshard/runs.jsonl
# ---------------------------------------------------------------------------

def test_reject_runs_jsonl(tmp_path):
    bad(tmp_path, ".openshard/runs.jsonl")


# ---------------------------------------------------------------------------
# Symlink final path rejection
# ---------------------------------------------------------------------------

@_can_symlink
def test_reject_symlink_target(tmp_path):
    real_file = tmp_path / "real.py"
    real_file.write_text("x = 1")
    link = tmp_path / "link.py"
    link.symlink_to(real_file)
    with pytest.raises(UnsafePathError):
        resolve_safe_repo_path(tmp_path, "link.py")


def test_allow_non_symlink_existing_file(tmp_path):
    real_file = tmp_path / "real.py"
    real_file.write_text("x = 1")
    result = ok(tmp_path, "real.py")
    assert result == real_file.resolve()


# ---------------------------------------------------------------------------
# Error type
# ---------------------------------------------------------------------------

def test_raises_unsafe_path_error_not_generic(tmp_path):
    with pytest.raises(UnsafePathError) as exc_info:
        resolve_safe_repo_path(tmp_path, "")
    assert isinstance(exc_info.value, Exception)


def test_error_message_is_informative(tmp_path):
    with pytest.raises(UnsafePathError, match="empty"):
        resolve_safe_repo_path(tmp_path, "")
