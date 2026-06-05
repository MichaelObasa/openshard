from __future__ import annotations

from openshard.analysis.repo import analyze_repo
from openshard.tui.plan_mode import (
    _detect_extra_stack,
    _gather_relevant_files,
    answer_plan_mode,
)

# --- Basic safety guarantees ---


def test_empty_task_with_path_returns_usage_hint(tmp_path):
    result = answer_plan_mode("", path=tmp_path)
    assert "/plan" in result
    assert not result.startswith("PLAN\n")


def test_repo_aware_says_no_files_changed(tmp_path):
    (tmp_path / "pyproject.toml").write_text("[project]\nname='x'\n")
    result = answer_plan_mode("refactor auth", path=tmp_path)
    assert "No files changed" in result


def test_repo_aware_says_no_provider_call(tmp_path):
    (tmp_path / "pyproject.toml").write_text("[project]\nname='x'\n")
    result = answer_plan_mode("refactor auth", path=tmp_path)
    assert "No provider call" in result


def test_plan_mode_does_not_write_files(tmp_path):
    before = set(p.name for p in tmp_path.iterdir())
    answer_plan_mode("refactor auth", path=tmp_path)
    after = set(p.name for p in tmp_path.iterdir())
    assert before == after


# --- Stack detection ---


def test_detects_python_from_pyproject_toml(tmp_path):
    (tmp_path / "pyproject.toml").write_text("[project]\nname='x'\n")
    (tmp_path / "main.py").write_text("print('hello')\n")
    result = answer_plan_mode("refactor auth", path=tmp_path)
    assert "python" in result.lower()


def test_detects_terraform_from_tf_file(tmp_path):
    (tmp_path / "main.tf").write_text('provider "aws" {}\n')
    result = answer_plan_mode("harden Terraform repo", path=tmp_path)
    assert "terraform" in result.lower()


def test_detects_github_actions(tmp_path):
    wf_dir = tmp_path / ".github" / "workflows"
    wf_dir.mkdir(parents=True)
    (wf_dir / "ci.yml").write_text("on: push\n")
    result = answer_plan_mode("improve CI", path=tmp_path)
    assert "github" in result.lower() or "ci.yml" in result


# --- Relevant files ---


def test_lists_package_file_in_output(tmp_path):
    (tmp_path / "pyproject.toml").write_text("[project]\n")
    result = answer_plan_mode("refactor", path=tmp_path)
    assert "pyproject.toml" in result


def test_skips_noisy_dirs(tmp_path):
    noisy = tmp_path / "node_modules" / "some-lib"
    noisy.mkdir(parents=True)
    (noisy / "index.js").write_text("module.exports = {}\n")
    result = answer_plan_mode("refactor", path=tmp_path)
    assert "node_modules" not in result


def test_relevant_files_bounded(tmp_path):
    for i in range(20):
        (tmp_path / f"module_{i}.py").write_text(f"x = {i}\n")
    facts = analyze_repo(tmp_path)
    extra = _detect_extra_stack(tmp_path)
    files = _gather_relevant_files(tmp_path, facts, extra)
    assert len(files) <= 10


# --- Checks and risk ---


def test_includes_suggested_checks_for_python(tmp_path):
    (tmp_path / "pyproject.toml").write_text("[tool.pytest.ini_options]\n")
    result = answer_plan_mode("refactor auth", path=tmp_path)
    assert "pytest" in result or "Checks" in result


def test_risk_level_appears_in_output(tmp_path):
    (tmp_path / "pyproject.toml").write_text("[project]\n")
    result = answer_plan_mode("refactor auth", path=tmp_path)
    assert "Risk level:" in result


def test_terraform_check_suggested_for_tf_file(tmp_path):
    (tmp_path / "main.tf").write_text('provider "aws" {}\n')
    result = answer_plan_mode("harden Terraform", path=tmp_path)
    assert "terraform validate" in result


# --- Fallback behaviour ---


def test_fallback_on_nonexistent_path():
    result = answer_plan_mode("refactor auth", path="/nonexistent/path/xyz_not_real_abc")
    assert "PLAN" in result
    assert "refactor auth" in result


def test_local_fallback_when_no_path():
    result = answer_plan_mode("refactor auth")
    assert "No repo scan" in result
    assert "no provider call" in result
    assert "no files changed" in result
