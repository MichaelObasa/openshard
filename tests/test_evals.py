import json
from pathlib import Path
from unittest.mock import patch

import pytest

from click.testing import CliRunner

from openshard.evals.registry import load_eval_tasks, validate_eval_task
from openshard.cli.main import cli
from openshard.execution.generator import ChangedFile, ExecutionResult
from openshard.providers.base import UsageStats


def test_loads_bundled_basic_evals():
    tasks = load_eval_tasks("basic")
    assert len(tasks) == 3
    ids = {t.id for t in tasks}
    assert ids == {"docs_update", "helper_function", "cli_flag"}


def test_validates_required_metadata_fields(tmp_path):
    task_dir = tmp_path / "sample_task"
    task_dir.mkdir()
    (task_dir / "prompt.txt").write_text("Do something.")
    (task_dir / "metadata.json").write_text(
        json.dumps(
            {
                "id": "sample_task",
                "title": "Sample Task",
                "category": "standard",
                "expected_files": ["some_file.py"],
                "verification_command": None,
            }
        )
    )
    task = validate_eval_task(task_dir)
    assert task.id == "sample_task"
    assert task.title == "Sample Task"
    assert task.category == "standard"
    assert task.expected_files == ["some_file.py"]
    assert task.verification_command is None
    assert task.prompt == "Do something."


def test_rejects_missing_prompt_txt(tmp_path):
    task_dir = tmp_path / "no_prompt"
    task_dir.mkdir()
    (task_dir / "metadata.json").write_text(
        json.dumps(
            {
                "id": "no_prompt",
                "title": "No Prompt",
                "category": "standard",
                "expected_files": [],
                "verification_command": None,
            }
        )
    )
    with pytest.raises(FileNotFoundError, match="prompt.txt"):
        validate_eval_task(task_dir)


def test_rejects_invalid_metadata_json(tmp_path):
    task_dir = tmp_path / "bad_meta"
    task_dir.mkdir()
    (task_dir / "prompt.txt").write_text("Do something.")
    (task_dir / "metadata.json").write_text("{ not valid json }")
    with pytest.raises(ValueError, match="Invalid metadata.json"):
        validate_eval_task(task_dir)


def test_rejects_missing_metadata_fields(tmp_path):
    task_dir = tmp_path / "incomplete_meta"
    task_dir.mkdir()
    (task_dir / "prompt.txt").write_text("Do something.")
    (task_dir / "metadata.json").write_text(json.dumps({"id": "incomplete_meta"}))
    with pytest.raises(ValueError, match="missing fields"):
        validate_eval_task(task_dir)


def test_eval_list_output_includes_fixture_ids():
    runner = CliRunner()
    result = runner.invoke(cli, ["eval", "list"])
    assert result.exit_code == 0, result.output
    assert "docs_update" in result.output
    assert "helper_function" in result.output
    assert "cli_flag" in result.output


def test_eval_list_with_suite_option():
    runner = CliRunner()
    result = runner.invoke(cli, ["eval", "list", "--suite", "basic"])
    assert result.exit_code == 0, result.output
    assert "docs_update" in result.output
    assert "helper_function" in result.output
    assert "cli_flag" in result.output


def test_eval_list_missing_suite_fails_cleanly():
    runner = CliRunner()
    result = runner.invoke(cli, ["eval", "list", "--suite", "nonexistent"])
    assert result.exit_code != 0
    assert "nonexistent" in result.output or "Error" in result.output


def test_eval_validate_succeeds_on_bundled_fixtures():
    runner = CliRunner()
    result = runner.invoke(cli, ["eval", "validate"])
    assert result.exit_code == 0, result.output
    assert "OK" in result.output
    assert "3" in result.output


def test_eval_validate_with_suite_option():
    runner = CliRunner()
    result = runner.invoke(cli, ["eval", "validate", "--suite", "basic"])
    assert result.exit_code == 0, result.output
    assert "OK" in result.output
    assert "3" in result.output
    assert "basic" in result.output


def test_eval_validate_missing_suite_fails_cleanly():
    runner = CliRunner()
    result = runner.invoke(cli, ["eval", "validate", "--suite", "nonexistent"])
    assert result.exit_code != 0
    assert "nonexistent" in result.output or "Error" in result.output


def test_readme_mentions_eval_commands():
    readme = (Path(__file__).parent.parent / "README.md").read_text(encoding="utf-8")
    assert "eval list" in readme
    assert "eval validate" in readme


# ---------------------------------------------------------------------------
# eval run tests — all mock ExecutionGenerator.generate (no real AI calls)
# ---------------------------------------------------------------------------

def _fake_result(files: list[ChangedFile] | None = None) -> ExecutionResult:
    return ExecutionResult(
        summary="done",
        files=files or [],
        notes=[],
        usage=UsageStats(prompt_tokens=10, completion_tokens=5, total_tokens=15),
    )


def test_eval_run_help_exits_zero():
    runner = CliRunner()
    result = runner.invoke(cli, ["eval", "run", "--help"])
    assert result.exit_code == 0
    assert "--suite" in result.output
    assert "--model" in result.output


def test_eval_run_missing_suite_fails_cleanly():
    runner = CliRunner()
    with patch("openshard.execution.generator.ExecutionGenerator.generate", return_value=_fake_result()):
        result = runner.invoke(cli, ["eval", "run", "--suite", "nonexistent"])
    assert result.exit_code != 0
    assert "nonexistent" in result.output or "Error" in result.output


def test_eval_run_writes_result_record(tmp_path, monkeypatch):
    log_path = tmp_path / ".openshard" / "eval-runs.jsonl"
    monkeypatch.chdir(tmp_path)

    with (
        patch("openshard.execution.generator.ExecutionGenerator.generate", return_value=_fake_result()),
        patch("openshard.run.pipeline._copy_cwd_to_workspace"),
        patch("openshard.verification.executor.run_verification_plan", return_value=(0, "ok")),
    ):
        runner = CliRunner()
        runner.invoke(cli, ["eval", "run", "--suite", "basic"])

    assert log_path.exists(), "eval-runs.jsonl was not created"
    lines = log_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 3  # one per task in basic suite

    required_keys = {
        "timestamp", "suite", "task_id", "model", "passed",
        "duration_seconds", "verification_attempted", "verification_passed",
        "verification_returncode", "verification_output",
        "files_written", "unsafe_files",
        "prompt_tokens", "completion_tokens", "total_tokens", "error",
    }
    for line in lines:
        record = json.loads(line)
        assert required_keys.issubset(record.keys()), f"missing keys in: {record}"
        assert record["suite"] == "basic"


def test_eval_run_prints_pass_fail(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    with (
        patch("openshard.execution.generator.ExecutionGenerator.generate", return_value=_fake_result()),
        patch("openshard.run.pipeline._copy_cwd_to_workspace"),
        patch("openshard.verification.executor.run_verification_plan", return_value=(0, "ok")),
    ):
        runner = CliRunner()
        result = runner.invoke(cli, ["eval", "run", "--suite", "basic"])

    task_ids = {"docs_update", "helper_function", "cli_flag"}
    for task_id in task_ids:
        assert task_id in result.output, f"{task_id} not in output"
    assert "PASS" in result.output or "FAIL" in result.output


def test_generated_unsafe_path_is_not_written(tmp_path, monkeypatch):
    unsafe_file = ChangedFile(path="../../etc/passwd", change_type="create", content="pwned", summary="unsafe")
    fake = _fake_result(files=[unsafe_file])

    monkeypatch.chdir(tmp_path)

    with (
        patch("openshard.execution.generator.ExecutionGenerator.generate", return_value=fake),
        patch("openshard.run.pipeline._copy_cwd_to_workspace"),
        patch("openshard.verification.executor.run_verification_plan", return_value=(0, "ok")),
    ):
        runner = CliRunner()
        runner.invoke(cli, ["eval", "run", "--suite", "basic"])

    log_path = tmp_path / ".openshard" / "eval-runs.jsonl"
    assert log_path.exists()
    for line in log_path.read_text(encoding="utf-8").strip().splitlines():
        record = json.loads(line)
        assert "../../etc/passwd" not in record["files_written"]
        assert "../../etc/passwd" in record["unsafe_files"] or record["files_written"] == []

    passwd_path = tmp_path / "etc" / "passwd"
    assert not passwd_path.exists(), "unsafe path was written to disk"
