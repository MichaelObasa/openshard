import json
from pathlib import Path

import pytest

from click.testing import CliRunner

from openshard.evals.registry import load_eval_tasks, validate_eval_task
from openshard.cli.main import cli


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
