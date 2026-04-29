import json
from pathlib import Path
from unittest.mock import patch

import pytest

from click.testing import CliRunner

from openshard.evals.registry import EvalTask, load_eval_tasks, validate_eval_task
from openshard.evals.runner import run_eval_task
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


# ---------------------------------------------------------------------------
# eval report tests — no AI, reads from a JSONL fixture written by the test
# ---------------------------------------------------------------------------

def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(r) for r in records), encoding="utf-8")


def _rec(
    task_id: str = "t1",
    model: str = "m1",
    suite: str = "basic",
    passed: bool = True,
    duration: float = 1.0,
    tokens: int = 100,
    unsafe: list[str] | None = None,
) -> dict:
    return {
        "timestamp": "2026-01-01T00:00:00Z",
        "suite": suite,
        "task_id": task_id,
        "model": model,
        "passed": passed,
        "duration_seconds": duration,
        "total_tokens": tokens,
        "unsafe_files": unsafe or [],
        "error": None,
        "verification_attempted": True,
        "verification_passed": passed,
        "verification_returncode": 0 if passed else 1,
        "verification_output": "",
        "files_written": [],
        "prompt_tokens": 0,
        "completion_tokens": 0,
    }


def test_eval_report_no_file(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    runner = CliRunner()
    result = runner.invoke(cli, ["eval", "report"])
    assert result.exit_code == 0, result.output
    assert "No eval runs found" in result.output


def test_eval_report_shows_summary(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    log = tmp_path / ".openshard" / "eval-runs.jsonl"
    _write_jsonl(log, [
        _rec(task_id="t1", passed=True),
        _rec(task_id="t2", passed=True),
        _rec(task_id="t3", passed=False),
    ])
    runner = CliRunner()
    result = runner.invoke(cli, ["eval", "report"])
    assert result.exit_code == 0, result.output
    assert "Total runs" in result.output
    assert "3" in result.output
    assert "Passed" in result.output
    assert "2" in result.output
    assert "Failed" in result.output
    assert "1" in result.output
    assert "66.7" in result.output


def test_eval_report_suite_filter(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    log = tmp_path / ".openshard" / "eval-runs.jsonl"
    _write_jsonl(log, [
        _rec(task_id="t1", suite="basic", passed=True),
        _rec(task_id="t2", suite="basic", passed=True),
        _rec(task_id="t3", suite="other", passed=False),
    ])
    runner = CliRunner()

    result = runner.invoke(cli, ["eval", "report", "--suite", "basic"])
    assert result.exit_code == 0, result.output
    assert "2" in result.output
    assert "t1" in result.output
    assert "t3" not in result.output

    result2 = runner.invoke(cli, ["eval", "report", "--suite", "other"])
    assert result2.exit_code == 0, result2.output
    assert "t3" in result2.output
    assert "t1" not in result2.output


def test_eval_report_model_filter(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    log = tmp_path / ".openshard" / "eval-runs.jsonl"
    _write_jsonl(log, [
        _rec(task_id="t1", model="m1", passed=True),
        _rec(task_id="t2", model="m2", passed=False),
    ])
    runner = CliRunner()

    result = runner.invoke(cli, ["eval", "report", "--model", "m1"])
    assert result.exit_code == 0, result.output
    assert "t1" in result.output
    assert "t2" not in result.output

    result2 = runner.invoke(cli, ["eval", "report", "--model", "m2"])
    assert result2.exit_code == 0, result2.output
    assert "t2" in result2.output
    assert "t1" not in result2.output


def test_eval_report_skips_malformed_lines(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    log = tmp_path / ".openshard" / "eval-runs.jsonl"
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text(
        json.dumps(_rec(task_id="good")) + "\n"
        + "\n"
        + "not valid json at all\n",
        encoding="utf-8",
    )
    runner = CliRunner()
    result = runner.invoke(cli, ["eval", "report"])
    assert result.exit_code == 0, result.output
    assert "good" in result.output
    assert "Total runs" in result.output


def test_eval_report_no_matching_filter(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    log = tmp_path / ".openshard" / "eval-runs.jsonl"
    _write_jsonl(log, [_rec(suite="basic")])
    runner = CliRunner()
    result = runner.invoke(cli, ["eval", "report", "--suite", "nosuchsuite"])
    assert result.exit_code == 0, result.output
    assert "No records match" in result.output


# ---------------------------------------------------------------------------
# fixture copying and suite-level sanity checks
# ---------------------------------------------------------------------------

def test_run_eval_task_copies_fixture_files(tmp_path):
    task_dir = tmp_path / "my_task"
    task_dir.mkdir()
    (task_dir / "prompt.txt").write_text("Do something.")
    (task_dir / "metadata.json").write_text(
        json.dumps({
            "id": "my_task",
            "title": "My Task",
            "category": "standard",
            "expected_files": ["foo.py"],
            "verification_command": None,
        })
    )
    fixtures = task_dir / "fixtures"
    fixtures.mkdir()
    (fixtures / "seed.txt").write_text("hello fixture")

    workspace = tmp_path / "workspace"
    workspace.mkdir()

    task = EvalTask(
        id="my_task",
        title="My Task",
        category="standard",
        expected_files=["foo.py"],
        verification_command=None,
        prompt="Do something.",
        task_dir=task_dir,
    )

    fake_result = ExecutionResult(
        summary="done",
        files=[],
        notes=[],
        usage=UsageStats(prompt_tokens=0, completion_tokens=0, total_tokens=0),
    )
    with patch("openshard.evals.runner.ExecutionGenerator") as mock_cls:
        mock_cls.return_value.generate.return_value = fake_result
        run_eval_task(task, model="test-model", suite="test", workspace_root=workspace)

    assert (workspace / "seed.txt").exists(), "fixture file was not copied into workspace"
    assert (workspace / "seed.txt").read_text() == "hello fixture"


def test_basic_suite_tasks_have_verification_commands():
    tasks = load_eval_tasks("basic")
    for task in tasks:
        assert task.verification_command is not None, (
            f"task {task.id!r} has no verification_command — eval run will always fail"
        )
