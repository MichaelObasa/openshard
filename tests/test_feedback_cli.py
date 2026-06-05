from __future__ import annotations

import datetime
import json
import tempfile
from pathlib import Path
from unittest.mock import patch

from click.testing import CliRunner

from openshard.cli.main import cli

_ONE_RUN = json.dumps({"timestamp": "2026-01-01T00:00:00", "task": "do X", "summary": "done"})


def _invoke(args: list[str], runs_content: str | None = _ONE_RUN) -> object:
    runner = CliRunner()
    with tempfile.TemporaryDirectory() as td:
        log_dir = Path(td) / ".openshard"
        log_dir.mkdir()
        log_path = log_dir / "runs.jsonl"
        if runs_content is not None:
            log_path.write_text(runs_content, encoding="utf-8")
        with patch("openshard.cli.main.Path.cwd", return_value=Path(td)):
            result = runner.invoke(cli, ["feedback"] + args)
    return result


def _invoke_and_read(args: list[str]) -> tuple[object, dict]:
    runner = CliRunner()
    with tempfile.TemporaryDirectory() as td:
        log_dir = Path(td) / ".openshard"
        log_dir.mkdir()
        log_path = log_dir / "runs.jsonl"
        log_path.write_text(_ONE_RUN, encoding="utf-8")
        with patch("openshard.cli.main.Path.cwd", return_value=Path(td)):
            result = runner.invoke(cli, ["feedback"] + args)
        entries = [json.loads(ln) for ln in log_path.read_text().splitlines() if ln.strip()]
    return result, entries[-1]


def test_feedback_writes_developer_feedback_to_entry():
    _, entry = _invoke_and_read(["--outcome", "accepted"])
    assert "developer_feedback" in entry
    assert entry["developer_feedback"]["outcome"] == "accepted"


def test_feedback_outcome_required():
    result = _invoke([])
    assert result.exit_code != 0


def test_feedback_invalid_outcome_rejected():
    result = _invoke(["--outcome", "loved-it"])
    assert result.exit_code != 0


def test_feedback_reason_is_free_text():
    long_reason = "missed the IAM policy check on the S3 bucket in us-east-2"
    _, entry = _invoke_and_read(["--outcome", "partial", "--reason", long_reason])
    assert entry["developer_feedback"]["reason"] == long_reason


def test_feedback_boolean_flags():
    _, entry = _invoke_and_read(["--outcome", "accepted", "--edited", "--ci-passed"])
    df = entry["developer_feedback"]
    assert df["edited"] is True
    assert df["ci_passed"] is True
    assert df["ci_failed"] is False


def test_feedback_no_runs_file_exits_nonzero():
    result = _invoke(["--outcome", "accepted"], runs_content=None)
    assert result.exit_code != 0


def test_feedback_empty_runs_file_exits_nonzero():
    result = _invoke(["--outcome", "accepted"], runs_content="")
    assert result.exit_code != 0


def test_feedback_schema_version_is_1():
    _, entry = _invoke_and_read(["--outcome", "rejected"])
    assert entry["developer_feedback"]["schema_version"] == 1


def test_feedback_source_is_cli():
    _, entry = _invoke_and_read(["--outcome", "retried"])
    assert entry["developer_feedback"]["source"] == "cli"


def test_feedback_recorded_at_is_iso8601():
    _, entry = _invoke_and_read(["--outcome", "abandoned"])
    recorded_at = entry["developer_feedback"]["recorded_at"]
    datetime.datetime.fromisoformat(recorded_at)  # raises if not valid
