from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import patch

from click.testing import CliRunner

from openshard.cli.main import cli


def _invoke(args: list[str], entries: list[dict] | None, *, env: dict | None = None):
    """Run `ci check` against a temp .openshard/runs.jsonl.

    entries=None means the runs file is absent entirely.
    """
    runner = CliRunner()
    with tempfile.TemporaryDirectory() as td:
        log_dir = Path(td) / ".openshard"
        log_dir.mkdir()
        if entries is not None:
            log_path = log_dir / "runs.jsonl"
            log_path.write_text(
                "".join(json.dumps(e) + "\n" for e in entries), encoding="utf-8"
            )
        with patch("openshard.cli.main.Path.cwd", return_value=Path(td)):
            result = runner.invoke(cli, ["ci", "check"] + args, env=env or {})
    return result


_PASS = {"verification_attempted": True, "verification_passed": True}
_FAIL = {"verification_attempted": True, "verification_passed": False}
_NOT_RUN = {"verification_attempted": False}
_APPROVAL_DENIED = {"approval_receipt": {"granted": False, "reason": "review"}}
_SECRET = {
    "verification_attempted": True,
    "verification_passed": True,
    "secret_scan_result": {
        "findings": [{"fingerprint": "fp1", "kind": "aws_key", "severity": "high"}]
    },
}


# --- skip / no-run states ---------------------------------------------------

def test_no_runs_file_skips():
    result = _invoke([], entries=None)
    assert result.exit_code == 0
    assert "OpenShard CI Check: skip" in result.output


def test_empty_history_skips():
    result = _invoke([], entries=[])
    assert result.exit_code == 0
    assert "OpenShard CI Check: skip" in result.output


# --- core verdicts ----------------------------------------------------------

def test_clean_run_passes():
    result = _invoke([], entries=[_PASS])
    assert result.exit_code == 0
    assert "OpenShard CI Check: pass" in result.output


def test_verification_failed_returns_exit_1():
    result = _invoke([], entries=[_FAIL])
    assert result.exit_code == 1
    assert "OpenShard CI Check: fail" in result.output
    assert "Verification failed" in result.output


def test_approval_denied_returns_exit_1():
    result = _invoke([], entries=[_APPROVAL_DENIED])
    assert result.exit_code == 1
    assert "Manual review required" in result.output


def test_not_run_warns_exit_0():
    result = _invoke([], entries=[_NOT_RUN])
    assert result.exit_code == 0
    assert "OpenShard CI Check: warn" in result.output


def test_strict_promotes_warn_to_fail():
    result = _invoke(["--strict"], entries=[_NOT_RUN])
    assert result.exit_code == 1
    assert "OpenShard CI Check: fail" in result.output


def test_secret_findings_warn_by_default():
    result = _invoke([], entries=[_SECRET])
    assert result.exit_code == 0
    assert "OpenShard CI Check: warn" in result.output


def test_secret_findings_fail_under_strict():
    result = _invoke(["--strict"], entries=[_SECRET])
    assert result.exit_code == 1


def test_uses_latest_entry():
    result = _invoke([], entries=[_FAIL, _PASS])
    assert result.exit_code == 0
    assert "OpenShard CI Check: pass" in result.output


# --- robustness -------------------------------------------------------------

def test_old_receipt_missing_fields_does_not_crash():
    result = _invoke([], entries=[{"task": "legacy"}])
    assert result.exit_code == 0
    assert "OpenShard CI Check: warn" in result.output


def test_corrupt_jsonl_line_is_skipped():
    runner = CliRunner()
    with tempfile.TemporaryDirectory() as td:
        log_dir = Path(td) / ".openshard"
        log_dir.mkdir()
        log_path = log_dir / "runs.jsonl"
        log_path.write_text(
            "{ this is not json\n" + json.dumps(_PASS) + "\n", encoding="utf-8"
        )
        with patch("openshard.cli.main.Path.cwd", return_value=Path(td)):
            result = runner.invoke(cli, ["ci", "check"])
    assert result.exit_code == 0
    assert "OpenShard CI Check: pass" in result.output


# --- JSON output ------------------------------------------------------------

def test_json_is_valid_and_complete():
    result = _invoke(["--json"], entries=[_FAIL])
    assert result.exit_code == 1
    payload = json.loads(result.output)  # raises if not valid JSON-only
    assert payload["schema_version"] == "1"
    assert payload["command"] == "ci check"
    assert payload["status"] == "fail"
    assert payload["exit_code"] == 1
    assert payload["checks"]["verification"] == "failed"
    assert isinstance(payload["reasons"], list)
    assert isinstance(payload["warnings"], list)


def test_json_skip_has_null_shard_id():
    result = _invoke(["--json"], entries=None)
    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["status"] == "skip"
    assert payload["shard_id"] is None
    assert payload["exit_code"] == 0


def test_json_pass_includes_shard_id():
    result = _invoke(["--json"], entries=[_PASS])
    payload = json.loads(result.output)
    assert payload["status"] == "pass"
    assert payload["shard_id"]  # non-empty


# --- github output ----------------------------------------------------------

def test_github_output_written_when_env_set():
    runner = CliRunner()
    with tempfile.TemporaryDirectory() as td:
        log_dir = Path(td) / ".openshard"
        log_dir.mkdir()
        (log_dir / "runs.jsonl").write_text(json.dumps(_FAIL) + "\n", encoding="utf-8")
        go_path = Path(td) / "gh_output.txt"
        go_path.write_text("", encoding="utf-8")
        with patch("openshard.cli.main.Path.cwd", return_value=Path(td)):
            result = runner.invoke(
                cli,
                ["ci", "check", "--github-output"],
                env={"GITHUB_OUTPUT": str(go_path)},
            )
        written = go_path.read_text(encoding="utf-8")
    assert result.exit_code == 1
    assert "openshard_ci_status=fail" in written
    assert "openshard_ci_exit_code=1" in written
    assert "openshard_ci_reasons_count=" in written


def test_github_output_missing_env_does_not_crash():
    # Empty string == treated as unset by _write_github_output; explicit so the
    # CI runner's own $GITHUB_OUTPUT cannot leak into this "missing env" case.
    result = _invoke(["--github-output"], entries=[_PASS], env={"GITHUB_OUTPUT": ""})
    assert result.exit_code == 0
    assert "GITHUB_OUTPUT is not set" in result.output


def test_json_github_output_reports_both_booleans():
    runner = CliRunner()
    with tempfile.TemporaryDirectory() as td:
        log_dir = Path(td) / ".openshard"
        log_dir.mkdir()
        (log_dir / "runs.jsonl").write_text(json.dumps(_PASS) + "\n", encoding="utf-8")
        go_path = Path(td) / "gh_output.txt"
        go_path.write_text("", encoding="utf-8")
        with patch("openshard.cli.main.Path.cwd", return_value=Path(td)):
            result = runner.invoke(
                cli,
                ["ci", "check", "--json", "--github-output"],
                env={"GITHUB_OUTPUT": str(go_path)},
            )
    payload = json.loads(result.output)
    assert payload["github_output_available"] is True
    assert payload["github_output_written"] is True


def test_json_github_output_missing_env_booleans_false():
    # Empty string == treated as unset; explicit so CI's $GITHUB_OUTPUT cannot leak in.
    result = _invoke(
        ["--json", "--github-output"], entries=[_PASS], env={"GITHUB_OUTPUT": ""}
    )
    payload = json.loads(result.output)
    assert payload["github_output_available"] is False
    assert payload["github_output_written"] is False


# --- leakage guards ---------------------------------------------------------

def test_no_secret_value_or_abs_path_leak():
    entry = {
        "verification_attempted": True,
        "verification_passed": True,
        "secret_scan_result": {
            "findings": [
                {
                    "fingerprint": "fp1",
                    "kind": "aws_key",
                    "severity": "high",
                    "path": "/home/user/secret.env",
                    "raw": "AKIAEXAMPLESECRETVALUE",
                }
            ]
        },
    }
    result = _invoke(["--json"], entries=[entry])
    assert "AKIAEXAMPLESECRETVALUE" not in result.output
    assert "/home/user/secret.env" not in result.output
