from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import patch

from click.testing import CliRunner

from openshard.cli.main import cli


def _invoke(args: list[str], entries: list[dict] | None):
    """Run `stats failures` against a temp .openshard/runs.jsonl.

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
            result = runner.invoke(cli, ["stats", "failures"] + args)
    return result


_FAIL = {"verification_attempted": True, "verification_passed": False}
_PASS = {"verification_attempted": True, "verification_passed": True}


# --- no-run states ----------------------------------------------------------

def test_missing_runs_file_human():
    result = _invoke([], entries=None)
    assert result.exit_code == 0
    assert "No run history found" in result.output


def test_missing_runs_file_json_not_found():
    result = _invoke(["--json"], entries=None)
    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["status"] == "not_found"
    assert payload["command"] == "stats failures"
    assert payload["runs_checked"] == 0
    assert payload["failures"] == []


def test_empty_file_json_not_found():
    result = _invoke(["--json"], entries=[])
    payload = json.loads(result.output)
    assert payload["status"] == "not_found"


# --- core output ------------------------------------------------------------

def test_human_output_lists_top_categories():
    result = _invoke([], entries=[_FAIL, _PASS])
    assert result.exit_code == 0
    assert "Failure taxonomy" in result.output
    assert "verification_failed" in result.output
    assert "runs checked: 2" in result.output


def test_json_is_valid_and_has_envelope():
    result = _invoke(["--json"], entries=[_FAIL, _PASS])
    payload = json.loads(result.output)  # must be valid JSON only
    assert payload["schema_version"] == "1"
    assert payload["command"] == "stats failures"
    assert payload["status"] == "ok"
    assert payload["runs_checked"] == 2
    assert payload["category_counts"]["verification_failed"] == 1
    assert payload["category_counts"]["no_failure_detected"] == 1
    assert payload["top_categories"][0]["category"] == "verification_failed"
    assert len(payload["failures"]) == 1
    assert payload["failures"][0]["category"] == "verification_failed"
    assert payload["failures"][0]["confidence"] == "high"


def test_limit_is_honored():
    entries = [_FAIL, _FAIL, _PASS]
    result = _invoke(["--json", "--limit", "1"], entries=entries)
    payload = json.loads(result.output)
    assert payload["runs_checked"] == 1
    # Only the most recent (a clean pass) is considered.
    assert payload["category_counts"]["no_failure_detected"] == 1
    assert payload["failures"] == []


def test_corrupt_lines_are_skipped():
    runner = CliRunner()
    with tempfile.TemporaryDirectory() as td:
        log_dir = Path(td) / ".openshard"
        log_dir.mkdir()
        log_path = log_dir / "runs.jsonl"
        log_path.write_text(
            json.dumps(_FAIL) + "\n{ not json }\n" + json.dumps(_PASS) + "\n",
            encoding="utf-8",
        )
        with patch("openshard.cli.main.Path.cwd", return_value=Path(td)):
            result = runner.invoke(cli, ["stats", "failures", "--json"])
    payload = json.loads(result.output)
    assert payload["runs_checked"] == 2


# --- no-leak guard ----------------------------------------------------------

def test_no_secret_value_or_abs_path_leak():
    entry = {
        "verification_attempted": True,
        "verification_passed": True,
        "error_class": "/home/user/secret.env",
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
