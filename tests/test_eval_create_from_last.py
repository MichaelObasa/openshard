from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import patch

from click.testing import CliRunner

from openshard.cli.main import cli
from openshard.evals.case_builder import (
    _safe_eval_id,
    _safe_task_text,
    build_eval_case,
    is_eligible,
)
from openshard.history.failures import FailureClassification, classify_failure
from openshard.history.shard_contract import build_shard_receipt


# --- CLI harness ------------------------------------------------------------

def _invoke(args: list[str], entries: list[dict] | None, *, prewrite=None):
    """Run `eval create-from-last` against a temp .openshard/.

    entries=None means the runs file is absent entirely. Returns (result, files)
    where files maps each "*.json" path (relative to the temp root, forward
    slashes) to its text content, captured before the temp dir is cleaned up.
    """
    runner = CliRunner()
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        (root / ".openshard").mkdir()
        if entries is not None:
            (root / ".openshard" / "runs.jsonl").write_text(
                "".join(json.dumps(e) + "\n" for e in entries), encoding="utf-8"
            )
        for rel, content in (prewrite or []):
            p = root / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(content, encoding="utf-8")
        with patch("openshard.cli.main.Path.cwd", return_value=root):
            result = runner.invoke(cli, ["eval", "create-from-last"] + args)
        files = {
            str(p.relative_to(root)).replace("\\", "/"): p.read_text(encoding="utf-8")
            for p in root.rglob("*.json")
        }
    return result, files


# A verification-failed run (eligible) and a clean accepted run (not eligible).
_FAIL = {
    "timestamp": "2026-04-13T06:24:08Z",
    "task": "fix the parser",
    "verification_attempted": True,
    "verification_passed": False,
}
_CLEAN = {
    "timestamp": "2026-04-13T06:24:08Z",
    "task": "add docs",
    "verification_attempted": True,
    "verification_passed": True,
}

# Deterministic id for a single-entry history dated 2026-04-13 (index 0).
_FAIL_EVAL_ID = "eval-shard-20260413-0001"
_FAIL_GENERATED = f".openshard/evals/generated/{_FAIL_EVAL_ID}.json"


def _generated(files: dict) -> dict:
    """Return the single generated eval case JSON, parsed."""
    matches = [v for k, v in files.items() if k.startswith(".openshard/evals/generated/")]
    assert len(matches) == 1, f"expected one generated case, got {list(files)}"
    return json.loads(matches[0])


# --- creation ---------------------------------------------------------------

def test_creates_eval_case_from_failed_run():
    result, files = _invoke([], entries=[_FAIL])
    assert result.exit_code == 0
    assert "Created eval case" in result.output
    case = _generated(files)
    assert case["schema_version"] == "1"
    assert case["source"] == "failed_shard_v1"
    assert case["eval_id"] == _FAIL_EVAL_ID
    assert case["source_shard_id"] == "shard-20260413-0001"
    assert case["failure_category"] == "verification_failed"
    assert case["failure_confidence"] == "high"
    assert case["signals"]["verification"] == "failed"
    assert case["expected_outcome"] == {
        "verification_should_pass": True,
        "manual_review_required": False,
    }
    assert case["constraints"]["redacted"] is True


def test_human_output_reports_key_fields():
    result, _ = _invoke([], entries=[_FAIL])
    assert "verification_failed" in result.output
    assert "shard-20260413-0001" in result.output
    assert _FAIL_EVAL_ID in result.output


# --- not eligible / no history ---------------------------------------------

def test_clean_run_is_no_op():
    result, files = _invoke([], entries=[_CLEAN])
    assert result.exit_code == 0
    assert "no failure/correction signal" in result.output
    assert files == {}  # nothing written


def test_clean_run_json_status_not_eligible():
    result, _ = _invoke(["--json"], entries=[_CLEAN])
    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["status"] == "not_eligible"
    assert payload["failure_category"] == "no_failure_detected"
    assert payload["eval_id"] is None


def test_no_history_is_no_op():
    result, files = _invoke([], entries=None)
    assert result.exit_code == 0
    assert "no run history" in result.output
    assert files == {}


def test_no_history_json_is_valid():
    result, _ = _invoke(["--json"], entries=None)
    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["status"] == "not_eligible"
    assert payload["warnings"]


def test_empty_history_is_no_op():
    result, files = _invoke([], entries=[])
    assert result.exit_code == 0
    assert files == {}


# --- robustness -------------------------------------------------------------

def test_corrupt_jsonl_line_is_skipped():
    runner = CliRunner()
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        (root / ".openshard").mkdir()
        (root / ".openshard" / "runs.jsonl").write_text(
            "{ not valid json\n" + json.dumps(_FAIL) + "\n", encoding="utf-8"
        )
        with patch("openshard.cli.main.Path.cwd", return_value=root):
            result = runner.invoke(cli, ["eval", "create-from-last"])
    assert result.exit_code == 0
    assert "Created eval case" in result.output


def test_old_minimal_receipt_does_not_crash():
    # Missing every newer field, but retried -> unknown_failure (eligible).
    minimal = {"task": "do a thing", "developer_feedback": {"outcome": "retried"}}
    result, files = _invoke([], entries=[minimal])
    assert result.exit_code == 0
    case = _generated(files)
    assert case["failure_category"] == "unknown_failure"


# --- overwrite protection ---------------------------------------------------

def test_existing_file_without_force_fails():
    result, _ = _invoke(
        [], entries=[_FAIL], prewrite=[(_FAIL_GENERATED, "{}\n")]
    )
    assert result.exit_code == 1
    assert "already exists" in result.output


def test_existing_file_json_error_status():
    result, _ = _invoke(
        ["--json"], entries=[_FAIL], prewrite=[(_FAIL_GENERATED, "{}\n")]
    )
    assert result.exit_code == 1
    payload = json.loads(result.output)
    assert payload["status"] == "error"
    assert any("already exists" in w for w in payload["warnings"])


def test_force_overwrites_existing_file():
    result, files = _invoke(
        ["--force"], entries=[_FAIL], prewrite=[(_FAIL_GENERATED, "{}\n")]
    )
    assert result.exit_code == 0
    case = _generated(files)
    assert case["eval_id"] == _FAIL_EVAL_ID  # real content, not the "{}" stub


# --- --output handling ------------------------------------------------------

def test_custom_relative_output_path():
    result, files = _invoke(["--output", "mycase.json"], entries=[_FAIL])
    assert result.exit_code == 0
    assert "mycase.json" in files
    assert json.loads(files["mycase.json"])["eval_id"] == _FAIL_EVAL_ID


def test_unsafe_absolute_output_falls_back_with_warning():
    result, files = _invoke(["--json", "--output", "/etc/evil.json"], entries=[_FAIL])
    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["status"] == "created"
    assert any("unsafe --output" in w for w in payload["warnings"])
    # Written to the default generated location, not the requested absolute path.
    assert any(k.startswith(".openshard/evals/generated/") for k in files)


def test_unsafe_traversal_output_falls_back():
    result, files = _invoke(["--output", "../escape.json"], entries=[_FAIL])
    assert result.exit_code == 0
    assert any(k.startswith(".openshard/evals/generated/") for k in files)
    assert "escape.json" not in {k.split("/")[-1] for k in files}


# --- JSON output ------------------------------------------------------------

def test_created_json_output_is_valid():
    result, _ = _invoke(["--json"], entries=[_FAIL])
    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["status"] == "created"
    assert payload["command"] == "eval create-from-last"
    assert payload["eval_id"] == _FAIL_EVAL_ID
    assert payload["failure_category"] == "verification_failed"


# --- no-leak ----------------------------------------------------------------

def test_no_leak_of_unsafe_content():
    secret = "ghp_ABCD1234567890abcdef0000"
    abs_path = "C:\\Users\\Michael\\secret\\app.py"
    leaky = {
        "timestamp": "2026-04-13T06:24:08Z",
        "task": f"fix bug in {abs_path} using token={secret} see /etc/passwd",
        "verification_attempted": True,
        "verification_passed": False,
        "error_message": "Traceback: secret leaked " + secret,
        "findings": [{"severity": "High", "message": "raw finding " + secret}],
        "files_detail": [{"path": abs_path, "change_type": "update"}],
    }
    result, files = _invoke(["--json"], entries=[leaky])
    assert result.exit_code == 0

    blob = result.output + "".join(files.values())
    assert secret not in blob
    assert abs_path not in blob
    assert "/etc/passwd" not in blob
    assert "Traceback" not in blob
    assert "raw finding" not in blob

    case = _generated(files)
    # task is sanitised down to safe words only.
    assert secret not in case["task"]
    assert "C:" not in case["task"]
    # only whitelisted top-level keys are present.
    assert set(case) == {
        "schema_version", "eval_id", "created_at", "source", "source_shard_id",
        "task", "failure_category", "failure_confidence", "failure_reasons",
        "signals", "expected_outcome", "constraints", "metadata",
    }


# --- pure builder unit tests ------------------------------------------------

def test_is_eligible_by_category():
    assert is_eligible(FailureClassification("s", "verification_failed", "high"))
    assert is_eligible(FailureClassification("s", "user_rejected", "high"))
    assert not is_eligible(FailureClassification("s", "no_failure_detected", "high"))


def test_safe_task_text_scrubs_and_falls_back():
    assert _safe_task_text(None) == "OpenShard failed run eval case"
    assert _safe_task_text("   ") == "OpenShard failed run eval case"
    out = _safe_task_text("fix C:\\a\\b.py token=secret_abc1234567890longvalue ok")
    assert "C:" not in out
    assert "token=" not in out
    assert "ok" in out
    # over-long input is capped.
    assert len(_safe_task_text("word " * 200)) <= 120


def test_safe_eval_id_rejects_path_like_values():
    # Path-like input is rejected outright (not scrubbed) -> fallback is used.
    assert _safe_eval_id("../../etc/passwd", "fallback-source") == "fallback-source"
    assert _safe_eval_id("C:\\Users\\Michael\\secret", "fallback-source") == "fallback-source"
    assert _safe_eval_id("a/b\\c", "fallback-source") == "fallback-source"
    assert _safe_eval_id("a..b", "fallback-source") == "fallback-source"
    # Genuinely safe values pass through unchanged.
    assert _safe_eval_id("normal-shard-123", "fallback-source") == "normal-shard-123"
    # Fallback is itself sanitised; if both are unsafe/empty -> "unknown".
    assert _safe_eval_id("", "2026-04-13T00:00:00") != ""
    assert _safe_eval_id(None, "") == "unknown"
    assert _safe_eval_id("../bad", "../also-bad") == "unknown"


def test_build_eval_case_only_reads_safe_fields():
    entry = {"timestamp": "2026-04-13T06:24:08Z", "task": "t",
             "verification_attempted": True, "verification_passed": False}
    receipt = build_shard_receipt(entry, index=0)
    classification = classify_failure(entry, receipt)
    case = build_eval_case(receipt, classification, "2026-06-03T00:00:00+00:00")
    assert case["created_at"] == "2026-06-03T00:00:00+00:00"
    assert case["signals"]["secret_scan_findings"] == 0
    assert isinstance(case["failure_reasons"], list)
