from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

from openshard.tui.state import get_guardrails, load_git_info, load_recent_runs

# ---------------------------------------------------------------------------
# load_git_info
# ---------------------------------------------------------------------------


def test_load_git_info_returns_project_name(tmp_path):
    info = load_git_info(tmp_path)
    assert info["project_name"] == tmp_path.name


def test_load_git_info_subprocess_exception_returns_unknown(tmp_path):
    with patch("openshard.tui.state.subprocess.run", side_effect=OSError("no git")):
        info = load_git_info(tmp_path)
    assert info["branch"] == "unknown"
    assert info["state"] == "unknown"


def test_load_git_info_nonzero_returncode_returns_unknown(tmp_path):
    mock = MagicMock()
    mock.returncode = 128
    mock.stdout = ""
    with patch("openshard.tui.state.subprocess.run", return_value=mock):
        info = load_git_info(tmp_path)
    assert info["branch"] == "unknown"
    assert info["state"] == "unknown"


def test_load_git_info_clean_state(tmp_path):
    mock = MagicMock()
    mock.returncode = 0
    mock.stdout = "## main...origin/main\n"
    with patch("openshard.tui.state.subprocess.run", return_value=mock):
        info = load_git_info(tmp_path)
    assert info["branch"] == "main"
    assert info["state"] == "clean"


def test_load_git_info_dirty_state(tmp_path):
    mock = MagicMock()
    mock.returncode = 0
    mock.stdout = "## main...origin/main\n M src/foo.py\n"
    with patch("openshard.tui.state.subprocess.run", return_value=mock):
        info = load_git_info(tmp_path)
    assert info["state"] == "dirty"


def test_load_git_info_branch_parsed_from_tracking_ref(tmp_path):
    mock = MagicMock()
    mock.returncode = 0
    mock.stdout = "## feat/my-feature...origin/feat/my-feature [ahead 2]\n"
    with patch("openshard.tui.state.subprocess.run", return_value=mock):
        info = load_git_info(tmp_path)
    assert info["branch"] == "feat/my-feature"


def test_load_git_info_empty_stdout_returns_unknown(tmp_path):
    mock = MagicMock()
    mock.returncode = 0
    mock.stdout = ""
    with patch("openshard.tui.state.subprocess.run", return_value=mock):
        info = load_git_info(tmp_path)
    assert info["branch"] == "unknown"
    assert info["state"] == "unknown"


# ---------------------------------------------------------------------------
# load_recent_runs
# ---------------------------------------------------------------------------


def test_load_recent_runs_missing_file(tmp_path):
    assert load_recent_runs(tmp_path) == []


def test_load_recent_runs_empty_file(tmp_path):
    (tmp_path / ".openshard").mkdir()
    (tmp_path / ".openshard" / "runs.jsonl").write_text("", encoding="utf-8")
    assert load_recent_runs(tmp_path) == []


def test_load_recent_runs_corrupt_lines_skipped(tmp_path):
    (tmp_path / ".openshard").mkdir()
    (tmp_path / ".openshard" / "runs.jsonl").write_text(
        "{bad json\n"
        '{"task": "good", "timestamp": "2026-01-01T00:00:00Z"}\n',
        encoding="utf-8",
    )
    runs = load_recent_runs(tmp_path)
    assert len(runs) == 1
    assert runs[0]["task"] == "good"


def test_load_recent_runs_status_passed(tmp_path):
    (tmp_path / ".openshard").mkdir()
    entry = {"task": "t", "timestamp": "2026-01-01T00:00:00Z", "verification_passed": True}
    (tmp_path / ".openshard" / "runs.jsonl").write_text(json.dumps(entry) + "\n", encoding="utf-8")
    runs = load_recent_runs(tmp_path)
    assert runs[0]["status"] == "passed"


def test_load_recent_runs_status_failed(tmp_path):
    (tmp_path / ".openshard").mkdir()
    entry = {"task": "t", "timestamp": "2026-01-01T00:00:00Z", "verification_passed": False}
    (tmp_path / ".openshard" / "runs.jsonl").write_text(json.dumps(entry) + "\n", encoding="utf-8")
    runs = load_recent_runs(tmp_path)
    assert runs[0]["status"] == "failed"


def test_load_recent_runs_status_readonly_via_form_factor(tmp_path):
    (tmp_path / ".openshard").mkdir()
    entry = {
        "task": "t",
        "timestamp": "2026-01-01T00:00:00Z",
        "form_factor": {"read_only": True},
    }
    (tmp_path / ".openshard" / "runs.jsonl").write_text(json.dumps(entry) + "\n", encoding="utf-8")
    runs = load_recent_runs(tmp_path)
    assert runs[0]["status"] == "read-only"


def test_load_recent_runs_status_readonly_via_zero_file_counts(tmp_path):
    (tmp_path / ".openshard").mkdir()
    entry = {
        "task": "t",
        "timestamp": "2026-01-01T00:00:00Z",
        "files_created": 0,
        "files_updated": 0,
        "files_deleted": 0,
    }
    (tmp_path / ".openshard" / "runs.jsonl").write_text(json.dumps(entry) + "\n", encoding="utf-8")
    runs = load_recent_runs(tmp_path)
    assert runs[0]["status"] == "read-only"


def test_load_recent_runs_status_unknown_for_old_record(tmp_path):
    (tmp_path / ".openshard").mkdir()
    entry = {"task": "t"}
    (tmp_path / ".openshard" / "runs.jsonl").write_text(json.dumps(entry) + "\n", encoding="utf-8")
    runs = load_recent_runs(tmp_path)
    assert runs[0]["status"] == "unknown"


def test_load_recent_runs_limit(tmp_path):
    (tmp_path / ".openshard").mkdir()
    entries = [{"task": f"task {i}", "timestamp": f"2026-01-{i+1:02d}T00:00:00Z"} for i in range(10)]
    text = "\n".join(json.dumps(e) for e in entries) + "\n"
    (tmp_path / ".openshard" / "runs.jsonl").write_text(text, encoding="utf-8")
    runs = load_recent_runs(tmp_path, limit=5)
    assert len(runs) == 5


def test_load_recent_runs_newest_first(tmp_path):
    (tmp_path / ".openshard").mkdir()
    entries = [
        {"task": "first", "timestamp": "2026-01-01T00:00:00Z"},
        {"task": "second", "timestamp": "2026-01-02T00:00:00Z"},
        {"task": "third", "timestamp": "2026-01-03T00:00:00Z"},
    ]
    text = "\n".join(json.dumps(e) for e in entries) + "\n"
    (tmp_path / ".openshard" / "runs.jsonl").write_text(text, encoding="utf-8")
    runs = load_recent_runs(tmp_path)
    assert runs[0]["task"] == "third"
    assert runs[-1]["task"] == "first"


def test_load_recent_runs_duration_formatted(tmp_path):
    (tmp_path / ".openshard").mkdir()
    entry = {"task": "t", "timestamp": "2026-01-01T00:00:00Z", "duration_seconds": 7.3}
    (tmp_path / ".openshard" / "runs.jsonl").write_text(json.dumps(entry) + "\n", encoding="utf-8")
    runs = load_recent_runs(tmp_path)
    assert runs[0]["duration"] == "7.3s"


def test_load_recent_runs_duration_absent(tmp_path):
    (tmp_path / ".openshard").mkdir()
    entry = {"task": "t", "timestamp": "2026-01-01T00:00:00Z"}
    (tmp_path / ".openshard" / "runs.jsonl").write_text(json.dumps(entry) + "\n", encoding="utf-8")
    runs = load_recent_runs(tmp_path)
    assert runs[0]["duration"] == "-"


# ---------------------------------------------------------------------------
# get_guardrails
# ---------------------------------------------------------------------------


def test_get_guardrails_returns_all_keys():
    g = get_guardrails()
    for key in ("Agent", "Model", "Sandbox", "Approval", "Checks", "Receipts"):
        assert key in g


def test_get_guardrails_agent_value():
    assert get_guardrails()["Agent"] == "OpenShard Native"


def test_get_guardrails_sandbox_on():
    assert get_guardrails()["Sandbox"] == "On"


def test_get_guardrails_approval_smart():
    assert get_guardrails()["Approval"] == "Smart"
