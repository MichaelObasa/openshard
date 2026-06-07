from __future__ import annotations

import json
import re
from pathlib import Path

from click.testing import CliRunner

from openshard.cli.main import cli
from openshard.history.memory import (
    MemoryEntry,
    _entry_to_dict,
    build_memory_entry,
    load_memory_entries,
    log_memory_entry,
)

_RUN_ENTRY = {
    "timestamp": "2026-01-01T00:00:00",
    "task": "refactor the auth module to use JWT tokens",
    "summary": "done",
}

_RUN_JSONL = json.dumps(_RUN_ENTRY)


def _write_run(td: Path) -> None:
    log_dir = td / ".openshard"
    log_dir.mkdir(parents=True, exist_ok=True)
    (log_dir / "runs.jsonl").write_text(_RUN_JSONL, encoding="utf-8")


def test_memory_entry_built_from_run_entry():
    entry = build_memory_entry(_RUN_ENTRY, "accepted", None)
    assert isinstance(entry, MemoryEntry)
    assert entry.outcome == "accepted"
    assert entry.task_short == "refactor the auth module to use JWT tokens"
    assert entry.run_id == "2026-01-01T00:00:00"
    assert entry.reason is None
    assert entry.schema_version == 1


def test_memory_entry_id_format():
    entry = build_memory_entry(_RUN_ENTRY, "rejected", "wrong file")
    assert re.match(r"^mem-\d{8}-\d{6}-\d{6}$", entry.entry_id)


def test_memory_log_and_load_roundtrip(tmp_path):
    entry = build_memory_entry(_RUN_ENTRY, "accepted", None)
    log_memory_entry(entry, cwd=tmp_path)
    loaded = load_memory_entries(cwd=tmp_path)
    assert len(loaded) == 1
    got = loaded[0]
    assert got.entry_id == entry.entry_id
    assert got.run_id == entry.run_id
    assert got.task_short == entry.task_short
    assert got.outcome == entry.outcome
    assert got.reason == entry.reason
    assert got.recorded_at == entry.recorded_at
    assert got.schema_version == entry.schema_version


def test_memory_load_skips_malformed_lines(tmp_path):
    memory_dir = tmp_path / ".openshard"
    memory_dir.mkdir()
    memory_file = memory_dir / "memory.jsonl"
    good = _entry_to_dict(build_memory_entry(_RUN_ENTRY, "accepted", None))
    memory_file.write_text(
        "not valid json\n" + json.dumps(good) + "\n{broken:\n",
        encoding="utf-8",
    )
    entries = load_memory_entries(cwd=tmp_path)
    assert len(entries) == 1
    assert entries[0].outcome == "accepted"


def test_memory_load_empty_file_returns_empty_list(tmp_path):
    memory_dir = tmp_path / ".openshard"
    memory_dir.mkdir()
    (memory_dir / "memory.jsonl").write_text("", encoding="utf-8")
    assert load_memory_entries(cwd=tmp_path) == []


def test_memory_load_missing_file_returns_empty_list(tmp_path):
    assert load_memory_entries(cwd=tmp_path) == []


def test_feedback_accept_creates_memory_entry():
    runner = CliRunner()
    with runner.isolated_filesystem() as td:
        td_path = Path(td)
        _write_run(td_path)
        from unittest.mock import patch
        with patch("openshard.cli.main.Path.cwd", return_value=td_path):
            with patch("openshard.history.memory.Path.cwd", return_value=td_path):
                result = runner.invoke(cli, ["feedback", "accept"])
        assert result.exit_code == 0
        memory_file = td_path / ".openshard" / "memory.jsonl"
        assert memory_file.exists()
        lines = [ln for ln in memory_file.read_text().splitlines() if ln.strip()]
        assert len(lines) == 1
        entry = json.loads(lines[0])
        assert entry["outcome"] == "accepted"


def test_feedback_reject_creates_memory_entry_with_reason():
    runner = CliRunner()
    with runner.isolated_filesystem() as td:
        td_path = Path(td)
        _write_run(td_path)
        from unittest.mock import patch
        with patch("openshard.cli.main.Path.cwd", return_value=td_path):
            with patch("openshard.history.memory.Path.cwd", return_value=td_path):
                result = runner.invoke(cli, ["feedback", "reject", "--reason", "wrong file"])
        assert result.exit_code == 0
        memory_file = td_path / ".openshard" / "memory.jsonl"
        assert memory_file.exists()
        lines = [ln for ln in memory_file.read_text().splitlines() if ln.strip()]
        assert len(lines) == 1
        entry = json.loads(lines[0])
        assert entry["outcome"] == "rejected"
        assert entry["reason"] == "wrong file"


def test_memory_list_exits_zero(tmp_path):
    runner = CliRunner()
    from unittest.mock import patch
    with patch("openshard.cli.main.Path.cwd", return_value=tmp_path):
        with patch("openshard.history.memory.Path.cwd", return_value=tmp_path):
            result = runner.invoke(cli, ["memory", "list"])
    assert result.exit_code == 0


def test_memory_list_shows_entries(tmp_path):
    _write_run(tmp_path)
    entry = build_memory_entry(_RUN_ENTRY, "accepted", None)
    log_memory_entry(entry, cwd=tmp_path)
    runner = CliRunner()
    from unittest.mock import patch
    with patch("openshard.cli.main.Path.cwd", return_value=tmp_path):
        with patch("openshard.history.memory.Path.cwd", return_value=tmp_path):
            result = runner.invoke(cli, ["memory", "list"])
    assert result.exit_code == 0
    assert "accepted" in result.output


def test_memory_stats_exits_zero(tmp_path):
    runner = CliRunner()
    from unittest.mock import patch
    with patch("openshard.cli.main.Path.cwd", return_value=tmp_path):
        with patch("openshard.history.memory.Path.cwd", return_value=tmp_path):
            result = runner.invoke(cli, ["memory", "stats"])
    assert result.exit_code == 0


def test_memory_stats_counts_outcomes(tmp_path):
    _write_run(tmp_path)
    log_memory_entry(build_memory_entry(_RUN_ENTRY, "accepted", None), cwd=tmp_path)
    log_memory_entry(build_memory_entry(_RUN_ENTRY, "rejected", "too slow"), cwd=tmp_path)
    runner = CliRunner()
    from unittest.mock import patch
    with patch("openshard.cli.main.Path.cwd", return_value=tmp_path):
        with patch("openshard.history.memory.Path.cwd", return_value=tmp_path):
            result = runner.invoke(cli, ["memory", "stats"])
    assert result.exit_code == 0
    assert "2" in result.output
    assert "1" in result.output


def test_memory_list_no_entries_shows_message(tmp_path):
    runner = CliRunner()
    from unittest.mock import patch
    with patch("openshard.cli.main.Path.cwd", return_value=tmp_path):
        with patch("openshard.history.memory.Path.cwd", return_value=tmp_path):
            result = runner.invoke(cli, ["memory", "list"])
    assert result.exit_code == 0
    assert "No memory entries" in result.output


def test_memory_stats_no_entries_shows_message(tmp_path):
    runner = CliRunner()
    from unittest.mock import patch
    with patch("openshard.cli.main.Path.cwd", return_value=tmp_path):
        with patch("openshard.history.memory.Path.cwd", return_value=tmp_path):
            result = runner.invoke(cli, ["memory", "stats"])
    assert result.exit_code == 0
    assert "No memory entries" in result.output
