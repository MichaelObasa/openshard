from __future__ import annotations

import json
from pathlib import Path

from click.testing import CliRunner

from openshard.cli.main import cli

_LOG_REL = Path(".openshard") / "runs.jsonl"


def _write_run(entry: dict | None = None) -> None:
    if entry is None:
        entry = {"timestamp": "2026-01-01T00:00:00", "task": "do X", "summary": "done"}
    _LOG_REL.parent.mkdir(parents=True, exist_ok=True)
    _LOG_REL.write_text(json.dumps(entry) + "\n", encoding="utf-8")


def _read_entries() -> list[dict]:
    return [json.loads(ln) for ln in _LOG_REL.read_text(encoding="utf-8").splitlines() if ln.strip()]


def _runner() -> CliRunner:
    return CliRunner()


def test_note_adds_note_to_last_entry():
    with _runner().isolated_filesystem():
        _write_run()
        _runner().invoke(cli, ["note", "hello"])
        entries = _read_entries()
        assert entries[-1]["notes"][0]["text"] == "hello"


def test_note_appends_not_overwrites():
    with _runner().isolated_filesystem():
        _write_run()
        _runner().invoke(cli, ["note", "first"])
        _runner().invoke(cli, ["note", "second"])
        entries = _read_entries()
        assert len(entries[-1]["notes"]) == 2
        assert entries[-1]["notes"][1]["text"] == "second"


def test_note_stores_recorded_at():
    with _runner().isolated_filesystem():
        _write_run()
        _runner().invoke(cli, ["note", "hello"])
        entries = _read_entries()
        recorded_at = entries[-1]["notes"][0]["recorded_at"]
        assert recorded_at and len(recorded_at) > 0


def test_note_stores_schema_version():
    with _runner().isolated_filesystem():
        _write_run()
        _runner().invoke(cli, ["note", "hello"])
        entries = _read_entries()
        assert entries[-1]["notes"][0]["schema_version"] == 1


def test_note_exits_zero_on_success():
    with _runner().isolated_filesystem():
        _write_run()
        result = _runner().invoke(cli, ["note", "hello"])
        assert result.exit_code == 0


def test_note_prints_confirmation():
    with _runner().isolated_filesystem():
        _write_run()
        result = _runner().invoke(cli, ["note", "hello"])
        assert "Note recorded" in result.output


def test_note_no_history_exits_nonzero():
    with _runner().isolated_filesystem():
        result = _runner().invoke(cli, ["note", "hello"])
        assert result.exit_code != 0
        assert "No run history found" in result.output


def test_note_empty_history_exits_nonzero():
    with _runner().isolated_filesystem():
        _LOG_REL.parent.mkdir(parents=True, exist_ok=True)
        _LOG_REL.write_text("", encoding="utf-8")
        result = _runner().invoke(cli, ["note", "hello"])
        assert result.exit_code != 0


def test_note_text_capped_at_500_chars():
    with _runner().isolated_filesystem():
        _write_run()
        long_text = "a" * 600
        _runner().invoke(cli, ["note", long_text])
        entries = _read_entries()
        assert len(entries[-1]["notes"][0]["text"]) <= 500


def test_note_secret_scrubbed():
    with _runner().isolated_filesystem():
        _write_run()
        secret_text = "key is sk-ant-api03-abcdefghijklmnopqrstuvwxyz0123456789"
        _runner().invoke(cli, ["note", secret_text])
        entries = _read_entries()
        stored = entries[-1]["notes"][0]["text"]
        assert "sk-ant-api03-abcdefghijklmnopqrstuvwxyz0123456789" not in stored


def test_last_more_shows_notes_section():
    with _runner().isolated_filesystem():
        _write_run()
        _runner().invoke(cli, ["note", "my test note"])
        result = _runner().invoke(cli, ["last", "--more"])
        assert "NOTE" in result.output
        assert "my test note" in result.output
