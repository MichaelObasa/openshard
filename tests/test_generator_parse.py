from __future__ import annotations

import json

import pytest

from openshard.execution.generator import (
    ChangedFile,
    ExecutionGenerator,
    ExecutionResult,
    _VALID_CHANGE_TYPES,
)
from openshard.providers.base import ProviderError


def _parse(raw: str) -> ExecutionResult:
    return ExecutionGenerator._parse(raw)


# ---------------------------------------------------------------------------
# _validate_file_object — type/shape failures
# ---------------------------------------------------------------------------

def test_validate_list_returns_none():
    notes: list[str] = []
    assert ExecutionGenerator._validate_file_object(["not", "a", "dict"], 0, notes) is None
    assert len(notes) == 1
    assert "dict" in notes[0]


def test_validate_string_returns_none():
    notes: list[str] = []
    assert ExecutionGenerator._validate_file_object("oops", 2, notes) is None
    assert "File[2]" in notes[0]


def test_validate_none_returns_none():
    notes: list[str] = []
    assert ExecutionGenerator._validate_file_object(None, 0, notes) is None
    assert len(notes) == 1


def test_validate_int_returns_none():
    notes: list[str] = []
    assert ExecutionGenerator._validate_file_object(42, 0, notes) is None


def test_validate_index_in_note():
    notes: list[str] = []
    ExecutionGenerator._validate_file_object("bad", 7, notes)
    assert "7" in notes[0]


# ---------------------------------------------------------------------------
# _validate_file_object — path field
# ---------------------------------------------------------------------------

def test_validate_missing_path_returns_none():
    notes: list[str] = []
    raw = {"change_type": "create", "content": "", "summary": ""}
    assert ExecutionGenerator._validate_file_object(raw, 0, notes) is None
    assert "path" in notes[0]


def test_validate_none_path_returns_none():
    notes: list[str] = []
    raw = {"path": None, "change_type": "create", "content": "", "summary": ""}
    assert ExecutionGenerator._validate_file_object(raw, 0, notes) is None


def test_validate_int_path_returns_none():
    notes: list[str] = []
    raw = {"path": 123, "change_type": "create", "content": "", "summary": ""}
    assert ExecutionGenerator._validate_file_object(raw, 0, notes) is None


def test_validate_empty_path_returns_none():
    notes: list[str] = []
    raw = {"path": "", "change_type": "create", "content": "", "summary": ""}
    assert ExecutionGenerator._validate_file_object(raw, 0, notes) is None
    assert "empty" in notes[0]


def test_validate_whitespace_path_returns_none():
    notes: list[str] = []
    raw = {"path": "   ", "change_type": "update", "content": "x", "summary": ""}
    assert ExecutionGenerator._validate_file_object(raw, 0, notes) is None


# ---------------------------------------------------------------------------
# _validate_file_object — change_type field
# ---------------------------------------------------------------------------

def test_validate_invalid_change_type_returns_none():
    notes: list[str] = []
    raw = {"path": "foo.py", "change_type": "modify", "content": "", "summary": ""}
    assert ExecutionGenerator._validate_file_object(raw, 0, notes) is None
    assert "change_type" in notes[0]
    assert "modify" in notes[0]


def test_validate_none_change_type_returns_none():
    notes: list[str] = []
    raw = {"path": "foo.py", "change_type": None, "content": "", "summary": ""}
    assert ExecutionGenerator._validate_file_object(raw, 0, notes) is None


def test_validate_missing_change_type_returns_none():
    notes: list[str] = []
    raw = {"path": "foo.py", "content": "x", "summary": "s"}
    assert ExecutionGenerator._validate_file_object(raw, 0, notes) is None


def test_validate_all_valid_change_types_accepted():
    for ct in _VALID_CHANGE_TYPES:
        notes: list[str] = []
        raw = {"path": "f.py", "change_type": ct, "content": "", "summary": ""}
        result = ExecutionGenerator._validate_file_object(raw, 0, notes)
        assert result is not None, f"Expected {ct!r} to be accepted"
        assert notes == []


# ---------------------------------------------------------------------------
# _validate_file_object — content field
# ---------------------------------------------------------------------------

def test_validate_int_content_returns_none():
    notes: list[str] = []
    raw = {"path": "foo.py", "change_type": "create", "content": 42, "summary": ""}
    assert ExecutionGenerator._validate_file_object(raw, 0, notes) is None
    assert "content" in notes[0]


def test_validate_null_content_returns_none():
    notes: list[str] = []
    raw = {"path": "foo.py", "change_type": "update", "content": None, "summary": ""}
    assert ExecutionGenerator._validate_file_object(raw, 0, notes) is None


def test_validate_list_content_returns_none():
    notes: list[str] = []
    raw = {"path": "foo.py", "change_type": "create", "content": ["a", "b"], "summary": ""}
    assert ExecutionGenerator._validate_file_object(raw, 0, notes) is None


def test_validate_missing_content_defaults_to_empty_string():
    notes: list[str] = []
    raw = {"path": "foo.py", "change_type": "create", "summary": "s"}
    result = ExecutionGenerator._validate_file_object(raw, 0, notes)
    assert isinstance(result, ChangedFile)
    assert result.content == ""
    assert notes == []


# ---------------------------------------------------------------------------
# _validate_file_object — summary field
# ---------------------------------------------------------------------------

def test_validate_int_summary_returns_none():
    notes: list[str] = []
    raw = {"path": "foo.py", "change_type": "delete", "content": "", "summary": 99}
    assert ExecutionGenerator._validate_file_object(raw, 0, notes) is None
    assert "summary" in notes[0]


def test_validate_null_summary_returns_none():
    notes: list[str] = []
    raw = {"path": "foo.py", "change_type": "update", "content": "x", "summary": None}
    assert ExecutionGenerator._validate_file_object(raw, 0, notes) is None


def test_validate_missing_summary_defaults_to_empty_string():
    notes: list[str] = []
    raw = {"path": "foo.py", "change_type": "update", "content": "x"}
    result = ExecutionGenerator._validate_file_object(raw, 0, notes)
    assert isinstance(result, ChangedFile)
    assert result.summary == ""
    assert notes == []


# ---------------------------------------------------------------------------
# _validate_file_object — valid objects
# ---------------------------------------------------------------------------

def test_validate_valid_create():
    notes: list[str] = []
    raw = {"path": "src/app.py", "change_type": "create", "content": "print(1)", "summary": "added"}
    result = ExecutionGenerator._validate_file_object(raw, 0, notes)
    assert isinstance(result, ChangedFile)
    assert result.path == "src/app.py"
    assert result.change_type == "create"
    assert result.content == "print(1)"
    assert result.summary == "added"
    assert notes == []


def test_validate_valid_update():
    notes: list[str] = []
    raw = {"path": "README.md", "change_type": "update", "content": "# hi", "summary": "updated"}
    result = ExecutionGenerator._validate_file_object(raw, 0, notes)
    assert isinstance(result, ChangedFile)
    assert notes == []


def test_validate_valid_delete_empty_content():
    notes: list[str] = []
    raw = {"path": "old.py", "change_type": "delete", "content": "", "summary": "removed"}
    result = ExecutionGenerator._validate_file_object(raw, 0, notes)
    assert isinstance(result, ChangedFile)
    assert result.change_type == "delete"
    assert notes == []


def test_validate_extra_keys_ignored():
    notes: list[str] = []
    raw = {
        "path": "foo.py",
        "change_type": "create",
        "content": "x",
        "summary": "s",
        "unknown_field": "whatever",
    }
    result = ExecutionGenerator._validate_file_object(raw, 0, notes)
    assert isinstance(result, ChangedFile)
    assert notes == []


# ---------------------------------------------------------------------------
# _parse — structural failures (ProviderError)
# ---------------------------------------------------------------------------

def test_parse_root_is_list_raises():
    payload = json.dumps([{"path": "a.py", "change_type": "create", "content": "", "summary": ""}])
    with pytest.raises(ProviderError, match="JSON object"):
        _parse(payload)


def test_parse_root_is_string_raises():
    with pytest.raises(ProviderError):
        _parse('"just a string"')


def test_parse_root_is_null_raises():
    with pytest.raises(ProviderError):
        _parse("null")


def test_parse_root_is_number_raises():
    with pytest.raises(ProviderError):
        _parse("42")


def test_parse_files_is_string_raises():
    payload = json.dumps({"summary": "ok", "files": "not-a-list", "notes": []})
    with pytest.raises(ProviderError, match="list"):
        _parse(payload)


def test_parse_files_is_dict_raises():
    payload = json.dumps({"summary": "ok", "files": {"path": "a.py"}, "notes": []})
    with pytest.raises(ProviderError):
        _parse(payload)


def test_parse_files_is_null_raises():
    payload = json.dumps({"summary": "ok", "files": None, "notes": []})
    with pytest.raises(ProviderError):
        _parse(payload)


# ---------------------------------------------------------------------------
# _parse — valid / backward-compat
# ---------------------------------------------------------------------------

def test_parse_valid_full_response():
    payload = json.dumps({
        "summary": "Created app",
        "files": [
            {"path": "app.py", "change_type": "create", "content": "x=1", "summary": "init"},
        ],
        "notes": ["run tests"],
    })
    result = _parse(payload)
    assert result.summary == "Created app"
    assert len(result.files) == 1
    assert result.files[0].path == "app.py"
    assert result.files[0].change_type == "create"
    assert result.files[0].content == "x=1"
    assert "run tests" in result.notes


def test_parse_empty_files_list():
    payload = json.dumps({"summary": "nothing", "files": [], "notes": []})
    result = _parse(payload)
    assert result.files == []


def test_parse_missing_files_key_returns_empty():
    payload = json.dumps({"summary": "ok", "notes": []})
    result = _parse(payload)
    assert result.files == []


def test_parse_missing_notes_key_returns_empty():
    payload = json.dumps({"summary": "ok", "files": []})
    result = _parse(payload)
    assert result.notes == []


def test_parse_markdown_fences_stripped():
    inner = json.dumps({"summary": "ok", "files": [], "notes": []})
    raw = f"```json\n{inner}\n```"
    result = _parse(raw)
    assert result.summary == "ok"


def test_parse_all_change_types_pass_through():
    payload = json.dumps({
        "summary": "s",
        "files": [
            {"path": "a.py", "change_type": "create", "content": "a", "summary": ""},
            {"path": "b.py", "change_type": "update", "content": "b", "summary": ""},
            {"path": "c.py", "change_type": "delete", "content": "", "summary": ""},
        ],
        "notes": [],
    })
    result = _parse(payload)
    assert len(result.files) == 3
    assert result.files[0].change_type == "create"
    assert result.files[1].change_type == "update"
    assert result.files[2].change_type == "delete"


def test_parse_non_string_summary_defaults_to_empty():
    payload = json.dumps({"summary": 42, "files": [], "notes": []})
    result = _parse(payload)
    assert result.summary == ""


def test_parse_non_string_notes_ignored():
    payload = json.dumps({"summary": "s", "files": [], "notes": [None, 1, "keep me", ""]})
    result = _parse(payload)
    assert result.notes == ["keep me"]


# ---------------------------------------------------------------------------
# _parse — mixed valid/invalid file lists
# ---------------------------------------------------------------------------

def test_parse_mixed_list_keeps_valid_skips_invalid():
    payload = json.dumps({
        "summary": "partial",
        "files": [
            {"path": "good.py", "change_type": "create", "content": "x", "summary": ""},
            "not a dict at all",
            {"path": "also_good.py", "change_type": "update", "content": "y", "summary": ""},
        ],
        "notes": [],
    })
    result = _parse(payload)
    assert len(result.files) == 2
    assert result.files[0].path == "good.py"
    assert result.files[1].path == "also_good.py"
    assert len(result.notes) == 1


def test_parse_invalid_change_type_skipped_with_note():
    payload = json.dumps({
        "summary": "s",
        "files": [
            {"path": "bad.py", "change_type": "modify", "content": "", "summary": ""},
            {"path": "good.py", "change_type": "create", "content": "x", "summary": ""},
        ],
        "notes": [],
    })
    result = _parse(payload)
    assert len(result.files) == 1
    assert result.files[0].path == "good.py"
    assert any("change_type" in n or "modify" in n for n in result.notes)


def test_parse_empty_path_skipped_with_note():
    payload = json.dumps({
        "summary": "s",
        "files": [
            {"path": "", "change_type": "create", "content": "x", "summary": ""},
            {"path": "real.py", "change_type": "create", "content": "x", "summary": ""},
        ],
        "notes": [],
    })
    result = _parse(payload)
    assert len(result.files) == 1
    assert result.files[0].path == "real.py"
    assert len(result.notes) >= 1


def test_parse_null_content_skipped_with_note():
    payload = json.dumps({
        "summary": "s",
        "files": [
            {"path": "x.py", "change_type": "create", "content": None, "summary": ""},
        ],
        "notes": [],
    })
    result = _parse(payload)
    assert result.files == []
    assert len(result.notes) == 1


def test_parse_all_invalid_produces_empty_files_with_notes():
    payload = json.dumps({
        "summary": "s",
        "files": [
            None,
            42,
            {"path": "", "change_type": "create", "content": "", "summary": ""},
        ],
        "notes": [],
    })
    result = _parse(payload)
    assert result.files == []
    assert len(result.notes) == 3


def test_parse_model_notes_preserved_alongside_validation_warnings():
    payload = json.dumps({
        "summary": "s",
        "files": [
            {"path": "bad.py", "change_type": "INVALID", "content": "", "summary": ""},
        ],
        "notes": ["deploy after merge"],
    })
    result = _parse(payload)
    assert any("deploy" in n for n in result.notes)
    assert any("change_type" in n or "INVALID" in n for n in result.notes)
