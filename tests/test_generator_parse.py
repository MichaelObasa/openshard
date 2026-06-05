from __future__ import annotations

import json

import pytest

from openshard.execution.generator import (
    _VALID_CHANGE_TYPES,
    ChangedFile,
    ExecutionGenerator,
    ExecutionResult,
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


# ---------------------------------------------------------------------------
# _recover_review_result — review-task fallback when JSON parsing fails
# ---------------------------------------------------------------------------

def _recover(raw: str) -> ExecutionResult:
    return ExecutionGenerator._recover_review_result(raw)


def test_recover_extracts_summary_from_truncated_json():
    raw = '{\n  "summary": "Found 12 issues worth addressing.\\n\\nCritical\\n- Bad CIDR",\n  "files": ['
    result = _recover(raw)
    assert "Found 12 issues" in result.summary
    assert "Critical" in result.summary
    assert "Bad CIDR" in result.summary
    assert result.files == []
    assert result.notes == []


def test_recover_unescapes_newlines_in_summary():
    raw = '{"summary": "Line 1\\nLine 2\\nLine 3"}'
    result = _recover(raw)
    assert "\n" in result.summary
    assert "Line 1" in result.summary
    assert "Line 3" in result.summary


def test_recover_unescapes_quotes_in_summary():
    raw = '{"summary": "He said \\"hello\\" world"}'
    result = _recover(raw)
    assert '"hello"' in result.summary


def test_recover_falls_back_to_raw_when_no_summary_field():
    raw = "The review found critical issues with your Terraform."
    result = _recover(raw)
    assert "critical issues" in result.summary
    assert result.files == []


def test_recover_strips_json_braces_when_no_summary_field():
    raw = '{"something": "else", "no_summary_here": true}'
    result = _recover(raw)
    assert result.files == []
    assert result.notes == []


def test_recover_strips_hidden_contract_from_summary():
    raw = json.dumps({
        "summary": "Found 5 issues.\n\nAfter completing your analysis output STRUCTURED_FINDINGS: []",
        "files": [],
    })
    # _recover is only called when _parse fails, but we can call it directly
    result = _recover(raw)
    assert "STRUCTURED_FINDINGS" not in result.summary
    assert "After completing your analysis" not in result.summary
    assert "Found 5 issues" in result.summary


def test_recover_strips_structured_findings_sentinel():
    raw = '{"summary": "Found issues.\\nSTRUCTURED_FINDINGS: [{\\\"severity\\\": \\\"High\\\"}]"}'
    result = _recover(raw)
    assert "STRUCTURED_FINDINGS" not in result.summary
    assert "Found issues" in result.summary


def test_recover_files_always_empty():
    raw = json.dumps({"summary": "review done", "files": [{"path": "a.py", "change_type": "create", "content": "x", "summary": ""}]})
    result = _recover(raw)
    assert result.files == []


# ---------------------------------------------------------------------------
# generate() with is_review_task — integration via mock provider
# ---------------------------------------------------------------------------

class _MockChatResponse:
    def __init__(self, content: str):
        self.content = content
        self.model = "test/model"
        self.usage = None


class _MockProvider:
    """Minimal BaseProvider shim that returns a fixed response."""

    def __init__(self, content: str):
        self._content = content

    def execute(self, model, prompt, system=None, max_tokens=None):
        return _MockChatResponse(self._content)

    def list_models(self):
        return []

    def get_model_info(self, model_id):
        return None


def _make_generator(provider_content: str) -> ExecutionGenerator:
    gen = ExecutionGenerator.__new__(ExecutionGenerator)
    gen.model = "test/model"
    gen.fixer_model = "test/model"
    gen._provider = _MockProvider(provider_content)
    gen._client = None
    return gen


def test_generate_review_task_recovers_from_truncated_json():
    """generate(is_review_task=True) must not raise when JSON is truncated."""
    truncated = '{\n  "summary": "Found 7 issues.\\n\\nCritical\\n- Open port 22",\n  "files": ['
    gen = _make_generator(truncated)
    result = gen.generate("review task STRUCTURED_FINDINGS:", is_review_task=True)
    assert "Found 7 issues" in result.summary
    assert result.files == []


def test_generate_review_task_recovers_with_no_json():
    """generate(is_review_task=True) falls back to raw text when no JSON at all."""
    gen = _make_generator("This is a plain text review with no JSON structure.")
    result = gen.generate("review STRUCTURED_FINDINGS:", is_review_task=True)
    assert "plain text review" in result.summary
    assert result.files == []


def test_generate_non_review_task_still_raises_on_bad_json():
    """generate(is_review_task=False) must still raise ProviderError on bad JSON."""
    gen = _make_generator("not json at all {{{{")
    with pytest.raises(ProviderError):
        gen.generate("normal task", is_review_task=False)


def test_generate_review_task_succeeds_on_valid_json():
    """generate(is_review_task=True) uses normal parse path when JSON is valid."""
    payload = json.dumps({
        "summary": "Found 3 issues.",
        "files": [],
        "notes": [],
    })
    gen = _make_generator(payload)
    result = gen.generate("review STRUCTURED_FINDINGS:", is_review_task=True)
    assert result.summary == "Found 3 issues."
    assert result.files == []


def test_generate_review_task_does_not_leak_structured_findings():
    """Recovered summary must not contain STRUCTURED_FINDINGS."""
    raw = '{"summary": "Issues found.\\nSTRUCTURED_FINDINGS: [{\\"severity\\": \\"Critical\\"}]"'
    gen = _make_generator(raw)
    result = gen.generate("review STRUCTURED_FINDINGS:", is_review_task=True)
    assert "STRUCTURED_FINDINGS" not in result.summary
    assert "Issues found" in result.summary
