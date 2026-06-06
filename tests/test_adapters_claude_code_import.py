"""Unit tests for openshard.adapters.claude_code_import."""

from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from openshard.adapters.claude_code_import import (
    _parse_git_changed_files,
    _sanitize_model,
    _sanitize_task,
    _scrub_notes_file,
    build_claude_code_import_entry,
    write_import_entry,
)
from openshard.history.metrics import load_runs

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_entry(repo_path: Path, **kwargs) -> dict:
    defaults = dict(task="Fix the bug", model=None, notes_file=None, repo_path=repo_path)
    defaults.update(kwargs)
    return build_claude_code_import_entry(
        defaults.pop("task"),
        model=defaults.pop("model"),
        notes_file=defaults.pop("notes_file"),
        repo_path=defaults.pop("repo_path"),
    )


# ---------------------------------------------------------------------------
# Provenance markers
# ---------------------------------------------------------------------------

class TestProvenanceMarkers(unittest.TestCase):

    def setUp(self):
        import tempfile
        self._tmp = tempfile.TemporaryDirectory()
        self.repo = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def test_import_source_is_claude_code(self):
        entry = _make_entry(self.repo)
        self.assertEqual(entry["import_source"], "claude_code")

    def test_import_method_is_v0(self):
        entry = _make_entry(self.repo)
        self.assertEqual(entry["import_method"], "openshard_import_v0")

    def test_executor_field(self):
        entry = _make_entry(self.repo)
        self.assertEqual(entry["executor"], "claude_code_import")

    def test_import_note_present_and_non_empty(self):
        entry = _make_entry(self.repo)
        note = entry.get("import_note", "")
        self.assertIsInstance(note, str)
        self.assertTrue(len(note) > 0)

    def test_import_note_mentions_claude_code(self):
        entry = _make_entry(self.repo)
        self.assertIn("Claude Code", entry["import_note"])

    def test_import_note_mentions_openshard(self):
        entry = _make_entry(self.repo)
        self.assertIn("OpenShard", entry["import_note"])


# ---------------------------------------------------------------------------
# Model handling
# ---------------------------------------------------------------------------

class TestModelHandling(unittest.TestCase):

    def setUp(self):
        import tempfile
        self._tmp = tempfile.TemporaryDirectory()
        self.repo = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def test_unknown_model_when_not_provided(self):
        entry = _make_entry(self.repo, model=None)
        self.assertEqual(entry["execution_model"], "unknown")

    def test_model_stored_when_provided(self):
        entry = _make_entry(self.repo, model="claude-sonnet-4-6")
        self.assertEqual(entry["execution_model"], "claude-sonnet-4-6")

    def test_sanitize_model_returns_unknown_for_none(self):
        self.assertEqual(_sanitize_model(None), "unknown")

    def test_sanitize_model_returns_unknown_for_empty(self):
        self.assertEqual(_sanitize_model(""), "unknown")

    def test_sanitize_model_passes_through_valid_slug(self):
        self.assertEqual(_sanitize_model("claude-opus-4-7"), "claude-opus-4-7")


# ---------------------------------------------------------------------------
# Cost and token fields must never appear
# ---------------------------------------------------------------------------

class TestNoFinancialFields(unittest.TestCase):

    def setUp(self):
        import tempfile
        self._tmp = tempfile.TemporaryDirectory()
        self.repo = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def test_no_estimated_cost(self):
        entry = _make_entry(self.repo)
        self.assertNotIn("estimated_cost", entry)

    def test_no_prompt_tokens(self):
        entry = _make_entry(self.repo)
        self.assertNotIn("prompt_tokens", entry)

    def test_no_completion_tokens(self):
        entry = _make_entry(self.repo)
        self.assertNotIn("completion_tokens", entry)

    def test_no_total_tokens(self):
        entry = _make_entry(self.repo)
        self.assertNotIn("total_tokens", entry)


# ---------------------------------------------------------------------------
# Verification fields
# ---------------------------------------------------------------------------

class TestVerificationFields(unittest.TestCase):

    def setUp(self):
        import tempfile
        self._tmp = tempfile.TemporaryDirectory()
        self.repo = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def test_verification_attempted_is_false(self):
        entry = _make_entry(self.repo)
        self.assertIs(entry["verification_attempted"], False)

    def test_verification_passed_is_none(self):
        entry = _make_entry(self.repo)
        self.assertIsNone(entry.get("verification_passed"))


# ---------------------------------------------------------------------------
# Content hash
# ---------------------------------------------------------------------------

class TestContentHash(unittest.TestCase):

    def setUp(self):
        import tempfile
        self._tmp = tempfile.TemporaryDirectory()
        self.repo = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def test_content_hash_present(self):
        entry = _make_entry(self.repo)
        self.assertIn("content_hash", entry)

    def test_content_hash_starts_with_sha256(self):
        entry = _make_entry(self.repo)
        self.assertTrue(entry["content_hash"].startswith("sha256:"))


# ---------------------------------------------------------------------------
# Blocked fields
# ---------------------------------------------------------------------------

class TestBlockedFields(unittest.TestCase):

    def setUp(self):
        import tempfile
        self._tmp = tempfile.TemporaryDirectory()
        self.repo = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def test_blocked_fields_stripped_by_coerce(self):
        # Patch the builder to inject a blocked field; coerce must strip it.
        from openshard.history.shard_schema import coerce_shard_entry
        dirty = {"task": "test", "raw_prompt": "SECRET", "import_source": "claude_code"}
        coerced = coerce_shard_entry(dirty)
        self.assertNotIn("raw_prompt", coerced)

    def test_schema_version_present(self):
        entry = _make_entry(self.repo)
        self.assertEqual(entry["schema_version"], "1.2")


# ---------------------------------------------------------------------------
# Task sanitization
# ---------------------------------------------------------------------------

class TestTaskSanitization(unittest.TestCase):

    def test_task_stored_normally(self):
        result = _sanitize_task("Fix the login bug")
        self.assertEqual(result, "Fix the login bug")

    def test_task_with_secret_is_scrubbed(self):
        result = _sanitize_task("Use sk-ant-abc12345678901234567890 to call API")
        self.assertNotIn("sk-ant-abc12345678901234567890", result)

    def test_empty_task_returns_placeholder(self):
        result = _sanitize_task("")
        self.assertEqual(result, "Claude Code session import")

    def test_non_string_task_returns_placeholder(self):
        result = _sanitize_task(None)  # type: ignore[arg-type]
        self.assertEqual(result, "Claude Code session import")

    def test_task_capped_at_500_chars(self):
        long_task = "a" * 600
        result = _sanitize_task(long_task)
        self.assertLessEqual(len(result), 500)


# ---------------------------------------------------------------------------
# git diff parsing
# ---------------------------------------------------------------------------

class TestParseGitChangedFiles(unittest.TestCase):

    def setUp(self):
        import tempfile
        self._tmp = tempfile.TemporaryDirectory()
        self.repo = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def test_returns_empty_outside_repo(self):
        files, source = _parse_git_changed_files(self.repo)
        self.assertEqual(files, [])
        self.assertEqual(source, "not_available")

    def test_classifies_add_as_create(self):
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "A\tsrc/new_file.py\n"
        with patch("subprocess.run", return_value=mock_result):
            files, source = _parse_git_changed_files(self.repo)
        self.assertEqual(len(files), 1)
        self.assertEqual(files[0]["change_type"], "create")
        self.assertEqual(files[0]["path"], "src/new_file.py")
        self.assertEqual(source, "git_diff_inferred")

    def test_classifies_modify_as_update(self):
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "M\tsrc/existing.py\n"
        with patch("subprocess.run", return_value=mock_result):
            files, source = _parse_git_changed_files(self.repo)
        self.assertEqual(files[0]["change_type"], "update")

    def test_classifies_delete_as_delete(self):
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "D\tsrc/removed.py\n"
        with patch("subprocess.run", return_value=mock_result):
            files, source = _parse_git_changed_files(self.repo)
        self.assertEqual(files[0]["change_type"], "delete")

    def test_caps_at_20_files(self):
        lines = "\n".join(f"M\tsrc/file{i}.py" for i in range(25))
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = lines
        with patch("subprocess.run", return_value=mock_result):
            files, _ = _parse_git_changed_files(self.repo)
        self.assertLessEqual(len(files), 20)

    def test_returns_empty_and_not_available_on_git_error(self):
        mock_result = MagicMock()
        mock_result.returncode = 128
        mock_result.stdout = ""
        with patch("subprocess.run", return_value=mock_result):
            files, source = _parse_git_changed_files(self.repo)
        self.assertEqual(files, [])
        self.assertEqual(source, "not_available")

    def test_files_source_git_diff_inferred_when_empty_diff(self):
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = ""
        with patch("subprocess.run", return_value=mock_result):
            files, source = _parse_git_changed_files(self.repo)
        self.assertEqual(files, [])
        self.assertEqual(source, "git_diff_inferred")

    def test_summary_field_is_honest(self):
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "M\tsrc/foo.py\n"
        with patch("subprocess.run", return_value=mock_result):
            files, _ = _parse_git_changed_files(self.repo)
        self.assertIn("inferred", files[0]["summary"])


# ---------------------------------------------------------------------------
# Notes file scrubbing
# ---------------------------------------------------------------------------

class TestScrubNotesFile(unittest.TestCase):

    def setUp(self):
        import tempfile
        self._tmp = tempfile.TemporaryDirectory()
        self.tmpdir = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def test_normal_notes_stored(self):
        notes = self.tmpdir / "notes.md"
        notes.write_text("Fixed the auth bug by rewriting the token check.", encoding="utf-8")
        result = _scrub_notes_file(notes)
        self.assertIn("auth bug", result)

    def test_secret_in_notes_is_redacted(self):
        notes = self.tmpdir / "notes.md"
        notes.write_text("Used key sk-ant-abcdefghijklmnopqrst12345 for testing.", encoding="utf-8")
        result = _scrub_notes_file(notes)
        self.assertNotIn("sk-ant-abcdefghijklmnopqrst12345", result)

    def test_notes_capped_at_summary_limit(self):
        notes = self.tmpdir / "notes.md"
        notes.write_text("x" * 1000, encoding="utf-8")
        result = _scrub_notes_file(notes)
        self.assertLessEqual(len(result), 300)

    def test_missing_file_returns_empty_string(self):
        result = _scrub_notes_file(self.tmpdir / "nonexistent.md")
        self.assertEqual(result, "")


# ---------------------------------------------------------------------------
# files_source field
# ---------------------------------------------------------------------------

class TestFilesSource(unittest.TestCase):

    def setUp(self):
        import tempfile
        self._tmp = tempfile.TemporaryDirectory()
        self.repo = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def test_files_source_not_available_outside_repo(self):
        entry = _make_entry(self.repo)
        self.assertEqual(entry["files_source"], "not_available")

    def test_files_source_git_diff_inferred_with_mocked_git(self):
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "M\tsrc/foo.py\n"
        with patch("subprocess.run", return_value=mock_result):
            entry = _make_entry(self.repo)
        self.assertEqual(entry["files_source"], "git_diff_inferred")


# ---------------------------------------------------------------------------
# write_import_entry
# ---------------------------------------------------------------------------

class TestWriteImportEntry(unittest.TestCase):

    def setUp(self):
        import tempfile
        self._tmp = tempfile.TemporaryDirectory()
        self.repo = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def test_creates_openshard_dir(self):
        entry = _make_entry(self.repo)
        write_import_entry(entry, self.repo)
        self.assertTrue((self.repo / ".openshard").is_dir())

    def test_creates_runs_jsonl(self):
        entry = _make_entry(self.repo)
        write_import_entry(entry, self.repo)
        self.assertTrue((self.repo / ".openshard" / "runs.jsonl").exists())

    def test_written_entry_appears_in_load_runs(self):
        import os
        orig = os.getcwd()
        os.chdir(self.repo)
        try:
            entry = _make_entry(self.repo)
            write_import_entry(entry, self.repo)
            runs = load_runs()
            self.assertEqual(len(runs), 1)
            self.assertEqual(runs[0]["import_source"], "claude_code")
        finally:
            os.chdir(orig)

    def test_written_entry_has_content_hash(self):
        import os
        orig = os.getcwd()
        os.chdir(self.repo)
        try:
            entry = _make_entry(self.repo)
            write_import_entry(entry, self.repo)
            runs = load_runs()
            self.assertIn("content_hash", runs[0])
        finally:
            os.chdir(orig)


if __name__ == "__main__":
    unittest.main()
