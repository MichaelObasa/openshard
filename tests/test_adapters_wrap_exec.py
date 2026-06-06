"""Unit tests for openshard.adapters.wrap_exec."""

from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from openshard.adapters.wrap_exec import (
    _parse_git_changed_files,
    _sanitize_model,
    _sanitize_task,
    build_wrap_entry,
    capture_pre_run_state,
    run_wrapped_command,
    write_wrap_entry,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fake_pre_state(**kwargs) -> dict:
    base = {
        "git_branch": "main",
        "git_head_commit_hash": "abc1234",
        "git_dirty": False,
        "captured_at": "2026-06-06T12:00:00Z",
    }
    base.update(kwargs)
    return base


def _make_entry(repo_path: Path, **kwargs) -> dict:
    defaults = dict(
        task="Fix the bug",
        model=None,
        pre_state=_fake_pre_state(),
        exit_code=0,
        repo_path=repo_path,
    )
    defaults.update(kwargs)
    return build_wrap_entry(
        defaults.pop("task"),
        model=defaults.pop("model"),
        pre_state=defaults.pop("pre_state"),
        exit_code=defaults.pop("exit_code"),
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

    def test_import_method_is_wrap_v0(self):
        entry = _make_entry(self.repo)
        self.assertEqual(entry["import_method"], "openshard_wrap_v0")

    def test_executor_field(self):
        entry = _make_entry(self.repo)
        self.assertEqual(entry["executor"], "claude_code_wrap")

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
# wrap_exit_code field
# ---------------------------------------------------------------------------

class TestWrapExitCode(unittest.TestCase):

    def setUp(self):
        import tempfile
        self._tmp = tempfile.TemporaryDirectory()
        self.repo = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def test_exit_code_zero_stored(self):
        entry = _make_entry(self.repo, exit_code=0)
        self.assertEqual(entry["wrap_exit_code"], 0)

    def test_exit_code_nonzero_stored(self):
        entry = _make_entry(self.repo, exit_code=1)
        self.assertEqual(entry["wrap_exit_code"], 1)

    def test_exit_code_127_stored(self):
        entry = _make_entry(self.repo, exit_code=127)
        self.assertEqual(entry["wrap_exit_code"], 127)

    def test_nonzero_exit_sets_wrap_status_subprocess_failed(self):
        entry = _make_entry(self.repo, exit_code=1)
        metadata = entry.get("metadata", {})
        self.assertEqual(metadata.get("wrap_status"), "subprocess_failed")

    def test_zero_exit_does_not_set_subprocess_failed(self):
        entry = _make_entry(self.repo, exit_code=0)
        metadata = entry.get("metadata", {})
        self.assertNotEqual(metadata.get("wrap_status"), "subprocess_failed")

    def test_nonzero_exit_does_not_set_verification_passed(self):
        entry = _make_entry(self.repo, exit_code=1)
        self.assertNotIn("verification_passed", entry)


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
        self.assertEqual(result, "Claude Code wrap session")

    def test_non_string_task_returns_placeholder(self):
        result = _sanitize_task(None)  # type: ignore[arg-type]
        self.assertEqual(result, "Claude Code wrap session")

    def test_task_capped_at_500_chars(self):
        long_task = "a" * 600
        result = _sanitize_task(long_task)
        self.assertLessEqual(len(result), 500)

    def test_secret_in_entry_task_is_scrubbed(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            entry = _make_entry(
                repo,
                task="Use sk-ant-api03-SECRETSECRET12345678901234 to auth",
            )
            self.assertNotIn("sk-ant-api03-SECRETSECRET12345678901234", entry["task"])


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
# Verification fields — honesty constraints
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

    def test_verification_passed_not_present_on_success(self):
        entry = _make_entry(self.repo, exit_code=0)
        self.assertNotIn("verification_passed", entry)

    def test_verification_passed_not_present_on_failure(self):
        entry = _make_entry(self.repo, exit_code=1)
        self.assertNotIn("verification_passed", entry)


# ---------------------------------------------------------------------------
# Content hash — coerce_shard_entry was called
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

    def test_schema_version_present(self):
        entry = _make_entry(self.repo)
        self.assertEqual(entry["schema_version"], "1.2")


# ---------------------------------------------------------------------------
# Blocked fields — coerce strips them
# ---------------------------------------------------------------------------

class TestBlockedFields(unittest.TestCase):

    def setUp(self):
        import tempfile
        self._tmp = tempfile.TemporaryDirectory()
        self.repo = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def test_blocked_fields_stripped_by_coerce(self):
        from openshard.history.shard_schema import coerce_shard_entry
        dirty = {"task": "test", "raw_prompt": "SECRET", "import_source": "claude_code"}
        coerced = coerce_shard_entry(dirty)
        self.assertNotIn("raw_prompt", coerced)

    def test_no_raw_diff_in_entry(self):
        entry = _make_entry(self.repo)
        self.assertNotIn("raw_diff", entry)

    def test_no_transcript_in_entry(self):
        entry = _make_entry(self.repo)
        self.assertNotIn("transcript", entry)


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


# ---------------------------------------------------------------------------
# capture_pre_run_state
# ---------------------------------------------------------------------------

class TestCapturePreRunState(unittest.TestCase):

    def setUp(self):
        import tempfile
        self._tmp = tempfile.TemporaryDirectory()
        self.repo = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def test_returns_dict(self):
        state = capture_pre_run_state(self.repo)
        self.assertIsInstance(state, dict)

    def test_has_git_branch_key(self):
        state = capture_pre_run_state(self.repo)
        self.assertIn("git_branch", state)

    def test_has_git_head_commit_hash_key(self):
        state = capture_pre_run_state(self.repo)
        self.assertIn("git_head_commit_hash", state)

    def test_has_git_dirty_key(self):
        state = capture_pre_run_state(self.repo)
        self.assertIn("git_dirty", state)

    def test_has_captured_at_key(self):
        state = capture_pre_run_state(self.repo)
        self.assertIn("captured_at", state)


# ---------------------------------------------------------------------------
# run_wrapped_command
# ---------------------------------------------------------------------------

class TestRunWrappedCommand(unittest.TestCase):

    def test_returns_zero_on_success(self):
        code = run_wrapped_command(["true"])
        self.assertEqual(code, 0)

    def test_returns_nonzero_on_failure(self):
        code = run_wrapped_command(["false"])
        self.assertNotEqual(code, 0)

    def test_returns_127_for_missing_command(self):
        code = run_wrapped_command(["__openshard_no_such_command_xyz__"])
        self.assertEqual(code, 127)

    def test_no_output_capture(self):
        # Ensure run_wrapped_command uses passthrough (no capture)
        import inspect

        import openshard.adapters.wrap_exec as mod
        src = inspect.getsource(mod.run_wrapped_command)
        self.assertNotIn("stdout=subprocess.PIPE", src)
        self.assertNotIn("stderr=subprocess.PIPE", src)
        self.assertNotIn("capture_output=True", src)


# ---------------------------------------------------------------------------
# write_wrap_entry
# ---------------------------------------------------------------------------

class TestWriteWrapEntry(unittest.TestCase):

    def setUp(self):
        import tempfile
        self._tmp = tempfile.TemporaryDirectory()
        self.repo = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def test_creates_openshard_dir(self):
        entry = _make_entry(self.repo)
        write_wrap_entry(entry, self.repo)
        self.assertTrue((self.repo / ".openshard").is_dir())

    def test_creates_runs_jsonl(self):
        entry = _make_entry(self.repo)
        write_wrap_entry(entry, self.repo)
        self.assertTrue((self.repo / ".openshard" / "runs.jsonl").exists())

    def test_written_entry_has_correct_import_source(self):
        import json
        import os
        orig = os.getcwd()
        os.chdir(self.repo)
        try:
            entry = _make_entry(self.repo)
            write_wrap_entry(entry, self.repo)
            runs_path = self.repo / ".openshard" / "runs.jsonl"
            data = json.loads(runs_path.read_text(encoding="utf-8").strip())
            self.assertEqual(data["import_source"], "claude_code")
        finally:
            os.chdir(orig)

    def test_written_entry_has_content_hash(self):
        import json
        import os
        orig = os.getcwd()
        os.chdir(self.repo)
        try:
            entry = _make_entry(self.repo)
            write_wrap_entry(entry, self.repo)
            runs_path = self.repo / ".openshard" / "runs.jsonl"
            data = json.loads(runs_path.read_text(encoding="utf-8").strip())
            self.assertIn("content_hash", data)
        finally:
            os.chdir(orig)


if __name__ == "__main__":
    unittest.main()
