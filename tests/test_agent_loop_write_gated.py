from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from openshard.native.agent_loop_types import AgentAction, ApprovalReceipt
from openshard.native.agent_loop_write_gated import GatedToolRunner, WriteGrant, _exec_write_file


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _action(kind: str = "write_file", args: dict | None = None,
            action_id: str = "a0") -> AgentAction:
    return AgentAction(
        action_id=action_id,
        iteration_index=0,
        kind=kind,
        args=args or {},
        rationale="test",
        needs_approval=(kind == "write_file"),
    )


def _grant(action_id: str = "a0", *, granted: bool = True) -> WriteGrant:
    return WriteGrant(
        receipt=ApprovalReceipt(request_id="r0", granted=granted, timestamp="t"),
        action_id=action_id,
    )


class _TempRepo:
    def setUp(self):
        self._td = tempfile.TemporaryDirectory()
        self.root = Path(self._td.name)
        (self.root / "existing.py").write_text("x = 1\n", encoding="utf-8")
        self.runner = GatedToolRunner(self.root)

    def tearDown(self):
        self._td.cleanup()


# ---------------------------------------------------------------------------
# WriteGrant validation
# ---------------------------------------------------------------------------

class TestGrantValidation(_TempRepo, unittest.TestCase):
    def test_no_grant_rejected(self):
        r = self.runner.run(_action("write_file", {"path": "out.py", "content": "x"}))
        self.assertFalse(r.ok)
        self.assertIn("WriteGrant", r.error)

    def test_denied_receipt_rejected(self):
        r = self.runner.run(
            _action("write_file", {"path": "out.py", "content": "x"}),
            grant=_grant(granted=False),
        )
        self.assertFalse(r.ok)
        self.assertIn("denied", r.error)

    def test_mismatched_action_id_rejected(self):
        a = _action("write_file", {"path": "out.py", "content": "x"}, action_id="a0")
        g = _grant(action_id="wrong")
        r = self.runner.run(a, grant=g)
        self.assertFalse(r.ok)
        self.assertIn("mismatch", r.error)

    def test_all_rejections_have_action_id(self):
        a = _action("write_file", {"path": "out.py"}, action_id="myid")
        r = self.runner.run(a)
        self.assertEqual(r.action_id, "myid")

    def test_all_rejections_raw_content_false(self):
        for grant_arg in (None, _grant(granted=False), _grant(action_id="bad")):
            with self.subTest(grant=grant_arg):
                r = self.runner.run(
                    _action("write_file", {"path": "out.py", "content": "x"}),
                    grant=grant_arg,
                )
                self.assertFalse(r.raw_content_stored)


# ---------------------------------------------------------------------------
# Successful write
# ---------------------------------------------------------------------------

class TestSuccessfulWrite(_TempRepo, unittest.TestCase):
    def test_writes_new_file(self):
        path = "new_file.py"
        r = self.runner.run(
            _action("write_file", {"path": path, "content": "hello = 1\n"}),
            grant=_grant(),
        )
        self.assertTrue(r.ok)
        self.assertTrue((self.root / path).exists())
        self.assertEqual((self.root / path).read_text(encoding="utf-8"), "hello = 1\n")

    def test_output_reports_char_count(self):
        content = "abc"
        r = self.runner.run(
            _action("write_file", {"path": "out.py", "content": content}),
            grant=_grant(),
        )
        self.assertIn("3", r.output)

    def test_overwrites_existing_file(self):
        r = self.runner.run(
            _action("write_file", {"path": "existing.py", "content": "y = 2\n"}),
            grant=_grant(),
        )
        self.assertTrue(r.ok)
        self.assertEqual((self.root / "existing.py").read_text(encoding="utf-8"), "y = 2\n")

    def test_creates_parent_dirs(self):
        r = self.runner.run(
            _action("write_file", {"path": "sub/deep/file.py", "content": "z"}),
            grant=_grant(),
        )
        self.assertTrue(r.ok)
        self.assertTrue((self.root / "sub" / "deep" / "file.py").exists())

    def test_empty_content_allowed(self):
        r = self.runner.run(
            _action("write_file", {"path": "empty.py", "content": ""}),
            grant=_grant(),
        )
        self.assertTrue(r.ok)

    def test_raw_content_stored_false_on_success(self):
        r = self.runner.run(
            _action("write_file", {"path": "out.py", "content": "x"}),
            grant=_grant(),
        )
        self.assertFalse(r.raw_content_stored)

    def test_duration_ms_non_negative(self):
        r = self.runner.run(
            _action("write_file", {"path": "out.py", "content": "x"}),
            grant=_grant(),
        )
        self.assertGreaterEqual(r.duration_ms, 0)


# ---------------------------------------------------------------------------
# Path safety with a valid grant
# ---------------------------------------------------------------------------

class TestWritePathSafety(_TempRepo, unittest.TestCase):
    def test_traversal_rejected_even_with_grant(self):
        r = self.runner.run(
            _action("write_file", {"path": "../../../tmp/evil.py", "content": "bad"}),
            grant=_grant(),
        )
        self.assertFalse(r.ok)
        self.assertFalse((self.root.parent / "tmp" / "evil.py").exists())

    def test_empty_path_rejected_even_with_grant(self):
        r = self.runner.run(
            _action("write_file", {"path": "", "content": "x"}),
            grant=_grant(),
        )
        self.assertFalse(r.ok)
        self.assertIn("non-empty", r.error)

    def test_missing_path_key_rejected(self):
        r = self.runner.run(
            _action("write_file", {"content": "x"}),
            grant=_grant(),
        )
        self.assertFalse(r.ok)


# ---------------------------------------------------------------------------
# Read-only actions still work through GatedToolRunner
# ---------------------------------------------------------------------------

class TestReadThroughDelegation(_TempRepo, unittest.TestCase):
    def test_list_files_works(self):
        r = self.runner.run(_action("list_files", {}))
        self.assertTrue(r.ok)
        self.assertIn("existing.py", r.output)

    def test_read_file_works(self):
        r = self.runner.run(_action("read_file", {"path": "existing.py"}))
        self.assertTrue(r.ok)
        self.assertIn("x = 1", r.output)

    def test_run_command_still_blocked(self):
        r = self.runner.run(_action("run_command", {}))
        self.assertFalse(r.ok)
        self.assertIn("blocked", r.error)

    def test_ask_human_still_rejected(self):
        r = self.runner.run(_action("ask_human", {}))
        self.assertFalse(r.ok)
        self.assertIn("loop signal", r.error)

    def test_grant_ignored_for_read_actions(self):
        # Passing a grant to a read action is harmless — it's ignored.
        r = self.runner.run(_action("list_files", {}), grant=_grant())
        self.assertTrue(r.ok)


# ---------------------------------------------------------------------------
# _exec_write_file unit tests (internal, but worth testing directly)
# ---------------------------------------------------------------------------

class TestExecWriteFile(unittest.TestCase):
    def setUp(self):
        self._td = tempfile.TemporaryDirectory()
        self.root = Path(self._td.name)

    def tearDown(self):
        self._td.cleanup()

    def test_empty_path_returns_error(self):
        r = _exec_write_file(self.root, "", "content", "aid")
        self.assertFalse(r.ok)

    def test_writes_and_returns_ok(self):
        r = _exec_write_file(self.root, "f.py", "hello", "aid")
        self.assertTrue(r.ok)
        self.assertEqual((self.root / "f.py").read_text(), "hello")

    def test_action_id_in_result(self):
        r = _exec_write_file(self.root, "f.py", "x", "myid")
        self.assertEqual(r.action_id, "myid")

    def test_raw_content_stored_false(self):
        r = _exec_write_file(self.root, "f.py", "x", "aid")
        self.assertFalse(r.raw_content_stored)


if __name__ == "__main__":
    unittest.main()
