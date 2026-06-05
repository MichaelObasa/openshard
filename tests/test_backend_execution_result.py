from __future__ import annotations

import json
import sys
import unittest
from dataclasses import asdict


class TestBackendExecutionResultDefaults(unittest.TestCase):
    def _result(self, **kwargs):
        from openshard.native.backends import BackendExecutionResult
        return BackendExecutionResult(**kwargs)

    def test_serializes_cleanly(self):
        result = self._result()
        serialized = json.dumps(asdict(result))
        self.assertIsInstance(serialized, str)
        self.assertIn("raw_content_stored", serialized)

    def test_raw_content_stored_defaults_false(self):
        result = self._result()
        self.assertFalse(result.raw_content_stored)

    def test_can_embed_in_run_history_metadata(self):
        result = self._result(
            backend_name="deepagents",
            mode="readonly_agent_proof",
            steps=["observe", "plan"],
            tools_used=["read_file"],
            warnings=["read-only mode"],
        )
        metadata = {"backend_result": asdict(result)}
        serialized = json.dumps(metadata)
        self.assertIn("deepagents", serialized)
        self.assertIn("raw_content_stored", serialized)

    def test_no_deepagents_import_required(self):
        with unittest.mock.patch.dict(sys.modules, {"deepagents": None}):
            from openshard.native.backends import BackendExecutionResult
            result = BackendExecutionResult(backend_name="deepagents")
        self.assertEqual(result.backend_name, "deepagents")

    def test_existing_native_backend_result_unchanged(self):
        from openshard.native.backends import NativeBackendResult
        result = NativeBackendResult()
        self.assertEqual(result.summary, "")
        self.assertEqual(result.notes, [])
        self.assertEqual(result.metadata, {})
        self.assertFalse(hasattr(result, "raw_content_stored"))

    def test_backend_contract_has_no_execution_side_effects(self):
        from openshard.native.backends import BackendExecutionResult
        result = BackendExecutionResult()
        for name in ("run", "execute", "write", "apply"):
            self.assertFalse(
                callable(getattr(result, name, None)),
                msg=f"BackendExecutionResult must not have a callable '{name}' method",
            )


import unittest.mock  # noqa: E402 — after class to avoid circular at module level

if __name__ == "__main__":
    unittest.main()
