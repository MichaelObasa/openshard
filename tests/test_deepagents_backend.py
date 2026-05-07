from __future__ import annotations

import json
import sys
import unittest
from dataclasses import asdict
from unittest.mock import MagicMock, patch

from openshard.native.backends import DeepAgentsAdapterMeta, build_deepagents_adapter_meta


class TestDeepAgentsAdapterMeta(unittest.TestCase):
    def test_defaults(self):
        meta = DeepAgentsAdapterMeta()
        self.assertFalse(meta.available)
        self.assertIsNone(meta.version)
        self.assertEqual(meta.adapter_class, "DeepAgentsNativeBackend")
        self.assertEqual(meta.mode, "unavailable")
        self.assertEqual(meta.notes, [])

    def test_json_serializable(self):
        meta = DeepAgentsAdapterMeta(
            available=True, version="1.2.3", mode="available_unconfigured", notes=["note"]
        )
        result = json.dumps(asdict(meta))
        self.assertIn("available_unconfigured", result)


class TestBuildDeepAgentsAdapterMeta(unittest.TestCase):
    def test_unavailable_when_not_installed(self):
        with patch.dict(sys.modules, {"deepagents": None}):
            result = build_deepagents_adapter_meta()
        self.assertFalse(result.available)
        self.assertEqual(result.mode, "unavailable")
        self.assertIsNone(result.version)
        self.assertTrue(len(result.notes) > 0)

    def test_available_unconfigured_when_installed(self):
        fake_da = MagicMock()
        fake_da.__version__ = "2.0.0"
        with patch.dict(sys.modules, {"deepagents": fake_da}):
            result = build_deepagents_adapter_meta()
        self.assertTrue(result.available)
        self.assertEqual(result.mode, "available_unconfigured")

    def test_captures_version_when_present(self):
        fake_da = MagicMock()
        fake_da.__version__ = "1.2.3"
        with patch.dict(sys.modules, {"deepagents": fake_da}):
            result = build_deepagents_adapter_meta()
        self.assertEqual(result.version, "1.2.3")

    def test_no_version_when_absent(self):
        fake_da = MagicMock(spec=[])  # no __version__ attribute
        with patch.dict(sys.modules, {"deepagents": fake_da}):
            result = build_deepagents_adapter_meta()
        self.assertIsNone(result.version)

    def test_never_raises_on_import_error(self):
        with patch.dict(sys.modules, {"deepagents": None}):
            result = build_deepagents_adapter_meta()
        self.assertIsInstance(result, DeepAgentsAdapterMeta)

    def test_never_raises_on_unexpected_error(self):
        with patch("builtins.__import__", side_effect=RuntimeError("unexpected")):
            result = build_deepagents_adapter_meta()
        self.assertIsInstance(result, DeepAgentsAdapterMeta)
        self.assertFalse(result.available)
        self.assertEqual(result.mode, "unavailable")

    def test_create_deep_agent_detected_but_not_called(self):
        fake_da = MagicMock()
        fake_da.__version__ = "3.0.0"
        fake_da.create_deep_agent = MagicMock()
        with patch.dict(sys.modules, {"deepagents": fake_da}):
            build_deepagents_adapter_meta()
        fake_da.create_deep_agent.assert_not_called()


class TestRunDeepAgentsAdapterPhase(unittest.TestCase):
    def _make_executor(self, backend_name="deepagents", experimental=False, deepagents_installed=True):
        from openshard.native.executor import NativeAgentExecutor
        fake_da = MagicMock()
        fake_da.__version__ = "1.0.0"
        modules = {"deepagents": fake_da if deepagents_installed else None}
        with patch.dict(sys.modules, modules):
            executor = NativeAgentExecutor(
                provider=MagicMock(),
                backend_name=backend_name,
                experimental_deepagents_run=experimental,
            )
        return executor, modules

    def test_phase_runs_when_backend_is_deepagents(self):
        fake_da = MagicMock()
        fake_da.__version__ = "1.0.0"
        with patch.dict(sys.modules, {"deepagents": fake_da}):
            from openshard.native.executor import NativeAgentExecutor
            executor = NativeAgentExecutor(provider=MagicMock(), backend_name="deepagents")
            executor._run_deepagents_adapter_phase()
        self.assertIsNotNone(executor.native_meta.deepagents_adapter)

    def test_phase_skipped_when_backend_is_builtin(self):
        from openshard.native.executor import NativeAgentExecutor
        executor = NativeAgentExecutor(provider=MagicMock(), backend_name="builtin")
        self.assertIsNone(executor.native_meta.deepagents_adapter)

    def test_phase_records_loop_step(self):
        fake_da = MagicMock()
        fake_da.__version__ = "1.0.0"
        with patch.dict(sys.modules, {"deepagents": fake_da}):
            from openshard.native.executor import NativeAgentExecutor
            executor = NativeAgentExecutor(provider=MagicMock(), backend_name="deepagents")
            executor._run_deepagents_adapter_phase()
        self.assertIn("deepagents_adapter", executor.native_meta.native_loop_steps)

    def test_unavailable_records_unavailable_mode(self):
        with patch.dict(sys.modules, {"deepagents": None}):
            from openshard.native.executor import NativeAgentExecutor
            executor = NativeAgentExecutor(provider=MagicMock(), backend_name="deepagents")
            executor._run_deepagents_adapter_phase()
        self.assertFalse(executor.native_meta.deepagents_adapter.available)
        self.assertEqual(executor.native_meta.deepagents_adapter.mode, "unavailable")

    def test_phase_is_independent_of_experimental_flag(self):
        fake_da = MagicMock()
        fake_da.__version__ = "1.0.0"
        with patch.dict(sys.modules, {"deepagents": fake_da}):
            from openshard.native.executor import NativeAgentExecutor
            executor = NativeAgentExecutor(
                provider=MagicMock(),
                backend_name="deepagents",
                experimental_deepagents_run=False,
            )
            executor._run_deepagents_adapter_phase()
        self.assertIsNotNone(executor.native_meta.deepagents_adapter)
        self.assertEqual(executor.native_meta.deepagents_adapter.mode, "available_unconfigured")

    def test_experimental_flag_does_not_populate_adapter(self):
        # deepagents_adapter is set by _run_deepagents_adapter_phase, not by _run_backend_proof_phase
        from openshard.native.executor import NativeAgentExecutor
        executor = NativeAgentExecutor(
            provider=MagicMock(),
            backend_name="builtin",
            experimental_deepagents_run=True,
        )
        self.assertIsNone(executor.native_meta.deepagents_adapter)


class TestRenderDeepAgentsAdapter(unittest.TestCase):
    def _render(self, deepagents_adapter):
        from openshard.cli.run_output import _render_native_demo_block
        from openshard.native.executor import NativeRunMeta
        meta = NativeRunMeta()
        meta.deepagents_adapter = deepagents_adapter
        return "\n".join(_render_native_demo_block(meta))

    def test_renders_available_unconfigured(self):
        adapter = DeepAgentsAdapterMeta(
            available=True, mode="available_unconfigured", notes=[]
        )
        out = self._render(adapter)
        self.assertIn("deepagents adapter: available unconfigured", out)

    def test_renders_unavailable(self):
        adapter = DeepAgentsAdapterMeta(available=False, mode="unavailable", notes=[])
        out = self._render(adapter)
        self.assertIn("deepagents adapter: unavailable", out)

    def test_renders_version_when_present(self):
        adapter = DeepAgentsAdapterMeta(
            available=True, version="1.2.3", mode="available_unconfigured", notes=[]
        )
        out = self._render(adapter)
        self.assertIn("[v1.2.3]", out)

    def test_no_version_suffix_when_absent(self):
        adapter = DeepAgentsAdapterMeta(available=True, mode="available_unconfigured", notes=[])
        out = self._render(adapter)
        self.assertNotIn("[v", out)

    def test_renders_notes(self):
        adapter = DeepAgentsAdapterMeta(
            available=False, mode="unavailable",
            notes=["Install deepagents to enable this experimental backend."]
        )
        out = self._render(adapter)
        self.assertIn("deepagents adapter note:", out)
        self.assertIn("Install deepagents", out)

    def test_no_crash_when_deepagents_adapter_is_none(self):
        from openshard.cli.run_output import _render_native_demo_block
        from openshard.native.executor import NativeRunMeta
        meta = NativeRunMeta()
        meta.deepagents_adapter = None
        out = "\n".join(_render_native_demo_block(meta))
        self.assertNotIn("deepagents adapter:", out)


class TestDeepAgentsAdapterRoundTrip(unittest.TestCase):
    def test_round_trip_via_native_meta_from_entry(self):
        from openshard.cli.run_output import _native_meta_from_entry
        entry = {
            "workflow": "native",
            "native_backend": "deepagents",
            "native_backend_available": True,
            "native_backend_notes": [],
            "deepagents_adapter": {
                "available": True,
                "version": "2.1.0",
                "adapter_class": "DeepAgentsNativeBackend",
                "mode": "available_unconfigured",
                "notes": [],
            },
        }
        meta = _native_meta_from_entry(entry)
        self.assertIsNotNone(meta.deepagents_adapter)
        self.assertEqual(meta.deepagents_adapter.mode, "available_unconfigured")
        self.assertEqual(meta.deepagents_adapter.version, "2.1.0")
        self.assertTrue(meta.deepagents_adapter.available)

    def test_missing_key_in_old_entry(self):
        from openshard.cli.run_output import _native_meta_from_entry
        entry = {
            "workflow": "native",
            "native_backend": "builtin",
            "native_backend_available": True,
            "native_backend_notes": [],
        }
        meta = _native_meta_from_entry(entry)
        self.assertIsNone(meta.deepagents_adapter)

    def test_round_trip_rendering_from_history(self):
        from openshard.cli.run_output import _native_meta_from_entry, _render_native_demo_block
        entry = {
            "workflow": "native",
            "native_backend": "deepagents",
            "native_backend_available": True,
            "native_backend_notes": [],
            "deepagents_adapter": {
                "available": False,
                "version": None,
                "adapter_class": "DeepAgentsNativeBackend",
                "mode": "unavailable",
                "notes": ["Install deepagents."],
            },
        }
        meta = _native_meta_from_entry(entry)
        out = "\n".join(_render_native_demo_block(meta))
        self.assertIn("deepagents adapter: unavailable", out)
        self.assertIn("deepagents adapter note: Install deepagents.", out)
