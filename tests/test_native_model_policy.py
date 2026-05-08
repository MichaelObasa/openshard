from __future__ import annotations

import json
import unittest
from dataclasses import asdict

from openshard.native.context import NativeModelPolicy, build_native_model_policy
from openshard.native.executor import NativeRunMeta
from openshard.cli.run_output import _native_meta_from_entry


class TestNativeModelPolicyDefaults(unittest.TestCase):
    def test_default_mode_is_auto(self):
        p = NativeModelPolicy()
        self.assertEqual(p.mode, "auto")

    def test_default_tiers_are_empty(self):
        p = NativeModelPolicy()
        self.assertEqual(p.allowed_tiers, [])
        self.assertEqual(p.disallowed_tiers, [])

    def test_default_flags(self):
        p = NativeModelPolicy()
        self.assertFalse(p.prefer_low_cost)
        self.assertFalse(p.require_open_source)
        self.assertFalse(p.require_local)
        self.assertTrue(p.allow_frontier)

    def test_default_warnings_empty(self):
        p = NativeModelPolicy()
        self.assertEqual(p.warnings, [])

    def test_json_roundtrip(self):
        p = NativeModelPolicy()
        d = asdict(p)
        raw = json.dumps(d)
        parsed = json.loads(raw)
        self.assertEqual(parsed["mode"], "auto")
        self.assertTrue(parsed["allow_frontier"])
        self.assertFalse(parsed["prefer_low_cost"])
        self.assertEqual(parsed["warnings"], [])


class TestBuildNativeModelPolicyAuto(unittest.TestCase):
    def test_none_produces_auto(self):
        p = build_native_model_policy(None)
        self.assertEqual(p.mode, "auto")
        self.assertTrue(p.allow_frontier)
        self.assertFalse(p.prefer_low_cost)
        self.assertFalse(p.require_open_source)
        self.assertFalse(p.require_local)
        self.assertEqual(p.warnings, [])

    def test_explicit_auto_matches_none(self):
        p_none = build_native_model_policy(None)
        p_auto = build_native_model_policy("auto")
        self.assertEqual(asdict(p_none), asdict(p_auto))


class TestBuildNativeModelPolicyCheapestSafe(unittest.TestCase):
    def test_mode_set(self):
        p = build_native_model_policy("cheapest-safe")
        self.assertEqual(p.mode, "cheapest-safe")

    def test_prefer_low_cost_true(self):
        p = build_native_model_policy("cheapest-safe")
        self.assertTrue(p.prefer_low_cost)

    def test_allow_frontier_true(self):
        p = build_native_model_policy("cheapest-safe")
        self.assertTrue(p.allow_frontier)

    def test_no_warnings(self):
        p = build_native_model_policy("cheapest-safe")
        self.assertEqual(p.warnings, [])


class TestBuildNativeModelPolicyFrontierHeavy(unittest.TestCase):
    def test_mode_set(self):
        p = build_native_model_policy("frontier-heavy")
        self.assertEqual(p.mode, "frontier-heavy")

    def test_allow_frontier_true(self):
        p = build_native_model_policy("frontier-heavy")
        self.assertTrue(p.allow_frontier)

    def test_prefer_low_cost_false(self):
        p = build_native_model_policy("frontier-heavy")
        self.assertFalse(p.prefer_low_cost)

    def test_no_warnings(self):
        p = build_native_model_policy("frontier-heavy")
        self.assertEqual(p.warnings, [])


class TestBuildNativeModelPolicyOpenSourceOnly(unittest.TestCase):
    def test_mode_set(self):
        p = build_native_model_policy("open-source-only")
        self.assertEqual(p.mode, "open-source-only")

    def test_require_open_source_true(self):
        p = build_native_model_policy("open-source-only")
        self.assertTrue(p.require_open_source)

    def test_allow_frontier_false(self):
        p = build_native_model_policy("open-source-only")
        self.assertFalse(p.allow_frontier)

    def test_frontier_reasoning_disallowed(self):
        p = build_native_model_policy("open-source-only")
        self.assertIn("frontier-reasoning-model", p.disallowed_tiers)

    def test_require_local_false(self):
        p = build_native_model_policy("open-source-only")
        self.assertFalse(p.require_local)

    def test_no_warnings(self):
        p = build_native_model_policy("open-source-only")
        self.assertEqual(p.warnings, [])


class TestBuildNativeModelPolicyLocalOnly(unittest.TestCase):
    def test_mode_set(self):
        p = build_native_model_policy("local-only")
        self.assertEqual(p.mode, "local-only")

    def test_require_local_true(self):
        p = build_native_model_policy("local-only")
        self.assertTrue(p.require_local)

    def test_require_open_source_true(self):
        p = build_native_model_policy("local-only")
        self.assertTrue(p.require_open_source)

    def test_allow_frontier_false(self):
        p = build_native_model_policy("local-only")
        self.assertFalse(p.allow_frontier)

    def test_frontier_reasoning_disallowed(self):
        p = build_native_model_policy("local-only")
        self.assertIn("frontier-reasoning-model", p.disallowed_tiers)

    def test_no_warnings(self):
        p = build_native_model_policy("local-only")
        self.assertEqual(p.warnings, [])


class TestBuildNativeModelPolicyCustom(unittest.TestCase):
    def test_mode_set(self):
        p = build_native_model_policy("custom")
        self.assertEqual(p.mode, "custom")

    def test_warning_emitted(self):
        p = build_native_model_policy("custom")
        self.assertEqual(len(p.warnings), 1)
        self.assertIn("not enforced in v1", p.warnings[0])

    def test_frontier_still_allowed_by_default(self):
        p = build_native_model_policy("custom")
        self.assertTrue(p.allow_frontier)


class TestBuildNativeModelPolicyUnknownMode(unittest.TestCase):
    def test_falls_back_to_auto(self):
        p = build_native_model_policy("nonexistent-mode")
        self.assertEqual(p.mode, "auto")

    def test_warning_emitted(self):
        p = build_native_model_policy("nonexistent-mode")
        self.assertEqual(len(p.warnings), 1)
        self.assertIn("defaulted to auto", p.warnings[0])

    def test_safe_defaults_preserved(self):
        p = build_native_model_policy("nonexistent-mode")
        self.assertTrue(p.allow_frontier)
        self.assertFalse(p.prefer_low_cost)
        self.assertFalse(p.require_open_source)
        self.assertFalse(p.require_local)


class TestNativeModelPolicyCLIChoices(unittest.TestCase):
    def test_all_valid_choice_values_accepted(self):
        for mode in ("auto", "cheapest-safe", "frontier-heavy", "open-source-only", "local-only", "custom"):
            try:
                p = build_native_model_policy(mode)
            except Exception as exc:
                self.fail(f"build_native_model_policy({mode!r}) raised {exc}")
            self.assertIsInstance(p, NativeModelPolicy)


class TestNativeModelPolicySerializes(unittest.TestCase):
    def test_asdict_produces_json_serializable_dict(self):
        for mode in ("auto", "cheapest-safe", "frontier-heavy", "open-source-only", "local-only", "custom"):
            p = build_native_model_policy(mode)
            d = asdict(p)
            try:
                json.dumps(d)
            except (TypeError, ValueError) as exc:
                self.fail(f"asdict({mode}) not JSON-serializable: {exc}")

    def test_asdict_has_expected_keys(self):
        p = build_native_model_policy("auto")
        d = asdict(p)
        for key in ("mode", "allowed_tiers", "disallowed_tiers", "prefer_low_cost",
                    "require_open_source", "require_local", "allow_frontier", "warnings"):
            self.assertIn(key, d)


class TestNativeRunMetaHasPolicyField(unittest.TestCase):
    def test_model_policy_defaults_to_none(self):
        meta = NativeRunMeta()
        self.assertIsNone(meta.model_policy)

    def test_model_policy_can_be_assigned(self):
        meta = NativeRunMeta()
        meta.model_policy = build_native_model_policy("cheapest-safe")
        self.assertIsNotNone(meta.model_policy)
        self.assertEqual(meta.model_policy.mode, "cheapest-safe")


class TestOldEntryWithoutPolicyNoCrash(unittest.TestCase):
    def _base_native_entry(self) -> dict:
        return {
            "workflow": "native",
            "executor": "native",
            "native_loop_steps": [],
            "native_loop_trace": [],
            "read_search_findings": [],
            "write_path": "pipeline",
        }

    def test_entry_without_model_policy_key_does_not_crash(self):
        entry = self._base_native_entry()
        self.assertNotIn("model_policy", entry)
        try:
            nm = _native_meta_from_entry(entry)
        except Exception as exc:
            self.fail(f"_native_meta_from_entry raised {exc}")
        self.assertIsNotNone(nm)

    def test_entry_without_model_policy_returns_none_for_field(self):
        entry = self._base_native_entry()
        nm = _native_meta_from_entry(entry)
        mp = getattr(nm, "model_policy", "MISSING")
        self.assertIsNone(mp)

    def test_entry_with_model_policy_dict_is_extracted(self):
        entry = self._base_native_entry()
        entry["model_policy"] = asdict(build_native_model_policy("cheapest-safe"))
        nm = _native_meta_from_entry(entry)
        mp = getattr(nm, "model_policy", None)
        self.assertIsNotNone(mp)
        mode = mp.get("mode") if isinstance(mp, dict) else getattr(mp, "mode", None)
        self.assertEqual(mode, "cheapest-safe")


if __name__ == "__main__":
    unittest.main()
