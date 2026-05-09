from __future__ import annotations

import json
import unittest
from dataclasses import asdict
from types import SimpleNamespace

from openshard.native.context import (
    NativeModelPolicyReceipt,
    NativeRoutingPreview,
    NativeRoutingReceipt,
    NativeRunTrustScore,
    build_native_routing_receipt,
)
from openshard.cli.run_output import _native_meta_from_entry, _render_native_demo_block


def _ns(**kwargs: object) -> SimpleNamespace:
    return SimpleNamespace(**kwargs)


def _build(**kwargs: object) -> NativeRoutingReceipt:
    return build_native_routing_receipt(**kwargs)


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

class TestNativeRoutingReceiptDefaults(unittest.TestCase):

    def setUp(self) -> None:
        self.receipt = NativeRoutingReceipt()

    def test_strategy_default(self):
        self.assertEqual(self.receipt.strategy, "")

    def test_planner_tier_default(self):
        self.assertEqual(self.receipt.planner_tier, "unknown")

    def test_executor_tier_default(self):
        self.assertEqual(self.receipt.executor_tier, "unknown")

    def test_validator_tier_default(self):
        self.assertEqual(self.receipt.validator_tier, "unknown")

    def test_policy_mode_default(self):
        self.assertEqual(self.receipt.policy_mode, "auto")

    def test_policy_affected_default(self):
        self.assertFalse(self.receipt.policy_affected)

    def test_blocked_count_default(self):
        self.assertEqual(self.receipt.blocked_count, 0)

    def test_trust_level_default(self):
        self.assertEqual(self.receipt.trust_level, "unknown")

    def test_confidence_default(self):
        self.assertEqual(self.receipt.confidence, "medium")

    def test_warnings_count_default(self):
        self.assertEqual(self.receipt.warnings_count, 0)

    def test_summary_default(self):
        self.assertEqual(self.receipt.summary, "")


# ---------------------------------------------------------------------------
# All-None inputs — no crash
# ---------------------------------------------------------------------------

class TestBuildNativeRoutingReceiptAllNone(unittest.TestCase):

    def test_returns_receipt(self):
        r = _build()
        self.assertIsInstance(r, NativeRoutingReceipt)

    def test_strategy_empty(self):
        self.assertEqual(_build().strategy, "")

    def test_trust_level_unknown(self):
        self.assertEqual(_build().trust_level, "unknown")

    def test_policy_affected_false(self):
        self.assertFalse(_build().policy_affected)

    def test_blocked_count_zero(self):
        self.assertEqual(_build().blocked_count, 0)

    def test_warnings_count_zero(self):
        self.assertEqual(_build().warnings_count, 0)

    def test_confidence_medium(self):
        self.assertEqual(_build().confidence, "medium")


# ---------------------------------------------------------------------------
# Derives fields from routing_preview
# ---------------------------------------------------------------------------

class TestBuildNativeRoutingReceiptFromRoutingPreview(unittest.TestCase):

    def _preview(self, **kwargs) -> NativeRoutingPreview:
        defaults = dict(
            strategy="frontier-heavy",
            planner_tier="frontier",
            executor_tier="fast",
            validator_tier="low-cost",
            policy_mode="strict",
            blocked_count=2,
            confidence="high",
            summary="frontier-heavy | planner=frontier executor=fast validator=low-cost | policy=strict",
            trust_level="good",
        )
        defaults.update(kwargs)
        return NativeRoutingPreview(**defaults)

    def test_strategy(self):
        r = _build(routing_preview=self._preview())
        self.assertEqual(r.strategy, "frontier-heavy")

    def test_planner_tier(self):
        r = _build(routing_preview=self._preview())
        self.assertEqual(r.planner_tier, "frontier")

    def test_executor_tier(self):
        r = _build(routing_preview=self._preview())
        self.assertEqual(r.executor_tier, "fast")

    def test_validator_tier(self):
        r = _build(routing_preview=self._preview())
        self.assertEqual(r.validator_tier, "low-cost")

    def test_policy_mode(self):
        r = _build(routing_preview=self._preview())
        self.assertEqual(r.policy_mode, "strict")

    def test_blocked_count(self):
        r = _build(routing_preview=self._preview())
        self.assertEqual(r.blocked_count, 2)

    def test_confidence(self):
        r = _build(routing_preview=self._preview())
        self.assertEqual(r.confidence, "high")

    def test_summary(self):
        r = _build(routing_preview=self._preview())
        self.assertIn("frontier-heavy", r.summary)

    def test_trust_from_preview_fallback(self):
        # run_trust_score absent — falls back to routing_preview.trust_level
        r = _build(routing_preview=self._preview(trust_level="fair"))
        self.assertEqual(r.trust_level, "fair")

    def test_dict_routing_preview(self):
        # Builder accepts dict-style routing_preview (from deserialized history)
        rp_dict = {
            "strategy": "cost-balanced",
            "planner_tier": "fast",
            "executor_tier": "fast",
            "validator_tier": "fast",
            "policy_mode": "auto",
            "blocked_count": 0,
            "confidence": "medium",
            "summary": "cost-balanced",
            "trust_level": "weak",
        }
        r = _build(routing_preview=rp_dict)
        self.assertEqual(r.strategy, "cost-balanced")
        self.assertEqual(r.trust_level, "weak")


# ---------------------------------------------------------------------------
# Derives policy fields from model_policy_receipt
# ---------------------------------------------------------------------------

class TestBuildNativeRoutingReceiptFromModelPolicyReceipt(unittest.TestCase):

    def _receipt(self, **kwargs) -> NativeModelPolicyReceipt:
        defaults = dict(affected_selection=True, warnings_count=3)
        defaults.update(kwargs)
        return NativeModelPolicyReceipt(**defaults)

    def test_policy_affected_true(self):
        r = _build(model_policy_receipt=self._receipt(affected_selection=True))
        self.assertTrue(r.policy_affected)

    def test_policy_affected_false(self):
        r = _build(model_policy_receipt=self._receipt(affected_selection=False))
        self.assertFalse(r.policy_affected)

    def test_warnings_count(self):
        r = _build(model_policy_receipt=self._receipt(warnings_count=5))
        self.assertEqual(r.warnings_count, 5)

    def test_warnings_count_zero(self):
        r = _build(model_policy_receipt=self._receipt(warnings_count=0))
        self.assertEqual(r.warnings_count, 0)

    def test_dict_policy_receipt(self):
        r = _build(model_policy_receipt={"affected_selection": True, "warnings_count": 2})
        self.assertTrue(r.policy_affected)
        self.assertEqual(r.warnings_count, 2)


# ---------------------------------------------------------------------------
# Derives trust_level from run_trust_score
# ---------------------------------------------------------------------------

class TestBuildNativeRoutingReceiptFromRunTrustScore(unittest.TestCase):

    def test_trust_level_from_score(self):
        rts = NativeRunTrustScore(score=85, level="strong")
        r = _build(run_trust_score=rts)
        self.assertEqual(r.trust_level, "strong")

    def test_trust_level_overrides_preview(self):
        rts = NativeRunTrustScore(score=30, level="weak")
        preview = NativeRoutingPreview(trust_level="good")
        r = _build(routing_preview=preview, run_trust_score=rts)
        self.assertEqual(r.trust_level, "weak")

    def test_dict_run_trust_score(self):
        r = _build(run_trust_score={"level": "fair", "score": 55})
        self.assertEqual(r.trust_level, "fair")

    def test_no_trust_score_falls_back_to_unknown(self):
        r = _build(run_trust_score=None)
        self.assertEqual(r.trust_level, "unknown")


# ---------------------------------------------------------------------------
# JSON serialization via asdict
# ---------------------------------------------------------------------------

class TestNativeRoutingReceiptAsdict(unittest.TestCase):

    def test_json_serializable(self):
        r = NativeRoutingReceipt(
            strategy="frontier-heavy",
            planner_tier="frontier",
            executor_tier="fast",
            validator_tier="low-cost",
            policy_mode="strict",
            policy_affected=True,
            blocked_count=2,
            trust_level="good",
            confidence="high",
            warnings_count=1,
            summary="frontier-heavy | policy=strict",
        )
        d = asdict(r)
        raw = json.dumps(d)
        parsed = json.loads(raw)
        self.assertEqual(parsed["strategy"], "frontier-heavy")
        self.assertEqual(parsed["trust_level"], "good")
        self.assertTrue(parsed["policy_affected"])
        self.assertEqual(parsed["blocked_count"], 2)

    def test_asdict_round_trip(self):
        r = _build()
        d = asdict(r)
        self.assertIn("strategy", d)
        self.assertIn("warnings_count", d)


# ---------------------------------------------------------------------------
# Old entries without routing_receipt do not crash
# ---------------------------------------------------------------------------

class TestNativeMetaFromEntryRoutingReceipt(unittest.TestCase):

    def _native_entry(self, **extra) -> dict:
        base = {"workflow": "native", "executor": "native"}
        base.update(extra)
        return base

    def test_missing_routing_receipt_returns_none(self):
        meta = _native_meta_from_entry(self._native_entry())
        self.assertIsNone(getattr(meta, "routing_receipt", None))

    def test_with_routing_receipt_dict(self):
        entry = self._native_entry(routing_receipt={
            "strategy": "cost-balanced",
            "planner_tier": "fast",
            "executor_tier": "fast",
            "validator_tier": "fast",
            "policy_mode": "auto",
            "policy_affected": False,
            "blocked_count": 0,
            "trust_level": "good",
            "confidence": "medium",
            "warnings_count": 0,
            "summary": "cost-balanced",
        })
        meta = _native_meta_from_entry(entry)
        rr = getattr(meta, "routing_receipt", None)
        self.assertIsNotNone(rr)
        self.assertEqual(getattr(rr, "strategy", None), "cost-balanced")

    def test_non_native_entry_returns_none(self):
        meta = _native_meta_from_entry({"task": "x"})
        self.assertIsNone(meta)


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def _meta_with_receipt(**kwargs) -> SimpleNamespace:
    defaults = dict(
        strategy="frontier-heavy",
        planner_tier="frontier",
        executor_tier="fast",
        validator_tier="low-cost",
        policy_mode="strict",
        policy_affected=True,
        blocked_count=2,
        trust_level="good",
        confidence="high",
        warnings_count=1,
        summary="frontier-heavy | policy=strict",
    )
    defaults.update(kwargs)
    return _ns(routing_receipt=_ns(**defaults))


class TestNativeRoutingReceiptRendering(unittest.TestCase):

    def test_hidden_at_default_detail(self):
        meta = _meta_with_receipt()
        lines = _render_native_demo_block(meta, detail="default")
        combined = "\n".join(lines)
        self.assertNotIn("routing receipt", combined)

    def test_hidden_at_more(self):
        meta = _meta_with_receipt()
        lines = _render_native_demo_block(meta, detail="more")
        combined = "\n".join(lines)
        self.assertNotIn("routing receipt", combined)

    def test_full_section_at_full(self):
        meta = _meta_with_receipt()
        lines = _render_native_demo_block(meta, detail="full")
        combined = "\n".join(lines)
        self.assertIn("[routing receipt]", combined)
        self.assertIn("strategy:", combined)
        self.assertIn("planner:", combined)
        self.assertIn("executor:", combined)
        self.assertIn("validator:", combined)
        self.assertIn("policy_mode:", combined)
        self.assertIn("policy_affected:", combined)
        self.assertIn("blocked_count:", combined)
        self.assertIn("trust:", combined)
        self.assertIn("confidence:", combined)
        self.assertIn("warnings_count:", combined)
        self.assertIn("summary:", combined)

    def test_policy_affected_yes_no_formatting(self):
        meta_yes = _meta_with_receipt(policy_affected=True)
        meta_no = _meta_with_receipt(policy_affected=False)
        lines_yes = _render_native_demo_block(meta_yes, detail="full")
        lines_no = _render_native_demo_block(meta_no, detail="full")
        self.assertIn("yes", "\n".join(lines_yes))
        self.assertIn("no", "\n".join(lines_no))

    def test_does_not_expose_raw_warning_text(self):
        # warnings_count is an int; raw warning strings must not appear
        meta = _meta_with_receipt(warnings_count=3)
        for detail in ("more", "full"):
            lines = _render_native_demo_block(meta, detail=detail)
            combined = "\n".join(lines)
            self.assertNotIn("raw warning", combined)
            # warnings_count integer is shown, not a list of strings
            if detail == "full":
                self.assertIn("warnings_count:", combined)
                self.assertIn("3", combined)

    def test_none_receipt_produces_no_receipt_lines(self):
        meta = _ns(routing_receipt=None)
        lines = _render_native_demo_block(meta, detail="more")
        combined = "\n".join(lines)
        self.assertNotIn("routing receipt", combined)

    def test_dict_receipt_renders(self):
        # Simulates deserialized history entry — receipt is a plain dict
        receipt_dict = {
            "strategy": "cost-balanced",
            "planner_tier": "fast",
            "executor_tier": "fast",
            "validator_tier": "fast",
            "policy_mode": "auto",
            "policy_affected": False,
            "blocked_count": 0,
            "trust_level": "fair",
            "confidence": "medium",
            "warnings_count": 0,
            "summary": "cost-balanced",
        }
        meta = _ns(routing_receipt=receipt_dict)
        lines = _render_native_demo_block(meta, detail="full")
        combined = "\n".join(lines)
        self.assertIn("[routing receipt]", combined)
        self.assertIn("cost-balanced", combined)
