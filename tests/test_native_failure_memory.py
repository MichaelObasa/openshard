"""Tests for NativeFailureMemory — lesson creation, serialization, and reconstruction."""

from __future__ import annotations

import json
import unittest
from dataclasses import asdict
from types import SimpleNamespace

from openshard.native.context import (
    NativeFailureMemory,
    build_native_failure_memory,
)


def _ns(**kwargs):
    return SimpleNamespace(**kwargs)


def _quality_score(level: str, score: int = 50):
    return _ns(level=level, score=score, warnings=[])


def _clarification(needed: bool):
    return _ns(needed=needed)


def _verification_loop(attempted: bool, passed: bool, truncated: bool = False):
    return _ns(attempted=attempted, passed=passed, truncated=truncated)


def _command_policy(blocked_count: int, needs_approval_count: int = 0):
    return _ns(blocked_count=blocked_count, needs_approval_count=needs_approval_count, warnings=[])


def _approval_request(requires_approval: bool, reason: str = "reason"):
    return _ns(requires_approval=requires_approval, reason=reason)


def _approval_receipt(granted: bool):
    return _ns(granted=granted)


def _budget_preview(would_exceed: bool, proposed: int = 5, budget: int = 3):
    return _ns(would_exceed_budget=would_exceed, proposed_files=proposed, budget_max_files=budget)


def _verification_plan(has_commands: bool):
    cmds = ["pytest"] if has_commands else []
    return _ns(suggested_verification_commands=cmds)


def _usage_summary(any_truncated: bool = False, components=None, warn_count: int = 0):
    return _ns(
        any_truncated=any_truncated,
        truncated_components=components or [],
        failure_warning_count=warn_count,
    )


def _build(**kwargs) -> NativeFailureMemory:
    defaults = dict(
        context_quality_score=None,
        clarification_request=None,
        verification_loop=None,
        command_policy_preview=None,
        approval_request=None,
        approval_receipt=None,
        change_budget_preview=None,
        verification_plan=None,
        context_usage_summary=None,
    )
    defaults.update(kwargs)
    return build_native_failure_memory(**defaults)


class TestLessonCreation(unittest.TestCase):
    def test_no_lessons_when_all_none(self):
        fm = _build()
        self.assertFalse(fm.has_lessons)
        self.assertEqual(fm.lessons, [])

    def test_no_lessons_when_all_clean(self):
        fm = _build(
            context_quality_score=_quality_score("good"),
            clarification_request=_clarification(False),
            verification_loop=_verification_loop(True, True),
            command_policy_preview=_command_policy(0),
            approval_request=_approval_request(False),
            change_budget_preview=_budget_preview(False),
            verification_plan=_verification_plan(True),
            context_usage_summary=_usage_summary(),
        )
        self.assertFalse(fm.has_lessons)
        self.assertEqual(fm.lessons, [])

    def test_weak_context_lesson(self):
        fm = _build(context_quality_score=_quality_score("weak", score=28))
        self.assertTrue(fm.has_lessons)
        self.assertEqual(len(fm.lessons), 1)
        self.assertEqual(fm.lessons[0].lesson_type, "weak_context")
        self.assertIn("28", fm.lessons[0].reason)

    def test_fair_context_does_not_trigger_weak_context(self):
        fm = _build(context_quality_score=_quality_score("fair", score=40))
        self.assertFalse(fm.has_lessons)

    def test_unknown_task_type_lesson(self):
        fm = _build(clarification_request=_clarification(True))
        self.assertTrue(fm.has_lessons)
        self.assertEqual(fm.lessons[0].lesson_type, "unknown_task_type")

    def test_clarification_not_needed_no_lesson(self):
        fm = _build(clarification_request=_clarification(False))
        self.assertFalse(fm.has_lessons)

    def test_failed_verification_lesson(self):
        fm = _build(verification_loop=_verification_loop(attempted=True, passed=False))
        self.assertEqual(fm.lessons[0].lesson_type, "failed_verification")

    def test_passed_verification_no_lesson(self):
        fm = _build(verification_loop=_verification_loop(attempted=True, passed=True))
        self.assertFalse(fm.has_lessons)

    def test_not_attempted_verification_no_lesson(self):
        fm = _build(verification_loop=_verification_loop(attempted=False, passed=False))
        self.assertFalse(fm.has_lessons)

    def test_unsafe_command_lesson(self):
        fm = _build(command_policy_preview=_command_policy(blocked_count=2))
        self.assertEqual(fm.lessons[0].lesson_type, "unsafe_command")
        self.assertIn("2", fm.lessons[0].reason)

    def test_zero_blocked_no_unsafe_lesson(self):
        fm = _build(command_policy_preview=_command_policy(blocked_count=0))
        self.assertFalse(fm.has_lessons)

    def test_approval_required_lesson(self):
        fm = _build(approval_request=_approval_request(True, reason="exceeds budget"))
        self.assertEqual(fm.lessons[0].lesson_type, "approval_required")
        self.assertIn("exceeds budget", fm.lessons[0].reason)

    def test_approval_not_required_no_lesson(self):
        fm = _build(approval_request=_approval_request(False))
        self.assertFalse(fm.has_lessons)

    def test_approval_rejected_lesson(self):
        fm = _build(
            approval_request=_approval_request(True),
            approval_receipt=_approval_receipt(granted=False),
        )
        types = [lesson.lesson_type for lesson in fm.lessons]
        self.assertIn("approval_rejected", types)

    def test_approval_rejected_not_triggered_without_request(self):
        fm = _build(
            approval_request=_approval_request(False),
            approval_receipt=_approval_receipt(granted=False),
        )
        types = [lesson.lesson_type for lesson in fm.lessons]
        self.assertNotIn("approval_rejected", types)

    def test_approval_rejected_not_triggered_when_granted(self):
        fm = _build(
            approval_request=_approval_request(True),
            approval_receipt=_approval_receipt(granted=True),
        )
        types = [lesson.lesson_type for lesson in fm.lessons]
        self.assertNotIn("approval_rejected", types)

    def test_patch_too_broad_lesson(self):
        fm = _build(change_budget_preview=_budget_preview(True, proposed=5, budget=3))
        self.assertEqual(fm.lessons[0].lesson_type, "patch_too_broad")
        self.assertIn("5", fm.lessons[0].reason)
        self.assertIn("3", fm.lessons[0].reason)

    def test_within_budget_no_lesson(self):
        fm = _build(change_budget_preview=_budget_preview(False))
        self.assertFalse(fm.has_lessons)

    def test_missing_verification_lesson(self):
        fm = _build(verification_plan=_verification_plan(has_commands=False))
        self.assertEqual(fm.lessons[0].lesson_type, "missing_verification")

    def test_has_commands_no_missing_verification(self):
        fm = _build(verification_plan=_verification_plan(has_commands=True))
        self.assertFalse(fm.has_lessons)

    def test_context_truncated_lesson(self):
        fm = _build(context_usage_summary=_usage_summary(any_truncated=True, components=["evidence"]))
        self.assertEqual(fm.lessons[0].lesson_type, "context_truncated")
        self.assertIn("evidence", fm.lessons[0].reason)

    def test_not_truncated_no_lesson(self):
        fm = _build(context_usage_summary=_usage_summary(any_truncated=False))
        self.assertFalse(fm.has_lessons)

    def test_warnings_present_lesson(self):
        fm = _build(context_usage_summary=_usage_summary(warn_count=3))
        self.assertEqual(fm.lessons[0].lesson_type, "warnings_present")
        self.assertIn("3", fm.lessons[0].reason)

    def test_zero_warnings_no_lesson(self):
        fm = _build(context_usage_summary=_usage_summary(warn_count=0))
        self.assertFalse(fm.has_lessons)

    def test_multiple_lessons_accumulate(self):
        fm = _build(
            context_quality_score=_quality_score("weak", 20),
            verification_plan=_verification_plan(False),
        )
        self.assertEqual(len(fm.lessons), 2)
        self.assertEqual(fm.lessons[0].lesson_type, "weak_context")
        self.assertEqual(fm.lessons[1].lesson_type, "missing_verification")


class TestSerialization(unittest.TestCase):
    def test_asdict_roundtrip_no_lessons(self):
        fm = _build()
        d = asdict(fm)
        self.assertFalse(d["has_lessons"])
        self.assertEqual(d["lessons"], [])

    def test_asdict_roundtrip_with_lessons(self):
        fm = _build(context_quality_score=_quality_score("weak", 25))
        d = asdict(fm)
        self.assertTrue(d["has_lessons"])
        self.assertEqual(len(d["lessons"]), 1)
        self.assertEqual(d["lessons"][0]["lesson_type"], "weak_context")
        self.assertIn("25", d["lessons"][0]["reason"])

    def test_json_serializable_no_lessons(self):
        fm = _build()
        json.dumps(asdict(fm))  # must not raise

    def test_json_serializable_with_lessons(self):
        fm = _build(
            context_quality_score=_quality_score("weak"),
            verification_plan=_verification_plan(False),
        )
        json.dumps(asdict(fm))  # must not raise

    def test_empty_memory_serializes_cleanly(self):
        fm = NativeFailureMemory()
        d = asdict(fm)
        self.assertEqual(d, {"lessons": [], "has_lessons": False})


class TestSavedRunReconstruction(unittest.TestCase):
    def _entry_with_failure_memory(self, lessons: list[dict]) -> dict:
        return {
            "workflow": "native",
            "executor": "native",
            "failure_memory": {
                "has_lessons": bool(lessons),
                "lessons": lessons,
            },
        }

    def test_entry_missing_failure_memory_is_safe(self):
        from openshard.cli.run_output import _native_meta_from_entry
        entry = {"workflow": "native", "executor": "native"}
        nm = _native_meta_from_entry(entry)
        self.assertIsNone(getattr(nm, "failure_memory", None))

    def test_entry_with_failure_memory_reconstructed(self):
        from openshard.cli.run_output import _native_meta_from_entry
        entry = self._entry_with_failure_memory([
            {"lesson_type": "weak_context", "reason": "score 20/100"},
        ])
        nm = _native_meta_from_entry(entry)
        fm = getattr(nm, "failure_memory", None)
        self.assertIsNotNone(fm)
        has_lessons = getattr(fm, "has_lessons", False)
        self.assertTrue(has_lessons)
        lessons = getattr(fm, "lessons", [])
        self.assertEqual(len(lessons), 1)

    def test_entry_with_no_lessons_reconstructed(self):
        from openshard.cli.run_output import _native_meta_from_entry
        entry = self._entry_with_failure_memory([])
        nm = _native_meta_from_entry(entry)
        fm = getattr(nm, "failure_memory", None)
        self.assertIsNotNone(fm)
        lessons = getattr(fm, "lessons", [])
        self.assertEqual(lessons, [])

    def test_non_native_entry_returns_none(self):
        from openshard.cli.run_output import _native_meta_from_entry
        entry = {"workflow": "opencode", "executor": "opencode"}
        nm = _native_meta_from_entry(entry)
        self.assertIsNone(nm)
