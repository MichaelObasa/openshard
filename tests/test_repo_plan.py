"""Tests for the pure repo-aware planner (openshard/planning/repo_plan.py)."""
from __future__ import annotations

import json
import unittest

from openshard.planning.repo_plan import (
    _safe_task_text,
    build_repo_aware_plan,
)

_UNSAFE = ("C:\\", "C:/", "/Users/", "/home/", "sk-", "AKIA")


def _repo_map(**summary) -> dict:
    base_summary = {
        "file_count": 1,
        "directory_count": 0,
        "languages": [],
        "frameworks": [],
        "package_managers": [],
        "test_commands": [],
    }
    base_summary.update(summary)
    return {
        "schema_version": "1",
        "source": "repo_map_v1",
        "git": {"branch": "main", "head_commit": "a" * 40, "dirty": False},
        "summary": base_summary,
        "important_files": [],
        "risky_areas": [],
        "warnings": [],
    }


class TestSafeTaskText(unittest.TestCase):
    def test_non_string_returns_placeholder(self):
        self.assertEqual(_safe_task_text(None), "OpenShard repo-aware plan")
        self.assertEqual(_safe_task_text(123), "OpenShard repo-aware plan")

    def test_normal_task_stays_readable(self):
        self.assertEqual(
            _safe_task_text("  Add tests for the auth module  "),
            "Add tests for the auth module",
        )

    def test_windows_absolute_path_redacted(self):
        out = _safe_task_text(r"fix bug in C:\Users\Michael\secret\app.py")
        self.assertNotIn("C:\\", out)
        self.assertNotIn("Michael", out)
        self.assertIn("<path>", out)
        self.assertIn("fix bug", out)

    def test_posix_absolute_path_redacted(self):
        for raw in ("/Users/michael/keys/id_rsa", "/home/michael/.env"):
            out = _safe_task_text(f"read {raw} please")
            self.assertNotIn("/Users/", out)
            self.assertNotIn("/home/", out)
            self.assertIn("<path>", out)

    def test_secret_like_tokens_redacted(self):
        cases = [
            "use sk-abcdef0123456789ABCDEF",
            "key AKIAIOSFODNN7EXAMPLE here",
            "token=ghp_supersecretvalue12345",
            "api_key: 0123456789abcdef0123456789abcdef0123",
        ]
        for raw in cases:
            out = _safe_task_text(raw)
            for needle in ("sk-", "AKIA", "ghp_supersecret", "0123456789abcdef0123456789abcdef"):
                self.assertNotIn(needle, out, msg=f"leaked from {raw!r}: {out!r}")

    def test_length_capped(self):
        out = _safe_task_text("word " * 200)
        self.assertLessEqual(len(out), 200)

    def test_empty_after_strip_uses_placeholder(self):
        self.assertEqual(_safe_task_text("   "), "OpenShard repo-aware plan")
        self.assertEqual(_safe_task_text("/Users/x"), "<path>")


class TestBuildRepoAwarePlan(unittest.TestCase):
    def test_repo_context_maps_from_map(self):
        rm = _repo_map(
            languages=["python"],
            frameworks=["flask"],
            package_managers=["pip"],
            test_commands=["python -m pytest"],
        )
        rm["important_files"] = ["pyproject.toml", "README.md"]
        plan = build_repo_aware_plan("add tests", rm, cache_hit=True)
        ctx = plan.repo_context
        self.assertEqual(ctx["languages"], ["python"])
        self.assertEqual(ctx["frameworks"], ["flask"])
        self.assertEqual(ctx["package_managers"], ["pip"])
        self.assertEqual(ctx["test_commands"], ["python -m pytest"])
        self.assertEqual(ctx["important_files"], ["pyproject.toml", "README.md"])
        self.assertTrue(ctx["cache_hit"])
        self.assertFalse(ctx["git_dirty"])

    def test_uses_test_command_in_steps(self):
        rm = _repo_map(languages=["python"], test_commands=["python -m pytest"])
        plan = build_repo_aware_plan("x", rm, cache_hit=False)
        joined = "\n".join(plan.plan_steps)
        self.assertIn("python -m pytest", joined)

    def test_no_test_command_generic_step_and_note(self):
        rm = _repo_map(languages=["python"])
        plan = build_repo_aware_plan("x", rm, cache_hit=False)
        self.assertTrue(any("No test command detected" in s for s in plan.plan_steps))
        self.assertTrue(any("No test command detected" in n for n in plan.safety_notes))

    def test_honest_wording_suggested_not_inspected(self):
        rm = _repo_map(languages=["python"])
        rm["important_files"] = ["pyproject.toml"]
        plan = build_repo_aware_plan("x", rm, cache_hit=False)
        joined = "\n".join(plan.plan_steps + plan.safety_notes)
        self.assertIn("suggested files", joined.lower())
        self.assertNotIn("files inspected", joined.lower())
        self.assertTrue(any("metadata-only" in n for n in plan.safety_notes))

    def test_dirty_repo_safety_note(self):
        rm = _repo_map()
        rm["git"]["dirty"] = True
        plan = build_repo_aware_plan("x", rm, cache_hit=False)
        self.assertTrue(plan.repo_context["git_dirty"])
        self.assertTrue(any("dirty" in n.lower() for n in plan.safety_notes))

    def test_non_git_safety_note(self):
        rm = _repo_map()
        rm["git"] = {"branch": None, "head_commit": None, "dirty": False}
        plan = build_repo_aware_plan("x", rm, cache_hit=False)
        self.assertTrue(any("not a git repository" in n.lower() for n in plan.safety_notes))

    def test_risky_areas_safety_note(self):
        rm = _repo_map()
        rm["risky_areas"] = ["auth.py", "config.py"]
        plan = build_repo_aware_plan("x", rm, cache_hit=False)
        self.assertTrue(any("risky" in n.lower() for n in plan.safety_notes))

    def test_no_stack_generic_note(self):
        rm = _repo_map()
        plan = build_repo_aware_plan("x", rm, cache_hit=False)
        self.assertTrue(any("No stack detected" in n for n in plan.safety_notes))

    def test_warnings_passed_through(self):
        rm = _repo_map()
        rm["warnings"] = ["dirty git tree; repo-map cache was rebuilt instead of reused"]
        plan = build_repo_aware_plan("x", rm, cache_hit=False)
        self.assertEqual(plan.warnings, rm["warnings"])

    def test_json_body_uses_safe_task(self):
        rm = _repo_map(languages=["python"])
        plan = build_repo_aware_plan(r"touch C:\Users\Michael\app.py sk-deadbeef0123456789", rm, cache_hit=False)
        body = plan.to_dict()
        text = json.dumps(body)
        for needle in _UNSAFE:
            self.assertNotIn(needle, text, msg=f"unsafe {needle!r} leaked: {text}")
        self.assertEqual(body["task"], plan.task)

    def test_no_leak_in_full_dict(self):
        rm = _repo_map(languages=["python"], test_commands=["python -m pytest"])
        rm["important_files"] = ["pyproject.toml"]
        rm["risky_areas"] = ["auth.py"]
        plan = build_repo_aware_plan("normal task", rm, cache_hit=True)
        text = json.dumps(plan.to_dict())
        for needle in _UNSAFE:
            self.assertNotIn(needle, text)


if __name__ == "__main__":
    unittest.main()
