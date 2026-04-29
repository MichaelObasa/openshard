from __future__ import annotations

import json
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from openshard.analysis.repo import RepoFacts
from openshard.cli.main import _log_run
from openshard.scoring.requirements import TaskRequirements
from openshard.scoring.scorer import ScoredRoutingResult, select_with_info
from openshard.providers.base import ModelInfo
from openshard.providers.manager import InventoryEntry
from openshard.skills.discovery import SkillDef
from openshard.skills.matcher import MatchedSkill


def _make_generator(model="openrouter/fast-model", fixer_model="openrouter/strong-model"):
    g = MagicMock()
    g.model = model
    g.fixer_model = fixer_model
    return g


def _make_scored(
    candidates=None,
    scores=None,
    used_fallback=False,
    category="standard",
) -> ScoredRoutingResult:
    ids = candidates or []
    return ScoredRoutingResult(
        category=category,
        requirements=TaskRequirements(),
        candidate_count=len(ids),
        selected_model=ids[0] if ids else None,
        selected_provider="openrouter",
        used_fallback=used_fallback,
        candidates=ids,
        scores=scores or {},
    )


def _make_repo_facts(**kwargs) -> RepoFacts:
    defaults = dict(
        languages=["python"],
        package_files=["pyproject.toml"],
        framework="pytest",
        test_command="pytest",
        risky_paths=["auth/tokens.py", "config/secrets.py"],
        changed_files=["src/foo.py"],
    )
    defaults.update(kwargs)
    return RepoFacts(**defaults)


def _make_entry(model_id: str) -> InventoryEntry:
    return InventoryEntry(
        provider="openrouter",
        model=ModelInfo(
            id=model_id,
            name=model_id,
            pricing={},
            context_window=None,
            max_output_tokens=None,
            supports_vision=False,
            supports_tools=False,
        ),
    )


class TestLogRunHistory(unittest.TestCase):

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self._tmpdir = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def _call(self, **kwargs):
        defaults = dict(
            start=time.time(),
            task="test task",
            generator=_make_generator(),
            retry_triggered=False,
            files=[],
            verification_attempted=False,
            verification_passed=None,
            workspace=None,
        )
        defaults.update(kwargs)
        with patch("openshard.cli.main.Path.cwd", return_value=self._tmpdir):
            _log_run(**defaults)

    def _read_entry(self) -> dict:
        log_path = self._tmpdir / ".openshard" / "runs.jsonl"
        lines = [ln for ln in log_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
        return json.loads(lines[-1])

    # --- routing_candidates ---

    def test_routing_candidates_stored(self):
        scored = _make_scored(
            candidates=["openrouter/fast", "openrouter/strong"],
            scores={"openrouter/fast": 12.0, "openrouter/strong": 11.5},
        )
        self._call(_scored=scored)
        entry = self._read_entry()
        self.assertEqual(entry["routing_candidates"], ["openrouter/fast", "openrouter/strong"])

    def test_routing_candidates_order_preserved(self):
        ids = ["openrouter/a", "openrouter/b", "openrouter/c"]
        scored = _make_scored(candidates=ids, scores={i: 10.0 for i in ids})
        self._call(_scored=scored)
        entry = self._read_entry()
        self.assertEqual(entry["routing_candidates"], ids)

    def test_routing_candidates_absent_on_fallback(self):
        scored = _make_scored(used_fallback=True)
        self._call(_scored=scored)
        entry = self._read_entry()
        self.assertNotIn("routing_candidates", entry)
        self.assertNotIn("routing_scores", entry)

    def test_routing_candidates_absent_when_no_scored(self):
        self._call()
        entry = self._read_entry()
        self.assertNotIn("routing_candidates", entry)

    # --- routing_scores ---

    def test_routing_scores_stored(self):
        scored = _make_scored(
            candidates=["openrouter/fast", "openrouter/strong"],
            scores={"openrouter/fast": 12.0, "openrouter/strong": 11.5},
        )
        self._call(_scored=scored)
        entry = self._read_entry()
        self.assertAlmostEqual(entry["routing_scores"]["openrouter/fast"], 12.0)
        self.assertAlmostEqual(entry["routing_scores"]["openrouter/strong"], 11.5)

    def test_routing_scores_absent_when_no_scored(self):
        self._call()
        entry = self._read_entry()
        self.assertNotIn("routing_scores", entry)

    # --- repo_facts ---

    def test_repo_facts_stored(self):
        facts = _make_repo_facts()
        self._call(repo_facts=facts)
        entry = self._read_entry()
        rf = entry["repo_facts"]
        self.assertEqual(rf["languages"], ["python"])
        self.assertEqual(rf["package_files"], ["pyproject.toml"])
        self.assertEqual(rf["framework"], "pytest")
        self.assertEqual(rf["test_command"], "pytest")

    def test_repo_facts_counts_and_samples(self):
        facts = _make_repo_facts(
            risky_paths=["a.py", "b.py", "c.py", "d.py"],
            changed_files=["x.py", "y.py", "z.py", "w.py"],
        )
        self._call(repo_facts=facts)
        entry = self._read_entry()
        rf = entry["repo_facts"]
        self.assertEqual(rf["risky_paths_count"], 4)
        self.assertEqual(len(rf["risky_paths_sample"]), 3)
        self.assertEqual(rf["changed_files_count"], 4)
        self.assertEqual(len(rf["changed_files_sample"]), 3)

    def test_repo_facts_sample_does_not_exceed_three(self):
        facts = _make_repo_facts(risky_paths=["a", "b", "c", "d", "e"])
        self._call(repo_facts=facts)
        entry = self._read_entry()
        self.assertLessEqual(len(entry["repo_facts"]["risky_paths_sample"]), 3)

    def test_repo_facts_none_framework_stored_as_null(self):
        facts = _make_repo_facts(framework=None, test_command=None)
        self._call(repo_facts=facts)
        entry = self._read_entry()
        self.assertIsNone(entry["repo_facts"]["framework"])
        self.assertIsNone(entry["repo_facts"]["test_command"])

    def test_repo_facts_absent_when_none(self):
        self._call()
        entry = self._read_entry()
        self.assertNotIn("repo_facts", entry)

    # --- backward compatibility ---

    def test_core_fields_always_present(self):
        self._call()
        entry = self._read_entry()
        for key in ("timestamp", "task", "execution_model", "retry_triggered",
                    "duration_seconds", "verification_attempted"):
            self.assertIn(key, entry)

    def test_no_new_fields_without_new_params(self):
        self._call()
        entry = self._read_entry()
        self.assertNotIn("routing_candidates", entry)
        self.assertNotIn("routing_scores", entry)
        self.assertNotIn("repo_facts", entry)

    def test_multiple_entries_appended(self):
        self._call(task="first")
        self._call(task="second")
        log_path = self._tmpdir / ".openshard" / "runs.jsonl"
        lines = [ln for ln in log_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
        self.assertEqual(len(lines), 2)
        self.assertEqual(json.loads(lines[0])["task"], "first")
        self.assertEqual(json.loads(lines[1])["task"], "second")

    # --- matched_skills ---

    def _make_skill(self, slug: str) -> SkillDef:
        return SkillDef(
            slug=slug, name=slug, description="", category="standard",
            keywords=[], languages=[], framework=None,
        )

    def test_matched_skills_logged(self):
        skills = [
            MatchedSkill(skill=self._make_skill("pytest-helper"), reasons=["keyword:pytest"]),
            MatchedSkill(skill=self._make_skill("django-views"), reasons=["framework:django", "category:standard"]),
        ]
        self._call(matched_skills=skills)
        entry = self._read_entry()
        self.assertEqual(entry["matched_skills"], ["pytest-helper", "django-views"])
        self.assertEqual(entry["matched_skill_reasons"]["pytest-helper"], ["keyword:pytest"])
        self.assertEqual(entry["matched_skill_reasons"]["django-views"], ["framework:django", "category:standard"])

    def test_no_skills_no_fields(self):
        self._call(matched_skills=[])
        entry = self._read_entry()
        self.assertNotIn("matched_skills", entry)
        self.assertNotIn("matched_skill_reasons", entry)

    # --- select_with_info integration ---

    def test_select_with_info_populates_candidates(self):
        entries = [_make_entry("openrouter/model-a"), _make_entry("openrouter/model-b")]
        result = select_with_info(entries, TaskRequirements(), "standard")
        self.assertGreater(len(result.candidates), 0)
        self.assertIn(result.selected_model, result.candidates)

    def test_select_with_info_populates_scores(self):
        entries = [_make_entry("openrouter/model-a"), _make_entry("openrouter/model-b")]
        result = select_with_info(entries, TaskRequirements(), "standard")
        self.assertGreater(len(result.scores), 0)
        self.assertIn(result.selected_model, result.scores)
        for score in result.scores.values():
            self.assertIsInstance(score, float)

    def test_select_with_info_fallback_has_empty_candidates_and_scores(self):
        entry = _make_entry("openrouter/no-vision")
        reqs = TaskRequirements(needs_vision=True)
        result = select_with_info([entry], reqs, "visual")
        self.assertTrue(result.used_fallback)
        self.assertEqual(result.candidates, [])
        self.assertEqual(result.scores, {})

    def test_select_with_info_scores_are_rounded(self):
        entries = [_make_entry("openrouter/model-a")]
        result = select_with_info(entries, TaskRequirements(), "standard")
        for score in result.scores.values():
            self.assertEqual(round(score, 3), score)

    # --- execution_profile ---

    def test_execution_profile_logged(self):
        from openshard.routing.profiles import ProfileDecision
        pd = ProfileDecision(profile="native_deep", reason="security category")
        self._call(profile_decision=pd)
        entry = self._read_entry()
        self.assertEqual(entry["execution_profile"], "native_deep")
        self.assertEqual(entry["execution_profile_reason"], "security category")

    def test_execution_profile_absent_when_none(self):
        self._call()
        entry = self._read_entry()
        self.assertNotIn("execution_profile", entry)
        self.assertNotIn("execution_profile_reason", entry)

    def test_execution_profile_light_logged(self):
        from openshard.routing.profiles import ProfileDecision
        pd = ProfileDecision(profile="native_light", reason="simple/safe task")
        self._call(profile_decision=pd)
        entry = self._read_entry()
        self.assertEqual(entry["execution_profile"], "native_light")

    def test_execution_profile_swarm_logged(self):
        from openshard.routing.profiles import ProfileDecision
        pd = ProfileDecision(profile="native_swarm", reason="explicit override")
        self._call(profile_decision=pd)
        entry = self._read_entry()
        self.assertEqual(entry["execution_profile"], "native_swarm")
        self.assertEqual(entry["execution_profile_reason"], "explicit override")


class TestVerificationPlanLogging(unittest.TestCase):

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self._tmpdir = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def _call(self, **kwargs):
        defaults = dict(
            start=time.time(),
            task="test task",
            generator=_make_generator(),
            retry_triggered=False,
            files=[],
            verification_attempted=False,
            verification_passed=None,
            workspace=None,
        )
        defaults.update(kwargs)
        with patch("openshard.cli.main.Path.cwd", return_value=self._tmpdir):
            _log_run(**defaults)

    def _read_entry(self) -> dict:
        log_path = self._tmpdir / ".openshard" / "runs.jsonl"
        lines = [ln for ln in log_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
        return json.loads(lines[-1])

    def _make_plan(self):
        from openshard.verification.plan import (
            CommandSafety, VerificationCommand, VerificationKind,
            VerificationPlan, VerificationSource,
        )
        cmd = VerificationCommand(
            name="tests",
            argv=["python", "-m", "pytest"],
            kind=VerificationKind.test,
            source=VerificationSource.detected,
            safety=CommandSafety.safe,
            reason="matches safe prefix: python -m pytest",
        )
        return VerificationPlan(commands=[cmd])

    def test_plan_with_commands_stored(self):
        self._call(verification_plan=self._make_plan())
        entry = self._read_entry()
        self.assertIn("verification_plan", entry)
        self.assertEqual(len(entry["verification_plan"]), 1)

    def test_stored_fields_match_command(self):
        self._call(verification_plan=self._make_plan())
        entry = self._read_entry()
        vc = entry["verification_plan"][0]
        self.assertEqual(vc["name"], "tests")
        self.assertEqual(vc["argv"], ["python", "-m", "pytest"])
        self.assertEqual(vc["kind"], "test")
        self.assertEqual(vc["source"], "detected")
        self.assertEqual(vc["safety"], "safe")
        self.assertIn("pytest", vc["reason"])

    def test_empty_plan_not_stored(self):
        from openshard.verification.plan import VerificationPlan
        self._call(verification_plan=VerificationPlan())
        entry = self._read_entry()
        self.assertNotIn("verification_plan", entry)

    def test_none_plan_not_stored(self):
        self._call(verification_plan=None)
        entry = self._read_entry()
        self.assertNotIn("verification_plan", entry)

    def test_existing_fields_unaffected(self):
        self._call(task="my task", verification_plan=self._make_plan())
        entry = self._read_entry()
        self.assertEqual(entry["task"], "my task")
        self.assertIn("timestamp", entry)
        self.assertIn("execution_model", entry)


if __name__ == "__main__":
    unittest.main()
