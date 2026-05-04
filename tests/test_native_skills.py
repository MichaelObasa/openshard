from __future__ import annotations

import unittest

from openshard.native.skills import (
    NativeSkill,
    NativeSkillMatch,
    list_builtin_skills,
    match_builtin_skills,
    selected_skill_names,
)

_EXPECTED_SKILL_NAMES = [
    "core-engineering-discipline",
    "repo-map",
    "context-discipline",
    "token-discipline",
    "safe-file-editing",
    "test-discovery",
    "verification-fix-loop",
    "diff-review",
    "security-sensitive-change",
]


class TestListBuiltinSkills(unittest.TestCase):
    def test_nonempty(self):
        self.assertGreater(len(list_builtin_skills()), 0)

    def test_contains_expected_skills(self):
        names = [s.name for s in list_builtin_skills()]
        for expected in _EXPECTED_SKILL_NAMES:
            self.assertIn(expected, names)

    def test_all_builtin_flag_set(self):
        for skill in list_builtin_skills():
            self.assertTrue(skill.built_in)

    def test_returns_copy(self):
        a = list_builtin_skills()
        b = list_builtin_skills()
        self.assertIsNot(a, b)


class TestMatchBuiltinSkills(unittest.TestCase):
    def test_match_selects_test_discovery(self):
        matches = match_builtin_skills("run the tests and fix failures")
        names = [m.skill.name for m in matches]
        self.assertIn("test-discovery", names)

    def test_match_selects_security_for_auth_tasks(self):
        matches = match_builtin_skills("update the auth and payment IAM config", max_skills=5)
        names = [m.skill.name for m in matches]
        self.assertIn("security-sensitive-change", names)

    def test_security_skill_scores_high(self):
        matches = match_builtin_skills("update auth and payment iam credential", max_skills=5)
        top = matches[0].skill.name
        self.assertEqual(top, "security-sensitive-change")

    def test_respects_max_skills(self):
        matches = match_builtin_skills("update auth and payment iam credential", max_skills=1)
        self.assertLessEqual(len(matches), 1)

    def test_max_skills_zero_returns_empty(self):
        matches = match_builtin_skills("run all tests and fix auth", max_skills=0)
        self.assertEqual(matches, [])

    def test_empty_task_returns_no_matches(self):
        self.assertEqual(match_builtin_skills(""), [])

    def test_whitespace_task_returns_no_matches(self):
        self.assertEqual(match_builtin_skills("   "), [])

    def test_returns_list_of_native_skill_match(self):
        matches = match_builtin_skills("run the tests")
        for m in matches:
            self.assertIsInstance(m, NativeSkillMatch)
            self.assertIsInstance(m.skill, NativeSkill)
            self.assertIsInstance(m.reason, str)
            self.assertIsInstance(m.score, float)

    def test_sort_is_deterministic(self):
        a = match_builtin_skills("run all tests and fix auth", max_skills=5)
        b = match_builtin_skills("run all tests and fix auth", max_skills=5)
        self.assertEqual([m.skill.name for m in a], [m.skill.name for m in b])

    def test_repo_facts_accepted(self):
        from openshard.analysis.repo import RepoFacts
        repo = RepoFacts(
            languages=["python"], package_files=[], framework=None,
            test_command=None, risky_paths=[], changed_files=[],
        )
        matches = match_builtin_skills("run the tests", repo_facts=repo)
        self.assertIsInstance(matches, list)


class TestSelectedSkillNames(unittest.TestCase):
    def test_returns_names(self):
        skills = list_builtin_skills()
        matches = [
            NativeSkillMatch(skill=skills[0], reason="test", score=1.0),
            NativeSkillMatch(skill=skills[1], reason="test", score=0.5),
        ]
        names = selected_skill_names(matches)
        self.assertEqual(names, [skills[0].name, skills[1].name])

    def test_empty_matches_returns_empty(self):
        self.assertEqual(selected_skill_names([]), [])
