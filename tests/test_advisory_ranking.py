from __future__ import annotations

import unittest

from openshard.routing.executor_advisory import (
    AdvisoryCandidate,
    ExecutorAdvisoryResult,
    rank_executors,
    render_executor_advisory,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _rank(**kwargs) -> ExecutorAdvisoryResult:
    defaults = dict(
        task="add a helper function",
        category="standard",
        risk_level="low",
        read_only=False,
        opencode_available=False,
        opencode_preference=False,
        risky_paths=None,
    )
    defaults.update(kwargs)
    return rank_executors(**defaults)


# ---------------------------------------------------------------------------
# 1. Native is highest-scored for a generic write task
# ---------------------------------------------------------------------------

class TestNativeDefaultForWriteTask(unittest.TestCase):
    def test_native_is_recommended_for_generic_write(self):
        result = _rank(task="add a helper function", category="standard")
        self.assertEqual(result.recommended.executor, "native")

    def test_native_score_above_alternatives_for_write(self):
        result = _rank(task="add a helper function", category="standard")
        for alt in result.alternatives:
            self.assertGreaterEqual(result.recommended.score, alt.score)


# ---------------------------------------------------------------------------
# 2. Read-only task ranks direct/plan-safe path at least as high as expected
# ---------------------------------------------------------------------------

class TestReadOnlyTaskRanking(unittest.TestCase):
    def test_direct_scores_highly_for_readonly(self):
        result = _rank(task="explain the auth module", category="standard", read_only=True)
        executors = {c.executor: c.score for c in [result.recommended] + result.alternatives}
        self.assertGreaterEqual(executors["direct"], 60.0)

    def test_native_remains_viable_for_readonly(self):
        result = _rank(task="what does this do", category="standard", read_only=True)
        executors = {c.executor: c.score for c in [result.recommended] + result.alternatives}
        self.assertIn("native", executors)


# ---------------------------------------------------------------------------
# 3. Security/auth task adds warning and boosts native
# ---------------------------------------------------------------------------

class TestSecurityTaskHandling(unittest.TestCase):
    def test_security_task_native_has_warning(self):
        result = _rank(task="fix the auth middleware", category="security", risk_level="high")
        nat = next(c for c in [result.recommended] + result.alternatives if c.executor == "native")
        self.assertTrue(
            any("policy" in w or "review" in w or "risk" in w for w in nat.warnings + result.warnings)
            or any("policy" in r or "check" in r or "enforce" in r for r in nat.reasons),
            "Expected native to have policy/check reasons for security task",
        )

    def test_security_task_native_outranks_direct(self):
        result = _rank(task="fix auth logic", category="security", risk_level="high", read_only=False)
        all_cands = {c.executor: c.score for c in [result.recommended] + result.alternatives}
        self.assertGreater(all_cands["native"], all_cands["direct"])


# ---------------------------------------------------------------------------
# 4. OpenCode unavailable → visible but score ≤ 50, warning present
# ---------------------------------------------------------------------------

class TestOpenCodeUnavailable(unittest.TestCase):
    def test_opencode_candidate_present_when_unavailable(self):
        result = _rank(opencode_available=False)
        executors = [c.executor for c in [result.recommended] + result.alternatives]
        self.assertIn("opencode", executors)

    def test_opencode_score_low_when_unavailable(self):
        result = _rank(opencode_available=False)
        oc = next(c for c in [result.recommended] + result.alternatives if c.executor == "opencode")
        self.assertLessEqual(oc.score, 50.0)

    def test_opencode_has_warning_when_unavailable(self):
        result = _rank(opencode_available=False)
        oc = next(c for c in [result.recommended] + result.alternatives if c.executor == "opencode")
        self.assertTrue(
            any("install" in w or "not found" in w or "PATH" in w for w in oc.warnings),
            f"Expected install warning, got: {oc.warnings}",
        )

    def test_opencode_available_false_on_candidate(self):
        result = _rank(opencode_available=False)
        oc = next(c for c in [result.recommended] + result.alternatives if c.executor == "opencode")
        self.assertFalse(oc.available)


# ---------------------------------------------------------------------------
# 5. OpenCode available → score > 50
# ---------------------------------------------------------------------------

class TestOpenCodeAvailable(unittest.TestCase):
    def test_opencode_score_above_50_when_available(self):
        result = _rank(opencode_available=True)
        oc = next(c for c in [result.recommended] + result.alternatives if c.executor == "opencode")
        self.assertGreater(oc.score, 50.0)

    def test_opencode_available_true_on_candidate(self):
        result = _rank(opencode_available=True)
        oc = next(c for c in [result.recommended] + result.alternatives if c.executor == "opencode")
        self.assertTrue(oc.available)


# ---------------------------------------------------------------------------
# 6. Explicit opencode preference boosts score when available
# ---------------------------------------------------------------------------

class TestOpenCodePreferenceAvailable(unittest.TestCase):
    def test_preference_boosts_opencode_score_when_available(self):
        without = _rank(opencode_available=True, opencode_preference=False)
        with_pref = _rank(opencode_available=True, opencode_preference=True)
        oc_without = next(c for c in [without.recommended] + without.alternatives if c.executor == "opencode")
        oc_with = next(c for c in [with_pref.recommended] + with_pref.alternatives if c.executor == "opencode")
        self.assertGreater(oc_with.score, oc_without.score)

    def test_preference_includes_explicit_preference_reason(self):
        result = _rank(opencode_available=True, opencode_preference=True)
        oc = next(c for c in [result.recommended] + result.alternatives if c.executor == "opencode")
        self.assertTrue(
            any("preference" in r.lower() for r in oc.reasons),
            f"Expected preference reason, got: {oc.reasons}",
        )


# ---------------------------------------------------------------------------
# 7. OpenCode preference still warns when unavailable
# ---------------------------------------------------------------------------

class TestOpenCodePreferenceUnavailable(unittest.TestCase):
    def test_preference_with_unavailable_still_warns(self):
        result = _rank(opencode_available=False, opencode_preference=True)
        oc = next(c for c in [result.recommended] + result.alternatives if c.executor == "opencode")
        self.assertTrue(
            any("install" in w or "not found" in w or "PATH" in w for w in oc.warnings),
            f"Expected install warning even with preference, got: {oc.warnings}",
        )

    def test_preference_with_unavailable_keeps_score_low(self):
        result = _rank(opencode_available=False, opencode_preference=True)
        oc = next(c for c in [result.recommended] + result.alternatives if c.executor == "opencode")
        self.assertLessEqual(oc.score, 50.0)


# ---------------------------------------------------------------------------
# 8. Ranking is deterministic
# ---------------------------------------------------------------------------

class TestDeterminism(unittest.TestCase):
    def test_same_inputs_produce_same_ranking(self):
        kwargs = dict(
            task="refactor the auth module",
            category="security",
            risk_level="high",
            read_only=False,
            opencode_available=True,
            opencode_preference=False,
            risky_paths=["src/auth.py"],
        )
        r1 = rank_executors(**kwargs)
        r2 = rank_executors(**kwargs)
        order1 = [r1.recommended.executor] + [a.executor for a in r1.alternatives]
        order2 = [r2.recommended.executor] + [a.executor for a in r2.alternatives]
        self.assertEqual(order1, order2)
        scores1 = [r1.recommended.score] + [a.score for a in r1.alternatives]
        scores2 = [r2.recommended.score] + [a.score for a in r2.alternatives]
        self.assertEqual(scores1, scores2)


# ---------------------------------------------------------------------------
# 9. Reasons are short strings (< 80 chars each)
# ---------------------------------------------------------------------------

class TestReasonLength(unittest.TestCase):
    def test_all_reasons_under_80_chars(self):
        result = _rank(
            task="implement new payment endpoint",
            category="security",
            risk_level="high",
            opencode_available=True,
        )
        for cand in [result.recommended] + result.alternatives:
            for reason in cand.reasons:
                self.assertLess(
                    len(reason), 80,
                    f"Reason too long ({len(reason)} chars): {reason!r}",
                )


# ---------------------------------------------------------------------------
# 10. Scores within 0–100
# ---------------------------------------------------------------------------

class TestScoreBounds(unittest.TestCase):
    def test_all_scores_within_bounds(self):
        for category in ("standard", "security", "complex", "boilerplate", "visual"):
            result = _rank(task="do something", category=category, opencode_available=True)
            for cand in [result.recommended] + result.alternatives:
                self.assertGreaterEqual(cand.score, 0.0, f"Score below 0 for {cand.executor}")
                self.assertLessEqual(cand.score, 100.0, f"Score above 100 for {cand.executor}")


# ---------------------------------------------------------------------------
# 11 & 12. No provider calls, no subprocess (rank_executors is pure)
# ---------------------------------------------------------------------------

class TestPurity(unittest.TestCase):
    def test_rank_executors_accepts_bool_input_not_subprocess(self):
        # rank_executors takes opencode_available as a plain bool — it never
        # calls detect_opencode() or runs a subprocess internally.
        import inspect

        import openshard.routing.executor_advisory as mod
        src = inspect.getsource(mod.rank_executors)
        # Check for actual subprocess call patterns, not just the word in a docstring
        self.assertNotIn("subprocess.run", src)
        self.assertNotIn("subprocess.Popen", src)
        self.assertNotIn("subprocess.call", src)
        self.assertNotIn("subprocess.check_output", src)
        self.assertNotIn("detect_opencode", src)
        self.assertNotIn("shutil.which", src)

    def test_rank_executors_has_no_network_imports(self):
        import inspect

        import openshard.routing.executor_advisory as mod
        src = inspect.getsource(mod)
        self.assertNotIn("urllib", src)
        self.assertNotIn("requests", src)
        self.assertNotIn("httpx", src)
        self.assertNotIn("aiohttp", src)


# ---------------------------------------------------------------------------
# 13. Rendered output does not expose raw prompts or command output
# ---------------------------------------------------------------------------

class TestRenderedOutput(unittest.TestCase):
    def test_render_does_not_contain_raw_task(self):
        task = "supersecret internal task description"
        result = _rank(task=task, opencode_available=True)
        rendered = "\n".join(render_executor_advisory(result))
        self.assertNotIn(task, rendered)

    def test_render_contains_advisory_only_notice(self):
        result = _rank()
        rendered = "\n".join(render_executor_advisory(result))
        self.assertIn("advisory only", rendered.lower())

    def test_render_contains_recommended_label(self):
        result = _rank()
        rendered = "\n".join(render_executor_advisory(result))
        self.assertIn("Recommended", rendered)


# ---------------------------------------------------------------------------
# 14. advisory_only always True
# ---------------------------------------------------------------------------

class TestAdvisoryOnlyFlag(unittest.TestCase):
    def test_advisory_only_is_always_true(self):
        for opencode in (True, False):
            result = _rank(opencode_available=opencode)
            self.assertIs(result.advisory_only, True)


# ---------------------------------------------------------------------------
# 15. version is always executor_advisory_v2
# ---------------------------------------------------------------------------

class TestVersionField(unittest.TestCase):
    def test_version_always_executor_advisory_v2(self):
        result = _rank()
        self.assertEqual(result.version, "executor_advisory_v2")


# ---------------------------------------------------------------------------
# Additional structural tests
# ---------------------------------------------------------------------------

class TestResultStructure(unittest.TestCase):
    def test_all_four_executors_present(self):
        result = _rank()
        all_executors = {result.recommended.executor} | {a.executor for a in result.alternatives}
        self.assertEqual(all_executors, {"native", "opencode", "direct", "staged"})

    def test_recommended_is_candidate_instance(self):
        result = _rank()
        self.assertIsInstance(result.recommended, AdvisoryCandidate)

    def test_alternatives_are_candidate_instances(self):
        result = _rank()
        for a in result.alternatives:
            self.assertIsInstance(a, AdvisoryCandidate)

    def test_alternatives_sorted_descending_by_score(self):
        result = _rank(opencode_available=True)
        scores = [a.score for a in result.alternatives]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_native_not_recommended_when_opencode_preferred_and_available_and_boilerplate(self):
        result = _rank(
            task="add a simple helper",
            category="boilerplate",
            read_only=False,
            opencode_available=True,
            opencode_preference=True,
        )
        # OpenCode can outrank native when explicitly preferred
        oc = next(c for c in [result.recommended] + result.alternatives if c.executor == "opencode")
        nat = next(c for c in [result.recommended] + result.alternatives if c.executor == "native")
        self.assertGreater(oc.score, nat.score)

    def test_render_returns_list_of_strings(self):
        result = _rank()
        lines = render_executor_advisory(result)
        self.assertIsInstance(lines, list)
        for line in lines:
            self.assertIsInstance(line, str)
