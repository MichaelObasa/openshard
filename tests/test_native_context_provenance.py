from __future__ import annotations

import unittest
from dataclasses import asdict
from types import SimpleNamespace

from openshard.native.context import (
    NativeContextProvenance,
    NativeContextSource,
    NativeContextUsageSummary,
    NativeEvidence,
    NativeFileContext,
    NativeFileSnippet,
    NativeObservation,
    NativePlan,
    NativeValidationContract,
    NativeContextQualityScore,
    NativeClarificationRequest,
    build_native_context_provenance,
    render_native_context_provenance,
)
from openshard.cli.run_output import _native_meta_from_entry, _render_native_demo_block


def _build(**kwargs) -> NativeContextProvenance:
    defaults = dict(
        repo_context_summary=None,
        observation=None,
        evidence=None,
        read_search_findings=None,
        file_context=None,
        context_packet=None,
        context_quality_score=None,
        context_quality_advisory=None,
        change_budget=None,
        plan=None,
        verification_plan=None,
        clarification_request=None,
        validation_contract=None,
        context_usage_summary=None,
        osn_loop=None,
        skills_context=None,
        injected_source_names=None,
    )
    defaults.update(kwargs)
    return build_native_context_provenance(**defaults)


def _meta(context_provenance=None) -> SimpleNamespace:
    return SimpleNamespace(
        workflow="native",
        repo_context_summary=None,
        observation=None,
        plan=None,
        write_path="pipeline",
        verification_loop=None,
        verification_command_summary=None,
        diff_review=None,
        final_report=None,
        native_loop_steps=[],
        native_loop_trace=[],
        native_backend=None,
        native_backend_available=True,
        native_backend_notes=[],
        native_backend_proof=None,
        read_search_findings=[],
        patch_proposal=None,
        command_policy_preview=None,
        context_packet=None,
        file_context=None,
        context_quality_score=None,
        context_quality_advisory=None,
        change_budget=None,
        change_budget_preview=None,
        change_budget_soft_gate=None,
        approval_request=None,
        approval_receipt=None,
        verification_plan=None,
        clarification_request=None,
        context_usage_summary=None,
        failure_memory=None,
        osn_loop=None,
        deepagents_adapter=None,
        validation_contract=None,
        context_provenance=context_provenance,
    )


class TestNativeContextSourceDefaults(unittest.TestCase):
    def test_source_defaults(self):
        s = NativeContextSource()
        self.assertEqual(s.name, "")
        self.assertFalse(s.used)
        self.assertFalse(s.injected)
        self.assertEqual(s.item_count, 0)
        self.assertEqual(s.summary, "")

    def test_provenance_defaults(self):
        p = NativeContextProvenance()
        self.assertEqual(p.sources, [])
        self.assertEqual(p.injected_sources, 0)
        self.assertEqual(p.used_sources, 0)
        self.assertEqual(p.total_items, 0)
        self.assertFalse(p.has_gaps)
        self.assertEqual(p.warnings, [])


class TestNativeContextProvenanceAsdict(unittest.TestCase):
    def test_source_asdict(self):
        s = NativeContextSource(name="repo_summary", used=True, injected=True, item_count=1, summary="1 summary")
        d = asdict(s)
        self.assertIsInstance(d, dict)
        self.assertEqual(d["name"], "repo_summary")
        self.assertTrue(d["used"])
        self.assertTrue(d["injected"])
        self.assertEqual(d["item_count"], 1)

    def test_provenance_asdict(self):
        p = _build()
        d = asdict(p)
        self.assertIsInstance(d, dict)
        self.assertIn("sources", d)
        self.assertIn("injected_sources", d)
        self.assertIn("used_sources", d)
        self.assertIn("total_items", d)
        self.assertIn("has_gaps", d)
        self.assertIn("warnings", d)
        self.assertIsInstance(d["sources"], list)

    def test_asdict_has_all_expected_keys(self):
        p = NativeContextProvenance()
        d = asdict(p)
        expected = {"sources", "injected_sources", "used_sources", "total_items", "has_gaps", "warnings"}
        self.assertEqual(set(d.keys()), expected)


class TestBuildWithAllNoneInputs(unittest.TestCase):
    def test_all_none_returns_valid_provenance(self):
        p = _build()
        self.assertIsInstance(p, NativeContextProvenance)
        self.assertEqual(p.used_sources, 0)
        self.assertEqual(p.injected_sources, 0)
        self.assertFalse(p.has_gaps)

    def test_all_none_sources_list_populated(self):
        p = _build()
        names = [s.name for s in p.sources]
        self.assertIn("repo_summary", names)
        self.assertIn("observation", names)
        self.assertIn("evidence", names)
        self.assertIn("read_search", names)
        self.assertIn("file_context", names)
        self.assertIn("context_packet", names)
        self.assertIn("context_quality", names)
        self.assertIn("advisory", names)
        self.assertIn("change_budget", names)
        self.assertIn("plan", names)
        self.assertIn("verification_plan", names)
        self.assertIn("clarification_request", names)
        self.assertIn("validation_contract", names)
        self.assertIn("context_usage_summary", names)
        self.assertIn("osn_loop", names)
        self.assertIn("skills_context", names)

    def test_all_none_no_source_used(self):
        p = _build()
        self.assertTrue(all(not s.used for s in p.sources))

    def test_all_none_no_source_injected(self):
        p = _build()
        self.assertTrue(all(not s.injected for s in p.sources))


class TestSourceCountsFromMetadata(unittest.TestCase):
    def test_observation_item_count(self):
        obs = NativeObservation(observed_tools=["read", "glob", "grep"])
        p = _build(observation=obs)
        src = next(s for s in p.sources if s.name == "observation")
        self.assertTrue(src.used)
        self.assertEqual(src.item_count, 3)
        self.assertIn("3 tools", src.summary)

    def test_evidence_item_count(self):
        ev = NativeEvidence(
            search_results=["result1", "result2"],
            file_snippets=[NativeFileSnippet(path="a.py"), NativeFileSnippet(path="b.py"), NativeFileSnippet(path="c.py")],
        )
        p = _build(evidence=ev)
        src = next(s for s in p.sources if s.name == "evidence")
        self.assertTrue(src.used)
        self.assertEqual(src.item_count, 5)
        self.assertIn("5 items", src.summary)

    def test_read_search_findings_count(self):
        p = _build(read_search_findings=["file:a.py", "file:b.py"])
        src = next(s for s in p.sources if s.name == "read_search")
        self.assertTrue(src.used)
        self.assertEqual(src.item_count, 2)
        self.assertIn("2 findings", src.summary)

    def test_file_context_count(self):
        fc = NativeFileContext(files_read=4)
        p = _build(file_context=fc)
        src = next(s for s in p.sources if s.name == "file_context")
        self.assertTrue(src.used)
        self.assertEqual(src.item_count, 4)
        self.assertIn("4 files", src.summary)

    def test_plan_step_count(self):
        plan = NativePlan(suggested_steps=["step1", "step2", "step3"])
        p = _build(plan=plan)
        src = next(s for s in p.sources if s.name == "plan")
        self.assertTrue(src.used)
        self.assertEqual(src.item_count, 3)
        self.assertIn("3 steps", src.summary)

    def test_plan_no_steps_defaults_to_1(self):
        plan = NativePlan(suggested_steps=[])
        p = _build(plan=plan)
        src = next(s for s in p.sources if s.name == "plan")
        self.assertTrue(src.used)
        self.assertEqual(src.item_count, 1)

    def test_repo_summary_item_count(self):
        from openshard.native.repo_context import NativeRepoContextSummary
        rcs = NativeRepoContextSummary(total_files=10)
        p = _build(repo_context_summary=rcs)
        src = next(s for s in p.sources if s.name == "repo_summary")
        self.assertTrue(src.used)
        self.assertEqual(src.item_count, 1)
        self.assertEqual(src.summary, "1 summary")

    def test_skills_context_used(self):
        p = _build(skills_context="some skill content")
        src = next(s for s in p.sources if s.name == "skills_context")
        self.assertTrue(src.used)
        self.assertEqual(src.item_count, 1)

    def test_skills_context_empty_not_used(self):
        p = _build(skills_context="")
        src = next(s for s in p.sources if s.name == "skills_context")
        self.assertFalse(src.used)
        self.assertEqual(src.item_count, 0)


class TestInjectedFlags(unittest.TestCase):
    def test_injected_flags_match_set(self):
        obs = NativeObservation(observed_tools=["read"])
        plan = NativePlan()
        p = _build(
            observation=obs,
            plan=plan,
            injected_source_names={"observation", "plan"},
        )
        obs_src = next(s for s in p.sources if s.name == "observation")
        plan_src = next(s for s in p.sources if s.name == "plan")
        repo_src = next(s for s in p.sources if s.name == "repo_summary")
        self.assertTrue(obs_src.injected)
        self.assertTrue(plan_src.injected)
        self.assertFalse(repo_src.injected)

    def test_plan_injected_when_in_set(self):
        plan = NativePlan(suggested_steps=["a", "b"])
        p = _build(plan=plan, injected_source_names={"plan"})
        src = next(s for s in p.sources if s.name == "plan")
        self.assertTrue(src.injected)

    def test_plan_not_injected_when_not_in_set(self):
        plan = NativePlan()
        p = _build(plan=plan, injected_source_names=set())
        src = next(s for s in p.sources if s.name == "plan")
        self.assertFalse(src.injected)

    def test_injected_count_matches(self):
        from openshard.native.repo_context import NativeRepoContextSummary
        rcs = NativeRepoContextSummary()
        obs = NativeObservation()
        p = _build(
            repo_context_summary=rcs,
            observation=obs,
            injected_source_names={"repo_summary", "observation"},
        )
        self.assertEqual(p.injected_sources, 2)

    def test_used_sources_count(self):
        obs = NativeObservation()
        plan = NativePlan()
        p = _build(observation=obs, plan=plan)
        self.assertEqual(p.used_sources, 2)

    def test_total_items_sum(self):
        obs = NativeObservation(observed_tools=["a", "b"])
        plan = NativePlan(suggested_steps=["s1"])
        p = _build(observation=obs, plan=plan)
        obs_items = next(s for s in p.sources if s.name == "observation").item_count
        plan_items = next(s for s in p.sources if s.name == "plan").item_count
        self.assertEqual(p.total_items, obs_items + plan_items)


class TestHasGaps(unittest.TestCase):
    def test_has_gaps_weak_context_quality(self):
        cqs = NativeContextQualityScore(level="weak")
        p = _build(context_quality_score=cqs)
        self.assertTrue(p.has_gaps)
        self.assertIn("context quality weak", p.warnings)

    def test_has_gaps_unknown_context_quality(self):
        cqs = NativeContextQualityScore(level="unknown")
        p = _build(context_quality_score=cqs)
        self.assertTrue(p.has_gaps)

    def test_no_gaps_good_context_quality(self):
        cqs = NativeContextQualityScore(level="good")
        p = _build(context_quality_score=cqs)
        self.assertFalse(p.has_gaps)

    def test_has_gaps_weak_validation_contract(self):
        vc = NativeValidationContract(strength="weak")
        p = _build(validation_contract=vc)
        self.assertTrue(p.has_gaps)
        self.assertIn("validation contract weak", p.warnings)

    def test_no_gaps_strong_validation_contract(self):
        vc = NativeValidationContract(strength="strong")
        p = _build(validation_contract=vc)
        self.assertFalse(p.has_gaps)

    def test_has_gaps_clarification_needed(self):
        cr = NativeClarificationRequest(needed=True)
        p = _build(clarification_request=cr)
        self.assertTrue(p.has_gaps)
        self.assertIn("clarification needed", p.warnings)

    def test_no_gaps_clarification_not_needed(self):
        cr = NativeClarificationRequest(needed=False)
        p = _build(clarification_request=cr)
        self.assertFalse(p.has_gaps)

    def test_has_gaps_file_context_truncated(self):
        fc = NativeFileContext(truncated=True)
        p = _build(file_context=fc)
        self.assertTrue(p.has_gaps)
        self.assertIn("file context truncated", p.warnings)

    def test_no_gaps_file_context_not_truncated(self):
        fc = NativeFileContext(truncated=False)
        p = _build(file_context=fc)
        self.assertFalse(p.has_gaps)

    def test_has_gaps_context_usage_summary_truncated(self):
        cus = NativeContextUsageSummary(any_truncated=True)
        p = _build(context_usage_summary=cus)
        self.assertTrue(p.has_gaps)
        self.assertIn("context truncated", p.warnings)

    def test_multiple_gaps_accumulate_warnings(self):
        cqs = NativeContextQualityScore(level="weak")
        vc = NativeValidationContract(strength="weak")
        p = _build(context_quality_score=cqs, validation_contract=vc)
        self.assertTrue(p.has_gaps)
        self.assertGreaterEqual(len(p.warnings), 2)


class TestWarningsSafety(unittest.TestCase):
    def test_warnings_are_compact_text(self):
        cqs = NativeContextQualityScore(level="weak")
        vc = NativeValidationContract(strength="weak")
        cr = NativeClarificationRequest(needed=True)
        fc = NativeFileContext(truncated=True)
        cus = NativeContextUsageSummary(any_truncated=True)
        p = _build(
            context_quality_score=cqs,
            validation_contract=vc,
            clarification_request=cr,
            file_context=fc,
            context_usage_summary=cus,
        )
        for w in p.warnings:
            self.assertIsInstance(w, str)
            self.assertLess(len(w), 60)

    def test_warnings_no_raw_paths(self):
        fc = NativeFileContext(truncated=True, paths=["src/secret/config.py"])
        p = _build(file_context=fc)
        for w in p.warnings:
            self.assertNotIn("src/secret", w)
            self.assertNotIn(".py", w)

    def test_no_gaps_no_warnings(self):
        p = _build()
        self.assertEqual(p.warnings, [])


class TestRenderer(unittest.TestCase):
    def test_renderer_returns_empty_for_none(self):
        self.assertEqual(render_native_context_provenance(None), "")

    def test_renderer_includes_summary_line(self):
        obs = NativeObservation(observed_tools=["read"])
        p = _build(observation=obs)
        out = render_native_context_provenance(p)
        self.assertIn("context provenance:", out)
        self.assertIn("sources", out)
        self.assertIn("injected", out)
        self.assertIn("items", out)

    def test_renderer_gaps_line_when_has_gaps(self):
        cqs = NativeContextQualityScore(level="weak")
        p = _build(context_quality_score=cqs)
        out = render_native_context_provenance(p)
        self.assertIn("context provenance gaps:", out)

    def test_renderer_no_gaps_line_when_no_gaps(self):
        cqs = NativeContextQualityScore(level="good")
        p = _build(context_quality_score=cqs)
        out = render_native_context_provenance(p)
        self.assertNotIn("context provenance gaps:", out)

    def test_renderer_does_not_expose_raw_content(self):
        fc = NativeFileContext(files_read=3, paths=["src/secret/config.py", "src/app.py"])
        p = _build(file_context=fc)
        out = render_native_context_provenance(p)
        self.assertNotIn("src/secret", out)
        self.assertNotIn("config.py", out)
        self.assertNotIn("app.py", out)

    def test_renderer_shows_only_warning_count_not_text(self):
        cqs = NativeContextQualityScore(level="weak")
        vc = NativeValidationContract(strength="weak")
        p = _build(context_quality_score=cqs, validation_contract=vc)
        out = render_native_context_provenance(p)
        self.assertNotIn("context quality weak", out)
        self.assertNotIn("validation contract weak", out)
        self.assertIn("warnings", out)


class TestRenderNativeDemoBlockProvenance(unittest.TestCase):
    def test_provenance_renders_in_more_detail(self):
        obs = NativeObservation(observed_tools=["read", "glob"])
        p = _build(observation=obs, injected_source_names={"observation"})
        meta = _meta(context_provenance=p)
        lines = _render_native_demo_block(meta, detail="more")
        combined = "\n".join(lines)
        self.assertIn("context provenance:", combined)

    def test_provenance_renders_in_full_detail(self):
        obs = NativeObservation(observed_tools=["read"])
        p = _build(observation=obs, injected_source_names={"observation"})
        meta = _meta(context_provenance=p)
        lines = _render_native_demo_block(meta, detail="full")
        combined = "\n".join(lines)
        self.assertIn("context provenance:", combined)
        self.assertIn("provenance source:", combined)

    def test_provenance_absent_from_default_detail(self):
        obs = NativeObservation(observed_tools=["read"])
        p = _build(observation=obs)
        meta = _meta(context_provenance=p)
        lines = _render_native_demo_block(meta, detail="default")
        combined = "\n".join(lines)
        self.assertNotIn("context provenance:", combined)

    def test_full_detail_source_lines_format(self):
        obs = NativeObservation(observed_tools=["read"])
        plan = NativePlan(suggested_steps=["s1", "s2"])
        p = _build(observation=obs, plan=plan, injected_source_names={"observation", "plan"})
        meta = _meta(context_provenance=p)
        lines = _render_native_demo_block(meta, detail="full")
        combined = "\n".join(lines)
        self.assertIn("provenance source: observation used=yes injected=yes", combined)
        self.assertIn("provenance source: plan used=yes injected=yes", combined)

    def test_dict_based_source_rendering(self):
        prov_dict = {
            "sources": [
                {"name": "repo_summary", "used": True, "injected": True, "item_count": 1, "summary": "1 summary"},
                {"name": "observation", "used": True, "injected": False, "item_count": 2, "summary": "2 tools"},
            ],
            "injected_sources": 1,
            "used_sources": 2,
            "total_items": 3,
            "has_gaps": False,
            "warnings": [],
        }
        entry = {"workflow": "native", "context_provenance": prov_dict}
        meta = _native_meta_from_entry(entry)
        self.assertIsNotNone(meta)
        lines = _render_native_demo_block(meta, detail="full")
        combined = "\n".join(lines)
        self.assertIn("provenance source: repo_summary", combined)
        self.assertIn("provenance source: observation", combined)

    def test_gaps_line_renders_in_more_detail(self):
        cqs = NativeContextQualityScore(level="weak")
        p = _build(context_quality_score=cqs)
        meta = _meta(context_provenance=p)
        lines = _render_native_demo_block(meta, detail="more")
        combined = "\n".join(lines)
        self.assertIn("context provenance gaps:", combined)

    def test_warning_text_not_rendered_in_block(self):
        cqs = NativeContextQualityScore(level="weak")
        p = _build(context_quality_score=cqs)
        meta = _meta(context_provenance=p)
        lines = _render_native_demo_block(meta, detail="more")
        combined = "\n".join(lines)
        self.assertNotIn("context quality weak", combined)

    def test_roundtrip_via_native_meta_from_entry(self):
        obs = NativeObservation(observed_tools=["read", "grep"])
        plan = NativePlan(suggested_steps=["step1"])
        p = _build(observation=obs, plan=plan, injected_source_names={"observation", "plan"})
        entry = {"workflow": "native", "context_provenance": asdict(p)}
        meta = _native_meta_from_entry(entry)
        prov = getattr(meta, "context_provenance", None)
        self.assertIsNotNone(prov)
        self.assertEqual(getattr(prov, "used_sources", None), p.used_sources)
        self.assertEqual(getattr(prov, "injected_sources", None), p.injected_sources)
        self.assertEqual(getattr(prov, "total_items", None), p.total_items)


if __name__ == "__main__":
    unittest.main()
