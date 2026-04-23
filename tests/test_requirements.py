from __future__ import annotations

import unittest

from openshard.execution.stages import Stage
from openshard.scoring.requirements import TaskRequirements, requirements_from_category, requirements_from_stage


class TestRequirementsFromCategory(unittest.TestCase):

    def test_security(self):
        req = requirements_from_category("security")
        self.assertTrue(req.security_sensitive)
        self.assertEqual(req.complexity, "complex")
        self.assertEqual(req.min_context_window, 100_000)
        self.assertIsNone(req.preferred_max_cost_per_m)

    def test_complex(self):
        req = requirements_from_category("complex")
        self.assertEqual(req.complexity, "complex")
        self.assertEqual(req.min_context_window, 100_000)
        self.assertFalse(req.security_sensitive)
        self.assertIsNone(req.preferred_max_cost_per_m)

    def test_visual(self):
        req = requirements_from_category("visual")
        self.assertTrue(req.needs_vision)
        self.assertFalse(req.security_sensitive)
        self.assertIsNone(req.preferred_max_cost_per_m)

    def test_boilerplate(self):
        req = requirements_from_category("boilerplate")
        self.assertEqual(req.complexity, "simple")
        self.assertEqual(req.preferred_max_cost_per_m, 2.0)
        self.assertFalse(req.security_sensitive)

    def test_standard(self):
        req = requirements_from_category("standard")
        self.assertEqual(req.complexity, "standard")
        self.assertEqual(req.preferred_max_cost_per_m, 1.5)
        self.assertIsNone(req.min_context_window)
        self.assertFalse(req.security_sensitive)


class TestTaskRequirements(unittest.TestCase):

    def test_default_field_values(self):
        req = TaskRequirements()
        self.assertIsNone(req.min_context_window)
        self.assertFalse(req.needs_vision)
        self.assertFalse(req.needs_tools)
        self.assertEqual(req.complexity, "standard")
        self.assertFalse(req.security_sensitive)
        self.assertIsNone(req.preferred_max_cost_per_m)

    def test_requirements_from_simple_stage(self):
        stage = Stage(stage_type="implementation", description="Add getter", complexity="simple")
        req = requirements_from_stage(stage)
        self.assertIsNone(req.min_context_window)
        self.assertFalse(req.needs_vision)
        self.assertFalse(req.needs_tools)
        self.assertEqual(req.complexity, "simple")
        self.assertFalse(req.security_sensitive)

    def test_requirements_from_complex_security_stage(self):
        stage = Stage(
            stage_type="implementation",
            description="Implement auth middleware",
            security_sensitive=True,
            complexity="complex",
        )
        req = requirements_from_stage(stage)
        self.assertEqual(req.min_context_window, 100_000)
        self.assertTrue(req.security_sensitive)
        self.assertEqual(req.complexity, "complex")
        self.assertFalse(req.needs_vision)
        self.assertFalse(req.needs_tools)
