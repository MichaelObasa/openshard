"""Tests for the 6 extended domain-specific workflow packs."""
from __future__ import annotations

import unittest

from click.testing import CliRunner

from openshard.cli.main import cli
from openshard.workflow_packs.packs import REQUIRED_FIELDS, load_packs

NEW_PACK_IDS = [
    "database-migration-review",
    "accessibility-audit",
    "kubernetes-security-review",
    "openapi-spec-review",
    "github-actions-review",
    "performance-hotspots",
]

# Packs that must have a non-empty execution_prompt_suffix
PACKS_WITH_SUFFIX = [
    "database-migration-review",
    "accessibility-audit",
    "openapi-spec-review",
    "github-actions-review",
]


class TestExtendedPacksLoad(unittest.TestCase):

    def setUp(self):
        self.packs = load_packs()
        self.pack_map = {p.id: p for p in self.packs}

    def test_all_new_packs_load(self):
        for pack_id in NEW_PACK_IDS:
            self.assertIn(pack_id, self.pack_map, f"Pack {pack_id!r} not found in loaded packs")

    def test_all_new_packs_have_required_fields(self):
        for pack_id in NEW_PACK_IDS:
            pack = self.pack_map[pack_id]
            for field in REQUIRED_FIELDS:
                self.assertTrue(
                    getattr(pack, field, None),
                    f"Pack {pack_id!r} missing or empty required field {field!r}",
                )

    def test_all_pack_ids_unique(self):
        ids = [p.id for p in self.packs]
        self.assertEqual(len(ids), len(set(ids)), "Duplicate pack IDs found")

    def test_kubernetes_security_review_has_native_workflow(self):
        pack = self.pack_map["kubernetes-security-review"]
        self.assertEqual(pack.workflow, "native")

    def test_performance_hotspots_has_no_execution_prompt_suffix(self):
        pack = self.pack_map["performance-hotspots"]
        self.assertFalse(
            pack.execution_prompt_suffix,
            "performance-hotspots should have no execution_prompt_suffix",
        )

    def test_packs_with_suffix_have_non_empty_execution_prompt_suffix(self):
        for pack_id in PACKS_WITH_SUFFIX:
            pack = self.pack_map[pack_id]
            self.assertTrue(
                pack.execution_prompt_suffix,
                f"Pack {pack_id!r} should have a non-empty execution_prompt_suffix",
            )


class TestExtendedPacksCLI(unittest.TestCase):

    def test_packs_list_includes_all_new_ids(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["packs", "list"])
        self.assertEqual(result.exit_code, 0)
        for pack_id in NEW_PACK_IDS:
            self.assertIn(pack_id, result.output, f"Pack {pack_id!r} not in packs list output")

    def test_packs_show_exits_zero_for_each_new_pack(self):
        runner = CliRunner()
        for pack_id in NEW_PACK_IDS:
            result = runner.invoke(cli, ["packs", "show", pack_id])
            self.assertEqual(
                result.exit_code,
                0,
                f"packs show {pack_id!r} exited {result.exit_code}: {result.output}",
            )


if __name__ == "__main__":
    unittest.main()
