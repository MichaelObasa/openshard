"""Tests for the 6 new builtin workflow packs."""
from __future__ import annotations

import unittest

from click.testing import CliRunner

from openshard.cli.main import cli
from openshard.workflow_packs.packs import REQUIRED_FIELDS, load_packs

NEW_PACK_IDS = [
    "code-review",
    "test-coverage-gaps",
    "dependency-audit",
    "api-design-review",
    "docker-security-review",
    "readme-audit",
]


class TestNewPacksLoad(unittest.TestCase):

    def setUp(self):
        self.packs = load_packs()
        self.pack_map = {p.id: p for p in self.packs}

    def test_all_new_packs_load(self):
        for pack_id in NEW_PACK_IDS:
            self.assertIn(pack_id, self.pack_map, f"Pack {pack_id!r} not found in loaded packs")

    def test_all_new_packs_have_required_fields(self):
        for pack in self.packs:
            if pack.id not in NEW_PACK_IDS:
                continue
            for field in REQUIRED_FIELDS:
                self.assertTrue(
                    getattr(pack, field, None),
                    f"Pack {pack.id!r} missing or empty required field {field!r}",
                )

    def test_all_pack_ids_unique(self):
        ids = [p.id for p in self.packs]
        self.assertEqual(len(ids), len(set(ids)), "Duplicate pack IDs found")

    def test_docker_security_review_has_native_workflow(self):
        pack = self.pack_map["docker-security-review"]
        self.assertEqual(pack.workflow, "native")

    def test_docker_security_review_has_execution_prompt_suffix(self):
        pack = self.pack_map["docker-security-review"]
        self.assertTrue(pack.execution_prompt_suffix, "execution_prompt_suffix should be non-empty")

    def test_code_review_has_no_workflow(self):
        pack = self.pack_map["code-review"]
        self.assertEqual(pack.workflow, "")


class TestNewPacksCLI(unittest.TestCase):

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
            self.assertEqual(result.exit_code, 0, f"packs show {pack_id!r} exited {result.exit_code}: {result.output}")

    def test_packs_run_help_exits_zero(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["packs", "run", "--help"])
        self.assertEqual(result.exit_code, 0)


if __name__ == "__main__":
    unittest.main()
