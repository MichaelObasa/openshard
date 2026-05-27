from __future__ import annotations

import unittest

from click.testing import CliRunner

from openshard.cli.main import cli


class TestCliPackage(unittest.TestCase):

    def test_cli_importable(self):
        self.assertIsNotNone(cli)

    def test_version_exits_zero(self):
        result = CliRunner().invoke(cli, ["--version"])
        self.assertEqual(result.exit_code, 0)

    def test_version_contains_version_string(self):
        import openshard
        result = CliRunner().invoke(cli, ["--version"])
        self.assertIn(openshard.__version__, result.output)

    def test_no_args_renders_home_exits_zero(self):
        result = CliRunner().invoke(cli, [])
        self.assertEqual(result.exit_code, 0)

    def test_package_version_importable(self):
        import openshard
        self.assertRegex(openshard.__version__, r"^\d+\.\d+")
