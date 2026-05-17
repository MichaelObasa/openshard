from __future__ import annotations

import sys
import unittest
from types import ModuleType
from unittest.mock import MagicMock, patch

from click.testing import CliRunner

from openshard.cli.main import cli


class TestCliTuiCommand(unittest.TestCase):

    def test_tui_appears_in_help(self):
        result = CliRunner().invoke(cli, ["--help"])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("tui", result.output)

    def test_tui_calls_openshard_tui_run(self):
        mock_instance = MagicMock()
        mock_class = MagicMock(return_value=mock_instance)
        mock_module = ModuleType("openshard.tui.app")
        mock_module.OpenShardTui = mock_class  # type: ignore[attr-defined]
        with patch.dict(sys.modules, {"openshard.tui.app": mock_module}):
            result = CliRunner().invoke(cli, ["tui"])
        mock_class.assert_called_once()
        mock_instance.run.assert_called_once()
        self.assertEqual(result.exit_code, 0)

    def test_tui_friendly_error_when_textual_missing(self):
        with patch.dict(sys.modules, {"openshard.tui.app": None}):  # type: ignore[dict-item]
            result = CliRunner().invoke(cli, ["tui"])
        self.assertNotEqual(result.exit_code, 0)
        self.assertIn("textual", result.output.lower())

    def test_demo_run_unaffected(self):
        result = CliRunner().invoke(cli, ["demo-run"])
        self.assertEqual(result.exit_code, 0)

    def test_run_still_requires_task_argument(self):
        result = CliRunner().invoke(cli, ["run"])
        self.assertNotEqual(result.exit_code, 0)
