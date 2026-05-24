from __future__ import annotations

import unittest

from click.testing import CliRunner

from openshard.cli.main import cli


class TestModelsListCommand(unittest.TestCase):
    def setUp(self) -> None:
        self.runner = CliRunner()

    def test_list_exits_zero(self) -> None:
        result = self.runner.invoke(cli, ["models", "list"])
        self.assertEqual(result.exit_code, 0)

    def test_list_includes_gemini_flash_lite(self) -> None:
        result = self.runner.invoke(cli, ["models", "list"])
        self.assertIn("google/gemini-3.1-flash-lite", result.output)

    def test_list_includes_grok_4_3(self) -> None:
        result = self.runner.invoke(cli, ["models", "list"])
        self.assertIn("x-ai/grok-4.3", result.output)

    def test_list_shows_header_columns(self) -> None:
        result = self.runner.invoke(cli, ["models", "list"])
        self.assertIn("ID", result.output)
        self.assertIn("Provider", result.output)
        self.assertIn("Tier", result.output)
        self.assertIn("Experimental", result.output)


class TestModelsShowCommand(unittest.TestCase):
    def setUp(self) -> None:
        self.runner = CliRunner()

    def test_show_known_model_exits_zero(self) -> None:
        result = self.runner.invoke(cli, ["models", "show", "google/gemini-3.1-flash-lite"])
        self.assertEqual(result.exit_code, 0)

    def test_show_displays_display_name(self) -> None:
        result = self.runner.invoke(cli, ["models", "show", "google/gemini-3.1-flash-lite"])
        self.assertIn("Google: Gemini 3.1 Flash Lite", result.output)

    def test_show_displays_provider(self) -> None:
        result = self.runner.invoke(cli, ["models", "show", "google/gemini-3.1-flash-lite"])
        self.assertIn("Google", result.output)

    def test_show_displays_tier(self) -> None:
        result = self.runner.invoke(cli, ["models", "show", "google/gemini-3.1-flash-lite"])
        self.assertIn("cheap", result.output)

    def test_show_displays_roles(self) -> None:
        result = self.runner.invoke(cli, ["models", "show", "google/gemini-3.1-flash-lite"])
        self.assertIn("cheap_control", result.output)

    def test_show_displays_capability_fields(self) -> None:
        result = self.runner.invoke(cli, ["models", "show", "google/gemini-3.1-flash-lite"])
        self.assertIn("Tools", result.output)
        self.assertIn("Structured", result.output)
        self.assertIn("Reasoning", result.output)
        self.assertIn("Multimodal", result.output)

    def test_show_unknown_model_exits_nonzero(self) -> None:
        result = self.runner.invoke(cli, ["models", "show", "unknown/model-x"])
        self.assertNotEqual(result.exit_code, 0)

    def test_show_unknown_model_shows_error(self) -> None:
        result = self.runner.invoke(cli, ["models", "show", "unknown/model-x"])
        self.assertIn("not found", result.output.lower())


class TestModelsRoleCommand(unittest.TestCase):
    def setUp(self) -> None:
        self.runner = CliRunner()

    def test_role_cheap_control_exits_zero(self) -> None:
        result = self.runner.invoke(cli, ["models", "role", "cheap_control"])
        self.assertEqual(result.exit_code, 0)

    def test_role_cheap_control_includes_gemini_flash_lite(self) -> None:
        result = self.runner.invoke(cli, ["models", "role", "cheap_control"])
        self.assertIn("google/gemini-3.1-flash-lite", result.output)

    def test_role_unknown_exits_zero(self) -> None:
        result = self.runner.invoke(cli, ["models", "role", "nonexistent_role_xyz"])
        self.assertEqual(result.exit_code, 0)

    def test_role_unknown_shows_no_models_found(self) -> None:
        result = self.runner.invoke(cli, ["models", "role", "nonexistent_role_xyz"])
        self.assertIn("No models found for role", result.output)


class TestModelsCapabilitiesCommand(unittest.TestCase):
    def setUp(self) -> None:
        self.runner = CliRunner()

    def test_capabilities_reasoning_exits_zero(self) -> None:
        result = self.runner.invoke(cli, ["models", "capabilities", "reasoning"])
        self.assertEqual(result.exit_code, 0)

    def test_capabilities_reasoning_includes_grok_4_3(self) -> None:
        result = self.runner.invoke(cli, ["models", "capabilities", "reasoning"])
        self.assertIn("x-ai/grok-4.3", result.output)

    def test_capabilities_unknown_shows_accepted_names(self) -> None:
        result = self.runner.invoke(cli, ["models", "capabilities", "unknown_cap"])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("tools", result.output)
        self.assertIn("reasoning", result.output)


class TestModelsExperimentalCommand(unittest.TestCase):
    def setUp(self) -> None:
        self.runner = CliRunner()

    def test_experimental_exits_zero(self) -> None:
        result = self.runner.invoke(cli, ["models", "experimental"])
        self.assertEqual(result.exit_code, 0)

    def test_experimental_includes_grok_build(self) -> None:
        result = self.runner.invoke(cli, ["models", "experimental"])
        self.assertIn("x-ai/grok-build-0.1", result.output)

    def test_experimental_includes_poolside_models(self) -> None:
        result = self.runner.invoke(cli, ["models", "experimental"])
        self.assertIn("poolside/laguna-xs.2:free", result.output)
        self.assertIn("poolside/laguna-m.1:free", result.output)


if __name__ == "__main__":
    unittest.main()
