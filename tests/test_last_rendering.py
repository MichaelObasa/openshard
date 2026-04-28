from __future__ import annotations

import unittest

import click
from click.testing import CliRunner

from openshard.cli.main import _model_label, _render_log_entry


def _render(entry: dict, detail: str = "more") -> str:
    @click.command()
    def cmd():
        _render_log_entry(entry, detail)

    return CliRunner().invoke(cmd).output


class TestLastRoutingDisplay(unittest.TestCase):

    def test_routing_section_shown_with_fields(self):
        entry = {
            "task": "do a thing",
            "routing_category": "standard",
            "routing_candidate_count": 5,
            "routing_selected_model": "openrouter/fast-model",
            "routing_selected_provider": "openrouter",
            "routing_used_fallback": False,
        }
        out = _render(entry)
        self.assertIn("[routing] category: standard", out)
        self.assertIn("candidates: 5", out)
        self.assertIn(_model_label("openrouter/fast-model"), out)
        self.assertIn("openrouter", out)

    def test_routing_section_fallback(self):
        entry = {
            "task": "do a thing",
            "routing_category": "visual",
            "routing_candidate_count": 0,
            "routing_selected_model": None,
            "routing_selected_provider": None,
            "routing_used_fallback": True,
        }
        out = _render(entry)
        self.assertIn("[routing] category: visual", out)
        self.assertIn("fallback (keyword routing)", out)

    def test_routing_section_absent_without_fields(self):
        entry = {"task": "do a thing"}
        out = _render(entry)
        self.assertNotIn("[routing]", out)


class TestLastProfileDisplay(unittest.TestCase):

    def test_more_shows_profile_when_present(self):
        entry = {
            "task": "do a thing",
            "execution_profile": "native_deep",
            "execution_profile_reason": "security category",
        }
        out = _render(entry, detail="more")
        self.assertIn("[profile] native_deep - security category", out)

    def test_full_shows_profile_when_present(self):
        entry = {
            "task": "do a thing",
            "execution_profile": "native_light",
            "execution_profile_reason": "simple/safe task",
        }
        out = _render(entry, detail="full")
        self.assertIn("[profile] native_light - simple/safe task", out)

    def test_profile_without_reason(self):
        entry = {
            "task": "do a thing",
            "execution_profile": "native_swarm",
        }
        out = _render(entry, detail="more")
        self.assertIn("[profile] native_swarm", out)

    def test_no_crash_on_entry_without_profile_fields(self):
        entry = {"task": "do a thing"}
        out = _render(entry, detail="more")
        self.assertNotIn("[profile]", out)

    def test_no_crash_on_entry_without_profile_fields_full(self):
        entry = {"task": "do a thing"}
        out = _render(entry, detail="full")
        self.assertNotIn("[profile]", out)

    def test_default_detail_does_not_show_profile(self):
        entry = {
            "task": "do a thing",
            "execution_profile": "native_deep",
            "execution_profile_reason": "security category",
        }
        out = _render(entry, detail="default")
        self.assertNotIn("[profile]", out)


if __name__ == "__main__":
    unittest.main()
