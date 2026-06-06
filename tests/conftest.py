"""Shared pytest fixtures for the OpenShard test suite."""
from __future__ import annotations

from unittest.mock import patch

import pytest


@pytest.fixture(autouse=True)
def _default_pipeline_provider():
    """Patch detect_provider at the pipeline import site for every test.

    Pipeline integration tests mock ExecutionGenerator to avoid real API
    calls, but they don't set any API key env var.  Without this fixture,
    detect_provider() raises ValueError and the pipeline exits before the
    mocked generator is ever reached, breaking those tests.

    Tests that exercise detect_provider() directly import it from
    openshard.config.settings and are unaffected by this patch.
    """
    with patch("openshard.run.pipeline.detect_provider", return_value="openrouter"):
        yield
