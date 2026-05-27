from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

_DEFAULTS: dict[str, Any] = {
    "model_tiers": [
        {"name": "fast",     "model": "anthropic/claude-haiku-4.5",  "max_tokens": 1024},
        {"name": "balanced", "model": "anthropic/claude-sonnet-4.6", "max_tokens": 4096},
        {"name": "powerful", "model": "anthropic/claude-opus-4.6",   "max_tokens": 8192},
    ],
    "planning_model":      "anthropic/claude-sonnet-4.6",
    "execution_model":     "anthropic/claude-sonnet-4.6",
    "fixer_model":         "anthropic/claude-sonnet-4.6",
    "workflow":            "auto",
    "approval_mode":       "smart",
    "cost_gate_threshold": 0.10,
    "executor":            "direct",
}


def _load_yaml(p: Path) -> dict[str, Any]:
    with p.open(encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def load_config(path: str | os.PathLike | None = None) -> dict[str, Any]:
    """Load and return the YAML configuration.

    Search order:
    1. *path* argument — raises FileNotFoundError if given but absent
    2. OPENSHARD_CONFIG environment variable — raises if set but absent
    3. .openshard/config.yml in the current working directory
    4. config.yml in the current working directory
    5. Bundled openshard/config/default_config.yml (importlib.resources)
    6. Built-in _DEFAULTS — never raises
    """
    if path:
        config_path = Path(path)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        return _load_yaml(config_path)

    env_path = os.environ.get("OPENSHARD_CONFIG", "")
    if env_path:
        config_path = Path(env_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        return _load_yaml(config_path)

    hidden = Path.cwd() / ".openshard" / "config.yml"
    if hidden.exists():
        return _load_yaml(hidden)

    cwd_cfg = Path.cwd() / "config.yml"
    if cwd_cfg.exists():
        return _load_yaml(cwd_cfg)

    try:
        from importlib.resources import files
        pkg_cfg = files("openshard.config").joinpath("default_config.yml")
        with pkg_cfg.open("r", encoding="utf-8") as fh:
            return yaml.safe_load(fh) or {}
    except Exception:
        pass

    return dict(_DEFAULTS)


def get_api_key() -> str:
    """Return the OpenRouter API key from the environment.

    Raises ``ValueError`` with a clear message if the variable is not set.
    """
    key = os.environ.get("OPENROUTER_API_KEY", "")
    if not key:
        raise ValueError(
            "OPENROUTER_API_KEY environment variable is not set.\n"
            "Export it before running:\n\n"
            "  export OPENROUTER_API_KEY=your_key_here\n\n"
            "Obtain a key from https://openrouter.ai/keys"
        )
    return key


def get_anthropic_api_key() -> str:
    """Return the Anthropic API key from the environment.

    Raises ``ValueError`` with a clear message if the variable is not set.
    """
    key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not key:
        raise ValueError(
            "ANTHROPIC_API_KEY environment variable is not set.\n"
            "Export it before running:\n\n"
            "  export ANTHROPIC_API_KEY=your_key_here\n\n"
            "Obtain a key from https://console.anthropic.com/settings/keys"
        )
    return key


def get_openai_api_key() -> str:
    """Return the OpenAI API key from the environment.

    Raises ``ValueError`` with a clear message if the variable is not set.
    """
    key = os.environ.get("OPENAI_API_KEY", "")
    if not key:
        raise ValueError(
            "OPENAI_API_KEY environment variable is not set.\n"
            "Export it before running:\n\n"
            "  export OPENAI_API_KEY=your_key_here\n\n"
            "Obtain a key from https://platform.openai.com/api-keys"
        )
    return key
