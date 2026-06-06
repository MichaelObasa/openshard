from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

_DEFAULTS: dict[str, Any] = {
    "model_tiers": [
        {"name": "fast",     "model": "anthropic/claude-haiku-4.5",  "max_tokens": 1024},
        {"name": "balanced", "model": "anthropic/claude-sonnet-4.6", "max_tokens": 4096},
        {"name": "powerful", "model": "anthropic/claude-opus-4.7",   "max_tokens": 8192},
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


def _inject_api_keys(config: dict[str, Any]) -> dict[str, Any]:
    """Inject API keys from environment variables into *config* (mutates and returns it).

    Checks OPENROUTER_API_KEY, ANTHROPIC_API_KEY, and OPENAI_API_KEY.  All
    keys found in the environment are stored so callers can inspect which
    providers are available without re-reading the environment.
    """
    openrouter_key = os.environ.get("OPENROUTER_API_KEY", "")
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY", "")
    openai_key = os.environ.get("OPENAI_API_KEY", "")

    if openrouter_key:
        config["openrouter_api_key"] = openrouter_key
    if anthropic_key:
        config["anthropic_api_key"] = anthropic_key
    if openai_key:
        config["openai_api_key"] = openai_key

    return config


def load_config(path: str | os.PathLike | None = None) -> dict[str, Any]:
    """Load and return the YAML configuration.

    Search order:
    1. *path* argument — raises FileNotFoundError if given but absent
    2. OPENSHARD_CONFIG environment variable — raises if set but absent
    3. .openshard/config.yml in the current working directory
    4. config.yml in the current working directory
    5. Bundled openshard/config/default_config.yml (importlib.resources) — with
       env-var API keys injected
    6. Built-in _DEFAULTS with env-var API keys injected — never raises

    When no user config is found (steps 5–6), API keys from the environment
    (OPENROUTER_API_KEY, ANTHROPIC_API_KEY, OPENAI_API_KEY) are injected into
    the returned dict so the tool works immediately after installation without
    running ``openshard init`` first.
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
            bundled = yaml.safe_load(fh) or {}
        return _inject_api_keys(bundled)
    except Exception:
        pass

    return _inject_api_keys(dict(_DEFAULTS))


def config_search_path(cwd: Path | None = None) -> Path:
    """Return the path where ``openshard init`` writes its config.

    This is always ``<cwd>/.openshard/config.yml`` — the highest-priority
    location in :func:`load_config`'s search order after explicit overrides.
    """
    base = cwd if cwd is not None else Path.cwd()
    return base / ".openshard" / "config.yml"


def find_config_path(cwd: Path | None = None) -> Path | None:
    """Return the config path :func:`load_config` would read, or ``None``.

    Mirrors :func:`load_config`'s search order for the discoverable
    locations: ``OPENSHARD_CONFIG`` env var, ``.openshard/config.yml`` and
    ``config.yml`` in *cwd*. The bundled default and built-in ``_DEFAULTS`` are
    not real on-disk user configs, so they are reported as "not found".
    """
    base = cwd if cwd is not None else Path.cwd()

    env_path = os.environ.get("OPENSHARD_CONFIG", "")
    if env_path:
        p = Path(env_path)
        return p if p.exists() else None

    hidden = base / ".openshard" / "config.yml"
    if hidden.exists():
        return hidden

    cwd_cfg = base / "config.yml"
    if cwd_cfg.exists():
        return cwd_cfg

    return None


def load_config_safe(
    path: str | os.PathLike | None = None,
    cwd: Path | None = None,
) -> tuple[dict[str, Any], bool, Path | None]:
    """Load config without raising; degrade to defaults on any failure.

    Returns ``(config, config_valid, resolved_path)`` where *config_valid* is
    ``False`` when an on-disk config exists but could not be parsed (malformed
    YAML, unreadable). In that case *config* falls back to ``_DEFAULTS`` so
    callers like ``doctor``/``config show`` keep working instead of crashing.
    """
    resolved = find_config_path(cwd=cwd)
    if resolved is None:
        try:
            return load_config(path=path), True, None
        except Exception:
            return dict(_DEFAULTS), True, None

    try:
        return _load_yaml(resolved), True, resolved
    except Exception:
        return dict(_DEFAULTS), False, resolved


def save_config(config: dict[str, Any], path: str | os.PathLike | None = None) -> Path:
    """Write *config* as YAML, creating the parent directory if needed.

    Defaults to ``<cwd>/.openshard/config.yml`` (see :func:`config_search_path`).
    Never writes secrets — API keys live only in the environment.
    """
    target = Path(path) if path is not None else config_search_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(config, fh, sort_keys=False, default_flow_style=False)
    return target


def get_onboarding(config: dict[str, Any]) -> dict[str, Any]:
    """Return the additive ``onboarding`` block from *config*, or ``{}``."""
    value = config.get("onboarding")
    return value if isinstance(value, dict) else {}


def get_api_key() -> str:
    """Return the OpenRouter API key from the environment.

    Raises ``ValueError`` with a clear message if the variable is not set.
    """
    key = os.environ.get("OPENROUTER_API_KEY", "")
    if not key:
        raise ValueError(
            "No API key configured. Set OPENROUTER_API_KEY, ANTHROPIC_API_KEY, or "
            "OPENAI_API_KEY, or run 'openshard init' to configure.\n\n"
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
