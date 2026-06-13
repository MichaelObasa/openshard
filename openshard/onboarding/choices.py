"""Shared onboarding choice constants.

Pure data — no imports from the rest of the codebase.
Both CLI (openshard/cli/ui/onboarding.py) and TUI (openshard/tui/onboarding_screen.py)
import from here to avoid a cross-layer dependency.

Each tuple is (label, value, note, is_planned).
"""
from __future__ import annotations

USER_TYPE_CHOICES: list[tuple[str, str, str, bool]] = [
    ("Human developer", "human", "", False),
    ("Agent / automation", "agent", "", False),
    ("Just exploring / demo", "demo", "", False),
]

EXECUTOR_CHOICES: list[tuple[str, str, str, bool]] = [
    ("OpenShard Native (default)", "native", "", False),
    ("Claude Code import/wrap", "claude_code", "", False),
    ("Codex / OpenAI", "codex", "", False),
    ("OpenCode (planned)", "opencode", "Not yet directly integrated. OpenShard will use local-only mode.", True),
    ("Goose (planned)", "goose", "Not yet directly integrated. OpenShard will use local-only mode.", True),
    ("Antigravity CLI (planned)", "antigravity", "Not yet directly integrated. OpenShard will use local-only mode.", True),
    ("Other / not sure yet", "other", "", False),
]

PROVIDER_ROUTE_CHOICES: list[tuple[str, str, str, bool]] = [
    ("OpenRouter aggregator", "openrouter", "Broadest model access through one key.", False),
    ("Direct provider API", "direct", "Connect directly to a provider's API.", False),
    ("Skip for now / demo mode", "demo", "No key required. Limited to local operations.", False),
]

DIRECT_PROVIDER_CHOICES: list[tuple[str, str, str, bool]] = [
    ("Anthropic (direct)", "anthropic", "Set ANTHROPIC_API_KEY.", False),
    ("OpenAI (direct)", "openai", "Set OPENAI_API_KEY.", False),
    ("Google Gemini (direct planned)", "google", "Direct support is planned. Uses local-only mode until available.", True),
    ("xAI Grok (direct planned)", "xai", "Direct support is planned. Uses local-only mode until available.", True),
    ("DeepSeek (direct planned)", "deepseek", "Direct support is planned. Uses local-only mode until available.", True),
    ("Moonshot / Kimi (direct planned)", "moonshot", "Direct support is planned. Uses local-only mode until available.", True),
    ("GLM / Zhipu (direct planned)", "glm", "Direct support is planned. Uses local-only mode until available.", True),
    ("MiniMax (direct planned)", "minimax", "Direct support is planned. Uses local-only mode until available.", True),
    ("Other / custom", "other", "", False),
]

SAFETY_PROFILE_CHOICES: list[tuple[str, str, str, bool]] = [
    (
        "Recommended",
        "recommended",
        "ask before risky actions · keep receipts · run checks where possible",
        False,
    ),
    (
        "Strict",
        "strict",
        "ask more often · safer for production repos · stronger review posture",
        False,
    ),
    (
        "Fast",
        "fast",
        "fewer prompts · still writes receipts · good for low-risk local work",
        False,
    ),
]

LOCAL_FIRST_NOTICE = (
    "OpenShard is local-first.\n\n"
    "  Your receipts stay on this machine by default.\n"
    "  API keys stay in your environment variables.\n"
    "  OpenShard does not send telemetry or use your runs for training.\n\n"
    "  You can enable team sync or telemetry later if you choose."
)

NEXT_COMMANDS = (
    "  openshard demo shard\n"
    "  openshard env\n"
    "  openshard run \"explain this repo\"\n"
    "  openshard last --more"
)
