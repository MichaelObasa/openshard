from __future__ import annotations

from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# ModelEntry — static capability profile for a single model.
# ---------------------------------------------------------------------------
# cost_class and latency_class are stable proxies for routing budget decisions.
# Exact token pricing lives in openshard/providers/openrouter.py (labelled
# snapshot). Do not hardcode volatile prices here.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ModelEntry:
    id: str
    display_name: str
    provider: str
    # Quality/role tier: cheap | mid | strong | frontier | experimental |
    #   code_specialist | small_coder | small | tiny | free_experimental |
    #   long_horizon | value_worker | open_weight | fast_reasoning
    tier: str
    roles: tuple[str, ...] = field(default_factory=tuple)
    experimental: bool = False

    # Capability profile — best-known static values, not live-fetched.
    context_length: int | None = None
    input_modalities: tuple[str, ...] = ("text",)
    output_modalities: tuple[str, ...] = ("text",)

    supports_tools: bool = False
    supports_structured_outputs: bool = False
    supports_reasoning: bool = False
    supports_multimodal: bool = False

    # fast | normal | slow | unknown
    latency_class: str = "unknown"
    # free | tiny | cheap | mid | expensive | unknown
    cost_class: str = "unknown"

    notes: str = ""


# ---------------------------------------------------------------------------
# Registry — all known models in a single list.
# ---------------------------------------------------------------------------
# Add new entries here. Do not scatter model IDs across routing files.
# Routing constants in openshard/routing/engine.py remain the authoritative
# source for *default routing decisions* — the registry is metadata only.
# ---------------------------------------------------------------------------

_REGISTRY: list[ModelEntry] = [
    # ------------------------------------------------------------------
    # Existing routing models — registered for metadata completeness.
    # Routing defaults in engine.py are unchanged.
    # ------------------------------------------------------------------
    ModelEntry(
        id="deepseek/deepseek-v4-flash",
        display_name="DeepSeek: V4 Flash",
        provider="DeepSeek",
        tier="cheap",
        roles=("cheap_control", "boilerplate"),
        experimental=False,
        context_length=128_000,
        supports_tools=True,
        supports_structured_outputs=True,
        latency_class="fast",
        cost_class="cheap",
    ),
    ModelEntry(
        id="z-ai/glm-5.1",
        display_name="Z-AI: GLM 5.1",
        provider="Z-AI",
        tier="mid",
        roles=("routine_engineering", "standard_coding"),
        experimental=False,
        context_length=128_000,
        supports_tools=True,
        supports_structured_outputs=True,
        latency_class="normal",
        cost_class="mid",
    ),
    ModelEntry(
        id="anthropic/claude-sonnet-4.6",
        display_name="Anthropic: Claude Sonnet 4.6",
        provider="Anthropic",
        tier="strong",
        roles=("planner", "reviewer", "frontier_alternative"),
        experimental=False,
        context_length=200_000,
        input_modalities=("text", "image"),
        supports_tools=True,
        supports_structured_outputs=True,
        supports_reasoning=True,
        supports_multimodal=True,
        latency_class="normal",
        cost_class="expensive",
    ),
    ModelEntry(
        id="anthropic/claude-opus-4.7",
        display_name="Anthropic: Claude Opus 4.7",
        provider="Anthropic",
        tier="frontier",
        roles=("escalation", "frontier_alternative"),
        experimental=False,
        context_length=200_000,
        input_modalities=("text", "image"),
        supports_tools=True,
        supports_structured_outputs=True,
        supports_reasoning=True,
        supports_multimodal=True,
        latency_class="slow",
        cost_class="expensive",
    ),
    ModelEntry(
        id="anthropic/claude-haiku-4.5",
        display_name="Anthropic: Claude Haiku 4.5",
        provider="Anthropic",
        tier="cheap",
        roles=("cheap_control", "summariser", "light_review", "claude_family_fast"),
        experimental=False,
        context_length=200_000,
        input_modalities=("text", "image"),
        supports_tools=True,
        supports_structured_outputs=True,
        supports_reasoning=False,
        supports_multimodal=True,
        latency_class="fast",
        cost_class="cheap",
    ),
    ModelEntry(
        id="moonshotai/kimi-k2.5",
        display_name="Moonshot AI: Kimi K2.5",
        provider="Moonshot AI",
        tier="mid",
        roles=("visual", "multimodal"),
        experimental=False,
        context_length=131_072,
        input_modalities=("text", "image"),
        supports_tools=True,
        supports_structured_outputs=True,
        supports_multimodal=True,
        latency_class="normal",
        cost_class="mid",
    ),
    ModelEntry(
        id="minimax/m2.7",
        display_name="MiniMax: M2.7",
        provider="MiniMax",
        tier="mid",
        roles=("complex", "long_context"),
        experimental=False,
        context_length=1_000_000,
        supports_tools=True,
        supports_structured_outputs=True,
        latency_class="normal",
        cost_class="mid",
    ),

    # ------------------------------------------------------------------
    # Core new models — non-experimental.
    # ------------------------------------------------------------------
    ModelEntry(
        id="google/gemini-3.1-flash-lite",
        display_name="Google: Gemini 3.1 Flash Lite",
        provider="Google",
        tier="cheap",
        roles=("cheap_control", "summariser", "session_inference", "feedback_inference", "light_review"),
        experimental=False,
        context_length=1_048_576,
        supports_tools=True,
        supports_structured_outputs=True,
        supports_multimodal=False,
        latency_class="fast",
        cost_class="cheap",
    ),
    ModelEntry(
        id="google/gemini-3.5-flash",
        display_name="Google: Gemini 3.5 Flash",
        provider="Google",
        tier="mid",
        roles=("routine_engineering", "planner", "reviewer", "agentic_mid"),
        experimental=False,
        context_length=1_048_576,
        input_modalities=("text", "image"),
        supports_tools=True,
        supports_structured_outputs=True,
        supports_multimodal=True,
        latency_class="fast",
        cost_class="cheap",
    ),
    ModelEntry(
        id="qwen/qwen3.7-max",
        display_name="Qwen: Qwen3.7 Max",
        provider="Qwen",
        tier="strong",
        roles=("planner", "reviewer", "routine_engineering", "coding", "productivity"),
        experimental=False,
        context_length=131_072,
        supports_tools=True,
        supports_structured_outputs=True,
        latency_class="normal",
        cost_class="mid",
    ),
    ModelEntry(
        id="x-ai/grok-4.3",
        display_name="xAI: Grok 4.3",
        provider="xAI",
        tier="strong",
        roles=("planner", "reviewer", "reasoning", "frontier_alternative"),
        experimental=False,
        context_length=131_072,
        supports_tools=True,
        supports_structured_outputs=True,
        supports_reasoning=True,
        latency_class="normal",
        cost_class="expensive",
    ),
    ModelEntry(
        id="~anthropic/claude-haiku-latest",
        display_name="Anthropic Claude Haiku Latest",
        provider="Anthropic",
        tier="cheap",
        roles=("cheap_control", "summariser", "light_review", "claude_family_fast"),
        experimental=False,
        context_length=200_000,
        input_modalities=("text", "image"),
        supports_tools=True,
        supports_structured_outputs=True,
        supports_multimodal=True,
        latency_class="fast",
        cost_class="cheap",
    ),
    ModelEntry(
        id="x-ai/grok-build-0.1",
        display_name="xAI: Grok Build 0.1",
        provider="xAI",
        tier="experimental",
        roles=("coding_agent", "agentic_engineering", "experimental_coding"),
        experimental=True,
        context_length=131_072,
        supports_tools=True,
        supports_structured_outputs=True,
        latency_class="normal",
        cost_class="mid",
    ),

    # ------------------------------------------------------------------
    # Experimental / specialist models.
    # ------------------------------------------------------------------
    ModelEntry(
        id="qwen/qwen3.6-flash",
        display_name="Qwen: Qwen3.6 Flash",
        provider="Qwen",
        tier="cheap",
        roles=("cheap_control", "summariser", "light_review", "routine_engineering"),
        experimental=True,
        context_length=131_072,
        supports_tools=True,
        supports_structured_outputs=True,
        latency_class="fast",
        cost_class="cheap",
    ),
    ModelEntry(
        id="qwen/qwen3-coder-30b-a3b-instruct",
        display_name="Qwen: Qwen3 Coder 30B A3B Instruct",
        provider="Qwen",
        tier="small_coder",
        roles=("code_generation", "code_review", "repo_understanding", "coding_agent"),
        experimental=True,
        context_length=131_072,
        supports_tools=True,
        supports_structured_outputs=True,
        latency_class="normal",
        cost_class="cheap",
    ),
    ModelEntry(
        id="mistralai/codestral-2508",
        display_name="Mistral: Codestral 2508",
        provider="Mistral",
        tier="code_specialist",
        roles=("code_generation", "code_correction", "test_generation", "small_coding_tasks"),
        experimental=True,
        context_length=32_768,
        supports_tools=True,
        supports_structured_outputs=True,
        latency_class="fast",
        cost_class="cheap",
    ),
    ModelEntry(
        id="google/gemma-4-26b-a4b-it",
        display_name="Google: Gemma 4 26B A4B",
        provider="Google",
        tier="small",
        roles=("cheap_control", "summariser", "structured_metadata", "light_review"),
        experimental=True,
        context_length=131_072,
        supports_tools=True,
        supports_structured_outputs=True,
        latency_class="normal",
        cost_class="cheap",
    ),
    ModelEntry(
        id="google/gemma-4-31b-it",
        display_name="Google: Gemma 4 31B",
        provider="Google",
        tier="small",
        roles=("cheap_control", "local_candidate", "summariser", "baseline_small_model"),
        experimental=True,
        context_length=131_072,
        supports_tools=True,
        supports_structured_outputs=True,
        latency_class="normal",
        cost_class="cheap",
    ),
    ModelEntry(
        id="ibm-granite/granite-4.1-8b",
        display_name="IBM: Granite 4.1 8B",
        provider="IBM",
        tier="tiny",
        roles=("metadata_extraction", "structured_output", "labels", "cheap_control"),
        experimental=True,
        context_length=8_192,
        supports_tools=True,
        supports_structured_outputs=True,
        latency_class="fast",
        cost_class="tiny",
    ),
    ModelEntry(
        id="stepfun/step-3.5-flash",
        display_name="StepFun: Step 3.5 Flash",
        provider="StepFun",
        tier="experimental",
        roles=("reasoning", "coding", "agentic_mid", "cheap_reasoning"),
        experimental=True,
        context_length=32_768,
        supports_tools=True,
        supports_structured_outputs=True,
        supports_reasoning=True,
        latency_class="fast",
        cost_class="cheap",
    ),
    ModelEntry(
        id="poolside/laguna-xs.2:free",
        display_name="Poolside: Laguna XS.2 (free)",
        provider="Poolside",
        tier="free_experimental",
        roles=("coding_agent", "benchmark_only", "experimental_coding"),
        experimental=True,
        context_length=None,
        supports_tools=False,
        supports_structured_outputs=False,
        latency_class="unknown",
        cost_class="free",
    ),
    ModelEntry(
        id="poolside/laguna-m.1:free",
        display_name="Poolside: Laguna M.1 (free)",
        provider="Poolside",
        tier="free_experimental",
        roles=("coding_agent", "benchmark_only", "experimental_coding"),
        experimental=True,
        context_length=None,
        supports_tools=False,
        supports_structured_outputs=False,
        latency_class="unknown",
        cost_class="free",
    ),

    # ------------------------------------------------------------------
    # OpenAI Frontier / Escalation — non-experimental.
    # ------------------------------------------------------------------
    ModelEntry(
        id="openai/gpt-5.5",
        display_name="OpenAI: GPT-5.5",
        provider="OpenAI",
        tier="frontier",
        roles=("escalation", "planner", "reviewer", "reasoning", "high_risk", "coding"),
        experimental=False,
        context_length=1_050_000,
        input_modalities=("text", "image", "file"),
        supports_tools=True,
        supports_structured_outputs=True,
        supports_reasoning=True,
        supports_multimodal=True,
        latency_class="normal",
        cost_class="expensive",
        notes="Frontier escalation model for complex, high-risk, ambiguous engineering tasks.",
    ),
    ModelEntry(
        id="openai/gpt-5.5-pro",
        display_name="OpenAI: GPT-5.5 Pro",
        provider="OpenAI",
        tier="frontier",
        roles=("escalation", "deep_review", "high_risk", "reasoning", "final_review"),
        experimental=False,
        context_length=1_050_000,
        input_modalities=("text", "image", "file"),
        supports_tools=True,
        supports_structured_outputs=True,
        supports_reasoning=True,
        supports_multimodal=True,
        latency_class="slow",
        cost_class="expensive",
        notes="Highest-cost OpenAI escalation lane. Use surgically.",
    ),
    ModelEntry(
        id="openai/gpt-5.4",
        display_name="OpenAI: GPT-5.4",
        provider="OpenAI",
        tier="strong",
        roles=("planner", "reviewer", "coding", "high_context", "routine_engineering"),
        experimental=False,
        context_length=1_050_000,
        input_modalities=("text", "image", "file"),
        supports_tools=True,
        supports_structured_outputs=True,
        supports_reasoning=True,
        supports_multimodal=True,
        latency_class="normal",
        cost_class="mid",
        notes="Strong high-context model for large repo/spec digestion and planning.",
    ),
    ModelEntry(
        id="openai/gpt-5.4-pro",
        display_name="OpenAI: GPT-5.4 Pro",
        provider="OpenAI",
        tier="frontier",
        roles=("escalation", "deep_review", "high_risk", "reasoning", "final_review"),
        experimental=False,
        context_length=1_050_000,
        input_modalities=("text", "image", "file"),
        supports_tools=True,
        supports_structured_outputs=True,
        supports_reasoning=True,
        supports_multimodal=True,
        latency_class="slow",
        cost_class="expensive",
        notes="Pro OpenAI lane for high-stakes reasoning and review.",
    ),

    # ------------------------------------------------------------------
    # OpenAI Efficient / Small — non-experimental.
    # ------------------------------------------------------------------
    ModelEntry(
        id="openai/gpt-5.4-mini",
        display_name="OpenAI: GPT-5.4 Mini",
        provider="OpenAI",
        tier="mid",
        roles=("value_worker", "routine_engineering", "coding", "test_generation", "docs", "lightweight_review"),
        experimental=False,
        context_length=400_000,
        input_modalities=("text", "image", "file"),
        supports_tools=True,
        supports_structured_outputs=True,
        supports_reasoning=True,
        supports_multimodal=True,
        latency_class="fast",
        cost_class="mid",
        notes="Efficient GPT-5.4 family model for routine engineering and high-throughput workloads.",
    ),
    ModelEntry(
        id="openai/gpt-5.4-nano",
        display_name="OpenAI: GPT-5.4 Nano",
        provider="OpenAI",
        tier="small",
        roles=("low_cost_control", "cheap_control", "lightweight_review", "summariser", "metadata_extraction", "fast_chat"),
        experimental=False,
        context_length=400_000,
        input_modalities=("text", "image", "file"),
        supports_tools=True,
        supports_structured_outputs=True,
        supports_reasoning=True,
        supports_multimodal=True,
        latency_class="fast",
        cost_class="cheap",
        notes="Lightweight GPT-5.4 family model for fast, low-cost control and metadata work.",
    ),
    ModelEntry(
        id="openai/gpt-5-mini",
        display_name="OpenAI: GPT-5 Mini",
        provider="OpenAI",
        tier="small",
        roles=("low_cost_control", "cheap_control", "lightweight_review", "summariser", "docs", "test_generation"),
        experimental=False,
        context_length=400_000,
        input_modalities=("text", "image", "file"),
        supports_tools=True,
        supports_structured_outputs=True,
        supports_reasoning=True,
        supports_multimodal=True,
        latency_class="fast",
        cost_class="cheap",
        notes="Compact GPT-5 model for lighter-weight reasoning and lower-cost workflow stages.",
    ),
    ModelEntry(
        id="openai/gpt-5-nano",
        display_name="OpenAI: GPT-5 Nano",
        provider="OpenAI",
        tier="tiny",
        roles=("low_cost_control", "cheap_control", "metadata_extraction", "labels", "fast_chat", "summariser"),
        experimental=False,
        context_length=400_000,
        input_modalities=("text", "image", "file"),
        supports_tools=True,
        supports_structured_outputs=True,
        supports_reasoning=True,
        supports_multimodal=True,
        latency_class="fast",
        cost_class="tiny",
        notes="Very fast, very cheap GPT model for small control-plane tasks.",
    ),

    # ------------------------------------------------------------------
    # Long-horizon / Agentic / Value Workers — non-experimental.
    # ------------------------------------------------------------------
    ModelEntry(
        id="moonshotai/kimi-k2.6",
        display_name="MoonshotAI: Kimi K2.6",
        provider="MoonshotAI",
        tier="long_horizon",
        roles=("swarm_worker", "long_horizon", "coding", "ui_generation", "multi_agent", "routine_engineering"),
        experimental=False,
        context_length=262_144,
        input_modalities=("text", "image"),
        supports_tools=True,
        supports_structured_outputs=True,
        supports_reasoning=True,
        supports_multimodal=True,
        latency_class="normal",
        cost_class="cheap",
        notes="Long-horizon coding and multi-agent orchestration candidate.",
    ),
    ModelEntry(
        id="deepseek/deepseek-v4-pro",
        display_name="DeepSeek: DeepSeek V4 Pro",
        provider="DeepSeek",
        tier="value_worker",
        roles=("value_worker", "coding", "high_context", "routine_engineering", "repo_understanding"),
        experimental=False,
        context_length=1_048_576,
        supports_tools=True,
        supports_structured_outputs=True,
        supports_reasoning=True,
        supports_multimodal=False,
        latency_class="normal",
        cost_class="cheap",
        notes="Price-sensitive large-context execution and repo-understanding candidate.",
    ),

    # ------------------------------------------------------------------
    # OpenAI Open-weight OSS — experimental.
    # ------------------------------------------------------------------
    ModelEntry(
        id="openai/gpt-oss-20b",
        display_name="OpenAI: GPT-OSS 20B",
        provider="OpenAI",
        tier="small",
        roles=("open_weight", "local_candidate", "low_cost_control", "cheap_control", "summariser", "metadata_extraction"),
        experimental=True,
        context_length=131_072,
        supports_tools=True,
        supports_structured_outputs=True,
        supports_reasoning=True,
        supports_multimodal=False,
        latency_class="fast",
        cost_class="tiny",
        notes="Open-weight small model candidate. Benchmark before trusting for routing.",
    ),
    ModelEntry(
        id="openai/gpt-oss-120b",
        display_name="OpenAI: GPT-OSS 120B",
        provider="OpenAI",
        tier="open_weight",
        roles=("open_weight", "reasoning", "coding", "reviewer", "local_candidate"),
        experimental=True,
        context_length=131_072,
        supports_tools=True,
        supports_structured_outputs=True,
        supports_reasoning=True,
        supports_multimodal=False,
        latency_class="normal",
        cost_class="tiny",
        notes="Larger open-weight reasoning/coding candidate. Benchmark before routing.",
    ),

    # ------------------------------------------------------------------
    # Experimental specialist / value models.
    # ------------------------------------------------------------------
    ModelEntry(
        id="inclusionai/ring-2.6-1t",
        display_name="inclusionAI: Ring-2.6 1T",
        provider="inclusionAI",
        tier="strong",
        roles=("reasoning", "coding", "agentic_mid", "value_worker", "reviewer"),
        experimental=True,
        context_length=262_144,
        supports_tools=True,
        supports_structured_outputs=True,
        supports_reasoning=True,
        supports_multimodal=False,
        latency_class="normal",
        cost_class="cheap",
        notes="Very cheap agentic/reasoning candidate. Keep experimental until benchmarked.",
    ),
    ModelEntry(
        id="minimax/minimax-m2.7",
        display_name="MiniMax: MiniMax M2.7",
        provider="MiniMax",
        tier="value_worker",
        roles=("value_worker", "coding", "docs", "product_engineering", "routine_engineering"),
        experimental=True,
        context_length=204_800,
        supports_tools=True,
        supports_structured_outputs=True,
        supports_reasoning=True,
        supports_multimodal=False,
        latency_class="normal",
        cost_class="cheap",
        notes="Budget mixed worker for product engineering, docs, and internal tools.",
    ),
    ModelEntry(
        id="inception/mercury-2",
        display_name="Inception: Mercury 2",
        provider="Inception",
        tier="fast_reasoning",
        roles=("cheap_reasoning", "verifier", "lightweight_review", "fast_chat", "metadata_extraction"),
        experimental=True,
        context_length=128_000,
        supports_tools=True,
        supports_structured_outputs=True,
        supports_reasoning=True,
        supports_multimodal=False,
        latency_class="fast",
        cost_class="cheap",
        notes="Extremely fast reasoning candidate. Benchmark for verifier/control tasks.",
    ),
]

# ---------------------------------------------------------------------------
# Role groups — canonical lists of model IDs per named role group.
# ---------------------------------------------------------------------------

ROLE_GROUPS: dict[str, list[str]] = {
    "cheap_control": [
        "google/gemini-3.1-flash-lite",
        "qwen/qwen3.6-flash",
        "~anthropic/claude-haiku-latest",
        "ibm-granite/granite-4.1-8b",
        "google/gemma-4-26b-a4b-it",
    ],
    "routine_engineering": [
        "google/gemini-3.5-flash",
        "qwen/qwen3.7-max",
        "mistralai/codestral-2508",
        "qwen/qwen3-coder-30b-a3b-instruct",
    ],
    "planner_reviewer": [
        "x-ai/grok-4.3",
        "qwen/qwen3.7-max",
        "google/gemini-3.5-flash",
    ],
    "experimental_coding_agent": [
        "x-ai/grok-build-0.1",
        "poolside/laguna-xs.2:free",
        "poolside/laguna-m.1:free",
        "stepfun/step-3.5-flash",
    ],
}

# ---------------------------------------------------------------------------
# Capability names accepted by models_by_capability() and supports().
# ---------------------------------------------------------------------------

_CAPABILITY_ATTRS: dict[str, str] = {
    "tools": "supports_tools",
    "structured_outputs": "supports_structured_outputs",
    "reasoning": "supports_reasoning",
    "multimodal": "supports_multimodal",
}

CAPABILITY_NAMES: tuple[str, ...] = tuple(_CAPABILITY_ATTRS)

# ---------------------------------------------------------------------------
# Index — built once at import time.
# ---------------------------------------------------------------------------

_INDEX: dict[str, ModelEntry] = {entry.id: entry for entry in _REGISTRY}


# ---------------------------------------------------------------------------
# Public helpers.
# ---------------------------------------------------------------------------


def get_model(model_id: str) -> ModelEntry | None:
    """Return the ModelEntry for *model_id*, or None if not registered."""
    return _INDEX.get(model_id)


def models_by_role(role: str) -> list[ModelEntry]:
    """Return all registered models whose *roles* tuple includes *role*."""
    return [e for e in _REGISTRY if role in e.roles]


def models_by_capability(capability: str) -> list[ModelEntry]:
    """Return all registered models that support *capability*.

    Accepted capability names: "tools", "structured_outputs", "reasoning",
    "multimodal". Returns an empty list for unrecognised capability strings.
    """
    attr = _CAPABILITY_ATTRS.get(capability)
    if attr is None:
        return []
    return [e for e in _REGISTRY if getattr(e, attr)]


def display_name_for(model_id: str, fallback: str | None = None) -> str:
    """Return the display name for *model_id*.

    If the model is not in the registry, returns *fallback* when provided,
    otherwise returns *model_id* unchanged.
    """
    entry = _INDEX.get(model_id)
    if entry is not None:
        return entry.display_name
    return fallback if fallback is not None else model_id


def is_experimental(model_id: str) -> bool:
    """Return True if *model_id* is registered and marked experimental."""
    entry = _INDEX.get(model_id)
    return entry.experimental if entry is not None else False


def supports(model_id: str, capability: str) -> bool:
    """Return True if *model_id* is registered and supports *capability*.

    Returns False for unknown model IDs or unrecognised capability strings.
    """
    entry = _INDEX.get(model_id)
    if entry is None:
        return False
    attr = _CAPABILITY_ATTRS.get(capability)
    if attr is None:
        return False
    return bool(getattr(entry, attr))


def all_models() -> list[ModelEntry]:
    """Return all registered models as a list."""
    return list(_REGISTRY)


def is_known_model(model_id: str) -> bool:
    """Return True if *model_id* is a registered model.

    This is the single source of truth for model existence. Prefer it over
    reaching into the private registry or comparing against hand-maintained
    lists elsewhere in the codebase.
    """
    return model_id in _INDEX


def require_model(model_id: str) -> str:
    """Return *model_id* if it is registered, else raise ValueError.

    Useful for asserting at import or startup that a hardcoded constant still
    points at a real registry entry.
    """
    if model_id not in _INDEX:
        raise ValueError(f"Unknown model id (not in registry): {model_id}")
    return model_id


def registry_ids() -> frozenset[str]:
    """Return the set of all registered model IDs.

    Lets callers and tests check membership without reaching into the private
    registry structures.
    """
    return frozenset(_INDEX)
