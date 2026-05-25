from __future__ import annotations

from openshard.models.registry import all_models, models_by_capability, models_by_role

_ASK_FALLBACK = (
    "Ask Mode v1 can answer OpenShard/product questions only.\n"
    "For repo analysis, use a normal task or /pack.\n"
    "For planning, use /plan once enabled."
)

_COMMANDS_TEXT = (
    "TUI commands:\n"
    "  /help                    Show help\n"
    "  /last                    Show the most recent run\n"
    "  /last more               Show more detail for the last run\n"
    "  /last full               Show full debug/audit detail\n"
    "  /feedback <outcome>      Record outcome for the last run\n"
    "  /clear                   Clear the output panel\n"
    "  /quit                    Exit the TUI\n"
    "  /packs                   List available workflow packs\n"
    "  /pack <id>               Load a workflow pack\n"
    "  /ask <question>          Ask OpenShard a product question\n"
    "\n"
    "Plain text is sent to the execution engine as a task."
)

_OPENSHARD_TEXT = (
    "OpenShard is a local CLI and TUI — the control layer for AI coding agents.\n"
    "It routes tasks to AI models, tracks structured run receipts (shards),\n"
    "and provides model routing, evidence provenance, and session history.\n"
    "\n"
    "Run `openshard --help` for full CLI options, or /help inside the TUI."
)

_RECEIPT_TEXT = (
    "A shard receipt is the structured record of a completed OpenShard run.\n"
    "It includes:\n"
    "  RESULT          — outcome and verification status\n"
    "  ACTIONS         — file and code operations performed\n"
    "  CHECK ACTIONS   — IaC, lint, or format verification steps\n"
    "  EVIDENCE        — files inspected, changed, or sourced\n"
    "\n"
    "Receipts are stored in .openshard/runs.jsonl.\n"
    "Use /last, /last more, or /last full to view them in the TUI."
)

_LAST_TEXT = (
    "  /last           — show the most recent run summary\n"
    "  /last more      — include check actions and evidence\n"
    "  /last full      — show the complete raw run output\n"
    "  /feedback       — record your outcome for the last run\n"
    "                    (accepted / rejected / partial / abandoned / retried)"
)

_PLAN_TEXT = (
    "/plan is planned for a future OpenShard release.\n"
    "It will generate a structured execution plan before committing to a run.\n"
    "Use `openshard plan <task>` from the CLI for the current planning surface."
)

_PACK_TEXT = (
    "Context packs attach a structured suffix and workflow to your next task.\n"
    "  /packs          — list available packs\n"
    "  /pack <id>      — select a pack and activate it for the next run\n"
    "Once activated, the pack context is appended to your task automatically."
)


def answer_ask_mode(question: str) -> str:
    """Return a local, deterministic answer for an Ask Mode question. No provider calls."""
    q = question.strip().lower()

    if "what models" in q or "which models" in q or "model roster" in q:
        return _answer_model_roster()
    if (
        "cheap control" in q
        or "cheap model" in q
        or "low cost" in q
        or "low-cost" in q
        or "lightweight" in q
        or "control model" in q
    ):
        return _answer_cheap_control()
    if "reasoning model" in q or "reasoning capable" in q:
        return _answer_reasoning_models()
    if "experimental model" in q or "experimental" in q:
        return _answer_experimental_models()
    if "what commands" in q or "available commands" in q:
        return _COMMANDS_TEXT
    if "what is openshard" in q or "what does openshard" in q or "about openshard" in q:
        return _OPENSHARD_TEXT
    if "receipt" in q or "shard receipt" in q or "what is a shard" in q:
        return _RECEIPT_TEXT
    if "/last" in q or "what is last" in q or "what does last" in q:
        return _LAST_TEXT
    if "/plan" in q or "what is plan" in q:
        return _PLAN_TEXT
    if "/pack" in q or "context pack" in q:
        return _PACK_TEXT
    return _ASK_FALLBACK


def _answer_model_roster() -> str:
    count = len(all_models())
    cheap = models_by_role("cheap_control")[:4]
    reasoning = models_by_capability("reasoning")[:4]
    lines = [
        f"OpenShard currently knows {count} registered models.",
        "",
        "OpenShard groups models by what they are useful for:",
        "  - Low-cost / control work",
        "  - Routine engineering",
        "  - Planning and review",
        "  - Reasoning-heavy tasks",
        "  - Experimental coding agents",
        "",
        "Low-cost / control examples:",
    ]
    for m in cheap:
        lines.append(f"  {m.display_name}")
    lines += ["", "Reasoning / review examples:"]
    for m in reasoning:
        lines.append(f"  {m.display_name}")
    lines += [
        "",
        "Use:",
        "  /ask low-cost models",
        "  /ask reasoning models",
        "  /ask experimental models",
        "  openshard models list",
    ]
    return "\n".join(lines)


def _answer_cheap_control() -> str:
    models = models_by_role("cheap_control")
    if not models:
        return "No low-cost / control models found in the current registry."
    col = max(len(m.display_name) for m in models) + 4
    lines = ["Low-cost / control models:", ""]
    for m in models:
        lines.append(f"  {m.display_name:<{col}}{m.cost_class} / {m.latency_class}")
    lines += ["", "Use `openshard models role cheap_control` for the raw registry role."]
    return "\n".join(lines)


def _answer_reasoning_models() -> str:
    models = models_by_capability("reasoning")
    if not models:
        return "No reasoning-capable models found in the current registry."
    lines = ["Reasoning-capable models:", ""]
    for m in models:
        lines.append(f"  {m.display_name}")
    lines += ["", "Use `openshard models capabilities reasoning` for full details."]
    return "\n".join(lines)


def _answer_experimental_models() -> str:
    models = [m for m in all_models() if m.experimental]
    if not models:
        return "No experimental models in the current registry."
    lines = ["Experimental models:", ""]
    for m in models:
        lines.append(f"  {m.display_name}")
    lines += ["", "Use `openshard models experimental` for full details."]
    return "\n".join(lines)
