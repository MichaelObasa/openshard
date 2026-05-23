from __future__ import annotations

from dataclasses import dataclass

from openshard.workflow_packs.builtin import _BUILTIN_PACKS

REQUIRED_FIELDS = {
    "id",
    "title",
    "category",
    "summary",
    "prompt",
    "recommended_context",
    "expected_receipt_value",
    "safety_notes",
    "tags",
}


@dataclass
class WorkflowPack:
    id: str
    title: str
    category: str
    summary: str
    prompt: str
    recommended_context: str
    expected_receipt_value: str
    safety_notes: str
    tags: list[str]
    execution_prompt_suffix: str = ""
    workflow: str = ""


def _validate_pack(raw: dict) -> WorkflowPack:
    missing = REQUIRED_FIELDS - set(raw.keys())
    if missing:
        raise ValueError(f"Pack {raw.get('id', '?')!r} missing fields: {sorted(missing)}")
    return WorkflowPack(
        id=raw["id"],
        title=raw["title"],
        category=raw["category"],
        summary=raw["summary"],
        prompt=raw["prompt"],
        recommended_context=raw["recommended_context"],
        expected_receipt_value=raw["expected_receipt_value"],
        safety_notes=raw["safety_notes"],
        tags=raw["tags"],
        execution_prompt_suffix=raw.get("execution_prompt_suffix", ""),
        workflow=raw.get("workflow", ""),
    )


def load_packs() -> list[WorkflowPack]:
    return [_validate_pack(raw) for raw in _BUILTIN_PACKS]


def get_pack(pack_id: str) -> WorkflowPack:
    for raw in _BUILTIN_PACKS:
        if raw.get("id") == pack_id:
            return _validate_pack(raw)
    raise KeyError(pack_id)
