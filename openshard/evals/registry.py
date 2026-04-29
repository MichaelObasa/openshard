from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

_EVALS_ROOT = Path(__file__).parent.parent.parent / "evals"

_REQUIRED_FIELDS = {"id", "title", "category", "expected_files", "verification_command"}


@dataclass
class EvalTask:
    id: str
    title: str
    category: str
    expected_files: list[str]
    verification_command: str | None
    prompt: str
    task_dir: Path


def validate_eval_task(task_dir: Path) -> EvalTask:
    prompt_path = task_dir / "prompt.txt"
    metadata_path = task_dir / "metadata.json"

    if not prompt_path.exists():
        raise FileNotFoundError(f"Missing prompt.txt in {task_dir}")

    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing metadata.json in {task_dir}")

    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid metadata.json in {task_dir}: {exc}") from exc

    missing = _REQUIRED_FIELDS - set(metadata.keys())
    if missing:
        raise ValueError(f"metadata.json in {task_dir} missing fields: {sorted(missing)}")

    return EvalTask(
        id=metadata["id"],
        title=metadata["title"],
        category=metadata["category"],
        expected_files=metadata["expected_files"],
        verification_command=metadata.get("verification_command"),
        prompt=prompt_path.read_text(encoding="utf-8"),
        task_dir=task_dir,
    )


def load_eval_tasks(suite: str = "basic") -> list[EvalTask]:
    suite_dir = _EVALS_ROOT / suite
    if not suite_dir.is_dir():
        raise FileNotFoundError(f"Eval suite not found: {suite_dir}")

    return [
        validate_eval_task(task_dir)
        for task_dir in sorted(suite_dir.iterdir())
        if task_dir.is_dir()
    ]
