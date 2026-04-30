from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

EVAL_RUNS_PATH = Path(".openshard/eval-runs.jsonl")


def load_eval_runs(path: Path) -> list[dict]:
    if not path.exists():
        return []

    records: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return records


@dataclass
class EvalStats:
    model: str
    suite: str
    task_id: str
    run_count: int
    pass_count: int
    fail_count: int
    pass_rate: float
    avg_duration: float
    avg_total_tokens: float | None
    unsafe_file_count: int


def compute_eval_stats(
    records: list[dict],
    *,
    suite: str | None = None,
    model: str | None = None,
    task: str | None = None,
) -> list[EvalStats]:
    if suite:
        records = [r for r in records if r.get("suite") == suite]
    if model:
        records = [r for r in records if r.get("model") == model]
    if task:
        records = [r for r in records if r.get("task_id") == task]

    groups: dict[tuple[str, str, str], list[dict]] = {}
    for r in records:
        m = r.get("model")
        s = r.get("suite")
        t = r.get("task_id")
        if not (m and s and t):
            continue

        key = (str(s), str(m), str(t))
        groups.setdefault(key, []).append(r)

    stats: list[EvalStats] = []
    for (sut, mdl, tid), runs in sorted(groups.items(), key=lambda kv: kv[0]):
        run_count = len(runs)
        pass_count = sum(1 for r in runs if r.get("passed") is True)
        fail_count = run_count - pass_count
        pass_rate = pass_count / run_count if run_count else 0.0

        durations = [
            r["duration_seconds"]
            for r in runs
            if isinstance(r.get("duration_seconds"), (int, float))
        ]
        avg_duration = sum(durations) / len(durations) if durations else 0.0

        token_values = [
            r["total_tokens"]
            for r in runs
            if isinstance(r.get("total_tokens"), (int, float))
        ]
        avg_total_tokens = sum(token_values) / len(token_values) if token_values else None

        unsafe_file_count = sum(
            len(r["unsafe_files"])
            for r in runs
            if isinstance(r.get("unsafe_files"), list)
        )

        stats.append(
            EvalStats(
                model=mdl,
                suite=sut,
                task_id=tid,
                run_count=run_count,
                pass_count=pass_count,
                fail_count=fail_count,
                pass_rate=pass_rate,
                avg_duration=avg_duration,
                avg_total_tokens=avg_total_tokens,
                unsafe_file_count=unsafe_file_count,
            )
        )

    return stats