"""Tests for the locked JSONL store (openshard.history.jsonl_store)."""

from __future__ import annotations

import json
import multiprocessing as mp
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from openshard.history.jsonl_store import (
    _lock_path_for,
    append_jsonl,
    write_jsonl,
)

# Records per writer in the concurrency tests.
_RECORDS_PER_WRITER = 50


# --- Top-level worker (must be importable / picklable for Windows `spawn`). ---
def _append_worker(args: tuple[str, int, int]) -> None:
    """Append ``count`` records tagged with ``proc`` to the shared file."""
    path_str, proc, count = args
    path = Path(path_str)
    for i in range(count):
        append_jsonl(path, {"id": f"proc_{proc}_rec_{i}", "proc": proc, "i": i})


def _read_lines(path: Path) -> list[str]:
    return [ln for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]


def test_locked_append_writes_well_formed_lines(tmp_path: Path) -> None:
    path = tmp_path / "x.jsonl"
    append_jsonl(path, {"a": 1})
    append_jsonl(path, {"a": 2})

    lines = _read_lines(path)
    assert [json.loads(ln) for ln in lines] == [{"a": 1}, {"a": 2}]
    # Sidecar lock file is created alongside the data file.
    assert _lock_path_for(path).exists()


def test_lock_released_after_serialization_error(tmp_path: Path) -> None:
    path = tmp_path / "x.jsonl"
    append_jsonl(path, {"ok": 1})

    # A non-serializable record must raise and write nothing.
    with pytest.raises(TypeError):
        append_jsonl(path, {"bad": object()})

    # The failed write left no partial line behind...
    assert [json.loads(ln) for ln in _read_lines(path)] == [{"ok": 1}]

    # ...and the lock was not leaked: a subsequent normal write still succeeds.
    append_jsonl(path, {"ok": 2})
    assert [json.loads(ln) for ln in _read_lines(path)] == [{"ok": 1}, {"ok": 2}]


def test_cross_process_appends_do_not_interleave(tmp_path: Path) -> None:
    """The load-bearing test: real processes contending for one file."""
    path = tmp_path / "shared.jsonl"
    n_procs = 4
    jobs = [(str(path), proc, _RECORDS_PER_WRITER) for proc in range(n_procs)]

    ctx = mp.get_context("spawn")  # spawn-safe everywhere, matches Windows
    with ctx.Pool(processes=n_procs) as pool:
        pool.map(_append_worker, jobs)

    lines = _read_lines(path)
    # Exact line count: nothing lost, nothing duplicated by torn writes.
    assert len(lines) == n_procs * _RECORDS_PER_WRITER
    # Every line is intact JSON (no interleaving mid-line).
    ids = [json.loads(ln)["id"] for ln in lines]
    # Every expected id appears exactly once.
    expected = {
        f"proc_{p}_rec_{i}"
        for p in range(n_procs)
        for i in range(_RECORDS_PER_WRITER)
    }
    counts = Counter(ids)
    assert set(counts) == expected
    assert all(c == 1 for c in counts.values())


def test_thread_appends_do_not_interleave(tmp_path: Path) -> None:
    """Secondary in-process smoke check of the same property."""
    path = tmp_path / "shared.jsonl"
    n_threads = 8
    with ThreadPoolExecutor(max_workers=n_threads) as ex:
        list(
            ex.map(
                _append_worker,
                [(str(path), t, _RECORDS_PER_WRITER) for t in range(n_threads)],
            )
        )

    lines = _read_lines(path)
    assert len(lines) == n_threads * _RECORDS_PER_WRITER
    assert all(isinstance(json.loads(ln), dict) for ln in lines)


def test_append_round_trips_through_real_reader(tmp_path: Path) -> None:
    """Format is unchanged: records written here load via the real reader."""
    import openshard.history.run_checkpoints as rc

    path = tmp_path / ".openshard" / "run_checkpoints.jsonl"
    events = [
        rc.NativeRunCheckpointEvent(run_id="r1", stage="plan", status="started"),
        rc.NativeRunCheckpointEvent(run_id="r1", stage="final", status="passed"),
    ]
    for ev in events:
        append_jsonl(path, rc._event_to_dict(ev))

    # Patch cwd-relative path resolution by writing where the reader looks.
    loaded = [
        rc._dict_to_event(json.loads(ln)) for ln in _read_lines(path)
    ]
    assert [(e.run_id, e.stage, e.status) for e in loaded] == [
        ("r1", "plan", "started"),
        ("r1", "final", "passed"),
    ]


def test_write_jsonl_is_crash_safe_and_shares_lock(tmp_path: Path) -> None:
    path = tmp_path / "runs.jsonl"
    append_jsonl(path, {"n": 0})

    records = [{"n": 1}, {"n": 2}, {"n": 3}]
    write_jsonl(path, records)

    assert [json.loads(ln) for ln in _read_lines(path)] == records
    # No leftover temp file from the atomic-replace rewrite.
    assert not (tmp_path / "runs.jsonl.tmp").exists()
    # Append and rewrite resolve to the same lock path (mutual exclusion).
    assert _lock_path_for(path) == path.with_name("runs.jsonl.lock")
