"""Cross-platform locked JSONL writes for OpenShard's local history store.

Every ``.openshard/*.jsonl`` record write funnels through this module so that
concurrent OpenShard processes writing the same history file cannot interleave
or tear each other's lines. Locking is stdlib-only: ``fcntl`` on Unix-like
systems and ``msvcrt`` on Windows, applied to a sidecar ``<file>.lock`` so we
never tangle with append-mode seek semantics on the data file descriptor.

Two helpers are exposed:

- ``append_jsonl(path, record)`` — append one record as a single JSON line.
- ``write_jsonl(path, records)`` — crash-safe whole-file rewrite (temp + replace).

Both derive the same lock path from the data file, so an append and a rewrite of
the same history file mutually exclude.
"""

from __future__ import annotations

import json
import os
import sys
from contextlib import contextmanager
from pathlib import Path

if sys.platform == "win32":
    import msvcrt
else:
    import fcntl


@contextmanager
def _file_lock(lock_path: Path):
    """Hold an exclusive cross-process lock on a sidecar ``.lock`` file.

    Acquisition blocks until the lock is held. Release is guaranteed in a
    ``finally`` even on exception, and the handle is closed in a nested
    ``finally`` so the OS lock is freed even if the unlock call itself raises.
    No path leaves a lock held, so there is no deadlock.
    """
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    fh = open(lock_path, "a+")
    try:
        if sys.platform == "win32":
            # msvcrt.locking locks a 1-byte range, so the sidecar must have at
            # least one byte; empty-file byte-range behavior is ambiguous.
            if os.fstat(fh.fileno()).st_size < 1:
                fh.write("\0")
                fh.flush()
            fh.seek(0)
            msvcrt.locking(fh.fileno(), msvcrt.LK_LOCK, 1)  # blocking exclusive
        else:
            fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
        yield
    finally:
        try:
            if sys.platform == "win32":
                fh.seek(0)
                msvcrt.locking(fh.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
        finally:
            fh.close()


def _lock_path_for(path: Path) -> Path:
    """Return the sidecar lock path for *path* (e.g. ``runs.jsonl.lock``)."""
    return path.with_name(path.name + ".lock")


def append_jsonl(path: Path, record: dict) -> None:
    """Append one record to *path* as a single locked, fsync'd JSON line.

    Serialization happens before the lock is acquired and before the file is
    opened, so a non-serializable record raises with nothing written and no
    lock taken.
    """
    path = Path(path)
    line = json.dumps(record) + "\n"  # serialize BEFORE locking / opening
    path.parent.mkdir(parents=True, exist_ok=True)
    with _file_lock(_lock_path_for(path)):
        with path.open("a", encoding="utf-8") as fh:
            fh.write(line)
            fh.flush()
            os.fsync(fh.fileno())


def write_jsonl(path: Path, records: list[dict]) -> None:
    """Crash-safe locked whole-file rewrite of *path* with *records*.

    Writes to a sibling temp file, fsyncs it, then ``os.replace``s it over the
    target (atomic on both Windows and POSIX) so the real file is never
    truncated mid-write. Uses the same sidecar lock as :func:`append_jsonl`, so
    a rewrite and a concurrent append of the same file serialize.
    """
    path = Path(path)
    blob = "".join(json.dumps(r) + "\n" for r in records)  # serialize BEFORE locking
    path.parent.mkdir(parents=True, exist_ok=True)
    with _file_lock(_lock_path_for(path)):
        tmp = path.with_name(path.name + ".tmp")
        try:
            with tmp.open("w", encoding="utf-8") as fh:
                fh.write(blob)
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp, path)  # atomic rename over target
        except BaseException:
            try:
                tmp.unlink()  # best-effort cleanup on failure
            except OSError:
                pass
            raise
