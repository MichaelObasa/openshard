"""Tests for openshard.history.shard_hash — Shard hash v1 (tamper-evidence).

Covers deterministic hashing, self-referential exclusion, content sensitivity,
the stamp-when-missing/preserve-when-present coercion rule, and the three
verification states (valid / mismatch / missing).
"""

from __future__ import annotations

import json

from openshard.history.shard_hash import (
    SHARD_HASH_FIELD,
    SHARD_HASH_VERSION,
    compute_shard_hash,
    stored_shard_hash,
    verify_shard_hash,
)
from openshard.history.shard_schema import coerce_shard_entry


def _entry(**kwargs) -> dict:
    base: dict = {
        "schema_version": "1.2",
        "shard_id": "shard-20260605-0001",
        "timestamp": "2026-06-05T00:00:00Z",
        "task": "do a thing",
        "execution_model": "test-model",
        "duration_seconds": 1.0,
        "files_created": 0,
        "files_updated": 0,
        "files_deleted": 0,
        "summary": "ok",
    }
    base.update(kwargs)
    return base


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


def test_constants():
    assert SHARD_HASH_FIELD == "content_hash"
    assert isinstance(SHARD_HASH_VERSION, str) and SHARD_HASH_VERSION


# ---------------------------------------------------------------------------
# 1. Deterministic
# ---------------------------------------------------------------------------


def test_compute_is_deterministic():
    entry = _entry()
    assert compute_shard_hash(entry) == compute_shard_hash(entry)


def test_compute_is_sha256_prefixed_hex():
    h = compute_shard_hash(_entry())
    assert h.startswith("sha256:")
    assert len(h) == len("sha256:") + 64
    int(h.split(":", 1)[1], 16)  # hex parse must succeed


def test_key_order_does_not_change_hash():
    entry = _entry()
    shuffled = dict(reversed(list(entry.items())))
    assert compute_shard_hash(entry) == compute_shard_hash(shuffled)


# ---------------------------------------------------------------------------
# 2. Excludes itself (no self-referential hashing)
# ---------------------------------------------------------------------------


def test_hash_excludes_content_hash_field():
    base = _entry()
    with_hash = dict(base, content_hash="sha256:deadbeef")
    assert compute_shard_hash(with_hash) == compute_shard_hash(base)


def test_changing_stored_hash_does_not_change_compute():
    base = _entry()
    a = compute_shard_hash(dict(base, content_hash="sha256:aaaa"))
    b = compute_shard_hash(dict(base, content_hash="sha256:bbbb"))
    assert a == b


# ---------------------------------------------------------------------------
# 3. Content sensitivity
# ---------------------------------------------------------------------------


def test_changing_a_field_changes_hash():
    assert compute_shard_hash(_entry(task="a")) != compute_shard_hash(_entry(task="b"))


def test_compute_never_raises_on_odd_types():
    # Sets aren't JSON-native; default=str must keep the call total.
    h = compute_shard_hash(_entry(weird={1, 2, 3}))
    assert h.startswith("sha256:")


# ---------------------------------------------------------------------------
# 4. coerce stamps when missing / preserves when present
# ---------------------------------------------------------------------------


def test_coerce_stamps_hash_when_missing():
    result = coerce_shard_entry(_entry())
    assert result[SHARD_HASH_FIELD].startswith("sha256:")


def test_coerce_stamped_hash_is_self_consistent():
    result = coerce_shard_entry(_entry())
    # The stamped value must equal a recompute over the coerced content.
    assert result[SHARD_HASH_FIELD] == compute_shard_hash(result)


def test_coerce_preserves_existing_hash():
    sentinel = "sha256:" + "0" * 64
    result = coerce_shard_entry(_entry(content_hash=sentinel))
    assert result[SHARD_HASH_FIELD] == sentinel


def test_coerced_entry_with_hash_is_json_serializable():
    result = coerce_shard_entry(_entry())
    parsed = json.loads(json.dumps(result))
    assert parsed[SHARD_HASH_FIELD].startswith("sha256:")


# ---------------------------------------------------------------------------
# stored_shard_hash
# ---------------------------------------------------------------------------


def test_stored_hash_returns_value_when_present():
    assert stored_shard_hash(_entry(content_hash="sha256:abc")) == "sha256:abc"


def test_stored_hash_returns_none_when_absent_or_empty():
    assert stored_shard_hash(_entry()) is None
    assert stored_shard_hash(_entry(content_hash="")) is None
    assert stored_shard_hash(_entry(content_hash=123)) is None  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# 5-7. verify_shard_hash — valid / mismatch / missing
# ---------------------------------------------------------------------------


def test_verify_valid_after_coerce():
    record = coerce_shard_entry(_entry())  # stamps a correct hash
    v = verify_shard_hash(record)
    assert v["status"] == "valid"
    assert v["matches"] is True
    assert v["present"] is True
    assert v["stored_hash"] == v["computed_hash"]


def test_verify_mismatch_when_content_edited_but_hash_kept():
    record = coerce_shard_entry(_entry())  # correct hash for original content
    record["task"] = "tampered task"       # edit content, keep old hash
    v = verify_shard_hash(record)
    assert v["status"] == "mismatch"
    assert v["matches"] is False
    assert v["present"] is True
    assert v["stored_hash"] != v["computed_hash"]


def test_verify_missing_for_legacy_record():
    v = verify_shard_hash(_entry())  # no content_hash at all
    assert v["status"] == "missing"
    assert v["present"] is False
    assert v["matches"] is None
    assert v["stored_hash"] is None
    assert v["computed_hash"].startswith("sha256:")  # still derivable


def test_verify_result_is_json_safe():
    json.dumps(verify_shard_hash(coerce_shard_entry(_entry())))
    json.dumps(verify_shard_hash(_entry()))
