"""Pure, deterministic construction of safe eval cases from failed Shards.

This module contains no I/O and makes no network/model calls. It takes an
already-built :class:`~openshard.history.shard_contract.ShardReceipt` and its
:class:`~openshard.history.failures.FailureClassification` and assembles a small,
redacted, versioned eval-case dict — the first local bridge from run history to
eval generation (not OSN-Bench, not a benchmark platform).

Safety: an eval case stores only safe structured metadata — the receipt
``shard_id`` (sanitised before any filename use), the classifier's already-safe
``category`` / ``confidence`` / ``reasons`` / ``signals``, and a sanitised copy
of the short task text. It never carries raw file contents, diffs, transcripts,
``error_message`` text, approval reasons, developer feedback reason text,
absolute local paths, or secret values.

The functions here never raise on hostile receipt data: anything uncertain is
scrubbed or replaced with a neutral placeholder rather than emitted.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from openshard.history.failures import FailureClassification
    from openshard.history.shard_contract import ShardReceipt

# Versioned, explicit contract. Bump only on breaking schema changes.
EVAL_CASE_SCHEMA_VERSION = "1"
EVAL_CASE_SOURCE = "failed_shard_v1"

# The terminal, not-a-failure taxonomy category (see history/failures.py).
_NO_FAILURE = "no_failure_detected"

# Neutral placeholder used when the task text is empty or scrubbed as unsafe.
_TASK_PLACEHOLDER = "OpenShard failed run eval case"

# Sanitisation caps.
_TASK_MAX_LEN = 120
_EVAL_ID_MAX_LEN = 48
_SECRET_TOKEN_LEN = 24


def is_eligible(classification: "FailureClassification") -> bool:
    """True when the run carries a useful learning signal worth an eval case.

    The failure taxonomy is the single source of eligibility truth: any category
    other than ``no_failure_detected`` means a failure/correction signal was
    detected (verification failed, policy denied, secret-scan finding, manual
    review/fix required, execution error, user rejected/abandoned, partial,
    verification not run, or a retry/unknown failure).
    """
    return bool(classification.category) and classification.category != _NO_FAILURE


def _looks_secret_like(token: str) -> bool:
    """Heuristic: a long opaque run with no spaces, or a key=value/secret: pair."""
    if "=" in token or ":" in token:
        # Drop key-ish assignments outright (api_key=..., token: ...).
        return True
    if len(token) >= _SECRET_TOKEN_LEN and not any(c.isspace() for c in token):
        # Long unbroken token with mixed alnum — likely a key/hash/credential.
        return any(c.isdigit() for c in token) and any(c.isalpha() for c in token)
    return False


def _safe_task_text(value: object) -> str:
    """Return safe, redacted task text suitable for storing in an eval case.

    Accepts only a string. Strips whitespace, removes path-like and secret-like
    tokens, caps length, and falls back to a neutral placeholder when the result
    is empty or unsafe. Never carries raw file contents, diffs, transcripts,
    ``error_message``, approval reasons, or feedback reason text.
    """
    if not isinstance(value, str):
        return _TASK_PLACEHOLDER

    safe_words: list[str] = []
    for word in value.split():
        # Drop absolute/relative paths and any token bearing a path separator:
        # drive-letter (C:\...), POSIX (/usr/...), or backslash/forward-slash runs.
        if "/" in word or "\\" in word:
            continue
        if len(word) >= 2 and word[1] == ":":  # drive-letter prefix like "C:"
            continue
        if _looks_secret_like(word):
            continue
        safe_words.append(word)

    cleaned = " ".join(safe_words).strip()
    if not cleaned:
        return _TASK_PLACEHOLDER
    if len(cleaned) > _TASK_MAX_LEN:
        cleaned = cleaned[:_TASK_MAX_LEN].rstrip()
    return cleaned or _TASK_PLACEHOLDER


def _safe_eval_id(value: object, fallback_source: str) -> str:
    """Return a short, filename-safe id fragment derived from an untrusted value.

    A path-like value (containing ``/``, ``\\``, ``..``, or a drive-letter prefix)
    is **rejected outright** — it never influences the id or filename. In that case,
    and when the value is missing/empty, we fall back to a sanitised fragment of
    ``fallback_source`` (e.g. the generation timestamp), or ``"unknown"`` when the
    fallback is itself unsafe or empty.
    """
    fragment = _sanitise_id_fragment(value)
    if fragment:
        return fragment
    fallback = _sanitise_id_fragment(fallback_source)
    return fallback or "unknown"


def _sanitise_id_fragment(value: object) -> str:
    """Reduce a value to a safe ``[A-Za-z0-9_-]`` fragment, or ``""`` if unsafe.

    Path-like input (containing ``/``, ``\\``, ``..``, or a drive-letter prefix)
    is rejected — returning ``""`` so the caller falls back — rather than scrubbed
    into a filename. Only genuinely non-path values are reduced to a safe fragment.
    """
    if not isinstance(value, str):
        return ""
    token = value.strip()
    if not token:
        return ""
    if "/" in token or "\\" in token or ".." in token or (len(token) >= 2 and token[1] == ":"):
        return ""
    kept = "".join(ch if (ch.isalnum() or ch in {"_", "-"}) else "-" for ch in token)
    # Collapse runs of '-' and trim leading/trailing separators.
    while "--" in kept:
        kept = kept.replace("--", "-")
    kept = kept.strip("-_")
    return kept[:_EVAL_ID_MAX_LEN].strip("-_")


def make_eval_id(receipt: "ShardReceipt", created_at: str) -> str:
    """Build a deterministic, filename-safe eval id from an untrusted shard id."""
    return "eval-" + _safe_eval_id(receipt.shard_id, created_at)


def build_eval_case(
    receipt: "ShardReceipt",
    classification: "FailureClassification",
    created_at: str,
) -> dict:
    """Assemble a safe, versioned eval-case dict from a receipt + classification.

    Reads only whitelisted safe fields. ``signals`` and ``reasons`` come straight
    from the failure taxonomy, which is already guaranteed not to leak secrets,
    absolute paths, raw error text, or feedback reason text.
    """
    signals = classification.signals if isinstance(classification.signals, dict) else {}

    return {
        "schema_version": EVAL_CASE_SCHEMA_VERSION,
        "eval_id": make_eval_id(receipt, created_at),
        "created_at": created_at,
        "source": EVAL_CASE_SOURCE,
        "source_shard_id": _safe_eval_id(receipt.shard_id, created_at),
        "task": _safe_task_text(receipt.task_short),
        "failure_category": classification.category,
        "failure_confidence": classification.confidence,
        "failure_reasons": list(classification.reasons or []),
        "signals": {
            "verification": signals.get("verification"),
            "policy_denied": bool(signals.get("policy_denied")),
            "manual_review_required": bool(signals.get("manual_review_required")),
            "secret_scan_findings": int(signals.get("secret_scan_findings") or 0),
            "feedback_outcome": signals.get("feedback_outcome"),
            "error_class": signals.get("error_class"),
        },
        "expected_outcome": {
            "verification_should_pass": True,
            "manual_review_required": False,
        },
        "constraints": {
            "no_raw_file_contents": True,
            "no_raw_transcripts": True,
            "redacted": True,
        },
        "metadata": {
            "verification_status": signals.get("verification"),
            "feedback_outcome": signals.get("feedback_outcome"),
            "error_class": signals.get("error_class"),
        },
    }
