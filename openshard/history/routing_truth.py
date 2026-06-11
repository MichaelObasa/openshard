"""Routing truth - an honest, read-only view of how a run actually routed.

OpenShard carries rich per-role model-selection metadata (planner / executor /
validator), but the default execution path almost always uses a *single*
runtime model. Per-role selection is advisory unless experimental tier dispatch
both ran (``tier_dispatch_receipt.enabled``) and was applied
(``tier_dispatch_receipt.applied``). The proof surfaces must not let a reader
conclude that three models were dispatched when one was.

This module derives that truth from fields already on a run entry, so old
records render safely with conservative defaults and no migration. It keeps two
things deliberately separate:

1. **Actual runtime model selection** - ``runtime_model`` / ``routing_mode`` /
   ``selection_source``. ``routing_mode`` always describes how the *real*
   runtime model was chosen; it is never set to an advisory-flavoured value.
2. **Role-routing truth** - ``role_dispatch_status`` / ``role_selection_mode``
   and the per-role recommended models plus their dispatched flags. The advisory
   nature of per-role metadata lives here, never in ``routing_mode``.

Design constraints (mirrors ``proof_contract.py``):

* Pure and deterministic. No I/O. ``build_routing_truth`` never raises.
* Never fabricates dispatch: a role is only ``*_dispatched=True`` when the run
  records that it actually ran.
* Output is status tokens, short model slugs, and plain-ASCII summary text only.
  No em dashes and no non-ASCII separators in emitted strings.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

# Runtime routing modes - how the real runtime model was chosen.
ROUTING_KEYWORD = "keyword"
ROUTING_SCORED = "scored"
ROUTING_USER_SELECTED = "user_selected"
ROUTING_TIER_DISPATCH = "tier_dispatch"
ROUTING_UNKNOWN = "unknown"

# Selection sources - the provenance of the runtime model choice.
SOURCE_DETERMINISTIC = "deterministic"
SOURCE_USER_SELECTED = "user_selected"
SOURCE_EXPERIMENTAL = "experimental"
SOURCE_UNKNOWN = "unknown"

# Role-routing truth.
DISPATCH_NONE = "not_dispatched"
DISPATCH_PARTIAL = "partially_dispatched"
DISPATCH_FULL = "dispatched"

ROLE_ADVISORY_ONLY = "advisory_only"
ROLE_DISPATCHED = "dispatched"
ROLE_UNAVAILABLE = "unavailable"

_ROLES = ("planner", "executor", "validator")


@dataclass
class RoutingTruth:
    """An honest, JSON-serialisable view of a run's routing."""

    # 1. Actual runtime model selection
    runtime_model: str | None
    routing_mode: str
    selection_source: str
    # 2. Role-routing truth
    role_dispatch_status: str
    role_selection_mode: str
    planner_model: str | None
    executor_model: str | None
    validator_model: str | None
    planner_dispatched: bool
    executor_dispatched: bool
    validator_dispatched: bool
    routing_truth_summary: str
    # 3. Advisory wiring provenance (added by PRs 1, 3, 4, 5).
    #    All fields have safe defaults so legacy Shards degrade gracefully.
    model_resolution: str = "unknown"       # "registry" | "hardcoded" | "unknown"
    feedback_routing_applied: bool = False
    mode_policy_applied: bool = False
    executor_source: str = "unknown"        # "advisory" | "override" | "heuristic" | "unknown"
    # 4. Provider-aware eligibility (recorded, not enforced). Safe defaults
    #    so legacy Shards degrade gracefully.
    available_providers: list[str] = field(default_factory=list)
    routing_constraints: dict | None = None


def _get(obj: object, key: str, default: object = None) -> object:
    """Read *key* from a dict or an attribute holder; tolerant of either."""
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _advisory_role_model(entry: dict, role: str) -> str | None:
    """Recommended tier/model for *role* from advisory per-role metadata.

    Prefers ``model_candidate_scoring.selected_by_role`` (final tier per role),
    then ``model_selection_decision.roles`` (role -> model_tier). Returns a tier
    name or model id, or None when no advisory metadata names this role.
    """
    mcs = entry.get("model_candidate_scoring")
    if mcs is not None:
        selected = _get(mcs, "selected_by_role", {}) or {}
        if isinstance(selected, dict):
            val = selected.get(role)
            if val:
                return str(val)

    msd = entry.get("model_selection_decision")
    if msd is not None:
        roles = _get(msd, "roles", None)
        if not isinstance(roles, list):  # type: ignore[attr-defined]  # _get returns object; narrowed to list below
            roles = []
        for r in roles:
            rname = _get(r, "role", "")
            if rname == role:
                tier = _get(r, "model_tier", "")
                if tier:
                    return str(tier)
    return None


def _has_advisory_role_metadata(entry: dict) -> bool:
    return (
        entry.get("model_candidate_scoring") is not None
        or entry.get("model_selection_decision") is not None
    )


def build_routing_truth(entry: object) -> RoutingTruth:
    """Derive a :class:`RoutingTruth` from a run entry. Never raises.

    Reads only fields already present on the entry, so legacy records degrade to
    conservative ``not_dispatched`` / ``unavailable`` defaults.
    """
    if not isinstance(entry, dict):
        entry = {}

    runtime_model = entry.get("execution_model") or entry.get("routing_selected_model")
    if runtime_model is not None:
        runtime_model = str(runtime_model)

    tdr = entry.get("tier_dispatch_receipt") or {}
    if not isinstance(tdr, dict):
        tdr = {k: _get(tdr, k) for k in (
            "enabled", "applied",
            "planner_model", "executor_model", "validator_model",
            "planner_model_actual", "executor_model_actual",
            "validator_model_actual", "validator_dispatch_status",
        )}
    dispatch_active = bool(tdr.get("enabled")) and bool(tdr.get("applied"))

    # --- Role-routing truth (independent of runtime routing) ---------------
    planner_dispatched = executor_dispatched = validator_dispatched = False
    planner_model = executor_model = validator_model = None

    if dispatch_active:
        planner_model = tdr.get("planner_model")
        executor_model = tdr.get("executor_model")
        validator_model = tdr.get("validator_model")
        planner_dispatched = bool(tdr.get("planner_model_actual"))
        executor_dispatched = bool(tdr.get("executor_model_actual"))
        validator_dispatched = bool(tdr.get("validator_model_actual")) or (
            tdr.get("validator_dispatch_status") == "applied"
        )
        dispatched_flags = [planner_dispatched, executor_dispatched, validator_dispatched]
        if all(dispatched_flags):
            role_dispatch_status = DISPATCH_FULL
        elif any(dispatched_flags):
            role_dispatch_status = DISPATCH_PARTIAL
        else:
            role_dispatch_status = DISPATCH_NONE
        role_selection_mode = ROLE_DISPATCHED if any(dispatched_flags) else ROLE_ADVISORY_ONLY
    elif _has_advisory_role_metadata(entry):
        planner_model = _advisory_role_model(entry, "planner")
        executor_model = _advisory_role_model(entry, "executor")
        validator_model = _advisory_role_model(entry, "validator")
        role_dispatch_status = DISPATCH_NONE
        role_selection_mode = ROLE_ADVISORY_ONLY
    else:
        role_dispatch_status = DISPATCH_NONE
        role_selection_mode = ROLE_UNAVAILABLE

    # --- Actual runtime model selection ------------------------------------
    if dispatch_active:
        routing_mode = ROUTING_TIER_DISPATCH
        selection_source = SOURCE_EXPERIMENTAL
    elif entry.get("user_selected_model"):
        # No entry field currently sets this; branch reserved for real
        # user-selected dispatch so the truth stays honest once it exists.
        routing_mode = ROUTING_USER_SELECTED
        selection_source = SOURCE_USER_SELECTED
    elif (
        entry.get("routing_selected_model")
        or entry.get("routing_scores")
        or entry.get("routing_candidates")
    ):
        routing_mode = ROUTING_SCORED
        selection_source = SOURCE_DETERMINISTIC
    elif (
        entry.get("routing_model")
        or entry.get("routing_rationale")
        or entry.get("routing_category")
    ):
        routing_mode = ROUTING_KEYWORD
        selection_source = SOURCE_DETERMINISTIC
    else:
        routing_mode = ROUTING_UNKNOWN
        selection_source = SOURCE_UNKNOWN

    summary = _build_summary(
        runtime_model,
        role_selection_mode,
        role_dispatch_status,
        planner_model,
        executor_model,
        validator_model,
    )

    # --- Advisory wiring provenance ----------------------------------------
    model_resolution = str(entry.get("model_resolution") or "unknown")
    feedback_routing_applied = bool(entry.get("feedback_routing_applied", False))
    mode_policy_applied = bool(entry.get("mode_policy_applied", False))
    executor_source = str(entry.get("executor_source") or "unknown")

    # --- Provider-aware eligibility (recorded, not enforced) ----------------
    _ap_raw = entry.get("available_providers")
    available_providers = (
        [str(p) for p in _ap_raw] if isinstance(_ap_raw, (list, tuple)) else []
    )
    _rc_raw = entry.get("routing_constraints")
    routing_constraints = _rc_raw if isinstance(_rc_raw, dict) else None

    return RoutingTruth(
        runtime_model=runtime_model,
        routing_mode=routing_mode,
        selection_source=selection_source,
        role_dispatch_status=role_dispatch_status,
        role_selection_mode=role_selection_mode,
        planner_model=planner_model,
        executor_model=executor_model,
        validator_model=validator_model,
        planner_dispatched=planner_dispatched,
        executor_dispatched=executor_dispatched,
        validator_dispatched=validator_dispatched,
        routing_truth_summary=summary,
        model_resolution=model_resolution,
        feedback_routing_applied=feedback_routing_applied,
        mode_policy_applied=mode_policy_applied,
        executor_source=executor_source,
        available_providers=available_providers,
        routing_constraints=routing_constraints,
    )


def _build_summary(
    runtime_model: str | None,
    role_selection_mode: str,
    role_dispatch_status: str,
    planner_model: str | None,
    executor_model: str | None,
    validator_model: str | None,
) -> str:
    """One plain-ASCII summary line. No em dashes, no non-ASCII separators."""
    model = runtime_model or "unknown"
    if role_selection_mode == ROLE_DISPATCHED:
        p = planner_model or "unknown"
        e = executor_model or "unknown"
        v = validator_model or "unknown"
        if role_dispatch_status == DISPATCH_PARTIAL:
            return (
                f"Model used per role (partial) - planner: {p}, executor: {e}, "
                f"validator: {v}"
            )
        return f"Model used per role - planner: {p}, executor: {e}, validator: {v} (dispatched)"
    if role_selection_mode == ROLE_ADVISORY_ONLY:
        return (
            f"Model used: {model}. Role routing: advisory only - "
            "planner/executor/validator were recommended, not dispatched"
        )
    return f"Model used: {model}. Role routing: not applicable"


def routing_truth_to_dict(rt: RoutingTruth) -> dict:
    """JSON-serialisable dict form of a :class:`RoutingTruth`."""
    return asdict(rt)


def render_routing_truth_lines(rt: RoutingTruth, detail: str) -> list[str]:
    """Human lines for the truth view. Plain ASCII; safe at any detail level.

    * ``default`` - the always-visible anti-overclaim line(s); empty when there
      is no per-role metadata to overclaim.
    * ``more`` / ``full`` - adds the explicit ``Model used`` line plus advisory
      wiring provenance when non-default values are present.
    """
    role_line = _role_line(rt)
    if detail == "default":
        return [role_line] if role_line else []

    lines: list[str] = []
    if rt.runtime_model:
        lines.append(f"Model used: {rt.runtime_model}")
    if role_line:
        lines.append(role_line)

    # Emit wiring provenance at more/full detail — only when values differ
    # from defaults so legacy Shards stay clean.
    _provenance_parts: list[str] = []
    if rt.model_resolution not in ("unknown", "hardcoded"):
        _provenance_parts.append(f"model={rt.model_resolution}")
    if rt.executor_source not in ("unknown", "heuristic"):
        _provenance_parts.append(f"executor={rt.executor_source}")
    if rt.feedback_routing_applied:
        _provenance_parts.append("feedback=applied")
    if rt.mode_policy_applied:
        _provenance_parts.append("mode_policy=applied")
    if _provenance_parts:
        lines.append(f"  Routing: {' | '.join(_provenance_parts)}")

    return lines


def _role_line(rt: RoutingTruth) -> str:
    if rt.role_selection_mode == ROLE_ADVISORY_ONLY:
        return (
            "Role routing: advisory only - planner/executor/validator were "
            "recommended, not dispatched"
        )
    if rt.role_selection_mode == ROLE_DISPATCHED:
        p = rt.planner_model or "unknown"
        e = rt.executor_model or "unknown"
        v = rt.validator_model or "unknown"
        if rt.role_dispatch_status == DISPATCH_PARTIAL:
            return (
                f"Role routing: partially dispatched - planner: {p}, executor: {e}, "
                f"validator: {v}"
            )
        return f"Role routing: dispatched - planner: {p}, executor: {e}, validator: {v}"
    return ""
