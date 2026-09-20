"""Future-only scoring for the Issue #44 h0-bypass controls.

The functions in this module score retained evaluator records.  They do not
select probes, donors, prefixes, or model outputs.  In particular, an authored
fixture value and an externally supplied probability vector remain diagnostic
controls rather than evidence that a model learned a semantic distribution.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
import math
from typing import Any

from .benchmark_v3_model import BOS_ID, FACTOR_ORDER, FACTOR_SIZES


CONTROL_TYPES = (
    "same_class_soft_shape",
    "alternate_class_one_hot",
    "alternate_class_donor_soft",
)
PREFIX_STRATA = ("bos_start_primary", "common_prefix_secondary")
GATE_PREFIX_REASONS = (
    "invalid_or_ambiguous_common_prefix",
    "target_slot_already_emitted_before_intervention",
    "target_gate_unreachable_after_prefix",
)
STRUCTURAL_UNAVAILABLE_REASONS = (
    "missing_donor",
    "unchanged_shape",
    *GATE_PREFIX_REASONS,
)


def _frame(value: Any, name: str) -> dict[str, str] | None:
    if value is None:
        return None
    if not isinstance(value, Mapping) or set(value) != set(FACTOR_ORDER):
        raise ValueError(f"{name} must be null or a complete four-factor frame")
    result = {field: value[field] for field in FACTOR_ORDER}
    if any(not isinstance(item, str) or not item for item in result.values()):
        raise ValueError(f"{name} factor values must be nonempty strings")
    return result


def _probability_map(value: Any, name: str) -> dict[str, list[float]]:
    if not isinstance(value, Mapping) or set(value) != set(FACTOR_ORDER):
        raise ValueError(f"{name} must contain exactly the four factor vectors")
    result: dict[str, list[float]] = {}
    for field in FACTOR_ORDER:
        vector = value[field]
        if not isinstance(vector, Sequence) or isinstance(vector, (str, bytes)):
            raise ValueError(f"{name}.{field} must be a probability vector")
        values = list(vector)
        if len(values) != FACTOR_SIZES[field]:
            raise ValueError(f"{name}.{field} has the wrong width")
        if any(type(item) not in (int, float) or not math.isfinite(float(item)) or float(item) < 0.0 for item in values):
            raise ValueError(f"{name}.{field} must contain finite nonnegative values")
        floats = [float(item) for item in values]
        if not math.isclose(sum(floats), 1.0, rel_tol=1e-6, abs_tol=1e-6):
            raise ValueError(f"{name}.{field} must sum to one")
        result[field] = floats
    return result


def _schedule(value: Any, name: str) -> list[list[bool]]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or not value:
        raise ValueError(f"{name} must be a nonempty gate schedule")
    result: list[list[bool]] = []
    for row in value:
        if not isinstance(row, Sequence) or isinstance(row, (str, bytes)):
            raise ValueError(f"{name} rows must be four-factor boolean vectors")
        vector = list(row)
        if len(vector) != len(FACTOR_ORDER) or any(type(item) is not bool for item in vector):
            raise ValueError(f"{name} rows must be four-factor boolean vectors")
        result.append(vector)
    return result


def _mask(value: Any, name: str, *, length: int) -> list[bool]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{name} must be a boolean mask")
    result = list(value)
    if len(result) != length or any(type(item) is not bool for item in result):
        raise ValueError(f"{name} must match its gate schedule")
    return result


def assigned_structural_unavailable_reason(record: Mapping[str, Any]) -> str | None:
    """Apply the frozen mutually exclusive priority to diagnostic flags."""

    if record.get("missing_donor") is True:
        return "missing_donor"
    if record.get("unchanged_shape") is True:
        return "unchanged_shape"
    if record.get("intervention_unavailable") is True:
        reason = record.get("intervention_unavailable_reason")
        if reason not in GATE_PREFIX_REASONS:
            raise ValueError("intervention_unavailable requires one gate/prefix reason")
        return str(reason)
    if record.get("intervention_unavailable_reason") is not None:
        raise ValueError("gate/prefix reason requires intervention_unavailable")
    return None


def validate_control_record(record: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the maps, traces, identity, and availability of one raw record."""

    if not isinstance(record, Mapping):
        raise TypeError("control record must be a mapping")
    control = record.get("control_type")
    if control not in CONTROL_TYPES:
        raise ValueError("unknown control type")
    stratum = record.get("prefix_stratum")
    if stratum not in PREFIX_STRATA:
        raise ValueError("unknown prefix stratum")
    factor = record.get("target_factor")
    if factor not in FACTOR_ORDER:
        raise ValueError("unknown target factor")
    prefix = record.get("prefix_token_ids")
    if not isinstance(prefix, Sequence) or isinstance(prefix, (str, bytes)) or not prefix or any(type(item) is not int for item in prefix):
        raise ValueError("prefix_token_ids must be a nonempty integer sequence")
    if stratum == "bos_start_primary" and list(prefix) != [BOS_ID]:
        raise ValueError("BOS-start primary prefix must contain BOS only")

    baseline_map = _probability_map(record.get("baseline_probability_map"), "baseline_probability_map")
    replacement_map = _probability_map(record.get("replacement_probability_map"), "replacement_probability_map")
    for other in FACTOR_ORDER:
        if other != factor and replacement_map[other] != baseline_map[other]:
            raise ValueError("non-target probability vector changed")

    expected = _schedule(record.get("expected_target_gate_schedule"), "expected_target_gate_schedule")
    _mask(record.get("expected_target_gate_mask"), "expected_target_gate_mask", length=len(expected))
    baseline_observed = _schedule(record.get("baseline_observed_target_gate_schedule"), "baseline_observed_target_gate_schedule")
    _mask(record.get("baseline_target_gate_mask"), "baseline_target_gate_mask", length=len(baseline_observed))
    intervention_observed = _schedule(record.get("intervention_observed_target_gate_schedule"), "intervention_observed_target_gate_schedule")
    _mask(record.get("intervention_target_gate_mask"), "intervention_target_gate_mask", length=len(intervention_observed))

    reason = assigned_structural_unavailable_reason(record)
    declared = record.get("structural_unavailable_reason")
    if declared != reason:
        raise ValueError("structural_unavailable_reason does not match frozen priority")
    if stratum == "bos_start_primary" and reason in GATE_PREFIX_REASONS:
        raise ValueError("BOS-start primary gate/prefix unavailability is a protocol violation")

    baseline = _frame(record.get("baseline_parsed_frame"), "baseline_parsed_frame")
    intervention = _frame(record.get("intervention_parsed_frame"), "intervention_parsed_frame")
    requested = record.get("requested_value")
    if not isinstance(requested, str) or not requested:
        raise ValueError("requested_value must be a nonempty canonical value")
    result = dict(record)
    result["baseline_probability_map"] = baseline_map
    result["replacement_probability_map"] = replacement_map
    result["baseline_parsed_frame"] = baseline
    result["intervention_parsed_frame"] = intervention
    result["structural_unavailable_reason"] = reason
    return result


def score_control_record(record: Mapping[str, Any]) -> dict[str, Any]:
    """Derive the exact frozen booleans for one retained control record."""

    result = validate_control_record(record)
    factor = str(result["target_factor"])
    requested = str(result["requested_value"])
    baseline = result["baseline_parsed_frame"]
    intervention = result["intervention_parsed_frame"]
    unavailable = result["structural_unavailable_reason"] is not None
    baseline_failed = baseline is None
    intervention_failed = intervention is None
    both = baseline is not None and intervention is not None
    already = baseline is not None and baseline[factor] == requested
    changed = both and baseline[factor] != intervention[factor]
    requested_success = intervention is not None and intervention[factor] == requested
    nontrivial_requested = both and baseline[factor] != requested and intervention[factor] == requested
    non_target_preserved = both and all(
        baseline[field] == intervention[field] for field in FACTOR_ORDER if field != factor
    )
    scored = not unavailable and both
    eligible = scored and not already
    result.update({
        "available": not unavailable,
        "baseline_parse_failed": baseline_failed,
        "intervention_parse_failed": intervention_failed,
        "baseline_already_requested": already,
        "target_changed": changed,
        "requested_value_success": requested_success,
        "requested_value_success_and_non_target_preserved": requested_success and non_target_preserved,
        "nontrivial_requested_success": nontrivial_requested,
        "non_target_preserved": non_target_preserved,
        "nontrivial_joint_success": nontrivial_requested and non_target_preserved,
        "scored": scored,
        "nontrivial_eligible": eligible,
    })
    return result


def _metric(count: int, denominator: int) -> dict[str, int | float | None]:
    return {"count": count, "denominator": denominator, "rate": (count / denominator if denominator else None)}


def aggregate_control_records(
    records: Sequence[Mapping[str, Any]], *, expected_scheduled_count: int | None = None,
) -> dict[str, Any]:
    """Aggregate one arm/seed/control/stratum without changing denominators."""

    if not records:
        raise ValueError("cannot aggregate an empty control record set")
    scored = [score_control_record(record) for record in records]
    identity = {(row["control_type"], row["prefix_stratum"]) for row in scored}
    if len(identity) != 1:
        raise ValueError("aggregate records must share control type and prefix stratum")
    scheduled = len(scored)
    if expected_scheduled_count is not None and scheduled != expected_scheduled_count:
        raise ValueError("scheduled control count differs from the frozen denominator")
    reasons = [row["structural_unavailable_reason"] for row in scored]
    missing = sum(reason == "missing_donor" for reason in reasons)
    unchanged = sum(reason == "unchanged_shape" for reason in reasons)
    intervention_unavailable = sum(reason in GATE_PREFIX_REASONS for reason in reasons)
    structural = missing + unchanged + intervention_unavailable
    available = scheduled - structural
    both = sum(row["scored"] for row in scored)
    eligible = sum(row["nontrivial_eligible"] for row in scored)
    control_type, prefix_stratum = next(iter(identity))

    def count(name: str, *, available_only: bool = False) -> int:
        return sum(bool(row[name]) and (row["available"] or not available_only) for row in scored)

    baseline_already_count = count("baseline_already_requested", available_only=True)
    baseline_parse_failed_count = count("baseline_parse_failed", available_only=True)
    intervention_parse_failed_count = count("intervention_parse_failed", available_only=True)
    target_changed_count = count("target_changed", available_only=True)
    non_target_preserved_count = count("non_target_preserved", available_only=True)
    requested_success_count = count("requested_value_success", available_only=True)
    requested_joint_count = count("requested_value_success_and_non_target_preserved", available_only=True)
    nontrivial_requested_count = count("nontrivial_requested_success", available_only=True)
    nontrivial_joint_count = count("nontrivial_joint_success", available_only=True)
    return {
        "control_type": control_type,
        "prefix_stratum": prefix_stratum,
        "scheduled_count": scheduled,
        "available_count": available,
        "structural_unavailable_count": structural,
        "intervention_unavailable_count": intervention_unavailable,
        "missing_count": missing,
        "unchanged_shape_count": unchanged,
        "scored_count": both,
        "nontrivial_eligible_count": eligible,
        "baseline_already_requested_count": baseline_already_count,
        "baseline_parse_failed_count": baseline_parse_failed_count,
        "intervention_parse_failed_count": intervention_parse_failed_count,
        "target_changed_count": target_changed_count,
        "requested_value_success_count": requested_success_count,
        "nontrivial_requested_success_count": nontrivial_requested_count,
        "non_target_preserved_count": non_target_preserved_count,
        "nontrivial_joint_success_count": nontrivial_joint_count,
        "denominator": scheduled,
        "baseline_already_requested": _metric(baseline_already_count, available),
        "baseline_parse_failed": _metric(baseline_parse_failed_count, available),
        "intervention_parse_failed": _metric(intervention_parse_failed_count, available),
        "target_changed": _metric(target_changed_count, both),
        "non_target_preserved": _metric(non_target_preserved_count, both),
        "requested_value_success": _metric(requested_success_count, available),
        "requested_value_success_and_non_target_preserved": _metric(requested_joint_count, available),
        "nontrivial_requested_success": _metric(nontrivial_requested_count, eligible),
        "nontrivial_joint_success_eligible": _metric(nontrivial_joint_count, eligible),
        "nontrivial_joint_success_scheduled": _metric(nontrivial_joint_count, scheduled),
    }


__all__ = [
    "CONTROL_TYPES",
    "GATE_PREFIX_REASONS",
    "PREFIX_STRATA",
    "STRUCTURAL_UNAVAILABLE_REASONS",
    "aggregate_control_records",
    "assigned_structural_unavailable_reason",
    "score_control_record",
    "validate_control_record",
]
