from __future__ import annotations

import copy

import pytest

from norishio_lm.benchmark_v3_h0_metrics import (
    aggregate_control_records,
    assigned_structural_unavailable_reason,
    score_control_record,
)
from norishio_lm.benchmark_v3_model import BOS_ID, FACTOR_ORDER, FACTOR_SIZES


def _maps() -> dict[str, list[float]]:
    return {
        field: [1.0, *([0.0] * (FACTOR_SIZES[field] - 1))]
        for field in FACTOR_ORDER
    }


def _schedule() -> list[list[bool]]:
    return [[True, True, True, True], [False, True, False, False]]


def _frame(participant: str = "p0") -> dict[str, str]:
    return {"participant": participant, "time": "t0", "event": "e0", "operator": "o0"}


def _record(**overrides):
    baseline_map = _maps()
    replacement = copy.deepcopy(baseline_map)
    replacement["participant"] = [0.0, 1.0, 0.0, 0.0, 0.0, 0.0]
    result = {
        "control_type": "alternate_class_one_hot",
        "prefix_stratum": "bos_start_primary",
        "prefix_token_ids": [BOS_ID],
        "target_factor": "participant",
        "requested_value": "p1",
        "baseline_probability_map": baseline_map,
        "replacement_probability_map": replacement,
        "expected_target_gate_schedule": _schedule(),
        "expected_target_gate_mask": [True, False],
        "baseline_observed_target_gate_schedule": _schedule(),
        "baseline_target_gate_mask": [True, False],
        "intervention_observed_target_gate_schedule": _schedule(),
        "intervention_target_gate_mask": [True, False],
        "baseline_parsed_frame": _frame(),
        "intervention_parsed_frame": _frame("p1"),
        "missing_donor": False,
        "unchanged_shape": False,
        "intervention_unavailable": False,
        "intervention_unavailable_reason": None,
        "structural_unavailable_reason": None,
    }
    result.update(overrides)
    return result


def test_nontrivial_joint_success_is_explicit() -> None:
    scored = score_control_record(_record())

    assert scored["baseline_already_requested"] is False
    assert scored["target_changed"] is True
    assert scored["requested_value_success"] is True
    assert scored["non_target_preserved"] is True
    assert scored["nontrivial_requested_success"] is True
    assert scored["nontrivial_joint_success"] is True


def test_already_requested_never_enters_nontrivial_eligible_denominator() -> None:
    row = _record(baseline_parsed_frame=_frame("p1"), intervention_parsed_frame=_frame("p1"))
    aggregate = aggregate_control_records([row], expected_scheduled_count=1)

    assert aggregate["baseline_already_requested"] == {"count": 1, "denominator": 1, "rate": 1.0}
    assert aggregate["nontrivial_eligible_count"] == 0
    assert aggregate["nontrivial_joint_success_eligible"]["rate"] is None
    assert aggregate["nontrivial_joint_success_scheduled"]["rate"] == 0.0


@pytest.mark.parametrize(
    ("overrides", "reason"),
    [
        ({"missing_donor": True, "control_type": "alternate_class_donor_soft", "structural_unavailable_reason": "missing_donor"}, "missing_donor"),
        ({"unchanged_shape": True, "control_type": "same_class_soft_shape", "structural_unavailable_reason": "unchanged_shape"}, "unchanged_shape"),
        ({
            "prefix_stratum": "common_prefix_secondary",
            "prefix_token_ids": [BOS_ID, 4],
            "intervention_unavailable": True,
            "intervention_unavailable_reason": "target_gate_unreachable_after_prefix",
            "structural_unavailable_reason": "target_gate_unreachable_after_prefix",
        }, "target_gate_unreachable_after_prefix"),
    ],
)
def test_structural_unavailable_reasons_are_exclusive(overrides, reason) -> None:
    row = _record(**overrides)
    assert assigned_structural_unavailable_reason(row) == reason
    aggregate = aggregate_control_records([row])
    assert aggregate["structural_unavailable_count"] == 1
    assert aggregate["available_count"] == 0
    assert aggregate["missing_count"] + aggregate["unchanged_shape_count"] + aggregate["intervention_unavailable_count"] == 1


def test_priority_assigns_only_missing_when_diagnostic_flags_overlap() -> None:
    row = _record(
        control_type="alternate_class_donor_soft",
        missing_donor=True,
        unchanged_shape=True,
        intervention_unavailable=True,
        intervention_unavailable_reason="target_gate_unreachable_after_prefix",
        structural_unavailable_reason="missing_donor",
    )
    assert score_control_record(row)["structural_unavailable_reason"] == "missing_donor"


def test_parse_failures_remain_in_available_denominator() -> None:
    row = _record(baseline_parsed_frame=None, intervention_parsed_frame=None)
    aggregate = aggregate_control_records([row])

    assert aggregate["available_count"] == 1
    assert aggregate["scored_count"] == 0
    assert aggregate["baseline_parse_failed"]["rate"] == 1.0
    assert aggregate["intervention_parse_failed"]["rate"] == 1.0


def test_rejects_changed_non_target_vector_and_malformed_trace() -> None:
    changed = _record()
    changed["replacement_probability_map"]["time"] = [0.0, 1.0, 0.0, 0.0, 0.0, 0.0]
    with pytest.raises(ValueError, match="non-target"):
        score_control_record(changed)

    bad_trace = _record(expected_target_gate_mask=[True])
    with pytest.raises(ValueError, match="match its gate schedule"):
        score_control_record(bad_trace)


def test_primary_gate_unavailability_is_protocol_violation() -> None:
    row = _record(
        intervention_unavailable=True,
        intervention_unavailable_reason="target_gate_unreachable_after_prefix",
        structural_unavailable_reason="target_gate_unreachable_after_prefix",
    )
    with pytest.raises(ValueError, match="protocol violation"):
        score_control_record(row)
