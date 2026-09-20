from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from norishio_lm.benchmark_v3_h0_intervention import (
    build_h0_intervention_descriptor,
    evaluate_h0_interventions,
    validate_h0_intervention_descriptor,
)
from norishio_lm.benchmark_v3_h0_model import build_future_h0_model
from norishio_lm.benchmark_v3_h0_metrics import score_control_record
from norishio_lm.benchmark_v3_model import FACTOR_ORDER, FACTOR_SIZES


@pytest.fixture(scope="module")
def evaluation() -> dict:
    descriptor = build_h0_intervention_descriptor()
    model = build_future_h0_model("H1L1", seed=7)
    return evaluate_h0_interventions(model, descriptor)


def test_untrained_engine_has_frozen_record_counts(evaluation: dict) -> None:
    assert evaluation["counts"] == {"source_swaps": 8, "controls": 24}
    assert len(evaluation["source_swaps"]) == 8
    assert len(evaluation["controls"]) == 24
    assert {record["control_type"] for record in evaluation["controls"]} == {
        "same_class_soft_shape", "alternate_class_one_hot", "alternate_class_donor_soft",
    }


def test_controls_hold_non_target_maps_and_select_the_frozen_classes(evaluation: dict) -> None:
    for record in evaluation["controls"]:
        factor = record["target_factor"]
        baseline = record["baseline_probability_map"]
        replacement = record["replacement_probability_map"]
        for other in FACTOR_ORDER:
            if other != factor:
                assert replacement[other] == baseline[other]
        requested = record["requested_class"]
        if record["control_type"] == "alternate_class_one_hot":
            assert replacement[factor] == [
                1.0 if index == requested else 0.0
                for index in range(FACTOR_SIZES[factor])
            ]
            assert record["actual_intervention_class"] == requested
        elif record["control_type"] == "same_class_soft_shape":
            assert max(range(len(replacement[factor])), key=lambda i: replacement[factor][i]) == record["baseline_argmax"]
            assert record["same_class_l1_distance"] > 0.0


def test_donor_soft_missing_is_retained_as_structural_unavailability(evaluation: dict) -> None:
    donor_records = [row for row in evaluation["controls"] if row["control_type"] == "alternate_class_donor_soft"]
    assert len(donor_records) == 8
    for row in donor_records:
        if row["missing_donor"]:
            assert row["structural_unavailable_reason"] == "missing_donor"
            assert row["intervention_generated"] is None
            assert row["available"] is False


def test_expected_and_observed_gate_traces_are_separate(evaluation: dict) -> None:
    assert all(len(row["expected_target_gate_schedule"]) == len(row["expected_target_gate_mask"]) for row in evaluation["controls"])
    assert all(len(row["baseline_observed_target_gate_schedule"]) == len(row["baseline_target_gate_mask"]) for row in evaluation["controls"])
    # The expected trace is authored-target preflight; generated traces come
    # from the model's actual BOS history and may miss the target gate.
    assert any(
        row["expected_target_gate_schedule"] != row["baseline_observed_target_gate_schedule"]
        for row in evaluation["controls"]
    )
    assert any(row["baseline_observed_gate_not_reached"] for row in evaluation["controls"])


def test_one_baseline_snapshot_is_reused_for_each_probe(evaluation: dict) -> None:
    grouped: dict[tuple[str, str], list[dict]] = {}
    for row in evaluation["controls"]:
        grouped.setdefault((row["support_class"], row["probe_row_id"]), []).append(row)
    assert len(grouped) == 8
    for rows in grouped.values():
        assert len(rows) == 3
        assert len({row["baseline_generated_digest"] for row in rows}) == 1
        assert len({row["baseline_probability_map_digest"] for row in rows}) == 1
        assert rows[0]["baseline_generated"] == rows[1]["baseline_generated"] == rows[2]["baseline_generated"]


def test_descriptor_input_tampering_is_rejected() -> None:
    descriptor = build_h0_intervention_descriptor()
    tampered = deepcopy(descriptor)
    tampered["expected_probes"]["unseen_pair"]["participant"]["expected_target_gate_mask"][0] = True
    with pytest.raises(ValueError, match="digest mismatch"):
        validate_h0_intervention_descriptor(tampered)


def test_published_canonical_descriptor_is_accepted() -> None:
    path = Path(__file__).parents[1] / "data" / "benchmark_v3_h0_confirmation" / "experiment-descriptor-v1.json"
    descriptor = json.loads(path.read_text(encoding="utf-8"))["descriptor"]
    normalized = validate_h0_intervention_descriptor(descriptor)
    assert normalized["canonical_descriptor_sha256"] == json.loads(path.read_text(encoding="utf-8"))["descriptor_sha256"]
    assert len(normalized["expected_probes"]["unseen_pair"]) == 4


def test_unchanged_shape_path_is_scored_as_structural_unavailability(evaluation: dict) -> None:
    row = deepcopy(next(item for item in evaluation["controls"] if item["control_type"] == "same_class_soft_shape"))
    row["replacement_probability_map"] = deepcopy(row["baseline_probability_map"])
    row["target_replacement_map"] = deepcopy(row["baseline_probability_map"])
    row["unchanged_shape"] = True
    row["structural_unavailable_reason"] = "unchanged_shape"
    scored = score_control_record(row)
    assert scored["available"] is False
    assert scored["structural_unavailable_reason"] == "unchanged_shape"
