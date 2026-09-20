from __future__ import annotations

import copy
import json

import pytest

from norishio_lm import benchmark_v3_h0_protocol as protocol
from norishio_lm.benchmark_v3_h0_fixture import build


def test_tracked_descriptor_is_literal_and_canonical() -> None:
    descriptor = protocol.load_protocol()
    assert descriptor == protocol.build_descriptor()
    assert protocol.protocol_digest(descriptor) == protocol.PROTOCOL_SHA256
    assert descriptor["inherited_future_protocol"]["descriptor_sha256"] == protocol.INHERITED_FUTURE_PROTOCOL_SHA256
    assert descriptor["fixture"]["content_digest_sha256"] == protocol.FIXTURE_CONTENT_SHA256
    assert descriptor["scope"]["final_split"] is None
    assert descriptor["scope"]["no_final_split"] is True


def test_primary_fsm_schedule_and_reachability_are_complete() -> None:
    descriptor = protocol.load_protocol()
    probes = descriptor["fsm"]["primary_probe_records"]
    assert len(probes) == 8
    assert len({item["row_id"] for item in probes}) == 8
    for item in probes:
        assert item["prefix_token_ids"] == [1]
        assert len(item["expected_target_gate_schedule"]) == len(item["expected_target_gate_mask"]) == len(item["byte_values"])
        assert item["target_gate_reachable_after_prefix"] is True
        assert item["intervention_unavailable"] is False


def test_donor_table_is_complete_and_lexicographic() -> None:
    descriptor = protocol.load_protocol()
    table = descriptor["donors"]["table"]
    assert set(table) == {"unseen_pair", "seen_pair/unseen_triple"}
    for support in table.values():
        assert len(support) == 4
        for probe in support.values():
            assert set(probe) == {"participant", "time", "event", "operator"}
            for factor in probe:
                assert probe[factor]
                for authored_class, donor in probe[factor].items():
                    assert donor["authored_class"] == authored_class
                    assert donor["factor"] == factor
                    assert donor["row_id"].endswith("-v0")


def test_valid_preflight_and_run_binding() -> None:
    result = protocol.preflight_protocol()
    assert result["preflight"] is True
    metadata = protocol.bind_run_metadata("issue44-H1L1-7")
    assert protocol.validate_run_metadata(metadata)["experiment_descriptor_sha256"] == protocol.PROTOCOL_SHA256


def test_seed_major_paired_run_and_evaluation_orders_are_frozen() -> None:
    training = protocol.load_protocol()["training"]
    assert training["run_order"] == [
        "H1L1 seed7", "H1L1_ANCHOR seed7",
        "H1L1 seed17", "H1L1_ANCHOR seed17",
        "H1L1 seed29", "H1L1_ANCHOR seed29",
    ]
    assert training["evaluation_order"] == [
        "ordinary_confirmation",
        "baseline_probability_maps_and_outputs",
        "donor_probability_maps",
        "internal_controls_in_fixture_support_factor_control_order",
        "source_swaps_in_fixture_support_factor_order",
    ]


def test_all_review_cutoffs_and_interpretation_branches_are_frozen() -> None:
    descriptor = protocol.load_protocol()
    thresholds = descriptor["selection"]["proposed_thresholds"]
    assert thresholds == {
        "preserved_max_drop": 0.05,
        "improved_min_gain": 0.10,
        "strong_head_min": 0.80,
        "weak_scheduled_joint_max": 0.05,
        "positive_scheduled_joint_min": 0.25,
        "same_class_material_effect_min": 0.10,
        "constant_source_close_max_gap": 0.05,
        "unseen_pair_near_zero_max": 0.02,
        "seen_vs_unseen_material_gap_min": 0.10,
    }
    branch_ids = {branch["id"] for branch in descriptor["decision_branches"]}
    assert {
        "same_class_shape_sensitivity", "constant_source_close",
        "fixture_specific_compositional_failure", "frame_recovery_without_surface_recovery",
        "one_hot_only_interpretation", "donor_soft_only_interpretation",
    } <= branch_ids


@pytest.mark.parametrize(
    ("path", "replacement", "message"),
    [
        (("fixture", "content_digest_sha256"), "0" * 64, "descriptor"),
        (("probes", "primary_prefix", "prefix_token_ids"), [1, 3], "descriptor"),
        (("donors", "selection_rule"), "reseat", "descriptor"),
        (("fsm", "primary_probe_records", 0, "expected_target_gate_mask", 14), False, "descriptor"),
        (("training", "budget", "max_attempts"), 7, "descriptor"),
        (("training", "run_order", 0), "H1L1 seed17", "descriptor"),
        (("training", "evaluation_order", 0), "source_swaps_in_fixture_support_factor_order", "descriptor"),
        (("model", "initial_state_by_seed", "7", "H1L1"), "f" * 64, "descriptor"),
    ],
)
def test_tampered_descriptor_is_rejected(path, replacement, message) -> None:
    value = copy.deepcopy(protocol.load_protocol())
    cursor = value
    for key in path[:-1]:
        cursor = cursor[key]
    cursor[path[-1]] = replacement
    with pytest.raises(ValueError, match=message):
        protocol.validate_protocol(value)


def test_tampered_fixture_is_rejected_before_preflight() -> None:
    bundle = copy.deepcopy(build())
    bundle["confirmation"][0]["inputs"]["text"] = "h0-source-tampered"
    with pytest.raises(ValueError, match="bundle differs|fixture|source"):
        protocol.preflight_protocol(fixture_bundle=bundle)


def test_missing_raw_field_is_rejected() -> None:
    raw = {field: None for field in protocol.RAW_REQUIRED_FIELDS}
    raw.pop("target_text")
    with pytest.raises(ValueError, match="missing raw fields"):
        protocol.preflight_protocol(raw_artifact=raw)


def test_descriptor_wrapper_has_exact_three_fields() -> None:
    payload = json.loads(protocol.PROTOCOL_PATH.read_text(encoding="utf-8"))
    assert set(payload) == {"schema", "descriptor_sha256", "descriptor"}
    assert payload["descriptor_sha256"] == protocol.PROTOCOL_SHA256
