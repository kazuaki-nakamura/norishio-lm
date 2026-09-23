"""Pre-training checks for the Issue #46 freeze and decision rules."""
from __future__ import annotations

from copy import deepcopy

import pytest

from norishio_lm import benchmark_v3_head_fixture as fixture
from norishio_lm.benchmark_v3_head_execute import _validate_saved_row, _vocabulary, decide
from norishio_lm.benchmark_v3_head_protocol import ARMS, SEEDS, build_descriptor


FIELDS = ("participant", "time", "event", "operator")


def _group(scores: dict[str, float]) -> dict:
    return {"fields": {factor: {"class_balanced_accuracy": value,
                                 "scheduled_count": 6, "unavailable_count": 0}
                       for factor, value in scores.items()}}


def _report(*, joint_train: float = 0.9, factor_train: float = 0.9,
            joint_confirm: float = 0.6, factor_confirm: float = 0.8,
            unseen: float = 0.6) -> dict:
    report = {"by_arm_seed": {}}
    for arm in ARMS:
        report["by_arm_seed"][arm] = {}
        for seed in SEEDS:
            train_scores = {factor: joint_train if arm == "JOINT" else factor_train
                            for factor in FIELDS}
            confirm_scores = {factor: joint_confirm if arm == "JOINT" else factor_confirm
                              for factor in FIELDS}
            report["by_arm_seed"][arm][str(seed)] = {
                "train_resubstitution": _group(train_scores),
                "confirmation": {"all": _group(confirm_scores),
                                 "unseen_pair": _group({factor: unseen for factor in FIELDS})},
            }
    return report


def _rows() -> list[dict]:
    return [{"arm": "FACTOR_ONLY", "seed": seed, "split": "confirmation",
             "predicted_frame": {"event": f"E{index}", "operator": f"O{index}"}}
            for seed in SEEDS for index in range(4)]


def test_descriptor_binds_disjoint_fixture_paired_model_and_six_runs() -> None:
    descriptor = build_descriptor()
    assert descriptor["fixture"]["train_rows"] == 384
    assert descriptor["fixture"]["confirmation_rows"] == 96
    assert descriptor["fixture"]["confirmation_support_counts"] == {
        "unseen_pair": 48, "seen_pair/unseen_triple": 48,
    }
    assert len(descriptor["training"]["run_order"]) == 6
    assert descriptor["training"]["max_total_optimizer_updates"] == 3600
    assert all(item["bitwise_equal"] and item["parameter_count"] == 32120
               for item in descriptor["arms"]["paired_initialization"].values())


def test_frozen_interference_branch_uses_both_atomic_factors_all_seeds() -> None:
    descriptor = build_descriptor()
    result = decide(_report(), _rows(), descriptor)
    assert result["branch"] == "joint_objective_interference_leading"
    assert result["predicates"]["interference"] is True
    report = _report()
    report["by_arm_seed"]["FACTOR_ONLY"]["29"]["confirmation"]["all"]["fields"]["time"]["class_balanced_accuracy"] = 0.6
    assert decide(report, _rows(), descriptor)["branch"] != "joint_objective_interference_leading"


def test_weak_single_factor_takes_narrow_branch_before_general_failure() -> None:
    descriptor = build_descriptor()
    report = _report(joint_train=0.3, factor_train=0.9)
    report["by_arm_seed"]["FACTOR_ONLY"]["7"]["train_resubstitution"]["fields"]["participant"]["class_balanced_accuracy"] = 0.4
    assert decide(report, _rows(), descriptor)["branch"] == "narrow_factor_follow_up"
    report["by_arm_seed"]["FACTOR_ONLY"]["7"]["train_resubstitution"]["fields"]["time"]["class_balanced_accuracy"] = 0.4
    assert decide(report, _rows(), descriptor)["branch"] == "basic_optimization_or_capacity_unresolved"


def test_frozen_composition_and_fixture_instability_are_separate() -> None:
    descriptor = build_descriptor()
    composition = _report(joint_confirm=0.7, factor_confirm=0.7, unseen=0.5)
    assert decide(composition, _rows(), descriptor)["branch"] == "compositional_generalization_leading"
    instability = _report(joint_confirm=0.9, factor_confirm=0.9, unseen=0.9)
    assert decide(instability, _rows(), descriptor)["branch"] == "fixture_specific_instability_leading"


def test_raw_row_validation_rejects_changed_gold_and_probability_binding() -> None:
    authored = fixture.build()["train"][0]
    vocabulary = _vocabulary()
    frame = authored["targets"]["frame"]
    argmax = {factor: vocabulary.ids[factor][frame[factor]] for factor in FIELDS}
    raw = {
        "arm": "JOINT", "seed": 7, "row_id": authored["id"], "split": "train",
        "support_group": authored["metadata"]["support_class"],
        "target_frame": frame, "factor_argmax": argmax,
        "factor_probability_vectors": {
            factor: [float(index == argmax[factor]) for index in range(len(vocabulary.values[factor]))]
            for factor in FIELDS
        },
        "predicted_frame": frame, "available": True,
    }
    _validate_saved_row(raw, authored, arm="JOINT", seed=7, split="train", vocabulary=vocabulary)
    changed = deepcopy(raw)
    changed["target_frame"] = {**frame, "participant": "invalid"}
    with pytest.raises(ValueError, match="metadata"):
        _validate_saved_row(changed, authored, arm="JOINT", seed=7, split="train", vocabulary=vocabulary)
    changed = deepcopy(raw)
    changed["factor_probability_vectors"]["participant"] = [0.5] * 6
    with pytest.raises(ValueError, match="probabilities"):
        _validate_saved_row(changed, authored, arm="JOINT", seed=7, split="train", vocabulary=vocabulary)
    changed = deepcopy(raw)
    changed["factor_argmax"]["participant"] = (argmax["participant"] + 1) % 6
    with pytest.raises(ValueError, match="argmax"):
        _validate_saved_row(changed, authored, arm="JOINT", seed=7, split="train", vocabulary=vocabulary)
