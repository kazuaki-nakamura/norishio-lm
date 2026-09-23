import pytest

from norishio_lm.benchmark_v3_head_metrics import aggregate_head_metrics


FACTORS = {
    "participant": ("p0", "p1"),
    "time": ("t0", "t1"),
    "event": ("e0", "e1"),
    "operator": ("o0", "o1"),
}


def row(arm, seed, split, gold, prediction=None, *, stratum=None):
    item = {"arm": arm, "seed": seed, "split": split, "gold_factors": gold}
    if prediction is not None:
        item["predicted_factors"] = prediction
    if stratum is not None:
        item["stratum"] = stratum
    return item


G0 = {"participant": "p0", "time": "t0", "event": "e0", "operator": "o0"}
G1 = {"participant": "p1", "time": "t1", "event": "e1", "operator": "o1"}


def test_atomic_balanced_metrics_keep_missing_class_and_unavailable_rows_explicitly():
    records = [
        row("JOINT", 7, "train", G0, G0),
        row("JOINT", 7, "train", G1),  # entire head unavailable
        row("JOINT", 7, "confirmation", G0, {**G0, "time": "t1"}, stratum="unseen_pair"),
    ]
    report = aggregate_head_metrics(records, FACTORS)
    train = report["by_arm_seed"]["JOINT"]["7"]["train_resubstitution"]
    participant = train["fields"]["participant"]
    assert participant["correct"] == 1
    assert participant["count"] == 1
    assert participant["scheduled_count"] == 2
    assert participant["unavailable_count"] == 1
    assert participant["class_balanced_accuracy"] == pytest.approx(1.0)
    assert participant["per_class"]["p0"]["status"] == "available"
    assert participant["per_class"]["p1"]["status"] == "unavailable"
    assert participant["per_class"]["p1"]["support"] == 1
    assert participant["per_class"]["p1"]["rate"] is None
    # A class with gold support but no prediction is unavailable; zero-support
    # expected classes are exercised by the empty confirmation stratum below.
    event = train["fields"]["event"]["per_class"]
    assert event["e1"]["status"] == "unavailable"
    assert event["e1"]["count"] == 0
    assert event["e1"]["rate"] is None


def test_zero_count_confirmation_stratum_is_present_and_unavailable():
    report = aggregate_head_metrics([row("JOINT", 7, "train", G0, G0)], FACTORS)
    empty = report["arm_aggregate"]["JOINT"]["confirmation"]["seen_pair/unseen_higher_order"]
    assert empty["rows"] == 0
    assert empty["joint_exact"] == {
        "correct": 0, "count": 0, "denominator": 0, "rate": None,
        "scheduled_count": 0, "unavailable_count": 0,
    }
    assert empty["fields"]["operator"]["per_class"]["o0"]["status"] == "missing"


def test_train_resubstitution_and_confirmation_support_are_separate():
    records = [
        row("JOINT", 7, "train", G0, G0),
        row("JOINT", 7, "confirmation", G1, G0, stratum="seen_pair/unseen_higher_order"),
    ]
    report = aggregate_head_metrics(records, FACTORS, supports={
        "train_resubstitution": {factor: {value: 4 for value in classes} for factor, classes in FACTORS.items()},
        "confirmation": {factor: {value: 2 for value in classes} for factor, classes in FACTORS.items()},
    })
    arm = report["arm_aggregate"]["JOINT"]
    assert arm["train_resubstitution"]["joint_exact"]["correct"] == 1
    assert arm["confirmation"]["all"]["joint_exact"]["correct"] == 0
    assert arm["confirmation"]["seen_pair/unseen_higher_order"]["rows"] == 1
    assert arm["confirmation"]["unseen_pair"]["rows"] == 0
    assert arm["train_resubstitution"]["fields"]["participant"]["per_class"]["p0"]["supplied_support"] == 4


def test_malformed_raw_is_rejected_without_inference_or_coercion():
    with pytest.raises(ValueError, match="split"):
        aggregate_head_metrics([row("JOINT", 7, "not-a-split", G0, G0)], FACTORS)
    with pytest.raises(ValueError, match="outside expected classes"):
        aggregate_head_metrics([row("JOINT", 7, "train", G0, {**G0, "event": "unknown"})], FACTORS)
    with pytest.raises(ValueError, match="must be a mapping"):
        aggregate_head_metrics([row("JOINT", 7, "train", G0, "bad")], FACTORS)
    with pytest.raises(ValueError, match="available"):
        aggregate_head_metrics([{**row("JOINT", 7, "train", G0, G0), "available": "yes"}], FACTORS)


def test_explicit_availability_can_make_one_factor_unavailable_and_custom_strata_remain_separate():
    records = [{**row("JOINT", 7, "confirmation", G0, G0, stratum="custom"),
                "available": {"participant": True, "time": False, "event": True, "operator": True}}]
    report = aggregate_head_metrics(records, FACTORS)
    confirmation = report["arm_aggregate"]["JOINT"]["confirmation"]
    assert confirmation["custom"]["rows"] == 1
    assert confirmation["custom"]["fields"]["time"]["available_count"] == 0


def test_paired_seed_directions_and_arm_aggregate_use_exact_counts():
    records = [
        row("JOINT", 7, "confirmation", G0, G0, stratum="unseen_pair"),
        row("FACTOR_ONLY", 7, "confirmation", G0, G1, stratum="unseen_pair"),
        row("JOINT", 17, "confirmation", G0, G1, stratum="unseen_pair"),
        row("FACTOR_ONLY", 17, "confirmation", G0, G0, stratum="unseen_pair"),
        row("JOINT", 29, "confirmation", G0, G0, stratum="unseen_pair"),
        row("FACTOR_ONLY", 29, "confirmation", G0, G0, stratum="unseen_pair"),
    ]
    report = aggregate_head_metrics(records, FACTORS)
    paired = report["paired_seed_directions"]
    assert paired["7"]["confirmation"]["unseen_pair"]["direction"] == "joint_better"
    assert paired["7"]["confirmation"]["unseen_pair"]["delta"] == pytest.approx(-1.0)
    assert paired["17"]["confirmation"]["unseen_pair"]["direction"] == "factor_only_better"
    assert paired["17"]["confirmation"]["unseen_pair"]["delta"] == pytest.approx(1.0)
    assert paired["29"]["confirmation"]["unseen_pair"]["direction"] == "tie"
    assert paired["29"]["confirmation"]["unseen_pair"]["delta"] == pytest.approx(0.0)
    assert paired["7"]["confirmation"]["unseen_pair"]["delta_definition"] == "FACTOR_ONLY - JOINT"
    joint = report["arm_aggregate"]["JOINT"]["confirmation"]["all"]["joint_exact"]
    assert joint["correct"] == 2
    assert joint["count"] == 3
    assert joint["rate"] == pytest.approx(2 / 3)


def test_paired_seed_directions_include_atomic_factor_directions():
    joint_prediction = {"participant": "p0", "time": "t1", "event": "e0", "operator": "o0"}
    factor_only_prediction = {"participant": "p1", "time": "t0", "event": "e0", "operator": "o0"}
    report = aggregate_head_metrics([
        row("JOINT", 7, "confirmation", G0, joint_prediction, stratum="unseen_pair"),
        row("FACTOR_ONLY", 7, "confirmation", G0, factor_only_prediction, stratum="unseen_pair"),
    ], FACTORS)

    paired = report["paired_seed_directions"]["7"]["confirmation"]["unseen_pair"]
    # The four-head exact result is a tie, while the atomic participant/time
    # directions point in opposite directions and must remain visible.
    assert paired["direction"] == "tie"
    assert paired["fields"]["participant"] == {
        "joint_rate": 1.0, "factor_only_rate": 0.0, "delta": -1.0,
        "direction": "joint_better", "delta_definition": "FACTOR_ONLY - JOINT",
    }
    assert paired["fields"]["time"]["direction"] == "factor_only_better"
    assert paired["fields"]["time"]["delta"] == pytest.approx(1.0)
