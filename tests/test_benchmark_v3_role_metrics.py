"""Decision regression checks on synthetic saved-head rows, without a model."""
from norishio_lm.benchmark_v3_role_metrics import FACTORS, summarize_role_heads


def _rows(*, rescue_time: bool):
    rows = []
    for arm in ("ROLE_ALIASED", "ROLE_DISJOINT"):
        for seed in (7, 17, 29):
            for index in range(480):
                gold = {"participant": "P0", "time": "T0", "event": "E0", "operator": "O0"}
                prediction = dict(gold)
                if arm == "ROLE_ALIASED":
                    prediction["participant"] = "P1"
                    prediction["time"] = "T1"
                elif not rescue_time:
                    prediction["time"] = "T1"
                split = "train" if index < 384 else "confirmation"
                rows.append({"arm": arm, "seed": seed, "split": split,
                             "support_group": "train" if split == "train" else
                             ("unseen_pair" if index < 432 else "seen_pair/unseen_triple"),
                             "target_frame": gold, "predicted_frame": prediction,
                             "available": True})
    return rows


def test_rescue_requires_both_strong_train_and_positive_paired_direction():
    classes = {"participant": ["P0", "P1"], "time": ["T0", "T1"],
               "event": ["E0"], "operator": ["O0"]}
    ceilings = dict.fromkeys(FACTORS, 0.5)
    both = summarize_role_heads(_rows(rescue_time=True), classes, ceilings)["decision"]
    assert both["branch"] == "participant_time_role_readout_rescued"
    assert both["rescue_factors"] == ["participant", "time"]
    one = summarize_role_heads(_rows(rescue_time=False), classes, ceilings)["decision"]
    assert one["branch"] == "one_factor_rescued_narrow_follow_up"
    assert one["rescue_factors"] == ["participant"]


def test_missing_saved_row_rejects_positive_decision():
    classes = {"participant": ["P0", "P1"], "time": ["T0", "T1"],
               "event": ["E0"], "operator": ["O0"]}
    ceilings = dict.fromkeys(FACTORS, 0.5)
    rows = _rows(rescue_time=True)
    rows.pop()
    try:
        summarize_role_heads(rows, classes, ceilings)
    except ValueError:
        pass
    else:
        raise AssertionError("incomplete run unexpectedly received a decision")


def test_event_only_rescue_is_not_hidden_by_participant_time_branch():
    classes = {"participant": ["P0", "P1"], "time": ["T0", "T1"],
               "event": ["E0", "E1"], "operator": ["O0"]}
    rows = _rows(rescue_time=True)
    for row in rows:
        if row["arm"] == "ROLE_ALIASED":
            row["predicted_frame"]["participant"] = "P0"
            row["predicted_frame"]["time"] = "T0"
            row["predicted_frame"]["event"] = "E1"
    ceilings = {"participant": 1.0, "time": 1.0, "event": 0.5, "operator": 1.0}
    decision = summarize_role_heads(rows, classes, ceilings)["decision"]
    assert decision["branch"] == "one_factor_rescued_narrow_follow_up"
    assert decision["rescue_factors"] == ["event"]
