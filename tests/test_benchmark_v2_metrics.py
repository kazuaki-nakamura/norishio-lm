import json

import pytest

from norishio_lm.benchmark_v2_metrics import score_benchmark_v2, score_intervention_locality


def frame(participant="p1", time="t1", event="meet", operator="want"):
    return {"participant": participant, "time": time, "event": event, "operator": operator}


def output(text, *, eos=True, utf8=True):
    return {"text": text, "ended_eos": eos, "valid_utf8": utf8, "invalid_special_tokens": []}


def parse(text):
    values = {"good": frame(), "other": frame("p2", "t2", "stay", "plan")}
    return values.get(text)


def test_common_metrics_and_train_only_seen_unseen_groups():
    rows = [
        {"target_frame": frame(), "target_text": "good", "generated": output("good")},
        {"target_frame": frame("p2", "t2", "stay", "plan"), "target_text": "other", "generated": output("other")},
        {"target_frame": frame("p2", "t1", "meet", "want"), "target_text": "missing", "generated": output("bad")},
    ]
    got = score_benchmark_v2(rows, parse, train_support=[{"target_frame": frame()}])
    assert got["all"]["fields"]["participant"]["accuracy"] == pytest.approx(2 / 3)
    assert got["all"]["fields"]["participant"]["balanced_accuracy"] == pytest.approx(3 / 4)
    assert got["all"]["pair_exact"] == {"correct": 2, "count": 3, "accuracy": pytest.approx(2 / 3)}
    assert got["groups"]["train_seen"]["rows"] == 1
    assert got["groups"]["train_unseen"]["rows"] == 2
    assert got["all"]["contingency_2x2"] == {
        "intermediate_correct": {"generation_correct": 0, "generation_incorrect": 0},
        "intermediate_incorrect": {"generation_correct": 0, "generation_incorrect": 0},
    }
    assert got["all"]["intermediate_unavailable_count"] == 3
    json.dumps(got, ensure_ascii=False)


def test_parse_failures_eos_and_duplicates_stay_in_denominators():
    rows = [
        {"target_frame": frame(), "target_text": "good", "generated": output("good")},
        {"target_frame": frame(), "target_text": "good", "generated": output("bad", eos=False)},
        {"target_frame": frame(), "target_text": "good", "generated": output("good")},
    ]
    got = score_benchmark_v2(rows, parse)
    assert got["all"]["rows"] == 3
    assert got["all"]["parseable_count"] == 2
    assert got["all"]["generation_frame_exact"] == {"correct": 2, "count": 3, "accuracy": pytest.approx(2 / 3)}
    assert got["duplicate_output_rows"] == [0, 2]
    assert got["all"]["generation_validity"]["unique_output"] == {"correct": 1, "count": 3, "accuracy": pytest.approx(1 / 3)}
    assert got["rows"][1]["rejection_reason"] == "nonEOS"


def test_declared_nonunique_output_is_still_scored_semantically():
    row = {"target_frame": frame(), "target_text": "good",
           "generated": {**output("good"), "unique_output": False}}
    got = score_benchmark_v2([row], parse)
    assert got["all"]["parseable_count"] == 1
    assert got["all"]["generation_frame_exact"]["correct"] == 1
    assert got["all"]["generation_validity"]["valid"]["correct"] == 1
    assert got["all"]["generation_validity"]["unique_output"]["correct"] == 0


def test_eos_and_utf8_diagnostics_are_independent_of_other_failures():
    rows = [
        {"target_frame": frame(), "target_text": "good",
         "generated": {"text": 3, "ended_eos": False, "valid_utf8": False}},
        {"target_frame": frame(), "target_text": "good",
         "generated": {"text": "good", "ended_eos": True, "valid_utf8": False}},
    ]
    got = score_benchmark_v2(rows, parse)["all"]["generation_validity"]
    assert got["eos"] == {"correct": 1, "count": 2, "accuracy": 0.5}
    assert got["utf8"] == {"correct": 0, "count": 2, "accuracy": 0.0}
    assert got["valid"] == {"correct": 0, "count": 2, "accuracy": 0.0}


def test_empty_subgroups_and_teacher_forced_diagnostic_are_separate():
    row = {"target_frame": frame(), "target_text": "good", "generated": output("good"),
           "teacher_forced_bytes": {"exact": True, "matched_bytes": 4, "expected_bytes": 4}}
    got = score_benchmark_v2([row], parse, train_support=[])
    assert got["groups"]["train_seen"]["fields"]["event"]["accuracy"] is None
    assert got["teacher_forced_bytes"]["diagnostic_only"] is True
    assert got["teacher_forced_bytes"]["byte_match_rate"] == 1
    assert "teacher_forced" not in got["all"]


def test_intervention_locality_uses_all_rows_as_denominator():
    interventions = [
        {"target_factor": "participant", "baseline_frame": frame(),
         "changed_frame": frame("p2")},
        {"target_factor": "time", "baseline": output("good"),
         "changed": output("bad"),},
    ]
    got = score_intervention_locality(interventions, parser=parse)
    assert got["by_factor"]["participant"]["preservation_rate"] == 1
    assert got["by_factor"]["time"]["preservation_rate"] == 0
    assert got["by_factor"]["time"]["rows"] == 1
    assert got["by_factor"]["time"]["parse_failures"] == 1


def test_generation_metrics_do_not_fall_back_to_correct_intermediate_frame():
    wrong = frame("p2", "t2", "stay", "plan")
    rows = [{"target_frame": frame(), "target_text": "wrong", "intermediate_frame": frame(),
             "generated": output("other")}]
    got = score_benchmark_v2(rows, lambda text: wrong)
    assert got["all"]["fields"]["participant"]["accuracy"] == 0
    assert got["all"]["pair_exact"]["correct"] == 0
    assert got["all"]["triple_exact"]["correct"] == 0
    assert got["all"]["intermediate"]["frame_exact"] == {"correct": 1, "count": 1, "accuracy": 1}
    assert got["all"]["contingency_2x2"]["intermediate_correct"] == {
        "generation_correct": 0, "generation_incorrect": 1}


def test_missing_intermediate_is_unavailable_instead_of_incorrect():
    rows = [{"target_frame": frame(), "target_text": "good", "generated": output("good")}]
    got = score_benchmark_v2(rows, parse)
    assert got["all"]["intermediate"]["available_count"] == 0
    assert got["all"]["intermediate_frame_exact"] == {"correct": 0, "count": 0, "accuracy": None}
    assert got["all"]["contingency_2x2"] == {
        "intermediate_correct": {"generation_correct": 0, "generation_incorrect": 0},
        "intermediate_incorrect": {"generation_correct": 0, "generation_incorrect": 0}}
    assert got["all"]["contingency_row_count"] == 0
    assert got["all"]["intermediate_unavailable_count"] == 1


def test_strong_validation_rejects_missing_fields_and_bad_metadata():
    with pytest.raises(ValueError):
        score_benchmark_v2([{"target_frame": {"participant": "p1"}, "target_text": "x",
                             "generated": output("x")}], parse)
    with pytest.raises(ValueError):
        score_benchmark_v2([{"target_frame": frame(), "target_text": "x",
                             "generated": {"text": "x", "ended_eos": "yes"}}], parse)
    bad_diagnostic = {"target_frame": frame(), "target_text": "good",
                      "generated": output("good"),
                      "teacher_forced_bytes": {"exact": "yes", "matched_bytes": -1,
                                               "expected_bytes": float("nan")}}
    with pytest.raises(ValueError):
        score_benchmark_v2([bad_diagnostic], parse)
    nonfinite = {"target_frame": frame(), "target_text": "good",
                 "generated": output("good"),
                 "teacher_forced_bytes": {"exact": True, "matched_bytes": 1,
                                          "expected_bytes": 1, "loss": float("nan")}}
    with pytest.raises(ValueError, match="non-finite"):
        score_benchmark_v2([nonfinite], parse)
