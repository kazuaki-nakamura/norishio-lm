from __future__ import annotations

import pytest

from norishio_lm.benchmark_v3_contract import INTERVENTION_RULE
from norishio_lm.benchmark_v3_metrics import (
    score_benchmark_v3,
    score_intermediate_probability_interventions,
    score_source_side_swaps,
)


def frame(participant="p1", time="t1", event="meet", operator="want"):
    return {"participant": participant, "time": time, "event": event, "operator": operator}


def output(text, *, eos=True, utf8=True, unique=True):
    return {"text": text, "ended_eos": eos, "valid_utf8": utf8, "unique_output": unique}


def parse(text):
    values = {
        "ok": frame(),
        "pair-known": frame(event="visit"),
        "changed": frame(participant="p2"),
    }
    if text == "bad":
        raise ValueError("not a frame")
    return values[text]


def test_v3_prioritises_free_generation_and_keeps_required_pair_groups():
    rows = [
        {"target_frame": frame(), "target_text": "ok", "generated": output("ok"),
         "intermediate_frame": {"canonical_frame": frame()}},
        {"target_frame": frame(event="visit"), "target_text": "pair-known",
         "generated": output("pair-known"), "intermediate_frame": frame(event="meet")},
        {"target_frame": frame(participant="p2"), "target_text": "changed",
         "generated": output("bad"), "intermediate_frame": None},
        {"target_frame": frame(participant="p3"), "target_text": "missing-eos",
         "generated": output("ok", eos=False)},
    ]
    report = score_benchmark_v3(
        rows, parse,
        train_support={"pairs": [["p1", "t1"]], "triples": [["p1", "t1", "meet"]]},
    )

    assert report["schema"] == "norishio.benchmark-v3.metrics.v1"
    assert report["required_groups"] == ["unseen_pair", "seen_pair"]
    assert report["group_counts"] == {
        "all": 4, "unseen_pair": 2, "seen_pair": 1, "excluded_seen_triple": 1,
    }
    assert report["all"]["free_generation_exact"] == {
        "correct": 2, "count": 4, "denominator": 4, "accuracy": 0.5,
    }
    assert report["all"]["triple_exact"]["correct"] == 2
    assert report["all"]["pair_exact"]["correct"] == 2
    assert report["all"]["atomic_balanced_accuracy"]["participant"] == pytest.approx(1 / 3)
    assert report["groups"]["seen_pair"]["rows"] == 1
    assert report["groups"]["unseen_pair"]["rows"] == 2
    assert report["all"]["generation_validity"]["eos"] == {
        "correct": 3, "count": 4, "denominator": 4, "accuracy": 0.75,
    }
    assert report["all"]["intermediate"]["frame_exact"] == {
        "correct": 1, "count": 2, "denominator": 2, "accuracy": 0.5,
    }
    assert report["space_metadata"]["intermediate_frame_space"] == "canonical_semantic_space"


def test_v3_auxiliary_teacher_forcing_and_empty_intermediate_are_null():
    row = {"target_frame": frame(), "target_text": "ok", "generated": output("ok"),
           "teacher_forced_bytes": {"exact": True, "matched_bytes": 2, "expected_bytes": 2}}
    report = score_benchmark_v3([row], parse, train_support=[])

    assert report["teacher_forced"]["diagnostic_only"] is True
    assert report["teacher_forced"]["byte_match_rate"] == 1
    assert report["all"]["intermediate"]["frame_exact"]["accuracy"] is None
    assert report["groups"]["seen_pair"]["rows"] == 0
    assert report["groups"]["seen_pair"]["free_generation_exact"]["accuracy"] is None


def test_source_side_swap_reports_target_change_preservation_and_parse_failures():
    report = score_source_side_swaps([
        {"factor": "participant", "baseline": output("ok"),
         "changed": output("changed"), "baseline_target_frame": frame(),
         "changed_target_frame": frame(participant="p2")},
        {"factor": "time", "baseline": output("ok"), "changed": output("bad")},
    ], parser=parse)

    participant = report["by_factor"]["participant"]
    assert participant["target_changed"] == {"correct": 1, "count": 1, "denominator": 1, "accuracy": 1}
    assert participant["non_target_preserved"] == {"correct": 1, "count": 1, "denominator": 1, "accuracy": 1}
    time = report["by_factor"]["time"]
    assert time["parse_failures"] == 1
    assert time["target_changed"]["accuracy"] == 0
    assert report["oracle"] is False


def test_true_intermediate_intervention_requires_fixed_one_hot_and_is_separate_from_swaps():
    baseline_probabilities = {
        "participant": [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "time": [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "event": [1.0, 0.0, 0.0, 0.0],
        "operator": [1.0, 0.0, 0.0, 0.0],
    }
    intervened_probabilities = {**baseline_probabilities,
                                "participant": [0.0, 1.0, 0.0, 0.0, 0.0, 0.0]}
    evidence = {"baseline_source_identity": "source-1", "intervened_source_identity": "source-1",
                "baseline_decoder_prefix": [1, 2], "intervened_decoder_prefix": [1, 2]}
    report = score_intermediate_probability_interventions([
        {"factor": "participant", "intervened_one_hot": [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
         "baseline_probabilities": baseline_probabilities,
         "intervened_probabilities": intervened_probabilities,
         **evidence, "baseline": output("ok"), "intervened": output("changed")},
    ], parser=parse)

    assert report["kind"] == "true_intermediate_probability_intervention"
    assert report["intervention"] == "fixed_artificial_one_hot"
    assert report["by_factor"]["participant"]["target_changed"]["correct"] == 1
    assert report["by_factor"]["participant"]["non_target_preserved"]["correct"] == 1
    assert report["oracle"] is False
    with pytest.raises(ValueError, match="fixed artificial one-hot"):
        score_intermediate_probability_interventions([
            {"factor": "participant", "intervened_one_hot": [0.4, 0.6, 0.0, 0.0, 0.0, 0.0],
             "baseline_probabilities": baseline_probabilities,
             "intervened_probabilities": intervened_probabilities,
             **evidence, "baseline": output("ok"), "intervened": output("changed")},
        ], parser=parse)


def test_future_intervention_rule_is_validated_and_retained():
    baseline_probabilities = {
        "participant": [0.1, 0.7, 0.2, 0.0, 0.0, 0.0],
        "time": [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "event": [1.0, 0.0, 0.0, 0.0],
        "operator": [1.0, 0.0, 0.0, 0.0],
    }
    intervened_probabilities = {
        **baseline_probabilities,
        "participant": [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
    }
    item = {
        "factor": "participant",
        "intervention_rule": INTERVENTION_RULE,
        "baseline_argmax_class": 1,
        "intervention_class": 2,
        "baseline_probabilities": baseline_probabilities,
        "intervened_probabilities": intervened_probabilities,
        "baseline_source_identity": "s",
        "intervened_source_identity": "s",
        "baseline_decoder_prefix": [3],
        "intervened_decoder_prefix": [3],
        "baseline": output("ok"),
        "intervened": output("changed"),
    }

    report = score_intermediate_probability_interventions([item], parser=parse)

    assert report["intervention_rules"] == [INTERVENTION_RULE]
    assert report["examples"][0]["baseline_argmax_class"] == 1
    assert report["examples"][0]["intervention_class"] == 2
    with pytest.raises(ValueError, match="alternate intervention rule"):
        score_intermediate_probability_interventions([
            {**item, "intervention_class": 0}
        ], parser=parse)


def test_true_intermediate_intervention_rejects_wrong_width_and_run_mismatch():
    probabilities = {
        "participant": [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "time": [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "event": [1.0, 0.0, 0.0, 0.0],
        "operator": [1.0, 0.0, 0.0, 0.0],
    }
    base = {"factor": "event", "baseline_probabilities": probabilities,
            "intervened_probabilities": {**probabilities, "event": [0.0, 1.0, 0.0, 0.0]},
            "intervened_one_hot": [0.0, 1.0, 0.0, 0.0],
            "baseline_source_identity": "s", "intervened_source_identity": "s",
            "baseline_decoder_prefix": [3], "intervened_decoder_prefix": [3],
            "baseline": output("ok"), "intervened": output("changed")}
    wrong_width = {**base, "intervened_one_hot": [0.0, 1.0]}
    with pytest.raises(ValueError, match="width 4"):
        score_intermediate_probability_interventions([wrong_width], parser=parse)
    with pytest.raises(ValueError, match="source identity"):
        score_intermediate_probability_interventions(
            [{**base, "intervened_source_identity": "other"}], parser=parse)
    with pytest.raises(ValueError, match="decoder prefix"):
        score_intermediate_probability_interventions(
            [{**base, "intervened_decoder_prefix": [4]}], parser=parse)


def test_true_intermediate_intervention_rejects_collateral_probability_change():
    probabilities = {
        "participant": [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "time": [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "event": [1.0, 0.0, 0.0, 0.0],
        "operator": [1.0, 0.0, 0.0, 0.0],
    }
    intervened = {**probabilities, "event": [0.0, 1.0, 0.0, 0.0],
                  "operator": [0.0, 1.0, 0.0, 0.0]}
    with pytest.raises(ValueError, match="non-target probability vector"):
        score_intermediate_probability_interventions([{
            "factor": "event", "intervened_one_hot": [0.0, 1.0, 0.0, 0.0],
            "baseline_probabilities": probabilities, "intervened_probabilities": intervened,
            "baseline_source_identity": "s", "intervened_source_identity": "s",
            "baseline_decoder_prefix": [3], "intervened_decoder_prefix": [3],
            "baseline": output("ok"), "intervened": output("changed"),
        }], parser=parse)


def test_canonical_head_does_not_decode_raw_code_space():
    with pytest.raises(ValueError):
        score_benchmark_v3([
            {"target_frame": frame(), "target_text": "ok", "generated": output("ok"),
             "intermediate_frame": {"raw_code_indices": [0, 0, 0, 0]}},
        ], parse)
