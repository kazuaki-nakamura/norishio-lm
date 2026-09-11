import pytest

pytest.importorskip("torch")

from norishio_lm.concept_model import ConceptVocabulary
from norishio_lm.pair_metrics import group_pair_scores, pair_support


def target(person, time, text="ok"):
    return {"text": text, "concept": {"participant": person, "time": time}}


def vocab():
    people = ["p1", "p2", "p3", "p4", "p5"]
    times = ["t1", "t2", "t3", "t4", "t5"]
    return ConceptVocabulary.fit([target(p, t) for p in people for t in times])


def test_pair_support_counts_asymmetric_tables_and_seen_rows():
    v = vocab()
    train = [target("p1", "t1")] * 2 + [target("p2", "t3")] * 3
    validation = [target("p1", "t1"), target("p2", "t2"), target("p5", "t5")]
    got = pair_support(train, validation, v)
    assert got["train_table5x5"][0][0] == 2
    assert got["train_table5x5"][1][2] == 3
    assert sum(map(sum, got["validation_table5x5"])) == 3
    assert got["validation_seen"] == [True, False, False]
    assert got["seen_rows"] == 1 and got["unseen_rows"] == 2
    assert got["class_values"]["participant"] == ["p1", "p2", "p3", "p4", "p5"]


def test_pair_support_rejects_unknown_class():
    v = vocab()
    with pytest.raises(ValueError):
        pair_support([target("unknown", "t1")], [], v)


def test_both_slots_requires_both_slots_and_parse_failures_stay_denominator():
    seed = {"people": ["p1", "p2"], "times": ["t1"], "conditions": [{
        "event": "MEET", "operators": ["WANT"], "agent": "SELF", "repeat": False,
        "target": "A{time}{person}Z",
    }]}
    good = {"text": "At1p1Z", "token_ids": [], "ended_eos": True, "valid_utf8": True, "invalid_special_tokens": []}
    one = {**good, "text": "At1p1Z"}
    mismatch = {**good, "text": "At1p1Z"}
    bad = {**good, "text": "not-a-template"}
    targets = [target("p1", "t1", "At1p1Z"), target("p2", "t1", "At1p1Z"), target("p1", "t1", "At1p1Z")]
    got = group_pair_scores([one, mismatch, bad], targets, seed, [True, True, False])
    assert got["all"]["both_slots"] == {"correct": 1, "count": 3, "accuracy": 1 / 3}
    assert got["unseen_pair"]["generation"]["rows"] == 1
    assert got["seen_pair"]["both_slots"]["accuracy"] == 0.5


def test_empty_subgroup_metrics_are_null_where_appropriate():
    seed = {"people": [], "times": [], "conditions": []}
    got = group_pair_scores([], [], seed, [])
    assert got["all"]["generation"] is None
    assert got["all"]["slots"]["parse_coverage"] is None
    assert got["all"]["both_slots"] == {"correct": 0, "count": 0, "accuracy": None}
