import json

import pytest
torch = pytest.importorskip("torch")

from norishio_lm.collapse_metrics import concept_head_stats, representation_stats
from norishio_lm.concept_model import CONCEPT_FIELDS, ConceptVocabulary


def _vocab():
    rows = [{"concept": {field: (("left", "right") if field == "operators" else f"{field}1")
                          for field in CONCEPT_FIELDS}},
            {"concept": {field: (("right", "left") if field == "operators" else f"{field}2")
                          for field in CONCEPT_FIELDS}}]
    return ConceptVocabulary.fit(rows), rows


def test_representation_stats_pair_distances_and_population_variance():
    got = representation_stats(torch.tensor([[0.0, 0.0], [3.0, 4.0], [3.0, 4.0]]))
    assert got["rows"] == 3 and got["dim"] == 2 and got["exact_unique_rows"] == 2
    assert got["pairwise_l2"]["min"] == pytest.approx(0)
    assert got["pairwise_l2"]["max"] == pytest.approx(5)
    assert got["pairwise_l2"]["count_le_tolerance"] == 1
    assert got["per_dim_population_variance"] == pytest.approx([2.0, 32 / 9])
    json.dumps(got)


def test_representation_stats_singleton_has_null_distances():
    got = representation_stats(torch.empty((0, 3)))
    assert got["pairwise_l2"]["min"] is None
    assert got["pairwise_l2"]["count_le_tolerance"] == 0
    assert got["per_dim_mean"] == [None] * 3
    assert got["per_dim_population_variance"] == [None] * 3


def test_concept_head_confusion_entropy_permutation_and_operator_order():
    vocab, targets = _vocab()
    widths = [len(vocab.fields[field]) + 1 for field in CONCEPT_FIELDS]
    probs = torch.cat([torch.tensor([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])[:, :width]
                       if width == 3 else torch.full((2, width), 1 / width)
                       for width in widths], dim=1)
    got = concept_head_stats(probs, targets, vocab, [1, 0])
    op = got["fields"]["operators"]
    assert op["gold_labels"] == [["left", "right"], ["right", "left"]]
    assert op["confusion_matrix"] == [[0, 1, 0], [0, 0, 1]]
    assert op["support"] == 2
    assert op["permutation_mean_l1"] == pytest.approx(2.0)
    assert op["class_support"] == [1, 1]
    assert op["prediction_column_ids"] == [0, 1, 2]
    assert "concept_metrics" in got and "unknown_gold_mask" in got["concept_metrics"]
    assert got["argmax_joint_pattern_count"] == 2
    assert len(got["argmax_joint_patterns"]) == 2
    json.dumps(got)


def test_unknown_and_missing_gold_are_excluded_from_confusion_but_retained_in_metrics():
    vocab, targets = _vocab()
    targets = list(targets) + [{"concept": {"event": "not-seen"}}, {}]
    width = sum(len(vocab.fields[field]) + 1 for field in CONCEPT_FIELDS)
    probs = torch.zeros((4, width))
    offset = 0
    for field in CONCEPT_FIELDS:
        probs[:, offset + 1] = 1
        offset += len(vocab.fields[field]) + 1
    got = concept_head_stats(probs, targets, vocab, [0, 1, 2, 3])
    event = got["fields"]["event"]
    assert event["support"] == 2
    assert sum(map(sum, event["confusion_matrix"])) == 2
    assert got["concept_metrics"]["fields"]["event"]["unseen_count"] == 1
    assert got["concept_metrics"]["fields"]["event"]["missing_count"] == 1


def test_probability_validation():
    vocab, targets = _vocab()
    width = sum(len(vocab.fields[field]) + 1 for field in CONCEPT_FIELDS)
    with pytest.raises(ValueError):
        concept_head_stats(torch.ones((2, width)), targets, vocab, [0, 1])
    with pytest.raises(ValueError):
        concept_head_stats(torch.zeros((2, width)), targets, vocab, [0, 0])


def test_empty_head_reports_null_permutation_distance():
    vocab, _ = _vocab()
    width = sum(len(vocab.fields[field]) + 1 for field in CONCEPT_FIELDS)
    got = concept_head_stats(torch.empty((0, width)), [], vocab, [])
    assert all(stats["permutation_mean_l1"] is None for stats in got["fields"].values())
