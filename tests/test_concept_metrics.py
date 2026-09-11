import pytest
torch = pytest.importorskip("torch")

from norishio_lm.concept_metrics import concept_metrics, majority_predictions, permutation_diagnostics
from norishio_lm.concept_model import CONCEPT_FIELDS, ConceptVocabulary


def _target(event="A", operators=("WANT", "NOT"), **extra):
    concept = {f: "x" for f in CONCEPT_FIELDS}
    concept.update(event=event, operators=list(operators), repeat_marked=False)
    concept.update(extra)
    return {"concept": concept}


def _id(vocab, field, value):
    return vocab.fields[field][tuple(value) if field == "operators" else value]


def test_missing_unseen_and_ordered_operator_classes_are_separate():
    train = [_target(event="A", operators=("WANT", "NOT")), _target(event="B", operators=("NOT", "WANT"))]
    vocab = ConceptVocabulary.fit(train)
    targets = [train[0], {"concept": {**train[0]["concept"], "event": "UNSEEN"}}, {"concept": None}]
    preds = {f: torch.tensor([_id(vocab, f, _target()["concept"][f]), 0, 0]) for f in CONCEPT_FIELDS}
    got = concept_metrics(preds, targets, vocab)
    assert got["fields"]["event"]["missing_count"] == 1
    assert got["fields"]["event"]["unseen_count"] == 1
    assert got["missing_gold_mask"]["event"] == [False, False, True]
    assert got["unknown_gold_mask"]["event"] == [False, True, False]
    classes = {tuple(item["value"]): item for item in got["fields"]["operators"]["classes"]}
    assert classes[("WANT", "NOT")]["count"] == 2
    assert classes[("NOT", "WANT")]["count"] == 0
    assert got["fields"]["event"]["correct"] == 1


def test_majority_uses_smallest_id_on_tie_and_zero_when_empty():
    train = [_target(event="B"), _target(event="A")]
    vocab = ConceptVocabulary.fit(train)
    out = majority_predictions(train, vocab, 3)
    assert out["event"].tolist() == [vocab.fields["event"]["A"]] * 3
    empty = majority_predictions([], vocab, 2)
    assert all(value.tolist() == [0, 0] for value in empty.values())


def test_balanced_accuracy_differs_from_micro_and_full_frame_excludes_incomplete():
    train = [_target(event="A"), _target(event="B")]
    vocab = ConceptVocabulary.fit(train)
    targets = [_target(event="A"), _target(event="A"), _target(event="B"), {"concept": None}]
    preds = {f: torch.tensor([_id(vocab, f, _target()["concept"][f])] * 4) for f in CONCEPT_FIELDS}
    preds["event"] = torch.tensor([_id(vocab, "event", "A"), _id(vocab, "event", "A"), _id(vocab, "event", "A"), 0])
    got = concept_metrics(preds, targets, vocab)
    assert got["fields"]["event"]["accuracy"] == pytest.approx(2 / 3)
    assert got["fields"]["event"]["balanced_accuracy"] == pytest.approx(0.5)
    assert got["global_micro"]["count"] == 21
    assert got["full_frame_exact_match"]["count"] == 3


def test_permutation_diagnostics_validates_and_counts_differences_by_field():
    train = [_target(event="A"), _target(event="B"), _target(event="A")]
    vocab = ConceptVocabulary.fit(train)
    got = permutation_diagnostics(train, torch.tensor([1, 0, 2]), vocab)
    assert got["moved_count"] == 2
    assert got["self_count"] == 1
    assert got["complete_comparable_pairs"] == 3
    assert got["semantically_different_known_complete_frames"] == 2
    assert got["differing_byfield"]["event"] == 2
    with pytest.raises(ValueError):
        permutation_diagnostics(train, [0, 0, 2], vocab)


def test_prediction_ids_outside_reserved_or_vocabulary_range_are_rejected():
    train = [_target()]
    vocab = ConceptVocabulary.fit(train)
    valid = {f: torch.tensor([0]) for f in CONCEPT_FIELDS}
    concept_metrics(valid, train, vocab)
    invalid = dict(valid)
    invalid["event"] = torch.tensor([-1])
    with pytest.raises(ValueError):
        concept_metrics(invalid, train, vocab)
    invalid["event"] = torch.tensor([len(vocab.fields["event"]) + 1])
    with pytest.raises(ValueError):
        concept_metrics(invalid, train, vocab)


def test_metrics_are_json_serializable_and_keep_ordered_operator_values():
    import json
    train = [_target(operators=("NOT", "WANT"))]
    vocab = ConceptVocabulary.fit(train)
    preds = {f: torch.tensor([_id(vocab, f, _target(operators=("NOT", "WANT"))["concept"][f])]) for f in CONCEPT_FIELDS}
    result = concept_metrics(preds, train, vocab)
    json.dumps(result)
    assert result["fields"]["operators"]["classes"][0]["value"] == ["NOT", "WANT"]
    assert result["unknown_gold_mask"]["event"] == [False]
