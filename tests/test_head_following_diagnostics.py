import pytest
torch = pytest.importorskip("torch")

from norishio_lm.concept_model import CONCEPT_FIELDS, ConceptVocabulary
from norishio_lm.head_following_diagnostics import (
    base_ablation_cases, confident_train_subsets, evaluate_condition, pair_following,
)


def _target(participant="p0", time="t0"):
    return {"concept": {f: (("OP",) if f == "operators" else ("STAY" if f == "event" else "x"))
                         for f in CONCEPT_FIELDS} | {"participant": participant, "time": time},
            "text": f"at {participant} {time}"}


def _vocab():
    return ConceptVocabulary.fit([_target(f"p{i}", f"t{i}") for i in range(5)])


def test_base_ablation_zeroes_only_named_old_blocks_and_keeps_heads_without_mutation():
    vocab = _vocab()
    width = sum(len(vocab.fields[f]) + 1 for f in CONCEPT_FIELDS)
    old = torch.ones(2, width)
    heads = torch.full((2, 10), .2)
    gold = torch.zeros(2, 10)
    old_before, heads_before = old.clone(), heads.clone()
    cases = base_ablation_cases(old, heads, gold, vocab)
    assert len(cases) == 8
    assert torch.equal(cases["full_predicted"][:, -10:], heads)
    assert torch.equal(cases["base_zero_both_gold"][:, -10:], gold)
    for field, value in (("event", cases["event_zero_predicted"]),
                         ("operators", cases["operators_zero_predicted"])):
        start = sum(len(vocab.fields[f]) + 1 for f in CONCEPT_FIELDS[:CONCEPT_FIELDS.index(field)])
        width_field = len(vocab.fields[field]) + 1
        assert not value[:, start:start + width_field].any()
    assert torch.equal(old, old_before) and torch.equal(heads, heads_before)


def test_confident_subsets_require_both_heads_and_joint_label_correctness():
    vocab = _vocab()
    targets = [_target("p0", "t0"), _target("p1", "t1"), _target("p2", "t2"), _target("p3", "t3")]
    p = {v: i for v, i in vocab.fields["participant"].items()}
    t = {v: i for v, i in vocab.fields["time"].items()}
    def row(pi, ti, pmax=.8, tmax=.8):
        value = torch.full((10,), (1 - pmax) / 4)
        value[pi] = pmax
        value[5:] = (1 - tmax) / 4
        value[5 + ti] = tmax
        return value
    heads = torch.stack([row(p["p0"] - 1, t["t0"] - 1), row(p["p1"] - 1, t["t2"] - 1),
                         row(p["p2"] - 1, t["t2"] - 1, .79), row(p["p3"] - 1, t["t3"] - 1)])
    result = confident_train_subsets(heads, targets, vocab)
    assert result == {"confident_correct": [0, 3], "confident_wrong": [1]}
    assert confident_train_subsets(torch.empty(0, 10), [], vocab) == {"confident_correct": [], "confident_wrong": []}
    with pytest.raises(ValueError): confident_train_subsets(heads, targets, vocab, threshold=True)


def test_pair_following_requires_full_parse_for_both_slots_and_uses_empty_rates():
    vocab = _vocab()
    seed = {"people": [f"p{i}" for i in range(5)], "times": [f"t{i}" for i in range(5)],
            "conditions": [{"event": "STAY", "operators": ["OP"], "agent": "x", "repeat": False,
                            "target": "at {person} {time}"}]}
    targets = [_target("p0", "t0")] * 3
    valid = {"text": "at p0 t0", "valid_utf8": True, "ended_eos": True, "invalid_special_tokens": []}
    partial = {"text": "at p0 t1", "valid_utf8": True, "ended_eos": True, "invalid_special_tokens": []}
    invalid = {**valid, "text": "not in grammar"}
    cells = pair_following([valid, partial, invalid], targets, vocab, seed)
    assert cells["0,0"]["rows"] == 3 and cells["0,0"]["both_correct"] == 1
    assert cells["0,0"]["rates"]["both"] == pytest.approx(1 / 3)
    assert cells["0,0"]["participant_correct"] == 2 and cells["0,0"]["time_correct"] == 1
    assert cells["1,1"]["rows"] == 0 and cells["1,1"]["rates"] is None


def test_empty_condition_returns_null_metrics_and_no_generations():
    vocab = _vocab()
    class Corpus:
        @staticmethod
        def seed_data():
            return {"people": [], "times": [], "conditions": []}
    result = evaluate_condition(object(), [], torch.empty(0, 10), object(), vocab, Corpus(), object())
    assert result["rows"] == 0 and result["lm"] is None and result["generation"] is None
    assert result["generations"] == [] and result["both_slots"]["accuracy"] is None


def test_confidence_selection_rejects_nan_and_unknown_low_confidence_labels():
    heads = torch.full((1, 10), .2)
    with pytest.raises(ValueError):
        confident_train_subsets(heads, [_target("unknown", "t0")], _vocab())
    heads[0, 0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        confident_train_subsets(heads, [_target()], _vocab())
