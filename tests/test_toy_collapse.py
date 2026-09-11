from copy import deepcopy

import pytest

torch = pytest.importorskip("torch")

from norishio_lm.concept_model import CONCEPT_FIELDS, ConceptVocabulary
from norishio_lm.tensorizer import SemanticTensorizer
from norishio_lm.toy_adapter import source_record
from norishio_lm.toy_collapse import oracle_probabilities, source_conditions, source_encodings
from norishio_lm.toy_evaluation import concept_ids, score_condition
from norishio_lm.toy_experiment import ToyModel, fixture_module


def setup():
    corpus, strict = fixture_module("toy_corpus"), fixture_module("strict_decoder")
    rows = corpus.build()["train"][:4]
    sources = [corpus.model_inputs(r) for r in rows]
    tensorizer = SemanticTensorizer.fit([source_record(s) for s in sources])
    vocab = ConceptVocabulary.fit([r["targets"] for r in rows])
    return corpus, strict, rows, sources, tensorizer, vocab, ToyModel("C", tensorizer, vocab).eval()


def test_sources_reject_gold_and_are_detached_without_gradients():
    _, _, _, sources, tensorizer, _, model = setup()
    latent, probabilities = source_encodings(model, sources, tensorizer)
    assert not latent.requires_grad and not probabilities.requires_grad
    assert all(p.grad is None for p in model.parameters())
    with pytest.raises(ValueError, match="source"):
        source_encodings(model, [{**sources[0], "targets": {}}], tensorizer)


def test_oracle_requires_complete_known_concepts_and_keeps_operator_order():
    first = {f: "x" for f in CONCEPT_FIELDS}
    first.update(operators=["NOT", "WANT"], repeat_marked=False)
    second = {**first, "operators": ["WANT", "NOT"]}
    vocab = ConceptVocabulary.fit([{"concept": first}, {"concept": second}])
    result = oracle_probabilities([first, second], vocab)
    ids = concept_ids(result, vocab)
    assert ids["operators"][0] != ids["operators"][1]
    with pytest.raises(ValueError):
        oracle_probabilities([{**first, "event": None}], vocab)
    with pytest.raises(ValueError):
        oracle_probabilities([{**first, "event": "unseen"}], vocab)
    with pytest.raises(ValueError):
        oracle_probabilities([{**first, "text": "gold reference"}], vocab)


def test_conditions_train_mean_and_permutation_are_global():
    prediction = torch.tensor([[.1, .9], [.2, .8], [.3, .7]])
    train = torch.tensor([[1., 0.], [.5, .5]])
    result = source_conditions(prediction, train, [2, 0, 1])
    assert torch.equal(result["train_mean"], torch.tensor([[.75, .25]] * 3))
    assert torch.equal(result["permuted"], prediction[[2, 0, 1]])
    assert set(result) == {"predicted", "train_mean", "permuted"}
    with pytest.raises(ValueError):
        source_conditions(prediction, train, [0, 0, 1])


def test_changed_gold_reference_cannot_change_normal_generation():
    corpus, strict, rows, sources, tensorizer, vocab, model = setup()
    rows = rows[:2]
    _, prediction = source_encodings(model, sources[:2], tensorizer)
    original = score_condition(model, rows, prediction, tensorizer, vocab, corpus, strict)
    changed = deepcopy(rows)
    for row in changed:
        row["targets"]["text"] = "different reference"
        row["targets"]["concept"] = {f: None for f in CONCEPT_FIELDS}
    altered = score_condition(model, changed, prediction, tensorizer, vocab, corpus, strict)
    assert [x["generation"] for x in original["examples"]] == [x["generation"] for x in altered["examples"]]
    assert original["lm"] != altered["lm"]
