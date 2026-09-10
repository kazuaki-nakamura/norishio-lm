from __future__ import annotations

from copy import deepcopy

import pytest

torch = pytest.importorskip("torch")

from norishio_lm.toy_adapter import source_record
from norishio_lm.toy_experiment import ToyModel, batch, fixture_module
from norishio_lm.tensorizer import SemanticTensorizer
from norishio_lm.concept_model import ConceptVocabulary


def _setup():
    corpus = fixture_module("toy_corpus")
    strict = fixture_module("strict_decoder")
    bundle = corpus.build()
    rows = bundle["train"][:4]
    tensorizer = SemanticTensorizer.fit(
        [source_record(corpus.model_inputs(row)) for row in bundle["train"]]
    )
    vocabulary = ConceptVocabulary.fit([row["targets"] for row in bundle["train"]])
    return corpus, strict, rows, tensorizer, vocabulary


def test_fixed_concept_intervention_is_invariant_to_source_swap() -> None:
    corpus, strict, rows, tensorizer, vocabulary = _setup()
    model = ToyModel("C", tensorizer, vocabulary).eval()
    ids, _, semantic = batch(rows, "C", tensorizer, corpus, strict)
    with torch.no_grad():
        _, auxiliary = model(ids, semantic)
        concepts = auxiliary.probabilities()
        first, _ = model(ids, semantic, intervention=concepts)
        changed = tensorizer.encode([source_record({"context": "changed", "text": "別のソース"})] * len(rows))
        second, _ = model(ids, changed, intervention=concepts)
    assert torch.equal(first["logits"], second["logits"])


def test_gold_metadata_changes_do_not_change_source_batch_or_forward() -> None:
    corpus, strict, rows, tensorizer, vocabulary = _setup()
    altered = deepcopy(rows)
    altered[0]["targets"]["sense"] = "changed_gold"
    altered[0]["targets"]["concept"] = None
    altered[0]["metadata"]["changed_gold"] = True
    altered[0]["id"] = "changed-id"
    original_batch = batch(rows, "B", tensorizer, corpus, strict)
    altered_batch = batch(altered, "B", tensorizer, corpus, strict)
    assert torch.equal(original_batch[0], altered_batch[0])
    assert torch.equal(original_batch[1], altered_batch[1])
    for name in original_batch[2].channels:
        assert torch.equal(original_batch[2].channels[name].ids, altered_batch[2].channels[name].ids)
        assert torch.equal(original_batch[2].channels[name].mask, altered_batch[2].channels[name].mask)
    model = ToyModel("B", tensorizer, vocabulary).eval()
    with torch.no_grad():
        first, _ = model(original_batch[0], original_batch[2])
        second, _ = model(altered_batch[0], altered_batch[2])
    assert torch.equal(first["logits"], second["logits"])


def test_heldout_source_encoding_cannot_grow_fitted_vocabularies() -> None:
    _, _, _, tensorizer, _ = _setup()
    before = tensorizer.vocabularies
    heldout = source_record({"context": "未見", "text": "未学習の入力"})
    tensorizer.encode([heldout])
    assert tensorizer.vocabularies == before


@pytest.mark.parametrize("pathway", ["A", "B", "C"])
def test_future_reference_cannot_change_earlier_logits(pathway: str) -> None:
    corpus, strict, rows, tensorizer, vocabulary = _setup()
    model = ToyModel(pathway, tensorizer, vocabulary).eval()
    ids, _, semantic = batch(rows[:1], pathway, tensorizer, corpus, strict)
    changed = ids.clone()
    changed[0, -1] = 4 if ids[0, -1] != 4 else 5
    with torch.no_grad():
        first, _ = model(ids, semantic)
        second, _ = model(changed, semantic)
    assert torch.equal(first["logits"][:, :-1], second["logits"][:, :-1])


def test_false_glyph_removal_is_exactly_invariant() -> None:
    from norishio_lm.toy_adapter import misleading_glyph_fixture_record

    corpus, strict, rows, tensorizer, vocabulary = _setup()
    model = ToyModel("C", tensorizer, vocabulary).eval()
    ids, _, plain = batch(rows[:1], "C", tensorizer, corpus, strict)
    perturbed = tensorizer.encode([misleading_glyph_fixture_record(corpus.model_inputs(rows[0]))])
    with torch.no_grad():
        first, _ = model(ids, plain, channels={"subcharacters": False})
        second, _ = model(ids, perturbed, channels={"subcharacters": False})
    assert torch.equal(first["logits"], second["logits"])
