from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from norishio_lm.concept_model import CONCEPT_FIELDS, ConceptVocabulary
from norishio_lm.toy_adapter import source_record
from norishio_lm.toy_experiment import ToyModel, fixture_module, batch
from norishio_lm.tensorizer import SemanticTensorizer
from norishio_lm.toy_controls import conditional_loss, majority_baseline, predict_concepts, state_digest, train_control


def _setup():
    corpus = fixture_module("toy_corpus")
    strict = fixture_module("strict_decoder")
    bundle = corpus.build()
    rows = bundle["train"][:4]
    tensorizer = SemanticTensorizer.fit([source_record(corpus.model_inputs(r)) for r in bundle["train"]])
    vocab = ConceptVocabulary.fit([r["targets"] for r in bundle["train"]])
    torch.manual_seed(17)
    return corpus, strict, bundle, rows, tensorizer, vocab


def test_majority_is_train_only_and_ties_use_smallest_id():
    train = [{"concept": {"event": "b"}}, {"concept": {"event": "a"}}]
    validation = [{"concept": {"event": "a"}}, {"concept": {"event": "b"}}]
    vocab = ConceptVocabulary.fit(train)
    result = majority_baseline(train, validation, vocab)
    field = result["fields"]["event"]
    assert field["id"] == min(vocab.fields["event"].values())
    assert field["count"] == 2 and field["correct"] == 1
    altered = majority_baseline(train, validation + [{"concept": {"event": "z"}}], vocab)
    assert altered["fields"]["event"]["id"] == field["id"]


def test_initial_mean_predictions_are_detached_and_normalized_per_field():
    corpus, _, bundle, _, tensorizer, vocab = _setup()
    initial = ToyModel("C", tensorizer, vocab)
    sources = [corpus.model_inputs(r) for r in bundle["train"][:3]]
    probabilities = predict_concepts(initial, sources, tensorizer, batch_size=2)
    assert not probabilities.requires_grad
    offset = 0
    for field in CONCEPT_FIELDS:
        width = len(vocab.fields[field]) + 1
        assert torch.allclose(probabilities[:, offset:offset + width].sum(1), torch.ones(3))
        offset += width
    assert not probabilities.mean(0, keepdim=True).requires_grad


def test_lm_only_from_fixed_intervention_has_no_encoder_gradient():
    corpus, strict, _, rows, tensorizer, vocab = _setup()
    model = ToyModel("C", tensorizer, vocab)
    ids, labels, semantic = batch(rows[:2], "C", tensorizer, corpus, strict)
    fixed = predict_concepts(model, [corpus.model_inputs(r) for r in rows[:2]], tensorizer).mean(0, keepdim=True)
    model.zero_grad(set_to_none=True)
    output, _ = model(ids, semantic, labels=labels, intervention=fixed.expand(2, -1))
    output["lm_loss"].backward()
    assert all(p.grad is None for p in model.encoder.parameters())


def test_train_control_clones_initial_state_and_reuses_schedule():
    corpus, strict, _, rows, tensorizer, vocab = _setup()
    initial = ToyModel("C", tensorizer, vocab)
    schedule = [[0, 1], [2, 3]]
    fixed = predict_concepts(initial, [corpus.model_inputs(r) for r in rows], tensorizer).mean(0, keepdim=True)
    original_digest = state_digest(initial)
    original_fixed = fixed.clone()
    _, predicted = train_control(initial, "predicted", schedule, rows, tensorizer, vocab, corpus, strict, fixed)
    _, constant = train_control(initial, "constant", schedule, rows, tensorizer, vocab, corpus, strict, fixed)
    assert predicted["initial_state_sha256"] == state_digest(initial)
    assert constant["initial_state_sha256"] == predicted["initial_state_sha256"]
    assert predicted["steps"] == constant["steps"] == len(schedule)
    assert state_digest(initial) == original_digest
    assert torch.equal(fixed, original_fixed)


def test_conditional_loss_is_invariant_to_batch_size_and_keeps_global_alignment():
    corpus, strict, bundle, _, tensorizer, vocab = _setup()
    rows = bundle["validation"][:5]
    model = ToyModel("C", tensorizer, vocab).eval()
    probs = predict_concepts(model, [corpus.model_inputs(r) for r in rows], tensorizer, batch_size=2)
    one = conditional_loss(model, rows, probs, tensorizer, corpus, strict, batch_size=1)
    all_rows = conditional_loss(model, rows, probs, tensorizer, corpus, strict, batch_size=len(rows))
    assert one["lm_tokens"] == all_rows["lm_tokens"]
    assert one["lm"] == pytest.approx(all_rows["lm"], abs=1e-7)
    reversed_rows = conditional_loss(model, rows, probs.flip(0), tensorizer, corpus, strict, batch_size=2)
    assert reversed_rows["lm"] != pytest.approx(one["lm"], abs=1e-8)


def test_majority_missing_labels_are_not_counted_as_negatives():
    train = [{"concept": {"event": "a"}}]
    vocab = ConceptVocabulary.fit(train)
    result = majority_baseline(train, [{"concept": None}], vocab)
    assert result["count"] == 0
    assert result["accuracy"] is None


def test_generation_exact_match_requires_raw_bytes_and_eos():
    from norishio_lm.toy_controls import generation_metrics

    rows = [
        {"token_ids": [101, 2], "ended_eos": True, "valid_utf8": True, "invalid_special_tokens": []},
        {"token_ids": [101], "ended_eos": False, "valid_utf8": True, "invalid_special_tokens": []},
        {"token_ids": [0, 101, 2], "ended_eos": True, "valid_utf8": True, "invalid_special_tokens": [0]},
    ]
    measured = generation_metrics(rows, ["a", "a", "a"])
    assert measured["exact_match"] == 1 / 3
    assert measured["eos_rate"] == 2 / 3
    assert measured["special_token_rows"] == 1
