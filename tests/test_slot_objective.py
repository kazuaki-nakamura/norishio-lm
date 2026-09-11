from __future__ import annotations

import copy

import pytest
torch = pytest.importorskip("torch")

from norishio_lm.concept_model import ConceptVocabulary
from norishio_lm.slot_objective import slot_ce, train_slot_control
from norishio_lm.tensorizer import SemanticTensorizer
from norishio_lm.toy_adapter import source_record
from norishio_lm.toy_controls import train_control
from norishio_lm.toy_experiment import ToyModel, fixture_module


def test_slot_ce_averages_union_and_excludes_eos_padding():
    logits = torch.zeros(1, 6, 260)
    labels = torch.tensor([[4, 5, 2, -100, -100, -100]])
    spans = [{"participant": {"start": 0, "end": 1}, "time": {"start": 1, "end": 2}}]
    expected = torch.nn.functional.cross_entropy(logits[0, :2], labels[0, :2])
    assert torch.allclose(slot_ce(logits, labels, spans), expected)


def test_slot_ce_rejects_overlap_bad_labels_and_invalid_logits():
    labels = torch.full((1, 4), 4, dtype=torch.long)
    overlap = [{"participant": {"start": 0, "end": 2}, "time": {"start": 1, "end": 3}}]
    with pytest.raises(ValueError):
        slot_ce(torch.zeros(1, 4, 260), labels, overlap)
    with pytest.raises(ValueError):
        slot_ce(torch.zeros(1, 4, 260), torch.tensor([[4, 2, 4, 4]]),
                [{"participant": {"start": 0, "end": 2}, "time": {"start": 2, "end": 3}}])
    bad = torch.zeros(1, 4, 260)
    bad[0, 0, 0] = float("nan")
    with pytest.raises(ValueError):
        slot_ce(bad, labels, [{"participant": {"start": 0, "end": 1}, "time": {"start": 1, "end": 2}}])


def test_slot_ce_gradient_reaches_lm_head_and_gru():
    logits = torch.randn(1, 5, 260, requires_grad=True)
    labels = torch.tensor([[4, 5, 6, 2, -100]])
    spans = [{"participant": {"start": 0, "end": 1}, "time": {"start": 1, "end": 3}}]
    slot_ce(logits, labels, spans).backward()
    assert logits.grad is not None and torch.count_nonzero(logits.grad).item() > 0
    corpus, strict = fixture_module("toy_corpus"), fixture_module("strict_decoder")
    bundle = corpus.build()
    rows = bundle["train"][:2]
    tensorizer = SemanticTensorizer.fit([source_record(corpus.model_inputs(r)) for r in bundle["train"]])
    vocab = ConceptVocabulary.fit([r["targets"] for r in bundle["train"]])
    model = ToyModel("C", tensorizer, vocab)
    ids, target_labels, semantic = __import__("norishio_lm.toy_experiment", fromlist=["batch"]).batch(
        rows, "C", tensorizer, corpus, strict)
    output, _ = model(ids, semantic, labels=target_labels)
    spans = [
        {"participant": {"start": 0, "end": 1}, "time": {"start": 1, "end": 2}}
        for _ in rows
    ]
    model.zero_grad(set_to_none=True)
    # Synthetic positions are valid byte labels in this gradient-path check.
    labels_for_check = target_labels.clone()
    labels_for_check[:, :2] = 4
    slot_ce(output["logits"], labels_for_check, spans).backward()
    assert model.decoder.lm_head.weight.grad is not None
    assert model.decoder.gru.weight_ih_l0.grad is not None


@pytest.mark.parametrize("mode", ["initial_only", "per_step_additive"])
def test_weight_zero_reproduces_train_control_state(mode):
    corpus, strict = fixture_module("toy_corpus"), fixture_module("strict_decoder")
    bundle = corpus.build()
    rows = bundle["train"][:4]
    tensorizer = SemanticTensorizer.fit([source_record(corpus.model_inputs(r)) for r in bundle["train"]])
    vocab = ConceptVocabulary.fit([r["targets"] for r in bundle["train"]])
    torch.manual_seed(19)
    initial = ToyModel("C", tensorizer, vocab, conditioning_mode=mode)
    fixed = torch.full((1, sum(len(v) + 1 for v in vocab.fields.values())), 0.1)
    schedule = [[0, 1], [2, 3]]
    expected, _ = train_control(initial, "predicted", schedule, rows, tensorizer, vocab, corpus, strict, fixed)
    actual, report = train_slot_control(initial, schedule, rows, tensorizer, vocab, corpus, strict, weight=0)
    assert all(torch.equal(a, b) for a, b in zip(expected.parameters(), actual.parameters()))
    assert report["trace"][0]["slot"] == 0.0


def test_invalid_weight_rejected():
    corpus, strict = fixture_module("toy_corpus"), fixture_module("strict_decoder")
    bundle = corpus.build()
    rows = bundle["train"][:1]
    tensorizer = SemanticTensorizer.fit([source_record(corpus.model_inputs(r)) for r in bundle["train"]])
    vocab = ConceptVocabulary.fit([r["targets"] for r in bundle["train"]])
    model = ToyModel("C", tensorizer, vocab)
    for weight in (-1, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            train_slot_control(model, [[0]], rows, tensorizer, vocab, corpus, strict, weight=weight)
