"""Boundary tests for the slot-only objective used by Issue 18.

These tests deliberately keep the objective small and inspectable: labels select
the authored participant/time byte spans, while the decoder remains responsible
for causal logits.  No training or benchmark claim is made here.
"""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")


def _fixture():
    from norishio_lm.tensorizer import SemanticTensorizer
    from norishio_lm.toy_adapter import source_record
    from norishio_lm.toy_experiment import ToyModel, fixture_module
    from norishio_lm.toy_spans import authored_slot_spans
    from norishio_lm.toy_collapse import source_encodings

    corpus, strict = fixture_module("toy_corpus"), fixture_module("strict_decoder")
    row = corpus.build()["train"][0]
    source = corpus.model_inputs(row)
    tensorizer = SemanticTensorizer.fit([source_record(source)])
    from norishio_lm.concept_model import ConceptVocabulary

    vocabulary = ConceptVocabulary.fit([row["targets"]])
    model = ToyModel("C", tensorizer, vocabulary, hidden_dim=8,
                     conditioning_mode="per_step_additive").eval()
    _, probabilities = source_encodings(model, [source], tensorizer)
    history = strict.target_history(row)
    input_ids = torch.tensor([history["input_ids"]], dtype=torch.long)
    labels = torch.tensor([history["labels"]], dtype=torch.long)
    spans = [authored_slot_spans(row["targets"], corpus.seed_data())]
    assert spans[0] is not None
    return model, probabilities, input_ids, labels, spans, row, source, tensorizer


def test_slot_ce_is_mean_cross_entropy_over_both_disjoint_authored_spans():
    from norishio_lm.slot_objective import slot_ce

    logits = torch.zeros(1, 5, 260)
    labels = torch.tensor([[7, 8, 9, 10, 11]])
    spans = [{"participant": {"start": 1, "end": 2}, "time": {"start": 3, "end": 5}}]
    logits[0, 1, 8] = 2.0
    logits[0, 3, 10] = 4.0
    logits[0, 4, 11] = 6.0
    expected = torch.nn.functional.cross_entropy(
        logits[0, [1, 3, 4]], labels[0, [1, 3, 4]], reduction="mean"
    )
    assert torch.allclose(slot_ce(logits, labels, spans), expected)


def test_slot_ce_backpropagates_decoder_gru_and_lm_head_but_masks_non_slot_positions():
    from norishio_lm.slot_objective import slot_ce

    model, probabilities, input_ids, labels, spans, *_ = _fixture()
    model.zero_grad(set_to_none=True)
    logits = model.decoder.decode_with_concept_intervention(input_ids, probabilities)["logits"]
    logits.retain_grad()
    loss = slot_ce(logits, labels, spans)
    loss.backward()

    assert model.decoder.gru.weight_ih_l0.grad is not None
    assert model.decoder.gru.weight_ih_l0.grad.abs().sum().item() > 0
    assert model.decoder.lm_head.weight.grad is not None
    assert model.decoder.lm_head.weight.grad.abs().sum().item() > 0
    slot_positions = {
        p
        for field in ("participant", "time")
        for p in range(spans[0][field]["start"], spans[0][field]["end"])
    }
    non_slot_positions = [p for p in range(labels.shape[1]) if p not in slot_positions]
    assert non_slot_positions
    assert torch.equal(logits.grad[0, non_slot_positions], torch.zeros_like(logits.grad[0, non_slot_positions]))


def test_future_reference_labels_cannot_change_current_decoder_logits():
    model, probabilities, input_ids, labels, spans, *_ = _fixture()
    from norishio_lm.slot_objective import slot_ce

    changed_labels = labels.clone()
    future = max(spans[0][field]["end"] for field in ("participant", "time"))
    changed_labels[:, future:] = (changed_labels[:, future:] + 31) % 260
    with torch.no_grad():
        first = model.decoder.decode_with_concept_intervention(input_ids, probabilities, labels=labels)["logits"]
        second = model.decoder.decode_with_concept_intervention(input_ids, probabilities, labels=changed_labels)["logits"]
    assert torch.equal(first, second)
    assert torch.equal(slot_ce(first, labels, spans), slot_ce(second, changed_labels, spans))


def test_future_input_bytes_do_not_change_earlier_slot_logits_and_source_rejects_target_key():
    model, probabilities, input_ids, labels, spans, row, source, tensorizer = _fixture()
    from norishio_lm.toy_collapse import source_encodings

    changed = input_ids.clone()
    boundary = spans[0]["participant"]["end"]
    changed[:, boundary:] = 5
    with torch.no_grad():
        first = model.decoder.decode_with_concept_intervention(input_ids, probabilities)["logits"]
        second = model.decoder.decode_with_concept_intervention(changed, probabilities)["logits"]
    assert torch.equal(first[:, :boundary], second[:, :boundary])
    assert not torch.equal(first[:, boundary:], second[:, boundary:])

    del labels
    with pytest.raises(ValueError):
        source_encodings(model, [{**source, "concept": row["targets"]["concept"]}], tensorizer)
