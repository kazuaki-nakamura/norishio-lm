"""Regression tests for stopping LM gradients at explicit slot distributions."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from norishio_lm.explicit_slot_model import slot_head_loss
from norishio_lm.local_slot_model import DuplicateRemovedModel
from norishio_lm.toy_adapter import source_record
from norishio_lm.toy_experiment import ToyModel
from test_explicit_slot_model import _setup


def _models_and_batch():
    initial, tensorizer, vocabulary = _setup()
    first = DuplicateRemovedModel(initial, vocabulary, new_seed=20, local=False,
                                  gradient_routing="end_to_end")
    second = DuplicateRemovedModel(initial, vocabulary, new_seed=20, local=False,
                                   gradient_routing="stop_slot_lm")
    semantic = tensorizer.encode([
        source_record({"context": "", "text": "source-a"}),
        source_record({"context": "", "text": "source-b"}),
    ])
    ids = torch.tensor([[1, 8, 9, 10], [1, 11, 12, 13]], dtype=torch.long)
    labels = torch.tensor([[8, 9, 10, 11], [11, 12, 13, 14]], dtype=torch.long)
    return first, second, vocabulary, semantic, ids, labels


def _forward(model, semantic, ids, labels):
    return model(ids, semantic, labels=labels)


def test_gradient_modes_have_same_initial_state_parameter_count_and_forward_values():
    end_to_end, stopped, _, semantic, ids, labels = _models_and_batch()

    assert sum(p.numel() for p in end_to_end.parameters()) == sum(
        p.numel() for p in stopped.parameters()
    )
    assert end_to_end.state_dict().keys() == stopped.state_dict().keys()
    for name, value in end_to_end.state_dict().items():
        assert torch.equal(value, stopped.state_dict()[name]), name

    out0, aux0, slots0 = _forward(end_to_end, semantic, ids, labels)
    out1, aux1, slots1 = _forward(stopped, semantic, ids, labels)
    assert torch.equal(out0["logits"], out1["logits"])
    assert torch.equal(out0["lm_loss"], out1["lm_loss"])
    for field in slots0:
        assert torch.equal(slots0[field], slots1[field])
    for field in aux0.concept_logits:
        assert torch.equal(aux0.concept_logits[field], aux1.concept_logits[field])


@pytest.mark.parametrize("mode, expect_slot_grad", [("end_to_end", True), ("stop_slot_lm", False)])
def test_lm_only_gradient_routing(mode, expect_slot_grad):
    initial, tensorizer, vocabulary = _setup()
    model = DuplicateRemovedModel(initial, vocabulary, new_seed=20, local=False,
                                  gradient_routing=mode)
    semantic = tensorizer.encode([
        source_record({"context": "", "text": "source-a"}),
        source_record({"context": "", "text": "source-b"}),
    ])
    ids = torch.tensor([[1, 8, 9, 10], [1, 11, 12, 13]], dtype=torch.long)
    labels = torch.tensor([[8, 9, 10, 11], [11, 12, 13, 14]], dtype=torch.long)
    output, _, _ = _forward(model, semantic, ids, labels)
    output["lm_loss"].backward()

    for field in ("participant", "time"):
        grad = model.slot_heads[field].weight.grad
        if expect_slot_grad:
            assert grad is not None and grad.abs().sum() > 0, field
        else:
            assert grad is None or grad.abs().sum() == 0, field

    # Stopping the slot path must not disable the decoder's own LM gradients.
    for parameter in (model.decoder.embedding.weight, model.decoder.gru.weight_ih_l0,
                      model.decoder.lm_head.weight):
        assert parameter.grad is not None and parameter.grad.abs().sum() > 0


def test_slot_ce_still_trains_heads_in_stop_slot_lm_mode():
    initial, tensorizer, vocabulary = _setup()
    model = DuplicateRemovedModel(initial, vocabulary, new_seed=20, local=False,
                                  gradient_routing="stop_slot_lm")
    semantic = tensorizer.encode([
        source_record({"context": "", "text": "source-a"}),
        source_record({"context": "", "text": "source-b"}),
    ])
    ids = torch.tensor([[1, 8, 9, 10], [1, 11, 12, 13]], dtype=torch.long)
    labels = torch.tensor([[8, 9, 10, 11], [11, 12, 13, 14]], dtype=torch.long)
    _, _, slot_logits = _forward(model, semantic, ids, labels)
    targets = [{"concept": {"participant": "participant-0", "time": "time-0"}}] * 2
    slot_head_loss(slot_logits, targets, vocabulary).backward()

    for field in ("participant", "time"):
        grad = model.slot_heads[field].weight.grad
        assert grad is not None and grad.abs().sum() > 0, field
    encoder_grad = model.encoder.encoders["surface"].embedding.weight.grad
    assert encoder_grad is not None and encoder_grad.abs().sum() > 0


def test_routing_does_not_allow_labels_or_gold_source_fields_into_predictions():
    model, _, _, semantic, ids, labels = _models_and_batch()
    with torch.no_grad():
        first, _, heads_first = _forward(model, semantic, ids, labels)
        changed, _, heads_changed = _forward(
            model, semantic, ids, labels.flip(0).roll(1, dims=1)
        )
    assert torch.equal(first["logits"], changed["logits"])
    for field in heads_first:
        assert torch.equal(heads_first[field], heads_changed[field])

    with pytest.raises(ValueError):
        source_record({"context": "", "text": "source-a", "targets": {"concept": {}}})
