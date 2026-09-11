from __future__ import annotations

import pytest
torch = pytest.importorskip("torch")

from norishio_lm.concept_model import ConceptTargets, ConceptVocabulary
from norishio_lm.explicit_slot_model import (
    ExplicitSlotModel, slot_head_loss, source_distributions,
)
from norishio_lm.tensorizer import CHANNELS, SemanticTensorizer
from norishio_lm.toy_adapter import source_record
from norishio_lm.toy_experiment import ToyModel


def _setup():
    tensorizer = SemanticTensorizer({name: {"<PAD>": 0, "<UNK>": 1, '"x"': 2}
                                     for name in CHANNELS})
    counts = {"event": 4, "operators": 4, "agent": 3, "participant": 5,
              "time": 5, "location": 3, "repeat_marked": 2}
    fields = {name: {f"{name}-{i}": i + 1 for i in range(count)}
              for name, count in counts.items()}
    vocab = ConceptVocabulary(fields, {"sense": 1}, {"sememe": 0})
    initial = ToyModel("C", tensorizer, vocab, hidden_dim=32,
                       conditioning_mode="per_step_additive")
    return initial, tensorizer, vocab


def _rows(n=3):
    return [{"context": "", "text": f"source-{i}"} for i in range(n)]


def test_common_parameters_and_parameter_delta():
    initial, _, vocab = _setup()
    model = ExplicitSlotModel(initial, vocab)
    old = dict(initial.state_dict())
    new = model.state_dict()
    for key, value in old.items():
        if key == "decoder.concept_projection.weight":
            assert torch.equal(new[key][:, :33], value)
        else:
            assert torch.equal(new[key], value)
    assert sum(p.numel() for p in model.parameters()) - sum(p.numel() for p in initial.parameters()) == 650


def test_seed_is_reproducible_without_global_rng_contamination():
    initial, _, vocab = _setup()
    torch.manual_seed(123)
    expected = torch.rand(4)
    torch.manual_seed(123)
    first = ExplicitSlotModel(initial, vocab, new_seed=20)
    observed = torch.rand(4)
    second = ExplicitSlotModel(initial, vocab, new_seed=20)
    assert torch.equal(observed, expected)
    for key, value in first.state_dict().items():
        assert torch.equal(value, second.state_dict()[key])


def test_slot_ce_gradients_reach_heads_and_encoder():
    initial, tensorizer, vocab = _setup()
    model = ExplicitSlotModel(initial, vocab)
    semantic = tensorizer.encode([source_record(s) for s in _rows(2)])
    targets = [{"concept": {"participant": "participant-0", "time": "time-0"}}
               for _ in range(2)]
    _, _, logits = model(torch.zeros(2, 2, dtype=torch.long), semantic)
    loss = slot_head_loss(logits, targets, vocab)
    loss.backward()
    assert model.slot_heads.participant.weight.grad.abs().sum() > 0
    assert model.slot_heads.time.weight.grad.abs().sum() > 0
    assert model.encoder.encoders["surface"].embedding.weight.grad is not None
    assert model.encoder.encoders["surface"].embedding.weight.grad.abs().sum() > 0


def test_lm_loss_reaches_slot_heads_via_soft_probabilities():
    initial, tensorizer, vocab = _setup()
    model = ExplicitSlotModel(initial, vocab)
    semantic = tensorizer.encode([source_record(s) for s in _rows(2)])
    labels = torch.tensor([[2, 3], [3, 4]])
    output, _, _ = model(torch.zeros(2, 2, dtype=torch.long), semantic, labels=labels)
    output["lm_loss"].backward()
    assert model.slot_heads.participant.weight.grad.abs().sum() > 0
    assert model.slot_heads.time.weight.grad.abs().sum() > 0


def test_causality_and_source_allowlist():
    initial, tensorizer, vocab = _setup()
    model = ExplicitSlotModel(initial, vocab).eval()
    semantic = tensorizer.encode([source_record(_rows(1)[0])])
    ids = torch.tensor([[2, 3, 4]])
    with torch.no_grad():
        first, _, _ = model(ids, semantic)
        changed, _, _ = model(torch.tensor([[2, 3, 5]]), semantic)
    assert torch.equal(first["logits"][:, :2], changed["logits"][:, :2])
    with pytest.raises(ValueError):
        source_distributions(model, [{"context": "", "text": "x", "targets": {}}], tensorizer)


def test_concept_targets_require_present_slot_masks():
    _, _, vocab = _setup()
    fields = {"participant": torch.tensor([1]), "time": torch.tensor([1])}
    masks = {"participant": torch.tensor([False]), "time": torch.tensor([True])}
    targets = ConceptTargets(fields, masks, {}, torch.tensor([True]), torch.tensor([0.]), torch.tensor([True]))
    with pytest.raises(ValueError):
        slot_head_loss({"participant": torch.zeros(1, 5), "time": torch.zeros(1, 5)}, targets, vocab)


def test_labels_affect_loss_but_never_head_probabilities_or_logits():
    initial, tensorizer, vocab = _setup()
    model = ExplicitSlotModel(initial, vocab).eval()
    semantic = tensorizer.encode([source_record(_rows(1)[0])])
    ids = torch.tensor([[1, 7, 8]])
    with torch.no_grad():
        a, old_a, heads_a = model(ids, semantic, labels=torch.tensor([[7, 8, 2]]))
        b, old_b, heads_b = model(ids, semantic, labels=torch.tensor([[9, 10, 2]]))
    assert torch.equal(a["logits"], b["logits"])
    assert torch.equal(old_a.probabilities(), old_b.probabilities())
    assert all(torch.equal(heads_a[f], heads_b[f]) for f in heads_a)
    assert a["lm_loss"] != b["lm_loss"]
