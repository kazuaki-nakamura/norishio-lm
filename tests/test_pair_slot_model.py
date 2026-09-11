from __future__ import annotations

import pytest
import math

torch = pytest.importorskip("torch")

from norishio_lm.concept_model import ConceptVocabulary
from norishio_lm.pair_slot_model import (
    PairSlotModel, pair_head_loss, pair_marginals, source_pair_distributions,
)
from norishio_lm.tensorizer import CHANNELS, SemanticTensorizer
from norishio_lm.toy_adapter import source_record
from norishio_lm.toy_experiment import ToyModel
from norishio_lm.explicit_slot_model import slot_head_loss
from norishio_lm.local_slot_model import DuplicateRemovedModel


def _setup() -> tuple[PairSlotModel, SemanticTensorizer, ConceptVocabulary]:
    tensorizer = SemanticTensorizer({name: {"<PAD>": 0, "<UNK>": 1, '"x"': 2}
                                     for name in CHANNELS})
    counts = {"event": 4, "operators": 4, "agent": 3, "participant": 5,
              "time": 5, "location": 3, "repeat_marked": 2}
    vocabulary = ConceptVocabulary(
        {name: {f"{name}-{i}": i + 1 for i in range(count)}
         for name, count in counts.items()}, {"sense": 1}, {"sememe": 0})
    initial = ToyModel("C", tensorizer, vocabulary, hidden_dim=32,
                       conditioning_mode="per_step_additive")
    return PairSlotModel(initial, vocabulary), tensorizer, vocabulary


def test_pair_forward_shapes_and_parameter_budget():
    model, tensorizer, _ = _setup()
    initial_count = sum(p.numel() for p in model.parameters())
    assert model.pair_head.weight.shape == (25, 32)
    assert sum(p.numel() for p in model.pair_head.parameters()) == 825
    semantic = tensorizer.encode([source_record({"context": "", "text": "x"})] * 2)
    ids = torch.ones((2, 4), dtype=torch.long)
    output, auxiliary, independent, pair = model(ids, semantic)
    assert output["logits"].shape == (2, 4, 260)
    assert auxiliary.probabilities().shape == (2, 33)
    assert independent["participant"].shape == independent["time"].shape == (2, 5)
    assert pair.shape == (2, 25)
    assert initial_count >= 825


def test_pair_marginals_and_pair_loss_validate_and_use_cartesian_ids():
    probabilities = torch.full((2, 25), 1 / 25)
    marginal = pair_marginals(probabilities)
    assert marginal.shape == (2, 10)
    assert torch.allclose(marginal[:, :5], torch.full((2, 5), 1 / 5))
    assert torch.allclose(marginal[:, 5:], torch.full((2, 5), 1 / 5))
    with pytest.raises(ValueError, match="normalized"):
        pair_marginals(torch.ones((1, 25)))
    with pytest.raises(ValueError, match="finite"):
        pair_marginals(torch.full((1, 25), float("nan")))
    _, _, vocabulary = _setup()
    logits = torch.zeros((1, 25), requires_grad=True)
    targets = [{"concept": {"participant": "participant-2", "time": "time-4"}}]
    loss = pair_head_loss(logits, targets, vocabulary)
    assert float(loss.detach()) == pytest.approx(math.log(25))
    loss.backward()
    assert logits.grad is not None and logits.grad.abs().sum() > 0
    assert int(logits.grad.argmin()) == 14
    assert pair_marginals(torch.empty(0, 25)).shape == (0, 10)
    with pytest.raises(ValueError, match="floating"):
        pair_marginals(torch.zeros(1, 25, dtype=torch.long))


def test_independent_heads_are_diagnostic_only_and_both_heads_get_gradients():
    model, tensorizer, vocabulary = _setup()
    semantic = tensorizer.encode([source_record({"context": "", "text": "x"})] * 2)
    ids = torch.tensor([[1, *[b + 4 for b in "私は、今日友人と".encode()]]] * 2)
    first = model(ids, semantic)
    clone, _, _ = _setup()
    clone.load_state_dict(model.state_dict())
    with torch.no_grad():
        clone.slot_heads.participant.bias[0].add_(10)
        clone.slot_heads.time.bias[1].sub_(10)
    second = clone(ids, semantic)
    assert torch.equal(first[0]["logits"], second[0]["logits"])
    targets = [
        {"concept": {"participant": "participant-1", "time": "time-1"}},
        {"concept": {"participant": "participant-2", "time": "time-2"}},
    ]
    loss = pair_head_loss(first[3], targets, vocabulary) + slot_head_loss(first[2], targets, vocabulary)
    loss.backward()
    assert model.pair_head.weight.grad is not None
    assert model.slot_heads.participant.weight.grad is not None
    assert model.slot_heads.time.weight.grad is not None


def test_source_pair_distributions_batch_and_reject_gold_leak():
    model, tensorizer, _ = _setup()
    sources = [{"context": "", "text": "x"}] * 17
    old, independent, pair, marginal = source_pair_distributions(model, sources, tensorizer)
    assert old.shape == (17, 33)
    assert independent.shape == (17, 10)
    assert pair.shape == (17, 25)
    assert marginal.shape == (17, 10)
    assert torch.allclose(marginal, pair_marginals(pair))
    with pytest.raises(ValueError, match="exactly a dict"):
        source_pair_distributions(model, [{"context": "", "text": "x", "gold": "bad"}], tensorizer)


def test_pair_keeps_initial_d_and_gate_causal_and_lm_reaches_pair_only():
    model, tensorizer, vocabulary = _setup()
    initial = ToyModel("C", tensorizer, vocabulary, conditioning_mode="per_step_additive")
    rng = torch.get_rng_state().clone()
    paired = PairSlotModel(initial, vocabulary)
    assert torch.equal(rng, torch.get_rng_state())
    baseline = DuplicateRemovedModel(initial, vocabulary, new_seed=20, local=True)
    assert all(torch.equal(v, paired.state_dict()[k]) for k, v in baseline.state_dict().items())
    assert sum(p.numel() for p in paired.parameters()) - sum(p.numel() for p in baseline.parameters()) == 825
    semantic = tensorizer.encode([source_record({"context": "", "text": "x"})])
    ids = torch.tensor([[1, *[b + 4 for b in "私は、今日友人と".encode()]]])
    changed = ids.clone()
    changed[:, -2:] = 4
    first = model(ids, semantic, labels=ids)
    second = model(changed, semantic)
    assert torch.equal(first[0]["logits"][:, :-2], second[0]["logits"][:, :-2])
    first[0]["lm_loss"].backward()
    assert float(model.pair_head.weight.grad.abs().sum()) > 0
    assert model.slot_heads.participant.weight.grad is None
    assert model.slot_heads.time.weight.grad is None
