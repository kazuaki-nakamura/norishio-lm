from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from norishio_lm.factorized_slot_model import FactorizedProjection, FactorizedSlotModel
from norishio_lm.toy_experiment import ToyModel
from norishio_lm.explicit_slot_model import ExplicitSlotModel
from norishio_lm.toy_adapter import source_record
from test_explicit_slot_model import _setup


def test_projection_copies_blocks_and_matches_shared():
    shared = torch.nn.Linear(43, 32)
    projection = FactorizedProjection(shared)
    x = torch.randn(4, 43)
    assert torch.allclose(projection(x), shared(x), atol=1e-6, rtol=1e-6)
    assert projection.in_features == 43 and projection.out_features == 32
    assert projection.participant_proj.bias is None
    assert sum(p.numel() for p in projection.parameters()) == sum(p.numel() for p in shared.parameters())
    assert torch.equal(projection.base_proj.weight, shared.weight[:, :33])
    assert torch.equal(projection.base_proj.bias, shared.bias)
    assert torch.equal(projection.participant_proj.weight, shared.weight[:, 33:38])
    assert torch.equal(projection.time_proj.weight, shared.weight[:, 38:43])


def test_factorized_model_preserves_rng_and_old_weights():
    initial, tensorizer, vocabulary = _setup()
    torch.manual_seed(123)
    expected = torch.rand(3)
    torch.manual_seed(123)
    model = FactorizedSlotModel(initial, vocabulary)
    assert torch.equal(expected, torch.rand(3))
    assert isinstance(model.decoder.concept_projection, FactorizedProjection)
    assert sum(p.numel() for p in model.parameters()) == sum(p.numel() for p in initial.parameters()) + 650


def test_zero_participant_input_removes_only_participant_branch():
    initial, _, vocabulary = _setup()
    model = FactorizedSlotModel(initial, vocabulary).eval()
    projection = model.decoder.concept_projection
    x = torch.randn(2, 43)
    zeroed = x.clone(); zeroed[:, 33:38] = 0
    expected = projection.base_proj(x[:, :33]) + projection.time_proj(x[:, 38:43])
    assert torch.allclose(projection(zeroed), expected)


def test_slot_gradients_and_future_token_causality():
    initial, tensorizer, vocabulary = _setup()
    model = FactorizedSlotModel(initial, vocabulary)
    semantic = tensorizer.encode([source_record({"context": "", "text": "x"})] * 2)
    ids = torch.tensor([[2, 3, 4], [2, 3, 5]])
    output, _, _ = model(ids, semantic, labels=torch.tensor([[3, 4, 5], [3, 4, 6]]))
    output["lm_loss"].backward()
    assert model.decoder.concept_projection.participant_proj.weight.grad is not None
    assert model.decoder.concept_projection.participant_proj.weight.grad.abs().sum() > 0
    with torch.no_grad():
        a, _, _ = model(ids, semantic)
        changed = ids.clone(); changed[0, 2] = 99
        b, _, _ = model(changed, semantic)
    assert torch.equal(a["logits"][:, :2], b["logits"][:, :2])
