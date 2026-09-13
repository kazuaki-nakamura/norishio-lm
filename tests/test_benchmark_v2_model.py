from __future__ import annotations

import pytest
import torch

from norishio_lm.benchmark_v2_model import (
    ARM_IDS,
    FACTOR_ORDER,
    BenchmarkV2Model,
)


DERANGEMENTS = {
    "participant": [2, 3, 4, 5, 0, 1],
    "time": [3, 4, 5, 0, 1, 2],
    "event": [2, 3, 0, 1],
    "operator": [3, 0, 1, 2],
}


def inputs(batch: int = 3) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    source = torch.tensor([[1, 14, 37, 0, 0], [1, 18, 29, 44, 3],
                           [1, 20, 21, 22, 3]], dtype=torch.long)[:batch]
    decoder = torch.tensor([[1, 22, 23, 2], [1, 25, 26, 2], [1, 27, 28, 2]],
                           dtype=torch.long)[:batch]
    factors = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0], [2, 3, 0, 1]],
                           dtype=torch.long)[:batch]
    return source, decoder, factors


def model(arm: str, seed: int = 7) -> BenchmarkV2Model:
    return BenchmarkV2Model(arm, seed=seed,
                            derangement_mappings=DERANGEMENTS if arm == "E" else None)


def test_frozen_parameter_counts_and_shapes() -> None:
    expected = {"A_G0": 29272, "A_G1": 29272, "B": 29848,
                "C": 29816, "D": 29532, "E": 29272}
    source, decoder, factors = inputs()
    for arm in ARM_IDS:
        candidate = model(arm)
        output = candidate(source, decoder,
                           local_gates=(torch.zeros(3, 4, 4, dtype=torch.bool)
                                        if arm == "B" else None),
                           factor_targets=(factors if arm != "D" else None))
        assert candidate.parameter_count() == expected[arm]
        assert output["logits"].shape == (3, 4, 260)
        assert output["latent"].shape == (3, 32)
        assert set(output["factor_logits"]) == (set(FACTOR_ORDER) if arm != "D" else set())


def test_source_mask_is_used_and_source_metadata_is_rejected() -> None:
    candidate = model("A_G0")
    source, _, _ = inputs()
    masked = candidate.source_predictions(source, source_mask=source.ne(0))
    changed_pad = source.clone()
    changed_pad[source.eq(0)] = 99
    changed = candidate.source_predictions(changed_pad, source_mask=source.ne(0))
    assert torch.equal(masked.latent, changed.latent)
    with pytest.raises((TypeError, ValueError)):
        candidate.source_predictions({"context": "", "text": "gold metadata"})  # type: ignore[arg-type]


@pytest.mark.parametrize("arm", ["A_G0", "A_G1", "C", "E"])
def test_factor_loss_is_source_only_and_labels_are_external(arm: str) -> None:
    source, decoder, factors = inputs()
    candidate = model(arm)
    first = candidate(source, decoder, labels=decoder, factor_targets=factors)
    second = candidate(source, decoder, labels=torch.zeros_like(decoder), factor_targets=factors)
    for field in FACTOR_ORDER:
        assert torch.equal(first["factor_logits"][field], second["factor_logits"][field])


def test_a_g1_detaches_only_lm_path_before_factor_heads() -> None:
    source, decoder, factors = inputs()
    for arm, expect_lm_grad in [("A_G0", True), ("A_G1", False)]:
        candidate = model(arm)
        output = candidate(source, decoder, labels=decoder, factor_targets=factors)
        output["lm_loss"].backward(retain_graph=True)
        lm_grads = [candidate.factor_heads[field].weight.grad for field in FACTOR_ORDER]
        assert all((grad is not None and bool(grad.abs().sum())) == expect_lm_grad
                   for grad in lm_grads)
        encoder_grad = candidate.encoder.projection.weight.grad
        assert (encoder_grad is not None and bool(encoder_grad.abs().sum())) == expect_lm_grad
        candidate.zero_grad(set_to_none=True)
        output["factor_loss"].backward()
        assert all(candidate.factor_heads[field].weight.grad is not None
                   and bool(candidate.factor_heads[field].weight.grad.abs().sum())
                   for field in FACTOR_ORDER)


def test_c_uses_hard_symbols_and_blocks_lm_grad_to_factor_heads() -> None:
    source, decoder, factors = inputs()
    candidate = model("C")
    output = candidate(source, decoder, labels=decoder, factor_targets=factors)
    assert output["symbol_indices"].shape == (3, 4)
    assert output["symbol_indices"].dtype == torch.long
    assert not any(parameter.numel() == 576 * 8 for parameter in candidate.parameters())
    output["lm_loss"].backward(retain_graph=True)
    assert all(candidate.factor_heads[field].weight.grad is None for field in FACTOR_ORDER)
    assert candidate.encoder.projection.weight.grad is None
    candidate.zero_grad(set_to_none=True)
    output["factor_loss"].backward()
    assert all(candidate.factor_heads[field].weight.grad is not None
               and bool(candidate.factor_heads[field].weight.grad.abs().sum())
               for field in FACTOR_ORDER)


def test_d_has_latent_adapter_and_no_factor_loss() -> None:
    source, decoder, factors = inputs()
    candidate = model("D")
    output = candidate(source, decoder, labels=decoder)
    assert candidate.latent_adapter[0].out_features == 24
    assert candidate.latent_adapter[2].out_features == 32
    assert "factor_loss" not in output
    with pytest.raises(ValueError, match="no factor"):
        candidate.factor_loss({}, factors)


def test_b_uses_four_bias_free_projections_and_prefix_gates() -> None:
    source, decoder, factors = inputs()
    candidate = model("B")
    assert all(projection.bias is None for projection in candidate.factor_projections.values())
    assert candidate.factor_projection_bias.shape == (32,)
    zeros = torch.zeros(3, 4, 4, dtype=torch.bool)
    participant = zeros.clone()
    participant[:, :, 0] = True
    zero_output = candidate(source, decoder, factor_targets=factors, local_gates=zeros)
    participant_output = candidate(
        source, decoder, factor_targets=factors, local_gates=participant
    )
    assert not torch.equal(zero_output["logits"], participant_output["logits"])
    with pytest.raises(ValueError, match="prefix-local"):
        candidate(source, decoder, factor_targets=factors)
    with pytest.raises(TypeError, match="boolean"):
        candidate(source, decoder, local_gates=zeros.float())


def test_e_requires_and_applies_explicit_derangements() -> None:
    with pytest.raises(ValueError, match="explicitly supplied"):
        BenchmarkV2Model("E")
    source, decoder, factors = inputs()
    candidate = model("E")
    output = candidate(source, decoder, labels=decoder, factor_targets=factors)
    expected = sum(torch.nn.functional.cross_entropy(
        output["factor_logits"][field],
        torch.tensor(DERANGEMENTS[field])[factors[:, index]],
    ) for index, field in enumerate(FACTOR_ORDER))
    assert torch.allclose(output["factor_loss"], expected)
    wrong = {**DERANGEMENTS, "event": [1, 2, 3, 0]}
    with pytest.raises(ValueError, match="frozen manifest"):
        BenchmarkV2Model("E", derangement_mappings=wrong)
