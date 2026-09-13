from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from norishio_lm.benchmark_v3_model import (
    ARM_IDS,
    FACTOR_ORDER,
    FACTOR_SIZES,
    MAIN_ARMS,
    BenchmarkV3Model,
)


def tensors(batch: int = 2, steps: int = 5):
    source = torch.tensor([[1, 14, 37, 3, 0], [1, 18, 29, 44, 3]], dtype=torch.long)[:batch]
    decoder = torch.tensor([[1, 22, 23, 2, 2], [1, 25, 26, 2, 2]], dtype=torch.long)[:batch, :steps]
    targets = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]], dtype=torch.long)[:batch]
    return source, decoder, targets


def test_main_arms_share_parameter_structure_and_gate_disabled_equivalence():
    models = {arm: BenchmarkV3Model(arm, seed=7) for arm in MAIN_ARMS}
    assert all(models["H0L0"].state_dict().keys() == model.state_dict().keys() for model in models.values())
    assert len({model.parameter_count() for model in models.values()}) == 1
    for name, parameter in models["H0L0"].named_parameters():
        assert torch.equal(parameter, models["H1L1"].state_dict()[name])
    source, decoder, _ = tensors()
    zero = torch.ones(2, decoder.shape[1], 4, dtype=torch.bool)
    l0 = models["H0L0"](source, decoder)
    l1 = models["H0L1"](source, decoder, local_gates=zero)
    assert torch.equal(l0.latent, l1.latent)
    assert torch.equal(l0.logits, l1.logits)
    assert all(torch.equal(l0.factor_logits[field], l1.factor_logits[field]) for field in FACTOR_ORDER)


def test_h0_h1_differ_only_by_factor_head_activation():
    h0 = BenchmarkV3Model("H0L0", seed=11)
    h1 = BenchmarkV3Model("H1L0", seed=11)
    assert h0.parameter_count() == h1.parameter_count()
    assert type(h0.factor_heads["event"].activation) is torch.nn.Identity
    assert type(h1.factor_heads["event"].activation) is torch.nn.Tanh
    for name, parameter in h0.named_parameters():
        assert torch.equal(parameter, h1.state_dict()[name])


def test_local_gates_are_step_local_and_future_gate_changes_do_not_change_past():
    source, decoder, _ = tensors()
    model = BenchmarkV3Model("H1L1", seed=3)
    gates = torch.ones(2, decoder.shape[1], 4, dtype=torch.bool)
    changed = gates.clone()
    changed[:, 3:, 0] = False
    first = model(source, decoder, local_gates=gates).logits
    second = model(source, decoder, local_gates=changed).logits
    assert torch.equal(first[:, :3], second[:, :3])
    with pytest.raises(ValueError, match="local_gates"):
        model(source, decoder, local_gates=torch.ones(2, decoder.shape[1] - 1, 4, dtype=torch.bool))


def test_d_aux_has_factor_heads_but_decoder_uses_latent_only():
    source, decoder, targets = tensors()
    model = BenchmarkV3Model("D_AUX", seed=5)
    output = model(source, decoder, factor_targets=targets)
    assert set(output.factor_logits) == set(FACTOR_ORDER)
    assert "factor_loss" in output and "logits" in output
    assert model.arm_metadata["decoder_factor_conditioning"] is False
    original = model._factor_logits
    model._factor_logits = lambda latent: {
        field: torch.zeros_like(original(latent)[field]) for field in FACTOR_ORDER
    }
    changed = model(source, decoder)
    assert torch.equal(output.logits, changed.logits)


def test_no_input_replaces_every_source_row_with_identical_fixed_tensor():
    source, decoder, _ = tensors()
    model = BenchmarkV3Model("NO_INPUT", seed=2)
    first = model(source, decoder)
    altered = source.flip(1)
    second = model(altered, decoder)
    assert first.source_ids.shape == (2, 2)
    assert torch.equal(first.source_ids, second.source_ids)
    assert torch.equal(first.latent, second.latent)
    assert torch.equal(first.logits, second.logits)


def test_canonical_probabilities_and_single_factor_intervention_are_validated():
    source, decoder, _ = tensors()
    model = BenchmarkV3Model("H0L0", seed=4)
    output = model(source, decoder)
    probabilities = output.factor_probs
    assert {field: tuple(value.shape) for field, value in probabilities.items()} == {
        "participant": (2, 6), "time": (2, 6), "event": (2, 4), "operator": (2, 4),
    }
    one_hot = torch.zeros(2, FACTOR_SIZES["event"])
    one_hot[:, 1] = 1
    intervened = model.intervene_probabilities(probabilities, "event", one_hot)
    assert torch.equal(intervened["event"], one_hot)
    assert all(torch.equal(intervened[field], probabilities[field]) for field in ("participant", "time", "operator"))
    with pytest.raises(ValueError, match="width 4"):
        model.intervene_probabilities(probabilities, "event", torch.tensor([[1.0, 0.0]]))
    assert output["factor_metadata"]["canonical_space"] == "canonical_semantic_space"
    assert output["factor_metadata"]["code_space"] == "canonical_factor_class_indices"


def test_common_loss_structure_and_factor_targets_are_available_for_all_arms():
    source, decoder, targets = tensors()
    for arm in ARM_IDS:
        gates = torch.ones(2, decoder.shape[1], 4, dtype=torch.bool) if arm.endswith("L1") else None
        output = BenchmarkV3Model(arm, seed=8)(source, decoder, labels=decoder,
                                               factor_targets=targets, local_gates=gates)
        assert set(("lm_loss", "factor_loss", "loss")) <= set(output)
        assert torch.isfinite(output["loss"])
