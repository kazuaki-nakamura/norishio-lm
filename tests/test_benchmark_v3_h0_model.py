from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from norishio_lm.benchmark_v3_h0_model import (
    H0_ARM_IDS,
    PARAMETER_COUNT,
    build_future_h0_model,
)
from norishio_lm.benchmark_v3_model import BenchmarkV3Model


def _inputs() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    source = torch.tensor(
        [[1, 14, 37, 3], [1, 18, 29, 44]], dtype=torch.long,
    )
    decoder = torch.tensor(
        [[1, 22, 23, 2], [1, 25, 26, 2]], dtype=torch.long,
    )
    gates = torch.ones(2, decoder.shape[1], 4, dtype=torch.bool)
    return source, decoder, gates


def test_future_arms_have_identical_initial_state_and_parameter_count() -> None:
    ordinary = build_future_h0_model("H1L1", seed=7)
    anchor = build_future_h0_model("H1L1_ANCHOR", seed=7)

    assert H0_ARM_IDS == ("H1L1", "H1L1_ANCHOR")
    assert ordinary.parameter_count() == PARAMETER_COUNT == 32_120
    assert anchor.parameter_count() == PARAMETER_COUNT
    assert tuple(ordinary.state_dict()) == tuple(anchor.state_dict())
    for name, tensor in ordinary.state_dict().items():
        assert torch.equal(tensor, anchor.state_dict()[name]), name


def test_h1l1_future_arm_is_bitwise_parity_with_historical_model() -> None:
    source, decoder, gates = _inputs()
    historical = BenchmarkV3Model("H1L1", seed=11)
    future = build_future_h0_model("H1L1", seed=11)

    old = historical(source, decoder, local_gates=gates)
    new = future(source, decoder, local_gates=gates)
    assert torch.equal(old.latent, new.latent)
    assert torch.equal(old.logits, new.logits)
    for field in ("participant", "time", "event", "operator"):
        assert torch.equal(old.factor_logits[field], new.factor_logits[field])
        assert torch.equal(old.factor_probs[field], new.factor_probs[field])
    assert future.arm_id == "H1L1"
    assert future.arm == "H1L1"
    assert future.h0_mode == "source_latent"


def test_anchor_h0_is_source_independent_but_factor_path_is_source_dependent() -> None:
    source, decoder, gates = _inputs()
    anchor = build_future_h0_model("H1L1_ANCHOR", seed=13)

    initial_state = anchor.decoder_initial_state(source)
    assert torch.equal(initial_state[0], initial_state[1])

    output = anchor(source, decoder, local_gates=gates)
    assert not torch.equal(output.factor_logits["participant"][0],
                           output.factor_logits["participant"][1])
    assert not torch.equal(output.factor_logits["time"][0],
                           output.factor_logits["time"][1])
    assert anchor.arm_id == "H1L1_ANCHOR"
    assert anchor.arm == "H1L1_ANCHOR"
    assert anchor.h0_mode == "anchor_bos_sep"


def test_anchor_h0_keeps_gradient_route_through_shared_encoder() -> None:
    source, _, _ = _inputs()
    anchor = build_future_h0_model("H1L1_ANCHOR", seed=17)

    anchor.decoder_initial_state(source).sum().backward()

    assert anchor.encoder.embedding.weight.grad is not None
    assert bool(anchor.encoder.embedding.weight.grad.abs().sum() > 0)


def test_one_hot_forward_keeps_legacy_intervention_parity() -> None:
    source, decoder, gates = _inputs()
    model = build_future_h0_model("H1L1", seed=19)
    one_hot = torch.tensor([0.0, 0.0, 1.0, 0.0])

    legacy = model.forward_with_intervention(
        source, decoder, "event", one_hot, local_gates=gates,
    )
    replacement = model.forward_with_probability_replacement(
        source, decoder, "event", one_hot, local_gates=gates,
    )
    assert torch.equal(legacy.logits, replacement.logits)
    assert torch.equal(
        legacy["intervened_factor_probs"]["event"],
        replacement["replaced_factor_probs"]["event"],
    )


def test_soft_replacement_accepts_vector_and_preserves_other_factors() -> None:
    source, decoder, gates = _inputs()
    model = build_future_h0_model("H1L1_ANCHOR", seed=23)
    baseline = model(source, decoder, local_gates=gates)
    h0_before = model.decoder_initial_state(source)
    soft = torch.tensor([0.05, 0.10, 0.20, 0.25, 0.15, 0.25])

    replaced = model.replace_factor_probability(
        baseline.factor_probs, "participant", soft,
    )
    assert torch.allclose(replaced["participant"], soft.expand(2, -1))
    for field in ("time", "event", "operator"):
        assert torch.equal(replaced[field], baseline.factor_probs[field])

    changed = model.forward_with_probability_replacement(
        source, decoder, "participant", soft, local_gates=gates,
    )
    assert torch.allclose(
        changed["replaced_factor_probs"]["participant"], soft.expand(2, -1),
    )
    assert torch.equal(
        h0_before,
        model.decoder_initial_state(source),
    )


@pytest.mark.parametrize(
    ("replacement", "message"),
    [
        (torch.tensor([0.2, 0.2, 0.2, 0.2, 0.2]), "width 6"),
        (torch.tensor([-0.1, 0.1, 0.2, 0.2, 0.2, 0.4]), "nonnegative"),
        (torch.tensor([float("nan"), 0.0, 0.0, 0.0, 0.0, 1.0]), "finite"),
        (torch.tensor([0.1, 0.1, 0.1, 0.1, 0.1, 0.1]), "sum to one"),
        (torch.ones(3, 6) / 6.0, "batch"),
    ],
)
def test_soft_replacement_rejects_invalid_vectors(replacement, message) -> None:
    source, decoder, gates = _inputs()
    model = build_future_h0_model("H1L1", seed=29)
    baseline = model(source, decoder, local_gates=gates)

    with pytest.raises((TypeError, ValueError), match=message):
        model.replace_factor_probability(
            baseline.factor_probs, "participant", replacement,
        )


@pytest.mark.parametrize("arm", ["", "H1L0", "H1L1_ANCHOR_EXTRA", "D_AUX"])
def test_invalid_future_arm_is_rejected(arm: str) -> None:
    with pytest.raises(ValueError, match="unknown future H0-path arm"):
        build_future_h0_model(arm, seed=7)
