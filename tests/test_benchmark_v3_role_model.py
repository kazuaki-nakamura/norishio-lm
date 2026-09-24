from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from norishio_lm.benchmark_v3_role_model import (
    ROLE_DISJOINT_CODE_BYTES,
    build_initialized_model,
    canonical_digit_token_id,
    corresponding_embedding_sequences,
    encode_role_digits,
    factor_only_training_step,
    paired_models,
    role_code_token_id,
)


def test_role_aliases_are_disjoint_and_map_to_canonical_digits() -> None:
    assert len(ROLE_DISJOINT_CODE_BYTES) == 20
    assert len(set(ROLE_DISJOINT_CODE_BYTES)) == 20
    assert encode_role_digits([0, 3], "ROLE_DISJOINT").tolist() == [role_code_token_id("ROLE_DISJOINT", 0), role_code_token_id("ROLE_DISJOINT", 3)]
    assert [role_code_token_id("ROLE_DISJOINT", digit) for digit in range(6)]
    assert role_code_token_id("ROLE_DISJOINT", 5) != role_code_token_id("ROLE_DISJOINT", 0)
    assert role_code_token_id("ROLE_ALIASED", 0) != role_code_token_id("ROLE_DISJOINT", 0)
    assert canonical_digit_token_id(0) == role_code_token_id("ROLE_ALIASED", 0)


def test_paired_initialization_is_bitwise_equal_and_parameter_matched() -> None:
    left, right, report = paired_models(7)
    assert report["bitwise_equal"] is True
    assert report["trainable_parameters"] == 32120
    assert report["state_sha256"]
    assert all(torch.equal(value, right.state_dict()[name]) for name, value in left.state_dict().items())
    # Clone-copying creates independent rows, rather than tying parameters.
    assert left.encoder.embedding.weight.data_ptr() != left.encoder.projection.weight.data_ptr()


def test_corresponding_role_sequences_equal_canonical_digit_rows() -> None:
    model = build_initialized_model(11)
    sequences = corresponding_embedding_sequences(model, [0, 3, 5, 1])
    assert torch.equal(sequences["ROLE_ALIASED"], sequences["ROLE_DISJOINT"])
    for index, digit in enumerate([0, 3, 5, 1]):
        assert torch.equal(sequences["ROLE_DISJOINT"][index], model.encoder.embedding.weight[canonical_digit_token_id(digit)])


def test_factor_only_step_leaves_decoder_without_gradients() -> None:
    model = build_initialized_model(13)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
    source = torch.tensor([[1, 65, 66, 3], [1, 75, 76, 3]], dtype=torch.long)
    targets = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]], dtype=torch.long)
    loss = factor_only_training_step(model, optimizer, source, targets)
    assert loss > 0
    assert all(parameter.grad is None for parameter in model.decoder.parameters())


def test_pair_outputs_match_on_synthetic_role_inputs() -> None:
    left, right, _ = paired_models(17)
    left.eval(); right.eval()
    digits = [0, 2, 5, 3]
    source_left = torch.stack([torch.cat((torch.tensor([1]), encode_role_digits(digits, "ROLE_ALIASED"), torch.tensor([3])))])
    source_right = torch.stack([torch.cat((torch.tensor([1]), encode_role_digits(digits, "ROLE_DISJOINT"), torch.tensor([3])))])
    with torch.no_grad():
        output_left = left(source_left)
        output_right = right(source_right)
    assert torch.equal(output_left.latent, output_right.latent)
    for factor in ("participant", "time", "event", "operator"):
        assert torch.equal(output_left.factor_probs[factor], output_right.factor_probs[factor])
