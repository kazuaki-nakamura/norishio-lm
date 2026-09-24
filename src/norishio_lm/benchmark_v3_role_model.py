"""Role-disjoint byte-code initialization for the H1L1 benchmark model.

The role code bytes are deliberately aliases for canonical digit bytes.  The
aliases are represented by separate vocabulary rows (there is no parameter
tying), but each row is cloned from its corresponding digit row before any
training.  This keeps the two paired codebook variants parameter matched while
allowing callers to use disjoint surface bytes for the two roles.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch
from torch import Tensor

from .benchmark_v3_model import BenchmarkV3Model, build_model
from .benchmark_v2_model import BYTE_OFFSET
from .benchmark_v3_checkpoint import state_sha256
from torch.nn.utils import clip_grad_norm_

ROLE_DISJOINT_CODE_BYTES = tuple(range(ord("A"), ord("T") + 1))
ROLE_DISJOINT_CODEBOOK: dict[int, int] = {
    **{value: index for index, value in enumerate(ROLE_DISJOINT_CODE_BYTES[0:6])},
    **{value: index for index, value in enumerate(ROLE_DISJOINT_CODE_BYTES[6:12])},
    **{value: index for index, value in enumerate(ROLE_DISJOINT_CODE_BYTES[12:16])},
    **{value: index for index, value in enumerate(ROLE_DISJOINT_CODE_BYTES[16:20])},
}
ROLE_FACTOR_CODEBOOKS = {
    "participant": dict(zip(ROLE_DISJOINT_CODE_BYTES[0:6], range(6))),
    "time": dict(zip(ROLE_DISJOINT_CODE_BYTES[6:12], range(6))),
    "event": dict(zip(ROLE_DISJOINT_CODE_BYTES[12:16], range(4))),
    "operator": dict(zip(ROLE_DISJOINT_CODE_BYTES[16:20], range(4))),
}
ROLE_ALIASED_CODEBOOK: dict[int, int] = {
    ord(str(index)): index for index in range(10)
}
ROLE_DISJOINT_CODEBOOKS: dict[str, dict[int, int]] = {
    "ROLE_ALIASED": ROLE_ALIASED_CODEBOOK,
    "ROLE_DISJOINT": ROLE_DISJOINT_CODEBOOK,
}
# A short alias is useful in descriptors and keeps the protocol vocabulary
# explicit for downstream code.
ROLE_DISJOINT = ROLE_DISJOINT_CODEBOOKS
ROLE_VARIANTS = tuple(ROLE_DISJOINT_CODEBOOKS)
# Descriptor-friendly names for callers that describe the two paired sides as
# A/B or source/target.  They reference the same immutable mappings.
ROLE_CODEBOOK_A = ROLE_DISJOINT_CODEBOOKS["ROLE_ALIASED"]
ROLE_CODEBOOK_B = ROLE_DISJOINT_CODEBOOKS["ROLE_DISJOINT"]


def canonical_digit_token_id(digit: int) -> int:
    """Return the model token id for an ASCII canonical digit (0 through 9)."""
    if type(digit) is not int or not 0 <= digit <= 9:
        raise ValueError("canonical digit must be an integer in [0, 9]")
    return BYTE_OFFSET + ord(str(digit))


def role_code_token_id(role: str, digit: int, factor: str | None = None) -> int:
    """Return the disjoint code-byte token id for ``role`` and ``digit``."""
    if role not in ROLE_DISJOINT_CODEBOOKS:
        raise ValueError(f"unknown role variant: {role!r}")
    if type(digit) is not int or digit < 0:
        raise ValueError("canonical digit must be a nonnegative integer")
    if role == "ROLE_DISJOINT":
        if factor is None:
            factor = "participant"
        if factor not in ROLE_FACTOR_CODEBOOKS or digit not in ROLE_FACTOR_CODEBOOKS[factor].values():
            raise ValueError("digit is invalid for the selected factor codebook")
        code_byte = next(byte for byte, value in ROLE_FACTOR_CODEBOOKS[factor].items() if value == digit)
    else:
        if digit > 9:
            raise ValueError("canonical digit must be an integer in [0, 9]")
        code_byte = next(byte for byte, value in ROLE_DISJOINT_CODEBOOKS[role].items() if value == digit)
    return BYTE_OFFSET + code_byte


def encode_role_digits(digits: Sequence[int], role: str = "ROLE_DISJOINT",
                       factors: Sequence[str] | None = None) -> Tensor:
    """Encode canonical digits as the selected role's disjoint code bytes."""
    if role not in ROLE_DISJOINT_CODEBOOKS:
        raise ValueError(f"unknown role variant: {role!r}")
    values = list(digits)
    selected = ["participant"] * len(values) if factors is None else list(factors)
    if len(selected) != len(values):
        raise ValueError("factors must match digits length")
    return torch.tensor([role_code_token_id(role, value, factor)
                         for value, factor in zip(values, selected, strict=True)], dtype=torch.long)


def _copy_role_rows(model: BenchmarkV3Model) -> None:
    """Clone every A..T embedding row from its canonical digit row."""
    embedding = model.encoder.embedding.weight
    with torch.no_grad():
        for role in ROLE_VARIANTS:
            for code_byte, digit in ROLE_DISJOINT_CODEBOOKS[role].items():
                embedding[BYTE_OFFSET + code_byte].copy_(embedding[canonical_digit_token_id(digit)])


def build_initialized_model(seed: int = 0) -> BenchmarkV3Model:
    """Build the paired H1L1 model and apply role-row clone initialization."""
    model = build_model("H1L1", seed=seed)
    _copy_role_rows(model)
    return model


def paired_models(seed: int = 0) -> tuple[BenchmarkV3Model, BenchmarkV3Model, dict[str, Any]]:
    """Return two independently constructed, bitwise-identical paired models."""
    left = build_initialized_model(seed)
    right = build_initialized_model(seed)
    left_state, right_state = left.state_dict(), right.state_dict()
    bitwise_equal = tuple(left_state) == tuple(right_state) and all(
        torch.equal(left_state[name], right_state[name]) for name in left_state
    )
    digest = state_sha256(left_state)
    report = {
        "seed": seed,
        "arm": "H1L1",
        "trainable_parameters": left.trainable_parameter_count(),
        "state_sha256": digest,
        "initial_state_sha256": digest,
        "bitwise_equal": bitwise_equal,
        "role_variants": list(ROLE_VARIANTS),
        "code_bytes": "A..T",
    }
    return left, right, report


def corresponding_embedding_sequences(model: BenchmarkV3Model, digits: Sequence[int]) -> dict[str, Tensor]:
    """Return role-code embedding sequences and require canonical alias equality."""
    if not isinstance(model, BenchmarkV3Model):
        raise TypeError("model must be a BenchmarkV3Model")
    encoded = {role: encode_role_digits(digits, role) for role in ROLE_VARIANTS}
    return {role: model.encoder.embedding(tokens) for role, tokens in encoded.items()}


def factor_only_training_step(model: BenchmarkV3Model, optimizer: torch.optim.Optimizer,
                              source_ids: Tensor, factor_targets: Tensor,
                              *, source_mask: Tensor | None = None) -> float:
    """Run one factor-only update using the caller's full-model optimizer.

    The decoder is absent from the computation graph, so its gradients remain
    ``None`` after the update.  The same optimizer parameter set and schedule
    can therefore be used by a paired JOINT/FACTOR_ONLY runner.
    """
    if not isinstance(model, BenchmarkV3Model) or model.arm != "H1L1":
        raise ValueError("factor-only training requires an H1L1 BenchmarkV3Model")
    expected = {id(parameter) for parameter in model.parameters()}
    supplied = {id(parameter) for group in optimizer.param_groups for parameter in group["params"]}
    if supplied != expected:
        raise ValueError("factor-only optimizer must contain every model parameter exactly")
    optimizer.zero_grad(set_to_none=True)
    output = model(source_ids, source_mask=source_mask, factor_targets=factor_targets)
    loss = output["factor_loss"]
    if not torch.isfinite(loss):
        raise ValueError("non-finite factor-only loss")
    loss.backward()
    clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    return float(loss.detach().cpu())


# Small compatibility spellings for protocol/executor code.  They intentionally
# point at the same implementation and do not add a model or parameter branch.
build_role_model = build_initialized_model
factor_only_step = factor_only_training_step


def paired_initialization_preflight(seed: int = 0) -> dict[str, Any]:
    """Return the paired initialization proof without exposing model objects."""
    _left, _right, report = paired_models(seed)
    return report


__all__ = ["ROLE_CODEBOOK_A", "ROLE_CODEBOOK_B", "ROLE_DISJOINT", "ROLE_DISJOINT_CODEBOOKS", "ROLE_DISJOINT_CODE_BYTES", "ROLE_FACTOR_CODEBOOKS",
           "ROLE_VARIANTS", "build_initialized_model", "canonical_digit_token_id",
           "build_role_model", "factor_only_step", "paired_initialization_preflight",
           "corresponding_embedding_sequences", "encode_role_digits",
           "factor_only_training_step", "paired_models", "role_code_token_id", "state_sha256"]
