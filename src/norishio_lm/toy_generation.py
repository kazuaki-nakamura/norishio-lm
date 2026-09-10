"""Greedy free running generation for the strict C concept pathway.

The decoder is seeded only with ``concept_probs``.  In particular, this API has
no source rows, labels, or gold targets, and it performs no sampling.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from .concept_model import TinyConceptDecoder


BOS = 1
EOS = 2
BYTE_OFFSET = 4
_RESERVED_SPECIALS = frozenset((0, BOS, 3))


@dataclass(frozen=True)
class GenerationResult:
    """One generated sequence and diagnostics for the toy byte vocabulary."""

    token_ids: list[int]
    text: str
    ended_eos: bool
    valid_utf8: bool
    invalid_special_tokens: list[int]


@torch.no_grad()
def greedy_generate(
    decoder: TinyConceptDecoder,
    concept_probs: Tensor,
    max_new_tokens: int = 128,
) -> list[GenerationResult]:
    """Generate one sequence per row from concept probabilities using argmax.

    This is the strict C pathway: the initial GRU state comes from the
    decoder's concept projection and each next-token decision consumes only the
    previously emitted token.  Reserved IDs are deliberately left unmasked so
    their occurrences can be reported in ``invalid_special_tokens``.
    """
    if not isinstance(decoder, TinyConceptDecoder):
        raise TypeError("decoder must be a TinyConceptDecoder")
    if not isinstance(max_new_tokens, int) or isinstance(max_new_tokens, bool):
        raise ValueError("max_new_tokens must be an integer")
    if concept_probs.ndim != 2:
        raise ValueError("concept_probs must have shape [B, D]")
    if concept_probs.shape[0] == 0:
        return []
    if max_new_tokens < 0:
        raise ValueError("max_new_tokens must be nonnegative")
    if decoder.concept_projection is None:
        raise ValueError("decoder must have a concept projection for pathway C")
    expected_dim = decoder.concept_projection.in_features
    if concept_probs.shape[1] != expected_dim:
        raise ValueError(f"concept_probs has dimension {concept_probs.shape[1]}, expected {expected_dim}")

    parameter = next(decoder.parameters())
    if concept_probs.device != parameter.device:
        raise ValueError("concept_probs and decoder must be on the same device")
    # The model's projection defines the appropriate computation dtype/device.
    probs = concept_probs.to(dtype=parameter.dtype)
    hidden = torch.tanh(decoder.concept_projection(probs)).unsqueeze(0)
    batch_size = concept_probs.shape[0]
    current = torch.full((batch_size,), BOS, dtype=torch.long, device=parameter.device)
    sequences: list[list[int]] = [[] for _ in range(batch_size)]
    finished = torch.zeros(batch_size, dtype=torch.bool, device=parameter.device)

    for _ in range(max_new_tokens):
        step_input = decoder.embedding(current).unsqueeze(1)
        next_output, next_hidden = decoder.gru(step_input, hidden)
        # Once a row has ended, retain its state and exclude it from all output.
        hidden = torch.where(
            finished.view(1, batch_size, 1), hidden, next_hidden
        )
        logits = decoder.lm_head(next_output[:, 0, :])
        next_tokens = logits.argmax(dim=-1)
        active = ~finished
        for index in active.nonzero(as_tuple=False).flatten().tolist():
            token = int(next_tokens[index])
            sequences[index].append(token)
            if token == EOS:
                finished[index] = True
        # The value for finished rows is immaterial, but BOS avoids propagating
        # an arbitrary reserved output if the batch continues for other rows.
        current = torch.where(finished, torch.full_like(current, BOS), next_tokens)
        if bool(finished.all()):
            break

    results: list[GenerationResult] = []
    for sequence in sequences:
        byte_values: list[int] = []
        invalid_special: list[int] = []
        valid = True
        ended = EOS in sequence and sequence.index(EOS) == len(sequence) - 1
        for token in sequence:
            if token == EOS:
                break
            if token in _RESERVED_SPECIALS:
                invalid_special.append(token)
            elif BYTE_OFFSET <= token <= BYTE_OFFSET + 255:
                byte_values.append(token - BYTE_OFFSET)
            else:
                # Out-of-range vocabulary entries cannot represent a byte.
                valid = False
        try:
            text = bytes(byte_values).decode("utf-8", errors="strict")
        except UnicodeDecodeError:
            text = bytes(byte_values).decode("utf-8", errors="replace")
            valid = False
        results.append(GenerationResult(sequence, text, ended, valid, invalid_special))
    return results


__all__ = ["GenerationResult", "greedy_generate"]
