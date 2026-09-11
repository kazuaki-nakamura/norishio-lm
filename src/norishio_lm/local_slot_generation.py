"""Greedy generation for the bounded local slot decoder."""
from __future__ import annotations

import torch
from torch import Tensor

from .local_slot_model import LocalSlotDecoder
from .toy_generation import BOS, EOS, BYTE_OFFSET, GenerationResult, greedy_generate


def _results(sequences: list[list[int]]) -> list[GenerationResult]:
    out: list[GenerationResult] = []
    for sequence in sequences:
        values: list[int] = []
        invalid: list[int] = []
        valid = True
        ended = EOS in sequence and sequence[-1] == EOS
        for token in sequence:
            if token == EOS:
                break
            if token in (0, BOS, 3):
                invalid.append(token)
            elif BYTE_OFFSET <= token <= BYTE_OFFSET + 255:
                values.append(token - BYTE_OFFSET)
            else:
                valid = False
        try:
            text = bytes(values).decode("utf-8", errors="strict")
        except UnicodeDecodeError:
            text = bytes(values).decode("utf-8", errors="replace")
            valid = False
        out.append(GenerationResult(sequence, text, ended, valid, invalid))
    return out


@torch.no_grad()
def greedy_generate_local(decoder: LocalSlotDecoder, concept_probs: Tensor,
                          max_new_tokens: int = 128) -> list[GenerationResult]:
    """Generate using the exact local prefix rule and greedy argmax decoding."""
    if not isinstance(decoder, LocalSlotDecoder):
        raise TypeError("decoder must be a LocalSlotDecoder")
    if not isinstance(max_new_tokens, int) or isinstance(max_new_tokens, bool) or max_new_tokens < 0:
        raise ValueError("max_new_tokens must be a nonnegative integer")
    if not decoder.local:
        return greedy_generate(decoder, concept_probs, max_new_tokens=max_new_tokens)
    if concept_probs.ndim != 2:
        raise ValueError("concept_probs must have shape [B, D]")
    if concept_probs.shape[0] == 0:
        return []
    parameter = next(decoder.parameters())
    if concept_probs.device != parameter.device:
        raise ValueError("concept_probs and decoder must be on the same device")
    expected = decoder.concept_projection.in_features if decoder.concept_projection is not None else -1
    if concept_probs.shape[1] != expected:
        raise ValueError(f"concept_probs has dimension {concept_probs.shape[1]}, expected {expected}")
    probs = concept_probs.to(dtype=parameter.dtype)
    batch = concept_probs.shape[0]
    sequences: list[list[int]] = [[] for _ in range(batch)]
    finished = torch.zeros(batch, dtype=torch.bool, device=parameter.device)
    for _ in range(max_new_tokens):
        max_len = max((len(s) for s in sequences), default=0)
        ids = torch.full((batch, max_len + 1), BOS, dtype=torch.long, device=parameter.device)
        for row, seq in enumerate(sequences):
            if seq:
                ids[row, 1:1 + len(seq)] = torch.tensor(seq, device=parameter.device)
        logits = decoder(ids, pathway="C", concept_probs=probs)["logits"][:, -1]
        next_tokens = logits.argmax(-1)
        active = ~finished
        for row in active.nonzero(as_tuple=False).flatten().tolist():
            token = int(next_tokens[row])
            sequences[row].append(token)
            if token == EOS:
                finished[row] = True
        if bool(finished.all()):
            break
    return _results(sequences)


__all__ = ["greedy_generate_local"]
