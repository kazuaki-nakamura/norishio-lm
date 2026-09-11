"""Greedy generation conditioned on a byte-token prefix.

The prefix is observed context: it is consumed by the decoder but never
included in the generated-decision diagnostics.  Once the prefix has been
consumed, decoding is strictly self-feeding and uses only tokens emitted by
the decoder itself.
"""
from __future__ import annotations

from typing import TypedDict

import torch
from torch import Tensor

from .concept_model import TinyConceptDecoder


BOS = 1
EOS = 2
BYTE_OFFSET = 4
VOCAB_SIZE = 260


class PrefixGenerationResult(TypedDict):
    """Diagnostics for one prefix-conditioned generation row."""

    token_ids: list[int]
    generated_token_ids: list[int]
    prefix_length: int
    decision_logits: Tensor
    ended_eos: bool


def _validate_prefixes(prefixes: list[list[int]], batch_size: int, cap: int) -> None:
    if not isinstance(prefixes, list) or len(prefixes) != batch_size:
        raise ValueError("prefixes must be a list with one list per concept row")
    for row in prefixes:
        if not isinstance(row, list):
            raise ValueError("each prefix must be a list of byte token IDs")
        if len(row) > cap:
            raise ValueError("prefix length cannot exceed max_total_tokens")
        for token in row:
            if isinstance(token, bool) or not isinstance(token, int) or not BYTE_OFFSET <= token < VOCAB_SIZE:
                raise ValueError("prefix token IDs must be integers in the byte range 4..259")


@torch.no_grad()
def prefix_generate(
    decoder: TinyConceptDecoder,
    concept_probs: Tensor,
    prefixes: list[list[int]],
    max_total_tokens: int = 128,
) -> list[PrefixGenerationResult]:
    """Greedily decode after consuming each row's byte-token prefix.

    ``max_total_tokens`` counts both observed prefix tokens and generated
    tokens.  The returned logits contain one ``[260]`` row per generated
    decision, including the decision that emits EOS when one is produced.
    """
    if not isinstance(decoder, TinyConceptDecoder):
        raise TypeError("decoder must be a TinyConceptDecoder")
    if not isinstance(max_total_tokens, int) or isinstance(max_total_tokens, bool) or max_total_tokens < 0:
        raise ValueError("max_total_tokens must be a nonnegative integer")
    if not isinstance(concept_probs, Tensor) or concept_probs.ndim != 2:
        raise ValueError("concept_probs must have shape [B, D]")
    if decoder.concept_projection is None:
        raise ValueError("decoder must have a concept projection for pathway C")
    if decoder.embedding.num_embeddings != VOCAB_SIZE:
        raise ValueError("prefix generation requires a 260-token decoder vocabulary")
    expected_dim = decoder.concept_projection.in_features
    if concept_probs.shape[1] != expected_dim:
        raise ValueError(f"concept_probs has dimension {concept_probs.shape[1]}, expected {expected_dim}")
    _validate_prefixes(prefixes, int(concept_probs.shape[0]), max_total_tokens)
    if concept_probs.shape[0] == 0:
        return []

    parameter = next(decoder.parameters())
    if concept_probs.device != parameter.device:
        raise ValueError("concept_probs and decoder must be on the same device")
    probs = concept_probs.to(dtype=parameter.dtype)
    results: list[PrefixGenerationResult] = []

    for row_index, prefix in enumerate(prefixes):
        # Full-prefix intervention is the reference for the first generated
        # decision, and also ensures custom decoder implementations observe
        # the public intervention API during warmup.
        warmup_ids = torch.tensor([[BOS, *prefix]], dtype=torch.long, device=parameter.device)
        warmup = decoder.decode_with_concept_intervention(warmup_ids, probs[row_index:row_index + 1])
        next_logits = warmup["logits"][0, -1]

        # Recover the same hidden state for cheap one-token GRU continuation.
        conditioning = torch.tanh(decoder.concept_projection(probs[row_index:row_index + 1]))
        hidden = conditioning.unsqueeze(0)
        warmup_embeddings = decoder.embedding(warmup_ids)
        if decoder.conditioning_mode == "per_step_additive":
            warmup_embeddings = warmup_embeddings + conditioning.unsqueeze(1)
        _, hidden = decoder.gru(warmup_embeddings, hidden)
        generated: list[int] = []
        decisions: list[Tensor] = []
        remaining = max_total_tokens - len(prefix)
        for step in range(remaining):
            decisions.append(next_logits.detach().clone())
            token = int(next_logits.argmax())
            generated.append(token)
            if token == EOS:
                break
            if step + 1 >= remaining:
                break
            step_input = decoder.embedding(torch.tensor([[token]], dtype=torch.long, device=parameter.device))
            if decoder.conditioning_mode == "per_step_additive":
                step_input = step_input + conditioning.unsqueeze(1)
            output, hidden = decoder.gru(step_input, hidden)
            next_logits = decoder.lm_head(output[:, -1, :])[0]

        logits = torch.stack(decisions, dim=0) if decisions else torch.empty((0, VOCAB_SIZE), dtype=parameter.dtype, device=parameter.device)
        results.append({
            "token_ids": [*prefix, *generated],
            "generated_token_ids": generated,
            "prefix_length": len(prefix),
            "decision_logits": logits,
            "ended_eos": bool(generated and generated[-1] == EOS),
        })
    return results


__all__ = ["PrefixGenerationResult", "prefix_generate"]
