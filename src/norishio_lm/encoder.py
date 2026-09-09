"""Small independent feature encoders with inspectable, masked gated fusion.

This module returns latent vectors, never inferred lexical labels. Import it
explicitly after installing the optional model dependency.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch
from torch import nn

from .tensorizer import CHANNELS, SemanticBatch


@dataclass(frozen=True)
class EncoderConfig:
    vocab_sizes: Mapping[str, int]
    embedding_dim: int = 16
    hidden_dim: int = 32

    def __post_init__(self) -> None:
        if set(self.vocab_sizes) != set(CHANNELS):
            raise ValueError("vocab_sizes must contain exactly CHANNELS")
        if any(type(n) is not int or n < 2 for n in self.vocab_sizes.values()):
            raise ValueError("vocabulary sizes must include PAD and UNK")
        if any(type(n) is not int or n < 1 for n in (self.embedding_dim, self.hidden_dim)):
            raise ValueError("embedding_dim and hidden_dim must be positive integers")
        object.__setattr__(self, "vocab_sizes", dict(self.vocab_sizes))


@dataclass(frozen=True)
class EncoderState:
    fused: torch.Tensor
    channel_states: torch.Tensor
    gates: torch.Tensor
    channel_mask: torch.Tensor


class FeatureEncoder(nn.Module):
    """Bag of feature embeddings; order and candidate linkage are metadata only."""

    def __init__(self, vocab_size: int, embedding_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.projection = nn.Linear(embedding_dim, hidden_dim)

    def forward(self, ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        weights = mask.unsqueeze(-1)
        features = self.embedding(ids.masked_fill(~mask, 0))
        pooled = (features * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1)
        return torch.tanh(self.projection(pooled)) * mask.any(dim=1, keepdim=True)


class MultiChannelEncoder(nn.Module):
    """Learn a gate per channel with no path from disabled features to fusion."""

    def __init__(self, config: EncoderConfig) -> None:
        super().__init__()
        self.config = EncoderConfig(config.vocab_sizes, config.embedding_dim, config.hidden_dim)
        self.encoders = nn.ModuleDict({name: FeatureEncoder(
            self.config.vocab_sizes[name], config.embedding_dim, config.hidden_dim)
            for name in CHANNELS})
        self.gate_heads = nn.ModuleDict({name: nn.Linear(config.hidden_dim, 1)
                                        for name in CHANNELS})

    def forward(self, batch: SemanticBatch, *,
                ablation: Mapping[str, bool] | None = None) -> EncoderState:
        switches = {} if ablation is None else dict(ablation)
        if set(switches) - set(CHANNELS) or any(type(v) is not bool for v in switches.values()):
            raise ValueError("ablation must map channel names to booleans; False disables")
        if set(batch.channels) != set(CHANNELS):
            raise ValueError("batch must contain exactly CHANNELS")
        states, present, logits = [], [], []
        batch_size = None
        device = next(self.parameters()).device
        for name in CHANNELS:
            channel = batch.channels[name]
            ids, mask = channel.ids, channel.mask
            if (ids.dtype != torch.long or mask.dtype != torch.bool or ids.ndim != 2
                    or mask.shape != ids.shape or ids.shape[0] < 1 or ids.shape[1] < 1):
                raise ValueError(f"{name}: expected nonempty long ids and bool mask [B,L]")
            if batch_size is None:
                batch_size = ids.shape[0]
            if ids.shape[0] != batch_size or ids.device != device or mask.device != device:
                raise ValueError(f"{name}: batch sizes and model/tensor devices must match")
            if ((ids < 0) | (ids >= self.config.vocab_sizes[name])).any():
                raise ValueError(f"{name}: feature ID outside configured vocabulary")
            if ((ids == 0) & mask).any():
                raise ValueError(f"{name}: PAD cannot be marked present")
            effective = mask & switches.get(name, True)
            state = self.encoders[name](ids, effective)
            states.append(state)
            present.append(effective.any(dim=1))
            logits.append(self.gate_heads[name](state).squeeze(-1))
        channel_states = torch.stack(states, dim=1)
        channel_mask = torch.stack(present, dim=1)
        scores = torch.stack(logits, dim=1)
        # Finite masking also defines all-disabled rows without softmax NaNs.
        scores = scores.masked_fill(~channel_mask, torch.finfo(scores.dtype).min)
        gates = torch.softmax(scores, dim=1) * channel_mask
        gates = gates / gates.sum(dim=1, keepdim=True).clamp_min(torch.finfo(gates.dtype).tiny)
        fused = (channel_states * gates.unsqueeze(-1)).sum(dim=1)
        return EncoderState(fused, channel_states, gates, channel_mask)
