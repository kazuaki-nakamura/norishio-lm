"""Factorized projection variant of the explicit participant/time slot model."""
from __future__ import annotations

from typing import Any

import torch
from torch import Tensor, nn

from .concept_model import ConceptVocabulary
from .explicit_slot_model import ExplicitSlotModel
from .toy_experiment import ToyModel


class FactorizedProjection(nn.Module):
    """Three additive linear blocks over the 33+5+5 input layout.

    The blocks are copied from an existing 43-to-32 projection.  This keeps
    the initial function algebraically equivalent while making each slot
    contribution independently inspectable.
    """

    in_features = 43
    out_features = 32

    def __init__(self, shared: nn.Linear) -> None:
        super().__init__()
        if not isinstance(shared, nn.Linear):
            raise TypeError("shared must be an nn.Linear")
        if (shared.in_features, shared.out_features) != (self.in_features, self.out_features):
            raise ValueError("shared projection must be Linear(43, 32)")
        # Linear constructors draw initial values even though every value is
        # immediately replaced. Keep those irrelevant draws out of the caller's
        # RNG stream as the explicit-slot constructor does.
        with torch.random.fork_rng(devices=[]):
            self.base_proj = nn.Linear(33, 32, bias=True)
            self.participant_proj = nn.Linear(5, 32, bias=False)
            self.time_proj = nn.Linear(5, 32, bias=False)
            with torch.no_grad():
                self.base_proj.weight.copy_(shared.weight[:, :33])
                self.base_proj.bias.copy_(shared.bias)
                self.participant_proj.weight.copy_(shared.weight[:, 33:38])
                self.time_proj.weight.copy_(shared.weight[:, 38:43])

    def forward(self, x: Tensor) -> Tensor:
        if x.shape[-1] != self.in_features:
            raise ValueError("factorized projection expects 43 input features")
        return (self.base_proj(x[..., :33])
                + self.participant_proj(x[..., 33:38])
                + self.time_proj(x[..., 38:43]))


class FactorizedSlotModel(ExplicitSlotModel):
    """Explicit slot model whose shared projection is split into three blocks."""

    def __init__(self, initial: ToyModel, vocabulary: ConceptVocabulary,
                 new_seed: int = 20) -> None:
        super().__init__(initial, vocabulary, new_seed=new_seed)
        shared = self.decoder.concept_projection
        if not isinstance(shared, nn.Linear):
            raise ValueError("explicit model projection must be linear")
        self.decoder.concept_projection = FactorizedProjection(shared)


__all__ = ["FactorizedProjection", "FactorizedSlotModel"]
