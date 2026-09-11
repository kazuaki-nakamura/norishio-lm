"""A small linear probe for frozen encoder latents.

The probe is deliberately separate from the encoder and only consumes detached
latents.  Its seven categorical heads correspond to the concept schema; a
head's class zero is reserved for the unknown/padding class.
"""
from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from typing import Any

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .concept_model import CONCEPT_FIELDS, ConceptVocabulary, encode_targets


class ToyProbe(nn.Module):
    """Seven independent linear categorical heads over standardized latents."""

    def __init__(self, input_dim: int, vocabulary: ConceptVocabulary) -> None:
        super().__init__()
        if input_dim <= 0:
            raise ValueError("input_dim must be positive")
        self.input_dim = int(input_dim)
        self.vocabulary = vocabulary
        self.field_heads = nn.ModuleDict(
            {field: nn.Linear(input_dim, len(vocabulary.fields[field]) + 1)
             for field in CONCEPT_FIELDS}
        )
        self.register_buffer("train_mean", torch.zeros(input_dim))
        self.register_buffer("train_std", torch.ones(input_dim))

    def normalize(self, latents: Tensor) -> Tensor:
        """Apply the training-only affine transform to new latent rows."""
        if latents.ndim != 2 or latents.shape[1] != self.input_dim:
            raise ValueError(f"latents must have shape [N, {self.input_dim}]")
        return (latents - self.train_mean) / self.train_std

    def forward(self, latents: Tensor) -> dict[str, Tensor]:
        standardized = self.normalize(latents)
        return {field: self.field_heads[field](standardized) for field in CONCEPT_FIELDS}


def _masked_field_loss(
    logits: Mapping[str, Tensor], targets: Sequence[Mapping[str, Any]], vocabulary: ConceptVocabulary,
) -> Tensor:
    encoded = encode_targets(targets, vocabulary)
    losses: list[Tensor] = []
    for field in CONCEPT_FIELDS:
        mask = encoded.masks[field].to(device=logits[field].device)
        if mask.any():
            ids = encoded.fields[field].to(device=logits[field].device)
            losses.append(F.cross_entropy(logits[field][mask], ids[mask]))
    if losses:
        return torch.stack(losses).mean()
    # Keep the zero attached to the graph so optimizer/backward remains valid.
    return sum((value.sum() for value in logits.values()), torch.zeros((), device=next(iter(logits.values())).device)) * 0


def train_probe(
    train_latents: Tensor,
    train_targets: Sequence[Mapping[str, Any]],
    vocabulary: ConceptVocabulary,
    *,
    seed: int = 7,
    steps: int = 300,
    learning_rate: float = 0.01,
) -> tuple[ToyProbe, dict[str, Any]]:
    """Fit a full-batch linear probe using training labels only.

    The encoder latent tensor is detached before it enters the optimization;
    no encoder/model reference is accepted by this API.
    """
    if not isinstance(train_latents, Tensor):
        raise TypeError("train_latents must be a torch.Tensor")
    if train_latents.dtype != torch.float32 or train_latents.device.type != "cpu":
        raise ValueError("train_latents must be a CPU float32 tensor")
    if not torch.isfinite(train_latents).all():
        raise ValueError("train_latents must contain only finite values")
    if train_latents.ndim != 2:
        raise ValueError("train_latents must have shape [N, D]")
    if len(train_targets) != train_latents.shape[0]:
        raise ValueError("train_latents and train_targets must have equal row counts")
    if not isinstance(steps, int) or isinstance(steps, bool) or steps < 0:
        raise ValueError("steps must be a nonnegative integer")
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ValueError("seed must be an integer")
    if learning_rate <= 0 or not torch.isfinite(torch.tensor(learning_rate)):
        raise ValueError("learning_rate must be finite and positive")

    started = time.perf_counter()
    frozen_latents = train_latents.detach()
    if frozen_latents.shape[0] == 0:
        raise ValueError("train_latents must contain at least one row")
    mean = frozen_latents.mean(dim=0)
    std = frozen_latents.std(dim=0, unbiased=False).clamp_min(1e-6)

    torch.manual_seed(seed)
    probe = ToyProbe(frozen_latents.shape[1], vocabulary)
    probe.train_mean.copy_(mean)
    probe.train_std.copy_(std)
    optimizer = torch.optim.Adam(probe.parameters(), lr=learning_rate)
    normalized = probe.normalize(frozen_latents)
    # Keep the training path explicit: normalized latents are detached and are
    # never optimizer parameters.
    normalized = normalized.detach()

    def loss_at_step() -> Tensor:
        logits = {field: probe.field_heads[field](normalized) for field in CONCEPT_FIELDS}
        return _masked_field_loss(logits, train_targets, vocabulary)

    with torch.no_grad():
        first_loss = float(loss_at_step().cpu())
    final_loss = first_loss
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        loss = loss_at_step()
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        final_loss = float(loss_at_step().cpu())
    probe.eval()

    metadata: dict[str, Any] = {
        "architecture": {
            "name": "linear_7_head",
            "heads": len(CONCEPT_FIELDS),
            "hidden_layers": 0,
            "capacity": "fixed_linear",
            "parameter_count": sum(parameter.numel() for parameter in probe.parameters()),
        },
        "budget": {"steps": steps, "learning_rate": learning_rate, "elapsed_seconds": time.perf_counter() - started,
                   "optimizer": "Adam", "full_batch": True},
        "train_normalization": {
            "mean": mean.detach().cpu().tolist(),
            "std": std.detach().cpu().tolist(),
            "std_unbiased": False,
            "std_floor": 1e-6,
        },
        "loss": {"first": first_loss, "final": final_loss},
        "input": {"detached": True, "shape": list(train_latents.shape), "dtype": str(train_latents.dtype)},
        "seed": seed,
        "fields": list(CONCEPT_FIELDS),
    }
    return probe, metadata
