"""Factorized pair probabilities and fixed unseen-pair diagnostics."""
from __future__ import annotations

from typing import Any, Mapping, Sequence

import torch
from torch import Tensor
import torch.nn.functional as F

from .concept_model import ConceptVocabulary
from .pair_head_metrics import pair_head_metrics


def factorized_pairs(heads: Tensor) -> Tensor:
    """Outer product of normalized participant5/time5; no new parameters."""
    if not isinstance(heads, Tensor) or heads.ndim != 2 or heads.shape[1] != 10:
        raise ValueError("heads must have shape [B,10]")
    if not heads.is_floating_point() or not bool(torch.isfinite(heads).all()) or bool((heads < 0).any()):
        raise ValueError("heads must be finite floating nonnegative probabilities")
    sums = heads.reshape(len(heads), 2, 5).sum(-1)
    if not torch.allclose(sums, torch.ones_like(sums), atol=1e-6, rtol=1e-6):
        raise ValueError("each head must sum to one")
    return (heads[:, :5, None] * heads[:, None, 5:]).reshape(len(heads), 25)


def factorized_log_probs(participant_logits: Tensor, time_logits: Tensor) -> Tensor:
    """Stable log q(p,t)=log p(p)+log p(t), including extreme logits."""
    if participant_logits.ndim != 2 or participant_logits.shape[1] != 5 or participant_logits.shape != time_logits.shape:
        raise ValueError("both logits must have matching [B,5] shape")
    if any(not x.is_floating_point() or not bool(torch.isfinite(x).all()) for x in (participant_logits, time_logits)):
        raise ValueError("logits must be finite floating tensors")
    return (F.log_softmax(participant_logits, -1)[:, :, None]
            + F.log_softmax(time_logits, -1)[:, None, :]).reshape(len(participant_logits), 25)


def pair_space_metrics(probabilities: Tensor, targets: Sequence[Mapping[str, Any]],
                       vocabulary: ConceptVocabulary, train_support: list[list[int]]) -> dict[str, Any]:
    """Fixed top-k, entropy and train-unseen mass, after standard validation."""
    result = pair_head_metrics(probabilities, targets, vocabulary, train_support)
    probs = probabilities.detach().cpu().double()
    unseen = torch.tensor([n == 0 for row in train_support for n in row], dtype=torch.bool)
    entropy = -(probs * probs.clamp_min(1e-12).log()).sum(-1)
    mass = probs[:, unseen].sum(-1)
    for name, group in result["groups"].items():
        rows = group["rows"]
        n = len(rows)
        indices = [r["row"] for r in rows]
        group["top_k"] = {str(k): {"correct": sum(r["gold_rank"] <= k for r in rows), "count": n,
            "coverage": sum(r["gold_rank"] <= k for r in rows) / n if n else None} for k in (1, 3, 5, 10)}
        group["entropy_mean"] = float(entropy[indices].mean()) if n else None
        group["train_unseen_mass_mean"] = float(mass[indices].mean()) if n else None
        for row in rows:
            row["entropy"] = float(entropy[row["row"]])
            row["train_unseen_mass"] = float(mass[row["row"]])
    result["entropy_units"] = "nats; probability log floor 1e-12"
    result["unseen_definition"] = "all pairs absent from train; no test labels used"
    result["unseen_pair_count"] = int(unseen.sum())
    return result
