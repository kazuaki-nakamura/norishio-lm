"""Joint participant/time pair head over the local duplicate-removed model."""
from __future__ import annotations

from typing import Any, Iterable, Mapping, Sequence

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .concept_model import ConceptTargets, ConceptVocabulary
from .local_slot_model import DuplicateRemovedModel
from .toy_adapter import source_record

PAIR_CLASSES = 5
PAIR_DIM = 25


def pair_marginals(pair_probabilities: Tensor) -> Tensor:
    """Validate joint probabilities and return participant/time marginals."""
    if not isinstance(pair_probabilities, Tensor) or pair_probabilities.ndim != 2 \
            or pair_probabilities.shape[1] != PAIR_DIM:
        raise ValueError("pair probabilities must have shape [B, 25]")
    if not pair_probabilities.is_floating_point():
        raise ValueError("pair probabilities must be floating-point")
    if not bool(torch.isfinite(pair_probabilities).all()) or bool((pair_probabilities < 0).any()):
        raise ValueError("pair probabilities must be finite and non-negative")
    sums = pair_probabilities.sum(dim=-1)
    if not bool(torch.allclose(sums, torch.ones_like(sums), atol=1e-5, rtol=1e-5)):
        raise ValueError("pair probabilities must be normalized")
    matrix = pair_probabilities.reshape(len(pair_probabilities), PAIR_CLASSES, PAIR_CLASSES)
    return torch.cat([matrix.sum(dim=-1), matrix.sum(dim=-2)], dim=-1)


def _pair_target_ids(targets: Any, vocabulary: ConceptVocabulary) -> Tensor:
    if isinstance(targets, ConceptTargets):
        p, t = targets.fields["participant"], targets.fields["time"]
        pm, tm = targets.masks.get("participant"), targets.masks.get("time")
        if (p.ndim != 1 or t.ndim != 1 or p.shape != t.shape or pm is None or tm is None
                or pm.shape != p.shape or tm.shape != t.shape or not bool(pm.all())
                or not bool(tm.all()) or (p < 1).any() or (p > 5).any()
                or (t < 1).any() or (t > 5).any()):
            raise ValueError("pair targets contain missing or unknown IDs")
        return (p - 1) * PAIR_CLASSES + (t - 1)
    if isinstance(targets, Mapping):
        targets = [targets]
    try:
        rows = list(targets)
    except TypeError as exc:
        raise TypeError("targets must be ConceptTargets or a sequence of mappings") from exc
    if not rows:
        raise ValueError("targets must not be empty")
    ids: list[int] = []
    for target in rows:
        if not isinstance(target, Mapping) or not isinstance(target.get("concept"), Mapping):
            raise ValueError("missing pair target")
        concept = target["concept"]
        p, t = concept.get("participant"), concept.get("time")
        if p not in vocabulary.fields["participant"] or t not in vocabulary.fields["time"]:
            raise ValueError("unknown or missing pair target")
        pi, ti = vocabulary.fields["participant"][p], vocabulary.fields["time"][t]
        if pi not in range(1, 6) or ti not in range(1, 6):
            raise ValueError("unknown pair target ID")
        ids.append((pi - 1) * PAIR_CLASSES + ti - 1)
    return torch.tensor(ids, dtype=torch.long)


class PairSlotModel(DuplicateRemovedModel):
    """Local model with a joint 5-by-5 participant/time prediction head."""

    def __init__(self, initial: nn.Module, vocabulary: ConceptVocabulary,
                 new_seed: int = 20, *, pair_seed: int = 28) -> None:
        super().__init__(initial, vocabulary, new_seed=new_seed, local=True)
        if type(pair_seed) is not int:
            raise TypeError("pair_seed must be an int")
        self.pair_seed = pair_seed
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(pair_seed)
            self.pair_head = nn.Linear(32, PAIR_DIM)

    def forward(self, ids: Tensor, semantic: Any, *, labels: Tensor | None = None
                ) -> tuple[dict[str, Tensor], Any, dict[str, Tensor], Tensor]:
        latent = self.encoder(semantic).fused
        auxiliary = self.decoder.bottleneck(latent)
        old_probs = auxiliary.probabilities()
        independent_logits = {
            field: self.slot_heads[field](latent) for field in ("participant", "time")
        }
        pair_logits = self.pair_head(latent)
        pair_probs = F.softmax(pair_logits, dim=-1)
        probabilities = torch.cat([old_probs, pair_marginals(pair_probs)], dim=-1)
        output = self.decoder.decode_with_concept_intervention(ids, probabilities, labels=labels)
        return output, auxiliary, independent_logits, pair_logits


def pair_slot_head_loss(pair_logits: Tensor, targets: Any,
                        vocabulary: ConceptVocabulary) -> Tensor:
    if pair_logits.ndim != 2 or pair_logits.shape[1] != PAIR_DIM:
        raise ValueError("pair logits must have shape [B, 25]")
    ids = _pair_target_ids(targets, vocabulary).to(pair_logits.device)
    if ids.shape[0] != pair_logits.shape[0]:
        raise ValueError("pair logits and targets batch sizes differ")
    return F.cross_entropy(pair_logits, ids)


# Public name used by the experiment objective specification.
pair_head_loss = pair_slot_head_loss


def source_pair_distributions(model: PairSlotModel,
                              sources: Sequence[Mapping[str, str]] | Iterable[Mapping[str, str]],
                              tensorizer: Any) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Return old, independent, joint, and joint-marginal distributions."""
    source_list = list(sources)
    if not source_list:
        raise ValueError("sources must not be empty")
    old_parts: list[Tensor] = []
    independent_parts: list[Tensor] = []
    pair_parts: list[Tensor] = []
    marginal_parts: list[Tensor] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(source_list), 16):
            records = [source_record(source) for source in source_list[start:start + 16]]
            latent = model.encoder(tensorizer.encode(records)).fused
            auxiliary = model.decoder.bottleneck(latent)
            independent_logits = {f: model.slot_heads[f](latent) for f in ("participant", "time")}
            independent = torch.cat([F.softmax(independent_logits[f], dim=-1)
                                     for f in ("participant", "time")], dim=-1)
            pair = F.softmax(model.pair_head(latent), dim=-1)
            old_parts.append(auxiliary.probabilities())
            independent_parts.append(independent)
            pair_parts.append(pair)
            marginal_parts.append(pair_marginals(pair))
    return (torch.cat(old_parts), torch.cat(independent_parts),
            torch.cat(pair_parts), torch.cat(marginal_parts))


__all__ = ["PAIR_CLASSES", "PAIR_DIM", "PairSlotModel", "pair_marginals",
           "pair_slot_head_loss", "pair_head_loss", "source_pair_distributions"]
