"""Explicit participant/time slot heads for the toy C pathway.

The slot predictions are an auxiliary, source-only interface.  They are
concatenated after the existing 33-way concept bottleneck and are never used
as teacher-forced decoder inputs.
"""
from __future__ import annotations

from copy import deepcopy
from typing import Any, Iterable, Mapping, Sequence

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .concept_model import ConceptTargets, ConceptVocabulary
from .toy_adapter import source_record


_SLOT_FIELDS = ("participant", "time")
_OLD_CONCEPT_DIM = 33
_SLOT_CLASSES = 5
_GRADIENT_ROUTING = {"end_to_end", "stop_slot_lm"}


class ExplicitSlotModel(nn.Module):
    """Pathway C model with explicit participant and time distributions."""

    pathway = "C"

    def __init__(self, initial: nn.Module, vocabulary: ConceptVocabulary,
                 new_seed: int = 20, *, gradient_routing: str = "end_to_end") -> None:
        super().__init__()
        if not hasattr(initial, "encoder") or not hasattr(initial, "decoder"):
            raise TypeError("initial must expose encoder and decoder")
        if getattr(initial, "pathway", None) != "C":
            raise ValueError("initial pathway must be C")
        if getattr(initial.decoder, "conditioning_mode", None) != "per_step_additive":
            raise ValueError("initial decoder conditioning_mode must be per_step_additive")
        if type(new_seed) is not int:
            raise TypeError("new_seed must be an int")
        if gradient_routing not in _GRADIENT_ROUTING:
            raise ValueError(f"gradient_routing must be one of {sorted(_GRADIENT_ROUTING)}")
        for field in _SLOT_FIELDS:
            values = vocabulary.fields.get(field)
            if values is None or set(values.values()) != set(range(1, _SLOT_CLASSES + 1)):
                raise ValueError(f"{field} vocabulary must contain exactly IDs 1..5")

        # Deepcopy is intentional: all pre-existing parameters remain bitwise
        # identical while the caller's initial model is left untouched.
        self.encoder = deepcopy(initial.encoder)
        self.decoder = deepcopy(initial.decoder)
        self.vocabulary = vocabulary
        self.gradient_routing = gradient_routing
        bottleneck = getattr(self.decoder, "bottleneck", None)
        old_projection = getattr(self.decoder, "concept_projection", None)
        if bottleneck is None or old_projection is None:
            raise ValueError("initial must be a C model with a concept bottleneck")
        if getattr(bottleneck, "vocabulary", None) != vocabulary:
            raise ValueError("provided vocabulary must match initial bottleneck vocabulary")
        if old_projection.in_features != _OLD_CONCEPT_DIM:
            raise ValueError(f"expected {_OLD_CONCEPT_DIM} old concept probabilities")
        hidden = old_projection.out_features
        if hidden != 32:
            raise ValueError("explicit slot model requires hidden dimension 32")

        # Keep all initialization draws local to this constructor.  The order
        # is part of the reproducibility contract: heads, then projection.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(new_seed)
            self.slot_heads = nn.ModuleDict({
                "participant": nn.Linear(hidden, _SLOT_CLASSES),
                "time": nn.Linear(hidden, _SLOT_CLASSES),
            })
            expanded = nn.Linear(_OLD_CONCEPT_DIM + 2 * _SLOT_CLASSES, hidden)
            with torch.no_grad():
                expanded.weight[:, :_OLD_CONCEPT_DIM].copy_(old_projection.weight)
                expanded.bias.copy_(old_projection.bias)
            self.decoder.concept_projection = expanded

    def forward(self, ids: Tensor, semantic: Any, *,
                labels: Tensor | None = None) -> tuple[dict[str, Tensor], Any, dict[str, Tensor]]:
        latent = self.encoder(semantic).fused
        auxiliary = self.decoder.bottleneck(latent)
        old_probs = auxiliary.probabilities()
        slot_logits = {field: self.slot_heads[field](latent) for field in _SLOT_FIELDS}
        slot_probs = [F.softmax(slot_logits[field], dim=-1) for field in _SLOT_FIELDS]
        decoder_slot_probs = [value.detach() if self.gradient_routing == "stop_slot_lm" else value
                              for value in slot_probs]
        probabilities = torch.cat([old_probs, *decoder_slot_probs], dim=-1)
        output = self.decoder.decode_with_concept_intervention(
            ids, probabilities, labels=labels
        )
        return output, auxiliary, slot_logits


@torch.no_grad()
def source_distributions(model: ExplicitSlotModel,
                         sources: Sequence[Mapping[str, str]] | Iterable[Mapping[str, str]],
                         tensorizer: Any) -> tuple[Tensor, Tensor]:
    """Return old and explicit slot probabilities for source-only inputs.

    Inputs are adapted in batches of at most 16 through the strict
    ``source_record`` allowlist.  Gold labels or metadata therefore fail at
    the boundary instead of entering inference accidentally.
    """
    source_list = list(sources)
    if not source_list:
        raise ValueError("sources must not be empty")
    old_parts: list[Tensor] = []
    slot_parts: list[Tensor] = []
    model.eval()
    for start in range(0, len(source_list), 16):
        records = [source_record(source) for source in source_list[start:start + 16]]
        semantic = tensorizer.encode(records)
        latent = model.encoder(semantic).fused
        auxiliary = model.decoder.bottleneck(latent)
        slot_logits = {field: model.slot_heads[field](latent) for field in _SLOT_FIELDS}
        old_parts.append(auxiliary.probabilities())
        slot_parts.append(torch.cat([F.softmax(slot_logits[f], dim=-1)
                                     for f in _SLOT_FIELDS], dim=-1))
    return torch.cat(old_parts, dim=0), torch.cat(slot_parts, dim=0)


def _target_ids(targets: Any, field: str, vocabulary: ConceptVocabulary) -> Tensor:
    if isinstance(targets, ConceptTargets):
        values = targets.fields[field]
        mask = targets.masks.get(field)
        if (mask is None or mask.shape != values.shape or not bool(mask.all())
                or values.ndim != 1 or (values < 1).any() or (values > _SLOT_CLASSES).any()):
            raise ValueError(f"{field} targets contain missing or unknown IDs")
        return values - 1
    if isinstance(targets, Mapping):
        targets = [targets]
    try:
        rows = list(targets)
    except TypeError as exc:
        raise TypeError("targets must be ConceptTargets or a sequence of mappings") from exc
    ids: list[int] = []
    value_to_id = vocabulary.fields[field]
    for target in rows:
        if not isinstance(target, Mapping) or not isinstance(target.get("concept"), Mapping):
            raise ValueError(f"missing {field} target")
        value = target["concept"].get(field)
        if value is None or value not in value_to_id:
            raise ValueError(f"unknown or missing {field} target")
        encoded = value_to_id[value]
        if encoded not in range(1, _SLOT_CLASSES + 1):
            raise ValueError(f"unknown {field} target ID")
        ids.append(encoded - 1)
    if not ids:
        raise ValueError("targets must not be empty")
    return torch.tensor(ids, dtype=torch.long)


def slot_head_loss(slot_logits: Mapping[str, Tensor], targets: Any,
                   vocabulary: ConceptVocabulary) -> Tensor:
    """Sum mean cross-entropies for participant and time heads."""
    losses = []
    for field in _SLOT_FIELDS:
        if field not in slot_logits:
            raise ValueError(f"missing {field} slot logits")
        logits = slot_logits[field]
        if logits.ndim != 2 or logits.shape[1] != _SLOT_CLASSES:
            raise ValueError(f"{field} logits must have shape [B, 5]")
        ids = _target_ids(targets, field, vocabulary).to(logits.device)
        if ids.shape[0] != logits.shape[0]:
            raise ValueError("slot logits and targets batch sizes differ")
        losses.append(F.cross_entropy(logits, ids))
    return sum(losses)


__all__ = ["ExplicitSlotModel", "slot_head_loss", "source_distributions"]
