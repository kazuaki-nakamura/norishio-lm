"""Bounded local conditioning for the duplicate-removed slot control."""
from __future__ import annotations

from typing import Any

import torch
from torch import Tensor, nn

from .concept_model import CONCEPT_FIELDS, ConceptVocabulary, TinyConceptDecoder
from .explicit_slot_model import ExplicitSlotModel


OLD_DIM = 33
SLOT_DIM = 5
HIDDEN_DIM = 32
PREFIXES = ("私は、".encode("utf-8"), "私が".encode("utf-8"))
BYTE_OFFSET = 4


def _kept_indices(vocabulary: ConceptVocabulary) -> tuple[int, ...]:
    fields = CONCEPT_FIELDS
    # The bottleneck probability layout follows CONCEPT_FIELDS, with an
    # additional id-zero class for every field.
    offsets: dict[str, tuple[int, int]] = {}
    at = 0
    for field in fields:
        width = len(vocabulary.fields[field]) + 1
        offsets[field] = (at, at + width)
        at += width
    if at != OLD_DIM:
        raise ValueError(f"expected 33 old concept probabilities, got {at}")
    removed = set(range(*offsets["participant"])) | set(range(*offsets["time"]))
    return tuple(i for i in range(OLD_DIM) if i not in removed)


class DuplicateRemovedProjection(nn.Module):
    """43-input projection retaining the 21 non-slot old probabilities."""

    in_features = 43
    out_features = HIDDEN_DIM

    def __init__(self, old: nn.Linear, kept_old_indices: tuple[int, ...]) -> None:
        super().__init__()
        if old.in_features != 43 or old.out_features != HIDDEN_DIM:
            raise ValueError("expected Linear(43, 32) projection")
        self.kept_old_indices = tuple(kept_old_indices)
        self.base_proj = nn.Linear(len(self.kept_old_indices), HIDDEN_DIM)
        self.participant_proj = nn.Linear(SLOT_DIM, HIDDEN_DIM, bias=False)
        self.time_proj = nn.Linear(SLOT_DIM, HIDDEN_DIM, bias=False)
        with torch.no_grad():
            self.base_proj.weight.copy_(old.weight[:, list(self.kept_old_indices)])
            self.base_proj.bias.copy_(old.bias)
            self.participant_proj.weight.copy_(old.weight[:, OLD_DIM:OLD_DIM + SLOT_DIM])
            self.time_proj.weight.copy_(old.weight[:, OLD_DIM + SLOT_DIM:])

    def parts(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        if x.shape[-1] != self.in_features:
            raise ValueError("duplicate-removed projection expects 43 input features")
        base = self.base_proj(x[..., list(self.kept_old_indices)])
        participant = self.participant_proj(x[..., OLD_DIM:OLD_DIM + SLOT_DIM])
        time = self.time_proj(x[..., OLD_DIM + SLOT_DIM:])
        return base, participant, time

    def forward(self, x: Tensor) -> Tensor:
        return sum(self.parts(x))


class LocalSlotDecoder(TinyConceptDecoder):
    """Decoder with a deterministic, prefix-gated local slot contribution."""

    def __init__(self, *args: Any, local: bool = False, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.local = local

    def _projection_parts(self, probs: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        projection = self.concept_projection
        if not isinstance(projection, DuplicateRemovedProjection):
            raise ValueError("local decoder requires DuplicateRemovedProjection")
        return projection.parts(probs)

    def initial_conditioning(self, concept_probs: Tensor) -> Tensor:
        base, participant, time = self._projection_parts(concept_probs)
        return torch.tanh(base if self.local else base + participant + time)

    @staticmethod
    def _gate_masks(input_ids: Tensor) -> tuple[Tensor, Tensor]:
        """Return [B,T,1] participant/time gates from consumed prefix bytes."""
        bsz, length = input_ids.shape
        device = input_ids.device
        p_gate = torch.zeros((bsz, length, 1), dtype=torch.bool, device=device)
        t_gate = torch.zeros_like(p_gate)
        consumed = input_ids[:, 1:]
        bos = input_ids[:, 0].eq(1)
        valid_tokens = (consumed >= BYTE_OFFSET) & (consumed <= BYTE_OFFSET + 255)
        valid_prior = torch.cat([torch.ones((bsz, 1), dtype=torch.bool, device=device),
                                 valid_tokens], dim=1).cumprod(dim=1).bool()
        positions = torch.arange(length, device=device)
        for prefix in PREFIXES:
            n = len(prefix)
            if consumed.shape[1] < n:
                continue
            encoded = torch.tensor([BYTE_OFFSET + x for x in prefix], device=device)
            match = bos & (consumed[:, :n] == encoded).all(dim=1)
            active = match[:, None] & valid_prior
            t = active & (positions[None, :] >= n) & (positions[None, :] < n + 6)
            p = active & (positions[None, :] >= n + 6) & (positions[None, :] < n + 12)
            t_gate[..., 0] |= t
            p_gate[..., 0] |= p
        return p_gate, t_gate

    def conditioning_for_prefix(self, input_ids: Tensor, concept_probs: Tensor) -> Tensor:
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [B, T]")
        base, participant, time = self._projection_parts(concept_probs)
        if not self.local:
            return torch.tanh(base + participant + time).unsqueeze(1).expand(-1, input_ids.shape[1], -1)
        pg, tg = self._gate_masks(input_ids)
        return torch.tanh(base[:, None, :] + pg * participant[:, None, :] + tg * time[:, None, :])

    def forward(self, input_ids: Tensor, *, pathway: str = "A", encoder_latent: Tensor | None = None,
                concept_probs: Tensor | None = None, labels: Tensor | None = None) -> dict[str, Tensor]:
        if self.local and pathway == "C":
            if concept_probs is None or encoder_latent is not None or self.concept_projection is None:
                raise ValueError("local C requires concept_probs and rejects encoder_latent")
            probs = concept_probs.to(dtype=self.embedding.weight.dtype)
            h0 = self.initial_conditioning(probs).unsqueeze(0)
            x = self.embedding(input_ids) + self.conditioning_for_prefix(input_ids, probs)
            hidden, _ = self.gru(x, h0)
            logits = self.lm_head(hidden)
            result = {"logits": logits}
            if labels is not None:
                valid = labels != -100
                result["lm_loss"] = (torch.nn.functional.cross_entropy(
                    logits.reshape(-1, logits.shape[-1]), labels.reshape(-1), ignore_index=-100)
                    if bool(valid.any()) else logits.sum() * 0)
            return result
        return super().forward(input_ids, pathway=pathway, encoder_latent=encoder_latent,
                              concept_probs=concept_probs, labels=labels)


class DuplicateRemovedModel(ExplicitSlotModel):
    """Explicit slot model with duplicate old participant/time groups removed."""

    def __init__(self, initial: nn.Module, vocabulary: ConceptVocabulary,
                 new_seed: int = 20, *, local: bool = False,
                 gradient_routing: str = "end_to_end") -> None:
        super().__init__(initial, vocabulary, new_seed=new_seed,
                         gradient_routing=gradient_routing)
        old = self.decoder.concept_projection
        if not isinstance(old, nn.Linear):
            raise ValueError("explicit model projection must be linear")
        kept = _kept_indices(vocabulary)
        # Projection constructors draw before their values are copied; keep
        # those irrelevant draws out of the caller's RNG stream.
        with torch.random.fork_rng(devices=[]):
            projection = DuplicateRemovedProjection(old, kept)
        decoder = self.decoder
        decoder.__class__ = LocalSlotDecoder
        decoder.local = local
        decoder.concept_projection = projection
        self.decoder = decoder
        self.kept_old_indices = kept


LocalConceptDecoder = LocalSlotDecoder
DuplicateRemovedDecoder = LocalSlotDecoder

__all__ = ["DuplicateRemovedModel", "DuplicateRemovedProjection", "LocalSlotDecoder",
           "LocalConceptDecoder", "DuplicateRemovedDecoder", "PREFIXES"]
