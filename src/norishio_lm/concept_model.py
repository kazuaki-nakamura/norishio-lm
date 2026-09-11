"""Small, explicit concept bottleneck components for the issue-3 toy task.

The schema in this module is a categorical annotation schema.  It deliberately
does not infer meaning from glyph decomposition.  ``None`` is represented by a
mask and is never converted to a negative class.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

import torch
from torch import Tensor, nn
import torch.nn.functional as F


CONCEPT_FIELDS = ("event", "operators", "agent", "participant", "time", "location", "repeat_marked")


def _key(value: Any) -> str:
    return repr(value)


@dataclass(frozen=True)
class ConceptVocabulary:
    """Vocabularies fit from training annotations only; id zero is reserved."""
    fields: Mapping[str, Mapping[Any, int]]
    senses: Mapping[str, int]
    sememes: Mapping[str, int]

    @classmethod
    def fit(cls, train_targets: Iterable[Mapping[str, Any]]) -> "ConceptVocabulary":
        vals = {f: set() for f in CONCEPT_FIELDS}
        senses: set[str] = set()
        sememes: set[str] = set()
        for target in train_targets:
            concept = target.get("concept")
            if isinstance(concept, Mapping):
                for field in CONCEPT_FIELDS:
                    value = concept.get(field)
                    if value is not None:
                        if field == "operators": value = tuple(value)
                        vals[field].add(value)
            if target.get("sense") is not None: senses.add(str(target["sense"]))
            if target.get("sememes") is not None: sememes.update(map(str, target["sememes"]))
        fields = {f: {v: i + 1 for i, v in enumerate(sorted(values, key=_key))} for f, values in vals.items()}
        return cls(fields, {v: i + 1 for i, v in enumerate(sorted(senses))},
                   {v: i for i, v in enumerate(sorted(sememes))})

    def inspect(self, values: Mapping[str, Tensor] | None = None) -> dict[str, Any]:
        """Return named categorical values (or logits' argmax values)."""
        if values is None: return {"concept": {f: dict(v) for f, v in self.fields.items()}, "sense": dict(self.senses), "sememes": dict(self.sememes)}
        out: dict[str, Any] = {}
        for name, logits in values.items():
            if name in self.fields:
                inv = {i: v for v, i in self.fields[name].items()}
                picks = [inv.get(int(x)) for x in logits.argmax(-1).reshape(-1)]
                out[name] = picks[0] if len(picks) == 1 else picks
            elif name == "sense":
                inv = {i: v for v, i in self.senses.items()}
                picks = [inv.get(int(x)) for x in logits.argmax(-1).reshape(-1)]
                out[name] = picks[0] if len(picks) == 1 else picks
            elif name == "sememes":
                rows = [[v for v, i in self.sememes.items() if bool(logits[j, i].sigmoid() >= .5)]
                        for j in range(logits.shape[0])]
                out[name] = rows[0] if len(rows) == 1 else rows
        return out


@dataclass
class ConceptTargets:
    fields: dict[str, Tensor]
    masks: dict[str, Tensor]
    sense: Tensor | None = None
    sense_mask: Tensor | None = None
    sememes: Tensor | None = None
    sememe_mask: Tensor | None = None


def encode_targets(targets: Sequence[Mapping[str, Any]], vocab: ConceptVocabulary) -> ConceptTargets:
    n = len(targets)
    device = torch.device("cpu")
    fields: dict[str, Tensor] = {}
    masks: dict[str, Tensor] = {}
    for field in CONCEPT_FIELDS:
        ids = torch.zeros(n, dtype=torch.long, device=device)
        mask = torch.zeros(n, dtype=torch.bool, device=device)
        for i, target in enumerate(targets):
            c = target.get("concept")
            value = c.get(field) if isinstance(c, Mapping) else None
            if value is not None:
                value = tuple(value) if field == "operators" else value
                if value in vocab.fields[field]:
                    ids[i] = vocab.fields[field][value]
                    mask[i] = True
        fields[field] = ids
        masks[field] = mask
    sense = torch.zeros(n, dtype=torch.long)
    sm = torch.zeros(n, dtype=torch.bool)
    sem = torch.zeros(n, len(vocab.sememes))
    semm = torch.zeros(n, dtype=torch.bool)
    for i, t in enumerate(targets):
        if t.get("sense") in vocab.senses:
            sense[i] = vocab.senses[t["sense"]]
            sm[i] = True
        if t.get("sememes") is not None:
            semm[i] = True
            for s in t["sememes"]:
                if str(s) in vocab.sememes: sem[i, vocab.sememes[str(s)]] = 1
    return ConceptTargets(fields, masks, sense, sm, sem, semm)


@dataclass
class BottleneckOutput:
    concept_logits: dict[str, Tensor]
    sense_logits: Tensor
    sememe_logits: Tensor

    def inspect(self, vocab: ConceptVocabulary) -> dict[str, Any]:
        return vocab.inspect({**self.concept_logits, "sense": self.sense_logits, "sememes": self.sememe_logits})

    def probabilities(self) -> Tensor:
        """Concatenate soft categorical probabilities for the C decoder."""
        return torch.cat([F.softmax(self.concept_logits[f], dim=-1) for f in CONCEPT_FIELDS], dim=-1)


class ConceptBottleneck(nn.Module):
    def __init__(self, hidden_dim: int = 32, vocabulary: ConceptVocabulary | None = None) -> None:
        super().__init__()
        self.vocabulary = vocabulary
        if vocabulary is None: raise ValueError("vocabulary fitted on training targets is required")
        self.field_heads = nn.ModuleDict({f: nn.Linear(hidden_dim, len(vocabulary.fields[f]) + 1) for f in CONCEPT_FIELDS})
        self.sense_head = nn.Linear(hidden_dim, len(vocabulary.senses) + 1)
        self.sememe_head = nn.Linear(hidden_dim, len(vocabulary.sememes))

    def forward(self, latent: Tensor) -> BottleneckOutput:
        return BottleneckOutput({f: head(latent) for f, head in self.field_heads.items()}, self.sense_head(latent), self.sememe_head(latent))

    def loss(self, output: BottleneckOutput, targets: ConceptTargets) -> dict[str, Tensor]:
        losses: dict[str, Tensor] = {}
        for f, logits in output.concept_logits.items():
            m = targets.masks[f]
            zero = logits.sum() * 0
            losses[f] = F.cross_entropy(logits[m], targets.fields[f][m]) if m.any() else zero
        m = targets.sense_mask
        zero = output.sense_logits.sum() * 0
        losses["sense"] = F.cross_entropy(output.sense_logits[m], targets.sense[m]) if m is not None and m.any() else zero
        m = targets.sememe_mask
        zero = output.sememe_logits.sum() * 0
        losses["sememes"] = (F.binary_cross_entropy_with_logits(output.sememe_logits[m], targets.sememes[m])
                              if m is not None and m.any() and output.sememe_logits.shape[-1] else zero)
        losses["concept"] = torch.stack([losses[f] for f in CONCEPT_FIELDS]).mean()
        return losses


class TinyConceptDecoder(nn.Module):
    """CPU GRU decoder with explicit A/B/C pathways and no C source bypass."""
    def __init__(self, vocab_size: int = 260, hidden_dim: int = 32,
                 bottleneck: ConceptBottleneck | None = None,
                 *, conditioning_mode: str = "initial_only") -> None:
        super().__init__()
        if conditioning_mode not in {"initial_only", "per_step_additive"}:
            raise ValueError("conditioning_mode must be initial_only or per_step_additive")
        self.conditioning_mode = conditioning_mode
        self.hidden_dim = hidden_dim
        self.embedding = nn.Embedding(vocab_size, hidden_dim, padding_idx=0)
        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.lm_head = nn.Linear(hidden_dim, vocab_size)
        self.source_projection = nn.Linear(hidden_dim, hidden_dim)
        self.bottleneck = bottleneck
        concept_dim = (sum(len(v) + 1 for v in bottleneck.vocabulary.fields.values())
                       if bottleneck is not None else 0)
        self.concept_projection = nn.Linear(concept_dim, hidden_dim) if concept_dim else None

    def forward(self, input_ids: Tensor, *, pathway: str = "A", encoder_latent: Tensor | None = None,
                concept_probs: Tensor | None = None, labels: Tensor | None = None) -> dict[str, Tensor]:
        if pathway not in {"A", "B", "C"}:
            raise ValueError("pathway must be A, B, or C")
        x = self.embedding(input_ids)
        h0 = None
        if pathway == "B":
            if encoder_latent is None or concept_probs is not None:
                raise ValueError("B requires encoder_latent and rejects concept_probs")
            h0 = torch.tanh(self.source_projection(encoder_latent)).unsqueeze(0)
        elif pathway == "C":
            if concept_probs is None or self.concept_projection is None or encoder_latent is not None:
                raise ValueError("C requires concept_probs and rejects encoder_latent")
            h0 = torch.tanh(self.concept_projection(concept_probs)).unsqueeze(0)
        elif encoder_latent is not None or concept_probs is not None:
            raise ValueError("A rejects encoder_latent and concept_probs")
        if self.conditioning_mode == "per_step_additive":
            if pathway != "C" or h0 is None:
                raise ValueError("per_step_additive conditioning requires pathway C")
            x = x + h0.squeeze(0).unsqueeze(1)
        hidden, _ = self.gru(x, h0)
        logits = self.lm_head(hidden)
        result = {"logits": logits}
        if labels is not None:
            valid = labels != -100
            result["lm_loss"] = (F.cross_entropy(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1), ignore_index=-100)
                                  if valid.any() else logits.sum() * 0)
        return result

    def decode_with_concept_intervention(self, input_ids: Tensor, concept_probs: Tensor, **kwargs: Any) -> dict[str, Tensor]:
        return self.forward(input_ids, pathway="C", concept_probs=concept_probs, **kwargs)

    def losses(self, output: Mapping[str, Tensor], *, targets: ConceptTargets | None = None,
               bottleneck_output: BottleneckOutput | None = None, weights: Mapping[str, float] | None = None) -> Tensor:
        weights = {"lm": 1., "sense": 1., "sememe": 1., "concept": 1., **(weights or {})}
        if set(weights) - {"lm", "sense", "sememe", "concept"}:
            raise ValueError("unknown loss weight")
        if any(not isinstance(value, (int, float)) or value < 0 or not torch.isfinite(torch.tensor(value))
               for value in weights.values()):
            raise ValueError("loss weights must be finite and nonnegative")
        total = output.get("lm_loss", next(iter(output.values())).sum() * 0) * weights["lm"]
        if targets is not None and bottleneck_output is not None and self.bottleneck is not None:
            aux = self.bottleneck.loss(bottleneck_output, targets)
            total = total + weights["concept"] * aux["concept"] + weights["sense"] * aux["sense"] + weights["sememe"] * aux["sememes"]
        return total
