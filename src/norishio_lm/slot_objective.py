"""Training objective for authored participant/time byte slots."""
from __future__ import annotations

from copy import deepcopy
import math
import time
from typing import Any, Mapping, Sequence

import torch
from torch import Tensor
import torch.nn.functional as F

from .concept_model import ConceptVocabulary, encode_targets
from .toy_controls import state_digest
from .toy_experiment import ToyModel, batch
from .toy_spans import authored_slot_spans
from .tensorizer import SemanticTensorizer


def _span_positions(spans: Sequence[Mapping[str, Mapping[str, int]]], batch_size: int,
                    sequence_length: int, labels: Tensor) -> Tensor:
    if len(spans) != batch_size:
        raise ValueError("one participant/time span mapping is required per batch row")
    selected: list[tuple[int, int]] = []
    for row, item in enumerate(spans):
        if not isinstance(item, Mapping):
            raise ValueError("slot spans must be mappings")
        positions: list[int] = []
        for name in ("participant", "time"):
            value = item.get(name)
            if not isinstance(value, Mapping):
                raise ValueError("participant and time spans are required")
            start, end = value.get("start"), value.get("end")
            if (isinstance(start, bool) or isinstance(end, bool) or
                    not isinstance(start, int) or not isinstance(end, int) or
                    start < 0 or end <= start or end > sequence_length):
                raise ValueError("slot span must be nonempty and within the sequence")
            current = list(range(start, end))
            if set(positions).intersection(current):
                raise ValueError("participant and time spans overlap")
            positions.extend(current)
        for position in positions:
            label = labels[row, position]
            if int(label) < 4 or int(label) > 259:
                raise ValueError("selected slot labels must be byte token IDs 4..259")
            selected.append((row, position))
    if not selected:
        raise ValueError("at least one slot byte is required")
    return torch.tensor(selected, dtype=torch.long, device=labels.device)


def slot_ce(logits: Tensor, labels: Tensor,
            spans: Sequence[Mapping[str, Mapping[str, int]]]) -> Tensor:
    """Mean byte cross-entropy over participant/time span positions."""
    if not isinstance(logits, Tensor) or not isinstance(labels, Tensor):
        raise TypeError("logits and labels must be torch tensors")
    if logits.ndim != 3 or logits.shape[-1] != 260:
        raise ValueError("logits must have shape [B, T, 260]")
    if labels.ndim != 2 or labels.shape != logits.shape[:2]:
        raise ValueError("labels must have shape [B, T]")
    if labels.dtype != torch.long:
        raise ValueError("labels must use torch.long")
    if not bool(torch.isfinite(logits).all()):
        raise ValueError("logits must be finite")
    selected = _span_positions(spans, logits.shape[0], logits.shape[1], labels)
    rows, positions = selected.unbind(dim=1)
    return F.cross_entropy(logits[rows, positions], labels[rows, positions])


def train_slot_control(initial: ToyModel, schedule: Sequence[Sequence[int]],
                       train_rows: list[dict[str, Any]], tensorizer: SemanticTensorizer,
                       vocabulary: ConceptVocabulary, corpus: Any, strict: Any,
                       weight: float = 1.0) -> tuple[ToyModel, dict[str, Any]]:
    """Train a cloned model with the four baseline losses plus weighted slot CE."""
    if not isinstance(weight, (int, float)) or isinstance(weight, bool) or weight < 0 or not math.isfinite(weight):
        raise ValueError("weight must be finite and nonnegative")
    model = deepcopy(initial)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
    seed = corpus.seed_data()
    train_spans = [authored_slot_spans(row["targets"], seed) for row in train_rows]
    if any(span is None for span in train_spans):
        raise ValueError("every training target must have one authored slot span")
    trace: list[dict[str, Any]] = []
    started = time.perf_counter()
    model.train()
    for step, indices in enumerate(schedule):
        rows = [train_rows[i] for i in indices]
        ids, labels, semantic = batch(rows, "C", tensorizer, corpus, strict)
        optimizer.zero_grad(set_to_none=True)
        output, aux = model(ids, semantic, labels=labels)
        targets = encode_targets([r["targets"] for r in rows], vocabulary)
        base = model.decoder.losses(output, targets=targets, bottleneck_output=aux)
        slot = slot_ce(output["logits"], labels, [train_spans[i] for i in indices]) if weight else output["logits"].sum() * 0
        total = base + weight * slot if weight else base
        total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step == 0 or (step + 1) % 10 == 0 or step + 1 == len(schedule):
            byte_count = sum(
                span[field]["end"] - span[field]["start"]
                for i in indices for span in [train_spans[i]] for field in ("participant", "time")
            )
            trace.append({"step": step + 1, "base": float(base.detach()),
                          "slot": float(slot.detach()), "total": float(total.detach()),
                          "byte_count": byte_count})
    return model, {"steps": len(schedule), "weight": float(weight),
                   "initial_state_sha256": state_digest(initial),
                   "parameters": sum(p.numel() for p in model.parameters()),
                   "training_seconds": time.perf_counter() - started, "trace": trace}


__all__ = ["slot_ce", "train_slot_control"]
