"""Byte-level diagnostics for participant/time spans."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch
from torch import Tensor


_FIELDS = ("participant", "time")


def _is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _validate_span_inputs(
    baseline_logits: Tensor,
    logits: Tensor,
    labels: Tensor,
    spans: Sequence[Mapping[str, Any]],
) -> tuple[int, int, int]:
    if not isinstance(baseline_logits, Tensor) or baseline_logits.ndim != 3:
        raise ValueError("baseline_logits must have shape [B, T, V]")
    if not isinstance(logits, Tensor) or logits.ndim != 3:
        raise ValueError("logits must have shape [B, T, V]")
    if tuple(baseline_logits.shape) != tuple(logits.shape):
        raise ValueError("baseline_logits and logits must have matching shapes")
    if not isinstance(labels, Tensor) or labels.ndim != 2:
        raise ValueError("labels must have shape [B, T]")
    batch, steps, vocab = map(int, logits.shape)
    if batch == 0:
        raise ValueError("batch must be nonempty")
    if tuple(labels.shape) != (batch, steps):
        raise ValueError("labels must align with logits as [B, T]")
    if not bool(torch.isfinite(baseline_logits.detach()).all()) or not bool(torch.isfinite(logits.detach()).all()):
        raise ValueError("logits must be finite")
    if labels.dtype.is_floating_point or labels.dtype.is_complex or labels.dtype == torch.bool:
        raise ValueError("labels must be an integer tensor")
    if not isinstance(spans, Sequence) or isinstance(spans, (str, bytes)) or len(spans) != batch:
        raise ValueError("spans must contain one mapping per batch row")
    for row, item in enumerate(spans):
        if not isinstance(item, Mapping) or set(item) != set(_FIELDS):
            raise ValueError(f"spans[{row}] must contain exactly participant and time")
        for field in _FIELDS:
            span = item[field]
            if not isinstance(span, Mapping) or set(span) != {"start", "end"}:
                raise ValueError(f"spans[{row}][{field!r}] must contain start and end")
            start, end = span["start"], span["end"]
            if not _is_int(start) or not _is_int(end) or start < 0 or end < 0 or start > steps or end > steps or start >= end:
                raise ValueError(f"spans[{row}][{field!r}] must be a nonempty half-open range within T")
    return batch, steps, vocab


def span_metrics(
    baseline_logits: Tensor,
    logits: Tensor,
    labels: Tensor,
    spans: Sequence[Mapping[str, Mapping[str, int]]],
) -> dict[str, Any]:
    """Measure intervened-vs-baseline next-byte behavior over two raw-byte spans.

    Decoder position ``j`` is scored against ``labels[..., j]``.  Byte IDs
    themselves are in the reserved range 4..259; there is no position offset.
    """
    batch, _, vocab = _validate_span_inputs(baseline_logits, logits, labels, spans)
    base = baseline_logits.detach().to(device="cpu", dtype=torch.float64)
    current = logits.detach().to(device="cpu", dtype=torch.float64)
    target = labels.detach().to(device="cpu", dtype=torch.long)
    base_logp = torch.log_softmax(base, dim=-1)
    current_logp = torch.log_softmax(current, dim=-1)
    fields: dict[str, dict[str, Any]] = {}
    examples: dict[str, list[list[dict[str, Any]]]] = {field: [] for field in _FIELDS}

    for field in _FIELDS:
        rows: list[list[dict[str, Any]]] = []
        values: list[dict[str, Any]] = []
        correct = 0
        for row in range(batch):
            row_values: list[dict[str, Any]] = []
            start = int(spans[row][field]["start"])
            end = int(spans[row][field]["end"])
            for position in range(start, end):
                model_position = position
                target_id = int(target[row, model_position].item())
                if target_id < 4 or target_id > 259 or target_id >= vocab:
                    raise ValueError("labels in scored spans must be byte IDs 4..259 and within V")
                row_logits = current[row, model_position]
                row_base_logits = base[row, model_position]
                probability = float(current_logp[row, model_position, target_id].exp().item())
                nll = float(-current_logp[row, model_position, target_id].item())
                rank = 1 + int((row_logits > row_logits[target_id]).sum().item())
                argmax_correct = bool(int(row_logits.argmax().item()) == target_id)
                logit_l1 = float((row_logits - row_base_logits).abs().mean().item())
                kl = float((base_logp[row, model_position].exp() * (base_logp[row, model_position] - current_logp[row, model_position])).sum().item())
                item = {
                    "position": position,
                    "target_id": target_id,
                    "probability": probability,
                    "rank": rank,
                    "argmax_correct": argmax_correct,
                    "logit_l1": logit_l1,
                    "kl": kl,
                    "nll": nll,
                }
                row_values.append(item)
                values.append(item)
                correct += int(argmax_correct)
            rows.append(row_values)
        count = len(values)
        fields[field] = {
            "byte_count": count,
            "row_count": batch,
            "mean_nll": sum(item["nll"] for item in values) / count,
            "mean_correct_probability": sum(item["probability"] for item in values) / count,
            "mean_correct_rank": sum(item["rank"] for item in values) / count,
            "correct_argmax": correct,
            "accuracy": correct / count,
            "mean_logit_l1": sum(item["logit_l1"] for item in values) / count,
            "mean_kl": sum(item["kl"] for item in values) / count,
        }
        examples[field] = rows
    result: dict[str, Any] = {field: fields[field] for field in _FIELDS}
    result["examples"] = examples
    return result
