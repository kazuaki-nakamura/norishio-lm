"""Metrics for a categorical participant/time pair head.

The pair vocabulary is the Cartesian product of the participant and time
classes in the supplied (train-fitted) :class:`ConceptVocabulary`.  Pair
index ``p * 5 + t`` uses zero-based columns while public class IDs remain
one-based.  Ties are resolved by the lowest pair index (the same rule as
``torch.argmax``), and that policy is included in the returned report.
"""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import torch
from torch import Tensor

from .concept_model import ConceptVocabulary


_FIELDS = ("participant", "time")
_CLASSES = 5
_PAIR_COUNT = 25
_EPS = 1e-12
_TIE_POLICY = "lowest_pair_index"


def _concept(row: Mapping[str, Any], index: int) -> Mapping[str, Any]:
    value: Any = row.get("concept") if isinstance(row, Mapping) else None
    if value is None and isinstance(row, Mapping) and isinstance(row.get("targets"), Mapping):
        value = row["targets"].get("concept")
    if not isinstance(value, Mapping):
        raise ValueError(f"target {index} has no concept mapping")
    return value


def _class_values(vocabulary: ConceptVocabulary, field: str) -> list[Any]:
    values = vocabulary.fields.get(field)
    if not isinstance(values, Mapping) or set(values.values()) != set(range(1, _CLASSES + 1)):
        raise ValueError(f"{field} vocabulary must contain exactly class IDs 1..5")
    return [value for value, _ in sorted(values.items(), key=lambda item: item[1])]


def _gold_pairs(
    targets: Sequence[Mapping[str, Any]], vocabulary: ConceptVocabulary
) -> tuple[list[int], list[tuple[Any, Any]]]:
    participant = {value: int(class_id) - 1 for value, class_id in vocabulary.fields["participant"].items()}
    times = {value: int(class_id) - 1 for value, class_id in vocabulary.fields["time"].items()}
    ids: list[int] = []
    pairs: list[tuple[Any, Any]] = []
    for row, target in enumerate(targets):
        concept = _concept(target, row)
        p_value, t_value = concept.get("participant"), concept.get("time")
        if p_value not in participant:
            raise ValueError(f"target {row} has unknown participant class {p_value!r}")
        if t_value not in times:
            raise ValueError(f"target {row} has unknown time class {t_value!r}")
        p, t = participant[p_value], times[t_value]
        ids.append(p * _CLASSES + t)
        pairs.append((p_value, t_value))
    return ids, pairs


def _probabilities(value: Tensor | Sequence[Sequence[float]], n: int) -> Tensor:
    if isinstance(value, Tensor):
        if value.ndim != 2 or tuple(value.shape) != (n, _PAIR_COUNT):
            raise ValueError(f"pair probabilities must have shape ({n}, 25)")
        if not value.is_floating_point():
            raise ValueError("pair probabilities must be floating-point")
        probs = value.detach().to(device="cpu", dtype=torch.float64)
    else:
        try:
            probs = torch.as_tensor(value, dtype=torch.float64)
        except (TypeError, ValueError, RuntimeError) as exc:
            raise ValueError("pair probabilities must be a rectangular numeric matrix") from exc
        if probs.ndim != 2 or tuple(probs.shape) != (n, _PAIR_COUNT):
            raise ValueError(f"pair probabilities must have shape ({n}, 25)")
    if not bool(torch.isfinite(probs).all().item()):
        raise ValueError("pair probabilities must contain only finite values")
    if bool((probs < 0).any().item()):
        raise ValueError("pair probabilities must be nonnegative")
    if n and not bool(torch.allclose(probs.sum(-1), torch.ones(n, dtype=probs.dtype), atol=1e-6, rtol=1e-6)):
        raise ValueError("each pair probability row must sum to one")
    return probs


def _ece(probs: Tensor, correct: Tensor) -> dict[str, Any]:
    n = probs.shape[0]
    confidence = probs.max(-1).values if n else torch.empty(0, dtype=torch.float64)
    bins: list[dict[str, Any]] = []
    for index in range(10):
        lower, upper = index / 10, (index + 1) / 10
        mask = (confidence >= lower) & ((confidence < upper) if index < 9 else (confidence <= 1.0))
        count = int(mask.sum().item())
        bins.append({
            "lower": lower, "upper": upper, "count": count,
            "confidence": float(confidence[mask].mean().item()) if count else None,
            "accuracy": float(correct[mask].double().mean().item()) if count else None,
        })
    value = sum(item["count"] / n * abs(item["accuracy"] - item["confidence"])
                for item in bins if item["count"]) if n else None
    return {"value": float(value) if value is not None else None, "bins": bins}


def _score_group(probs: Tensor, gold: list[int], row_ids: list[int]) -> dict[str, Any]:
    n = len(gold)
    if n:
        index = torch.tensor(row_ids, dtype=torch.long)
        selected = probs[index]
        gold_tensor = torch.tensor(gold, dtype=torch.long)
        prediction = selected.argmax(-1)
        correct = prediction == gold_tensor
    else:
        selected = torch.empty((0, _PAIR_COUNT), dtype=torch.float64)
        prediction = torch.empty(0, dtype=torch.long)
        correct = torch.empty(0, dtype=torch.bool)
    confusion = [[0 for _ in range(_PAIR_COUNT)] for _ in range(_PAIR_COUNT)]
    support = [0] * _PAIR_COUNT
    for truth, predicted in zip(gold, prediction.tolist()):
        support[truth] += 1
        confusion[truth][predicted] += 1
    rows: list[dict[str, Any]] = []
    for local, (truth, predicted) in enumerate(zip(gold, prediction.tolist())):
        values = selected[local].tolist()
        # Sort by (-probability, index), making ties reproducible and explicit.
        order = sorted(range(_PAIR_COUNT), key=lambda column: (-values[column], column))
        rows.append({
            "row": row_ids[local], "gold_pair_index": truth,
            "gold_pair_class_id": truth + 1, "prediction_pair_index": predicted,
            "prediction_pair_class_id": predicted + 1,
            "gold_probability": float(values[truth]),
            "gold_rank": order.index(truth) + 1,
            "confidence": float(values[predicted]), "correct": bool(predicted == truth),
            "probabilities": [float(value) for value in values],
        })
    if n:
        one_hot = torch.zeros_like(selected)
        one_hot[torch.arange(n), torch.tensor(gold)] = 1.0
        gold_probability = selected[torch.arange(n), torch.tensor(gold)].clamp_min(_EPS)
        brier = float((selected - one_hot).square().sum(-1).mean().item())
        nll = float(-gold_probability.log().mean().item())
        accuracy = float(correct.double().mean().item())
    else:
        brier = nll = accuracy = None
    return {
        "count": n, "accuracy": accuracy, "nll": nll, "brier": brier,
        "mean_gold_probability": sum(row["gold_probability"] for row in rows) / n if n else None,
        "mean_gold_rank": sum(row["gold_rank"] for row in rows) / n if n else None,
        "ece": _ece(selected, correct), "confusion": confusion,
        "confusion_matrix": confusion, "support": support, "rows": rows,
        "tie_policy": _TIE_POLICY,
    }


def pair_head_metrics(
    probabilities: Tensor | Sequence[Sequence[float]],
    targets: Sequence[Mapping[str, Any]],
    vocabulary: ConceptVocabulary,
    train_support: Sequence[Sequence[int]] | Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Score 25-way participant/time probabilities and train-pair splits.

    ``train_support`` is normally the train-derived 5x5 count table.  For
    compatibility with callers that retain annotations, a sequence of train
    target mappings is also accepted.  It is used only to mark each evaluated
    gold pair as ``train_seen`` or ``train_unseen``.
    """
    n = len(targets)
    participant_values = _class_values(vocabulary, "participant")
    time_values = _class_values(vocabulary, "time")
    gold, pairs = _gold_pairs(targets, vocabulary)
    probs = _probabilities(probabilities, n)
    if train_support is None:
        seen: set[int] = set()
    elif train_support and isinstance(train_support[0], Mapping):
        train_gold, _ = _gold_pairs(train_support, vocabulary)  # type: ignore[arg-type]
        seen = set(train_gold)
    else:
        try:
            table = [list(row) for row in train_support]  # type: ignore[arg-type]
        except TypeError as exc:
            raise ValueError("train_support must be a 5x5 count table") from exc
        if len(table) != _CLASSES or any(len(row) != _CLASSES for row in table):
            raise ValueError("train_support must be a 5x5 count table")
        if any((not isinstance(count, int)) or count < 0 for row in table for count in row):
            raise ValueError("train_support counts must be nonnegative integers")
        seen = {p * _CLASSES + t for p, row in enumerate(table) for t, count in enumerate(row) if count}
    row_groups = {
        "all": list(range(n)),
        "train_seen": [row for row, pair in enumerate(gold) if pair in seen],
        "train_unseen": [row for row, pair in enumerate(gold) if pair not in seen],
    }
    groups = {
        name: _score_group(probs, [gold[row] for row in rows], rows)
        for name, rows in row_groups.items()
    }
    return {
        "pair_count": _PAIR_COUNT, "class_count": _CLASSES,
        "pair_order": "participant_class_id_then_time_class_id",
        "class_values": {"participant": participant_values, "time": time_values},
        "tie_policy": _TIE_POLICY, "train_pair_support": sorted(seen),
        "train_seen_pairs": [
            {"pair_index": pair, "participant": participant_values[pair // _CLASSES],
             "time": time_values[pair % _CLASSES]} for pair in sorted(seen)
        ],
        "groups": groups,
        "all": groups["all"], "train_seen": groups["train_seen"],
        "train_unseen": groups["train_unseen"],
    }


__all__ = ["pair_head_metrics"]
