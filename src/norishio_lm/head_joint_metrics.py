"""Auditable metrics for the participant/time categorical heads.

The two heads are evaluated independently; joint probabilities are the outer
product of their probabilities and therefore are an independence diagnostic,
not a learned joint distribution.
"""
from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import torch
from torch import Tensor

from .concept_model import ConceptVocabulary


_FIELDS = ("participant", "time")
_EPS = 1e-12


def _finite_number(value: Tensor) -> bool:
    return bool(torch.isfinite(value).all().item())


def _gold_ids(targets: Sequence[Mapping[str, Any]], vocabulary: ConceptVocabulary) -> tuple[list[int], list[int]]:
    ids: dict[str, list[int]] = {field: [] for field in _FIELDS}
    for row, target in enumerate(targets):
        concept = target.get("concept") if isinstance(target, Mapping) else None
        if not isinstance(concept, Mapping):
            raise ValueError(f"target {row} has no concept")
        for field in _FIELDS:
            value = concept.get(field)
            if value is None:
                raise ValueError(f"target {row} has no known {field} class")
            if field == "operators":
                value = tuple(value)
            class_id = vocabulary.fields[field].get(value)
            if class_id not in range(1, 6):
                raise ValueError(f"target {row} has unknown {field} class {value!r}")
            ids[field].append(int(class_id) - 1)
    return ids["participant"], ids["time"]


def _validate_heads(heads: Tensor, n: int, vocabulary: ConceptVocabulary) -> tuple[Tensor, Tensor]:
    if not isinstance(heads, Tensor) or heads.ndim != 2 or tuple(heads.shape) != (n, 10):
        raise ValueError(f"heads must have shape ({n}, 10)")
    if not heads.is_floating_point() or not _finite_number(heads):
        raise ValueError("heads must contain finite floating-point probabilities")
    if any(set(vocabulary.fields[field].values()) != set(range(1, 6)) for field in _FIELDS):
        raise ValueError("participant and time vocabularies must contain exactly IDs 1..5")
    value = heads.detach().to(device="cpu", dtype=torch.float64)
    if bool((value < 0).any().item()):
        raise ValueError("head probabilities must be nonnegative")
    sums = value.reshape(n, 2, 5).sum(-1)
    if n and not bool(torch.allclose(sums, torch.ones_like(sums), atol=1e-6, rtol=1e-6)):
        raise ValueError("each head probability row must sum to one")
    return value[:, :5], value[:, 5:]


def _head_metrics(prob: Tensor, gold: list[int]) -> dict[str, Any]:
    n = len(gold)
    pred = prob.argmax(1) if n else torch.empty(0, dtype=torch.long)
    correct = pred == torch.tensor(gold, dtype=torch.long)
    confusion = [[0 for _ in range(5)] for _ in range(5)]
    support = [0] * 5
    for g, p in zip(gold, pred.tolist()):
        support[g] += 1
        confusion[g][p] += 1
    recalls = [confusion[i][i] / support[i] for i in range(5) if support[i]]
    brier = None
    nll = None
    if n:
        onehot = torch.zeros_like(prob)
        onehot[torch.arange(n), torch.tensor(gold)] = 1.0
        brier = float(((prob - onehot).square().sum(1)).mean().item())
        nll = float((-prob[torch.arange(n), torch.tensor(gold)].clamp_min(_EPS).log()).mean().item())
    bins = []
    confidence = prob.max(1).values if n else torch.empty(0, dtype=torch.float64)
    for index in range(10):
        lower, upper = index / 10, (index + 1) / 10
        mask = (confidence >= lower) & ((confidence < upper) if index < 9 else (confidence <= 1.0))
        count = int(mask.sum().item())
        bins.append({
            "lower": lower, "upper": upper, "count": count,
            "confidence": float(confidence[mask].mean().item()) if count else None,
            "accuracy": float(correct[mask].double().mean().item()) if count else None,
        })
    ece = (sum(item["count"] / n * abs(item["accuracy"] - item["confidence"])
               for item in bins if item["count"]) if n else None)
    rows = []
    for i, g in enumerate(gold):
        p = int(pred[i])
        vals = prob[i].tolist()
        ordered = sorted(vals, reverse=True)
        rows.append({
            "row": i, "truth_class_id": g + 1, "prediction_class_id": p + 1,
            "truth_column": g, "prediction_column": p,
            "confidence": float(vals[p]), "correct": bool(p == g),
            "margin": float(ordered[0] - ordered[1]),
            "logprob": float(math.log(max(vals[g], _EPS))),
            "entropy": float(-(sum(x * math.log(max(x, _EPS)) for x in vals))),
            "probabilities": [float(x) for x in vals],
        })
    return {
        "accuracy": float(correct.double().mean().item()) if n else None,
        "balanced_accuracy": float(sum(recalls) / len(recalls)) if recalls else None,
        "confusion": confusion, "confusion_matrix": confusion, "support": support,
        "brier": brier, "nll": nll, "ece": {"value": float(ece) if ece is not None else None, "bins": bins}, "raw_rows": rows,
    }


def head_joint_metrics(heads: Tensor, targets: Sequence[Mapping[str, Any]], vocabulary: ConceptVocabulary) -> dict[str, Any]:
    """Return JSON-safe participant/time head and independence-joint metrics."""
    n = len(targets)
    participant_gold, time_gold = _gold_ids(targets, vocabulary)
    participant, time = _validate_heads(heads, n, vocabulary)
    p_metrics = _head_metrics(participant, participant_gold)
    t_metrics = _head_metrics(time, time_gold)
    p_pred = participant.argmax(1) if n else torch.empty(0, dtype=torch.long)
    t_pred = time.argmax(1) if n else torch.empty(0, dtype=torch.long)
    p_ok = p_pred == torch.tensor(participant_gold, dtype=torch.long)
    t_ok = t_pred == torch.tensor(time_gold, dtype=torch.long)
    categories = {
        "correct_both": int((p_ok & t_ok).sum()), "participant_only": int((p_ok & ~t_ok).sum()),
        "time_only": int((~p_ok & t_ok).sum()), "both_wrong": int((~p_ok & ~t_ok).sum()),
    }
    cells = []
    participant_values = {int(class_id): value for value, class_id in vocabulary.fields["participant"].items()}
    time_values = {int(class_id): value for value, class_id in vocabulary.fields["time"].items()}
    for pg in range(5):
        row = []
        for tg in range(5):
            mask = (torch.tensor(participant_gold) == pg) & (torch.tensor(time_gold) == tg)
            indices = mask.nonzero().reshape(-1)
            cell_counts = {name: int((mask & value).sum()) for name, value in {
                "correct_both": p_ok & t_ok, "participant_only": p_ok & ~t_ok,
                "time_only": ~p_ok & t_ok, "both_wrong": ~p_ok & ~t_ok,
            }.items()}
            count = int(mask.sum())
            rates = {name: (value / count if count else None) for name, value in cell_counts.items()}
            row.append({"participant_class_id": pg + 1, "time_class_id": tg + 1,
                        "participant_class_value": participant_values[pg + 1],
                        "time_class_value": time_values[tg + 1],
                        "participant_column": pg, "time_column": tg,
                        "count": count, "categories": cell_counts, "rates": rates,
                        "rows": indices.tolist()})
        cells.append(row)
    rows = []
    for i, (pg, tg) in enumerate(zip(participant_gold, time_gold)):
        product = torch.outer(participant[i], time[i])
        flat = product.reshape(-1)
        order = torch.argsort(flat, descending=True)
        gold_product = float(product[pg, tg])
        p_entropy = float(-(sum(float(x) * math.log(max(float(x), _EPS)) for x in participant[i])))
        t_entropy = float(-(sum(float(x) * math.log(max(float(x), _EPS)) for x in time[i])))
        rows.append({
            "row": i, "participant_truth_class_id": pg + 1, "time_truth_class_id": tg + 1,
            "participant_prediction_class_id": int(p_pred[i]) + 1, "time_prediction_class_id": int(t_pred[i]) + 1,
            "correct_both": bool(p_ok[i] & t_ok[i]), "participant_correct": bool(p_ok[i]), "time_correct": bool(t_ok[i]),
            "gold_joint_logprob": float(math.log(max(float(participant[i, pg]), _EPS))
                                      + math.log(max(float(time[i, tg]), _EPS))),
            "joint_entropy": p_entropy + t_entropy,
            "joint_top_margin": float(flat[order[0]] - flat[order[1]]),
        })
    error_p = (~p_ok).double()
    error_t = (~t_ok).double()
    covariance = float(((error_p - error_p.mean()) * (error_t - error_t.mean())).mean().item()) if n else None
    if n and error_p.std(unbiased=False) > 0 and error_t.std(unbiased=False) > 0:
        correlation = float(torch.corrcoef(torch.stack((error_p, error_t)))[0, 1].item())
    else:
        correlation = None
    exact = {"correct_both": categories["correct_both"], "participant_only": categories["participant_only"],
             "time_only": categories["time_only"], "both_wrong": categories["both_wrong"],
             "correct": categories["correct_both"], "count": n,
             "accuracy": categories["correct_both"] / n if n else None}
    joint = {"exact": exact, "categories": categories, "gold_pair_cells": cells,
                      "gold_pairs": cells, "raw_rows": rows,
                      "error_binary_covariance": covariance, "error_binary_correlation": correlation,
                      "independence_approximation": True}
    return {"participant": p_metrics, "time": t_metrics,
            "per_head": {"participant": p_metrics, "time": t_metrics},
            "joint": joint, "joint_exact": exact, "raw_rows": rows}
