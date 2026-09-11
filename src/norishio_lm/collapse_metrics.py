"""Small, JSON-safe diagnostics for representation and concept-head collapse."""
from __future__ import annotations

import math
from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any

import torch
from torch import Tensor

from .concept_metrics import concept_metrics
from .concept_model import CONCEPT_FIELDS, ConceptVocabulary


def _number(value: Tensor | float) -> float:
    return float(value.item() if isinstance(value, Tensor) else value)


def representation_stats(values: Tensor, *, tolerance: float = 1e-6) -> dict[str, Any]:
    """Return bounded diversity and per-dimension statistics for ``[N, D]`` values."""
    if not isinstance(values, Tensor) or values.ndim != 2:
        raise ValueError("values must be a two-dimensional tensor")
    if not math.isfinite(float(tolerance)) or tolerance < 0:
        raise ValueError("tolerance must be finite and nonnegative")
    x = values.detach().to(device="cpu", dtype=torch.float64)
    if not bool(torch.isfinite(x).all()):
        raise ValueError("values must be finite")
    n, d = map(int, x.shape)
    unique = int(torch.unique(x, dim=0).shape[0]) if n else 0
    pair_count = n * (n - 1) // 2
    if pair_count:
        distances = torch.pdist(x, p=2)
        pairwise: dict[str, Any] = {
            "min": _number(distances.min()), "mean": _number(distances.mean()),
            "max": _number(distances.max()),
            "count_le_tolerance": int((distances <= tolerance).sum()),
            "pair_count": pair_count,
            "tolerance": float(tolerance),
        }
    else:
        pairwise = {"min": None, "mean": None, "max": None,
                    "count_le_tolerance": 0, "pair_count": pair_count,
                    "tolerance": float(tolerance)}
    mean = x.mean(dim=0) if n else torch.zeros(d, dtype=torch.float64)
    variance = x.var(dim=0, unbiased=False) if n else torch.zeros(d, dtype=torch.float64)
    return {"rows": n, "dim": d, "exact_unique_rows": unique,
            "pairwise_l2": pairwise,
            "per_dim_mean": [_number(v) for v in mean] if n else [None] * d,
            "per_dim_population_variance": [_number(v) for v in variance] if n else [None] * d}


def _target_value(target: Mapping[str, Any], field: str) -> Any:
    concept = target.get("concept")
    value = concept.get(field) if isinstance(concept, Mapping) else None
    return tuple(value) if field == "operators" and value is not None else value


def _validate_permutation(permutation: Sequence[int] | Tensor, n: int) -> list[int]:
    values = permutation.detach().cpu().tolist() if isinstance(permutation, Tensor) else list(permutation)
    if len(values) != n or any(type(value) is not int for value in values) or sorted(values) != list(range(n)):
        raise ValueError("permutation must be a bijection over probability rows")
    return values


def concept_head_stats(probabilities: Tensor, targets: Sequence[Mapping[str, Any]],
                       vocabulary: ConceptVocabulary,
                       permutation: Sequence[int] | Tensor) -> dict[str, Any]:
    """Summarize categorical probabilities, predictions, labels, and row permutation effects."""
    if not isinstance(probabilities, Tensor) or probabilities.ndim != 2:
        raise ValueError("probabilities must be a two-dimensional tensor")
    n = len(targets)
    expected = sum(len(vocabulary.fields[field]) + 1 for field in CONCEPT_FIELDS)
    if tuple(probabilities.shape) != (n, expected):
        raise ValueError(f"probabilities must have shape ({n}, {expected})")
    p = probabilities.detach().to(device="cpu", dtype=torch.float64)
    if not bool(torch.isfinite(p).all()) or bool((p < 0).any()):
        raise ValueError("probabilities must be finite and nonnegative")
    perm = _validate_permutation(permutation, n)
    fields: dict[str, Any] = {}
    predictions: dict[str, Tensor] = {}
    chunks: dict[str, Tensor] = {}
    offset = 0
    for field in CONCEPT_FIELDS:
        width = len(vocabulary.fields[field]) + 1
        chunk = p[:, offset:offset + width]
        if not bool(torch.allclose(chunk.sum(dim=1), torch.ones(n, dtype=chunk.dtype), atol=1e-5, rtol=1e-5)):
            raise ValueError(f"probabilities[{field!r}] rows must sum to one")
        chunks[field] = chunk
        predictions[field] = chunk.argmax(dim=1).to(dtype=torch.long)
        offset += width
    for field in CONCEPT_FIELDS:
        mapping = vocabulary.fields[field]
        inverse = {int(class_id): value for value, class_id in mapping.items()}
        known = []
        for target in targets:
            value = _target_value(target, field)
            known.append(value in mapping if value is not None else False)
        matrix = [[0 for _ in range(len(mapping) + 1)] for _ in range(len(mapping))]
        for i, target in enumerate(targets):
            value = _target_value(target, field)
            if value not in mapping:
                continue
            matrix[int(mapping[value]) - 1][int(predictions[field][i])] += 1
        chunk = chunks[field]
        entropy = -(chunk * torch.where(chunk > 0, chunk.log(), torch.zeros_like(chunk))).sum(dim=1)
        variance = chunk.var(dim=0, unbiased=False) if n else torch.zeros(chunk.shape[1], dtype=chunk.dtype)
        paired_l1 = ((chunk - chunk[perm]).abs().sum(dim=1).mean().item() if n else None)
        labels = [list(inverse[i]) if isinstance(inverse[i], tuple) else inverse[i]
                  for i in range(1, len(mapping) + 1)]
        fields[field] = {
            "gold_labels": labels, "confusion_matrix": matrix,
            "support": int(sum(known)),
            "class_support": [sum(matrix[row]) for row in range(len(mapping))],
            "prediction_column_ids": list(range(len(mapping) + 1)),
            "prediction_entropy_mean": _number(entropy.mean()) if n else None,
            "prediction_entropy_variance": _number(entropy.var(unbiased=False)) if n else None,
            "probability_variance": [_number(v) for v in variance] if n else [None] * chunk.shape[1],
            "permutation_mean_l1": float(paired_l1) if paired_l1 is not None else None,
        }
    joint = Counter(tuple(int(predictions[field][i]) for field in CONCEPT_FIELDS) for i in range(n))
    joint_rows = [{"pattern": list(pattern), "count": int(count)} for pattern, count in sorted(joint.items())]
    pred_stats = concept_metrics(predictions, targets, vocabulary)
    return {"fields": fields, "concept_metrics": pred_stats,
            "argmax_joint_pattern_count": len(joint_rows),
            "argmax_joint_patterns": joint_rows,
            "fullsoft_representation_stats": representation_stats(p),
            "predictions": {field: predictions[field].tolist() for field in CONCEPT_FIELDS}}
