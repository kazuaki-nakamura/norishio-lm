"""Evaluation helpers for the seven categorical concept channels."""
from __future__ import annotations

from collections import Counter
from typing import Any, Mapping, Sequence

import torch
from torch import Tensor

from .concept_model import CONCEPT_FIELDS, ConceptVocabulary, encode_targets


def _value(field: str, target: Mapping[str, Any]) -> Any:
    concept = target.get("concept")
    if not isinstance(concept, Mapping) or concept.get(field) is None:
        return None
    raw = concept[field]
    return tuple(raw) if field == "operators" else raw


def _predictions(predictions: Mapping[str, Tensor], n: int, vocabulary: ConceptVocabulary) -> dict[str, Tensor]:
    if set(predictions) != set(CONCEPT_FIELDS):
        missing = sorted(set(CONCEPT_FIELDS) - set(predictions))
        extra = sorted(set(predictions) - set(CONCEPT_FIELDS))
        raise ValueError(f"prediction fields must be exactly {CONCEPT_FIELDS}; missing={missing}, extra={extra}")
    result: dict[str, Tensor] = {}
    for field in CONCEPT_FIELDS:
        value = predictions[field]
        if not isinstance(value, Tensor) or value.ndim != 1 or value.shape[0] != n:
            raise ValueError(f"predictions[{field!r}] must be a one-dimensional tensor of length {n}")
        if value.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
            raise ValueError(f"predictions[{field!r}] must contain integer ids")
        result[field] = value.detach().to(device="cpu", dtype=torch.long)
        if result[field].numel() and (int(result[field].min()) < 0 or int(result[field].max()) > len(vocabulary.fields[field])):
            raise ValueError(f"predictions[{field!r}] contains an id outside 0..{len(vocabulary.fields[field])}")
    return result


def concept_metrics(predictions: Mapping[str, Tensor], targets: Sequence[Mapping[str, Any]], vocabulary: ConceptVocabulary) -> dict[str, Any]:
    """Compute class, field, micro, macro, and complete-frame metrics."""
    n = len(targets)
    pred = _predictions(predictions, n, vocabulary)
    fields: dict[str, Any] = {}
    field_accuracy: dict[str, float | None] = {}
    balanced_accuracy: dict[str, float | None] = {}
    unknown_gold_mask: dict[str, list[bool]] = {}
    missing_gold_mask: dict[str, list[bool]] = {}
    total_correct = total_known = 0
    for field in CONCEPT_FIELDS:
        mapping = vocabulary.fields[field]
        class_stats: dict[Any, dict[str, Any]] = {
            value: {"value": value, "id": int(class_id), "correct": 0, "count": 0, "recall": None}
            for value, class_id in mapping.items()
        }
        missing = unseen = known = correct = 0
        unknown = torch.zeros(n, dtype=torch.bool)
        missing_mask = torch.zeros(n, dtype=torch.bool)
        for i, target in enumerate(targets):
            value = _value(field, target)
            if value is None:
                missing += 1
                missing_mask[i] = True
                continue
            class_id = mapping.get(value)
            if class_id is None:
                unseen += 1
                unknown[i] = True
                continue
            known += 1
            entry = class_stats[value]
            entry["count"] += 1
            if int(pred[field][i]) == int(class_id):
                correct += 1
                entry["correct"] += 1
        recalls = []
        for entry in class_stats.values():
            if entry["count"]:
                entry["recall"] = entry["correct"] / entry["count"]
                recalls.append(entry["recall"])
        accuracy = correct / known if known else None
        balanced = sum(recalls) / len(recalls) if recalls else None
        field_accuracy[field] = accuracy
        balanced_accuracy[field] = balanced
        # JSON uses arrays for classes: tuple-valued operator labels remain
        # ordered, but are represented as JSON arrays rather than dict keys.
        classes = []
        for value, entry in class_stats.items():
            serial_value = list(value) if isinstance(value, tuple) else value
            classes.append({**entry, "value": serial_value})
        unknown_list = unknown.tolist()
        missing_list = missing_mask.tolist()
        fields[field] = {"classes": classes, "accuracy": accuracy, "balanced_accuracy": balanced,
                         "correct": correct, "count": known, "known_count": known,
                         "missing_count": missing, "unseen_count": unseen,
                         "unknown_gold_mask": unknown_list, "missing_gold_mask": missing_list}
        unknown_gold_mask[field] = unknown_list
        missing_gold_mask[field] = missing_list
        total_correct += correct
        total_known += known
    eligible = [score for score in balanced_accuracy.values() if score is not None]
    full_correct = full_count = 0
    for i, target in enumerate(targets):
        known_frame = all(_value(field, target) in vocabulary.fields[field] for field in CONCEPT_FIELDS)
        if not known_frame:
            continue
        full_count += 1
        if all(int(pred[field][i]) == vocabulary.fields[field][_value(field, target)] for field in CONCEPT_FIELDS):
            full_correct += 1
    micro = total_correct / total_known if total_known else None
    macro = sum(eligible) / len(eligible) if eligible else None
    exact = {"correct": full_correct, "count": full_count,
             "accuracy": full_correct / full_count if full_count else None}
    return {"fields": fields, "field_accuracy": field_accuracy,
            "balanced_accuracy": balanced_accuracy, "unknown_gold_mask": unknown_gold_mask,
            "missing_gold_mask": missing_gold_mask,
            "global_micro": {"correct": total_correct, "count": total_known, "accuracy": micro},
            "micro": {"correct": total_correct, "count": total_known, "accuracy": micro},
            "macro": macro, "full_frame_exact_match": exact}


def majority_predictions(train_targets: Sequence[Mapping[str, Any]], vocabulary: ConceptVocabulary, n: int) -> dict[str, Tensor]:
    """Return the training-majority known id for each field, repeated ``n`` times."""
    if type(n) is not int or n < 0:
        raise ValueError("n must be a nonnegative integer")
    encoded = encode_targets(train_targets, vocabulary)
    out: dict[str, Tensor] = {}
    for field in CONCEPT_FIELDS:
        ids = encoded.fields[field][encoded.masks[field]].tolist()
        counts = Counter(ids)
        winner = min(counts, key=lambda value: (-counts[value], value)) if counts else 0
        out[field] = torch.full((n,), int(winner), dtype=torch.long)
    return out


def permutation_diagnostics(targets: Sequence[Mapping[str, Any]], permutation: Sequence[int] | Tensor, vocabulary: ConceptVocabulary) -> dict[str, Any]:
    """Measure label-frame changes under a validated row permutation."""
    n = len(targets)
    if isinstance(permutation, Tensor):
        if permutation.ndim != 1:
            raise ValueError("permutation must be one-dimensional")
        permutation = permutation.detach().cpu().tolist()
    else:
        permutation = list(permutation)
    if len(permutation) != n or any(type(x) is not int for x in permutation):
        raise ValueError("permutation must contain exactly one integer for every target row")
    if sorted(permutation) != list(range(n)):
        raise ValueError("permutation must be a bijection over target row indices")
    complete = [all(_value(field, target) in vocabulary.fields[field] for field in CONCEPT_FIELDS) for target in targets]
    comparable = [i for i, j in enumerate(permutation) if complete[i] and complete[j]]
    different = [i for i in comparable if any(_value(f, targets[i]) != _value(f, targets[permutation[i]]) for f in CONCEPT_FIELDS)]
    differing_byfield = {f: sum(_value(f, targets[i]) != _value(f, targets[permutation[i]]) for i in comparable) for f in CONCEPT_FIELDS}
    return {"moved_count": sum(i != j for i, j in enumerate(permutation)),
            "self_count": sum(i == j for i, j in enumerate(permutation)),
            "complete_comparable_pairs": len(comparable),
            "semantically_different_known_complete_frames": len(different),
            "differing_byfield": differing_byfield}
