"""Participant/time pair support and generation diagnostics.

Pair membership is derived from the training annotations only.  Generation
scoring delegates parsing and the ordinary metrics to the existing toy
scorers, so this module does not introduce a second grammar or scorer.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from .concept_model import ConceptVocabulary
from .toy_controls import generation_metrics
from .toy_slots import score_slots


def _concept(row: Mapping[str, Any]) -> Mapping[str, Any]:
    value: Any = row.get("concept")
    if value is None and isinstance(row.get("targets"), Mapping):
        value = row["targets"].get("concept")
    if not isinstance(value, Mapping):
        raise ValueError("each target must contain a concept mapping")
    return value


def _class_values(vocabulary: ConceptVocabulary, field: str) -> list[Any]:
    return [value for value, _ in sorted(vocabulary.fields[field].items(), key=lambda item: item[1])]


def _pair(row: Mapping[str, Any], vocabulary: ConceptVocabulary) -> tuple[Any, Any]:
    concept = _concept(row)
    values: list[Any] = []
    for field in ("participant", "time"):
        value = concept.get(field)
        if value is None or value not in vocabulary.fields[field]:
            raise ValueError(f"unknown {field} value: {value!r}")
        values.append(value)
    return values[0], values[1]


def _table(rows: Sequence[Mapping[str, Any]], participants: Sequence[Any], times: Sequence[Any], vocabulary: ConceptVocabulary) -> list[list[int]]:
    pids = {value: i for i, value in enumerate(participants)}
    tids = {value: i for i, value in enumerate(times)}
    result = [[0 for _ in times] for _ in participants]
    for row in rows:
        person, time = _pair(row, vocabulary)
        result[pids[person]][tids[time]] += 1
    return result


def pair_support(
    train_targets: list[Mapping[str, Any]],
    validation_targets: list[Mapping[str, Any]],
    vocabulary: ConceptVocabulary,
) -> dict[str, Any]:
    """Count participant/time support using vocabulary classes fit on train."""
    participants = _class_values(vocabulary, "participant")
    times = _class_values(vocabulary, "time")
    if len(participants) != 5 or len(times) != 5:
        raise ValueError("participant and time vocabularies must each contain five classes")
    train_table = _table(train_targets, participants, times, vocabulary)
    validation_table = _table(validation_targets, participants, times, vocabulary)
    seen_pairs = {
        (participants[i], times[j])
        for i, row in enumerate(train_table)
        for j, count in enumerate(row)
        if count
    }
    validation_seen = [
        (_pair(row, vocabulary) in seen_pairs) for row in validation_targets
    ]
    return {
        "train_table5x5": train_table,
        "validation_table5x5": validation_table,
        "class_values": {"participant": participants, "time": times},
        "validation_seen": validation_seen,
        "seen_rows": sum(validation_seen),
        "unseen_rows": len(validation_seen) - sum(validation_seen),
    }


def _text(target: Mapping[str, Any]) -> str:
    value: Any = target.get("text")
    if value is None and isinstance(target.get("targets"), Mapping):
        value = target["targets"].get("text")
    if not isinstance(value, str):
        raise ValueError("each target must contain reference text")
    return value


def _both_slots(slot_result: Mapping[str, Any], targets: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    correct = 0
    for example, target in zip(slot_result["examples"], targets):
        parsed = example.get("parsed")
        if parsed is None:
            continue
        concept = _concept(target)
        if (parsed.get("participant") == concept.get("participant") and
                parsed.get("time") == concept.get("time")):
            correct += 1
    count = len(targets)
    return {"correct": correct, "count": count, "accuracy": correct / count if count else None}


def _group(generated: Sequence[Mapping[str, Any]], targets: Sequence[Mapping[str, Any]], seed: Mapping[str, Any]) -> dict[str, Any]:
    rows = len(targets)
    if len(generated) != rows:
        raise ValueError("generated and targets must have equal length")
    slots = score_slots(generated, targets, seed)
    return {
        "generation": generation_metrics(list(generated), [_text(target) for target in targets]) if rows else None,
        "slots": slots,
        "both_slots": _both_slots(slots, targets),
        "rows": rows,
    }


def group_pair_scores(
    generated: list[dict[str, Any]],
    targets: list[Mapping[str, Any]],
    seed_grammar: Mapping[str, Any],
    seen: list[bool],
) -> dict[str, Any]:
    """Score all rows and fixed seen/unseen pair subgroups."""
    if len(generated) != len(targets) or len(seen) != len(targets):
        raise ValueError("generated, targets, and seen must have equal length")
    groups = {
        "all": list(range(len(targets))),
        "seen_pair": [i for i, value in enumerate(seen) if value],
        "unseen_pair": [i for i, value in enumerate(seen) if not value],
    }
    return {
        name: _group([generated[i] for i in indexes], [targets[i] for i in indexes], seed_grammar)
        for name, indexes in groups.items()
    }


__all__ = ["pair_support", "group_pair_scores"]
