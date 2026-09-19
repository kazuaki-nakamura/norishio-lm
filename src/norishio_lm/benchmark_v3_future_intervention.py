"""Future-only alternate-value intervention helpers for benchmark-v3."""
from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

import torch
from torch import Tensor

from .benchmark_v3_contract import INTERVENTION_RULE
from .benchmark_v3_metrics import (
    FIELDS,
    _check_one_hot,
    score_intermediate_probability_interventions,
)


FACTOR_WIDTHS = {"participant": 6, "time": 6, "event": 4, "operator": 4}


def alternate_intervention_one_hot(
    probabilities: Mapping[str, Tensor], factor: str, *, index: int = 0,
) -> tuple[Tensor, int, int]:
    """Choose ``(baseline_argmax + 1) % width`` for a future probe."""

    if factor not in FIELDS:
        raise ValueError(f"unknown factor {factor!r}")
    selected = probabilities.get(factor)
    if not isinstance(selected, Tensor) or selected.ndim != 2:
        raise ValueError("factor probabilities must be rank-2 tensors")
    if type(index) is not int or not 0 <= index < selected.shape[0]:
        raise ValueError("probability row index is out of range")
    width = selected.shape[1]
    if width != FACTOR_WIDTHS[factor]:
        raise ValueError(f"factor probability width must be {FACTOR_WIDTHS[factor]}")
    baseline_class = int(selected[index].argmax(dim=-1).item())
    intervention_class = (baseline_class + 1) % width
    one_hot = torch.zeros((1, width), dtype=selected.dtype, device=selected.device)
    one_hot[0, intervention_class] = 1.0
    return one_hot, baseline_class, intervention_class


def _argmax(values: Any, *, factor: str, index: int) -> int:
    if (
        not isinstance(values, Sequence)
        or isinstance(values, (str, bytes))
        or len(values) != FACTOR_WIDTHS[factor]
    ):
        raise ValueError(f"interventions[{index}] baseline factor width is invalid")
    return max(range(len(values)), key=values.__getitem__)


def score_future_intermediate_probability_interventions(
    interventions: Sequence[Mapping[str, Any]], *, parser: Callable[[str], Any],
) -> dict[str, Any]:
    """Validate alternate-class metadata, then use the historical base scorer."""

    checked: list[Mapping[str, Any]] = []
    metadata: list[dict[str, Any]] = []
    for index, item in enumerate(interventions):
        if not isinstance(item, Mapping):
            raise ValueError(f"interventions[{index}] must be a mapping")
        factor = item.get("factor")
        if factor not in FIELDS:
            raise ValueError(f"interventions[{index}] has unknown factor")
        baseline = item.get("baseline_probabilities")
        if not isinstance(baseline, Mapping):
            raise ValueError(f"interventions[{index}] requires baseline probabilities")
        baseline_class = _argmax(baseline.get(factor), factor=factor, index=index)
        expected_class = (baseline_class + 1) % FACTOR_WIDTHS[factor]
        baseline_metadata = item.get("baseline_argmax_class")
        intervention_metadata = item.get("intervention_class")
        if type(baseline_metadata) is not int:
            raise ValueError(f"interventions[{index}] baseline argmax metadata must be an int")
        if type(intervention_metadata) is not int:
            raise ValueError(f"interventions[{index}] intervention class metadata must be an int")
        if item.get("intervention_rule") != INTERVENTION_RULE:
            raise ValueError(f"interventions[{index}] has unknown intervention rule")
        if baseline_metadata != baseline_class:
            raise ValueError(f"interventions[{index}] baseline argmax metadata mismatch")
        if intervention_metadata != expected_class:
            raise ValueError(f"interventions[{index}] alternate intervention class mismatch")
        intervened = item.get("intervened_probabilities")
        if not isinstance(intervened, Mapping):
            raise ValueError(f"interventions[{index}] requires intervened probabilities")
        actual_vector = _check_one_hot(
            intervened.get(factor),
            f"interventions[{index}].intervened_probabilities.{factor}",
            FACTOR_WIDTHS[factor],
        )
        actual_class = actual_vector.index(1.0)
        if actual_class == baseline_class:
            raise ValueError(f"interventions[{index}] actual intervention class must differ from baseline argmax")
        if actual_class != intervention_metadata:
            raise ValueError(f"interventions[{index}] actual intervention class mismatch")
        metadata.append({
            "intervention_rule": INTERVENTION_RULE,
            "baseline_argmax_class": baseline_class,
            "intervention_class": expected_class,
        })
        checked.append(item)

    report = score_intermediate_probability_interventions(checked, parser=parser)
    report["schema"] = "norishio.issue39.future-intermediate-probability-intervention.v1"
    report["intervention_rule"] = INTERVENTION_RULE
    for example, values in zip(report["examples"], metadata, strict=True):
        example.update(values)
    return report


__all__ = [
    "FACTOR_WIDTHS",
    "alternate_intervention_one_hot",
    "score_future_intermediate_probability_interventions",
]
