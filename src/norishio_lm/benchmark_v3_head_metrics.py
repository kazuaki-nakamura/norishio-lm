"""Deterministic aggregation for canonical benchmark-v3 factor-head rows.

This module deliberately consumes retained raw rows only.  It does not load a
model or checkpoint, run inference, inspect generated text, or rank arms.  A
row identifies its ``arm`` and ``seed`` and contains a gold canonical factor
mapping plus an optional predicted mapping.  The public aggregator keeps
training resubstitution and confirmation evidence separate and computes
micro-counted arm aggregates so that seed sizes cannot silently reweight a
result.

The accepted row aliases are intentionally small and evaluator-facing:
``split`` (``train``/``confirmation``), ``stratum`` (confirmation only),
``target_frame``/``gold``/``target`` and
``predicted_frame``/``prediction``/``predicted``/``head_prediction``.  A
caller may also pass the explicit ``gold_factors`` and ``predicted_factors``
keys.  Factor names and class values remain caller supplied; no vocabulary or
unfinished fixture is imported here.
"""
from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any


DEFAULT_STRATA = ("unseen_pair", "seen_pair/unseen_higher_order")
DEFAULT_ARMS = ("JOINT", "FACTOR_ONLY")
_MISSING = object()
_SPLIT_ALIASES = {
    "train": "train_resubstitution",
    "training": "train_resubstitution",
    "train_resubstitution": "train_resubstitution",
    "resubstitution": "train_resubstitution",
    "confirmation": "confirmation",
    "confirm": "confirmation",
    "validation": "confirmation",
    "eval": "confirmation",
}
_STRATUM_ALIASES = {
    "unseen_pair": "unseen_pair",
    "unseen-pair": "unseen_pair",
    "seen_pair/unseen_higher_order": "seen_pair/unseen_higher_order",
    "seen_pair_unseen_higher_order": "seen_pair/unseen_higher_order",
    "seen_pair/unseen_triple": "seen_pair/unseen_higher_order",
    "seen_pair": "seen_pair/unseen_higher_order",
    "seen-pair/unseen-higher-order": "seen_pair/unseen_higher_order",
}


def _rate(correct: int, count: int) -> float | None:
    return correct / count if count else None


def _counts(correct: int, count: int, *, scheduled: int | None = None,
            unavailable: int | None = None) -> dict[str, Any]:
    result: dict[str, Any] = {
        "correct": int(correct), "count": int(count), "denominator": int(count), "rate": _rate(correct, count),
    }
    if scheduled is not None:
        result["scheduled_count"] = int(scheduled)
    if unavailable is not None:
        result["unavailable_count"] = int(unavailable)
    return result


def _is_seq(value: Any) -> bool:
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray))


def _first(mapping: Mapping[str, Any], names: Iterable[str]) -> Any:
    for name in names:
        if name in mapping:
            return mapping[name]
    return _MISSING


def _factor_mapping(value: Any, factors: tuple[str, ...], *, label: str, index: int,
                    allow_none: bool = True) -> dict[str, Any] | None:
    if value is None and allow_none:
        return None
    if isinstance(value, Mapping):
        # Some retained rows wrap the canonical mapping in one of these keys.
        for key in ("canonical", "canonical_frame", "factors", "frame"):
            nested = value.get(key)
            if isinstance(nested, Mapping) and not any(f in value for f in factors):
                value = nested
                break
        result = {factor: value[factor] for factor in factors if factor in value}
        return result
    if _is_seq(value):
        if len(value) != len(factors):
            raise ValueError(f"rows[{index}].{label} must have {len(factors)} values")
        return dict(zip(factors, value))
    raise ValueError(f"rows[{index}].{label} must be a mapping, sequence, or null")


def _class_list(value: Any, factor: str, *, label: str) -> tuple[Any, ...]:
    if isinstance(value, Mapping):
        # A support map may be {class: count}; preserve its key order only for
        # diagnostics, while duplicate class keys are impossible by mapping.
        value = tuple(value.keys())
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Iterable):
        raise ValueError(f"{label}[{factor!r}] must be an iterable of expected classes")
    classes = tuple(value)
    if len(set(classes)) != len(classes):
        raise ValueError(f"{label}[{factor!r}] contains duplicate classes")
    return classes


def _normalise_expected(value: Mapping[str, Any], factors: tuple[str, ...], *, label: str) -> dict[str, tuple[Any, ...]]:
    if not isinstance(value, Mapping) or set(value) != set(factors):
        raise ValueError(f"{label} must contain exactly the factor names {list(factors)!r}")
    return {factor: _class_list(value[factor], factor, label=label) for factor in factors}


def _normalise_supports(value: Any, factors: tuple[str, ...]) -> dict[str, Any]:
    """Return optional caller-supplied support counts without inventing support.

    Supports may be ``{factor: {class: count}}`` for all rows, or nested under
    ``train_resubstitution``/``confirmation``/stratum.  They are diagnostics;
    observed gold rows remain the denominator for rates.
    """
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError("supports must be a mapping")
    if set(value).issubset(set(factors)):
        return {factor: value.get(factor) for factor in factors}
    return {str(key): _normalise_supports(item, factors) for key, item in value.items()}


@dataclass(frozen=True)
class _Row:
    index: int
    arm: str
    seed: int | str
    split: str
    stratum: str | None
    gold: dict[str, Any]
    prediction: dict[str, Any] | None


def _normalise_rows(raw_records: Sequence[Mapping[str, Any]], factors: tuple[str, ...],
                    expected: Mapping[str, tuple[Any, ...]]) -> list[_Row]:
    if not _is_seq(raw_records):
        raise ValueError("records must be a sequence of mappings")
    rows: list[_Row] = []
    for index, record in enumerate(raw_records):
        if not isinstance(record, Mapping):
            raise ValueError(f"records[{index}] must be a mapping")
        arm = record.get("arm")
        if not isinstance(arm, str) or not arm:
            raise ValueError(f"records[{index}].arm must be a nonempty string")
        if "seed" not in record or isinstance(record["seed"], bool) or not isinstance(record["seed"], (int, str)):
            raise ValueError(f"records[{index}].seed must be an int or string")
        split_value = record.get("split", record.get("dataset_split"))
        if not isinstance(split_value, str) or split_value.lower() not in _SPLIT_ALIASES:
            raise ValueError(f"records[{index}].split must be train or confirmation")
        split = _SPLIT_ALIASES[split_value.lower()]
        stratum_value = record.get("stratum", record.get("support_group"))
        if split == "confirmation":
            if not isinstance(stratum_value, str) or not stratum_value:
                raise ValueError(f"records[{index}].stratum is required for confirmation rows")
            stratum = _STRATUM_ALIASES.get(stratum_value.lower(), stratum_value)
        else:
            stratum = None
        gold_value = _first(record, ("gold_factors", "target_frame", "gold", "target", "targets"))
        if gold_value is _MISSING:
            raise ValueError(f"records[{index}] has no canonical gold factor mapping")
        gold = _factor_mapping(gold_value, factors, label="gold", index=index, allow_none=False)
        assert gold is not None
        if set(gold) != set(factors):
            missing = [factor for factor in factors if factor not in gold]
            raise ValueError(f"records[{index}].gold is missing factors {missing!r}")
        for factor in factors:
            if gold[factor] not in expected[factor]:
                raise ValueError(f"records[{index}] gold {factor} class is outside expected classes")
        prediction_value = _first(record, ("predicted_factors", "predicted_frame", "prediction",
                                           "predicted", "head_prediction", "factor_prediction",
                                           "factor_argmax", "predicted_classes"))
        prediction = None if prediction_value is _MISSING else _factor_mapping(
            prediction_value, factors, label="prediction", index=index)
        if prediction is not None:
            # A partial prediction is unavailable for the missing factors, but
            # unknown values are malformed raw data and must fail loudly.
            for factor, predicted in prediction.items():
                if factor not in factors:
                    raise ValueError(f"records[{index}].prediction has unknown factor {factor!r}")
                if predicted not in expected[factor]:
                    raise ValueError(f"records[{index}] prediction {factor} class is outside expected classes")
        availability = record.get("available", record.get("head_available", _MISSING))
        if availability is not _MISSING:
            if isinstance(availability, bool):
                if not availability:
                    prediction = None
            elif isinstance(availability, Mapping):
                if prediction is not None:
                    prediction = {factor: value for factor, value in prediction.items()
                                  if availability.get(factor, True) is True}
            else:
                raise ValueError(f"records[{index}].available must be boolean or a factor mapping")
        rows.append(_Row(index, arm, record["seed"], split, stratum, gold, prediction))
    return rows


def _metric_for_rows(rows: Sequence[_Row], factors: tuple[str, ...], expected: Mapping[str, tuple[Any, ...]],
                     supports: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Compute metrics for one arm/seed/split/stratum group."""
    scheduled = len(rows)
    fields: dict[str, Any] = {}
    for factor in factors:
        class_rows: dict[Any, list[_Row]] = {value: [] for value in expected[factor]}
        correct = 0
        available = 0
        for row in rows:
            prediction = row.prediction
            if prediction is None or factor not in prediction:
                continue
            available += 1
            class_rows[row.gold[factor]].append(row)
            if prediction[factor] == row.gold[factor]:
                correct += 1
        # Gold supports are counted even if the corresponding prediction is
        # unavailable.  This makes missing head output visible in each class.
        observed_support = {value: sum(row.gold[factor] == value for row in rows)
                            for value in expected[factor]}
        per_class: dict[str, Any] = {}
        recalls: list[float] = []
        for value in expected[factor]:
            available_class = len(class_rows[value])
            class_correct = sum(r.prediction is not None and r.prediction.get(factor) == value
                                for r in class_rows[value])
            support_count = observed_support[value]
            rate = _rate(class_correct, available_class)
            status = "missing" if support_count == 0 else ("unavailable" if available_class == 0 else "available")
            if rate is not None:
                recalls.append(rate)
            item: dict[str, Any] = {
                "value": value, "correct": class_correct, "count": available_class,
                "denominator": available_class,
                "rate": rate, "support": support_count, "available_count": available_class,
                "unavailable_count": support_count - available_class, "status": status,
            }
            if supports and factor in supports and isinstance(supports[factor], Mapping):
                item["supplied_support"] = supports[factor].get(value, 0)
            per_class[str(value)] = item
        fields[factor] = {
            "correct": correct, "count": available, "denominator": available, "rate": _rate(correct, available),
            "scheduled_count": scheduled, "available_count": available,
            "unavailable_count": scheduled - available,
            "class_balanced_accuracy": _rate(sum(recalls), len(recalls)) if recalls else None,
            "per_class": per_class,
            "class_support": list(per_class.values()),
        }
    complete_available = sum(row.prediction is not None and set(row.prediction) == set(factors) for row in rows)
    complete_correct = sum(
        row.prediction is not None and set(row.prediction) == set(factors)
        and all(row.prediction[factor] == row.gold[factor] for factor in factors)
        for row in rows
    )
    result: dict[str, Any] = {
        "rows": scheduled, "scheduled_count": scheduled,
        "available_count": complete_available, "unavailable_count": scheduled - complete_available,
        "fields": fields,
        "atomic": {factor: fields[factor] for factor in factors},
        "joint_exact": _counts(complete_correct, complete_available,
                                scheduled=scheduled, unavailable=scheduled - complete_available),
        "four_factor_joint_exact": _counts(complete_correct, complete_available,
                                            scheduled=scheduled, unavailable=scheduled - complete_available),
    }
    return result


def _aggregate_group(rows: Sequence[_Row], factors: tuple[str, ...], expected: Mapping[str, tuple[Any, ...]],
                     supports: Mapping[str, Any]) -> dict[str, Any]:
    return _metric_for_rows(rows, factors, expected, supports)


def _merge_metrics(metrics: Sequence[dict[str, Any]], factors: tuple[str, ...]) -> dict[str, Any]:
    """Merge exact counts across seeds/strata; rates are computed afterward."""
    scheduled = sum(int(item["scheduled_count"]) for item in metrics)
    available = sum(int(item["available_count"]) for item in metrics)
    correct = sum(int(item["joint_exact"]["correct"]) for item in metrics)
    fields: dict[str, Any] = {}
    for factor in factors:
        fmetrics = [item["fields"][factor] for item in metrics]
        fscheduled = sum(int(item["scheduled_count"]) for item in fmetrics)
        favailable = sum(int(item["count"]) for item in fmetrics)
        fcorrect = sum(int(item["correct"]) for item in fmetrics)
        class_values = sorted({str(value) for item in fmetrics for value in item["per_class"]})
        per_class: dict[str, Any] = {}
        for value in class_values:
            items = [item["per_class"][value] for item in fmetrics if value in item["per_class"]]
            c = sum(int(item["correct"]) for item in items)
            n = sum(int(item["count"]) for item in items)
            support = sum(int(item["support"]) for item in items)
            per_class[value] = {"value": items[0]["value"], "correct": c, "count": n,
                                "denominator": n,
                                "rate": _rate(c, n), "support": support,
                                "available_count": n, "unavailable_count": support - n,
                                "status": "missing" if support == 0 else ("unavailable" if n == 0 else "available")}
        recalls = [item["rate"] for item in per_class.values() if item["rate"] is not None]
        fields[factor] = {"correct": fcorrect, "count": favailable, "denominator": favailable,
                          "rate": _rate(fcorrect, favailable),
                          "scheduled_count": fscheduled, "available_count": favailable,
                          "unavailable_count": fscheduled - favailable,
                          "class_balanced_accuracy": sum(recalls) / len(recalls) if recalls else None,
                          "per_class": per_class, "class_support": list(per_class.values())}
    return {"rows": scheduled, "scheduled_count": scheduled, "available_count": available,
            "unavailable_count": scheduled - available, "fields": fields,
            "atomic": {factor: fields[factor] for factor in factors},
            "joint_exact": _counts(correct, available, scheduled=scheduled,
                                    unavailable=scheduled - available),
            "four_factor_joint_exact": _counts(correct, available, scheduled=scheduled,
                                                unavailable=scheduled - available)}


def _scalar_direction(joint: Mapping[str, Any], factor_only: Mapping[str, Any]) -> dict[str, Any]:
    """Compare one atomic metric using the preregistered arm order."""
    joint_rate = joint["rate"]
    factor_only_rate = factor_only["rate"]
    if joint_rate is None or factor_only_rate is None:
        return {"joint_rate": joint_rate, "factor_only_rate": factor_only_rate,
                "delta": None, "direction": "unavailable",
                "delta_definition": "FACTOR_ONLY - JOINT"}
    delta = factor_only_rate - joint_rate
    return {
        "joint_rate": joint_rate, "factor_only_rate": factor_only_rate,
        "delta": delta,
        "direction": "factor_only_better" if delta > 0 else ("joint_better" if delta < 0 else "tie"),
        "delta_definition": "FACTOR_ONLY - JOINT",
    }


def _direction(joint: dict[str, Any], factor_only: dict[str, Any],
               factors: Sequence[str] = ()) -> dict[str, Any]:
    j = joint["joint_exact"]
    f = factor_only["joint_exact"]
    if not j["count"] or not f["count"]:
        result = {"direction": "unavailable", "joint_rate": j["rate"],
                "factor_only_rate": f["rate"], "delta": None,
                "delta_definition": "FACTOR_ONLY - JOINT"}
    else:
        # The preregistered comparison is FACTOR_ONLY - JOINT.  Keep the arm
        # rates beside the delta so a direction never hides the underlying counts.
        delta = f["rate"] - j["rate"]
        result = {"direction": "factor_only_better" if delta > 0 else ("joint_better" if delta < 0 else "tie"),
                  "joint_rate": j["rate"], "factor_only_rate": f["rate"], "delta": delta,
                  "delta_definition": "FACTOR_ONLY - JOINT"}
    result["fields"] = {
        factor: _scalar_direction(joint["fields"][factor], factor_only["fields"][factor])
        for factor in factors
    }
    return result


def aggregate_head_metrics(
    records: Sequence[Mapping[str, Any]],
    expected_factor_classes: Mapping[str, Any],
    *,
    supports: Mapping[str, Any] | None = None,
    factors: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Aggregate retained canonical factor-head rows.

    ``expected_factor_classes`` is required so zero-support classes can be
    reported explicitly.  ``supports`` is optional and copied into per-class
    diagnostics when supplied.  It never changes a denominator.  The returned
    ``train_resubstitution`` and ``confirmation`` trees are separate in every
    arm/seed entry; confirmation includes the two required strata even when
    one has zero rows.
    """
    if factors is None:
        if not isinstance(expected_factor_classes, Mapping):
            raise ValueError("expected_factor_classes must be a mapping")
        factors_tuple = tuple(expected_factor_classes.keys())
    else:
        factors_tuple = tuple(factors)
    if not factors_tuple or len(set(factors_tuple)) != len(factors_tuple):
        raise ValueError("factors must be a nonempty sequence of unique names")
    expected = _normalise_expected(expected_factor_classes, factors_tuple, label="expected_factor_classes")
    normalised = _normalise_rows(records, factors_tuple, expected)
    normalised_supports = _normalise_supports(supports, factors_tuple)
    arms = sorted({row.arm for row in normalised})
    seeds = sorted({row.seed for row in normalised}, key=lambda value: (str(type(value)), value))
    observed_strata = sorted({row.stratum for row in normalised
                               if row.split == "confirmation" and row.stratum is not None})
    confirmation_strata = list(DEFAULT_STRATA) + [stratum for stratum in observed_strata
                                                   if stratum not in DEFAULT_STRATA]
    by_arm_seed: dict[str, dict[str, Any]] = {}
    for arm in arms:
        by_arm_seed[arm] = {}
        for seed in seeds:
            selected = [row for row in normalised if row.arm == arm and row.seed == seed]
            train = [row for row in selected if row.split == "train_resubstitution"]
            confirmation = [row for row in selected if row.split == "confirmation"]
            by_arm_seed[arm][str(seed)] = {
                "seed": seed,
                "train_resubstitution": _aggregate_group(train, factors_tuple, expected,
                                                          normalised_supports.get("train_resubstitution", normalised_supports)),
                "confirmation": {
                    "all": _aggregate_group(confirmation, factors_tuple, expected,
                                             normalised_supports.get("confirmation", normalised_supports)),
                    **{stratum: _aggregate_group(
                        [row for row in confirmation if row.stratum == stratum], factors_tuple, expected,
                        normalised_supports.get(stratum, normalised_supports.get("confirmation", normalised_supports)))
                        for stratum in confirmation_strata},
                },
            }
    # Aggregate exact counts over all rows for an arm, preserving the split and
    # stratum boundaries in the public report.
    arm_aggregates: dict[str, Any] = {}
    for arm in arms:
        selected = [row for row in normalised if row.arm == arm]
        train = [row for row in selected if row.split == "train_resubstitution"]
        confirmation = [row for row in selected if row.split == "confirmation"]
        arm_aggregates[arm] = {
            "train_resubstitution": _aggregate_group(train, factors_tuple, expected,
                                                      normalised_supports.get("train_resubstitution", normalised_supports)),
            "confirmation": {
                "all": _aggregate_group(confirmation, factors_tuple, expected,
                                         normalised_supports.get("confirmation", normalised_supports)),
                **{stratum: _aggregate_group(
                    [row for row in confirmation if row.stratum == stratum], factors_tuple, expected,
                    normalised_supports.get(stratum, normalised_supports.get("confirmation", normalised_supports)))
                    for stratum in confirmation_strata},
            },
        }
    paired: dict[str, Any] = {}
    if "JOINT" in by_arm_seed and "FACTOR_ONLY" in by_arm_seed:
        for seed in seeds:
            key = str(seed)
            paired[key] = {}
            for split in ("train_resubstitution", "confirmation"):
                names = ["all"] if split == "train_resubstitution" else ["all", *confirmation_strata]
                paired[key][split] = {name: _direction(
                    by_arm_seed["JOINT"][key][split] if split == "train_resubstitution" else by_arm_seed["JOINT"][key][split][name],
                    by_arm_seed["FACTOR_ONLY"][key][split] if split == "train_resubstitution" else by_arm_seed["FACTOR_ONLY"][key][split][name],
                    factors_tuple,
                ) for name in names}
    return {
        "schema": "norishio.benchmark-v3.head-metrics.v1",
        "factors": list(factors_tuple),
        "expected_factor_classes": {factor: list(expected[factor]) for factor in factors_tuple},
        "arms": arms,
        "seeds": list(seeds),
        "by_arm_seed": by_arm_seed,
        "arm_aggregate": arm_aggregates,
        "paired_seed_directions": paired,
        "supports": supports,
        "row_count": len(normalised),
    }


# A descriptive alias is useful to callers that prefer the full term.
aggregate_benchmark_v3_head_metrics = aggregate_head_metrics
aggregate_factor_head_metrics = aggregate_head_metrics
factor_head_metrics = aggregate_head_metrics


__all__ = ["DEFAULT_ARMS", "DEFAULT_STRATA", "aggregate_head_metrics",
           "aggregate_benchmark_v3_head_metrics", "aggregate_factor_head_metrics",
           "factor_head_metrics"]
