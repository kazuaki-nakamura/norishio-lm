"""Model-independent metrics for benchmark-v3 Phase 0.

The scorer is deliberately shared by every future v3 arm.  A row may carry a
generation, a canonical intermediate frame, both, or neither.  Missing
intermediate predictions remain unavailable (``None``), rather than becoming
incorrect predictions.  Source-side swaps and true intermediate probability
interventions have separate schemas so an oracle frame cannot be mistaken for
an intervention produced by a model.
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
import math
from typing import Any, Callable

from .benchmark_v2_metrics import (
    FIELDS,
    _MISSING,
    _first,
    _freeze,
    _generation_diagnostics,
    _jsonable,
    _parse,
    _support_sets,
    _teacher_forced,
    _frame as _v2_frame,
)

Parser = Callable[[str], Any]
PAIR = ("participant", "time")
TRIPLE = ("participant", "time", "event")
SCHEMA = "norishio.benchmark-v3.metrics.v1"
CANONICAL_SPACE = "canonical_semantic_space"
CODE_SPACE = "model_code_space"
FACTOR_WIDTHS = {"participant": 6, "time": 6, "event": 4, "operator": 4}


def _is_sequence(value: Any) -> bool:
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray))


def _ratio(correct: int, count: int) -> float | None:
    return correct / count if count else None


def _exact(correct: int, count: int) -> dict[str, Any]:
    return {"correct": correct, "count": count, "denominator": count,
            "accuracy": _ratio(correct, count)}


def _frame(value: Any, *, label: str) -> dict[str, Any]:
    """Read only a canonical frame; raw/code predictions are never decoded here."""
    if isinstance(value, Mapping):
        for key in ("canonical_frame", "canonical", "semantic_frame"):
            if isinstance(value.get(key), Mapping):
                value = value[key]
                break
    return _v2_frame(value, label=label)


def _target_frame(row: Mapping[str, Any], index: int) -> dict[str, Any]:
    value = _first(row, ("target_frame", "target", "gold_frame", "frame", "targets"))
    if value is _MISSING and all(field in row for field in FIELDS):
        value = row
    return _frame(value, label=f"rows[{index}].target_frame")


def _target_text(row: Mapping[str, Any], index: int) -> str:
    value = _first(row, ("target_text", "reference_text", "expected_text", "gold_text"))
    if value is _MISSING:
        target = row.get("target", row.get("targets"))
        if isinstance(target, Mapping):
            value = _first(target, ("text", "target_text", "reference_text"))
    if not isinstance(value, str):
        raise ValueError(f"rows[{index}].target_text must be a string")
    try:
        value.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise ValueError(f"rows[{index}].target_text must be valid UTF-8") from exc
    return value


def _intermediate_frame(row: Mapping[str, Any], index: int) -> dict[str, Any] | None:
    value = _first(row, ("intermediate_frame", "predicted_frame", "model_frame",
                         "frame_prediction", "canonical_intermediate_frame",
                         "intermediate_prediction", "head_prediction", "intermediate"))
    if value is _MISSING or value is None:
        return None
    return _frame(value, label=f"rows[{index}].intermediate_frame")


def _generation(row: Mapping[str, Any], index: int) -> Mapping[str, Any]:
    value = _first(row, ("generated", "generation", "prediction", "output"))
    if value is not _MISSING:
        if isinstance(value, str):
            return {"text": value}
        if not isinstance(value, Mapping):
            raise ValueError(f"rows[{index}].generated must be a mapping or string")
        inherited = {key: row[key] for key in
                     ("ended_eos", "eos", "valid_utf8", "unique_output",
                      "invalid_special_tokens") if key in row and key not in value}
        return {**inherited, **value}
    text = _first(row, ("generated_text", "output_text", "prediction_text"))
    if text is _MISSING:
        raise ValueError(f"rows[{index}] has no generated output")
    return {"text": text, **{key: row[key] for key in
                              ("ended_eos", "eos", "valid_utf8", "unique_output",
                               "invalid_special_tokens") if key in row}}


def _matches(frame: Mapping[str, Any] | None, target: Mapping[str, Any], fields: Sequence[str]) -> bool:
    return frame is not None and all(frame[field] == target[field] for field in fields)


def _atomic(rows: Sequence[dict[str, Any]], indices: Sequence[int]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    accuracies: dict[str, float | None] = {}
    balanced: dict[str, float | None] = {}
    for field in FIELDS:
        support: dict[Any, int] = {}
        correct_by_class: dict[Any, int] = {}
        correct = 0
        for index in indices:
            gold = rows[index]["target"][field]
            support[gold] = support.get(gold, 0) + 1
            if rows[index]["parsed"] is not None and rows[index]["parsed"][field] == gold:
                correct += 1
                correct_by_class[gold] = correct_by_class.get(gold, 0) + 1
        recalls = [_ratio(correct_by_class.get(value, 0), count)
                   for value, count in support.items() if count]
        accuracies[field] = _ratio(correct, len(indices))
        balanced[field] = sum(recalls) / len(recalls) if recalls else None
        result[field] = {
            "correct": correct, "count": len(indices), "denominator": len(indices),
            "accuracy": accuracies[field], "balanced_accuracy": balanced[field],
            "class_support": [{"value": _jsonable(value), "count": count,
                               "correct": correct_by_class.get(value, 0),
                               "recall": _ratio(correct_by_class.get(value, 0), count)}
                              for value, count in support.items()],
        }
    return {"fields": result, "accuracy": accuracies, "balanced_accuracy": balanced}


def _score_group(indices: Sequence[int], rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    indices = list(indices)
    count = len(indices)
    parseable = sum(rows[i]["parsed"] is not None for i in indices)
    atomic = _atomic(rows, indices)
    intermediate_indices = [i for i in indices if rows[i]["intermediate"] is not None]
    intermediate_exact = sum(_matches(rows[i]["intermediate"], rows[i]["target"], FIELDS)
                             for i in intermediate_indices)
    intermediate_fields = _atomic(
        [{**row, "parsed": row["intermediate"]} for row in rows], indices)
    frame_exact = sum(_matches(rows[i]["parsed"], rows[i]["target"], FIELDS) for i in indices)
    pair_exact = sum(_matches(rows[i]["parsed"], rows[i]["target"], PAIR) for i in indices)
    triple_exact = sum(_matches(rows[i]["parsed"], rows[i]["target"], TRIPLE) for i in indices)
    text_exact = sum(rows[i]["text"] == rows[i]["target_text"] for i in indices)
    free_exact = sum(rows[i]["valid_output"] and rows[i]["text"] == rows[i]["target_text"] and
                     _matches(rows[i]["parsed"], rows[i]["target"], FIELDS) for i in indices)
    matrix = {"intermediate_correct": {"generation_correct": 0, "generation_incorrect": 0},
              "intermediate_incorrect": {"generation_correct": 0, "generation_incorrect": 0}}
    for i in indices:
        if rows[i]["intermediate"] is None:
            continue
        h = "intermediate_correct" if _matches(rows[i]["intermediate"], rows[i]["target"], FIELDS) else "intermediate_incorrect"
        g = "generation_correct" if _matches(rows[i]["parsed"], rows[i]["target"], FIELDS) else "generation_incorrect"
        matrix[h][g] += 1
    validity = {
        "valid": _exact(sum(rows[i]["valid_output"] for i in indices), count),
        "eos": _exact(sum(rows[i]["eos_valid"] for i in indices), count),
        "utf8": _exact(sum(rows[i]["utf8_valid"] for i in indices), count),
        "unique_output": _exact(sum(rows[i]["unique_output"] for i in indices), count),
    }
    intermediate = {
        "available_count": len(intermediate_indices), "count": len(intermediate_indices),
        "row_denominator": count, "denominator": len(intermediate_indices),
        "fields": intermediate_fields["fields"],
        "atomic_accuracy": intermediate_fields["accuracy"],
        "atomic_balanced_accuracy": intermediate_fields["balanced_accuracy"],
        "frame_exact": _exact(intermediate_exact, len(intermediate_indices)),
    }
    return {
        "rows": count, "count": count, "denominator": count,
        "parseable_count": parseable, "parse_coverage": _ratio(parseable, count),
        "fields": atomic["fields"], "atomic_accuracy": atomic["accuracy"],
        "atomic_balanced_accuracy": atomic["balanced_accuracy"],
        "pair_exact": _exact(pair_exact, count), "triple_exact": _exact(triple_exact, count),
        "generation_frame_exact": _exact(frame_exact, count),
        "frame_exact": _exact(frame_exact, count),
        "free_generation_exact": _exact(free_exact, count),
        "exact_target_text": _exact(text_exact, count),
        "target_text_exact": _exact(text_exact, count),
        "exact_text": _exact(text_exact, count),
        "generation_validity": validity,
        "intermediate": intermediate,
        "intermediate_frame_exact": intermediate["frame_exact"],
        "head_vs_generation_2x2": matrix,
        "contingency_2x2": matrix,
        "contingency_row_count": len(intermediate_indices),
        "intermediate_unavailable_count": count - len(intermediate_indices),
    }


def _normalise_rows(rows: Sequence[Mapping[str, Any]], parser: Parser) -> list[dict[str, Any]]:
    if not _is_sequence(rows):
        raise ValueError("rows must be a sequence of mappings")
    if not callable(parser):
        raise ValueError("parser must be callable")
    result: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise ValueError(f"rows[{index}] must be a mapping")
        target = _target_frame(row, index)
        target_text = _target_text(row, index)
        intermediate = _intermediate_frame(row, index)
        diagnostics = _generation_diagnostics(_generation(row, index), index)
        result.append({"target": target, "target_text": target_text, "text": diagnostics["text"],
                       "valid_output": diagnostics["valid"], "eos_valid": diagnostics["eos"],
                       "utf8_valid": diagnostics["utf8"], "unique_output": diagnostics["unique"],
                       "reason": diagnostics["reason"], "parsed": None,
                       "intermediate": intermediate})
    seen: dict[str, list[int]] = {}
    for index, row in enumerate(result):
        if row["text"] is not None:
            seen.setdefault(row["text"], []).append(index)
    duplicate_indices = {i for group in seen.values() if len(group) > 1 for i in group}
    for index, row in enumerate(result):
        if index in duplicate_indices:
            row["unique_output"] = False
        row["parsed"], reason = _parse(parser, row["text"], row["reason"], index)
        if reason is not None:
            row["reason"] = reason
    return result


def score_benchmark_v3(
    rows: Sequence[Mapping[str, Any]],
    parser: Parser,
    *,
    train_support: Any = None,
    source_side_swaps: Sequence[Mapping[str, Any]] | None = None,
    intermediate_probability_interventions: Sequence[Mapping[str, Any]] | None = None,
    interventions: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Score all v3 arms using one model-independent canonical scorer."""
    normalized = _normalise_rows(rows, parser)
    all_indices = list(range(len(normalized)))
    pair_support, triple_support = _support_sets(train_support)
    seen_pair = [i for i, row in enumerate(normalized)
                 if tuple(row["target"][field] for field in PAIR) in pair_support and
                 tuple(row["target"][field] for field in TRIPLE) not in triple_support]
    unseen_pair = [i for i, row in enumerate(normalized)
                   if tuple(row["target"][field] for field in PAIR) not in pair_support]
    excluded_seen_triple = [i for i in all_indices if i not in seen_pair and i not in unseen_pair]
    groups = {"all": _score_group(all_indices, normalized),
              "unseen_pair": _score_group(unseen_pair, normalized),
              "seen_pair": _score_group(seen_pair, normalized)}
    result: dict[str, Any] = {
        "schema": SCHEMA, "version": "benchmark-v3-metrics-1", "row_count": len(normalized),
        "fields": list(FIELDS),
        "space_metadata": {"code_space": CODE_SPACE, "canonical_space": CANONICAL_SPACE,
                            "generation_frame_space": CANONICAL_SPACE,
                            "intermediate_frame_space": CANONICAL_SPACE},
        "train_support": {"provided": train_support is not None,
                           "pair_count": len(pair_support), "triple_count": len(triple_support)},
        "groups": groups, "train_support_groups": groups, "all": groups["all"],
        "required_groups": ["unseen_pair", "seen_pair"],
        "group_counts": {"all": len(all_indices), "unseen_pair": len(unseen_pair),
                         "seen_pair": len(seen_pair), "excluded_seen_triple": len(excluded_seen_triple)},
        "rows": [{"index": i, "parseable": row["parsed"] is not None,
                  "valid_output": row["valid_output"], "unique_output": row["unique_output"],
                  "rejection_reason": row["reason"]} for i, row in enumerate(normalized)],
        "duplicate_output_rows": [i for i, row in enumerate(normalized)
                                   if row["unique_output"] is False],
        "teacher_forced": _teacher_forced(rows),
        "teacher_forced_auxiliary": _teacher_forced(rows),
    }
    # Stable top-level aliases for audit consumers.
    result.update({"atomic_accuracy": groups["all"]["atomic_accuracy"],
                   "atomic_balanced_accuracy": groups["all"]["atomic_balanced_accuracy"],
                   "pair_exact": groups["all"]["pair_exact"],
                   "triple_exact": groups["all"]["triple_exact"],
                   "free_generation_exact": groups["all"]["free_generation_exact"],
                   "exact_target_text": groups["all"]["exact_target_text"],
        "head_vs_generation_2x2": groups["all"]["head_vs_generation_2x2"],
        "parse_coverage": groups["all"]["parse_coverage"],
        "generation_validity": groups["all"]["generation_validity"]})
    if source_side_swaps is not None:
        result["source_side_swap"] = score_source_side_swaps(source_side_swaps, parser=parser)
    if intermediate_probability_interventions is None and interventions is not None:
        intermediate_probability_interventions = interventions
    if intermediate_probability_interventions is not None:
        result["intermediate_probability_intervention"] = score_intermediate_probability_interventions(
            intermediate_probability_interventions, parser=parser)
    return _jsonable(result)


def _swap_output(item: Mapping[str, Any], keys: Sequence[str], index: int) -> Mapping[str, Any]:
    value = _first(item, keys)
    if value is _MISSING:
        raise ValueError(f"swaps[{index}] is missing {' or '.join(keys)}")
    if isinstance(value, str):
        return {"text": value}
    if not isinstance(value, Mapping):
        raise ValueError(f"swaps[{index}] output must be a mapping or string")
    return value


def score_source_side_swaps(
    swaps: Sequence[Mapping[str, Any]], *, parser: Parser,
) -> dict[str, Any]:
    """Score source-side one-factor swaps, with parse failures in the denominator."""
    if not _is_sequence(swaps) or not callable(parser):
        raise ValueError("swaps must be a sequence and parser must be callable")
    examples: list[dict[str, Any]] = []
    for index, item in enumerate(swaps):
        if not isinstance(item, Mapping):
            raise ValueError(f"swaps[{index}] must be a mapping")
        factor = _first(item, ("factor", "target_factor", "changed_factor"))
        if factor not in FIELDS:
            raise ValueError(f"swaps[{index}] has unknown factor {factor!r}")
        before = _swap_output(item, ("baseline", "before", "baseline_generation", "baseline_frame"), index)
        after = _swap_output(item, ("changed", "after", "changed_generation", "swapped_generation", "changed_frame"), index)
        bdiag = _generation_diagnostics(before, index) if not all(field in before for field in FIELDS) else None
        adiag = _generation_diagnostics(after, index) if not all(field in after for field in FIELDS) else None
        bt, br = (bdiag["text"], bdiag["reason"]) if bdiag else (None, None)
        at, ar = (adiag["text"], adiag["reason"]) if adiag else (None, None)
        try:
            bframe = _frame(before, label=f"swaps[{index}].baseline_frame") if bdiag is None else None
            aframe = _frame(after, label=f"swaps[{index}].changed_frame") if adiag is None else None
            if bdiag is not None:
                bframe, br = _parse(parser, bt, br, index)
            if adiag is not None:
                aframe, ar = _parse(parser, at, ar, index)
        except Exception as exc:
            bframe = aframe = None
            br = ar = f"parser_error:{type(exc).__name__}"
        failures = [reason for reason in (br, ar) if reason is not None]
        actual_changed = bool(bframe is not None and aframe is not None and bframe[factor] != aframe[factor])
        preserved = bool(bframe is not None and aframe is not None and
                         all(bframe[field] == aframe[field] for field in FIELDS if field != factor))
        baseline_target = _first(item, ("baseline_target_frame", "before_target_frame"))
        changed_target = _first(item, ("changed_target_frame", "after_target_frame", "target_frame"))
        expected_changed: bool | None = None
        if baseline_target is not _MISSING and changed_target is not _MISSING:
            btarget = _frame(baseline_target, label=f"swaps[{index}].baseline_target_frame")
            atarget = _frame(changed_target, label=f"swaps[{index}].changed_target_frame")
            expected_changed = btarget[factor] != atarget[factor]
        examples.append({"index": index, "factor": factor, "target_changed": actual_changed,
                         "expected_target_changed": expected_changed,
                         "non_target_preserved": preserved,
                         "parse_failure": bool(failures), "parse_failures": failures})
    by_factor: dict[str, Any] = {}
    for factor in FIELDS:
        subset = [row for row in examples if row["factor"] == factor]
        n = len(subset)
        comparable = [row for row in subset if row["expected_target_changed"] is not None and not row["parse_failure"]]
        correct_change = sum(row["target_changed"] == row["expected_target_changed"] for row in comparable)
        by_factor[factor] = {"rows": n, "count": n, "denominator": n,
                             "target_changed": _exact(sum(row["target_changed"] for row in subset), n),
                             "non_target_preserved": _exact(sum(row["non_target_preserved"] for row in subset), n),
                             "target_change_correct": _exact(correct_change, len(comparable)),
                             "parse_failures": sum(row["parse_failure"] for row in subset)}
    return _jsonable({"schema": "norishio.benchmark-v3.source-side-swap.v1",
                      "kind": "source_side_swap", "oracle": False,
                      "space_metadata": {"code_space": CODE_SPACE, "canonical_space": CANONICAL_SPACE},
                      "rows": len(examples), "by_factor": by_factor,
                      "parse_failures": sum(row["parse_failure"] for row in examples),
                      "examples": examples})


def _check_probability(value: Any, label: str, width: int) -> list[float]:
    if not _is_sequence(value) or len(value) != width:
        raise ValueError(f"{label} must have width {width}")
    try:
        vector = [float(item) for item in value]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be numeric") from exc
    if any(not math.isfinite(item) or item < 0 for item in vector) or not math.isclose(sum(vector), 1.0, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError(f"{label} must be a finite probability simplex")
    return vector


def _check_one_hot(value: Any, label: str, width: int) -> list[float]:
    vector = _check_probability(value, label, width)
    if sum(item == 1.0 for item in vector) != 1 or any(item not in (0.0, 1.0) for item in vector):
        raise ValueError(f"{label} must be a fixed artificial one-hot vector")
    return vector


def _required_evidence(item: Mapping[str, Any], index: int) -> None:
    """Require paired run evidence before comparing intermediate outputs."""
    before_source = _first(item, ("baseline_source_identity", "source_identity_before",
                                  "baseline_source", "before_source_identity"))
    after_source = _first(item, ("intervened_source_identity", "source_identity_after",
                                 "intervened_source", "after_source_identity"))
    if before_source is _MISSING or after_source is _MISSING:
        evidence_before = item.get("baseline_evidence", item.get("baseline"))
        evidence_after = item.get("intervened_evidence", item.get("intervened"))
        if not isinstance(evidence_after, Mapping):
            evidence_after = item.get("changed", item.get("after"))
        if isinstance(evidence_before, Mapping) and isinstance(evidence_after, Mapping):
            before_source = evidence_before.get("source_identity", _MISSING)
            after_source = evidence_after.get("source_identity", _MISSING)
            before_prefix = evidence_before.get("decoder_prefix", _MISSING)
            after_prefix = evidence_after.get("decoder_prefix", _MISSING)
        else:
            before_prefix = _MISSING
            after_prefix = _MISSING
    else:
        before_prefix = _first(item, ("baseline_decoder_prefix", "decoder_prefix_before",
                                       "baseline_prefix", "before_decoder_prefix"))
        after_prefix = _first(item, ("intervened_decoder_prefix", "decoder_prefix_after",
                                     "intervened_prefix", "after_decoder_prefix"))
    if before_source is _MISSING or after_source is _MISSING:
        raise ValueError(f"interventions[{index}] requires baseline and intervened source identity")
    if isinstance(before_source, str):
        source_present = bool(before_source.strip())
    else:
        source_present = before_source is not None and bool(before_source)
    if isinstance(after_source, str):
        after_present = bool(after_source.strip())
    else:
        after_present = after_source is not None and bool(after_source)
    if not source_present or not after_present or before_source != after_source:
        raise ValueError(f"interventions[{index}] requires identical nonempty source identity")
    if before_prefix is _MISSING or after_prefix is _MISSING:
        raise ValueError(f"interventions[{index}] requires baseline and intervened decoder prefix")
    if before_prefix is None or after_prefix is None or before_prefix != after_prefix:
        raise ValueError(f"interventions[{index}] requires identical decoder prefix")


def _probability_maps(item: Mapping[str, Any], index: int) -> tuple[dict[str, list[float]], dict[str, list[float]]]:
    baseline = _first(item, ("baseline_probabilities", "baseline_intermediate_probabilities"))
    intervened = _first(item, ("intervened_probabilities", "intervention_probabilities",
                               "intermediate_probabilities"))
    if not isinstance(baseline, Mapping) or not isinstance(intervened, Mapping):
        raise ValueError(f"interventions[{index}] requires baseline and intervened probabilities")
    missing = [field for field in FIELDS if field not in baseline or field not in intervened]
    if missing:
        raise ValueError(f"interventions[{index}] probabilities must cover all factors: {missing}")
    baseline_checked = {field: _check_probability(baseline[field], f"interventions[{index}].baseline_probabilities.{field}", FACTOR_WIDTHS[field]) for field in FIELDS}
    intervened_checked = {field: _check_probability(intervened[field], f"interventions[{index}].intervened_probabilities.{field}", FACTOR_WIDTHS[field]) for field in FIELDS}
    return baseline_checked, intervened_checked


def score_intermediate_probability_interventions(
    interventions: Sequence[Mapping[str, Any]], *, parser: Parser,
) -> dict[str, Any]:
    """Score true intermediate interventions supplied as fixed artificial one-hot vectors.

    This report is intentionally separate from source-side swaps and oracle
    diagnostics.  It validates the intervention representation, but does not
    claim that an oracle frame was generated by a model.
    """
    if not _is_sequence(interventions) or not callable(parser):
        raise ValueError("interventions must be a sequence and parser must be callable")
    examples: list[dict[str, Any]] = []
    for index, item in enumerate(interventions):
        if not isinstance(item, Mapping):
            raise ValueError(f"interventions[{index}] must be a mapping")
        if item.get("oracle", False) is True or item.get("source") == "oracle":
            raise ValueError("oracle interventions must use a separate diagnostic schema")
        factor = _first(item, ("factor", "target_factor", "changed_factor"))
        if factor not in FIELDS:
            raise ValueError(f"interventions[{index}] has unknown factor {factor!r}")
        _required_evidence(item, index)
        baseline_probabilities, intervened_probabilities = _probability_maps(item, index)
        one_hot = _first(item, ("intervened_one_hot", "intervention_one_hot", "one_hot"))
        if one_hot is _MISSING:
            one_hot = intervened_probabilities[factor]
        if one_hot is _MISSING:
            raise ValueError(f"interventions[{index}] is missing fixed artificial one-hot")
        vector = _check_one_hot(one_hot, f"interventions[{index}].one_hot", FACTOR_WIDTHS[factor])
        if intervened_probabilities[factor] != vector:
            raise ValueError(f"interventions[{index}] selected probability vector must equal declared one-hot")
        if baseline_probabilities[factor] == intervened_probabilities[factor]:
            raise ValueError(f"interventions[{index}] selected factor probability vector did not change")
        for non_target in FIELDS:
            if non_target != factor and baseline_probabilities[non_target] != intervened_probabilities[non_target]:
                raise ValueError(f"interventions[{index}] changed non-target probability vector {non_target!r}")
        before = _swap_output(item, ("baseline", "before", "baseline_generation", "baseline_frame"), index)
        after = _swap_output(item, ("intervened", "changed", "after", "intervention_generation", "intervened_frame"), index)
        if all(field in before for field in FIELDS):
            bframe, br = _frame(before, label=f"interventions[{index}].baseline_frame"), None
        else:
            bt = _generation_diagnostics(before, index)
            bframe, br = _parse(parser, bt["text"], bt["reason"], index)
        if all(field in after for field in FIELDS):
            aframe, ar = _frame(after, label=f"interventions[{index}].intervened_frame"), None
        else:
            at = _generation_diagnostics(after, index)
            aframe, ar = _parse(parser, at["text"], at["reason"], index)
        failures = [reason for reason in (br, ar) if reason is not None]
        changed = bool(bframe is not None and aframe is not None and bframe[factor] != aframe[factor])
        preserved = bool(bframe is not None and aframe is not None and
                         all(bframe[field] == aframe[field] for field in FIELDS if field != factor))
        examples.append({"index": index, "factor": factor, "one_hot": vector,
                         "target_changed": changed, "non_target_preserved": preserved,
                         "parse_failure": bool(failures), "parse_failures": failures})
    by_factor: dict[str, Any] = {}
    for factor in FIELDS:
        subset = [row for row in examples if row["factor"] == factor]
        n = len(subset)
        by_factor[factor] = {"rows": n, "count": n, "denominator": n,
                             "target_changed": _exact(sum(row["target_changed"] for row in subset), n),
                             "non_target_preserved": _exact(sum(row["non_target_preserved"] for row in subset), n),
                             "parse_failures": sum(row["parse_failure"] for row in subset)}
    return _jsonable({"schema": "norishio.benchmark-v3.intermediate-probability-intervention.v1",
                      "kind": "true_intermediate_probability_intervention", "oracle": False,
                      "intervention": "fixed_artificial_one_hot",
                      "space_metadata": {"code_space": CODE_SPACE, "canonical_space": CANONICAL_SPACE},
                      "rows": len(examples), "by_factor": by_factor,
                      "parse_failures": sum(row["parse_failure"] for row in examples),
                      "examples": examples})


# Short aliases make the schemas discoverable to future runners without arm-specific code.
score_v3 = score_benchmark_v3
score_rows = score_benchmark_v3
score_source_side_swap = score_source_side_swaps
score_intermediate_probability_intervention = score_intermediate_probability_interventions
score_source_side_swap_diagnostic = score_source_side_swaps
score_intermediate_intervention = score_intermediate_probability_interventions

__all__ = ["FIELDS", "PAIR", "TRIPLE", "score_benchmark_v3", "score_v3", "score_rows",
           "score_source_side_swaps", "score_source_side_swap",
           "score_source_side_swap_diagnostic", "score_intermediate_intervention",
           "score_intermediate_probability_interventions",
           "score_intermediate_probability_intervention"]
