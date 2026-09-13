"""Common, model-independent metrics for frozen benchmark-v2 rows.

The scorer intentionally knows only four categorical frame factors and a
caller-supplied strict parser.  It does not infer semantics, tokenize text,
or turn teacher-forced diagnostics into free-generation results.
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
import math
from typing import Any, Callable


FIELDS = ("participant", "time", "event", "operator")
_PAIR = ("participant", "time")
_TRIPLE = ("participant", "time", "event")
_MISSING = object()
Parser = Callable[[str], Any]


def _is_sequence(value: Any) -> bool:
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray))


def _jsonable(value: Any) -> Any:
    """Convert ordinary tuple/set values to JSON-compatible values."""
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, set):
        return [_jsonable(item) for item in sorted(value, key=repr)]
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("non-finite floats are not JSON serializable")
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise ValueError(f"value {value!r} is not JSON serializable")


def _freeze(value: Any) -> Any:
    """Make frame values comparable, including ordered operator lists."""
    if isinstance(value, Mapping):
        return tuple(sorted((str(key), _freeze(item)) for key, item in value.items()))
    if isinstance(value, (tuple, list)):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, set):
        return tuple(sorted((_freeze(item) for item in value), key=repr))
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("frame values must be finite")
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise ValueError(f"frame value {value!r} is not a supported JSON value")


def _field_value(frame: Mapping[str, Any], field: str) -> Any:
    key = "operators" if field == "operator" and "operator" not in frame else field
    if key not in frame or frame[key] is None:
        raise ValueError(f"frame is missing non-null field {field!r}")
    return _freeze(frame[key])


def _frame(value: Any, *, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping")
    if isinstance(value.get("targets"), Mapping):
        value = value["targets"]
    if isinstance(value.get("frame"), Mapping):
        value = value["frame"]
    if isinstance(value.get("concept"), Mapping):
        value = value["concept"]
    result: dict[str, Any] = {}
    for field in FIELDS:
        result[field] = _field_value(value, field)
    return result


def _first(mapping: Mapping[str, Any], keys: Sequence[str], default: Any = _MISSING) -> Any:
    for key in keys:
        if key in mapping:
            return mapping[key]
    return default


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
                          "frame_prediction", "intermediate"))
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


def _generation_diagnostics(generation: Mapping[str, Any], index: int) -> dict[str, Any]:
    value = _first(generation, ("text", "generated_text", "output_text"))
    text_valid = isinstance(value, str)
    encodable = False
    if text_valid:
        try:
            value.encode("utf-8")
            encodable = True
        except UnicodeEncodeError:
            pass
    utf8 = _first(generation, ("valid_utf8", "utf8"))
    if utf8 is not _MISSING and type(utf8) is not bool:
        raise ValueError(f"generated output {index} valid_utf8 must be bool")
    eos = _first(generation, ("ended_eos", "eos", "terminated_eos"))
    if eos is not _MISSING and type(eos) is not bool:
        raise ValueError(f"generated output {index} ended_eos must be bool")
    invalid = generation.get("invalid_special_tokens", ())
    if invalid is None:
        invalid = ()
    if isinstance(invalid, (str, bytes)) or not isinstance(invalid, Iterable):
        raise ValueError(f"generated output {index} invalid_special_tokens must be iterable")
    special_valid = not list(invalid)
    unique = _first(generation, ("unique_output", "unique"))
    if unique is not _MISSING and type(unique) is not bool:
        raise ValueError(f"generated output {index} unique_output must be bool")
    utf8_valid = encodable and utf8 is not False
    eos_valid = eos is True
    reason = ("invalid_text" if not text_valid else
              "invalid_utf8" if not utf8_valid else
              "missing_eos" if eos is _MISSING else
              "nonEOS" if not eos_valid else
              "invalid_special_tokens" if not special_valid else None)
    return {"text": value if text_valid and utf8_valid and eos_valid and special_valid else None,
            "reason": reason, "valid": reason is None, "eos": eos_valid,
            "utf8": utf8_valid, "special_tokens": special_valid,
            "unique": unique is not False}


def _generation_text(generation: Mapping[str, Any], index: int) -> tuple[str | None, str | None]:
    diagnostics = _generation_diagnostics(generation, index)
    return diagnostics["text"], diagnostics["reason"]


def _parse(parser: Parser, text: str | None, reason: str | None, index: int) -> tuple[dict[str, Any] | None, str | None]:
    if reason is not None or text is None:
        return None, reason
    try:
        parsed = parser(text)
    except Exception as exc:  # strict parser failures remain ordinary row failures
        return None, f"parser_error:{type(exc).__name__}"
    if isinstance(parsed, Mapping) and "frames" in parsed:
        parsed = parsed["frames"]
    if _is_sequence(parsed):
        if len(parsed) != 1:
            return None, "non_unique_parse"
        parsed = parsed[0]
    if not isinstance(parsed, Mapping):
        return None, "parse_failure"
    try:
        return _frame(parsed, label=f"parsed frame {index}"), None
    except ValueError:
        return None, "invalid_parse_frame"


def _ratio(correct: int, count: int) -> float | None:
    return correct / count if count else None


def _exact(correct: int, count: int) -> dict[str, Any]:
    return {"correct": correct, "count": count, "accuracy": _ratio(correct, count)}


def _frame_matches(frame: Mapping[str, Any] | None, target: Mapping[str, Any], fields: Sequence[str]) -> bool:
    return frame is not None and all(frame[field] == target[field] for field in fields)


def _score_indices(indices: Sequence[int], rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    n = len(indices)
    parseable = sum(rows[i]["parsed"] is not None for i in indices)
    fields: dict[str, Any] = {}
    field_accuracy: dict[str, float | None] = {}
    balanced_accuracy: dict[str, float | None] = {}
    for field in FIELDS:
        support: dict[Any, int] = {}
        correct_by_class: dict[Any, int] = {}
        correct = 0
        for i in indices:
            gold = rows[i]["target"][field]
            support[gold] = support.get(gold, 0) + 1
            parsed = rows[i]["parsed"]
            if parsed is not None and parsed[field] == gold:
                correct += 1
                correct_by_class[gold] = correct_by_class.get(gold, 0) + 1
        recalls = [_ratio(correct_by_class.get(value, 0), count)
                   for value, count in support.items() if count]
        balanced = sum(recalls) / len(recalls) if recalls else None
        field_accuracy[field] = _ratio(correct, n)
        balanced_accuracy[field] = balanced
        fields[field] = {"correct": correct, "count": n,
                         "accuracy": field_accuracy[field],
                         "balanced_accuracy": balanced,
                         "class_support": [{"value": _jsonable(value), "count": count,
                                            "correct": correct_by_class.get(value, 0),
                                            "recall": _ratio(correct_by_class.get(value, 0), count)}
                                           for value, count in support.items()]}

    pair_correct = sum(_frame_matches(rows[i]["parsed"], rows[i]["target"], _PAIR) for i in indices)
    triple_correct = sum(_frame_matches(rows[i]["parsed"], rows[i]["target"], _TRIPLE) for i in indices)
    generation_intermediate = sum(_frame_matches(rows[i]["parsed"], rows[i]["target"], FIELDS)
                                  for i in indices)
    intermediate_indices = [i for i in indices if rows[i]["intermediate"] is not None]
    intermediate_field_accuracy: dict[str, float | None] = {}
    intermediate_balanced_accuracy: dict[str, float | None] = {}
    intermediate_fields: dict[str, Any] = {}
    for field in FIELDS:
        support: dict[Any, int] = {}
        correct_by_class: dict[Any, int] = {}
        correct = 0
        for i in intermediate_indices:
            gold = rows[i]["target"][field]
            support[gold] = support.get(gold, 0) + 1
            if rows[i]["intermediate"][field] == gold:
                correct += 1
                correct_by_class[gold] = correct_by_class.get(gold, 0) + 1
        recalls = [_ratio(correct_by_class.get(value, 0), count)
                   for value, count in support.items() if count]
        balanced = sum(recalls) / len(recalls) if recalls else None
        accuracy = _ratio(correct, len(intermediate_indices))
        intermediate_field_accuracy[field] = accuracy
        intermediate_balanced_accuracy[field] = balanced
        intermediate_fields[field] = {"correct": correct, "count": len(intermediate_indices),
                                      "row_denominator": n, "accuracy": accuracy,
                                      "balanced_accuracy": balanced}
    intermediate_pair = sum(_frame_matches(rows[i]["intermediate"], rows[i]["target"], _PAIR)
                            for i in indices)
    intermediate_triple = sum(_frame_matches(rows[i]["intermediate"], rows[i]["target"], _TRIPLE)
                              for i in indices)
    intermediate_exact = sum(_frame_matches(rows[i]["intermediate"], rows[i]["target"], FIELDS)
                             for i in indices)
    text_exact = sum(rows[i]["text"] == rows[i]["target_text"] for i in indices)
    generation_frame = generation_intermediate
    generation_correct = sum(rows[i]["valid_output"] and rows[i]["text"] == rows[i]["target_text"] and
                             rows[i]["parsed"] is not None and
                             all(rows[i]["parsed"][field] == rows[i]["target"][field] for field in FIELDS)
                             for i in indices)
    matrix = {"intermediate_correct": {"generation_correct": 0, "generation_incorrect": 0},
              "intermediate_incorrect": {"generation_correct": 0, "generation_incorrect": 0}}
    for i in indices:
        predicted = rows[i]["intermediate"]
        if predicted is None:
            continue
        intermediate_group = ("intermediate_correct"
                              if _frame_matches(predicted, rows[i]["target"], FIELDS)
                              else "intermediate_incorrect")
        generation_ok = _frame_matches(rows[i]["parsed"], rows[i]["target"], FIELDS)
        matrix[intermediate_group][
            "generation_correct" if generation_ok else "generation_incorrect"] += 1
    return {
        "rows": n,
        "parseable_count": parseable,
        "parse_coverage": _ratio(parseable, n),
        "fields": fields,
        "field_accuracy": field_accuracy,
        "balanced_accuracy": balanced_accuracy,
        "participant_time_pair_exact": _exact(pair_correct, n),
        "participant_time_event_triple_exact": _exact(triple_correct, n),
        "pair_exact": _exact(pair_correct, n),
        "triple_exact": _exact(triple_correct, n),
        "intermediate": {"available_count": len(intermediate_indices), "row_denominator": n,
                          "fields": intermediate_fields,
                          "field_accuracy": intermediate_field_accuracy,
                          "balanced_accuracy": intermediate_balanced_accuracy,
                          "participant_time_pair_exact": _exact(intermediate_pair, len(intermediate_indices)),
                          "participant_time_event_triple_exact": _exact(intermediate_triple, len(intermediate_indices)),
                          "frame_exact": _exact(intermediate_exact, len(intermediate_indices))},
        "intermediate_frame_exact": _exact(intermediate_exact, len(intermediate_indices)),
        "exact_target_text": _exact(text_exact, n),
        "target_text_exact": _exact(text_exact, n),
        "generation_frame_exact": _exact(generation_frame, n),
        "free_generation_exact": _exact(generation_correct, n),
        "generation_validity": {
            "valid": _exact(sum(rows[i]["valid_output"] for i in indices), n),
            "eos": _exact(sum(rows[i]["eos_valid"] for i in indices), n),
            "utf8": _exact(sum(rows[i]["utf8_valid"] for i in indices), n),
            "unique_output": _exact(sum(rows[i]["unique_output"] for i in indices), n),
        },
        "contingency_2x2": matrix,
        "contingency_row_count": len(intermediate_indices),
        "intermediate_unavailable_count": n - len(intermediate_indices),
    }


def _support_sets(train_support: Any) -> tuple[set[tuple[Any, ...]], set[tuple[Any, ...]]]:
    if train_support is None:
        return set(), set()
    value = train_support
    if isinstance(value, Mapping):
        for key in ("rows", "train_rows", "targets", "frames"):
            if key in value:
                value = value[key]
                break
        else:
            pairs = value.get("pairs", value.get("participant_time", ()))
            triples = value.get("triples", value.get("participant_time_event", ()))
            if isinstance(pairs, (str, bytes)) or isinstance(triples, (str, bytes)):
                raise ValueError("train_support pairs and triples must be sequences")
            pair_set: set[tuple[Any, ...]] = set()
            triple_set: set[tuple[Any, ...]] = set()
            for pair in pairs:
                if not _is_sequence(pair) or len(pair) != len(_PAIR):
                    raise ValueError("train_support pairs must contain two values")
                pair_set.add(tuple(_freeze(x) for x in pair))
            for triple in triples:
                if not _is_sequence(triple) or len(triple) != len(_TRIPLE):
                    raise ValueError("train_support triples must contain three values")
                triple_set.add(tuple(_freeze(x) for x in triple))
            return pair_set, triple_set
    if isinstance(value, (str, bytes)) or not isinstance(value, Iterable):
        raise ValueError("train_support must be a sequence of rows or a support mapping")
    pairs: set[tuple[Any, ...]] = set()
    triples: set[tuple[Any, ...]] = set()
    for index, item in enumerate(value):
        if _is_sequence(item) and len(item) in (len(_PAIR), len(_TRIPLE)):
            frozen = tuple(_freeze(x) for x in item)
            if len(frozen) == len(_PAIR):
                pairs.add(frozen)
            else:
                triples.add(frozen)
            continue
        frame = _frame(item.get("target_frame", item.get("target", item)) if isinstance(item, Mapping) else item,
                      label=f"train_support[{index}]")
        pairs.add(tuple(frame[field] for field in _PAIR))
        triples.add(tuple(frame[field] for field in _TRIPLE))
    return pairs, triples


def _teacher_forced(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any] | None:
    supplied: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        value = _first(row, ("teacher_forced_bytes", "teacher_forced_diagnostic"))
        if value is _MISSING:
            value = row.get("teacher_forced")
        if value is not None:
            if not isinstance(value, Mapping):
                raise ValueError(f"rows[{index}].teacher_forced_bytes must be a mapping")
            required = {"exact", "matched_bytes", "expected_bytes"}
            if not required.issubset(value):
                raise ValueError(f"rows[{index}].teacher_forced_bytes is incomplete")
            exact = value["exact"]
            matched = value["matched_bytes"]
            expected = value["expected_bytes"]
            if type(exact) is not bool:
                raise ValueError(f"rows[{index}] teacher-forced exact must be bool")
            if type(matched) is not int or type(expected) is not int:
                raise ValueError(f"rows[{index}] teacher-forced byte counts must be integers")
            if matched < 0 or expected < 0 or matched > expected:
                raise ValueError(f"rows[{index}] teacher-forced byte counts are invalid")
            if exact != (matched == expected):
                raise ValueError(f"rows[{index}] teacher-forced exact disagrees with byte counts")
            supplied.append(dict(value))
    if not supplied:
        return None
    exact = [value["exact"] for value in supplied]
    matched = [value["matched_bytes"] for value in supplied]
    expected = [value["expected_bytes"] for value in supplied]
    return {"diagnostic_only": True, "rows": len(supplied), "row_denominator": len(rows),
            "coverage": _ratio(len(supplied), len(rows)),
            "exact_rows": sum(exact),
            "exact_rate": _ratio(sum(exact), len(supplied)),
            "matched_bytes": sum(matched),
            "expected_bytes": sum(expected),
            "byte_match_rate": _ratio(sum(matched), sum(expected)),
            "rows_data": [_jsonable(dict(value)) for value in supplied]}


def score_benchmark_v2(
    rows: Sequence[Mapping[str, Any]],
    parser: Parser,
    *,
    train_support: Any = None,
    interventions: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Score frozen rows with a strict parser and train-only support splits."""
    if not _is_sequence(rows):
        raise ValueError("rows must be a sequence of mappings")
    if not callable(parser):
        raise ValueError("parser must be callable")
    normalized: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise ValueError(f"rows[{index}] must be a mapping")
        target = _target_frame(row, index)
        target_text = _target_text(row, index)
        intermediate = _intermediate_frame(row, index)
        generation = _generation(row, index)
        diagnostics = _generation_diagnostics(generation, index)
        normalized.append({"target": target, "target_text": target_text,
                           "text": diagnostics["text"],
                           "valid_output": diagnostics["valid"],
                           "eos_valid": diagnostics["eos"],
                           "utf8_valid": diagnostics["utf8"],
                           "unique_output": diagnostics["unique"],
                           "reason": diagnostics["reason"], "parsed": None,
                           "intermediate": intermediate})
    seen_texts: dict[str, list[int]] = {}
    for index, item in enumerate(normalized):
        if item["text"] is not None:
            seen_texts.setdefault(item["text"], []).append(index)
    duplicate_indices = {index for indexes in seen_texts.values() if len(indexes) > 1 for index in indexes}
    for index, item in enumerate(normalized):
        if index in duplicate_indices:
            item["unique_output"] = False
        item["parsed"], parse_reason = _parse(parser, item["text"], item["reason"], index)
        if parse_reason is not None:
            item["reason"] = parse_reason
    pair_support, triple_support = _support_sets(train_support)
    all_indices = list(range(len(normalized)))
    groups: dict[str, dict[str, Any]] = {"all": _score_indices(all_indices, normalized)}
    if train_support is not None:
        seen_pair = [i for i, item in enumerate(normalized)
                     if tuple(item["target"][field] for field in _PAIR) in pair_support]
        unseen_pair = [i for i in all_indices if i not in seen_pair]
        seen_triple = [i for i, item in enumerate(normalized)
                       if tuple(item["target"][field] for field in _TRIPLE) in triple_support]
        unseen_triple = [i for i in all_indices if i not in seen_triple]
    else:
        seen_pair = unseen_pair = seen_triple = unseen_triple = []
    groups.update({"train_seen": _score_indices(seen_pair, normalized),
                   "train_unseen": _score_indices(unseen_pair, normalized),
                   "seen_pair": _score_indices(seen_pair, normalized),
                   "unseen_pair": _score_indices(unseen_pair, normalized),
                   "train_seen_triple": _score_indices(seen_triple, normalized),
                   "train_unseen_triple": _score_indices(unseen_triple, normalized)})
    groups["seen"] = groups["train_seen"]
    groups["unseen"] = groups["train_unseen"]
    result: dict[str, Any] = {"version": "benchmark-v2-metrics-1", "row_count": len(normalized),
                              "fields": list(FIELDS), "groups": groups, "all": groups["all"],
                              "seen": groups["train_seen"], "unseen": groups["train_unseen"],
                              "train_support": {"provided": train_support is not None,
                                                 "seen_pair_count": len(pair_support),
                                                 "seen_triple_count": len(triple_support)},
                              "duplicate_output_rows": sorted(duplicate_indices),
                              "rows": [{"index": index, "parseable": item["parsed"] is not None,
                                        "valid_output": item["valid_output"],
                                        "unique_output": item["unique_output"],
                                        "rejection_reason": item["reason"]}
                                       for index, item in enumerate(normalized)],
                              "teacher_forced_bytes": _teacher_forced(rows)}
    # Canonical tournament-facing aliases. Detailed denominator-bearing data
    # stays under ``groups``/``all`` while these keys match the frozen contract.
    result.update({
        "atomic_accuracy": groups["all"]["field_accuracy"],
        "atomic_balanced_accuracy": groups["all"]["balanced_accuracy"],
        "pair_exact": groups["all"]["pair_exact"],
        "triple_exact": groups["all"]["triple_exact"],
        "train_support_groups": groups,
        "exact_target_text": groups["all"]["exact_target_text"],
        "eos": groups["all"]["generation_validity"]["eos"],
        "utf8": groups["all"]["generation_validity"]["utf8"],
        "unique_output": groups["all"]["generation_validity"]["unique_output"],
        "intermediate_generation_2x2": groups["all"]["contingency_2x2"],
        "teacher_forced_bytes_separate": result["teacher_forced_bytes"],
    })
    if interventions is not None:
        result["intervention_locality"] = score_intervention_locality(interventions, parser=parser)
    return _jsonable(result)


def score_intervention_locality(
    interventions: Sequence[Mapping[str, Any]],
    *,
    parser: Parser | None = None,
) -> dict[str, Any]:
    """Score preservation of every non-target factor after one-factor changes."""
    if not _is_sequence(interventions):
        raise ValueError("interventions must be a sequence of mappings")
    if parser is not None and not callable(parser):
        raise ValueError("parser must be callable")
    parsed_rows: list[dict[str, Any]] = []
    for index, item in enumerate(interventions):
        if not isinstance(item, Mapping):
            raise ValueError(f"interventions[{index}] must be a mapping")
        factor = _first(item, ("target_factor", "changed_factor", "factor", "requested_factor"))
        if not isinstance(factor, str) or factor not in FIELDS:
            raise ValueError(f"interventions[{index}] has unknown target factor {factor!r}")
        before = _first(item, ("baseline_frame", "before_frame"))
        after = _first(item, ("changed_frame", "after_frame", "intervention_frame"))
        reasons: list[str] = []
        if before is _MISSING or after is _MISSING:
            if parser is None:
                raise ValueError(f"interventions[{index}] needs frames or parser plus outputs")
            baseline_output = _first(item, ("baseline", "before", "baseline_generation"))
            changed_output = _first(item, ("changed", "after", "intervention", "intervened_generation"))
            if not isinstance(baseline_output, Mapping) or not isinstance(changed_output, Mapping):
                raise ValueError(f"interventions[{index}] must provide baseline and changed outputs")
            btext, breason = _generation_text(baseline_output, index)
            atext, areason = _generation_text(changed_output, index)
            before, breason2 = _parse(parser, btext, breason, index)
            after, areason2 = _parse(parser, atext, areason, index)
            reasons.extend(x for x in (breason2, areason2) if x is not None)
        else:
            try:
                before = _frame(before, label=f"interventions[{index}].baseline_frame")
                after = _frame(after, label=f"interventions[{index}].changed_frame")
            except ValueError:
                reasons.append("invalid_frame")
                before = after = None
        preserved = bool(before is not None and after is not None and
                         all(before[field] == after[field] for field in FIELDS if field != factor))
        changed = bool(before is not None and after is not None and before[factor] != after[factor])
        parsed_rows.append({"factor": factor, "preserved": preserved, "changed": changed,
                            "parse_failure": bool(reasons), "rejection_reasons": reasons})
    by_factor: dict[str, Any] = {}
    for factor in FIELDS:
        subset = [row for row in parsed_rows if row["factor"] == factor]
        count = len(subset)
        by_factor[factor] = {"rows": count, "preserved_non_target": sum(row["preserved"] for row in subset),
                             "preservation_rate": _ratio(sum(row["preserved"] for row in subset), count),
                             "target_changed": sum(row["changed"] for row in subset),
                             "target_change_rate": _ratio(sum(row["changed"] for row in subset), count),
                             "parse_failures": sum(row["parse_failure"] for row in subset)}
    return _jsonable({"rows": len(parsed_rows), "by_factor": by_factor,
                      "examples": parsed_rows,
                      "parse_failures": sum(row["parse_failure"] for row in parsed_rows)})


score_v2 = score_benchmark_v2
score_free_generation = score_benchmark_v2
score_rows = score_benchmark_v2
intervention_locality = score_intervention_locality

__all__ = ["FIELDS", "score_benchmark_v2", "score_v2", "score_rows", "score_free_generation",
           "score_intervention_locality", "intervention_locality"]
