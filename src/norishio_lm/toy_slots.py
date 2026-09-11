"""Conservative slot parsing and scoring for the authored toy grammar.

This module deliberately recognizes only the literal templates in the authored
seed.  It is a mechanical frame checker, not a Japanese semantic parser.
"""
from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any

_FIELDS = ("event", "operators", "agent", "participant", "time", "location", "repeat_marked")


def _template_pattern(template: str, people: Sequence[str], times: Sequence[str]) -> re.Pattern[str]:
    """Compile one seed template, replacing only its two named slots."""
    pieces: list[str] = []
    position = 0
    for match in re.finditer(r"\{(person|time)\}", template):
        pieces.append(re.escape(template[position : match.start()]))
        values = people if match.group(1) == "person" else times
        # Values are escaped and longest-first to avoid prefix alternatives.
        alternatives = "|".join(re.escape(value) for value in sorted(values, key=len, reverse=True))
        if not alternatives:
            alternatives = r"(?!)"
        pieces.append(f"(?P<{match.group(1)}>{alternatives})")
        position = match.end()
    pieces.append(re.escape(template[position:]))
    return re.compile("".join(pieces))


def _frame(condition: Mapping[str, Any], captures: Mapping[str, str]) -> dict[str, Any]:
    event = condition["event"]
    return {
        "event": event,
        "operators": list(condition["operators"]),
        "agent": condition["agent"],
        "participant": captures["person"],
        "time": captures["time"],
        "location": "HERE" if event == "STAY" else "UNSPECIFIED",
        "repeat_marked": bool(condition["repeat"]),
    }


def parse_authored_output(text: str, seed: Mapping[str, Any]) -> dict[str, Any] | None:
    """Parse a complete authored target-template output, or return ``None``.

    Matching is a full-string match.  A candidate is accepted when multiple
    spelling variants identify the same frame; ambiguity between distinct
    frames is rejected.
    """
    if not isinstance(text, str) or not isinstance(seed, Mapping):
        return None
    people = seed.get("people")
    times = seed.get("times")
    conditions = seed.get("conditions")
    if not isinstance(people, Sequence) or isinstance(people, (str, bytes)):
        return None
    if not isinstance(times, Sequence) or isinstance(times, (str, bytes)):
        return None
    if not isinstance(conditions, Sequence) or isinstance(conditions, (str, bytes)):
        return None
    people = [value for value in people if isinstance(value, str)]
    times = [value for value in times if isinstance(value, str)]
    matches: dict[tuple[Any, ...], dict[str, Any]] = {}
    for condition in conditions:
        if not isinstance(condition, Mapping):
            continue
        template = condition.get("target")
        if not isinstance(template, str):
            continue
            # A malformed template is simply not an authored grammar rule.
        try:
            pattern = _template_pattern(template, people, times)
            match = pattern.fullmatch(text)
        except (re.error, IndexError, KeyError):
            continue
        if match is None or "person" not in match.groupdict() or "time" not in match.groupdict():
            continue
        frame = _frame(condition, match.groupdict())
        key = tuple(frame[field] if field != "operators" else tuple(frame[field]) for field in _FIELDS)
        matches[key] = frame
    if len(matches) != 1:
        return None
    return next(iter(matches.values()))


def _generated_text(generation: Mapping[str, Any]) -> tuple[str | None, str | None]:
    """Return text and a mechanical rejection reason for one generation."""
    if not isinstance(generation, Mapping):
        return None, "invalid_generation"
    text = generation.get("text")
    if not isinstance(text, str):
        return None, "invalid_text"
    if generation.get("valid_utf8") is not True:
        return None, "invalid_utf8"
    invalid = generation.get("invalid_special_tokens")
    if invalid:
        return None, "invalid_special_tokens"
    if generation.get("ended_eos") is not True:
        return None, "nonEOS"
    return text, None


def _gold(row: Mapping[str, Any]) -> Mapping[str, Any] | None:
    value = row.get("concept", row.get("target", row))
    if isinstance(value, Mapping) and isinstance(value.get("concept"), Mapping):
        value = value["concept"]
    return value if isinstance(value, Mapping) else None


def score_slots(
    generations: Sequence[Mapping[str, Any]],
    targets: Sequence[Mapping[str, Any]],
    seed: Mapping[str, Any],
) -> dict[str, Any]:
    """Score parsed frames while retaining parse failures and per-row examples."""
    if len(generations) != len(targets):
        raise ValueError("generations and targets must have equal length")
    total = len(generations)
    rows: list[dict[str, Any]] = []
    correct = {field: 0 for field in _FIELDS}
    observed = {field: 0 for field in _FIELDS}
    parsed_observed = {field: 0 for field in _FIELDS}
    parseable = 0
    evaluable = 0
    fullframe_correct = 0
    known_fullframe = 0
    for index in range(total):
        generation = generations[index]
        target = targets[index]
        text, rejection = _generated_text(generation)
        parsed = None if rejection else parse_authored_output(text or "", seed)
        if parsed is None and rejection is None:
            rejection = "no_unique_frame"
        gold = _gold(target)
        fields: dict[str, bool | None] = {}
        if gold is not None:
            for field in _FIELDS:
                if field in gold and gold[field] is not None:
                    observed[field] += 1
                    if parsed is not None:
                        parsed_observed[field] += 1
                    gold_value = list(gold[field]) if field == "operators" else gold[field]
                    fields[field] = parsed is not None and parsed.get(field) == gold_value
                    if fields[field]:
                        correct[field] += 1
                else:
                    fields[field] = None
            known = all(field in gold and gold[field] is not None for field in _FIELDS)
            if known:
                known_fullframe += 1
            if parsed is not None and known:
                evaluable += 1
                if all(fields[field] for field in _FIELDS):
                    fullframe_correct += 1
        if parsed is not None:
            parseable += 1
        rows.append({"parsed": parsed, "rejection_reason": rejection, "fields": fields})
    denominator = total
    return {
        "rows": total,
        "parseable_count": parseable,
        "parse_coverage": (parseable / denominator if denominator else None),
        "fields": {field: {"correct": correct[field], "count": observed[field],
                            "accuracy": (correct[field] / observed[field] if observed[field] else None),
                            "evaluable_count": parsed_observed[field],
                            "conditional_accuracy": (correct[field] / parsed_observed[field] if parsed_observed[field] else None)}
                    for field in _FIELDS},
        "evaluable_count": evaluable,
        "conditional_accuracy": (fullframe_correct / evaluable if evaluable else None),
        "fullframe_correct": fullframe_correct,
        "fullframe_count": known_fullframe,
        "fullframe_accuracy": (fullframe_correct / known_fullframe if known_fullframe else None),
        "examples": [{"parsed": row["parsed"], "rejection_reason": row["rejection_reason"]} for row in rows],
    }


__all__ = ["parse_authored_output", "score_slots"]
